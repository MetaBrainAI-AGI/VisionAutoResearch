//! vision_autoresearch — native Rust port of Karpathy's keep-or-revert autoresearch loop,
//! GENERALIZED (KB autoresearch.md + SEAMS).
//!
//! Karpathy's `autoresearch` (released 2026-03-07; github.com/karpathy/autoresearch) runs an
//! agent loop over a single editable artifact (`train.py`) scored by a FROZEN evaluator
//! (`evaluate_bpb()` in `prepare.py`): propose ONE change -> commit -> run the frozen scorer
//! -> if the scalar metric strictly improves KEEP the commit (new baseline), else `git reset`
//! (revert). The LLM is both the mutation operator and the selection pressure. The
//! generalization contract: a problem is autoresearchable IFF (1) ONE bounded editable
//! artifact, (2) a FROZEN un-gameable evaluator, (3) a SCALAR metric — "the constraint is the
//! metric, not the domain."
//!
//! This crate is that loop, native Rust:
//!   * `metric`      — Direction + strict keep-or-revert gate.
//!   * `evaluator`   — frozen `Evaluator` trait + `ScalarEvaluator` (generic) +
//!                     `VectorBtEvaluator` (vectorized backtest: sharpe/sortino/total_return/
//!                     max_drawdown/calmar/win_rate/cvar over a scenario ensemble).
//!   * `scenario`    — `ScenarioSource` trait + `MirofishScenarioSource` (native MC rollouts).
//!   * `proposer`    — the mutation operator (`Proposer` trait + Local/Variation proposers).
//!   * `attempt_log` — JSONL attempt log (results.tsv) + the git `Ratchet` (keep/revert).
//!   * `engine`      — `Run` + `run_iteration`/`run_loop` linear ratchet + RAYON scenario
//!                     `sweep`/`spawn_swarm` scale-out.
//!
//! The `python` feature adds a PyO3 facade (`AutoResearchNative` #[pyclass] + helper
//! #[pyfunction]s) so `modules/vp_autoresearch.py` and the dashboard `/api/autoresearch` can
//! drive the native loop, mirroring `metabrain_core` (allow_threads, kernel-call counters).

pub mod attempt_log;
pub mod engine;
pub mod evaluator;
pub mod metric;
pub mod proposer;
pub mod scenario;

// Re-export the public surface so downstream Rust crates (and the facade) get a flat API.
pub use attempt_log::{AttemptLog, AttemptRow, GitRatchet, NoopRatchet, Ratchet};
pub use engine::{
    scenario_sweep, spawn_swarm, sweep, vectorbt_evaluator_from_source, IterationResult, Run,
    StopCondition, SwarmMode, SwarmResult, SweepResult, AR_ITERATIONS,
};
pub use evaluator::{
    bt_metric_of_path, Artifact, BtMetric, Evaluator, ScalarEvaluator, VectorBtEvaluator,
};
pub use metric::{Direction, KeepOrRevert, Metric};
pub use proposer::{Candidate, LocalProposer, Proposer, VariationProposer};
pub use scenario::{MirofishScenarioSource, ScenarioSource};

// ───────────────────────── PyO3 facade (feature = "python") ─────────────────────────
#[cfg(feature = "python")]
mod pyfacade {
    use crate::attempt_log::{AttemptLog, NoopRatchet};
    use crate::engine::{spawn_swarm, vectorbt_evaluator_from_source, Run, StopCondition, SwarmMode};
    use crate::evaluator::{BtMetric, Evaluator, ScalarEvaluator, VectorBtEvaluator};
    use crate::metric::{Direction, Metric};
    use crate::proposer::LocalProposer;
    use crate::scenario::MirofishScenarioSource;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use std::sync::atomic::Ordering;

    const AR_VERSION: &str = "1.0.0";

    /// Drive a VectorBT-evaluator autoresearch run end-to-end and return a result dict. The
    /// scenario ensemble is generated natively from a MiroFish rollout source (start/drift/vol),
    /// the candidate `target` is a strategy-parameter vector, and the loop ratchets it to
    /// optimize the chosen backtest metric. This is the single chokepoint the Python facade +
    /// dashboard call.
    #[pyclass(name = "AutoResearchNative", module = "vision_autoresearch")]
    pub struct AutoResearchNative {}

    #[pymethods]
    impl AutoResearchNative {
        #[new]
        fn new() -> Self {
            AutoResearchNative {}
        }

        fn name(&self) -> &'static str {
            "AutoResearchNative"
        }
        fn version(&self) -> &'static str {
            AR_VERSION
        }
        fn iterations(&self) -> u64 {
            crate::engine::AR_ITERATIONS.load(Ordering::Relaxed)
        }

        /// run_backtest(target, start, drift, vol, metric="sharpe", n_scenarios=200,
        /// horizon=12, iterations=100, step=0.2, seed=42) -> result dict.
        ///
        /// Generalized autoresearch over a MiroFish scenario ensemble + a vectorized backtest
        /// evaluator. Returns {goal, metric, direction, iterations, n_kept, best_value,
        /// best_target, git_history, attempts:[...]}. The keep-or-revert ratchet runs natively;
        /// the GIL is released for the parallel sweep/evaluation.
        #[allow(clippy::too_many_arguments)]
        #[pyo3(signature = (target, start=100.0, drift=0.001, vol=0.02, metric="sharpe",
            n_scenarios=200, horizon=12, iterations=100, step=0.2, seed=42))]
        fn run_backtest<'py>(
            &self,
            py: Python<'py>,
            target: Vec<f64>,
            start: f64,
            drift: f64,
            vol: f64,
            metric: &str,
            n_scenarios: usize,
            horizon: usize,
            iterations: usize,
            step: f64,
            seed: u64,
        ) -> PyResult<Bound<'py, PyDict>> {
            let bt = BtMetric::from_str(metric);
            // Build the frozen evaluator (scenario ensemble baked in) off the GIL.
            let (run_results, n_kept, baseline, target_out, git_hist, metric_name, direction): (
                Vec<crate::engine::IterationResult>,
                usize,
                f64,
                Vec<f64>,
                Vec<String>,
                String,
                &'static str,
            ) = py.allow_threads(move || {
                let src = MirofishScenarioSource::new(start, drift, vol);
                let ev = vectorbt_evaluator_from_source(&src, n_scenarios, horizon, seed, bt);
                let dir = ev.metric().direction.as_str();
                let mname = ev.metric().name.clone();
                let prop = LocalProposer::new(step, seed).with_bounds(-3.0, 3.0);
                let mut run = Run::new(target, ev, prop, NoopRatchet::new(), AttemptLog::new())
                    .with_goal(format!("autoresearch backtest: optimize {}", mname));
                let results = run.run_loop(StopCondition::MaxIterations(iterations.max(1)));
                (
                    results,
                    run.n_kept(),
                    run.baseline_value(),
                    run.target.clone(),
                    run.git_history(),
                    mname,
                    dir,
                )
            });

            let d = PyDict::new_bound(py);
            d.set_item("engine", "rust")?;
            d.set_item("version", AR_VERSION)?;
            d.set_item("metric", metric_name)?;
            d.set_item("direction", direction)?;
            d.set_item("iterations", run_results.len())?;
            d.set_item("n_kept", n_kept)?;
            d.set_item("best_value", baseline)?;
            d.set_item("best_target", target_out)?;
            d.set_item("git_history", git_hist)?;

            let attempts = PyList::empty_bound(py);
            for r in &run_results {
                let row = PyDict::new_bound(py);
                row.set_item("iter", r.iter)?;
                row.set_item("commit7", &r.commit7)?;
                row.set_item("metric_value", r.metric_value)?;
                row.set_item("baseline_value", r.baseline_value)?;
                row.set_item("peak_mem_gb", r.peak_mem_gb)?;
                row.set_item("status", r.status.as_str())?;
                row.set_item("crashed", r.crashed)?;
                row.set_item("description", &r.description)?;
                attempts.append(row)?;
            }
            d.set_item("attempts", attempts)?;
            Ok(d)
        }

        /// run_scalar(target, target_vector, iterations=100, step=0.2, seed=42) -> dict.
        /// The GENERIC scalar evaluator path: minimize SSE of `target` toward `target_vector`
        /// (a frozen, un-gameable parabola). Proves the loop generalizes beyond backtests.
        #[pyo3(signature = (target, target_vector, iterations=100, step=0.2, seed=42))]
        fn run_scalar<'py>(
            &self,
            py: Python<'py>,
            target: Vec<f64>,
            target_vector: Vec<f64>,
            iterations: usize,
            step: f64,
            seed: u64,
        ) -> PyResult<Bound<'py, PyDict>> {
            let (n_iters, n_kept, best, best_target, git_hist): (usize, usize, f64, Vec<f64>, Vec<String>) =
                py.allow_threads(move || {
                    let ev = ScalarEvaluator::sse_to_target(target_vector);
                    let prop = LocalProposer::new(step, seed).with_bounds(-10.0, 10.0);
                    let mut run = Run::new(target, ev, prop, NoopRatchet::new(), AttemptLog::new())
                        .with_goal("autoresearch scalar: minimize SSE to target");
                    let results = run.run_loop(StopCondition::MaxIterations(iterations.max(1)));
                    (results.len(), run.n_kept(), run.baseline_value(), run.target.clone(), run.git_history())
                });
            let d = PyDict::new_bound(py);
            d.set_item("engine", "rust")?;
            d.set_item("metric", "sse")?;
            d.set_item("direction", "minimize")?;
            d.set_item("iterations", n_iters)?;
            d.set_item("n_kept", n_kept)?;
            d.set_item("best_value", best)?;
            d.set_item("best_target", best_target)?;
            d.set_item("git_history", git_hist)?;
            Ok(d)
        }

        /// sweep(start, drift, vol, metric, n_candidates, n_scenarios, horizon, step, seed) ->
        /// list of ranked {index, metric_value, description, artifact}. The rayon parallel
        /// scenario sweep (broad/shallow swarm), best-first.
        #[allow(clippy::too_many_arguments)]
        #[pyo3(signature = (start=100.0, drift=0.001, vol=0.02, metric="sharpe",
            n_candidates=32, n_scenarios=128, horizon=12, step=0.5, seed=42))]
        fn sweep<'py>(
            &self,
            py: Python<'py>,
            start: f64,
            drift: f64,
            vol: f64,
            metric: &str,
            n_candidates: usize,
            n_scenarios: usize,
            horizon: usize,
            step: f64,
            seed: u64,
        ) -> PyResult<Bound<'py, PyList>> {
            let bt = BtMetric::from_str(metric);
            let ranked = py.allow_threads(move || {
                let src = MirofishScenarioSource::new(start, drift, vol);
                let ev = vectorbt_evaluator_from_source(&src, n_scenarios, horizon, seed, bt);
                let prop = LocalProposer::new(step, seed).with_bounds(-3.0, 3.0);
                spawn_swarm(&ev, &prop, &vec![0.0, 0.0], n_candidates, SwarmMode::Subagent, &[])
            });
            let out = PyList::empty_bound(py);
            for s in &ranked.ranked {
                let row = PyDict::new_bound(py);
                row.set_item("index", s.index)?;
                row.set_item("metric_value", s.metric_value)?;
                row.set_item("description", &s.description)?;
                row.set_item("artifact", s.artifact.clone())?;
                out.append(row)?;
            }
            Ok(out)
        }
    }

    /// Lifetime native loop-iteration count (evidence the Rust loop is live).
    #[pyfunction]
    pub fn autoresearch_iterations() -> u64 {
        crate::engine::AR_ITERATIONS.load(Ordering::Relaxed)
    }

    /// Compute a single backtest metric over a returns series (vectorized kernel exposed for
    /// the evaluator-as-scalar-metric facade `vision_autoresearch_eval.evaluate`).
    #[pyfunction]
    #[pyo3(signature = (returns, metric="sharpe"))]
    pub fn backtest_metric(returns: Vec<f64>, metric: &str) -> f64 {
        crate::evaluator::bt_metric_of_path(BtMetric::from_str(metric), &returns)
    }

    /// Reduce a SCENARIO ENSEMBLE (paths matrix of returns) to ONE scalar by the chosen metric
    /// (cross-scenario mean) — the `vision_autoresearch_eval.evaluate(scenarios, metric)`
    /// kernel. `paths_are_prices=False` => rows are returns. Mirrors VectorBtEvaluator.score.
    #[pyfunction]
    #[pyo3(signature = (scenarios, metric="sharpe", paths_are_prices=false))]
    pub fn evaluate_scenarios(scenarios: Vec<Vec<f64>>, metric: &str, paths_are_prices: bool) -> f64 {
        let ev = VectorBtEvaluator::new(BtMetric::from_str(metric), scenarios, paths_are_prices);
        // a neutral candidate (full long, no momentum) so the scalar reflects the ensemble.
        ev.score(&vec![1.0, 0.0])
    }

    /// Helper to expose direction parsing to the facade.
    #[pyfunction]
    pub fn metric_direction(metric: &str) -> String {
        let _ = Metric::new(metric, Direction::Minimize); // touch to keep import live
        BtMetric::from_str(metric).direction().as_str().to_string()
    }

    /// The PyO3 module — `import vision_autoresearch`.
    #[pymodule]
    fn vision_autoresearch(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<AutoResearchNative>()?;
        m.add_function(wrap_pyfunction!(autoresearch_iterations, m)?)?;
        m.add_function(wrap_pyfunction!(backtest_metric, m)?)?;
        m.add_function(wrap_pyfunction!(evaluate_scenarios, m)?)?;
        m.add_function(wrap_pyfunction!(metric_direction, m)?)?;
        Ok(())
    }
}
