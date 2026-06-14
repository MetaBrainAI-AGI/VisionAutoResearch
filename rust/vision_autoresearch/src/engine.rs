//! engine.rs — the AutoResearch engine: the keep-or-revert LOOP + the rayon SCENARIO SWEEP.
//!
//! This is the generalized port of Karpathy's loop (KB autoresearch.md "THE EXACT LOOP"):
//!   loop:
//!     1-3. read state + history, propose ONE candidate (the mutation operator)
//!     4-5. apply the edit to the artifact, git commit (candidate)
//!     6.   run the FROZEN evaluator (5-min budget there; here: a deterministic scorer call)
//!     7.   on crash: log failure, revert, retry (failures are data)
//!     8.   append a row to the attempt log
//!     9.   DECIDE: strictly improved -> KEEP commit (new baseline); else git reset (revert)
//!
//! The engine OWNS the evaluator; the proposer does not — that is the anti-gaming isolation.
//!
//! Two execution modes:
//!   * `run_iteration` / `run_loop` — the sequential linear ratchet (one artifact, advanced).
//!   * `sweep` — a RAYON-PARALLEL scenario/variation sweep: score MANY candidates concurrently
//!     against the frozen evaluator, then pick the best (broad-and-shallow exploration). This
//!     is the multi-agent "subagent" scale-out (KB §multi-agent) reduced to a parallel map.

use crate::attempt_log::{make_row, AttemptLog, AttemptRow, Ratchet};
use crate::evaluator::{Artifact, Evaluator};
use crate::metric::{Direction, KeepOrRevert, Metric};
use crate::proposer::{Candidate, Proposer};
use crate::scenario::ScenarioSource;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Lifetime engine-iteration counter (evidence the native loop ran).
pub static AR_ITERATIONS: AtomicU64 = AtomicU64::new(0);

/// When to stop the overnight loop.
#[derive(Debug, Clone, Copy)]
pub enum StopCondition {
    /// Stop after N iterations.
    MaxIterations(usize),
    /// Stop once `n` consecutive iterations produce no improvement (the ratchet stalled).
    NoImprovementFor(usize),
    /// Run exactly once.
    Once,
}

/// The result of ONE loop iteration (KB reuse: IterationResult).
#[derive(Debug, Clone)]
pub struct IterationResult {
    pub iter: usize,
    pub commit7: String,
    pub metric_value: f64,
    pub baseline_value: f64,
    pub peak_mem_gb: f64,
    pub status: KeepOrRevert,
    pub crashed: bool,
    pub description: String,
}

impl IterationResult {
    pub fn kept(&self) -> bool {
        !self.crashed && self.status.is_keep()
    }
}

/// The "Run" definition (the prompt's `Run{target, evaluator, metric, goal, conditions,
/// strategies/variations}`). `target` is the editable artifact (baseline); the evaluator +
/// proposer + ratchet are the moving parts; `goal`/`conditions` are human-readable context
/// (the program.md analogue) carried for the log/dashboard.
pub struct Run<E: Evaluator, P: Proposer, R: Ratchet> {
    /// The current baseline artifact (advances on every KEEP).
    pub target: Artifact,
    /// The FROZEN evaluator (engine-owned; proposer never sees it).
    pub evaluator: E,
    /// The mutation operator.
    pub proposer: P,
    /// The git (or noop) ratchet.
    pub ratchet: R,
    /// The attempt log (results.tsv analogue).
    pub log: AttemptLog,
    /// Human-readable research goal (program.md analogue).
    pub goal: String,
    /// Human-readable constraints/conditions.
    pub conditions: Vec<String>,
    /// The best metric value seen so far (the live baseline value). Seeded to "worst".
    baseline_value: f64,
}

impl<E: Evaluator, P: Proposer, R: Ratchet> Run<E, P, R> {
    pub fn new(target: Artifact, evaluator: E, proposer: P, ratchet: R, log: AttemptLog) -> Self {
        let worst = evaluator.metric().worst();
        Run {
            target,
            evaluator,
            proposer,
            ratchet,
            log,
            goal: String::new(),
            conditions: Vec::new(),
            baseline_value: worst,
        }
    }

    pub fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goal = goal.into();
        self
    }
    pub fn with_conditions(mut self, conditions: Vec<String>) -> Self {
        self.conditions = conditions;
        self
    }

    pub fn metric(&self) -> &Metric {
        self.evaluator.metric()
    }

    pub fn baseline_value(&self) -> f64 {
        self.baseline_value
    }

    /// Score the current baseline once and lock it as the starting value (so the first
    /// candidate is compared against a REAL baseline, not just "worst"). Called by run_loop
    /// before the first iteration when the baseline is non-empty.
    pub fn establish_baseline(&mut self) {
        if !self.target.is_empty() {
            let v = self.evaluator.score(&self.target);
            if v.is_finite() {
                self.baseline_value = v;
            }
        }
    }

    /// ONE propose -> apply -> commit -> evaluate(frozen) -> keep-or-revert -> log cycle.
    /// This is the verbatim autoresearch step. `safe_score` catches a panicking evaluator as
    /// a CRASH (status="crash", revert, "failures are data") rather than aborting the loop.
    pub fn run_iteration(&mut self, iter: usize) -> IterationResult {
        AR_ITERATIONS.fetch_add(1, Ordering::Relaxed);

        // 1-3. propose ONE candidate from baseline + recent history (the mutation operator).
        let recent = self.log.recent(16).to_vec();
        let candidate: Candidate = self.proposer.propose(&self.target, &recent, iter);

        // 4. apply the edit (the candidate IS the new artifact); 5. git commit (candidate).
        let commit7 = self
            .ratchet
            .commit_candidate(&format!("autoresearch[{}]: {}", iter, candidate.description));

        // 6. run the FROZEN evaluator, catching a panic as a crash.
        let (new_value, crashed) = safe_score(&self.evaluator, &candidate.artifact);

        // 9. DECIDE: strict-improvement gate, direction-aware. A crash never keeps.
        let decision = if crashed {
            KeepOrRevert::Revert
        } else {
            self.metric().decide(self.baseline_value, new_value)
        };

        // apply the ratchet (keep advances branch; revert resets HEAD~1).
        let _ = self.ratchet.apply(decision);

        // on KEEP: advance baseline artifact + baseline value.
        if !crashed && decision.is_keep() {
            self.target = candidate.artifact.clone();
            self.baseline_value = new_value;
        }

        // 8. append the attempt row (results.tsv analogue).
        let row: AttemptRow = make_row(
            iter,
            commit7.clone(),
            self.metric().name.as_str(),
            new_value,
            self.baseline_value,
            est_peak_mem_gb(&candidate.artifact),
            Some(decision),
            crashed,
            candidate.description.clone(),
        );
        self.log.append(row);

        IterationResult {
            iter,
            commit7,
            metric_value: new_value,
            baseline_value: self.baseline_value,
            peak_mem_gb: est_peak_mem_gb(&candidate.artifact),
            status: decision,
            crashed,
            description: candidate.description,
        }
    }

    /// The overnight loop: iterate until `stop` fires. Returns every IterationResult.
    pub fn run_loop(&mut self, stop: StopCondition) -> Vec<IterationResult> {
        self.establish_baseline();
        let mut results = Vec::new();
        let mut no_improve = 0usize;
        let mut i = 0usize;
        loop {
            let r = self.run_iteration(i);
            let kept = r.kept();
            results.push(r);
            if kept {
                no_improve = 0;
            } else {
                no_improve += 1;
            }
            i += 1;
            let should_stop = match stop {
                StopCondition::Once => true,
                StopCondition::MaxIterations(n) => i >= n,
                StopCondition::NoImprovementFor(n) => no_improve >= n,
            };
            if should_stop {
                break;
            }
        }
        results
    }

    /// The validated-improvement git history (the ratchet output = kept commits only).
    pub fn git_history(&self) -> Vec<String> {
        self.ratchet.kept_history()
    }

    pub fn n_kept(&self) -> usize {
        self.log.n_kept()
    }
}

/// Score an artifact, catching an evaluator panic as a CRASH (NaN, crashed=true). Mirrors
/// "on crash: log failure, retry — failures are data".
fn safe_score<E: Evaluator>(evaluator: &E, artifact: &Artifact) -> (f64, bool) {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| evaluator.score(artifact)));
    match res {
        Ok(v) if v.is_finite() => (v, false),
        Ok(v) => (v, true), // NaN/inf -> treated as a crash (can't win)
        Err(_) => (f64::NAN, true),
    }
}

/// A coarse, deterministic "peak memory" proxy for the attempt log (the autoresearch row
/// carries peak_mem_GB). Real runs override this from the evaluator harness; here it scales
/// with artifact size so the column is populated + reproducible.
fn est_peak_mem_gb(artifact: &Artifact) -> f64 {
    // 8 bytes/knob, +1 GB base; expressed in GB to 3 dp.
    let bytes = artifact.len() as f64 * 8.0;
    ((1.0 + bytes / 1e9) * 1000.0).round() / 1000.0
}

// ───────────────────────── RAYON SCENARIO SWEEP ─────────────────────────

/// One scored candidate from a parallel sweep.
#[derive(Debug, Clone)]
pub struct SweepResult {
    pub index: usize,
    pub artifact: Artifact,
    pub metric_value: f64,
    pub description: String,
}

/// Run a SWEEP: score `candidates` against the frozen `evaluator` IN PARALLEL (rayon), then
/// return them ranked best-first by the metric direction. This is the broad/shallow
/// "subagent swarm" scale-out — every variation/scenario evaluated concurrently. The winner
/// is `ranked[0]` (the best candidate to promote into the linear ratchet as the new target).
pub fn sweep<E: Evaluator>(evaluator: &E, candidates: &[Candidate]) -> Vec<SweepResult> {
    let direction = evaluator.metric().direction;
    let mut scored: Vec<SweepResult> = candidates
        .par_iter()
        .enumerate()
        .map(|(i, c)| {
            let (v, _crashed) = safe_score(evaluator, &c.artifact);
            SweepResult {
                index: i,
                artifact: c.artifact.clone(),
                metric_value: v,
                description: c.description.clone(),
            }
        })
        .collect();
    // sort best-first; NaN to the back.
    scored.sort_by(|a, b| rank_cmp(direction, a.metric_value, b.metric_value));
    scored
}

/// Generate `n` candidate variations via the proposer and sweep them in parallel against the
/// frozen evaluator — the one-call "scenario sweep" the prompt asks for (run every
/// variation/scenario concurrently). Returns ranked results, best first.
pub fn scenario_sweep<E: Evaluator, P: Proposer>(
    evaluator: &E,
    proposer: &P,
    baseline: &Artifact,
    n: usize,
    history: &[AttemptRow],
) -> Vec<SweepResult> {
    let candidates: Vec<Candidate> = (0..n)
        .map(|k| proposer.propose(baseline, history, k))
        .collect();
    sweep(evaluator, &candidates)
}

/// Direction-aware comparator for ranking (best first). NaN sorts last.
fn rank_cmp(direction: Direction, a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Equal,
        (true, false) => Greater, // a worse
        (false, true) => Less,    // a better
        (false, false) => match direction {
            Direction::Minimize => a.partial_cmp(&b).unwrap_or(Equal),
            Direction::Maximize => b.partial_cmp(&a).unwrap_or(Equal),
        },
    }
}

/// Swarm scale-out mode (KB §multi-agent): SUBAGENT = broad/shallow parallel proposers (the
/// `sweep`); AGENT_TEAM = fewer, coupled candidates (a smaller, diverse sweep). Both reduce to
/// a parallel scored map here; the difference is only how many/how diverse the candidates are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwarmMode {
    Subagent,
    AgentTeam,
}

/// The result of a swarm sweep: the ranked candidates + which mode produced them.
#[derive(Debug, Clone)]
pub struct SwarmResult {
    pub mode_subagent: bool,
    pub ranked: Vec<SweepResult>,
}

/// Spawn a parallel swarm of `n` proposals (Subagent = n broad; AgentTeam = min(n,4) diverse)
/// and return the ranked results. Use this when a single linear ratchet stalls
/// (StopCondition::NoImprovementFor) to explore broadly before resuming the ratchet.
pub fn spawn_swarm<E: Evaluator, P: Proposer>(
    evaluator: &E,
    proposer: &P,
    baseline: &Artifact,
    n: usize,
    mode: SwarmMode,
    history: &[AttemptRow],
) -> SwarmResult {
    let count = match mode {
        SwarmMode::Subagent => n.max(1),
        SwarmMode::AgentTeam => n.min(4).max(1), // fewer, coupled specialists
    };
    let ranked = scenario_sweep(evaluator, proposer, baseline, count, history);
    SwarmResult {
        mode_subagent: mode == SwarmMode::Subagent,
        ranked,
    }
}

/// Build a VectorBT evaluator straight off a ScenarioSource — the "Mirofish rollouts ->
/// vectorized backtest -> scalar" wiring (KB SEAMS §1B + §5) in one call.
pub fn vectorbt_evaluator_from_source<S: ScenarioSource>(
    source: &S,
    n_scenarios: usize,
    horizon: usize,
    seed: u64,
    bt: crate::evaluator::BtMetric,
) -> crate::evaluator::VectorBtEvaluator {
    let paths = source.rollout(n_scenarios, horizon, seed);
    crate::evaluator::VectorBtEvaluator::new(bt, paths, source.paths_are_prices())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempt_log::NoopRatchet;
    use crate::evaluator::ScalarEvaluator;
    use crate::proposer::{LocalProposer, VariationProposer};

    #[test]
    fn loop_ratchets_toward_target_minimize() {
        // SSE-to-target evaluator: minimize. The loop should monotonically improve (or hold)
        // the baseline and end strictly better than the starting baseline.
        let target = vec![1.0, -2.0, 3.0];
        let ev = ScalarEvaluator::sse_to_target(target.clone());
        let prop = LocalProposer::new(0.4, 2026).with_bounds(-5.0, 5.0);
        let mut run = Run::new(
            vec![0.0, 0.0, 0.0],
            ev,
            prop,
            NoopRatchet::new(),
            AttemptLog::new(),
        )
        .with_goal("minimize SSE to target");

        run.establish_baseline();
        let start = run.baseline_value();
        let results = run.run_loop(StopCondition::MaxIterations(200));
        assert_eq!(results.len(), 200);
        // baseline must have strictly improved (lower SSE) and never regressed past start.
        assert!(run.baseline_value() < start, "loop did not improve: {} -> {}", start, run.baseline_value());
        // at least one keep happened.
        assert!(run.n_kept() >= 1);
        // git history = kept commits only.
        assert_eq!(run.git_history().len(), run.n_kept());
    }

    #[test]
    fn keep_advances_revert_holds_baseline() {
        // VariationProposer feeding a known-better then a known-worse candidate.
        let ev = ScalarEvaluator::sse_to_target(vec![0.0]); // minimize x^2 (target 0)
        let prop = VariationProposer::new(vec![vec![0.5], vec![3.0], vec![0.1]]);
        let mut run = Run::new(vec![1.0], ev, prop, NoopRatchet::new(), AttemptLog::new());
        run.establish_baseline(); // baseline = 1.0^2 = 1.0

        // iter 0 -> 0.5 (sse 0.25 < 1.0) KEEP, baseline -> 0.25, target -> [0.5]
        let r0 = run.run_iteration(0);
        assert!(r0.kept());
        assert!((run.baseline_value() - 0.25).abs() < 1e-9);

        // iter 1 -> 3.0 (sse 9.0 > 0.25) REVERT, baseline + target unchanged
        let r1 = run.run_iteration(1);
        assert!(!r1.kept());
        assert!((run.baseline_value() - 0.25).abs() < 1e-9);
        assert_eq!(run.target, vec![0.5]);

        // iter 2 -> 0.1 (sse 0.01 < 0.25) KEEP, baseline -> 0.01
        let r2 = run.run_iteration(2);
        assert!(r2.kept());
        assert!((run.baseline_value() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn crash_is_logged_and_reverted_not_fatal() {
        // an evaluator that panics on a specific artifact -> caught as crash, loop continues.
        let ev = ScalarEvaluator::new(
            "panicky",
            Metric::val_bpb(),
            |a: &Artifact| {
                if a.first().copied().unwrap_or(0.0) > 100.0 {
                    panic!("boom");
                }
                a.first().copied().unwrap_or(0.0).abs()
            },
        );
        let prop = VariationProposer::new(vec![vec![999.0], vec![0.1]]);
        let mut run = Run::new(vec![1.0], ev, prop, NoopRatchet::new(), AttemptLog::new());
        run.establish_baseline();
        let r0 = run.run_iteration(0); // 999 -> panic -> crash
        assert!(r0.crashed);
        assert!(!r0.kept());
        // the loop survived; next iter scores fine.
        let r1 = run.run_iteration(1); // 0.1 -> keep
        assert!(!r1.crashed);
        assert!(r1.kept());
        // attempt log recorded a crash row.
        assert!(run.log.all().iter().any(|r| r.status == "crash"));
    }

    #[test]
    fn no_improvement_stop_fires_when_stalled() {
        // proposer that only ever proposes a WORSE candidate -> never keeps -> stall stop.
        let ev = ScalarEvaluator::sse_to_target(vec![0.0]);
        let prop = VariationProposer::new(vec![vec![5.0]]); // always sse 25 > baseline 0
        let mut run = Run::new(vec![0.0], ev, prop, NoopRatchet::new(), AttemptLog::new());
        let results = run.run_loop(StopCondition::NoImprovementFor(3));
        // baseline 0 is already optimal -> first 3 (worse) attempts stall -> stop at 3.
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| !r.kept()));
    }

    #[test]
    fn sweep_ranks_best_first_in_parallel() {
        let ev = ScalarEvaluator::sse_to_target(vec![0.0]); // minimize x^2
        let cands = vec![
            crate::proposer::Candidate { artifact: vec![5.0], description: "far".into() },
            crate::proposer::Candidate { artifact: vec![0.1], description: "near".into() },
            crate::proposer::Candidate { artifact: vec![2.0], description: "mid".into() },
        ];
        let ranked = sweep(&ev, &cands);
        assert_eq!(ranked.len(), 3);
        // best = smallest sse = the 0.1 candidate
        assert_eq!(ranked[0].description, "near");
        assert!(ranked[0].metric_value < ranked[1].metric_value);
    }

    #[test]
    fn swarm_subagent_is_broad_agentteam_is_capped() {
        let ev = ScalarEvaluator::sse_to_target(vec![0.0]);
        let prop = LocalProposer::new(0.5, 11);
        let sub = spawn_swarm(&ev, &prop, &vec![1.0], 10, SwarmMode::Subagent, &[]);
        let team = spawn_swarm(&ev, &prop, &vec![1.0], 10, SwarmMode::AgentTeam, &[]);
        assert_eq!(sub.ranked.len(), 10); // broad
        assert_eq!(team.ranked.len(), 4); // capped, coupled
        assert!(sub.mode_subagent);
        assert!(!team.mode_subagent);
    }

    #[test]
    fn vectorbt_evaluator_from_mirofish_source_scores() {
        use crate::evaluator::BtMetric;
        use crate::scenario::MirofishScenarioSource;
        let src = MirofishScenarioSource::new(100.0, 0.001, 0.02);
        let ev = vectorbt_evaluator_from_source(&src, 64, 12, 42, BtMetric::Sharpe);
        let a = ev.score(&vec![1.0, 0.0]);
        let b = ev.score(&vec![-1.0, 0.5]);
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() > 1e-9, "evaluator not responsive over mirofish scenarios");
    }
}
