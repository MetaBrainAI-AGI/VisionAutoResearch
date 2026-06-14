//! evaluator.rs — the FROZEN, un-gameable Evaluator trait + two concrete evaluators.
//!
//! ## Anti-gaming guarantee (the load-bearing autoresearch invariant)
//! In Karpathy's loop, `train.py` *imports* `evaluate_bpb()` from the FROZEN `prepare.py`
//! and the agent has NO code path into the scorer — so the metric cannot be gamed (KB
//! autoresearch.md, "EVALUATOR ISOLATION"). We mirror that here: the `Evaluator` trait is
//! the scorer, the engine OWNS it, and the proposer/mutation operator can ONLY edit the
//! `Artifact` (the candidate vector). A proposer never holds an `&dyn Evaluator`, so it has
//! no path to influence the score. A problem is autoresearchable IFF: (1) ONE bounded
//! editable artifact, (2) a FROZEN evaluator that cannot be gamed, (3) a SCALAR metric.
//!
//! Two concrete evaluators are provided, per the task contract:
//!   * `ScalarEvaluator`  — a generic frozen scalar fn (any pluggable fitness closure).
//!   * `VectorBtEvaluator`— a vectorized-backtest evaluator: turns a Mirofish SCENARIO
//!     ensemble (paths matrix) into one scalar via a metric registry
//!     {sharpe, sortino, total_return, max_drawdown, calmar, win_rate, cvar}.
//!
//! VectorBT itself is NOT in the repo (KB SEAMS §5) so the math is authored fresh, Rust-
//! native (rayon over paths), mirroring VisionRustify's RUST_KERNEL pattern. A vectorbt
//! Python facade is an OPTIONAL EXEMPT add-on at the .py layer — the Rust kernel is default.

use crate::metric::Metric;
use rayon::prelude::*;

/// The bounded, agent-EDITABLE artifact. In Karpathy's loop this is `train.py`; here it is a
/// vector of f64 "knobs" (hyperparameters / strategy params / weights). The single-artifact
/// constraint is what prevents the proposer from "refactoring the universe".
pub type Artifact = Vec<f64>;

/// A frozen scorer. `score()` is the ONLY way to turn an artifact into the scalar metric, and
/// it is owned by the engine — the proposer never sees it. Implementors MUST be deterministic
/// for a given artifact (so keep-or-revert is reproducible) and side-effect-free on the repo.
pub trait Evaluator: Send + Sync {
    /// Score one candidate artifact -> the scalar metric value (NaN signals a crashed/invalid
    /// evaluation, which the keep-or-revert gate treats as "never wins").
    fn score(&self, artifact: &Artifact) -> f64;

    /// The metric this evaluator produces (name + direction). The engine reads `direction`
    /// to decide keep-vs-revert; the proposer never reads it.
    fn metric(&self) -> &Metric;

    /// Human label for the attempt log.
    fn name(&self) -> &str {
        "evaluator"
    }
}

// ───────────────────────── ScalarEvaluator (generic) ─────────────────────────

/// A generic scalar evaluator backed by a frozen closure. This is the "any metric you care
/// about that is reasonably efficient to evaluate" general case (KB generalization contract).
pub struct ScalarEvaluator {
    metric: Metric,
    name: String,
    f: Box<dyn Fn(&Artifact) -> f64 + Send + Sync>,
}

impl ScalarEvaluator {
    pub fn new(
        name: impl Into<String>,
        metric: Metric,
        f: impl Fn(&Artifact) -> f64 + Send + Sync + 'static,
    ) -> Self {
        ScalarEvaluator {
            metric,
            name: name.into(),
            f: Box::new(f),
        }
    }

    /// A common default: minimize the (negative) sum of squared error toward a fixed target
    /// vector — a frozen, un-gameable parabola the loop can ratchet toward.
    pub fn sse_to_target(target: Artifact) -> Self {
        ScalarEvaluator::new("sse_to_target", Metric::new("sse", crate::metric::Direction::Minimize), move |a| {
            let n = a.len().min(target.len());
            let mut acc = 0.0f64;
            for i in 0..n {
                let d = a[i] - target[i];
                acc += d * d;
            }
            // length mismatch is penalized so the artifact can't shrink to win
            acc += (a.len() as f64 - target.len() as f64).abs();
            acc
        })
    }
}

impl Evaluator for ScalarEvaluator {
    fn score(&self, artifact: &Artifact) -> f64 {
        (self.f)(artifact)
    }
    fn metric(&self) -> &Metric {
        &self.metric
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ───────────────────────── VectorBtEvaluator (backtest) ─────────────────────────

/// The risk/return metric registry the VectorBT-style evaluator supports. Each is a pure,
/// vectorized fn over a returns series; the evaluator reduces a SCENARIO ENSEMBLE (many
/// paths) to ONE scalar so keep-or-revert is binary (KB SEAMS §5 metric registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtMetric {
    Sharpe,
    Sortino,
    TotalReturn,
    MaxDrawdown,
    Calmar,
    WinRate,
    Cvar,
}

impl BtMetric {
    pub fn from_str(s: &str) -> BtMetric {
        match s.trim().to_lowercase().as_str() {
            "sortino" => BtMetric::Sortino,
            "total_return" | "totalreturn" | "return" => BtMetric::TotalReturn,
            "max_drawdown" | "maxdrawdown" | "drawdown" | "mdd" => BtMetric::MaxDrawdown,
            "calmar" => BtMetric::Calmar,
            "win_rate" | "winrate" => BtMetric::WinRate,
            "cvar" | "es" => BtMetric::Cvar,
            _ => BtMetric::Sharpe,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            BtMetric::Sharpe => "sharpe",
            BtMetric::Sortino => "sortino",
            BtMetric::TotalReturn => "total_return",
            BtMetric::MaxDrawdown => "max_drawdown",
            BtMetric::Calmar => "calmar",
            BtMetric::WinRate => "win_rate",
            BtMetric::Cvar => "cvar",
        }
    }
    /// Direction the metric should be optimized in.
    pub fn direction(&self) -> crate::metric::Direction {
        use crate::metric::Direction::*;
        match self {
            // max_drawdown & cvar are losses -> we report them as magnitudes to MINIMIZE.
            BtMetric::MaxDrawdown | BtMetric::Cvar => Minimize,
            _ => Maximize,
        }
    }
}

// ── pure vectorized metric kernels over a returns series ──

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64], m: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0);
    var.sqrt()
}

/// Sharpe of a returns series (per-step, rf=0). 0 when vol is ~0.
pub fn sharpe(returns: &[f64]) -> f64 {
    let m = mean(returns);
    let sd = std_dev(returns, m);
    if sd <= 1e-12 {
        0.0
    } else {
        m / sd
    }
}

/// Sortino — like Sharpe but only DOWNSIDE deviation in the denominator.
pub fn sortino(returns: &[f64]) -> f64 {
    let m = mean(returns);
    let downside: Vec<f64> = returns.iter().filter(|r| **r < 0.0).cloned().collect();
    if downside.is_empty() {
        // no downside -> strongly favorable; cap to avoid +inf
        return if m > 0.0 { 1e6 } else { 0.0 };
    }
    let dd = (downside.iter().map(|r| r * r).sum::<f64>() / downside.len() as f64).sqrt();
    if dd <= 1e-12 {
        0.0
    } else {
        m / dd
    }
}

/// Total compounded return over the path (product of (1+r) - 1).
pub fn total_return(returns: &[f64]) -> f64 {
    let mut eq = 1.0f64;
    for r in returns {
        eq *= 1.0 + r;
    }
    eq - 1.0
}

/// Maximum drawdown MAGNITUDE (>=0) of the equity curve. Lower is better.
pub fn max_drawdown(returns: &[f64]) -> f64 {
    let mut eq = 1.0f64;
    let mut peak = 1.0f64;
    let mut mdd = 0.0f64;
    for r in returns {
        eq *= 1.0 + r;
        if eq > peak {
            peak = eq;
        }
        let dd = (peak - eq) / peak;
        if dd > mdd {
            mdd = dd;
        }
    }
    mdd
}

/// Calmar = total_return / max_drawdown. Higher is better.
pub fn calmar(returns: &[f64]) -> f64 {
    let mdd = max_drawdown(returns);
    if mdd <= 1e-12 {
        let tr = total_return(returns);
        return if tr > 0.0 { 1e6 } else { 0.0 };
    }
    total_return(returns) / mdd
}

/// Fraction of steps with a positive return.
pub fn win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let wins = returns.iter().filter(|r| **r > 0.0).count();
    wins as f64 / returns.len() as f64
}

/// CVaR (Expected Shortfall) at 5% — mean of the worst 5% of returns, reported as a positive
/// MAGNITUDE of loss. Lower is better.
pub fn cvar(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = ((sorted.len() as f64 * 0.05).ceil() as usize).max(1);
    let tail: f64 = sorted[..k.min(sorted.len())].iter().sum::<f64>() / k as f64;
    // tail is the avg of the worst returns (negative) -> report magnitude of the loss
    (-tail).max(0.0)
}

/// Compute one BtMetric over a single returns series.
pub fn bt_metric_of_path(m: BtMetric, returns: &[f64]) -> f64 {
    match m {
        BtMetric::Sharpe => sharpe(returns),
        BtMetric::Sortino => sortino(returns),
        BtMetric::TotalReturn => total_return(returns),
        BtMetric::MaxDrawdown => max_drawdown(returns),
        BtMetric::Calmar => calmar(returns),
        BtMetric::WinRate => win_rate(returns),
        BtMetric::Cvar => cvar(returns),
    }
}

/// A vectorized-backtest evaluator. The candidate `Artifact` is a STRATEGY-PARAMETER vector
/// (e.g. weights / thresholds). `scenarios` is the Mirofish ensemble: a paths matrix where
/// each inner vec is one per-step PRICE-or-RETURN path. The evaluator:
///   1. converts the candidate params + each scenario path into a strategy returns series,
///   2. computes the chosen BtMetric per path (rayon-parallel over paths),
///   3. aggregates the per-path metrics to ONE scalar (the fitness used to rank/select).
/// The aggregation is the cross-scenario MEAN (robust central fitness across the ensemble).
pub struct VectorBtEvaluator {
    metric: Metric,
    bt: BtMetric,
    /// Pre-bound scenario ensemble (paths). Frozen at construction — the agent edits the
    /// candidate params, never these paths (the data is part of the un-gameable evaluator).
    scenarios: Vec<Vec<f64>>,
    /// If true, scenario rows are treated as PRICE levels and converted to step returns;
    /// if false, they are already returns.
    paths_are_prices: bool,
}

impl VectorBtEvaluator {
    pub fn new(bt: BtMetric, scenarios: Vec<Vec<f64>>, paths_are_prices: bool) -> Self {
        VectorBtEvaluator {
            metric: Metric::new(bt.name(), bt.direction()),
            bt,
            scenarios,
            paths_are_prices,
        }
    }

    /// Map a candidate param vector over one price/return path -> a strategy returns series.
    /// Strategy model (simple, frozen, vectorizable): a per-step target EXPOSURE in [-1,1]
    /// derived from the candidate's first param as a constant tilt plus a momentum overlay
    /// scaled by the second param. exposure_t * market_return_t = strategy return_t.
    /// This is deliberately simple but RESPONSIVE to the candidate params, so different
    /// proposals genuinely produce different fitness (no degenerate constant scorer).
    fn strategy_returns(&self, params: &Artifact, path: &[f64]) -> Vec<f64> {
        // market step returns
        let mkt: Vec<f64> = if self.paths_are_prices {
            path.windows(2)
                .map(|w| if w[0].abs() > 1e-12 { w[1] / w[0] - 1.0 } else { 0.0 })
                .collect()
        } else {
            path.to_vec()
        };
        let tilt = params.get(0).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let mom_scale = params.get(1).copied().unwrap_or(0.0).clamp(-2.0, 2.0);
        let mut out = Vec::with_capacity(mkt.len());
        let mut prev = 0.0f64;
        for r in &mkt {
            // exposure = constant tilt + momentum overlay (sign of last market move)
            let exposure = (tilt + mom_scale * prev.signum() * 0.5).clamp(-1.0, 1.0);
            out.push(exposure * r);
            prev = *r;
        }
        out
    }
}

impl Evaluator for VectorBtEvaluator {
    fn score(&self, artifact: &Artifact) -> f64 {
        if self.scenarios.is_empty() {
            return f64::NAN;
        }
        let bt = self.bt;
        // rayon over scenario paths — the Gate-2 parallel win for the sweep.
        let per_path: Vec<f64> = self
            .scenarios
            .par_iter()
            .map(|path| {
                let rets = self.strategy_returns(artifact, path);
                bt_metric_of_path(bt, &rets)
            })
            .collect();
        // aggregate to one scalar = cross-scenario mean (robust ensemble fitness)
        let valid: Vec<f64> = per_path.into_iter().filter(|v| v.is_finite()).collect();
        if valid.is_empty() {
            f64::NAN
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        }
    }
    fn metric(&self) -> &Metric {
        &self.metric
    }
    fn name(&self) -> &str {
        self.bt.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharpe_basic_and_zero_vol() {
        // constant positive returns -> zero vol -> sharpe defined as 0 (no risk-adjusted info)
        assert_eq!(sharpe(&[0.01, 0.01, 0.01]), 0.0);
        // a series with positive mean and some vol -> positive sharpe
        let s = sharpe(&[0.02, -0.01, 0.03, 0.01]);
        assert!(s > 0.0, "sharpe={}", s);
    }

    #[test]
    fn total_return_and_drawdown() {
        let r = vec![0.1, -0.5, 0.1];
        // (1.1)(0.5)(1.1) - 1 = 0.605 - 1 = -0.395
        assert!((total_return(&r) - (-0.395)).abs() < 1e-9);
        // peak after first step = 1.1, trough = 0.55 -> mdd = (1.1-0.55)/1.1 = 0.5
        assert!((max_drawdown(&r) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn win_rate_and_cvar() {
        assert!((win_rate(&[0.1, -0.2, 0.3, 0.0]) - 0.5).abs() < 1e-12);
        // worst 5% of 20 -> 1 element -> the min; magnitude of -0.2 = 0.2
        let mut r = vec![0.01; 20];
        r[5] = -0.2;
        assert!((cvar(&r) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn sortino_no_downside_is_favorable() {
        assert_eq!(sortino(&[0.01, 0.02, 0.03]), 1e6);
        assert_eq!(sortino(&[-0.01, -0.02]), sortino(&[-0.01, -0.02])); // deterministic
    }

    #[test]
    fn calmar_uses_return_over_drawdown() {
        let r = vec![0.2, -0.1, 0.15];
        let c = calmar(&r);
        let expect = total_return(&r) / max_drawdown(&r);
        assert!((c - expect).abs() < 1e-9);
    }

    #[test]
    fn vectorbt_evaluator_is_frozen_and_responsive() {
        // two scenario paths of step-returns
        let scen = vec![
            vec![0.02, -0.01, 0.03, 0.01, -0.02, 0.04],
            vec![0.01, 0.02, -0.03, 0.02, 0.01, -0.01],
        ];
        let ev = VectorBtEvaluator::new(BtMetric::Sharpe, scen, false);
        let a = ev.score(&vec![1.0, 0.0]); // full long tilt
        let b = ev.score(&vec![-1.0, 0.0]); // full short tilt
        assert!(a.is_finite() && b.is_finite());
        // different candidate params -> different fitness (no degenerate scorer)
        assert!((a - b).abs() > 1e-9, "evaluator not responsive: a={} b={}", a, b);
        // metric direction is Maximize for sharpe
        assert_eq!(ev.metric().direction, crate::metric::Direction::Maximize);
    }

    #[test]
    fn scalar_sse_evaluator_ratchets_toward_target() {
        let ev = ScalarEvaluator::sse_to_target(vec![1.0, 2.0, 3.0]);
        let far = ev.score(&vec![0.0, 0.0, 0.0]);
        let near = ev.score(&vec![0.9, 1.9, 3.1]);
        assert!(near < far, "closer artifact should score lower (minimize)");
        assert_eq!(ev.metric().direction, crate::metric::Direction::Minimize);
    }
}
