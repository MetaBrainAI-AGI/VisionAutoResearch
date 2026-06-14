//! metric.rs — the SCALAR metric + direction-aware keep-or-revert decision.
//!
//! Karpathy's autoresearch loop turns "is this candidate better?" into a single binary
//! decision by reducing every run to ONE scalar (val_bpb, lower better) and a fixed
//! direction. We generalize: a `Metric` carries a name + a `Direction` (Minimize|Maximize),
//! and `improved(prev, new)` is the STRICT-improvement gate that drives keep-or-revert.
//!
//! Cited contract (KB autoresearch.md): "If val_bpb improved (lower), you advance the
//! branch, keeping the git commit. If val_bpb is equal or worse, you git reset back to
//! where you started." So the gate is STRICT (`equal -> revert`), and direction-aware.

use serde::{Deserialize, Serialize};

/// Optimization direction for a scalar metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Lower is better (e.g. val_bpb, max_drawdown, validation loss).
    Minimize,
    /// Higher is better (e.g. sharpe, total_return, accuracy).
    Maximize,
}

impl Direction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Direction::Minimize => "minimize",
            Direction::Maximize => "maximize",
        }
    }

    /// Parse from a facade string; defaults to Minimize on anything unknown
    /// (val_bpb / loss is the canonical autoresearch metric, lower-better).
    pub fn from_str(s: &str) -> Direction {
        match s.trim().to_lowercase().as_str() {
            "maximize" | "max" | "higher" | "up" => Direction::Maximize,
            _ => Direction::Minimize,
        }
    }
}

/// The keep-or-revert verdict for one candidate evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeepOrRevert {
    /// Strictly improved -> advance the branch, keep the commit (new baseline).
    Keep,
    /// Equal or worse -> git reset back to where we started.
    Revert,
}

impl KeepOrRevert {
    pub fn as_str(&self) -> &'static str {
        match self {
            KeepOrRevert::Keep => "keep",
            KeepOrRevert::Revert => "revert",
        }
    }
    pub fn is_keep(&self) -> bool {
        matches!(self, KeepOrRevert::Keep)
    }
}

/// A named scalar metric with an optimization direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub direction: Direction,
}

impl Metric {
    pub fn new(name: impl Into<String>, direction: Direction) -> Self {
        Metric {
            name: name.into(),
            direction,
        }
    }

    /// The canonical autoresearch metric: validation bits-per-byte, lower better.
    pub fn val_bpb() -> Self {
        Metric::new("val_bpb", Direction::Minimize)
    }

    /// STRICT improvement gate. NaN never improves (a crashed/invalid run can never win).
    /// `equal -> false` (revert) per the verbatim rule. This is the selection pressure.
    pub fn improved(&self, prev: f64, new: f64) -> bool {
        if new.is_nan() {
            return false;
        }
        if prev.is_nan() {
            // No valid baseline yet -> any valid candidate is an improvement.
            return true;
        }
        match self.direction {
            Direction::Minimize => new < prev,
            Direction::Maximize => new > prev,
        }
    }

    /// Direction-aware keep-or-revert decision over (previous baseline, new candidate).
    pub fn decide(&self, prev: f64, new: f64) -> KeepOrRevert {
        if self.improved(prev, new) {
            KeepOrRevert::Keep
        } else {
            KeepOrRevert::Revert
        }
    }

    /// True if `a` is at least as good as `b` (direction-aware) — used to rank a sweep.
    /// NaN sorts as worst.
    pub fn at_least_as_good(&self, a: f64, b: f64) -> bool {
        if a.is_nan() {
            return false;
        }
        if b.is_nan() {
            return true;
        }
        match self.direction {
            Direction::Minimize => a <= b,
            Direction::Maximize => a >= b,
        }
    }

    /// The "worst possible" sentinel value for this direction (used as the seed baseline
    /// when there is no prior result yet). +inf for minimize, -inf for maximize.
    pub fn worst(&self) -> f64 {
        match self.direction {
            Direction::Minimize => f64::INFINITY,
            Direction::Maximize => f64::NEG_INFINITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_strict_keep_on_lower_only() {
        let m = Metric::val_bpb();
        assert_eq!(m.decide(1.0, 0.9), KeepOrRevert::Keep); // lower -> keep
        assert_eq!(m.decide(1.0, 1.0), KeepOrRevert::Revert); // equal -> revert (strict)
        assert_eq!(m.decide(1.0, 1.1), KeepOrRevert::Revert); // worse -> revert
    }

    #[test]
    fn maximize_strict_keep_on_higher_only() {
        let m = Metric::new("sharpe", Direction::Maximize);
        assert_eq!(m.decide(1.0, 1.1), KeepOrRevert::Keep);
        assert_eq!(m.decide(1.0, 1.0), KeepOrRevert::Revert);
        assert_eq!(m.decide(1.0, 0.9), KeepOrRevert::Revert);
    }

    #[test]
    fn nan_candidate_never_keeps_and_no_baseline_always_keeps() {
        let m = Metric::val_bpb();
        assert_eq!(m.decide(1.0, f64::NAN), KeepOrRevert::Revert); // crashed run can't win
        // No baseline yet (prev = worst/NaN) -> a valid candidate is kept.
        assert!(m.improved(f64::NAN, 5.0));
        assert!(m.improved(m.worst(), 5.0));
    }

    #[test]
    fn worst_sentinel_is_direction_aware() {
        assert_eq!(Metric::val_bpb().worst(), f64::INFINITY);
        assert_eq!(
            Metric::new("ret", Direction::Maximize).worst(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn direction_parse_defaults_to_minimize() {
        assert_eq!(Direction::from_str("maximize"), Direction::Maximize);
        assert_eq!(Direction::from_str("MAX"), Direction::Maximize);
        assert_eq!(Direction::from_str("whatever"), Direction::Minimize);
        assert_eq!(Direction::from_str("minimize"), Direction::Minimize);
    }
}
