//! scenario.rs — the ScenarioSource trait + a native MiroFish-rollout generator.
//!
//! The evaluator needs a SCENARIO ENSEMBLE to score a candidate against. The richest seam
//! (KB SEAMS §1B) is `local_timesfm_mirofish_combo._mirofish_simulate(point, std, scenarios,
//! seed)`: N Monte-Carlo trajectories = a drift + gaussian-shock random walk anchored on a
//! forecast. The Python version only keeps the TERMINAL value per path; an AutoResearch
//! evaluator wants the FULL per-step path, so this native port RETAINS every step (the KB
//! explicitly recommends extending `_mirofish_simulate` to keep intermediate `v`).
//!
//! `ScenarioSource` is pluggable so a different generator (real market windows, a TimesFM
//! backend, a synthetic stress set) can drop in. The MiroFish source is reproducible
//! (SplitMix64, mirroring `mirofish_fwd_native.rs`) so a sweep is deterministic per seed.

use rayon::prelude::*;

/// A generator of scenario PATHS. Each path is a per-step series (prices or returns) the
/// evaluator scores. `rollout(n, horizon, seed)` returns `n` paths of length `horizon`.
pub trait ScenarioSource: Send + Sync {
    fn rollout(&self, n: usize, horizon: usize, seed: u64) -> Vec<Vec<f64>>;
    fn name(&self) -> &str {
        "scenario_source"
    }
    /// True if the emitted rows are PRICE levels (vs already-step-returns). The evaluator
    /// uses this to convert correctly.
    fn paths_are_prices(&self) -> bool {
        true
    }
}

// ── SplitMix64 (portable, reproducible) — identical constants to mirofish_fwd_native.rs ──

#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A uniform double in [0,1) from the SplitMix64 stream (top 53 bits / 2^53).
#[inline]
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64_next(state) >> 11) as f64 / 9007199254740992.0_f64
}

/// One standard-normal draw via Box-Muller (cos branch). Two uniforms -> one normal.
#[inline]
fn next_normal(state: &mut u64) -> f64 {
    let mut u1 = next_unit(state);
    if u1 < 1e-12 {
        u1 = 1e-12; // avoid ln(0)
    }
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Per-path seed derivation (mirrors mirofish_fwd_native.rs `path_seed`) so paths are
/// decorrelated AND the whole ensemble is reproducible from one base seed.
#[inline]
fn path_seed(base_seed: u64, k: usize) -> u64 {
    base_seed ^ (k as u64).wrapping_mul(0x9E3779B97F4A7C15)
}

/// MiroFish Monte-Carlo rollout source — drift + gaussian-shock random walk on PRICE levels,
/// anchored at `start` with per-step `drift` and shock std `vol`. Native port of
/// `_mirofish_simulate`, extended to retain the full per-step path (not just the terminal).
pub struct MirofishScenarioSource {
    pub start: f64,
    pub drift: f64,
    pub vol: f64,
}

impl MirofishScenarioSource {
    pub fn new(start: f64, drift: f64, vol: f64) -> Self {
        MirofishScenarioSource {
            start: start.max(1e-9),
            drift,
            vol: vol.max(0.0),
        }
    }

    /// Convenience constructor from a recent history vector (anchors start at last value and
    /// estimates drift+vol from the history step returns) — mirrors how the combo forecast
    /// seeds the simulation from `forecast_mean`/`forecast_std`.
    pub fn from_history(history: &[f64]) -> Self {
        if history.len() < 2 {
            return MirofishScenarioSource::new(history.last().copied().unwrap_or(1.0), 0.0, 0.01);
        }
        let rets: Vec<f64> = history
            .windows(2)
            .map(|w| if w[0].abs() > 1e-12 { w[1] / w[0] - 1.0 } else { 0.0 })
            .collect();
        let m = rets.iter().sum::<f64>() / rets.len() as f64;
        let var = rets.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / rets.len() as f64;
        MirofishScenarioSource::new(*history.last().unwrap(), m, var.sqrt().max(1e-6))
    }

    fn one_path(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut v = self.start;
        let mut path = Vec::with_capacity(n);
        for _ in 0..n {
            let shock = next_normal(&mut state) * self.vol;
            // multiplicative random walk: v *= (1 + drift + shock), clamped > 0
            v *= 1.0 + self.drift + shock;
            if v < 1e-9 {
                v = 1e-9;
            }
            path.push(v); // RETAIN every step (the per-step path the evaluator needs)
        }
        path
    }
}

impl ScenarioSource for MirofishScenarioSource {
    fn rollout(&self, n: usize, horizon: usize, seed: u64) -> Vec<Vec<f64>> {
        let np = n.max(1);
        let h = horizon.max(1);
        // rayon over the ensemble — each path is independent + reproducibly seeded.
        (0..np)
            .into_par_iter()
            .map(|k| self.one_path(h, path_seed(seed, k)))
            .collect()
    }
    fn name(&self) -> &str {
        "mirofish_rollout"
    }
    fn paths_are_prices(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_shape_and_reproducibility() {
        let src = MirofishScenarioSource::new(100.0, 0.001, 0.02);
        let a = src.rollout(50, 8, 42);
        let b = src.rollout(50, 8, 42);
        assert_eq!(a.len(), 50);
        assert!(a.iter().all(|p| p.len() == 8));
        // same seed -> identical ensemble
        for (pa, pb) in a.iter().zip(b.iter()) {
            for (x, y) in pa.iter().zip(pb.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn different_seed_gives_different_paths() {
        let src = MirofishScenarioSource::new(100.0, 0.0, 0.02);
        let a = src.rollout(10, 8, 1);
        let b = src.rollout(10, 8, 2);
        // at least one terminal value should differ
        let diff = a
            .iter()
            .zip(b.iter())
            .any(|(pa, pb)| (pa.last().unwrap() - pb.last().unwrap()).abs() > 1e-9);
        assert!(diff, "different seeds produced identical paths");
    }

    #[test]
    fn from_history_estimates_drift_and_vol() {
        // a gently rising history -> positive drift, small vol, no panic
        let hist = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let src = MirofishScenarioSource::from_history(&hist);
        assert!(src.drift > 0.0);
        assert!(src.vol >= 0.0);
        let paths = src.rollout(5, 4, 7);
        assert_eq!(paths.len(), 5);
        assert!(paths.iter().all(|p| p.iter().all(|v| *v > 0.0)));
    }

    #[test]
    fn prices_stay_positive() {
        // even high vol must not produce non-positive prices (clamped multiplicative walk)
        let src = MirofishScenarioSource::new(50.0, -0.05, 0.5);
        let paths = src.rollout(20, 30, 99);
        assert!(paths.iter().all(|p| p.iter().all(|v| *v > 0.0)));
    }
}
