//! proposer.rs — the mutation operator (Karpathy's LLM, generalized to a `Proposer` trait).
//!
//! In autoresearch the LLM is BOTH the mutation operator and the selection pressure (KB: a
//! "linear ratchet, not a population EA"). The selection pressure is the keep-or-revert gate
//! (metric.rs). The MUTATION OPERATOR is a `Proposer`: given the current baseline artifact +
//! the attempt history (`results.tsv`), it emits ONE candidate edit to the artifact. The
//! single-artifact constraint is enforced structurally — a proposer only ever returns a new
//! `Artifact`, it has no `&dyn Evaluator`, so it can't game the score.
//!
//! `LocalProposer` is a frozen, reproducible perturbation strategy (so the LOOP and the SWEEP
//! are testable without an LLM). A real deployment swaps in an `LlmProposer` via the Python
//! facade (it reads program.md + results.tsv and edits the artifact), keeping the same trait.

use crate::evaluator::Artifact;

/// One proposed candidate: the mutated artifact + a human description (logged to results.tsv).
#[derive(Debug, Clone)]
pub struct Candidate {
    pub artifact: Artifact,
    pub description: String,
}

/// The mutation operator. `propose()` reads the baseline + the recent attempt history and
/// returns ONE candidate. `n` is the iteration index (used to vary the perturbation).
pub trait Proposer: Send + Sync {
    fn propose(&self, baseline: &Artifact, history: &[crate::attempt_log::AttemptRow], n: usize)
        -> Candidate;
    fn name(&self) -> &str {
        "proposer"
    }
}

// ── SplitMix64 (reproducible perturbation) ──
#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
#[inline]
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64_next(state) >> 11) as f64 / 9007199254740992.0_f64
}

/// A frozen, reproducible local-search proposer: perturbs each artifact knob by a
/// gaussian-ish step scaled by `step`, seeded by `base_seed ^ n` so each iteration explores
/// a different neighbourhood deterministically. This stands in for the LLM mutation operator
/// in tests + headless runs; the loop's keep-or-revert ratchet does the actual optimization.
pub struct LocalProposer {
    pub step: f64,
    pub base_seed: u64,
    /// Hard bounds applied to every knob (keeps proposals in a sane box).
    pub lo: f64,
    pub hi: f64,
}

impl LocalProposer {
    pub fn new(step: f64, base_seed: u64) -> Self {
        LocalProposer {
            step,
            base_seed,
            lo: -10.0,
            hi: 10.0,
        }
    }
    pub fn with_bounds(mut self, lo: f64, hi: f64) -> Self {
        self.lo = lo;
        self.hi = hi;
        self
    }
}

impl Proposer for LocalProposer {
    fn propose(
        &self,
        baseline: &Artifact,
        _history: &[crate::attempt_log::AttemptRow],
        n: usize,
    ) -> Candidate {
        let mut state = self.base_seed ^ (n as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let mut artifact = baseline.clone();
        // perturb every knob; (uniform-2)*step is a symmetric step in [-step, step]
        for v in artifact.iter_mut() {
            let delta = (next_unit(&mut state) * 2.0 - 1.0) * self.step;
            *v = (*v + delta).clamp(self.lo, self.hi);
        }
        // an empty baseline can't be optimized — seed one knob so the loop has something to move
        if artifact.is_empty() {
            artifact.push((next_unit(&mut state) * 2.0 - 1.0).clamp(self.lo, self.hi));
        }
        Candidate {
            artifact,
            description: format!("local-perturb step={:.3} iter={}", self.step, n),
        }
    }
    fn name(&self) -> &str {
        "local_proposer"
    }
}

/// A fixed-list proposer: replays a predefined set of candidate variations in order, then
/// cycles. Useful for an explicit "strategies/variations" list on the Run (the prompt's
/// `strategies/variations` field) and for deterministic SWEEP tests.
pub struct VariationProposer {
    pub variations: Vec<Artifact>,
}

impl VariationProposer {
    pub fn new(variations: Vec<Artifact>) -> Self {
        VariationProposer { variations }
    }
}

impl Proposer for VariationProposer {
    fn propose(
        &self,
        baseline: &Artifact,
        _history: &[crate::attempt_log::AttemptRow],
        n: usize,
    ) -> Candidate {
        if self.variations.is_empty() {
            return Candidate {
                artifact: baseline.clone(),
                description: "no-op (empty variation list)".into(),
            };
        }
        let idx = n % self.variations.len();
        Candidate {
            artifact: self.variations[idx].clone(),
            description: format!("variation[{}]", idx),
        }
    }
    fn name(&self) -> &str {
        "variation_proposer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_proposer_is_reproducible_and_bounded() {
        let p = LocalProposer::new(0.5, 123).with_bounds(-2.0, 2.0);
        let base = vec![0.0, 0.0, 0.0];
        let a = p.propose(&base, &[], 0);
        let b = p.propose(&base, &[], 0);
        // same (baseline, n) -> identical proposal
        assert_eq!(a.artifact, b.artifact);
        // different iteration -> (generally) different proposal
        let c = p.propose(&base, &[], 1);
        assert_ne!(a.artifact, c.artifact);
        // bounds respected
        assert!(a.artifact.iter().all(|v| *v >= -2.0 && *v <= 2.0));
    }

    #[test]
    fn local_proposer_seeds_empty_baseline() {
        let p = LocalProposer::new(1.0, 7);
        let c = p.propose(&vec![], &[], 3);
        assert!(!c.artifact.is_empty(), "must seed a knob for an empty baseline");
    }

    #[test]
    fn variation_proposer_cycles() {
        let p = VariationProposer::new(vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(p.propose(&vec![], &[], 0).artifact, vec![1.0]);
        assert_eq!(p.propose(&vec![], &[], 1).artifact, vec![2.0]);
        assert_eq!(p.propose(&vec![], &[], 2).artifact, vec![3.0]);
        assert_eq!(p.propose(&vec![], &[], 3).artifact, vec![1.0]); // cycles
    }
}
