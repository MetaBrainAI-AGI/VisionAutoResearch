//! keep_or_revert — the VisionAutoResearch loop in 30 seconds.
//!
//! A faithful, runnable demo of Andrej Karpathy's keep-or-revert research loop
//! (the "automate research as an optimization over one editable artifact against a
//! frozen evaluator" idea), ported to Rust with a frozen, un-gameable scalar metric.
//!
//! The loop PROPOSES a change to the editable artifact, SCORES it against a FROZEN
//! evaluator, and KEEPS the change only if the scalar metric improves — otherwise it
//! REVERTS to the prior baseline. Over many iterations the baseline ratchets
//! monotonically toward the target the proposer never sees.
//!
//!     cargo run --release --example keep_or_revert
//!
//! Inventor credit: Andrej Karpathy (https://github.com/karpathy). This Rust port
//! adds rayon parallel scenario sweeps, a MiroFish scenario lab, and VectorBT-style
//! backtest metrics (see `scenario_sweep`, `spawn_swarm`, `VectorBtEvaluator`).

use vision_autoresearch::{
    AttemptLog, LocalProposer, NoopRatchet, Run, ScalarEvaluator, StopCondition,
};

fn main() {
    // The FROZEN evaluator: minimize sum-of-squared-error toward a fixed target the
    // proposer never sees. This is the "un-gameable metric near physical truth" the
    // keep-or-revert contract requires — you can only win by genuinely getting closer.
    let target = vec![1.0, -2.0, 3.0, 0.5];
    let evaluator = ScalarEvaluator::sse_to_target(target.clone());

    // The proposer: seeded, bounded random local perturbations of the current baseline.
    let proposer = LocalProposer::new(0.4, 2026).with_bounds(-5.0, 5.0);

    // The Run: editable artifact (baseline) + evaluator + proposer + ratchet + log.
    let mut run = Run::new(
        vec![0.0, 0.0, 0.0, 0.0], // start far from the target
        evaluator,
        proposer,
        NoopRatchet::new(),
        AttemptLog::new(),
    )
    .with_goal("minimize SSE to a frozen target vector");

    run.establish_baseline();
    let start = run.baseline_value();
    println!("VisionAutoResearch — keep-or-revert loop (Rust port of Karpathy's idea)");
    println!("  target (hidden from the proposer): {target:?}");
    println!("  start baseline SSE: {start:.5}\n");

    let results = run.run_loop(StopCondition::MaxIterations(200));

    // Show only the iterations that KEPT — i.e. a real improvement landed and ratcheted.
    let mut shown = 0;
    for r in &results {
        if r.kept() {
            println!(
                "  iter {:>3}  KEEP   metric {:.5}  baseline -> {:.5}  [{}]",
                r.iter, r.metric_value, r.baseline_value, r.commit7
            );
            shown += 1;
        }
    }
    if shown == 0 {
        println!("  (no improvement kept — try a different seed or step size)");
    }

    let reverted = results.len() - run.n_kept();
    let pct_closer = if start > 0.0 {
        (1.0 - run.baseline_value() / start) * 100.0
    } else {
        0.0
    };
    println!(
        "\nDONE: {} iterations — {} kept, {} reverted.",
        results.len(),
        run.n_kept(),
        reverted
    );
    println!(
        "  baseline ratcheted {start:.5} -> {:.5}  ({pct_closer:.1}% closer to target)",
        run.baseline_value()
    );
    println!(
        "  kept commits in the ratchet history: {}",
        run.git_history().len()
    );
}
