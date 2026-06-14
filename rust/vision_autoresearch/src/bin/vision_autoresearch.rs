//! vision_autoresearch — standalone CLI for the keep-or-revert loop + TimesFM walk-forward.
//!
//! Subcommands:
//!   * `run`         — keep-or-revert loop on a config (a MiroFish scenario backtest, or a
//!                     walk-forward over a series file). Ratchets one artifact to a better scalar.
//!   * `walkforward` — load a per-step series from a CSV or JSON file, run the TimesFM walk-forward
//!                     evaluator on it, and print the out-of-sample scalar + per-fold breakdown.
//!   * `example`     — the 30-second keep-or-revert demo (no inputs).
//!
//! All numeric flags are `--key value`; unknown flags are ignored so the surface is forgiving.
//! No external arg-parsing crate — this keeps the CLI dependency-free (only the rlib + std).

use std::env;
use std::fs;
use std::process;

use vision_autoresearch::{
    AttemptLog, BtMetric, Evaluator, ForecasterKind, ForecasterSpec, LocalProposer,
    MirofishScenarioSource, NoopRatchet, Run, StopCondition, WalkForward,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("example");
    let rest = &args[2.min(args.len())..];
    let code = match sub {
        "run" => cmd_run(rest),
        "walkforward" | "wf" => cmd_walkforward(rest),
        "example" | "demo" => cmd_example(),
        "-h" | "--help" | "help" => {
            print_help();
            0
        }
        other => {
            eprintln!("unknown subcommand `{other}`\n");
            print_help();
            2
        }
    };
    process::exit(code);
}

fn print_help() {
    println!(
        "vision_autoresearch — keep-or-revert autoresearch + TimesFM walk-forward\n\n\
         USAGE:\n  \
           vision_autoresearch <SUBCOMMAND> [flags]\n\n\
         SUBCOMMANDS:\n  \
           run            keep-or-revert loop. Flags: --mode <backtest|walkforward>\n                 \
                          [--series <file>] [--metric sharpe|sortino|total_return|max_drawdown|calmar]\n                 \
                          [--iterations N] [--train-len W] [--horizon H] [--step S]\n                 \
                          [--forecaster naive|ewma|holt] [--n-scenarios N] [--seed S] [--step-size X]\n  \
           walkforward    eval a series file (CSV/JSON). Flags: --series <file> [--metric ...]\n                 \
                          [--train-len W] [--horizon H] [--step S] [--forecaster ...]\n                 \
                          [--tilt X] [--mom X] [--alpha A] [--beta B] [--overfit-penalty P]\n  \
           example        the 30-second keep-or-revert demo\n\n\
         SERIES FILE: a JSON array of numbers, or one-number-per-line / comma-separated CSV."
    );
}

// ───────────────────────── flag parsing helpers (dependency-free) ─────────────────────────

fn flag<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == key {
            return args.get(i + 1).map(|s| s.as_str());
        }
        // also accept --key=value
        if let Some(v) = args[i].strip_prefix(&format!("{key}=")) {
            return Some(v);
        }
        i += 1;
    }
    None
}

fn flag_f64(args: &[String], key: &str, default: f64) -> f64 {
    flag(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn flag_usize(args: &[String], key: &str, default: usize) -> usize {
    flag(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn flag_u64(args: &[String], key: &str, default: u64) -> u64 {
    flag(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn flag_str<'a>(args: &'a [String], key: &str, default: &'a str) -> &'a str {
    flag(args, key).unwrap_or(default)
}

// ───────────────────────── series loading (CSV or JSON) ─────────────────────────

/// Load a per-step numeric series from `path`. Accepts:
///   * a JSON array: `[100.0, 101.2, …]`
///   * CSV / plain text: numbers separated by newlines, commas, or whitespace.
/// Non-numeric tokens are skipped. Errors are returned as a message string.
fn load_series(path: &str) -> Result<Vec<f64>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("cannot read `{path}`: {e}"))?;
    let trimmed = raw.trim_start();
    // JSON array path
    if trimmed.starts_with('[') {
        let v: Vec<f64> = serde_json::from_str(trimmed)
            .map_err(|e| format!("`{path}` looks like JSON but failed to parse as a number array: {e}"))?;
        if v.is_empty() {
            return Err(format!("`{path}` parsed to an empty series"));
        }
        return Ok(v);
    }
    // CSV / whitespace / newline-separated
    let nums: Vec<f64> = raw
        .split(|c: char| c == ',' || c.is_whitespace() || c == ';')
        .filter_map(|tok| {
            let t = tok.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<f64>().ok()
            }
        })
        .collect();
    if nums.is_empty() {
        return Err(format!("`{path}` contained no parseable numbers"));
    }
    Ok(nums)
}

// ───────────────────────── subcommand: example ─────────────────────────

fn cmd_example() -> i32 {
    use vision_autoresearch::ScalarEvaluator;
    let target = vec![1.0, -2.0, 3.0, 0.5];
    let evaluator = ScalarEvaluator::sse_to_target(target.clone());
    let proposer = LocalProposer::new(0.4, 2026).with_bounds(-5.0, 5.0);
    let mut run = Run::new(
        vec![0.0, 0.0, 0.0, 0.0],
        evaluator,
        proposer,
        NoopRatchet::new(),
        AttemptLog::new(),
    )
    .with_goal("minimize SSE to a frozen target vector");
    run.establish_baseline();
    let start = run.baseline_value();
    let results = run.run_loop(StopCondition::MaxIterations(200));
    let pct = if start > 0.0 {
        (1.0 - run.baseline_value() / start) * 100.0
    } else {
        0.0
    };
    println!("vision_autoresearch example — keep-or-revert (Karpathy's loop, native Rust)");
    println!("  target (hidden): {target:?}");
    println!(
        "  {} iterations — {} kept, {} reverted",
        results.len(),
        run.n_kept(),
        results.len() - run.n_kept()
    );
    println!(
        "  baseline SSE {start:.5} -> {:.5}  ({pct:.1}% closer)",
        run.baseline_value()
    );
    0
}

// ───────────────────────── subcommand: walkforward ─────────────────────────

fn cmd_walkforward(args: &[String]) -> i32 {
    let series_path = match flag(args, "--series") {
        Some(p) => p,
        None => {
            eprintln!("walkforward: --series <file> is required");
            return 2;
        }
    };
    let series = match load_series(series_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("walkforward: {e}");
            return 1;
        }
    };

    let bt = BtMetric::from_str(flag_str(args, "--metric", "sharpe"));
    let train_len = flag_usize(args, "--train-len", (series.len() / 4).max(8));
    let horizon = flag_usize(args, "--horizon", (series.len() / 12).max(4));
    let step = flag_usize(args, "--step", horizon.max(1));
    let kind = ForecasterKind::from_str(flag_str(args, "--forecaster", "holt"));
    let overfit_penalty = flag_f64(args, "--overfit-penalty", 0.5);
    let tilt = flag_f64(args, "--tilt", 1.0);
    let mom = flag_f64(args, "--mom", 0.0);
    let alpha = flag_f64(args, "--alpha", 0.4);
    let beta = flag_f64(args, "--beta", 0.1);

    let wf = WalkForward::new(
        bt,
        series.clone(),
        true, // series files are PRICE levels by default
        train_len,
        horizon,
        step,
        ForecasterSpec::new(2, kind),
    )
    .with_overfit_penalty(overfit_penalty);

    let artifact = vec![tilt, mom, alpha, beta];
    let score = wf.score(&artifact);

    println!("vision_autoresearch walkforward — `{series_path}`");
    println!("  series length      : {}", series.len());
    println!("  metric             : {} ({})", bt.name(), wf.metric().direction.as_str());
    println!("  forecaster         : {}", kind.name());
    println!("  train_len/horizon  : {train_len} / {horizon}  (step {step})");
    println!("  folds (OOS)        : {}", wf.n_folds());
    println!("  artifact           : tilt={tilt} mom={mom} alpha={alpha} beta={beta}");
    println!("  overfit penalty    : {overfit_penalty}");
    println!("  ---------------------------------------------");
    if score.is_finite() {
        println!("  WALK-FORWARD SCORE : {score:.6}  (strictly out-of-sample)");
        0
    } else {
        println!("  WALK-FORWARD SCORE : NaN  (too short for any fold? need len >= train_len + horizon)");
        1
    }
}

// ───────────────────────── subcommand: run (keep-or-revert) ─────────────────────────

fn cmd_run(args: &[String]) -> i32 {
    let mode = flag_str(args, "--mode", "backtest");
    let bt = BtMetric::from_str(flag_str(args, "--metric", "sharpe"));
    let iterations = flag_usize(args, "--iterations", 100);
    let seed = flag_u64(args, "--seed", 42);
    let step_size = flag_f64(args, "--step-size", 0.2);

    match mode {
        "walkforward" | "wf" => {
            let series_path = match flag(args, "--series") {
                Some(p) => p,
                None => {
                    eprintln!("run --mode walkforward: --series <file> is required");
                    return 2;
                }
            };
            let series = match load_series(series_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("run: {e}");
                    return 1;
                }
            };
            let train_len = flag_usize(args, "--train-len", (series.len() / 4).max(8));
            let horizon = flag_usize(args, "--horizon", (series.len() / 12).max(4));
            let step = flag_usize(args, "--step", horizon.max(1));
            let kind = ForecasterKind::from_str(flag_str(args, "--forecaster", "holt"));
            let wf = WalkForward::new(
                bt,
                series,
                true,
                train_len,
                horizon,
                step,
                ForecasterSpec::new(2, kind),
            );
            // artifact = [tilt, mom, alpha, beta]; co-tune strategy + forecaster knobs.
            let prop = LocalProposer::new(step_size, seed).with_bounds(-1.0, 1.0);
            let mut run = Run::new(
                vec![0.0, 0.0, 0.4, 0.1],
                wf,
                prop,
                NoopRatchet::new(),
                AttemptLog::new(),
            )
            .with_goal("walk-forward: co-tune strategy + forecaster");
            run.establish_baseline();
            report_run("walkforward", &mut run, iterations)
        }
        _ => {
            // MiroFish scenario backtest mode.
            let start = flag_f64(args, "--start", 100.0);
            let drift = flag_f64(args, "--drift", 0.001);
            let vol = flag_f64(args, "--vol", 0.02);
            let n_scenarios = flag_usize(args, "--n-scenarios", 200);
            let horizon = flag_usize(args, "--horizon", 12);
            let src = MirofishScenarioSource::new(start, drift, vol);
            let ev = vision_autoresearch::vectorbt_evaluator_from_source(&src, n_scenarios, horizon, seed, bt);
            let prop = LocalProposer::new(step_size, seed).with_bounds(-3.0, 3.0);
            let mut run = Run::new(
                vec![0.0, 0.0],
                ev,
                prop,
                NoopRatchet::new(),
                AttemptLog::new(),
            )
            .with_goal(format!("backtest: optimize {}", bt.name()));
            run.establish_baseline();
            report_run("backtest", &mut run, iterations)
        }
    }
}

/// Run the loop and print a compact report. Generic over the evaluator/proposer the Run holds.
fn report_run<E, P, R>(mode: &str, run: &mut Run<E, P, R>, iterations: usize) -> i32
where
    E: vision_autoresearch::Evaluator,
    P: vision_autoresearch::Proposer,
    R: vision_autoresearch::Ratchet,
{
    let start = run.baseline_value();
    let dir = run.metric().direction.as_str();
    let name = run.metric().name.clone();
    let results = run.run_loop(StopCondition::MaxIterations(iterations.max(1)));
    println!("vision_autoresearch run — mode={mode}  metric={name} ({dir})");
    println!(
        "  {} iterations — {} kept, {} reverted",
        results.len(),
        run.n_kept(),
        results.len() - run.n_kept()
    );
    let s = if start.is_finite() { format!("{start:.6}") } else { "worst".into() };
    println!("  baseline {name}: {s} -> {:.6}", run.baseline_value());
    println!("  best artifact      : {:?}", run.target);
    println!("  kept commits       : {}", run.git_history().len());
    0
}
