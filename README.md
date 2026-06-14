# VisionAutoResearch

**A Rust-native, parallel autonomous-research engine** — open source, free.

> **Built on Andrej Karpathy's idea.** VisionAutoResearch is a Rust port + generalization of
> **[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)** by **Andrej Karpathy**.
> The original three-file Python project introduced the *keep-or-revert overnight research loop*;
> all credit for the idea goes to him — please ⭐ and read [the original](https://github.com/karpathy/autoresearch).

## What it is
Give it **one editable target + a frozen evaluator + a scalar metric + a goal**, and it runs:

```
propose a variation  ->  apply  ->  evaluate (frozen)  ->  keep if the metric improved, else revert  ->  log  ->  repeat
```

The output is a **git history of validated improvements** plus a full attempt log — exactly
Karpathy's pattern, re-implemented in **native Rust**.

## What this version adds
- 🦀 **Native Rust** — no Python runtime required; ship a single pre-compiled binary.
- ⚡ **Parallel processing** — a `rayon`-powered scenario sweep runs *every* variation/scenario
  concurrently across all cores (not one-at-a-time overnight).
- 🐟 **MiroFish Simulation Lab integration** — forward / Monte-Carlo rollouts generate the
  scenario space, so the loop explores far more of the search surface per wall-clock hour.
- 📈 **VectorBT evaluator** — plug in a vectorized backtest as the frozen scalar metric for
  trading-strategy and quantitative search (any evaluator that returns a number works).

## Quick start (30 seconds)
The crate lives in [`rust/vision_autoresearch`](rust/vision_autoresearch). Run the bundled
example — a keep-or-revert loop ratcheting toward a frozen target:

```bash
git clone https://github.com/MetaBrainAI-AGI/VisionAutoResearch.git
cd VisionAutoResearch/rust/vision_autoresearch
cargo run --release --example keep_or_revert
```

```
VisionAutoResearch — keep-or-revert loop (Rust port of Karpathy's idea)
  target (hidden from the proposer): [1.0, -2.0, 3.0, 0.5]
  start baseline SSE: 14.25000

  iter   0  KEEP   metric 12.98446  baseline -> 12.98446  [0000001]
  iter   6  KEEP   metric 12.30771  baseline -> 12.30771  [0000007]
  ...
DONE: 200 iterations — N kept, M reverted.
  baseline ratcheted 14.25000 -> ...  (closer to target)
```

Use it as a library — implement the `Evaluator`, `Proposer`, and `Ratchet` traits for your own
artifact, or use the built-in `ScalarEvaluator` / `VectorBtEvaluator` + `LocalProposer`. Run the
tests with `cargo test --release`.

## Status
Engine + crate are **live in this repo** (`cargo build`/`test`/`run --example` all green). The
optional `pyo3` feature exposes the evaluator as a native Python kernel for embedding hosts.

## License
MIT — free to use, modify, and distribute. See [`LICENSE`](LICENSE). Original pattern © Andrej Karpathy
(`karpathy/autoresearch`, MIT); this Rust port © MetaBrainAGI.

— **VisionAutoResearch** · part of the [VisionPRIME](https://github.com/MetaBrainAI-AGI) ecosystem · by MetaBrainAGI
