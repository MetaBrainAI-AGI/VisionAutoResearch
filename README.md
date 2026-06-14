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

## Status
The Rust engine is in active build. This repo is the public home; the binary + crate land here.

## License
MIT — free to use, modify, and distribute. See [`LICENSE`](LICENSE). Original pattern © Andrej Karpathy
(`karpathy/autoresearch`, MIT); this Rust port © MetaBrainAGI.

— **VisionAutoResearch** · part of the [VisionPRIME](https://github.com/MetaBrainAI-AGI) ecosystem · by MetaBrainAGI
