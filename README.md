# VisionAutoResearch

**Rust-native autonomous research loop for VisionPRIME** — a native port of
[Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch) keep-or-revert
pattern, generalized to *any* optimize-against-a-metric problem.

## The loop
Give it **one editable target + a frozen evaluator + a scalar metric + a goal**, then it runs:

```
propose a variation  ->  apply  ->  evaluate (frozen)  ->  keep if the metric improved, else revert  ->  log  ->  repeat
```

The output is a **git history of validated improvements** plus a full attempt log — exactly
Karpathy's overnight pattern, but native Rust with a **rayon-parallel scenario sweep** so it runs
*every* variation/scenario concurrently.

## VisionPRIME integration
- **Mirofish** — the scenario source (forward / monte-carlo rollouts feed the sweep).
- **VectorBT evaluator** — vectorized backtest → the scalar metric (for strategy search).
- **Dashboard** `/autoresearch` — configure target, evaluator, metric, conditions, goals,
  strategies/variations, and **schedule** runs.
- Driven by the VisionPRIME runtime; scenarios → keep-or-revert → validated improvements.

## Status
Private. Source is compiled and shipped pre-built (Rust-native standard); the long
from-source build lives only in the master-dev tooling, never shipped.

— MetaBrainAGI · VisionPRIME
