//! timesfm_walkforward.rs — a RUST-NATIVE, rayon-parallel TimesFM WALK-FORWARD evaluator.
//!
//! ## What this is
//! A frozen [`Evaluator`] (`WalkForward`) that rolls an in-sample window over a per-step series,
//! conditions a [`Forecaster`] on each window, forecasts the next `horizon` steps, and scores the
//! candidate strategy on the STRICTLY OUT-OF-SAMPLE realized path — never on data the forecaster
//! saw. The per-fold backtest scalars are aggregated to ONE number (the same un-gameable scalar
//! the keep-or-revert ratchet selects on), and an even/odd held-out consistency penalty is folded
//! in so a candidate that wins by overfitting one phase is demoted.
//!
//! Because `WalkForward` implements the crate's [`Evaluator`] trait, it drops straight into
//! [`Run`](crate::Run) (keep-or-revert) and the rayon [`sweep`](crate::sweep)/[`spawn_swarm`]
//! — no engine change. The candidate [`Artifact`] carries the strategy params PLUS the
//! forecaster hyper-knobs, so the loop tunes BOTH the strategy and the forecaster against one
//! scalar.
//!
//! ## The rolling-window algorithm (`WalkForward::score`)
//! Given a series `S` of length `T`, an in-sample window `W` = `train_len`, a forecast `H` =
//! `horizon`, and a `step`: for each fold origin `t = W, W+step, … while t+H <= T`:
//!   1. in-sample = `S[t-W .. t]`;
//!   2. condition/fit the forecaster on that slice and forecast the next `H` points;
//!   3. realized OUT-OF-SAMPLE path = `S[t .. t+H]` (never seen during fit — the un-gameable part);
//!   4. derive a strategy returns series from `params` + the forecast-direction overlay vs. the
//!      realized OOS path;
//!   5. score that OOS returns series with the chosen [`BtMetric`] kernel -> one per-fold scalar.
//! Aggregate the per-fold scalars to one mean (robust central fitness, matching
//! `VectorBtEvaluator`'s cross-ensemble mean). When an ENSEMBLE of scenario paths is supplied
//! (e.g. MiroFish rollouts) the outer rayon is over scenarios and folds run inner-sequential.
//!
//! ## The overfit guard (the un-gameable invariant)
//! Two mechanisms, both computed INSIDE `score()` so the proposer has no path to influence them:
//!   * (A) STRICT OUT-OF-SAMPLE ONLY — the per-fold metric uses only `S[t .. t+H]`.
//!   * (B) EVEN/ODD held-out consistency — split folds into A=even / B=odd; a candidate that wins
//!     on one half but not the other is penalized by `|score_A - score_B|` (direction-normalized),
//!     scaled by `overfit_penalty`. Consistent generalizers keep their score; phase-overfitters
//!     are demoted toward (and past) a candidate that wins both halves modestly.
//!
//! ## ONNX TimesFM (feature = "timesfm-onnx")
//! The native baselines (Naive/EWMA/Holt) ship NOW with zero new required deps and pass
//! `cargo test`. An optional [`OnnxForecaster`] mirrors `metabrain_core/src/onnx_predict.rs`'s
//! proven load-once/run pattern. KEY DECISION (flagged): the crate's established ONNX convention
//! is **`tract-onnx`** (pure-Rust, NO native onnxruntime dep) — see metabrain_core. The task's
//! `ort` feature name is honored as the cargo FEATURE name (`timesfm-onnx`), but the dependency it
//! gates is `tract-onnx`, because a hard `ort`/onnxruntime native lib would violate the crate's
//! no-native-dep posture and would NOT match the proven seam. Confirm if `ort` is truly required.

use crate::evaluator::{bt_metric_of_path, Artifact, BtMetric, Evaluator};
use crate::metric::{Direction, Metric};
use rayon::prelude::*;

// ───────────────────────────── Forecaster trait + native impls ─────────────────────────────

/// A point forecaster. `forecast(history, horizon)` conditions on the in-sample `history` slice
/// and returns the next `horizon` point predictions (same units as `history`). Implementors MUST
/// be deterministic for a given `(history, horizon)` so the walk-forward is reproducible, and
/// MUST NOT see any out-of-sample data (the engine only ever hands them the in-sample slice).
pub trait Forecaster: Send + Sync {
    /// Condition on `history` and predict the next `horizon` points.
    fn forecast(&self, history: &[f64], horizon: usize) -> Vec<f64>;

    /// Human label for logs.
    fn name(&self) -> &str {
        "forecaster"
    }
}

/// NAIVE (random-walk) forecaster: the last observed value, repeated `horizon` times. The
/// canonical forecasting baseline — any model must beat this to earn its keep. Zero hyper-knobs.
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveForecaster;

impl Forecaster for NaiveForecaster {
    fn forecast(&self, history: &[f64], horizon: usize) -> Vec<f64> {
        let last = history.last().copied().unwrap_or(0.0);
        vec![last; horizon]
    }
    fn name(&self) -> &str {
        "naive"
    }
}

/// EWMA (exponentially-weighted moving average) forecaster. One knob `alpha in (0,1]`: the level
/// is `l_t = alpha*x_t + (1-alpha)*l_{t-1}`, and the flat forecast repeats the final level. A
/// larger `alpha` tracks recent data faster; a smaller `alpha` smooths more.
#[derive(Debug, Clone, Copy)]
pub struct EwmaForecaster {
    pub alpha: f64,
}

impl EwmaForecaster {
    pub fn new(alpha: f64) -> Self {
        // clamp to a sane open interval so the recursion is always well-defined
        EwmaForecaster {
            alpha: alpha.clamp(1e-3, 1.0),
        }
    }
}

impl Default for EwmaForecaster {
    fn default() -> Self {
        EwmaForecaster::new(0.3)
    }
}

impl Forecaster for EwmaForecaster {
    fn forecast(&self, history: &[f64], horizon: usize) -> Vec<f64> {
        if history.is_empty() {
            return vec![0.0; horizon];
        }
        let a = self.alpha;
        let mut level = history[0];
        for &x in &history[1..] {
            level = a * x + (1.0 - a) * level;
        }
        vec![level; horizon]
    }
    fn name(&self) -> &str {
        "ewma"
    }
}

/// HOLT (double-exponential smoothing) forecaster — level + trend. Two knobs: `alpha` (level
/// smoothing) and `beta` (trend smoothing). The `h`-step forecast is `level + h*trend`, so unlike
/// EWMA it extrapolates a slope. This is the strongest native baseline and the one whose two
/// knobs the keep-or-revert loop can co-tune with the strategy params.
#[derive(Debug, Clone, Copy)]
pub struct HoltForecaster {
    pub alpha: f64,
    pub beta: f64,
}

impl HoltForecaster {
    pub fn new(alpha: f64, beta: f64) -> Self {
        HoltForecaster {
            alpha: alpha.clamp(1e-3, 1.0),
            beta: beta.clamp(0.0, 1.0),
        }
    }
}

impl Default for HoltForecaster {
    fn default() -> Self {
        HoltForecaster::new(0.4, 0.1)
    }
}

impl Forecaster for HoltForecaster {
    fn forecast(&self, history: &[f64], horizon: usize) -> Vec<f64> {
        let n = history.len();
        if n == 0 {
            return vec![0.0; horizon];
        }
        if n == 1 {
            return vec![history[0]; horizon];
        }
        let (a, b) = (self.alpha, self.beta);
        // standard Holt initialization: level = first obs, trend = first observed slope.
        let mut level = history[0];
        let mut trend = history[1] - history[0];
        for &x in &history[1..] {
            let prev_level = level;
            level = a * x + (1.0 - a) * (prev_level + trend);
            trend = b * (level - prev_level) + (1.0 - b) * trend;
        }
        (1..=horizon).map(|h| level + (h as f64) * trend).collect()
    }
    fn name(&self) -> &str {
        "holt"
    }
}

/// Build a boxed [`Forecaster`] from a string kind + the candidate's forecaster hyper-knobs.
/// `alpha`/`beta` are read from the artifact tail (see [`ForecasterSpec`]); unknown kinds fall
/// back to Holt (the most expressive native baseline).
pub fn make_forecaster(kind: ForecasterKind, alpha: f64, beta: f64) -> Box<dyn Forecaster> {
    match kind {
        ForecasterKind::Naive => Box::new(NaiveForecaster),
        ForecasterKind::Ewma => Box::new(EwmaForecaster::new(alpha)),
        ForecasterKind::Holt => Box::new(HoltForecaster::new(alpha, beta)),
        #[cfg(feature = "timesfm-onnx")]
        ForecasterKind::TimesFmOnnx => {
            // The ONNX path is constructed elsewhere (it needs a model handle); fall back to
            // Holt here so `make_forecaster` stays infallible. Use `WalkForward::with_onnx`.
            Box::new(HoltForecaster::new(alpha, beta))
        }
    }
}

/// Which forecaster a [`WalkForward`] evaluator drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecasterKind {
    Naive,
    Ewma,
    Holt,
    #[cfg(feature = "timesfm-onnx")]
    TimesFmOnnx,
}

impl ForecasterKind {
    pub fn from_str(s: &str) -> ForecasterKind {
        match s.trim().to_lowercase().as_str() {
            "naive" | "rw" | "random_walk" => ForecasterKind::Naive,
            "ewma" | "ewm" | "ema" => ForecasterKind::Ewma,
            #[cfg(feature = "timesfm-onnx")]
            "timesfm" | "timesfm-onnx" | "onnx" => ForecasterKind::TimesFmOnnx,
            _ => ForecasterKind::Holt,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            ForecasterKind::Naive => "naive",
            ForecasterKind::Ewma => "ewma",
            ForecasterKind::Holt => "holt",
            #[cfg(feature = "timesfm-onnx")]
            ForecasterKind::TimesFmOnnx => "timesfm-onnx",
        }
    }
}

// ───────────────────────────── Artifact layout (strategy + forecaster) ─────────────────────────────

/// How a candidate [`Artifact`] is split into the STRATEGY params and the FORECASTER hyper-knobs.
/// Layout: `[ strategy_0, strategy_1, …, strategy_{k-1}, alpha, beta ]`. The first `n_strategy`
/// knobs are the strategy params (exactly what `VectorBtEvaluator` consumes — `tilt`, `mom_scale`);
/// the final two are the forecaster `alpha`/`beta`. This lets the keep-or-revert loop co-tune the
/// strategy AND the forecaster against one scalar. If the artifact is shorter than expected,
/// missing knobs default sensibly so a bare `[tilt, mom]` artifact still works.
#[derive(Debug, Clone, Copy)]
pub struct ForecasterSpec {
    /// Number of leading strategy knobs (default 2: tilt + momentum-scale).
    pub n_strategy: usize,
    /// Forecaster family.
    pub kind: ForecasterKind,
}

impl Default for ForecasterSpec {
    fn default() -> Self {
        ForecasterSpec {
            n_strategy: 2,
            kind: ForecasterKind::Holt,
        }
    }
}

impl ForecasterSpec {
    pub fn new(n_strategy: usize, kind: ForecasterKind) -> Self {
        ForecasterSpec {
            n_strategy: n_strategy.max(1),
            kind,
        }
    }

    /// The strategy-param slice of an artifact (the first `n_strategy` knobs).
    pub fn strategy_params<'a>(&self, artifact: &'a Artifact) -> &'a [f64] {
        let k = self.n_strategy.min(artifact.len());
        &artifact[..k]
    }

    /// The forecaster `(alpha, beta)` read from the artifact tail. Defaults: alpha=0.4, beta=0.1.
    pub fn forecaster_knobs(&self, artifact: &Artifact) -> (f64, f64) {
        let alpha = artifact.get(self.n_strategy).copied().unwrap_or(0.4);
        let beta = artifact.get(self.n_strategy + 1).copied().unwrap_or(0.1);
        (alpha, beta)
    }

    /// Build the boxed forecaster this artifact specifies.
    pub fn build_forecaster(&self, artifact: &Artifact) -> Box<dyn Forecaster> {
        let (alpha, beta) = self.forecaster_knobs(artifact);
        make_forecaster(self.kind, alpha, beta)
    }
}

// ───────────────────────────── strategy returns from a forecast overlay ─────────────────────────────

/// Convert a per-step PRICE-or-RETURN window to step RETURNS.
fn to_returns(path: &[f64], are_prices: bool) -> Vec<f64> {
    if are_prices {
        path.windows(2)
            .map(|w| if w[0].abs() > 1e-12 { w[1] / w[0] - 1.0 } else { 0.0 })
            .collect()
    } else {
        path.to_vec()
    }
}

/// The frozen, vectorizable strategy model used inside each fold. Identical in spirit to
/// `VectorBtEvaluator::strategy_returns` (constant tilt + momentum overlay) but ADDS a
/// forecast-direction overlay: a GOOD forecast (whose sign matches the realized OOS move) earns
/// positive exposure, a BAD one is punished. This is what makes a better forecaster score higher
/// — the metric rewards forecast accuracy via realized OOS returns, never via lookahead.
///
/// `params`     = strategy knobs (tilt, mom_scale).
/// `realized`   = OOS step returns (what actually happened — the un-gameable truth).
/// `fc_dir`     = per-step forecast direction in {-1,0,+1} (sign of forecast point-over-point),
///                aligned to `realized`. `fc_weight` scales how much the forecast tilts exposure.
fn strategy_returns_with_forecast(
    params: &[f64],
    realized: &[f64],
    fc_dir: &[f64],
    fc_weight: f64,
) -> Vec<f64> {
    let tilt = params.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let mom_scale = params.get(1).copied().unwrap_or(0.0).clamp(-2.0, 2.0);
    let mut out = Vec::with_capacity(realized.len());
    let mut prev = 0.0f64;
    for (i, r) in realized.iter().enumerate() {
        let dir = fc_dir.get(i).copied().unwrap_or(0.0);
        // exposure = constant tilt + momentum overlay + forecast-direction overlay, clamped.
        let exposure =
            (tilt + mom_scale * prev.signum() * 0.5 + fc_weight * dir).clamp(-1.0, 1.0);
        out.push(exposure * r);
        prev = *r;
    }
    out
}

/// Per-step forecast DIRECTION aligned to a realized OOS window. The forecaster predicts the next
/// `H` price/return points from the in-sample slice; the direction at step `i` is the sign of the
/// forecast's point-over-point change (for prices: `sign(point[i] - prev_anchor)`; for returns:
/// `sign(point[i])`). `anchor` is the last in-sample observation (the forecaster's starting point).
fn forecast_direction(points: &[f64], anchor: f64, are_prices: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(points.len());
    let mut prev = anchor;
    for &p in points {
        let dir = if are_prices { p - prev } else { p };
        out.push(dir.signum());
        prev = p;
    }
    out
}

// ───────────────────────────── WalkForward evaluator ─────────────────────────────

/// One per-fold result (kept for diagnostics + the held-out split).
#[derive(Debug, Clone, Copy)]
pub struct FoldScore {
    pub origin: usize,
    pub metric_value: f64,
}

/// A FROZEN walk-forward evaluator. Holds a long per-step series (or a SCENARIO ENSEMBLE of them),
/// rolls a `train_len` window with `step`, forecasts `horizon`, and scores STRICTLY out-of-sample.
/// The candidate [`Artifact`] supplies both the strategy params and the forecaster knobs (see
/// [`ForecasterSpec`]). Implements [`Evaluator`], so it drops into `Run` and the rayon sweep.
pub struct WalkForward {
    metric: Metric,
    bt: BtMetric,
    /// The scenario ensemble: one or more long per-step series. A single realized history is
    /// just `vec![history]`. Frozen at construction (part of the un-gameable evaluator).
    series_set: Vec<Vec<f64>>,
    are_prices: bool,
    train_len: usize,
    horizon: usize,
    step: usize,
    spec: ForecasterSpec,
    /// How strongly the forecast-direction overlay tilts strategy exposure (default 0.5).
    fc_weight: f64,
    /// Even/odd held-out consistency penalty weight (default 0.5). 0 disables the guard.
    overfit_penalty: f64,
    /// Optional ONNX forecaster (feature-gated). When set it overrides `spec.kind`.
    #[cfg(feature = "timesfm-onnx")]
    onnx: Option<std::sync::Arc<OnnxForecaster>>,
}

impl WalkForward {
    /// Build a walk-forward evaluator over a SINGLE realized series.
    pub fn new(
        bt: BtMetric,
        series: Vec<f64>,
        are_prices: bool,
        train_len: usize,
        horizon: usize,
        step: usize,
        spec: ForecasterSpec,
    ) -> Self {
        Self::from_ensemble(bt, vec![series], are_prices, train_len, horizon, step, spec)
    }

    /// Build a walk-forward evaluator over a SCENARIO ENSEMBLE (e.g. MiroFish rollouts). The
    /// outer rayon then parallelizes over scenarios.
    pub fn from_ensemble(
        bt: BtMetric,
        series_set: Vec<Vec<f64>>,
        are_prices: bool,
        train_len: usize,
        horizon: usize,
        step: usize,
        spec: ForecasterSpec,
    ) -> Self {
        WalkForward {
            metric: Metric::new(bt.name(), bt.direction()),
            bt,
            series_set,
            are_prices,
            train_len: train_len.max(2),
            horizon: horizon.max(1),
            step: step.max(1),
            spec,
            fc_weight: 0.5,
            overfit_penalty: 0.5,
            #[cfg(feature = "timesfm-onnx")]
            onnx: None,
        }
    }

    pub fn with_fc_weight(mut self, w: f64) -> Self {
        self.fc_weight = w;
        self
    }

    pub fn with_overfit_penalty(mut self, p: f64) -> Self {
        self.overfit_penalty = p.max(0.0);
        self
    }

    /// Attach an ONNX TimesFM forecaster (feature = "timesfm-onnx"); overrides the native kind.
    #[cfg(feature = "timesfm-onnx")]
    pub fn with_onnx(mut self, onnx: OnnxForecaster) -> Self {
        self.onnx = Some(std::sync::Arc::new(onnx));
        self
    }

    /// Build the forecaster this evaluator uses for a given artifact (ONNX overrides the kind).
    fn forecaster_for(&self, artifact: &Artifact) -> Box<dyn Forecaster> {
        #[cfg(feature = "timesfm-onnx")]
        {
            if let Some(o) = &self.onnx {
                return Box::new(ArcForecaster(o.clone()));
            }
        }
        self.spec.build_forecaster(artifact)
    }

    /// The fold origins for a series of length `t`: `train_len, train_len+step, … while o+H<=T`.
    fn fold_origins(&self, t: usize) -> Vec<usize> {
        let mut origins = Vec::new();
        let mut o = self.train_len;
        while o + self.horizon <= t {
            origins.push(o);
            o += self.step;
        }
        origins
    }

    /// Score ONE fold of ONE series: fit on `[o-W..o]`, forecast H, score the realized
    /// `[o..o+H]` OOS returns. Returns NaN when the fold is malformed (skipped by the aggregator).
    fn score_fold(&self, series: &[f64], forecaster: &dyn Forecaster, params: &[f64], o: usize) -> f64 {
        let w = self.train_len;
        if o < w || o + self.horizon > series.len() {
            return f64::NAN;
        }
        let in_sample = &series[o - w..o];
        let realized_window = &series[o..o + self.horizon];

        // 2. forecast the next H points from the in-sample slice (never sees realized_window).
        let points = forecaster.forecast(in_sample, self.horizon);
        let anchor = *in_sample.last().unwrap_or(&0.0);
        let fc_dir = forecast_direction(&points, anchor, self.are_prices);

        // 3-4. realized OOS step returns + strategy returns under the forecast overlay.
        // For prices we need the anchor->window transition, so prepend the anchor to compute the
        // first realized step return; align fc_dir to the resulting returns length.
        let realized_returns = if self.are_prices {
            let mut withanchor = Vec::with_capacity(self.horizon + 1);
            withanchor.push(anchor);
            withanchor.extend_from_slice(realized_window);
            to_returns(&withanchor, true) // length == horizon
        } else {
            realized_window.to_vec()
        };
        let strat = strategy_returns_with_forecast(params, &realized_returns, &fc_dir, self.fc_weight);

        // 5. score the OOS strategy returns with the chosen metric kernel.
        bt_metric_of_path(self.bt, &strat)
    }

    /// All per-fold scores across the whole ensemble (rayon-parallel). Each entry carries its
    /// fold origin so the even/odd held-out split can partition them.
    fn all_fold_scores(&self, artifact: &Artifact) -> Vec<FoldScore> {
        let params = self.spec.strategy_params(artifact);
        // Outer parallelism over scenarios; folds run inner-sequential (each scenario builds its
        // own forecaster handle so the boxed trait object is not shared across threads).
        self.series_set
            .par_iter()
            .flat_map_iter(|series| {
                let forecaster = self.forecaster_for(artifact);
                let origins = self.fold_origins(series.len());
                origins.into_iter().map(move |o| FoldScore {
                    origin: o,
                    metric_value: self.score_fold(series, forecaster.as_ref(), params, o),
                })
            })
            .collect()
    }

    /// Mean of the finite per-fold metric values (NaN folds skipped). Returns NaN if none finite.
    fn mean_finite(scores: &[FoldScore]) -> f64 {
        let finite: Vec<f64> = scores
            .iter()
            .map(|f| f.metric_value)
            .filter(|v| v.is_finite())
            .collect();
        if finite.is_empty() {
            f64::NAN
        } else {
            finite.iter().sum::<f64>() / finite.len() as f64
        }
    }

    /// EVEN/ODD held-out consistency: split folds by parity of their position in the origin
    /// sequence (interleaved so both halves span the whole series), take each half's mean, and
    /// return `(score_A_even, score_B_odd)`. Used to demote phase-overfitters.
    fn even_odd_means(&self, scores: &[FoldScore]) -> (f64, f64) {
        // Group fold-positions per scenario by interleaving on the ordinal index within each
        // scenario's origin list — but since we flattened, recover parity from a stable ordinal:
        // we re-derive parity from the origin value relative to train_len/step so it's stable.
        let mut even = Vec::new();
        let mut odd = Vec::new();
        for f in scores {
            if !f.metric_value.is_finite() {
                continue;
            }
            // ordinal of this origin within the rolling sequence: (origin - train_len)/step
            let ord = (f.origin.saturating_sub(self.train_len)) / self.step;
            if ord % 2 == 0 {
                even.push(f.metric_value);
            } else {
                odd.push(f.metric_value);
            }
        }
        let m = |v: &Vec<f64>| {
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        (m(&even), m(&odd))
    }

    /// The single aggregated scalar fitness with the overfit guard folded in. Public so callers
    /// (and tests) can inspect it directly; `Evaluator::score` delegates here.
    pub fn walk_forward_score(&self, artifact: &Artifact) -> f64 {
        let scores = self.all_fold_scores(artifact);
        let base = Self::mean_finite(&scores);
        if !base.is_finite() {
            return f64::NAN;
        }
        if self.overfit_penalty <= 0.0 {
            return base;
        }
        let (a, b) = self.even_odd_means(&scores);
        if !a.is_finite() || !b.is_finite() {
            // can't form a held-out split (too few folds) -> return the raw OOS mean.
            return base;
        }
        // Direction-aware penalty: subtract the inconsistency from the achieved fitness so the
        // proposer cannot win by overfitting one phase. For Maximize we subtract; for Minimize
        // (lower-is-better) we ADD the penalty so an overfitter's score gets WORSE either way.
        let inconsistency = (a - b).abs() * self.overfit_penalty;
        match self.metric.direction {
            Direction::Maximize => base - inconsistency,
            Direction::Minimize => base + inconsistency,
        }
    }

    /// Number of folds the current config produces over the whole ensemble (diagnostic).
    pub fn n_folds(&self) -> usize {
        self.series_set
            .iter()
            .map(|s| self.fold_origins(s.len()).len())
            .sum()
    }
}

impl Evaluator for WalkForward {
    fn score(&self, artifact: &Artifact) -> f64 {
        if self.series_set.is_empty() {
            return f64::NAN;
        }
        self.walk_forward_score(artifact)
    }
    fn metric(&self) -> &Metric {
        &self.metric
    }
    fn name(&self) -> &str {
        self.bt.name()
    }
}

// ───────────────────────────── ONNX TimesFM forecaster (feature = "timesfm-onnx") ─────────────────────────────

#[cfg(feature = "timesfm-onnx")]
mod onnx_impl {
    //! Optional ONNX TimesFM forecaster. Mirrors metabrain_core/src/onnx_predict.rs's proven
    //! load-once/run pattern using `tract-onnx` (pure-Rust, NO native onnxruntime dep). The
    //! cargo feature is named `timesfm-onnx` (honoring the task's `ort` intent as a feature
    //! name), but the dependency it gates is `tract-onnx` to match the crate's no-native-dep
    //! convention. The model is expected to take a fixed-length context window `[1, context_len]`
    //! and emit `[1, horizon]` point forecasts (TimesFM's standard signature).

    use super::Forecaster;
    use std::sync::Arc;
    use tract_onnx::prelude::*;

    type Runnable =
        SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

    /// A loaded, optimized, runnable ONNX TimesFM model. Built ONCE; `forecast` runs it per fold.
    pub struct OnnxForecaster {
        model: Runnable,
        context_len: usize,
        horizon: usize,
    }

    impl OnnxForecaster {
        /// Load + optimize + make-runnable `path`, pinned to a `[1, context_len]` input fact and
        /// expected to emit `horizon` points. Done ONCE (not per call), exactly like
        /// OnnxPredictNative.
        pub fn new(path: &str, context_len: usize, horizon: usize) -> TractResult<Self> {
            let model = tract_onnx::onnx()
                .model_for_path(path)?
                .with_input_fact(0, f32::fact([1, context_len]).into())?
                .into_optimized()?
                .into_runnable()?;
            Ok(OnnxForecaster {
                model,
                context_len: context_len.max(1),
                horizon: horizon.max(1),
            })
        }

        fn run(&self, ctx: &[f32]) -> TractResult<Vec<f32>> {
            let input = tract_ndarray::Array2::from_shape_vec((1, self.context_len), ctx.to_vec())?;
            let tensor: Tensor = input.into();
            let result = self.model.run(tvec!(tensor.into()))?;
            let view = result[0].to_array_view::<f32>()?;
            Ok(view.iter().copied().collect())
        }
    }

    impl Forecaster for OnnxForecaster {
        fn forecast(&self, history: &[f64], horizon: usize) -> Vec<f64> {
            // build a fixed-length context: left-pad (with the first value) or take the last
            // `context_len` observations, matching TimesFM's fixed context window.
            let mut ctx = vec![history.first().copied().unwrap_or(0.0) as f32; self.context_len];
            let take = history.len().min(self.context_len);
            for (i, &x) in history[history.len() - take..].iter().enumerate() {
                ctx[self.context_len - take + i] = x as f32;
            }
            match self.run(&ctx) {
                Ok(out) => {
                    let mut v: Vec<f64> = out.iter().map(|&x| x as f64).collect();
                    // The model emits its OWN fixed horizon (`self.horizon`); first normalize the
                    // raw output to that declared length, then resize to whatever the caller asked
                    // for. Folding `self.horizon` in makes the declared model horizon load-bearing
                    // (a model that over/under-emits is reconciled to its contract before reuse).
                    let last = v.last().copied().unwrap_or(0.0);
                    v.resize(self.horizon, last);
                    v.resize(horizon.max(1), last);
                    v
                }
                // fail-open to a naive forecast so a bad model handle never aborts a sweep.
                Err(_) => vec![history.last().copied().unwrap_or(0.0); horizon],
            }
        }
        fn name(&self) -> &str {
            "timesfm-onnx"
        }
    }

    /// A `Forecaster` wrapper that shares an `Arc<OnnxForecaster>` across rayon scenario threads
    /// (the model is `Send + Sync` once runnable, so sharing is sound and load-once is preserved).
    pub struct ArcForecaster(pub Arc<OnnxForecaster>);

    impl Forecaster for ArcForecaster {
        fn forecast(&self, history: &[f64], horizon: usize) -> Vec<f64> {
            self.0.forecast(history, horizon)
        }
        fn name(&self) -> &str {
            "timesfm-onnx-arc"
        }
    }
}

#[cfg(feature = "timesfm-onnx")]
pub use onnx_impl::{ArcForecaster, OnnxForecaster};

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Direction;

    // a strictly-rising price series -> trend forecasters should beat naive on a long strategy.
    fn rising_series(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64).collect()
    }

    #[test]
    fn naive_ewma_holt_forecast_shapes_and_values() {
        let h = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(NaiveForecaster.forecast(&h, 3), vec![4.0, 4.0, 4.0]);
        let e = EwmaForecaster::new(1.0).forecast(&h, 2); // alpha=1 -> tracks last value
        assert!((e[0] - 4.0).abs() < 1e-9 && e.len() == 2);
        // Holt on a perfect linear ramp should extrapolate the slope ~+1 per step.
        let ho = HoltForecaster::new(0.5, 0.5).forecast(&h, 3);
        assert_eq!(ho.len(), 3);
        assert!(ho[2] > ho[0], "Holt should extrapolate an upward trend: {ho:?}");
        assert!(ho[0] > 4.0, "Holt next point should exceed last obs on a ramp: {ho:?}");
    }

    #[test]
    fn fold_origins_are_strictly_out_of_sample_and_rolling() {
        let wf = WalkForward::new(
            BtMetric::Sharpe,
            rising_series(40),
            true,
            10, // train_len
            5,  // horizon
            5,  // step
            ForecasterSpec::default(),
        );
        let origins = wf.fold_origins(40);
        // first origin = train_len = 10; each fold's [o..o+H] never overlaps its in-sample [o-W..o]
        assert_eq!(origins.first().copied(), Some(10));
        assert!(origins.iter().all(|&o| o + 5 <= 40));
        // rolling by step=5: 10,15,20,25,30,35 (35+5=40 ok) -> 6 folds
        assert_eq!(origins, vec![10, 15, 20, 25, 30, 35]);
        assert_eq!(wf.n_folds(), 6);
    }

    #[test]
    fn walk_forward_aggregates_to_one_finite_scalar() {
        let wf = WalkForward::new(
            BtMetric::TotalReturn,
            rising_series(60),
            true,
            12,
            6,
            3,
            ForecasterSpec::new(2, ForecasterKind::Holt),
        );
        // candidate: long tilt + a positive forecaster (alpha,beta) tail
        let artifact = vec![1.0, 0.0, 0.4, 0.2];
        let s = wf.score(&artifact);
        assert!(s.is_finite(), "walk-forward must aggregate to a finite scalar, got {s}");
        // on a rising series a long+trend strategy should produce a positive total return.
        assert!(s > 0.0, "long strategy on a rising series should be profitable: {s}");
    }

    #[test]
    fn better_forecaster_scores_at_least_as_well_as_naive_on_trend() {
        // On a clean upward trend the Holt (trend-aware) forecast direction is reliably +1, so
        // the forecast overlay should not HURT vs. naive — and typically helps. We assert Holt is
        // not worse than naive by more than numerical noise (monotone-improvement spirit).
        let series = rising_series(80);
        let mk = |kind| {
            WalkForward::new(BtMetric::TotalReturn, series.clone(), true, 16, 8, 4, ForecasterSpec::new(2, kind))
                .with_overfit_penalty(0.0) // isolate forecaster effect from the held-out term
        };
        let artifact = vec![0.5, 0.0, 0.4, 0.2];
        let holt = mk(ForecasterKind::Holt).score(&artifact);
        let naive = mk(ForecasterKind::Naive).score(&artifact);
        assert!(holt.is_finite() && naive.is_finite());
        assert!(holt >= naive - 1e-9, "trend-aware forecaster should not underperform naive on a pure trend: holt={holt} naive={naive}");
    }

    #[test]
    fn monotonic_improvement_on_improving_strategy_knob() {
        // Hold the forecaster fixed; sweep the strategy tilt from short->long on a rising series.
        // Total return must be monotonically non-decreasing as tilt goes -1 -> 0 -> +1.
        let wf = WalkForward::new(
            BtMetric::TotalReturn,
            rising_series(70),
            true,
            14,
            7,
            7,
            ForecasterSpec::new(2, ForecasterKind::Holt),
        )
        .with_fc_weight(0.0) // isolate the tilt knob from the forecast overlay
        .with_overfit_penalty(0.0);
        let short = wf.score(&vec![-1.0, 0.0, 0.4, 0.2]);
        let flat = wf.score(&vec![0.0, 0.0, 0.4, 0.2]);
        let long = wf.score(&vec![1.0, 0.0, 0.4, 0.2]);
        assert!(short < flat, "short should lose on a rising series: short={short} flat={flat}");
        assert!(flat <= long, "long should win on a rising series: flat={flat} long={long}");
        assert!(long > 0.0, "long on a rising series must be profitable: {long}");
    }

    #[test]
    fn overfit_guard_demotes_an_inconsistent_candidate() {
        // Build a series whose EVEN folds and ODD folds behave very differently: a regime that
        // alternates between a strong up-block and a flat-block, so a candidate's per-fold metric
        // is high on one parity and low on the other -> a large even/odd gap -> the guard demotes.
        // Compare the SAME candidate scored with the guard ON vs OFF: ON must be <= OFF for a
        // Maximize metric (penalty subtracts), proving the held-out term bites.
        let mut series = Vec::new();
        // 6 blocks of 12 steps; even blocks rise hard, odd blocks are flat. train_len=12,horizon=6,
        // step=12 so each fold origin lands on a block boundary and inherits that block's regime.
        for b in 0..7 {
            let base = 100.0 + (b as f64) * 5.0;
            for i in 0..12 {
                if b % 2 == 0 {
                    series.push(base + i as f64 * 2.0); // strong rise
                } else {
                    series.push(base); // flat
                }
            }
        }
        let artifact = vec![1.0, 0.0, 0.4, 0.2];
        let guarded = WalkForward::from_ensemble(
            BtMetric::TotalReturn,
            vec![series.clone()],
            true,
            12,
            6,
            12,
            ForecasterSpec::new(2, ForecasterKind::Holt),
        )
        .with_overfit_penalty(1.0);
        let unguarded = WalkForward::from_ensemble(
            BtMetric::TotalReturn,
            vec![series],
            true,
            12,
            6,
            12,
            ForecasterSpec::new(2, ForecasterKind::Holt),
        )
        .with_overfit_penalty(0.0);

        let g = guarded.score(&artifact);
        let u = unguarded.score(&artifact);
        assert!(g.is_finite() && u.is_finite(), "g={g} u={u}");
        // the held-out penalty must make the guarded (Maximize) score no better than unguarded,
        // and strictly worse when the even/odd gap is real.
        assert!(g <= u + 1e-12, "overfit guard must not inflate score: guarded={g} unguarded={u}");
        assert!(g < u, "with a real even/odd regime gap the guard must demote: guarded={g} unguarded={u}");
    }

    #[test]
    fn forecaster_spec_splits_artifact_correctly() {
        let spec = ForecasterSpec::new(2, ForecasterKind::Holt);
        let art = vec![0.7, -0.3, 0.55, 0.22];
        assert_eq!(spec.strategy_params(&art), &[0.7, -0.3]);
        assert_eq!(spec.forecaster_knobs(&art), (0.55, 0.22));
        // a bare [tilt, mom] artifact still yields sensible forecaster defaults.
        let bare = vec![1.0, 0.0];
        assert_eq!(spec.forecaster_knobs(&bare), (0.4, 0.1));
    }

    #[test]
    fn empty_series_set_scores_nan() {
        let wf = WalkForward::from_ensemble(
            BtMetric::Sharpe,
            vec![],
            true,
            10,
            5,
            5,
            ForecasterSpec::default(),
        );
        assert!(wf.score(&vec![1.0, 0.0, 0.4, 0.2]).is_nan());
        assert_eq!(wf.metric().direction, Direction::Maximize);
    }

    #[test]
    fn implements_evaluator_and_drops_into_run() {
        // prove WalkForward is a drop-in Evaluator by running the keep-or-revert loop on it.
        use crate::attempt_log::{AttemptLog, NoopRatchet};
        use crate::engine::{Run, StopCondition};
        use crate::proposer::LocalProposer;
        let wf = WalkForward::new(
            BtMetric::TotalReturn,
            rising_series(60),
            true,
            12,
            6,
            6,
            ForecasterSpec::new(2, ForecasterKind::Holt),
        );
        let prop = LocalProposer::new(0.3, 7).with_bounds(-1.0, 1.0);
        let mut run = Run::new(vec![-1.0, 0.0, 0.4, 0.2], wf, prop, NoopRatchet::new(), AttemptLog::new())
            .with_goal("tune strategy+forecaster via walk-forward");
        run.establish_baseline();
        let start = run.baseline_value();
        let results = run.run_loop(StopCondition::MaxIterations(120));
        assert_eq!(results.len(), 120);
        // baseline (Maximize total_return) should ratchet UP from a short-tilt start on a riser.
        assert!(run.baseline_value() >= start, "walk-forward ratchet regressed: {start} -> {}", run.baseline_value());
        assert!(run.n_kept() >= 1, "expected at least one kept improvement");
    }
}
