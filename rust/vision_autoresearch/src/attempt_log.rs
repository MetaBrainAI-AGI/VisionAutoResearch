//! attempt_log.rs — the attempt log (Karpathy's `results.tsv`) + the git RATCHET.
//!
//! Two outputs of the loop (KB autoresearch.md):
//!   1. the GIT HISTORY = validated wins only (kept commits) — the linear ratchet, and
//!   2. the ATTEMPT LOG = the full attempt record (hash, metric, peak_mem, status, desc),
//!      git-UNtracked. We persist it as JSONL (one row per attempt) for the dashboard
//!      `/api/autoresearch` passthrough and for the proposer to read recent history.
//!
//! The GIT RATCHET implements the verbatim rule: each candidate is `git commit`ed; on a KEEP
//! the commit advances the branch (new baseline); on a REVERT we `git reset HEAD~1`. The git
//! calls are isolated here behind `GitRatchet` so the pure loop core stays testable (a
//! `NoopRatchet` is provided for in-memory/test runs that don't touch a repo).

use crate::metric::KeepOrRevert;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One attempt row — the JSONL analogue of a `results.tsv` line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptRow {
    pub iter: usize,
    /// 7-char short commit hash (or "-" when not under git).
    pub commit7: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub baseline_value: f64,
    pub peak_mem_gb: f64,
    /// "keep" | "revert" | "crash".
    pub status: String,
    pub description: String,
    pub ts: u64,
}

impl AttemptRow {
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

fn now_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The attempt log: an in-memory ring of rows + an optional JSONL file it appends to.
pub struct AttemptLog {
    rows: Vec<AttemptRow>,
    path: Option<PathBuf>,
}

impl AttemptLog {
    pub fn new() -> Self {
        AttemptLog {
            rows: Vec::new(),
            path: None,
        }
    }

    /// Create a log that also appends every row to a JSONL file.
    pub fn with_file(path: impl Into<PathBuf>) -> Self {
        AttemptLog {
            rows: Vec::new(),
            path: Some(path.into()),
        }
    }

    /// Append a row (in-memory + file if configured). File errors are swallowed (the
    /// in-memory record is authoritative) — fail-open like the harness facades.
    pub fn append(&mut self, row: AttemptRow) {
        if let Some(p) = &self.path {
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "{}", row.to_json_line());
            }
        }
        self.rows.push(row);
    }

    /// The most recent `n` rows (newest last), for the proposer to read.
    pub fn recent(&self, n: usize) -> &[AttemptRow] {
        let len = self.rows.len();
        let start = len.saturating_sub(n);
        &self.rows[start..]
    }

    pub fn all(&self) -> &[AttemptRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn n_kept(&self) -> usize {
        self.rows.iter().filter(|r| r.status == "keep").count()
    }
}

impl Default for AttemptLog {
    fn default() -> Self {
        AttemptLog::new()
    }
}

/// The git ratchet abstraction: commit a candidate, then keep (advance) or revert (reset).
/// Isolated so the loop core is testable without a repo.
pub trait Ratchet: Send {
    /// Commit the current working tree as a candidate; returns the 7-char short hash.
    fn commit_candidate(&mut self, message: &str) -> String;
    /// Apply the keep-or-revert decision. KEEP = leave the commit (advance branch);
    /// REVERT = `git reset HEAD~1` back to the prior baseline.
    fn apply(&mut self, decision: KeepOrRevert) -> bool;
    /// The list of kept (validated-improvement) short hashes so far.
    fn kept_history(&self) -> Vec<String>;
}

/// A no-op ratchet for in-memory runs (no git). Records synthetic hashes so the attempt log
/// + tests still see a consistent shape.
pub struct NoopRatchet {
    counter: u64,
    last: String,
    kept: Vec<String>,
}

impl NoopRatchet {
    pub fn new() -> Self {
        NoopRatchet {
            counter: 0,
            last: "-------".to_string(),
            kept: Vec::new(),
        }
    }
}

impl Default for NoopRatchet {
    fn default() -> Self {
        NoopRatchet::new()
    }
}

impl Ratchet for NoopRatchet {
    fn commit_candidate(&mut self, _message: &str) -> String {
        self.counter += 1;
        // deterministic synthetic 7-char "hash"
        self.last = format!("{:07x}", self.counter & 0xFFFFFFF);
        self.last.clone()
    }
    fn apply(&mut self, decision: KeepOrRevert) -> bool {
        if decision.is_keep() {
            self.kept.push(self.last.clone());
        }
        true
    }
    fn kept_history(&self) -> Vec<String> {
        self.kept.clone()
    }
}

/// The real git ratchet — runs `git` in `repo` to commit/keep/reset, exactly mirroring the
/// autoresearch loop (commit candidate -> on keep leave it, on revert `git reset HEAD~1`).
/// All git invocations hardcode the program + argv (no shell interpolation) per the repo
/// security rule. Errors fall back to a synthetic hash and a logged failure (fail-open).
pub struct GitRatchet {
    repo: PathBuf,
    kept: Vec<String>,
    last: String,
}

impl GitRatchet {
    pub fn new(repo: impl Into<PathBuf>) -> Self {
        GitRatchet {
            repo: repo.into(),
            kept: Vec::new(),
            last: "-------".to_string(),
        }
    }

    fn git(&self, args: &[&str]) -> Option<String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(args)
            .output()
            .ok()?;
        if out.status.success() {
            Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
        } else {
            None
        }
    }

    fn short_head(&self) -> String {
        self.git(&["rev-parse", "--short=7", "HEAD"])
            .unwrap_or_else(|| "-------".to_string())
    }
}

impl Ratchet for GitRatchet {
    fn commit_candidate(&mut self, message: &str) -> String {
        // stage everything, then commit. `--allow-empty` so a no-diff candidate still records
        // a node in history (the attempt log still wants its hash); `--no-verify` would skip
        // hooks — we DO run hooks (commit discipline), so we omit it.
        let _ = self.git(&["add", "-A"]);
        let _ = self.git(&["commit", "--allow-empty", "-m", message]);
        self.last = self.short_head();
        self.last.clone()
    }
    fn apply(&mut self, decision: KeepOrRevert) -> bool {
        match decision {
            KeepOrRevert::Keep => {
                self.kept.push(self.last.clone());
                true
            }
            KeepOrRevert::Revert => {
                // git reset back to where we started (drop the candidate commit + its tree).
                self.git(&["reset", "--hard", "HEAD~1"]).is_some()
            }
        }
    }
    fn kept_history(&self) -> Vec<String> {
        self.kept.clone()
    }
}

/// Helper to build an AttemptRow from a completed evaluation.
#[allow(clippy::too_many_arguments)]
pub fn make_row(
    iter: usize,
    commit7: String,
    metric_name: &str,
    metric_value: f64,
    baseline_value: f64,
    peak_mem_gb: f64,
    decision: Option<KeepOrRevert>,
    crashed: bool,
    description: String,
) -> AttemptRow {
    let status = if crashed {
        "crash".to_string()
    } else {
        match decision {
            Some(KeepOrRevert::Keep) => "keep".to_string(),
            Some(KeepOrRevert::Revert) => "revert".to_string(),
            None => "revert".to_string(),
        }
    };
    AttemptRow {
        iter,
        commit7,
        metric_name: metric_name.to_string(),
        metric_value,
        baseline_value,
        peak_mem_gb,
        status,
        description,
        ts: now_ts(),
    }
}

/// Does `path` look like a git work-tree root we can ratchet on?
pub fn is_git_repo(path: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attempt_log_appends_and_recents() {
        let mut log = AttemptLog::new();
        for i in 0..5 {
            log.append(make_row(
                i,
                format!("{:07x}", i),
                "val_bpb",
                1.0 - i as f64 * 0.1,
                1.0,
                0.5,
                Some(KeepOrRevert::Keep),
                false,
                format!("attempt {}", i),
            ));
        }
        assert_eq!(log.len(), 5);
        assert_eq!(log.recent(2).len(), 2);
        assert_eq!(log.recent(2)[1].iter, 4);
        assert_eq!(log.n_kept(), 5);
    }

    #[test]
    fn noop_ratchet_tracks_kept_only() {
        let mut r = NoopRatchet::new();
        let h1 = r.commit_candidate("c1");
        assert!(r.apply(KeepOrRevert::Keep));
        let _h2 = r.commit_candidate("c2");
        assert!(r.apply(KeepOrRevert::Revert));
        let kept = r.kept_history();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0], h1);
    }

    #[test]
    fn row_status_reflects_decision_and_crash() {
        let keep = make_row(0, "a".into(), "m", 0.5, 1.0, 0.0, Some(KeepOrRevert::Keep), false, "d".into());
        assert_eq!(keep.status, "keep");
        let rev = make_row(0, "a".into(), "m", 1.5, 1.0, 0.0, Some(KeepOrRevert::Revert), false, "d".into());
        assert_eq!(rev.status, "revert");
        let crash = make_row(0, "-".into(), "m", f64::NAN, 1.0, 0.0, None, true, "boom".into());
        assert_eq!(crash.status, "crash");
    }

    #[test]
    fn json_line_roundtrips() {
        let row = make_row(2, "abc1234".into(), "sharpe", 1.23, 1.0, 0.7, Some(KeepOrRevert::Keep), false, "x".into());
        let line = row.to_json_line();
        let back: AttemptRow = serde_json::from_str(&line).unwrap();
        assert_eq!(back.commit7, "abc1234");
        assert_eq!(back.status, "keep");
        assert!((back.metric_value - 1.23).abs() < 1e-12);
    }
}
