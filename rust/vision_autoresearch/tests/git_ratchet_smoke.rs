//! Integration smoke test for the REAL GitRatchet against a throwaway temp git repo.
//! Proves: commit_candidate advances HEAD, KEEP leaves the commit (history grows), REVERT
//! `git reset --hard HEAD~1` drops the candidate commit (history shrinks back). This is the
//! one path the unit tests deliberately stub with NoopRatchet (to avoid touching the repo).

use std::process::Command;
use vision_autoresearch::attempt_log::{is_git_repo, GitRatchet, Ratchet};
use vision_autoresearch::metric::KeepOrRevert;

fn git(repo: &std::path::Path, args: &[&str]) {
    let ok = Command::new("git").arg("-C").arg(repo).args(args).output().unwrap().status.success();
    assert!(ok, "git {:?} failed", args);
}

fn count_commits(repo: &std::path::Path) -> usize {
    let out = Command::new("git").arg("-C").arg(repo)
        .args(["rev-list", "--count", "HEAD"]).output().unwrap();
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
}

#[test]
fn git_ratchet_keep_advances_revert_resets() {
    // unique temp dir
    let mut dir = std::env::temp_dir();
    dir.push(format!("var_git_ratchet_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    git(&dir, &["init", "-q"]);
    git(&dir, &["config", "user.email", "t@t.t"]);
    git(&dir, &["config", "user.name", "t"]);
    git(&dir, &["config", "commit.gpgsign", "false"]);
    std::fs::write(dir.join("artifact.txt"), "base").unwrap();
    git(&dir, &["add", "-A"]);
    git(&dir, &["commit", "-q", "-m", "base"]);

    assert!(is_git_repo(&dir));
    let base_count = count_commits(&dir);

    let mut r = GitRatchet::new(dir.clone());

    // candidate 1: write a change, commit, KEEP -> history grows by 1.
    std::fs::write(dir.join("artifact.txt"), "candidate-1").unwrap();
    let h1 = r.commit_candidate("autoresearch[0]: c1");
    assert_eq!(h1.len(), 7, "short hash should be 7 chars, got {:?}", h1);
    assert!(r.apply(KeepOrRevert::Keep));
    assert_eq!(count_commits(&dir), base_count + 1, "KEEP should advance history");

    // candidate 2: write a change, commit, REVERT -> history back to base_count+1.
    std::fs::write(dir.join("artifact.txt"), "candidate-2").unwrap();
    let _h2 = r.commit_candidate("autoresearch[1]: c2");
    assert_eq!(count_commits(&dir), base_count + 2);
    assert!(r.apply(KeepOrRevert::Revert), "revert should run git reset");
    assert_eq!(count_commits(&dir), base_count + 1, "REVERT should drop the candidate commit");

    // working tree was reset to the kept candidate-1 content.
    let content = std::fs::read_to_string(dir.join("artifact.txt")).unwrap();
    assert_eq!(content, "candidate-1", "reset --hard should restore kept tree");

    // kept_history == only the kept hash.
    let kept = r.kept_history();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0], h1);

    let _ = std::fs::remove_dir_all(&dir);
}
