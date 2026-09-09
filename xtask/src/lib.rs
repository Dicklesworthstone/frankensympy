//! Source identity primitives for development gate receipts, not attestation.
//! The runner and separate validator share encoding, never execution verdicts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSnapshot {
    pub commit: String,
    pub tree: String,
    pub inputs_digest: String,
    pub files: usize,
}

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has workspace parent")
        .to_path_buf()
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "git failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

pub fn file_digest(path: &Path) -> Result<String, String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > 256 * 1024 * 1024 {
        return Err(format!("not a bounded regular input: {}", path.display()));
    }
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(&mut file).map_err(|e| e.to_string())?;
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn source_snapshot(root: &Path) -> Result<SourceSnapshot, String> {
    // Deliberately excludes gate output receipts and tracker state. Includes
    // untracked source so a worktree overlay cannot masquerade as clean HEAD.
    // External path dependencies and machine execution attestation remain
    // separate obligations: this is a repository-input identity, not a release.
    let files = git(
        root,
        &[
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            // Cargo discovers these root-package inputs without explicit
            // manifest target entries, so additions must invalidate receipts.
            "build.rs",
            "tests",
            "benches",
            "examples",
            ".cargo",
            "crates",
            "src",
            "python",
            "tools",
            "scripts",
            "registries",
            "xtask",
            "artifacts/conformance",
        ],
    )?;
    let mut hashes = BTreeMap::new();
    for name in files.split('\0').filter(|name| !name.is_empty()) {
        if hashes.len() >= 100_000 {
            return Err("source file count exceeds limit".into());
        }
        hashes.insert(name.to_string(), file_digest(&root.join(name))?);
    }
    if hashes.is_empty() || hashes.len() > 100_000 {
        return Err("invalid source file count".into());
    }
    for required in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        if !hashes.contains_key(required) {
            return Err(format!("missing required source input: {required}"));
        }
    }
    let extension = root.join("python/fsym_python.so");
    if extension.exists() {
        hashes.insert("python/fsym_python.so".into(), file_digest(&extension)?);
    }
    let raw = serde_json::to_vec(&hashes).map_err(|e| e.to_string())?;
    Ok(SourceSnapshot {
        commit: git(root, &["rev-parse", "HEAD"])?.trim().to_string(),
        tree: git(root, &["rev-parse", "HEAD^{tree}"])?.trim().to_string(),
        inputs_digest: blake3::hash(&raw).to_hex().to_string(),
        files: hashes.len(),
    })
}

pub fn profile_digest(root: &Path, profile: &str) -> Result<String, String> {
    if profile.is_empty()
        || !profile
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-.".contains(&b))
    {
        return Err("unsafe profile ID".into());
    }
    file_digest(
        &root
            .join("tools/conformance-lab/profiles")
            .join(format!("{profile}.toml")),
    )
}
