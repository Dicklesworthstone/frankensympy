//! Source identity primitives for development gate receipts, not attestation.
//! The runner and separate validator share encoding, never execution verdicts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::Read;
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
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let opened = file.metadata().map_err(|e| e.to_string())?;
    if !opened.is_file() || opened.len() > 256 * 1024 * 1024 {
        return Err(format!("not a bounded regular input: {}", path.display()));
    }
    digest_reader(file, 256 * 1024 * 1024)
}

fn digest_reader(reader: impl Read, limit: u64) -> Result<String, String> {
    // Metadata is only a preflight; a file can grow after that check. Read
    // at most one sentinel byte beyond the cap, and never return its digest.
    let mut reader = reader.take(limit.checked_add(1).ok_or("input byte limit overflow")?);
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_reader(&mut reader)
        .map_err(|e| e.to_string())?;
    if reader.limit() == 0 {
        return Err("input exceeds byte limit".into());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn text_reader(reader: impl Read, limit: u64) -> Result<String, String> {
    let mut reader = reader.take(limit.checked_add(1).ok_or("input byte limit overflow")?);
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    if reader.limit() == 0 {
        return Err("input exceeds byte limit".into());
    }
    Ok(text)
}

/// Load a development receipt with a cap on bytes read, not just metadata size.
pub fn read_receipt(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let opened = file.metadata().map_err(|e| e.to_string())?;
    if !opened.is_file() || opened.len() > 16 * 1024 * 1024 {
        return Err("receipt must be a regular file within 16 MiB".into());
    }
    text_reader(file, 16 * 1024 * 1024)
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

#[cfg(test)]
mod tests {
    use super::{digest_reader, text_reader};
    use std::io::{self, Cursor, Read};

    #[test]
    fn readers_enforce_actual_stream_length() {
        // A stream may be longer than the length admitted before reading.
        // Keep it finite so the unbounded implementation fails, not hangs.
        let bytes = [b'x'; 32];
        let mut input = Cursor::new(bytes);
        assert!(digest_reader(&mut input, 8).is_err());
        assert_eq!(input.position(), 9);
        input.set_position(0);
        assert!(text_reader(&mut input, 8).is_err());
        assert_eq!(input.position(), 9);
    }

    #[test]
    fn readers_preserve_boundary_values_and_errors() {
        for length in [0, 1, 8] {
            let bytes = vec![b'x'; length];
            assert_eq!(
                digest_reader(bytes.as_slice(), 8).unwrap(),
                blake3::hash(&bytes).to_hex().to_string()
            );
            assert_eq!(text_reader(bytes.as_slice(), 8).unwrap().as_bytes(), bytes);
        }
        assert!(text_reader([0xff].as_slice(), 8).is_err());
        assert_eq!(text_reader([].as_slice(), 0).unwrap(), "");
        assert!(text_reader(b"x".as_slice(), 0).is_err());
        assert!(digest_reader(b"x".as_slice(), 0).is_err());
        assert!(digest_reader([].as_slice(), u64::MAX).is_err());
        assert!(text_reader([].as_slice(), u64::MAX).is_err());
    }

    #[test]
    fn readers_propagate_io_errors() {
        struct FailedRead;
        impl Read for FailedRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("injected read failure"))
            }
        }
        assert!(
            digest_reader(FailedRead, 8)
                .unwrap_err()
                .contains("injected read failure")
        );
        assert!(
            text_reader(FailedRead, 8)
                .unwrap_err()
                .contains("injected read failure")
        );
    }
}
