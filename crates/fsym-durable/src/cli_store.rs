//! Pinned FrankenSQLite CLI storage lane (`fra-rc-durable-m5e` adapter).
//!
//! Implements the same [`DurableStore`] contract as the canonical file lane
//! on top of the pinned `fsqlite` CLI binary (registries/dependencies.toml:
//! frankensqlite @ abdc4dc8), following the fsym-formal external-tool
//! precedent: the crate carries NO library dependency on frankensqlite —
//! every operation is one owned `-batch -c` subprocess over a single
//! file-backed database, and the binary path is supplied by the caller.
//! The database is a container: digest/schema/universe/dependency
//! validation happens in this crate after readback, and mathematical
//! re-verification remains the runtime resume path's obligation.
//!
//! Rows store the record wire bytes hex-encoded (all inlined SQL literals
//! are hex or enforced-safe charsets, so no quoting can reach the SQL
//! layer). Publication is a single transaction (INSERT INTO committed +
//! DELETE FROM staging), matching the file lane's atomic-rename
//! guarantee. The caller builds the binary from the pinned worktree with
//! `cargo build -p fsqlite-cli --locked -F fsqlite-core/native`; the
//! default build's pager is memory-only and rejects file-backed handles.

#![forbid(unsafe_code)]

use crate::file_store::{CLEANUP_RESERVE_LIMIT, safe_schema_segment};
use crate::store::{DurableStore, PreparedHandle};
use crate::{DurableError, DurableRecord, hex_decode, hex_lower};
use std::path::PathBuf;
use std::process::Command;

pub struct FsqliteCliStore {
    cli: PathBuf,
    db: PathBuf,
}

const STAGING: &str = "staging";
const COMMITTED: &str = "committed";

impl FsqliteCliStore {
    /// Opens (and lazily initializes) the records database. The CLI binary
    /// must exist and be executable; the database file's parent directory
    /// must exist.
    pub fn open(cli: impl Into<PathBuf>, db: impl Into<PathBuf>) -> Result<Self, DurableError> {
        let cli = cli.into();
        let db = db.into();
        if !cli.is_file() {
            return Err(DurableError::Backend(format!(
                "pinned fsqlite CLI not found at {}",
                cli.display()
            )));
        }
        if let Some(parent) = db.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                DurableError::Backend(format!(
                    "cannot create database parent {}: {error}",
                    parent.display()
                ))
            })?;
        }
        let store = Self { cli, db };
        store.run(
            "CREATE TABLE IF NOT EXISTS durable_records (scope TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL, wire TEXT NOT NULL, PRIMARY KEY (scope, kind, name));",
        )?;
        Ok(store)
    }

    /// One owned synchronous `-batch -c` subprocess. Returns stdout lines
    /// with the CLI's single-quote value wrapping stripped.
    fn run(&self, sql: &str) -> Result<Vec<String>, DurableError> {
        let output = Command::new(&self.cli)
            .arg(&self.db)
            .arg("-batch")
            .arg("-c")
            .arg(sql)
            .output()
            .map_err(|error| DurableError::Backend(format!("CLI spawn failed: {error}")))?;
        if !output.status.success() {
            return Err(DurableError::Backend(format!(
                "CLI exited {:?}: {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                line.strip_prefix('\'')
                    .and_then(|rest| rest.strip_suffix('\''))
                    .unwrap_or(line)
                    .to_string()
            })
            .collect())
    }

    fn scope(universe_id: [u8; 32], payload_schema: &str) -> Result<String, DurableError> {
        Ok(format!(
            "{}_{}",
            hex_lower(&universe_id),
            safe_schema_segment(payload_schema)?
        ))
    }

    fn staging_count(&self, scope: &str) -> Result<usize, DurableError> {
        let rows = self.run(&format!(
            "SELECT COUNT(*) FROM durable_records WHERE scope='{scope}' AND kind='{STAGING}';"
        ))?;
        rows.first()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| DurableError::Backend("staging COUNT returned no parsable row".into()))
    }

    fn select_staged(
        &self,
        scope: &str,
        handle: &PreparedHandle,
    ) -> Result<DurableRecord, DurableError> {
        let name = handle.as_str();
        let rows = self.run(&format!(
            "SELECT wire FROM durable_records WHERE scope='{scope}' AND kind='{STAGING}' AND name='{name}';"
        ))?;
        let wire_hex = rows.into_iter().next().ok_or(DurableError::Absent)?;
        let wire = crate::hex_decode(&wire_hex)?;
        DurableRecord::from_wire(&wire)
    }
}

impl DurableStore for FsqliteCliStore {
    fn prepare(&self, record: &DurableRecord) -> Result<PreparedHandle, DurableError> {
        let scope = Self::scope(record.universe_id, &record.payload_schema)?;
        let held = self.staging_count(&scope)?;
        if held >= CLEANUP_RESERVE_LIMIT {
            return Err(DurableError::InsufficientCleanupReserve {
                held,
                limit: CLEANUP_RESERVE_LIMIT,
            });
        }
        let handle = PreparedHandle {
            name: format!(
                "{}-{}.record",
                hex_lower(&record.record_digest),
                record.payload.len()
            ),
        };
        let wire_hex = hex_lower(&record.to_wire()?);
        self.run(&format!(
            "INSERT INTO durable_records (scope, kind, name, wire) VALUES ('{scope}', '{STAGING}', '{}', '{wire_hex}');",
            handle.as_str()
        ))?;
        Ok(handle)
    }

    fn verify_prepared(&self, handle: &PreparedHandle) -> Result<DurableRecord, DurableError> {
        // Handles are content-addressed names; the staging scan mirrors the
        // file lane (first match wins, digests verified on readback).
        let scopes = self.run(&format!(
            "SELECT DISTINCT scope FROM durable_records WHERE kind='{STAGING}' AND name='{}';",
            handle.as_str()
        ))?;
        for scope in scopes {
            if let Ok(record) = self.select_staged(&scope, handle) {
                return Ok(record);
            }
        }
        Err(DurableError::UnknownHandle(handle.name.clone()))
    }

    fn commit(&self, handle: &PreparedHandle) -> Result<(), DurableError> {
        let record = self.verify_prepared(handle)?;
        let scope = Self::scope(record.universe_id, &record.payload_schema)?;
        let name = handle.as_str();
        // Idempotent re-commit of the identical content-addressed record.
        let committed: Vec<String> = self.run(&format!(
            "SELECT wire FROM durable_records WHERE scope='{scope}' AND kind='{COMMITTED}' AND name='{name}';"
        ))?;
        if let Some(wire_hex) = committed.into_iter().next() {
            let wire = crate::hex_decode(&wire_hex)?;
            return if DurableRecord::from_wire(&wire)? == record {
                Ok(())
            } else {
                Err(DurableError::Backend(
                    "committed row exists at this handle with divergent bytes".into(),
                ))
            };
        }
        self.run(&format!(
            "BEGIN; INSERT INTO durable_records (scope, kind, name, wire) SELECT scope, '{COMMITTED}', name, wire FROM durable_records WHERE scope='{scope}' AND kind='{STAGING}' AND name='{name}'; DELETE FROM durable_records WHERE scope='{scope}' AND kind='{STAGING}' AND name='{name}'; COMMIT;"
        ))?;
        Ok(())
    }

    fn load_committed(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<Option<DurableRecord>, DurableError> {
        let scope = Self::scope(universe_id, payload_schema)?;
        let rows = self.run(&format!(
            "SELECT wire FROM durable_records WHERE scope='{scope}' AND kind='{COMMITTED}' ORDER BY name;"
        ))?;
        match rows.into_iter().next() {
            Some(wire_hex) => Ok(Some(DurableRecord::from_wire(&hex_decode(&wire_hex)?)?)),
            None => Ok(None),
        }
    }

    fn release(&self, handle: &PreparedHandle) -> Result<(), DurableError> {
        self.run(&format!(
            "DELETE FROM durable_records WHERE kind='{STAGING}' AND name='{}';",
            handle.as_str()
        ))?;
        Ok(())
    }

    fn staging_len(&self) -> Result<usize, DurableError> {
        let rows = self.run(&format!(
            "SELECT COUNT(*) FROM durable_records WHERE kind='{STAGING}';"
        ))?;
        rows.first()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| DurableError::Backend("staging COUNT returned no parsable row".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DependencyManifest;
    use std::path::Path;

    fn cli_path() -> Option<PathBuf> {
        if let Ok(env) = std::env::var("FSQLITE_CLI_BIN")
            && Path::new(&env).is_file()
        {
            return Some(PathBuf::from(env));
        }
        for candidate in [
            "/data/tmp/cargo-target/debug/fsqlite",
            "/data/projects/frankensqlite-pin/target/debug/fsqlite",
        ] {
            if Path::new(candidate).is_file() {
                return Some(PathBuf::from(candidate));
            }
        }
        None
    }

    fn unique_db(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fsym-cli-store-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join("records.db")
    }

    fn sample(universe: [u8; 32], payload: &[u8]) -> DurableRecord {
        DurableRecord::new(
            universe,
            "fsym.portfolio.factor_race.continuation.v1",
            payload.to_vec(),
            DependencyManifest::empty(),
        )
        .expect("record")
    }

    #[test]
    fn prepare_verify_commit_load_round_trip_through_pinned_cli() {
        let Some(cli) = cli_path() else {
            eprintln!(
                "skipped: pinned fsqlite CLI binary not found (set FSQLITE_CLI_BIN or build \
                 the pinned worktree with -F fsqlite-core/native)"
            );
            return;
        };
        let db = unique_db("roundtrip");
        let store = FsqliteCliStore::open(&cli, &db).expect("store");
        let record = sample([7; 32], b"cli-payload");
        let handle = store.prepare(&record).expect("prepare");
        assert_eq!(store.staging_len().expect("len"), 1);
        let verified = store.verify_prepared(&handle).expect("verify");
        assert_eq!(verified, record);
        store.commit(&handle).expect("commit");
        assert_eq!(store.staging_len().expect("len"), 0);
        // Actual disk readback through a brand-new store instance over the
        // same database file: persistence, not in-process state.
        let reopened = FsqliteCliStore::open(&cli, &db).expect("reopen");
        let loaded = reopened
            .load_committed(record.universe_id, &record.payload_schema)
            .expect("load")
            .expect("committed record");
        assert_eq!(loaded, record);
        assert!(
            reopened
                .load_committed([4; 32], &record.payload_schema)
                .expect("foreign universe")
                .is_none()
        );
        // Idempotent re-commit of the identical content-addressed record.
        let again = store.prepare(&record).expect("re-prepare");
        store.commit(&again).expect("idempotent re-commit");
        std::fs::remove_dir_all(db.parent().expect("parent")).expect("cleanup");
    }

    #[test]
    fn cleanup_reserve_is_enforced_until_release() {
        let Some(cli) = cli_path() else {
            eprintln!("skipped: pinned fsqlite CLI binary not found");
            return;
        };
        let db = unique_db("reserve");
        let store = FsqliteCliStore::open(&cli, &db).expect("store");
        let mut handles = Vec::new();
        for index in 0..CLEANUP_RESERVE_LIMIT {
            let record = DurableRecord::new(
                [6; 32],
                "fsym.portfolio.factor_race.continuation.v1",
                format!("payload-{index}").into_bytes(),
                DependencyManifest::empty(),
            )
            .expect("record");
            handles.push(store.prepare(&record).expect("prepare under reserve"));
        }
        let overflow = DurableRecord::new(
            [6; 32],
            "fsym.portfolio.factor_race.continuation.v1",
            b"payload-over".to_vec(),
            DependencyManifest::empty(),
        )
        .expect("record");
        assert!(matches!(
            store.prepare(&overflow),
            Err(DurableError::InsufficientCleanupReserve { .. })
        ));
        store.release(&handles[0]).expect("release");
        store.prepare(&overflow).expect("prepare after release");
        std::fs::remove_dir_all(db.parent().expect("parent")).expect("cleanup");
    }
}
