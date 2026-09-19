//! Canonical file-backed reference lane with actual disk readback.
//!
//! Layout under one root directory:
//! ```text
//! <root>/<universe_hex>/<payload_schema>/staging/<handle>.record
//! <root>/<universe_hex>/<payload_schema>/committed/<handle>.record
//! ```
//! Commit is a same-filesystem rename of the staged file, so a reader either
//! observes the full previous state or the full new state. Every read path
//! re-reads the bytes from disk and re-validates; a stored digest flag alone
//! is never trusted. Staging is bounded by `CLEANUP_RESERVE_LIMIT`: when the
//! limit is held, `prepare` refuses with `InsufficientCleanupReserve` until
//! records are released.

use crate::store::{DurableStore, PreparedHandle};
use crate::{DurableError, DurableRecord, hex_lower};
use std::fs;
use std::path::{Path, PathBuf};

/// Staged records held per payload-schema namespace before prepare refuses.
pub const CLEANUP_RESERVE_LIMIT: usize = 8;

pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, DurableError> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|error| {
            DurableError::Backend(format!(
                "cannot open durable store root {}: {error}",
                root.display()
            ))
        })?;
        Ok(Self { root })
    }

    fn namespace(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<PathBuf, DurableError> {
        let dir = self
            .root
            .join(hex_lower(&universe_id))
            .join(safe_schema_segment(payload_schema)?);
        fs::create_dir_all(&dir).map_err(|error| {
            DurableError::Backend(format!(
                "cannot create namespace {}: {error}",
                dir.display()
            ))
        })?;
        Ok(dir)
    }

    fn staging_dir(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<PathBuf, DurableError> {
        let dir = self.namespace(universe_id, payload_schema)?.join("staging");
        fs::create_dir_all(&dir).map_err(|error| {
            DurableError::Backend(format!("cannot create staging {}: {error}", dir.display()))
        })?;
        Ok(dir)
    }

    fn committed_dir(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<PathBuf, DurableError> {
        let dir = self
            .namespace(universe_id, payload_schema)?
            .join("committed");
        fs::create_dir_all(&dir).map_err(|error| {
            DurableError::Backend(format!(
                "cannot create committed {}: {error}",
                dir.display()
            ))
        })?;
        Ok(dir)
    }
}

/// Payload schemas are bounded ids; only their safe subset may become a path
/// segment so a hostile schema id can never traverse out of the namespace.
pub(crate) fn safe_schema_segment(schema: &str) -> Result<String, DurableError> {
    if schema.is_empty() || schema.len() > 256 {
        return Err(DurableError::MalformedRecord(
            "payload schema id must contain 1..=256 bytes".into(),
        ));
    }
    let mut segment = String::with_capacity(schema.len());
    for character in schema.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
            segment.push(character);
        } else {
            return Err(DurableError::MalformedRecord(format!(
                "payload schema id contains a character that cannot be a store segment: {character:?}"
            )));
        }
    }
    Ok(segment)
}

fn read_record(path: &Path) -> Result<DurableRecord, DurableError> {
    let wire = fs::read(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => DurableError::Absent,
        _ => DurableError::Backend(format!("cannot read {}: {error}", path.display())),
    })?;
    DurableRecord::from_wire(&wire)
}

impl DurableStore for FileStore {
    fn prepare(&self, record: &DurableRecord) -> Result<PreparedHandle, DurableError> {
        let staging = self.staging_dir(record.universe_id, &record.payload_schema)?;
        let held = fs::read_dir(&staging)
            .map_err(|error| {
                DurableError::Backend(format!(
                    "cannot list staging {}: {error}",
                    staging.display()
                ))
            })?
            .filter_map(|entry| entry.ok())
            .count();
        if held >= CLEANUP_RESERVE_LIMIT {
            return Err(DurableError::InsufficientCleanupReserve {
                held,
                limit: CLEANUP_RESERVE_LIMIT,
            });
        }
        // Canonical handle: record digest hex plus payload length, so two
        // prepares of identical content collide harmlessly and any payload
        // change produces a distinct staging name.
        let handle = PreparedHandle {
            name: format!(
                "{}-{}.record",
                hex_lower(&record.record_digest),
                record.payload.len()
            ),
        };
        let target = staging.join(&handle.name);
        let wire = record.to_wire()?;
        // Write to a temp sibling, sync, then rename: the observed staging
        // name only ever contains fully written bytes.
        let temp = staging.join(format!(".tmp-{}", handle.name));
        {
            let written = fs::File::create(&temp).map_err(|error| {
                DurableError::Backend(format!("cannot write temp staging: {error}"))
            })?;
            use std::io::Write as _;
            let mut written = written;
            written.write_all(&wire).map_err(|error| {
                DurableError::Backend(format!("cannot write staged bytes: {error}"))
            })?;
            written.sync_all().map_err(|error| {
                DurableError::Backend(format!("cannot sync staged bytes: {error}"))
            })?;
        }
        fs::rename(&temp, &target).map_err(|error| {
            DurableError::Backend(format!("cannot finalize staging name: {error}"))
        })?;
        Ok(handle)
    }

    fn verify_prepared(&self, handle: &PreparedHandle) -> Result<DurableRecord, DurableError> {
        // The staging namespace is keyed by universe/schema; scan each staging
        // directory for the handle name. Handles are content-addressed, so a
        // scan is bounded by the cleanup reserve across namespaces.
        let mut found = None;
        for universe_dir in fs::read_dir(&self.root)
            .map_err(|error| DurableError::Backend(format!("cannot list root: {error}")))?
        {
            let universe_dir = universe_dir
                .map_err(|error| DurableError::Backend(format!("root entry: {error}")))?;
            let universe_path = universe_dir.path();
            if !universe_path.is_dir() {
                // Non-namespace files (run markers, sidecars) live in the
                // root alongside namespaces; they are never scanned.
                continue;
            }
            for schema_dir in fs::read_dir(&universe_path)
                .map_err(|error| DurableError::Backend(format!("cannot list namespace: {error}")))?
            {
                let schema_dir = schema_dir
                    .map_err(|error| DurableError::Backend(format!("namespace entry: {error}")))?;
                let candidate = schema_dir.path().join("staging").join(&handle.name);
                if candidate.is_file() {
                    found = Some(candidate);
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let path = found.ok_or_else(|| DurableError::UnknownHandle(handle.name.clone()))?;
        read_record(&path)
    }

    fn commit(&self, handle: &PreparedHandle) -> Result<(), DurableError> {
        let record = self.verify_prepared(handle)?;
        let staging = self.staging_dir(record.universe_id, &record.payload_schema)?;
        let committed = self.committed_dir(record.universe_id, &record.payload_schema)?;
        let source = staging.join(&handle.name);
        let target = committed.join(&handle.name);
        if target.exists() {
            // Identical content-addressed re-commit is idempotent; anything
            // else at that name would have failed verify above.
            return Ok(());
        }
        fs::rename(&source, &target)
            .map_err(|error| DurableError::Backend(format!("atomic commit rename failed: {error}")))
    }

    fn load_committed(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<Option<DurableRecord>, DurableError> {
        let committed = self
            .namespace(universe_id, payload_schema)?
            .join("committed");
        let entries = match fs::read_dir(&committed) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(DurableError::Backend(format!(
                    "cannot list committed {}: {error}",
                    committed.display()
                )));
            }
        };
        let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
        for entry in entries.filter_map(|entry| entry.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let modified = fs::metadata(&path)
                .and_then(|meta| meta.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if best.as_ref().is_none_or(|(time, _)| modified > *time) {
                best = Some((modified, path));
            }
        }
        match best {
            Some((_, path)) => Ok(Some(read_record(&path)?)),
            None => Ok(None),
        }
    }

    fn release(&self, handle: &PreparedHandle) -> Result<(), DurableError> {
        // Release scans staging namespaces exactly like verify; a released
        // handle that is absent is a no-op success.
        for universe_dir in fs::read_dir(&self.root)
            .map_err(|error| DurableError::Backend(format!("cannot list root: {error}")))?
        {
            let universe_dir = universe_dir
                .map_err(|error| DurableError::Backend(format!("root entry: {error}")))?;
            let universe_path = universe_dir.path();
            if !universe_path.is_dir() {
                continue;
            }
            for schema_dir in fs::read_dir(&universe_path)
                .map_err(|error| DurableError::Backend(format!("cannot list namespace: {error}")))?
            {
                let schema_dir = schema_dir
                    .map_err(|error| DurableError::Backend(format!("namespace entry: {error}")))?;
                let candidate = schema_dir.path().join("staging").join(&handle.name);
                if candidate.is_file() {
                    fs::remove_file(&candidate).map_err(|error| {
                        DurableError::Backend(format!("cannot release staged record: {error}"))
                    })?;
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn staging_len(&self) -> Result<usize, DurableError> {
        let mut total = 0;
        for universe_dir in fs::read_dir(&self.root)
            .map_err(|error| DurableError::Backend(format!("cannot list root: {error}")))?
        {
            let universe_dir = universe_dir
                .map_err(|error| DurableError::Backend(format!("root entry: {error}")))?;
            let universe_path = universe_dir.path();
            if !universe_path.is_dir() {
                continue;
            }
            for schema_dir in fs::read_dir(&universe_path)
                .map_err(|error| DurableError::Backend(format!("cannot list namespace: {error}")))?
            {
                let schema_dir = schema_dir
                    .map_err(|error| DurableError::Backend(format!("namespace entry: {error}")))?;
                let staging = schema_dir.path().join("staging");
                if let Ok(entries) = fs::read_dir(&staging) {
                    total += entries.filter_map(|entry| entry.ok()).count();
                }
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DependencyManifest;

    fn unique_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fsym-durable-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create temp root");
        dir
    }

    fn sample() -> DurableRecord {
        DurableRecord::new(
            [3; 32],
            "fsym.portfolio.factor_race.continuation.v1",
            b"durable-payload".to_vec(),
            DependencyManifest::new([("pin", "v1")]),
        )
        .expect("record")
    }

    #[test]
    fn prepare_verify_commit_and_load_round_trip_from_disk() {
        let root = unique_root("roundtrip");
        let store = FileStore::open(&root).expect("store");
        let record = sample();
        let handle = store.prepare(&record).expect("prepare");
        assert_eq!(store.staging_len().expect("len"), 1);
        let verified = store.verify_prepared(&handle).expect("verify");
        assert_eq!(verified, record);
        store.commit(&handle).expect("commit");
        assert_eq!(store.staging_len().expect("len"), 0);
        let loaded = store
            .load_committed(record.universe_id, &record.payload_schema)
            .expect("load")
            .expect("committed record present");
        assert_eq!(loaded, record);
        assert!(
            store
                .load_committed([4; 32], &record.payload_schema)
                .expect("load foreign universe")
                .is_none()
        );
        fs::remove_dir_all(&root).expect("cleanup temp root");
    }

    #[test]
    fn cleanup_reserve_is_enforced_until_release() {
        let root = unique_root("reserve");
        let store = FileStore::open(&root).expect("store");
        let mut handles = Vec::new();
        for index in 0..CLEANUP_RESERVE_LIMIT {
            let record = DurableRecord::new(
                [5; 32],
                "fsym.portfolio.factor_race.continuation.v1",
                format!("payload-{index}").into_bytes(),
                DependencyManifest::empty(),
            )
            .expect("record");
            handles.push(store.prepare(&record).expect("prepare under reserve"));
        }
        let record = DurableRecord::new(
            [5; 32],
            "fsym.portfolio.factor_race.continuation.v1",
            b"payload-over".to_vec(),
            DependencyManifest::empty(),
        )
        .expect("record");
        assert!(matches!(
            store.prepare(&record),
            Err(DurableError::InsufficientCleanupReserve { .. })
        ));
        store.release(&handles[0]).expect("release");
        store.prepare(&record).expect("prepare after release");
        fs::remove_dir_all(&root).expect("cleanup temp root");
    }

    #[test]
    fn hostile_schema_segment_is_refused() {
        assert!(safe_schema_segment("../../etc").is_err());
        assert!(safe_schema_segment("fsym.portfolio.factor_race.continuation.v1").is_ok());
    }
}
