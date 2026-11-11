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
use crate::{DurableError, DurableRecord, MAX_RECORD_WIRE_BYTES, hex_decode, hex_lower};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Staged records held per payload-schema namespace before prepare refuses.
pub const CLEANUP_RESERVE_LIMIT: usize = 8;
const MAX_SCAN_ENTRIES: usize = 16_384;

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
        let root = fs::canonicalize(&root).map_err(|error| {
            DurableError::Backend(format!("cannot resolve store root: {error}"))
        })?;
        sync_directory(&root)?;
        if let Some(parent) = root.parent() {
            sync_directory(parent)?;
        }
        Ok(Self { root })
    }

    fn namespace(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<PathBuf, DurableError> {
        let schema = safe_schema_segment(payload_schema)?;
        let universe = self.root.join(hex_lower(&universe_id));
        ensure_directory(&universe)?;
        let dir = universe.join(schema);
        ensure_directory(&dir)?;
        Ok(dir)
    }

    fn staging_dir(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<PathBuf, DurableError> {
        let dir = self.namespace(universe_id, payload_schema)?.join("staging");
        ensure_directory(&dir)?;
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
        ensure_directory(&dir)?;
        Ok(dir)
    }

    /// Load the exact published record named by an immutable content digest.
    /// This bounded discovery API never chooses a checkpoint by filesystem
    /// iteration order or modification time. Callers that know the namespace
    /// and payload length should use [`Self::load_committed_exact`] instead.
    pub fn load_committed_record(
        &self,
        record_digest: [u8; 32],
    ) -> Result<DurableRecord, DurableError> {
        let prefix = format!("{}-", hex_lower(&record_digest));
        let mut found = None;
        let mut scanned = 0usize;
        self.visit_namespaces(|universe, schema, namespace| {
            let directory = namespace.join("committed");
            if !existing_directory(&directory)? {
                return Ok(());
            }
            for entry in read_directory(&directory)? {
                scan_step(&mut scanned)?;
                let entry = entry.map_err(directory_error)?;
                let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                    continue;
                };
                if !name.starts_with(&prefix) {
                    continue;
                }
                let handle = PreparedHandle::from_name(name);
                let record = read_bound_record(&entry.path(), universe, schema, &handle)?;
                if found.as_ref().is_some_and(|prior| prior != &record) {
                    return Err(DurableError::MalformedRecord(
                        "ambiguous committed record digest".into(),
                    ));
                }
                found = Some(record);
            }
            Ok(())
        })?;
        found.ok_or(DurableError::Absent)
    }

    /// Load one known immutable record without scanning namespaces or files.
    /// Every supplied key component is checked against the actual decoded
    /// record; callers still validate dependencies and mathematical contents.
    /// Reads additionally allow at most `4 * payload_len + 4096` wire bytes,
    /// bounding byte-array encoding plus the small manifest used by this API.
    /// Larger metadata envelopes require the general digest lookup instead.
    pub fn load_committed_exact(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
        record_digest: [u8; 32],
        payload_len: usize,
    ) -> Result<DurableRecord, DurableError> {
        let schema = safe_schema_segment(payload_schema)?;
        let handle = PreparedHandle::from_name(format!(
            "{}-{payload_len}.record",
            hex_lower(&record_digest)
        ));
        handle.validate()?;
        let universe = self.root.join(hex_lower(&universe_id));
        let namespace = universe.join(schema);
        let committed = namespace.join("committed");
        for directory in [&universe, &namespace, &committed] {
            if !existing_directory(directory)? {
                return Err(DurableError::Absent);
            }
        }
        read_bound_record_with_limit(
            &committed.join(handle.as_str()),
            universe_id,
            payload_schema,
            &handle,
            small_wire_limit(payload_len),
        )
    }

    /// Read back one caller-known staged record without namespace discovery.
    /// A re-digested substitution is rejected against the complete expected
    /// record, not only against metadata retained by the backend.
    pub fn verify_prepared_record(
        &self,
        expected: &DurableRecord,
    ) -> Result<DurableRecord, DurableError> {
        let handle = expected_handle(expected)?;
        let staging = self.record_area(expected, "staging")?;
        let readback = read_bound_record_with_limit(
            &staging.join(handle.as_str()),
            expected.universe_id,
            &expected.payload_schema,
            &handle,
            small_wire_limit(expected.payload.len()),
        )?;
        require_expected_record(readback, expected)
    }

    /// Publish one caller-known staged record using direct bounded readback.
    /// Repeating a completed commit revalidates the exact committed bytes.
    pub fn commit_record(&self, expected: &DurableRecord) -> Result<(), DurableError> {
        let handle = expected_handle(expected)?;
        let staging = match self.record_area(expected, "staging") {
            Ok(staging) => staging,
            Err(DurableError::Absent) => {
                return self.verify_committed_retry(expected);
            }
            Err(error) => return Err(error),
        };
        match self.verify_prepared_record(expected) {
            Ok(_) => {}
            Err(DurableError::Absent) => return self.verify_committed_retry(expected),
            Err(error) => return Err(error),
        }
        self.publish_record(
            &staging.join(handle.as_str()),
            expected,
            &handle,
            small_wire_limit(expected.payload.len()),
        )
    }

    /// Release only a caller-known staging path, without scanning unrelated
    /// namespaces or trusting the candidate bytes stored at that path.
    pub fn release_prepared_record(&self, expected: &DurableRecord) -> Result<(), DurableError> {
        let handle = expected_handle(expected)?;
        let staging = match self.record_area(expected, "staging") {
            Ok(staging) => staging,
            Err(DurableError::Absent) => return Ok(()),
            Err(error) => return Err(error),
        };
        release_staging_file(&staging, &handle)
    }

    fn record_area(&self, record: &DurableRecord, area: &str) -> Result<PathBuf, DurableError> {
        let schema = safe_schema_segment(&record.payload_schema)?;
        let universe = self.root.join(hex_lower(&record.universe_id));
        let namespace = universe.join(schema);
        let area = namespace.join(area);
        for directory in [&universe, &namespace, &area] {
            if !existing_directory(directory)? {
                return Err(DurableError::Absent);
            }
        }
        Ok(area)
    }

    fn verify_committed_retry(&self, expected: &DurableRecord) -> Result<(), DurableError> {
        let record = self.load_committed_exact(
            expected.universe_id,
            &expected.payload_schema,
            expected.record_digest,
            expected.payload.len(),
        )?;
        require_expected_record(record, expected)?;
        let committed = self.record_area(expected, "committed")?;
        let handle = expected_handle(expected)?;
        sync_file(&committed.join(handle.as_str()))?;
        sync_directory(&committed)
    }

    fn publish_record(
        &self,
        source: &Path,
        record: &DurableRecord,
        handle: &PreparedHandle,
        wire_limit: usize,
    ) -> Result<(), DurableError> {
        let staging = source
            .parent()
            .ok_or_else(|| DurableError::MalformedRecord("prepared record has no parent".into()))?;
        let committed = self.committed_dir(record.universe_id, &record.payload_schema)?;
        let target = committed.join(handle.as_str());
        match read_bound_record_with_limit(
            &target,
            record.universe_id,
            &record.payload_schema,
            handle,
            wire_limit,
        ) {
            Ok(existing) => {
                require_expected_record(existing, record)?;
                sync_file(&target)?;
                fs::remove_file(source).map_err(directory_error)?;
                sync_directory(&committed)?;
                sync_directory(staging)?;
                return Ok(());
            }
            Err(DurableError::Absent) => {}
            Err(error) => return Err(error),
        }
        // A recovered or idempotently prepared staging file is read back just
        // like a fresh one; flush its actual contents before moving its name.
        sync_file(source)?;
        fs::rename(source, &target).map_err(|error| {
            DurableError::Backend(format!("atomic commit rename failed: {error}"))
        })?;
        sync_directory(&committed)?;
        sync_directory(staging)
    }

    fn visit_namespaces(
        &self,
        mut visit: impl FnMut([u8; 32], &str, &Path) -> Result<(), DurableError>,
    ) -> Result<(), DurableError> {
        let mut scanned = 0usize;
        for entry in read_directory(&self.root)? {
            scan_step(&mut scanned)?;
            let entry = entry.map_err(directory_error)?;
            let Some(universe) = parse_universe(&entry.file_name().to_string_lossy()) else {
                continue;
            };
            let universe_path = entry.path();
            if !existing_directory(&universe_path)? {
                continue;
            }
            for entry in read_directory(&universe_path)? {
                scan_step(&mut scanned)?;
                let entry = entry.map_err(directory_error)?;
                let schema = entry.file_name();
                let schema = schema.to_str().ok_or_else(|| {
                    DurableError::MalformedRecord("non-UTF-8 durable schema directory".into())
                })?;
                safe_schema_segment(schema)?;
                let namespace = entry.path();
                if existing_directory(&namespace)? {
                    visit(universe, schema, &namespace)?;
                }
            }
        }
        Ok(())
    }

    fn prepared_record(
        &self,
        handle: &PreparedHandle,
    ) -> Result<(PathBuf, DurableRecord), DurableError> {
        handle.validate()?;
        let mut found = None;
        self.visit_namespaces(|universe, schema, namespace| {
            let staging = namespace.join("staging");
            if !existing_directory(&staging)? {
                return Ok(());
            }
            let path = staging.join(handle.as_str());
            match read_bound_record(&path, universe, schema, handle) {
                Ok(record) => {
                    if found.is_some() {
                        return Err(DurableError::MalformedRecord(
                            "ambiguous prepared record handle".into(),
                        ));
                    }
                    found = Some((path, record));
                }
                Err(DurableError::Absent) => {}
                Err(error) => return Err(error),
            }
            Ok(())
        })?;
        found.ok_or_else(|| DurableError::UnknownHandle(handle.name.clone()))
    }
}

/// Payload schemas are bounded ids; only their safe subset may become a path
/// segment so a hostile schema id can never traverse out of the namespace.
pub(crate) fn safe_schema_segment(schema: &str) -> Result<String, DurableError> {
    if schema.is_empty() || schema.len() > 256 || matches!(schema, "." | "..") {
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

fn read_record_bounded(path: &Path, wire_limit: usize) -> Result<DurableRecord, DurableError> {
    let metadata = fs::symlink_metadata(path).map_err(read_error)?;
    if !metadata.file_type().is_file() {
        return Err(DurableError::MalformedRecord(
            "durable record path must be a regular file".into(),
        ));
    }
    if metadata.len() > wire_limit as u64 {
        return Err(DurableError::RecordTooLarge {
            actual: usize::try_from(metadata.len()).unwrap_or(usize::MAX),
            limit: wire_limit,
        });
    }
    let file = fs::File::open(path).map_err(read_error)?;
    let mut wire = Vec::new();
    // A file can grow after metadata inspection. The reader's independent
    // bound prevents that race from turning read_to_end into an unbounded
    // allocation, and the extra byte distinguishes oversized from truncated.
    file.take(wire_limit as u64 + 1)
        .read_to_end(&mut wire)
        .map_err(read_error)?;
    if wire.len() > wire_limit {
        return Err(DurableError::RecordTooLarge {
            actual: wire.len(),
            limit: wire_limit,
        });
    }
    DurableRecord::from_wire(&wire)
}

fn small_wire_limit(payload_len: usize) -> usize {
    payload_len
        .saturating_mul(4)
        .saturating_add(4096)
        .min(MAX_RECORD_WIRE_BYTES)
}

fn expected_handle(record: &DurableRecord) -> Result<PreparedHandle, DurableError> {
    record.validate(record.universe_id, &record.dependencies)?;
    safe_schema_segment(&record.payload_schema)?;
    let handle = PreparedHandle::from_name(format!(
        "{}-{}.record",
        hex_lower(&record.record_digest),
        record.payload.len()
    ));
    handle.validate()?;
    Ok(handle)
}

fn require_expected_record(
    record: DurableRecord,
    expected: &DurableRecord,
) -> Result<DurableRecord, DurableError> {
    if &record != expected {
        return Err(DurableError::MalformedRecord(
            "stored record differs from the complete expected record".into(),
        ));
    }
    Ok(record)
}

fn release_staging_file(staging: &Path, handle: &PreparedHandle) -> Result<(), DurableError> {
    let candidate = staging.join(handle.as_str());
    match fs::symlink_metadata(&candidate) {
        Ok(metadata) if metadata.file_type().is_file() => {
            fs::remove_file(&candidate).map_err(directory_error)?;
            sync_directory(staging)?;
        }
        Ok(_) => {
            return Err(DurableError::MalformedRecord(
                "released record must be a regular staging file".into(),
            ));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(read_error(error)),
    }
    Ok(())
}

fn read_bound_record(
    path: &Path,
    universe: [u8; 32],
    schema: &str,
    handle: &PreparedHandle,
) -> Result<DurableRecord, DurableError> {
    read_bound_record_with_limit(path, universe, schema, handle, MAX_RECORD_WIRE_BYTES)
}

fn read_bound_record_with_limit(
    path: &Path,
    universe: [u8; 32],
    schema: &str,
    handle: &PreparedHandle,
    wire_limit: usize,
) -> Result<DurableRecord, DurableError> {
    let (expected_digest, expected_len) = handle.parts()?;
    let record = read_record_bounded(path, wire_limit)?;
    if record.universe_id != universe {
        return Err(DurableError::UniverseMismatch {
            expected: universe,
            found: record.universe_id,
        });
    }
    if record.payload_schema != schema {
        return Err(DurableError::SchemaMismatch {
            expected: schema.into(),
            found: record.payload_schema,
        });
    }
    if record.record_digest != expected_digest {
        return Err(DurableError::DigestMismatch {
            expected: expected_digest,
            computed: record.record_digest,
        });
    }
    if record.payload.len() != expected_len {
        return Err(DurableError::MalformedRecord(
            "stored payload length differs from its record handle".into(),
        ));
    }
    Ok(record)
}

fn read_error(error: io::Error) -> DurableError {
    if error.kind() == io::ErrorKind::NotFound {
        DurableError::Absent
    } else {
        DurableError::Backend(format!("durable file read failed: {error}"))
    }
}

fn directory_error(error: io::Error) -> DurableError {
    DurableError::Backend(format!("durable directory entry failed: {error}"))
}

fn read_directory(path: &Path) -> Result<fs::ReadDir, DurableError> {
    fs::read_dir(path).map_err(directory_error)
}

fn existing_directory(path: &Path) -> Result<bool, DurableError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => Ok(true),
        Ok(_) => Err(DurableError::MalformedRecord(
            "durable namespace must be a directory, not a link or file".into(),
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(directory_error(error)),
    }
}

fn ensure_directory(path: &Path) -> Result<(), DurableError> {
    if !existing_directory(path)? {
        match fs::create_dir(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                existing_directory(path)?;
            }
            Err(error) => return Err(directory_error(error)),
        }
        sync_directory(path)?;
        if let Some(parent) = path.parent() {
            sync_directory(parent)?;
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), DurableError> {
    fs::File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| DurableError::Backend(format!("durable directory sync failed: {error}")))
}

fn sync_file(path: &Path) -> Result<(), DurableError> {
    fs::File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(read_error)
}

fn write_staging_file(
    temp: &Path,
    target: &Path,
    write_payload: impl FnOnce(&mut fs::File) -> Result<(), DurableError>,
) -> Result<(), DurableError> {
    // No cleanup runs unless this invocation actually created the file. A
    // create_new collision must leave the pre-existing path untouched.
    let mut written = fs::File::options()
        .write(true)
        .create_new(true)
        .open(temp)
        .map_err(|error| DurableError::Backend(format!("cannot write temp staging: {error}")))?;
    let write_result = write_payload(&mut written).and_then(|()| {
        written
            .sync_all()
            .map_err(|error| DurableError::Backend(format!("cannot sync staged bytes: {error}")))
    });
    // Close the file before unlinking it, including on platforms that refuse
    // deletion of an open file. No fallible operation can bypass this close.
    drop(written);
    let operation = write_result.and_then(|()| {
        fs::rename(temp, target).map_err(|error| {
            DurableError::Backend(format!("cannot finalize staging name: {error}"))
        })
    });
    operation.map_err(|error| cleanup_failed_staging(temp, error))
}

fn cleanup_failed_staging(temp: &Path, operation: DurableError) -> DurableError {
    let cleanup = match fs::remove_file(temp) {
        Ok(()) => temp.parent().map_or(Ok(()), sync_directory),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(directory_error(error)),
    };
    match cleanup {
        Ok(()) => operation,
        Err(cleanup) => DurableError::Backend(format!(
            "staging operation failed: {operation}; cleanup of its owned temporary file failed: {cleanup}"
        )),
    }
}

fn parse_universe(name: &str) -> Option<[u8; 32]> {
    if name.len() != 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    hex_decode(name).ok()?.try_into().ok()
}

fn scan_step(scanned: &mut usize) -> Result<(), DurableError> {
    *scanned += 1;
    if *scanned > MAX_SCAN_ENTRIES {
        return Err(DurableError::Backend(
            "durable store directory scan limit exceeded; use an exact lookup".into(),
        ));
    }
    Ok(())
}

impl DurableStore for FileStore {
    fn prepare(&self, record: &DurableRecord) -> Result<PreparedHandle, DurableError> {
        // Validate and bound all bytes before creating a staging namespace.
        let wire = record.to_wire()?;
        safe_schema_segment(&record.payload_schema)?;
        let handle = PreparedHandle {
            name: format!(
                "{}-{}.record",
                hex_lower(&record.record_digest),
                record.payload.len()
            ),
        };
        handle.validate()?;
        let staging = self.staging_dir(record.universe_id, &record.payload_schema)?;
        let target = staging.join(&handle.name);
        match read_bound_record_with_limit(
            &target,
            record.universe_id,
            &record.payload_schema,
            &handle,
            wire.len(),
        ) {
            Ok(existing) if &existing == record => return Ok(handle),
            Ok(_) => {
                return Err(DurableError::MalformedRecord(
                    "prepared record differs at its content address".into(),
                ));
            }
            Err(DurableError::Absent) => {}
            Err(error) => return Err(error),
        }
        let mut held = 0usize;
        for entry in read_directory(&staging)? {
            entry.map_err(directory_error)?;
            held += 1;
            if held >= CLEANUP_RESERVE_LIMIT {
                return Err(DurableError::InsufficientCleanupReserve {
                    held,
                    limit: CLEANUP_RESERVE_LIMIT,
                });
            }
        }
        // Write to a temp sibling, sync, then rename: the observed staging
        // name only ever contains fully written bytes. create_new prevents
        // stale temporary paths or symbolic links from redirecting writes.
        static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
        let temp = staging.join(format!(
            ".tmp-{}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed),
            handle.name
        ));
        write_staging_file(&temp, &target, |written| {
            use std::io::Write as _;
            written.write_all(&wire).map_err(|error| {
                DurableError::Backend(format!("cannot write staged bytes: {error}"))
            })
        })?;
        // A failure here occurs after rename made the staging record visible.
        // Keep that indeterminate state explicit; caller-owned rollback may
        // release its known staging record, rather than treating it as a temp.
        sync_directory(&staging)?;
        Ok(handle)
    }

    fn verify_prepared(&self, handle: &PreparedHandle) -> Result<DurableRecord, DurableError> {
        self.prepared_record(handle).map(|(_, record)| record)
    }

    fn commit(&self, handle: &PreparedHandle) -> Result<(), DurableError> {
        let (expected_digest, expected_len) = handle.parts()?;
        let (source, record) = match self.prepared_record(handle) {
            Ok(prepared) => prepared,
            Err(DurableError::UnknownHandle(_)) => {
                // A completed commit may be retried after its staging name
                // disappeared. Its published bytes still have to validate.
                let existing = self.load_committed_record(expected_digest)?;
                if existing.payload.len() != expected_len {
                    return Err(DurableError::MalformedRecord(
                        "committed payload length differs from retry handle".into(),
                    ));
                }
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        self.publish_record(&source, &record, handle, MAX_RECORD_WIRE_BYTES)
    }

    fn load_committed(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<Option<DurableRecord>, DurableError> {
        let schema = safe_schema_segment(payload_schema)?;
        let universe = self.root.join(hex_lower(&universe_id));
        let namespace = universe.join(schema);
        let committed = namespace.join("committed");
        for directory in [&universe, &namespace, &committed] {
            if !existing_directory(directory)? {
                return Ok(None);
            }
        }
        // Retain this legacy "latest" API for existing callers. Stable name
        // ordering resolves equal mtimes; resumable search uses exact keys.
        let mut best: Option<(std::time::SystemTime, String)> = None;
        let mut scanned = 0usize;
        for entry in read_directory(&committed)? {
            scan_step(&mut scanned)?;
            let entry = entry.map_err(directory_error)?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.ends_with(".record") {
                continue;
            }
            PreparedHandle::from_name(name).validate()?;
            let metadata = fs::symlink_metadata(entry.path()).map_err(read_error)?;
            if !metadata.file_type().is_file() {
                return Err(DurableError::MalformedRecord(
                    "committed record must be a regular file".into(),
                ));
            }
            let modified = metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            let key = (modified, name.to_owned());
            if best.as_ref().is_none_or(|prior| &key > prior) {
                best = Some(key);
            }
        }
        best.map(|(_, name)| {
            let handle = PreparedHandle::from_name(name);
            read_bound_record(
                &committed.join(handle.as_str()),
                universe_id,
                payload_schema,
                &handle,
            )
        })
        .transpose()
    }

    fn release(&self, handle: &PreparedHandle) -> Result<(), DurableError> {
        handle.validate()?;
        // Corrupt candidate bytes may still be discarded, but the requested
        // basename and every parent directory must belong to this store.
        self.visit_namespaces(|_, _, namespace| {
            let staging = namespace.join("staging");
            if !existing_directory(&staging)? {
                return Ok(());
            }
            release_staging_file(&staging, handle)
        })
    }

    fn staging_len(&self) -> Result<usize, DurableError> {
        let mut total = 0usize;
        self.visit_namespaces(|_, _, namespace| {
            let staging = namespace.join("staging");
            if existing_directory(&staging)? {
                for entry in read_directory(&staging)? {
                    entry.map_err(directory_error)?;
                    scan_step(&mut total)?;
                }
            }
            Ok(())
        })?;
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

    #[test]
    fn direct_record_publication_and_cleanup_ignore_unrelated_corrupt_namespaces() {
        let root = unique_root("direct-record");
        let store = FileStore::open(&root).expect("store");
        let unrelated = root.join(hex_lower(&[99; 32]));
        fs::create_dir(&unrelated).expect("foreign universe");
        let sentinel = unrelated.join("foreign-schema");
        fs::write(&sentinel, b"not-a-schema-directory").expect("corrupt foreign namespace");

        let record = sample();
        let handle = store.prepare(&record).expect("direct prepare");
        assert_eq!(store.verify_prepared_record(&record).unwrap(), record);
        store.commit_record(&record).expect("direct publish");
        store
            .commit_record(&record)
            .expect("idempotent direct retry");
        assert_eq!(
            store
                .load_committed_exact(
                    record.universe_id,
                    &record.payload_schema,
                    record.record_digest,
                    record.payload.len(),
                )
                .unwrap(),
            record
        );
        store.prepare(&record).expect("re-stage");
        store
            .release_prepared_record(&record)
            .expect("direct release");
        store
            .release_prepared_record(&record)
            .expect("release absent");
        let staging_path = root
            .join(hex_lower(&record.universe_id))
            .join(&record.payload_schema)
            .join("staging")
            .join(handle.as_str());
        assert!(!staging_path.exists());
        assert_eq!(fs::read(sentinel).unwrap(), b"not-a-schema-directory");
    }

    #[test]
    fn direct_readback_and_commit_preflight_the_small_envelope_bound() {
        let root = unique_root("direct-bound");
        let store = FileStore::open(&root).expect("store");
        let record = sample();
        let handle = store.prepare(&record).expect("prepare");
        let path = root
            .join(hex_lower(&record.universe_id))
            .join(&record.payload_schema)
            .join("staging")
            .join(handle.as_str());
        let limit = small_wire_limit(record.payload.len());
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(limit as u64 + 1)
            .unwrap();
        for result in [
            store.verify_prepared_record(&record).map(|_| ()),
            store.commit_record(&record),
            store.prepare(&record).map(|_| ()),
        ] {
            assert!(matches!(result, Err(DurableError::RecordTooLarge { .. })));
        }
        store.release_prepared_record(&record).unwrap();
        assert!(!path.exists());
        assert!(matches!(
            store.load_committed_exact(
                record.universe_id,
                &record.payload_schema,
                record.record_digest,
                record.payload.len(),
            ),
            Err(DurableError::Absent)
        ));
    }

    #[test]
    fn partial_write_failures_release_owned_temps_without_exhausting_staging() {
        use std::io::Write as _;

        let root = unique_root("failed-write-cleanup");
        let store = FileStore::open(&root).unwrap();
        let record = sample();
        let staging = store
            .staging_dir(record.universe_id, &record.payload_schema)
            .unwrap();
        let target = staging.join("not-published.record");
        for attempt in 0..(CLEANUP_RESERVE_LIMIT + 2) {
            let temp = staging.join(format!(".tmp-failed-write-{attempt}"));
            let operation = DurableError::Backend("injected failure after partial write".into());
            let failure = write_staging_file(&temp, &target, |file| {
                file.write_all(b"partial record bytes")
                    .map_err(read_error)?;
                Err(operation.clone())
            })
            .unwrap_err();
            assert_eq!(failure, operation);
            assert!(!temp.exists());
            assert!(!target.exists());
            assert_eq!(store.staging_len().unwrap(), 0);
        }
        let handle = store.prepare(&record).expect("reserve remains available");
        assert_eq!(store.verify_prepared(&handle).unwrap(), record);
    }

    #[test]
    fn rename_failure_cleans_only_the_new_temp_and_preserves_existing_targets() {
        use std::io::Write as _;

        let root = unique_root("failed-rename-cleanup");
        let temp = root.join(".tmp-owned");
        let target = root.join("existing-directory");
        fs::create_dir(&target).unwrap();
        let sentinel = target.join("sentinel");
        fs::write(&sentinel, b"preserve target").unwrap();
        let unrelated = root.join(".tmp-other-attempt");
        fs::write(&unrelated, b"preserve other temp").unwrap();
        let failure = write_staging_file(&temp, &target, |file| {
            file.write_all(b"complete record").map_err(read_error)
        })
        .unwrap_err();
        assert!(failure.to_string().contains("cannot finalize staging name"));
        assert!(!temp.exists());
        assert_eq!(fs::read(sentinel).unwrap(), b"preserve target");
        assert_eq!(fs::read(unrelated).unwrap(), b"preserve other temp");
    }

    #[test]
    fn failed_temp_creation_never_claims_or_deletes_the_existing_file() {
        let root = unique_root("temp-create-collision");
        let temp = root.join(".tmp-existing");
        let target = root.join("not-published.record");
        fs::write(&temp, b"preexisting bytes").unwrap();
        let failure = write_staging_file(&temp, &target, |_| {
            panic!("writer must not run when create_new refuses")
        })
        .unwrap_err();
        assert!(failure.to_string().contains("cannot write temp staging"));
        assert_eq!(fs::read(temp).unwrap(), b"preexisting bytes");
        assert!(!target.exists());
    }

    #[test]
    fn failed_temp_cleanup_reports_both_errors_without_deleting_a_directory() {
        let changed_temp = unique_root("temp-cleanup-failure");
        let operation = DurableError::Backend("original write failure".into());
        let failure = cleanup_failed_staging(&changed_temp, operation);
        let message = failure.to_string();
        assert!(message.contains("original write failure"));
        assert!(message.contains("cleanup of its owned temporary file failed"));
        assert!(changed_temp.is_dir());
    }
}
