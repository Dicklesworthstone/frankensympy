//! Independent adversarial checks for records admitted by durable search recovery.
//!
//! Checksums establish byte identity only. These tests deliberately supply
//! checksum-consistent invalid schemas and foreign dependencies, and exercise
//! the actual file boundary rather than an in-memory substitute.

#![forbid(unsafe_code)]

use fsym_durable::{
    DependencyManifest, DurableError, DurableRecord, DurableStore, FileStore,
    MAX_RECORD_WIRE_BYTES, PreparedHandle, hex_lower,
};
use serde_json::json;
use std::fs::{self, File, FileTimes};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

const SCHEMA: &str = "fsym.root-search.checkpoint.v1";

fn temp_root(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "fsym-record-boundaries-{label}-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

fn dependencies() -> DependencyManifest {
    DependencyManifest::new([("engine", "root-search-v1")])
}

fn record(payload: &[u8]) -> DurableRecord {
    DurableRecord::new([7; 32], SCHEMA, payload.to_vec(), dependencies()).unwrap()
}

fn stored_path(
    root: &Path,
    record: &DurableRecord,
    area: &str,
    handle: &PreparedHandle,
) -> PathBuf {
    root.join(hex_lower(&record.universe_id))
        .join(&record.payload_schema)
        .join(area)
        .join(handle.as_str())
}

fn commit(store: &FileStore, record: &DurableRecord) -> PreparedHandle {
    let handle = store.prepare(record).unwrap();
    assert_eq!(store.verify_prepared(&handle).unwrap(), *record);
    store.commit(&handle).unwrap();
    handle
}

// Recompute the public record-format digest after deliberately changing a
// schema field. A rejection must therefore come from schema admission, not a
// stale checksum that would hide a missing schema check.
fn redigest(record: &mut DurableRecord) {
    record.payload_digest = *blake3::hash(&record.payload).as_bytes();
    let mut identity = serde_json::to_vec(&json!({
        "schema_version": record.schema_version,
        "record_schema": record.record_schema,
        "universe_id": hex_lower(&record.universe_id),
        "payload_schema": record.payload_schema,
        "payload_digest": hex_lower(&record.payload_digest),
        "dependencies": record.dependencies.entries,
    }))
    .unwrap();
    identity.push(b'\n');
    record.record_digest = *blake3::hash(&identity).as_bytes();
}

#[test]
fn canonical_records_round_trip_and_noncanonical_encodings_refuse() {
    let expected = record(b"{\"candidate\":true}");
    let wire = expected.to_wire().unwrap();
    assert_eq!(DurableRecord::from_wire(&wire).unwrap(), expected);
    let pretty = serde_json::to_vec_pretty(&expected).unwrap();
    assert_ne!(pretty, wire);
    let mut spaced = b" \n".to_vec();
    spaced.extend_from_slice(&wire);
    // Deliberately move a field instead of relying on serde_json::Value's map
    // order, which changes when another workspace crate enables preserve_order.
    let remaining_fields = std::str::from_utf8(&wire)
        .unwrap()
        .strip_prefix("{\"schema_version\":1,")
        .unwrap()
        .strip_suffix('}')
        .unwrap();
    let reordered = format!("{{{remaining_fields},\"schema_version\":1}}").into_bytes();
    assert_ne!(reordered, wire);
    for altered in [pretty, spaced, reordered] {
        assert!(
            DurableRecord::from_wire(&altered).is_err(),
            "a byte-distinct encoding must not masquerade as canonical"
        );
    }
}

#[test]
fn duplicate_and_unknown_fields_refuse_without_last_value_wins() {
    let expected = record(b"candidate-state");
    let wire = String::from_utf8(expected.to_wire().unwrap()).unwrap();
    let duplicate_root = wire.replacen(
        "\"schema_version\":1",
        "\"schema_version\":1,\"schema_version\":1",
        1,
    );
    let duplicate_dependency = wire.replacen(
        "\"engine\":\"root-search-v1\"",
        "\"engine\":\"untrusted\",\"engine\":\"root-search-v1\"",
        1,
    );
    assert_ne!(duplicate_root, wire);
    assert_ne!(duplicate_dependency, wire);
    let mut unknown = serde_json::to_value(&expected).unwrap();
    unknown["persisted_verified"] = json!(true);
    for altered in [
        duplicate_root.into_bytes(),
        duplicate_dependency.into_bytes(),
        serde_json::to_vec(&unknown).unwrap(),
    ] {
        assert!(DurableRecord::from_wire(&altered).is_err());
    }
}

#[test]
fn checksum_consistent_unsupported_or_empty_schemas_cannot_enter_storage() {
    let original = record(b"candidate-state");
    let root = temp_root("schema");
    let store = FileStore::open(&root).unwrap();
    for mutation in 0..4 {
        let mut invalid = original.clone();
        match mutation {
            0 => invalid.schema_version += 1,
            1 => invalid.record_schema = "fsym.durable.record.future".into(),
            2 => invalid.payload_schema.clear(),
            3 => invalid.payload.clear(),
            _ => unreachable!(),
        }
        redigest(&mut invalid);
        let wire = serde_json::to_vec(&invalid).unwrap();
        assert!(DurableRecord::from_wire(&wire).is_err());
        assert!(invalid.validate([7; 32], &dependencies()).is_err());
        assert!(store.prepare(&invalid).is_err());
        assert_eq!(store.staging_len().unwrap(), 0);
    }
}

#[test]
fn dependency_binding_requires_the_complete_exact_manifest() {
    let mut expected = record(b"candidate-state");
    assert!(expected.validate([7; 32], &dependencies()).is_ok());
    assert!(
        expected
            .validate([7; 32], &DependencyManifest::empty())
            .is_err()
    );
    let foreign = DependencyManifest::new([("engine", "root-search-v2")]);
    assert!(expected.validate([7; 32], &foreign).is_err());
    expected
        .dependencies
        .entries
        .insert("extra".into(), "pin".into());
    redigest(&mut expected);
    let from_wire = DurableRecord::from_wire(&serde_json::to_vec(&expected).unwrap()).unwrap();
    assert!(from_wire.validate([7; 32], &dependencies()).is_err());
    assert!(from_wire.validate([7; 32], &expected.dependencies).is_ok());
    assert!(from_wire.validate([8; 32], &expected.dependencies).is_err());
}

#[test]
fn dot_absolute_and_traversing_schema_segments_are_refused() {
    let parent = temp_root("schema-path");
    let root = parent.join("store");
    let store = FileStore::open(&root).unwrap();
    let absolute = parent.join("escaped").display().to_string();
    for schema in [".", "..", "../escaped", "a/../../escaped", &absolute] {
        if let Ok(invalid) = DurableRecord::new([7; 32], schema, vec![1], dependencies()) {
            assert!(store.prepare(&invalid).is_err(), "schema {schema:?}");
        }
    }
    assert!(!parent.join("escaped").exists());
    assert_eq!(store.staging_len().unwrap(), 0);
}

#[test]
fn reconstructed_handles_cannot_read_or_release_an_outside_file() {
    let parent = temp_root("handle-path");
    let root = parent.join("store");
    let store = FileStore::open(&root).unwrap();
    let expected = record(b"candidate-state");
    let live = store.prepare(&expected).unwrap();
    let sentinel = parent.join("outside.record");
    let sentinel_bytes = expected.to_wire().unwrap();
    fs::write(&sentinel, &sentinel_bytes).unwrap();
    let absolute = sentinel.display().to_string();
    for name in [
        "",
        ".",
        "..",
        "../outside.record",
        "../../outside.record",
        &absolute,
    ] {
        let invalid = PreparedHandle::from_name(name);
        assert!(store.verify_prepared(&invalid).is_err(), "read {name:?}");
        assert!(store.commit(&invalid).is_err(), "commit {name:?}");
        assert!(store.release(&invalid).is_err(), "release {name:?}");
        assert_eq!(fs::read(&sentinel).unwrap(), sentinel_bytes);
        assert_eq!(store.verify_prepared(&live).unwrap(), expected);
    }
}

#[test]
fn oversized_physical_records_are_refused_at_staging_and_committed_reads() {
    let root = temp_root("oversized-file");
    let store = FileStore::open(&root).unwrap();
    let staged = record(b"staged");
    let staged_handle = store.prepare(&staged).unwrap();
    let staged_file = stored_path(&root, &staged, "staging", &staged_handle);
    File::options()
        .write(true)
        .open(staged_file)
        .unwrap()
        .set_len(MAX_RECORD_WIRE_BYTES as u64 + 1)
        .unwrap();
    assert!(matches!(
        store.verify_prepared(&staged_handle),
        Err(DurableError::RecordTooLarge { .. })
    ));
    let committed = record(b"committed");
    let committed_handle = commit(&store, &committed);
    let committed_file = stored_path(&root, &committed, "committed", &committed_handle);
    File::options()
        .write(true)
        .open(committed_file)
        .unwrap()
        .set_len(MAX_RECORD_WIRE_BYTES as u64 + 1)
        .unwrap();
    assert!(matches!(
        store.load_committed_record(committed.record_digest),
        Err(DurableError::RecordTooLarge { .. })
    ));
}

#[test]
fn an_identical_commit_retry_is_idempotent_without_leaking_staging_slots() {
    let root = temp_root("idempotence");
    let store = FileStore::open(&root).unwrap();
    let expected = record(b"same-candidate");
    for _ in 0..12 {
        let handle = store.prepare(&expected).unwrap();
        store.commit(&handle).unwrap();
        assert_eq!(store.staging_len().unwrap(), 0);
        assert_eq!(
            store.load_committed_record(expected.record_digest).unwrap(),
            expected
        );
    }
}

#[test]
fn a_corrupt_committed_target_is_not_a_successful_idempotent_retry() {
    let root = temp_root("corrupt-target");
    let store = FileStore::open(&root).unwrap();
    let expected = record(b"candidate-state");
    let first = commit(&store, &expected);
    let target = stored_path(&root, &expected, "committed", &first);
    fs::write(&target, b"interrupted or corrupted record").unwrap();
    match store.prepare(&expected) {
        Err(_) => {}
        Ok(retry) => assert!(store.commit(&retry).is_err()),
    }
    assert!(store.load_committed_record(expected.record_digest).is_err());
    assert_eq!(
        fs::read(target).unwrap(),
        b"interrupted or corrupted record"
    );
}

#[test]
fn exact_record_selection_ignores_modification_order_and_staged_candidates() {
    let root = temp_root("exact-selection");
    let store = FileStore::open(&root).unwrap();
    let old = record(b"first-candidate");
    let newer = record(b"second-candidate");
    let unpublished = record(b"unpublished-candidate");
    let old_handle = commit(&store, &old);
    let new_handle = commit(&store, &newer);
    store.prepare(&unpublished).unwrap();
    assert!(matches!(
        store.load_committed_record(unpublished.record_digest),
        Err(DurableError::Absent)
    ));
    assert!(matches!(
        store.load_committed_record([0; 32]),
        Err(DurableError::Absent)
    ));
    let old_path = stored_path(&root, &old, "committed", &old_handle);
    let new_path = stored_path(&root, &newer, "committed", &new_handle);
    for (old_time, new_time) in [(20, 10), (10, 20), (10, 10)] {
        for (path, seconds) in [(&old_path, old_time), (&new_path, new_time)] {
            File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_times(
                    FileTimes::new()
                        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)),
                )
                .unwrap();
        }
        assert_eq!(store.load_committed_record(old.record_digest).unwrap(), old);
        assert_eq!(
            store.load_committed_record(newer.record_digest).unwrap(),
            newer
        );
    }
}

#[test]
fn a_valid_foreign_record_under_the_requested_filename_is_not_the_requested_object() {
    let root = temp_root("record-substitution");
    let store = FileStore::open(&root).unwrap();
    let expected = record(b"first-candidate");
    let foreign = record(b"other-candidate");
    let handle = commit(&store, &expected);
    fs::write(
        stored_path(&root, &expected, "committed", &handle),
        foreign.to_wire().unwrap(),
    )
    .unwrap();
    assert!(store.load_committed_record(expected.record_digest).is_err());
    assert!(matches!(
        store.load_committed_record(foreign.record_digest),
        Err(DurableError::Absent)
    ));
}
