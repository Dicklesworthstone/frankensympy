//! Storage-neutral durable checkpoint boundary (WS15, bead `fra-rc-durable-m5e`).
//!
//! Wraps canonical typed-checkpoint wire bytes (produced by
//! `fsym_runtime::TypedCheckpoint`) in a self-describing durable record and
//! publishes it through an explicit prepare → verify → commit sequence.
//! Decoding a record — from any lane — re-validates its canonical digest,
//! schema identity, universe binding, and dependency manifest before the
//! payload is handed back. Storage never defines mathematical truth: the
//! runtime resume path independently re-verifies every candidate, and no
//! persisted boolean can stand in for that verification.
//!
//! Two lanes implement [`store::DurableStore`]:
//! - [`file_store::FileStore`] — canonical reference lane; staging directory
//!   plus same-filesystem atomic rename commit; used by the fresh-process
//!   crash matrix with actual disk readback.
//! - `fsqlite_store::FsqliteStore` (feature `fsqlite`) — narrow adapter over
//!   the pinned FrankenSQLite commit admitted in
//!   `registries/dependencies.toml`.
//!
//! Persistence is optional and outside the algebraic hot path
//! (CONSTITUTION.md §7.8): nothing in this crate participates in term
//! equality, proof validity, or evidence promotion.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod cli_store;
pub mod file_store;
pub mod store;

pub use cli_store::FsqliteCliStore;
pub use file_store::FileStore;
pub use store::{DurableStore, PreparedHandle};

// NOTE: the narrow FrankenSQLite adapter lane is intentionally absent pending
// a written admission review: the pinned frankensqlite commit resolves
// asupersync 0.5.0 from the registry, so embedding it today would place a
// second asupersync build unit in this workspace's graph. The constitution
// names asupersync as the only async runtime and requires written admission
// for donor dependencies; see fra-rc-durable-m5e on the tracker.

/// Wire schema of the durable record envelope itself.
pub const DURABLE_RECORD_SCHEMA: &str = "fsym.durable.record.v1";
pub const DURABLE_RECORD_SCHEMA_VERSION: u32 = 1;
/// Default maximum serialized record size admitted at any trust boundary.
pub const MAX_RECORD_WIRE_BYTES: usize = 4 * 1024 * 1024;

/// Typed refusal for every durable-boundary failure. Variants carry the
/// observed defect; nothing here is ever converted into a candidate result.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DurableError {
    #[error("record exceeds durable wire bound: {actual} > {limit} bytes")]
    RecordTooLarge { actual: usize, limit: usize },
    #[error("malformed durable record: {0}")]
    MalformedRecord(String),
    #[error("durable record digest mismatch: expected {expected:02x?}, computed {computed:02x?}")]
    DigestMismatch {
        expected: [u8; 32],
        computed: [u8; 32],
    },
    #[error("durable record schema mismatch: expected {expected}, found {found}")]
    SchemaMismatch { expected: String, found: String },
    #[error("durable record belongs to universe {found:02x?}, not the presenting {expected:02x?}")]
    UniverseMismatch { expected: [u8; 32], found: [u8; 32] },
    #[error("dependency manifest mismatch for {name}: expected {expected}, found {found}")]
    DependencyMismatch {
        name: String,
        expected: String,
        found: String,
    },
    #[error("dependency manifest is missing pinned entry {0}")]
    DependencyMissing(String),
    #[error("staging cleanup reserve exhausted: {held} records held, limit {limit}")]
    InsufficientCleanupReserve { held: usize, limit: usize },
    #[error("prepared record {0} is not available for the requested operation")]
    UnknownHandle(String),
    #[error("requested durable record is absent")]
    Absent,
    #[error("durable storage backend failure: {0}")]
    Backend(String),
}

/// Pinned provenance of everything the payload's correctness depends on.
///
/// Entries are name → pinned identity (commit, digest, or version string) and
/// are stored sorted; `DurableRecord::validate` compares the full map, so a
/// record produced under different tooling refuses rather than silently
/// resuming.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyManifest {
    pub entries: BTreeMap<String, String>,
}

impl DependencyManifest {
    pub fn empty() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn new(entries: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>) -> Self {
        Self {
            entries: entries
                .into_iter()
                .map(|(name, pin)| (name.into(), pin.into()))
                .collect(),
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        let serialized = serde_json::to_vec(&self.entries)
            .expect("BTreeMap serialization cannot fail in [u8;32] digest context");
        *blake3::hash(&serialized).as_bytes()
    }
}

/// Canonical durable record envelope. Field order and canonical JSON
/// serialization are identity: `record_digest` covers every field, and the
/// encoding admitted at the boundary must round-trip byte-identically
/// (`deny_unknown_fields` plus a canonical encode equality check).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableRecord {
    pub schema_version: u32,
    pub record_schema: String,
    /// Digest binding the record to one immutable input universe: the exact
    /// payload schema, input expression, assumptions context, and dependency
    /// pins a resume presents. Different universe → typed refusal.
    pub universe_id: [u8; 32],
    /// Expected payload checkpoint schema (for example
    /// `fsym.portfolio.factor_race.continuation.v1`).
    pub payload_schema: String,
    pub payload: Vec<u8>,
    pub payload_digest: [u8; 32],
    pub dependencies: DependencyManifest,
    pub record_digest: [u8; 32],
}

impl DurableRecord {
    /// Assembles and self-digests a record after bounding every field.
    pub fn new(
        universe_id: [u8; 32],
        payload_schema: impl Into<String>,
        payload: Vec<u8>,
        dependencies: DependencyManifest,
    ) -> Result<Self, DurableError> {
        let payload_schema = payload_schema.into();
        if payload_schema.is_empty() || payload_schema.len() > 256 {
            return Err(DurableError::MalformedRecord(
                "payload schema id must contain 1..=256 bytes".into(),
            ));
        }
        if payload.is_empty() {
            return Err(DurableError::MalformedRecord(
                "durable records cannot carry an empty payload".into(),
            ));
        }
        if payload.len() > MAX_RECORD_WIRE_BYTES {
            return Err(DurableError::RecordTooLarge {
                actual: payload.len(),
                limit: MAX_RECORD_WIRE_BYTES,
            });
        }
        let mut record = Self {
            schema_version: DURABLE_RECORD_SCHEMA_VERSION,
            record_schema: DURABLE_RECORD_SCHEMA.into(),
            universe_id,
            payload_schema,
            payload,
            payload_digest: [0; 32],
            dependencies,
            record_digest: [0; 32],
        };
        record.payload_digest = *blake3::hash(&record.payload).as_bytes();
        record.record_digest = record.compute_record_digest()?;
        Ok(record)
    }

    fn compute_record_digest(&self) -> Result<[u8; 32], DurableError> {
        // Canonical tree over every field except record_digest itself.
        let canonical = serde_json::json!({
            "schema_version": self.schema_version,
            "record_schema": self.record_schema,
            "universe_id": hex_lower(&self.universe_id),
            "payload_schema": self.payload_schema,
            "payload_digest": hex_lower(&self.payload_digest),
            "dependencies": self.dependencies.entries,
        });
        let mut serialized = serde_json::to_vec(&canonical).map_err(|error| {
            DurableError::MalformedRecord(format!("canonical record encoding failed: {error}"))
        })?;
        if serialized.len() > MAX_RECORD_WIRE_BYTES {
            return Err(DurableError::RecordTooLarge {
                actual: serialized.len(),
                limit: MAX_RECORD_WIRE_BYTES,
            });
        }
        serialized.push(b'\n');
        Ok(*blake3::hash(&serialized).as_bytes())
    }

    /// Full boundary validation: schema identity, payload digest, record
    /// digest, and the exact dependency manifest presented as `expected`.
    /// Returns the validated payload bytes.
    pub fn validate(
        &self,
        expected_universe: [u8; 32],
        expected_dependencies: &DependencyManifest,
    ) -> Result<&[u8], DurableError> {
        if self.schema_version != DURABLE_RECORD_SCHEMA_VERSION {
            return Err(DurableError::SchemaMismatch {
                expected: format!("schema_version {DURABLE_RECORD_SCHEMA_VERSION}"),
                found: format!("schema_version {}", self.schema_version),
            });
        }
        if self.record_schema != DURABLE_RECORD_SCHEMA {
            return Err(DurableError::SchemaMismatch {
                expected: DURABLE_RECORD_SCHEMA.into(),
                found: self.record_schema.clone(),
            });
        }
        if self.universe_id != expected_universe {
            return Err(DurableError::UniverseMismatch {
                expected: expected_universe,
                found: self.universe_id,
            });
        }
        let computed_payload = *blake3::hash(&self.payload).as_bytes();
        if computed_payload != self.payload_digest {
            return Err(DurableError::DigestMismatch {
                expected: self.payload_digest,
                computed: computed_payload,
            });
        }
        let computed_record = self.compute_record_digest()?;
        if computed_record != self.record_digest {
            return Err(DurableError::DigestMismatch {
                expected: self.record_digest,
                computed: computed_record,
            });
        }
        for (name, expected_pin) in &expected_dependencies.entries {
            match self.dependencies.entries.get(name) {
                Some(found) if found == expected_pin => {}
                Some(found) => {
                    return Err(DurableError::DependencyMismatch {
                        name: name.clone(),
                        expected: expected_pin.clone(),
                        found: found.clone(),
                    });
                }
                None => return Err(DurableError::DependencyMissing(name.clone())),
            }
        }
        if self.payload.len() > MAX_RECORD_WIRE_BYTES {
            return Err(DurableError::RecordTooLarge {
                actual: self.payload.len(),
                limit: MAX_RECORD_WIRE_BYTES,
            });
        }
        Ok(&self.payload)
    }

    /// Canonical wire encoding; decode must round-trip byte-identically.
    pub fn to_wire(&self) -> Result<Vec<u8>, DurableError> {
        let wire = serde_json::to_vec(self).map_err(|error| {
            DurableError::MalformedRecord(format!("record serialization failed: {error}"))
        })?;
        if wire.len() > MAX_RECORD_WIRE_BYTES {
            return Err(DurableError::RecordTooLarge {
                actual: wire.len(),
                limit: MAX_RECORD_WIRE_BYTES,
            });
        }
        Ok(wire)
    }

    /// Strict decode: unknown fields refuse; the decoded record is then
    /// re-digest-checked so no transport flip survives.
    pub fn from_wire(wire: &[u8]) -> Result<Self, DurableError> {
        if wire.len() > MAX_RECORD_WIRE_BYTES {
            return Err(DurableError::RecordTooLarge {
                actual: wire.len(),
                limit: MAX_RECORD_WIRE_BYTES,
            });
        }
        let record: Self = serde_json::from_slice(wire).map_err(|error| {
            DurableError::MalformedRecord(format!("record decode refused: {error}"))
        })?;
        let computed_payload = *blake3::hash(&record.payload).as_bytes();
        if computed_payload != record.payload_digest {
            return Err(DurableError::DigestMismatch {
                expected: record.payload_digest,
                computed: computed_payload,
            });
        }
        let computed_record = record.compute_record_digest()?;
        if computed_record != record.record_digest {
            return Err(DurableError::DigestMismatch {
                expected: record.record_digest,
                computed: computed_record,
            });
        }
        Ok(record)
    }
}

/// Computes the canonical universe identifier for a factor-race frontier:
/// the payload schema, the BLAKE3 of the input expression serialization, the
/// assumptions context digest, and the dependency manifest digest. Every
/// input that can change resumed semantics participates.
pub fn factor_race_universe_id(
    payload_schema: &str,
    input_expr_digest: [u8; 32],
    context_digest: [u8; 32],
    dependencies: &DependencyManifest,
) -> [u8; 32] {
    let canonical = serde_json::json!({
        "payload_schema": payload_schema,
        "input_digest": hex_lower(&input_expr_digest),
        "context_digest": hex_lower(&context_digest),
        "dependencies": dependencies.entries,
    });
    let mut serialized = serde_json::to_vec(&canonical)
        .expect("canonical universe serialization cannot fail in [u8;32] digest context");
    serialized.push(b'\n');
    *blake3::hash(&serialized).as_bytes()
}

/// Decodes a lowercase hex string produced by [`hex_lower`].
pub fn hex_decode(value: &str) -> Result<Vec<u8>, DurableError> {
    if !value.len().is_multiple_of(2) {
        return Err(DurableError::MalformedRecord(
            "hex string has odd length".into(),
        ));
    }
    let mut out = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for pair in bytes.chunks(2) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| DurableError::MalformedRecord("non-hex digit".into()))?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| DurableError::MalformedRecord("non-hex digit".into()))?;
        out.push(((high << 4) | low) as u8);
    }
    Ok(out)
}

pub fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from_digit((byte >> 4) as u32, 16).expect("hex digit"));
        out.push(char::from_digit((byte & 0xf) as u32, 16).expect("hex digit"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> DurableRecord {
        DurableRecord::new(
            [7; 32],
            "fsym.portfolio.factor_race.continuation.v1",
            b"payload-bytes".to_vec(),
            DependencyManifest::new([("toolchain", "rust-1.100.0-nightly")]),
        )
        .expect("valid record")
    }

    #[test]
    fn round_trip_preserves_digests_and_validates() {
        let record = sample_record();
        let wire = record.to_wire().expect("wire");
        let decoded = DurableRecord::from_wire(&wire).expect("decode");
        assert_eq!(decoded, record, "wire round trip must be identity");
        let payload = decoded
            .validate([7; 32], &record.dependencies)
            .expect("validation");
        assert_eq!(payload, b"payload-bytes");
    }

    #[test]
    fn every_field_flip_is_refused() {
        let record = sample_record();
        let dependencies = record.dependencies.clone();
        // Payload byte flip breaks both payload and record digests.
        let mut flipped = record.clone();
        flipped.payload[0] ^= 1;
        assert!(matches!(
            flipped.validate([7; 32], &dependencies),
            Err(DurableError::DigestMismatch { .. })
        ));
        // Universe divergence is a typed refusal, not a digest error.
        assert!(matches!(
            record.validate([9; 32], &dependencies),
            Err(DurableError::UniverseMismatch { .. })
        ));
        // Dependency drift refuses per pin; the manifest is sorted, so the
        // unexpected extra pin (absent from the presenting expectation) is
        // reported first as a missing entry, before the pin value mismatch.
        let mut other_deps = DependencyManifest::new([("toolchain", "rust-other")]);
        other_deps.entries.insert("extra".into(), "pin".into());
        assert!(matches!(
            record.validate([7; 32], &other_deps),
            Err(DurableError::DependencyMissing(_))
        ));
        let mismatch_only = DependencyManifest::new([("toolchain", "rust-other")]);
        assert!(matches!(
            record.validate([7; 32], &mismatch_only),
            Err(DurableError::DependencyMismatch { .. })
        ));
        let missing = DependencyManifest::new([("absent", "pin")]);
        assert!(matches!(
            record.validate([7; 32], &missing),
            Err(DurableError::DependencyMissing(_))
        ));
        // Re-digesting the envelope with a stale record_digest refuses even
        // when the payload itself is intact.
        let mut stale = record.clone();
        stale.record_digest = [0; 32];
        let wire = serde_json::to_vec(&stale).expect("serialize");
        assert!(matches!(
            DurableRecord::from_wire(&wire),
            Err(DurableError::DigestMismatch { .. })
        ));
        // Unknown fields refuse at the boundary.
        let mut poison = serde_json::to_value(&record).unwrap();
        poison["persisted_verified"] = serde_json::json!(true);
        let wire = serde_json::to_vec(&poison).unwrap();
        assert!(matches!(
            DurableRecord::from_wire(&wire),
            Err(DurableError::MalformedRecord(_))
        ));
    }

    #[test]
    fn empty_payload_and_oversized_bounds_refuse_at_construction() {
        assert!(matches!(
            DurableRecord::new([1; 32], "schema", Vec::new(), DependencyManifest::empty()),
            Err(DurableError::MalformedRecord(_))
        ));
        assert!(matches!(
            DurableRecord::new([1; 32], "", b"x".to_vec(), DependencyManifest::empty()),
            Err(DurableError::MalformedRecord(_))
        ));
        let big = vec![0u8; MAX_RECORD_WIRE_BYTES + 1];
        let error = DurableRecord::new([1; 32], "schema", big, DependencyManifest::empty())
            .expect_err("oversized record refused");
        assert!(matches!(error, DurableError::RecordTooLarge { .. }));
    }
}
