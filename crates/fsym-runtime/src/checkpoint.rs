//! Typed normalized checkpoints and state snapshots (WS13 / architecture §7.8).
//!
//! Checkpoints capture normalized mathematical and execution state, sequence IDs,
//! and remaining budget allowances. Checkpoints are NEVER process memory dumps.

#![forbid(unsafe_code)]

use fsym_budget::Dimension;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Write};
use thiserror::Error;

const CHECKPOINT_SCHEMA_VERSION: u32 = 3;
const MAX_CHECKPOINT_SCHEMA_ID_BYTES: usize = 128;
const MAX_CHECKPOINT_CANONICAL_BYTES: usize = 8 * 1024 * 1024;
const MAX_CHECKPOINT_WIRE_BYTES: usize = MAX_CHECKPOINT_CANONICAL_BYTES + 1024;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CheckpointError {
    #[error("Checkpoint schema ID must contain 1..={0} bytes")]
    InvalidSchemaId(usize),
    #[error("Checkpoint canonical encoding exceeds {limit} bytes")]
    CanonicalEncodingTooLarge { limit: usize },
    #[error("Checkpoint serialization allocation failed")]
    AllocationFailure,
    #[error("Checkpoint serialization failed")]
    SerializationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterFailure {
    SizeLimit,
    Allocation,
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    failure: Option<WriterFailure>,
}

impl BoundedJsonWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            failure: None,
        }
    }

    fn into_result(self, serialization_succeeded: bool) -> Result<Vec<u8>, CheckpointError> {
        match (serialization_succeeded, self.failure) {
            (true, None) => Ok(self.bytes),
            (_, Some(WriterFailure::SizeLimit)) => {
                Err(CheckpointError::CanonicalEncodingTooLarge {
                    limit: MAX_CHECKPOINT_CANONICAL_BYTES,
                })
            }
            (_, Some(WriterFailure::Allocation)) => Err(CheckpointError::AllocationFailure),
            (false, None) => Err(CheckpointError::SerializationFailed),
        }
    }
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(buffer.len()) else {
            self.failure = Some(WriterFailure::SizeLimit);
            return Err(io::Error::other("checkpoint encoding size limit exceeded"));
        };
        if new_len > MAX_CHECKPOINT_CANONICAL_BYTES {
            self.failure = Some(WriterFailure::SizeLimit);
            return Err(io::Error::other("checkpoint encoding size limit exceeded"));
        }
        if self.bytes.try_reserve(buffer.len()).is_err() {
            self.failure = Some(WriterFailure::Allocation);
            return Err(io::Error::other("checkpoint encoding allocation failed"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

enum CanonicalJson {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

impl Serialize for CanonicalJson {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => value.serialize(serializer),
            Self::String(value) => serializer.serialize_str(value),
            Self::Array(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(value)?;
                }
                sequence.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

struct CanonicalJsonVisitor;

impl<'de> Visitor<'de> for CanonicalJsonVisitor {
    type Value = CanonicalJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(CanonicalJson::Null)
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(CanonicalJson::Null)
    }

    fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
        Ok(CanonicalJson::Bool(value))
    }

    fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
        Ok(CanonicalJson::Number(value.into()))
    }

    fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
        Ok(CanonicalJson::Number(value.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
        serde_json::Number::from_f64(value)
            .map(CanonicalJson::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(CanonicalJson::String(value.to_owned()))
    }

    fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
        Ok(CanonicalJson::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(CanonicalJson::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entries = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if entries.contains_key(&key) {
                return Err(serde::de::Error::custom("duplicate JSON object key"));
            }
            let value = map.next_value()?;
            entries.insert(key, value);
        }
        Ok(CanonicalJson::Object(entries))
    }
}

impl<'de> Deserialize<'de> for CanonicalJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(CanonicalJsonVisitor)
    }
}

/// Typed normalized checkpoint with BLAKE3 cryptographic integrity digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedCheckpoint<T> {
    pub schema_version: u32,
    pub payload_schema: String,
    pub checkpoint_seq: u64,
    pub payload: T,
    pub remaining_budget: BTreeMap<Dimension, u64>,
    pub verifier_remaining: u64,
    pub content_digest: [u8; 32],
}

impl<T: Serialize> TypedCheckpoint<T> {
    /// Create a new typed checkpoint and compute its canonical digest.
    pub fn new(
        payload_schema: impl Into<String>,
        checkpoint_seq: u64,
        payload: T,
        remaining_budget: BTreeMap<Dimension, u64>,
        verifier_remaining: u64,
    ) -> Result<Self, CheckpointError> {
        let payload_schema = payload_schema.into();
        validate_schema_id(&payload_schema)?;
        let content_digest = checkpoint_digest(
            CHECKPOINT_SCHEMA_VERSION,
            &payload_schema,
            checkpoint_seq,
            &payload,
            &remaining_budget,
            verifier_remaining,
        )?;

        Ok(Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            payload_schema,
            checkpoint_seq,
            payload,
            remaining_budget,
            verifier_remaining,
            content_digest,
        })
    }

    /// Check integrity digest of this checkpoint.
    pub fn verify_integrity(&self) -> bool {
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION
            || validate_schema_id(&self.payload_schema).is_err()
        {
            return false;
        }
        checkpoint_digest(
            self.schema_version,
            &self.payload_schema,
            self.checkpoint_seq,
            &self.payload,
            &self.remaining_budget,
            self.verifier_remaining,
        )
        .is_ok_and(|digest| digest == self.content_digest)
    }
}

/// Validates the exact serialized checkpoint representation before storage.
///
/// Deserializing the payload through `serde_json::Value` first would silently
/// collapse duplicate object keys. Parsing it directly as `CanonicalJson`
/// preserves the checkpoint's fail-closed duplicate-key rule at the wire
/// boundary as well as during initial construction.
pub(crate) fn serialized_checkpoint_has_valid_integrity(serialized: &[u8]) -> bool {
    if serialized.len() > MAX_CHECKPOINT_WIRE_BYTES {
        return false;
    }
    serde_json::from_slice::<TypedCheckpoint<CanonicalJson>>(serialized)
        .is_ok_and(|checkpoint| checkpoint.verify_integrity())
}

fn checkpoint_digest<T: Serialize>(
    schema_version: u32,
    payload_schema: &str,
    checkpoint_seq: u64,
    payload: &T,
    remaining_budget: &BTreeMap<Dimension, u64>,
    verifier_remaining: u64,
) -> Result<[u8; 32], CheckpointError> {
    // Bound the generic serializer before constructing the canonical tree.
    // Without this admission pass, an untrusted payload could allocate an
    // arbitrarily large tree before the size limit below was observed.
    let mut admission_writer = BoundedJsonWriter::new();
    let admission_succeeded = serde_json::to_writer(
        &mut admission_writer,
        &(
            schema_version,
            payload_schema,
            checkpoint_seq,
            payload,
            remaining_budget,
            verifier_remaining,
        ),
    )
    .is_ok();
    let admitted_fields = admission_writer.into_result(admission_succeeded)?;

    // The canonical tree rejects duplicate keys recursively and uses `BTreeMap`
    // ordering, avoiding both last-key-wins digest collisions and process-random
    // `HashMap` iteration order in otherwise generic payloads.
    let canonical_value: CanonicalJson = serde_json::from_slice(&admitted_fields)
        .map_err(|_| CheckpointError::SerializationFailed)?;
    let mut canonical_writer = BoundedJsonWriter::new();
    let canonical_succeeded =
        serde_json::to_writer(&mut canonical_writer, &canonical_value).is_ok();
    let canonical_fields = canonical_writer.into_result(canonical_succeeded)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.checkpoint.v3:");
    hasher.update(&canonical_fields);
    Ok(*hasher.finalize().as_bytes())
}

fn validate_schema_id(schema_id: &str) -> Result<(), CheckpointError> {
    if schema_id.is_empty() || schema_id.len() > MAX_CHECKPOINT_SCHEMA_ID_BYTES {
        Err(CheckpointError::InvalidSchemaId(
            MAX_CHECKPOINT_SCHEMA_ID_BYTES,
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct DuplicateKeyPayload;

    impl Serialize for DuplicateKeyPayload {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("duplicate", &1_u8)?;
            map.serialize_entry("duplicate", &2_u8)?;
            map.end()
        }
    }

    #[test]
    fn remaining_budget_is_integrity_bound() {
        let mut budget = BTreeMap::new();
        budget.insert(Dimension::ComputeSteps, 10);
        let mut checkpoint = TypedCheckpoint::new("test.state.v1", 7, "state", budget, 2).unwrap();
        assert!(checkpoint.verify_integrity());

        checkpoint
            .remaining_budget
            .insert(Dimension::ComputeSteps, 1_000_000);

        assert!(!checkpoint.verify_integrity());
    }

    #[test]
    fn unordered_payload_maps_have_stable_digests_and_roundtrip() {
        let mut first_payload = std::collections::HashMap::new();
        first_payload.insert("b".to_string(), 2u64);
        first_payload.insert("a".to_string(), 1u64);
        let mut second_payload = std::collections::HashMap::new();
        second_payload.insert("a".to_string(), 1u64);
        second_payload.insert("b".to_string(), 2u64);

        let first =
            TypedCheckpoint::new("test.map.v1", 1, first_payload, BTreeMap::new(), 0).unwrap();
        let second =
            TypedCheckpoint::new("test.map.v1", 1, second_payload, BTreeMap::new(), 0).unwrap();
        assert_eq!(first.content_digest, second.content_digest);

        let wire = serde_json::to_vec(&first).unwrap();
        let restored: TypedCheckpoint<std::collections::HashMap<String, u64>> =
            serde_json::from_slice(&wire).unwrap();
        assert!(restored.verify_integrity());
    }

    #[test]
    fn schema_identity_is_mandatory_and_integrity_bound() {
        assert_eq!(
            TypedCheckpoint::new("", 1, "state", BTreeMap::new(), 0),
            Err(CheckpointError::InvalidSchemaId(
                MAX_CHECKPOINT_SCHEMA_ID_BYTES
            ))
        );

        let mut checkpoint =
            TypedCheckpoint::new("test.state.v1", 1, "state", BTreeMap::new(), 0).unwrap();
        checkpoint.payload_schema = "test.other.v1".to_string();
        assert!(!checkpoint.verify_integrity());
    }

    #[test]
    fn oversized_payload_is_refused_before_canonical_tree_construction() {
        let oversized = "x".repeat(MAX_CHECKPOINT_CANONICAL_BYTES + 1);
        assert_eq!(
            TypedCheckpoint::new("test.oversized.v1", 1, oversized, BTreeMap::new(), 0),
            Err(CheckpointError::CanonicalEncodingTooLarge {
                limit: MAX_CHECKPOINT_CANONICAL_BYTES
            })
        );
    }

    #[test]
    fn duplicate_payload_keys_are_refused_instead_of_collapsed() {
        assert_eq!(
            TypedCheckpoint::new(
                "test.duplicate-key.v1",
                1,
                DuplicateKeyPayload,
                BTreeMap::new(),
                0,
            ),
            Err(CheckpointError::SerializationFailed)
        );
    }
}
