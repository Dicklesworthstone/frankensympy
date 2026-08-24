//! Typed normalized checkpoints and state snapshots (WS13 / architecture §7.8).
//!
//! Checkpoints capture normalized mathematical and execution state, sequence IDs,
//! and remaining budget allowances. Checkpoints are NEVER process memory dumps.

#![forbid(unsafe_code)]

use fsym_budget::Dimension;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

const CHECKPOINT_SCHEMA_VERSION: u32 = 3;
const MAX_CHECKPOINT_SCHEMA_ID_BYTES: usize = 128;
const MAX_CHECKPOINT_CANONICAL_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CheckpointError {
    #[error("Checkpoint schema ID must contain 1..={0} bytes")]
    InvalidSchemaId(usize),
    #[error("Checkpoint canonical encoding exceeds {limit} bytes")]
    CanonicalEncodingTooLarge { limit: usize },
    #[error("Checkpoint serialization failed: {0}")]
    Serialization(String),
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

fn checkpoint_digest<T: Serialize>(
    schema_version: u32,
    payload_schema: &str,
    checkpoint_seq: u64,
    payload: &T,
    remaining_budget: &BTreeMap<Dimension, u64>,
    verifier_remaining: u64,
) -> Result<[u8; 32], CheckpointError> {
    // Convert through `Value`: serde_json's default map representation is key-sorted, avoiding
    // process-random `HashMap` iteration order in an otherwise generic payload.
    let canonical_value = serde_json::to_value(&(
        schema_version,
        payload_schema,
        checkpoint_seq,
        payload,
        remaining_budget,
        verifier_remaining,
    ))
    .map_err(|error| CheckpointError::Serialization(error.to_string()))?;
    let canonical_fields = serde_json::to_vec(&canonical_value)
        .map_err(|error| CheckpointError::Serialization(error.to_string()))?;
    if canonical_fields.len() > MAX_CHECKPOINT_CANONICAL_BYTES {
        return Err(CheckpointError::CanonicalEncodingTooLarge {
            limit: MAX_CHECKPOINT_CANONICAL_BYTES,
        });
    }
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

        let first = TypedCheckpoint::new(
            "test.map.v1",
            1,
            first_payload,
            BTreeMap::new(),
            0,
        )
        .unwrap();
        let second = TypedCheckpoint::new(
            "test.map.v1",
            1,
            second_payload,
            BTreeMap::new(),
            0,
        )
        .unwrap();
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
}
