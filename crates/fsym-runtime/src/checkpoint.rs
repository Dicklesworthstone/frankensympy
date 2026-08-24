//! Typed normalized checkpoints and state snapshots (WS13 / architecture §7.8).
//!
//! Checkpoints capture normalized mathematical and execution state, sequence IDs,
//! and remaining budget allowances. Checkpoints are NEVER process memory dumps.

#![forbid(unsafe_code)]

use fsym_budget::Dimension;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Typed normalized checkpoint with BLAKE3 cryptographic integrity digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedCheckpoint<T> {
    pub checkpoint_seq: u64,
    pub payload: T,
    pub remaining_budget: BTreeMap<Dimension, u64>,
    pub verifier_remaining: u64,
    pub content_digest: [u8; 32],
}

impl<T: Serialize> TypedCheckpoint<T> {
    /// Create a new typed checkpoint and compute its canonical digest.
    pub fn new(
        checkpoint_seq: u64,
        payload: T,
        remaining_budget: BTreeMap<Dimension, u64>,
        verifier_remaining: u64,
    ) -> Self {
        let serialized_payload =
            serde_json::to_vec(&payload).expect("payload must be serializable");
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.checkpoint.v1:");
        hasher.update(&checkpoint_seq.to_le_bytes());
        hasher.update(&serialized_payload);
        hasher.update(&verifier_remaining.to_le_bytes());
        let content_digest = *hasher.finalize().as_bytes();

        Self {
            checkpoint_seq,
            payload,
            remaining_budget,
            verifier_remaining,
            content_digest,
        }
    }

    /// Check integrity digest of this checkpoint.
    pub fn verify_integrity(&self) -> bool {
        let serialized_payload = match serde_json::to_vec(&self.payload) {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.checkpoint.v1:");
        hasher.update(&self.checkpoint_seq.to_le_bytes());
        hasher.update(&serialized_payload);
        hasher.update(&self.verifier_remaining.to_le_bytes());
        *hasher.finalize().as_bytes() == self.content_digest
    }
}
