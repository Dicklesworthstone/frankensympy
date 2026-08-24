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
        let content_digest = checkpoint_digest(
            checkpoint_seq,
            &payload,
            &remaining_budget,
            verifier_remaining,
        )
        .expect("checkpoint fields must be serializable");

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
        checkpoint_digest(
            self.checkpoint_seq,
            &self.payload,
            &self.remaining_budget,
            self.verifier_remaining,
        )
        .is_ok_and(|digest| digest == self.content_digest)
    }
}

fn checkpoint_digest<T: Serialize>(
    checkpoint_seq: u64,
    payload: &T,
    remaining_budget: &BTreeMap<Dimension, u64>,
    verifier_remaining: u64,
) -> Result<[u8; 32], serde_json::Error> {
    let canonical_fields = serde_json::to_vec(&(
        checkpoint_seq,
        payload,
        remaining_budget,
        verifier_remaining,
    ))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.checkpoint.v2:");
    hasher.update(&canonical_fields);
    Ok(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaining_budget_is_integrity_bound() {
        let mut budget = BTreeMap::new();
        budget.insert(Dimension::ComputeSteps, 10);
        let mut checkpoint = TypedCheckpoint::new(7, "state", budget, 2);
        assert!(checkpoint.verify_integrity());

        checkpoint
            .remaining_budget
            .insert(Dimension::ComputeSteps, 1_000_000);

        assert!(!checkpoint.verify_integrity());
    }
}
