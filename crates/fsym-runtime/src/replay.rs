//! Deterministic execution tracing and bit-for-bit replay engine (WS13).

#![forbid(unsafe_code)]

use fsym_budget::Dimension;
use serde::{Deserialize, Serialize};

/// An individual trace entry in a deterministic computation replay log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayEvent {
    pub step_index: u64,
    pub event_name: String,
    pub dimension_charges: Vec<(Dimension, u64)>,
    pub outcome_digest: [u8; 32],
}

/// Complete deterministic replay transcript recording seeds, choices, and outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayLog {
    pub initial_seed: u64,
    pub strategy_name: String,
    pub events: Vec<ReplayEvent>,
    pub final_digest: [u8; 32],
}

impl ReplayLog {
    /// Create a new replay log.
    pub fn new(initial_seed: u64, strategy_name: impl Into<String>) -> Self {
        Self {
            initial_seed,
            strategy_name: strategy_name.into(),
            events: Vec::new(),
            final_digest: [0u8; 32],
        }
    }

    /// Record a step in the trace.
    pub fn record_event(
        &mut self,
        name: impl Into<String>,
        dimension_charges: Vec<(Dimension, u64)>,
        payload: &[u8],
    ) {
        let step_index = self.events.len() as u64;
        let event_name = name.into();
        let canonical_event =
            serde_json::to_vec(&(step_index, &event_name, &dimension_charges, payload))
                .expect("replay event fields must be serializable");
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.replay.event.v2:");
        hasher.update(&canonical_event);
        let outcome_digest = *hasher.finalize().as_bytes();

        self.events.push(ReplayEvent {
            step_index,
            event_name,
            dimension_charges,
            outcome_digest,
        });
    }

    /// Seal and finalize the replay transcript digest.
    pub fn finalize(&mut self) -> [u8; 32] {
        let digest = self.computed_final_digest();
        self.final_digest = digest;
        digest
    }

    /// Verify that every replay field remains bound to the sealed digest.
    pub fn verify_integrity(&self) -> bool {
        self.final_digest == self.computed_final_digest()
    }

    /// Verify that a candidate replay log matches this reference transcript bit-for-bit.
    pub fn verify_replay_match(&self, candidate: &ReplayLog) -> bool {
        self.verify_integrity() && candidate.verify_integrity() && self == candidate
    }

    fn computed_final_digest(&self) -> [u8; 32] {
        let canonical_log =
            serde_json::to_vec(&(self.initial_seed, &self.strategy_name, &self.events))
                .expect("replay log fields must be serializable");
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.replay.log.v2:");
        hasher.update(&canonical_log);
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_charges_are_integrity_bound() {
        let mut log = ReplayLog::new(11, "strategy");
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome");
        log.finalize();
        assert!(log.verify_integrity());

        log.events[0].dimension_charges[0].1 = 300;

        assert!(!log.verify_integrity());
    }
}
