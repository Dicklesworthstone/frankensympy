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
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.replay.event.v1:");
        hasher.update(&step_index.to_le_bytes());
        hasher.update(payload);
        let outcome_digest = *hasher.finalize().as_bytes();

        self.events.push(ReplayEvent {
            step_index,
            event_name: name.into(),
            dimension_charges,
            outcome_digest,
        });
    }

    /// Seal and finalize the replay transcript digest.
    pub fn finalize(&mut self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.replay.log.v1:");
        hasher.update(&self.initial_seed.to_le_bytes());
        hasher.update(self.strategy_name.as_bytes());
        for ev in &self.events {
            hasher.update(&ev.step_index.to_le_bytes());
            hasher.update(ev.event_name.as_bytes());
            hasher.update(&ev.outcome_digest);
        }
        let digest = *hasher.finalize().as_bytes();
        self.final_digest = digest;
        digest
    }

    /// Verify that a candidate replay log matches this reference transcript bit-for-bit.
    pub fn verify_replay_match(&self, candidate: &ReplayLog) -> bool {
        self == candidate && self.final_digest == candidate.final_digest
    }
}
