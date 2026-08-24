//! Deterministic execution transcript records and bit-for-bit comparison (WS13).
//!
//! This module binds recorded inputs and outcomes; it does not itself re-execute a strategy.

#![forbid(unsafe_code)]

use fsym_budget::Dimension;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_REPLAY_EVENTS: usize = 16_384;
const MAX_EVENT_NAME_BYTES: usize = 256;
const MAX_STRATEGY_NAME_BYTES: usize = 256;
const MAX_EVENT_PAYLOAD_BYTES: usize = 1_048_576;
const MAX_EVENT_CHARGES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReplayError {
    #[error("Replay {resource} exceeds limit {limit}")]
    LimitExceeded {
        resource: &'static str,
        limit: usize,
    },
    #[error("Replay event index mismatch: expected {expected}, got {actual}")]
    StepIndexMismatch { expected: u64, actual: u64 },
    #[error("Replay event contains a zero-valued dimension charge")]
    ZeroCharge,
    #[error("Replay serialization failed: {0}")]
    Serialization(String),
}

/// An individual trace entry in a deterministic computation replay log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayEvent {
    pub step_index: u64,
    pub event_name: String,
    pub dimension_charges: Vec<(Dimension, u64)>,
    pub payload: Vec<u8>,
    pub outcome_digest: [u8; 32],
}

/// Complete deterministic replay transcript recording seeds, choices, and outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayLog {
    pub initial_seed: u64,
    pub strategy_name: String,
    pub events: Vec<ReplayEvent>,
    pub final_digest: [u8; 32],
}

impl ReplayLog {
    /// Create a new replay log.
    pub fn new(initial_seed: u64, strategy_name: impl Into<String>) -> Result<Self, ReplayError> {
        let strategy_name = strategy_name.into();
        check_len(
            strategy_name.len(),
            MAX_STRATEGY_NAME_BYTES,
            "strategy-name bytes",
        )?;
        Ok(Self {
            initial_seed,
            strategy_name,
            events: Vec::new(),
            final_digest: [0u8; 32],
        })
    }

    /// Record a step in the trace.
    pub fn record_event(
        &mut self,
        name: impl Into<String>,
        dimension_charges: Vec<(Dimension, u64)>,
        payload: &[u8],
    ) -> Result<(), ReplayError> {
        let next_event_count =
            self.events
                .len()
                .checked_add(1)
                .ok_or(ReplayError::LimitExceeded {
                    resource: "event count",
                    limit: MAX_REPLAY_EVENTS,
                })?;
        check_len(next_event_count, MAX_REPLAY_EVENTS, "event count")?;
        let event_name = name.into();
        check_len(event_name.len(), MAX_EVENT_NAME_BYTES, "event-name bytes")?;
        check_len(
            dimension_charges.len(),
            MAX_EVENT_CHARGES,
            "dimension-charge count",
        )?;
        check_len(
            payload.len(),
            MAX_EVENT_PAYLOAD_BYTES,
            "event-payload bytes",
        )?;
        if dimension_charges.iter().any(|(_, amount)| *amount == 0) {
            return Err(ReplayError::ZeroCharge);
        }
        let step_index =
            u64::try_from(self.events.len()).map_err(|_| ReplayError::LimitExceeded {
                resource: "event count",
                limit: MAX_REPLAY_EVENTS,
            })?;
        let payload = payload.to_vec();
        let outcome_digest = event_digest(step_index, &event_name, &dimension_charges, &payload)?;

        self.events.push(ReplayEvent {
            step_index,
            event_name,
            dimension_charges,
            payload,
            outcome_digest,
        });
        Ok(())
    }

    /// Seal and finalize the replay transcript digest.
    pub fn finalize(&mut self) -> Result<[u8; 32], ReplayError> {
        self.validate_events()?;
        let digest = self.computed_final_digest()?;
        self.final_digest = digest;
        Ok(digest)
    }

    /// Verify that every replay field remains bound to the sealed digest.
    pub fn verify_integrity(&self) -> bool {
        self.validate_events().is_ok()
            && self
                .computed_final_digest()
                .is_ok_and(|digest| self.final_digest == digest)
    }

    /// Verify that a candidate replay log matches this reference transcript bit-for-bit.
    pub fn verify_replay_match(&self, candidate: &ReplayLog) -> bool {
        self.verify_integrity() && candidate.verify_integrity() && self == candidate
    }

    fn validate_events(&self) -> Result<(), ReplayError> {
        check_len(
            self.strategy_name.len(),
            MAX_STRATEGY_NAME_BYTES,
            "strategy-name bytes",
        )?;
        if self.events.len() > MAX_REPLAY_EVENTS {
            return Err(ReplayError::LimitExceeded {
                resource: "event count",
                limit: MAX_REPLAY_EVENTS,
            });
        }
        for (index, event) in self.events.iter().enumerate() {
            let expected = u64::try_from(index).map_err(|_| ReplayError::LimitExceeded {
                resource: "event count",
                limit: MAX_REPLAY_EVENTS,
            })?;
            if event.step_index != expected {
                return Err(ReplayError::StepIndexMismatch {
                    expected,
                    actual: event.step_index,
                });
            }
            check_len(
                event.event_name.len(),
                MAX_EVENT_NAME_BYTES,
                "event-name bytes",
            )?;
            check_len(
                event.dimension_charges.len(),
                MAX_EVENT_CHARGES,
                "dimension-charge count",
            )?;
            check_len(
                event.payload.len(),
                MAX_EVENT_PAYLOAD_BYTES,
                "event-payload bytes",
            )?;
            if event
                .dimension_charges
                .iter()
                .any(|(_, amount)| *amount == 0)
            {
                return Err(ReplayError::ZeroCharge);
            }
            if event.outcome_digest
                != event_digest(
                    event.step_index,
                    &event.event_name,
                    &event.dimension_charges,
                    &event.payload,
                )?
            {
                return Err(ReplayError::Serialization(
                    "event digest does not match its recorded fields".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn computed_final_digest(&self) -> Result<[u8; 32], ReplayError> {
        let canonical_log =
            serde_json::to_vec(&(self.initial_seed, &self.strategy_name, &self.events))
                .map_err(|error| ReplayError::Serialization(error.to_string()))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.replay.log.v2:");
        hasher.update(&canonical_log);
        Ok(*hasher.finalize().as_bytes())
    }
}

fn check_len(actual: usize, limit: usize, resource: &'static str) -> Result<(), ReplayError> {
    if actual > limit {
        Err(ReplayError::LimitExceeded { resource, limit })
    } else {
        Ok(())
    }
}

fn event_digest(
    step_index: u64,
    event_name: &str,
    dimension_charges: &[(Dimension, u64)],
    payload: &[u8],
) -> Result<[u8; 32], ReplayError> {
    let canonical_event = serde_json::to_vec(&(step_index, event_name, dimension_charges, payload))
        .map_err(|error| ReplayError::Serialization(error.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.replay.event.v2:");
    hasher.update(&canonical_event);
    Ok(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_charges_are_integrity_bound() {
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome")
            .unwrap();
        log.finalize().unwrap();
        assert!(log.verify_integrity());

        log.events[0].dimension_charges[0].1 = 300;

        assert!(!log.verify_integrity());
    }

    #[test]
    fn payload_mutation_is_detected_without_a_reference_log() {
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome")
            .unwrap();
        log.finalize().unwrap();

        log.events[0].payload[0] ^= 0xff;

        assert!(!log.verify_integrity());
    }

    #[test]
    fn duplicate_or_out_of_order_indices_cannot_be_resealed() {
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome")
            .unwrap();
        log.events[0].step_index = 7;

        assert_eq!(
            log.finalize(),
            Err(ReplayError::StepIndexMismatch {
                expected: 0,
                actual: 7,
            })
        );
    }
}
