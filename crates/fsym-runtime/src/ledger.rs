//! Bounded in-memory hash-chain model for a future WS15 ledger adapter.
//!
//! This module provides integrity-checked ephemeral records only. It does not
//! perform durable I/O, transactions, crash recovery, or authoritative
//! publication; those remain responsibilities of the planned `fsym-ledger`
//! storage-neutral boundary and its optional persistence adapters.

#![forbid(unsafe_code)]

use crate::checkpoint::TypedCheckpoint;
use serde::Serialize;
use std::io::{self, Write};
use thiserror::Error;

/// Default maximum number of records retained by one ephemeral ledger.
pub const MAX_LEDGER_RECORDS: usize = 100_000;
/// Default maximum payload size of one record.
pub const MAX_LEDGER_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
/// Default aggregate payload size retained by one ephemeral ledger.
pub const MAX_LEDGER_TOTAL_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

/// Admission limits for in-memory ledger growth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerLimits {
    pub max_records: usize,
    pub max_payload_bytes: usize,
    pub max_total_payload_bytes: usize,
}

impl Default for LedgerLimits {
    fn default() -> Self {
        Self {
            max_records: MAX_LEDGER_RECORDS,
            max_payload_bytes: MAX_LEDGER_PAYLOAD_BYTES,
            max_total_payload_bytes: MAX_LEDGER_TOTAL_PAYLOAD_BYTES,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LedgerError {
    #[error("Ledger entry sequence out of order: expected {0}, got {1}")]
    SequenceMismatch(u64, u64),
    #[error("Ledger sequence cannot be represented as a u64")]
    SequenceExhausted,
    #[error("Ledger integrity failure: entry digest or chain state is corrupted")]
    CorruptedEntry,
    #[error("Entry not found for sequence ID {0}")]
    NotFound(u64),
    #[error("Ledger record limit exceeded ({0})")]
    RecordLimitExceeded(usize),
    #[error("Ledger payload byte limit exceeded ({0})")]
    PayloadLimitExceeded(usize),
    #[error("Ledger aggregate payload byte limit exceeded ({0})")]
    TotalPayloadLimitExceeded(usize),
    #[error("Ledger allocation failed")]
    AllocationFailure,
    #[error("Checkpoint serialization failed")]
    SerializationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterFailure {
    SizeLimit,
    Allocation,
}

struct BoundedPayloadWriter {
    bytes: Vec<u8>,
    max_bytes: usize,
    failure: Option<WriterFailure>,
}

impl BoundedPayloadWriter {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
            failure: None,
        }
    }
}

impl Write for BoundedPayloadWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(buffer.len()) else {
            self.failure = Some(WriterFailure::SizeLimit);
            return Err(io::Error::other("ledger payload size limit exceeded"));
        };
        if new_len > self.max_bytes {
            self.failure = Some(WriterFailure::SizeLimit);
            return Err(io::Error::other("ledger payload size limit exceeded"));
        }
        if self.bytes.try_reserve(buffer.len()).is_err() {
            self.failure = Some(WriterFailure::Allocation);
            return Err(io::Error::other("ledger payload allocation failed"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// An immutable record in an ephemeral hash chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerRecord {
    seq: u64,
    payload_digest: [u8; 32],
    payload: Vec<u8>,
    prev_record_hash: [u8; 32],
    record_hash: [u8; 32],
}

impl LedgerRecord {
    fn new(seq: u64, payload: Vec<u8>, prev_record_hash: [u8; 32]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.ledger.payload:");
        hasher.update(&payload);
        let payload_digest = *hasher.finalize().as_bytes();

        let mut record_hasher = blake3::Hasher::new();
        record_hasher.update(b"fsym.ledger.record.v1:");
        record_hasher.update(&seq.to_le_bytes());
        record_hasher.update(&payload_digest);
        record_hasher.update(&prev_record_hash);
        let record_hash = *record_hasher.finalize().as_bytes();

        Self {
            seq,
            payload_digest,
            payload,
            prev_record_hash,
            record_hash,
        }
    }

    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn payload_digest(&self) -> [u8; 32] {
        self.payload_digest
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn previous_record_hash(&self) -> [u8; 32] {
        self.prev_record_hash
    }

    pub fn record_hash(&self) -> [u8; 32] {
        self.record_hash
    }

    pub fn verify(&self, expected_previous: [u8; 32]) -> bool {
        if self.prev_record_hash != expected_previous {
            return false;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.ledger.payload:");
        hasher.update(&self.payload);
        if *hasher.finalize().as_bytes() != self.payload_digest {
            return false;
        }

        let mut record_hasher = blake3::Hasher::new();
        record_hasher.update(b"fsym.ledger.record.v1:");
        record_hasher.update(&self.seq.to_le_bytes());
        record_hasher.update(&self.payload_digest);
        record_hasher.update(&self.prev_record_hash);
        *record_hasher.finalize().as_bytes() == self.record_hash
    }
}

/// A bounded ephemeral append-only hash chain.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EphemeralLedger {
    records: Vec<LedgerRecord>,
    latest_hash: [u8; 32],
    total_payload_bytes: usize,
}

impl EphemeralLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[LedgerRecord] {
        &self.records
    }

    pub fn latest_hash(&self) -> [u8; 32] {
        self.latest_hash
    }

    pub fn total_payload_bytes(&self) -> usize {
        self.total_payload_bytes
    }

    /// Appends one payload under the default in-memory admission limits.
    pub fn append(&mut self, payload: Vec<u8>) -> Result<u64, LedgerError> {
        self.append_with_limits(payload, LedgerLimits::default())
    }

    /// Appends one payload under caller-provided in-memory admission limits.
    pub fn append_with_limits(
        &mut self,
        payload: Vec<u8>,
        limits: LedgerLimits,
    ) -> Result<u64, LedgerError> {
        if self.records.len() >= limits.max_records {
            return Err(LedgerError::RecordLimitExceeded(limits.max_records));
        }
        if payload.len() > limits.max_payload_bytes {
            return Err(LedgerError::PayloadLimitExceeded(limits.max_payload_bytes));
        }
        let new_total = self.total_payload_bytes.checked_add(payload.len()).ok_or(
            LedgerError::TotalPayloadLimitExceeded(limits.max_total_payload_bytes),
        )?;
        if new_total > limits.max_total_payload_bytes {
            return Err(LedgerError::TotalPayloadLimitExceeded(
                limits.max_total_payload_bytes,
            ));
        }
        let seq = u64::try_from(self.records.len()).map_err(|_| LedgerError::SequenceExhausted)?;
        self.records
            .try_reserve(1)
            .map_err(|_| LedgerError::AllocationFailure)?;

        let record = LedgerRecord::new(seq, payload, self.latest_hash);
        let record_hash = record.record_hash;
        self.records.push(record);
        self.latest_hash = record_hash;
        self.total_payload_bytes = new_total;
        Ok(seq)
    }

    /// Validates record sequence, payload digests, chain links, head, and the
    /// cached aggregate byte count.
    pub fn validate_chain(&self) -> Result<(), LedgerError> {
        let mut previous = [0u8; 32];
        let mut total_payload_bytes = 0_usize;
        for (index, record) in self.records.iter().enumerate() {
            let expected = u64::try_from(index).map_err(|_| LedgerError::SequenceExhausted)?;
            if record.seq != expected {
                return Err(LedgerError::SequenceMismatch(expected, record.seq));
            }
            if !record.verify(previous) {
                return Err(LedgerError::CorruptedEntry);
            }
            total_payload_bytes = total_payload_bytes
                .checked_add(record.payload.len())
                .ok_or(LedgerError::CorruptedEntry)?;
            previous = record.record_hash;
        }
        if previous != self.latest_hash || total_payload_bytes != self.total_payload_bytes {
            return Err(LedgerError::CorruptedEntry);
        }
        Ok(())
    }

    pub fn verify_chain(&self) -> bool {
        self.validate_chain().is_ok()
    }

    pub fn get(&self, seq: u64) -> Result<&LedgerRecord, LedgerError> {
        let index = usize::try_from(seq).map_err(|_| LedgerError::NotFound(seq))?;
        let record = self.records.get(index).ok_or(LedgerError::NotFound(seq))?;
        if record.seq != seq {
            return Err(LedgerError::SequenceMismatch(seq, record.seq));
        }
        Ok(record)
    }

    /// Appends an integrity-valid typed checkpoint under default limits.
    pub fn append_checkpoint<T: Serialize>(
        &mut self,
        checkpoint: &TypedCheckpoint<T>,
    ) -> Result<u64, LedgerError> {
        self.append_checkpoint_with_limits(checkpoint, LedgerLimits::default())
    }

    /// Appends an integrity-valid typed checkpoint under caller limits.
    pub fn append_checkpoint_with_limits<T: Serialize>(
        &mut self,
        checkpoint: &TypedCheckpoint<T>,
        limits: LedgerLimits,
    ) -> Result<u64, LedgerError> {
        if self.records.len() >= limits.max_records {
            return Err(LedgerError::RecordLimitExceeded(limits.max_records));
        }
        let remaining_total = limits
            .max_total_payload_bytes
            .checked_sub(self.total_payload_bytes)
            .ok_or(LedgerError::TotalPayloadLimitExceeded(
                limits.max_total_payload_bytes,
            ))?;
        if remaining_total == 0 {
            return Err(LedgerError::TotalPayloadLimitExceeded(
                limits.max_total_payload_bytes,
            ));
        }
        if limits.max_payload_bytes == 0 {
            return Err(LedgerError::PayloadLimitExceeded(0));
        }
        let serialized_limit = limits.max_payload_bytes.min(remaining_total);
        let mut writer = BoundedPayloadWriter::new(serialized_limit);
        if serde_json::to_writer(&mut writer, checkpoint).is_err() {
            return Err(match writer.failure {
                Some(WriterFailure::Allocation) => LedgerError::AllocationFailure,
                Some(WriterFailure::SizeLimit) if remaining_total < limits.max_payload_bytes => {
                    LedgerError::TotalPayloadLimitExceeded(limits.max_total_payload_bytes)
                }
                Some(WriterFailure::SizeLimit) => {
                    LedgerError::PayloadLimitExceeded(limits.max_payload_bytes)
                }
                None => LedgerError::SerializationFailed,
            });
        }
        let stored_checkpoint: TypedCheckpoint<serde_json::Value> =
            serde_json::from_slice(&writer.bytes).map_err(|_| LedgerError::SerializationFailed)?;
        if !stored_checkpoint.verify_integrity() {
            return Err(LedgerError::CorruptedEntry);
        }
        self.append_with_limits(writer.bytes, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Dimension;
    use serde::Serializer;
    use std::cell::Cell;
    use std::collections::BTreeMap;

    struct StatefulPayload {
        serializations: Cell<u64>,
    }

    impl Serialize for StatefulPayload {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let value = self.serializations.get();
            self.serializations.set(value + 1);
            serializer.serialize_u64(value)
        }
    }

    #[test]
    fn rejected_appends_do_not_mutate_chain_state() {
        let mut ledger = EphemeralLedger::new();
        let limits = LedgerLimits {
            max_records: 1,
            max_payload_bytes: 2,
            max_total_payload_bytes: 2,
        };
        assert_eq!(
            ledger.append_with_limits(vec![1, 2, 3], limits),
            Err(LedgerError::PayloadLimitExceeded(2))
        );
        assert!(ledger.is_empty());
        assert_eq!(ledger.append_with_limits(vec![1, 2], limits), Ok(0));
        let head = ledger.latest_hash();
        assert_eq!(
            ledger.append_with_limits(vec![], limits),
            Err(LedgerError::RecordLimitExceeded(1))
        );
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.latest_hash(), head);
        assert!(ledger.verify_chain());

        let aggregate_limits = LedgerLimits {
            max_records: 3,
            max_payload_bytes: 2,
            max_total_payload_bytes: 2,
        };
        assert_eq!(
            ledger.append_with_limits(vec![3], aggregate_limits),
            Err(LedgerError::TotalPayloadLimitExceeded(2))
        );
        assert_eq!(ledger.len(), 1);
        assert!(ledger.verify_chain());
    }

    #[test]
    fn tampering_is_detected_by_typed_validation() {
        let mut ledger = EphemeralLedger::new();
        ledger.append(b"audit".to_vec()).unwrap();
        ledger.records[0].payload[0] ^= u8::MAX;
        assert_eq!(ledger.validate_chain(), Err(LedgerError::CorruptedEntry));
    }

    #[test]
    fn checkpoint_verification_binds_the_exact_stored_serialization() {
        let checkpoint = TypedCheckpoint::new(
            "test.stateful.v1",
            0,
            StatefulPayload {
                serializations: Cell::new(0),
            },
            BTreeMap::<Dimension, u64>::new(),
            0,
        )
        .unwrap();
        let mut ledger = EphemeralLedger::new();
        assert_eq!(
            ledger.append_checkpoint(&checkpoint),
            Err(LedgerError::CorruptedEntry)
        );
        assert!(ledger.is_empty());
    }
}
