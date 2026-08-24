//! Durable ledger and journal for checkpoints and audit records (WS15).

#![forbid(unsafe_code)]

use crate::checkpoint::TypedCheckpoint;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LedgerError {
    #[error("Ledger entry sequence out of order: expected {0}, got {1}")]
    SequenceMismatch(u64, u64),
    #[error("Ledger integrity failure: entry digest corrupted")]
    CorruptedEntry,
    #[error("Entry not found for sequence ID {0}")]
    NotFound(u64),
}

/// An immutable record in the durable ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerRecord {
    pub seq: u64,
    pub payload_digest: [u8; 32],
    pub payload: Vec<u8>,
    pub prev_record_hash: [u8; 32],
    pub record_hash: [u8; 32],
}

impl LedgerRecord {
    pub fn new(seq: u64, payload: Vec<u8>, prev_record_hash: [u8; 32]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.ledger.payload:");
        hasher.update(&payload);
        let payload_digest = *hasher.finalize().as_bytes();

        let mut r_hasher = blake3::Hasher::new();
        r_hasher.update(b"fsym.ledger.record.v1:");
        r_hasher.update(&seq.to_le_bytes());
        r_hasher.update(&payload_digest);
        r_hasher.update(&prev_record_hash);
        let record_hash = *r_hasher.finalize().as_bytes();

        Self {
            seq,
            payload_digest,
            payload,
            prev_record_hash,
            record_hash,
        }
    }

    pub fn verify(&self, expected_prev: [u8; 32]) -> bool {
        if self.prev_record_hash != expected_prev {
            return false;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.ledger.payload:");
        hasher.update(&self.payload);
        if *hasher.finalize().as_bytes() != self.payload_digest {
            return false;
        }

        let mut r_hasher = blake3::Hasher::new();
        r_hasher.update(b"fsym.ledger.record.v1:");
        r_hasher.update(&self.seq.to_le_bytes());
        r_hasher.update(&self.payload_digest);
        r_hasher.update(&self.prev_record_hash);
        *r_hasher.finalize().as_bytes() == self.record_hash
    }
}

/// Durable Append-Only Ledger.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableLedger {
    pub records: Vec<LedgerRecord>,
    pub latest_hash: [u8; 32],
}

impl DurableLedger {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            latest_hash: [0u8; 32],
        }
    }

    /// Appends a new payload record to the ledger.
    pub fn append(&mut self, payload: Vec<u8>) -> u64 {
        let seq = self.records.len() as u64;
        let record = LedgerRecord::new(seq, payload, self.latest_hash);
        self.latest_hash = record.record_hash;
        self.records.push(record);
        seq
    }

    /// Verifies the entire hash chain of the ledger from genesis.
    pub fn verify_chain(&self) -> bool {
        let mut prev = [0u8; 32];
        for (i, r) in self.records.iter().enumerate() {
            if r.seq != i as u64 {
                return false;
            }
            if !r.verify(prev) {
                return false;
            }
            prev = r.record_hash;
        }
        prev == self.latest_hash
    }

    /// Appends a typed checkpoint to the ledger.
    pub fn append_checkpoint<T: Serialize>(
        &mut self,
        checkpoint: &TypedCheckpoint<T>,
    ) -> Result<u64, LedgerError> {
        if !checkpoint.verify_integrity() {
            return Err(LedgerError::CorruptedEntry);
        }
        let serialized = serde_json::to_vec(checkpoint).map_err(|_| LedgerError::CorruptedEntry)?;
        Ok(self.append(serialized))
    }
}
