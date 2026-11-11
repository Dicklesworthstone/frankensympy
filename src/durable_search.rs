//! Opt-in file persistence for source-bound root-search continuations.
//!
//! A stored record is candidate state. Its digest locates immutable bytes; it
//! neither authenticates a sender nor proves any root. Restore checks the
//! receiving program, context, box, options and partial cover again, and final
//! acceptance still requires the independent complete-cover verifier.

#![forbid(unsafe_code)]

use crate::certification::{CertificationError, ResidualProblem};
use crate::search_checkpoint::{
    MAX_ROOT_SEARCH_CHECKPOINT_BYTES, export_root_search_checkpoint, restore_root_search_checkpoint,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_calculus::root_isolation::{RootIsolationOptions, RootIsolationSearch};
use fsym_core::RealBall;
use fsym_durable::{
    DependencyManifest, DurableError, DurableRecord, DurableStore, FileStore,
    MAX_RECORD_WIRE_BYTES, hex_lower,
};
use fsym_proof_kernel::capsule::object_digest;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;
use thiserror::Error;

pub const DURABLE_ROOT_SEARCH_SCHEMA: &str = "fsym.root_search.checkpoint.v1";

/// Exact immutable record identity, never a filename or a request for "latest".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredRootSearchCheckpoint {
    pub record_digest: [u8; 32],
    pub payload_bytes: usize,
}

#[derive(Debug, Error)]
pub enum DurableSearchError {
    #[error("root-search persistence is not enabled by the session host")]
    Disabled,
    #[error("stored root-search checkpoint has an invalid byte bound")]
    InvalidLength,
    #[error("stored root-search checkpoint differs from the requested record")]
    RecordMismatch,
    #[error("durable root-search operation refused: {0}")]
    Storage(#[from] DurableError),
    #[error("root-search checkpoint validation refused: {0}")]
    Certification(#[from] CertificationError),
    #[error("root-search storage resource accounting refused: {0}")]
    Meter(#[from] MeterError),
    #[error("staged root-search cleanup failed after {operation}: {cleanup}")]
    Cleanup {
        operation: String,
        cleanup: DurableError,
    },
}

/// An in-memory, fully encoded candidate awaiting an explicit disk commit.
/// Preparing does not write files, so output admission can precede publication.
pub struct PreparedRootSearchCheckpoint {
    record: DurableRecord,
    key: StoredRootSearchCheckpoint,
    wire_bytes: usize,
}

impl PreparedRootSearchCheckpoint {
    pub fn key(&self) -> StoredRootSearchCheckpoint {
        self.key
    }
}

/// Host-owned filesystem capability. Request envelopes cannot select paths.
pub struct RootSearchCheckpointStore {
    store: FileStore,
}

// The format and complete locked dependency universe must match. This is a
// portability policy, not a claim that a digest authenticates the executable.
// Source-level mathematical checks remain mandatory on every restoration.
pub(crate) fn dependencies() -> &'static DependencyManifest {
    static DEPENDENCIES: OnceLock<DependencyManifest> = OnceLock::new();
    DEPENDENCIES.get_or_init(|| {
        DependencyManifest::new([
            ("root_search_checkpoint_schema", "1".to_owned()),
            (
                "workspace_lock",
                hex_lower(&object_digest(include_bytes!("../Cargo.lock"))),
            ),
            (
                "donor_sources",
                hex_lower(&object_digest(include_bytes!(
                    "../registries/donor_sources.toml"
                ))),
            ),
        ])
    })
}

fn charge_io(meter: &mut impl BudgetMeter, bytes: usize) -> Result<(), DurableSearchError> {
    meter.checkpoint()?;
    // These are cumulative conservative work/allocation charges, not a claim
    // to preempt synchronous OS I/O. No protected verifier authority is used.
    meter.charge_batch(&[
        (Dimension::ComputeSteps, bytes as u64 + 1),
        (Dimension::MemoryBytes, bytes as u64 * 4 + 4096),
        (Dimension::AllocationCount, 32),
    ])?;
    Ok(())
}

pub(crate) fn charge_dependencies(meter: &mut impl BudgetMeter) -> Result<(), MeterError> {
    // Charge even on a cache hit; first-use hashing cannot become free work or
    // make the allowance depend on another session having warmed the cache.
    let bytes = (include_bytes!("../Cargo.lock").len()
        + include_bytes!("../registries/donor_sources.toml").len()) as u64;
    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, bytes + 1),
        (Dimension::MemoryBytes, bytes * 4 + 4096),
        (Dimension::AllocationCount, 32),
    ])
}

impl RootSearchCheckpointStore {
    /// Opens an explicitly selected, host-controlled local directory.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, DurableSearchError> {
        Ok(Self {
            store: FileStore::open(root)?,
        })
    }

    /// Captures candidate progress and accounts for its bounded durable wire
    /// encoding. No disk effects occur until [`Self::commit`].
    pub fn prepare(
        &self,
        search: &mut RootIsolationSearch,
        context_digest: [u8; 32],
        meter: &mut impl BudgetMeter,
    ) -> Result<PreparedRootSearchCheckpoint, DurableSearchError> {
        let payload = export_root_search_checkpoint(search, context_digest, meter)?;
        let payload_bytes = payload.len();
        // Byte-array JSON can expand to four bytes per source byte. Bound and
        // charge that envelope before serialization, including pinned metadata.
        let wire_bound = payload_bytes
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(4096))
            .filter(|bytes| *bytes <= MAX_RECORD_WIRE_BYTES)
            .ok_or(DurableSearchError::InvalidLength)?;
        charge_io(meter, wire_bound)?;
        charge_dependencies(meter)?;
        let record = DurableRecord::new(
            context_digest,
            DURABLE_ROOT_SEARCH_SCHEMA,
            payload,
            dependencies().clone(),
        )?;
        let wire_bytes = record.to_wire()?.len();
        let key = StoredRootSearchCheckpoint {
            record_digest: record.record_digest,
            payload_bytes,
        };
        meter.checkpoint()?;
        Ok(PreparedRootSearchCheckpoint {
            record,
            key,
            wire_bytes,
        })
    }

    /// Stages, reads back, verifies and atomically commits an exact checkpoint.
    /// Cancellation before the final commit releases staging; successful commit
    /// is the last cancellation boundary. An OS sync failure after rename may
    /// leave a visible record; callers can inspect/retry its exact key.
    pub fn commit(
        &self,
        prepared: PreparedRootSearchCheckpoint,
        meter: &mut impl BudgetMeter,
    ) -> Result<StoredRootSearchCheckpoint, DurableSearchError> {
        // Reserve all bounded disk/codec work before the first filesystem
        // effect. Cleanup does not depend on obtaining fresh generator budget.
        charge_io(meter, prepared.wire_bytes.saturating_mul(4))?;
        let attempt = (|| {
            self.store.prepare(&prepared.record)?;
            meter.checkpoint()?;
            let readback = self.store.verify_prepared_record(&prepared.record)?;
            readback.validate(prepared.record.universe_id, dependencies())?;
            if readback != prepared.record {
                return Err(DurableSearchError::RecordMismatch);
            }
            meter.checkpoint()?;
            self.store.commit_record(&prepared.record)?;
            Ok(prepared.key)
        })();
        match attempt {
            Ok(key) => Ok(key),
            Err(error) => {
                if let Err(cleanup) = self.store.release_prepared_record(&prepared.record) {
                    return Err(DurableSearchError::Cleanup {
                        operation: error.to_string(),
                        cleanup,
                    });
                }
                Err(error)
            }
        }
    }

    /// Loads one exact candidate checkpoint and independently restores it.
    /// Caller-owned ordinary accounting is cumulative; stored counters cannot
    /// replenish it or supply protected verifier authority.
    pub fn restore(
        &self,
        key: StoredRootSearchCheckpoint,
        problem: &ResidualProblem,
        domain: &[RealBall],
        options: RootIsolationOptions,
        context_digest: [u8; 32],
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationSearch, DurableSearchError> {
        if key.payload_bytes == 0 || key.payload_bytes > MAX_ROOT_SEARCH_CHECKPOINT_BYTES {
            return Err(DurableSearchError::InvalidLength);
        }
        // Exact lookup preflights the physical file against this bound before
        // reading, so a forged small handle cannot cause an uncharged large read.
        let wire_bound = (key.payload_bytes * 4 + 4096).min(MAX_RECORD_WIRE_BYTES);
        charge_io(meter, wire_bound)?;
        charge_dependencies(meter)?;
        let record = self.store.load_committed_exact(
            context_digest,
            DURABLE_ROOT_SEARCH_SCHEMA,
            key.record_digest,
            key.payload_bytes,
        )?;
        let payload = record.validate(context_digest, dependencies())?;
        meter.checkpoint()?;
        Ok(restore_root_search_checkpoint(
            payload,
            problem,
            domain,
            options,
            context_digest,
            meter,
        )?)
    }
}
