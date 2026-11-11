//! Opt-in, repairable file checkpoints for candidate root-isolation searches.
//!
//! Source and repair symbols are independently addressed durable records. The
//! exact manifest is published last, after all symbol records have been read
//! back and committed. Cancellation or failure before that point can leave at
//! most eighty unreferenced symbol records per attempt; it cannot publish a
//! partial manifest. This module does not garbage-collect shared records.
//!
//! RaptorQ restores bytes, never mathematical authority. Re-encoding the recovered
//! payload must reproduce the exact manifest, including metadata for erased
//! symbols. The payload then passes the ordinary portable checkpoint's source,
//! context, domain, options, canonical encoding and independent prefix checks.
//! Filesystem access and the codec are synchronous: bounded work is prepaid,
//! with cooperative cancellation between operations, not forced interruption
//! of a running filesystem call or codec operation.

#![forbid(unsafe_code)]

use crate::certification::{CertificationError, ResidualProblem};
use crate::durable_search::{DURABLE_ROOT_SEARCH_SCHEMA, charge_dependencies, dependencies};
use crate::search_checkpoint::{
    MAX_ROOT_SEARCH_CHECKPOINT_BYTES, export_root_search_checkpoint, restore_root_search_checkpoint,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_calculus::root_isolation::{RootIsolationOptions, RootIsolationSearch};
use fsym_core::RealBall;
use fsym_durable::{DurableError, DurableRecord, DurableStore, FileStore, MAX_RECORD_WIRE_BYTES};
use fsym_runtime::{RepairError, RepairSidecar};
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::marker::PhantomData;
use std::path::PathBuf;
use thiserror::Error;

pub const ROOT_SEARCH_REPAIR_MANIFEST_SCHEMA: &str = "fsym.root_search.repair_manifest.v1";
pub const ROOT_SEARCH_SOURCE_CHUNK_SCHEMA: &str = "fsym.root_search.source_symbol.v1";
pub const ROOT_SEARCH_REPAIR_CHUNK_SCHEMA: &str = "fsym.root_search.repair_symbol.v1";
pub const MAX_ROOT_SEARCH_REPAIR_PAYLOAD_BYTES: usize = 1024 * 1024;
pub const MAX_ROOT_SEARCH_REPAIR_MANIFEST_BYTES: usize = 64 * 1024;
const MAX_SOURCE_CHUNKS: usize = 64;
const REPAIR_CHUNKS: usize = 16;
const MIN_SYMBOL_BYTES: usize = 256;
const MAX_SYMBOL_BYTES: usize = 16 * 1024;
const CHUNK_HEADER_BYTES: usize = 4;
const MANIFEST_VERSION: u32 = 1;
const CODEC_VERSION: u16 = 1;

/// Exact manifest identity and bounded lengths. This is not an authorization
/// token; the receiving host separately owns the filesystem capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairableRootSearchCheckpoint {
    pub manifest_digest: [u8; 32],
    pub manifest_bytes: usize,
    pub payload_bytes: usize,
}

/// Observed symbol losses during one successful restoration. These counts
/// describe storage recovery and do not certify any root or complete cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootSearchRepairReport {
    pub source_chunks: usize,
    pub repair_chunks: usize,
    pub source_missing: usize,
    pub source_corrupt: usize,
    pub repair_missing: usize,
    pub repair_corrupt: usize,
    pub recovered: bool,
}

#[derive(Debug, Error)]
pub enum RepairableSearchError {
    #[error("repairable root-search checkpoint has an invalid byte bound")]
    InvalidLength,
    #[error("repairable root-search manifest is invalid: {0}")]
    InvalidManifest(&'static str),
    #[error("repairable root-search record differs from its prepared bytes")]
    RecordMismatch,
    #[error("repairable root-search storage refused: {0}")]
    Storage(#[from] DurableError),
    #[error("repairable root-search codec refused: {0}")]
    Repair(#[from] RepairError),
    #[error("repairable root-search checkpoint validation refused: {0}")]
    Certification(#[from] CertificationError),
    #[error("repairable root-search resource accounting refused: {0}")]
    Meter(#[from] MeterError),
    #[error("repairable root-search manifest encoding refused: {0}")]
    Json(#[from] serde_json::Error),
    #[error("staged repairable checkpoint cleanup failed after {operation}: {cleanup}")]
    Cleanup {
        operation: String,
        cleanup: DurableError,
    },
}

/// Fully encoded private state. Preparation has no filesystem effects, so a
/// caller can admit the response before requesting the final publication.
pub struct PreparedRepairableRootSearchCheckpoint {
    symbol_records: Vec<DurableRecord>,
    manifest_record: DurableRecord,
    key: RepairableRootSearchCheckpoint,
    wire_bound: usize,
}

impl PreparedRepairableRootSearchCheckpoint {
    pub fn key(&self) -> RepairableRootSearchCheckpoint {
        self.key
    }
}

/// A host-selected local directory. Request data never selects a path.
pub struct RepairableRootSearchCheckpointStore {
    store: FileStore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChunkKey {
    record_digest: [u8; 32],
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairManifest {
    schema_version: u32,
    checkpoint_schema: String,
    context_digest: [u8; 32],
    original_len: usize,
    symbol_size: usize,
    num_source_symbols: usize,
    codec_schema_version: u16,
    codec_seed: u64,
    canonical_digest: [u8; 32],
    #[serde(deserialize_with = "bounded_vec::<_, _, MAX_SOURCE_CHUNKS>")]
    source_symbol_digests: Vec<[u8; 32]>,
    #[serde(deserialize_with = "bounded_vec::<_, _, REPAIR_CHUNKS>")]
    repair_symbol_digests: Vec<[u8; 32]>,
    #[serde(deserialize_with = "bounded_vec::<_, _, MAX_SOURCE_CHUNKS>")]
    sources: Vec<ChunkKey>,
    #[serde(deserialize_with = "bounded_vec::<_, _, REPAIR_CHUNKS>")]
    repairs: Vec<ChunkKey>,
}

// Bound vectors while deserializing, before an attacker-controlled array can
// grow a Vec. Size hints never authorize allocations beyond the fixed policy.
fn bounded_vec<'de, D, T, const MAX: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Bounded<T, const MAX: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const MAX: usize> Visitor<'de> for Bounded<T, MAX> {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "an array containing at most {MAX} entries")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX));
            while let Some(value) = sequence.next_element()? {
                if values.len() == MAX {
                    return Err(de::Error::custom("repair manifest array exceeds policy"));
                }
                values.push(value);
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(Bounded::<T, MAX>(PhantomData))
}

#[derive(Clone, Copy)]
struct Layout {
    symbol_bytes: usize,
    source_chunks: usize,
}

impl Layout {
    fn for_payload(payload_bytes: usize) -> Result<Self, RepairableSearchError> {
        if payload_bytes == 0
            || payload_bytes > MAX_ROOT_SEARCH_REPAIR_PAYLOAD_BYTES
            || payload_bytes > MAX_ROOT_SEARCH_CHECKPOINT_BYTES
        {
            return Err(RepairableSearchError::InvalidLength);
        }
        let symbol_bytes = payload_bytes
            .div_ceil(MAX_SOURCE_CHUNKS)
            .max(MIN_SYMBOL_BYTES);
        let source_chunks = payload_bytes.div_ceil(symbol_bytes);
        if symbol_bytes > MAX_SYMBOL_BYTES || source_chunks > MAX_SOURCE_CHUNKS {
            return Err(RepairableSearchError::InvalidLength);
        }
        Ok(Self {
            symbol_bytes,
            source_chunks,
        })
    }

    fn chunk_bytes(self) -> usize {
        self.symbol_bytes + CHUNK_HEADER_BYTES
    }
}

fn wire_bound(payload_bytes: usize) -> Result<usize, RepairableSearchError> {
    payload_bytes
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4096))
        .filter(|bytes| *bytes <= MAX_RECORD_WIRE_BYTES)
        .ok_or(RepairableSearchError::InvalidLength)
}

fn charge_io(
    meter: &mut impl BudgetMeter,
    bytes: usize,
    records: usize,
) -> Result<(), RepairableSearchError> {
    meter.checkpoint()?;
    let bytes = u64::try_from(bytes).map_err(|_| RepairableSearchError::InvalidLength)?;
    let records = u64::try_from(records).map_err(|_| RepairableSearchError::InvalidLength)?;
    let steps = bytes
        .checked_mul(4)
        .and_then(|amount| amount.checked_add(records * 1024 + 1))
        .ok_or(RepairableSearchError::InvalidLength)?;
    let memory = bytes
        .checked_mul(8)
        .and_then(|amount| amount.checked_add(records * 4096 + 4096))
        .ok_or(RepairableSearchError::InvalidLength)?;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, steps),
        (Dimension::MemoryBytes, memory),
        (Dimension::AllocationCount, records * 64 + 64),
    ])?;
    Ok(())
}

fn charge_codec(meter: &mut impl BudgetMeter, layout: Layout) -> Result<(), RepairableSearchError> {
    meter.checkpoint()?;
    // Reserve matrix elimination and symbol-row work, including the codec's
    // constraint rows and transient copies. This bounded policy never exposes
    // the codec's much larger general-purpose limits to request data.
    let rows = (layout.source_chunks + 64) as u64;
    let width = layout.symbol_bytes as u64;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, rows * rows * (rows + width) * 4),
        (
            Dimension::MemoryBytes,
            rows * rows * 128 + rows * width * 16 + 64 * 1024,
        ),
        (Dimension::AllocationCount, rows * rows + 256),
    ])?;
    Ok(())
}

impl RepairableRootSearchCheckpointStore {
    /// Opens a trusted host directory. This constructor creates the directory;
    /// individual preparations remain entirely in memory.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, RepairableSearchError> {
        Ok(Self {
            store: FileStore::open(root)?,
        })
    }

    pub fn prepare(
        &self,
        search: &mut RootIsolationSearch,
        context_digest: [u8; 32],
        meter: &mut impl BudgetMeter,
    ) -> Result<PreparedRepairableRootSearchCheckpoint, RepairableSearchError> {
        let payload = export_root_search_checkpoint(search, context_digest, meter)?;
        self.prepare_payload(&payload, context_digest, meter)
    }

    fn prepare_payload(
        &self,
        payload: &[u8],
        context_digest: [u8; 32],
        meter: &mut impl BudgetMeter,
    ) -> Result<PreparedRepairableRootSearchCheckpoint, RepairableSearchError> {
        let layout = Layout::for_payload(payload.len())?;
        charge_dependencies(meter)?;
        charge_codec(meter, layout)?;
        let sidecar = RepairSidecar::encode(payload, layout.symbol_bytes, REPAIR_CHUNKS)?;
        meter.checkpoint()?;
        let symbol_count = layout.source_chunks + REPAIR_CHUNKS;
        let symbols_wire_bound = wire_bound(layout.chunk_bytes())? * symbol_count;
        charge_io(
            meter,
            symbols_wire_bound + wire_bound(MAX_ROOT_SEARCH_REPAIR_MANIFEST_BYTES)?,
            symbol_count + 1,
        )?;

        let mut symbol_records = Vec::with_capacity(symbol_count);
        let mut sources = Vec::with_capacity(layout.source_chunks);
        for index in 0..layout.source_chunks {
            meter.checkpoint()?;
            let start = index * layout.symbol_bytes;
            let end = (start + layout.symbol_bytes).min(payload.len());
            let mut chunk = vec![0; layout.chunk_bytes()];
            chunk[..CHUNK_HEADER_BYTES].copy_from_slice(&(index as u32).to_le_bytes());
            chunk[CHUNK_HEADER_BYTES..CHUNK_HEADER_BYTES + end - start]
                .copy_from_slice(&payload[start..end]);
            let record = DurableRecord::new(
                context_digest,
                ROOT_SEARCH_SOURCE_CHUNK_SCHEMA,
                chunk,
                dependencies().clone(),
            )?;
            sources.push(ChunkKey {
                record_digest: record.record_digest,
            });
            symbol_records.push(record);
        }
        let mut repairs = Vec::with_capacity(REPAIR_CHUNKS);
        for (esi, symbol) in &sidecar.repair_symbols {
            meter.checkpoint()?;
            let mut chunk = Vec::with_capacity(layout.chunk_bytes());
            chunk.extend_from_slice(&esi.to_le_bytes());
            chunk.extend_from_slice(symbol);
            let record = DurableRecord::new(
                context_digest,
                ROOT_SEARCH_REPAIR_CHUNK_SCHEMA,
                chunk,
                dependencies().clone(),
            )?;
            repairs.push(ChunkKey {
                record_digest: record.record_digest,
            });
            symbol_records.push(record);
        }
        let manifest = RepairManifest {
            schema_version: MANIFEST_VERSION,
            checkpoint_schema: DURABLE_ROOT_SEARCH_SCHEMA.into(),
            context_digest,
            original_len: payload.len(),
            symbol_size: sidecar.symbol_size,
            num_source_symbols: sidecar.num_source_symbols,
            codec_schema_version: sidecar.schema_version,
            codec_seed: sidecar.codec_seed,
            canonical_digest: sidecar.canonical_digest,
            source_symbol_digests: sidecar.source_symbol_digests,
            repair_symbol_digests: sidecar.repair_symbol_digests,
            sources,
            repairs,
        };
        let manifest_bytes = serde_json::to_vec(&manifest)?;
        if manifest_bytes.len() > MAX_ROOT_SEARCH_REPAIR_MANIFEST_BYTES {
            return Err(RepairableSearchError::InvalidLength);
        }
        let manifest_len = manifest_bytes.len();
        let manifest_record = DurableRecord::new(
            context_digest,
            ROOT_SEARCH_REPAIR_MANIFEST_SCHEMA,
            manifest_bytes,
            dependencies().clone(),
        )?;
        let key = RepairableRootSearchCheckpoint {
            manifest_digest: manifest_record.record_digest,
            manifest_bytes: manifest_len,
            payload_bytes: payload.len(),
        };
        meter.checkpoint()?;
        Ok(PreparedRepairableRootSearchCheckpoint {
            symbol_records,
            manifest_record,
            key,
            wire_bound: symbols_wire_bound + wire_bound(manifest_len)?,
        })
    }

    /// Commits symbols first and the exact manifest last. Failure can leave
    /// bounded orphan symbol records; it never substitutes an older checkpoint
    /// or reports cancellation after successful manifest publication.
    pub fn commit(
        &self,
        prepared: PreparedRepairableRootSearchCheckpoint,
        meter: &mut impl BudgetMeter,
    ) -> Result<RepairableRootSearchCheckpoint, RepairableSearchError> {
        charge_io(
            meter,
            prepared.wire_bound * 4,
            prepared.symbol_records.len() + 1,
        )?;
        for record in &prepared.symbol_records {
            self.commit_record(record, meter)?;
        }
        self.commit_record(&prepared.manifest_record, meter)?;
        // No cancellation check after the authoritative manifest is visible.
        Ok(prepared.key)
    }

    fn commit_record(
        &self,
        record: &DurableRecord,
        meter: &mut impl BudgetMeter,
    ) -> Result<(), RepairableSearchError> {
        meter.checkpoint()?;
        let attempt = (|| {
            self.store.prepare(record)?;
            meter.checkpoint()?;
            let readback = self.store.verify_prepared_record(record)?;
            readback.validate(record.universe_id, dependencies())?;
            if &readback != record {
                return Err(RepairableSearchError::RecordMismatch);
            }
            meter.checkpoint()?;
            self.store.commit_record(record)?;
            Ok(())
        })();
        if let Err(error) = attempt {
            if let Err(cleanup) = self.store.release_prepared_record(record) {
                return Err(RepairableSearchError::Cleanup {
                    operation: error.to_string(),
                    cleanup,
                });
            }
            return Err(error);
        }
        Ok(())
    }

    /// Restores the exact candidate after byte recovery and independent source
    /// checks. One decoder pass and one deterministic encoder pass are prepaid
    /// on the ordinary meter. Backend/permission errors remain errors; only
    /// absent or invalid symbol records are erasures. A bad manifest refuses.
    #[allow(clippy::too_many_arguments)]
    pub fn restore(
        &self,
        key: RepairableRootSearchCheckpoint,
        problem: &ResidualProblem,
        domain: &[RealBall],
        options: RootIsolationOptions,
        context_digest: [u8; 32],
        meter: &mut impl BudgetMeter,
    ) -> Result<(RootIsolationSearch, RootSearchRepairReport), RepairableSearchError> {
        let (payload, report) = self.restore_payload(key, context_digest, meter)?;
        let search = restore_root_search_checkpoint(
            &payload,
            problem,
            domain,
            options,
            context_digest,
            meter,
        )?;
        Ok((search, report))
    }

    fn restore_payload(
        &self,
        key: RepairableRootSearchCheckpoint,
        context_digest: [u8; 32],
        meter: &mut impl BudgetMeter,
    ) -> Result<(Vec<u8>, RootSearchRepairReport), RepairableSearchError> {
        let layout = Layout::for_payload(key.payload_bytes)?;
        if key.manifest_bytes == 0 || key.manifest_bytes > MAX_ROOT_SEARCH_REPAIR_MANIFEST_BYTES {
            return Err(RepairableSearchError::InvalidLength);
        }
        charge_dependencies(meter)?;
        charge_io(meter, wire_bound(key.manifest_bytes)?, 1)?;
        let record = self.store.load_committed_exact(
            context_digest,
            ROOT_SEARCH_REPAIR_MANIFEST_SCHEMA,
            key.manifest_digest,
            key.manifest_bytes,
        )?;
        let bytes = record.validate(context_digest, dependencies())?;
        let manifest: RepairManifest = serde_json::from_slice(bytes)?;
        validate_manifest(&manifest, key, context_digest, layout)?;
        if serde_json::to_vec(&manifest)?.as_slice() != bytes {
            return Err(RepairableSearchError::InvalidManifest(
                "noncanonical encoding",
            ));
        }
        meter.checkpoint()?;
        let symbol_count = layout.source_chunks + REPAIR_CHUNKS;
        charge_io(
            meter,
            wire_bound(layout.chunk_bytes())? * symbol_count,
            symbol_count,
        )?;
        charge_codec(meter, layout)?;
        let mut report = RootSearchRepairReport {
            source_chunks: layout.source_chunks,
            repair_chunks: REPAIR_CHUNKS,
            source_missing: 0,
            source_corrupt: 0,
            repair_missing: 0,
            repair_corrupt: 0,
            recovered: false,
        };
        let mut sources = Vec::with_capacity(layout.source_chunks);
        for (index, key) in manifest.sources.iter().enumerate() {
            meter.checkpoint()?;
            match self.load_chunk(
                context_digest,
                ROOT_SEARCH_SOURCE_CHUNK_SCHEMA,
                *key,
                layout,
                index as u32,
            )? {
                ChunkRead::Present(symbol) => sources.push(Some(symbol)),
                ChunkRead::Missing => {
                    report.source_missing += 1;
                    sources.push(None);
                }
                ChunkRead::Corrupt => {
                    report.source_corrupt += 1;
                    sources.push(None);
                }
            }
        }
        let mut repair_symbols = Vec::with_capacity(REPAIR_CHUNKS);
        let mut repair_symbol_digests = Vec::with_capacity(REPAIR_CHUNKS);
        for (index, key) in manifest.repairs.iter().enumerate() {
            meter.checkpoint()?;
            let esi = (layout.source_chunks + index) as u32;
            match self.load_chunk(
                context_digest,
                ROOT_SEARCH_REPAIR_CHUNK_SCHEMA,
                *key,
                layout,
                esi,
            )? {
                ChunkRead::Present(symbol) => {
                    repair_symbols.push((esi, symbol));
                    repair_symbol_digests.push(manifest.repair_symbol_digests[index]);
                }
                ChunkRead::Missing => report.repair_missing += 1,
                ChunkRead::Corrupt => report.repair_corrupt += 1,
            }
        }
        let available =
            sources.iter().filter(|symbol| symbol.is_some()).count() + repair_symbols.len();
        if available < layout.source_chunks {
            return Err(RepairError::InsufficientSymbols(available, layout.source_chunks).into());
        }
        // The outer manifest always names all sixteen repairs. The codec
        // envelope contains the available subset with its original ESIs and
        // expected digests; missing repairs do not require fabricated bytes.
        let sidecar = RepairSidecar {
            schema_version: manifest.codec_schema_version,
            original_len: manifest.original_len,
            symbol_size: manifest.symbol_size,
            num_source_symbols: manifest.num_source_symbols,
            codec_seed: manifest.codec_seed,
            source_symbol_digests: manifest.source_symbol_digests,
            repair_symbols,
            repair_symbol_digests,
            canonical_digest: manifest.canonical_digest,
        };
        meter.checkpoint()?;
        let payload = sidecar.reconstruct(&sources, &sidecar.repair_symbols)?;
        meter.checkpoint()?;
        // The codec validates hashes for received symbols, but an erased
        // symbol's claimed hash/reference otherwise goes unused. Recreate the
        // complete deterministic manifest in memory and compare its exact key,
        // so a forged erased-symbol digest or address cannot be accepted merely
        // because the remaining equations recover the expected payload bytes.
        // This performs no disk writes and pays all encoding/accounting again.
        drop(sources);
        drop(sidecar);
        let regenerated = self.prepare_payload(&payload, context_digest, meter)?;
        if regenerated.key() != key {
            return Err(RepairableSearchError::InvalidManifest(
                "recovered payload does not reproduce the exact manifest",
            ));
        }
        drop(regenerated);
        report.recovered = report.source_missing + report.source_corrupt != 0;
        Ok((payload, report))
    }

    fn load_chunk(
        &self,
        context_digest: [u8; 32],
        schema: &str,
        key: ChunkKey,
        layout: Layout,
        expected_esi: u32,
    ) -> Result<ChunkRead, RepairableSearchError> {
        let read = (|| {
            let record = self.store.load_committed_exact(
                context_digest,
                schema,
                key.record_digest,
                layout.chunk_bytes(),
            )?;
            record.validate(context_digest, dependencies())?;
            Ok::<_, DurableError>(record)
        })();
        match read {
            Ok(record) => {
                if record.payload[..CHUNK_HEADER_BYTES] != expected_esi.to_le_bytes() {
                    return Ok(ChunkRead::Corrupt);
                }
                Ok(ChunkRead::Present(
                    record.payload[CHUNK_HEADER_BYTES..].to_vec(),
                ))
            }
            Err(DurableError::Absent) => Ok(ChunkRead::Missing),
            Err(
                DurableError::RecordTooLarge { .. }
                | DurableError::MalformedRecord(_)
                | DurableError::DigestMismatch { .. }
                | DurableError::SchemaMismatch { .. }
                | DurableError::UniverseMismatch { .. }
                | DurableError::DependencyMismatch { .. }
                | DurableError::DependencyMissing(_)
                | DurableError::DependencyUnexpected(_),
            ) => Ok(ChunkRead::Corrupt),
            Err(error) => Err(error.into()),
        }
    }
}

enum ChunkRead {
    Present(Vec<u8>),
    Missing,
    Corrupt,
}

fn validate_manifest(
    manifest: &RepairManifest,
    key: RepairableRootSearchCheckpoint,
    context_digest: [u8; 32],
    layout: Layout,
) -> Result<(), RepairableSearchError> {
    if manifest.schema_version != MANIFEST_VERSION
        || manifest.codec_schema_version != CODEC_VERSION
        || manifest.checkpoint_schema != DURABLE_ROOT_SEARCH_SCHEMA
    {
        return Err(RepairableSearchError::InvalidManifest(
            "schema or codec version mismatch",
        ));
    }
    if manifest.context_digest != context_digest || manifest.original_len != key.payload_bytes {
        return Err(RepairableSearchError::InvalidManifest(
            "context or payload length mismatch",
        ));
    }
    if manifest.symbol_size != layout.symbol_bytes
        || manifest.num_source_symbols != layout.source_chunks
        || manifest.sources.len() != layout.source_chunks
        || manifest.source_symbol_digests.len() != layout.source_chunks
        || manifest.repairs.len() != REPAIR_CHUNKS
        || manifest.repair_symbol_digests.len() != REPAIR_CHUNKS
    {
        return Err(RepairableSearchError::InvalidManifest(
            "symbol layout differs from bounded policy",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, Unbounded};
    use fsym_durable::hex_lower;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;

    const CONTEXT: [u8; 32] = [7; 32];

    fn directory() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "fsym-root-repair-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn payload() -> Vec<u8> {
        (0..8192)
            .map(|index| ((index * 73 + index / 11) % 251) as u8)
            .collect()
    }

    fn record_path(root: &Path, record: &DurableRecord) -> PathBuf {
        root.join(hex_lower(&record.universe_id))
            .join(&record.payload_schema)
            .join("committed")
            .join(format!(
                "{}-{}.record",
                hex_lower(&record.record_digest),
                record.payload.len(),
            ))
    }

    fn erase(path: &Path) {
        // Preserve the fixture bytes while removing the exact lookup location.
        fs::rename(path, path.with_extension("erased")).unwrap();
    }

    fn publish_manifest(
        store: &RepairableRootSearchCheckpointStore,
        bytes: Vec<u8>,
        payload_bytes: usize,
    ) -> RepairableRootSearchCheckpoint {
        let record = DurableRecord::new(
            CONTEXT,
            ROOT_SEARCH_REPAIR_MANIFEST_SCHEMA,
            bytes,
            dependencies().clone(),
        )
        .unwrap();
        let key = RepairableRootSearchCheckpoint {
            manifest_digest: record.record_digest,
            manifest_bytes: record.payload.len(),
            payload_bytes,
        };
        store.store.prepare(&record).unwrap();
        store.store.verify_prepared_record(&record).unwrap();
        store.store.commit_record(&record).unwrap();
        key
    }

    #[test]
    fn disk_source_and_repair_losses_reconstruct_the_exact_original_bytes() {
        let root = directory();
        let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
        let payload = payload();
        let prepared = store
            .prepare_payload(&payload, CONTEXT, &mut Unbounded)
            .unwrap();
        let layout = Layout::for_payload(payload.len()).unwrap();
        let paths: Vec<_> = prepared
            .symbol_records
            .iter()
            .map(|record| record_path(&root, record))
            .collect();
        let key = store.commit(prepared, &mut Unbounded).unwrap();
        erase(&paths[0]);
        fs::write(&paths[1], b"corrupt source record").unwrap();
        erase(&paths[layout.source_chunks]);
        fs::write(&paths[layout.source_chunks + 1], b"corrupt repair record").unwrap();

        let reopened = RepairableRootSearchCheckpointStore::open(&root).unwrap();
        let (recovered, report) = reopened
            .restore_payload(key, CONTEXT, &mut Unbounded)
            .unwrap();
        assert_eq!(recovered, payload);
        assert_eq!(report.source_missing, 1);
        assert_eq!(report.source_corrupt, 1);
        assert_eq!(report.repair_missing, 1);
        assert_eq!(report.repair_corrupt, 1);
        assert!(report.recovered);
    }

    #[test]
    fn intact_sources_need_no_repair_records_and_excess_source_loss_refuses() {
        for erase_repairs in [true, false] {
            let root = directory();
            let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
            let payload = payload();
            let prepared = store
                .prepare_payload(&payload, CONTEXT, &mut Unbounded)
                .unwrap();
            let layout = Layout::for_payload(payload.len()).unwrap();
            let paths: Vec<_> = prepared
                .symbol_records
                .iter()
                .map(|record| record_path(&root, record))
                .collect();
            let key = store.commit(prepared, &mut Unbounded).unwrap();
            if erase_repairs {
                for path in &paths[layout.source_chunks..] {
                    erase(path);
                }
                let (restored, report) =
                    store.restore_payload(key, CONTEXT, &mut Unbounded).unwrap();
                assert_eq!(restored, payload);
                assert_eq!(report.repair_missing, REPAIR_CHUNKS);
                assert!(!report.recovered);
            } else {
                for path in &paths[..REPAIR_CHUNKS + 1] {
                    erase(path);
                }
                assert!(matches!(
                    store.restore_payload(key, CONTEXT, &mut Unbounded),
                    Err(RepairableSearchError::Repair(
                        RepairError::InsufficientSymbols(_, _)
                    ))
                ));
            }
        }
    }

    #[test]
    fn codec_seed_and_every_available_symbol_digest_remain_checked() {
        let root = directory();
        let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
        let payload = payload();
        let prepared = store
            .prepare_payload(&payload, CONTEXT, &mut Unbounded)
            .unwrap();
        let manifest_bytes = prepared.manifest_record.payload.clone();
        store.commit(prepared, &mut Unbounded).unwrap();
        for mutation in 0..3 {
            let mut manifest: RepairManifest = serde_json::from_slice(&manifest_bytes).unwrap();
            match mutation {
                0 => manifest.codec_seed ^= 1,
                1 => manifest.source_symbol_digests[0][0] ^= 1,
                _ => manifest.repair_symbol_digests[0][0] ^= 1,
            }
            let key = publish_manifest(
                &store,
                serde_json::to_vec(&manifest).unwrap(),
                payload.len(),
            );
            let error = store
                .restore_payload(key, CONTEXT, &mut Unbounded)
                .unwrap_err();
            assert!(matches!(
                (mutation, error),
                (
                    0,
                    RepairableSearchError::Repair(RepairError::InvalidManifest(
                        "codec seed mismatch"
                    ))
                ) | (
                    1,
                    RepairableSearchError::Repair(RepairError::SourceSymbolDigestMismatch(0))
                ) | (
                    2,
                    RepairableSearchError::Repair(RepairError::RepairSymbolDigestMismatch(_))
                )
            ));
        }
    }

    #[test]
    fn erased_symbol_hashes_and_addresses_must_match_the_recovered_payload() {
        let root = directory();
        let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
        let payload = payload();
        let prepared = store
            .prepare_payload(&payload, CONTEXT, &mut Unbounded)
            .unwrap();
        let original = prepared.manifest_record.payload.clone();
        let source_count = Layout::for_payload(payload.len()).unwrap().source_chunks;
        let missing_source = record_path(&root, &prepared.symbol_records[0]);
        let missing_repair = record_path(&root, &prepared.symbol_records[source_count]);
        let original_key = store.commit(prepared, &mut Unbounded).unwrap();
        erase(&missing_source);
        erase(&missing_repair);
        assert_eq!(
            store
                .restore_payload(original_key, CONTEXT, &mut Unbounded)
                .unwrap()
                .0,
            payload,
        );
        for mutation in 0..4 {
            let mut manifest: RepairManifest = serde_json::from_slice(&original).unwrap();
            match mutation {
                0 => manifest.source_symbol_digests[0][0] ^= 1,
                1 => manifest.repair_symbol_digests[0][0] ^= 1,
                2 => manifest.sources[0].record_digest[0] ^= 1,
                _ => manifest.repairs[0].record_digest[0] ^= 1,
            }
            let key = publish_manifest(
                &store,
                serde_json::to_vec(&manifest).unwrap(),
                payload.len(),
            );
            assert!(matches!(
                store.restore_payload(key, CONTEXT, &mut Unbounded),
                Err(RepairableSearchError::InvalidManifest(
                    "recovered payload does not reproduce the exact manifest"
                ))
            ));
        }
    }

    #[test]
    fn manifest_bounds_canonical_encoding_and_context_are_not_repairable_erasures() {
        let root = directory();
        let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
        let payload = payload();
        let prepared = store
            .prepare_payload(&payload, CONTEXT, &mut Unbounded)
            .unwrap();
        let original = prepared.manifest_record.payload.clone();
        let manifest_path = record_path(&root, &prepared.manifest_record);
        let original_key = store.commit(prepared, &mut Unbounded).unwrap();
        for mutation in 0..4 {
            let mut manifest: RepairManifest = serde_json::from_slice(&original).unwrap();
            match mutation {
                0 => manifest.num_source_symbols = usize::MAX,
                1 => manifest.codec_schema_version += 1,
                2 => manifest.sources = vec![manifest.sources[0]; MAX_SOURCE_CHUNKS + 1],
                _ => manifest.context_digest[0] ^= 1,
            }
            let key = publish_manifest(
                &store,
                serde_json::to_vec(&manifest).unwrap(),
                payload.len(),
            );
            assert!(matches!(
                store.restore_payload(key, CONTEXT, &mut Unbounded),
                Err(RepairableSearchError::InvalidManifest(_) | RepairableSearchError::Json(_))
            ));
        }
        let mut noncanonical = vec![b' '];
        noncanonical.extend_from_slice(&original);
        let key = publish_manifest(&store, noncanonical, payload.len());
        assert!(matches!(
            store.restore_payload(key, CONTEXT, &mut Unbounded),
            Err(RepairableSearchError::InvalidManifest(
                "noncanonical encoding"
            ))
        ));
        fs::write(manifest_path, b"corrupt manifest record").unwrap();
        assert!(matches!(
            store.restore_payload(original_key, CONTEXT, &mut Unbounded),
            Err(RepairableSearchError::Storage(_))
        ));
    }

    #[test]
    fn recovered_checkpoint_still_requires_the_original_mathematical_source() {
        let root = directory();
        let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
        let problem =
            ResidualProblem::parse_sources(&["x^2-2".into()], &["x".into()], &mut Unbounded)
                .unwrap();
        let other =
            ResidualProblem::parse_sources(&["x^2-3".into()], &["x".into()], &mut Unbounded)
                .unwrap();
        let domain = vec![RealBall::new(0.into(), 2.into()).unwrap()];
        let options = RootIsolationOptions {
            precision_digits: 16,
            max_leaves: 128,
        };
        let mut search = problem
            .begin_isolation_search(&domain, options, &mut Unbounded)
            .unwrap();
        let prepared = store.prepare(&mut search, CONTEXT, &mut Unbounded).unwrap();
        let source_path = record_path(&root, &prepared.symbol_records[0]);
        let key = store.commit(prepared, &mut Unbounded).unwrap();
        erase(&source_path);
        assert!(matches!(
            store.restore(key, &other, &domain, options, CONTEXT, &mut Unbounded),
            Err(RepairableSearchError::Certification(
                CertificationError::SourceMismatch
            ))
        ));
        let (restored, report) = store
            .restore(key, &problem, &domain, options, CONTEXT, &mut Unbounded)
            .unwrap();
        assert_eq!(restored.residuals(), problem.residuals());
        assert_eq!(restored.variables(), problem.variables());
        assert_eq!(restored.domain(), domain);
        assert!(report.recovered);
    }

    struct CancelAfter {
        checkpoints: usize,
    }

    impl BudgetMeter for CancelAfter {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.checkpoints == 0 {
                return Err(MeterError::Cancelled);
            }
            self.checkpoints -= 1;
            Ok(())
        }
    }

    #[test]
    fn cancellation_and_exhaustion_cannot_publish_a_manifest_or_spend_verifier_budget() {
        for cancel in [true, false] {
            let root = directory();
            let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
            let prepared = store
                .prepare_payload(&payload(), CONTEXT, &mut Unbounded)
                .unwrap();
            let key = prepared.key();
            let manifest_path = record_path(&root, &prepared.manifest_record);
            let first_source = record_path(&root, &prepared.symbol_records[0]);
            if cancel {
                // Charge boundary plus one symbol's prepare/readback/commit
                // checkpoints: the next symbol refuses, before the manifest.
                assert!(matches!(
                    store.commit(prepared, &mut CancelAfter { checkpoints: 4 }),
                    Err(RepairableSearchError::Meter(MeterError::Cancelled))
                ));
                assert!(first_source.exists());
            } else {
                let mut budget = Budget::new(BudgetLimits::uniform(0, 1000));
                assert!(matches!(
                    store.commit(prepared, &mut budget),
                    Err(RepairableSearchError::Meter(MeterError::Budget(_)))
                ));
                assert_eq!(budget.verifier_remaining(), 1000);
                assert!(!first_source.exists());
            }
            assert!(!manifest_path.exists());
            assert!(matches!(
                store.restore_payload(key, CONTEXT, &mut Unbounded),
                Err(RepairableSearchError::Storage(DurableError::Absent))
            ));
        }
    }
}
