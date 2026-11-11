//! Bounded staging and transfer of opaque artifact bytes.
//!
//! Content IDs and chunk checksums establish byte integrity only. Callers must
//! independently decode the canonical artifact and verify its mathematical
//! statement before publishing any accepted object.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_proof_kernel::capsule::object_digest;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, ops::Range};
use thiserror::Error;

pub use crate::certification::MAX_ROOT_CERTIFICATE_BYTES as MAX_ARTIFACT_BYTES;
pub const ARTIFACT_STREAM_SCHEMA_VERSION: u32 = 1;
/// Hex encoding leaves each chunk well below the 64 KiB NDJSON frame limit.
pub const ARTIFACT_CHUNK_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ArtifactTransferId(Box<str>);

impl ArtifactTransferId {
    pub fn new(value: impl Into<String>) -> Result<Self, ArtifactTransferError> {
        let value = value.into();
        if value.len() > 128
            || !value.strip_prefix("transfer:").is_some_and(|tail| {
                !tail.is_empty()
                    && tail.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')
                    })
            })
        {
            return Err(ArtifactTransferError::InvalidLabel);
        }
        // Retain only the bounded label, never a caller's spare String capacity.
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ArtifactTransferId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// A full artifact-domain content digest, with an explicit artifact kind prefix.
/// This is an untrusted content claim until the complete bytes have been checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArtifactContentId([u8; 32]);

impl ArtifactContentId {
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Display for ArtifactContentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("artifact:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for ArtifactContentId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ArtifactContentId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ContentIdVisitor;
        impl serde::de::Visitor<'_> for ContentIdVisitor {
            type Value = ArtifactContentId;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("artifact: followed by 64 lowercase hexadecimal digits")
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                let hex = value
                    .strip_prefix("artifact:")
                    .filter(|hex| hex.len() == 64)
                    .ok_or_else(|| E::custom(ArtifactTransferError::InvalidContentId))?;
                let mut digest = [0; 32];
                for (slot, pair) in digest.iter_mut().zip(hex.as_bytes().as_chunks::<2>().0) {
                    *slot = hex_pair(pair)
                        .ok_or_else(|| E::custom(ArtifactTransferError::InvalidContentId))?;
                }
                Ok(ArtifactContentId(digest))
            }
        }
        deserializer.deserialize_str(ContentIdVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    RootCertificate,
    RootIsolation,
    IntegralCertificate,
    RootSearchCheckpoint,
    StationaryCertificate,
    BoxOptimization,
    ParametricFamily,
}

impl ArtifactKind {
    const fn tag(self) -> u8 {
        match self {
            Self::RootCertificate => 1,
            Self::RootIsolation => 2,
            Self::IntegralCertificate => 3,
            Self::RootSearchCheckpoint => 4,
            Self::StationaryCertificate => 5,
            Self::BoxOptimization => 6,
            Self::ParametricFamily => 7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub kind: ArtifactKind,
    pub total_bytes: usize,
    pub content_id: ArtifactContentId,
}

impl ArtifactManifest {
    pub fn chunk_count(&self) -> usize {
        self.total_bytes.div_ceil(ARTIFACT_CHUNK_BYTES)
    }

    fn validate(&self) -> Result<(), ArtifactTransferError> {
        if self.schema_version != ARTIFACT_STREAM_SCHEMA_VERSION {
            return Err(ArtifactTransferError::UnsupportedVersion {
                found: self.schema_version,
                supported: ARTIFACT_STREAM_SCHEMA_VERSION,
            });
        }
        check_length(self.total_bytes)
    }

    fn range(&self, index: usize) -> Result<Range<usize>, ArtifactTransferError> {
        if index >= self.chunk_count() {
            return Err(ArtifactTransferError::InvalidIndex {
                index,
                chunk_count: self.chunk_count(),
            });
        }
        // Only validated manifests enter transfer state, so these are <=1 MiB.
        let start = index * ARTIFACT_CHUNK_BYTES;
        Ok(start..(start + ARTIFACT_CHUNK_BYTES).min(self.total_bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactChunk {
    pub index: usize,
    pub digest: [u8; 32],
    pub data_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArtifactProgress {
    pub received_bytes: usize,
    pub next_chunk: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ArtifactTransferError {
    #[error("artifact transfer stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("invalid bounded transfer: label")]
    InvalidLabel,
    #[error("artifact content ID must contain 64 lowercase hexadecimal digits")]
    InvalidContentId,
    #[error("unsupported artifact stream schema {found}; expected {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("artifact size {actual} exceeds maximum {max}")]
    ByteLimit { actual: usize, max: usize },
    #[error("an artifact cannot be empty")]
    EmptyArtifact,
    #[error("chunk index {index} is outside the {chunk_count}-chunk artifact")]
    InvalidIndex { index: usize, chunk_count: usize },
    #[error("expected chunk {expected}, received future chunk {found}")]
    OutOfOrder { expected: usize, found: usize },
    #[error("hex payload length {actual} differs from expected {expected}")]
    ChunkLength { expected: usize, actual: usize },
    #[error("chunk payload must use canonical lowercase hexadecimal digits")]
    InvalidHex,
    #[error("chunk digest does not match its bytes")]
    ChunkDigestMismatch,
    #[error("complete artifact digest does not match its content ID")]
    FinalDigestMismatch,
    #[error("repeated chunk differs from the already received bytes")]
    ConflictingChunk,
    #[error("artifact is incomplete: received {received} of {expected} bytes")]
    Incomplete { received: usize, expected: usize },
    #[error("artifact kind differs from the requested replay operation")]
    KindMismatch,
    #[error("artifact transfer allocation refused")]
    AllocationFailure,
}

fn check_length(length: usize) -> Result<(), ArtifactTransferError> {
    if length == 0 {
        return Err(ArtifactTransferError::EmptyArtifact);
    }
    if length > MAX_ARTIFACT_BYTES {
        return Err(ArtifactTransferError::ByteLimit {
            actual: length,
            max: MAX_ARTIFACT_BYTES,
        });
    }
    Ok(())
}

fn charge(
    meter: &mut impl BudgetMeter,
    steps: usize,
    bytes: usize,
    allocations: u64,
) -> Result<(), ArtifactTransferError> {
    meter.checkpoint()?;
    let charges = [
        (Dimension::ComputeSteps, steps as u64),
        (Dimension::MemoryBytes, bytes as u64),
        (Dimension::AllocationCount, allocations),
    ];
    let mut nonzero = [(Dimension::ComputeSteps, 1); 3];
    let mut length = 0;
    for entry in charges {
        if entry.1 != 0 {
            nonzero[length] = entry;
            length += 1;
        }
    }
    meter.charge_batch(&nonzero[..length])?;
    Ok(())
}

fn digest(bytes: &[u8], meter: &mut impl BudgetMeter) -> Result<[u8; 32], ArtifactTransferError> {
    // Every call is bounded by 1 MiB (16 KiB for a chunk). The reused hasher
    // allocates no payload buffer; safe points bracket this synchronous unit.
    charge(meter, bytes.len() + 64, 0, 0)?;
    let digest = object_digest(bytes);
    meter.checkpoint()?;
    Ok(digest)
}

/// Derive the content ID from a bounded byte payload and its artifact kind.
/// With `H = capsule::object_digest`, the fixed stack preimage is
/// `"fsym.artifact-content.v1\0" || kind_tag || len_u64_le || H(bytes)`.
/// Stable kind tags distinguish root certificates (1), complete root isolation
/// (2), integrals (3), candidate search checkpoints (4), and stationary points (5).
pub fn artifact_content_id(
    kind: ArtifactKind,
    bytes: &[u8],
    meter: &mut impl BudgetMeter,
) -> Result<ArtifactContentId, ArtifactTransferError> {
    meter.checkpoint()?;
    check_length(bytes.len())?;
    const DOMAIN: &[u8] = b"fsym.artifact-content.v1\0";
    let payload = digest(bytes, meter)?;
    let mut frame = [0; DOMAIN.len() + 1 + 8 + 32];
    frame[..DOMAIN.len()].copy_from_slice(DOMAIN);
    frame[DOMAIN.len()] = kind.tag();
    frame[DOMAIN.len() + 1..DOMAIN.len() + 9].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    frame[DOMAIN.len() + 9..].copy_from_slice(&payload);
    Ok(ArtifactContentId::from_digest(digest(&frame, meter)?))
}

/// Derive an indexed checksum bound to the complete artifact's content ID.
/// The fixed stack preimage is `"fsym.artifact-chunk.v1\0" || content_digest ||
/// index_u64_le || len_u64_le || H(bytes)`, using the same `H` as content IDs.
pub fn artifact_chunk_digest(
    content_id: ArtifactContentId,
    index: usize,
    bytes: &[u8],
    meter: &mut impl BudgetMeter,
) -> Result<[u8; 32], ArtifactTransferError> {
    meter.checkpoint()?;
    let chunk_count = MAX_ARTIFACT_BYTES.div_ceil(ARTIFACT_CHUNK_BYTES);
    if index >= chunk_count {
        return Err(ArtifactTransferError::InvalidIndex { index, chunk_count });
    }
    if bytes.len() > ARTIFACT_CHUNK_BYTES {
        return Err(ArtifactTransferError::ByteLimit {
            actual: bytes.len(),
            max: ARTIFACT_CHUNK_BYTES,
        });
    }
    const DOMAIN: &[u8] = b"fsym.artifact-chunk.v1\0";
    let payload = digest(bytes, meter)?;
    let mut frame = [0; DOMAIN.len() + 32 + 8 + 8 + 32];
    frame[..DOMAIN.len()].copy_from_slice(DOMAIN);
    frame[DOMAIN.len()..DOMAIN.len() + 32].copy_from_slice(&content_id.digest());
    frame[DOMAIN.len() + 32..DOMAIN.len() + 40].copy_from_slice(&(index as u64).to_le_bytes());
    frame[DOMAIN.len() + 40..DOMAIN.len() + 48]
        .copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    frame[DOMAIN.len() + 48..].copy_from_slice(&payload);
    digest(&frame, meter)
}

fn hex_pair(pair: &[u8]) -> Option<u8> {
    fn nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
    Some((nibble(pair[0])? << 4) | nibble(pair[1])?)
}

fn encode_hex(bytes: &[u8], meter: &mut impl BudgetMeter) -> Result<String, ArtifactTransferError> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    charge(meter, bytes.len() * 2 + 1, bytes.len() * 2, 1)?;
    let mut encoded = String::new();
    encoded
        .try_reserve_exact(bytes.len() * 2)
        .map_err(|_| ArtifactTransferError::AllocationFailure)?;
    for block in bytes.chunks(512) {
        meter.checkpoint()?;
        for &byte in block {
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 15) as usize] as char);
        }
    }
    meter.checkpoint()?;
    Ok(encoded)
}

fn decode_hex(hex: &str, meter: &mut impl BudgetMeter) -> Result<Vec<u8>, ArtifactTransferError> {
    // Callers checked the exact expected even length against a bounded chunk.
    charge(meter, hex.len() + 1, hex.len() / 2, 1)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(hex.len() / 2)
        .map_err(|_| ArtifactTransferError::AllocationFailure)?;
    for block in hex.as_bytes().chunks(1024) {
        meter.checkpoint()?;
        for pair in block.as_chunks::<2>().0 {
            bytes.push(hex_pair(pair).ok_or(ArtifactTransferError::InvalidHex)?);
        }
    }
    meter.checkpoint()?;
    Ok(bytes)
}

/// Immutable bounded bytes with indexed, repeatable reads. This is not a receipt.
pub struct ArtifactExport {
    manifest: ArtifactManifest,
    bytes: Vec<u8>,
}

impl ArtifactExport {
    pub fn new(
        kind: ArtifactKind,
        bytes: Vec<u8>,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, ArtifactTransferError> {
        meter.checkpoint()?;
        check_length(bytes.len())?;
        if bytes.capacity() > MAX_ARTIFACT_BYTES {
            return Err(ArtifactTransferError::ByteLimit {
                actual: bytes.capacity(),
                max: MAX_ARTIFACT_BYTES,
            });
        }
        // Admit retention on this caller's ledger even if the bytes were
        // constructed under another meter. No payload clone is performed.
        charge(meter, 1, bytes.capacity(), 0)?;
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_STREAM_SCHEMA_VERSION,
            kind,
            total_bytes: bytes.len(),
            content_id: artifact_content_id(kind, &bytes, meter)?,
        };
        Ok(Self { manifest, bytes })
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    pub fn chunk(
        &self,
        index: usize,
        meter: &mut impl BudgetMeter,
    ) -> Result<ArtifactChunk, ArtifactTransferError> {
        meter.checkpoint()?;
        let bytes = &self.bytes[self.manifest.range(index)?];
        Ok(ArtifactChunk {
            index,
            digest: artifact_chunk_digest(self.manifest.content_id, index, bytes, meter)?,
            data_hex: encode_hex(bytes, meter)?,
        })
    }
}

/// An ordered staging prefix. It exposes complete bytes only after an integrity
/// check, and never grants mathematical acceptance or consumes a verifier lease.
pub struct ArtifactImport {
    manifest: ArtifactManifest,
    bytes: Vec<u8>,
}

impl ArtifactImport {
    pub fn new(
        manifest: ArtifactManifest,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, ArtifactTransferError> {
        meter.checkpoint()?;
        manifest.validate()?;
        charge(meter, 1, manifest.total_bytes, 1)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(manifest.total_bytes)
            .map_err(|_| ArtifactTransferError::AllocationFailure)?;
        meter.checkpoint()?;
        Ok(Self { manifest, bytes })
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    pub fn progress(&self) -> ArtifactProgress {
        ArtifactProgress {
            received_bytes: self.bytes.len(),
            next_chunk: self.bytes.len().div_ceil(ARTIFACT_CHUNK_BYTES),
        }
    }

    /// Predict the bounded acknowledgment before a Session admits its response.
    /// Payload validation still occurs in `append`; this does not mutate state.
    pub fn preview_append(&self, index: usize) -> Result<ArtifactProgress, ArtifactTransferError> {
        let range = self.manifest.range(index)?;
        let progress = self.progress();
        if index > progress.next_chunk {
            return Err(ArtifactTransferError::OutOfOrder {
                expected: progress.next_chunk,
                found: index,
            });
        }
        if index < progress.next_chunk {
            return Ok(progress);
        }
        Ok(ArtifactProgress {
            received_bytes: range.end,
            next_chunk: index + 1,
        })
    }

    /// Exact old-index retries are charged and return current progress.
    /// Every refusal, including cancellation, leaves the staging prefix intact.
    pub fn append(
        &mut self,
        index: usize,
        expected_digest: &[u8; 32],
        data_hex: &str,
        meter: &mut impl BudgetMeter,
    ) -> Result<ArtifactProgress, ArtifactTransferError> {
        meter.checkpoint()?;
        let next = self.preview_append(index)?;
        let range = self.manifest.range(index)?;
        if data_hex.len() != range.len() * 2 {
            return Err(ArtifactTransferError::ChunkLength {
                expected: range.len() * 2,
                actual: data_hex.len(),
            });
        }
        let bytes = decode_hex(data_hex, meter)?;
        if artifact_chunk_digest(self.manifest.content_id, index, &bytes, meter)?
            != *expected_digest
        {
            return Err(ArtifactTransferError::ChunkDigestMismatch);
        }
        charge(meter, bytes.len() + 1, 0, 0)?;
        if index < self.progress().next_chunk {
            if self.bytes[range] != bytes {
                return Err(ArtifactTransferError::ConflictingChunk);
            }
            meter.checkpoint()?;
            return Ok(next);
        }
        // The full buffer was reserved at begin. No allocation, fallible work,
        // or cancelling safe point follows this final bounded publication.
        self.bytes.extend_from_slice(&bytes);
        Ok(next)
    }

    /// Validate the entire content ID on each replay attempt. Returned bytes
    /// remain untrusted input to the caller's canonical mathematical verifier.
    pub fn checked_bytes(
        &self,
        expected_kind: ArtifactKind,
        meter: &mut impl BudgetMeter,
    ) -> Result<&[u8], ArtifactTransferError> {
        meter.checkpoint()?;
        if self.manifest.kind != expected_kind {
            return Err(ArtifactTransferError::KindMismatch);
        }
        if self.bytes.len() != self.manifest.total_bytes {
            return Err(ArtifactTransferError::Incomplete {
                received: self.bytes.len(),
                expected: self.manifest.total_bytes,
            });
        }
        if artifact_content_id(self.manifest.kind, &self.bytes, meter)? != self.manifest.content_id
        {
            return Err(ArtifactTransferError::FinalDigestMismatch);
        }
        Ok(&self.bytes)
    }
}
