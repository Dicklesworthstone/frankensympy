//! The storage-neutral two-phase durable boundary.
//!
//! Every lane implements the same explicit publication sequence:
//! [`DurableStore::prepare`] writes to a staging namespace,
//! [`DurableStore::verify_prepared`] re-reads the staged bytes and runs full
//! boundary validation, and [`DurableStore::commit`] publishes atomically or
//! refuses. Crash testing drives real process death at each boundary; nothing
//! is implicitly published and nothing staged is trusted.

use crate::{DurableError, DurableRecord, MAX_RECORD_WIRE_BYTES, hex_decode};
use std::fmt;

/// Opaque staging handle minted by [`DurableStore::prepare`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedHandle {
    pub(crate) name: String,
}

impl PreparedHandle {
    pub fn as_str(&self) -> &str {
        &self.name
    }

    /// Reconstructs a handle previously surfaced by this crate (marker files,
    /// receipts). Handles are content-addressed names, never capabilities.
    pub fn from_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Check the content-addressed basename before it is used by a backend.
    /// Reconstructing a handle does not grant path or SQL authority.
    pub fn validate(&self) -> Result<(), DurableError> {
        self.parts().map(|_| ())
    }

    pub(crate) fn parts(&self) -> Result<([u8; 32], usize), DurableError> {
        let invalid = || DurableError::MalformedRecord("invalid prepared record handle".into());
        if self.name.len() > 92 {
            return Err(invalid());
        }
        let (digest, length) = self.name.split_once('-').ok_or_else(invalid)?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid());
        }
        let length = length.strip_suffix(".record").ok_or_else(invalid)?;
        if length.is_empty()
            || !length.bytes().all(|byte| byte.is_ascii_digit())
            || length.starts_with('0')
        {
            return Err(invalid());
        }
        let payload_len = length.parse::<usize>().map_err(|_| invalid())?;
        if payload_len > MAX_RECORD_WIRE_BYTES {
            return Err(invalid());
        }
        let digest = hex_decode(digest)?.try_into().map_err(|_| invalid())?;
        Ok((digest, payload_len))
    }
}

impl fmt::Display for PreparedHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

pub trait DurableStore {
    /// Bounds and stages one record. Refuses when the staging cleanup reserve
    /// is exhausted (`InsufficientCleanupReserve`) so abandoned staging can
    /// never grow without bound.
    fn prepare(&self, record: &DurableRecord) -> Result<PreparedHandle, DurableError>;

    /// Re-reads the staged bytes from the backend and runs full boundary
    /// validation (digests, schema, universe, dependency pins) before the
    /// record may be committed.
    fn verify_prepared(&self, handle: &PreparedHandle) -> Result<DurableRecord, DurableError>;

    /// Atomically publishes a verified staged record. Non-atomic or partial
    /// publication must be impossible in the implementing lane.
    fn commit(&self, handle: &PreparedHandle) -> Result<(), DurableError>;

    /// Loads the committed record for one universe and payload schema, if any.
    fn load_committed(
        &self,
        universe_id: [u8; 32],
        payload_schema: &str,
    ) -> Result<Option<DurableRecord>, DurableError>;

    /// Discards a staged record without publishing.
    fn release(&self, handle: &PreparedHandle) -> Result<(), DurableError>;

    /// Number of records currently held in the staging namespace.
    fn staging_len(&self) -> Result<usize, DurableError>;
}
