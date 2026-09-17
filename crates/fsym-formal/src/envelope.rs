use crate::{
    MAX_CAPSULE_BYTES, MAX_JSON_BYTES, MAX_SOURCE_BYTES, ProjectionError as E, ProjectionReceipt,
};
use serde::{Deserialize, Serialize};

/// Transport data only: deserialization grants no checked status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionEnvelope {
    pub profile_id: String,
    pub checker_pin: String,
    pub environment: String,
    pub capsule_hex: String,
    pub native_claim_root: String,
    pub statement_root: String,
    pub lean_source: String,
    pub receipt: ProjectionReceipt,
    pub native_verdict: String,
}
impl ProjectionEnvelope {
    /// Bounded strict transport decoding. Independently check before use.
    pub fn from_json(bytes: &[u8]) -> Result<Self, E> {
        if bytes.len() > MAX_JSON_BYTES {
            return Err(E::ResourceExhausted("JSON bytes"));
        }
        let result: Self = serde_json::from_slice(bytes).map_err(|_| E::MalformedInput)?;
        result.check_sizes()?;
        Ok(result)
    }
    pub(crate) fn check_sizes(&self) -> Result<(), E> {
        if self.capsule_hex.len() > MAX_CAPSULE_BYTES * 2 {
            return Err(E::ResourceExhausted("capsule bytes"));
        }
        if self.lean_source.len() > MAX_SOURCE_BYTES {
            return Err(E::ResourceExhausted("source bytes"));
        }
        Ok(())
    }
}

/// Minted only by the independent native-to-statement checker. This is not
/// foreign-kernel evidence, and cannot be deserialized or constructed publicly.
#[derive(Debug, Clone)]
pub struct CheckedProjection {
    envelope: ProjectionEnvelope,
}
impl CheckedProjection {
    pub(crate) fn new(envelope: ProjectionEnvelope) -> Self {
        Self { envelope }
    }
    pub fn envelope(&self) -> &ProjectionEnvelope {
        &self.envelope
    }
    pub fn into_envelope(self) -> ProjectionEnvelope {
        self.envelope
    }
}

pub fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 15) as usize] as char);
    }
    out
}
/// Strict lower-case hex; an explicit byte cap is applied before allocation.
pub fn decode_hex(text: &str, max_bytes: usize) -> Result<Vec<u8>, E> {
    if text.len() / 2 > max_bytes {
        return Err(E::ResourceExhausted("hex bytes"));
    }
    if !text.len().is_multiple_of(2) {
        return Err(E::MalformedInput);
    }
    fn nibble(b: u8) -> Result<u8, E> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            _ => Err(E::MalformedInput),
        }
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(nibble(pair[0])? * 16 + nibble(pair[1])?))
        .collect()
}
