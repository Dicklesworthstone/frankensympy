//! Typed, non-interchangeable identifiers and canonical digest preimage
//! domains for FrankenSymPy.
//!
//! Layer: L0 (typed IDs). This crate contains no term logic and no
//! persistence, per `docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md` §4.1.
//!
//! Invariants enforced here:
//!
//! - IDs of different kinds are distinct types; a parsed payload of one kind
//!   can never be reinterpreted as another kind.
//! - Identifier `0` is reserved and never issued (fail-closed sentinel).
//! - Parsing accepts only the canonical `Display` spelling and rejects unknown
//!   kind prefixes instead of guessing.
//! - Digest preimages are framed with an explicit domain tag so that bytes
//!   hashed under one identity domain can never collide with bytes framed
//!   under another domain.
//!
//! Stable identities exclude scheduling, wall time, memory addresses, and
//! cache state. Callers derive the numeric payloads from content upstream;
//! this crate only types, validates, formats, and frames them.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

const MAX_DIAGNOSTIC_BYTES: usize = 128;
const MAX_U64_DECIMAL_DIGITS: usize = 20;

fn bounded_text(text: &str) -> String {
    if text.len() <= MAX_DIAGNOSTIC_BYTES {
        return text.to_owned();
    }
    let mut summary = String::with_capacity(MAX_DIAGNOSTIC_BYTES + 16);
    for character in text.chars() {
        if summary.len() + character.len_utf8() > MAX_DIAGNOSTIC_BYTES {
            break;
        }
        summary.push(character);
    }
    summary.push_str("…<truncated>");
    summary
}

fn bounded_binary_kind(bytes: &[u8]) -> String {
    let visible_len = bytes.len().min(MAX_DIAGNOSTIC_BYTES);
    let mut visible = String::with_capacity(visible_len + 16);
    for byte in &bytes[..visible_len] {
        if byte.is_ascii_graphic() {
            visible.push(*byte as char);
        } else {
            visible.push('.');
        }
    }
    if bytes.len() > visible_len {
        visible.push_str("…<truncated>");
    }
    visible
}

/// Errors arising from identifier construction, decoding, or cross-kind misuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdError {
    /// Payload value `0` is reserved and may not be used for a live entity.
    ReservedZero,

    /// Identifier text did not start with the expected kind prefix followed by
    /// `-` and a decimal integer payload.
    UnknownKind { found: String },

    /// The kind prefix matched, but the following payload was not valid decimal
    /// or exceeded [`u64::MAX`].
    InvalidPayload { found: String },

    /// Binary frame had an unrecognized kind prefix, wrong length prefix, or
    /// was truncated.
    MalformedPayload { found: String },
}

impl fmt::Display for IdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReservedZero => write!(f, "identifier payload 0 is reserved as a sentinel"),
            Self::UnknownKind { found } => {
                write!(f, "unknown identifier kind in \"{}\"", bounded_text(found))
            }
            Self::InvalidPayload { found } => {
                write!(
                    f,
                    "invalid identifier payload in \"{}\"",
                    bounded_text(found)
                )
            }
            Self::MalformedPayload { found } => {
                write!(
                    f,
                    "malformed binary identifier frame: {}",
                    bounded_text(found)
                )
            }
        }
    }
}

impl std::error::Error for IdError {}

/// Defines a newtype identifier with a fixed textual kind prefix.
///
/// The generated type implements `Display`, `FromStr`, and `preimage_domain`,
/// and refuses to construct from the reserved payload `0`.
macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name(u64);

        impl $name {
            /// Fixed textual kind prefix used in parsing, formatting, and as
            /// the digest preimage domain separator.
            pub const KIND: &'static str = $prefix;

            /// Constructs an identifier from a raw payload, rejecting the
            /// reserved value `0`.
            pub fn new(payload: u64) -> Result<Self, IdError> {
                if payload == 0 {
                    Err(IdError::ReservedZero)
                } else {
                    Ok(Self(payload))
                }
            }

            /// Returns the raw payload. Callers must not reinterpret a
            /// payload across kinds.
            pub fn raw(self) -> u64 {
                self.0
            }

            /// Canonical digest preimage domain for this identifier kind.
            pub fn preimage_domain(self) -> &'static str {
                Self::KIND
            }

            /// Canonical self-describing binary form: little-endian
            /// length-prefixed kind tag, then the little-endian payload.
            /// Identical across processes and architectures.
            pub fn to_binary(self) -> Vec<u8> {
                let mut out = Vec::with_capacity(Self::KIND.len() + 17);
                out.extend_from_slice(&(Self::KIND.len() as u64).to_le_bytes());
                out.extend_from_slice(Self::KIND.as_bytes());
                out.extend_from_slice(&self.0.to_le_bytes());
                out
            }

            /// Decodes the canonical binary form, failing closed on any
            /// other kind, truncated layout, or reserved payload.
            pub fn from_binary(bytes: &[u8]) -> Result<Self, IdError> {
                let malformed = |len: usize| IdError::MalformedPayload {
                    found: format!("<binary:{} bytes>", len),
                };
                let Some(tag_len_bytes) = bytes.get(..8) else {
                    return Err(malformed(bytes.len()));
                };
                let Ok(tag_len_arr) = <[u8; 8]>::try_from(tag_len_bytes) else {
                    return Err(malformed(bytes.len()));
                };
                let Ok(kind_len) = usize::try_from(u64::from_le_bytes(tag_len_arr)) else {
                    return Err(malformed(bytes.len()));
                };
                let Some(rest) = bytes.get(8..) else {
                    return Err(malformed(bytes.len()));
                };
                let Some(expected_rest_len) = kind_len.checked_add(8) else {
                    return Err(malformed(bytes.len()));
                };
                if rest.len() != expected_rest_len {
                    return Err(malformed(bytes.len()));
                }
                let (kind, payload) = rest.split_at(kind_len);
                if kind != Self::KIND.as_bytes() {
                    return Err(IdError::UnknownKind {
                        found: bounded_binary_kind(kind),
                    });
                }
                let Ok(payload_arr) = <[u8; 8]>::try_from(payload) else {
                    return Err(malformed(bytes.len()));
                };
                Self::new(u64::from_le_bytes(payload_arr))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}-{}", Self::KIND, self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = IdError;

            fn from_str(text: &str) -> Result<Self, Self::Err> {
                let max_text_len = Self::KIND
                    .len()
                    .checked_add(1 + MAX_U64_DECIMAL_DIGITS)
                    .ok_or_else(|| IdError::MalformedPayload {
                        found: bounded_text(text),
                    })?;
                if text.len() > max_text_len {
                    return Err(IdError::MalformedPayload {
                        found: bounded_text(text),
                    });
                }
                let Some(rest) = text.strip_prefix(Self::KIND) else {
                    return Err(IdError::UnknownKind {
                        found: bounded_text(text),
                    });
                };
                let Some(payload) = rest.strip_prefix('-') else {
                    return Err(IdError::UnknownKind {
                        found: bounded_text(text),
                    });
                };
                if payload.is_empty()
                    || !payload.bytes().all(|byte| byte.is_ascii_digit())
                    || (payload.len() > 1 && payload.starts_with('0'))
                {
                    return Err(IdError::MalformedPayload {
                        found: bounded_text(text),
                    });
                }
                let raw: u64 = payload
                    .parse()
                    .map_err(|_| IdError::MalformedPayload {
                        found: bounded_text(text),
                    })?;
                Self::new(raw)
            }
        }
    };
}

define_id!(
    /// Identity of a node in the Python-facing surface object graph.
    SurfaceId,
    "surface"
);
define_id!(
    /// Identity of a term in the native semantic DAG. Distinct from any
    /// surface handle, arena slot, or graph vertex.
    TermId,
    "term"
);
define_id!(
    /// Identity of an exact mathematical domain (e.g. a polynomial ring).
    DomainId,
    "domain"
);
define_id!(
    /// Identity of an immutable assumptions context.
    ContextId,
    "context"
);
define_id!(
    /// Identity of a registered deterministic rewrite rule.
    RuleId,
    "rule"
);
define_id!(
    /// Identity of a registered candidate-generating algorithm.
    AlgorithmId,
    "algorithm"
);
define_id!(
    /// Identity of a registered independent verifier.
    VerifierId,
    "verifier"
);
define_id!(
    /// Identity of a typed claim in the claim lattice.
    ClaimId,
    "claim"
);
define_id!(
    /// Identity of one derivation recorded in the evidence graph.
    DerivationId,
    "derivation"
);
define_id!(
    /// Identity of a verification or execution receipt.
    ReceiptId,
    "receipt"
);
define_id!(
    /// Identity of a published checkpoint.
    CheckpointId,
    "checkpoint"
);
define_id!(
    /// Identity of a replay/repair bundle.
    BundleId,
    "bundle"
);
define_id!(
    /// Identity of an agent-native semantic workspace.
    WorkspaceId,
    "workspace"
);
define_id!(
    /// Identity of a workspace branch forked for speculative work.
    BranchId,
    "branch"
);

/// Default allocation bound for an encoded digest preimage.
pub const MAX_PREIMAGE_BYTES: usize = 1024 * 1024;

/// Errors produced while framing a canonical digest preimage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreimageError {
    /// A component or aggregate encoded length overflowed its representation.
    LengthOverflow,
    /// The encoded preimage exceeded the caller's allocation limit.
    SizeLimitExceeded { limit: usize },
    /// Reserving the admitted output allocation failed.
    AllocationFailure,
}

impl fmt::Display for PreimageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthOverflow => write!(f, "canonical preimage length overflow"),
            Self::SizeLimitExceeded { limit } => {
                write!(f, "canonical preimage exceeds byte limit {limit}")
            }
            Self::AllocationFailure => write!(f, "canonical preimage allocation failed"),
        }
    }
}

impl std::error::Error for PreimageError {}

/// Frames byte slices into a canonical digest preimage under an explicit
/// domain tag.
///
/// Framing is length-prefixed at every level, so no concatenation ambiguity
/// exists: two different `(domain, parts)` inputs always frame to different
/// bytes whenever they differ in any component boundary. Hashing happens in
/// higher layers; this function only produces the canonical input bytes.
/// The complete encoded length is checked before one bounded allocation.
pub fn frame_preimage(domain: &str, parts: &[&[u8]]) -> Result<Vec<u8>, PreimageError> {
    frame_preimage_with_limit(domain, parts, MAX_PREIMAGE_BYTES)
}

/// Frames a canonical digest preimage under a caller-provided byte limit.
pub fn frame_preimage_with_limit(
    domain: &str,
    parts: &[&[u8]],
    max_bytes: usize,
) -> Result<Vec<u8>, PreimageError> {
    u64::try_from(parts.len()).map_err(|_| PreimageError::LengthOverflow)?;
    let mut encoded_len = framed_chunk_len(domain.len())?
        .checked_add(framed_chunk_len(8)?)
        .ok_or(PreimageError::LengthOverflow)?;
    if encoded_len > max_bytes {
        return Err(PreimageError::SizeLimitExceeded { limit: max_bytes });
    }
    for part in parts {
        encoded_len = encoded_len
            .checked_add(framed_chunk_len(part.len())?)
            .ok_or(PreimageError::LengthOverflow)?;
        if encoded_len > max_bytes {
            return Err(PreimageError::SizeLimitExceeded { limit: max_bytes });
        }
    }

    let mut out = Vec::new();
    out.try_reserve_exact(encoded_len)
        .map_err(|_| PreimageError::AllocationFailure)?;
    push_chunk(&mut out, domain.as_bytes())?;
    push_chunk(
        &mut out,
        &u64::try_from(parts.len())
            .map_err(|_| PreimageError::LengthOverflow)?
            .to_le_bytes(),
    )?;
    for part in parts {
        push_chunk(&mut out, part)?;
    }
    Ok(out)
}

fn framed_chunk_len(payload_len: usize) -> Result<usize, PreimageError> {
    u64::try_from(payload_len).map_err(|_| PreimageError::LengthOverflow)?;
    payload_len
        .checked_add(8)
        .ok_or(PreimageError::LengthOverflow)
}

fn push_chunk(out: &mut Vec<u8>, chunk: &[u8]) -> Result<(), PreimageError> {
    let len = u64::try_from(chunk.len()).map_err(|_| PreimageError::LengthOverflow)?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(chunk);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_display_and_parse() {
        let id = TermId::new(42).expect("nonzero");
        assert_eq!(id.to_string(), "term-42");
        let parsed: TermId = "term-42".parse().expect("valid");
        assert_eq!(parsed, id);
        assert_eq!(parsed.raw(), 42);
    }

    #[test]
    fn zero_payload_is_reserved() {
        assert_eq!(TermId::new(0), Err(IdError::ReservedZero));
        assert_eq!("term-0".parse::<TermId>(), Err(IdError::ReservedZero));
    }

    #[test]
    fn unknown_prefix_fails_closed() {
        // A ContextId string must never parse into a TermId.
        assert_eq!(
            "context-7".parse::<TermId>(),
            Err(IdError::UnknownKind {
                found: "context-7".to_string()
            })
        );
        // Prefixes sharing a leading substring are still rejected.
        assert_eq!(
            "terms-7".parse::<TermId>(),
            Err(IdError::UnknownKind {
                found: "terms-7".to_string()
            })
        );
        // Garbage never parses.
        assert!("".parse::<TermId>().is_err());
        assert!("term-".parse::<TermId>().is_err());
        assert!("term-x".parse::<TermId>().is_err());
    }

    #[test]
    fn negative_and_overflowing_payloads_are_malformed_or_reserved() {
        assert!("term--1".parse::<TermId>().is_err());
        assert!("term-99999999999999999999999".parse::<TermId>().is_err());
        assert!("term-01".parse::<TermId>().is_err());
        assert!("term-+1".parse::<TermId>().is_err());
    }

    #[test]
    fn kinds_have_unique_domains() {
        let kinds = [
            SurfaceId::KIND,
            TermId::KIND,
            DomainId::KIND,
            ContextId::KIND,
            RuleId::KIND,
            AlgorithmId::KIND,
            VerifierId::KIND,
            ClaimId::KIND,
            DerivationId::KIND,
            ReceiptId::KIND,
            CheckpointId::KIND,
            BundleId::KIND,
            WorkspaceId::KIND,
            BranchId::KIND,
        ];
        for (i, kind) in kinds.iter().enumerate() {
            for other in kinds.iter().skip(i + 1) {
                assert_ne!(kind, other, "kinds must be unique");
            }
        }
    }

    #[test]
    fn ordering_is_by_payload_within_a_kind() {
        let a = TermId::new(1).unwrap();
        let b = TermId::new(2).unwrap();
        assert!(a < b);
    }

    #[test]
    fn framing_is_unambiguous_across_boundaries() {
        let ab = frame_preimage("d", &[b"a", b"bc"]).unwrap();
        let abc = frame_preimage("d", &[b"ab", b"c"]).unwrap();
        // Same concatenated bytes but different component boundaries must
        // frame differently.
        assert_ne!(ab, abc);
    }

    #[test]
    fn framing_separates_domains() {
        let parts: [&[u8]; 1] = [b"payload"];
        let a = frame_preimage("term", &parts).unwrap();
        let b = frame_preimage("surface", &parts).unwrap();
        assert_ne!(a, b, "distinct identity domains must never share preimages");
    }

    #[test]
    fn framing_is_deterministic_and_length_prefixed() {
        let parts: [&[u8]; 2] = [b"x", b"yz"];
        let f1 = frame_preimage("term", &parts).unwrap();
        let f2 = frame_preimage("term", &parts).unwrap();
        assert_eq!(f1, f2);
        // Head is the length-prefixed domain tag itself.
        assert_eq!(&f1[..8], &4u64.to_le_bytes());
        assert_eq!(&f1[8..12], b"term");
        // Next comes the part count, itself framed as a chunk: an 8-byte
        // length prefix followed by the little-endian count.
        assert_eq!(&f1[12..20], &8u64.to_le_bytes());
        assert_eq!(&f1[20..28], &2u64.to_le_bytes());
    }

    #[test]
    fn empty_parts_frame_is_still_well_formed() {
        let f = frame_preimage("claim", &[]).unwrap();
        assert_eq!(&f[..8], &5u64.to_le_bytes());
        assert_eq!(&f[8..13], b"claim");
        // Zero parts still carry a framed (length-prefixed) count.
        assert_eq!(&f[13..21], &8u64.to_le_bytes());
        assert_eq!(&f[21..29], &0u64.to_le_bytes());
        assert_eq!(f.len(), 29);
    }

    #[test]
    fn framing_preflights_the_exact_encoded_size() {
        let parts: [&[u8]; 1] = [b"payload"];
        let encoded = frame_preimage("term", &parts).unwrap();
        assert_eq!(
            frame_preimage_with_limit("term", &parts, encoded.len()).unwrap(),
            encoded
        );
        assert_eq!(
            frame_preimage_with_limit("term", &parts, encoded.len() - 1),
            Err(PreimageError::SizeLimitExceeded {
                limit: encoded.len() - 1
            })
        );
    }

    #[test]
    fn binary_form_roundtrips_and_is_self_describing() {
        let id = TermId::new(0xDEAD_BEEF).unwrap();
        let bytes = id.to_binary();
        // Kind tag is framed, so the blob names its own kind.
        assert_eq!(&bytes[..8], &4u64.to_le_bytes());
        assert_eq!(&bytes[8..12], b"term");
        assert_eq!(bytes.len(), 12 + 8);
        let decoded = TermId::from_binary(&bytes).unwrap();
        assert_eq!(decoded, id);
    }

    #[test]
    fn binary_form_rejects_foreign_kind_and_truncation() {
        let bytes = ContextId::new(9).unwrap().to_binary();
        let err = TermId::from_binary(&bytes).unwrap_err();
        assert_eq!(
            err,
            IdError::UnknownKind {
                found: "context".to_string()
            }
        );
        // Every truncation must fail closed, not decode partially.
        let full = TermId::new(7).unwrap().to_binary();
        for cut in 0..full.len() {
            assert!(TermId::from_binary(&full[..cut]).is_err());
        }
        // Trailing garbage is a malformed layout, not extra data.
        let mut padded = full.clone();
        padded.push(0);
        assert!(TermId::from_binary(&padded).is_err());
    }

    #[test]
    fn hostile_binary_lengths_and_diagnostics_are_bounded() {
        let hostile_length = [u8::MAX; 16];
        assert!(TermId::from_binary(&hostile_length).is_err());

        let kind_len = 4096_u64;
        let kind_len_usize = usize::try_from(kind_len).unwrap();
        let mut oversized_kind = Vec::new();
        oversized_kind.extend_from_slice(&kind_len.to_le_bytes());
        oversized_kind.extend(std::iter::repeat_n(b'x', kind_len_usize));
        oversized_kind.extend_from_slice(&7_u64.to_le_bytes());
        let error = TermId::from_binary(&oversized_kind).unwrap_err();
        assert!(matches!(&error, IdError::UnknownKind { .. }));
        if let IdError::UnknownKind { found } = error {
            assert!(found.len() <= MAX_DIAGNOSTIC_BYTES + "…<truncated>".len());
        }
        oversized_kind
            .get_mut(8..8 + kind_len_usize)
            .unwrap()
            .fill(u8::MAX);
        let error = TermId::from_binary(&oversized_kind).unwrap_err();
        assert!(matches!(&error, IdError::UnknownKind { .. }));
        if let IdError::UnknownKind { found } = error {
            assert!(found.len() <= MAX_DIAGNOSTIC_BYTES + "…<truncated>".len());
        }

        let oversized_text = "x".repeat(4096);
        let error = oversized_text.parse::<TermId>().unwrap_err();
        assert!(matches!(&error, IdError::MalformedPayload { .. }));
        if let IdError::MalformedPayload { found } = error {
            assert!(found.len() <= MAX_DIAGNOSTIC_BYTES + "…<truncated>".len());
        }
    }
}
