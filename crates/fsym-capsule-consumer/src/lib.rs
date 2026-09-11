//! # fsym-capsule-consumer
//!
//! Minimal offline consumer for verifier-complete FrankenSymPy capsules
//! (`docs/PORTABLE_CLAIM_CERTIFICATE_AND_ARTIFACT_PROTOCOL.md` §7.1).
//!
//! This crate exists to demonstrate the public embeddability promise: checking a
//! capsule needs the capsule bytes and nothing else. Its dependency closure is
//! deliberately tiny — `fsym-proof-kernel` and whatever that crate itself
//! needs — and contains no generator, planner, runtime, Python, persistence, or
//! network crate. There is no resolver hook, no ambient time, no entropy, and
//! no fallback: a missing object or a malformed length is a refusal.
//!
//! Layer: L2 (portable verifier consumer).
//!
//! ```no_run
//! # let capsule_bytes: Vec<u8> = Vec::new();
//! let report = fsym_capsule_consumer::verify_offline(&capsule_bytes, None, 1_000);
//! if report.is_verified() {
//!     // accept the claim
//! } else {
//!     // refused, refuted, or inconclusive: never an acceptance
//! }
//! ```

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use fsym_proof_kernel::capsule::{
    CAPSULE_SCHEMA_VERSION, CapsuleError, CapsuleVerdict, DEFAULT_FUEL, verify_capsule,
};

/// Stable outcome labels for callers that only need the verdict class.
pub const OUTCOME_VERIFIED: &str = "verified";
pub const OUTCOME_REFUTED: &str = "refuted";
pub const OUTCOME_INCONCLUSIVE: &str = "inconclusive";
pub const OUTCOME_REFUSED: &str = "refused";

/// What an offline check produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleReport {
    /// One of the `OUTCOME_*` labels.
    pub outcome: &'static str,
    /// Digest of the checked claim, when the capsule decoded far enough.
    pub claim_digest: Option<[u8; 32]>,
    /// Objects resolved during the check.
    pub objects: usize,
    /// Coefficient multiplications charged.
    pub multiplications: u64,
    /// Deterministic diagnostic: refusal reason or refutation detail.
    pub detail: alloc::string::String,
}

impl CapsuleReport {
    /// True only for `verified`. Nothing else is an acceptance.
    pub fn is_verified(&self) -> bool {
        self.outcome == OUTCOME_VERIFIED
    }

    /// True when the check could not reach a conclusion (fuel exhaustion).
    pub fn is_inconclusive(&self) -> bool {
        self.outcome == OUTCOME_INCONCLUSIVE
    }
}

impl core::fmt::Display for CapsuleReport {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "{} (objects={}, multiplications={}, detail={})",
            self.outcome, self.objects, self.multiplications, self.detail
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CapsuleReport {}

/// Check a capsule offline against an independently supplied claim root.
///
/// `expected_claim_root` is the authority for *which* statement is being
/// accepted; `None` checks the capsule's own claim (useful for inspection, not
/// for accepting a claim). `fuel` bounds coefficient multiplications, and
/// exhaustion returns `inconclusive` — never an acceptance and never a
/// rejection.
pub fn verify_offline(
    bytes: &[u8],
    expected_claim_root: Option<[u8; 32]>,
    fuel: u64,
) -> CapsuleReport {
    match verify_capsule(bytes, expected_claim_root, fuel) {
        CapsuleVerdict::Verified {
            claim_digest,
            objects,
            multiplications,
        } => CapsuleReport {
            outcome: OUTCOME_VERIFIED,
            claim_digest: Some(claim_digest),
            objects,
            multiplications,
            detail: alloc::string::String::new(),
        },
        CapsuleVerdict::Refuted { reason } => CapsuleReport {
            outcome: OUTCOME_REFUTED,
            claim_digest: None,
            objects: 0,
            multiplications: 0,
            detail: reason,
        },
        CapsuleVerdict::Inconclusive { needed_at_least } => CapsuleReport {
            outcome: OUTCOME_INCONCLUSIVE,
            claim_digest: None,
            objects: 0,
            multiplications: needed_at_least,
            detail: alloc::format!("fuel exhausted after {needed_at_least} multiplications"),
        },
        CapsuleVerdict::Refused(error) => CapsuleReport {
            outcome: OUTCOME_REFUSED,
            claim_digest: None,
            objects: 0,
            multiplications: 0,
            detail: describe(&error),
        },
    }
}

/// Convenience wrapper using the default fuel budget.
pub fn verify_offline_default(bytes: &[u8]) -> CapsuleReport {
    verify_offline(bytes, None, DEFAULT_FUEL)
}

/// Schema version this consumer was built against.
pub fn supported_schema_version() -> u16 {
    CAPSULE_SCHEMA_VERSION
}

/// Deterministic, allocation-light diagnostic text for a refusal.
#[cfg(feature = "std")]
pub fn describe(error: &CapsuleError) -> alloc::string::String {
    alloc::format!("{error:?}")
}

#[cfg(not(feature = "std"))]
pub fn describe(error: &CapsuleError) -> alloc::string::String {
    use alloc::string::ToString;
    match error {
        CapsuleError::Malformed(reason) => alloc::format!("malformed: {reason}"),
        CapsuleError::UnknownSchema(what) => alloc::format!("unknown schema: {what}"),
        CapsuleError::DuplicateObjectId(id) => alloc::format!("duplicate object id: {id}"),
        CapsuleError::ObjectDigestMismatch(id) => alloc::format!("object digest mismatch: {id}"),
        CapsuleError::MissingObject(id) => alloc::format!("missing object: {id}"),
        CapsuleError::DuplicateFactor(id) => alloc::format!("duplicate factor: {id}"),
        CapsuleError::ClaimRootMismatch => "claim root mismatch".to_string(),
        CapsuleError::Inconsistent(reason) => alloc::format!("inconsistent: {reason}"),
    }
}
