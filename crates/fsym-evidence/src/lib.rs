//! # fsym-evidence
//!
//! Evidence envelopes, verification receipts, candidate/verified namespaces,
//! and evidence lattice non-conversion rules for FrankenSymPy (WS06).
//!
//! Layer: L2 (evidence).
//! Per `docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md` §6.10, this crate enforces
//! the separation of unverified candidates from certified accepted values.

#![forbid(unsafe_code)]

pub mod envelope;
pub mod lattice;
pub mod namespace;
pub mod receipt;

pub use envelope::EvidenceEnvelope;
pub use lattice::{LatticeError, validate_evidence_transition};
pub use namespace::{CandidateNamespace, NamespaceError, VerifiedNamespace};
pub use receipt::VerificationReceipt;
