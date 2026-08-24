//! # fsym-proof-kernel
//!
//! Small, trusted proof kernel and derivation checker for FrankenSymPy (WS06).
//!
//! Layer: L2 (claims and proof kernel).
//! Per `docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md` §6.8, this crate contains
//! the small trusted proof-term checker:
//! - Equality, symmetry, transitivity, and congruence;
//! - Immutable assumptions context checking and side-condition verification;
//! - Capture-safe substitution;
//! - Definitional reductions and arithmetic evaluations;
//! - Fail-closed certificate lemma syntax pending a trusted family dispatcher;
//! - Independent, stateless derivation tree verification.

#![forbid(unsafe_code)]

pub mod claim;
pub mod kernel;
pub mod mutation;
pub mod rule;

pub use claim::{Claim, ClaimKind};
pub use kernel::{
    DerivationStep, DerivationTree, KernelError, ProofKernel, verify_derivation_independent,
};
pub use rule::{ProofRule, StepId};
