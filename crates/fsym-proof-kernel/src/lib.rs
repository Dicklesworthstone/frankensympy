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
//! - Certificate lemma verification via trusted family dispatchers (e.g. `RealBall`), failing closed for unregistered families;
//! - Independent, stateless derivation tree verification.

#![forbid(unsafe_code)]

pub mod claim;
pub mod kernel;
pub mod mutation;
pub mod rule;

pub use claim::{Claim, ClaimKind};
pub use kernel::{
    DerivationStep, DerivationTree, KernelError, MAX_DERIVATION_STEPS, ProofKernel,
    claim_verification_units, derivation_verification_units, expression_verification_units,
    verify_derivation_independent,
};
pub use rule::{ProofRule, StepId};
