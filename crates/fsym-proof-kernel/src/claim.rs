//! Typed mathematical claims for the FrankenSymPy proof kernel (WS06).
//!
//! Layer: L2 (claims and proof kernel).
//! Claims define the exact mathematical proposition being established,
//! separated from any generator search trace or heuristic candidate.

#![forbid(unsafe_code)]

use fsym_assumptions::{Domain, Predicate};
use fsym_core::Expr;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Distinct mathematical claim categories.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ClaimKind {
    Equality,
    PredicateHold,
    DomainMembership,
    NonZero,
    AlgebraicIdentity,
}

/// A typed, checkable mathematical proposition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Claim {
    /// Exact symbolic equality between two expressions: $\vdash a = b$.
    Equality { lhs: Expr, rhs: Expr },
    /// Predicate entitlement for an expression: $\vdash P(e)$.
    PredicateHold { expr: Expr, predicate: Predicate },
    /// Exact domain membership: $\vdash e \in \mathcal{D}$.
    DomainMembership { expr: Expr, domain: Domain },
    /// Non-zeroness claim: $\vdash e \neq 0$.
    NonZero(Expr),
    /// Universal algebraic identity across all valid bindings: $\forall \bar{x}, L(\bar{x}) = R(\bar{x})$.
    AlgebraicIdentity { lhs: Expr, rhs: Expr },
}

impl Claim {
    /// Create an equality claim.
    pub fn equality(lhs: Expr, rhs: Expr) -> Self {
        Claim::Equality { lhs, rhs }
    }

    /// Create a predicate holding claim.
    pub fn predicate(expr: Expr, predicate: Predicate) -> Self {
        Claim::PredicateHold { expr, predicate }
    }

    /// Create a domain membership claim.
    pub fn domain_membership(expr: Expr, domain: Domain) -> Self {
        Claim::DomainMembership { expr, domain }
    }

    /// Create a non-zero claim.
    pub fn non_zero(expr: Expr) -> Self {
        Claim::NonZero(expr)
    }

    /// Return the category of this claim.
    pub fn kind(&self) -> ClaimKind {
        match self {
            Claim::Equality { .. } => ClaimKind::Equality,
            Claim::PredicateHold { .. } => ClaimKind::PredicateHold,
            Claim::DomainMembership { .. } => ClaimKind::DomainMembership,
            Claim::NonZero(_) => ClaimKind::NonZero,
            Claim::AlgebraicIdentity { .. } => ClaimKind::AlgebraicIdentity,
        }
    }

    /// Canonical BLAKE3 content digest of this claim.
    pub fn digest(&self) -> [u8; 32] {
        let serialized = serde_json::to_vec(self).expect("claim is serializable");
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.claim.v1:");
        hasher.update(&serialized);
        *hasher.finalize().as_bytes()
    }

    /// Access the LHS expression if this is an equality or algebraic identity.
    pub fn lhs(&self) -> Option<&Expr> {
        match self {
            Claim::Equality { lhs, .. } | Claim::AlgebraicIdentity { lhs, .. } => Some(lhs),
            _ => None,
        }
    }

    /// Access the RHS expression if this is an equality or algebraic identity.
    pub fn rhs(&self) -> Option<&Expr> {
        match self {
            Claim::Equality { rhs, .. } | Claim::AlgebraicIdentity { rhs, .. } => Some(rhs),
            _ => None,
        }
    }
}

impl fmt::Display for Claim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Claim::Equality { lhs, rhs } => write!(f, "{lhs} = {rhs}"),
            Claim::PredicateHold { expr, predicate } => write!(f, "{predicate:?}({expr})"),
            Claim::DomainMembership { expr, domain } => write!(f, "{expr} in {domain}"),
            Claim::NonZero(expr) => write!(f, "{expr} != 0"),
            Claim::AlgebraicIdentity { lhs, rhs } => write!(f, "Identity: {lhs} === {rhs}"),
        }
    }
}
