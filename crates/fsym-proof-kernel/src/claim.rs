//! Typed mathematical claims for the FrankenSymPy proof kernel (WS06).
//!
//! Layer: L2 (claims and proof kernel).
//! Claims define the exact mathematical proposition being established,
//! separated from any generator search trace or heuristic candidate.

#![forbid(unsafe_code)]

use fsym_assumptions::{Domain, Predicate};
use fsym_core::{BigInt, Expr};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

/// Bound for the synchronous reference modular irreducibility checker.
pub const MAX_MODULAR_IRREDUCIBILITY_DEGREE: usize = 64;

/// Distinct mathematical claim categories.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ClaimKind {
    Equality,
    PredicateHold,
    DomainMembership,
    NonZero,
    AlgebraicIdentity,
    IrreducibleModPrime,
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
    /// The exact polynomial with ascending coefficients is irreducible in F_p[x].
    ///
    /// Coefficients must be canonical residues `0 <= c < prime`, with a nonzero
    /// leading coefficient, and `prime` must be a prime of at most 32 bits.
    /// This claim says nothing about non-zeroness at a symbolic argument, real
    /// or complex roots, or irreducibility over a different field.
    IrreducibleModPrime {
        #[serde(deserialize_with = "bounded_modular_coefficients")]
        coeffs: Vec<BigInt>,
        #[serde(deserialize_with = "bounded_modular_integer")]
        prime: BigInt,
    },
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
            Claim::IrreducibleModPrime { .. } => ClaimKind::IrreducibleModPrime,
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
            Claim::IrreducibleModPrime { coeffs, prime } => {
                write!(f, "Irreducible over F_{prime}: coefficients {coeffs:?}")
            }
        }
    }
}

// The claim and its certificate use the same bounded wire representation.
// Reject each oversize scalar before retaining the next sequence element.
pub(crate) fn bounded_modular_integer<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BigInt, D::Error> {
    let integer = BigInt::deserialize(deserializer)?;
    if integer.bits() > 32 {
        return Err(serde::de::Error::custom(
            "modular irreducibility integer exceeds 32 bits",
        ));
    }
    Ok(integer)
}

#[derive(Deserialize)]
struct ModularInteger(#[serde(deserialize_with = "bounded_modular_integer")] BigInt);

pub(crate) fn bounded_modular_coefficients<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<BigInt>, D::Error> {
    struct Coefficients;
    impl<'de> Visitor<'de> for Coefficients {
        type Value = Vec<BigInt>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "at most {} modular polynomial coefficients",
                MAX_MODULAR_IRREDUCIBILITY_DEGREE + 1
            )
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            const MAX: usize = MAX_MODULAR_IRREDUCIBILITY_DEGREE + 1;
            if sequence.size_hint().is_some_and(|size| size > MAX) {
                return Err(serde::de::Error::custom(
                    "modular irreducibility degree limit exceeded",
                ));
            }
            let mut coefficients = Vec::new();
            loop {
                if coefficients.len() == MAX {
                    if sequence.next_element::<IgnoredAny>()?.is_some() {
                        return Err(serde::de::Error::custom(
                            "modular irreducibility degree limit exceeded",
                        ));
                    }
                    return Ok(coefficients);
                }
                match sequence.next_element::<ModularInteger>()? {
                    Some(coefficient) => coefficients.push(coefficient.0),
                    None => return Ok(coefficients),
                }
            }
        }
    }
    deserializer.deserialize_seq(Coefficients)
}
