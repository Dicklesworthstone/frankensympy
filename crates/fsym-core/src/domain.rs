//! Declared intern-time mathematical domain for WS04 terms.
//!
//! Distinct from [`crate::sort::Sort`] (kind / numeric-tower inference) and from
//! the richer ring/field constructors in `fsym-assumptions`. Constitution §7.4:
//! domain, sort, assumptions context, branch policy, and compatibility facts
//! stay separate. This type is the intern identity's declared universe.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

/// Declared intern universe for a semantic term.
///
/// Default intern (`TermDag::insert_node`) uses [`TermDomain::Expression`].
/// Narrower domains are identity-relevant: the same operator payload interned
/// at `Integer` is not the same term as that payload interned at `Expression`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum TermDomain {
    /// Exact integers \(\mathbb{Z}\).
    Integer,
    /// Exact rationals \(\mathbb{Q}\).
    Rational,
    /// Reals \(\mathbb{R}\).
    Real,
    /// Complex numbers \(\mathbb{C}\).
    Complex,
    /// Unrestricted expression universe. Default intern domain.
    #[default]
    Expression,
}

impl TermDomain {
    /// Canonical tag byte mixed into the term preimage.
    pub const fn tag(self) -> u8 {
        match self {
            TermDomain::Integer => 0,
            TermDomain::Rational => 1,
            TermDomain::Real => 2,
            TermDomain::Complex => 3,
            TermDomain::Expression => 4,
        }
    }

    /// Whether this domain is a numeric tower member rather than the catch-all
    /// expression universe.
    pub const fn is_numeric(self) -> bool {
        !matches!(self, TermDomain::Expression)
    }
}

impl fmt::Display for TermDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermDomain::Integer => write!(f, "Integer"),
            TermDomain::Rational => write!(f, "Rational"),
            TermDomain::Real => write!(f, "Real"),
            TermDomain::Complex => write!(f, "Complex"),
            TermDomain::Expression => write!(f, "Expression"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_tags_are_stable_and_distinct() {
        let tags = [
            TermDomain::Integer.tag(),
            TermDomain::Rational.tag(),
            TermDomain::Real.tag(),
            TermDomain::Complex.tag(),
            TermDomain::Expression.tag(),
        ];
        let unique: std::collections::BTreeSet<u8> = tags.into_iter().collect();
        assert_eq!(unique.len(), tags.len());
        assert_eq!(TermDomain::default(), TermDomain::Expression);
        assert!(TermDomain::Integer.is_numeric());
        assert!(!TermDomain::Expression.is_numeric());
    }
}
