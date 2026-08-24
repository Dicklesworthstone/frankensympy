//! Four-way truth value representation for WS04 assumptions reasoning.
//!
//! Internal truth distinguishes:
//! - [`TruthValue::EntailedTrue`]: Derivable from active facts and algebraic properties.
//! - [`TruthValue::EntailedFalse`]: Refutable from active facts and algebraic properties.
//! - [`TruthValue::Unknown`]: Undetermined by available facts; never coerced to false.
//! - [`TruthValue::Contradictory`]: Inconsistent or overdetermined assumptions.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{BitAnd, BitOr, Not};

/// Four-valued truth model for symbolic predicates and side conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TruthValue {
    /// True by deductive entailment.
    EntailedTrue,
    /// False by deductive entailment.
    EntailedFalse,
    /// Genuinely unknown under active facts; cannot be assumed false.
    Unknown,
    /// Inconsistent or contradictory under active facts.
    Contradictory,
}

impl TruthValue {
    /// Whether this value is definitely true.
    pub const fn is_entailed_true(self) -> bool {
        matches!(self, Self::EntailedTrue)
    }

    /// Whether this value is definitely false.
    pub const fn is_entailed_false(self) -> bool {
        matches!(self, Self::EntailedFalse)
    }

    /// Whether this value is undetermined.
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Whether this value represents an explicit contradiction.
    pub const fn is_contradictory(self) -> bool {
        matches!(self, Self::Contradictory)
    }

    /// Whether this value has a decisive 2-valued outcome (`Some(true)` or `Some(false)`).
    pub const fn is_decided(self) -> bool {
        matches!(self, Self::EntailedTrue | Self::EntailedFalse)
    }

    /// Converts to standard `Option<bool>` for backward compatibility.
    /// Both `Unknown` and `Contradictory` map to `None`.
    pub const fn to_option_bool(self) -> Option<bool> {
        match self {
            Self::EntailedTrue => Some(true),
            Self::EntailedFalse => Some(false),
            Self::Unknown | Self::Contradictory => None,
        }
    }

    /// Constructs from a standard boolean.
    pub const fn from_bool(b: bool) -> Self {
        if b {
            Self::EntailedTrue
        } else {
            Self::EntailedFalse
        }
    }

    /// Constructs from an `Option<bool>`.
    pub const fn from_option_bool(b: Option<bool>) -> Self {
        match b {
            Some(true) => Self::EntailedTrue,
            Some(false) => Self::EntailedFalse,
            None => Self::Unknown,
        }
    }

    /// Logical conjunction in four-valued logic.
    /// Contradictions propagate; false dominates over unknown.
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Contradictory, _) | (_, Self::Contradictory) => Self::Contradictory,
            (Self::EntailedFalse, _) | (_, Self::EntailedFalse) => Self::EntailedFalse,
            (Self::EntailedTrue, Self::EntailedTrue) => Self::EntailedTrue,
            _ => Self::Unknown,
        }
    }

    /// Logical disjunction in four-valued logic.
    /// Contradictions propagate; true dominates over unknown.
    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::Contradictory, _) | (_, Self::Contradictory) => Self::Contradictory,
            (Self::EntailedTrue, _) | (_, Self::EntailedTrue) => Self::EntailedTrue,
            (Self::EntailedFalse, Self::EntailedFalse) => Self::EntailedFalse,
            _ => Self::Unknown,
        }
    }

    /// Logical implication: `A -> B == !A || B`.
    pub fn implies(self, other: Self) -> Self {
        (!self).or(other)
    }
}

impl Not for TruthValue {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::EntailedTrue => Self::EntailedFalse,
            Self::EntailedFalse => Self::EntailedTrue,
            Self::Unknown => Self::Unknown,
            Self::Contradictory => Self::Contradictory,
        }
    }
}

impl BitAnd for TruthValue {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}

impl BitOr for TruthValue {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

impl fmt::Display for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntailedTrue => write!(f, "True"),
            Self::EntailedFalse => write!(f, "False"),
            Self::Unknown => write!(f, "Unknown"),
            Self::Contradictory => write!(f, "Contradictory"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_way_truth_identities() {
        use TruthValue::*;

        assert_eq!(!EntailedTrue, EntailedFalse);
        assert_eq!(!EntailedFalse, EntailedTrue);
        assert_eq!(!Unknown, Unknown);
        assert_eq!(!Contradictory, Contradictory);

        // Conjunction
        assert_eq!(EntailedTrue & EntailedTrue, EntailedTrue);
        assert_eq!(EntailedTrue & EntailedFalse, EntailedFalse);
        assert_eq!(EntailedTrue & Unknown, Unknown);
        assert_eq!(EntailedFalse & Unknown, EntailedFalse);
        assert_eq!(Contradictory & EntailedTrue, Contradictory);
        assert_eq!(Contradictory & EntailedFalse, Contradictory);

        // Disjunction
        assert_eq!(EntailedTrue | EntailedFalse, EntailedTrue);
        assert_eq!(EntailedFalse | EntailedFalse, EntailedFalse);
        assert_eq!(EntailedFalse | Unknown, Unknown);
        assert_eq!(EntailedTrue | Unknown, EntailedTrue);
        assert_eq!(Contradictory | EntailedTrue, Contradictory);

        // Implication
        assert_eq!(EntailedTrue.implies(EntailedTrue), EntailedTrue);
        assert_eq!(EntailedTrue.implies(EntailedFalse), EntailedFalse);
        assert_eq!(EntailedFalse.implies(EntailedFalse), EntailedTrue);
        assert_eq!(Unknown.implies(EntailedTrue), EntailedTrue);
    }
}
