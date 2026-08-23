//! # fsym-sets
//!
//! Symbolic sets: intervals, finite sets, unions, intersections, complements, and conditions.

#![forbid(unsafe_code)]

use fsym_core::Expr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SetError {
    #[error("Invalid interval bounds: start ({0}) > end ({1})")]
    InvalidInterval(String, String),
    #[error("Operation not supported between given sets")]
    UnsupportedOperation,
}

/// Symbolic set representation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SymSet {
    /// Empty set ∅.
    EmptySet,
    /// Universal set 𝕌.
    UniversalSet,
    /// Real numeric interval: [start, end], (start, end), [start, end), (start, end].
    Interval {
        start: Expr,
        end: Expr,
        left_open: bool,
        right_open: bool,
    },
    /// Explicit finite discrete set: {x_1, x_2, ...}.
    FiniteSet(BTreeSet<Expr>),
    /// Union of sets: S_1 ∪ S_2 ∪ ...
    Union(Vec<SymSet>),
    /// Intersection of sets: S_1 ∩ S_2 ∩ ...
    Intersection(Vec<SymSet>),
    /// Complement of set: U \ S.
    Complement(Box<SymSet>),
}

impl SymSet {
    pub fn empty() -> Self {
        SymSet::EmptySet
    }

    pub fn universal() -> Self {
        SymSet::UniversalSet
    }

    pub fn interval_closed(start: Expr, end: Expr) -> Self {
        SymSet::Interval {
            start,
            end,
            left_open: false,
            right_open: false,
        }
    }

    pub fn interval_open(start: Expr, end: Expr) -> Self {
        SymSet::Interval {
            start,
            end,
            left_open: true,
            right_open: true,
        }
    }

    pub fn finite(elements: impl IntoIterator<Item = Expr>) -> Self {
        let set: BTreeSet<Expr> = elements.into_iter().collect();
        if set.is_empty() {
            SymSet::EmptySet
        } else {
            SymSet::FiniteSet(set)
        }
    }

    pub fn union(self, other: SymSet) -> Self {
        match (self, other) {
            (SymSet::EmptySet, s) | (s, SymSet::EmptySet) => s,
            (SymSet::UniversalSet, _) | (_, SymSet::UniversalSet) => SymSet::UniversalSet,
            (SymSet::Union(mut a), SymSet::Union(b)) => {
                a.extend(b);
                SymSet::Union(a)
            }
            (SymSet::Union(mut a), b) | (b, SymSet::Union(mut a)) => {
                a.push(b);
                SymSet::Union(a)
            }
            (a, b) => SymSet::Union(vec![a, b]),
        }
    }

    pub fn intersection(self, other: SymSet) -> Self {
        match (self, other) {
            (SymSet::EmptySet, _) | (_, SymSet::EmptySet) => SymSet::EmptySet,
            (SymSet::UniversalSet, s) | (s, SymSet::UniversalSet) => s,
            (SymSet::Intersection(mut a), SymSet::Intersection(b)) => {
                a.extend(b);
                SymSet::Intersection(a)
            }
            (SymSet::Intersection(mut a), b) | (b, SymSet::Intersection(mut a)) => {
                a.push(b);
                SymSet::Intersection(a)
            }
            (a, b) => SymSet::Intersection(vec![a, b]),
        }
    }
}

impl fmt::Display for SymSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymSet::EmptySet => write!(f, "EmptySet"),
            SymSet::UniversalSet => write!(f, "UniversalSet"),
            SymSet::Interval {
                start,
                end,
                left_open,
                right_open,
            } => {
                let l = if *left_open { "(" } else { "[" };
                let r = if *right_open { ")" } else { "]" };
                write!(f, "Interval{}{}, {}{}", l, start, end, r)
            }
            SymSet::FiniteSet(elems) => {
                let s = elems
                    .iter()
                    .map(|e| format!("{}", e))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "FiniteSet({})", s)
            }
            SymSet::Union(sets) => {
                let s = sets
                    .iter()
                    .map(|set| format!("{}", set))
                    .collect::<Vec<_>>()
                    .join(" | ");
                write!(f, "Union({})", s)
            }
            SymSet::Intersection(sets) => {
                let s = sets
                    .iter()
                    .map(|set| format!("{}", set))
                    .collect::<Vec<_>>()
                    .join(" & ");
                write!(f, "Intersection({})", s)
            }
            SymSet::Complement(inner) => write!(f, "Complement({})", inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_set_and_interval() {
        let set = SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(2)]);
        let interval = SymSet::interval_closed(Expr::from_i64(0), Expr::from_i64(5));
        let union = set.union(interval);
        assert!(matches!(union, SymSet::Union(_)));
    }
}
