//! # fsym-sets
//!
//! Symbolic sets: intervals, finite sets, unions, intersections, complements, and conditions.

#![forbid(unsafe_code)]

use fsym_core::{BigRational, Constant, Expr};
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

    /// Construct a closed interval while validating numeric bounds.
    ///
    /// Inverted numeric bounds yield [`SetError::InvalidInterval`]; symbolic
    /// bounds cannot be validated eagerly and are accepted.
    pub fn interval_checked(start: Expr, end: Expr) -> Result<Self, SetError> {
        if let (Some(s), Some(e)) = (numeric_value(&start), numeric_value(&end))
            && s > e
        {
            return Err(SetError::InvalidInterval(
                format!("{}", start),
                format!("{}", end),
            ));
        }
        Ok(SymSet::interval_closed(start, end))
    }

    /// Three-valued membership: `Some(decision)` or `None` when the set is
    /// too symbolic to decide for this element.
    pub fn contains(&self, elem: &Expr) -> Option<bool> {
        match self {
            SymSet::EmptySet => Some(false),
            SymSet::UniversalSet => Some(true),
            SymSet::FiniteSet(elems) => Some(elems.contains(elem)),
            SymSet::Interval {
                start,
                end,
                left_open,
                right_open,
            } => {
                use std::cmp::Ordering;
                let el = numeric_value(elem)?;
                let start_cmp = cmp_to_bound(&el, start)?;
                let end_cmp = cmp_to_bound(&el, end)?;
                let above_start =
                    start_cmp == Ordering::Greater || (!*left_open && start_cmp == Ordering::Equal);
                let below_end =
                    end_cmp == Ordering::Less || (!*right_open && end_cmp == Ordering::Equal);
                Some(above_start && below_end)
            }
            SymSet::Union(parts) => {
                let mut all_false = true;
                for part in parts {
                    match part.contains(elem) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => all_false = false,
                    }
                }
                if all_false { Some(false) } else { None }
            }
            SymSet::Intersection(parts) => {
                let mut all_true = true;
                for part in parts {
                    match part.contains(elem) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => all_true = false,
                    }
                }
                if all_true { Some(true) } else { None }
            }
            SymSet::Complement(inner) => inner.contains(elem).map(|b| !b),
        }
    }

    /// Relative complement against the universal set, simplified by De
    /// Morgan's laws where the shape allows it.
    pub fn complement(self) -> Self {
        match self {
            SymSet::EmptySet => SymSet::UniversalSet,
            SymSet::UniversalSet => SymSet::EmptySet,
            SymSet::Complement(inner) => *inner,
            SymSet::Interval {
                start,
                end,
                left_open,
                right_open,
            } => SymSet::Union(vec![
                SymSet::Interval {
                    start: Expr::Const(Constant::NegativeInfinity),
                    end: start,
                    left_open: true,
                    right_open: !left_open,
                },
                SymSet::Interval {
                    start: end,
                    end: Expr::Const(Constant::Infinity),
                    left_open: !right_open,
                    right_open: true,
                },
            ]),
            SymSet::FiniteSet(_) => SymSet::Complement(Box::new(self)),
            SymSet::Union(parts) => {
                SymSet::Intersection(parts.into_iter().map(SymSet::complement).collect())
            }
            SymSet::Intersection(parts) => {
                SymSet::Union(parts.into_iter().map(SymSet::complement).collect())
            }
        }
    }

    /// Set difference computed as `self ∩ otherᶜ`.
    pub fn difference(self, other: SymSet) -> Self {
        self.intersection(other.complement())
    }
}

/// Exact numeric view of an expression, if it is a rational constant.
fn numeric_value(e: &Expr) -> Option<BigRational> {
    match e {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Ordering of an exact element against an interval bound, including
/// infinite bounds emitted by `complement`.
fn cmp_to_bound(el: &BigRational, bound: &Expr) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    match bound {
        Expr::Integer(n) => Some(el.cmp(&BigRational::from_integer(n.clone()))),
        Expr::Rational(r) => Some(el.cmp(r)),
        Expr::Const(Constant::Infinity) => Some(Ordering::Less),
        Expr::Const(Constant::NegativeInfinity) => Some(Ordering::Greater),
        _ => None,
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

    #[test]
    fn test_membership_finite_and_intervals() {
        let s = SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(3)]);
        assert_eq!(s.contains(&Expr::from_i64(3)), Some(true));
        assert_eq!(s.contains(&Expr::from_i64(2)), Some(false));

        let closed = SymSet::interval_closed(Expr::from_i64(0), Expr::from_i64(5));
        assert_eq!(closed.contains(&Expr::from_i64(0)), Some(true));
        assert_eq!(closed.contains(&Expr::from_i64(5)), Some(true));
        let open = SymSet::interval_open(Expr::from_i64(0), Expr::from_i64(5));
        assert_eq!(open.contains(&Expr::from_i64(0)), Some(false));
        // Symbolic element against numeric bounds is undecidable, not false.
        assert_eq!(open.contains(&Expr::symbol("x")), None);
    }

    #[test]
    fn test_interval_with_symbolic_bounds_undecidable() {
        let iv = SymSet::interval_closed(Expr::symbol("a"), Expr::from_i64(10));
        assert_eq!(iv.contains(&Expr::from_i64(3)), None);
    }

    #[test]
    fn test_complement_boundaries_and_involution() {
        let iv = SymSet::interval_closed(Expr::from_i64(0), Expr::from_i64(5));
        let comp = iv.complement();
        // Endpoints of a closed interval are excluded from its complement.
        assert_eq!(comp.contains(&Expr::from_i64(0)), Some(false));
        assert_eq!(comp.contains(&Expr::from_i64(5)), Some(false));
        assert_eq!(comp.contains(&Expr::from_i64(-1)), Some(true));
        assert_eq!(comp.contains(&Expr::from_i64(7)), Some(true));
        assert_eq!(comp.contains(&Expr::from_i64(3)), Some(false));
    }

    #[test]
    fn test_finite_set_double_complement_is_identity() {
        let s = SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(9)]);
        assert_eq!(s.clone().complement().complement(), s);
        // Empty/universal complements swap.
        assert_eq!(SymSet::EmptySet.complement(), SymSet::UniversalSet);
    }

    #[test]
    fn test_de_morgan_complements_agree_on_membership() {
        let a = SymSet::finite(vec![Expr::from_i64(1)]);
        let b = SymSet::interval_closed(Expr::from_i64(0), Expr::from_i64(10));
        let lhs = a.clone().union(b.clone()).complement();
        let rhs = a.complement().intersection(b.complement());
        for p in [-1i64, 0, 1, 5, 10, 11] {
            assert_eq!(
                lhs.contains(&Expr::from_i64(p)),
                rhs.contains(&Expr::from_i64(p)),
                "mismatch at {p}"
            );
        }
    }

    #[test]
    fn test_difference_punches_holes() {
        let span = SymSet::interval_closed(Expr::from_i64(0), Expr::from_i64(10));
        let hole = SymSet::finite(vec![Expr::from_i64(5)]);
        let d = span.difference(hole);
        assert_eq!(d.contains(&Expr::from_i64(5)), Some(false));
        assert_eq!(d.contains(&Expr::from_i64(4)), Some(true));
        assert_eq!(d.contains(&Expr::from_i64(11)), Some(false));
    }

    #[test]
    fn test_checked_interval_rejects_inverted_bounds() {
        assert!(matches!(
            SymSet::interval_checked(Expr::from_i64(5), Expr::from_i64(1)),
            Err(SetError::InvalidInterval(_, _))
        ));
        assert!(matches!(
            SymSet::interval_checked(Expr::from_i64(1), Expr::from_i64(5)),
            Ok(SymSet::Interval { .. })
        ));
        // Symbolic bounds cannot be validated eagerly.
        assert!(SymSet::interval_checked(Expr::symbol("a"), Expr::symbol("b")).is_ok());
    }
}
