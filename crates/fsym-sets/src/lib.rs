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

    pub fn interval_left_open(start: Expr, end: Expr) -> Self {
        SymSet::Interval {
            start,
            end,
            left_open: true,
            right_open: false,
        }
    }

    pub fn interval_right_open(start: Expr, end: Expr) -> Self {
        SymSet::Interval {
            start,
            end,
            left_open: false,
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

    /// Construct a closed interval while validating exact extended-real bounds.
    ///
    /// Inverted exact bounds yield [`SetError::InvalidInterval`]; symbolic
    /// bounds cannot be validated eagerly and are accepted.
    pub fn interval_checked(start: Expr, end: Expr) -> Result<Self, SetError> {
        if let (Some(s), Some(e)) = (exact_bound(&start), exact_bound(&end))
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
            SymSet::FiniteSet(elems) => {
                if elems.contains(elem) {
                    return Some(true);
                }

                let elem_numeric = numeric_value(elem);
                let mut all_distinct = true;
                for member in elems {
                    match (elem_numeric.as_ref(), numeric_value(member)) {
                        (Some(candidate), Some(member)) if candidate == &member => {
                            return Some(true);
                        }
                        (Some(_), Some(_)) => {}
                        _ => all_distinct = false,
                    }
                }

                // Structural inequality alone does not refute mathematical
                // equality. For example, a symbol in `{x}` may equal `1`
                // under a later substitution. Only exact numeric pairs are
                // currently strong enough to establish non-membership.
                if all_distinct { Some(false) } else { None }
            }
            SymSet::Interval {
                start,
                end,
                left_open,
                right_open,
            } => {
                use std::cmp::Ordering;
                // Decide interval emptiness before inspecting the element. An inverted
                // exact interval, an interval concentrated at infinity, or a zero-width
                // finite interval with either endpoint open contains nothing even when the
                // candidate element is symbolic.
                if interval_empty_status(start, end, *left_open, *right_open) == Some(true) {
                    return Some(false);
                }
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

    /// Symmetric difference computed as `(self \ other) ∪ (other \ self)`.
    pub fn symmetric_difference(self, other: SymSet) -> Self {
        self.clone()
            .difference(other.clone())
            .union(other.difference(self))
    }

    /// Decides whether the set is definitively empty (`Some(true)`), non-empty (`Some(false)`),
    /// or undetermined under symbolic parameters (`None`).
    pub fn is_empty_set(&self) -> Option<bool> {
        match self {
            SymSet::EmptySet => Some(true),
            SymSet::UniversalSet => Some(false),
            SymSet::FiniteSet(elems) => Some(elems.is_empty()),
            SymSet::Interval {
                start,
                end,
                left_open,
                right_open,
            } => interval_empty_status(start, end, *left_open, *right_open)
                .or_else(|| (start == end && (*left_open || *right_open)).then_some(true)),
            SymSet::Union(parts) => {
                let mut all_empty = true;
                for part in parts {
                    match part.is_empty_set() {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => all_empty = false,
                    }
                }
                if all_empty { Some(true) } else { None }
            }
            SymSet::Intersection(parts) => {
                for part in parts {
                    if part.is_empty_set() == Some(true) {
                        return Some(true);
                    }
                }
                None
            }
            SymSet::Complement(inner) => match inner.as_ref() {
                SymSet::UniversalSet => Some(true),
                SymSet::EmptySet => Some(false),
                _ => None,
            },
        }
    }

    /// Determines whether `self` is a subset of `other` (`self ⊆ other`).
    ///
    /// Returns `Some(true)` if provably a subset, `Some(false)` if provably not,
    /// and `None` if undetermined due to symbolic parameters.
    pub fn is_subset(&self, other: &SymSet) -> Option<bool> {
        if self == other {
            return Some(true);
        }
        if let Some(true) = self.is_empty_set() {
            return Some(true);
        }
        if let Some(true) = other.is_empty_set() {
            return self.is_empty_set();
        }

        match (self, other) {
            (SymSet::EmptySet, _) => Some(true),
            (_, SymSet::UniversalSet) => Some(true),
            (SymSet::UniversalSet, SymSet::EmptySet) => Some(false),
            (SymSet::FiniteSet(a_elems), SymSet::FiniteSet(b_elems)) => {
                let mut all_found = true;
                for a in a_elems {
                    if !b_elems.contains(a) {
                        if let Some(a_num) = numeric_value(a) {
                            let mut found_match = false;
                            let mut has_symbolic = false;
                            for b in b_elems {
                                if let Some(b_num) = numeric_value(b) {
                                    if a_num == b_num {
                                        found_match = true;
                                        break;
                                    }
                                } else {
                                    has_symbolic = true;
                                }
                            }
                            if !found_match && !has_symbolic {
                                return Some(false);
                            }
                            if !found_match {
                                all_found = false;
                            }
                        } else {
                            all_found = false;
                        }
                    }
                }
                if all_found { Some(true) } else { None }
            }
            (SymSet::FiniteSet(elems), other_set) => {
                let mut all_contained = true;
                for elem in elems {
                    match other_set.contains(elem) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => all_contained = false,
                    }
                }
                if all_contained { Some(true) } else { None }
            }
            (
                SymSet::Interval {
                    start: a_s,
                    end: a_e,
                    left_open: a_lo,
                    right_open: a_ro,
                },
                SymSet::Interval {
                    start: b_s,
                    end: b_e,
                    left_open: b_lo,
                    right_open: b_ro,
                },
            ) => {
                if let (Some(as_v), Some(ae_v), Some(bs_v), Some(be_v)) = (
                    numeric_value(a_s),
                    numeric_value(a_e),
                    numeric_value(b_s),
                    numeric_value(b_e),
                ) {
                    let start_ok = match bs_v.cmp(&as_v) {
                        std::cmp::Ordering::Less => true,
                        std::cmp::Ordering::Equal => !*b_lo || *a_lo,
                        std::cmp::Ordering::Greater => false,
                    };
                    let end_ok = match ae_v.cmp(&be_v) {
                        std::cmp::Ordering::Less => true,
                        std::cmp::Ordering::Equal => !*b_ro || *a_ro,
                        std::cmp::Ordering::Greater => false,
                    };
                    Some(start_ok && end_ok)
                } else {
                    None
                }
            }
            (SymSet::Union(parts), other_set) => {
                let mut all_subset = true;
                for part in parts {
                    match part.is_subset(other_set) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => all_subset = false,
                    }
                }
                if all_subset { Some(true) } else { None }
            }
            (self_set, SymSet::Intersection(parts)) => {
                let mut all_subset = true;
                for part in parts {
                    match self_set.is_subset(part) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => all_subset = false,
                    }
                }
                if all_subset { Some(true) } else { None }
            }
            _ => None,
        }
    }

    /// Determines whether `self` is a superset of `other` (`self ⊇ other`).
    pub fn is_superset(&self, other: &SymSet) -> Option<bool> {
        other.is_subset(self)
    }

    /// Determines whether `self` and `other` are disjoint (`self ∩ other = ∅`).
    pub fn is_disjoint(&self, other: &SymSet) -> Option<bool> {
        if self.is_empty_set() == Some(true) || other.is_empty_set() == Some(true) {
            return Some(true);
        }
        match (self, other) {
            (SymSet::FiniteSet(elems), other_set) | (other_set, SymSet::FiniteSet(elems)) => {
                let mut all_disjoint = true;
                for elem in elems {
                    match other_set.contains(elem) {
                        Some(true) => return Some(false),
                        Some(false) => {}
                        None => all_disjoint = false,
                    }
                }
                if all_disjoint { Some(true) } else { None }
            }
            (
                SymSet::Interval {
                    start: a_s,
                    end: a_e,
                    left_open: a_lo,
                    right_open: a_ro,
                },
                SymSet::Interval {
                    start: b_s,
                    end: b_e,
                    left_open: b_lo,
                    right_open: b_ro,
                },
            ) => {
                if let (Some(as_v), Some(ae_v), Some(bs_v), Some(be_v)) = (
                    numeric_value(a_s),
                    numeric_value(a_e),
                    numeric_value(b_s),
                    numeric_value(b_e),
                ) {
                    let a_strictly_left_of_b = ae_v < bs_v || (ae_v == bs_v && (*a_ro || *b_lo));
                    let b_strictly_left_of_a = be_v < as_v || (be_v == as_v && (*b_ro || *a_lo));
                    Some(a_strictly_left_of_b || b_strictly_left_of_a)
                } else {
                    None
                }
            }
            (SymSet::Union(parts), other_set) | (other_set, SymSet::Union(parts)) => {
                let mut all_disjoint = true;
                for part in parts {
                    match part.is_disjoint(other_set) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => all_disjoint = false,
                    }
                }
                if all_disjoint { Some(true) } else { None }
            }
            _ => None,
        }
    }

    /// Computes the 1D Lebesgue measure / length of the set.
    pub fn measure(&self) -> Option<Expr> {
        match self {
            SymSet::EmptySet => Some(Expr::from_i64(0)),
            SymSet::UniversalSet => Some(Expr::Const(Constant::Infinity)),
            SymSet::FiniteSet(_) => Some(Expr::from_i64(0)),
            SymSet::Interval {
                start,
                end,
                left_open,
                right_open,
            } => match interval_empty_status(start, end, *left_open, *right_open) {
                Some(true) => Some(Expr::from_i64(0)),
                Some(false) => match (exact_bound(start), exact_bound(end)) {
                    (Some(ExactBound::Finite(s)), Some(ExactBound::Finite(e))) => {
                        let diff = e - s;
                        if diff.is_integer() {
                            Some(Expr::Integer(diff.to_integer()))
                        } else {
                            Some(Expr::Rational(diff))
                        }
                    }
                    (Some(_), Some(_)) => Some(Expr::Const(Constant::Infinity)),
                    _ => None,
                },
                None if start == end => Some(Expr::from_i64(0)),
                None => None,
            },
            SymSet::Complement(inner) => match inner.as_ref() {
                SymSet::EmptySet => Some(Expr::Const(Constant::Infinity)),
                SymSet::UniversalSet => Some(Expr::from_i64(0)),
                _ => None,
            },
            _ => None,
        }
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ExactBound {
    NegativeInfinity,
    Finite(BigRational),
    Infinity,
}

/// Exact extended-real view of an interval endpoint.
fn exact_bound(expr: &Expr) -> Option<ExactBound> {
    match expr {
        Expr::Integer(value) => Some(ExactBound::Finite(BigRational::from_integer(value.clone()))),
        Expr::Rational(value) => Some(ExactBound::Finite(value.clone())),
        Expr::Const(Constant::NegativeInfinity) => Some(ExactBound::NegativeInfinity),
        Expr::Const(Constant::Infinity) => Some(ExactBound::Infinity),
        _ => None,
    }
}

/// Decides interval emptiness when both endpoints have exact extended-real order.
/// Infinite endpoints are not themselves real members, so a zero-width interval
/// at either infinity is empty even when both endpoint flags are closed.
fn interval_empty_status(
    start: &Expr,
    end: &Expr,
    left_open: bool,
    right_open: bool,
) -> Option<bool> {
    let start = exact_bound(start)?;
    let end = exact_bound(end)?;
    Some(match start.cmp(&end) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => {
            !matches!(start, ExactBound::Finite(_)) || left_open || right_open
        }
    })
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
    fn symbolic_finite_set_membership_remains_unknown() {
        let symbolic = SymSet::finite([Expr::symbol("x")]);
        assert_eq!(symbolic.contains(&Expr::from_i64(1)), None);
        assert_eq!(symbolic.contains(&Expr::symbol("y")), None);
        assert_eq!(symbolic.contains(&Expr::symbol("x")), Some(true));

        // Complement must not turn an unresolved equality into a proof of
        // membership.
        assert_eq!(
            symbolic.clone().complement().contains(&Expr::from_i64(1)),
            None
        );

        let mixed = SymSet::finite([Expr::from_i64(1), Expr::symbol("x")]);
        assert_eq!(mixed.contains(&Expr::from_i64(1)), Some(true));
        assert_eq!(mixed.contains(&Expr::from_i64(2)), None);
    }

    #[test]
    fn test_interval_with_symbolic_bounds_undecidable() {
        let iv = SymSet::interval_closed(Expr::symbol("a"), Expr::from_i64(10));
        assert_eq!(iv.contains(&Expr::from_i64(3)), None);
    }

    #[test]
    fn numerically_empty_intervals_are_false_before_element_evaluation() {
        let symbolic_element = Expr::symbol("x");
        let open_point = SymSet::interval_open(Expr::from_i64(0), Expr::from_i64(0));
        assert_eq!(open_point.contains(&symbolic_element), Some(false));

        let half_open_point = SymSet::Interval {
            start: Expr::from_i64(2),
            end: Expr::from_i64(2),
            left_open: false,
            right_open: true,
        };
        assert_eq!(half_open_point.contains(&symbolic_element), Some(false));

        // The unchecked convenience constructors preserve their existing API, but
        // an inverted numeric interval still has a decidable empty membership set.
        let inverted = SymSet::interval_closed(Expr::from_i64(5), Expr::from_i64(1));
        assert_eq!(inverted.contains(&symbolic_element), Some(false));
    }

    #[test]
    fn interval_measure_requires_proven_extended_real_order() {
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let negative_infinity = Expr::Const(Constant::NegativeInfinity);
        let infinity = Expr::Const(Constant::Infinity);

        assert_eq!(
            SymSet::interval_closed(x.clone(), zero.clone()).measure(),
            None,
            "a conditional symbolic ordering must not produce an unguarded negative length"
        );
        assert_eq!(
            SymSet::interval_closed(zero.clone(), x.clone()).measure(),
            None
        );
        assert_eq!(
            SymSet::interval_closed(x.clone(), x).measure(),
            Some(zero.clone()),
            "equal symbolic endpoints have zero measure regardless of singleton emptiness"
        );

        for point_at_infinity in [
            SymSet::interval_closed(infinity.clone(), infinity.clone()),
            SymSet::interval_closed(negative_infinity.clone(), negative_infinity.clone()),
        ] {
            assert_eq!(point_at_infinity.is_empty_set(), Some(true));
            assert_eq!(point_at_infinity.contains(&Expr::symbol("y")), Some(false));
            assert_eq!(point_at_infinity.measure(), Some(zero.clone()));
        }

        let all_reals = SymSet::interval_open(negative_infinity.clone(), infinity.clone());
        assert_eq!(all_reals.is_empty_set(), Some(false));
        assert_eq!(all_reals.contains(&zero), Some(true));
        assert_eq!(all_reals.measure(), Some(infinity.clone()));

        assert!(matches!(
            SymSet::interval_checked(infinity, Expr::from_i64(1)),
            Err(SetError::InvalidInterval(_, _))
        ));
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

    #[test]
    fn test_half_open_intervals_and_symmetric_difference() {
        let left_open = SymSet::interval_left_open(Expr::from_i64(0), Expr::from_i64(5));
        assert_eq!(left_open.contains(&Expr::from_i64(0)), Some(false));
        assert_eq!(left_open.contains(&Expr::from_i64(5)), Some(true));

        let right_open = SymSet::interval_right_open(Expr::from_i64(0), Expr::from_i64(5));
        assert_eq!(right_open.contains(&Expr::from_i64(0)), Some(true));
        assert_eq!(right_open.contains(&Expr::from_i64(5)), Some(false));

        // Symmetric difference of {1, 2, 3} and {2, 3, 4} is {1, 4}
        let s1 = SymSet::finite(vec![
            Expr::from_i64(1),
            Expr::from_i64(2),
            Expr::from_i64(3),
        ]);
        let s2 = SymSet::finite(vec![
            Expr::from_i64(2),
            Expr::from_i64(3),
            Expr::from_i64(4),
        ]);
        let sym_diff = s1.symmetric_difference(s2);
        assert_eq!(sym_diff.contains(&Expr::from_i64(1)), Some(true));
        assert_eq!(sym_diff.contains(&Expr::from_i64(4)), Some(true));
        assert_eq!(sym_diff.contains(&Expr::from_i64(2)), Some(false));
        assert_eq!(sym_diff.contains(&Expr::from_i64(3)), Some(false));
    }

    #[test]
    fn test_is_empty_set() {
        assert_eq!(SymSet::EmptySet.is_empty_set(), Some(true));
        assert_eq!(SymSet::UniversalSet.is_empty_set(), Some(false));
        assert_eq!(
            SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(2)]).is_empty_set(),
            Some(false)
        );

        let valid_iv = SymSet::interval_closed(Expr::from_i64(1), Expr::from_i64(5));
        assert_eq!(valid_iv.is_empty_set(), Some(false));

        let inverted_iv = SymSet::interval_closed(Expr::from_i64(5), Expr::from_i64(1));
        assert_eq!(inverted_iv.is_empty_set(), Some(true));

        let open_point = SymSet::interval_open(Expr::from_i64(2), Expr::from_i64(2));
        assert_eq!(open_point.is_empty_set(), Some(true));

        let closed_point = SymSet::interval_closed(Expr::from_i64(2), Expr::from_i64(2));
        assert_eq!(closed_point.is_empty_set(), Some(false));
    }

    #[test]
    fn test_subset_and_superset_relations() {
        let empty = SymSet::EmptySet;
        let univ = SymSet::UniversalSet;
        let s12 = SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(2)]);
        let s123 = SymSet::finite(vec![
            Expr::from_i64(1),
            Expr::from_i64(2),
            Expr::from_i64(3),
        ]);

        assert_eq!(empty.is_subset(&s12), Some(true));
        assert_eq!(s12.is_subset(&univ), Some(true));
        assert_eq!(univ.is_subset(&empty), Some(false));

        assert_eq!(s12.is_subset(&s123), Some(true));
        assert_eq!(s123.is_subset(&s12), Some(false));
        assert_eq!(s123.is_superset(&s12), Some(true));

        let iv05 = SymSet::interval_closed(Expr::from_i64(0), Expr::from_i64(5));
        let iv13 = SymSet::interval_closed(Expr::from_i64(1), Expr::from_i64(3));
        let iv13_open = SymSet::interval_open(Expr::from_i64(1), Expr::from_i64(3));

        assert_eq!(iv13.is_subset(&iv05), Some(true));
        assert_eq!(iv05.is_subset(&iv13), Some(false));
        assert_eq!(iv13_open.is_subset(&iv13), Some(true));
        assert_eq!(iv13.is_subset(&iv13_open), Some(false));

        // Finite set in interval
        assert_eq!(s12.is_subset(&iv05), Some(true));
        let s_outside = SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(10)]);
        assert_eq!(s_outside.is_subset(&iv05), Some(false));
    }

    #[test]
    fn test_disjoint_relations() {
        let s12 = SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(2)]);
        let s34 = SymSet::finite(vec![Expr::from_i64(3), Expr::from_i64(4)]);
        let s23 = SymSet::finite(vec![Expr::from_i64(2), Expr::from_i64(3)]);

        assert_eq!(s12.is_disjoint(&s34), Some(true));
        assert_eq!(s12.is_disjoint(&s23), Some(false));

        let iv02 = SymSet::interval_closed(Expr::from_i64(0), Expr::from_i64(2));
        let iv35 = SymSet::interval_closed(Expr::from_i64(3), Expr::from_i64(5));
        let iv25_open = SymSet::interval_open(Expr::from_i64(2), Expr::from_i64(5));
        let iv25_closed = SymSet::interval_closed(Expr::from_i64(2), Expr::from_i64(5));

        assert_eq!(iv02.is_disjoint(&iv35), Some(true));
        assert_eq!(iv02.is_disjoint(&iv25_open), Some(true));
        assert_eq!(iv02.is_disjoint(&iv25_closed), Some(false));
    }

    #[test]
    fn test_set_measure() {
        assert_eq!(SymSet::EmptySet.measure(), Some(Expr::from_i64(0)));
        assert_eq!(
            SymSet::UniversalSet.measure(),
            Some(Expr::Const(Constant::Infinity))
        );
        let finite = SymSet::finite(vec![Expr::from_i64(1), Expr::from_i64(2)]);
        assert_eq!(finite.measure(), Some(Expr::from_i64(0)));

        let iv15 = SymSet::interval_closed(Expr::from_i64(1), Expr::from_i64(5));
        assert_eq!(iv15.measure(), Some(Expr::from_i64(4)));

        let iv_half =
            SymSet::interval_open(Expr::rational(1, 2).unwrap(), Expr::rational(7, 2).unwrap());
        assert_eq!(iv_half.measure(), Some(Expr::from_i64(3)));
    }
}
