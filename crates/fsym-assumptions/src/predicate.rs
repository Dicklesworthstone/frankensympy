//! Mathematical predicates and inference lattice for WS04 assumptions reasoning.
//!
//! Deductive predicate hierarchy covering numeric towers, sign bands, primality,
//! parity, algebraicity, and finiteness.

use fsym_core::{Constant, Expr};
use num_traits::{Signed, Zero};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Mathematical predicates governing symbols and expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Predicate {
    // Numeric tower
    Complex,
    Real,
    Rational,
    Integer,
    Algebraic,
    Transcendental,

    // Sign bands
    Positive,
    Negative,
    NonNegative,
    NonPositive,
    Zero,
    NonZero,

    // Number-theoretic
    Prime,
    Even,
    Odd,

    // Boundedness
    Finite,
    Infinite,
}

impl Predicate {
    /// Direct deductive consequences in the numeric lattice.
    ///
    /// Excludes `Prime => Odd` because 2 is prime and even.
    pub fn direct_consequences(self) -> &'static [Predicate] {
        use Predicate::*;
        match self {
            Complex => &[],
            Real => &[Complex],
            Rational => &[Real, Complex, Algebraic, Finite],
            Integer => &[Rational, Real, Complex, Algebraic, Finite],
            Algebraic => &[Complex],
            Transcendental => &[Complex],
            Even => &[Integer, Rational, Real, Complex, Algebraic, Finite],
            Odd => &[Integer, Rational, Real, Complex, Algebraic, Finite],
            Prime => &[
                Integer,
                Positive,
                NonNegative,
                NonZero,
                Rational,
                Real,
                Complex,
                Algebraic,
                Finite,
            ],
            Positive => &[NonNegative, NonZero, Real, Complex],
            Negative => &[NonPositive, NonZero, Real, Complex],
            NonNegative => &[Real, Complex],
            NonPositive => &[Real, Complex],
            Zero => &[
                NonNegative,
                NonPositive,
                Rational,
                Real,
                Complex,
                Algebraic,
                Finite,
                Even,
            ],
            NonZero => &[Complex],
            Finite => &[Complex],
            Infinite => &[Complex],
        }
    }

    /// Full reflexive-transitive closure of consequences licensed by this predicate.
    pub fn closure(self) -> BTreeSet<Predicate> {
        let mut seen = BTreeSet::from([self]);
        let mut queue = vec![self];
        while let Some(current) = queue.pop() {
            for next in current.direct_consequences() {
                if seen.insert(*next) {
                    queue.push(*next);
                }
            }
        }
        seen
    }

    /// Whether predicate `a` logically entails predicate `b`.
    pub fn entails(a: Predicate, b: Predicate) -> bool {
        a.closure().contains(&b)
    }

    /// Predicates directly contradicted by `self`.
    pub fn direct_contradictions(self) -> &'static [Predicate] {
        use Predicate::*;
        match self {
            Zero => &[NonZero, Positive, Negative, Infinite],
            NonZero => &[Zero],
            Positive => &[Negative, Zero, NonPositive],
            Negative => &[Positive, Zero, NonNegative],
            Even => &[Odd],
            Odd => &[Even],
            Algebraic => &[Transcendental],
            Transcendental => &[Algebraic, Rational, Integer, Prime, Even, Odd],
            Finite => &[Infinite],
            Infinite => &[Finite, Zero, Integer, Rational, Prime],
            _ => &[],
        }
    }

    /// Whether predicate `a` directly or transitively contradicts predicate `b`.
    pub fn contradicts(a: Predicate, b: Predicate) -> bool {
        let a_closure = a.closure();
        let b_closure = b.closure();

        for x in &a_closure {
            for c in x.direct_contradictions() {
                if b_closure.contains(c) {
                    return true;
                }
            }
        }
        false
    }
}

/// Computes the inherent facts known from an exact expression leaf.
pub fn inherent_facts(expr: &Expr) -> Option<BTreeSet<Predicate>> {
    match expr {
        Expr::Integer(n) => {
            let mut facts = Predicate::Integer.closure();
            if n.is_zero() {
                facts.extend(Predicate::Zero.closure());
            } else {
                facts.extend(Predicate::NonZero.closure());
                facts.extend(if n > &num_bigint::BigInt::from(0) {
                    Predicate::Positive.closure()
                } else {
                    Predicate::Negative.closure()
                });
                let two = num_bigint::BigInt::from(2);
                let even = (n % &two).is_zero();
                facts.extend(
                    if even {
                        Predicate::Even
                    } else {
                        Predicate::Odd
                    }
                    .closure(),
                );
            }
            Some(facts)
        }
        Expr::Rational(r) => {
            let mut facts = Predicate::Rational.closure();
            if r.is_zero() {
                facts.extend(Predicate::Zero.closure());
            } else {
                facts.extend(Predicate::NonZero.closure());
                facts.extend(if r.is_positive() {
                    Predicate::Positive.closure()
                } else {
                    Predicate::Negative.closure()
                });
            }
            Some(facts)
        }
        Expr::Const(Constant::Pi) | Expr::Const(Constant::E) => {
            let mut facts = Predicate::Transcendental.closure();
            facts.extend(Predicate::Real.closure());
            facts.extend(Predicate::Positive.closure());
            facts.extend(Predicate::Finite.closure());
            Some(facts)
        }
        Expr::Const(Constant::I) => {
            let mut facts = Predicate::Algebraic.closure();
            facts.extend(Predicate::Complex.closure());
            facts.extend(Predicate::NonZero.closure());
            facts.extend(Predicate::Finite.closure());
            Some(facts)
        }
        Expr::Const(Constant::Infinity) => {
            let mut facts = Predicate::Infinite.closure();
            facts.extend(Predicate::Positive.closure());
            Some(facts)
        }
        Expr::Const(Constant::NegativeInfinity) => {
            let mut facts = Predicate::Infinite.closure();
            facts.extend(Predicate::Negative.closure());
            Some(facts)
        }
        Expr::Const(Constant::ComplexInfinity) => {
            let facts = Predicate::Infinite.closure();
            Some(facts)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_entailment_soundness() {
        use Predicate::*;
        assert!(Predicate::entails(Integer, Real));
        assert!(Predicate::entails(Integer, Rational));
        assert!(Predicate::entails(Integer, Algebraic));
        assert!(Predicate::entails(Rational, Finite));
        assert!(Predicate::entails(Prime, NonZero));
        assert!(Predicate::entails(Prime, Positive));
        assert!(!Predicate::entails(Prime, Odd), "2 is prime and even");
        assert!(Predicate::entails(Zero, Even));
    }

    #[test]
    fn contradiction_symmetry() {
        use Predicate::*;
        assert!(Predicate::contradicts(Positive, Negative));
        assert!(Predicate::contradicts(Negative, Positive));
        assert!(Predicate::contradicts(Even, Odd));
        assert!(Predicate::contradicts(Odd, Even));
        assert!(Predicate::contradicts(Algebraic, Transcendental));
        assert!(Predicate::contradicts(Transcendental, Rational));
        assert!(Predicate::contradicts(Finite, Infinite));
        assert!(Predicate::contradicts(Infinite, Zero));
    }
}
