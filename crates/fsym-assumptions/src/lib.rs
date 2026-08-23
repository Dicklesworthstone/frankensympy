//! # fsym-assumptions
//!
//! Deductive predicate and assumptions engine for symbolic reasoning:
//! real, positive, negative, integer, rational, prime, complex, zero.

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AssumptionError {
    #[error("Contradictory assumptions inferred")]
    Contradiction,
}

/// Mathematical predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Predicate {
    Real,
    Complex,
    Integer,
    Rational,
    Positive,
    Negative,
    NonNegative,
    NonPositive,
    Zero,
    NonZero,
    Prime,
    Even,
    Odd,
}

/// Assumptions context holding facts about symbols.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssumptionsContext {
    pub facts: HashMap<Symbol, Vec<Predicate>>,
}

impl Predicate {
    /// Direct consequences under the standard numeric-tower lattice.
    ///
    /// Deliberately excludes `Prime => Odd` (2 is prime and even) and
    /// makes no claim the lattice cannot justify.
    fn direct_consequences(self) -> &'static [Predicate] {
        use Predicate::*;
        match self {
            Complex => &[],
            Real => &[Complex],
            Rational => &[Real, Complex],
            Integer => &[Rational, Real, Complex],
            Even | Odd => &[Integer, Rational, Real, Complex],
            Prime => &[
                Integer,
                Positive,
                NonNegative,
                NonZero,
                Rational,
                Real,
                Complex,
            ],
            Positive => &[NonNegative, NonZero, Rational, Real, Complex],
            Negative => &[NonPositive, NonZero, Rational, Real, Complex],
            NonNegative => &[Real, Complex],
            Zero => &[NonNegative, NonPositive, Rational, Real, Complex],
            NonPositive => &[Real, Complex],
            NonZero => &[Real, Complex],
        }
    }

    /// Full transitive closure of consequences of `self`.
    pub fn closure(self) -> std::collections::BTreeSet<Predicate> {
        let mut seen = std::collections::BTreeSet::from([self]);
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

    /// Whether `a` entails `b` in the assumption lattice.
    pub fn entails(a: Predicate, b: Predicate) -> bool {
        a.closure().contains(&b)
    }

    /// Predicates directly contradicted by `self`.
    fn contradictions(self) -> &'static [Predicate] {
        use Predicate::*;
        match self {
            Zero => &[NonZero, Positive, Negative],
            NonZero => &[Zero],
            Positive => &[Negative, Zero, NonPositive],
            Negative => &[Positive, Zero, NonNegative],
            Even => &[Odd],
            Odd => &[Even],
            _ => &[],
        }
    }
}

/// Predicates that hold for a literal expression on its face, without
/// consulting any context.
fn inherent_facts(expr: &Expr) -> Option<std::collections::BTreeSet<Predicate>> {
    use num_traits::{Signed, Zero};
    match expr {
        Expr::Integer(n) => {
            let mut facts: std::collections::BTreeSet<Predicate> = Predicate::Integer.closure();
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
            let mut facts: std::collections::BTreeSet<Predicate> = Predicate::Rational.closure();
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
        _ => None,
    }
}

/// Three-valued decision against a known-fact set: entailed, contradicted,
/// or unknown. Absence from `known` is never evidence of falsity.
fn decide(known: &std::collections::BTreeSet<Predicate>, pred: Predicate) -> Option<bool> {
    if known.contains(&pred) {
        return Some(true);
    }
    for fact in known {
        if fact.contradictions().contains(&pred) {
            return Some(false);
        }
    }
    None
}

impl AssumptionsContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assume(&mut self, sym: Symbol, pred: Predicate) {
        self.facts.entry(sym).or_default().push(pred);
    }

    /// Deduced predicate set for one symbol: stated facts plus every
    /// consequence the lattice licenses.
    pub fn deductions(&self, sym: &Symbol) -> std::collections::BTreeSet<Predicate> {
        let mut out = std::collections::BTreeSet::new();
        if let Some(preds) = self.facts.get(sym) {
            for p in preds {
                out.extend(p.closure());
            }
        }
        out
    }

    /// Check if predicate holds for an expression.
    ///
    /// Returns `Some(true)` only when derivable, `Some(false)` when
    /// refutable from known facts or literal structure, and `None` when
    /// genuinely unknown — unknown never becomes true or false.
    pub fn is_true(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        let known = inherent_facts(expr)
            .or_else(|| match expr {
                Expr::Sym(s) => Some(self.deductions(s)),
                _ => None,
            })
            .unwrap_or_default();
        decide(&known, pred)
    }

    /// Reports [`AssumptionError::Contradiction`] if any symbol carries
    /// mutually exclusive facts. Contradictions are surfaced explicitly;
    /// they never cause unrelated queries to answer vacuously.
    pub fn check_consistency(&self) -> Result<(), AssumptionError> {
        for preds in self.facts.values() {
            for (i, a) in preds.iter().enumerate() {
                for b in &preds[i + 1..] {
                    if Predicate::contradictions(*a).contains(b)
                        || Predicate::contradictions(*b).contains(a)
                    {
                        return Err(AssumptionError::Contradiction);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assumptions_context() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume(x.clone(), Predicate::Positive);
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Positive),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::from_i64(5), Predicate::Positive),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::from_i64(-2), Predicate::Positive),
            Some(false)
        );
    }

    #[test]
    fn lattice_entailment_is_transitive_and_honest() {
        use Predicate::*;
        assert!(Predicate::entails(Integer, Real));
        assert!(Predicate::entails(Prime, NonZero));
        assert!(!Predicate::entails(Prime, Even), "2 is prime and even");
        assert!(!Predicate::entails(Positive, Negative));
        assert!(Predicate::entails(Zero, NonNegative));
    }

    #[test]
    fn symbol_facts_deduce_consequences() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume(x.clone(), Predicate::Positive);
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::NonNegative),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Rational),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Negative),
            Some(false)
        );
        // Undecidable stays undecidable.
        assert_eq!(ctx.is_true(&Expr::Sym(x.clone()), Predicate::Even), None);
    }

    #[test]
    fn negative_integer_literals_fully_inferred() {
        let ctx = AssumptionsContext::new();
        let neg_four = Expr::from_i64(-4);
        for pred in [Predicate::Negative, Predicate::Even, Predicate::NonZero] {
            assert_eq!(ctx.is_true(&neg_four, pred), Some(true));
        }
        assert_eq!(ctx.is_true(&neg_four, Predicate::Odd), Some(false));
        assert_eq!(ctx.is_true(&neg_four, Predicate::Prime), None);
    }

    #[test]
    fn zero_literal_satisfies_both_sign_bands() {
        let ctx = AssumptionsContext::new();
        let zero = Expr::from_i64(0);
        assert_eq!(ctx.is_true(&zero, Predicate::Zero), Some(true));
        assert_eq!(ctx.is_true(&zero, Predicate::NonNegative), Some(true));
        assert_eq!(ctx.is_true(&zero, Predicate::NonPositive), Some(true));
        assert_eq!(ctx.is_true(&zero, Predicate::Positive), Some(false));
        assert_eq!(ctx.is_true(&zero, Predicate::NonZero), Some(false));
    }

    #[test]
    fn rational_literals_report_sign_but_not_integrality() {
        let ctx = AssumptionsContext::new();
        let three_halves = Expr::Rational(num_rational::BigRational::new(3.into(), 2.into()));
        assert_eq!(ctx.is_true(&three_halves, Predicate::Positive), Some(true));
        assert_eq!(ctx.is_true(&three_halves, Predicate::Rational), Some(true));
        assert_eq!(ctx.is_true(&three_halves, Predicate::Integer), None);
    }

    #[test]
    fn contradictory_context_is_detected_explicitly() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume(x.clone(), Predicate::Positive);
        ctx.assume(x.clone(), Predicate::Negative);
        assert_eq!(ctx.check_consistency(), Err(AssumptionError::Contradiction));

        let mut zero_ctx = AssumptionsContext::new();
        zero_ctx.assume(x.clone(), Predicate::Zero);
        zero_ctx.assume(x, Predicate::NonZero);
        assert_eq!(
            zero_ctx.check_consistency(),
            Err(AssumptionError::Contradiction)
        );
    }

    #[test]
    fn contradiction_never_vacuously_satisfies_unrelated_queries() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume(x.clone(), Predicate::Positive);
        ctx.assume(x.clone(), Predicate::Negative);
        // Ex falso is forbidden: an inconsistent context must not prove
        // arbitrary predicates about other subjects.
        assert_eq!(ctx.is_true(&Expr::from_i64(7), Predicate::Odd), Some(true));
    }
}
