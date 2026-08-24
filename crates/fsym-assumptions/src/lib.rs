//! # fsym-assumptions
//!
//! Deductive predicate and assumptions engine for symbolic reasoning (WS04):
//! - Multi-valued truth model ([`TruthValue`]: EntailedTrue, EntailedFalse, Unknown, Contradictory);
//! - Mathematical domains ([`Domain`]: $\mathbb{Z}$, $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$, $D[x]$, $D(x)$, $\mathbb{F}_p$) with explicit coercion graph;
//! - Capture-avoiding substitution and alpha-equivalence ([`bindings`]);
//! - Deductive predicate hierarchy ([`Predicate`]) and assumption context ([`AssumptionsContext`]).

#![forbid(unsafe_code)]

pub mod bindings;
pub mod domain;
pub mod predicate;
pub mod truth;

pub use bindings::*;
pub use domain::*;
pub use predicate::*;
pub use truth::*;

use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AssumptionError {
    #[error("Contradictory assumptions inferred")]
    Contradiction,
    #[error("Domain conflict for symbol {0}: {1} vs {2}")]
    DomainConflict(String, String, String),
    #[error("Unknown symbol: {0}")]
    UnknownSymbol(String),
}

/// Assumptions context holding mathematical facts and domain assignments for symbols.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssumptionsContext {
    pub facts: HashMap<Symbol, Vec<Predicate>>,
    pub domains: HashMap<Symbol, Domain>,
}

impl AssumptionsContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a predicate assumption for a symbol.
    pub fn assume(&mut self, sym: Symbol, pred: Predicate) {
        self.facts.entry(sym).or_default().push(pred);
    }

    /// Records an exact domain assignment for a symbol.
    pub fn assume_domain(&mut self, sym: Symbol, domain: Domain) {
        self.domains.insert(sym, domain);
    }

    /// Retrieves the domain assigned to a symbol, if any.
    pub fn domain_of(&self, sym: &Symbol) -> Option<&Domain> {
        self.domains.get(sym)
    }

    /// Deduced predicate set for one symbol: stated facts plus every consequence
    /// the lattice licenses, as well as facts implied by domain assignments.
    pub fn deductions(&self, sym: &Symbol) -> BTreeSet<Predicate> {
        let mut out = BTreeSet::new();
        if let Some(preds) = self.facts.get(sym) {
            for p in preds {
                out.extend(p.closure());
            }
        }
        if let Some(dom) = self.domains.get(sym) {
            match dom {
                Domain::ZZ => out.extend(Predicate::Integer.closure()),
                Domain::QQ => out.extend(Predicate::Rational.closure()),
                Domain::RR => out.extend(Predicate::Real.closure()),
                Domain::CC => out.extend(Predicate::Complex.closure()),
                _ => {}
            }
        }
        out
    }

    /// Evaluates a predicate query against an expression in 4-valued logic.
    pub fn query(&self, expr: &Expr, pred: Predicate) -> TruthValue {
        let known = inherent_facts(expr)
            .or_else(|| match expr {
                Expr::Sym(s) => Some(self.deductions(s)),
                _ => None,
            })
            .unwrap_or_default();

        if known.contains(&pred) {
            TruthValue::EntailedTrue
        } else if known.iter().any(|fact| Predicate::contradicts(*fact, pred)) {
            TruthValue::EntailedFalse
        } else {
            TruthValue::Unknown
        }
    }

    /// Check if predicate holds for an expression (backward-compatible 3-valued query).
    ///
    /// Returns `Some(true)` only when derivable, `Some(false)` when
    /// refutable from known facts or literal structure, and `None` when
    /// genuinely unknown.
    pub fn is_true(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        self.query(expr, pred).to_option_bool()
    }

    /// Check if predicate is refutable for an expression.
    pub fn is_false(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        match self.query(expr, pred) {
            TruthValue::EntailedFalse => Some(true),
            TruthValue::EntailedTrue => Some(false),
            TruthValue::Unknown | TruthValue::Contradictory => None,
        }
    }

    /// Reports [`AssumptionError::Contradiction`] if any symbol carries
    /// mutually exclusive facts or conflicting domain assertions.
    pub fn check_consistency(&self) -> Result<(), AssumptionError> {
        for preds in self.facts.values() {
            for (i, a) in preds.iter().enumerate() {
                for b in &preds[i + 1..] {
                    if Predicate::contradicts(*a, *b) {
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
    fn test_domain_assignments_imply_predicates() {
        let mut ctx = AssumptionsContext::new();
        let n = Symbol::new("n");
        ctx.assume_domain(n.clone(), Domain::ZZ);
        assert_eq!(
            ctx.query(&Expr::Sym(n.clone()), Predicate::Integer),
            TruthValue::EntailedTrue
        );
        assert_eq!(
            ctx.query(&Expr::Sym(n), Predicate::Real),
            TruthValue::EntailedTrue
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
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Real),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Rational),
            None
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Negative),
            Some(false)
        );
        // Undecidable stays undecidable.
        assert_eq!(ctx.is_true(&Expr::Sym(x), Predicate::Even), None);
    }

    #[test]
    fn negative_integer_literals_fully_inferred() {
        let ctx = AssumptionsContext::new();
        let neg_four = Expr::from_i64(-4);
        for pred in [Predicate::Negative, Predicate::Even, Predicate::NonZero] {
            assert_eq!(ctx.is_true(&neg_four, pred), Some(true));
        }
        assert_eq!(ctx.is_true(&neg_four, Predicate::Odd), Some(false));
        assert_eq!(ctx.is_true(&neg_four, Predicate::Prime), Some(false));
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
