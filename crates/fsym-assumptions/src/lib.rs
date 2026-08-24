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
use std::sync::Arc;
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
    facts: HashMap<Symbol, Vec<Predicate>>,
    domains: HashMap<Symbol, Domain>,
}

/// An immutable, thread-safe snapshot of an [`AssumptionsContext`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmutableAssumptionsSnapshot {
    inner: Arc<AssumptionsContext>,
}

impl ImmutableAssumptionsSnapshot {
    /// Evaluates a predicate query against the immutable snapshot in 4-valued logic.
    pub fn query(&self, expr: &Expr, pred: Predicate) -> TruthValue {
        self.inner.query(expr, pred)
    }

    /// Check if predicate holds for an expression under the immutable snapshot.
    pub fn is_true(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        self.inner.is_true(expr, pred)
    }

    /// Retrieves the domain assigned to a symbol under this snapshot.
    pub fn domain_of(&self, sym: &Symbol) -> Option<&Domain> {
        self.inner.domain_of(sym)
    }
}

impl AssumptionsContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an immutable, thread-safe snapshot of this context.
    pub fn snapshot(&self) -> ImmutableAssumptionsSnapshot {
        ImmutableAssumptionsSnapshot {
            inner: Arc::new(self.clone()),
        }
    }

    /// Records a predicate assumption for a symbol.
    pub fn assume(&mut self, sym: Symbol, pred: Predicate) {
        self.facts.entry(sym).or_default().push(pred);
    }

    /// Records an exact domain assignment for a symbol, emitting [`AssumptionError::DomainConflict`]
    /// if an incompatible domain is already registered.
    pub fn assume_domain(&mut self, sym: Symbol, domain: Domain) -> Result<(), AssumptionError> {
        if let Some(existing) = self.domains.get(&sym) {
            if existing != &domain {
                if let Some(common) = common_domain(existing, &domain) {
                    if common == Domain::ExpressionDomain && *existing != Domain::ExpressionDomain {
                        return Err(AssumptionError::DomainConflict(
                            sym.name,
                            existing.to_string(),
                            domain.to_string(),
                        ));
                    }
                    self.domains.insert(sym, common);
                    return Ok(());
                } else {
                    return Err(AssumptionError::DomainConflict(
                        sym.name,
                        existing.to_string(),
                        domain.to_string(),
                    ));
                }
            }
        } else {
            self.domains.insert(sym, domain);
        }
        Ok(())
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
        if let Expr::Sym(s) = expr
            && let Some(preds) = self.facts.get(s)
        {
            for (i, a) in preds.iter().enumerate() {
                for b in &preds[i + 1..] {
                    if Predicate::contradicts(*a, *b) {
                        return TruthValue::Contradictory;
                    }
                }
            }
        }

        let known = inherent_facts(expr)
            .or_else(|| match expr {
                Expr::Sym(s) => Some(self.deductions(s)),
                _ => None,
            })
            .unwrap_or_default();

        let has_pred = known.contains(&pred);
        let has_contradiction = known.iter().any(|fact| Predicate::contradicts(*fact, pred));

        if has_pred && has_contradiction {
            TruthValue::Contradictory
        } else if has_pred {
            TruthValue::EntailedTrue
        } else if has_contradiction {
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
        ctx.assume_domain(n.clone(), Domain::ZZ).unwrap();
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
    fn test_domain_conflict_detection() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume_domain(x.clone(), Domain::FiniteField { characteristic: 5 })
            .unwrap();
        let conflict = ctx.assume_domain(x.clone(), Domain::ZZ);
        assert!(matches!(conflict, Err(AssumptionError::DomainConflict(..))));
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

        // 4-way query reports Contradictory
        assert_eq!(
            ctx.query(&Expr::Sym(x.clone()), Predicate::Positive),
            TruthValue::Contradictory
        );

        let mut zero_ctx = AssumptionsContext::new();
        zero_ctx.assume(x.clone(), Predicate::Zero);
        zero_ctx.assume(x.clone(), Predicate::NonZero);
        assert_eq!(
            zero_ctx.check_consistency(),
            Err(AssumptionError::Contradiction)
        );
        assert_eq!(
            zero_ctx.query(&Expr::Sym(x), Predicate::Zero),
            TruthValue::Contradictory
        );
    }

    #[test]
    fn immutable_snapshot_preserves_query_behavior() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume(x.clone(), Predicate::Positive);

        let snap = ctx.snapshot();
        assert_eq!(
            snap.query(&Expr::Sym(x), Predicate::Positive),
            TruthValue::EntailedTrue
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
