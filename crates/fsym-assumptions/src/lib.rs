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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl AssumptionsContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assume(&mut self, sym: Symbol, pred: Predicate) {
        self.facts.entry(sym).or_default().push(pred);
    }

    /// Check if predicate holds for expression.
    pub fn is_true(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        match (expr, pred) {
            (Expr::Integer(_n), Predicate::Integer) => Some(true),
            (Expr::Integer(_n), Predicate::Rational) => Some(true),
            (Expr::Integer(_n), Predicate::Real) => Some(true),
            (Expr::Integer(n), Predicate::Positive) => Some(n > &num_bigint::BigInt::from(0)),
            (Expr::Integer(n), Predicate::Zero) => Some(n == &num_bigint::BigInt::from(0)),
            (Expr::Rational(_), Predicate::Rational) => Some(true),
            (Expr::Rational(_), Predicate::Real) => Some(true),
            (Expr::Sym(s), p) => {
                if let Some(preds) = self.facts.get(s)
                    && preds.contains(&p)
                {
                    return Some(true);
                }
                None
            }
            _ => None,
        }
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
}
