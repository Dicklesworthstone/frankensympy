//! # fsym-logic
//!
//! Boolean algebra, truth tables, normal forms (CNF, DNF), and SAT solving.

#![forbid(unsafe_code)]

use fsym_core::Symbol;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LogicError {
    #[error("Variable not found in truth assignment: {0}")]
    UnassignedVariable(String),
    #[error("Truth table exceeds supported variable count: {0} > 20")]
    TableTooLarge(usize),
}

/// Propositional logic formula.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoolExpr {
    /// Boolean constant: True or False.
    Const(bool),
    /// Boolean variable symbol.
    Var(Symbol),
    /// Logical NOT (¬A).
    Not(Box<BoolExpr>),
    /// Logical AND (A ∧ B ∧ ...).
    And(Vec<BoolExpr>),
    /// Logical OR (A ∨ B ∨ ...).
    Or(Vec<BoolExpr>),
    /// Logical Implication (A → B).
    Implies(Box<BoolExpr>, Box<BoolExpr>),
    /// Logical Equivalence (A ↔ B).
    Equivalent(Box<BoolExpr>, Box<BoolExpr>),
}

impl BoolExpr {
    pub fn var(name: impl Into<String>) -> Self {
        BoolExpr::Var(Symbol::new(name))
    }

    pub fn and(self, other: BoolExpr) -> Self {
        match (self, other) {
            (BoolExpr::Const(false), _) | (_, BoolExpr::Const(false)) => BoolExpr::Const(false),
            (BoolExpr::Const(true), b) | (b, BoolExpr::Const(true)) => b,
            (BoolExpr::And(mut a), BoolExpr::And(b)) => {
                a.extend(b);
                BoolExpr::And(a)
            }
            (BoolExpr::And(mut a), b) | (b, BoolExpr::And(mut a)) => {
                a.push(b);
                BoolExpr::And(a)
            }
            (a, b) => BoolExpr::And(vec![a, b]),
        }
    }

    pub fn or(self, other: BoolExpr) -> Self {
        match (self, other) {
            (BoolExpr::Const(true), _) | (_, BoolExpr::Const(true)) => BoolExpr::Const(true),
            (BoolExpr::Const(false), b) | (b, BoolExpr::Const(false)) => b,
            (BoolExpr::Or(mut a), BoolExpr::Or(b)) => {
                a.extend(b);
                BoolExpr::Or(a)
            }
            (BoolExpr::Or(mut a), b) | (b, BoolExpr::Or(mut a)) => {
                a.push(b);
                BoolExpr::Or(a)
            }
            (a, b) => BoolExpr::Or(vec![a, b]),
        }
    }

    /// Evaluate expression under variable assignment.
    pub fn evaluate(&self, env: &HashMap<Symbol, bool>) -> Result<bool, LogicError> {
        match self {
            BoolExpr::Const(b) => Ok(*b),
            BoolExpr::Var(s) => env
                .get(s)
                .copied()
                .ok_or_else(|| LogicError::UnassignedVariable(s.name.clone())),
            BoolExpr::Not(inner) => Ok(!inner.evaluate(env)?),
            BoolExpr::And(terms) => {
                for t in terms {
                    if !t.evaluate(env)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            BoolExpr::Or(terms) => {
                for t in terms {
                    if t.evaluate(env)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            BoolExpr::Implies(a, b) => Ok(!a.evaluate(env)? || b.evaluate(env)?),
            BoolExpr::Equivalent(a, b) => Ok(a.evaluate(env)? == b.evaluate(env)?),
        }
    }

    /// Collect all boolean variables in the expression.
    pub fn variables(&self) -> Vec<Symbol> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_vars(&self, acc: &mut Vec<Symbol>) {
        match self {
            BoolExpr::Const(_) => {}
            BoolExpr::Var(s) => acc.push(s.clone()),
            BoolExpr::Not(inner) => inner.collect_vars(acc),
            BoolExpr::And(terms) | BoolExpr::Or(terms) => {
                for t in terms {
                    t.collect_vars(acc);
                }
            }
            BoolExpr::Implies(a, b) | BoolExpr::Equivalent(a, b) => {
                a.collect_vars(acc);
                b.collect_vars(acc);
            }
        }
    }
}

impl std::ops::Not for BoolExpr {
    type Output = BoolExpr;

    fn not(self) -> BoolExpr {
        match self {
            BoolExpr::Const(b) => BoolExpr::Const(!b),
            BoolExpr::Not(inner) => *inner,
            other => BoolExpr::Not(Box::new(other)),
        }
    }
}

impl fmt::Display for BoolExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolExpr::Const(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            BoolExpr::Var(s) => write!(f, "{}", s),
            BoolExpr::Not(inner) => write!(f, "~{}", inner),
            BoolExpr::And(terms) => {
                let s = terms
                    .iter()
                    .map(|t| format!("{}", t))
                    .collect::<Vec<_>>()
                    .join(" & ");
                write!(f, "({})", s)
            }
            BoolExpr::Or(terms) => {
                let s = terms
                    .iter()
                    .map(|t| format!("{}", t))
                    .collect::<Vec<_>>()
                    .join(" | ");
                write!(f, "({})", s)
            }
            BoolExpr::Implies(a, b) => write!(f, "({} >> {})", a, b),
            BoolExpr::Equivalent(a, b) => write!(f, "({} <=> {})", a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logic_eval() {
        let p = Symbol::new("p");
        let q = Symbol::new("q");
        let expr = BoolExpr::var("p").and(!BoolExpr::var("q"));
        let mut env = HashMap::new();
        env.insert(p.clone(), true);
        env.insert(q.clone(), false);
        assert_eq!(expr.evaluate(&env), Ok(true));
    }
}
