//! # fsym-calculus
//!
//! Symbolic differentiation, integration, limits, and series expansion.

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use fsym_simplify::simplify;
use num_bigint::BigInt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CalculusError {
    #[error("Cannot differentiate non-differentiable term: {0}")]
    NonDifferentiable(String),
    #[error("Integration not computable symbolically: {0}")]
    IntegrationFailed(String),
}

/// Compute the symbolic derivative of an expression with respect to a symbol: ∂expr / ∂var.
pub fn diff(expr: &Expr, var: &Symbol) -> Expr {
    let unsimplified = match expr {
        Expr::Sym(s) => {
            if s == var {
                Expr::from_i64(1)
            } else {
                Expr::from_i64(0)
            }
        }
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => Expr::from_i64(0),
        Expr::Add(terms) => {
            let diff_terms: Vec<Expr> = terms.iter().map(|t| diff(t, var)).collect();
            Expr::Add(diff_terms)
        }
        Expr::Mul(factors) => {
            // Product rule: d(f*g*h) = f'*g*h + f*g'*h + f*g*h'
            let mut add_terms = Vec::new();
            for i in 0..factors.len() {
                let mut prod_factors = Vec::new();
                for (j, factor) in factors.iter().enumerate() {
                    if i == j {
                        prod_factors.push(diff(factor, var));
                    } else {
                        prod_factors.push(factor.clone());
                    }
                }
                add_terms.push(Expr::Mul(prod_factors));
            }
            Expr::Add(add_terms)
        }
        Expr::Pow(base, exp) => {
            // d(u^v) = u^v * (v' * ln(u) + v * u' / u)
            // Special case when exponent is constant integer n: d(u^n) = n * u^(n-1) * u'
            if let Expr::Integer(n) = exp.as_ref() {
                let n_minus_1 = n - BigInt::from(1);
                let du = diff(base, var);
                Expr::Mul(vec![
                    Expr::Integer(n.clone()),
                    Expr::Pow(base.clone(), Arc::new(Expr::Integer(n_minus_1))),
                    du,
                ])
            } else {
                // General chain rule fallback
                Expr::Function(
                    "diff".to_string(),
                    vec![expr.clone(), Expr::Sym(var.clone())],
                )
            }
        }
        Expr::Function(name, args) => {
            // Elementary derivatives
            if name == "sin" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![Expr::Function("cos".to_string(), vec![u.clone()]), du])
            } else if name == "cos" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("sin".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "exp" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![Expr::Function("exp".to_string(), vec![u.clone()]), du])
            } else {
                Expr::Function(
                    "diff".to_string(),
                    vec![expr.clone(), Expr::Sym(var.clone())],
                )
            }
        }
    };
    simplify(&unsimplified)
}

/// Compute the N-th derivative: d^n(expr) / d(var)^n.
pub fn diff_n(expr: &Expr, var: &Symbol, n: usize) -> Expr {
    let mut current = expr.clone();
    for _ in 0..n {
        current = diff(&current, var);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_polynomial() {
        let x = Symbol::new("x");
        // f(x) = x^3 + 2*x + 5
        let expr = Expr::Add(vec![
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(3))),
            Expr::Mul(vec![Expr::from_i64(2), Expr::symbol("x")]),
            Expr::from_i64(5),
        ]);
        let d = diff(&expr, &x);
        // df/dx = 3*x^2 + 2
        let free = d.free_symbols();
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].name, "x");
    }

    #[test]
    fn test_diff_trig() {
        let x = Symbol::new("x");
        let expr = Expr::Function("sin".to_string(), vec![Expr::symbol("x")]);
        let d = diff(&expr, &x);
        assert_eq!(
            d,
            Expr::Function("cos".to_string(), vec![Expr::symbol("x")])
        );
    }
}
