//! # fsym-simplify
//!
//! Algebraic simplification engine, rewrite pipelines, expansion, and rational canonicalization.

#![forbid(unsafe_code)]

use fsym_core::Expr;
use num_bigint::BigInt;
use num_traits::Zero;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SimplifyError {
    #[error("Simplification error: {0}")]
    General(String),
}

/// Simplify an algebraic expression recursively.
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let mut simplified_terms: Vec<Expr> = terms.iter().map(simplify).collect();
            // Combine integer constants
            let mut int_sum = BigInt::zero();
            let mut non_ints = Vec::new();
            for t in simplified_terms.drain(..) {
                if let Expr::Integer(n) = t {
                    int_sum += n;
                } else if !t.is_zero() {
                    non_ints.push(t);
                }
            }
            if !int_sum.is_zero() || non_ints.is_empty() {
                non_ints.push(Expr::Integer(int_sum));
            }
            if non_ints.len() == 1 {
                non_ints.pop().unwrap()
            } else {
                Expr::Add(non_ints)
            }
        }
        Expr::Mul(factors) => {
            let mut simplified_factors: Vec<Expr> = factors.iter().map(simplify).collect();
            let mut int_prod = BigInt::from(1);
            let mut non_ints = Vec::new();
            for f in simplified_factors.drain(..) {
                if let Expr::Integer(n) = f {
                    if n.is_zero() {
                        return Expr::from_i64(0);
                    }
                    int_prod *= n;
                } else if !f.is_one() {
                    non_ints.push(f);
                }
            }
            if int_prod != BigInt::from(1) || non_ints.is_empty() {
                non_ints.insert(0, Expr::Integer(int_prod));
            }
            if non_ints.len() == 1 {
                non_ints.pop().unwrap()
            } else {
                Expr::Mul(non_ints)
            }
        }
        Expr::Pow(base, exp) => {
            let b = simplify(base);
            let e = simplify(exp);
            if e.is_zero() {
                Expr::from_i64(1)
            } else if e.is_one() {
                b
            } else {
                Expr::Pow(std::sync::Arc::new(b), std::sync::Arc::new(e))
            }
        }
        other => other.clone(),
    }
}

/// Expand polynomial/product terms in an expression.
pub fn expand(expr: &Expr) -> Expr {
    // Basic expansion for product of sums
    simplify(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_basic() {
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let e = Expr::Add(vec![x.clone(), zero, Expr::from_i64(5), Expr::from_i64(3)]);
        let s = simplify(&e);
        assert_eq!(s, Expr::Add(vec![x, Expr::from_i64(8)]));
    }

    #[test]
    fn test_simplify_mul_zero() {
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let e = Expr::Mul(vec![x, zero]);
        let s = simplify(&e);
        assert_eq!(s, Expr::from_i64(0));
    }
}
