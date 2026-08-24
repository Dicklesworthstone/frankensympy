//! # fsym-polys
//!
//! Polynomial algebra, polynomial rings, multivariate sparse representations,
//! polynomial identity testing (PIT), and verified identity certificates (WS08).

#![forbid(unsafe_code)]

pub mod identity;
pub mod multivariate;
pub mod univariate;

pub use identity::*;
pub use multivariate::*;
pub use univariate::*;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PolyError {
    #[error("Division by zero polynomial")]
    DivisionByZero,
    #[error("Incompatible polynomial ring generators: expected {0}, got {1}")]
    IncompatibleGenerators(String, String),
    #[error("Non-polynomial expression: {0}")]
    NonPolynomialExpression(String),
    #[error("Polynomial identity verification failed: {0}")]
    IdentityCheckFailed(String),
    #[error("General polynomial error: {0}")]
    General(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_assumptions::ImmutableAssumptionsSnapshot;
    use fsym_budget::Unbounded;
    use fsym_core::{BigInt, BigRational, Expr, Symbol};
    use fsym_proof_kernel::verify_derivation_independent;
    use std::sync::Arc;

    #[test]
    fn test_univariate_div_rem_identity() {
        let x = Symbol::new("x");
        // A(x) = 2*x^3 + 3*x^2 + x + 5
        let a = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(5)),
                BigRational::from_integer(BigInt::from(1)),
                BigRational::from_integer(BigInt::from(3)),
                BigRational::from_integer(BigInt::from(2)),
            ],
        );
        // B(x) = x + 1
        let b = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(1)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );

        let (q, r) = a.div_rem(&b).unwrap();
        // Check A = Q * B + R
        let q_times_b = q.mul(&b).unwrap();
        let reconstructed = q_times_b.add(&r).unwrap();
        assert_eq!(reconstructed, a);
    }

    #[test]
    fn test_multivariate_product_and_derivatives() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let gens = vec![x.clone(), y.clone()];

        // P(x, y) = x^2 + 2*x*y + y^2
        let expr = Expr::Add(vec![
            Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
            Expr::Mul(vec![
                Expr::from_i64(2),
                Expr::Sym(x.clone()),
                Expr::Sym(y.clone()),
            ]),
            Expr::Pow(Arc::new(Expr::Sym(y.clone())), Arc::new(Expr::from_i64(2))),
        ]);

        let poly = MultivariatePoly::from_expr(&expr, &gens).unwrap();
        assert_eq!(poly.total_degree(), Some(2));
        assert_eq!(poly.degree_in(0), 2);
        assert_eq!(poly.degree_in(1), 2);

        // ∂P/∂x = 2*x + 2*y
        let d_dx = poly.derivative(0).unwrap();
        assert_eq!(d_dx.total_degree(), Some(1));

        // Evaluate at x=3, y=4 -> (3+4)^2 = 49
        let eval_pt = vec![
            BigRational::from_integer(BigInt::from(3)),
            BigRational::from_integer(BigInt::from(4)),
        ];
        let val = poly.eval(&eval_pt).unwrap();
        assert_eq!(val, BigRational::from_integer(BigInt::from(49)));
    }

    #[test]
    fn test_schwartz_zippel_detects_non_identity_with_witness() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let gens = vec![x.clone(), y.clone()];

        // P = x * y - (x + y)  (not identically zero)
        let expr = Expr::Add(vec![
            Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
            Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(x.clone())]),
            Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(y.clone())]),
        ]);

        let poly = MultivariatePoly::from_expr(&expr, &gens).unwrap();
        let witness = schwartz_zippel_test(&poly, 10, 42).unwrap();
        assert!(
            witness.is_some(),
            "Schwartz-Zippel must find non-zero witness"
        );
        let w = witness.unwrap();
        assert_eq!(poly.eval(&w.point).unwrap(), w.evaluated_value);
    }

    #[test]
    fn test_verified_polynomial_identity_certificate() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let gens = vec![x.clone(), y.clone()];
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut meter = Unbounded;

        // (x + y)^2
        let lhs = Expr::Pow(
            Arc::new(Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())])),
            Arc::new(Expr::from_i64(2)),
        );

        // x^2 + 2*x*y + y^2
        let rhs = Expr::Add(vec![
            Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
            Expr::Mul(vec![
                Expr::from_i64(2),
                Expr::Sym(x.clone()),
                Expr::Sym(y.clone()),
            ]),
            Expr::Pow(Arc::new(Expr::Sym(y.clone())), Arc::new(Expr::from_i64(2))),
        ]);

        let envelope = verify_polynomial_identity(&lhs, &rhs, &gens, &context, &mut meter).unwrap();
        assert!(
            verify_derivation_independent(envelope.derivation.as_ref().unwrap(), &context).is_ok()
        );
    }

    #[test]
    fn test_negative_corpus_identity_refusal() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let gens = vec![x.clone(), y.clone()];
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut meter = Unbounded;

        // x + y != x * y
        let lhs = Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]);
        let rhs = Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]);

        let res = verify_polynomial_identity(&lhs, &rhs, &gens, &context, &mut meter);
        assert!(matches!(res, Err(PolyError::IdentityCheckFailed(_))));
    }
}
