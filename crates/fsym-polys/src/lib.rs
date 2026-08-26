//! # fsym-polys
//!
//! Polynomial algebra, polynomial rings, multivariate sparse representations,
//! polynomial identity testing (PIT), and verified identity certificates (WS08).

#![forbid(unsafe_code)]

pub mod factorization;
pub mod gcd;
pub mod groebner;
pub mod identity;
pub mod multivariate;
pub mod univariate;

pub use factorization::*;
pub use gcd::*;
pub use groebner::*;
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
    use num_traits::{One, Zero};
    use std::collections::BTreeMap;
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

        // Positive powers of zero remain ordinary polynomial evaluation,
        // even though negative powers of zero are rejected by BigRational.
        let zero_pt = vec![BigRational::zero(), BigRational::zero()];
        assert_eq!(poly.eval(&zero_pt).unwrap(), BigRational::zero());
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

        let envelope = verify_polynomial_identity(
            &lhs,
            &rhs,
            &gens,
            &context,
            fsym_id::ReceiptId::new(1).unwrap(),
            &mut meter,
        )
        .unwrap();
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

        let res = verify_polynomial_identity(
            &lhs,
            &rhs,
            &gens,
            &context,
            fsym_id::ReceiptId::new(2).unwrap(),
            &mut meter,
        );
        assert!(matches!(res, Err(PolyError::IdentityCheckFailed(_))));
    }

    #[test]
    fn multivariate_conversion_refuses_exponents_that_do_not_fit_u32() {
        let x = Symbol::new("x");
        let exponent = BigInt::from(u64::from(u32::MAX) + 1);
        let expr = Expr::Pow(
            Arc::new(Expr::Sym(x.clone())),
            Arc::new(Expr::Integer(exponent)),
        );

        assert!(matches!(
            MultivariatePoly::from_expr(&expr, &[x]),
            Err(PolyError::NonPolynomialExpression(_))
        ));
    }

    #[test]
    fn verified_identity_refuses_duplicate_generators() {
        let x = Symbol::new("x");
        let expr = Expr::Sym(x.clone());
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());

        assert!(matches!(
            verify_polynomial_identity(
                &expr,
                &expr,
                &[x.clone(), x],
                &context,
                fsym_id::ReceiptId::new(3).unwrap(),
                &mut Unbounded,
            ),
            Err(PolyError::General(message)) if message.contains("duplicate")
        ));
    }

    #[test]
    fn metered_multivariate_zero_product_does_not_issue_a_zero_charge() {
        let x = Symbol::new("x");
        let zero = MultivariatePoly::zero(vec![x.clone()]);
        let variable = MultivariatePoly::var(vec![x.clone()], &x).unwrap();
        let mut meter = fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(1, 0));

        assert_eq!(
            zero.metered_mul(&variable, &mut meter).unwrap(),
            MultivariatePoly::zero(vec![x])
        );
    }

    #[test]
    fn multivariate_json_wire_roundtrips_without_non_string_map_keys() {
        let x = Symbol::new("x");
        let poly = MultivariatePoly::var(vec![x.clone()], &x).unwrap();
        let wire = serde_json::to_value(&poly).unwrap();
        assert!(wire["terms"].is_array());

        let restored: MultivariatePoly = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(restored, poly);

        let mut malformed = wire;
        malformed["terms"][0]["exponents"] = serde_json::json!([1, 0]);
        assert!(serde_json::from_value::<MultivariatePoly>(malformed).is_err());
    }

    #[test]
    fn test_extended_gcd_bezout_certificate_verified() {
        let x = Symbol::new("x");
        // A(x) = (x - 1)*(x - 2) = x^2 - 3x + 2
        let a = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(2)),
                BigRational::from_integer(BigInt::from(-3)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );
        // B(x) = (x - 2)*(x - 3) = x^2 - 5x + 6
        let b = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(6)),
                BigRational::from_integer(BigInt::from(-5)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );

        let cert = a.extended_gcd(&b).unwrap();
        // Expect monic gcd = x - 2
        let expected_gcd = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(-2)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );
        assert_eq!(cert.gcd, expected_gcd);
        assert!(verify_bezout_certificate(&a, &b, &cert).is_ok());
    }

    #[test]
    fn test_mutant_tampered_bezout_rejected() {
        let x = Symbol::new("x");
        let a = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(2)),
                BigRational::from_integer(BigInt::from(-3)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );
        let b = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(6)),
                BigRational::from_integer(BigInt::from(-5)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );

        let mut cert = a.extended_gcd(&b).unwrap();
        // Tamper U coefficient
        cert.u.coeffs[0] = BigRational::from_integer(BigInt::from(999));
        assert!(verify_bezout_certificate(&a, &b, &cert).is_err());
    }

    #[test]
    fn test_square_free_decomposition_and_verification() {
        let x = Symbol::new("x");
        // P(x) = (x - 1)^2 * (x + 2) = (x^2 - 2x + 1)*(x + 2) = x^3 - 3x + 2
        let p = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(2)),
                BigRational::from_integer(BigInt::from(-3)),
                BigRational::from_integer(BigInt::from(0)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );

        let factorization = square_free_decomposition(&p).unwrap();
        assert!(verify_factorization_certificate(&p, &factorization).is_ok());
        assert_eq!(factorization.factors.len(), 2);
    }

    #[test]
    fn test_mutant_tampered_factorization_rejected() {
        let x = Symbol::new("x");
        let p = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(2)),
                BigRational::from_integer(BigInt::from(-3)),
                BigRational::from_integer(BigInt::from(0)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );

        let mut factorization = square_free_decomposition(&p).unwrap();
        // Tamper factor power
        factorization.factors[0].multiplicity = 99;
        assert!(verify_factorization_certificate(&p, &factorization).is_err());
    }

    #[test]
    fn factorization_verifier_rejects_noncanonical_certificate_shapes() {
        let x = Symbol::new("x");
        let x_poly = UnivariatePoly::new(
            x.clone(),
            vec![BigRational::from_integer(0.into()), BigRational::one()],
        );

        let non_monic = FactorizationResult {
            scale: BigRational::new(1.into(), 2.into()),
            factors: vec![FactorTerm {
                poly: UnivariatePoly::new(
                    x.clone(),
                    vec![
                        BigRational::from_integer(0.into()),
                        BigRational::from_integer(2.into()),
                    ],
                ),
                multiplicity: 1,
            }],
        };
        assert!(verify_factorization_certificate(&x_poly, &non_monic).is_err());

        let one = UnivariatePoly::one(x.clone());
        let zero_multiplicity = FactorizationResult {
            scale: BigRational::one(),
            factors: vec![FactorTerm {
                poly: x_poly.clone(),
                multiplicity: 0,
            }],
        };
        assert!(verify_factorization_certificate(&one, &zero_multiplicity).is_err());

        #[cfg(target_pointer_width = "64")]
        {
            let wrapped_multiplicity = FactorizationResult {
                scale: BigRational::one(),
                factors: vec![FactorTerm {
                    poly: x_poly,
                    multiplicity: (u32::MAX as usize) + 1,
                }],
            };
            assert!(verify_factorization_certificate(&one, &wrapped_multiplicity).is_err());
        }
    }

    #[test]
    fn univariate_boundaries_reject_noncanonical_wire_and_exponent_narrowing() {
        let x = Symbol::new("x");
        let poly = UnivariatePoly::new(
            x.clone(),
            vec![BigRational::from_integer(1.into()), BigRational::one()],
        );
        let mut wire = serde_json::to_value(&poly).unwrap();
        wire.get_mut("coeffs")
            .and_then(serde_json::Value::as_array_mut)
            .unwrap()
            .push(serde_json::to_value(BigRational::from_integer(0.into())).unwrap());
        assert!(serde_json::from_value::<UnivariatePoly>(wire).is_err());

        let oversized_power = Expr::Pow(
            Arc::new(Expr::Sym(x.clone())),
            Arc::new(Expr::Integer(BigInt::from(u64::from(u32::MAX) + 1))),
        );
        assert!(matches!(
            UnivariatePoly::from_expr(&oversized_power, &x),
            Err(PolyError::NonPolynomialExpression(_))
        ));

        let excessive_dense_power = Expr::Pow(
            Arc::new(Expr::Sym(x.clone())),
            Arc::new(Expr::from_i64(65_536)),
        );
        assert!(matches!(
            UnivariatePoly::from_expr(&excessive_dense_power, &x),
            Err(PolyError::General(_))
        ));

        assert!(matches!(
            UnivariatePoly::monomial(x, BigRational::one(), usize::MAX),
            Err(PolyError::General(_))
        ));

        let invalid = UnivariatePoly {
            gen_sym: Symbol::new("x"),
            coeffs: Vec::new(),
        };
        let one = UnivariatePoly::one(Symbol::new("x"));
        assert!(invalid.add(&one).is_err());
        assert!(invalid.sub(&one).is_err());
        assert!(invalid.mul(&one).is_err());
        assert!(invalid.div_rem(&one).is_err());
        assert!(invalid.gcd(&one).is_err());
        assert!(invalid.extended_gcd(&one).is_err());
        assert!(square_free_decomposition(&invalid).is_err());
        assert!(serde_json::to_value(&invalid).is_err());

        let oversized_wire = serde_json::json!({
            "gen_sym": Symbol::new("x"),
            "coeffs": vec![BigRational::from_integer(0.into()); 65_537],
        });
        assert!(serde_json::from_value::<UnivariatePoly>(oversized_wire).is_err());
    }

    #[test]
    fn test_groebner_basis_computation_and_ideal_membership() {
        // Ideal I = <x^2 + y, x*y + x> in Q[x, y]
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let gens = vec![x.clone(), y.clone()];

        // f1 = x^2 + y
        let mut t1 = BTreeMap::new();
        t1.insert(vec![2, 0], BigRational::one());
        t1.insert(vec![0, 1], BigRational::one());
        let f1 = MultivariatePoly::new(gens.clone(), t1);

        // f2 = x*y + x
        let mut t2 = BTreeMap::new();
        t2.insert(vec![1, 1], BigRational::one());
        t2.insert(vec![1, 0], BigRational::one());
        let f2 = MultivariatePoly::new(gens.clone(), t2);

        let gb = groebner_basis(&[f1.clone(), f2.clone()], TermOrder::Lex).unwrap();
        assert!(!gb.is_empty());

        // Check ideal membership of original generators
        assert!(ideal_membership(&f1, &gb, TermOrder::Lex).unwrap());
        assert!(ideal_membership(&f2, &gb, TermOrder::Lex).unwrap());

        // Check non-member polynomial is correctly rejected
        // g = y + 5
        let mut tg = BTreeMap::new();
        tg.insert(vec![0, 1], BigRational::one());
        tg.insert(vec![0, 0], BigRational::from_integer(BigInt::from(5)));
        let g = MultivariatePoly::new(gens.clone(), tg);
        assert!(!ideal_membership(&g, &gb, TermOrder::Lex).unwrap());
    }

    #[test]
    fn test_variable_elimination() {
        // System:
        // x - y^2 = 0
        // x - z = 0
        // Eliminate x -> yields y^2 - z = 0
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let z = Symbol::new("z");
        let gens = vec![x.clone(), y.clone(), z.clone()];

        // f1 = x - y^2
        let mut t1 = BTreeMap::new();
        t1.insert(vec![1, 0, 0], BigRational::one());
        t1.insert(vec![0, 2, 0], BigRational::from_integer(BigInt::from(-1)));
        let f1 = MultivariatePoly::new(gens.clone(), t1);

        // f2 = x - z
        let mut t2 = BTreeMap::new();
        t2.insert(vec![1, 0, 0], BigRational::one());
        t2.insert(vec![0, 0, 1], BigRational::from_integer(BigInt::from(-1)));
        let f2 = MultivariatePoly::new(gens.clone(), t2);

        let elim = eliminate(&[f1, f2], std::slice::from_ref(&x)).unwrap();
        assert!(!elim.is_empty());
        // Verify that none of the polynomials in elim contain x (deg in x is 0)
        for p in &elim {
            assert_eq!(p.degree_in(0), 0);
        }
    }

    #[test]
    fn multivariate_division_refuses_incompatible_rings() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let dividend = MultivariatePoly::one(vec![x.clone()]);
        let divisor = MultivariatePoly::one(vec![y]);

        assert!(matches!(
            dividend.div_rem(&[divisor], TermOrder::Lex),
            Err(PolyError::IncompatibleGenerators(_, _))
        ));
        assert!(matches!(
            groebner_basis(
                &[dividend, MultivariatePoly::one(vec![x, Symbol::new("z")])],
                TermOrder::Lex,
            ),
            Err(PolyError::IncompatibleGenerators(_, _))
        ));
    }

    #[test]
    fn graded_orders_do_not_overflow_total_degree() {
        let large = [u32::MAX, u32::MAX];
        let smaller = [u32::MAX, u32::MAX - 1];
        assert_eq!(
            TermOrder::DegLex.compare_monomials(&large, &smaller),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            TermOrder::DegRevLex.compare_monomials(&large, &smaller),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn elimination_refuses_unknown_variables() {
        let x = Symbol::new("x");
        let poly = MultivariatePoly::one(vec![x]);
        let missing = Symbol::new("missing");
        assert!(matches!(
            eliminate(&[poly], &[missing]),
            Err(PolyError::IncompatibleGenerators(_, _))
        ));
    }

    #[test]
    fn test_factor_polynomial_complete() {
        let x = Symbol::new("x");

        // 1. Quadratic: x^2 - 1 = (x - 1)(x + 1)
        let p1 = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(-1)),
                BigRational::zero(),
                BigRational::one(),
            ],
        );
        let res1 = factor_polynomial(&p1).unwrap();
        assert_eq!(res1.factors.len(), 2);
        assert_eq!(res1.scale, BigRational::one());
        assert!(verify_factorization_certificate(&p1, &res1).is_ok());

        // 2. Cubic with 3 roots: (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
        let p2 = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(-6)),
                BigRational::from_integer(BigInt::from(11)),
                BigRational::from_integer(BigInt::from(-6)),
                BigRational::one(),
            ],
        );
        let res2 = factor_polynomial(&p2).unwrap();
        assert_eq!(res2.factors.len(), 3);
        assert!(verify_factorization_certificate(&p2, &res2).is_ok());

        // 3. Repeated roots: 3*(x - 2)^2 = 3*(x^2 - 4x + 4) = 3x^2 - 12x + 12
        let p3 = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(12)),
                BigRational::from_integer(BigInt::from(-12)),
                BigRational::from_integer(BigInt::from(3)),
            ],
        );
        let res3 = factor_polynomial(&p3).unwrap();
        assert_eq!(res3.scale, BigRational::from_integer(BigInt::from(3)));
        assert_eq!(res3.factors.len(), 1);
        assert_eq!(res3.factors[0].multiplicity, 2);
        assert!(verify_factorization_certificate(&p3, &res3).is_ok());

        // 4. Irreducible quadratic: x^2 + 1
        let p4 = UnivariatePoly::new(
            x.clone(),
            vec![BigRational::one(), BigRational::zero(), BigRational::one()],
        );
        let res4 = factor_polynomial(&p4).unwrap();
        assert_eq!(res4.factors.len(), 1);
        assert_eq!(res4.factors[0].poly, p4);
        assert!(verify_factorization_certificate(&p4, &res4).is_ok());
    }

    #[test]
    fn test_groebner_basis_certificate_and_verification() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let gens = vec![x.clone(), y.clone()];

        // Ideal I = <x^2 + y, x*y + x> under DegLex
        // f1 = x^2 + y
        let mut t1 = BTreeMap::new();
        t1.insert(vec![2, 0], BigRational::one());
        t1.insert(vec![0, 1], BigRational::one());
        let f1 = MultivariatePoly::new(gens.clone(), t1);

        // f2 = x*y + x
        let mut t2 = BTreeMap::new();
        t2.insert(vec![1, 1], BigRational::one());
        t2.insert(vec![1, 0], BigRational::one());
        let f2 = MultivariatePoly::new(gens, t2);

        let initial = vec![f1, f2];
        let cert = groebner_basis_with_certificate(&initial, TermOrder::DegLex).unwrap();
        assert_eq!(
            cert.basis,
            groebner_basis(&initial, TermOrder::DegLex).unwrap()
        );
        assert!(verify_groebner_certificate(&initial, &cert).is_ok());

        // Mutants:
        // 1. Non-monic basis element
        let mut non_monic_cert = cert.clone();
        let mut tampered_terms = non_monic_cert.basis[0].terms.clone();
        for val in tampered_terms.values_mut() {
            *val *= BigRational::from_integer(2.into());
        }
        non_monic_cert.basis[0].terms = tampered_terms;
        assert!(verify_groebner_certificate(&initial, &non_monic_cert).is_err());

        // 2. Incomplete basis (missing generator)
        let incomplete_cert = GroebnerBasisCertificate {
            order: TermOrder::DegLex,
            basis: vec![cert.basis[0].clone()],
            input_ideal_witnesses: vec![cert.input_ideal_witnesses[0].clone()],
        };
        assert!(verify_groebner_certificate(&initial, &incomplete_cert).is_err());

        // 3. A larger ideal is not the same ideal: <1> contains the input ideal, but the unit
        // polynomial cannot be expressed as a linear combination of these generators.
        let forged_unit_ideal = GroebnerBasisCertificate {
            order: TermOrder::DegLex,
            basis: vec![MultivariatePoly::one(initial[0].generators.clone())],
            input_ideal_witnesses: vec![vec![
                MultivariatePoly::zero(initial[0].generators.clone());
                initial.len()
            ]],
        };
        assert!(verify_groebner_certificate(&initial, &forged_unit_ideal).is_err());

        // 4. Membership witnesses are checked as exact identities, not trusted metadata.
        let mut tampered_witness_cert = cert.clone();
        tampered_witness_cert.input_ideal_witnesses[0][0] = tampered_witness_cert
            .input_ideal_witnesses[0][0]
            .add(&MultivariatePoly::one(initial[0].generators.clone()))
            .unwrap();
        assert!(verify_groebner_certificate(&initial, &tampered_witness_cert).is_err());
    }
}
