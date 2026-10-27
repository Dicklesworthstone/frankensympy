//! Public definite integration and recheckable certificate regressions.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::integrate::definite;
use fsym_calculus::rational_definite::{
    RationalDefiniteIntegralError as Error, metered_rational_definite_integral,
    rational_definite_integral,
};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_definite::{
    RationalDefiniteIntegralVerificationError, RationalIntegralEndpoint as Endpoint,
    verify_rational_definite_integral,
};
use std::sync::Arc;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}
fn p(values: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(Symbol::new("x"), values.iter().map(|&n| q(n)).collect())
}
fn finite(n: i64) -> Endpoint {
    Endpoint::Finite(q(n))
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn fraction(top: &UnivariatePoly, bottom: &UnivariatePoly) -> Expr {
    Expr::Mul(vec![top.to_expr(), power(bottom.to_expr(), -1)])
}
fn exact(value: Expr) -> BigRational {
    match value {
        Expr::Integer(n) => BigRational::from_integer(n),
        Expr::Rational(q) => q,
        other => panic!("expected rational, got {other}"),
    }
}
fn derivative(a: &UnivariatePoly, b: &UnivariatePoly) -> (UnivariatePoly, UnivariatePoly) {
    (
        a.derivative()
            .mul(b)
            .unwrap()
            .sub(&a.mul(&b.derivative()).unwrap())
            .unwrap(),
        b.mul(b).unwrap(),
    )
}

#[test]
fn integrates_irreducible_repeated_denominators_without_root_splitting() {
    // Derivatives of (x^3 + 2x + 1)/(x^2 + c)^m: exact finite
    // endpoint answers over a family with irreducible repeated factors.
    let a = p(&[1, 2, 0, 1]);
    for c in [1i64, 2, 5] {
        for m in 1..=3 {
            let b = p(&[c, 0, 1]).pow(m).unwrap();
            let (n, d) = derivative(&a, &b);
            for (lo, hi) in [(-2i64, 1i64), (0, 3), (3, 0), (1, 1)] {
                let result = rational_definite_integral(&n, &d, &finite(lo), &finite(hi)).unwrap();
                let expected = a.eval(&q(hi)) / b.eval(&q(hi)) - a.eval(&q(lo)) / b.eval(&q(lo));
                assert_eq!(result.value(), &expected);
                verify_rational_definite_integral(
                    &n.coeffs,
                    &d.coeffs,
                    &finite(lo),
                    &finite(hi),
                    result.certificate(),
                    &mut Unbounded,
                )
                .unwrap();
                assert_eq!(
                    exact(
                        definite(
                            &fraction(&n, &d),
                            &Symbol::new("x"),
                            &Expr::from_i64(lo),
                            &Expr::from_i64(hi)
                        )
                        .unwrap()
                    ),
                    expected
                );
            }
        }
    }
}

#[test]
fn improper_and_two_infinite_endpoints_are_checked_separately() {
    // F=x/(1+x^2), so the derivative has integral 0 over R and
    // integral 1/2 from -infinity to 1. No principal-value cancellation.
    let a = p(&[0, 1]);
    let b = p(&[1, 0, 1]);
    let (n, d) = derivative(&a, &b);
    for (lo, hi, expected) in [
        (Endpoint::NegativeInfinity, Endpoint::PositiveInfinity, q(0)),
        (
            Endpoint::NegativeInfinity,
            finite(1),
            BigRational::new(1.into(), 2.into()),
        ),
        (
            finite(1),
            Endpoint::PositiveInfinity,
            BigRational::new((-1).into(), 2.into()),
        ),
        (Endpoint::PositiveInfinity, Endpoint::NegativeInfinity, q(0)),
    ] {
        assert_eq!(
            rational_definite_integral(&n, &d, &lo, &hi)
                .unwrap()
                .value(),
            &expected
        );
    }
    assert_eq!(
        exact(
            definite(
                &fraction(&n, &d),
                &Symbol::new("x"),
                &Expr::Const(Constant::NegativeInfinity),
                &Expr::Const(Constant::Infinity)
            )
            .unwrap()
        ),
        q(0)
    );
    // Separate endpoint limits diverge: an algebraic subtraction of two
    // infinities must never turn this into a certificate for zero.
    assert!(matches!(
        rational_definite_integral(
            &p(&[1]),
            &p(&[1]),
            &Endpoint::NegativeInfinity,
            &Endpoint::PositiveInfinity
        ),
        Err(Error::DivergentEndpoint)
    ));
}

#[test]
fn original_interior_holes_survive_cancellation_reciprocals_and_held_powers() {
    let x = Expr::symbol("x");
    let variable = Symbol::new("x");
    for source in [
        Expr::Mul(vec![x.clone(), power(x.clone(), -1)]),
        power(power(x.clone(), -1), -1),
        power(power(x.clone(), -1), 0),
        Expr::Mul(vec![Expr::from_i64(0), power(x.clone(), -1)]),
    ] {
        assert!(definite(&source, &variable, &Expr::from_i64(-1), &Expr::from_i64(1)).is_none());
    }
    // A removable endpoint exclusion admits the convergent improper limit.
    let cancelled = Expr::Mul(vec![x.clone(), power(x.clone(), -1)]);
    assert_eq!(
        exact(
            definite(
                &cancelled,
                &variable,
                &Expr::from_i64(0),
                &Expr::from_i64(2)
            )
            .unwrap()
        ),
        q(2)
    );
    let inverse = power(power(x, -1), -1);
    assert_eq!(
        exact(definite(&inverse, &variable, &Expr::from_i64(0), &Expr::from_i64(2)).unwrap()),
        q(2)
    );
    let invalid = Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]);
    assert!(definite(&invalid, &variable, &Expr::from_i64(1), &Expr::from_i64(2)).is_none());
}

#[test]
fn poles_and_nonrational_primitives_are_not_given_finite_certificates() {
    let (n, d) = derivative(&p(&[1]), &p(&[-2, 0, 1]));
    assert!(matches!(
        rational_definite_integral(&n, &d, &finite(1), &finite(2)),
        Err(Error::Verification(
            RationalDefiniteIntegralVerificationError::InteriorSingularity
        ))
    ));
    assert!(matches!(
        rational_definite_integral(&p(&[1]), &p(&[0, 0, 1]), &finite(0), &finite(1)),
        Err(Error::DivergentEndpoint)
    ));
    assert!(matches!(
        rational_definite_integral(&p(&[1]), &p(&[1, 0, 1]), &finite(0), &finite(1)),
        Err(Error::UnsupportedPrimitive)
    ));
    // Existing logarithm/arctangent integration remains available.
    for text in ["1/x", "1/(1+x**2)"] {
        assert!(
            definite(
                &fsym_core::parse(text).unwrap(),
                &Symbol::new("x"),
                &Expr::from_i64(1),
                &Expr::from_i64(2)
            )
            .is_some()
        );
    }
    assert_eq!(
        definite(
            &fraction(&p(&[1]), &p(&[0, 0, 1])),
            &Symbol::new("x"),
            &Expr::from_i64(0),
            &Expr::from_i64(1)
        ),
        Some(Expr::Const(Constant::Infinity))
    );
}

#[test]
fn large_exact_endpoints_and_small_coefficients_do_not_use_binary64() {
    let huge = BigRational::from_integer(BigInt::from(10).pow(40));
    let upper = &huge + q(1);
    let result = rational_definite_integral(
        &p(&[1]),
        &p(&[0, 0, 1]),
        &Endpoint::Finite(huge.clone()),
        &Endpoint::Finite(upper.clone()),
    )
    .unwrap();
    assert_eq!(result.value(), &(q(1) / huge - q(1) / upper));
    let tiny = BigRational::new(1.into(), BigInt::from(10).pow(100));
    let n = UnivariatePoly::new(Symbol::new("x"), vec![tiny.clone()]);
    assert_eq!(
        rational_definite_integral(&n, &p(&[1]), &finite(0), &finite(1))
            .unwrap()
            .value(),
        &tiny
    );
}

#[test]
fn divergent_endpoint_subtraction_preserves_sign_and_rejects_infinity_cancellation() {
    let x = Symbol::new("x");
    for (source, lower, upper, expected) in [
        ("1/x**2", 0, 1, Constant::Infinity),
        ("-1/x**2", 0, 1, Constant::NegativeInfinity),
        ("1/x**2", 1, 0, Constant::NegativeInfinity),
        ("1/x**2", -1, 0, Constant::Infinity),
    ] {
        assert_eq!(
            definite(
                &fsym_core::parse(source).unwrap(),
                &x,
                &Expr::from_i64(lower),
                &Expr::from_i64(upper)
            ),
            Some(Expr::Const(expected)),
            "{source} over ({lower}, {upper})",
        );
    }
    let negative_infinity = Expr::Const(Constant::NegativeInfinity);
    let positive_infinity = Expr::Const(Constant::Infinity);
    assert_eq!(
        definite(
            &Expr::from_i64(1),
            &x,
            &negative_infinity,
            &positive_infinity
        ),
        Some(positive_infinity.clone())
    );
    assert_eq!(
        definite(
            &Expr::from_i64(1),
            &x,
            &positive_infinity,
            &negative_infinity
        ),
        Some(negative_infinity.clone())
    );
    // The two separate integrals diverge with opposite signs. A symmetric
    // principal value is not an ordinary definite integral.
    assert!(
        definite(
            &Expr::Sym(x.clone()),
            &x,
            &negative_infinity,
            &positive_infinity
        )
        .is_none()
    );
}

struct Cancelled;
impl BudgetMeter for Cancelled {
    fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn separate_generator_and_verifier_cancellation_never_publishes() {
    assert!(matches!(
        metered_rational_definite_integral(
            &p(&[1]),
            &p(&[0, 0, 1]),
            &finite(1),
            &finite(2),
            &mut Cancelled,
            &mut Unbounded
        ),
        Err(Error::Meter(MeterError::Cancelled))
    ));
    assert!(matches!(
        metered_rational_definite_integral(
            &p(&[1]),
            &p(&[0, 0, 1]),
            &finite(1),
            &finite(2),
            &mut Unbounded,
            &mut Cancelled
        ),
        Err(Error::Verification(
            RationalDefiniteIntegralVerificationError::Meter(MeterError::Cancelled)
        ))
    ));
}
