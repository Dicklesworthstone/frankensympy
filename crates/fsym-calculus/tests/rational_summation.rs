//! Public summation, exact endpoint, domain, and independent-checker regressions.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_sum::{
    RationalSummationError, metered_rational_antidifference, rational_antidifference, rational_sum,
};
use fsym_calculus::summation::summation;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_sum::{
    RationalSumUpper, RationalSumVerificationError, verify_rational_antidifference,
};
use num_traits::One;
use std::sync::Arc;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}

fn poly(values: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(Symbol::new("k"), values.iter().map(|&n| q(n)).collect())
}

fn difference(a: &UnivariatePoly, b: &UnivariatePoly) -> (UnivariatePoly, UnivariatePoly) {
    let next_a = a.shift(&BigRational::one()).unwrap();
    let next_b = b.shift(&BigRational::one()).unwrap();
    (
        next_a
            .mul(b)
            .unwrap()
            .sub(&a.mul(&next_b).unwrap())
            .unwrap(),
        b.mul(&next_b).unwrap(),
    )
}

fn raw_power(base: Expr, power: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(power)))
}

fn raw_fraction(n: &UnivariatePoly, d: &UnivariatePoly) -> Expr {
    Expr::Mul(vec![n.to_expr(), raw_power(d.to_expr(), -1)])
}

fn exact(expr: Expr) -> BigRational {
    match expr {
        Expr::Integer(n) => BigRational::from_integer(n),
        Expr::Rational(r) => r,
        other => panic!("expected exact rational result, got {other}"),
    }
}

#[test]
fn repeated_fractional_poles_support_finite_and_infinite_sums() {
    for offset in [1i64, 3, 5] {
        for order in 1..=4 {
            // F(k) = -1/(2k+offset)^order. No pole lies at an integer.
            let a = poly(&[-1]);
            let b = poly(&[offset, 2]).pow(order).unwrap();
            let (n, d) = difference(&a, &b);
            let certificate = rational_antidifference(&n, &d).unwrap();
            verify_rational_antidifference(&n.coeffs, &d.coeffs, &certificate, &mut Unbounded)
                .unwrap();
            for end in [0i64, 1, 7, 10_000] {
                let sum =
                    rational_sum(&n, &d, &0.into(), &RationalSumUpper::Finite(end.into())).unwrap();
                let expected =
                    a.eval(&q(end + 1)) / b.eval(&q(end + 1)) - a.eval(&q(0)) / b.eval(&q(0));
                assert_eq!(sum.value, expected);
            }
            let infinite =
                rational_sum(&n, &d, &0.into(), &RationalSumUpper::PositiveInfinity).unwrap();
            assert_eq!(infinite.value, q(1) / b.eval(&q(0)));
            assert_eq!(
                exact(
                    summation(
                        &raw_fraction(&n, &d),
                        &n.gen_sym,
                        &Expr::from_i64(0),
                        &Expr::Const(Constant::Infinity)
                    )
                    .unwrap()
                ),
                infinite.value
            );
        }
    }
}

#[test]
fn handles_polynomial_quotients_and_multiple_shift_orbits() {
    // F(k) = (k^5+2k+3)/((2k+1)^2*(3k+1)). This mixes the
    // 1/2+Z and 1/3+Z pole orbits and has a polynomial quotient.
    let a = poly(&[3, 2, 0, 0, 0, 1]);
    let b = poly(&[1, 2]).pow(2).unwrap().mul(&poly(&[1, 3])).unwrap();
    let (n, d) = difference(&a, &b);
    for (lo, hi) in [(-5i64, 4i64), (0, 0), (100, 103), (8, 1), (0, -1)] {
        let result =
            rational_sum(&n, &d, &lo.into(), &RationalSumUpper::Finite(hi.into())).unwrap();
        let expected = a.eval(&q(hi + 1)) / b.eval(&q(hi + 1)) - a.eval(&q(lo)) / b.eval(&q(lo));
        assert_eq!(result.value, expected);
    }
    assert!(rational_sum(&n, &d, &0.into(), &RationalSumUpper::PositiveInfinity).is_err());
}

#[test]
fn public_whole_expression_telescopes_before_linearity() {
    let k = Symbol::new("k");
    let x = Expr::Sym(k.clone());
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let f = Expr::Add(vec![
        raw_power(Expr::Add(vec![x.clone(), half.clone()]), -2),
        Expr::Mul(vec![
            Expr::from_i64(-1),
            raw_power(Expr::Add(vec![x, half, Expr::from_i64(2)]), -2),
        ]),
    ]);
    // Infinite sum is 1/(1/2)^2 + 1/(3/2)^2.
    assert_eq!(
        exact(summation(&f, &k, &Expr::from_i64(0), &Expr::Const(Constant::Infinity)).unwrap()),
        BigRational::new(40.into(), 9.into())
    );
    assert!(summation(&f, &k, &Expr::from_i64(0), &Expr::symbol("n")).is_none());
}

#[test]
fn original_poles_survive_cancellation_and_nested_reciprocals() {
    let k = Symbol::new("k");
    let x = Expr::Sym(k.clone());
    let cancelled = Expr::Mul(vec![x.clone(), raw_power(x.clone(), -1)]);
    assert!(summation(&cancelled, &k, &Expr::from_i64(-1), &Expr::from_i64(2)).is_none());
    assert_eq!(
        exact(summation(&cancelled, &k, &Expr::from_i64(1), &Expr::from_i64(5000)).unwrap()),
        q(5000)
    );
    let nested = raw_power(raw_power(x.clone(), -1), -1);
    assert!(summation(&nested, &k, &Expr::from_i64(0), &Expr::from_i64(2)).is_none());
    assert_eq!(
        exact(summation(&nested, &k, &Expr::from_i64(1), &Expr::from_i64(3)).unwrap()),
        q(6)
    );
    let held_zero_power = raw_power(raw_power(x, -1), 0);
    assert!(summation(&held_zero_power, &k, &Expr::from_i64(0), &Expr::from_i64(2)).is_none());
    let zero_denominator = Expr::Mul(vec![Expr::from_i64(0), raw_power(Expr::from_i64(0), -1)]);
    assert!(
        summation(
            &zero_denominator,
            &k,
            &Expr::from_i64(1),
            &Expr::from_i64(2)
        )
        .is_none()
    );
}

#[test]
fn unsupported_residues_and_shift_spans_refuse_without_partial_answers() {
    // 1/k^2 has no rational antidifference; its short finite sums still
    // work through exact direct evaluation and its zeta rule remains live.
    assert!(rational_antidifference(&poly(&[1]), &poly(&[0, 0, 1])).is_err());
    let f = raw_power(Expr::symbol("k"), -2);
    assert_eq!(
        exact(
            summation(
                &f,
                &Symbol::new("k"),
                &Expr::from_i64(1),
                &Expr::from_i64(3)
            )
            .unwrap()
        ),
        BigRational::new(49.into(), 36.into())
    );
    assert!(
        summation(
            &f,
            &Symbol::new("k"),
            &Expr::from_i64(1),
            &Expr::Const(Constant::Infinity)
        )
        .is_some()
    );
    assert!(
        summation(
            &f,
            &Symbol::new("k"),
            &Expr::from_i64(0),
            &Expr::Const(Constant::Infinity)
        )
        .is_none()
    );
    assert!(
        summation(
            &f,
            &Symbol::new("k"),
            &Expr::from_i64(1_000_000_000_000),
            &Expr::Const(Constant::Infinity)
        )
        .is_none()
    );
    let den = poly(&[1, 1]).mul(&poly(&[66, 1])).unwrap();
    assert!(rational_antidifference(&poly(&[65]), &den).is_err());
    assert!(rational_antidifference(&poly(&[1]), &poly(&[1, 0, 1])).is_err());
}

#[test]
fn arbitrary_precision_integer_endpoints_are_not_converted_to_f64() {
    let a = poly(&[-1]);
    let b = poly(&[1, 2]).pow(2).unwrap();
    let (n, d) = difference(&a, &b);
    let end = BigInt::from(10).pow(30);
    let result = rational_sum(&n, &d, &0.into(), &RationalSumUpper::Finite(end.clone())).unwrap();
    let after = BigRational::from_integer(end + BigInt::one());
    assert_eq!(result.value, q(1) - q(1) / b.eval(&after));
}

#[test]
fn geometric_convergence_is_exact_on_both_sides_of_one() {
    let k = Symbol::new("k");
    let denominator = BigInt::from(10).pow(40);
    for sign in [-1i64, 1] {
        let numerator = &denominator + BigInt::from(sign);
        let ratio = BigRational::new(numerator, denominator.clone());
        let f = Expr::Pow(
            Arc::new(Expr::Rational(ratio.clone())),
            Arc::new(Expr::Sym(k.clone())),
        );
        let result = summation(&f, &k, &Expr::from_i64(0), &Expr::Const(Constant::Infinity));
        if sign < 0 {
            assert_eq!(
                exact(result.unwrap()),
                BigRational::from_integer(denominator.clone())
            );
        } else {
            assert!(result.is_none());
        }
    }
    let shifted_zero = Expr::Pow(
        Arc::new(Expr::from_i64(0)),
        Arc::new(Expr::Add(vec![Expr::Sym(k.clone()), Expr::from_i64(-1)])),
    );
    assert!(
        summation(
            &shifted_zero,
            &k,
            &Expr::from_i64(1),
            &Expr::Const(Constant::Infinity)
        )
        .is_none()
    );
    let alternating = Expr::Pow(Arc::new(Expr::from_i64(-1)), Arc::new(Expr::Sym(k.clone())));
    assert!(
        summation(
            &alternating,
            &k,
            &Expr::from_i64(0),
            &Expr::Const(Constant::Infinity)
        )
        .is_none()
    );
}

struct Cancelled;
impl BudgetMeter for Cancelled {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn generator_and_verifier_cancellation_do_not_publish_results() {
    let (n, d) = difference(&poly(&[-1]), &poly(&[1, 2]).pow(2).unwrap());
    assert!(matches!(
        metered_rational_antidifference(&n, &d, &mut Cancelled, &mut Unbounded),
        Err(RationalSummationError::Meter(MeterError::Cancelled))
    ));
    assert!(matches!(
        metered_rational_antidifference(&n, &d, &mut Unbounded, &mut Cancelled),
        Err(RationalSummationError::Verification(
            RationalSumVerificationError::Meter(MeterError::Cancelled)
        ))
    ));
}
