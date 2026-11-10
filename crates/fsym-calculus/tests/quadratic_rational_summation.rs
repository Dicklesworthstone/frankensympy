//! Quadratic shift orbits, complete integer domains, and public summation.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_sum::{
    RationalSummationError, metered_rational_antidifference, rational_antidifference, rational_sum,
};
use fsym_calculus::summation::summation;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, parse};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_sum::{
    RationalSumUpper, RationalSumVerificationError, verify_rational_antidifference,
    verify_rational_sum,
};
use std::sync::Arc;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}

fn poly(values: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(
        Symbol::new("k"),
        values.iter().map(|&value| q(value)).collect(),
    )
}

fn difference(a: &UnivariatePoly, b: &UnivariatePoly) -> (UnivariatePoly, UnivariatePoly) {
    let next_a = a.shift(&q(1)).unwrap();
    let next_b = b.shift(&q(1)).unwrap();
    (
        next_a
            .mul(b)
            .unwrap()
            .sub(&a.mul(&next_b).unwrap())
            .unwrap(),
        b.mul(&next_b).unwrap(),
    )
}

fn exact(expression: Expr) -> BigRational {
    match expression {
        Expr::Integer(value) => BigRational::from_integer(value),
        Expr::Rational(value) => value,
        other => panic!("expected an exact rational sum, got {other}"),
    }
}

fn raw_power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn raw_fraction(n: &UnivariatePoly, d: &UnivariatePoly) -> Expr {
    Expr::Mul(vec![n.to_expr(), raw_power(d.to_expr(), -1)])
}

#[test]
fn repeated_quadratics_and_linear_numerators_telescope_on_the_integer_lattice() {
    let half = BigRational::new(1.into(), 2.into());
    let factors = [
        poly(&[1, 0, 1]),
        poly(&[-2, 0, 1]),
        poly(&[1, 0, 1]).shift(&half).unwrap(),
    ];
    for factor in factors {
        for order in 1..=3 {
            // Every numerator is nonconstant: translating just the factor,
            // without also translating its residue, would give a wrong sum.
            let a = poly(&[-2, 3]);
            let b = factor.pow(order).unwrap();
            let (n, d) = difference(&a, &b);
            let certificate = rational_antidifference(&n, &d).unwrap();
            assert_eq!(certificate.input_quadratics.len(), 2);
            assert!(certificate.input_poles.is_empty());
            verify_rational_antidifference(&n.coeffs, &d.coeffs, &certificate, &mut Unbounded)
                .unwrap();
            for (lo, hi) in [(-5, 8), (0, 0), (100, 103), (8, 1), (0, -1)] {
                let expected = a.eval(&q(hi + 1)) / b.eval(&q(hi + 1))
                    - a.eval(&q(lo)) / b.eval(&q(lo));
                assert_eq!(
                    verify_rational_sum(
                        &n.coeffs,
                        &d.coeffs,
                        &certificate,
                        &lo.into(),
                        &RationalSumUpper::Finite(hi.into()),
                        &mut Unbounded,
                    )
                    .unwrap(),
                    expected
                );
            }
            assert_eq!(
                verify_rational_sum(
                    &n.coeffs,
                    &d.coeffs,
                    &certificate,
                    &0.into(),
                    &RationalSumUpper::PositiveInfinity,
                    &mut Unbounded,
                )
                .unwrap(),
                -a.eval(&q(0)) / b.eval(&q(0))
            );
        }
    }
}

#[test]
fn mixed_linear_quadratic_and_fractional_shift_orbits_preserve_polynomial_parts() {
    let half = BigRational::new(1.into(), 2.into());
    let quadratic = poly(&[1, 0, 1]);
    // Q(k) and Q(k+1/2) have the same centered factor, but their distinct
    // residue classes modulo Z must never be merged into one shift orbit.
    let b = quadratic
        .mul(&quadratic.shift(&half).unwrap())
        .unwrap()
        .mul(&poly(&[1, 2]))
        .unwrap();
    let a = poly(&[2, -1, 0, 0, 0, 0, 1]);
    let (n, d) = difference(&a, &b);
    let certificate = rational_antidifference(&n, &d).unwrap();
    assert_eq!(certificate.input_poles.len(), 2);
    assert_eq!(certificate.input_quadratics.len(), 4);
    for (lo, hi) in [(-8, 9), (0, 0), (10_000, 20_000), (8, 1)] {
        assert_eq!(
            verify_rational_sum(
                &n.coeffs,
                &d.coeffs,
                &certificate,
                &lo.into(),
                &RationalSumUpper::Finite(hi.into()),
                &mut Unbounded,
            )
            .unwrap(),
            a.eval(&q(hi + 1)) / b.eval(&q(hi + 1)) - a.eval(&q(lo)) / b.eval(&q(lo))
        );
    }
    assert!(
        verify_rational_sum(
            &n.coeffs,
            &d.coeffs,
            &certificate,
            &0.into(),
            &RationalSumUpper::PositiveInfinity,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn missing_intermediate_shifts_are_filled_with_the_correct_polynomial_residue() {
    let q0 = poly(&[2, 1, 1]);
    let q3 = q0.shift(&q(3)).unwrap();
    let p0 = poly(&[2, 3]);
    let p3 = p0.shift(&q(3)).unwrap();
    // P(k)/Q(k)^2 - P(k+3)/Q(k+3)^2 has a three-term rational primitive.
    let d0 = q0.pow(2).unwrap();
    let d3 = q3.pow(2).unwrap();
    let n = p0
        .mul(&d3)
        .unwrap()
        .sub(&p3.mul(&d0).unwrap())
        .unwrap();
    let d = d0.mul(&d3).unwrap();
    let result = rational_sum(&n, &d, &0.into(), &RationalSumUpper::PositiveInfinity).unwrap();
    let expected = (0..3)
        .map(|k| p0.eval(&q(k)) / d0.eval(&q(k)))
        .fold(q(0), |sum, term| sum + term);
    assert_eq!(result.value, expected);
    assert_eq!(result.certificate.antidifference_quadratics.len(), 3);
    assert!(
        result
            .certificate
            .antidifference_quadratics
            .iter()
            .all(|factor| factor.multiplicity == 2)
    );
}

#[test]
fn large_exact_endpoints_and_existing_public_summation_use_the_new_lane() {
    let k = Symbol::new("k");
    let source = parse("1/(k^2+1)-1/((k+1)^2+1)").unwrap();
    assert_eq!(
        exact(
            summation(
                &source,
                &k,
                &Expr::from_i64(0),
                &Expr::Const(Constant::Infinity),
            )
            .unwrap()
        ),
        q(1)
    );
    let end = BigInt::from(10).pow(40);
    let after = BigRational::from_integer(&end + BigInt::one());
    assert_eq!(
        exact(summation(&source, &k, &Expr::from_i64(0), &Expr::Integer(end)).unwrap()),
        q(1) - q(1) / (&after * &after + q(1))
    );
    let source = parse("1/(k^2-2)-1/((k+1)^2-2)").unwrap();
    assert_eq!(
        exact(
            summation(
                &source,
                &k,
                &Expr::from_i64(0),
                &Expr::Const(Constant::Infinity),
            )
            .unwrap()
        ),
        BigRational::new((-1).into(), 2.into())
    );
    let b = UnivariatePoly::new(
        k.clone(),
        vec![BigRational::new((-1).into(), 8.into()), q(0), q(1)],
    );
    let (n, d) = difference(&poly(&[-1]), &b);
    assert_eq!(
        exact(
            summation(
                &raw_fraction(&n, &d),
                &k,
                &Expr::from_i64(0),
                &Expr::Const(Constant::Infinity),
            )
            .unwrap()
        ),
        q(-8)
    );
}

#[test]
fn original_cancelled_linear_poles_survive_quadratic_telescoping() {
    let k = Symbol::new("k");
    let x = Expr::Sym(k.clone());
    let source = parse("1/(k^2+1)-1/((k+1)^2+1)").unwrap();
    let holes = [
        Expr::Mul(vec![x.clone(), raw_power(x.clone(), -1)]),
        raw_power(raw_power(x.clone(), -1), 0),
    ];
    for hole in holes {
        let held = Expr::Mul(vec![source.clone(), hole]);
        assert!(
            summation(&held, &k, &Expr::from_i64(0), &Expr::from_i64(10_000)).is_none()
        );
        assert!(
            summation(
                &held,
                &k,
                &Expr::from_i64(0),
                &Expr::Const(Constant::Infinity),
            )
            .is_none()
        );
        assert_eq!(
            exact(
                summation(
                    &held,
                    &k,
                    &Expr::from_i64(1),
                    &Expr::Const(Constant::Infinity),
                )
                .unwrap()
            ),
            BigRational::new(1.into(), 2.into())
        );
        let held_zero = Expr::Mul(vec![Expr::from_i64(0), held]);
        assert!(
            summation(
                &held_zero,
                &k,
                &Expr::from_i64(0),
                &Expr::Const(Constant::Infinity),
            )
            .is_none()
        );
    }
}

#[test]
fn nonzero_polynomial_residues_and_nonmatching_quadratics_refuse_without_partial_values() {
    for source in [
        "1/(k^2+1)",
        "1/(k^2+1)-1/(k^2+2)",
        "k/(k^2+1)-k/((k+1)^2+1)",
    ] {
        assert!(
            summation(
                &parse(source).unwrap(),
                &Symbol::new("k"),
                &Expr::from_i64(0),
                &Expr::Const(Constant::Infinity),
            )
            .is_none(),
            "{source}"
        );
    }
    let q0 = poly(&[1, 0, 1]);
    let q65 = q0.shift(&q(65)).unwrap();
    let n = q65.sub(&q0).unwrap();
    let d = q0.mul(&q65).unwrap();
    assert!(matches!(
        rational_antidifference(&n, &d),
        Err(RationalSummationError::Refused("integer-shift span limit"))
    ));
    assert!(rational_antidifference(&poly(&[1]), &q0.pow(17).unwrap()).is_err());
    assert!(rational_antidifference(&poly(&[1]), &poly(&[1, 1, 0, 1])).is_err());
}

#[derive(Default)]
struct Checkpoints {
    seen: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Checkpoints {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.seen += 1;
        if self.cancel_at == Some(self.seen) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
}

#[test]
fn new_generator_and_checker_share_no_budget_reset_and_refuse_cancelled_publication() {
    let (n, d) = difference(&poly(&[-1]), &poly(&[1, 0, 1]));
    let mut generation = Budget::new(BudgetLimits::uniform(u64::MAX, 37));
    let mut verification = Budget::new(BudgetLimits::uniform(u64::MAX, 41));
    let good = metered_rational_antidifference(&n, &d, &mut generation, &mut verification).unwrap();
    let mut limits = BudgetLimits::uniform(u64::MAX, 37);
    for dimension in Dimension::ALL {
        limits.dimensions[dimension.index()] = u64::MAX - generation.remaining(dimension);
    }
    let mut exact = Budget::new(limits);
    assert_eq!(
        metered_rational_antidifference(&n, &d, &mut exact, &mut Unbounded).unwrap(),
        good
    );
    assert_eq!(exact.verifier_remaining(), 37);
    assert_eq!(verification.verifier_remaining(), 41);
    let mut short = limits;
    short.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        metered_rational_antidifference(&n, &d, &mut Budget::new(short), &mut Unbounded),
        Err(RationalSummationError::Meter(_))
    ));
    let mut generation = Checkpoints::default();
    let mut verification = Checkpoints::default();
    metered_rational_antidifference(&n, &d, &mut generation, &mut verification).unwrap();
    for cancel_at in [
        1,
        generation.seen / 2,
        generation.seen - 1,
        generation.seen,
    ] {
        let mut stopped = Checkpoints {
            seen: 0,
            cancel_at: Some(cancel_at),
        };
        assert!(matches!(
            metered_rational_antidifference(&n, &d, &mut stopped, &mut Unbounded),
            Err(RationalSummationError::Meter(MeterError::Cancelled))
        ));
    }
    for cancel_at in [
        1,
        verification.seen / 2,
        verification.seen - 1,
        verification.seen,
    ] {
        let mut stopped = Checkpoints {
            seen: 0,
            cancel_at: Some(cancel_at),
        };
        assert!(matches!(
            metered_rational_antidifference(&n, &d, &mut Unbounded, &mut stopped),
            Err(RationalSummationError::Verification(
                RationalSumVerificationError::Meter(MeterError::Cancelled)
            )) | Err(RationalSummationError::Meter(MeterError::Cancelled))
        ));
    }
}
