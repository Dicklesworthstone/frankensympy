//! Exact coefficient, independent-verifier, and public integration regressions.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::hermite::{HermiteError, hermite_reduce, metered_hermite_reduce};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_integral::{
    RationalIntegralReduction, RationalIntegralVerificationError,
    verify_rational_integral_reduction,
};
use num_traits::{One, Zero};

fn poly(values: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(
        Symbol::new("x"),
        values
            .iter()
            .map(|&v| BigRational::from_integer(v.into()))
            .collect(),
    )
}

fn fraction(numerator: &UnivariatePoly, denominator: &UnivariatePoly) -> Expr {
    numerator.to_expr() * Expr::pow(denominator.to_expr(), Expr::from_i64(-1))
}

fn check_reduction(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
) -> RationalIntegralReduction {
    let result = hermite_reduce(numerator, denominator).expect("Hermite reduction");
    verify_rational_integral_reduction(
        &numerator.coeffs,
        &denominator.coeffs,
        &result,
        &mut Unbounded,
    )
    .expect("independent coefficient verification");
    let encoded = serde_json::to_vec(&result).unwrap();
    let decoded: RationalIntegralReduction = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(result, decoded);
    verify_rational_integral_reduction(
        &numerator.coeffs,
        &denominator.coeffs,
        &decoded,
        &mut Unbounded,
    )
    .unwrap();
    result
}

#[test]
fn integrates_high_degree_repeated_poles_without_irreducible_factorization() {
    // G = (x^2 + 3x + 2)/(x^5 + x + 1)^m. Supply G' as an expanded
    // rational function, with no recognizable syntactic substitution.
    for coefficients in [
        &[1, 1, 0, 0, 0, 1][..],
        &[3, -2, 0, 1][..],
        &[1, 1, 1, 0, 1][..],
    ] {
        let base = poly(coefficients);
        for multiplicity in 1..=3u32 {
            let a = poly(&[2, 3, 1]);
            let b = base.pow(multiplicity).unwrap();
            let numerator = a
                .derivative()
                .mul(&b)
                .unwrap()
                .sub(&a.mul(&b.derivative()).unwrap())
                .unwrap();
            let denominator = b.pow(2).unwrap();
            let result = check_reduction(&numerator, &denominator);
            assert!(result.remainder_numerator.iter().all(Zero::is_zero));
            let actual_num = UnivariatePoly::new(a.gen_sym.clone(), result.rational_numerator);
            let actual_den = UnivariatePoly::new(a.gen_sym.clone(), result.rational_denominator);
            assert_eq!(actual_num.mul(&b).unwrap(), a.mul(&actual_den).unwrap());
            let input = fraction(&numerator, &denominator);
            let antiderivative = fsym_calculus::integrate(&input, &a.gen_sym)
                .expect("public integration must reach the Hermite lane");
            // Inspect the actual public Expr, independently of the reduction
            // certificate: these proper rational primitives vanish at infinity,
            // so A/B is the uniquely normalized expected expression.
            let cleared = fsym_simplify::try_cancel(&(antiderivative * b.to_expr())).unwrap();
            let public_numerator = UnivariatePoly::from_expr(&cleared, &a.gen_sym).unwrap();
            assert_eq!(public_numerator, a);
        }
    }
}

#[test]
fn repeated_pole_reduction_keeps_nonzero_square_free_remainders() {
    // 1/(x^2+1)^2 = (x/(2(x^2+1)))' + 1/(2(x^2+1)).
    let den = poly(&[1, 0, 1]).pow(2).unwrap();
    let result = check_reduction(&poly(&[1]), &den);
    let half = BigRational::new(1.into(), 2.into());
    assert_eq!(
        result.rational_numerator,
        vec![BigRational::zero(), half.clone()]
    );
    assert_eq!(result.rational_denominator, poly(&[1, 0, 1]).coeffs);
    assert_eq!(result.remainder_numerator, vec![half]);
    assert_eq!(result.remainder_denominator, poly(&[1, 0, 1]).coeffs);
    assert!(
        !verify_rational_integral_reduction(
            &poly(&[1]).coeffs,
            &den.coeffs,
            &result,
            &mut Unbounded
        )
        .unwrap()
        .is_rational_antiderivative()
    );

    // An unresolved quintic remainder must not disappear just because a
    // rational portion of the integral was successfully computed.
    let quintic = poly(&[1, 1, 0, 0, 0, 1]);
    let result = check_reduction(&poly(&[1]), &quintic.pow(2).unwrap());
    assert!(!result.remainder_numerator.iter().all(Zero::is_zero));
    let integrand = fraction(&poly(&[1]), &quintic.pow(2).unwrap());
    assert!(fsym_calculus::integrate::antiderivative(&integrand, &quintic.gen_sym).is_none());
}

#[test]
fn reductions_handle_cancellation_improper_fractions_and_multiple_pole_orders() {
    let factors = [poly(&[-1, 1]), poly(&[1, 0, 1]), poly(&[1, 1, 0, 1])];
    let denominator = factors[0]
        .pow(3)
        .unwrap()
        .mul(&factors[1].pow(2).unwrap())
        .unwrap()
        .mul(&factors[2])
        .unwrap();
    for numerator in [
        poly(&[0]),
        poly(&[1]),
        poly(&[-2, 3, 0, 5]),
        denominator
            .mul(&poly(&[1, 2, 3]))
            .unwrap()
            .add(&poly(&[1, 1]))
            .unwrap(),
        denominator.clone(),
    ] {
        check_reduction(&numerator, &denominator);
        let scalar = UnivariatePoly::new(
            Symbol::new("x"),
            vec![BigRational::new((-7).into(), 13.into())],
        );
        let scaled = check_reduction(
            &numerator.mul(&scalar).unwrap(),
            &denominator.mul(&scalar).unwrap(),
        );
        let original = check_reduction(&numerator, &denominator);
        assert_eq!(scaled.rational_numerator, original.rational_numerator);
        assert_eq!(scaled.rational_denominator, original.rational_denominator);
        assert_eq!(scaled.remainder_numerator, original.remainder_numerator);
        assert_eq!(scaled.remainder_denominator, original.remainder_denominator);
    }
}

#[test]
fn verifier_rejects_mutated_derivatives_poles_square_freeness_and_inputs() {
    let numerator = poly(&[1]);
    let denominator = poly(&[1, 0, 1]).pow(2).unwrap();
    let good = check_reduction(&numerator, &denominator);
    for field in 0..8 {
        let mut bad = good.clone();
        let coefficients = match field {
            0 => &mut bad.rational_numerator,
            1 => &mut bad.rational_denominator,
            2 => &mut bad.remainder_numerator,
            3 => &mut bad.remainder_denominator,
            4 => &mut bad.rational_pole_cofactor,
            5 => &mut bad.remainder_pole_cofactor,
            6 => &mut bad.square_free_u,
            _ => &mut bad.square_free_v,
        };
        coefficients[0] += BigRational::one();
        assert!(
            verify_rational_integral_reduction(
                &numerator.coeffs,
                &denominator.coeffs,
                &bad,
                &mut Unbounded
            )
            .is_err(),
            "certificate field {field}"
        );
    }
    assert!(
        verify_rational_integral_reduction(
            &poly(&[2]).coeffs,
            &denominator.coeffs,
            &good,
            &mut Unbounded
        )
        .is_err()
    );
    let mut bad = good.clone();
    bad.rational_denominator = vec![BigRational::zero()];
    assert!(
        verify_rational_integral_reduction(
            &numerator.coeffs,
            &denominator.coeffs,
            &bad,
            &mut Unbounded
        )
        .is_err()
    );
    bad = good.clone();
    bad.rational_numerator.push(BigRational::zero());
    assert!(
        verify_rational_integral_reduction(
            &numerator.coeffs,
            &denominator.coeffs,
            &bad,
            &mut Unbounded
        )
        .is_err()
    );
    bad.rational_numerator.clear();
    assert!(
        verify_rational_integral_reduction(
            &numerator.coeffs,
            &denominator.coeffs,
            &bad,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct CancelMeter {
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CancelMeter {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        assert_ne!(amount, 0);
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn both_generator_and_verifier_can_stop_without_publishing_a_reduction() {
    let numerator = poly(&[1, 1]);
    let denominator = poly(&[1, 0, 1]).pow(2).unwrap();
    let mut generation = CancelMeter::default();
    let mut verification = CancelMeter::default();
    metered_hermite_reduce(&numerator, &denominator, &mut generation, &mut verification).unwrap();
    for checkpoint in 1..=generation.checkpoints {
        let mut cancelled = CancelMeter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_hermite_reduce(&numerator, &denominator, &mut cancelled, &mut Unbounded),
            Err(HermiteError::Meter(MeterError::Cancelled))
        ));
    }
    for checkpoint in 1..=verification.checkpoints {
        let mut cancelled = CancelMeter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_hermite_reduce(&numerator, &denominator, &mut Unbounded, &mut cancelled),
            Err(HermiteError::Verification(
                RationalIntegralVerificationError::Meter(MeterError::Cancelled)
            ))
        ));
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(metered_hermite_reduce(&numerator, &denominator, &mut tiny, &mut Unbounded).is_err());
    let mut reserved_verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_hermite_reduce(
            &numerator,
            &denominator,
            &mut Unbounded,
            &mut reserved_verifier
        )
        .is_err()
    );
    assert!(hermite_reduce(&numerator, &poly(&[0])).is_err());
    let huge = UnivariatePoly::new(
        Symbol::new("x"),
        vec![BigRational::from_integer(BigInt::one() << 4_097u32)],
    );
    assert!(hermite_reduce(&huge, &denominator).is_err());
}
