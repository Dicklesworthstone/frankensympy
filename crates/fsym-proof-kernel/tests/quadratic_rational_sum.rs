//! Independent quadratic-domain witnesses; no calculus/factorization generator.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::rational_sum::{
    MAX_SUM_COEFFICIENTS, RationalAntidifference, RationalSumPole, RationalSumQuadratic,
    RationalSumUpper, RationalSumVerificationError, verify_rational_antidifference,
    verify_rational_sum,
};

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}

fn quadratic(constant: BigRational, linear: BigRational) -> RationalSumQuadratic {
    RationalSumQuadratic {
        constant,
        linear,
        multiplicity: 1,
    }
}

fn example(
    constant: BigRational,
    linear: BigRational,
) -> (Vec<BigRational>, Vec<BigRational>, RationalAntidifference) {
    // F(k) = -1/(k²+b*k+c), so ΔF = (2k+b+1)/(q(k)*q(k+1)).
    let next_constant = &constant + &linear + q(1);
    let next_linear = &linear + q(2);
    let numerator = vec![&linear + q(1), q(2)];
    let denominator = vec![
        &constant * &next_constant,
        &constant * &next_linear + &linear * &next_constant,
        &constant + &linear * &next_linear + &next_constant,
        &linear + &next_linear,
        q(1),
    ];
    let certificate = RationalAntidifference {
        numerator: vec![q(-1)],
        denominator: vec![constant.clone(), linear.clone(), q(1)],
        input_poles: vec![],
        antidifference_poles: vec![],
        input_quadratics: vec![
            quadratic(constant.clone(), linear.clone()),
            quadratic(next_constant, next_linear),
        ],
        antidifference_quadratics: vec![quadratic(constant, linear)],
    };
    (numerator, denominator, certificate)
}

fn finite(
    numerator: &[BigRational],
    denominator: &[BigRational],
    certificate: &RationalAntidifference,
    lo: i64,
    hi: i64,
) -> Result<BigRational, RationalSumVerificationError> {
    verify_rational_sum(
        numerator,
        denominator,
        certificate,
        &lo.into(),
        &RationalSumUpper::Finite(hi.into()),
        &mut Unbounded,
    )
}

#[test]
fn negative_and_nonsquare_positive_discriminants_have_exact_integer_sum_domains() {
    for constant in [q(1), q(-2), BigRational::new((-1).into(), 8.into())] {
        // The last discriminant is 1/2: a square numerator alone is not a
        // rational-square proof. The middle case has genuine real poles,
        // but none at an integer, so it is still a valid integer sum.
        let (numerator, denominator, certificate) = example(constant.clone(), q(0));
        verify_rational_antidifference(&numerator, &denominator, &certificate, &mut Unbounded)
            .unwrap();
        for (lo, hi) in [(0, 0), (-7, 9), (3, 20), (8, 1), (0, -1)] {
            let left = q(lo * lo) + &constant;
            let right = q((hi + 1) * (hi + 1)) + &constant;
            assert_eq!(
                finite(&numerator, &denominator, &certificate, lo, hi).unwrap(),
                q(1) / left - q(1) / right
            );
        }
        assert_eq!(
            verify_rational_sum(
                &numerator,
                &denominator,
                &certificate,
                &0.into(),
                &RationalSumUpper::PositiveInfinity,
                &mut Unbounded,
            )
            .unwrap(),
            q(1) / constant
        );
    }
}

#[test]
fn exact_discriminants_cannot_be_rounded_into_a_double_root() {
    let epsilon = BigRational::new(3.into(), BigInt::one() << 121u32);
    let quarter = BigRational::new(1.into(), 4.into());
    for constant in [&quarter + &epsilon, &quarter - &epsilon] {
        let (numerator, denominator, certificate) = example(constant.clone(), q(-1));
        assert_eq!(
            finite(&numerator, &denominator, &certificate, 0, 0).unwrap(),
            q(0)
        );
        assert_eq!(
            verify_rational_sum(
                &numerator,
                &denominator,
                &certificate,
                &0.into(),
                &RationalSumUpper::PositiveInfinity,
                &mut Unbounded,
            )
            .unwrap(),
            q(1) / constant
        );
    }
}

#[test]
fn reducible_or_square_discriminant_quadratics_cannot_hide_rational_poles() {
    for constant in [q(0), q(-1), BigRational::new((-1).into(), 4.into())] {
        let (numerator, denominator, certificate) = example(constant, q(0));
        // These certificates have the correct rational difference identity.
        // They must still reject because the asserted quadratic factors do
        // have rational roots (including a double root when constant=0).
        assert!(matches!(
            verify_rational_antidifference(&numerator, &denominator, &certificate, &mut Unbounded),
            Err(RationalSumVerificationError::Invalid(
                "quadratic denominator has rational roots"
            ))
        ));
    }
}

#[test]
fn incomplete_mutated_duplicate_and_oversized_quadratic_witnesses_refuse() {
    let (numerator, denominator, good) = example(q(1), q(0));
    for mutation in 0..12 {
        let mut bad = good.clone();
        match mutation {
            0 => {
                bad.input_quadratics.pop();
            }
            1 => bad.input_quadratics[0].constant = q(2),
            2 => bad.input_quadratics[1].linear = q(-2),
            3 => bad.input_quadratics[0].multiplicity = 0,
            4 => bad.input_quadratics[0].multiplicity = u32::MAX,
            5 => bad.input_quadratics[1] = bad.input_quadratics[0].clone(),
            6 => bad.antidifference_quadratics.clear(),
            7 => bad.antidifference_quadratics[0].constant = q(-1),
            8 => bad.numerator[0] = q(1),
            9 => bad.denominator[0] = q(2),
            10 => {
                bad.input_quadratics = vec![bad.input_quadratics[0].clone(); MAX_SUM_COEFFICIENTS];
            }
            11 => {
                bad.input_quadratics[0].constant =
                    BigRational::from_integer(BigInt::one() << 16_385u32);
            }
            _ => unreachable!(),
        }
        assert!(
            verify_rational_antidifference(&numerator, &denominator, &bad, &mut Unbounded).is_err(),
            "quadratic witness mutation {mutation}"
        );
    }
    let mut changed_source = numerator.clone();
    changed_source[0] = q(7);
    assert!(
        verify_rational_antidifference(&changed_source, &denominator, &good, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn both_original_removable_poles_and_antidifference_endpoint_poles_are_checked() {
    let (mut numerator, mut denominator, mut certificate) = example(q(1), q(0));
    // The source is k*ΔF/k. Its hole at k=0 survives exact cancellation.
    numerator.insert(0, q(0));
    denominator.insert(0, q(0));
    certificate.input_poles.push(RationalSumPole {
        root: q(0),
        multiplicity: 1,
    });
    verify_rational_antidifference(&numerator, &denominator, &certificate, &mut Unbounded)
        .unwrap();
    assert!(finite(&numerator, &denominator, &certificate, 0, 10).is_err());
    assert!(finite(&numerator, &denominator, &certificate, 2, -1).is_err());
    assert_eq!(
        finite(&numerator, &denominator, &certificate, 1, 2).unwrap(),
        BigRational::new(2.into(), 5.into())
    );
    let (numerator, denominator, mut certificate) = example(q(1), q(0));
    // A candidate F = -k/[k*(k²+1)] is not defined at a required endpoint,
    // even though its source and its rational difference identity are valid.
    certificate.numerator.insert(0, q(0));
    certificate.denominator.insert(0, q(0));
    certificate.antidifference_poles.push(RationalSumPole {
        root: q(0),
        multiplicity: 1,
    });
    verify_rational_antidifference(&numerator, &denominator, &certificate, &mut Unbounded)
        .unwrap();
    assert!(finite(&numerator, &denominator, &certificate, 0, 2).is_err());
    assert!(finite(&numerator, &denominator, &certificate, -2, -1).is_err());
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
fn discriminant_arithmetic_and_final_publication_obey_actual_budgets_and_cancellation() {
    let (numerator, denominator, certificate) =
        example(BigRational::new((-1).into(), 8.into()), q(0));
    let run = |meter: &mut dyn BudgetMeter| {
        // The reference API deliberately takes a concrete meter. A forwarding
        // wrapper allows the same claim to exercise each real ledger below.
        struct Forward<'a>(&'a mut dyn BudgetMeter);
        impl BudgetMeter for Forward<'_> {
            fn checkpoint(&mut self) -> Result<(), MeterError> {
                self.0.checkpoint()
            }
            fn charge(&mut self, d: Dimension, n: u64) -> Result<(), MeterError> {
                self.0.charge(d, n)
            }
            fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
                self.0.charge_batch(charges)
            }
        }
        verify_rational_sum(
            &numerator,
            &denominator,
            &certificate,
            &0.into(),
            &RationalSumUpper::PositiveInfinity,
            &mut Forward(meter),
        )
    };
    let mut measured = Budget::new(BudgetLimits::uniform(u64::MAX, 37));
    assert_eq!(run(&mut measured).unwrap(), q(-8));
    let mut limits = BudgetLimits::uniform(u64::MAX, 37);
    for dimension in Dimension::ALL {
        let used = u64::MAX - measured.remaining(dimension);
        limits.dimensions[dimension.index()] = used;
    }
    let mut exact = Budget::new(limits);
    assert_eq!(run(&mut exact).unwrap(), q(-8));
    assert_eq!(exact.verifier_remaining(), 37);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut short = limits;
        short.dimensions[dimension.index()] -= 1;
        assert!(matches!(
            run(&mut Budget::new(short)),
            Err(RationalSumVerificationError::Meter(_))
        ));
    }
    let mut checkpoints = Checkpoints::default();
    run(&mut checkpoints).unwrap();
    // Initial admission, discriminant/identity arithmetic, endpoint work,
    // and the last publication checkpoint all retain typed cancellation.
    for cancel_at in [
        1,
        checkpoints.seen / 4,
        checkpoints.seen / 2,
        checkpoints.seen - 1,
        checkpoints.seen,
    ] {
        let mut stopped = Checkpoints {
            seen: 0,
            cancel_at: Some(cancel_at),
        };
        assert!(matches!(
            run(&mut stopped),
            Err(RationalSumVerificationError::Meter(MeterError::Cancelled))
        ));
    }
}
