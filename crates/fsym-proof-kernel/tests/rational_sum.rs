//! Hand-built certificates and adversarial checks, without the generator crate.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::rational_sum::{
    MAX_SUM_COEFFICIENTS, RationalAntidifference, RationalSumPole, RationalSumUpper,
    RationalSumVerificationError, verify_rational_antidifference, verify_rational_sum,
};

fn q(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

fn coefficients(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|&n| q(n)).collect()
}

fn certificate() -> RationalAntidifference {
    // N/D = 1/(k+1)^2 - 1/(k+2)^2, F = -1/(k+1)^2.
    RationalAntidifference {
        numerator: coefficients(&[-1]),
        denominator: coefficients(&[1, 2, 1]),
        input_poles: vec![
            RationalSumPole {
                root: q(-1),
                multiplicity: 2,
            },
            RationalSumPole {
                root: q(-2),
                multiplicity: 2,
            },
        ],
        antidifference_poles: vec![RationalSumPole {
            root: q(-1),
            multiplicity: 2,
        }],
    }
}

#[test]
fn independently_checks_difference_and_finite_infinite_endpoints() {
    let n = coefficients(&[3, 2]);
    let d = coefficients(&[4, 12, 13, 6, 1]);
    let good = certificate();
    verify_rational_antidifference(&n, &d, &good, &mut Unbounded).unwrap();
    for upper in 0i64..20 {
        let actual = verify_rational_sum(
            &n,
            &d,
            &good,
            &BigInt::from(0),
            &RationalSumUpper::Finite(BigInt::from(upper)),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(
            actual,
            q(1) - BigRational::new(1.into(), ((upper + 2) * (upper + 2)).into())
        );
    }
    assert_eq!(
        verify_rational_sum(
            &n,
            &d,
            &good,
            &BigInt::from(0),
            &RationalSumUpper::PositiveInfinity,
            &mut Unbounded,
        )
        .unwrap(),
        q(1)
    );
    assert!(
        verify_rational_sum(
            &n,
            &d,
            &good,
            &BigInt::from(-2),
            &RationalSumUpper::PositiveInfinity,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn rejects_mutated_identity_poles_multiplicity_and_noncanonical_inputs() {
    let n = coefficients(&[3, 2]);
    let d = coefficients(&[4, 12, 13, 6, 1]);
    let good = certificate();
    for mutation in 0..9 {
        let mut bad = good.clone();
        match mutation {
            0 => bad.numerator[0] = q(1),
            1 => bad.denominator[0] = q(2),
            2 => {
                bad.input_poles.pop();
            }
            3 => bad.input_poles[0].multiplicity = 1,
            4 => bad.input_poles[0].root = q(-3),
            5 => bad.antidifference_poles[0].multiplicity = u32::MAX,
            6 => bad.denominator.push(q(0)),
            7 => bad.numerator.clear(),
            8 => bad.numerator = vec![q(1); MAX_SUM_COEFFICIENTS + 1],
            _ => unreachable!(),
        }
        assert!(
            verify_rational_antidifference(&n, &d, &bad, &mut Unbounded).is_err(),
            "mutation {mutation}"
        );
    }
    assert!(
        verify_rational_antidifference(&n, &coefficients(&[0]), &good, &mut Unbounded).is_err()
    );
    assert!(
        verify_rational_antidifference(&coefficients(&[4, 2]), &d, &good, &mut Unbounded).is_err()
    );
}

#[test]
fn original_removable_poles_and_reversed_ranges_are_preserved() {
    // k/k is identically one on its domain, but it is undefined at zero.
    let input = coefficients(&[0, 1]);
    let good = RationalAntidifference {
        numerator: coefficients(&[0, 1]),
        denominator: coefficients(&[1]),
        input_poles: vec![RationalSumPole {
            root: q(0),
            multiplicity: 1,
        }],
        antidifference_poles: vec![],
    };
    let finite = |lower: i64, upper: i64| {
        verify_rational_sum(
            &input,
            &input,
            &good,
            &lower.into(),
            &RationalSumUpper::Finite(upper.into()),
            &mut Unbounded,
        )
    };
    assert_eq!(finite(1, 10).unwrap(), q(10));
    assert_eq!(finite(3, 0).unwrap(), q(-2));
    assert_eq!(finite(0, -1).unwrap(), q(0));
    assert!(finite(-1, 1).is_err());
    assert!(finite(2, -1).is_err());
    assert!(
        verify_rational_sum(
            &input,
            &input,
            &good,
            &1.into(),
            &RationalSumUpper::PositiveInfinity,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn fractional_poles_do_not_exclude_integer_points_between_them() {
    // 1/(2k+1)^2 - 1/(2k+3)^2 has no integer pole, including at negative k.
    let n = coefficients(&[8, 8]);
    let d = coefficients(&[9, 48, 88, 64, 16]);
    let good = RationalAntidifference {
        numerator: coefficients(&[-1]),
        denominator: coefficients(&[1, 4, 4]),
        input_poles: vec![
            RationalSumPole {
                root: BigRational::new((-1).into(), 2.into()),
                multiplicity: 2,
            },
            RationalSumPole {
                root: BigRational::new((-3).into(), 2.into()),
                multiplicity: 2,
            },
        ],
        antidifference_poles: vec![RationalSumPole {
            root: BigRational::new((-1).into(), 2.into()),
            multiplicity: 2,
        }],
    };
    assert_eq!(
        verify_rational_sum(
            &n,
            &d,
            &good,
            &0.into(),
            &RationalSumUpper::PositiveInfinity,
            &mut Unbounded
        )
        .unwrap(),
        q(1)
    );
    assert_eq!(
        verify_rational_sum(
            &n,
            &d,
            &good,
            &(-1).into(),
            &RationalSumUpper::Finite(0.into()),
            &mut Unbounded
        )
        .unwrap(),
        BigRational::new(8.into(), 9.into())
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
fn verifier_cancellation_is_typed_and_never_returns_a_sum() {
    assert!(matches!(
        verify_rational_sum(
            &coefficients(&[3, 2]),
            &coefficients(&[4, 12, 13, 6, 1]),
            &certificate(),
            &0.into(),
            &RationalSumUpper::PositiveInfinity,
            &mut Cancelled,
        ),
        Err(RationalSumVerificationError::Meter(MeterError::Cancelled))
    ));
}
