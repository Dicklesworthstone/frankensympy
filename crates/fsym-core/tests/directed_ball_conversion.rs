//! The rational ball is formed exactly before its endpoints are rounded outward.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded,
};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational, RealBall};

const SIGN_BIT: u64 = 1 << 63;

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn power_of_two(exponent: u32) -> BigInt {
    BigInt::from(1i64) << exponent
}

fn exact_float(bits: u64) -> BigRational {
    assert!(f64::from_bits(bits).is_finite());
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (significand, exponent) = if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction + (1u64 << 52), exponent - 1075)
    };
    let significand = BigInt::from(significand);
    let signed = if bits & SIGN_BIT == 0 {
        significand
    } else {
        -significand
    };
    if exponent >= 0 {
        BigRational::from_integer(signed << exponent as u32)
    } else {
        BigRational::new(signed, power_of_two(exponent.unsigned_abs()))
    }
}

fn assert_tight_ball_export(ball: &RealBall, floats: (f64, f64)) {
    let (lower, upper) = floats;
    assert!(!lower.is_nan() && !upper.is_nan());
    assert!(lower <= upper);
    let exact_lower = ball.lower();
    let exact_upper = ball.upper();
    if lower.is_finite() {
        assert!(exact_float(lower.to_bits()) <= exact_lower);
        let successor = if lower.to_bits() == SIGN_BIT {
            1
        } else if lower.is_sign_negative() {
            lower.to_bits() - 1
        } else {
            lower.to_bits() + 1
        };
        if f64::from_bits(successor).is_finite() {
            assert!(exact_lower < exact_float(successor));
        }
    } else {
        assert_eq!(lower, f64::NEG_INFINITY);
        assert!(exact_lower < exact_float((-f64::MAX).to_bits()));
    }
    if upper.is_finite() {
        assert!(exact_upper <= exact_float(upper.to_bits()));
        let predecessor = if upper.to_bits() == 0 {
            SIGN_BIT | 1
        } else if upper.is_sign_negative() {
            upper.to_bits() + 1
        } else {
            upper.to_bits() - 1
        };
        if f64::from_bits(predecessor).is_finite() {
            assert!(exact_float(predecessor) < exact_upper);
        }
    } else {
        assert_eq!(upper, f64::INFINITY);
        assert!(exact_float(f64::MAX.to_bits()) < exact_upper);
    }
}

fn converted(ball: &RealBall) -> (u64, u64) {
    let floats = ball
        .metered_to_f64_bounds(&mut Unbounded)
        .expect("unbounded rational arithmetic succeeds")
        .expect("canonical ball endpoints have binary64 enclosures");
    assert_tight_ball_export(ball, floats);
    (floats.0.to_bits(), floats.1.to_bits())
}

#[test]
fn directed_endpoints_cover_the_whole_ball_instead_of_only_its_midpoint() {
    let ball = RealBall::new(rational(1, 5), rational(1, 10)).unwrap();
    assert_eq!(
        converted(&ball),
        (0x3fb9_9999_9999_9999, 0x3fd3_3333_3333_3334)
    );
    assert_eq!(
        converted(&ball.neg()),
        (0xbfd3_3333_3333_3334, 0xbfb9_9999_9999_9999)
    );
    let point = RealBall::exact(rational(1, 10));
    assert_eq!(
        converted(&point),
        (0x3fb9_9999_9999_9999, 0x3fb9_9999_9999_999a)
    );
    assert_eq!(
        converted(&RealBall::from_i64(3)),
        (3.0f64.to_bits(), 3.0f64.to_bits())
    );
}

#[test]
fn exact_cancellation_of_huge_midpoint_and_radius_precedes_float_conversion() {
    let huge = BigRational::from_integer(power_of_two(2000));
    let radius = &huge - rational(1, 10);
    let ball = RealBall::new(huge, radius).unwrap();
    assert_eq!(
        converted(&ball),
        (0x3fb9_9999_9999_9999, f64::INFINITY.to_bits())
    );
    assert_eq!(
        converted(&ball.neg()),
        (f64::NEG_INFINITY.to_bits(), 0xbfb9_9999_9999_9999)
    );
}

#[test]
fn signed_subnormal_and_exact_zero_endpoints_do_not_collapse_a_nonzero_ball() {
    let tiny = BigRational::new(1i64.into(), power_of_two(1076));
    assert_eq!(
        converted(&RealBall::new(rational(0, 1), tiny.clone()).unwrap()),
        (SIGN_BIT | 1, 1)
    );
    assert_eq!(
        converted(&RealBall::new(tiny.clone(), tiny.clone()).unwrap()),
        (0, 1)
    );
    assert_eq!(
        converted(&RealBall::new(-&tiny, tiny.clone()).unwrap()),
        (SIGN_BIT | 1, 0)
    );
    assert_eq!(converted(&RealBall::exact(-tiny)), (SIGN_BIT | 1, SIGN_BIT));
    assert_eq!(converted(&RealBall::from_i64(0)), (0, 0));
}

struct ObservedMeter {
    budget: Budget,
    checkpoints: usize,
    cancel_at: Option<usize>,
    cancelled: bool,
}

impl ObservedMeter {
    fn new(limits: BudgetLimits, cancel_at: Option<usize>) -> Self {
        Self {
            budget: Budget::new(limits),
            checkpoints: 0,
            cancel_at,
            cancelled: false,
        }
    }

    fn unlimited(cancel_at: Option<usize>) -> Self {
        Self::new(BudgetLimits::uniform(u64::MAX, 37), cancel_at)
    }

    fn spent_steps(&self) -> u64 {
        u64::MAX - self.budget.remaining(Dimension::ComputeSteps)
    }
}

impl BudgetMeter for ObservedMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.cancelled, "arithmetic continued after cancellation");
        self.budget.charge(dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.cancelled, "arithmetic continued after cancellation");
        self.budget.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.cancelled, "a cancelled conversion kept running");
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.cancelled = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn each_endpoint_formation_and_conversion_phase_consumes_the_same_ledger() {
    let ball = RealBall::new(rational(7, 3), rational(2, 5)).unwrap();
    let mut phases = ObservedMeter::unlimited(None);
    let lower = ball
        .midpoint()
        .metered_sub(ball.radius(), &mut phases)
        .unwrap();
    let after_lower_formation = phases.spent_steps();
    let upper = ball
        .midpoint()
        .metered_add(ball.radius(), &mut phases)
        .unwrap();
    let after_upper_formation = phases.spent_steps();
    lower.metered_to_f64_bounds(&mut phases).unwrap().unwrap();
    let after_lower_conversion = phases.spent_steps();
    upper.metered_to_f64_bounds(&mut phases).unwrap().unwrap();
    let total = phases.spent_steps();
    assert!(0 < after_lower_formation);
    assert!(after_lower_formation < after_upper_formation);
    assert!(after_upper_formation < after_lower_conversion);
    assert!(after_lower_conversion < total);

    for allowance in [
        after_lower_formation,
        after_upper_formation,
        after_lower_conversion,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 37);
        limits.dimensions[Dimension::ComputeSteps.index()] = allowance;
        let mut meter = ObservedMeter::new(limits, None);
        let ledger = meter.budget.ledger_identity();
        assert!(matches!(
            ball.metered_to_f64_bounds(&mut meter),
            Err(RationalArithmeticError::Meter(MeterError::Budget(
                BudgetError::Exhausted {
                    dimension: Dimension::ComputeSteps,
                    ..
                }
            )))
        ));
        assert_eq!(meter.budget.ledger_identity(), ledger);
        assert_eq!(meter.budget.remaining(Dimension::ComputeSteps), 0);
        assert_eq!(meter.budget.verifier_remaining(), 37);
    }
    let mut exact_limits = BudgetLimits::uniform(u64::MAX, 37);
    exact_limits.dimensions[Dimension::ComputeSteps.index()] = total;
    let mut exact_meter = ObservedMeter::new(exact_limits, None);
    let floats = ball
        .metered_to_f64_bounds(&mut exact_meter)
        .unwrap()
        .unwrap();
    assert_tight_ball_export(&ball, floats);
    assert_eq!(exact_meter.budget.remaining(Dimension::ComputeSteps), 0);
}

#[test]
fn allocation_refusals_retain_the_typed_operational_error() {
    let ball = RealBall::new(rational(7, 3), rational(2, 5)).unwrap();
    for refused in [Dimension::MemoryBytes, Dimension::AllocationCount] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 37);
        limits.dimensions[refused.index()] = 0;
        let mut meter = ObservedMeter::new(limits, None);
        assert!(matches!(
            ball.metered_to_f64_bounds(&mut meter),
            Err(RationalArithmeticError::Meter(MeterError::Budget(
                BudgetError::Exhausted { dimension, .. }
            ))) if dimension == refused
        ));
        assert_eq!(meter.budget.verifier_remaining(), 37);
    }
}

#[test]
fn every_safe_point_including_final_publication_can_cancel_the_complete_interval() {
    let ball = RealBall::new(
        BigRational::from_integer(power_of_two(52) + 1i64),
        rational(1, 2),
    )
    .unwrap();
    let mut baseline = ObservedMeter::unlimited(None);
    let floats = ball.metered_to_f64_bounds(&mut baseline).unwrap().unwrap();
    assert_tight_ball_export(&ball, floats);
    assert!(baseline.checkpoints > 20);
    for checkpoint in 1..=baseline.checkpoints {
        let mut meter = ObservedMeter::unlimited(Some(checkpoint));
        assert_eq!(
            ball.metered_to_f64_bounds(&mut meter),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
        assert_eq!(meter.checkpoints, checkpoint);
        assert_eq!(meter.budget.verifier_remaining(), 37);
    }
}
