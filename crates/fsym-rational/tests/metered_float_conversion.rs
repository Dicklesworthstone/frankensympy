//! Exact binary64 rounding and resource refusal at the governed rational boundary.
//!
//! The rounding oracle decodes neighboring IEEE values into exact dyadic rationals;
//! it does not call either floating conversion implementation to derive an expected answer.

use fsym_bigint::BigInt;
use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_rational::BigRational;

const SIGN_BIT: u64 = 1 << 63;

fn power_of_two(exponent: u32) -> BigInt {
    BigInt::from(1i64) << exponent
}

fn zero() -> BigRational {
    BigRational::from_integer(BigInt::from(0i64))
}

fn converted_bits(value: &BigRational) -> u64 {
    value
        .metered_to_f64(&mut Unbounded)
        .expect("unbounded conversion has no operational refusal")
        .expect("a canonical rational has a binary64 rounding")
        .to_bits()
}

fn assert_both_signs(numerator: BigInt, denominator: BigInt, positive_bits: u64) {
    assert_ne!(numerator, BigInt::from(0i64));
    let positive = BigRational::new(numerator.clone(), denominator.clone());
    let negative = BigRational::new(-numerator, denominator);
    assert_eq!(converted_bits(&positive), positive_bits);
    // Unlike compiled bytecode, the arithmetic conversion preserves signed underflow.
    assert_eq!(converted_bits(&negative), positive_bits | SIGN_BIT);
}

#[test]
fn overflowing_components_and_double_rounding_counterexamples_keep_exact_quotients() {
    for height in [1024, 2000, 4095] {
        let denominator = power_of_two(height);
        assert_both_signs(&denominator + 1i64, denominator, 1.0f64.to_bits());
    }

    let base = power_of_two(53);
    // Independent component rounding incorrectly gives 1 and its successor respectively.
    assert_both_signs(base.clone(), &base + 1i64, 0x3fef_ffff_ffff_ffff);
    assert_both_signs(&base + 2i64, &base + 1i64, 0x3ff0_0000_0000_0000);
}

#[test]
fn normal_midpoints_choose_even_significands_and_carry_across_binades() {
    let denominator = power_of_two(53);
    for (offset, expected) in [(1i64, 0u64), (3, 2), (5, 2), (7, 4)] {
        assert_both_signs(
            &denominator + offset,
            denominator.clone(),
            0x3ff0_0000_0000_0000 + expected,
        );
    }
    // The midpoint immediately below 2 rounds into the next binade.
    assert_both_signs(power_of_two(54) - 1i64, denominator, 2.0f64.to_bits());
    assert_both_signs(
        power_of_two(53) + 1i64,
        BigInt::from(1i64),
        0x4340_0000_0000_0000,
    );
}

#[test]
fn subnormal_ties_and_underflow_preserve_the_arithmetic_sign_bit() {
    for (numerator, denominator_power, expected) in [
        (1i64, 1074, 1),
        (1, 1075, 0),
        (3, 1075, 2),
        (5, 1075, 2),
        (7, 1075, 4),
        (1, 2000, 0),
    ] {
        assert_both_signs(
            numerator.into(),
            power_of_two(denominator_power),
            expected,
        );
    }
    // A tie between the greatest subnormal and the least normal chooses the latter.
    assert_both_signs(
        power_of_two(53) - 1i64,
        power_of_two(1075),
        0x0010_0000_0000_0000,
    );
    assert_eq!(converted_bits(&zero()), 0);
}

#[test]
fn overflow_is_decided_after_exact_rounding_at_the_last_finite_cell() {
    let max_finite = (power_of_two(53) - 1i64) << 971u32;
    let overflow_midpoint = (power_of_two(54) - 1i64) << 970u32;
    assert_both_signs(max_finite, BigInt::from(1i64), f64::MAX.to_bits());
    assert_both_signs(
        &overflow_midpoint - 1i64,
        BigInt::from(1i64),
        f64::MAX.to_bits(),
    );
    assert_both_signs(
        &overflow_midpoint * 3i64 - 1i64,
        3i64.into(),
        f64::MAX.to_bits(),
    );
    assert_both_signs(
        overflow_midpoint,
        BigInt::from(1i64),
        f64::INFINITY.to_bits(),
    );
    assert_both_signs(
        power_of_two(2050) + 1i64,
        power_of_two(1024),
        f64::INFINITY.to_bits(),
    );
}

fn exact_positive_float(bits: u64) -> BigRational {
    assert!(bits < f64::INFINITY.to_bits());
    let fraction = bits & ((1u64 << 52) - 1);
    let biased_exponent = (bits >> 52) as i32;
    let (significand, exponent) = if biased_exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), biased_exponent - 1023 - 52)
    };
    if exponent >= 0 {
        BigRational::from_integer(BigInt::from(significand) << exponent as u32)
    } else {
        BigRational::new(BigInt::from(significand), power_of_two((-exponent) as u32))
    }
}

#[test]
fn governed_results_belong_to_independently_constructed_exact_rounding_cells() {
    let base = power_of_two(53);
    let two = BigRational::from_integer(2i64.into());
    for scale in [-1075i32, -1022, -52, 0, 52, 900, 1023] {
        for numerator_offset in -2i64..=2 {
            for denominator_offset in -2i64..=2 {
                let mut numerator = &base + numerator_offset;
                let mut denominator = &base + denominator_offset;
                if scale >= 0 {
                    numerator = numerator << scale as u32;
                } else {
                    denominator = denominator << (-scale) as u32;
                }
                let exact = BigRational::new(numerator, denominator);
                let bits = converted_bits(&exact);
                let rounded = exact_positive_float(bits);
                let lower = if bits == 0 {
                    zero()
                } else {
                    (exact_positive_float(bits - 1) + rounded.clone()) / &two
                };
                let upper = (rounded + exact_positive_float(bits + 1)) / &two;
                assert!(exact >= lower && exact <= upper);
                if exact == lower || exact == upper {
                    assert_eq!(bits & 1, 0, "midpoint rounding must choose even");
                }
            }
        }
    }
}

struct ObservedMeter {
    budget: Budget,
    checkpoints: usize,
    cancel_at: Option<usize>,
    cancelled: bool,
    refused_batches: usize,
}

impl ObservedMeter {
    fn new(limits: BudgetLimits, cancel_at: Option<usize>) -> Self {
        Self {
            budget: Budget::new(limits),
            checkpoints: 0,
            cancel_at,
            cancelled: false,
            refused_batches: 0,
        }
    }

    fn unlimited(cancel_at: Option<usize>) -> Self {
        Self::new(BudgetLimits::uniform(u64::MAX, 37), cancel_at)
    }
}

impl BudgetMeter for ObservedMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.cancelled, "work continued after cancellation");
        self.budget.charge(dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.cancelled, "work continued after cancellation");
        let before = self.budget.snapshot();
        let result = self.budget.charge_batch(charges);
        if result.is_err() {
            self.refused_batches += 1;
            assert_eq!(self.budget.snapshot(), before, "a refused batch was partial");
        }
        result
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
fn every_checkpoint_including_final_publication_can_cancel_without_returning_a_float() {
    // An exact midpoint with exponent 52 avoids large scaling factors while exercising
    // division, remainder comparison, ties-to-even, and final result publication.
    let value = BigRational::new(power_of_two(53) + 1i64, 2i64.into());
    let mut baseline = ObservedMeter::unlimited(None);
    assert_eq!(
        value.metered_to_f64(&mut baseline).unwrap().unwrap(),
        (1u64 << 52) as f64,
    );
    assert!(baseline.checkpoints > 20);
    for checkpoint in 1..=baseline.checkpoints {
        let mut meter = ObservedMeter::unlimited(Some(checkpoint));
        assert_eq!(value.metered_to_f64(&mut meter), Err(MeterError::Cancelled));
        assert_eq!(meter.checkpoints, checkpoint);
        assert_eq!(meter.budget.verifier_remaining(), 37);
    }
}

#[test]
fn exact_fit_trait_object_uses_one_ledger_and_cannot_retry_with_fresh_allowances() {
    let value = BigRational::new(7i64.into(), 3i64.into());
    let mut baseline = ObservedMeter::unlimited(None);
    assert_eq!(
        value.metered_to_f64(&mut baseline).unwrap().unwrap().to_bits(),
        0x4002_aaaa_aaaa_aaab,
    );
    let mut spent = [0; DIMENSION_COUNT];
    for dimension in Dimension::ALL {
        spent[dimension.index()] = u64::MAX - baseline.budget.remaining(dimension);
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(spent[dimension.index()] > 0);
    }
    let mut budget = Budget::new(BudgetLimits {
        dimensions: spent,
        verifier_pool: 37,
    });
    let identity = budget.ledger_identity();
    let meter: &mut dyn BudgetMeter = &mut budget;
    assert_eq!(
        value.metered_to_f64(meter).unwrap().unwrap().to_bits(),
        0x4002_aaaa_aaaa_aaab,
    );
    assert_eq!(budget.snapshot().dimensions, [0; DIMENSION_COUNT]);
    assert_eq!(budget.ledger_identity(), identity);
    assert_eq!(budget.verifier_remaining(), 37);
    let exhausted = budget.snapshot();
    assert!(matches!(
        value.metered_to_f64(&mut budget),
        Err(MeterError::Budget(BudgetError::Exhausted {
            dimension: Dimension::ComputeSteps,
            ..
        }))
    ));
    assert_eq!(budget.snapshot(), exhausted);
}

#[test]
fn memory_and_allocation_refusals_are_atomic_for_each_work_unit() {
    for numerator in [7i64, -7] {
        let value = BigRational::new(numerator.into(), 3i64.into());
        for refused in [Dimension::MemoryBytes, Dimension::AllocationCount] {
            let mut limits = BudgetLimits::uniform(u64::MAX, 37);
            limits.dimensions[refused.index()] = 0;
            let mut meter = ObservedMeter::new(limits, None);
            assert!(matches!(
                value.metered_to_f64(&mut meter),
                Err(MeterError::Budget(BudgetError::Exhausted { dimension, .. }))
                    if dimension == refused
            ));
            assert_eq!(meter.refused_batches, 1);
            assert_eq!(meter.budget.verifier_remaining(), 37);
        }
    }
}

#[test]
fn metadata_extremes_need_no_coefficient_clone_and_still_check_publication() {
    let huge = power_of_two(32_768);
    let values = [
        (zero(), 0),
        (BigRational::from_integer(huge.clone()), f64::INFINITY.to_bits()),
        (
            BigRational::from_integer(-huge.clone()),
            f64::NEG_INFINITY.to_bits(),
        ),
        (BigRational::new(BigInt::from(1i64), huge.clone()), 0),
        (BigRational::new(BigInt::from(-1i64), huge), SIGN_BIT),
    ];
    for (value, expected_bits) in values {
        let mut limits = BudgetLimits::uniform(0, 37);
        limits.dimensions[Dimension::ComputeSteps.index()] = 1;
        let mut meter = ObservedMeter::new(limits, None);
        assert_eq!(
            value.metered_to_f64(&mut meter).unwrap().unwrap().to_bits(),
            expected_bits,
        );
        assert_eq!(meter.budget.snapshot().dimensions, [0; DIMENSION_COUNT]);
        assert_eq!(meter.budget.verifier_remaining(), 37);
        for checkpoint in [1, meter.checkpoints] {
            let mut cancelled = ObservedMeter::new(limits, Some(checkpoint));
            assert_eq!(
                value.metered_to_f64(&mut cancelled),
                Err(MeterError::Cancelled),
            );
        }
    }
}

#[test]
fn a_large_finite_quotient_refuses_inside_its_borrowed_coefficient_comparison() {
    let denominator = power_of_two(32_768);
    let value = BigRational::new(&denominator + 1i64, denominator);
    // The values have identical leading limbs; cancellation must reach the comparison
    // itself before a giant coefficient clone or an unmetered conversion can happen.
    let mut cancelled = ObservedMeter::unlimited(Some(128));
    assert_eq!(value.metered_to_f64(&mut cancelled), Err(MeterError::Cancelled));
    assert_eq!(cancelled.checkpoints, 128);
    assert_eq!(cancelled.budget.remaining(Dimension::MemoryBytes), u64::MAX);
    assert_eq!(cancelled.budget.remaining(Dimension::AllocationCount), u64::MAX);

    let mut limits = BudgetLimits::uniform(0, 37);
    limits.dimensions[Dimension::ComputeSteps.index()] = 64;
    let mut exhausted = ObservedMeter::new(limits, None);
    assert!(matches!(
        value.metered_to_f64(&mut exhausted),
        Err(MeterError::Budget(BudgetError::Exhausted {
            dimension: Dimension::ComputeSteps,
            ..
        }))
    ));
    assert_eq!(exhausted.budget.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(exhausted.budget.verifier_remaining(), 37);
}
