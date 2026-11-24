//! Independent signed arithmetic and resource checks for the direct binary-scaling boundary.

use fsym_bigint::{BigInt, MeteredShiftError, metered_multiply, metered_pow, metered_shift_left};
use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
    Unbounded,
};
use proptest::prelude::*;

struct ObservedMeter {
    budget: Budget,
    checkpoints: usize,
    cancel_at: Option<usize>,
    cancelled: bool,
    refused_batches: usize,
}

impl ObservedMeter {
    fn unlimited(cancel_at: Option<usize>) -> Self {
        Self::new(BudgetLimits::uniform(u64::MAX, 31), cancel_at)
    }

    fn new(limits: BudgetLimits, cancel_at: Option<usize>) -> Self {
        Self {
            budget: Budget::new(limits),
            checkpoints: 0,
            cancel_at,
            cancelled: false,
            refused_batches: 0,
        }
    }

    fn spent(&self, dimension: Dimension) -> u64 {
        u64::MAX - self.budget.remaining(dimension)
    }

    fn exact_limits(&self) -> BudgetLimits {
        let mut dimensions = [0; DIMENSION_COUNT];
        for dimension in Dimension::ALL {
            dimensions[dimension.index()] = self.spent(dimension);
        }
        BudgetLimits {
            dimensions,
            verifier_pool: 31,
        }
    }
}

impl BudgetMeter for ObservedMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.cancelled, "a cancelled shift continued charging");
        self.budget.charge(dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.cancelled, "a cancelled shift continued charging");
        let before = self.budget.snapshot();
        let result = self.budget.charge_batch(charges);
        if result.is_err() {
            self.refused_batches += 1;
            assert_eq!(self.budget.snapshot(), before, "refused batch was partial");
        }
        result
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.cancelled, "a cancelled shift continued executing");
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
fn signed_digit_carries_match_the_separate_substrate_shift() {
    for bytes in [
        vec![1],
        vec![u8::MAX; 4],
        vec![u8::MAX; 17],
        vec![0x55, 0xaa, 0, 0, 0xff, 0, 0x80, 1],
        vec![0xff; 129],
    ] {
        let magnitude = BigInt::from_bytes_le(&bytes);
        for value in [magnitude.clone(), -magnitude] {
            for shift in [
                0u32, 1, 7, 8, 31, 32, 33, 63, 64, 65, 95, 96, 127, 1024, 1075,
            ] {
                let expected = &value << shift;
                let shifted = metered_shift_left(&value, u64::from(shift), &mut Unbounded)
                    .expect("bounded exact shift");
                assert_eq!(shifted, expected, "{value} shifted by {shift}");
                assert_eq!(shifted.bits(), value.bits() + u64::from(shift));
                assert_eq!(&shifted >> shift, value);
            }
        }
    }
    // This output crosses two input digits and adds a new most-significant digit.
    assert_eq!(
        metered_shift_left(&BigInt::from(u64::MAX), 1, &mut Unbounded).unwrap(),
        BigInt::from_str_radix("36893488147419103230", 10).unwrap()
    );
}

#[test]
fn canonical_zero_ignores_huge_shift_without_storage_and_still_checks_cancellation() {
    let zero = BigInt::from(0i64);
    for shift in [0, 64, u64::from(u32::MAX), u64::MAX] {
        let mut meter = ObservedMeter::new(BudgetLimits::uniform(0, 31), None);
        assert_eq!(
            metered_shift_left(&zero, shift, &mut meter),
            Ok(zero.clone())
        );
        assert_eq!(meter.budget.snapshot().dimensions, [0; DIMENSION_COUNT]);
        assert_eq!(meter.budget.verifier_remaining(), 31);
        for checkpoint in 1..=meter.checkpoints {
            let mut cancelled = ObservedMeter::new(BudgetLimits::uniform(0, 31), Some(checkpoint));
            assert_eq!(
                metered_shift_left(&zero, shift, &mut cancelled),
                Err(MeteredShiftError::Meter(MeterError::Cancelled))
            );
        }
    }
}

#[test]
fn unrepresentable_bit_height_fails_before_storage_admission() {
    let value = BigInt::from(-3i64);
    let mut meter = ObservedMeter::unlimited(None);
    assert_eq!(
        metered_shift_left(&value, u64::MAX, &mut meter),
        Err(MeteredShiftError::SizeOverflow)
    );
    assert_eq!(meter.spent(Dimension::MemoryBytes), 0);
    assert_eq!(meter.spent(Dimension::AllocationCount), 0);
    for checkpoint in 1..=meter.checkpoints {
        let mut cancelled = ObservedMeter::unlimited(Some(checkpoint));
        assert_eq!(
            metered_shift_left(&value, u64::MAX, &mut cancelled),
            Err(MeteredShiftError::Meter(MeterError::Cancelled))
        );
    }
}

#[test]
fn half_gigabyte_shift_request_is_refused_before_reserving_a_buffer() {
    let mut limits = BudgetLimits::uniform(u64::MAX, 31);
    limits.dimensions[Dimension::MemoryBytes.index()] = 4096;
    let mut meter = ObservedMeter::new(limits, None);
    assert!(matches!(
        metered_shift_left(&BigInt::from(1i64), u64::from(u32::MAX), &mut meter),
        Err(MeteredShiftError::Meter(MeterError::Budget(
            BudgetError::Exhausted {
                dimension: Dimension::MemoryBytes,
                requested: 536_870_912,
                remaining: 4096,
            }
        )))
    ));
    assert_eq!(meter.refused_batches, 1);
    assert_eq!(meter.budget.remaining(Dimension::MemoryBytes), 4096);
    assert_eq!(meter.budget.remaining(Dimension::AllocationCount), u64::MAX);
    assert_eq!(
        meter.budget.remaining(Dimension::ComputeSteps),
        u64::MAX - 1
    );
    assert_eq!(meter.budget.verifier_remaining(), 31);
}

#[test]
fn every_safe_point_including_output_copy_and_publication_can_cancel() {
    let value = -(BigInt::from(1i64) << 129u32) + 17i64;
    for shift in [0, 1, 32, 127] {
        let mut baseline = ObservedMeter::unlimited(None);
        assert!(metered_shift_left(&value, shift, &mut baseline).is_ok());
        assert!(baseline.checkpoints > 10);
        for checkpoint in 1..=baseline.checkpoints {
            let mut cancelled = ObservedMeter::unlimited(Some(checkpoint));
            assert_eq!(
                metered_shift_left(&value, shift, &mut cancelled),
                Err(MeteredShiftError::Meter(MeterError::Cancelled))
            );
            assert_eq!(cancelled.checkpoints, checkpoint);
            assert_eq!(cancelled.budget.verifier_remaining(), 31);
            assert!(
                cancelled.spent(Dimension::ComputeSteps) <= baseline.spent(Dimension::ComputeSteps)
            );
        }
    }
}

#[test]
fn exact_and_one_short_allowances_include_final_native_repacking() {
    let value = (BigInt::from(1i64) << 97u32) - 1i64;
    for shift in [0, 31, 32, 63, 128] {
        let mut baseline = ObservedMeter::unlimited(None);
        let expected = metered_shift_left(&value, shift, &mut baseline).unwrap();
        let exact_limits = baseline.exact_limits();
        let mut exact = Budget::new(exact_limits);
        let identity = exact.ledger_identity();
        assert_eq!(metered_shift_left(&value, shift, &mut exact), Ok(expected));
        assert_eq!(exact.snapshot().dimensions, [0; DIMENSION_COUNT]);
        assert_eq!(exact.ledger_identity(), identity);
        assert_eq!(exact.verifier_remaining(), 31);
        let depleted = exact.snapshot();
        assert!(matches!(
            metered_shift_left(&value, shift, &mut exact),
            Err(MeteredShiftError::Meter(MeterError::Budget(
                BudgetError::Exhausted {
                    dimension: Dimension::ComputeSteps,
                    ..
                }
            )))
        ));
        assert_eq!(exact.snapshot(), depleted);
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut limits = exact_limits;
            limits.dimensions[dimension.index()] -= 1;
            let mut short = ObservedMeter::new(limits, None);
            assert!(matches!(
                metered_shift_left(&value, shift, &mut short),
                Err(MeteredShiftError::Meter(MeterError::Budget(BudgetError::Exhausted {
                    dimension: refused,
                    ..
                }))) if refused == dimension
            ));
            assert_eq!(short.budget.verifier_remaining(), 31);
        }
    }
}

#[test]
fn direct_scaling_avoids_power_construction_and_multiplication_work() {
    // Compare exact outputs and intrinsic charges only. This is maintenance evidence, not a
    // timing benchmark, an algorithm-crossover profile, or an external performance claim.
    for height in [128u32, 1024, 4096] {
        for shift in [31u32, 64, 1024, 1075] {
            let value = (BigInt::from(1i64) << height) - 1i64;
            let mut direct = ObservedMeter::unlimited(None);
            let shifted = metered_shift_left(&value, u64::from(shift), &mut direct).unwrap();
            let mut composed = ObservedMeter::unlimited(None);
            let factor = metered_pow(&BigInt::from(2i64), shift, &mut composed).unwrap();
            let product = metered_multiply(&value, &factor, &mut composed).unwrap();
            assert_eq!(shifted, product);
            for dimension in [
                Dimension::ComputeSteps,
                Dimension::MemoryBytes,
                Dimension::AllocationCount,
            ] {
                assert!(
                    direct.spent(dimension) < composed.spent(dimension),
                    "{dimension} at {height}/{shift}"
                );
            }
            let output_digits = shifted.bits().div_ceil(32);
            assert!(direct.spent(Dimension::ComputeSteps) <= 2 * output_digits + 1);
            assert!(direct.spent(Dimension::AllocationCount) <= 2);
        }
    }
}

proptest! {
    #[test]
    fn broad_signed_sources_match_substrate_and_additive_shift_composition(
        source in proptest::collection::vec(any::<u8>(), 0..256),
        left in 0u32..2048,
        right in 0u32..2048,
    ) {
        let value = BigInt::from_signed_bytes_be(&source);
        let shifted = metered_shift_left(&value, u64::from(left + right), &mut Unbounded).unwrap();
        prop_assert_eq!(&shifted, &(&value << (left + right)));
        let intermediate = metered_shift_left(&value, u64::from(left), &mut Unbounded).unwrap();
        prop_assert_eq!(shifted, metered_shift_left(&intermediate, u64::from(right), &mut Unbounded).unwrap());
    }
}
