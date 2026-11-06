//! Tight IEEE binary64 enclosures of exact rationals, including operational refusal.
//!
//! Expected intervals come from exact dyadic endpoints and rational interpolation. The oracle
//! neither calls a float conversion to construct an expected endpoint nor samples an epsilon.

use fsym_bigint::BigInt;
use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
    Unbounded,
};
use fsym_rational::BigRational;

const SIGN_BIT: u64 = 1 << 63;
const FRACTION_MASK: u64 = (1 << 52) - 1;

fn power_of_two(exponent: u32) -> BigInt {
    BigInt::from(1i64) << exponent
}

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn exact_finite_float(bits: u64) -> BigRational {
    assert!(f64::from_bits(bits).is_finite());
    let fraction = bits & FRACTION_MASK;
    let biased_exponent = ((bits >> 52) & 0x7ff) as i32;
    let (mantissa, power) = if biased_exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction + (1u64 << 52), biased_exponent - 1023 - 52)
    };
    let numerator = BigInt::from(mantissa);
    let numerator = if bits & SIGN_BIT == 0 {
        numerator
    } else {
        -numerator
    };
    if power >= 0 {
        BigRational::from_integer(numerator << power as u32)
    } else {
        BigRational::new(numerator, power_of_two(power.unsigned_abs()))
    }
}

fn assert_tight_bounds(value: &BigRational, bounds: (f64, f64)) {
    let (lower, upper) = bounds;
    assert!(!lower.is_nan() && !upper.is_nan());
    assert!(lower <= upper);
    if lower.is_finite() {
        assert!(exact_finite_float(lower.to_bits()) <= *value);
    } else {
        assert_eq!(lower, f64::NEG_INFINITY);
    }
    if upper.is_finite() {
        assert!(*value <= exact_finite_float(upper.to_bits()));
    } else {
        assert_eq!(upper, f64::INFINITY);
    }
    if lower.to_bits() == upper.to_bits() {
        assert_eq!(*value, exact_finite_float(lower.to_bits()));
    } else {
        // Bit adjacency is checked independently of f64::next_up/next_down. Every inexact
        // result has the same sign at both ends, including signed underflow and overflow.
        let successor = if lower.is_sign_negative() {
            lower.to_bits() - 1
        } else {
            lower.to_bits() + 1
        };
        assert_eq!(upper.to_bits(), successor);
        if lower.is_finite() {
            assert!(exact_finite_float(lower.to_bits()) < *value);
        }
        if upper.is_finite() {
            assert!(*value < exact_finite_float(upper.to_bits()));
        }
    }
}

fn converted(value: &BigRational) -> (u64, u64) {
    let bounds = value
        .metered_to_f64_bounds(&mut Unbounded)
        .expect("unbounded enclosure does not exhaust or cancel")
        .expect("a canonical rational has a directed enclosure");
    assert_tight_bounds(value, bounds);
    (bounds.0.to_bits(), bounds.1.to_bits())
}

fn assert_both_signs(value: &BigRational, lower_bits: u64, upper_bits: u64) {
    assert!(*value > rational(0, 1));
    assert_eq!(converted(value), (lower_bits, upper_bits));
    assert_eq!(
        converted(&-value),
        (upper_bits | SIGN_BIT, lower_bits | SIGN_BIT)
    );
}

#[test]
fn exact_dyadics_keep_equal_endpoints_and_canonical_zero() {
    for bits in [
        0,
        1,
        2,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3fef_ffff_ffff_ffff,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0001,
        42.0f64.to_bits(),
        f64::MAX.to_bits(),
    ] {
        let value = exact_finite_float(bits);
        assert_eq!(converted(&value), (bits, bits));
        let unbounded = value.to_f64_bounds().unwrap();
        assert_eq!((unbounded.0.to_bits(), unbounded.1.to_bits()), (bits, bits));
        if bits != 0 {
            assert_eq!(converted(&-value), (bits | SIGN_BIT, bits | SIGN_BIT));
        }
    }
    // Canonical rationals do not carry an IEEE sign on exact zero.
    assert_eq!(converted(&exact_finite_float(SIGN_BIT)), (0, 0));
}

#[test]
fn non_dyadic_values_and_huge_coefficients_have_exact_tight_enclosures() {
    for (value, lower, upper) in [
        (
            rational(1, 10),
            0x3fb9_9999_9999_9999,
            0x3fb9_9999_9999_999a,
        ),
        (rational(1, 3), 0x3fd5_5555_5555_5555, 0x3fd5_5555_5555_5556),
        (rational(7, 3), 0x4002_aaaa_aaaa_aaaa, 0x4002_aaaa_aaaa_aaab),
    ] {
        assert_both_signs(&value, lower, upper);
    }
    let one_bits = 1.0f64.to_bits();
    for height in [1024, 2000, 4095] {
        let denominator = power_of_two(height);
        let value = BigRational::new(&denominator + 1i64, denominator);
        assert_both_signs(&value, one_bits, one_bits + 1);
    }
    // Converting the two coefficients separately would lose the direction in these cases.
    let base = power_of_two(53);
    assert_both_signs(
        &BigRational::new(base.clone(), &base + 1i64),
        one_bits - 1,
        one_bits,
    );
    assert_both_signs(
        &BigRational::new(&base + 2i64, &base + 1i64),
        one_bits,
        one_bits + 1,
    );
}

#[test]
fn subnormal_underflow_and_the_normal_boundary_preserve_tightness_and_sign() {
    for (numerator, denominator_power, lower, upper) in [
        (1i64, 1075, 0u64, 1u64),
        (3, 1075, 1, 2),
        (5, 1075, 2, 3),
        (7, 1075, 3, 4),
        (1, 2000, 0, 1),
    ] {
        assert_both_signs(
            &BigRational::new(numerator.into(), power_of_two(denominator_power)),
            lower,
            upper,
        );
    }
    assert_both_signs(
        &BigRational::new(power_of_two(53) - 1i64, power_of_two(1075)),
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
    );
}

#[test]
fn overflow_bounds_cover_values_on_both_sides_of_the_nearest_rounding_threshold() {
    let maximum = (power_of_two(53) - 1i64) << 971u32;
    let midpoint = (power_of_two(54) - 1i64) << 970u32;
    for value in [
        BigRational::from_integer(&maximum + 1i64),
        BigRational::new(&maximum * 3i64 + 1i64, 3i64.into()),
        BigRational::from_integer(&midpoint - 1i64),
        BigRational::from_integer(midpoint),
        BigRational::new(power_of_two(2050) + 1i64, power_of_two(1024)),
    ] {
        assert_both_signs(&value, f64::MAX.to_bits(), f64::INFINITY.to_bits());
    }
}

#[test]
fn exact_interpolation_between_ieee_neighbors_always_recovers_those_neighbors() {
    let mut lower_bits = vec![0, 1, FRACTION_MASK - 1, FRACTION_MASK];
    for exponent in [1u64, 2, 25, 511, 1022, 1023, 1024, 1076, 1534, 2045, 2046] {
        for fraction in [0, 1, 0x0005_5555_5555_5555, FRACTION_MASK - 1] {
            lower_bits.push((exponent << 52) | fraction);
        }
    }
    for bits in lower_bits {
        let lower = exact_finite_float(bits);
        let upper = exact_finite_float(bits + 1);
        for numerator in [1i64, 2, 3] {
            let weight = rational(numerator, 4);
            let value = &lower + (&upper - &lower) * weight;
            assert_both_signs(&value, bits, bits + 1);
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

    fn spent(&self) -> [u64; DIMENSION_COUNT] {
        let mut spent = [0; DIMENSION_COUNT];
        for dimension in Dimension::ALL {
            spent[dimension.index()] = u64::MAX - self.budget.remaining(dimension);
        }
        spent
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
            assert_eq!(self.budget.snapshot(), before);
        }
        result
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.cancelled, "a cancelled enclosure kept running");
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.cancelled = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn near_integer_midpoint() -> BigRational {
    BigRational::new(power_of_two(53) + 1i64, 2i64.into())
}

#[test]
fn every_checkpoint_can_cancel_before_either_endpoint_is_published() {
    let midpoint = near_integer_midpoint();
    for value in [
        midpoint.clone(),
        -midpoint,
        BigRational::from_integer(power_of_two(52) + 1i64),
    ] {
        let mut baseline = ObservedMeter::unlimited(None);
        let bounds = value.metered_to_f64_bounds(&mut baseline).unwrap().unwrap();
        assert_tight_bounds(&value, bounds);
        assert!(baseline.checkpoints > 20);
        for checkpoint in 1..=baseline.checkpoints {
            let mut meter = ObservedMeter::unlimited(Some(checkpoint));
            assert_eq!(
                value.metered_to_f64_bounds(&mut meter),
                Err(MeterError::Cancelled)
            );
            assert_eq!(meter.checkpoints, checkpoint);
            assert_eq!(meter.budget.verifier_remaining(), 37);
        }
    }
}

#[test]
fn the_directed_comparison_cannot_reuse_the_nearest_phases_spent_allowance() {
    let value = near_integer_midpoint();
    let mut nearest_meter = ObservedMeter::unlimited(None);
    value.metered_to_f64(&mut nearest_meter).unwrap().unwrap();
    let spent = nearest_meter.spent();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 37);
        limits.dimensions[dimension.index()] = spent[dimension.index()];
        let mut meter = ObservedMeter::new(limits, None);
        assert!(matches!(
            value.metered_to_f64_bounds(&mut meter),
            Err(MeterError::Budget(BudgetError::Exhausted { dimension: exhausted, .. }))
                if exhausted == dimension
        ));
        assert!(meter.checkpoints > nearest_meter.checkpoints);
        assert_eq!(meter.budget.verifier_remaining(), 37);
        if dimension != Dimension::ComputeSteps {
            assert_eq!(meter.refused_batches, 1);
        }
    }
}

#[test]
fn exact_fit_trait_object_charges_one_ledger_and_never_resets_it() {
    let value = rational(7, 3);
    let mut baseline = ObservedMeter::unlimited(None);
    let expected = value.metered_to_f64_bounds(&mut baseline).unwrap().unwrap();
    let spent = baseline.spent();
    let mut budget = Budget::new(BudgetLimits {
        dimensions: spent,
        verifier_pool: 37,
    });
    let identity = budget.ledger_identity();
    let meter: &mut dyn BudgetMeter = &mut budget;
    assert_eq!(value.metered_to_f64_bounds(meter), Ok(Some(expected)));
    assert_eq!(budget.snapshot().dimensions, [0; DIMENSION_COUNT]);
    assert_eq!(budget.ledger_identity(), identity);
    assert_eq!(budget.verifier_remaining(), 37);
    let exhausted = budget.snapshot();
    assert!(matches!(
        value.metered_to_f64_bounds(&mut budget),
        Err(MeterError::Budget(BudgetError::Exhausted {
            dimension: Dimension::ComputeSteps,
            ..
        }))
    ));
    assert_eq!(budget.snapshot(), exhausted);
}

#[test]
fn height_only_terminal_cases_allocate_nothing_and_still_check_publication() {
    let huge = power_of_two(32_768);
    let values = [
        (rational(0, 1), (0, 0)),
        (
            BigRational::from_integer(huge.clone()),
            (f64::MAX.to_bits(), f64::INFINITY.to_bits()),
        ),
        (
            BigRational::from_integer(-huge.clone()),
            (f64::NEG_INFINITY.to_bits(), (-f64::MAX).to_bits()),
        ),
        (BigRational::new(1i64.into(), huge.clone()), (0, 1)),
        (
            BigRational::new((-1i64).into(), huge),
            (SIGN_BIT | 1, SIGN_BIT),
        ),
    ];
    for (value, expected) in values {
        let mut limits = BudgetLimits::uniform(0, 37);
        limits.dimensions[Dimension::ComputeSteps.index()] = 2;
        let mut meter = ObservedMeter::new(limits, None);
        let bounds = value.metered_to_f64_bounds(&mut meter).unwrap().unwrap();
        assert_eq!((bounds.0.to_bits(), bounds.1.to_bits()), expected);
        assert_tight_bounds(&value, bounds);
        assert_eq!(meter.budget.snapshot().dimensions, [0; DIMENSION_COUNT]);
        assert_eq!(meter.budget.verifier_remaining(), 37);
        for checkpoint in 1..=meter.checkpoints {
            let mut cancelled = ObservedMeter::new(limits, Some(checkpoint));
            assert_eq!(
                value.metered_to_f64_bounds(&mut cancelled),
                Err(MeterError::Cancelled),
            );
        }
    }
}
