//! Real Lambert-W bytecode and stable first/second derivatives.
//!
//! The binary64 input is retained as an exact dyadic rational before the
//! existing certified real-branch kernel admits its domain. In particular,
//! the rounded binary64 value `-1.0 / E` lies below the real branch point and
//! must not be clamped to -1. Returned binary64 values are approximations;
//! the surrounding compiled program does not acquire an enclosure receipt.

use super::{EvalError, charge_evaluation_steps};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::ball::{LambertWError, RealBall};
use fsym_core::{BigInt, BigRational};

#[derive(Clone, Copy)]
pub(super) enum LambertOrder {
    Value,
    First,
    Second,
}

/// Adapt an unsized caller meter without creating or replenishing a budget.
struct CallerMeter<'a, M: BudgetMeter + ?Sized>(&'a mut M);

impl<M: BudgetMeter + ?Sized> BudgetMeter for CallerMeter<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.0.charge(dimension, amount)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.0.checkpoint()
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.0.charge_batch(charges)
    }
}

fn failed(error: LambertWError) -> EvalError {
    match error {
        LambertWError::Meter(error)
        | LambertWError::Arithmetic(RationalArithmeticError::Meter(error)) => {
            EvalError::Meter(error)
        }
        LambertWError::Arithmetic(RationalArithmeticError::SizeOverflow) => {
            EvalError::DimensionOverflow
        }
        LambertWError::Arithmetic(RationalArithmeticError::AllocationFailure) => {
            EvalError::AllocationFailure
        }
        other => EvalError::LambertW(other),
    }
}

fn arithmetic(error: RationalArithmeticError) -> EvalError {
    failed(error.into())
}

fn finite_rational<M: BudgetMeter + ?Sized>(
    rational: &BigRational,
    meter: &mut M,
) -> Result<f64, EvalError> {
    rational
        .metered_to_f64(meter)?
        .filter(|value| value.is_finite())
        .ok_or(EvalError::NonFiniteValue)
}

fn exact_input<M: BudgetMeter>(value: f64, meter: &mut M) -> Result<BigRational, EvalError> {
    meter.checkpoint()?;
    if !value.is_finite() {
        return Err(EvalError::NonFiniteValue);
    }
    // A finite double needs at most 1075 denominator bits or 1024 numerator
    // bits. Precharge the bounded shifts and their temporary storage before
    // constructing the exact ratio; normalization is separately metered.
    meter.charge_batch(&[
        (Dimension::ComputeSteps, 128),
        (Dimension::MemoryBytes, 1024),
        (Dimension::AllocationCount, 8),
    ])?;
    meter.checkpoint()?;
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mantissa, shift) = if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), exponent - 1023 - 52)
    };
    let mut numerator = BigInt::from(mantissa);
    let mut denominator = BigInt::from(1);
    if shift >= 0 {
        numerator = numerator << shift as u32;
    } else {
        denominator = denominator << shift.unsigned_abs();
    }
    if value.is_sign_negative() {
        numerator = -numerator;
    }
    BigRational::metered_new(&numerator, &denominator, meter).map_err(arithmetic)
}

pub(super) fn evaluate_lambert<M: BudgetMeter + ?Sized>(
    value: f64,
    branch: i32,
    order: LambertOrder,
    meter: &mut M,
) -> Result<f64, EvalError> {
    let result = evaluate(value, branch, order, &mut CallerMeter(meter))?;
    // Lambert-W bytecode retains its finite real output contract even when
    // the surrounding scalar program does not request the finite-only lane.
    if !result.is_finite() {
        return Err(EvalError::NonFiniteValue);
    }
    meter.checkpoint()?;
    Ok(result)
}

fn evaluate<M: BudgetMeter>(
    value: f64,
    branch: i32,
    order: LambertOrder,
    meter: &mut M,
) -> Result<f64, EvalError> {
    charge_evaluation_steps(meter, 1)?;
    if !value.is_finite() {
        return Err(EvalError::NonFiniteValue);
    }
    if branch == 0 && value == 0.0 {
        return Ok(match order {
            LambertOrder::Value => value,
            LambertOrder::First => 1.0,
            LambertOrder::Second => -2.0,
        });
    }
    let input = exact_input(value, meter)?;
    // 24 decimal places resolve 1+W at the closest admissible binary64
    // argument above -1/e. Curvature on W-1 also has cancellation in W+2
    // near its inflection at -2/e^2. Select extra precision before the one
    // kernel call in that small window; this is only an accuracy choice,
    // and exact rational domain admission and all resource caps still apply.
    let digits = if branch == -1
        && matches!(order, LambertOrder::Second)
        && (-0.28..=-0.26).contains(&value)
    {
        32
    } else {
        24
    };
    // The native kernel retains relative accuracy for tiny principal
    // arguments, including binary64 subnormals.
    let enclosure = RealBall::exact(input)
        .metered_lambert_w(branch, digits, meter)
        .map_err(failed)?;
    evaluate_midpoint(value, branch, order, enclosure.midpoint(), meter)
}

// Only successful real-kernel admission reaches this arithmetic in production.
// Keeping it separate also lets tests cover representable derivative tails
// without promising that every such inverse solve fits the kernel's budget.
fn evaluate_midpoint<M: BudgetMeter>(
    value: f64,
    branch: i32,
    order: LambertOrder,
    w: &BigRational,
    meter: &mut M,
) -> Result<f64, EvalError> {
    let rounded = finite_rational(w, meter)?;
    if matches!(order, LambertOrder::Value) {
        return Ok(rounded);
    }

    // Form 1+w before rounding: rounding w alone near -1 would lose several
    // digits of the small denominator and corrupt both derivatives.
    let one = BigRational::from_integer(BigInt::from(1));
    let denominator = w.metered_add(&one, meter).map_err(arithmetic)?;
    let denominator_f64 = finite_rational(&denominator, meter)?;
    if (branch == 0 && denominator_f64 <= 0.0) || (branch == -1 && denominator_f64 >= 0.0) {
        return Err(EvalError::NonsmoothSource("LambertW"));
    }
    charge_evaluation_steps(meter, 24)?;
    let slope = if branch == 0 {
        // exp(-w)/(1+w) is regular at zero and finite for real W0. Its
        // numerator cannot overflow because w > -1 on this domain.
        (-rounded).exp() / denominator_f64
    } else {
        // exp(-w) can overflow for W-1(-1e-308) even though its derivative
        // remains finite. W/(x*(1+W)) in this order avoids that intermediate.
        let ratio = w.metered_div(&denominator, meter).map_err(arithmetic)?;
        finite_rational(&ratio, meter)? / value
    };
    if matches!(order, LambertOrder::First) {
        return Ok(slope);
    }
    let two = BigRational::from_integer(BigInt::from(2));
    let numerator = w.metered_add(&two, meter).map_err(arithmetic)?;
    let ratio = numerator
        .metered_div(&denominator, meter)
        .map_err(arithmetic)?;
    let ratio = finite_rational(&ratio, meter)?;
    // W'' = -(W')^2*(W+2)/(W+1). Do not first form exp(-2W), x*x,
    // or (1+W)^3. The final finite-real policy rejects actual overflow.
    Ok(-slope * (slope * ratio))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

    #[test]
    fn midpoint_jets_avoid_overflow_and_preserve_representable_tail_slopes() {
        // Independent high-precision W values are rounded only for this
        // arithmetic test. They are not accepted inverse enclosures: the
        // production entry point must first succeed in the certified kernel.
        let lower_w = -715.769_566_923_421_3_f64;
        assert!((-lower_w).exp().is_infinite());
        let w = exact_input(lower_w, &mut Unbounded).unwrap();
        let slope =
            evaluate_midpoint(-1e-308, -1, LambertOrder::First, &w, &mut Unbounded).unwrap();
        let expected = -1.001_399_052_290_802_5e308;
        assert!(slope.is_finite());
        assert!((slope - expected).abs() <= 3e-15 * expected.abs());
        let curvature =
            evaluate_midpoint(-1e-308, -1, LambertOrder::Second, &w, &mut Unbounded).unwrap();
        assert!(curvature.is_infinite() && curvature.is_sign_negative());

        let w = exact_input(703.227_033_104_770_2, &mut Unbounded).unwrap();
        let slope =
            evaluate_midpoint(f64::MAX, 0, LambertOrder::First, &w, &mut Unbounded).unwrap();
        assert!(slope > 0.0 && slope.is_subnormal());
        assert!((slope - 5.554_785_652_925_27e-309).abs() <= 200.0 * f64::from_bits(1));
    }

    #[test]
    fn exact_input_conversion_preserves_every_binary64_regime() {
        for value in [
            -f64::MAX,
            -1.0,
            -f64::MIN_POSITIVE,
            -f64::from_bits(1),
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            0.1,
            1.0,
            f64::MAX,
        ] {
            let rational = exact_input(value, &mut Unbounded).unwrap();
            assert_eq!(rational.to_f64(), Some(value));
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                exact_input(value, &mut Unbounded),
                Err(EvalError::NonFiniteValue)
            );
        }
    }
}
