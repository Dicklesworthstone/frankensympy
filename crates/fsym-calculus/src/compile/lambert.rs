//! Real Lambert-W bytecode evaluation through the bounded exact inverse kernel.
//!
//! A binary64 input is converted to its exact dyadic rational, never a decimal
//! approximation. The existing branch-aware rational-ball solver supplies a
//! tightly enclosed value; rounding its midpoint produces an ordinary numeric
//! approximation. Derivative jets are evaluated without the artificial W0(0)
//! pole. These floats carry no certificate: source-bound C1/C2 verification is
//! performed separately by fsym-core's original-expression interval checker.

use super::{EvalError, charge_evaluation_steps};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::{BigInt, BigRational, LambertWError, RealBall};

#[derive(Clone, Copy)]
pub(super) enum LambertOrder {
    Value,
    First,
    Second,
}

/// Adapt a possibly unsized request meter without creating or resetting any
/// allowance. The inverse kernel retains its own intrinsic work ceilings too.
struct ForwardMeter<'a, M: BudgetMeter + ?Sized>(&'a mut M);

impl<M: BudgetMeter + ?Sized> BudgetMeter for ForwardMeter<'_, M> {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.0.checkpoint()
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.0.charge(dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.0.charge_batch(charges)
    }
}

fn evaluation_error(error: LambertWError) -> EvalError {
    match error {
        LambertWError::Meter(error) => EvalError::Meter(error),
        other => EvalError::LambertW(other),
    }
}

fn exact_binary64<M: BudgetMeter + ?Sized>(
    value: f64,
    meter: &mut M,
) -> Result<BigRational, EvalError> {
    meter.checkpoint()?;
    if !value.is_finite() {
        return Err(EvalError::NonFiniteValue);
    }
    // A finite binary64 value needs at most 1075 bits in either raw
    // coefficient. Precharge their bounded construction before allocation;
    // reduction and all subsequent arithmetic use the governed kernels.
    meter.charge_batch(&[
        (Dimension::ComputeSteps, 128),
        (Dimension::MemoryBytes, 1024),
        (Dimension::AllocationCount, 8),
    ])?;
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), exponent_bits - 1075)
    };
    let (numerator, denominator) = if exponent >= 0 {
        (
            BigInt::from(significand) << exponent as u32,
            BigInt::from(1),
        )
    } else {
        (
            BigInt::from(significand),
            BigInt::from(1) << (-exponent) as u32,
        )
    };
    let numerator = if value.is_sign_negative() {
        -numerator
    } else {
        numerator
    };
    let result = BigRational::metered_new(&numerator, &denominator, &mut ForwardMeter(meter))
        .map_err(LambertWError::from)
        .map_err(evaluation_error)?;
    meter.checkpoint()?;
    Ok(result)
}

pub(super) fn evaluate_lambert<M: BudgetMeter + ?Sized>(
    argument: f64,
    branch: i32,
    order: LambertOrder,
    meter: &mut M,
) -> Result<f64, EvalError> {
    charge_evaluation_steps(meter, 1)?;
    if !argument.is_finite() {
        return Err(EvalError::NonFiniteValue);
    }
    if branch == 0 && argument == 0.0 {
        let value = match order {
            LambertOrder::Value => argument,
            LambertOrder::First => 1.0,
            LambertOrder::Second => -2.0,
        };
        charge_evaluation_steps(meter, 1)?;
        return Ok(value);
    }
    let point = RealBall::exact(exact_binary64(argument, meter)?);
    // The kernel keeps relative accuracy for tiny W0 arguments and proves the
    // complete real branch domain, including the irrational -1/e boundary.
    let enclosure = point
        .metered_lambert_w(branch, 17, &mut ForwardMeter(meter))
        .map_err(evaluation_error)?;
    let w = enclosure
        .midpoint()
        .metered_to_f64(meter)?
        .filter(|value| value.is_finite())
        .ok_or(EvalError::NonFiniteValue)?;
    let denominator = 1.0 + w;
    if (branch == 0 && denominator <= 0.0) || (branch == -1 && denominator >= 0.0) {
        return Err(EvalError::NonsmoothSource("LambertW"));
    }
    charge_evaluation_steps(meter, 16)?;
    let value = match order {
        LambertOrder::Value => w,
        LambertOrder::First | LambertOrder::Second => {
            let slope = if branch == 0 {
                // This analytic form retains W0'(0)=1 without dividing by
                // the original argument, including tiny subnormal inputs.
                (-w).exp() / denominator
            } else {
                // W_-1 is separated from zero. Dividing the bounded ratio
                // by x avoids an overflowing exp(-w) before normalization.
                (w / denominator) / argument
            };
            if matches!(order, LambertOrder::First) {
                slope
            } else {
                // W''=-(W+2)/(W+1)*(W')^2. Scale before multiplying the
                // slopes, avoiding premature square overflow/underflow.
                (-((w + 2.0) / denominator) * slope) * slope
            }
        }
    };
    if !value.is_finite() {
        return Err(EvalError::NonFiniteValue);
    }
    charge_evaluation_steps(meter, 1)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

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
            let rational = exact_binary64(value, &mut Unbounded).unwrap();
            assert_eq!(rational.to_f64(), Some(value));
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                exact_binary64(value, &mut Unbounded),
                Err(EvalError::NonFiniteValue)
            );
        }
    }
}
