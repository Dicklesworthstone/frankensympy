//! Exact domain decisions for the approximate binary64 convenience evaluator.
//!
//! A rounded float cannot establish integrality, parity, or the sign of an
//! underflowed rational. Literal classification borrows the original value;
//! compound rational sources use a bounded exact lane before approximation.
//! Extreme exact magnitudes are scaled before logarithms, square roots and
//! rational powers, so an intermediate zero or infinity does not replace a
//! representable result. Ordinary finite magnitudes keep the binary64 lane.
//! This is not a certified enclosure or a supervised evaluation API.

use crate::{BigInt, BigRational, Constant, CoreError, Expr};
use fsym_bigint::{
    metered_add, metered_cmp, metered_div_rem, metered_shift_left, metered_subtract,
};
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension};
use num_traits::{One, ToPrimitive, Zero};
use std::cmp::Ordering;

const MAX_FOLD_BITS: u64 = 4096;
const MAX_FOLD_NODES: usize = 1024;
const MAX_FOLD_DEPTH: usize = 64;

struct Exact {
    sign: Ordering,
    integer: bool,
    odd: bool,
}

pub(crate) struct Input {
    pub(crate) value: f64,
    exact: Option<Exact>,
}

impl Input {
    pub(crate) fn is_negative(&self) -> bool {
        match &self.exact {
            Some(exact) => exact.sign == Ordering::Less,
            None => !self.value.is_nan() && self.value.is_sign_negative(),
        }
    }

    pub(crate) fn require_sign(&self) -> Result<(), CoreError> {
        if self.exact.is_none() && self.value == 0.0 {
            return Err(CoreError::InvalidOperation(
                "cannot establish the exact source sign after binary64 underflow or cancellation"
                    .into(),
            ));
        }
        Ok(())
    }

    fn needs_magnitude_recovery(&self) -> bool {
        self.exact
            .as_ref()
            .is_some_and(|exact| exact.sign != Ordering::Equal)
            && (self.value == 0.0 || self.value.is_infinite())
    }
}

fn classify(numerator: &BigInt, integer: bool, value: Option<f64>) -> Result<Input, CoreError> {
    let sign = if numerator.is_zero() {
        Ordering::Equal
    } else if numerator.is_negative() {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    Ok(Input {
        value: value.ok_or_else(|| {
            CoreError::InvalidOperation("exact value cannot be converted to f64".into())
        })?,
        exact: Some(Exact {
            sign,
            integer,
            odd: integer && !(numerator & &BigInt::one()).is_zero(),
        }),
    })
}

/// Fold only exact, fully inspected rational sources. Work and intermediate
/// storage remain bounded even when the surrounding convenience API has no
/// caller meter. Exhaustion is terminal, never permission to decide the same
/// source's domain from a less accurate evaluation.
struct RationalFold {
    remaining_nodes: usize,
    meter: Budget,
}

enum FoldError {
    NotRational,
    Limit,
}

impl RationalFold {
    fn new() -> Self {
        let mut limits = BudgetLimits::uniform(1_000_000, 0);
        limits.dimensions[Dimension::MemoryBytes.index()] = 4 * 1024 * 1024;
        limits.dimensions[Dimension::AllocationCount.index()] = 8192;
        Self {
            remaining_nodes: MAX_FOLD_NODES,
            meter: Budget::new(limits),
        }
    }

    fn value(&mut self, expression: &Expr, depth: usize) -> Result<BigRational, FoldError> {
        if depth > MAX_FOLD_DEPTH || self.remaining_nodes == 0 {
            return Err(FoldError::Limit);
        }
        self.remaining_nodes -= 1;
        self.meter
            .charge(Dimension::ComputeSteps, 1)
            .map_err(|_| FoldError::Limit)?;
        let value = match expression {
            Expr::Integer(value) => {
                self.copy(value.bits())?;
                BigRational::from_integer(value.clone())
            }
            Expr::Rational(value) => {
                self.copy(value.height().max_bits())?;
                value.clone()
            }
            Expr::Add(children) | Expr::Mul(children) => {
                let sum = matches!(expression, Expr::Add(_));
                let mut value = if sum {
                    BigRational::zero()
                } else {
                    BigRational::one()
                };
                // Never short-circuit a zero product: every original child
                // must be exact and defined before this is an exact value.
                for child in children {
                    let right = self.value(child, depth + 1)?;
                    value = if sum {
                        value
                            .metered_add(&right, &mut self.meter)
                            .map_err(|_| FoldError::Limit)?
                    } else {
                        value
                            .metered_mul(&right, &mut self.meter)
                            .map_err(|_| FoldError::Limit)?
                    };
                    if value.height().max_bits() > MAX_FOLD_BITS {
                        return Err(FoldError::Limit);
                    }
                }
                value
            }
            Expr::Pow(base, exponent) => {
                let base = self.value(base, depth + 1)?;
                let exponent = self.value(exponent, depth + 1)?;
                if !exponent.is_integer() {
                    return Err(FoldError::NotRational);
                }
                let exponent = exponent.numer().to_i32().ok_or(FoldError::Limit)?;
                if base
                    .height()
                    .max_bits()
                    .checked_mul(u64::from(exponent.unsigned_abs()))
                    .ok_or(FoldError::Limit)?
                    > MAX_FOLD_BITS
                {
                    return Err(FoldError::Limit);
                }
                // This convenience evaluator deliberately retains 0**0 == 1.
                // Negative powers of zero still refuse exact folding.
                if base.is_zero() && exponent < 0 {
                    return Err(FoldError::NotRational);
                }
                base.metered_pow(exponent, &mut self.meter)
                    .map_err(|_| FoldError::Limit)?
            }
            Expr::Function(name, args) if args.len() == 1 => {
                let argument = &args[0];
                match (name.as_str(), argument) {
                    ("log" | "ln", Expr::Const(Constant::E)) => BigRational::one(),
                    ("sin" | "tan", Expr::Const(Constant::Pi)) => BigRational::zero(),
                    ("cos", Expr::Const(Constant::Pi)) => -BigRational::one(),
                    _ => {
                        let argument = self.value(argument, depth + 1)?;
                        match name.as_str() {
                            "sin" | "tan" if argument.is_zero() => BigRational::zero(),
                            "cos" | "exp" if argument.is_zero() => BigRational::one(),
                            "log" | "ln" if argument.is_one() => BigRational::zero(),
                            "sqrt" => {
                                // A bounded exact-square check, never floating
                                // recognition of a nearly integral square root.
                                let bits = argument.height().max_bits().max(1);
                                self.meter
                                    .charge(Dimension::ComputeSteps, bits * bits)
                                    .map_err(|_| FoldError::Limit)?;
                                argument.exact_sqrt().ok_or(FoldError::NotRational)?
                            }
                            _ => return Err(FoldError::NotRational),
                        }
                    }
                }
            }
            _ => return Err(FoldError::NotRational),
        };
        Ok(value)
    }

    fn copy(&mut self, bits: u64) -> Result<(), FoldError> {
        if bits > MAX_FOLD_BITS {
            return Err(FoldError::Limit);
        }
        self.meter
            .charge_batch(&[
                (Dimension::ComputeSteps, bits.div_ceil(64).max(1)),
                (Dimension::MemoryBytes, bits.div_ceil(8).max(1) * 2 + 64),
                (Dimension::AllocationCount, 2),
            ])
            .map_err(|_| FoldError::Limit)
    }

    fn recovery_source(&mut self, expression: &Expr) -> Result<BigRational, CoreError> {
        self.value(expression, 0).map_err(|error| match error {
            FoldError::Limit => recovery_limit(),
            FoldError::NotRational => CoreError::InvalidOperation(
                "bounded magnitude recovery requires an exact rational source and exponent".into(),
            ),
        })
    }
}

fn recovery_limit() -> CoreError {
    CoreError::InvalidOperation(
        "exact magnitude recovery exceeds the bounded rational evaluation limits".into(),
    )
}

/// The exact source is m * 2^k with 1 <= m < 2. Only m is rounded here;
/// scaling and exponent splitting retain k as an exact integer.
struct ScaledMagnitude {
    mantissa: f64,
    exponent: i32,
}

impl ScaledMagnitude {
    fn new(value: BigRational, fold: &mut RationalFold) -> Result<Self, CoreError> {
        let value = if value.numer().is_negative() {
            -value
        } else {
            value
        };
        let mut exponent = i32::try_from(value.numer().bits()).map_err(|_| recovery_limit())?
            - i32::try_from(value.denom().bits()).map_err(|_| recovery_limit())?;
        let mut numerator = metered_shift_left(
            value.numer(),
            u64::from(exponent.min(0).unsigned_abs()),
            &mut fold.meter,
        )
        .map_err(|_| recovery_limit())?;
        let denominator = metered_shift_left(
            value.denom(),
            u64::from(exponent.max(0).unsigned_abs()),
            &mut fold.meter,
        )
        .map_err(|_| recovery_limit())?;
        if metered_cmp(&numerator, &denominator, &mut fold.meter).map_err(|_| recovery_limit())?
            == Ordering::Less
        {
            // Normalization needs at most one bit beyond the admitted source.
            numerator =
                metered_shift_left(&numerator, 1, &mut fold.meter).map_err(|_| recovery_limit())?;
            exponent -= 1;
        }
        // Round the normalized ratio directly to its 53-bit significand.
        // Reducing a new rational here would run an unnecessary large GCD
        // after scaling, exhausting the work budget on simple decimal powers.
        let shifted =
            metered_shift_left(&numerator, 52, &mut fold.meter).map_err(|_| recovery_limit())?;
        let (significand, remainder) = metered_div_rem(&shifted, &denominator, &mut fold.meter)
            .map_err(|_| recovery_limit())?
            .ok_or_else(recovery_limit)?;
        let mut significand = significand.to_u64().ok_or_else(recovery_limit)?;
        let twice_remainder =
            metered_shift_left(&remainder, 1, &mut fold.meter).map_err(|_| recovery_limit())?;
        let remainder_order = metered_cmp(&twice_remainder, &denominator, &mut fold.meter)
            .map_err(|_| recovery_limit())?;
        if remainder_order == Ordering::Greater
            || (remainder_order == Ordering::Equal && significand & 1 != 0)
        {
            significand += 1;
        }
        let mantissa = (significand as f64) * 2.0_f64.powi(-52);
        Ok(Self { mantissa, exponent })
    }

    fn square_root(&self) -> f64 {
        let mut mantissa = self.mantissa;
        let mut exponent = self.exponent;
        if exponent.rem_euclid(2) != 0 {
            mantissa *= 2.0;
            exponent -= 1;
        }
        scale_binary(mantissa.sqrt(), exponent / 2)
    }

    fn logarithm(&self) -> f64 {
        f64::from(self.exponent).mul_add(std::f64::consts::LN_2, self.mantissa.ln())
    }
}

/// Round to the binary64 range only at the final multiplication. In
/// particular, forming 2^-1075 first would erase half-subnormal ties.
fn scale_binary(mut mantissa: f64, mut exponent: i32) -> f64 {
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    if exponent > 1023 {
        f64::INFINITY
    } else if exponent < -1075 {
        0.0
    } else if exponent < -1022 {
        (mantissa * 2.0_f64.powi(exponent + 1022)) * f64::MIN_POSITIVE
    } else {
        mantissa * 2.0_f64.powi(exponent)
    }
}

pub(crate) fn square_root(expression: &Expr, input: &Input) -> Result<f64, CoreError> {
    if !input.needs_magnitude_recovery() {
        return Ok(input.value.sqrt());
    }
    let mut fold = RationalFold::new();
    let value = fold.recovery_source(expression)?;
    Ok(ScaledMagnitude::new(value, &mut fold)?.square_root())
}

pub(crate) fn logarithm(expression: &Expr, input: &Input) -> Result<f64, CoreError> {
    if !input.needs_magnitude_recovery() {
        return Ok(input.value.ln());
    }
    let mut fold = RationalFold::new();
    let value = fold.recovery_source(expression)?;
    Ok(ScaledMagnitude::new(value, &mut fold)?.logarithm())
}

fn recovered_power_magnitude(
    base_source: &Expr,
    exponent_source: &Expr,
    base: &Input,
    exponent: &Input,
) -> Result<Option<f64>, CoreError> {
    if !base.needs_magnitude_recovery() {
        return Ok(None);
    }
    let mut fold = RationalFold::new();
    let value = fold.recovery_source(base_source)?;
    let scaled = ScaledMagnitude::new(value, &mut fold)?;
    let explicit_infinite_sign = match exponent_source {
        Expr::Const(Constant::Infinity) => Some(Ordering::Greater),
        Expr::Const(Constant::NegativeInfinity) => Some(Ordering::Less),
        _ => None,
    };
    if let Some(sign) = explicit_infinite_sign {
        let grows = (scaled.exponent > 0) == (sign == Ordering::Greater);
        return Ok(Some(if grows { f64::INFINITY } else { 0.0 }));
    }
    let Some(exact) = &exponent.exact else {
        return Err(CoreError::InvalidOperation(
            "bounded magnitude recovery requires an exact rational exponent".into(),
        ));
    };
    if exponent.value == 0.0 {
        // With <=4096-bit source coefficients, even a nonzero exponent that
        // rounds to zero changes the result by far less than one ulp at one.
        return Ok(Some(1.0));
    }
    if exponent.value.is_infinite() {
        // The source magnitude is far from one. A literal exponent outside
        // binary64 range therefore saturates without copying its coefficient.
        let grows = (scaled.exponent > 0) == (exact.sign == Ordering::Greater);
        return Ok(Some(if grows { f64::INFINITY } else { 0.0 }));
    }
    let rational_exponent = fold.recovery_source(exponent_source)?;
    let integer_scale = BigRational::from_integer(BigInt::from(scaled.exponent));
    let product = integer_scale
        .metered_mul(&rational_exponent, &mut fold.meter)
        .map_err(|_| recovery_limit())?;
    let adjacent = product
        .metered_add(&rational_exponent, &mut fold.meter)
        .map_err(|_| recovery_limit())?;
    let (lower, upper) = if exact.sign == Ordering::Less {
        (&adjacent, &product)
    } else {
        (&product, &adjacent)
    };
    // k*y and (k+1)*y enclose log2(q^y). Saturate before an intermediate
    // mantissa power can overflow and subsequently meet an underflowed scale.
    if lower
        .metered_cmp(
            &BigRational::from_integer(BigInt::from(1024)),
            &mut fold.meter,
        )
        .map_err(|_| recovery_limit())?
        != Ordering::Less
    {
        return Ok(Some(f64::INFINITY));
    }
    if upper
        .metered_cmp(
            &BigRational::from_integer(BigInt::from(-1075)),
            &mut fold.meter,
        )
        .map_err(|_| recovery_limit())?
        != Ordering::Greater
    {
        return Ok(Some(0.0));
    }
    if rational_exponent.numer().is_one() && rational_exponent.denom() == &BigInt::from(2) {
        return Ok(Some(scaled.square_root()));
    }
    // Split k*y exactly into its floor and fractional part. Rounding k*y
    // first would amplify a lost low bit into avoidable relative error.
    let (mut whole, mut remainder) =
        metered_div_rem(product.numer(), product.denom(), &mut fold.meter)
            .map_err(|_| recovery_limit())?
            .ok_or_else(recovery_limit)?;
    if remainder.is_negative() {
        whole = metered_subtract(&whole, &BigInt::one(), &mut fold.meter)
            .map_err(|_| recovery_limit())?;
        remainder = metered_add(&remainder, product.denom(), &mut fold.meter)
            .map_err(|_| recovery_limit())?;
    }
    let whole = whole.to_i32().ok_or_else(recovery_limit)?;
    let fraction = BigRational::metered_new(&remainder, product.denom(), &mut fold.meter)
        .map_err(|_| recovery_limit())?
        .metered_to_f64(&mut fold.meter)
        .map_err(|_| recovery_limit())?
        .ok_or_else(recovery_limit)?;
    let mantissa = scaled.mantissa.powf(exponent.value) * 2.0_f64.powf(fraction);
    if !mantissa.is_finite() || mantissa <= 0.0 {
        return Err(recovery_limit());
    }
    Ok(Some(scale_binary(mantissa, whole)))
}

pub(crate) fn input(expression: &Expr) -> Result<Input, CoreError> {
    // Large literal integers need no exact arithmetic or copied coefficient
    // buffer to establish parity, even beyond the compound-fold bit envelope.
    match expression {
        Expr::Integer(value) => return classify(value, true, value.to_f64()),
        Expr::Rational(value) => {
            return classify(value.numer(), value.is_integer(), value.to_f64());
        }
        _ => {}
    }
    match RationalFold::new().value(expression, 0) {
        Ok(value) => return classify(value.numer(), value.is_integer(), value.to_f64()),
        Err(FoldError::Limit) => {
            return Err(CoreError::InvalidOperation(
                "exact source classification exceeds the bounded rational evaluation limits".into(),
            ));
        }
        Err(FoldError::NotRational) => {}
    }
    Ok(Input {
        value: expression.evalf()?,
        exact: None,
    })
}

pub(crate) fn power(base_source: &Expr, exponent_source: &Expr) -> Result<f64, CoreError> {
    let base = input(base_source)?;
    let exponent = input(exponent_source)?;
    // Keep the convenience evaluator's explicit NaN/powf conventions. Exact
    // classification never turns an undefined source child into a rational.
    if base.value.is_nan() || exponent.value.is_nan() {
        return Ok(base.value.powf(exponent.value));
    }
    base.require_sign()?;
    if base.is_negative() {
        let Some(exact) = &exponent.exact else {
            return Err(CoreError::InvalidOperation(
                "cannot establish the exact exponent of a negative-base power within the rational evaluation limits".into(),
            ));
        };
        if !exact.integer {
            return Err(CoreError::InvalidOperation(
                "a negative base to a noninteger power is not real-valued".into(),
            ));
        }
        let magnitude = recovered_power_magnitude(base_source, exponent_source, &base, &exponent)?
            .unwrap_or_else(|| base.value.abs().powf(exponent.value));
        return Ok(if exact.odd { -magnitude } else { magnitude });
    }
    if base
        .exact
        .as_ref()
        .is_some_and(|base| base.sign == Ordering::Equal)
    {
        if let Some(exact) = &exponent.exact {
            return Ok(match exact.sign {
                Ordering::Less => f64::INFINITY,
                Ordering::Equal => 1.0,
                Ordering::Greater => 0.0,
            });
        }
        if exponent.value == 0.0 {
            return Err(CoreError::InvalidOperation(
                "cannot establish the exact exponent sign of a zero-base power".into(),
            ));
        }
    }
    Ok(
        recovered_power_magnitude(base_source, exponent_source, &base, &exponent)?
            .unwrap_or_else(|| base.value.powf(exponent.value)),
    )
}
