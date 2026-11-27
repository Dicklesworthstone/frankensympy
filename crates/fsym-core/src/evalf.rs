//! Exact domain decisions for the approximate binary64 convenience evaluator.
//!
//! A rounded float cannot establish integrality, parity, or the sign of an
//! underflowed rational. Literal classification borrows the original value;
//! compound rational sources use a bounded exact lane before approximation.
//! This is not a certified enclosure or a supervised evaluation API.

use crate::{BigInt, BigRational, Constant, CoreError, Expr};
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

pub(crate) fn power(base: &Expr, exponent: &Expr) -> Result<f64, CoreError> {
    let base = input(base)?;
    let exponent = input(exponent)?;
    // Keep the convenience evaluator's explicit NaN/powf conventions. Exact
    // classification never turns an undefined source child into a rational.
    if base.value.is_nan() || exponent.value.is_nan() {
        return Ok(base.value.powf(exponent.value));
    }
    base.require_sign()?;
    if base.is_negative() {
        let Some(exact) = exponent.exact else {
            return Err(CoreError::InvalidOperation(
                "cannot establish the exact exponent of a negative-base power within the rational evaluation limits".into(),
            ));
        };
        if !exact.integer {
            return Err(CoreError::InvalidOperation(
                "a negative base to a noninteger power is not real-valued".into(),
            ));
        }
        let magnitude = base.value.abs().powf(exponent.value);
        return Ok(if exact.odd { -magnitude } else { magnitude });
    }
    if base
        .exact
        .as_ref()
        .is_some_and(|base| base.sign == Ordering::Equal)
    {
        if let Some(exact) = exponent.exact {
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
    Ok(base.value.powf(exponent.value))
}
