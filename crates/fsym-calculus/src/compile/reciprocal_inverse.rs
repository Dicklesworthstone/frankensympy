//! Principal real reciprocal-inverse functions and their analytic jets.
//!
//! The value domains and open smooth domains are checked before evaluation.
//! Scaled formulas avoid a reciprocal, square or cube overflowing before a
//! representable value or derivative is obtained. These binary64 kernels
//! produce approximations; original-source ball verification is independent.

use super::{EvalError, charge_evaluation_steps};
use fsym_budget::BudgetMeter;

#[derive(Clone, Copy)]
pub(super) enum ReciprocalInverse {
    Acot,
    Acoth,
    Asec,
    Acsc,
    Asech,
    Acsch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ReciprocalInverseOrder {
    Value,
    First,
    Second,
}

impl ReciprocalInverse {
    fn name(self) -> &'static str {
        match self {
            Self::Acot => "acot",
            Self::Acoth => "acoth",
            Self::Asec => "asec",
            Self::Acsc => "acsc",
            Self::Asech => "asech",
            Self::Acsch => "acsch",
        }
    }
}

/// Evaluate one already-lowered real operation under the caller's ledger.
/// The original value and every private jet retain their own domain even if
/// a later multiplication by zero would otherwise conceal a singularity.
pub(super) fn evaluate_reciprocal_inverse<M: BudgetMeter + ?Sized>(
    value: f64,
    function: ReciprocalInverse,
    order: ReciprocalInverseOrder,
    smooth_only: bool,
    meter: &mut M,
) -> Result<f64, EvalError> {
    charge_evaluation_steps(meter, 32)?;
    if !value.is_finite() {
        return Err(EvalError::NonFiniteValue);
    }
    let magnitude = value.abs();
    let smooth = smooth_only || order != ReciprocalInverseOrder::Value;
    let result = match function {
        ReciprocalInverse::Acot => {
            // SymPy's principal acot is atan(1/x) off zero, with acot(0)=pi/2.
            // Its negative-side jump at zero is not a differentiable point.
            if smooth && value == 0.0 {
                return Err(EvalError::NonsmoothSource(function.name()));
            }
            match order {
                ReciprocalInverseOrder::Value => {
                    if value == 0.0 {
                        std::f64::consts::FRAC_PI_2
                    } else if magnitude <= 1.0 {
                        std::f64::consts::FRAC_PI_2.copysign(value) - value.atan()
                    } else {
                        value.recip().atan()
                    }
                }
                ReciprocalInverseOrder::First | ReciprocalInverseOrder::Second => {
                    if magnitude <= 1.0 {
                        let denominator = 1.0 + value * value;
                        if order == ReciprocalInverseOrder::First {
                            -denominator.recip()
                        } else {
                            (2.0 * value / denominator) / denominator
                        }
                    } else {
                        let reciprocal = value.recip();
                        let scaled = reciprocal / (1.0 + reciprocal * reciprocal);
                        if order == ReciprocalInverseOrder::First {
                            -reciprocal * scaled
                        } else {
                            (2.0 * reciprocal * scaled) * scaled
                        }
                    }
                }
            }
        }
        ReciprocalInverse::Acoth => {
            if magnitude <= 1.0 {
                return Err(EvalError::NonFiniteValue);
            }
            match order {
                ReciprocalInverseOrder::Value => {
                    if magnitude <= 2.0 {
                        // Subtract the pole before rounding a reciprocal.
                        // Otherwise atanh amplifies the reciprocal's error.
                        (0.5 * (2.0 / (magnitude - 1.0)).ln_1p()).copysign(value)
                    } else {
                        value.recip().atanh()
                    }
                }
                ReciprocalInverseOrder::First => {
                    // -1/((|x|-1)(|x|+1)); no rounded x*x-1 near a pole.
                    -((magnitude - 1.0).recip() / (magnitude + 1.0))
                }
                ReciprocalInverseOrder::Second => {
                    // 2*x/(x*x-1)^2, with every division performed before
                    // the next potentially overflowing denominator product.
                    let curvature = (((2.0 * (magnitude / (magnitude + 1.0))) / (magnitude + 1.0))
                        / (magnitude - 1.0))
                        / (magnitude - 1.0);
                    curvature.copysign(value)
                }
            }
        }
        ReciprocalInverse::Asec | ReciprocalInverse::Acsc => {
            if magnitude < 1.0 {
                return Err(EvalError::NonFiniteValue);
            }
            if smooth && magnitude == 1.0 {
                return Err(EvalError::NonsmoothSource(function.name()));
            }
            let is_secant = matches!(function, ReciprocalInverse::Asec);
            let reciprocal = magnitude.recip();
            // Retain the distance from |x|=1 in both values and derivatives;
            // acos/asin of a rounded reciprocal lose accuracy near that edge.
            let norm = ((magnitude - 1.0) / magnitude).sqrt() * (1.0 + reciprocal).sqrt();
            if order == ReciprocalInverseOrder::Value {
                let signed_reciprocal = reciprocal.copysign(value);
                if is_secant {
                    norm.atan2(signed_reciprocal)
                } else {
                    signed_reciprocal.atan2(norm)
                }
            } else {
                let scaled = reciprocal / norm;
                let derivative = if order == ReciprocalInverseOrder::First {
                    scaled * reciprocal
                } else {
                    let curvature = ((2.0 - reciprocal * reciprocal) * scaled) * scaled * scaled;
                    -curvature.copysign(value)
                };
                if is_secant { derivative } else { -derivative }
            }
        }
        ReciprocalInverse::Asech => {
            if value <= 0.0 || value > 1.0 {
                return Err(EvalError::NonFiniteValue);
            }
            if smooth && value == 1.0 {
                return Err(EvalError::NonsmoothSource(function.name()));
            }
            let norm = (1.0 - value).sqrt() * (1.0 + value).sqrt();
            match order {
                // acosh(1/x)=log(1+sqrt(1-x*x))-log(x), including tiny x
                // whose reciprocal overflows although the value is finite.
                ReciprocalInverseOrder::Value => norm.ln_1p() - value.ln(),
                ReciprocalInverseOrder::First => -(norm.recip() / value),
                ReciprocalInverseOrder::Second => {
                    // A fused numerator retains curvature near x=1/sqrt(2).
                    let numerator = (-2.0 * value).mul_add(value, 1.0);
                    ((((numerator / norm) / norm) / norm) / value) / value
                }
            }
        }
        ReciprocalInverse::Acsch => {
            if value == 0.0 {
                return Err(EvalError::NonFiniteValue);
            }
            match order {
                ReciprocalInverseOrder::Value => {
                    let result = if magnitude <= 1.0 {
                        magnitude.hypot(1.0).ln_1p() - magnitude.ln()
                    } else {
                        magnitude.recip().asinh()
                    };
                    result.copysign(value)
                }
                ReciprocalInverseOrder::First => -(magnitude.hypot(1.0).recip() / magnitude),
                ReciprocalInverseOrder::Second => {
                    let curvature = if magnitude <= 1.0 {
                        let norm = magnitude.hypot(1.0);
                        ((((1.0 + 2.0 * magnitude * magnitude) / norm) / norm) / norm)
                            / magnitude
                            / magnitude
                    } else {
                        let reciprocal = magnitude.recip();
                        let scaled = reciprocal / reciprocal.hypot(1.0);
                        ((2.0 + reciprocal * reciprocal) * scaled) * scaled * scaled
                    };
                    curvature.copysign(value)
                }
            }
        }
    };
    meter.checkpoint()?;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(EvalError::NonFiniteValue)
    }
}
