//! Original-source real reciprocal-inverse values and derivative bounds.
//!
//! The checker composes u = 1/x with an independently enclosed primitive.
//! All operations use the same cumulative Work ledger. The generator's private
//! numerical derivative nodes and floating-point formulas are never inputs.

use super::{RootBoxError, Work, checked};
use crate::{BigRational, RealBall};
use fsym_budget::BudgetMeter;
use num_traits::{One, Zero};

pub(super) const CIRCULAR_ENDPOINT: &str =
    "reciprocal inverse circular argument is not strictly outside [-1, 1]";
pub(super) const SECH_ENDPOINT: &str = "asech argument is not strictly inside (0, 1)";

impl<M: BudgetMeter> Work<'_, M> {
    pub(super) fn reciprocal_inverse(
        &mut self,
        argument: &RealBall,
        name: &str,
    ) -> Result<(RealBall, RealBall), RootBoxError> {
        self.arithmetic(&[argument], &[])?;
        let lower = argument.lower();
        let upper = argument.upper();
        let one = BigRational::one();
        let primitive = match name {
            "acot" => {
                if argument.contains_zero() {
                    return Err(RootBoxError::Domain(
                        "acot argument meets its zero branch cut",
                    ));
                }
                "atan"
            }
            "acoth" => {
                if lower <= one && upper >= -BigRational::one() {
                    return Err(RootBoxError::Domain(
                        "acoth argument is not strictly outside [-1, 1]",
                    ));
                }
                "atanh"
            }
            "asec" | "acsc" => {
                if lower <= one && upper >= -BigRational::one() {
                    return Err(RootBoxError::Domain(CIRCULAR_ENDPOINT));
                }
                if name == "asec" { "acos" } else { "asin" }
            }
            "asech" => {
                if lower <= BigRational::zero() || upper >= one {
                    return Err(RootBoxError::Domain(SECH_ENDPOINT));
                }
                "acosh"
            }
            "acsch" => {
                if argument.contains_zero() {
                    return Err(RootBoxError::Domain(
                        "acsch argument interval contains zero",
                    ));
                }
                "asinh"
            }
            _ => return Err(RootBoxError::Unsupported("reciprocal inverse function")),
        };
        let inverse = self.inv(argument)?;
        let (value, inner_slope) = self.smooth_unary_rule(primitive, &inverse)?;
        let inverse_square = self.power(&inverse, 2)?;
        let slope = self.mul(&inner_slope, &inverse_square)?.neg();
        Ok((checked(value)?, checked(slope)?))
    }

    pub(super) fn reciprocal_inverse_curvature(
        &mut self,
        argument: &RealBall,
        name: &str,
        slope: &RealBall,
    ) -> Result<RealBall, RootBoxError> {
        // Called only after the complete original argument passed the smooth
        // value/first-derivative rule above.
        if matches!(name, "acot" | "acoth") {
            let square = self.power(slope, 2)?;
            let twice_argument = self.mul(&RealBall::from_i64(2), argument)?;
            return self.mul(&twice_argument, &square);
        }
        let inverse = self.inv(argument)?;
        let square = self.power(&inverse, 2)?;
        let cube = self.mul(&inverse, &square)?;
        let denominator = match name {
            "asec" | "acsc" => self.sub(&RealBall::from_i64(1), &square)?,
            "asech" => self.sub(&square, &RealBall::from_i64(1))?,
            "acsch" => self.add(&RealBall::from_i64(1), &square)?,
            _ => return Err(RootBoxError::Unsupported("reciprocal inverse curvature")),
        };
        let correction = self.div(&cube, &denominator)?;
        let leading = self.mul(&RealBall::from_i64(-2), &inverse)?;
        let logarithmic_slope = if matches!(name, "asec" | "acsc") {
            self.sub(&leading, &correction)?
        } else {
            self.add(&leading, &correction)?
        };
        self.mul(slope, &logarithmic_slope)
    }

    /// Finite continuous endpoints only. This value cannot inhabit a smooth jet.
    pub(super) fn closed_reciprocal_inverse_value(
        &mut self,
        argument: &RealBall,
        name: &str,
    ) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[argument], &[])?;
        let lower = argument.lower();
        let upper = argument.upper();
        match name {
            "asec" | "acsc" if lower >= BigRational::one() || upper <= -BigRational::one() => {}
            "asech" if lower > BigRational::zero() && upper <= BigRational::one() => {}
            "asec" | "acsc" | "asech" => {
                return Err(RootBoxError::Domain(
                    "reciprocal inverse argument is outside its closed real domain",
                ));
            }
            _ => {
                return Err(RootBoxError::Unsupported(
                    "closed reciprocal inverse function",
                ));
            }
        }
        // Prepay exact reciprocal construction and the existing bounded
        // endpoint sqrt/log/atan kernels, retaining their own hard limits.
        for _ in 0..12 {
            self.elementary(argument)?;
        }
        let value = match name {
            "asec" => argument.asec(self.precision)?,
            "acsc" => argument.acsc(self.precision)?,
            _ => argument.asech(self.precision)?,
        };
        self.step()?;
        checked(value)
    }
}
