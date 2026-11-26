//! Principal real reciprocal-inverse functions on complete exact intervals.
//!
//! The six functions compose an exact reciprocal with an existing certified
//! inverse kernel. No midpoint chooses a branch. The SymPy principal acot
//! branch is discontinuous at zero; its singleton value is pi/2, while an
//! interval meeting both sides (or a zero endpoint) is conservatively refused.
//! Values at the finite asec/acsc/asech endpoints do not imply smoothness.

use super::{BallError, RealBall, checked_work_bits, ensure_endpoint_construction_fits};
use num_traits::Zero;

impl RealBall {
    /// Leave the same guard precision as the inverse circular/hyperbolic kernels.
    pub const MAX_RECIPROCAL_INVERSE_PRECISION_DIGITS: u32 =
        Self::MAX_INVERSE_CIRCULAR_PRECISION_DIGITS;

    /// Principal real arccotangent, with the isolated value acot(0) = pi/2.
    pub fn acot(&self, precision_digits: u32) -> Result<Self, BallError> {
        admit(self, precision_digits)?;
        if self.midpoint().is_zero() && self.radius().is_zero() {
            return Self::pi(precision_digits)?.div(&Self::from_i64(2));
        }
        apply_admitted(self, precision_digits, Self::atan)
    }

    /// Principal real inverse hyperbolic cotangent on either component |x| > 1.
    pub fn acoth(&self, precision_digits: u32) -> Result<Self, BallError> {
        apply(self, precision_digits, Self::atanh)
    }

    /// Principal real arcsecant on either closed component |x| >= 1.
    pub fn asec(&self, precision_digits: u32) -> Result<Self, BallError> {
        apply(self, precision_digits, Self::acos)
    }

    /// Principal real arccosecant on either closed component |x| >= 1.
    pub fn acsc(&self, precision_digits: u32) -> Result<Self, BallError> {
        apply(self, precision_digits, Self::asin)
    }

    /// Principal real inverse hyperbolic secant on 0 < x <= 1.
    pub fn asech(&self, precision_digits: u32) -> Result<Self, BallError> {
        apply(self, precision_digits, Self::acosh)
    }

    /// Principal real inverse hyperbolic cosecant on either component x != 0.
    pub fn acsch(&self, precision_digits: u32) -> Result<Self, BallError> {
        apply(self, precision_digits, Self::asinh)
    }
}

fn admit(ball: &RealBall, precision: u32) -> Result<(), BallError> {
    RealBall::precision_target(precision)?;
    if precision > RealBall::MAX_RECIPROCAL_INVERSE_PRECISION_DIGITS {
        return Err(BallError::ArgumentOutsideDeclaredEnvelope(
            "reciprocal inverse guard precision exceeds the declared ceiling".into(),
        ));
    }
    ensure_endpoint_construction_fits(ball, "RealBall reciprocal inverse input")?;
    let height = ball
        .lower()
        .height()
        .max_bits()
        .max(ball.upper().height().max_bits());
    // Cover endpoint reciprocals and their hull before allocation. The chosen
    // inverse kernel performs its own independent, tighter composition checks.
    checked_work_bits(
        "RealBall reciprocal inverse composition",
        [
            height.saturating_mul(8),
            u64::from(precision).saturating_mul(16) + 256,
        ],
    )?;
    Ok(())
}

fn apply(
    ball: &RealBall,
    precision: u32,
    evaluate: fn(&RealBall, u32) -> Result<RealBall, BallError>,
) -> Result<RealBall, BallError> {
    admit(ball, precision)?;
    apply_admitted(ball, precision, evaluate)
}

fn apply_admitted(
    ball: &RealBall,
    precision: u32,
    evaluate: fn(&RealBall, u32) -> Result<RealBall, BallError>,
) -> Result<RealBall, BallError> {
    if ball.contains_zero() {
        // Avoid formatting an arbitrarily large input in the rejection path.
        return Err(BallError::ArgumentOutsideDeclaredEnvelope(
            "reciprocal inverse interval contains zero".into(),
        ));
    }
    let inverse = ball.inv()?;
    ensure_endpoint_construction_fits(&inverse, "RealBall reciprocal inverse argument")?;
    let result = evaluate(&inverse, precision)?;
    ensure_endpoint_construction_fits(&result, "RealBall reciprocal inverse result")?;
    Ok(result)
}
