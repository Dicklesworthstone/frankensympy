//! Principal real inverse-circular enclosures built from certified primitives.
//!
//! The half-angle identity
//! asin(x) = 2 atan(x / (1 + sqrt(1 - x^2)))
//! follows from the principal branches and half-angle formulas in DLMF
//! https://dlmf.nist.gov/4.23 and https://dlmf.nist.gov/4.21.E23.
//! Its denominator stays at least one throughout the closed domain [-1, 1].
//! Endpoint evaluation and monotonicity avoid dependency inflation across a
//! wide input interval; no floating approximation supplies an interval bound.

use super::{
    BallError, BigInt, BigRational, RealBall, checked_work_bits, ensure_endpoint_construction_fits,
};
use num_traits::{One, Zero};

impl RealBall {
    /// Four internal guard digits share the transcendental precision ceiling.
    pub const MAX_INVERSE_CIRCULAR_PRECISION_DIGITS: u32 =
        Self::MAX_TRANSCENDENTAL_PRECISION_DIGITS - 4;

    /// Enclose the principal real arcsine on the complete closed input ball.
    ///
    /// Every input value must lie in [-1, 1]. Requested precision is in
    /// 1..=MAX_INVERSE_CIRCULAR_PRECISION_DIGITS decimal digits. A positive
    /// input radius remains part of the result; precision controls numerical
    /// enclosure error, not the mathematical variation across the input.
    /// This value-only operation admits the endpoints. Derivative and root
    /// checkers separately require a neighborhood inside (-1, 1).
    pub fn asin(&self, precision_digits: u32) -> Result<Self, BallError> {
        let (lower, upper, working_digits) = inverse_circular_input(self, precision_digits)?;
        let low = asin_endpoint(&lower, working_digits)?;
        let high = if lower == upper {
            low.clone()
        } else {
            asin_endpoint(&upper, working_digits)?
        };
        interval_hull(low.lower(), high.upper())
    }

    /// Enclose the principal real arccosine on the complete closed input ball.
    ///
    /// Domain and precision limits match asin. The decreasing extension
    /// follows from acos(x) = pi/2 - asin(x), using certified pi and interval
    /// subtraction. No endpoint is clipped into the real domain.
    pub fn acos(&self, precision_digits: u32) -> Result<Self, BallError> {
        let (lower, upper, working_digits) = inverse_circular_input(self, precision_digits)?;
        if lower.is_one() && upper.is_one() {
            return Ok(Self::from_i64(0));
        }
        let half_pi = Self::pi(working_digits)?.div(&Self::from_i64(2))?;
        let low = asin_endpoint(&lower, working_digits)?;
        let high = if lower == upper {
            low.clone()
        } else {
            asin_endpoint(&upper, working_digits)?
        };
        let result = half_pi.sub(&interval_hull(low.lower(), high.upper())?);
        ensure_endpoint_construction_fits(&result, "RealBall arccosine result")?;
        Ok(result)
    }
}

fn inverse_circular_input(
    ball: &RealBall,
    precision_digits: u32,
) -> Result<(BigRational, BigRational, u32), BallError> {
    RealBall::precision_target(precision_digits)?;
    if precision_digits > RealBall::MAX_INVERSE_CIRCULAR_PRECISION_DIGITS {
        return Err(BallError::ArgumentOutsideDeclaredEnvelope(
            "inverse circular guard precision exceeds the declared ceiling".into(),
        ));
    }
    ensure_endpoint_construction_fits(ball, "RealBall inverse circular input")?;
    let lower = ball.lower();
    let upper = ball.upper();
    if lower < -BigRational::one() || upper > BigRational::one() {
        return Err(BallError::ArgumentOutsideDeclaredEnvelope(
            "inverse circular input is not contained in [-1, 1]".into(),
        ));
    }
    // Preflight exact squares, sums and quotient endpoint construction before
    // allocating their rational intermediates. sqrt has its own tighter
    // shift/integer-work guard as well.
    let height = lower.height().max_bits().max(upper.height().max_bits());
    checked_work_bits(
        "RealBall inverse circular composition",
        [
            height.saturating_mul(8),
            u64::from(precision_digits).saturating_mul(16) + 256,
        ],
    )?;
    Ok((lower, upper, precision_digits + 2))
}

fn asin_endpoint(value: &BigRational, digits: u32) -> Result<RealBall, BallError> {
    if value.is_zero() {
        return Ok(RealBall::from_i64(0));
    }
    let x = RealBall::exact(value.clone());
    let radicand = RealBall::from_i64(1).sub(&x.pow(2)?);
    let root = radicand.sqrt(digits * 4 + 64)?;
    let denominator = RealBall::from_i64(1).add(&root);
    ensure_endpoint_construction_fits(&denominator, "RealBall inverse circular denominator")?;
    let half_tangent = x.div(&denominator)?;
    ensure_endpoint_construction_fits(&half_tangent, "RealBall inverse circular argument")?;
    let result = half_tangent.atan(digits)?.mul(&RealBall::from_i64(2));
    ensure_endpoint_construction_fits(&result, "RealBall arcsine result")?;
    Ok(result)
}

fn interval_hull(lower: BigRational, upper: BigRational) -> Result<RealBall, BallError> {
    checked_work_bits(
        "RealBall inverse circular hull",
        [lower.height().max_bits(), upper.height().max_bits(), 4],
    )?;
    let two = BigRational::from_integer(BigInt::from(2));
    RealBall::new((&lower + &upper) / &two, (upper - lower) / two)
}
