//! Principal real inverse-hyperbolic enclosures from exact endpoint arithmetic.
//!
//! The logarithmic formulas in DLMF https://dlmf.nist.gov/4.37.iv,
//! restricted to their real domains, are increasing. Evaluating the two
//! endpoints therefore retains interval variation without dependency inflation.
//! Odd symmetry is applied before evaluating asinh/atanh at a negative point;
//! in particular, asinh never subtracts nearly equal large quantities.

use super::{
    BallError, BigRational, RealBall, checked_work_bits, ensure_endpoint_construction_fits,
};
use num_traits::{One, Signed, Zero};

impl RealBall {
    /// Guard digits reserved for the square-root/logarithm compositions.
    pub const MAX_INVERSE_HYPERBOLIC_PRECISION_DIGITS: u32 =
        Self::MAX_TRANSCENDENTAL_PRECISION_DIGITS - 4;

    /// Enclose the principal real asinh on the complete input ball.
    ///
    /// All finite real arguments are admissible within the arithmetic work
    /// envelope. Precision controls numerical enclosure error, not the
    /// mathematical variation across a positive-width input interval.
    pub fn asinh(&self, precision_digits: u32) -> Result<Self, BallError> {
        let (lower, upper, digits) = input(self, precision_digits)?;
        endpoints(&lower, &upper, digits, asinh_endpoint)
    }

    /// Enclose the principal real acosh for an entire ball contained in [1, inf).
    ///
    /// The value at one is exactly zero. Derivative/certificate consumers
    /// must separately establish a smooth neighborhood strictly above one.
    pub fn acosh(&self, precision_digits: u32) -> Result<Self, BallError> {
        let (lower, upper, digits) = input(self, precision_digits)?;
        if lower < BigRational::one() {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(
                "acosh input is not contained in [1, infinity)".into(),
            ));
        }
        endpoints(&lower, &upper, digits, acosh_endpoint)
    }

    /// Enclose principal real atanh for a ball strictly contained in (-1, 1).
    /// Endpoints at either pole, or any portion outside the domain, refuse.
    pub fn atanh(&self, precision_digits: u32) -> Result<Self, BallError> {
        let (lower, upper, digits) = input(self, precision_digits)?;
        if lower <= -BigRational::one() || upper >= BigRational::one() {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(
                "atanh input is not strictly contained in (-1, 1)".into(),
            ));
        }
        endpoints(&lower, &upper, digits, atanh_endpoint)
    }
}

fn input(ball: &RealBall, precision: u32) -> Result<(BigRational, BigRational, u32), BallError> {
    RealBall::precision_target(precision)?;
    if precision > RealBall::MAX_INVERSE_HYPERBOLIC_PRECISION_DIGITS {
        return Err(BallError::ArgumentOutsideDeclaredEnvelope(
            "inverse hyperbolic guard precision exceeds the declared ceiling".into(),
        ));
    }
    ensure_endpoint_construction_fits(ball, "RealBall inverse hyperbolic input")?;
    let lower = ball.lower();
    let upper = ball.upper();
    let height = lower.height().max_bits().max(upper.height().max_bits());
    // Preflight the rational squares/sums/ratios before their intermediates
    // are allocated. Each existing sqrt/ln kernel enforces its own envelope.
    checked_work_bits(
        "RealBall inverse hyperbolic composition",
        [
            height.saturating_mul(8),
            u64::from(precision).saturating_mul(16) + 256,
        ],
    )?;
    Ok((lower, upper, precision + 2))
}

fn endpoints(
    lower: &BigRational,
    upper: &BigRational,
    digits: u32,
    evaluate: fn(&BigRational, u32) -> Result<RealBall, BallError>,
) -> Result<RealBall, BallError> {
    let low = evaluate(lower, digits)?;
    if lower == upper {
        return Ok(low);
    }
    let high = evaluate(upper, digits)?;
    let lower = low.lower();
    let upper = high.upper();
    checked_work_bits(
        "RealBall inverse hyperbolic hull",
        [lower.height().max_bits(), upper.height().max_bits(), 4],
    )?;
    let two = BigRational::from_integer(2.into());
    RealBall::new((&lower + &upper) / &two, (upper - lower) / two)
}

fn asinh_endpoint(value: &BigRational, digits: u32) -> Result<RealBall, BallError> {
    if value.is_zero() {
        return Ok(RealBall::from_i64(0));
    }
    let x = RealBall::exact(value.abs());
    let square_plus_one = x.pow(2)?.add(&RealBall::from_i64(1));
    let root = square_plus_one.sqrt(digits * 4 + 64)?;
    let argument = x.add(&root);
    ensure_endpoint_construction_fits(&argument, "RealBall asinh logarithm")?;
    let result = argument.ln(digits)?;
    ensure_endpoint_construction_fits(&result, "RealBall asinh result")?;
    Ok(if value.is_negative() {
        result.neg()
    } else {
        result
    })
}

fn acosh_endpoint(value: &BigRational, digits: u32) -> Result<RealBall, BallError> {
    if value.is_one() {
        return Ok(RealBall::from_i64(0));
    }
    let x = RealBall::exact(value.clone());
    let square_minus_one = x.pow(2)?.sub(&RealBall::from_i64(1));
    let root = square_minus_one.sqrt(digits * 4 + 64)?;
    let argument = x.add(&root);
    ensure_endpoint_construction_fits(&argument, "RealBall acosh logarithm")?;
    let result = argument.ln(digits)?;
    ensure_endpoint_construction_fits(&result, "RealBall acosh result")?;
    Ok(result)
}

fn atanh_endpoint(value: &BigRational, digits: u32) -> Result<RealBall, BallError> {
    if value.is_zero() {
        return Ok(RealBall::from_i64(0));
    }
    let x = value.abs();
    let ratio = (BigRational::one() + &x) / (BigRational::one() - x);
    let argument = RealBall::exact(ratio);
    ensure_endpoint_construction_fits(&argument, "RealBall atanh logarithm")?;
    let result = argument.ln(digits)?.div(&RealBall::from_i64(2))?;
    ensure_endpoint_construction_fits(&result, "RealBall atanh result")?;
    Ok(if value.is_negative() {
        result.neg()
    } else {
        result
    })
}
