//! Principal real argument of a rational rectangle, without floating bounds.
//!
//! Away from the origin and the negative real axis, the partial derivatives
//! of theta = atan2(y, x) are x/(x^2+y^2) and -y/(x^2+y^2). Along each edge of
//! a rectangle their signs are constant. The extrema therefore occur at its
//! corners, including rectangles crossing either coordinate axis away from
//! the cut. Each corner uses certified atan and pi on the correct quadrant.

use super::{
    BallError, BigRational, RealBall, atan_rational, checked_work_bits,
    ensure_endpoint_construction_fits,
};
use num_traits::{Signed, Zero};

impl RealBall {
    /// Two composition guards and atan's two pi guards share one ceiling.
    pub const MAX_ATAN2_PRECISION_DIGITS: u32 = Self::MAX_TRANSCENDENTAL_PRECISION_DIGITS - 4;

    /// Enclose the principal real angle `atan2(self, x)` on the entire rectangle.
    ///
    /// The argument order is `(y, x)` and the value range is (-pi, pi]. Boxes
    /// containing the origin or meeting the negative-axis cut refuse. The
    /// sole cut exception is an exactly zero y with a strictly negative x:
    /// its value is pi. Derivative and certificate consumers must separately
    /// exclude that exception, since it has no continuous full neighborhood.
    ///
    /// Precision is in 1..=MAX_ATAN2_PRECISION_DIGITS decimal digits and
    /// controls numerical enclosure error, never the input variation. Exact
    /// rational endpoint and composition work is preflighted before allocation;
    /// this bounded convenience API has the same fixed-work contract as atan.
    pub fn atan2(&self, x: &Self, precision_digits: u32) -> Result<Self, BallError> {
        Self::precision_target(precision_digits)?;
        if precision_digits > Self::MAX_ATAN2_PRECISION_DIGITS {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(
                "atan2 guard precision exceeds the declared ceiling".into(),
            ));
        }
        // Adding rational endpoints may multiply their denominators. A
        // subsequent ratio and atan reduction may form up to cubic powers
        // before the existing kernel rounds onto its bounded rational grid.
        // Admit that complete composition before constructing either endpoint.
        let endpoint_height = |ball: &Self| {
            ball.midpoint()
                .height()
                .max_bits()
                .saturating_add(ball.radius().height().max_bits())
                .saturating_add(2)
        };
        checked_work_bits(
            "RealBall atan2 composition",
            [
                endpoint_height(self)
                    .max(endpoint_height(x))
                    .saturating_mul(16),
                u64::from(precision_digits).saturating_mul(16) + 256,
            ],
        )?;
        let y_lower = self.lower();
        let y_upper = self.upper();
        let x_lower = x.lower();
        let x_upper = x.upper();
        let y_has_zero = !y_lower.is_positive() && !y_upper.is_negative();
        let x_has_zero = !x_lower.is_positive() && !x_upper.is_negative();
        if y_has_zero && x_has_zero {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(
                "atan2 rectangle contains the origin".into(),
            ));
        }
        let digits = precision_digits + 2;
        if y_lower.is_zero() && y_upper.is_zero() {
            return if x_upper.is_negative() {
                Self::pi(digits)
            } else {
                Ok(Self::from_i64(0))
            };
        }
        if y_has_zero && x_lower.is_negative() {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(
                "atan2 rectangle meets the negative-axis branch cut".into(),
            ));
        }

        let mut pi = None;
        let first = endpoint(&y_lower, &x_lower, digits, &mut pi)?;
        let mut lower = first.lower();
        let mut upper = first.upper();
        for (y_index, y) in [&y_lower, &y_upper].into_iter().enumerate() {
            if y_index == 1 && y_lower == y_upper {
                continue;
            }
            for (x_index, x) in [&x_lower, &x_upper].into_iter().enumerate() {
                if (y_index == 0 && x_index == 0) || (x_index == 1 && x_lower == x_upper) {
                    continue;
                }
                let angle = endpoint(y, x, digits, &mut pi)?;
                lower = lower.min(angle.lower());
                upper = upper.max(angle.upper());
            }
        }
        checked_work_bits(
            "RealBall atan2 interval hull",
            [lower.height().max_bits(), upper.height().max_bits(), 4],
        )?;
        let two = BigRational::from_integer(2.into());
        Self::new((&lower + &upper) / &two, (upper - lower) / two)
    }
}

fn cached_pi(digits: u32, cache: &mut Option<RealBall>) -> Result<&RealBall, BallError> {
    if cache.is_none() {
        *cache = Some(RealBall::pi(digits)?);
    }
    Ok(cache.as_ref().expect("pi was populated above"))
}

fn endpoint(
    y: &BigRational,
    x: &BigRational,
    digits: u32,
    pi: &mut Option<RealBall>,
) -> Result<RealBall, BallError> {
    if y.is_zero() {
        // Origin/cut admission precedes all corner construction.
        return if x.is_positive() {
            Ok(RealBall::from_i64(0))
        } else {
            Ok(cached_pi(digits, pi)?.clone())
        };
    }
    if x.is_zero() {
        let half_pi = cached_pi(digits, pi)?.div(&RealBall::from_i64(2))?;
        return Ok(if y.is_negative() {
            half_pi.neg()
        } else {
            half_pi
        });
    }
    let x_abs = x.abs();
    let y_abs = y.abs();
    // Use a ratio <= 1 even for extreme aspect ratios. This does not replace
    // either input by a rounded value or widen its admitted domain.
    let mut acute = if y_abs <= x_abs {
        atan_rational(&(y_abs / x_abs), digits)?
    } else {
        let angle = atan_rational(&(x_abs / y_abs), digits)?;
        cached_pi(digits, pi)?
            .div(&RealBall::from_i64(2))?
            .sub(&angle)
    };
    if x.is_negative() {
        acute = cached_pi(digits, pi)?.sub(&acute);
    }
    if y.is_negative() {
        acute = acute.neg();
    }
    ensure_endpoint_construction_fits(&acute, "RealBall atan2 corner result")?;
    Ok(acute)
}
