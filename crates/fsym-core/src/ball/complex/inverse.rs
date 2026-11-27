//! Principal inverse functions on complete rational rectangles.
//!
//! The logarithmic identities follow DLMF sections 4.23 and 4.37:
//! https://dlmf.nist.gov/4.23 and https://dlmf.nist.gov/4.37.
//! Each logarithm and square root keeps its own principal branch. Exact
//! axes use the SymPy 1.14 branch values rather than signed-zero conventions.
//! A rectangle crossing or touching a cut refuses unless the complete
//! rectangle lies on that cut's axis. These are value enclosures, not
//! analyticity or derivative certificates.

use super::{ComplexBall, RealBall, finish, hull, preflight};
use crate::{BallError, BigRational};
use num_traits::{One, Signed, Zero};

impl ComplexBall {
    /// Enclose principal asin on the complete rectangle.
    ///
    /// Nonreal rectangles meeting either real cut outside (-1, 1) refuse.
    /// Exact real intervals, including intervals through the branch points,
    /// retain their complete value hull. In particular, asin(x) has negative
    /// imaginary part for real x > 1 and positive imaginary part for x < -1.
    /// Bounds and precision match [Self::exp]; no midpoint selects a branch.
    pub fn asin(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(self, precision_digits, "ComplexBall inverse sine")?;
        if exact_zero(self.imag()) {
            return real_inverse_circular(self.real(), precision_digits, false);
        }
        if exact_zero(self.real()) {
            return finish(RealBall::from_i64(0), self.imag().asinh(precision_digits)?);
        }
        reject_real_cuts(self, "inverse sine rectangle meets a branch cut")?;
        let square = self.pow(2)?;
        let radicand = Self::from_i64(1).checked_add(&square.neg())?;
        let root = radicand.sqrt(precision_digits)?;
        let rotated = Self::new(self.imag().neg(), self.real().clone());
        let argument = rotated.checked_add(&root)?;
        let logarithm = argument.ln(precision_digits)?;
        // -i Log(i z + sqrt(1-z^2)).
        finish(logarithm.imag, logarithm.real.neg())
    }

    /// Enclose principal acos, preserving exact zero at real x = 1.
    ///
    /// The domain envelope matches [Self::asin]. Exact real x > 1 has
    /// positive imaginary part; real x < -1 has real part pi and negative
    /// imaginary part. Off the real axis, acos(z) = pi/2 - asin(z).
    pub fn acos(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(self, precision_digits, "ComplexBall inverse cosine")?;
        if exact_zero(self.imag()) {
            return real_inverse_circular(self.real(), precision_digits, true);
        }
        let sine = self.asin(precision_digits)?;
        let half_pi = RealBall::pi(precision_digits + 2)?.div(&RealBall::from_i64(2))?;
        let value = Self::from_real(half_pi).checked_add(&sine.neg())?;
        finish(value.real, value.imag)
    }

    /// Enclose principal atan, excluding every rectangle containing +i or -i.
    ///
    /// Non-axis rectangles meeting the imaginary cuts at |Im(z)| >= 1
    /// refuse, including one-sided contact. Exact imaginary-axis intervals
    /// excluding the poles retain the principal cut values. The rotation
    /// atan(z) = -i atanh(i z) keeps both logarithms and their branches.
    pub fn atan(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(self, precision_digits, "ComplexBall inverse tangent")?;
        if exact_zero(self.imag()) {
            return finish(self.real().atan(precision_digits)?, RealBall::from_i64(0));
        }
        let rotated = Self::new(self.imag().neg(), self.real().clone());
        let value = rotated.atanh(precision_digits)?;
        finish(value.imag, value.real.neg())
    }

    /// Enclose principal asinh with its two imaginary-axis cuts.
    ///
    /// The rotation asinh(z) = -i asin(i z) retains the exact cut values:
    /// asinh(2i) has positive real part and imaginary part pi/2, while
    /// asinh(-2i) is its negative. Exact real intervals stay exactly real.
    pub fn asinh(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(
            self,
            precision_digits,
            "ComplexBall inverse hyperbolic sine",
        )?;
        if exact_zero(self.imag()) {
            return finish(self.real().asinh(precision_digits)?, RealBall::from_i64(0));
        }
        let rotated = Self::new(self.imag().neg(), self.real().clone());
        let value = rotated.asin(precision_digits)?;
        finish(value.imag, value.real.neg())
    }

    /// Enclose principal acosh with nonnegative real part.
    ///
    /// Nonreal rectangles meeting the real cut (-infinity, 1] refuse.
    /// Exact real intervals retain their whole value hull, including at
    /// either branch point. Negative real x < -1 has imaginary part +pi.
    /// The two square roots below cannot be collapsed into sqrt(z^2-1):
    /// that would choose the wrong branch on the negative real axis.
    pub fn acosh(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(
            self,
            precision_digits,
            "ComplexBall inverse hyperbolic cosine",
        )?;
        if exact_zero(self.imag()) {
            let lower = self.real().lower();
            let upper = self.real().upper();
            let one = BigRational::one();
            let minimum = if lower > one {
                lower.clone()
            } else if upper < -&one {
                -&upper
            } else {
                one.clone()
            };
            let maximum = lower.abs().max(upper.abs()).max(one);
            let real = hull(minimum, maximum)?.acosh(precision_digits)?;
            let imag = clipped_unit_interval(self.real())?.acos(precision_digits)?;
            return finish(real, imag);
        }
        if self.imag().contains_zero() && self.real().lower() <= BigRational::one() {
            return Err(outside(
                "inverse hyperbolic cosine rectangle meets a branch cut",
            ));
        }
        let minus = self.checked_add(&Self::from_i64(-1))?;
        let plus = self.checked_add(&Self::from_i64(1))?;
        let left_root = minus.sqrt(precision_digits)?;
        let right_root = plus.sqrt(precision_digits)?;
        let product = left_root.checked_mul(&right_root)?;
        let argument = self.checked_add(&product)?;
        argument.ln(precision_digits)
    }

    /// Enclose principal atanh, excluding every rectangle containing +1 or -1.
    ///
    /// Nonreal rectangles touching either real cut outside (-1, 1) refuse.
    /// Exact real intervals outside the poles retain their cut values:
    /// Im(atanh(x)) = -pi/2 for x > 1 and +pi/2 for x < -1.
    /// The difference of logarithms is never replaced by a single logarithm
    /// of their quotient, which would lose the declared branch.
    pub fn atanh(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(
            self,
            precision_digits,
            "ComplexBall inverse hyperbolic tangent",
        )?;
        let one = BigRational::one();
        if self.imag().contains_zero()
            && (self.real().contains(&one) || self.real().contains(&-&one))
        {
            return Err(outside(
                "inverse hyperbolic tangent rectangle contains a pole",
            ));
        }
        if exact_zero(self.imag()) {
            if self.real().lower() > -&one && self.real().upper() < one {
                return finish(self.real().atanh(precision_digits)?, RealBall::from_i64(0));
            }
            // Outside the poles, atanh(1/x) gives the real component
            // without subtracting nearly equal large logarithms.
            let real = self.real().inv()?.atanh(precision_digits)?;
            let half_pi = RealBall::pi(precision_digits + 2)?.div(&RealBall::from_i64(2))?;
            let imag = if self.real().is_positive() {
                half_pi.neg()
            } else {
                half_pi
            };
            return finish(real, imag);
        }
        if exact_zero(self.real()) {
            return finish(RealBall::from_i64(0), self.imag().atan(precision_digits)?);
        }
        reject_real_cuts(
            self,
            "inverse hyperbolic tangent rectangle meets a branch cut",
        )?;
        let plus = Self::from_i64(1).checked_add(self)?;
        let minus = Self::from_i64(1).checked_add(&self.neg())?;
        let first = plus.ln(precision_digits)?;
        let second = minus.ln(precision_digits)?;
        let difference = first.checked_add(&second.neg())?;
        let value = difference.checked_div(&Self::from_i64(2))?;
        finish(value.real, value.imag)
    }
}

fn exact_zero(value: &RealBall) -> bool {
    value.midpoint().is_zero() && value.radius().is_zero()
}

fn outside(reason: &str) -> BallError {
    BallError::ArgumentOutsideDeclaredEnvelope(reason.into())
}

fn reject_real_cuts(value: &ComplexBall, reason: &str) -> Result<(), BallError> {
    let one = BigRational::one();
    if value.imag().contains_zero()
        && (value.real().lower() <= -&one || value.real().upper() >= one)
    {
        return Err(outside(reason));
    }
    Ok(())
}

/// The real component of asin on the real axis is asin(clamp(x,-1,1)).
/// This is an exact branch formula, not clipping a nonreal input into a
/// real-only kernel: the omitted part is retained in the imaginary hull.
fn clipped_unit_interval(value: &RealBall) -> Result<RealBall, BallError> {
    let one = BigRational::one();
    hull(
        value.lower().max(-&one).min(one.clone()),
        value.upper().max(-&one).min(one),
    )
}

fn real_inverse_circular(
    value: &RealBall,
    digits: u32,
    cosine: bool,
) -> Result<ComplexBall, BallError> {
    let clipped = clipped_unit_interval(value)?;
    let real = if cosine {
        clipped.acos(digits)?
    } else {
        clipped.asin(digits)?
    };
    // g(x)=acosh(-x) below -1, zero inside [-1,1], -acosh(x)
    // above 1 is decreasing on the complete real axis.
    let low = asin_real_imaginary(&value.upper(), digits)?;
    let high = asin_real_imaginary(&value.lower(), digits)?;
    let imag = hull(low.lower(), high.upper())?;
    finish(real, if cosine { imag.neg() } else { imag })
}

fn asin_real_imaginary(value: &BigRational, digits: u32) -> Result<RealBall, BallError> {
    let one = BigRational::one();
    if value > &one {
        Ok(RealBall::exact(value.clone()).acosh(digits)?.neg())
    } else if value < &-one {
        RealBall::exact(-value).acosh(digits)
    } else {
        Ok(RealBall::from_i64(0))
    }
}
