//! Complex elementary functions on complete rational rectangles.
//!
//! Logarithm uses log(z) = log(x^2 + y^2)/2 + i atan2(y, x).
//! Square root uses u^2 = (|z| + x)/2 and v^2 = (|z| - x)/2,
//! with u >= 0 and the sign of v fixed by the principal branch.
//! All endpoints are rational enclosures; no floating approximation controls
//! a domain decision, a branch decision, or an error bound.

use super::{
    BallError, BigInt, BigRational, ComplexBall, RealBall, checked_work_bits,
    ensure_endpoint_construction_fits,
};
use num_traits::{Signed, ToPrimitive, Zero};

impl ComplexBall {
    /// The component transcendental kernels and branch calculations share
    /// this ceiling, including every internal guard digit.
    pub const MAX_ELEMENTARY_PRECISION_DIGITS: u32 =
        RealBall::MAX_TRANSCENDENTAL_PRECISION_DIGITS - 8;

    /// Enclose exp(x + i y) = exp(x) (cos(y) + i sin(y)) over the
    /// complete input rectangle. Monotone exact-endpoint exponentials
    /// preserve real input variation; precision controls the component
    /// kernels' rounding, not the width of the whole image.
    ///
    /// Precision is in 1..=MAX_ELEMENTARY_PRECISION_DIGITS. Input,
    /// intermediates and output are subject to the intrinsic arithmetic
    /// envelope, not a caller-owned resource or cancellation budget.
    pub fn exp(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(
            self,
            precision_digits,
            "ComplexBall exponential composition",
        )?;
        let working = precision_digits + 4;
        let magnitude = exponential_range(self.real(), working)?;
        let phase = Self::new(self.imag().cos(working)?, self.imag().sin(working)?);
        let result = Self::from_real(magnitude).checked_mul(&phase)?;
        finish(result.real, result.imag)
    }

    /// Enclose sin(x + i y) = sin(x) cosh(y) + i cos(x) sinh(y) over
    /// the complete rectangle. A positive input width remains in the
    /// result, including the imaginary component's exponential variation.
    /// Precision and intrinsic arithmetic bounds match [Self::exp].
    pub fn sin(&self, precision_digits: u32) -> Result<Self, BallError> {
        self.circular(precision_digits, false)
    }

    /// Enclose cos(x + i y) = cos(x) cosh(y) - i sin(x) sinh(y) over
    /// the complete rectangle, under the same bounds as [Self::exp].
    pub fn cos(&self, precision_digits: u32) -> Result<Self, BallError> {
        self.circular(precision_digits, true)
    }

    fn circular(&self, precision_digits: u32, cosine: bool) -> Result<Self, BallError> {
        preflight(self, precision_digits, "ComplexBall circular composition")?;
        let working = precision_digits + 4;
        let sine = self.real().sin(working)?;
        let cosine_value = self.real().cos(working)?;
        if self.imag().midpoint().is_zero() && self.imag().radius().is_zero() {
            return finish(
                if cosine { cosine_value } else { sine },
                RealBall::from_i64(0),
            );
        }
        let positive = exponential_range(self.imag(), working)?;
        let negative = exponential_range(&self.imag().neg(), working)?;
        // Bound the sum/difference and division by two before constructing
        // their rational intermediates. The two exponentials can have
        // unrelated endpoint denominators.
        let height = |value: &RealBall| {
            value
                .midpoint()
                .height()
                .max_bits()
                .max(value.radius().height().max_bits())
        };
        checked_work_bits(
            "ComplexBall hyperbolic combination",
            [height(&positive)
                .saturating_add(height(&negative))
                .saturating_add(2)
                .saturating_mul(8)],
        )?;
        let two = RealBall::from_i64(2);
        let cosh = positive.add(&negative).div(&two)?;
        let sinh = positive.sub(&negative).div(&two)?;
        // Each product has three radius terms. Include their transient
        // denominator products and both final component allocations.
        checked_work_bits(
            "ComplexBall circular products",
            [height(&sine)
                .max(height(&cosine_value))
                .saturating_add(height(&cosh).max(height(&sinh)))
                .saturating_add(2)
                .saturating_mul(16)],
        )?;
        let real_factor = if cosine { &cosine_value } else { &sine };
        let imag_factor = if cosine { &sine } else { &cosine_value };
        let real = real_factor.mul(&cosh);
        let imag = imag_factor.mul(&sinh);
        finish(real, if cosine { imag.neg() } else { imag })
    }

    /// Enclose the principal power `z^w = exp(w * Log(z))` over both complete
    /// input rectangles, using the logarithm branch with argument in (-pi, pi].
    ///
    /// An exact signed-i32 integer exponent uses algebraic powering and does
    /// not impose a logarithmic branch cut. An exactly zero base is admitted
    /// only for a strictly positive, exactly real exponent enclosure. Other
    /// noninteger powers require a base rectangle excluding zero and obey
    /// the same negative-axis branch restrictions as [`Self::ln`]. In
    /// particular, a negative real point has its principal complex root,
    /// never the real odd root selected by an unrelated real-root convention.
    ///
    /// Precision is decimal digits in 1..=MAX_ELEMENTARY_PRECISION_DIGITS.
    /// Input uncertainty and amplification by the exponent remain in the
    /// result; the requested digits do not promise an absolute output width.
    /// Every composition is bounded before its rational products. This is
    /// an intrinsic work envelope, not a caller-owned cancellation budget.
    pub fn pow_principal(&self, exponent: &Self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(self, precision_digits, "ComplexBall principal-power base")?;
        preflight(
            exponent,
            precision_digits,
            "ComplexBall principal-power exponent",
        )?;
        let real_exponent =
            exponent.imag().midpoint().is_zero() && exponent.imag().radius().is_zero();
        if real_exponent
            && exponent.real().radius().is_zero()
            && exponent.real().midpoint().denom().is_one()
            && let Some(integer) = exponent.real().midpoint().numer().to_i32()
        {
            return self.pow(integer);
        }
        let zero_base = self.real().midpoint().is_zero()
            && self.real().radius().is_zero()
            && self.imag().midpoint().is_zero()
            && self.imag().radius().is_zero();
        if zero_base && real_exponent && exponent.real().lower().is_positive() {
            return Ok(Self::from_i64(0));
        }
        let logarithm = self.ln(precision_digits)?;
        checked_work_bits(
            "ComplexBall principal-power logarithm product",
            [logarithm
                .power_height()
                .saturating_add(exponent.power_height())
                .saturating_add(2)
                .saturating_mul(16)],
        )?;
        let scaled = logarithm.mul(exponent);
        preflight(
            &scaled,
            precision_digits,
            "ComplexBall principal-power exponential input",
        )?;
        scaled.exp(precision_digits)
    }

    /// Enclose the principal complex logarithm of the complete rectangle.
    ///
    /// The imaginary component lies on the principal argument branch
    /// (-pi, pi]. Rectangles containing zero refuse. A nonreal rectangle
    /// meeting the negative real axis also refuses, consistently with
    /// RealBall::atan2. An exactly real, strictly negative interval is
    /// admitted with imaginary component pi.
    ///
    /// Precision is decimal digits in 1..=MAX_ELEMENTARY_PRECISION_DIGITS.
    /// Input variation remains part of the result: precision controls the
    /// component kernels' numerical errors, not the width of the image.
    /// The intrinsic work preflight is a fixed bound for this convenience
    /// API, not a caller-owned resource meter or an analyticity certificate.
    pub fn ln(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(self, precision_digits, "ComplexBall logarithm composition")?;
        if self.contains_zero() {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(
                "principal complex logarithm rectangle contains zero".into(),
            ));
        }
        // Admit the complete branch domain before computing either value.
        let angle = self.imag().atan2(self.real(), precision_digits + 2)?;
        let norm_squared = self.norm_sq()?;
        ensure_endpoint_construction_fits(&norm_squared, "ComplexBall logarithm squared modulus")?;
        let magnitude = norm_squared
            .ln(precision_digits + 2)?
            .div(&RealBall::from_i64(2))?;
        finish(magnitude, angle)
    }

    /// Enclose the principal complex square root of the complete rectangle.
    ///
    /// The real part is nonnegative. For nonreal arguments the imaginary
    /// part has the sign of the input imaginary part; on the negative real
    /// axis it is nonnegative. Nonreal rectangles meeting that branch cut
    /// refuse, including one-sided rectangles touching the cut. Exact real
    /// intervals (including zero and intervals through zero) retain their
    /// defined principal values. Boxes in the closed right half-plane may
    /// contain zero; accepting their values does not prove differentiability
    /// at zero.
    ///
    /// Precision is decimal digits in 1..=MAX_ELEMENTARY_PRECISION_DIGITS,
    /// unlike the binary precision argument of RealBall::sqrt. All inner
    /// square roots round outward. Extra modulus precision also accounts
    /// for the square root's sensitivity at zero. Positive input widths
    /// cannot be removed by increasing precision.
    pub fn sqrt(&self, precision_digits: u32) -> Result<Self, BallError> {
        preflight(
            self,
            precision_digits,
            "ComplexBall square-root composition",
        )?;
        let bits = precision_digits * 4 + 32;
        let x_lower = self.real().lower();
        let x_upper = self.real().upper();
        let y_lower = self.imag().lower();
        let y_upper = self.imag().upper();

        if y_lower.is_zero() && y_upper.is_zero() {
            if !x_lower.is_negative() {
                return finish(self.real().sqrt(bits)?, RealBall::from_i64(0));
            }
            if !x_upper.is_positive() {
                return finish(RealBall::from_i64(0), self.real().neg().sqrt(bits)?);
            }
            // The real-axis image is the union of a nonnegative real segment
            // and a nonnegative imaginary segment. Its rectangular hull is
            // a valid value enclosure, with no analyticity claim at zero.
            return finish(
                hull(BigRational::zero(), x_upper)?.sqrt(bits)?,
                hull(BigRational::zero(), -x_lower)?.sqrt(bits)?,
            );
        }

        if !y_lower.is_positive() && !y_upper.is_negative() && x_lower.is_negative() {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(
                "principal complex square-root rectangle meets the negative-axis branch cut".into(),
            ));
        }

        // sqrt is only Holder-continuous at zero: doubling the modulus's
        // working bits bounds that amplification even when |z| +/- x is
        // tiny. The exact outward bounds, rather than the guard count,
        // establish enclosure validity.
        let modulus = self.abs_ball(bits * 2 + 8)?;
        ensure_endpoint_construction_fits(&modulus, "ComplexBall square-root modulus")?;
        let real = nonnegative_half(modulus.add(self.real()))?.sqrt(bits)?;
        let imaginary_magnitude = nonnegative_half(modulus.sub(self.real()))?.sqrt(bits)?;
        let imag = if !y_lower.is_negative() {
            imaginary_magnitude
        } else if !y_upper.is_positive() {
            imaginary_magnitude.neg()
        } else {
            // This case crosses the positive real axis. Both signs occur;
            // a single sign selected from the midpoint would be unsound.
            let bound = imaginary_magnitude.upper();
            hull(-&bound, bound)?
        };
        finish(real, imag)
    }
}

/// RealBall::exp retains its absolute output-radius contract. Its monotone
/// interval extension instead evaluates exact endpoints and preserves the
/// entire outward hull, including genuine input variation. All complex
/// exponential compositions use this path.
fn exponential_range(value: &RealBall, digits: u32) -> Result<RealBall, BallError> {
    ensure_endpoint_construction_fits(value, "ComplexBall exponential input")?;
    if value.radius().is_zero() {
        return value.exp(digits);
    }
    let lower = RealBall::exact(value.lower()).exp(digits)?;
    let upper = RealBall::exact(value.upper()).exp(digits)?;
    ensure_endpoint_construction_fits(&lower, "ComplexBall exponential lower endpoint")?;
    ensure_endpoint_construction_fits(&upper, "ComplexBall exponential upper endpoint")?;
    hull(lower.lower(), upper.upper())
}

/// Bound the complete exact-rational composition before endpoint copies,
/// squares, products or square-root shifts. The deliberately conservative
/// factor covers denominator products in nested endpoint/hull construction;
/// component kernels retain their own tighter per-operation checks.
fn preflight(value: &ComplexBall, digits: u32, operation: &'static str) -> Result<(), BallError> {
    if digits == 0 || digits > ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS {
        return Err(BallError::ArgumentOutsideDeclaredEnvelope(format!(
            "complex elementary precision {digits} digits (supported: 1..={})",
            ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS,
        )));
    }
    let endpoint_height = |part: &RealBall| {
        part.midpoint()
            .height()
            .max_bits()
            .saturating_add(part.radius().height().max_bits())
            .saturating_add(2)
    };
    checked_work_bits(
        operation,
        [
            endpoint_height(value.real())
                .max(endpoint_height(value.imag()))
                .saturating_mul(128),
            u64::from(digits).saturating_mul(64) + 256,
        ],
    )?;
    Ok(())
}

fn hull(lower: BigRational, upper: BigRational) -> Result<RealBall, BallError> {
    checked_work_bits(
        "ComplexBall elementary interval hull",
        [lower.height().max_bits(), upper.height().max_bits(), 4],
    )?;
    let two = BigRational::from_integer(BigInt::from(2));
    RealBall::new((&lower + &upper) / &two, (upper - lower) / two)
}

/// The true quantities |z| +/- Re(z) are nonnegative everywhere. Interval
/// dependency can make a computed lower endpoint negative; intersecting
/// with this proved domain is valid. A negative upper endpoint would be an
/// internal contradiction and is refused, never silently repaired.
fn nonnegative_half(value: RealBall) -> Result<RealBall, BallError> {
    ensure_endpoint_construction_fits(&value, "ComplexBall square-root radicand")?;
    let lower = value.lower().max(BigRational::zero());
    let upper = value.upper();
    if upper.is_negative() {
        return Err(BallError::ArgumentOutsideDeclaredEnvelope(
            "principal complex square-root radicand has a negative upper bound".into(),
        ));
    }
    let two = BigRational::from_integer(BigInt::from(2));
    hull(lower / &two, upper / two)
}

fn finish(real: RealBall, imag: RealBall) -> Result<ComplexBall, BallError> {
    ensure_endpoint_construction_fits(&real, "ComplexBall elementary real result")?;
    ensure_endpoint_construction_fits(&imag, "ComplexBall elementary imaginary result")?;
    Ok(ComplexBall::new(real, imag))
}
