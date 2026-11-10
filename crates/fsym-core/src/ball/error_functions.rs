//! Real error-function enclosures with exact rational acceptance checks.
//!
//! The small-argument series is DLMF 7.6.1:
//! https://dlmf.nist.gov/7.6.E1 . The positive complementary tail uses the
//! Gaussian continued fraction in https://dlmf.nist.gov/7.9.E1 .
//!
//! To see the finite continued-fraction bounds directly, put
//! I_n(x) = integral_0^infinity t^n exp(-t^2 - 2*x*t) dt, x > 0.
//! Integration by parts gives r_n = I_n/I_(n-1) = n/(2*(x+r_(n+1)))
//! and sqrt(pi)*exp(x^2)*erfc(x) = 1/(x+r_1). Every r_n is positive;
//! hence 0 < r_(N+1) < (N+1)/(2*x). Propagating that interval backward
//! encloses the exact tail at every finite depth. No agreement between
//! approximate convergents is used as proof.
//!
//! Both endpoints are rounded outward on a fixed decimal grid after each
//! step. The final exact width check includes every such rounding error.

use super::{
    BallError, BigInt, BigRational, RealBall, bigdiv_floor, checked_work_bits,
    ensure_endpoint_construction_fits, ten_pow_rational,
};
use num_traits::{One, Signed, Zero};

const GUARD_DIGITS: u32 = 40;
const MAX_TAYLOR_TERMS: u32 = 1024;
const MAX_CONTINUED_FRACTION_DEPTH: u32 = 4096;
const TAIL_CUTOFF: i64 = 32;

impl RealBall {
    /// Maximum requested precision for the bounded error-function kernels.
    /// Internal guard digits are additional to this public 256-digit limit.
    pub const MAX_ERROR_FUNCTION_PRECISION_DIGITS: u32 = 256;

    /// Enclose erf on the complete real input ball.
    ///
    /// Endpoint evaluation uses monotonicity; the output keeps mathematical
    /// input variation. Exact point inputs have absolute radius at most
    /// 10^-(precision_digits+2). Near zero the normalized series also retains
    /// relative accuracy instead of rounding a tiny nonzero argument away.
    pub fn erf(&self, precision_digits: u32) -> Result<Self, BallError> {
        error_function(self, precision_digits, false, &mut |_, _, _| Ok::<(), BallError>(()))
    }

    /// Enclose erfc on the complete real input ball.
    ///
    /// Positive tails through x=32 are evaluated directly and have relative
    /// width at most 10^-(precision_digits+2). Beyond 32, monotonicity gives
    /// [0, erfc(32).upper]; this retains the requested absolute accuracy,
    /// without asserting a positive lower bound below the bounded tail lane.
    pub fn erfc(&self, precision_digits: u32) -> Result<Self, BallError> {
        error_function(self, precision_digits, true, &mut |_, _, _| Ok::<(), BallError>(()))
    }

    /// Internal entry point for original-source checkers. The new Taylor and
    /// continued-fraction loops checkpoint at every iteration. Existing
    /// elementary kernels remain bounded primitive stages charged before
    /// entry; their inner big-integer operations are not interruptible here.
    pub(crate) fn error_function_with_work<E: From<BallError>>(
        &self,
        digits: u32,
        complementary: bool,
        work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
    ) -> Result<Self, E> {
        error_function(self, digits, complementary, work)
    }

    /// Relative positive exponential used by Gaussian values and derivatives.
    /// Range-reduce before invoking the absolute exp kernel, then square
    /// outward-rounded intervals. Every stage shares the original meter.
    /// Wide inputs keep their complete monotone endpoint range.
    pub(crate) fn positive_exp_with_work<E: From<BallError>>(
        &self,
        digits: u32,
        work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
    ) -> Result<Self, E> {
        work(1, 0, 0)?;
        Self::precision_target(digits)?;
        ensure_endpoint_construction_fits(self, "RealBall Gaussian exponential input")?;
        let lower = self.lower();
        let upper = self.upper();
        if lower.is_negative() {
            return Err(refusal("Gaussian exponential requires a nonnegative argument").into());
        }
        let mut evaluate = |value: &BigRational| -> Result<RealBall, E> {
            work(1, 0, 0)?;
            if value.is_zero() {
                return Ok(Self::from_i64(1));
            }
            let point = Self::exact(value.clone());
            if point.decimal_magnitude_bound() > Self::MAX_EXP_DECIMAL_MAGNITUDE {
                return Err(refusal("Gaussian exponential magnitude exceeds its envelope").into());
            }
            let mut reduced = value.clone();
            let half = BigRational::new(1.into(), 2.into());
            let two = BigRational::from_integer(2.into());
            let mut squarings = 0u32;
            while reduced > half {
                recurrence_work(work, value.height().max_bits(), digits)?;
                reduced /= &two;
                squarings += 1;
            }
            let working = digits + 16 + 2 * squarings;
            let grid_digits = working + 12;
            let grid = BigInt::from(10).pow(grid_digits);
            // For 0 < reduced <= 1/2 the existing exp kernel has no inner
            // range reduction and its positive magnitude guard is one.
            // It therefore starts at n=working+12 Taylor terms. This is
            // already sufficient: 2^n*(n+1)! >= 10^(n-4), so its Lagrange
            // tail is below the required 10^-(working+6). Prepay every term,
            // including its factorial and outward-rounding temporaries.
            for _ in 0..working + 12 {
                recurrence_work(work, reduced.height().max_bits(), working)?;
            }
            let mut result = round_interval(Self::exact(reduced).exp(working)?, &grid)?;
            for _ in 0..squarings {
                let height = result.midpoint().height().max_bits()
                    .max(result.radius().height().max_bits());
                checked_work_bits(
                    "RealBall relative Gaussian exponential square",
                    [height.saturating_mul(4), u64::from(grid_digits) * 4 + 4],
                )?;
                recurrence_work(work, height, working)?;
                result = round_interval(result.mul(&result), &grid)?;
            }
            if result.lower() <= BigRational::zero()
                || result.width() > result.lower() * ten_pow_rational(digits + 4)
            {
                return Err(refusal("Gaussian relative exponential target was not met").into());
            }
            work(1, 0, 0)?;
            Ok(result)
        };
        let first = evaluate(&lower)?;
        if lower == upper {
            work(1, 0, 0)?;
            return Ok(first);
        }
        let last = evaluate(&upper)?;
        let result = hull(first.lower(), last.upper())?;
        work(1, 0, 0)?;
        Ok(result)
    }
}

fn refusal(message: &str) -> BallError {
    BallError::ArgumentOutsideDeclaredEnvelope(message.into())
}

fn error_function<E: From<BallError>>(
    input: &RealBall,
    digits: u32,
    complementary: bool,
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
) -> Result<RealBall, E> {
    work(1, 0, 0)?;
    RealBall::precision_target(digits)?;
    if digits > RealBall::MAX_ERROR_FUNCTION_PRECISION_DIGITS {
        return Err(refusal("error-function precision exceeds the declared ceiling").into());
    }
    ensure_endpoint_construction_fits(input, "RealBall error-function input")?;
    let lower = input.lower();
    let upper = input.upper();
    let height = lower.height().max_bits().max(upper.height().max_bits());
    // Include the original exact square, endpoint rounding, division and
    // interval compositions before constructing their rational intermediates.
    checked_work_bits(
        "RealBall error-function composition",
        [
            height.saturating_mul(16),
            u64::from(digits + GUARD_DIGITS).saturating_mul(64) + 1024,
        ],
    )?;
    let mut endpoint = |x: &BigRational| -> Result<RealBall, E> {
        work(1, 0, 0)?;
        let (erf, erfc) = positive_pair(&x.abs(), digits, work)?;
        Ok(if complementary {
            if x.is_negative() {
                RealBall::from_i64(2).sub(&erfc)
            } else {
                erfc
            }
        } else if x.is_negative() {
            erf.neg()
        } else {
            erf
        })
    };
    let first = endpoint(&lower)?;
    if lower == upper {
        work(1, 0, 0)?;
        return Ok(first);
    }
    let last = endpoint(&upper)?;
    // erf is increasing; erfc is decreasing throughout the real line.
    let result = if complementary {
        hull(last.lower(), first.upper())?
    } else {
        hull(first.lower(), last.upper())?
    };
    work(1, 0, 0)?;
    Ok(result)
}

fn hull(lower: BigRational, upper: BigRational) -> Result<RealBall, BallError> {
    checked_work_bits(
        "RealBall error-function interval",
        [lower.height().max_bits(), upper.height().max_bits(), 4],
    )?;
    if lower > upper {
        return Err(refusal("inconsistent error-function interval bounds"));
    }
    let two = BigRational::from_integer(2.into());
    RealBall::new((&lower + &upper) / &two, (upper - lower) / two)
}

/// Round both endpoints, including the radius, rather than only the midpoint.
/// This prevents hidden denominator growth during repeated interval division.
fn round_interval(ball: RealBall, scale: &BigInt) -> Result<RealBall, BallError> {
    ensure_endpoint_construction_fits(&ball, "RealBall error-function grid")?;
    checked_work_bits(
        "RealBall error-function grid products",
        [
            ball.midpoint().height().max_bits(),
            ball.radius().height().max_bits(),
            scale.bits(),
            4,
        ],
    )?;
    let lower = ball.lower() * BigRational::from_integer(scale.clone());
    let upper = ball.upper() * BigRational::from_integer(scale.clone());
    let low = bigdiv_floor(lower.numer(), lower.denom());
    let high = -bigdiv_floor(&(-upper.numer()), upper.denom());
    hull(
        BigRational::new(low, scale.clone()),
        BigRational::new(high, scale.clone()),
    )
}

fn positive_pair<E: From<BallError>>(
    x: &BigRational,
    digits: u32,
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
) -> Result<(RealBall, RealBall), E> {
    if x.is_zero() {
        return Ok((RealBall::from_i64(0), RealBall::from_i64(1)));
    }
    let one = RealBall::from_i64(1);
    let cutoff = BigRational::from_integer(TAIL_CUTOFF.into());
    let (erf, erfc) = if x < &BigRational::from_integer(4.into()) {
        let erf = taylor_erf(x, digits, work)?;
        let erfc = one.sub(&erf);
        (erf, erfc)
    } else {
        let tail_argument = x.min(&cutoff);
        let tail = complementary_tail(tail_argument, digits, work)?;
        let erfc = if x > &cutoff {
            hull(BigRational::zero(), tail.upper())?
        } else {
            tail
        };
        (one.sub(&erfc), erfc)
    };
    // Intersect only with mathematically established ranges for x >= 0.
    let range = hull(BigRational::zero(), BigRational::one())?;
    let erf = erf
        .intersect(&range)
        .ok_or_else(|| refusal("erf range enclosure did not intersect [0,1]"))?;
    let erfc = erfc
        .intersect(&range)
        .ok_or_else(|| refusal("erfc range enclosure did not intersect [0,1]"))?;
    let target = ten_pow_rational(digits + 2);
    if erf.radius() > &target || erfc.radius() > &target {
        return Err(refusal("error-function absolute enclosure target was not met").into());
    }
    // This exact relative check also protects tiny erf arguments. Values
    // beyond the explicit positive-tail lane only promise absolute accuracy.
    if erf.lower() <= BigRational::zero()
        || erf.width() > erf.lower() * &target
        || (x <= &cutoff
            && (erfc.lower() <= BigRational::zero()
                || erfc.width() > erfc.lower() * &target))
    {
        return Err(refusal("error-function relative enclosure target was not met").into());
    }
    Ok((erf, erfc))
}

/// Four limb-weighted arithmetic allowances per bounded recurrence step.
/// The byte/allocation allowances cover endpoint rounding and its temporary
/// rationals. They accumulate across all tried depths and both endpoints.
fn recurrence_work<E>(
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
    input_bits: u64,
    digits: u32,
) -> Result<(), E> {
    let width = input_bits.max(u64::from(digits + GUARD_DIGITS) * 4 + 32)
        .div_ceil(64) + 1;
    work(256 * (width + 1).pow(2), 1024 * (width + 1), 128)
}

/// Existing bounded pi/sqrt/exp kernels retain their intrinsic envelopes.
/// Reserve their conservative elementary allowance before entering each.
fn elementary_work<E>(
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
    input_bits: u64,
    digits: u32,
    kernels: u64,
) -> Result<(), E> {
    let width = input_bits.div_ceil(64) + u64::from(digits) + 1;
    work(
        kernels * 64 * width.pow(2),
        kernels * 4096 * width,
        kernels * 64,
    )
}

fn normalization(digits: u32) -> Result<RealBall, BallError> {
    let working = digits + GUARD_DIGITS;
    let root_pi = RealBall::pi(working)?.sqrt(working * 4 + 64)?;
    RealBall::from_i64(2).div(&root_pi)
}

/// Factor x outside the series, so x near zero never meets an absolute
/// rounding threshold before its nonzero scale has been retained.
fn taylor_erf<E: From<BallError>>(
    x: &BigRational,
    digits: u32,
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
) -> Result<RealBall, E> {
    let grid = BigInt::from(10).pow(digits + GUARD_DIGITS);
    let square = x * x;
    let square_ball = RealBall::exact(square.clone());
    let target = ten_pow_rational(digits + 32);
    let mut term = RealBall::from_i64(1);
    let mut sum = RealBall::from_i64(0);
    for n in 0..MAX_TAYLOR_TERMS {
        recurrence_work(work, square.height().max_bits(), digits)?;
        sum = round_interval(
            if n.is_multiple_of(2) {
                sum.add(&term)
            } else {
                sum.sub(&term)
            },
            &grid,
        )?;
        let coefficient = BigRational::new(
            BigInt::from(2 * n + 1),
            BigInt::from((n + 1) * (2 * n + 3)),
        );
        let next = round_interval(
            term.mul(&square_ball)
                .mul(&RealBall::exact(coefficient)),
            &grid,
        )?;
        let omitted = next.upper();
        // For every j >= n+1, a_(j+1)/a_j < x^2/(j+1) <= 1.
        // Only then is the first omitted term an alternating-tail bound.
        if square <= BigRational::from_integer(BigInt::from(n + 2)) && omitted <= target {
            elementary_work(work, 4, digits + GUARD_DIGITS, 2)?;
            let bounded_sum = RealBall::new(
                sum.midpoint().clone(),
                sum.radius().clone() + omitted,
            )?;
            return Ok(bounded_sum
                .mul(&normalization(digits)?)
                .mul(&RealBall::exact(x.clone())));
        }
        term = next;
    }
    Err(refusal("error-function Taylor iteration bound exceeded").into())
}

/// A direct positive erfc tail, without subtracting a number close to one.
fn complementary_tail<E: From<BallError>>(
    x: &BigRational,
    digits: u32,
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
) -> Result<RealBall, E> {
    let ratio = gaussian_mills_ratio(x, digits, work)?;
    elementary_work(work, 4, digits + GUARD_DIGITS, 2)?;
    let working = digits + GUARD_DIGITS;
    let root_pi = RealBall::pi(working)?.sqrt(working * 4 + 64)?;
    // Positive exp followed by inversion retains relative precision when
    // exp(-x^2) is hundreds of orders below the absolute working target.
    let exponential = RealBall::exact(x * x).positive_exp_with_work(digits + 8, work)?;
    recurrence_work(work, x.height().max_bits(), digits)?;
    Ok(ratio.div(&root_pi)?.div(&exponential)?)
}

fn gaussian_mills_ratio<E: From<BallError>>(
    x: &BigRational,
    digits: u32,
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
) -> Result<RealBall, E> {
    let grid = BigInt::from(10).pow(digits + GUARD_DIGITS);
    let target = ten_pow_rational(digits + 10);
    let argument = RealBall::exact(x.clone());
    let mut depth = 32u32;
    while depth <= MAX_CONTINUED_FRACTION_DEPTH {
        recurrence_work(work, x.height().max_bits(), digits)?;
        let upper_tail = BigRational::from_integer(BigInt::from(depth + 1))
            / (BigRational::from_integer(2.into()) * x);
        let mut tail = round_interval(hull(BigRational::zero(), upper_tail)?, &grid)?;
        for n in (1..=depth).rev() {
            recurrence_work(work, x.height().max_bits(), digits)?;
            let denominator = argument.add(&tail);
            tail = round_interval(
                RealBall::exact(BigRational::new(BigInt::from(n), BigInt::from(2)))
                    .div(&denominator)?,
                &grid,
            )?;
        }
        recurrence_work(work, x.height().max_bits(), digits)?;
        let ratio = round_interval(argument.add(&tail).inv()?, &grid)?;
        if ratio.lower() > BigRational::zero()
            && ratio.width() <= ratio.lower() * &target
        {
            return Ok(ratio);
        }
        depth *= 2;
    }
    Err(refusal("Gaussian continued-fraction iteration bound exceeded").into())
}
