//! Entire real sinc and its first two derivatives, including the origin.
//!
//! The identity sinc(x) = integral_0^1 cos(t*x) dt establishes both the
//! removable singularity and the global bounds |sinc^(k)(x)| <= 1/(k+1).
//! A point jet is computed by exact, outward-rounded alternating series for
//! |x| <= 1 and by certified sine/cosine identities otherwise. The mean value
//! theorem extends each point enclosure over the complete input ball using
//! the next derivative's global bound. No floating samples, numerical
//! differentiation, or derivative program is used by these kernels.

use super::{
    BallError, BigRational, RealBall, checked_work_bits, ensure_endpoint_construction_fits,
    round_outward_onto_grid, ten_pow_rational,
};
use num_traits::{One, Signed, Zero};

const GUARD_DIGITS: u32 = 16;
const MAX_TERMS: u32 = 512;

impl RealBall {
    /// Precision ceiling for the bounded sinc point-jet series.
    pub const MAX_SINC_PRECISION_DIGITS: u32 = 256;

    /// Enclose the real analytic sinc, with sinc(0) = 1, on this entire ball.
    ///
    /// Precision controls the absolute point-evaluation error, not variation
    /// across the input ball. Wide inputs may retain the global [-1, 1]
    /// range. The existing sine/cosine range-reduction envelope applies to
    /// nonzero midpoints outside [-1, 1].
    pub fn sinc(&self, precision_digits: u32) -> Result<Self, BallError> {
        let [value, _, _] =
            self.sinc_jet_with_work(precision_digits, &mut |_, _, _| Ok::<(), BallError>(()))?;
        Ok(value)
    }

    /// Original-source interval checkers share their cumulative ledger with
    /// every series iteration, arithmetic composition and publication point.
    pub(crate) fn sinc_jet_with_work<E: From<BallError>>(
        &self,
        digits: u32,
        work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
    ) -> Result<[Self; 3], E> {
        work(1, 0, 0)?;
        Self::precision_target(digits)?;
        if digits > Self::MAX_SINC_PRECISION_DIGITS {
            return Err(refusal("sinc precision exceeds the declared ceiling").into());
        }
        ensure_endpoint_construction_fits(self, "RealBall sinc input")?;
        let height = self
            .midpoint()
            .height()
            .max_bits()
            .max(self.radius().height().max_bits());
        checked_work_bits(
            "RealBall sinc composition",
            [
                height.saturating_mul(16),
                u64::from(digits + GUARD_DIGITS).saturating_mul(64) + 1024,
            ],
        )?;
        arithmetic_work(work, height, digits)?;
        let mut jet = point_jet(self.midpoint(), digits, work)?;
        for (order, value) in jet.iter_mut().enumerate() {
            arithmetic_work(work, height, digits)?;
            // The next derivative is bounded by integral_0^1 t^(k+1) dt.
            let variation = self.radius() / BigRational::from_integer(((order + 2) as u64).into());
            let bound = BigRational::new(1.into(), ((order + 1) as u64).into());
            let lower = (value.lower() - &variation).max(-&bound);
            let upper = (value.upper() + variation).min(bound);
            if lower > upper {
                return Err(refusal("sinc point enclosure contradicts its global bound").into());
            }
            let two = BigRational::from_integer(2.into());
            *value = Self::new((&lower + &upper) / &two, (upper - lower) / two)?;
            ensure_endpoint_construction_fits(value, "RealBall sinc output")?;
        }
        work(1, 0, 0)?;
        Ok(jet)
    }
}

fn refusal(message: &str) -> BallError {
    BallError::ArgumentOutsideDeclaredEnvelope(message.into())
}

fn arithmetic_work<E>(
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
    input_bits: u64,
    digits: u32,
) -> Result<(), E> {
    let width = input_bits
        .saturating_mul(2)
        .max(u64::from(digits + GUARD_DIGITS) * 4 + 32)
        .div_ceil(64)
        + 2;
    // Includes all three coefficient recurrences, outward endpoint rounding,
    // and temporary rational storage. These are cumulative logical charges.
    work(1024 * width.pow(2), 4096 * width, 512)
}

fn point_jet<E: From<BallError>>(
    x: &BigRational,
    digits: u32,
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
) -> Result<[RealBall; 3], E> {
    work(1, 0, 0)?;
    if x.is_zero() {
        return Ok([
            RealBall::from_i64(1),
            RealBall::from_i64(0),
            RealBall::exact(BigRational::new((-1).into(), 3.into())),
        ]);
    }
    let height = x.height().max_bits();
    arithmetic_work(work, height, digits)?;
    let jet = if x.abs() <= BigRational::one() {
        taylor_jet(x, digits, work)?
    } else {
        // The existing certified trig kernels are bounded primitive stages.
        // Reserve both elementary allowances before entering either kernel.
        let width = height.div_ceil(64) + u64::from(digits + 8) + 1;
        work(128 * width.pow(2), 8192 * width, 128)?;
        let argument = RealBall::exact(x.clone());
        let sine = argument.sin(digits + 8)?;
        let cosine = argument.cos(digits + 8)?;
        arithmetic_work(work, height, digits)?;
        let value = sine.div(&argument)?;
        let slope = cosine.sub(&value).div(&argument)?;
        // x*sinc''(x) + 2*sinc'(x) + x*sinc(x) = 0, used only at |x|>1.
        let curvature = value
            .neg()
            .sub(&slope.mul(&RealBall::from_i64(2)).div(&argument)?);
        [value, slope, curvature]
    };
    let target = ten_pow_rational(digits + 2);
    for value in &jet {
        arithmetic_work(work, height, digits)?;
        if value.radius() > &target {
            return Err(refusal("sinc point enclosure missed the requested precision").into());
        }
    }
    work(1, 0, 0)?;
    Ok(jet)
}

fn taylor_jet<E: From<BallError>>(
    x: &BigRational,
    digits: u32,
    work: &mut impl FnMut(u64, u64, u64) -> Result<(), E>,
) -> Result<[RealBall; 3], E> {
    let height = x.height().max_bits();
    arithmetic_work(work, height, digits)?;
    let square = RealBall::exact(x * x);
    let third = RealBall::exact(BigRational::new(1.into(), 3.into()));
    // sinc = A; sinc' = -x*B; sinc'' = -C. Factoring x out of B
    // preserves tiny nonzero slopes without absolute-grid underflow.
    let mut terms = [RealBall::from_i64(1), third.clone(), third];
    let mut sums = std::array::from_fn(|_| RealBall::from_i64(0));
    let target = ten_pow_rational(digits + 8);
    let grid = digits + GUARD_DIGITS;
    for n in 0..MAX_TERMS {
        arithmetic_work(work, height, digits)?;
        let factors = [
            BigRational::new(1.into(), ((2 * n + 2) * (2 * n + 3)).into()),
            BigRational::new(1.into(), ((2 * n + 2) * (2 * n + 5)).into()),
            BigRational::new(
                (2 * n + 3).into(),
                ((2 * n + 1) * (2 * n + 2) * (2 * n + 5)).into(),
            ),
        ];
        let mut complete = true;
        for order in 0..3 {
            sums[order] = round_outward_onto_grid(
                if n.is_multiple_of(2) {
                    sums[order].add(&terms[order])
                } else {
                    sums[order].sub(&terms[order])
                },
                grid,
            )?;
            terms[order] = round_outward_onto_grid(
                terms[order]
                    .mul(&square)
                    .mul(&RealBall::exact(factors[order].clone())),
                grid,
            )?;
            complete &= terms[order].upper() <= target;
        }
        if complete {
            // For |x|<=1, each exact term sequence decreases monotonically.
            // Its first omitted term bounds the alternating remainder; the
            // sum radii already include all preceding grid-rounding errors.
            arithmetic_work(work, height, digits)?;
            for order in 0..3 {
                sums[order] = RealBall::new(
                    sums[order].midpoint().clone(),
                    sums[order].radius() + terms[order].upper(),
                )?;
            }
            let [value, normalized_slope, negative_curvature] = sums;
            return Ok([
                value,
                normalized_slope.mul(&RealBall::exact(-x)),
                negative_curvature.neg(),
            ]);
        }
    }
    Err(refusal("sinc Taylor iteration bound exceeded").into())
}
