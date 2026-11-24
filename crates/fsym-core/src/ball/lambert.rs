//! Certified enclosures of the two finite real Lambert-W branches.
//!
//! All root refinements retain a bracket proved by monotonicity. Interval
//! Newton is only an accelerator: its residual and derivative are enclosed,
//! rounded outward, and intersected with that bracket before acceptance.
//! No floating-point approximation participates in acceptance.
//!
//! For z > 0 solve v*exp(v)=z on v >= 0. For z < 0 solve
//! v/exp(v)=-z, with v=-W(z), on [0,1] or [1,infinity).
//! The complete input interval must lie in the chosen branch's real domain.
//! The irrational branch point is handled by exact expression folding; a
//! rational ball straddling it is never replaced by its midpoint.

use super::{BallError, BigInt, BigRational, RealBall, round_outward_onto_grid};
use crate::arith::RationalArithmeticError;
use fsym_budget::{BudgetError, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use thiserror::Error;

const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 32_768;
const MAX_STEPS: u64 = 512_000_000;
const MAX_BYTES: u64 = 256 * 1024 * 1024;
const MAX_ALLOCATIONS: u64 = 4_000_000;
const MAX_ITERATIONS: usize = 1024;
const GUARD_DIGITS: u32 = 24;

/// A real Lambert-W enclosure was refused before a result was published.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LambertWError {
    /// At least one point of the complete input ball is proved outside the
    /// selected finite real branch. A failed bounded comparison, unsupported
    /// branch or exhausted computation never has this mathematical meaning.
    #[error("Lambert-W input is outside the finite real domain of branch {branch}")]
    OutsideRealDomain { branch: i32 },
    #[error("Lambert-W ball operation failed: {0}")]
    Ball(#[from] BallError),
    #[error("Lambert-W evaluation stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("Lambert-W exact arithmetic failed: {0}")]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for LambertWError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            other => Self::Arithmetic(other),
        }
    }
}

impl RealBall {
    /// Maximum requested decimal precision for this bounded real kernel.
    pub const MAX_LAMBERT_W_PRECISION_DIGITS: u32 = 128;

    /// Enclose W_0 or W_{-1} on the complete input ball.
    ///
    /// Branch 0 accepts real arguments at or above -1/e. Branch -1 accepts
    /// arguments from -1/e up to, but excluding, zero; its limit at zero is
    /// not a finite real ball. Rational endpoints cannot equal -1/e.
    ///
    /// Each endpoint approximation has width at most 10^-(digits+2).
    /// Principal values with |z| < 1 retain that relative accuracy in |z|.
    /// A wide input retains its mathematical image width. Inputs and all
    /// intermediates have finite height bounds, and both endpoint solves
    /// share the fixed work/allocation ceilings of the metered entry point.
    pub fn lambert_w(&self, branch: i32, digits: u32) -> Result<Self, BallError> {
        self.metered_lambert_w(branch, digits, &mut Unbounded)
            .map_err(|error| match error {
                LambertWError::Ball(error) => error,
                other => BallError::ArgumentOutsideDeclaredEnvelope(other.to_string()),
            })
    }

    /// Cumulative, cancellation-aware real Lambert-W evaluation.
    ///
    /// The caller's meter is shared by domain admission, both endpoint
    /// solves, every exponential/refinement, and final publication. The
    /// intrinsic ceilings are 512 million compute steps, 256 MiB cumulative
    /// allocation bytes, and four million allocations, even with Unbounded.
    /// Original midpoint/radius and constructed endpoints are at most 4096
    /// bits; rational arithmetic is preflighted at 32768 work bits. A
    /// difficult valid input can refuse within these finite limits.
    pub fn metered_lambert_w<M: BudgetMeter>(
        &self,
        branch: i32,
        digits: u32,
        meter: &mut M,
    ) -> Result<Self, LambertWError> {
        let mut work = Work::new(meter);
        work.reserve(64, 512, 8)?;
        if !matches!(branch, 0 | -1) {
            return Err(refuse("only the real branches 0 and -1 are supported"));
        }
        if digits == 0 || digits > Self::MAX_LAMBERT_W_PRECISION_DIGITS {
            return Err(refuse("Lambert-W precision must be in 1..=128"));
        }
        for value in [self.midpoint(), self.radius()] {
            work.input_height(value)?;
        }
        let input = Interval::from_ball(self, &mut work)?;
        work.input_height(&input.lo)?;
        work.input_height(&input.hi)?;
        admit_domain(&input, branch, &mut work)?;

        let first = endpoint(&input.lo, branch, digits, &mut work)?;
        let output = if work.compare(&input.lo, &input.hi)? == Ordering::Equal {
            first
        } else {
            let last = endpoint(&input.hi, branch, digits, &mut work)?;
            // W_0 increases; W_{-1} decreases on its entire real domain.
            if branch == 0 {
                Interval {
                    lo: first.lo,
                    hi: last.hi,
                }
            } else {
                Interval {
                    lo: last.lo,
                    hi: first.hi,
                }
            }
        };
        let result = output.into_ball(&mut work)?;
        work.checkpoint()?;
        Ok(result)
    }
}

/// An adapter, never a fresh caller budget. Its fixed counters accumulate
/// alongside the supplied meter; no endpoint or precision retry resets them.
struct Work<'a, M: BudgetMeter> {
    meter: &'a mut M,
    used: [u64; DIMENSION_COUNT],
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; DIMENSION_COUNT],
        }
    }

    fn reserve(&mut self, steps: u64, bytes: u64, allocations: u64) -> Result<(), LambertWError> {
        let mut charges = [(Dimension::ComputeSteps, 1); 3];
        let mut length = 0;
        for charge in [
            (Dimension::ComputeSteps, steps),
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, allocations),
        ] {
            if charge.1 != 0 {
                charges[length] = charge;
                length += 1;
            }
        }
        if length == 0 {
            self.checkpoint()?;
        } else {
            self.charge_batch(&charges[..length])?;
        }
        Ok(())
    }

    fn input_height(&mut self, value: &BigRational) -> Result<(), LambertWError> {
        self.checkpoint()?;
        if value.height().max_bits() > MAX_INPUT_BITS {
            return Err(refuse("Lambert-W input rational height exceeds 4096 bits"));
        }
        Ok(())
    }

    fn preflight(&mut self, bits: u64) -> Result<(), LambertWError> {
        self.checkpoint()?;
        if bits > MAX_WORK_BITS {
            return Err(BallError::ResourceLimitExceeded {
                operation: "RealBall Lambert-W rational work",
                required_bits: bits,
                limit_bits: MAX_WORK_BITS,
            }
            .into());
        }
        Ok(())
    }

    fn binary(&mut self, a: &BigRational, b: &BigRational) -> Result<(), LambertWError> {
        self.preflight(
            a.height()
                .max_bits()
                .saturating_add(b.height().max_bits())
                .saturating_add(2),
        )
    }

    fn compare(&mut self, a: &BigRational, b: &BigRational) -> Result<Ordering, LambertWError> {
        self.binary(a, b)?;
        let limbs = (a.height().max_bits() + b.height().max_bits()).div_ceil(64) + 1;
        self.reserve(16 * limbs * limbs, 128 * limbs, 16)?;
        Ok(a.cmp(b))
    }

    fn copy(&mut self, value: &BigRational) -> Result<BigRational, LambertWError> {
        self.preflight(value.height().max_bits())?;
        let limbs = value.height().total_limbs().max(1);
        self.reserve(8 * limbs, 16 * limbs, 2)?;
        Ok(value.clone())
    }

    fn negate(&mut self, value: &BigRational) -> Result<BigRational, LambertWError> {
        let copy = self.copy(value)?;
        Ok(-copy)
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, LambertWError> {
        self.binary(a, b)?;
        Ok(a.metered_add(b, self)?)
    }

    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, LambertWError> {
        self.binary(a, b)?;
        Ok(a.metered_sub(b, self)?)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, LambertWError> {
        self.binary(a, b)?;
        Ok(a.metered_mul(b, self)?)
    }

    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, LambertWError> {
        self.binary(a, b)?;
        Ok(a.metered_div(b, self)?)
    }

    fn decimal_target(&mut self, digits: u32) -> Result<BigRational, LambertWError> {
        let bits = u64::from(digits) * 4 + 4;
        self.preflight(bits)?;
        let limbs = bits.div_ceil(64) + 1;
        self.reserve(64 * limbs * limbs, 128 * limbs, 16)?;
        Ok(BigRational::new(
            BigInt::one(),
            BigInt::from(10).pow(digits),
        ))
    }

    fn exp(&mut self, interval: &Interval, digits: u32) -> Result<Interval, LambertWError> {
        let bits = interval.lo.height().max_bits() + interval.hi.height().max_bits() + 4;
        self.preflight(bits)?;
        let limbs = bits.div_ceil(64) + 1;
        self.reserve(256 * limbs * limbs, 1024 * limbs, 128)?;
        let input = interval.to_ball(self)?;
        let result = input.positive_exp_with_work(digits, &mut |steps, bytes, allocations| {
            self.reserve(steps, bytes, allocations)
        })?;
        Interval::from_ball(&result, self)
    }

    fn round(&mut self, interval: Interval, digits: u32) -> Result<Interval, LambertWError> {
        let ball = interval.into_ball(self)?;
        let bits = ball
            .midpoint()
            .height()
            .max_bits()
            .saturating_add(ball.radius().height().max_bits())
            .saturating_add(u64::from(digits) * 4 + 4);
        self.preflight(bits)?;
        // Prepay floor/ceil products, quotient work, grid construction and
        // temporary rational reductions in the existing bounded primitive.
        let limbs = bits.div_ceil(64) + 1;
        self.reserve(256 * limbs * limbs, 1024 * limbs, 128)?;
        let rounded = round_outward_onto_grid(ball, digits)?;
        Interval::from_ball(&rounded, self)
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.meter.checkpoint()?;
        let mut totals = [0u64; DIMENSION_COUNT];
        for &(dimension, amount) in charges {
            if amount == 0 {
                return Err(BudgetError::ZeroCharge.into());
            }
            let slot = &mut totals[dimension.index()];
            *slot = slot
                .checked_add(amount)
                .ok_or(BudgetError::ChargeOverflow { dimension })?;
        }
        for dimension in Dimension::ALL {
            let limit = match dimension {
                Dimension::ComputeSteps => MAX_STEPS,
                Dimension::MemoryBytes => MAX_BYTES,
                Dimension::AllocationCount => MAX_ALLOCATIONS,
                _ => u64::MAX,
            };
            let remaining = limit - self.used[dimension.index()];
            if totals[dimension.index()] > remaining {
                return Err(BudgetError::Exhausted {
                    dimension,
                    requested: totals[dimension.index()],
                    remaining,
                }
                .into());
            }
        }
        self.meter.charge_batch(charges)?;
        for (used, total) in self.used.iter_mut().zip(totals) {
            *used += total;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

struct Interval {
    lo: BigRational,
    hi: BigRational,
}

impl Interval {
    fn point<M: BudgetMeter>(
        value: &BigRational,
        work: &mut Work<'_, M>,
    ) -> Result<Self, LambertWError> {
        Ok(Self {
            lo: work.copy(value)?,
            hi: work.copy(value)?,
        })
    }

    fn from_ball<M: BudgetMeter>(
        ball: &RealBall,
        work: &mut Work<'_, M>,
    ) -> Result<Self, LambertWError> {
        Ok(Self {
            lo: work.sub(ball.midpoint(), ball.radius())?,
            hi: work.add(ball.midpoint(), ball.radius())?,
        })
    }

    fn to_ball<M: BudgetMeter>(&self, work: &mut Work<'_, M>) -> Result<RealBall, LambertWError> {
        work.reserve(8, 128, 4)?;
        let sum = work.add(&self.lo, &self.hi)?;
        let difference = work.sub(&self.hi, &self.lo)?;
        let two = BigRational::from_integer(2.into());
        Ok(RealBall::new(
            work.div(&sum, &two)?,
            work.div(&difference, &two)?,
        )?)
    }

    fn into_ball<M: BudgetMeter>(self, work: &mut Work<'_, M>) -> Result<RealBall, LambertWError> {
        self.to_ball(work)
    }

    fn width<M: BudgetMeter>(&self, work: &mut Work<'_, M>) -> Result<BigRational, LambertWError> {
        work.sub(&self.hi, &self.lo)
    }

    fn negate<M: BudgetMeter>(self, work: &mut Work<'_, M>) -> Result<Self, LambertWError> {
        Ok(Self {
            lo: work.negate(&self.hi)?,
            hi: work.negate(&self.lo)?,
        })
    }
}

fn refuse(message: &str) -> LambertWError {
    BallError::ArgumentOutsideDeclaredEnvelope(message.into()).into()
}

fn admit_domain<M: BudgetMeter>(
    input: &Interval,
    branch: i32,
    work: &mut Work<'_, M>,
) -> Result<(), LambertWError> {
    work.reserve(32, 512, 16)?;
    if branch == -1 && !input.hi.is_negative() {
        return Err(LambertWError::OutsideRealDomain { branch });
    }
    if !input.lo.is_negative() {
        return Ok(());
    }
    let magnitude = work.negate(&input.lo)?;
    // 2 < e < 3 provides cheap, exact admissions/refusals.
    let third = BigRational::new(1.into(), 3.into());
    if work.compare(&magnitude, &third)? != Ordering::Greater {
        return Ok(());
    }
    let half = BigRational::new(1.into(), 2.into());
    if work.compare(&magnitude, &half)? != Ordering::Less {
        return Err(LambertWError::OutsideRealDomain { branch });
    }
    let reciprocal = work.div(&BigRational::one(), &magnitude)?;
    let one = Interval::point(&BigRational::one(), work)?;
    let maximum = RealBall::MAX_LAMBERT_W_PRECISION_DIGITS + GUARD_DIGITS;
    let mut digits = 8u32;
    loop {
        let exp_one = work.exp(&one, digits)?;
        if work.compare(&exp_one.hi, &reciprocal)? == Ordering::Less {
            return Ok(());
        }
        if work.compare(&exp_one.lo, &reciprocal)? == Ordering::Greater {
            return Err(LambertWError::OutsideRealDomain { branch });
        }
        if digits == maximum {
            return Err(refuse(
                "the input cannot be separated from -1/e within the bounded precision",
            ));
        }
        digits = digits.saturating_mul(2).min(maximum);
    }
}

fn initial_bracket<M: BudgetMeter>(
    argument: &BigRational,
    negative: bool,
    branch: i32,
    work: &mut Work<'_, M>,
) -> Result<Interval, LambertWError> {
    work.reserve(32, 512, 16)?;
    let one = BigRational::one();
    if !negative {
        if work.compare(argument, &one)? != Ordering::Greater {
            // For 0 <= v < 1, exp(v) <= 1/(1-v). Thus
            // z/(1+z) <= W_0(z) <= z, also useful for tiny z.
            let denominator = work.add(&one, argument)?;
            return Ok(Interval {
                lo: work.div(argument, &denominator)?,
                hi: work.copy(argument)?,
            });
        }
        // If b=bits(p)-bits(q)+1, then z=p/q < 2^b. Since e>2
        // and b>=1, b*exp(b)>z. Use value magnitude, not payload height.
        let bits = argument.numer().bits() - argument.denom().bits() + 1;
        let upper = BigRational::from_integer(BigInt::from(bits));
        Ok(Interval {
            lo: BigRational::zero(),
            hi: upper,
        })
    } else if branch == 0 {
        let quarter = BigRational::new(1.into(), 4.into());
        let upper = if work.compare(argument, &quarter)? != Ordering::Greater {
            // With t<=1/4 and u=t/(1-2t)<=1/2,
            // u*exp(-u) >= u*(1-u) >= t. Hence t<=v<=u.
            let doubled = work.add(argument, argument)?;
            let denominator = work.sub(&one, &doubled)?;
            work.div(argument, &denominator)?
        } else {
            BigRational::one()
        };
        Ok(Interval {
            lo: work.copy(argument)?,
            hi: upper,
        })
    } else {
        // For b=bits(q)-bits(p)+1, t=p/q > 2^-b. At B=2(b+1),
        // B*exp(-B) < B*2^-B <= 2^-b < t, since b>=1.
        let bits = argument.denom().bits() - argument.numer().bits() + 1;
        let upper = 2 * (bits + 1);
        Ok(Interval {
            lo: BigRational::one(),
            hi: BigRational::from_integer(BigInt::from(upper)),
        })
    }
}

fn residual<M: BudgetMeter>(
    midpoint: &BigRational,
    argument: &BigRational,
    negative: bool,
    digits: u32,
    work: &mut Work<'_, M>,
) -> Result<Interval, LambertWError> {
    work.reserve(16, 256, 8)?;
    let point = Interval::point(midpoint, work)?;
    let exponential = work.exp(&point, digits)?;
    let (lo, hi) = if negative {
        (
            work.div(midpoint, &exponential.hi)?,
            work.div(midpoint, &exponential.lo)?,
        )
    } else {
        (
            work.mul(midpoint, &exponential.lo)?,
            work.mul(midpoint, &exponential.hi)?,
        )
    };
    Ok(Interval {
        lo: work.sub(&lo, argument)?,
        hi: work.sub(&hi, argument)?,
    })
}

fn derivative<M: BudgetMeter>(
    bracket: &Interval,
    negative: bool,
    digits: u32,
    work: &mut Work<'_, M>,
) -> Result<Option<Interval>, LambertWError> {
    work.reserve(16, 256, 8)?;
    let one = BigRational::one();
    if negative {
        let lo = work.sub(&one, &bracket.hi)?;
        let hi = work.sub(&one, &bracket.lo)?;
        // A bracket touching v=1 has a zero derivative endpoint. Bisection
        // must first remove it; dividing by this interval would be unsound.
        if !lo.is_negative() && !lo.is_zero() {
            let exponential = work.exp(bracket, digits)?;
            return Ok(Some(Interval {
                lo: work.div(&lo, &exponential.hi)?,
                hi: work.div(&hi, &exponential.lo)?,
            }));
        }
        if hi.is_negative() {
            let exponential = work.exp(bracket, digits)?;
            return Ok(Some(Interval {
                lo: work.div(&lo, &exponential.lo)?,
                hi: work.div(&hi, &exponential.hi)?,
            }));
        }
        Ok(None)
    } else {
        // (1+v)*exp(v) is strictly increasing for v>=0.
        let exponential = work.exp(bracket, digits)?;
        let lo = work.add(&one, &bracket.lo)?;
        let hi = work.add(&one, &bracket.hi)?;
        Ok(Some(Interval {
            lo: work.mul(&lo, &exponential.lo)?,
            hi: work.mul(&hi, &exponential.hi)?,
        }))
    }
}

fn newton_intersection<M: BudgetMeter>(
    bracket: &Interval,
    midpoint: &BigRational,
    residual: &Interval,
    derivative: &Interval,
    grid_digits: u32,
    work: &mut Work<'_, M>,
) -> Result<Interval, LambertWError> {
    work.reserve(16, 256, 8)?;
    let quotients = [
        work.div(&residual.lo, &derivative.lo)?,
        work.div(&residual.lo, &derivative.hi)?,
        work.div(&residual.hi, &derivative.lo)?,
        work.div(&residual.hi, &derivative.hi)?,
    ];
    let mut low = 0;
    let mut high = 0;
    for index in 1..quotients.len() {
        if work.compare(&quotients[index], &quotients[low])? == Ordering::Less {
            low = index;
        }
        if work.compare(&quotients[index], &quotients[high])? == Ordering::Greater {
            high = index;
        }
    }
    let image = Interval {
        lo: work.sub(midpoint, &quotients[high])?,
        hi: work.sub(midpoint, &quotients[low])?,
    };
    // Drop large transient quotients before allocating the outward grid.
    drop(quotients);
    let mut image = work.round(image, grid_digits)?;
    if work.compare(&image.lo, &bracket.lo)? == Ordering::Less {
        image.lo = work.copy(&bracket.lo)?;
    }
    if work.compare(&image.hi, &bracket.hi)? == Ordering::Greater {
        image.hi = work.copy(&bracket.hi)?;
    }
    if work.compare(&image.lo, &image.hi)? == Ordering::Greater {
        return Err(refuse("certified Lambert-W Newton intersection was empty"));
    }
    Ok(image)
}

fn endpoint<M: BudgetMeter>(
    z: &BigRational,
    branch: i32,
    digits: u32,
    work: &mut Work<'_, M>,
) -> Result<Interval, LambertWError> {
    work.reserve(64, 512, 16)?;
    if z.is_zero() {
        return Interval::point(z, work);
    }
    let negative = z.is_negative();
    let argument = if negative {
        work.negate(z)?
    } else {
        work.copy(z)?
    };
    let mut target = work.decimal_target(digits + 2)?;
    let one = BigRational::one();
    let small_principal = branch == 0 && work.compare(&argument, &one)? == Ordering::Less;
    if small_principal {
        target = work.mul(&target, &argument)?;
    }
    // A decimal grid fine enough for relative accuracy on tiny arguments.
    // If k = bits(q)-bits(p)+1, p/q > 2^-k > 10^-ceil(k/3).
    let extra = if small_principal {
        argument
            .denom()
            .bits()
            .saturating_add(1)
            .saturating_sub(argument.numer().bits())
            .div_ceil(3) as u32
    } else {
        0
    };
    let grid_digits = digits + GUARD_DIGITS + extra;
    let mut bracket = initial_bracket(&argument, negative, branch, work)?;
    let increasing = !negative || branch == 0;
    let mut working_digits = 8u32;
    let maximum = RealBall::MAX_LAMBERT_W_PRECISION_DIGITS + GUARD_DIGITS;
    let two = BigRational::from_integer(2.into());
    let three = BigRational::from_integer(3.into());
    let four = BigRational::from_integer(4.into());

    for _ in 0..MAX_ITERATIONS {
        work.reserve(8, 128, 4)?;
        let width = bracket.width(work)?;
        if work.compare(&width, &target)? != Ordering::Greater {
            return if negative {
                bracket.negate(work)
            } else {
                Ok(bracket)
            };
        }
        let sum = work.add(&bracket.lo, &bracket.hi)?;
        let midpoint = work.div(&sum, &two)?;
        let value = residual(&midpoint, &argument, negative, working_digits, work)?;
        if let Some(slope) = derivative(&bracket, negative, working_digits, work)? {
            let next = newton_intersection(&bracket, &midpoint, &value, &slope, grid_digits, work)?;
            let next_width = next.width(work)?;
            let scaled_next = work.mul(&next_width, &four)?;
            let scaled_old = work.mul(&width, &three)?;
            if work.compare(&scaled_next, &scaled_old)? != Ordering::Greater {
                bracket = next;
                continue;
            }
        }
        let sign = if value.hi.is_negative() {
            Some(Ordering::Less)
        } else if !value.lo.is_negative() && !value.lo.is_zero() {
            Some(Ordering::Greater)
        } else {
            None
        };
        if let Some(sign) = sign {
            if (sign == Ordering::Less) == increasing {
                bracket.lo = midpoint;
            } else {
                bracket.hi = midpoint;
            }
        } else if working_digits < maximum {
            working_digits = working_digits.saturating_mul(2).min(maximum);
        } else {
            return Err(refuse(
                "Lambert-W residual could not establish the requested precision",
            ));
        }
    }
    Err(refuse("Lambert-W iteration ceiling exhausted"))
}
