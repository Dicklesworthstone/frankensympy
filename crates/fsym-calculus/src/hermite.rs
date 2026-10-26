//! Factor-free Hermite reduction for rational functions over `Q(x)`.
//!
//! Repeated poles are reduced using square-free decomposition and Bezout
//! identities; irreducible factorization is unnecessary. An independent
//! proof-kernel checker verifies the rational derivative, the explicit
//! square-free remainder, and preservation of the input pole set.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_polys::{PolyError, UnivariatePoly, metered_square_free_decomposition};
use fsym_proof_kernel::rational_integral::{
    RationalIntegralReduction, RationalIntegralVerificationError,
    verify_rational_integral_reduction,
};
use num_traits::Zero;
use thiserror::Error;

const MAX_DEGREE: usize = 128;
const MAX_COEFFICIENT_BITS: u64 = 4_096;

#[derive(Debug, Error)]
pub enum HermiteError {
    #[error("rational integration reduction refused: {0}")]
    Refused(&'static str),
    #[error(transparent)]
    Polynomial(#[from] PolyError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Verification(#[from] RationalIntegralVerificationError),
}

type Result<T> = std::result::Result<T, HermiteError>;

/// The polynomial substrate reports its own errors as `PolyError`. Preserve
/// the original typed meter interruption rather than classifying cancellation
/// or exhaustion as an algebraic failure at that boundary.
struct ForwardMeter<'a, M> {
    inner: &'a mut M,
    stopped: Option<MeterError>,
}

impl<M: BudgetMeter> BudgetMeter for ForwardMeter<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        let result = self.inner.charge(dimension, amount);
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let result = self.inner.charge_batch(charges);
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

fn validate(poly: &UnivariatePoly) -> Result<()> {
    poly.validate_shape()?;
    if poly.degree().unwrap_or(0) > MAX_DEGREE
        || poly
            .coeffs
            .iter()
            .any(|c| c.numer().bits().max(c.denom().bits()) > MAX_COEFFICIENT_BITS)
    {
        return Err(HermiteError::Refused(
            "polynomial degree or coefficient-height limit",
        ));
    }
    Ok(())
}

/// Prepay a bounded dense polynomial batch before entering the polynomial
/// substrate. That substrate's Euclidean routines are synchronous, so safe
/// points occur between batches, not inside each rational operation. Charges
/// conservatively cover coefficient growth and transient polynomial storage.
fn batch(meter: &mut impl BudgetMeter, polys: &[&UnivariatePoly]) -> Result<()> {
    meter.checkpoint()?;
    let mut width = 1u64;
    let mut bits = 1u64;
    for poly in polys {
        validate(poly)?;
        width = width.max(poly.coeffs.len() as u64);
        bits = bits.max(
            poly.coeffs
                .iter()
                .map(|c| c.numer().bits().max(c.denom().bits()))
                .max()
                .unwrap_or(1),
        );
    }
    let work = width.saturating_pow(3).saturating_mul(64);
    let growth = bits
        .saturating_add(16)
        .saturating_mul(width.saturating_pow(2))
        .saturating_mul(8);
    meter.charge_batch(&[
        (Dimension::ComputeSteps, work),
        (Dimension::AllocationCount, work),
        (
            Dimension::MemoryBytes,
            work.saturating_mul(growth.div_ceil(8).max(1)),
        ),
    ])?;
    Ok(())
}

fn exact_quotient(a: &UnivariatePoly, b: &UnivariatePoly) -> Result<UnivariatePoly> {
    let (quotient, remainder) = a.div_rem(b)?;
    if !remainder.is_zero() {
        return Err(HermiteError::Refused("nonexact Hermite division"));
    }
    Ok(quotient)
}

fn scale(p: &UnivariatePoly, c: BigRational) -> Result<UnivariatePoly> {
    Ok(p.mul(&UnivariatePoly::new(p.gen_sym.clone(), vec![c]))?)
}

fn normalize_fraction(
    numerator: UnivariatePoly,
    denominator: UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<(UnivariatePoly, UnivariatePoly)> {
    batch(meter, &[&numerator, &denominator])?;
    if denominator.is_zero() {
        return Err(HermiteError::Refused("zero denominator"));
    }
    let gcd = numerator.gcd(&denominator)?;
    let numerator = exact_quotient(&numerator, &gcd)?;
    let denominator = exact_quotient(&denominator, &gcd)?;
    let leading = denominator.leading_coeff().recip();
    Ok((
        scale(&numerator, leading.clone())?,
        scale(&denominator, leading)?,
    ))
}

fn add_fraction(
    left: (UnivariatePoly, UnivariatePoly),
    numerator: UnivariatePoly,
    denominator: UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<(UnivariatePoly, UnivariatePoly)> {
    batch(meter, &[&left.0, &left.1, &numerator, &denominator])?;
    // Use the lcm denominator rather than multiplying repeated pole factors.
    let gcd = left.1.gcd(&denominator)?;
    let left_scale = exact_quotient(&denominator, &gcd)?;
    let right_scale = exact_quotient(&left.1, &gcd)?;
    let top = left
        .0
        .mul(&left_scale)?
        .add(&numerator.mul(&right_scale)?)?;
    let bottom = left.1.mul(&left_scale)?;
    normalize_fraction(top, bottom, meter)
}

/// Reduce a rational integrand and independently verify the result. This is
/// an explicit unbounded-meter convenience entry point for synchronous callers.
/// A nonzero remainder is retained, not represented as a complete integral.
pub fn hermite_reduce(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
) -> Result<RationalIntegralReduction> {
    metered_hermite_reduce(numerator, denominator, &mut Unbounded, &mut Unbounded)
}

/// Generator and verifier have separate meters; generation cannot consume
/// the caller's verifier reservation. Cancellation or exhaustion publishes
/// no partial reduction and never retries with an unbounded meter.
pub fn metered_hermite_reduce(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<RationalIntegralReduction> {
    batch(generator_meter, &[numerator, denominator])?;
    if numerator.gen_sym != denominator.gen_sym {
        return Err(HermiteError::Refused("different polynomial generators"));
    }
    let x = numerator.gen_sym.clone();
    let (num, den) = normalize_fraction(numerator.clone(), denominator.clone(), generator_meter)?;
    batch(generator_meter, &[&num, &den])?;
    let (quotient, proper) = num.div_rem(&den)?;
    let mut rational = (
        quotient.integrate(BigRational::zero())?,
        UnivariatePoly::one(x.clone()),
    );
    let mut remainder = (
        UnivariatePoly::zero(x.clone()),
        UnivariatePoly::one(x.clone()),
    );

    if !proper.is_zero() {
        let mut forwarded = ForwardMeter {
            inner: generator_meter,
            stopped: None,
        };
        let decomposition = metered_square_free_decomposition(&den, &mut forwarded);
        if let Some(error) = forwarded.stopped {
            return Err(error.into());
        }
        let decomposition = decomposition?;
        for factor in decomposition.factors {
            let p = factor.poly;
            let multiplicity = factor.multiplicity;
            batch(generator_meter, &[&proper, &den, &p])?;
            let power = p.pow(
                u32::try_from(multiplicity)
                    .map_err(|_| HermiteError::Refused("pole multiplicity overflow"))?,
            )?;
            let other = exact_quotient(&den, &power)?;
            let inverse = other.extended_gcd(&power)?;
            if !inverse.gcd.is_one() {
                return Err(HermiteError::Refused("non-coprime square-free components"));
            }
            let mut local = proper.mul(&inverse.u)?.div_rem(&power)?.1;
            let derivative = p.derivative();
            batch(generator_meter, &[&p, &derivative, &local])?;
            let derivative_inverse = derivative.extended_gcd(&p)?;
            if !derivative_inverse.gcd.is_one() {
                return Err(HermiteError::Refused("component is not square-free"));
            }
            for order in (2..=multiplicity).rev() {
                batch(generator_meter, &[&local, &p, &derivative_inverse.u])?;
                let order_minus_one = BigRational::from_integer(BigInt::from((order - 1) as u64));
                // A/p^m = (B/p^(m-1))' + C/p^(m-1), choosing
                // B = -A * (p')^(-1)/(m-1) modulo p.
                let reduced = local.mul(&derivative_inverse.u)?.div_rem(&p)?.1;
                let b = scale(&reduced, -order_minus_one.recip())?;
                let adjusted = local.add(&scale(&b.mul(&derivative)?, order_minus_one)?)?;
                local = exact_quotient(&adjusted, &p)?.sub(&b.derivative())?;
                let bottom = p.pow(
                    u32::try_from(order - 1)
                        .map_err(|_| HermiteError::Refused("pole multiplicity overflow"))?,
                )?;
                rational = add_fraction(rational, b, bottom, generator_meter)?;
            }
            remainder = add_fraction(remainder, local, p, generator_meter)?;
        }
    }

    batch(
        generator_meter,
        &[
            &rational.0,
            &rational.1,
            &remainder.0,
            &remainder.1,
            denominator,
        ],
    )?;
    let square_free = remainder.1.extended_gcd(&remainder.1.derivative())?;
    if !square_free.gcd.is_one() {
        return Err(HermiteError::Refused("remainder retained a repeated pole"));
    }
    let witness = RationalIntegralReduction {
        rational_pole_cofactor: exact_quotient(denominator, &rational.1)?.coeffs,
        remainder_pole_cofactor: exact_quotient(denominator, &remainder.1)?.coeffs,
        rational_numerator: rational.0.coeffs,
        rational_denominator: rational.1.coeffs,
        remainder_numerator: remainder.0.coeffs,
        remainder_denominator: remainder.1.coeffs,
        square_free_u: square_free.u.coeffs,
        square_free_v: square_free.v.coeffs,
    };
    generator_meter.checkpoint()?;
    verify_rational_integral_reduction(
        &numerator.coeffs,
        &denominator.coeffs,
        &witness,
        verifier_meter,
    )?;
    generator_meter.checkpoint()?;
    Ok(witness)
}
