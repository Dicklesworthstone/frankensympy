//! Exact real definite rational integrals with logarithmic endpoint cancellation.
//!
//! The generator uses Hermite reduction and partial fractions over Q. A
//! dependency-separated proof-kernel checker verifies the complete primitive,
//! all original poles and each finite or infinite endpoint. Quadratic
//! reciprocals use real logarithmic/arctangent branches; a higher-degree
//! factor is admitted only when its numerator is a logarithmic derivative.

use crate::hermite::{HermiteError, metered_hermite_reduce};
use crate::rational_limits::{RationalLimitError, metered_rational_local_order};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use fsym_polys::{PolyError, UnivariatePoly, metered_complete_factorization};
use fsym_proof_kernel::elementary_rational_definite::{
    ElementaryRationalDefiniteIntegralCertificate, ElementaryRationalDefiniteIntegralVerification,
    ElementaryRationalDefiniteIntegralVerificationError, ElementaryRationalTerm,
    verify_elementary_rational_definite_integral,
};
use fsym_proof_kernel::rational_definite::{
    RationalDefiniteIntegralVerificationError, RationalIntegralEndpoint,
    verify_rational_integral_domain,
};
use fsym_proof_kernel::rational_limit::RationalLocalCertificate;
use num_traits::{One, Zero};
use thiserror::Error;

const MAX_DEGREE: usize = 32;
const MAX_INPUT_BITS: u64 = 4_096;
const MAX_WORK_BITS: u64 = 65_536;

#[derive(Debug, Error)]
pub enum ElementaryRationalDefiniteIntegralError {
    #[error("elementary rational definite integration refused: {0}")]
    Refused(&'static str),
    #[error("the rational primitive requires an unsupported algebraic logarithm")]
    UnsupportedPrimitive,
    #[error("the elementary primitive has no finite endpoint limit")]
    DivergentEndpoint,
    #[error(transparent)]
    Hermite(#[from] HermiteError),
    #[error(transparent)]
    Local(#[from] RationalLimitError),
    #[error(transparent)]
    Domain(#[from] RationalDefiniteIntegralVerificationError),
    #[error(transparent)]
    Verification(ElementaryRationalDefiniteIntegralVerificationError),
    #[error(transparent)]
    Polynomial(#[from] PolyError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for ElementaryRationalDefiniteIntegralError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

impl From<ElementaryRationalDefiniteIntegralVerificationError>
    for ElementaryRationalDefiniteIntegralError
{
    fn from(error: ElementaryRationalDefiniteIntegralVerificationError) -> Self {
        match error {
            ElementaryRationalDefiniteIntegralVerificationError::DivergentEndpoint => {
                Self::DivergentEndpoint
            }
            ElementaryRationalDefiniteIntegralVerificationError::Meter(error) => Self::Meter(error),
            error => Self::Verification(error),
        }
    }
}

type Result<T> = std::result::Result<T, ElementaryRationalDefiniteIntegralError>;

const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

struct CappedMeter<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
}

impl<'a, M: BudgetMeter> CappedMeter<'a, M> {
    fn new(inner: &'a mut M) -> Self {
        Self {
            inner,
            used: [0; 3],
            stopped: None,
        }
    }
}

impl<M: BudgetMeter> BudgetMeter for CappedMeter<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = (|| {
            let mut next = self.used;
            for &(dimension, amount) in charges {
                let (index, limit) = match dimension {
                    Dimension::ComputeSteps => (0, MAX_COMPUTE),
                    Dimension::MemoryBytes => (1, MAX_MEMORY),
                    Dimension::AllocationCount => (2, MAX_COMPUTE),
                    _ => continue,
                };
                let remaining = limit.saturating_sub(next[index]);
                if amount > remaining {
                    return Err(MeterError::Budget(BudgetError::Exhausted {
                        dimension,
                        requested: amount,
                        remaining,
                    }));
                }
                next[index] += amount;
            }
            self.inner.charge_batch(charges)?;
            self.used = next;
            Ok(())
        })();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

/// Source-bound exact result. The untrusted certificate can be checked again
/// by the independent proof kernel; fields cannot manufacture acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementaryRationalDefiniteIntegral {
    certificate: ElementaryRationalDefiniteIntegralCertificate,
    verification: ElementaryRationalDefiniteIntegralVerification,
}

impl ElementaryRationalDefiniteIntegral {
    pub fn value(&self) -> &Expr {
        self.verification.value()
    }

    pub fn certificate(&self) -> &ElementaryRationalDefiniteIntegralCertificate {
        &self.certificate
    }

    pub fn verification(&self) -> &ElementaryRationalDefiniteIntegralVerification {
        &self.verification
    }
}

/// Preserve the original interruption when a polynomial routine wraps a
/// meter failure in its own error type. All phases share the caller's meter.
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

fn height(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate(poly: &UnivariatePoly, max_bits: u64) -> Result<()> {
    poly.validate_shape()?;
    if poly.gen_sym.name.len() > 65_536
        || poly.degree().unwrap_or(0) > MAX_DEGREE
        || poly.coeffs.iter().any(|value| height(value) > max_bits)
    {
        return Err(ElementaryRationalDefiniteIntegralError::Refused(
            "polynomial degree or coefficient-height limit",
        ));
    }
    Ok(())
}

/// Prepay the synchronous Euclidean batch, including conservative coefficient
/// growth. Cancellation is checked between batches, never claimed inside an
/// uncooperative operation.
fn batch(polynomials: &[&UnivariatePoly], meter: &mut impl BudgetMeter) -> Result<()> {
    meter.checkpoint()?;
    let mut width = 1u64;
    let mut bits = 1u64;
    let mut symbol_bytes = 0u64;
    for poly in polynomials {
        validate(poly, MAX_WORK_BITS)?;
        symbol_bytes = symbol_bytes.max(poly.gen_sym.name.len() as u64);
        width = width.max(poly.coeffs.len() as u64);
        bits = bits.max(poly.coeffs.iter().map(height).max().unwrap_or(1));
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
            work.saturating_mul(growth.div_ceil(8).max(1).saturating_add(2 * symbol_bytes)),
        ),
    ])?;
    Ok(())
}

fn copy_coefficients(
    coefficients: &[BigRational],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<BigRational>> {
    meter.checkpoint()?;
    let bytes = coefficients.iter().fold(0u64, |total, value| {
        total.saturating_add(64 + 2 * height(value).div_ceil(8))
    });
    meter.charge_batch(&[
        (Dimension::ComputeSteps, coefficients.len() as u64),
        (Dimension::MemoryBytes, bytes),
        (
            Dimension::AllocationCount,
            1 + 2 * coefficients.len() as u64,
        ),
    ])?;
    Ok(coefficients.to_vec())
}

fn copy_local(
    certificate: &RationalLocalCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<RationalLocalCertificate> {
    Ok(RationalLocalCertificate {
        numerator_order: certificate.numerator_order,
        denominator_order: certificate.denominator_order,
        numerator_unit: copy_coefficients(&certificate.numerator_unit, meter)?,
        denominator_unit: copy_coefficients(&certificate.denominator_unit, meter)?,
    })
}

fn exact_quotient(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
) -> Result<UnivariatePoly> {
    let (quotient, remainder) = numerator.div_rem(denominator)?;
    if !remainder.is_zero() {
        return Err(ElementaryRationalDefiniteIntegralError::Refused(
            "nonexact primitive factor division",
        ));
    }
    Ok(quotient)
}

fn logarithmic_coefficient(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<Option<BigRational>> {
    batch(&[numerator, denominator], meter)?;
    let derivative = denominator.derivative();
    if numerator.degree() != derivative.degree() || derivative.is_zero() {
        return Ok(None);
    }
    let coefficient = numerator
        .leading_coeff()
        .metered_div(&derivative.leading_coeff(), meter)?;
    let multiple = derivative.mul(&UnivariatePoly::new(
        denominator.gen_sym.clone(),
        vec![coefficient.clone()],
    ))?;
    Ok((multiple == *numerator).then_some(coefficient))
}

pub fn elementary_rational_definite_integral(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
) -> Result<ElementaryRationalDefiniteIntegral> {
    metered_elementary_rational_definite_integral(
        numerator,
        denominator,
        lower,
        upper,
        &mut Unbounded,
        &mut Unbounded,
    )
}

/// Complete supported elementary primitive with exact real endpoint limits.
/// The original denominator must retain the source's excluded points, or the
/// caller must independently check every original guard on the same interval.
/// A generator refusal never resets either budget or accepts a partial result.
pub fn metered_elementary_rational_definite_integral(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<ElementaryRationalDefiniteIntegral> {
    // Intrinsic caps also apply to the convenience API's unbounded meters.
    // Caller ledgers continue across all nested generator/checker operations.
    let mut generator = CappedMeter::new(generator_meter);
    let mut verifier = CappedMeter::new(verifier_meter);
    let result = generate_and_verify(
        numerator,
        denominator,
        lower,
        upper,
        &mut generator,
        &mut verifier,
    );
    if let Some(error) = generator.stopped.or(verifier.stopped) {
        return Err(ElementaryRationalDefiniteIntegralError::Meter(error));
    }
    result
}

fn generate_and_verify(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<ElementaryRationalDefiniteIntegral> {
    generator_meter.checkpoint()?;
    validate(numerator, MAX_INPUT_BITS)?;
    validate(denominator, MAX_INPUT_BITS)?;
    if numerator.gen_sym != denominator.gen_sym {
        return Err(ElementaryRationalDefiniteIntegralError::Refused(
            "different polynomial generators",
        ));
    }
    // Domain admission precedes cancellation and any factor search.
    verify_rational_integral_domain(&denominator.coeffs, lower, upper, verifier_meter)?;
    let primitive =
        metered_hermite_reduce(numerator, denominator, generator_meter, verifier_meter)?;
    generator_meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            256 + 4 * numerator.gen_sym.name.len() as u64,
        ),
        (Dimension::AllocationCount, 8),
    ])?;
    let remainder = UnivariatePoly::new(
        numerator.gen_sym.clone(),
        copy_coefficients(&primitive.remainder_numerator, generator_meter)?,
    );
    let remainder_denominator = UnivariatePoly::new(
        numerator.gen_sym.clone(),
        copy_coefficients(&primitive.remainder_denominator, generator_meter)?,
    );
    let mut terms = Vec::new();
    if !remainder.is_zero() {
        // A whole logarithmic derivative needs no irreducible factor search,
        // including high-degree irreducible polynomial arguments.
        if let Some(coefficient) =
            logarithmic_coefficient(&remainder, &remainder_denominator, generator_meter)?
        {
            generator_meter.charge_batch(&[
                (Dimension::MemoryBytes, 256),
                (Dimension::AllocationCount, 4),
            ])?;
            terms.push(ElementaryRationalTerm {
                denominator: copy_coefficients(&remainder_denominator.coeffs, generator_meter)?,
                pole_cofactor: vec![BigRational::one()],
                logarithm_coefficient: coefficient,
                reciprocal_coefficient: BigRational::zero(),
            });
        } else {
            let mut forwarded = ForwardMeter {
                inner: generator_meter,
                stopped: None,
            };
            let factorization =
                metered_complete_factorization(&remainder_denominator, &mut forwarded);
            if let Some(error) = forwarded.stopped {
                return Err(error.into());
            }
            let factorization = factorization?;
            if !factorization.scale.is_one() {
                return Err(ElementaryRationalDefiniteIntegralError::Refused(
                    "nonmonic Hermite remainder denominator",
                ));
            }
            for factor in factorization.factors {
                if factor.multiplicity != 1 {
                    return Err(ElementaryRationalDefiniteIntegralError::Refused(
                        "nonsquarefree Hermite remainder denominator",
                    ));
                }
                let denominator = factor.poly;
                batch(
                    &[&remainder, &remainder_denominator, &denominator],
                    generator_meter,
                )?;
                let cofactor = exact_quotient(&remainder_denominator, &denominator)?;
                let inverse = cofactor.extended_gcd(&denominator)?;
                if !inverse.gcd.is_one() {
                    return Err(ElementaryRationalDefiniteIntegralError::Refused(
                        "noncoprime primitive factors",
                    ));
                }
                let local = remainder.mul(&inverse.u)?.div_rem(&denominator)?.1;
                let (logarithm, reciprocal) = match denominator.degree() {
                    Some(1) => (local.coeffs[0].clone(), BigRational::zero()),
                    Some(2) => {
                        let linear = local
                            .coeffs
                            .get(1)
                            .cloned()
                            .unwrap_or_else(BigRational::zero);
                        let logarithm =
                            linear.metered_div(&BigRational::from(2), generator_meter)?;
                        let shifted =
                            logarithm.metered_mul(&denominator.coeffs[1], generator_meter)?;
                        let reciprocal = local.coeffs[0].metered_sub(&shifted, generator_meter)?;
                        (logarithm, reciprocal)
                    }
                    _ => (
                        logarithmic_coefficient(&local, &denominator, generator_meter)?
                            .ok_or(ElementaryRationalDefiniteIntegralError::UnsupportedPrimitive)?,
                        BigRational::zero(),
                    ),
                };
                generator_meter.charge_batch(&[
                    (Dimension::MemoryBytes, 256),
                    (Dimension::AllocationCount, 4),
                ])?;
                terms.push(ElementaryRationalTerm {
                    denominator: denominator.coeffs,
                    pole_cofactor: cofactor.coeffs,
                    logarithm_coefficient: logarithm,
                    reciprocal_coefficient: reciprocal,
                });
            }
        }
    }
    generator_meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            256 + 4 * numerator.gen_sym.name.len() as u64,
        ),
        (Dimension::AllocationCount, 8),
    ])?;
    let rational_numerator = UnivariatePoly::new(
        numerator.gen_sym.clone(),
        copy_coefficients(&primitive.rational_numerator, generator_meter)?,
    );
    let rational_denominator = UnivariatePoly::new(
        numerator.gen_sym.clone(),
        copy_coefficients(&primitive.rational_denominator, generator_meter)?,
    );
    let lower_point = lower.metered_limit_point(generator_meter)?;
    let left = metered_rational_local_order(
        &rational_numerator,
        &rational_denominator,
        &lower_point,
        generator_meter,
        verifier_meter,
    )?;
    let upper_point = upper.metered_limit_point(generator_meter)?;
    let right = metered_rational_local_order(
        &rational_numerator,
        &rational_denominator,
        &upper_point,
        generator_meter,
        verifier_meter,
    )?;
    let certificate = ElementaryRationalDefiniteIntegralCertificate {
        primitive,
        terms,
        lower_limit: copy_local(left.certificate(), generator_meter)?,
        upper_limit: copy_local(right.certificate(), generator_meter)?,
    };
    let verification = verify_elementary_rational_definite_integral(
        &numerator.coeffs,
        &denominator.coeffs,
        lower,
        upper,
        &certificate,
        verifier_meter,
    )?;
    generator_meter.checkpoint()?;
    Ok(ElementaryRationalDefiniteIntegral {
        certificate,
        verification,
    })
}
