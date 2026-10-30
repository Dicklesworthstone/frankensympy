//! Exact real definite integrals whose Hermite remainder vanishes.
//!
//! This composes factor-free rational integration with independently checked
//! endpoint limits and an independent Sturm domain check. Irreducible repeated
//! denominators need not be split into roots. A nonzero logarithmic/arctangent
//! remainder is explicitly unsupported by this certificate family.

use crate::elementary_rational_definite::{
    ElementaryRationalDefiniteIntegralError, elementary_rational_definite_integral,
};
use crate::hermite::{HermiteError, metered_hermite_reduce};
use crate::rational_limits::{RationalLimitError, metered_rational_local_order};
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_definite::{
    RationalDefiniteIntegralCertificate, RationalDefiniteIntegralVerification,
    RationalDefiniteIntegralVerificationError, RationalIntegralEndpoint,
    verify_rational_definite_integral, verify_rational_integral_domain,
};
use fsym_proof_kernel::rational_limit::{
    LocalDirection, RationalDirectionalLimit, RationalLocalCertificate,
};
use num_traits::Zero;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RationalDefiniteIntegralError {
    #[error("rational definite integration refused: {0}")]
    Refused(&'static str),
    #[error("the integral requires a nonrational primitive")]
    UnsupportedPrimitive,
    #[error("the rational primitive does not have finite endpoint limits")]
    DivergentEndpoint,
    #[error(transparent)]
    Hermite(#[from] HermiteError),
    #[error(transparent)]
    Elementary(#[from] ElementaryRationalDefiniteIntegralError),
    #[error(transparent)]
    Local(#[from] RationalLimitError),
    #[error(transparent)]
    Verification(#[from] RationalDefiniteIntegralVerificationError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalDefiniteIntegralError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalDefiniteIntegralError>;

/// Checked result plus a portable untrusted witness for a later independent
/// recheck. No public constructor or unchecked decoder creates an acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalDefiniteIntegral {
    certificate: RationalDefiniteIntegralCertificate,
    verification: RationalDefiniteIntegralVerification,
}

impl RationalDefiniteIntegral {
    pub fn value(&self) -> &BigRational {
        self.verification.value()
    }

    pub fn certificate(&self) -> &RationalDefiniteIntegralCertificate {
        &self.certificate
    }

    pub fn verification(&self) -> &RationalDefiniteIntegralVerification {
        &self.verification
    }
}

fn reserve_endpoint_certificate(
    certificate: &RationalLocalCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    meter.checkpoint()?;
    let coefficients = certificate
        .numerator_unit
        .iter()
        .chain(&certificate.denominator_unit);
    let count = certificate.numerator_unit.len() + certificate.denominator_unit.len();
    let mut bytes = 128u64;
    for value in coefficients {
        let bits = value.numer().bits().max(value.denom().bits()).max(1);
        bytes = bytes.saturating_add(64 + 2 * bits.div_ceil(8));
    }
    meter.charge_batch(&[
        (Dimension::AllocationCount, 2 + 2 * count as u64),
        (Dimension::MemoryBytes, bytes),
    ])?;
    Ok(())
}

pub fn rational_definite_integral(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
) -> Result<RationalDefiniteIntegral> {
    metered_rational_definite_integral(
        numerator,
        denominator,
        lower,
        upper,
        &mut Unbounded,
        &mut Unbounded,
    )
}

/// Generator and checker consume separate caller meters. The original
/// denominator must include any domain guards erased by rational cancellation.
/// There is no unbounded retry after a refusal, cancellation or exhaustion.
pub fn metered_rational_definite_integral(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<RationalDefiniteIntegral> {
    generator_meter.checkpoint()?;
    if numerator.gen_sym != denominator.gen_sym {
        return Err(RationalDefiniteIntegralError::Refused(
            "different polynomial generators",
        ));
    }
    // Domain failures are checked before trying a different primitive family;
    // even an algebraically removable original interior hole stays excluded.
    verify_rational_integral_domain(&denominator.coeffs, lower, upper, verifier_meter)?;
    let primitive =
        metered_hermite_reduce(numerator, denominator, generator_meter, verifier_meter)?;
    if !primitive.remainder_numerator.iter().all(Zero::is_zero) {
        return Err(RationalDefiniteIntegralError::UnsupportedPrimitive);
    }
    let copied_coefficients =
        (primitive.rational_numerator.len() + primitive.rational_denominator.len()) as u64;
    generator_meter.charge_batch(&[
        (Dimension::AllocationCount, 4 + 2 * copied_coefficients),
        (
            Dimension::MemoryBytes,
            copied_coefficients
                .saturating_mul(64 + 2 * 65_536u64.div_ceil(8))
                .saturating_add(128 + 2 * numerator.gen_sym.name.len() as u64),
        ),
    ])?;
    let top = UnivariatePoly::new(
        numerator.gen_sym.clone(),
        primitive.rational_numerator.clone(),
    );
    let bottom = UnivariatePoly::new(
        numerator.gen_sym.clone(),
        primitive.rational_denominator.clone(),
    );
    let lower_point = lower.metered_limit_point(generator_meter)?;
    let left =
        metered_rational_local_order(&top, &bottom, &lower_point, generator_meter, verifier_meter)?;
    let upper_point = upper.metered_limit_point(generator_meter)?;
    let right =
        metered_rational_local_order(&top, &bottom, &upper_point, generator_meter, verifier_meter)?;
    let left_bits = left
        .verification()
        .leading_coefficient()
        .numer()
        .bits()
        .max(left.verification().leading_coefficient().denom().bits())
        .max(1);
    let right_bits = right
        .verification()
        .leading_coefficient()
        .numer()
        .bits()
        .max(right.verification().leading_coefficient().denom().bits())
        .max(1);
    generator_meter.charge_batch(&[
        (Dimension::AllocationCount, 4),
        (
            Dimension::MemoryBytes,
            128 + 2 * left_bits.div_ceil(8) + 2 * right_bits.div_ceil(8),
        ),
    ])?;
    let RationalDirectionalLimit::Finite(left_value) =
        left.verification().limit(LocalDirection::Both)
    else {
        return Err(RationalDefiniteIntegralError::DivergentEndpoint);
    };
    let RationalDirectionalLimit::Finite(right_value) =
        right.verification().limit(LocalDirection::Both)
    else {
        return Err(RationalDefiniteIntegralError::DivergentEndpoint);
    };
    let value = right_value.metered_sub(&left_value, generator_meter)?;
    reserve_endpoint_certificate(left.certificate(), generator_meter)?;
    reserve_endpoint_certificate(right.certificate(), generator_meter)?;
    let certificate = RationalDefiniteIntegralCertificate {
        primitive,
        lower_limit: left.certificate().clone(),
        upper_limit: right.certificate().clone(),
        value,
    };
    let verification = verify_rational_definite_integral(
        &numerator.coeffs,
        &denominator.coeffs,
        lower,
        upper,
        &certificate,
        verifier_meter,
    )?;
    generator_meter.checkpoint()?;
    Ok(RationalDefiniteIntegral {
        certificate,
        verification,
    })
}

fn endpoint(value: &Expr) -> Option<RationalIntegralEndpoint> {
    Some(match value {
        Expr::Integer(integer) => {
            RationalIntegralEndpoint::Finite(BigRational::from_integer(integer.clone()))
        }
        Expr::Rational(rational) => RationalIntegralEndpoint::Finite(rational.clone()),
        Expr::Const(Constant::Infinity) => RationalIntegralEndpoint::PositiveInfinity,
        Expr::Const(Constant::NegativeInfinity) => RationalIntegralEndpoint::NegativeInfinity,
        _ => return None,
    })
}

/// `None` permits existing nonrational primitive families. A recognized
/// rational domain/resource failure is terminal and cannot be erased by
/// simplification. Divergent endpoint values retain the older extended-real
/// API behavior, without receiving this finite-integral certificate.
pub(crate) fn try_definite(
    expression: &Expr,
    variable: &Symbol,
    lower: &Expr,
    upper: &Expr,
) -> Option<Result<Expr>> {
    let (lower, upper) = (endpoint(lower)?, endpoint(upper)?);
    let quotient = match crate::rational_sum::is_rational_quotient(expression, variable) {
        Some(quotient) => quotient,
        None => {
            return Some(Err(RationalDefiniteIntegralError::Refused(
                "source expression traversal limit",
            )));
        }
    };
    let Some((numerator, denominator)) =
        crate::rational_sum::original_rational_parts(expression, variable)
    else {
        return quotient.then_some(Err(RationalDefiniteIntegralError::Refused(
            "original rational source or coefficient limit",
        )));
    };
    // Complete the logarithmic/atan remainder before the legacy endpoint
    // simplifier sees it. In particular, coupled logarithms may have finite
    // limits at infinity even though each separate logarithm diverges.
    // The older factor-free rational-only lane retains its larger degree
    // envelope; no algorithm is retried after a meter or domain failure.
    if numerator.degree().unwrap_or(0) <= 32 && denominator.degree().unwrap_or(0) <= 32 {
        return match elementary_rational_definite_integral(&numerator, &denominator, &lower, &upper)
        {
            Ok(result) => Some(Ok(result.value().clone())),
            Err(
                ElementaryRationalDefiniteIntegralError::UnsupportedPrimitive
                | ElementaryRationalDefiniteIntegralError::DivergentEndpoint,
            ) => None,
            Err(error) => Some(Err(error.into())),
        };
    }
    match rational_definite_integral(&numerator, &denominator, &lower, &upper) {
        Ok(result) => Some(Ok(fsym_core::elementary::rational_expr(
            result.value().clone(),
        ))),
        Err(
            RationalDefiniteIntegralError::UnsupportedPrimitive
            | RationalDefiniteIntegralError::DivergentEndpoint,
        ) => None,
        Err(error) => Some(Err(error)),
    }
}
