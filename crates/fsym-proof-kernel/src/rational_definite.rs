//! Checked real definite integrals with a rational antiderivative.
//!
//! The checker establishes the derivative identity, the absence of zeros of
//! the original denominator in the open integration interval, and finite
//! endpoint limits. Endpoints may be rational or either infinity. Thus an
//! accepted value is an ordinary integral or a convergent improper integral,
//! never a Cauchy principal value. Interior removable holes are conservatively
//! refused: a caller must split such an integral explicitly.
//!
//! Sturm arithmetic here is a bounded reference implementation independent
//! of the polynomial and calculus generators. Endpoint and derivative
//! certificates are checked by their existing independent kernel families.

use crate::rational_integral::{
    RationalIntegralReduction, RationalIntegralVerificationError,
    verify_rational_integral_reduction,
};
use crate::rational_limit::{
    LocalDirection, RationalDirectionalLimit, RationalLimitPoint, RationalLocalCertificate,
    RationalLocalVerificationError, verify_rational_local_order,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::BigRational;
use fsym_core::arith::RationalArithmeticError;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use thiserror::Error;

pub const MAX_DEFINITE_COEFFICIENTS: usize = 129;
const MAX_INPUT_BITS: u64 = 4_096;
const MAX_WORK_BITS: u64 = 32_768;
const MAX_REFERENCE_STEPS: u64 = 2_000_000;
const MAX_REFERENCE_ALLOCATION: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RationalIntegralEndpoint {
    NegativeInfinity,
    Finite(BigRational),
    PositiveInfinity,
}

impl RationalIntegralEndpoint {
    pub fn limit_point(&self) -> RationalLimitPoint {
        match self {
            Self::Finite(value) => RationalLimitPoint::Finite(value.clone()),
            Self::NegativeInfinity | Self::PositiveInfinity => RationalLimitPoint::Infinity,
        }
    }

    /// Account for the finite rational copy before allocating its magnitude
    /// storage. Infinite endpoints have no coefficient payload.
    pub fn metered_limit_point(&self, meter: &mut impl BudgetMeter) -> Result<RationalLimitPoint> {
        meter.checkpoint()?;
        if let Self::Finite(value) = self {
            valid_scalar(value, MAX_INPUT_BITS)?;
            meter.charge_batch(&[
                (Dimension::AllocationCount, 2),
                (Dimension::MemoryBytes, 64 + 2 * bits(value).div_ceil(8)),
            ])?;
        }
        let point = self.limit_point();
        meter.checkpoint()?;
        Ok(point)
    }
}

/// Transportable, untrusted witness. A serialized receipt never replaces
/// rechecking against the original coefficient vectors and endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalDefiniteIntegralCertificate {
    pub primitive: RationalIntegralReduction,
    pub lower_limit: RationalLocalCertificate,
    pub upper_limit: RationalLocalCertificate,
    pub value: BigRational,
}

/// A successful check; private fields prevent constructing accepted values
/// from unverified transport data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalDefiniteIntegralVerification {
    value: BigRational,
}

impl RationalDefiniteIntegralVerification {
    pub fn value(&self) -> &BigRational {
        &self.value
    }
}

#[derive(Debug, Error)]
pub enum RationalDefiniteIntegralVerificationError {
    #[error("invalid rational definite integral: {0}")]
    Invalid(&'static str),
    #[error("original rational denominator vanishes inside the integration interval")]
    InteriorSingularity,
    #[error("the rational primitive has no finite endpoint limit")]
    DivergentEndpoint,
    #[error(transparent)]
    Reduction(#[from] RationalIntegralVerificationError),
    #[error(transparent)]
    Endpoint(#[from] RationalLocalVerificationError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalDefiniteIntegralVerificationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalDefiniteIntegralVerificationError>;

fn invalid(message: &'static str) -> RationalDefiniteIntegralVerificationError {
    RationalDefiniteIntegralVerificationError::Invalid(message)
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn valid_scalar(value: &BigRational, limit: u64) -> Result<()> {
    if !value.denom().is_positive() || bits(value) > limit {
        return Err(invalid("invalid rational or coefficient height limit"));
    }
    Ok(())
}

fn validate(poly: &[BigRational]) -> Result<()> {
    if poly.is_empty() || poly.len() > MAX_DEFINITE_COEFFICIENTS {
        return Err(invalid("polynomial coefficient count limit"));
    }
    if poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        return Err(invalid("noncanonical polynomial coefficients"));
    }
    for value in poly {
        valid_scalar(value, MAX_INPUT_BITS)?;
    }
    Ok(())
}

fn trim(poly: &mut Vec<BigRational>) {
    while poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        poly.pop();
    }
}

struct Reference<'a, M> {
    meter: &'a mut M,
    remaining: u64,
    allocation: u64,
}

impl<M: BudgetMeter> Reference<'_, M> {
    fn work(&mut self, count: u64) -> Result<()> {
        self.meter.checkpoint()?;
        self.remaining = self
            .remaining
            .checked_sub(count.max(1))
            .ok_or_else(|| invalid("Sturm reference work limit"))?;
        self.meter.charge(Dimension::ComputeSteps, count.max(1))?;
        Ok(())
    }

    fn reserve(&mut self, count: usize) -> Result<()> {
        if count > MAX_DEFINITE_COEFFICIENTS {
            return Err(invalid("Sturm polynomial growth limit"));
        }
        let bytes = (count as u64).saturating_mul(64 + 2 * MAX_WORK_BITS.div_ceil(8));
        self.allocation = self
            .allocation
            .checked_add(bytes)
            .ok_or_else(|| invalid("Sturm allocation limit"))?;
        if self.allocation > MAX_REFERENCE_ALLOCATION {
            return Err(invalid("Sturm cumulative allocation limit"));
        }
        self.work(count as u64)?;
        self.meter.charge_batch(&[
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, (2 * count + 1) as u64),
        ])?;
        Ok(())
    }

    fn result(&mut self, value: BigRational) -> Result<BigRational> {
        valid_scalar(&value, MAX_WORK_BITS)?;
        Ok(value)
    }

    fn scalar_work(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        let growth = bits(a).saturating_add(bits(b)).saturating_add(1);
        if growth > MAX_WORK_BITS {
            return Err(invalid("Sturm scalar growth preflight"));
        }
        self.allocation = self
            .allocation
            .checked_add(8 * (64 + growth.div_ceil(8)))
            .ok_or_else(|| invalid("Sturm scalar allocation limit"))?;
        if self.allocation > MAX_REFERENCE_ALLOCATION {
            return Err(invalid("Sturm cumulative allocation limit"));
        }
        self.work(1)
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.scalar_work(a, b)?;
        let value = a.metered_add(b, self.meter)?;
        self.result(value)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.scalar_work(a, b)?;
        let value = a.metered_mul(b, self.meter)?;
        self.result(value)
    }

    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.scalar_work(a, b)?;
        let value = a.metered_sub(b, self.meter)?;
        self.result(value)
    }

    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.scalar_work(a, b)?;
        let value = a.metered_div(b, self.meter)?;
        self.result(value)
    }

    fn copy(&mut self, poly: &[BigRational]) -> Result<Vec<BigRational>> {
        self.reserve(poly.len())?;
        Ok(poly.to_vec())
    }

    fn derivative(&mut self, poly: &[BigRational]) -> Result<Vec<BigRational>> {
        self.reserve(poly.len().saturating_sub(1).max(1))?;
        let mut output = Vec::with_capacity(poly.len().saturating_sub(1).max(1));
        for (index, value) in poly.iter().enumerate().skip(1) {
            output.push(self.mul(value, &BigRational::from_integer((index as u64).into()))?);
        }
        if output.is_empty() {
            output.push(BigRational::zero());
        }
        Ok(output)
    }

    fn remainder(
        &mut self,
        numerator: &[BigRational],
        denominator: &[BigRational],
    ) -> Result<Vec<BigRational>> {
        let mut remainder = self.copy(numerator)?;
        let lead = denominator
            .last()
            .ok_or_else(|| invalid("empty Sturm divisor"))?;
        if lead.is_zero() {
            return Err(invalid("zero Sturm divisor"));
        }
        while remainder.len() >= denominator.len() && !remainder.iter().all(Zero::is_zero) {
            self.work(1)?;
            let shift = remainder.len() - denominator.len();
            let factor = self.div(remainder.last().unwrap(), lead)?;
            for (index, value) in denominator.iter().enumerate() {
                let product = self.mul(&factor, value)?;
                remainder[index + shift] = self.sub(&remainder[index + shift], &product)?;
            }
            trim(&mut remainder);
        }
        Ok(remainder)
    }

    fn evaluate(&mut self, poly: &[BigRational], point: &BigRational) -> Result<BigRational> {
        let mut value = BigRational::zero();
        for coefficient in poly.iter().rev() {
            let product = self.mul(&value, point)?;
            value = self.add(&product, coefficient)?;
        }
        Ok(value)
    }

    /// Sign just inside the interval. Repeated endpoint roots require the
    /// first nonzero Taylor coefficient, not dropping zero sequence entries.
    fn endpoint_sign(
        &mut self,
        poly: &[BigRational],
        point: &RationalIntegralEndpoint,
        from_right: bool,
    ) -> Result<Ordering> {
        match point {
            RationalIntegralEndpoint::NegativeInfinity
            | RationalIntegralEndpoint::PositiveInfinity => {
                let sign = poly.last().unwrap().cmp(&BigRational::zero());
                if matches!(point, RationalIntegralEndpoint::NegativeInfinity)
                    && poly.len().is_multiple_of(2)
                {
                    Ok(sign.reverse())
                } else {
                    Ok(sign)
                }
            }
            RationalIntegralEndpoint::Finite(point) => {
                let mut current = self.copy(poly)?;
                let mut order = 0usize;
                loop {
                    let sign = self.evaluate(&current, point)?.cmp(&BigRational::zero());
                    if sign != Ordering::Equal || current.len() == 1 {
                        return Ok(if !from_right && !order.is_multiple_of(2) {
                            sign.reverse()
                        } else {
                            sign
                        });
                    }
                    current = self.derivative(&current)?;
                    order += 1;
                }
            }
        }
    }

    fn variations(
        &mut self,
        chain: &[Vec<BigRational>],
        point: &RationalIntegralEndpoint,
        from_right: bool,
    ) -> Result<usize> {
        let mut previous = None;
        let mut changes = 0usize;
        for poly in chain {
            let sign = self.endpoint_sign(poly, point, from_right)?;
            if sign != Ordering::Equal {
                if previous.is_some_and(|last| last != sign) {
                    changes += 1;
                }
                previous = Some(sign);
            }
        }
        Ok(changes)
    }
}

/// Check that the original rational domain is regular on the open interval.
/// Endpoint zeros are allowed only as potential improper endpoints; the full
/// integral checker separately requires finite primitive limits there.
pub fn verify_rational_integral_domain(
    denominator: &[BigRational],
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    meter.checkpoint()?;
    validate(denominator)?;
    if denominator.iter().all(Zero::is_zero) {
        return Err(invalid("zero original denominator"));
    }
    for endpoint in [lower, upper] {
        if let RationalIntegralEndpoint::Finite(value) = endpoint {
            valid_scalar(value, MAX_INPUT_BITS)?;
        }
    }
    let mut reference = Reference {
        meter,
        remaining: MAX_REFERENCE_STEPS,
        allocation: 0,
    };
    if lower == upper {
        return match lower {
            RationalIntegralEndpoint::Finite(point)
                if !reference.evaluate(denominator, point)?.is_zero() =>
            {
                Ok(())
            }
            _ => Err(invalid(
                "degenerate interval has no defined finite endpoint",
            )),
        };
    }
    let (left, right) = if lower < upper {
        (lower, upper)
    } else {
        (upper, lower)
    };
    let mut chain = vec![reference.copy(denominator)?];
    if denominator.len() > 1 {
        chain.push(reference.derivative(denominator)?);
        while chain.last().unwrap().len() > 1 {
            let n = chain.len();
            let remainder = reference.remainder(&chain[n - 2], &chain[n - 1])?;
            if remainder.iter().all(Zero::is_zero) {
                break;
            }
            reference.reserve(remainder.len())?;
            chain.push(remainder.into_iter().map(|value| -value).collect());
        }
    }
    let left_variations = reference.variations(&chain, left, true)?;
    let right_variations = reference.variations(&chain, right, false)?;
    if left_variations < right_variations {
        return Err(invalid("Sturm variation invariant"));
    }
    if left_variations != right_variations {
        return Err(RationalDefiniteIntegralVerificationError::InteriorSingularity);
    }
    reference.meter.checkpoint()?;
    Ok(())
}

/// Verify an exact finite definite-integral claim without trusting the
/// antiderivative generator, endpoint proposals, or serialized value.
pub fn verify_rational_definite_integral(
    numerator: &[BigRational],
    denominator: &[BigRational],
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
    certificate: &RationalDefiniteIntegralCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<RationalDefiniteIntegralVerification> {
    validate(numerator)?;
    valid_scalar(&certificate.value, MAX_WORK_BITS)?;
    verify_rational_integral_domain(denominator, lower, upper, meter)?;
    if !verify_rational_integral_reduction(numerator, denominator, &certificate.primitive, meter)?
        .is_rational_antiderivative()
    {
        return Err(invalid(
            "nonzero Hermite remainder is not a complete primitive",
        ));
    }
    let primitive = &certificate.primitive;
    let lower_point = lower.metered_limit_point(meter)?;
    let lower_check = verify_rational_local_order(
        &primitive.rational_numerator,
        &primitive.rational_denominator,
        &lower_point,
        &certificate.lower_limit,
        meter,
    )?;
    let upper_point = upper.metered_limit_point(meter)?;
    let upper_check = verify_rational_local_order(
        &primitive.rational_numerator,
        &primitive.rational_denominator,
        &upper_point,
        &certificate.upper_limit,
        meter,
    )?;
    // `limit` copies a checked leading coefficient when the local order is
    // zero; reserve both possible copies before constructing either value.
    meter.charge_batch(&[
        (Dimension::AllocationCount, 4),
        (
            Dimension::MemoryBytes,
            128 + 2 * bits(lower_check.leading_coefficient()).div_ceil(8)
                + 2 * bits(upper_check.leading_coefficient()).div_ceil(8),
        ),
    ])?;
    let RationalDirectionalLimit::Finite(left) = lower_check.limit(LocalDirection::Both) else {
        return Err(RationalDefiniteIntegralVerificationError::DivergentEndpoint);
    };
    let RationalDirectionalLimit::Finite(right) = upper_check.limit(LocalDirection::Both) else {
        return Err(RationalDefiniteIntegralVerificationError::DivergentEndpoint);
    };
    let value = right.metered_sub(&left, meter)?;
    valid_scalar(&value, MAX_WORK_BITS)?;
    if value != certificate.value {
        return Err(invalid(
            "claimed integral does not equal the checked endpoint difference",
        ));
    }
    meter.checkpoint()?;
    Ok(RationalDefiniteIntegralVerification { value })
}
