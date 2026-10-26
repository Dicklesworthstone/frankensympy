//! Independent certificates for rational integration over `Q(x)`.
//!
//! A reduction proves `N/D = (A/B)' + R/S`, that `S` is square-free,
//! and that neither output denominator introduces a new pole. It does not
//! establish a definite integral, convergence, or a logarithmic branch.
//! The checker uses its own dense coefficient arithmetic, with no dependency
//! on the polynomial or calculus generators.

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_RATIONAL_REDUCTION_COEFFICIENTS: usize = 257;
const MAX_INTERMEDIATE_COEFFICIENTS: usize = 4 * MAX_RATIONAL_REDUCTION_COEFFICIENTS;
const MAX_COEFFICIENT_BITS: u64 = 65_536;
const MAX_REFERENCE_STEPS: u64 = 10_000_000;

/// Untrusted dense coefficient vectors, in ascending degree order. Zero is
/// `[0]`; trailing zeros and empty vectors are rejected by the verifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalIntegralReduction {
    #[serde(deserialize_with = "bounded_coefficients")]
    pub rational_numerator: Vec<BigRational>,
    #[serde(deserialize_with = "bounded_coefficients")]
    pub rational_denominator: Vec<BigRational>,
    #[serde(deserialize_with = "bounded_coefficients")]
    pub remainder_numerator: Vec<BigRational>,
    #[serde(deserialize_with = "bounded_coefficients")]
    pub remainder_denominator: Vec<BigRational>,
    /// `D = B * rational_pole_cofactor`.
    #[serde(deserialize_with = "bounded_coefficients")]
    pub rational_pole_cofactor: Vec<BigRational>,
    /// `D = S * remainder_pole_cofactor`.
    #[serde(deserialize_with = "bounded_coefficients")]
    pub remainder_pole_cofactor: Vec<BigRational>,
    /// `square_free_u * S + square_free_v * S' = 1`.
    #[serde(deserialize_with = "bounded_coefficients")]
    pub square_free_u: Vec<BigRational>,
    #[serde(deserialize_with = "bounded_coefficients")]
    pub square_free_v: Vec<BigRational>,
}

fn bounded_coefficients<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<BigRational>, D::Error> {
    struct Coefficients;
    impl<'de> serde::de::Visitor<'de> for Coefficients {
        type Value = Vec<BigRational>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                formatter,
                "at most {MAX_RATIONAL_REDUCTION_COEFFICIENTS} coefficients"
            )
        }

        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            if seq.size_hint().unwrap_or(0) > MAX_RATIONAL_REDUCTION_COEFFICIENTS {
                return Err(serde::de::Error::custom(
                    "rational reduction coefficient limit",
                ));
            }
            let mut coefficients = Vec::new();
            while coefficients.len() < MAX_RATIONAL_REDUCTION_COEFFICIENTS {
                match seq.next_element::<BigRational>()? {
                    Some(value) => {
                        // Reject an oversized coefficient before retaining it or
                        // decoding the next one. The owned integer decoder also
                        // bounds each individual magnitude before allocation.
                        if coefficient_bits(&value) > MAX_COEFFICIENT_BITS {
                            return Err(serde::de::Error::custom(
                                "rational reduction coefficient height limit",
                            ));
                        }
                        coefficients.push(value);
                    }
                    None => return Ok(coefficients),
                }
            }
            if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                return Err(serde::de::Error::custom(
                    "rational reduction coefficient limit",
                ));
            }
            Ok(coefficients)
        }
    }
    deserializer.deserialize_seq(Coefficients)
}

/// A successful local verification, deliberately neither constructible nor
/// deserializable by a caller. Retained certificates must be checked again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalIntegralVerification {
    rational_antiderivative: bool,
}

impl RationalIntegralVerification {
    /// True only when the checked remainder is identically zero.
    pub fn is_rational_antiderivative(&self) -> bool {
        self.rational_antiderivative
    }
}

#[derive(Debug, Error)]
pub enum RationalIntegralVerificationError {
    #[error("invalid rational integral reduction: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalIntegralVerificationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type VerificationResult<T> = std::result::Result<T, RationalIntegralVerificationError>;

fn invalid(reason: &'static str) -> RationalIntegralVerificationError {
    RationalIntegralVerificationError::Invalid(reason)
}

fn coefficient_bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate(poly: &[BigRational]) -> VerificationResult<()> {
    if poly.is_empty() || poly.len() > MAX_RATIONAL_REDUCTION_COEFFICIENTS {
        return Err(invalid("coefficient count is outside the supported range"));
    }
    if poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        return Err(invalid("noncanonical trailing zero"));
    }
    if poly.iter().any(|c| {
        c.denom().is_zero() || c.denom().is_negative() || coefficient_bits(c) > MAX_COEFFICIENT_BITS
    }) {
        return Err(invalid("invalid or oversized rational coefficient"));
    }
    Ok(())
}

fn trim(mut poly: Vec<BigRational>) -> Vec<BigRational> {
    while poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        poly.pop();
    }
    poly
}

struct Reference<'a, M> {
    meter: &'a mut M,
    remaining: u64,
}

impl<M: BudgetMeter> Reference<'_, M> {
    fn work(&mut self, steps: u64) -> VerificationResult<()> {
        self.meter.checkpoint()?;
        self.remaining = self
            .remaining
            .checked_sub(steps)
            .ok_or_else(|| invalid("reference arithmetic work limit"))?;
        self.meter.charge(Dimension::ComputeSteps, steps.max(1))?;
        Ok(())
    }

    fn reserve(&mut self, len: usize, bits: u64) -> VerificationResult<()> {
        if len > MAX_INTERMEDIATE_COEFFICIENTS || bits > 4 * MAX_COEFFICIENT_BITS {
            return Err(invalid("reference arithmetic growth limit"));
        }
        self.work(len as u64)?;
        self.meter.charge_batch(&[
            (
                Dimension::MemoryBytes,
                (len as u64)
                    .saturating_mul(64 + 2 * bits.div_ceil(8))
                    .max(1),
            ),
            (Dimension::AllocationCount, (2 * len + 1) as u64),
        ])?;
        Ok(())
    }

    fn product(
        &mut self,
        a: &[BigRational],
        b: &[BigRational],
    ) -> VerificationResult<Vec<BigRational>> {
        let len = a.len() + b.len() - 1;
        let bits_a = a.iter().map(coefficient_bits).max().unwrap_or(1);
        let bits_b = b.iter().map(coefficient_bits).max().unwrap_or(1);
        self.reserve(len, bits_a + bits_b)?;
        let mut result = vec![BigRational::zero(); len];
        for (i, ai) in a.iter().enumerate() {
            self.work((b.len() as u64).saturating_mul(2))?;
            for (j, bj) in b.iter().enumerate() {
                let product = ai.metered_mul(bj, self.meter)?;
                result[i + j] = result[i + j].metered_add(&product, self.meter)?;
                if coefficient_bits(&result[i + j]) > MAX_COEFFICIENT_BITS {
                    return Err(invalid("reference coefficient growth limit"));
                }
            }
        }
        Ok(trim(result))
    }

    fn sum(
        &mut self,
        a: &[BigRational],
        b: &[BigRational],
        subtract: bool,
    ) -> VerificationResult<Vec<BigRational>> {
        let len = a.len().max(b.len());
        let bits = a.iter().chain(b).map(coefficient_bits).max().unwrap_or(1);
        self.reserve(len, 2 * bits + 1)?;
        let mut result = vec![BigRational::zero(); len];
        for (i, value) in result.iter_mut().enumerate() {
            self.work(2)?;
            if let Some(ai) = a.get(i) {
                *value = value.metered_add(ai, self.meter)?;
            }
            if let Some(bi) = b.get(i) {
                if subtract {
                    *value = value.metered_sub(bi, self.meter)?;
                } else {
                    *value = value.metered_add(bi, self.meter)?;
                }
            }
            if coefficient_bits(value) > MAX_COEFFICIENT_BITS {
                return Err(invalid("reference coefficient growth limit"));
            }
        }
        Ok(trim(result))
    }

    fn derivative(&mut self, a: &[BigRational]) -> VerificationResult<Vec<BigRational>> {
        let len = a.len().saturating_sub(1).max(1);
        let bits = a.iter().map(coefficient_bits).max().unwrap_or(1) + 16;
        self.reserve(len, bits)?;
        let mut result = vec![BigRational::zero(); len];
        for (degree, coefficient) in a.iter().enumerate().skip(1) {
            self.work(1)?;
            let multiplier = BigRational::from_integer(BigInt::from(degree as u64));
            result[degree - 1] = coefficient.metered_mul(&multiplier, self.meter)?;
            if coefficient_bits(&result[degree - 1]) > MAX_COEFFICIENT_BITS {
                return Err(invalid("reference coefficient growth limit"));
            }
        }
        Ok(trim(result))
    }
}

/// Verify the reduction against the original, caller-supplied `N/D`.
/// All equalities are exact coefficient identities, never samples. The pole
/// witnesses preserve the original domain even when input factors cancel.
pub fn verify_rational_integral_reduction(
    numerator: &[BigRational],
    denominator: &[BigRational],
    witness: &RationalIntegralReduction,
    meter: &mut impl BudgetMeter,
) -> VerificationResult<RationalIntegralVerification> {
    meter.checkpoint()?;
    let polynomials = [
        numerator,
        denominator,
        &witness.rational_numerator,
        &witness.rational_denominator,
        &witness.remainder_numerator,
        &witness.remainder_denominator,
        &witness.rational_pole_cofactor,
        &witness.remainder_pole_cofactor,
        &witness.square_free_u,
        &witness.square_free_v,
    ];
    for polynomial in polynomials {
        meter.charge(Dimension::ComputeSteps, polynomial.len().max(1) as u64)?;
        validate(polynomial)?;
    }
    for den in [
        denominator,
        &witness.rational_denominator,
        &witness.remainder_denominator,
    ] {
        if den.iter().all(Zero::is_zero) {
            return Err(invalid("zero denominator"));
        }
    }
    let mut reference = Reference {
        meter,
        remaining: MAX_REFERENCE_STEPS,
    };
    let a = &witness.rational_numerator;
    let b = &witness.rational_denominator;
    let r = &witness.remainder_numerator;
    let s = &witness.remainder_denominator;
    if reference.product(b, &witness.rational_pole_cofactor)? != denominator
        || reference.product(s, &witness.remainder_pole_cofactor)? != denominator
    {
        return Err(invalid("output denominator introduces an unproved pole"));
    }
    let ds = reference.derivative(s)?;
    let us = reference.product(&witness.square_free_u, s)?;
    let vds = reference.product(&witness.square_free_v, &ds)?;
    if reference.sum(&us, &vds, false)? != [BigRational::one()] {
        return Err(invalid(
            "remainder denominator is not certified square-free",
        ));
    }
    let da = reference.derivative(a)?;
    let db = reference.derivative(b)?;
    let dab = reference.product(&da, b)?;
    let adb = reference.product(a, &db)?;
    let derivative_numerator = reference.sum(&dab, &adb, true)?;
    let b_squared = reference.product(b, b)?;
    let derivative_term = reference.product(&derivative_numerator, s)?;
    let remainder_term = reference.product(r, &b_squared)?;
    let rhs_numerator = reference.sum(&derivative_term, &remainder_term, false)?;
    let rhs = reference.product(denominator, &rhs_numerator)?;
    let denominator_product = reference.product(&b_squared, s)?;
    let lhs = reference.product(numerator, &denominator_product)?;
    if lhs != rhs {
        return Err(invalid("rational derivative/remainder identity failed"));
    }
    reference.meter.checkpoint()?;
    Ok(RationalIntegralVerification {
        rational_antiderivative: r.iter().all(Zero::is_zero),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

    fn coefficients(values: &[i64]) -> Vec<BigRational> {
        values
            .iter()
            .map(|&value| BigRational::from_integer(value.into()))
            .collect()
    }

    fn reciprocal_derivative() -> RationalIntegralReduction {
        // Hand-constructed witness: 1/x^2 = (-1/x)' + 0/1.
        RationalIntegralReduction {
            rational_numerator: coefficients(&[-1]),
            rational_denominator: coefficients(&[0, 1]),
            remainder_numerator: coefficients(&[0]),
            remainder_denominator: coefficients(&[1]),
            rational_pole_cofactor: coefficients(&[0, 1]),
            remainder_pole_cofactor: coefficients(&[0, 0, 1]),
            square_free_u: coefficients(&[1]),
            square_free_v: coefficients(&[0]),
        }
    }

    #[test]
    fn hand_constructed_witness_is_checked_without_a_generator() {
        assert!(
            verify_rational_integral_reduction(
                &coefficients(&[1]),
                &coefficients(&[0, 0, 1]),
                &reciprocal_derivative(),
                &mut Unbounded,
            )
            .unwrap()
            .is_rational_antiderivative()
        );

        // The derivative identity alone holds for 1/1 = (0/x)' + 1/1,
        // but the witness introduces an input-domain hole at x = 0.
        let mut extra_pole = reciprocal_derivative();
        extra_pole.rational_numerator = coefficients(&[0]);
        extra_pole.remainder_numerator = coefficients(&[1]);
        extra_pole.rational_pole_cofactor = coefficients(&[1]);
        extra_pole.remainder_pole_cofactor = coefficients(&[1]);
        assert!(
            verify_rational_integral_reduction(
                &coefficients(&[1]),
                &coefficients(&[1]),
                &extra_pole,
                &mut Unbounded,
            )
            .is_err()
        );

        // The identity and pole witnesses hold, but x^2 is not square-free.
        let mut repeated_remainder = reciprocal_derivative();
        repeated_remainder.rational_numerator = coefficients(&[0]);
        repeated_remainder.rational_denominator = coefficients(&[1]);
        repeated_remainder.rational_pole_cofactor = coefficients(&[0, 0, 1]);
        repeated_remainder.remainder_numerator = coefficients(&[1]);
        repeated_remainder.remainder_denominator = coefficients(&[0, 0, 1]);
        repeated_remainder.remainder_pole_cofactor = coefficients(&[1]);
        assert!(
            verify_rational_integral_reduction(
                &coefficients(&[1]),
                &coefficients(&[0, 0, 1]),
                &repeated_remainder,
                &mut Unbounded,
            )
            .is_err()
        );
    }

    #[test]
    fn serialized_witness_bounds_each_coefficient_and_vector() {
        let witness = reciprocal_derivative();
        let encoded = serde_json::to_vec(&witness).unwrap();
        assert_eq!(
            serde_json::from_slice::<RationalIntegralReduction>(&encoded).unwrap(),
            witness
        );

        let mut oversized = witness.clone();
        oversized.rational_numerator = vec![BigRational::from_integer(
            BigInt::one() << MAX_COEFFICIENT_BITS as u32,
        )];
        let encoded = serde_json::to_vec(&oversized).unwrap();
        assert!(serde_json::from_slice::<RationalIntegralReduction>(&encoded).is_err());

        oversized = witness;
        oversized.rational_numerator =
            vec![BigRational::zero(); MAX_RATIONAL_REDUCTION_COEFFICIENTS + 1];
        let encoded = serde_json::to_vec(&oversized).unwrap();
        assert!(serde_json::from_slice::<RationalIntegralReduction>(&encoded).is_err());
    }
}
