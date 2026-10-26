//! Independent local-order certificates for rational functions over `Q(x)`.
//!
//! The local coordinate is `t = x - a` at a finite rational point and `t = 1/x`
//! at infinity. A witness factors the transformed numerator and denominator
//! as powers of `t` times polynomials nonzero at zero. The checker reconstructs
//! the original coefficient vectors without using a polynomial/limit generator.
//! This proves a punctured real-neighborhood asymptotic and its directional
//! extended-real limit, not a value at an excluded point or a general analytic
//! series/convergence claim.

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::BigRational;
use fsym_core::arith::RationalArithmeticError;
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_LOCAL_COEFFICIENTS: usize = 129;
pub const MAX_LOCAL_COEFFICIENT_BITS: u64 = 16_384;
const MAX_REFERENCE_STEPS: u64 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RationalLimitPoint {
    Finite(BigRational),
    /// Both ends of the real line use `t = 1/x`; choose the sign of `t`
    /// separately when asking the checked receipt for a directional limit.
    Infinity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalDirection {
    Positive,
    Negative,
    Both,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalDirectionalLimit {
    Finite(BigRational),
    PositiveInfinity,
    NegativeInfinity,
    DoesNotExist,
}

/// Untrusted witness. Coefficients are in ascending degree order; zero is
/// exactly `[0]`, and all other vectors must have nonzero last coefficients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalLocalCertificate {
    /// `None` is reserved for the identically zero numerator.
    pub numerator_order: Option<usize>,
    pub denominator_order: usize,
    #[serde(deserialize_with = "bounded_coefficients")]
    pub numerator_unit: Vec<BigRational>,
    #[serde(deserialize_with = "bounded_coefficients")]
    pub denominator_unit: Vec<BigRational>,
}

fn bounded_coefficients<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<BigRational>, D::Error> {
    struct Coefficients;
    impl<'de> serde::de::Visitor<'de> for Coefficients {
        type Value = Vec<BigRational>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                formatter,
                "at most {MAX_LOCAL_COEFFICIENTS} rational coefficients"
            )
        }

        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            if seq.size_hint().unwrap_or(0) > MAX_LOCAL_COEFFICIENTS {
                return Err(serde::de::Error::custom("local coefficient count limit"));
            }
            let mut coefficients = Vec::new();
            while coefficients.len() < MAX_LOCAL_COEFFICIENTS {
                match seq.next_element::<BigRational>()? {
                    Some(value) => {
                        if coefficient_bits(&value) > MAX_LOCAL_COEFFICIENT_BITS {
                            return Err(serde::de::Error::custom("local coefficient height limit"));
                        }
                        coefficients.push(value);
                    }
                    None => return Ok(coefficients),
                }
            }
            if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                return Err(serde::de::Error::custom("local coefficient count limit"));
            }
            Ok(coefficients)
        }
    }
    deserializer.deserialize_seq(Coefficients)
}

/// A successful local check. Private fields prevent a deserialized or
/// caller-constructed witness from manufacturing an accepted asymptotic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalLocalVerification {
    order: Option<i64>,
    leading_coefficient: BigRational,
}

impl RationalLocalVerification {
    /// The exact exponent in `N/D ~ leading_coefficient * t^order`, or
    /// `None` for the identically zero rational function.
    pub fn order(&self) -> Option<i64> {
        self.order
    }

    pub fn leading_coefficient(&self) -> &BigRational {
        &self.leading_coefficient
    }

    pub fn limit(&self, direction: LocalDirection) -> RationalDirectionalLimit {
        let Some(order) = self.order else {
            return RationalDirectionalLimit::Finite(BigRational::zero());
        };
        if order > 0 {
            return RationalDirectionalLimit::Finite(BigRational::zero());
        }
        if order == 0 {
            return RationalDirectionalLimit::Finite(self.leading_coefficient.clone());
        }
        let odd = order % 2 != 0;
        if odd && direction == LocalDirection::Both {
            return RationalDirectionalLimit::DoesNotExist;
        }
        let negative =
            self.leading_coefficient.is_negative() ^ (odd && direction == LocalDirection::Negative);
        if negative {
            RationalDirectionalLimit::NegativeInfinity
        } else {
            RationalDirectionalLimit::PositiveInfinity
        }
    }
}

#[derive(Debug, Error)]
pub enum RationalLocalVerificationError {
    #[error("invalid rational local-order certificate: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalLocalVerificationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalLocalVerificationError>;

fn invalid(reason: &'static str) -> RationalLocalVerificationError {
    RationalLocalVerificationError::Invalid(reason)
}

fn coefficient_bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate(poly: &[BigRational]) -> Result<()> {
    if poly.is_empty() || poly.len() > MAX_LOCAL_COEFFICIENTS {
        return Err(invalid("coefficient count limit"));
    }
    if poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        return Err(invalid("noncanonical trailing zero"));
    }
    if poly.iter().any(|c| {
        c.denom().is_zero()
            || c.denom().is_negative()
            || coefficient_bits(c) > MAX_LOCAL_COEFFICIENT_BITS
    }) {
        return Err(invalid("invalid rational or coefficient height limit"));
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
}

impl<M: BudgetMeter> Reference<'_, M> {
    fn work(&mut self, amount: u64) -> Result<()> {
        self.meter.checkpoint()?;
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or_else(|| invalid("reference work limit"))?;
        self.meter.charge(Dimension::ComputeSteps, amount)?;
        Ok(())
    }

    fn reserve(&mut self, count: usize) -> Result<()> {
        if count > MAX_LOCAL_COEFFICIENTS {
            return Err(invalid("reference polynomial degree limit"));
        }
        self.work(count as u64)?;
        self.meter.charge_batch(&[
            (Dimension::AllocationCount, (2 * count + 1) as u64),
            (
                Dimension::MemoryBytes,
                count as u64 * (64 + 2 * MAX_LOCAL_COEFFICIENT_BITS.div_ceil(8)),
            ),
        ])?;
        Ok(())
    }

    fn sum(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.work(1)?;
        let result = a.metered_add(b, self.meter)?;
        if coefficient_bits(&result) > MAX_LOCAL_COEFFICIENT_BITS {
            return Err(invalid("reference coefficient growth limit"));
        }
        Ok(result)
    }

    fn product(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.work(1)?;
        let result = a.metered_mul(b, self.meter)?;
        if coefficient_bits(&result) > MAX_LOCAL_COEFFICIENT_BITS {
            return Err(invalid("reference coefficient growth limit"));
        }
        Ok(result)
    }

    /// Reconstruct sum c_j (x-a)^j by successively forming the powers of
    /// x-a. The generator instead computes local coefficients by repeated
    /// synthetic division; it is not invoked here.
    fn reconstruct_finite(
        &mut self,
        local: &[BigRational],
        center: &BigRational,
    ) -> Result<Vec<BigRational>> {
        self.reserve(local.len())?;
        let mut output = vec![BigRational::zero(); local.len()];
        self.reserve(1)?;
        let mut power = vec![BigRational::one()];
        let negative_center = -center.clone();
        for (degree, coefficient) in local.iter().enumerate() {
            for (index, term) in power.iter().enumerate() {
                let scaled = self.product(coefficient, term)?;
                output[index] = self.sum(&output[index], &scaled)?;
            }
            if degree + 1 < local.len() {
                self.reserve(power.len() + 1)?;
                let mut next = vec![BigRational::zero(); power.len() + 1];
                for (index, term) in power.iter().enumerate() {
                    let shifted = self.product(term, &negative_center)?;
                    next[index] = self.sum(&next[index], &shifted)?;
                    next[index + 1] = self.sum(&next[index + 1], term)?;
                }
                power = next;
            }
        }
        trim(&mut output);
        Ok(output)
    }

    fn check_polynomial(
        &mut self,
        original: &[BigRational],
        unit: &[BigRational],
        order: usize,
        point: &RationalLimitPoint,
        common_degree: usize,
    ) -> Result<()> {
        let count = order
            .checked_add(unit.len())
            .ok_or_else(|| invalid("local-order overflow"))?;
        self.reserve(count)?;
        let mut local = vec![BigRational::zero(); order];
        local.extend_from_slice(unit);
        let reconstructed = match point {
            RationalLimitPoint::Finite(center) => self.reconstruct_finite(&local, center)?,
            RationalLimitPoint::Infinity => {
                if local.len() > common_degree + 1 {
                    return Err(invalid("reversed polynomial degree mismatch"));
                }
                self.reserve(common_degree + 1)?;
                local.resize(common_degree + 1, BigRational::zero());
                local.reverse();
                trim(&mut local);
                local
            }
        };
        if reconstructed != original {
            return Err(invalid(
                "local factorization does not reconstruct original polynomial",
            ));
        }
        Ok(())
    }
}

/// Check both local factorizations against the original numerator and
/// denominator. No cancellation of original denominator factors is required:
/// isolated exclusions are retained and the conclusion is punctured-local.
pub fn verify_rational_local_order(
    numerator: &[BigRational],
    denominator: &[BigRational],
    point: &RationalLimitPoint,
    witness: &RationalLocalCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<RationalLocalVerification> {
    meter.checkpoint()?;
    for poly in [
        numerator,
        denominator,
        &witness.numerator_unit,
        &witness.denominator_unit,
    ] {
        validate(poly)?;
        meter.charge(Dimension::ComputeSteps, poly.len() as u64)?;
    }
    if let RationalLimitPoint::Finite(center) = point {
        validate(std::slice::from_ref(center))?;
    }
    if denominator == [BigRational::zero()] {
        return Err(invalid("identically zero original denominator"));
    }
    if witness.denominator_unit[0].is_zero() {
        return Err(invalid("denominator unit vanishes at local origin"));
    }
    let numerator_zero = numerator == [BigRational::zero()];
    if numerator_zero {
        if witness.numerator_order.is_some() || witness.numerator_unit != [BigRational::zero()] {
            return Err(invalid("zero numerator requires the explicit zero witness"));
        }
    } else if witness.numerator_order.is_none() || witness.numerator_unit[0].is_zero() {
        return Err(invalid(
            "nonzero numerator requires a nonvanishing local unit",
        ));
    }
    let common_degree = numerator.len().max(denominator.len()) - 1;
    let mut reference = Reference {
        meter,
        remaining: MAX_REFERENCE_STEPS,
    };
    reference.check_polynomial(
        denominator,
        &witness.denominator_unit,
        witness.denominator_order,
        point,
        common_degree,
    )?;
    let receipt = if let Some(numerator_order) = witness.numerator_order {
        reference.check_polynomial(
            numerator,
            &witness.numerator_unit,
            numerator_order,
            point,
            common_degree,
        )?;
        let leading_coefficient =
            witness.numerator_unit[0].metered_div(&witness.denominator_unit[0], reference.meter)?;
        if coefficient_bits(&leading_coefficient) > MAX_LOCAL_COEFFICIENT_BITS {
            return Err(invalid("leading coefficient growth limit"));
        }
        RationalLocalVerification {
            order: Some(numerator_order as i64 - witness.denominator_order as i64),
            leading_coefficient,
        }
    } else {
        RationalLocalVerification {
            order: None,
            leading_coefficient: BigRational::zero(),
        }
    };
    reference.meter.checkpoint()?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, Unbounded};
    use fsym_core::BigInt;

    fn coefficients(values: &[i64]) -> Vec<BigRational> {
        values
            .iter()
            .map(|&v| BigRational::from_integer(v.into()))
            .collect()
    }

    fn finite_witness() -> RationalLocalCertificate {
        RationalLocalCertificate {
            numerator_order: Some(3),
            denominator_order: 2,
            numerator_unit: coefficients(&[3, 2]),
            denominator_unit: coefficients(&[5, -1]),
        }
    }

    #[test]
    fn rational_limit_manual_local_factorization_and_pole_directions() {
        // N=(x-2)^3*(3+2*(x-2)); D=(x-2)^2*(5-(x-2)).
        // Coefficients are hand-expanded, not produced by a generator.
        let n = coefficients(&[8, -28, 30, -13, 2]);
        let d = coefficients(&[28, -32, 11, -1]);
        let point = RationalLimitPoint::Finite(BigRational::from_integer(2.into()));
        let witness = finite_witness();
        let receipt =
            verify_rational_local_order(&n, &d, &point, &witness, &mut Unbounded).unwrap();
        assert_eq!(receipt.order(), Some(1));
        assert_eq!(
            receipt.leading_coefficient(),
            &BigRational::new(3.into(), 5.into())
        );
        assert_eq!(
            receipt.limit(LocalDirection::Both),
            RationalDirectionalLimit::Finite(BigRational::zero())
        );

        let pole = RationalLocalCertificate {
            numerator_order: Some(0),
            denominator_order: 3,
            numerator_unit: coefficients(&[1]),
            denominator_unit: coefficients(&[1]),
        };
        let receipt = verify_rational_local_order(
            &coefficients(&[1]),
            &coefficients(&[-8, 12, -6, 1]),
            &point,
            &pole,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(
            receipt.limit(LocalDirection::Positive),
            RationalDirectionalLimit::PositiveInfinity
        );
        assert_eq!(
            receipt.limit(LocalDirection::Negative),
            RationalDirectionalLimit::NegativeInfinity
        );
        assert_eq!(
            receipt.limit(LocalDirection::Both),
            RationalDirectionalLimit::DoesNotExist
        );

        let zero = RationalLocalCertificate {
            numerator_order: None,
            numerator_unit: coefficients(&[0]),
            ..pole
        };
        let receipt = verify_rational_local_order(
            &coefficients(&[0]),
            &coefficients(&[-8, 12, -6, 1]),
            &point,
            &zero,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(receipt.order(), None);
        assert_eq!(
            receipt.limit(LocalDirection::Both),
            RationalDirectionalLimit::Finite(BigRational::zero())
        );
    }

    #[test]
    fn rational_limit_manual_reciprocal_coordinate_at_infinity() {
        let witness = RationalLocalCertificate {
            numerator_order: Some(0),
            denominator_order: 1,
            numerator_unit: coefficients(&[-2]),
            denominator_unit: coefficients(&[1, 0, 1]),
        };
        let receipt = verify_rational_local_order(
            &coefficients(&[0, 0, 0, -2]),
            &coefficients(&[1, 0, 1]),
            &RationalLimitPoint::Infinity,
            &witness,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(receipt.order(), Some(-1));
        assert_eq!(
            receipt.limit(LocalDirection::Positive),
            RationalDirectionalLimit::NegativeInfinity
        );
        assert_eq!(
            receipt.limit(LocalDirection::Negative),
            RationalDirectionalLimit::PositiveInfinity
        );
    }

    #[test]
    fn rational_limit_rejects_mutated_orders_units_originals_and_point() {
        let n = coefficients(&[8, -28, 30, -13, 2]);
        let d = coefficients(&[28, -32, 11, -1]);
        let point = RationalLimitPoint::Finite(BigRational::from_integer(2.into()));
        let witness = finite_witness();
        let mut mutants = Vec::new();
        let mut bad = witness.clone();
        bad.numerator_order = Some(2);
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.denominator_order = 3;
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.numerator_order = None;
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.numerator_unit[0] = BigRational::zero();
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.denominator_unit[0] = BigRational::zero();
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.numerator_unit[1] = BigRational::from_integer(3.into());
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.denominator_unit[1] = BigRational::from_integer(1.into());
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.numerator_unit.push(BigRational::one());
        mutants.push(bad);
        for bad in mutants {
            assert!(verify_rational_local_order(&n, &d, &point, &bad, &mut Unbounded).is_err());
        }
        let epsilon = BigRational::new(BigInt::one(), BigInt::from(10).pow(80));
        let mut changed_n = n.clone();
        changed_n[0] = changed_n[0].clone() + epsilon.clone();
        let mut changed_d = d.clone();
        changed_d[0] = changed_d[0].clone() + epsilon;
        assert!(
            verify_rational_local_order(&changed_n, &d, &point, &witness, &mut Unbounded).is_err()
        );
        assert!(
            verify_rational_local_order(&n, &changed_d, &point, &witness, &mut Unbounded).is_err()
        );
        assert!(
            verify_rational_local_order(
                &n,
                &d,
                &RationalLimitPoint::Finite(BigRational::one()),
                &witness,
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            verify_rational_local_order(
                &n,
                &d,
                &RationalLimitPoint::Infinity,
                &witness,
                &mut Unbounded
            )
            .is_err()
        );
    }

    #[test]
    fn rational_limit_malformed_bounded_transport_and_verifier_budget() {
        let n = coefficients(&[8, -28, 30, -13, 2]);
        let d = coefficients(&[28, -32, 11, -1]);
        let point = RationalLimitPoint::Finite(BigRational::from_integer(2.into()));
        let witness = finite_witness();
        let mut malformed = Vec::new();
        let mut bad = witness.clone();
        bad.numerator_unit.clear();
        malformed.push(bad);
        let mut bad = witness.clone();
        bad.numerator_unit.push(BigRational::zero());
        malformed.push(bad);
        let mut bad = witness.clone();
        bad.denominator_order = usize::MAX;
        malformed.push(bad);
        let mut bad = witness.clone();
        bad.numerator_unit = vec![BigRational::one(); MAX_LOCAL_COEFFICIENTS + 1];
        malformed.push(bad);
        let mut bad = witness.clone();
        bad.numerator_unit[0] =
            BigRational::from_integer(BigInt::one() << MAX_LOCAL_COEFFICIENT_BITS as u32);
        malformed.push(bad);
        for bad in malformed {
            assert!(verify_rational_local_order(&n, &d, &point, &bad, &mut Unbounded).is_err());
        }
        assert!(
            verify_rational_local_order(&n, &coefficients(&[0]), &point, &witness, &mut Unbounded)
                .is_err()
        );
        assert!(verify_rational_local_order(&[], &d, &point, &witness, &mut Unbounded).is_err());
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            verify_rational_local_order(&n, &d, &point, &witness, &mut budget),
            Err(RationalLocalVerificationError::Meter(_))
        ));
        let encoded = serde_json::to_vec(&witness).unwrap();
        let decoded: RationalLocalCertificate = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, witness);
        verify_rational_local_order(&n, &d, &point, &decoded, &mut Unbounded).unwrap();
        let mut value = serde_json::to_value(&witness).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("verified".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<RationalLocalCertificate>(value).is_err());
        let mut oversized = witness;
        oversized.denominator_unit = vec![BigRational::one(); MAX_LOCAL_COEFFICIENTS + 1];
        assert!(
            serde_json::from_slice::<RationalLocalCertificate>(
                &serde_json::to_vec(&oversized).unwrap()
            )
            .is_err()
        );
    }
}
