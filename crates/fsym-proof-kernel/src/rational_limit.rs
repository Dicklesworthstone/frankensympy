//! Independent local-order and formal Laurent certificates over `Q(x)`.
//!
//! The local coordinate is `t = x - a` at a finite rational point and `t = 1/x`
//! at infinity. A witness factors the transformed numerator and denominator
//! as powers of `t` times polynomials nonzero at zero. The checker reconstructs
//! the original coefficient vectors without using a polynomial/limit generator.
//! This proves a punctured real-neighborhood asymptotic and its directional
//! extended-real limit, not a value at an excluded point or a general analytic
//! series/convergence claim.
//! Formal coefficient certificates additionally check a truncated polynomial
//! convolution, and any exact finite expansion must pass the full identity.

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::BigRational;
use fsym_core::arith::RationalArithmeticError;
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_LOCAL_COEFFICIENTS: usize = 129;
pub const MAX_LOCAL_COEFFICIENT_BITS: u64 = 16_384;
/// Output truncation bound; this does not enlarge the polynomial/local-unit
/// input envelope governed by `MAX_LOCAL_COEFFICIENTS`.
pub const MAX_SERIES_COEFFICIENTS: usize = 256;
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

/// Untrusted formal Laurent truncation in the coordinate of `local`.
/// Coefficients start at the independently checked local order. A nonexact
/// truncation includes every coefficient below the caller's requested order,
/// including zeros. An exact expansion may omit trailing zero coefficients,
/// but must satisfy the full rational identity, not just a truncated one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalSeriesCertificate {
    pub local: RationalLocalCertificate,
    #[serde(deserialize_with = "bounded_series_coefficients")]
    pub coefficients: Vec<BigRational>,
    pub exact: bool,
}

fn bounded_series_coefficients<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<BigRational>, D::Error> {
    struct Coefficients;
    impl<'de> serde::de::Visitor<'de> for Coefficients {
        type Value = Vec<BigRational>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                formatter,
                "at most {MAX_SERIES_COEFFICIENTS} Laurent coefficients"
            )
        }

        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            if seq.size_hint().unwrap_or(0) > MAX_SERIES_COEFFICIENTS {
                return Err(serde::de::Error::custom("Laurent coefficient count limit"));
            }
            let mut coefficients = Vec::new();
            while coefficients.len() < MAX_SERIES_COEFFICIENTS {
                match seq.next_element::<BigRational>()? {
                    Some(value) => {
                        if coefficient_bits(&value) > MAX_LOCAL_COEFFICIENT_BITS {
                            return Err(serde::de::Error::custom(
                                "Laurent coefficient height limit",
                            ));
                        }
                        coefficients.push(value);
                    }
                    None => return Ok(coefficients),
                }
            }
            if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                return Err(serde::de::Error::custom("Laurent coefficient count limit"));
            }
            Ok(coefficients)
        }
    }
    deserializer.deserialize_seq(Coefficients)
}

/// Checked formal coefficients, not a claim about convergence of a general
/// analytic series. Private fields prevent manufacturing an accepted tail or
/// marking a truncated result exact without the independent identity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalSeriesVerification {
    start_order: i64,
    coefficients: Vec<BigRational>,
    remainder_order: Option<i64>,
}

impl RationalSeriesVerification {
    pub fn start_order(&self) -> i64 {
        self.start_order
    }

    pub fn coefficients(&self) -> &[BigRational] {
        &self.coefficients
    }

    /// `None` proves an exact finite Laurent expression; otherwise all
    /// coefficients strictly below this absolute order have been checked.
    pub fn remainder_order(&self) -> Option<i64> {
        self.remainder_order
    }
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

    fn reserve_series(&mut self, count: usize) -> Result<()> {
        if count > MAX_SERIES_COEFFICIENTS {
            return Err(invalid("Laurent output coefficient count limit"));
        }
        self.work(count.max(1) as u64)?;
        self.meter.charge_batch(&[
            (Dimension::AllocationCount, (2 * count + 1) as u64),
            (
                Dimension::MemoryBytes,
                count.max(1) as u64 * (64 + 2 * MAX_LOCAL_COEFFICIENT_BITS.div_ceil(8)),
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

/// Check a formal Laurent truncation against the original `N/D`, expansion
/// point, and requested absolute order. After independently checking the
/// local units `U,V`, verify `V*C = U mod t^m` coefficient by coefficient.
/// An exactness claim additionally checks the complete polynomial identity.
/// No coefficient-generation recurrence is called by this verifier.
pub fn verify_rational_series(
    numerator: &[BigRational],
    denominator: &[BigRational],
    point: &RationalLimitPoint,
    absolute_order: i64,
    witness: &RationalSeriesCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<RationalSeriesVerification> {
    meter.checkpoint()?;
    if witness.coefficients.len() > MAX_SERIES_COEFFICIENTS {
        return Err(invalid("Laurent coefficient count limit"));
    }
    for coefficient in &witness.coefficients {
        validate(std::slice::from_ref(coefficient))?;
    }
    let local = verify_rational_local_order(numerator, denominator, point, &witness.local, meter)?;
    let mut reference = Reference {
        meter,
        remaining: MAX_REFERENCE_STEPS,
    };
    reference.reserve_series(witness.coefficients.len())?;
    let Some(start_order) = local.order() else {
        if !witness.exact || !witness.coefficients.is_empty() {
            return Err(invalid(
                "zero rational function requires an exact empty expansion",
            ));
        }
        reference.meter.checkpoint()?;
        return Ok(RationalSeriesVerification {
            start_order: 0,
            coefficients: Vec::new(),
            remainder_order: None,
        });
    };
    let required = if absolute_order <= start_order {
        0
    } else {
        absolute_order
            .checked_sub(start_order)
            .and_then(|count| usize::try_from(count).ok())
            .filter(|&count| count <= MAX_SERIES_COEFFICIENTS)
            .ok_or_else(|| invalid("requested Laurent precision exceeds the coefficient bound"))?
    };
    if witness.exact {
        if witness.coefficients.len() > required
            || witness.coefficients.is_empty()
            || witness.coefficients.last().is_some_and(Zero::is_zero)
        {
            return Err(invalid("noncanonical exact Laurent coefficient vector"));
        }
    } else if witness.coefficients.len() != required {
        return Err(invalid("missing or extra requested Laurent coefficients"));
    }
    let u = &witness.local.numerator_unit;
    let v = &witness.local.denominator_unit;
    let checked_count = if witness.exact {
        u.len().max(v.len() + witness.coefficients.len() - 1)
    } else {
        required
    };
    for degree in 0..checked_count {
        let mut coefficient = BigRational::zero();
        for (j, vj) in v.iter().enumerate().take(degree + 1) {
            if let Some(c) = witness.coefficients.get(degree - j) {
                let product = reference.product(vj, c)?;
                coefficient = reference.sum(&coefficient, &product)?;
            }
        }
        if coefficient != u.get(degree).cloned().unwrap_or_else(BigRational::zero) {
            return Err(invalid(
                "Laurent coefficients fail the original rational identity",
            ));
        }
    }
    let receipt = RationalSeriesVerification {
        start_order,
        coefficients: witness.coefficients.clone(),
        remainder_order: if witness.exact {
            None
        } else {
            Some(absolute_order)
        },
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

    fn fibonacci_series_witness() -> RationalSeriesCertificate {
        RationalSeriesCertificate {
            local: RationalLocalCertificate {
                numerator_order: Some(0),
                denominator_order: 0,
                numerator_unit: coefficients(&[1]),
                denominator_unit: coefficients(&[1, -1, -1]),
            },
            coefficients: coefficients(&[1, 1, 2, 3, 5, 8]),
            exact: false,
        }
    }

    #[test]
    fn rational_series_manual_truncations_exact_tail_and_zero() {
        let point = RationalLimitPoint::Finite(BigRational::zero());
        let witness = fibonacci_series_witness();
        let receipt = verify_rational_series(
            &coefficients(&[1]),
            &coefficients(&[1, -1, -1]),
            &point,
            6,
            &witness,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(receipt.start_order(), 0);
        assert_eq!(receipt.coefficients(), coefficients(&[1, 1, 2, 3, 5, 8]));
        assert_eq!(receipt.remainder_order(), Some(6));

        // (2+3t)/(t^2*(1-t)) = 2/t^2 + 5/t + 5 + 5t + 5t^2 + O(t^3).
        let pole = RationalSeriesCertificate {
            local: RationalLocalCertificate {
                numerator_order: Some(0),
                denominator_order: 2,
                numerator_unit: coefficients(&[2, 3]),
                denominator_unit: coefficients(&[1, -1]),
            },
            coefficients: coefficients(&[2, 5, 5, 5, 5]),
            exact: false,
        };
        let receipt = verify_rational_series(
            &coefficients(&[2, 3]),
            &coefficients(&[0, 0, 1, -1]),
            &point,
            3,
            &pole,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(receipt.start_order(), -2);
        assert_eq!(receipt.remainder_order(), Some(3));

        // (1-t^2)/(1-t) = 1+t exactly; requested order is higher than the
        // finite expansion, and exactness requires the whole identity.
        let exact = RationalSeriesCertificate {
            local: RationalLocalCertificate {
                numerator_order: Some(0),
                denominator_order: 0,
                numerator_unit: coefficients(&[1, 0, -1]),
                denominator_unit: coefficients(&[1, -1]),
            },
            coefficients: coefficients(&[1, 1]),
            exact: true,
        };
        assert_eq!(
            verify_rational_series(
                &coefficients(&[1, 0, -1]),
                &coefficients(&[1, -1]),
                &point,
                5,
                &exact,
                &mut Unbounded
            )
            .unwrap()
            .remainder_order(),
            None
        );
        let zero = RationalSeriesCertificate {
            local: RationalLocalCertificate {
                numerator_order: None,
                numerator_unit: coefficients(&[0]),
                ..exact.local
            },
            coefficients: vec![],
            exact: true,
        };
        let receipt = verify_rational_series(
            &coefficients(&[0]),
            &coefficients(&[1, -1]),
            &point,
            -3,
            &zero,
            &mut Unbounded,
        )
        .unwrap();
        assert!(receipt.coefficients().is_empty());
        assert_eq!(receipt.remainder_order(), None);
    }

    #[test]
    fn rational_series_rejects_coefficient_precision_and_exactness_mutants() {
        let point = RationalLimitPoint::Finite(BigRational::zero());
        let n = coefficients(&[1]);
        let d = coefficients(&[1, -1, -1]);
        let witness = fibonacci_series_witness();
        let epsilon = BigRational::new(BigInt::one(), BigInt::from(10).pow(100));
        let mut mutants = Vec::new();
        let mut bad = witness.clone();
        bad.coefficients[0] = bad.coefficients[0].clone() + epsilon.clone();
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.coefficients[5] = bad.coefficients[5].clone() + epsilon;
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.coefficients.pop();
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.coefficients.push(BigRational::one());
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.exact = true;
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.local.numerator_order = Some(1);
        mutants.push(bad);
        let mut bad = witness.clone();
        bad.local.denominator_unit[1] = BigRational::one();
        mutants.push(bad);
        for bad in mutants {
            assert!(verify_rational_series(&n, &d, &point, 6, &bad, &mut Unbounded).is_err());
        }
        assert!(verify_rational_series(&n, &d, &point, 7, &witness, &mut Unbounded).is_err());
        assert!(verify_rational_series(&n, &d, &point, 5, &witness, &mut Unbounded).is_err());
        assert!(
            verify_rational_series(
                &n,
                &d,
                &RationalLimitPoint::Infinity,
                6,
                &witness,
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            verify_rational_series(&coefficients(&[2]), &d, &point, 6, &witness, &mut Unbounded)
                .is_err()
        );

        let zeros = RationalSeriesCertificate {
            local: RationalLocalCertificate {
                numerator_order: Some(0),
                denominator_order: 0,
                numerator_unit: coefficients(&[1]),
                denominator_unit: coefficients(&[1, 0, 1]),
            },
            coefficients: coefficients(&[1, 0, -1, 0]),
            exact: false,
        };
        verify_rational_series(
            &n,
            &coefficients(&[1, 0, 1]),
            &point,
            4,
            &zeros,
            &mut Unbounded,
        )
        .unwrap();
        let mut missing_zero = zeros;
        missing_zero.coefficients.pop();
        assert!(
            verify_rational_series(
                &n,
                &coefficients(&[1, 0, 1]),
                &point,
                4,
                &missing_zero,
                &mut Unbounded
            )
            .is_err()
        );
    }

    #[test]
    fn rational_series_bounded_transport_and_verifier_budget() {
        let point = RationalLimitPoint::Finite(BigRational::zero());
        let witness = fibonacci_series_witness();
        let n = coefficients(&[1]);
        let d = coefficients(&[1, -1, -1]);
        let encoded = serde_json::to_vec(&witness).unwrap();
        let decoded: RationalSeriesCertificate = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, witness);
        verify_rational_series(&n, &d, &point, 6, &decoded, &mut Unbounded).unwrap();
        let mut unknown = serde_json::to_value(&witness).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("accepted".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<RationalSeriesCertificate>(unknown).is_err());
        let mut oversized = witness.clone();
        oversized.coefficients = vec![BigRational::one(); MAX_SERIES_COEFFICIENTS + 1];
        assert!(verify_rational_series(&n, &d, &point, 257, &oversized, &mut Unbounded).is_err());
        assert!(
            serde_json::from_slice::<RationalSeriesCertificate>(
                &serde_json::to_vec(&oversized).unwrap()
            )
            .is_err()
        );
        let mut huge = witness.clone();
        huge.coefficients[0] =
            BigRational::from_integer(BigInt::one() << MAX_LOCAL_COEFFICIENT_BITS as u32);
        assert!(verify_rational_series(&n, &d, &point, 6, &huge, &mut Unbounded).is_err());
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            verify_rational_series(&n, &d, &point, 6, &witness, &mut budget),
            Err(RationalLocalVerificationError::Meter(_))
        ));
    }
}
