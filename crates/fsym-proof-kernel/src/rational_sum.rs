//! Independent checking of rational antidifferences and integer sums.
//!
//! The identity `F(k + 1) - F(k) = N(k)/D(k)` is checked by dense
//! coefficient arithmetic, independently of the polynomial/calculus generators.
//! A sum additionally checks every original denominator pole and every
//! antidifference pole on its integer range. Infinite sums are admitted only
//! when the exact rational endpoint limit is finite. No analytic continuation
//! across a pole, principal value, or symbolic-bound claim is made.

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};
use thiserror::Error;

pub const MAX_SUM_COEFFICIENTS: usize = 129;
pub const MAX_SUM_COEFFICIENT_BITS: u64 = 16_384;
const MAX_INTERMEDIATE_COEFFICIENTS: usize = 4 * MAX_SUM_COEFFICIENTS;
const MAX_REFERENCE_STEPS: u64 = 4_000_000;

/// An exact rational root of a denominator, with its positive multiplicity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalSumPole {
    pub root: BigRational,
    pub multiplicity: u32,
}

/// Untrusted certificate for `F = numerator/denominator`. Coefficients are
/// ascending, nonempty, and without trailing zeros. Both pole lists must
/// reconstruct their entire denominator, including removable input poles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalAntidifference {
    pub numerator: Vec<BigRational>,
    pub denominator: Vec<BigRational>,
    pub input_poles: Vec<RationalSumPole>,
    pub antidifference_poles: Vec<RationalSumPole>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalSumUpper {
    Finite(BigInt),
    PositiveInfinity,
}

#[derive(Debug, Error)]
pub enum RationalSumVerificationError {
    #[error("rational summation verification refused: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalSumVerificationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalSumVerificationError>;

fn invalid(reason: &'static str) -> RationalSumVerificationError {
    RationalSumVerificationError::Invalid(reason)
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate(poly: &[BigRational]) -> Result<()> {
    if poly.is_empty() || poly.len() > MAX_SUM_COEFFICIENTS {
        return Err(invalid("coefficient count limit"));
    }
    if poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        return Err(invalid("noncanonical trailing zero"));
    }
    for coefficient in poly {
        if coefficient.denom().is_zero()
            || coefficient.denom().is_negative()
            || bits(coefficient) > MAX_SUM_COEFFICIENT_BITS
        {
            return Err(invalid("invalid or oversized coefficient"));
        }
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
    fn work(&mut self, amount: u64) -> Result<()> {
        self.meter.checkpoint()?;
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or_else(|| invalid("reference work limit"))?;
        self.meter.charge(Dimension::ComputeSteps, amount.max(1))?;
        Ok(())
    }

    fn reserve(&mut self, count: usize) -> Result<()> {
        if count > MAX_INTERMEDIATE_COEFFICIENTS {
            return Err(invalid("intermediate coefficient count limit"));
        }
        self.work(count.max(1) as u64)?;
        self.meter.charge_batch(&[
            (Dimension::AllocationCount, (count as u64).max(1)),
            (
                Dimension::MemoryBytes,
                (count as u64).max(1) * (64 + 2 * MAX_SUM_COEFFICIENT_BITS.div_ceil(8)),
            ),
        ])?;
        Ok(())
    }

    fn check_value(&self, value: BigRational) -> Result<BigRational> {
        if bits(&value) > MAX_SUM_COEFFICIENT_BITS {
            Err(invalid("reference coefficient growth limit"))
        } else {
            Ok(value)
        }
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.work(1)?;
        let value = a.metered_add(b, self.meter)?;
        self.check_value(value)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.work(1)?;
        let value = a.metered_mul(b, self.meter)?;
        self.check_value(value)
    }

    fn product(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Vec<BigRational>> {
        self.reserve(a.len() + b.len() - 1)?;
        let mut result = vec![BigRational::zero(); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                let product = self.mul(ai, bj)?;
                result[i + j] = self.add(&result[i + j], &product)?;
            }
        }
        Ok(trim(result))
    }

    fn difference(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Vec<BigRational>> {
        self.reserve(a.len().max(b.len()))?;
        let mut result = vec![BigRational::zero(); a.len().max(b.len())];
        for (i, coefficient) in result.iter_mut().enumerate() {
            let ai = a.get(i).cloned().unwrap_or_else(BigRational::zero);
            let bi = b.get(i).cloned().unwrap_or_else(BigRational::zero);
            *coefficient = self.add(&ai, &(-bi))?;
        }
        Ok(trim(result))
    }

    /// Horner composition with `x + 1`, without the generator's shift code.
    fn shift_one(&mut self, poly: &[BigRational]) -> Result<Vec<BigRational>> {
        self.reserve(poly.len())?;
        let mut result = vec![BigRational::zero(); poly.len()];
        for coefficient in poly.iter().rev() {
            for i in (1..poly.len()).rev() {
                result[i] = self.add(&result[i], &result[i - 1])?;
            }
            result[0] = self.add(&result[0], coefficient)?;
        }
        Ok(trim(result))
    }

    fn evaluate(&mut self, poly: &[BigRational], x: &BigRational) -> Result<BigRational> {
        let mut result = BigRational::zero();
        for coefficient in poly.iter().rev() {
            result = self.mul(&result, x)?;
            result = self.add(&result, coefficient)?;
        }
        Ok(result)
    }

    fn quotient(
        &mut self,
        numerator: BigRational,
        denominator: BigRational,
    ) -> Result<BigRational> {
        self.work(1)?;
        if denominator.is_zero() {
            return Err(invalid("undefined rational endpoint"));
        }
        let value = numerator.metered_div(&denominator, self.meter)?;
        self.check_value(value)
    }

    fn check_poles(&mut self, polynomial: &[BigRational], poles: &[RationalSumPole]) -> Result<()> {
        if polynomial.last().is_none_or(Zero::is_zero) {
            return Err(invalid("zero denominator"));
        }
        if poles.len() >= MAX_SUM_COEFFICIENTS {
            return Err(invalid("pole count limit"));
        }
        let mut degree = 0usize;
        for (i, pole) in poles.iter().enumerate() {
            if pole.multiplicity == 0
                || pole.root.denom().is_zero()
                || pole.root.denom().is_negative()
                || bits(&pole.root) > MAX_SUM_COEFFICIENT_BITS
            {
                return Err(invalid("invalid pole or multiplicity"));
            }
            if poles[..i].iter().any(|p| p.root == pole.root) {
                return Err(invalid("duplicate pole"));
            }
            degree = degree
                .checked_add(pole.multiplicity as usize)
                .ok_or_else(|| invalid("pole multiplicity overflow"))?;
            if degree >= polynomial.len() {
                return Err(invalid("pole degree exceeds denominator degree"));
            }
        }
        if degree + 1 != polynomial.len() {
            return Err(invalid("incomplete denominator pole list"));
        }
        let mut reconstructed = vec![polynomial.last().expect("validated nonempty").clone()];
        for pole in poles {
            for _ in 0..pole.multiplicity {
                reconstructed =
                    self.product(&reconstructed, &[-pole.root.clone(), BigRational::one()])?;
            }
        }
        if reconstructed != polynomial {
            return Err(invalid("denominator pole factorization mismatch"));
        }
        Ok(())
    }

    fn check_antidifference(
        &mut self,
        numerator: &[BigRational],
        denominator: &[BigRational],
        certificate: &RationalAntidifference,
    ) -> Result<()> {
        for poly in [
            numerator,
            denominator,
            &certificate.numerator,
            &certificate.denominator,
        ] {
            validate(poly)?;
        }
        self.check_poles(denominator, &certificate.input_poles)?;
        self.check_poles(&certificate.denominator, &certificate.antidifference_poles)?;
        let a_shift = self.shift_one(&certificate.numerator)?;
        let b_shift = self.shift_one(&certificate.denominator)?;
        let left = self.product(&a_shift, &certificate.denominator)?;
        let right = self.product(&certificate.numerator, &b_shift)?;
        let difference = self.difference(&left, &right)?;
        let lhs = self.product(denominator, &difference)?;
        let product_denominator = self.product(&certificate.denominator, &b_shift)?;
        let rhs = self.product(numerator, &product_denominator)?;
        if lhs != rhs {
            return Err(invalid("rational difference identity mismatch"));
        }
        Ok(())
    }
}

/// Checks only the rational identity and both complete pole lists. This
/// does not grant a definite or infinite sum without the domain checks below.
pub fn verify_rational_antidifference(
    numerator: &[BigRational],
    denominator: &[BigRational],
    certificate: &RationalAntidifference,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    Reference {
        meter,
        remaining: MAX_REFERENCE_STEPS,
    }
    .check_antidifference(numerator, denominator, certificate)
}

fn has_integer_pole(poles: &[RationalSumPole], lower: &BigInt, upper: Option<&BigInt>) -> bool {
    poles.iter().any(|pole| {
        pole.root.is_integer()
            && pole.root.numer() >= lower
            && upper.is_none_or(|hi| pole.root.numer() <= hi)
    })
}

/// Independently evaluates a checked rational telescoping sum. Finite
/// reversed ranges obey `sum(a,b) = -sum(b+1,a-1)`; an empty range is zero.
/// Bounds and coefficients remain arbitrary-precision within declared limits.
pub fn verify_rational_sum(
    numerator: &[BigRational],
    denominator: &[BigRational],
    certificate: &RationalAntidifference,
    lower: &BigInt,
    upper: &RationalSumUpper,
    meter: &mut impl BudgetMeter,
) -> Result<BigRational> {
    if lower.bits() > 4096
        || matches!(upper, RationalSumUpper::Finite(value) if value.bits() > 4096)
    {
        return Err(invalid("integer endpoint height limit"));
    }
    let mut reference = Reference {
        meter,
        remaining: MAX_REFERENCE_STEPS,
    };
    reference.check_antidifference(numerator, denominator, certificate)?;
    let (lo, hi, negate) = match upper {
        RationalSumUpper::Finite(hi) if hi < lower => {
            let next = hi + BigInt::one();
            if &next == lower {
                return Ok(BigRational::zero());
            }
            (next, Some(lower - BigInt::one()), true)
        }
        RationalSumUpper::Finite(hi) => (lower.clone(), Some(hi.clone()), false),
        RationalSumUpper::PositiveInfinity => (lower.clone(), None, false),
    };
    if has_integer_pole(&certificate.input_poles, &lo, hi.as_ref()) {
        return Err(invalid("original summand has an integer pole in the range"));
    }
    let after_hi = hi.as_ref().map(|value| value + BigInt::one());
    if has_integer_pole(&certificate.antidifference_poles, &lo, after_hi.as_ref()) {
        return Err(invalid("antidifference has an integer pole in the range"));
    }
    let start = BigRational::from_integer(lo);
    let start_numerator = reference.evaluate(&certificate.numerator, &start)?;
    let start_denominator = reference.evaluate(&certificate.denominator, &start)?;
    let start_value = reference.quotient(start_numerator, start_denominator)?;
    let end_value = if let Some(end) = after_hi {
        let end = BigRational::from_integer(end);
        let top = reference.evaluate(&certificate.numerator, &end)?;
        let bottom = reference.evaluate(&certificate.denominator, &end)?;
        reference.quotient(top, bottom)?
    } else {
        match certificate
            .numerator
            .len()
            .cmp(&certificate.denominator.len())
        {
            std::cmp::Ordering::Less => BigRational::zero(),
            std::cmp::Ordering::Equal => reference.quotient(
                certificate
                    .numerator
                    .last()
                    .expect("validated nonempty")
                    .clone(),
                certificate
                    .denominator
                    .last()
                    .expect("validated nonempty")
                    .clone(),
            )?,
            std::cmp::Ordering::Greater => {
                return Err(invalid("no finite rational limit at infinity"));
            }
        }
    };
    let value = reference.add(&end_value, &(-start_value))?;
    Ok(if negate { -value } else { value })
}
