//! Canonical arbitrary-precision rational arithmetic (WS03 / architecture doc §5.4).
//!
//! The provisional `num-rational` substrate is private to this crate. Higher layers use only
//! [`BigRational`] and [`fsym_bigint::BigInt`], keeping rational representation and normalization
//! policy independently replaceable from the integer substrate.
//!
//! This boundary owns the canonical rational value type, exact height metadata, finite simple
//! continued fractions, and cross-cancelled scalar arithmetic with cancellation-first metered
//! lanes. Pair-returning rational reconstruction support remains in `fsym-modular`; this crate
//! wraps only its canonical result in the owned rational value type through the explicitly
//! designated L1 reconstruction-support edge.

#![forbid(unsafe_code)]

use fsym_bigint::{
    BigInt, NonZeroBigInt, gcd, metered_add as metered_bigint_add, metered_div_rem_nonzero,
    metered_gcd, metered_multiply, metered_subtract as metered_bigint_subtract,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use num_traits::{Num, One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::str::FromStr;

/// Owned, canonical arbitrary-precision rational.
///
/// Values are reduced and their denominators are positive. The wrapped substrate is private so
/// persisted and semantic consumers cannot depend on its concrete representation.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct BigRational(num_rational::Ratio<BigInt>);

/// Exact coefficient-height metadata for a canonical rational.
///
/// Bit counts describe the absolute numerator and positive denominator. Zero therefore has a
/// numerator height of zero bits, while every denominator has at least one bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalHeight {
    numerator_bits: u64,
    denominator_bits: u64,
}

/// Typed failure from cancellation-first rational arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RationalArithmeticError {
    /// A budget was exhausted or the owning region was cancelled.
    Meter(MeterError),
    /// Division or remainder was requested with a zero right-hand operand.
    DivisionByZero,
    /// A mathematically exact internal quotient failed its invariant check.
    InvariantViolation(&'static str),
}

impl fmt::Display for RationalArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meter(error) => fmt::Display::fmt(error, f),
            Self::DivisionByZero => f.write_str("rational division by zero"),
            Self::InvariantViolation(message) => {
                write!(f, "rational arithmetic invariant violated: {message}")
            }
        }
    }
}

impl std::error::Error for RationalArithmeticError {}

impl From<MeterError> for RationalArithmeticError {
    fn from(error: MeterError) -> Self {
        Self::Meter(error)
    }
}

impl RationalHeight {
    /// Magnitude bits in the canonical numerator.
    pub fn numerator_bits(self) -> u64 {
        self.numerator_bits
    }

    /// Magnitude bits in the positive canonical denominator.
    pub fn denominator_bits(self) -> u64 {
        self.denominator_bits
    }

    /// The conventional rational height: the larger coefficient bit length.
    pub fn max_bits(self) -> u64 {
        self.numerator_bits.max(self.denominator_bits)
    }

    /// Total u64 limbs needed by the two canonical coefficients.
    pub fn total_limbs(self) -> u64 {
        self.numerator_bits
            .div_ceil(fsym_bigint::LIMB_BITS)
            .saturating_add(self.denominator_bits.div_ceil(fsym_bigint::LIMB_BITS))
    }
}

impl<'de> Deserialize<'de> for BigRational {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = num_rational::Ratio::<BigInt>::deserialize(deserializer)?;
        if !value.denom().is_positive() {
            return Err(serde::de::Error::custom(
                "rational denominator must be positive",
            ));
        }
        if value.numer().gcd(value.denom()) != BigInt::one() {
            return Err(serde::de::Error::custom(
                "rational numerator and denominator must be coprime",
            ));
        }
        Ok(Self(value))
    }
}

impl BigRational {
    /// Constructs a reduced rational with a positive denominator.
    ///
    /// # Panics
    ///
    /// Panics when `denom` is zero.
    pub fn new(numer: BigInt, denom: BigInt) -> Self {
        Self(num_rational::Ratio::new(numer, denom))
    }

    /// Constructs a rational whose denominator is one.
    pub fn from_integer(value: BigInt) -> Self {
        Self(num_rational::Ratio::from_integer(value))
    }

    /// Returns the canonical numerator.
    pub fn numer(&self) -> &BigInt {
        self.0.numer()
    }

    /// Returns the positive canonical denominator.
    pub fn denom(&self) -> &BigInt {
        self.0.denom()
    }

    /// Returns whether the denominator is one.
    pub fn is_integer(&self) -> bool {
        self.0.is_integer()
    }

    /// Truncates toward zero.
    pub fn to_integer(&self) -> BigInt {
        self.0.to_integer()
    }

    /// Returns the reciprocal.
    ///
    /// # Panics
    ///
    /// Panics when this rational is zero.
    pub fn recip(&self) -> Self {
        Self(self.0.recip())
    }

    /// Raises this rational to a signed integer power.
    pub fn pow(&self, exponent: i32) -> Self {
        Self(num_rational::Ratio::<BigInt>::pow(&self.0, exponent))
    }

    /// Adds two canonical rationals while cancelling denominator factors before multiplication.
    pub fn cross_cancelled_add(&self, rhs: &Self) -> Self {
        cross_cancelled_sum(self, rhs, false)
    }

    /// Subtracts two canonical rationals while cancelling denominator factors before
    /// multiplication.
    pub fn cross_cancelled_sub(&self, rhs: &Self) -> Self {
        cross_cancelled_sum(self, rhs, true)
    }

    /// Multiplies two canonical rationals after cancelling both numerator/denominator crosses.
    pub fn cross_cancelled_mul(&self, rhs: &Self) -> Self {
        let left_cross = gcd(self.numer(), rhs.denom());
        let right_cross = gcd(rhs.numer(), self.denom());
        let numerator = (self.numer() / &left_cross) * (rhs.numer() / &right_cross);
        let denominator = (self.denom() / &right_cross) * (rhs.denom() / &left_cross);
        Self(num_rational::Ratio::new_raw(numerator, denominator))
    }

    /// Divides two canonical rationals after cancelling numerator and denominator crosses.
    ///
    /// # Panics
    ///
    /// Panics when `rhs` is zero, matching the ordinary division-operator contract.
    pub fn cross_cancelled_div(&self, rhs: &Self) -> Self {
        if rhs.is_zero() {
            return Self(num_rational::Ratio::new(
                self.numer().clone(),
                BigInt::zero(),
            ));
        }
        let numerator_cross = gcd(self.numer(), rhs.numer());
        let denominator_cross = gcd(self.denom(), rhs.denom());
        let mut numerator = (self.numer() / &numerator_cross) * (rhs.denom() / &denominator_cross);
        let mut denominator =
            (self.denom() / &denominator_cross) * (rhs.numer() / &numerator_cross);
        if denominator.is_negative() {
            numerator = -numerator;
            denominator = -denominator;
        }
        Self(num_rational::Ratio::new_raw(numerator, denominator))
    }

    /// Computes the truncating rational remainder with an LCM-sized common denominator.
    ///
    /// # Panics
    ///
    /// Panics when `rhs` is zero, matching the ordinary remainder-operator contract.
    pub fn cross_cancelled_rem(&self, rhs: &Self) -> Self {
        if rhs.is_zero() {
            return Self(num_rational::Ratio::new(
                self.numer().clone(),
                BigInt::zero(),
            ));
        }
        let denominator_gcd = gcd(self.denom(), rhs.denom());
        let left_denominator = self.denom() / &denominator_gcd;
        let right_denominator = rhs.denom() / &denominator_gcd;
        let left_scaled = self.numer() * &right_denominator;
        let right_scaled = rhs.numer() * &left_denominator;
        let (_, remainder) = left_scaled.div_rem(&right_scaled);
        let common_denominator = self.denom() * right_denominator;
        let reduction = gcd(&remainder, &common_denominator);
        let numerator = remainder / &reduction;
        let denominator = common_denominator / reduction;
        Self(num_rational::Ratio::new_raw(numerator, denominator))
    }

    /// Reconstructs the unique bounded rational representative of `residue (mod modulus)`.
    ///
    /// This proves only the bounded congruence contract documented by
    /// [`fsym_modular::rational_reconstruct`]. It does not recognize or certify an external
    /// numerical approximation.
    pub fn reconstruct_modular(residue: &BigInt, modulus: &BigInt) -> Option<Self> {
        let (numerator, denominator) = fsym_modular::rational_reconstruct(residue, modulus)?;
        Some(Self(num_rational::Ratio::new_raw(numerator, denominator)))
    }

    /// Cancellation-first bounded modular rational reconstruction.
    ///
    /// Both successful and refused computed results observe a final checkpoint before
    /// publication. As in [`Self::reconstruct_modular`], the result establishes a bounded
    /// congruence representative, not recognition of an external numerical approximation.
    pub fn metered_reconstruct_modular<M: BudgetMeter>(
        residue: &BigInt,
        modulus: &BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        let reconstructed = fsym_modular::metered_rational_reconstruct(residue, modulus, meter)?
            .map(|(numerator, denominator)| {
                Self(num_rational::Ratio::new_raw(numerator, denominator))
            });
        metered_finish(reconstructed, meter)
    }

    /// Cancellation-first, cross-cancelled rational addition.
    pub fn metered_add<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Self, RationalArithmeticError> {
        metered_sum(self, rhs, false, meter)
    }

    /// Cancellation-first, cross-cancelled rational subtraction.
    pub fn metered_sub<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Self, RationalArithmeticError> {
        metered_sum(self, rhs, true, meter)
    }

    /// Cancellation-first rational multiplication with both crosses reduced first.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Self, RationalArithmeticError> {
        meter.checkpoint()?;
        let left_cross = metered_gcd(self.numer(), rhs.denom(), meter)?;
        let right_cross = metered_gcd(rhs.numer(), self.denom(), meter)?;
        let left_numerator = metered_exact_quotient(
            self.numer(),
            &left_cross,
            "left numerator cross-cancellation",
            meter,
        )?;
        let right_numerator = metered_exact_quotient(
            rhs.numer(),
            &right_cross,
            "right numerator cross-cancellation",
            meter,
        )?;
        let left_denominator = metered_exact_quotient(
            self.denom(),
            &right_cross,
            "left denominator cross-cancellation",
            meter,
        )?;
        let right_denominator = metered_exact_quotient(
            rhs.denom(),
            &left_cross,
            "right denominator cross-cancellation",
            meter,
        )?;
        let numerator = metered_multiply(&left_numerator, &right_numerator, meter)?;
        let denominator = metered_multiply(&left_denominator, &right_denominator, meter)?;
        rational_metered_finish(
            Self(num_rational::Ratio::new_raw(numerator, denominator)),
            meter,
        )
    }

    /// Cancellation-first rational division with typed zero-divisor refusal.
    pub fn metered_div<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Self, RationalArithmeticError> {
        meter.checkpoint()?;
        if rhs.is_zero() {
            return rational_metered_error(RationalArithmeticError::DivisionByZero, meter);
        }
        let numerator_cross = metered_gcd(self.numer(), rhs.numer(), meter)?;
        let denominator_cross = metered_gcd(self.denom(), rhs.denom(), meter)?;
        let left_numerator = metered_exact_quotient(
            self.numer(),
            &numerator_cross,
            "division numerator cross-cancellation",
            meter,
        )?;
        let right_numerator = metered_exact_quotient(
            rhs.numer(),
            &numerator_cross,
            "division divisor cross-cancellation",
            meter,
        )?;
        let left_denominator = metered_exact_quotient(
            self.denom(),
            &denominator_cross,
            "division left-denominator cross-cancellation",
            meter,
        )?;
        let right_denominator = metered_exact_quotient(
            rhs.denom(),
            &denominator_cross,
            "division right-denominator cross-cancellation",
            meter,
        )?;
        let mut numerator = metered_multiply(&left_numerator, &right_denominator, meter)?;
        let mut denominator = metered_multiply(&left_denominator, &right_numerator, meter)?;
        if denominator.is_negative() {
            numerator = metered_negate(numerator, meter)?;
            denominator = metered_negate(denominator, meter)?;
        }
        rational_metered_finish(
            Self(num_rational::Ratio::new_raw(numerator, denominator)),
            meter,
        )
    }

    /// Cancellation-first truncating rational remainder with typed zero-divisor refusal.
    pub fn metered_rem<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Self, RationalArithmeticError> {
        meter.checkpoint()?;
        if rhs.is_zero() {
            return rational_metered_error(RationalArithmeticError::DivisionByZero, meter);
        }
        let denominator_gcd = metered_gcd(self.denom(), rhs.denom(), meter)?;
        let left_denominator = metered_exact_quotient(
            self.denom(),
            &denominator_gcd,
            "remainder left-denominator cancellation",
            meter,
        )?;
        let right_denominator = metered_exact_quotient(
            rhs.denom(),
            &denominator_gcd,
            "remainder right-denominator cancellation",
            meter,
        )?;
        let left_scaled = metered_multiply(self.numer(), &right_denominator, meter)?;
        let right_scaled = metered_multiply(rhs.numer(), &left_denominator, meter)?;
        let Some(right_scaled) = NonZeroBigInt::new(&right_scaled) else {
            return Err(RationalArithmeticError::InvariantViolation(
                "nonzero remainder divisor became zero",
            ));
        };
        let (_, remainder) = metered_div_rem_nonzero(&left_scaled, right_scaled, meter)?;
        let common_denominator = metered_multiply(self.denom(), &right_denominator, meter)?;
        let reduction = metered_gcd(&remainder, &common_denominator, meter)?;
        let numerator = metered_exact_quotient(
            &remainder,
            &reduction,
            "remainder numerator normalization",
            meter,
        )?;
        let denominator = metered_exact_quotient(
            &common_denominator,
            &reduction,
            "remainder denominator normalization",
            meter,
        )?;
        rational_metered_finish(
            Self(num_rational::Ratio::new_raw(numerator, denominator)),
            meter,
        )
    }

    /// Returns exact numerator/denominator height metadata.
    pub fn height(&self) -> RationalHeight {
        RationalHeight {
            numerator_bits: self.numer().bits(),
            denominator_bits: self.denom().bits(),
        }
    }

    /// Expands this value into its finite simple continued fraction.
    ///
    /// Division uses mathematical floor semantics, so negative values receive the standard
    /// representation with a possibly negative first coefficient and positive remainders.
    pub fn continued_fraction(&self) -> Vec<BigInt> {
        let mut numerator = self.numer().clone();
        let mut denominator = self.denom().clone();
        let mut coefficients = Vec::new();
        loop {
            let (mut quotient, mut remainder) = numerator.div_rem(&denominator);
            if remainder.is_negative() {
                quotient -= 1i64;
                remainder += &denominator;
            }
            coefficients.push(quotient);
            if remainder.is_zero() {
                return coefficients;
            }
            numerator = denominator;
            denominator = remainder;
        }
    }

    /// Cancellation-first continued-fraction expansion with coefficient-height accounting.
    pub fn metered_continued_fraction<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<Vec<BigInt>, MeterError> {
        meter.checkpoint()?;
        let mut numerator = metered_clone(self.numer(), meter)?;
        let mut denominator = metered_clone(self.denom(), meter)?;
        let coefficient_capacity = continued_fraction_coefficient_bound(self.denom().bits());
        let coefficient_bytes = u64::try_from(coefficient_capacity)
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::try_from(std::mem::size_of::<BigInt>()).unwrap_or(u64::MAX));
        meter.charge_batch(&[
            (Dimension::MemoryBytes, coefficient_bytes),
            (Dimension::AllocationCount, 1),
        ])?;
        let mut coefficients = Vec::with_capacity(coefficient_capacity);
        loop {
            meter.checkpoint()?;
            let Some(divisor) = NonZeroBigInt::new(&denominator) else {
                return metered_finish(coefficients, meter);
            };
            let (mut quotient, mut remainder) =
                metered_div_rem_nonzero(&numerator, divisor, meter)?;
            if remainder.is_negative() {
                let one = metered_one(meter)?;
                quotient = metered_bigint_subtract(&quotient, &one, meter)?;
                remainder = metered_bigint_add(&remainder, &denominator, meter)?;
            }
            charge_persisted_coefficient(&quotient, meter)?;
            debug_assert!(coefficients.len() < coefficient_capacity);
            coefficients.push(quotient);
            if remainder.is_zero() {
                return metered_finish(coefficients, meter);
            }
            numerator = denominator;
            denominator = remainder;
        }
    }

    /// Reconstructs a rational from a nonempty finite integer continued fraction.
    ///
    /// Generated simple continued fractions are accepted, as are generalized integer coefficient
    /// lists whose nested divisions are defined. Returns `None` when a zero suffix would require
    /// division by zero.
    pub fn from_continued_fraction(coefficients: &[BigInt]) -> Option<Self> {
        let (last, prefix) = coefficients.split_last()?;
        let mut numerator = last.clone();
        let mut denominator = BigInt::one();
        for coefficient in prefix.iter().rev() {
            if numerator.is_zero() {
                return None;
            }
            let next_numerator = coefficient * &numerator + &denominator;
            denominator = numerator;
            numerator = next_numerator;
        }
        if denominator.is_negative() {
            numerator = -numerator;
            denominator = -denominator;
        }
        Some(Self(num_rational::Ratio::new_raw(numerator, denominator)))
    }

    /// Cancellation-first reconstruction from a finite integer continued fraction.
    pub fn metered_from_continued_fraction<M: BudgetMeter>(
        coefficients: &[BigInt],
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        let Some((last, prefix)) = coefficients.split_last() else {
            return metered_finish(None, meter);
        };
        let mut numerator = metered_clone(last, meter)?;
        let mut denominator = metered_one(meter)?;
        for coefficient in prefix.iter().rev() {
            meter.checkpoint()?;
            if numerator.is_zero() {
                return metered_finish(None, meter);
            }
            let product = metered_multiply(coefficient, &numerator, meter)?;
            let next_numerator = metered_bigint_add(&product, &denominator, meter)?;
            denominator = numerator;
            numerator = next_numerator;
        }
        if denominator.is_negative() {
            numerator = metered_negate(numerator, meter)?;
            denominator = metered_negate(denominator, meter)?;
        }
        let value = Self(num_rational::Ratio::new_raw(numerator, denominator));
        metered_finish(Some(value), meter)
    }
}

fn cross_cancelled_sum(lhs: &BigRational, rhs: &BigRational, subtract: bool) -> BigRational {
    let denominator_gcd = gcd(lhs.denom(), rhs.denom());
    let left_denominator = lhs.denom() / &denominator_gcd;
    let right_denominator = rhs.denom() / &denominator_gcd;
    let left_scaled = lhs.numer() * &right_denominator;
    let right_scaled = rhs.numer() * &left_denominator;
    let combined = if subtract {
        left_scaled - right_scaled
    } else {
        left_scaled + right_scaled
    };
    let reduction = gcd(&combined, &denominator_gcd);
    let numerator = combined / &reduction;
    let denominator = left_denominator * (rhs.denom() / reduction);
    BigRational(num_rational::Ratio::new_raw(numerator, denominator))
}

fn metered_sum<M: BudgetMeter>(
    lhs: &BigRational,
    rhs: &BigRational,
    subtract: bool,
    meter: &mut M,
) -> Result<BigRational, RationalArithmeticError> {
    meter.checkpoint()?;
    let denominator_gcd = metered_gcd(lhs.denom(), rhs.denom(), meter)?;
    let left_denominator = metered_exact_quotient(
        lhs.denom(),
        &denominator_gcd,
        "sum left-denominator cancellation",
        meter,
    )?;
    let right_denominator = metered_exact_quotient(
        rhs.denom(),
        &denominator_gcd,
        "sum right-denominator cancellation",
        meter,
    )?;
    let left_scaled = metered_multiply(lhs.numer(), &right_denominator, meter)?;
    let right_scaled = metered_multiply(rhs.numer(), &left_denominator, meter)?;
    let combined = if subtract {
        metered_bigint_subtract(&left_scaled, &right_scaled, meter)?
    } else {
        metered_bigint_add(&left_scaled, &right_scaled, meter)?
    };
    let reduction = metered_gcd(&combined, &denominator_gcd, meter)?;
    let numerator =
        metered_exact_quotient(&combined, &reduction, "sum numerator normalization", meter)?;
    let right_reduced = metered_exact_quotient(
        rhs.denom(),
        &reduction,
        "sum denominator normalization",
        meter,
    )?;
    let denominator = metered_multiply(&left_denominator, &right_reduced, meter)?;
    rational_metered_finish(
        BigRational(num_rational::Ratio::new_raw(numerator, denominator)),
        meter,
    )
}

fn metered_exact_quotient<M: BudgetMeter>(
    value: &BigInt,
    divisor: &BigInt,
    invariant: &'static str,
    meter: &mut M,
) -> Result<BigInt, RationalArithmeticError> {
    let Some(divisor) = NonZeroBigInt::new(divisor) else {
        return Err(RationalArithmeticError::InvariantViolation(invariant));
    };
    let (quotient, remainder) = metered_div_rem_nonzero(value, divisor, meter)?;
    if !remainder.is_zero() {
        return Err(RationalArithmeticError::InvariantViolation(invariant));
    }
    Ok(quotient)
}

/// Publishes a fully classified rational value only after a terminal checkpoint.
fn rational_metered_finish<T, M: BudgetMeter>(
    value: T,
    meter: &mut M,
) -> Result<T, RationalArithmeticError> {
    meter.checkpoint()?;
    Ok(value)
}

/// Publishes a fully classified rational refusal only after a terminal checkpoint.
fn rational_metered_error<T, M: BudgetMeter>(
    error: RationalArithmeticError,
    meter: &mut M,
) -> Result<T, RationalArithmeticError> {
    meter.checkpoint()?;
    Err(error)
}

/// Publishes a fully classified value only after a terminal checkpoint.
fn metered_finish<T, M: BudgetMeter>(value: T, meter: &mut M) -> Result<T, MeterError> {
    meter.checkpoint()?;
    Ok(value)
}

/// Every two Euclidean divisions at least halve the positive divisor: if the first remainder is
/// above half, the following remainder is below half. Two slots per denominator bit plus the
/// initial quotient therefore bound every canonical rational's finite expansion.
fn continued_fraction_coefficient_bound(denominator_bits: u64) -> usize {
    usize::try_from(denominator_bits.saturating_mul(2).saturating_add(1)).unwrap_or(usize::MAX)
}

fn metered_clone<M: BudgetMeter>(value: &BigInt, meter: &mut M) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            value.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let cloned = value.clone();
    meter.checkpoint()?;
    Ok(cloned)
}

fn metered_one<M: BudgetMeter>(meter: &mut M) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge_batch(&[(Dimension::MemoryBytes, 8), (Dimension::AllocationCount, 1)])?;
    let one = BigInt::one();
    meter.checkpoint()?;
    Ok(one)
}

fn charge_persisted_coefficient<M: BudgetMeter>(
    value: &BigInt,
    meter: &mut M,
) -> Result<(), MeterError> {
    meter.checkpoint()?;
    meter.charge(
        Dimension::MemoryBytes,
        value.limb_count().max(1).saturating_mul(8),
    )?;
    meter.checkpoint()
}

fn metered_negate<M: BudgetMeter>(value: BigInt, meter: &mut M) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge(Dimension::ComputeSteps, value.limb_count().max(1))?;
    let result = -value;
    meter.checkpoint()?;
    Ok(result)
}

impl fmt::Debug for BigRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for BigRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl From<BigInt> for BigRational {
    fn from(value: BigInt) -> Self {
        Self::from_integer(value)
    }
}

impl From<i64> for BigRational {
    fn from(value: i64) -> Self {
        Self::from_integer(BigInt::from(value))
    }
}

impl Zero for BigRational {
    fn zero() -> Self {
        Self(num_rational::Ratio::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for BigRational {
    fn one() -> Self {
        Self(num_rational::Ratio::one())
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

macro_rules! impl_rational_binary_op {
    ($trait:ident, $method:ident, $owned_method:ident) => {
        impl $trait for BigRational {
            type Output = BigRational;

            fn $method(self, rhs: BigRational) -> Self::Output {
                self.$owned_method(&rhs)
            }
        }

        impl $trait<&BigRational> for BigRational {
            type Output = BigRational;

            fn $method(self, rhs: &BigRational) -> Self::Output {
                self.$owned_method(rhs)
            }
        }

        impl $trait<BigRational> for &BigRational {
            type Output = BigRational;

            fn $method(self, rhs: BigRational) -> Self::Output {
                self.$owned_method(&rhs)
            }
        }

        impl $trait<&BigRational> for &BigRational {
            type Output = BigRational;

            fn $method(self, rhs: &BigRational) -> Self::Output {
                self.$owned_method(rhs)
            }
        }
    };
}

impl_rational_binary_op!(Add, add, cross_cancelled_add);
impl_rational_binary_op!(Sub, sub, cross_cancelled_sub);
impl_rational_binary_op!(Mul, mul, cross_cancelled_mul);
impl_rational_binary_op!(Div, div, cross_cancelled_div);
impl_rational_binary_op!(Rem, rem, cross_cancelled_rem);

macro_rules! impl_rational_assign_op {
    ($trait:ident, $method:ident, $owned_method:ident) => {
        impl $trait for BigRational {
            fn $method(&mut self, rhs: BigRational) {
                *self = self.$owned_method(&rhs);
            }
        }

        impl $trait<&BigRational> for BigRational {
            fn $method(&mut self, rhs: &BigRational) {
                *self = self.$owned_method(rhs);
            }
        }
    };
}

impl_rational_assign_op!(AddAssign, add_assign, cross_cancelled_add);
impl_rational_assign_op!(SubAssign, sub_assign, cross_cancelled_sub);
impl_rational_assign_op!(MulAssign, mul_assign, cross_cancelled_mul);
impl_rational_assign_op!(DivAssign, div_assign, cross_cancelled_div);
impl_rational_assign_op!(RemAssign, rem_assign, cross_cancelled_rem);

impl Neg for BigRational {
    type Output = BigRational;

    fn neg(self) -> Self::Output {
        BigRational(-self.0)
    }
}

impl Neg for &BigRational {
    type Output = BigRational;

    fn neg(self) -> Self::Output {
        BigRational(-&self.0)
    }
}

impl Num for BigRational {
    type FromStrRadixErr = String;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        num_rational::Ratio::from_str_radix(src, radix)
            .map(Self)
            .map_err(|error| error.to_string())
    }
}

impl FromStr for BigRational {
    type Err = String;

    fn from_str(src: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(src, 10)
    }
}

impl Signed for BigRational {
    fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    fn abs_sub(&self, other: &Self) -> Self {
        Self(self.0.abs_sub(&other.0))
    }

    fn signum(&self) -> Self {
        Self(self.0.signum())
    }

    fn is_positive(&self) -> bool {
        self.0.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.0.is_negative()
    }
}

impl ToPrimitive for BigRational {
    fn to_i64(&self) -> Option<i64> {
        self.to_integer().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.to_integer().to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.numer().to_f64()? / self.denom().to_f64()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetError, BudgetLimits, Unbounded};
    use proptest::prelude::*;

    #[derive(Debug, Default)]
    struct CheckpointMeter {
        checkpoints: usize,
        cancel_at: Option<usize>,
        charged: bool,
    }

    impl CheckpointMeter {
        fn cancelling_at(checkpoint: usize) -> Self {
            Self {
                checkpoints: 0,
                cancel_at: Some(checkpoint),
                charged: false,
            }
        }
    }

    impl BudgetMeter for CheckpointMeter {
        fn charge(&mut self, _dimension: Dimension, amount: u64) -> Result<(), MeterError> {
            self.charged |= amount != 0;
            Ok(())
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            self.charged |= charges.iter().any(|(_, amount)| *amount != 0);
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints = self.checkpoints.saturating_add(1);
            if self.cancel_at == Some(self.checkpoints) {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn owned_rational_is_canonical_and_supports_arithmetic() {
        let value = BigRational::new(BigInt::from(-6), BigInt::from(-8));
        assert_eq!(value.numer(), &BigInt::from(3));
        assert_eq!(value.denom(), &BigInt::from(4));
        assert_eq!(
            &value + &BigRational::new(BigInt::from(1), BigInt::from(4)),
            BigRational::one()
        );
        assert_eq!(value.recip(), BigRational::new(4.into(), 3.into()));
        assert_eq!(value.pow(-2), BigRational::new(16.into(), 9.into()));
    }

    #[test]
    fn owned_operator_and_assignment_paths_use_canonical_cross_cancelled_results() {
        let lhs = BigRational::new(BigInt::from(-14), BigInt::from(15));
        let rhs = BigRational::new(BigInt::from(21), BigInt::from(22));

        assert_eq!(
            lhs.cross_cancelled_add(&rhs),
            BigRational::new(BigInt::from(7), BigInt::from(330))
        );
        assert_eq!(
            lhs.cross_cancelled_sub(&rhs),
            BigRational::new(BigInt::from(-623), BigInt::from(330))
        );
        assert_eq!(
            lhs.cross_cancelled_mul(&rhs),
            BigRational::new(BigInt::from(-49), BigInt::from(55))
        );
        assert_eq!(
            lhs.cross_cancelled_div(&rhs),
            BigRational::new(BigInt::from(-44), BigInt::from(45))
        );
        assert_eq!(
            lhs.cross_cancelled_rem(&rhs),
            BigRational::new(BigInt::from(-14), BigInt::from(15))
        );

        let shared_denominator = BigInt::from(15);
        let reduction_lhs = BigRational::new(BigInt::one(), shared_denominator.clone());
        let add_rhs = BigRational::new(BigInt::from(13), &shared_denominator * 2i64);
        let sub_rhs = BigRational::new(BigInt::from(-13), &shared_denominator * 2i64);
        assert_eq!(
            reduction_lhs.cross_cancelled_add(&add_rhs),
            BigRational::new(BigInt::one(), BigInt::from(2))
        );
        assert_eq!(
            reduction_lhs.cross_cancelled_sub(&sub_rhs),
            BigRational::new(BigInt::one(), BigInt::from(2))
        );
        let negative_divisor = BigRational::new(BigInt::from(-4), BigInt::from(5));
        let signed_quotient = BigRational::new(BigInt::from(2), BigInt::from(3))
            .cross_cancelled_div(&negative_divisor);
        assert_eq!(
            signed_quotient,
            BigRational::new(BigInt::from(-5), BigInt::from(6))
        );
        assert!(signed_quotient.denom().is_positive());

        assert_eq!(&lhs + &rhs, lhs.cross_cancelled_add(&rhs));
        assert_eq!(&lhs - &rhs, lhs.cross_cancelled_sub(&rhs));
        assert_eq!(&lhs * &rhs, lhs.cross_cancelled_mul(&rhs));
        assert_eq!(&lhs / &rhs, lhs.cross_cancelled_div(&rhs));
        assert_eq!(&lhs % &rhs, lhs.cross_cancelled_rem(&rhs));

        let mut assigned = lhs.clone();
        assigned += &rhs;
        assert_eq!(assigned, lhs.cross_cancelled_add(&rhs));
        assigned = lhs.clone();
        assigned -= rhs.clone();
        assert_eq!(assigned, lhs.cross_cancelled_sub(&rhs));
        assigned = lhs.clone();
        assigned *= &rhs;
        assert_eq!(assigned, lhs.cross_cancelled_mul(&rhs));
        assigned = lhs.clone();
        assigned /= rhs.clone();
        assert_eq!(assigned, lhs.cross_cancelled_div(&rhs));
        assigned = lhs.clone();
        assigned %= &rhs;
        assert_eq!(assigned, lhs.cross_cancelled_rem(&rhs));
    }

    #[test]
    fn metered_scalar_lanes_match_owned_operators_and_refuse_zero_divisors() {
        let lhs = BigRational::new(BigInt::from(-14), BigInt::from(15));
        let rhs = BigRational::new(BigInt::from(21), BigInt::from(22));
        let mut meter = Unbounded;
        assert_eq!(lhs.metered_add(&rhs, &mut meter), Ok(&lhs + &rhs));
        let mut meter = Unbounded;
        assert_eq!(lhs.metered_sub(&rhs, &mut meter), Ok(&lhs - &rhs));
        let mut meter = Unbounded;
        assert_eq!(lhs.metered_mul(&rhs, &mut meter), Ok(&lhs * &rhs));
        let mut meter = Unbounded;
        assert_eq!(lhs.metered_div(&rhs, &mut meter), Ok(&lhs / &rhs));
        let mut meter = Unbounded;
        assert_eq!(lhs.metered_rem(&rhs, &mut meter), Ok(&lhs % &rhs));

        let zero = BigRational::zero();
        let mut meter = Unbounded;
        assert_eq!(
            lhs.metered_div(&zero, &mut meter),
            Err(RationalArithmeticError::DivisionByZero)
        );
        let mut meter = Unbounded;
        assert_eq!(
            lhs.metered_rem(&zero, &mut meter),
            Err(RationalArithmeticError::DivisionByZero)
        );

        let mut cancelled = CheckpointMeter::cancelling_at(1);
        assert_eq!(
            lhs.metered_div(&zero, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
        let mut cancelled = CheckpointMeter::cancelling_at(1);
        assert_eq!(
            lhs.metered_rem(&zero, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
    }

    #[test]
    fn metered_scalar_lanes_refuse_budget_and_terminal_cancellation() {
        let common = BigInt::from(2).pow(256) - 1i64;
        let lhs = BigRational::new(common.clone(), BigInt::from(3).pow(161));
        let rhs = BigRational::new(BigInt::from(3).pow(161), common);

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            lhs.metered_mul(&rhs, &mut budget),
            Err(RationalArithmeticError::Meter(MeterError::Budget(_)))
        ));

        let mut measured = CheckpointMeter::default();
        assert_eq!(lhs.metered_add(&rhs, &mut measured).unwrap(), &lhs + &rhs);
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            lhs.metered_add(&rhs, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );

        let mut measured = CheckpointMeter::default();
        assert_eq!(
            lhs.metered_mul(&rhs, &mut measured).unwrap(),
            BigRational::one()
        );
        assert!(measured.checkpoints > 1_000);
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints * 3 / 4);
        assert_eq!(
            lhs.metered_mul(&rhs, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );

        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            lhs.metered_mul(&rhs, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
    }

    #[test]
    fn classified_fast_terminals_checkpoint_without_charging_work() {
        let value = BigRational::new(BigInt::from(7), BigInt::from(11));
        let zero = BigRational::zero();

        for operation in [
            BigRational::metered_div::<CheckpointMeter>,
            BigRational::metered_rem::<CheckpointMeter>,
        ] {
            let mut measured = CheckpointMeter::default();
            assert_eq!(
                operation(&value, &zero, &mut measured),
                Err(RationalArithmeticError::DivisionByZero)
            );
            assert_eq!(measured.checkpoints, 2);
            assert!(!measured.charged);

            let mut cancelled = CheckpointMeter::cancelling_at(2);
            assert_eq!(
                operation(&value, &zero, &mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled))
            );
            assert_eq!(cancelled.checkpoints, 2);
            assert!(!cancelled.charged);
        }

        let mut measured = CheckpointMeter::default();
        assert_eq!(
            BigRational::metered_from_continued_fraction(&[], &mut measured),
            Ok(None)
        );
        assert_eq!(measured.checkpoints, 2);
        assert!(!measured.charged);

        let mut cancelled = CheckpointMeter::cancelling_at(2);
        assert_eq!(
            BigRational::metered_from_continued_fraction(&[], &mut cancelled),
            Err(MeterError::Cancelled)
        );
        assert_eq!(cancelled.checkpoints, 2);
        assert!(!cancelled.charged);
    }

    #[test]
    fn deserialization_rejects_noncanonical_raw_ratios() {
        for raw in [
            (BigInt::from(2), BigInt::from(4)),
            (BigInt::from(1), BigInt::from(-2)),
            (BigInt::zero(), BigInt::from(2)),
            (BigInt::one(), BigInt::zero()),
        ] {
            let encoded = serde_json::to_vec(&raw).expect("raw pair serializes");
            assert!(serde_json::from_slice::<BigRational>(&encoded).is_err());
        }

        let canonical = (BigInt::from(-1), BigInt::from(2));
        let encoded = serde_json::to_vec(&canonical).expect("canonical pair serializes");
        assert_eq!(
            serde_json::from_slice::<BigRational>(&encoded).expect("canonical pair deserializes"),
            BigRational::new(canonical.0, canonical.1)
        );
    }

    #[test]
    fn serialization_preserves_the_canonical_pair_wire_shape() {
        let value = BigRational::new(BigInt::from(-6), BigInt::from(8));
        assert_eq!(
            serde_json::to_vec(&value).expect("rational serializes"),
            serde_json::to_vec(&(value.numer(), value.denom())).expect("pair serializes")
        );
    }

    #[test]
    fn height_reports_exact_canonical_coefficient_bits() {
        let height = BigRational::new(BigInt::from(-3), BigInt::from(4)).height();
        assert_eq!(height.numerator_bits(), 2);
        assert_eq!(height.denominator_bits(), 3);
        assert_eq!(height.max_bits(), 3);
        assert_eq!(height.total_limbs(), 2);

        let zero_height = BigRational::zero().height();
        assert_eq!(zero_height.numerator_bits(), 0);
        assert_eq!(zero_height.denominator_bits(), 1);
    }

    #[test]
    fn modular_reconstruction_wraps_only_canonical_bounded_results() {
        let modulus = BigInt::from(101);
        assert_eq!(
            BigRational::reconstruct_modular(&BigInt::from(26), &modulus),
            Some(BigRational::new(BigInt::from(3), BigInt::from(4)))
        );
        assert_eq!(
            BigRational::reconstruct_modular(&BigInt::zero(), &modulus),
            Some(BigRational::zero())
        );
        assert_eq!(
            BigRational::reconstruct_modular(&BigInt::from(3), &BigInt::from(7)),
            None
        );
        assert_eq!(
            BigRational::reconstruct_modular(&BigInt::one(), &BigInt::one()),
            None
        );

        let mut meter = Unbounded;
        assert_eq!(
            BigRational::reconstruct_modular(&BigInt::from(2), &BigInt::from(2)),
            None
        );
        assert_eq!(
            BigRational::metered_reconstruct_modular(
                &BigInt::from(2),
                &BigInt::from(2),
                &mut meter,
            )
            .unwrap(),
            None
        );

        let expected_zero = Some(BigRational::zero());
        let mut meter = Unbounded;
        assert_eq!(
            BigRational::reconstruct_modular(&BigInt::from(3), &BigInt::from(3)),
            expected_zero
        );
        assert_eq!(
            BigRational::metered_reconstruct_modular(
                &BigInt::from(3),
                &BigInt::from(3),
                &mut meter,
            )
            .unwrap(),
            expected_zero
        );
    }

    #[test]
    fn metered_modular_reconstruction_checks_before_every_publication_class() {
        for (residue, modulus) in [
            (BigInt::from(26), BigInt::from(101)),
            (BigInt::from(3), BigInt::from(7)),
            (BigInt::one(), BigInt::one()),
        ] {
            let mut measured = CheckpointMeter::default();
            let expected =
                BigRational::metered_reconstruct_modular(&residue, &modulus, &mut measured)
                    .expect("unbounded checkpoint meter admits reconstruction");
            assert_eq!(
                expected,
                BigRational::reconstruct_modular(&residue, &modulus)
            );
            let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
            assert_eq!(
                BigRational::metered_reconstruct_modular(&residue, &modulus, &mut cancelled,),
                Err(MeterError::Cancelled)
            );
        }

        let modulus = (BigInt::one() << 127) - 1i64;
        let denominator = BigInt::from(37);
        let inverse = fsym_modular::mod_inverse(&denominator, &modulus)
            .expect("denominator is invertible modulo the Mersenne prime");
        let residue = (&BigInt::from(-23) * inverse) % &modulus;
        let mut measured = CheckpointMeter::default();
        assert!(
            BigRational::metered_reconstruct_modular(&residue, &modulus, &mut measured)
                .expect("unbounded checkpoint meter admits reconstruction")
                .is_some()
        );
        assert!(measured.checkpoints > 100);
        let mut cancelled =
            CheckpointMeter::cancelling_at(measured.checkpoints.saturating_mul(3) / 4);
        assert_eq!(
            BigRational::metered_reconstruct_modular(&residue, &modulus, &mut cancelled),
            Err(MeterError::Cancelled)
        );
    }

    #[test]
    fn continued_fractions_use_floor_semantics_and_round_trip() {
        let positive = BigRational::new(BigInt::from(415), BigInt::from(93));
        let positive_coefficients = positive.continued_fraction();
        assert_eq!(positive_coefficients, [4, 2, 6, 7].map(BigInt::from));
        assert!(
            positive_coefficients
                .iter()
                .skip(1)
                .all(BigInt::is_positive)
        );
        assert!(positive_coefficients.last().is_some_and(|last| last > &1));

        let negative = BigRational::new(BigInt::from(-415), BigInt::from(93));
        assert_eq!(
            negative.continued_fraction(),
            [-5, 1, 1, 6, 7].map(BigInt::from)
        );
        assert_eq!(
            BigRational::from_continued_fraction(&negative.continued_fraction()),
            Some(negative)
        );
        assert_eq!(BigRational::from_continued_fraction(&[]), None);
        assert_eq!(
            BigRational::from_continued_fraction(&[BigInt::one(), BigInt::zero()]),
            None
        );
    }

    #[test]
    fn metered_continued_fractions_match_and_obey_budget() {
        let value = BigRational::new(BigInt::from(-415), BigInt::from(93));
        let expected = value.continued_fraction();

        let mut meter = Unbounded;
        let actual = value.metered_continued_fraction(&mut meter).unwrap();
        assert_eq!(actual, expected);

        let mut meter = Unbounded;
        assert_eq!(
            BigRational::metered_from_continued_fraction(&actual, &mut meter).unwrap(),
            Some(value)
        );

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            BigRational::new(BigInt::from(415), BigInt::from(93))
                .metered_continued_fraction(&mut budget),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            BigRational::metered_from_continued_fraction(&actual, &mut budget),
            Err(MeterError::Budget(_))
        ));
    }

    #[test]
    fn metered_expansion_preflights_the_bounded_coefficient_buffer() {
        let value = BigRational::new(BigInt::from(415), BigInt::from(93));
        let clone_bytes = value
            .numer()
            .limb_count()
            .max(1)
            .saturating_add(value.denom().limb_count().max(1))
            .saturating_mul(8);
        let coefficient_bytes =
            u64::try_from(continued_fraction_coefficient_bound(value.denom().bits()))
                .unwrap_or(u64::MAX)
                .saturating_mul(u64::try_from(std::mem::size_of::<BigInt>()).unwrap_or(u64::MAX));
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::MemoryBytes.index()] = clone_bytes
            .saturating_add(coefficient_bytes)
            .saturating_sub(1);
        let mut budget = Budget::new(limits);

        assert_eq!(
            value.metered_continued_fraction(&mut budget),
            Err(MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::MemoryBytes,
                requested: coefficient_bytes,
                remaining: coefficient_bytes.saturating_sub(1),
            }))
        );
    }

    #[test]
    fn continued_fraction_lanes_support_late_cancellation() {
        let (mut previous, mut current) = (BigInt::zero(), BigInt::one());
        for _ in 0..180 {
            let next = &previous + &current;
            previous = current;
            current = next;
        }
        let value = BigRational::new(current, previous);

        let mut measured = CheckpointMeter::default();
        let coefficients = value.metered_continued_fraction(&mut measured).unwrap();
        assert!(coefficients.len() > 100);
        assert!(measured.checkpoints > 1_000);

        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            value.metered_continued_fraction(&mut cancelled),
            Err(MeterError::Cancelled)
        );

        let late_checkpoint = measured.checkpoints.saturating_mul(3) / 4;
        let mut cancelled = CheckpointMeter::cancelling_at(late_checkpoint);
        assert_eq!(
            value.metered_continued_fraction(&mut cancelled),
            Err(MeterError::Cancelled)
        );

        let mut measured = CheckpointMeter::default();
        assert_eq!(
            BigRational::metered_from_continued_fraction(&coefficients, &mut measured).unwrap(),
            Some(value)
        );
        let final_checkpoint = measured.checkpoints;
        let mut cancelled = CheckpointMeter::cancelling_at(final_checkpoint);
        assert_eq!(
            BigRational::metered_from_continued_fraction(&coefficients, &mut cancelled),
            Err(MeterError::Cancelled)
        );
        let late_checkpoint = measured.checkpoints.saturating_mul(3) / 4;
        let mut cancelled = CheckpointMeter::cancelling_at(late_checkpoint);
        assert_eq!(
            BigRational::metered_from_continued_fraction(&coefficients, &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let undefined = [BigInt::one(), BigInt::zero()];
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            BigRational::metered_from_continued_fraction(&undefined, &mut measured).unwrap(),
            None
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            BigRational::metered_from_continued_fraction(&undefined, &mut cancelled),
            Err(MeterError::Cancelled)
        );
    }

    proptest! {
        #[test]
        fn modular_reconstruction_round_trips_unique_small_rationals(
            numerator in -50i64..51,
            denominator in 1u64..51,
        ) {
            fn scalar_gcd(mut lhs: u64, mut rhs: u64) -> u64 {
                while rhs != 0 {
                    let remainder = lhs % rhs;
                    lhs = rhs;
                    rhs = remainder;
                }
                lhs
            }

            prop_assume!(scalar_gcd(numerator.unsigned_abs(), denominator) == 1);
            let modulus = BigInt::from(1_000_003);
            let denominator = BigInt::from(denominator);
            let inverse = fsym_modular::mod_inverse(&denominator, &modulus).unwrap();
            let product = BigInt::from(numerator) * inverse;
            let residue = ((&product % &modulus) + &modulus) % &modulus;
            let expected = BigRational::new(BigInt::from(numerator), denominator);
            prop_assert_eq!(
                BigRational::reconstruct_modular(&residue, &modulus),
                Some(expected.clone())
            );
            let mut meter = Unbounded;
            prop_assert_eq!(
                BigRational::metered_reconstruct_modular(&residue, &modulus, &mut meter).unwrap(),
                Some(expected)
            );
        }

        #[test]
        fn normalization_is_value_preserving_and_coprime(
            numerator in any::<i64>(),
            denominator in any::<i64>().prop_filter("nonzero denominator", |value| *value != 0),
        ) {
            let original_numerator = BigInt::from(numerator);
            let original_denominator = BigInt::from(denominator);
            let rational = BigRational::new(
                original_numerator.clone(),
                original_denominator.clone(),
            );

            prop_assert!(rational.denom().is_positive());
            prop_assert_eq!(rational.numer().gcd(rational.denom()), BigInt::one());
            prop_assert_eq!(
                rational.numer() * &original_denominator,
                rational.denom() * &original_numerator,
            );

            let encoded = serde_json::to_vec(&rational).expect("canonical rational serializes");
            let decoded: BigRational =
                serde_json::from_slice(&encoded).expect("canonical rational deserializes");
            prop_assert_eq!(decoded, rational);
        }

        #[test]
        fn continued_fraction_round_trips_broad_signed_rationals(
            numerator in any::<i64>(),
            denominator in any::<i64>().prop_filter("nonzero denominator", |value| *value != 0),
        ) {
            let rational = BigRational::new(BigInt::from(numerator), BigInt::from(denominator));
            let coefficients = rational.continued_fraction();
            prop_assert_eq!(
                BigRational::from_continued_fraction(&coefficients),
                Some(rational.clone())
            );

            let mut meter = Unbounded;
            prop_assert_eq!(
                rational.metered_continued_fraction(&mut meter).unwrap(),
                coefficients.clone()
            );
            let mut meter = Unbounded;
            prop_assert_eq!(
                BigRational::metered_from_continued_fraction(&coefficients, &mut meter).unwrap(),
                Some(rational)
            );
        }

        #[test]
        fn generalized_integer_continued_fractions_remain_canonical(
            coefficients in proptest::collection::vec(any::<i32>(), 1..20),
        ) {
            let coefficients: Vec<BigInt> = coefficients.into_iter().map(BigInt::from).collect();
            if let Some(rational) = BigRational::from_continued_fraction(&coefficients) {
                prop_assert!(rational.denom().is_positive());
                prop_assert_eq!(
                    fsym_bigint::gcd(rational.numer(), rational.denom()),
                    BigInt::one()
                );
                prop_assert_eq!(
                    BigRational::new(rational.numer().clone(), rational.denom().clone()),
                    rational
                );
            }
        }

        #[test]
        fn owned_and_metered_scalar_lanes_match_naive_canonical_arithmetic(
            left_numerator in any::<i64>(),
            left_denominator in any::<i64>().prop_filter("nonzero left denominator", |value| *value != 0),
            right_numerator in any::<i64>(),
            right_denominator in any::<i64>().prop_filter("nonzero right denominator", |value| *value != 0),
        ) {
            let lhs = BigRational::new(
                BigInt::from(left_numerator),
                BigInt::from(left_denominator),
            );
            let rhs = BigRational::new(
                BigInt::from(right_numerator),
                BigInt::from(right_denominator),
            );
            let common_denominator = lhs.denom() * rhs.denom();
            let left_scaled = lhs.numer() * rhs.denom();
            let right_scaled = rhs.numer() * lhs.denom();
            let expected_add = BigRational::new(
                &left_scaled + &right_scaled,
                common_denominator.clone(),
            );
            let expected_sub = BigRational::new(
                &left_scaled - &right_scaled,
                common_denominator.clone(),
            );
            let expected_mul = BigRational::new(
                lhs.numer() * rhs.numer(),
                common_denominator.clone(),
            );

            prop_assert_eq!(&lhs + &rhs, expected_add.clone());
            prop_assert_eq!(&lhs - &rhs, expected_sub.clone());
            prop_assert_eq!(&lhs * &rhs, expected_mul.clone());

            let mut meter = Unbounded;
            prop_assert_eq!(lhs.metered_add(&rhs, &mut meter).unwrap(), expected_add);
            let mut meter = Unbounded;
            prop_assert_eq!(lhs.metered_sub(&rhs, &mut meter).unwrap(), expected_sub);
            let mut meter = Unbounded;
            prop_assert_eq!(lhs.metered_mul(&rhs, &mut meter).unwrap(), expected_mul);

            if !rhs.is_zero() {
                let expected_div = BigRational::new(
                    lhs.numer() * rhs.denom(),
                    lhs.denom() * rhs.numer(),
                );
                let (_, naive_remainder) = left_scaled.div_rem(&right_scaled);
                let expected_rem = BigRational::new(naive_remainder, common_denominator);
                prop_assert_eq!(&lhs / &rhs, expected_div.clone());
                prop_assert_eq!(&lhs % &rhs, expected_rem.clone());

                let mut meter = Unbounded;
                prop_assert_eq!(lhs.metered_div(&rhs, &mut meter).unwrap(), expected_div);
                let mut meter = Unbounded;
                prop_assert_eq!(lhs.metered_rem(&rhs, &mut meter).unwrap(), expected_rem);
            }

            for value in [
                lhs.cross_cancelled_add(&rhs),
                lhs.cross_cancelled_sub(&rhs),
                lhs.cross_cancelled_mul(&rhs),
            ] {
                prop_assert!(value.denom().is_positive());
                prop_assert_eq!(gcd(value.numer(), value.denom()), BigInt::one());
            }
        }
    }
}
