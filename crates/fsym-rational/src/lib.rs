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
    BigInt, NonZeroBigInt, gcd, metered_add as metered_bigint_add,
    metered_cmp as metered_bigint_cmp, metered_div_rem_nonzero, metered_gcd, metered_multiply,
    metered_pow as metered_bigint_pow, metered_subtract as metered_bigint_subtract,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use num_traits::{Num, One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Deserializer, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::str::FromStr;

/// Owned, canonical arbitrary-precision rational.
///
/// Values are reduced and their denominators are positive. The wrapped substrate is private so
/// persisted and semantic consumers cannot depend on its concrete representation.
#[derive(Clone, Default, Serialize)]
#[serde(transparent)]
pub struct BigRational(num_rational::Ratio<BigInt>);

impl PartialEq for BigRational {
    fn eq(&self, other: &Self) -> bool {
        self.numer() == other.numer() && self.denom() == other.denom()
    }
}

impl Eq for BigRational {}

impl PartialOrd for BigRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigRational {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_canonical_rationals(self, other)
    }
}

impl Hash for BigRational {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.numer().hash(state);
        self.denom().hash(state);
    }
}

fn compare_canonical_rationals(lhs: &BigRational, rhs: &BigRational) -> Ordering {
    let mut left_numerator = lhs.numer().clone();
    let mut left_denominator = lhs.denom().clone();
    let mut right_numerator = rhs.numer().clone();
    let mut right_denominator = rhs.denom().clone();
    let mut reverse = false;

    loop {
        let (left_integer, left_remainder) =
            div_mod_floor_positive_denominator(&left_numerator, &left_denominator);
        let (right_integer, right_remainder) =
            div_mod_floor_positive_denominator(&right_numerator, &right_denominator);
        let integer_order = left_integer.cmp(&right_integer);
        if integer_order != Ordering::Equal {
            return orient_ordering(integer_order, reverse);
        }

        let remainder_order = match (left_remainder.is_zero(), right_remainder.is_zero()) {
            (true, true) => return Ordering::Equal,
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        };
        if let Some(ordering) = remainder_order {
            return orient_ordering(ordering, reverse);
        }

        left_numerator = left_denominator;
        left_denominator = left_remainder;
        right_numerator = right_denominator;
        right_denominator = right_remainder;
        reverse = !reverse;
    }
}

fn div_mod_floor_positive_denominator(
    numerator: &BigInt,
    denominator: &BigInt,
) -> (BigInt, BigInt) {
    debug_assert!(denominator.is_positive());
    let (mut quotient, mut remainder) = numerator.div_rem(denominator);
    if remainder.is_negative() {
        quotient -= 1i64;
        remainder += denominator;
    }
    (quotient, remainder)
}

fn orient_ordering(ordering: Ordering, reverse: bool) -> Ordering {
    if reverse {
        ordering.reverse()
    } else {
        ordering
    }
}

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
    /// A checked rational work-buffer size could not be represented.
    SizeOverflow,
    /// A preflighted rational work-buffer reservation was refused by the allocator.
    AllocationFailure,
    /// A mathematically exact internal quotient failed its invariant check.
    InvariantViolation(&'static str),
}

impl fmt::Display for RationalArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meter(error) => fmt::Display::fmt(error, f),
            Self::DivisionByZero => f.write_str("rational division by zero"),
            Self::SizeOverflow => f.write_str("rational work-buffer size overflow"),
            Self::AllocationFailure => f.write_str("rational work-buffer allocation refused"),
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

    /// Constructs a reduced rational with a positive denominator under the caller's meter.
    ///
    /// A zero denominator is refused before normalization begins. GCD reduction, both exact
    /// quotients, sign normalization, and final publication all use governed arithmetic lanes.
    pub fn metered_new<M: BudgetMeter>(
        numer: &BigInt,
        denom: &BigInt,
        meter: &mut M,
    ) -> Result<Self, RationalArithmeticError> {
        meter.checkpoint()?;
        if denom.is_zero() {
            return rational_metered_error(RationalArithmeticError::DivisionByZero, meter);
        }

        let reduction = metered_gcd(numer, denom, meter)?;
        let mut numerator = metered_exact_quotient(
            numer,
            &reduction,
            "constructor numerator normalization",
            meter,
        )?;
        let mut denominator = metered_exact_quotient(
            denom,
            &reduction,
            "constructor denominator normalization",
            meter,
        )?;
        if denominator.is_negative() {
            numerator = metered_negate(numerator, meter)?;
            denominator = metered_negate(denominator, meter)?;
        }
        rational_metered_finish(
            Self(num_rational::Ratio::new_raw(numerator, denominator)),
            meter,
        )
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

    /// Compares two canonical rationals through a cancellation-first continued-fraction lane.
    ///
    /// The comparator never materializes cross-products. It iterates over reciprocal remainders,
    /// so adversarial Euclidean chains consume governed loop work without consuming call stack.
    /// A final checkpoint separates the fully classified ordering from publication.
    pub fn metered_cmp<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Ordering, RationalArithmeticError> {
        metered_compare_canonical_rationals(self, other, meter)
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

    /// Cancellation-first signed rational exponentiation.
    ///
    /// Negative powers of zero are refused with [`RationalArithmeticError::DivisionByZero`]
    /// instead of reaching the substrate panic path. Numerator and denominator powers use the
    /// bigint-owned governed binary lane; reciprocal sign normalization and final publication are
    /// checkpointed.
    pub fn metered_pow<M: BudgetMeter>(
        &self,
        exponent: i32,
        meter: &mut M,
    ) -> Result<Self, RationalArithmeticError> {
        meter.checkpoint()?;
        if exponent.is_negative() && self.is_zero() {
            return rational_metered_error(RationalArithmeticError::DivisionByZero, meter);
        }

        let magnitude = exponent.unsigned_abs();
        let numerator_power = metered_bigint_pow(self.numer(), magnitude, meter)?;
        let denominator_power = metered_bigint_pow(self.denom(), magnitude, meter)?;
        let (mut numerator, mut denominator) = if exponent.is_negative() {
            (denominator_power, numerator_power)
        } else {
            (numerator_power, denominator_power)
        };
        if denominator.is_negative() {
            numerator = metered_negate(numerator, meter)?;
            denominator = metered_negate(denominator, meter)?;
        }
        rational_metered_finish(
            Self(num_rational::Ratio::new_raw(numerator, denominator)),
            meter,
        )
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
        let remainder = metered_scaled_remainder(&left_scaled, &right_scaled, meter)?;
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

    /// Cancellation-first continued-fraction expansion with incremental, fallible coefficient
    /// storage and coefficient-height accounting.
    pub fn metered_continued_fraction<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<Vec<BigInt>, RationalArithmeticError> {
        meter.checkpoint()?;
        let coefficient_limit = match continued_fraction_coefficient_bound(self.denom().bits()) {
            Ok(limit) => limit,
            Err(error) => return rational_metered_error(error, meter),
        };
        self.metered_continued_fraction_with_limit(coefficient_limit, meter)
    }

    fn metered_continued_fraction_with_limit<M: BudgetMeter>(
        &self,
        coefficient_limit: usize,
        meter: &mut M,
    ) -> Result<Vec<BigInt>, RationalArithmeticError> {
        let mut numerator = metered_clone(self.numer(), meter)?;
        let mut denominator = metered_clone(self.denom(), meter)?;
        let mut coefficients = Vec::new();
        loop {
            meter.checkpoint()?;
            let Some(divisor) = NonZeroBigInt::new(&denominator) else {
                return rational_metered_finish(coefficients, meter);
            };
            let (mut quotient, mut remainder) =
                metered_div_rem_nonzero(&numerator, divisor, meter)?;
            if remainder.is_negative() {
                let one = metered_one(meter)?;
                quotient = metered_bigint_subtract(&quotient, &one, meter)?;
                remainder = metered_bigint_add(&remainder, &denominator, meter)?;
            }
            reserve_next_continued_fraction_slot(&mut coefficients, coefficient_limit, meter)?;
            charge_persisted_coefficient(&quotient, meter)?;
            coefficients.push(quotient);
            if remainder.is_zero() {
                return rational_metered_finish(coefficients, meter);
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

fn metered_compare_canonical_rationals<M: BudgetMeter>(
    lhs: &BigRational,
    rhs: &BigRational,
    meter: &mut M,
) -> Result<Ordering, RationalArithmeticError> {
    meter.checkpoint()?;
    let mut left_numerator = metered_clone(lhs.numer(), meter)?;
    let mut left_denominator = metered_clone(lhs.denom(), meter)?;
    let mut right_numerator = metered_clone(rhs.numer(), meter)?;
    let mut right_denominator = metered_clone(rhs.denom(), meter)?;
    let mut reverse = false;

    loop {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let (left_integer, left_remainder) =
            metered_div_mod_floor_positive_denominator(&left_numerator, &left_denominator, meter)?;
        let (right_integer, right_remainder) = metered_div_mod_floor_positive_denominator(
            &right_numerator,
            &right_denominator,
            meter,
        )?;
        let integer_order = metered_bigint_cmp(&left_integer, &right_integer, meter)?;
        if integer_order != Ordering::Equal {
            return rational_metered_finish(orient_ordering(integer_order, reverse), meter);
        }

        let remainder_order = match (left_remainder.is_zero(), right_remainder.is_zero()) {
            (true, true) => return rational_metered_finish(Ordering::Equal, meter),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        };
        if let Some(ordering) = remainder_order {
            return rational_metered_finish(orient_ordering(ordering, reverse), meter);
        }

        left_numerator = left_denominator;
        left_denominator = left_remainder;
        right_numerator = right_denominator;
        right_denominator = right_remainder;
        reverse = !reverse;
    }
}

fn metered_div_mod_floor_positive_denominator<M: BudgetMeter>(
    numerator: &BigInt,
    denominator: &BigInt,
    meter: &mut M,
) -> Result<(BigInt, BigInt), RationalArithmeticError> {
    let denominator = metered_require_nonzero(
        denominator,
        "canonical rational comparison denominator became zero",
        meter,
    )?;
    let (mut quotient, mut remainder) = metered_div_rem_nonzero(numerator, denominator, meter)?;
    if remainder.is_negative() {
        let one = metered_one(meter)?;
        quotient = metered_bigint_subtract(&quotient, &one, meter)?;
        remainder = metered_bigint_add(&remainder, denominator.get(), meter)?;
    }
    Ok((quotient, remainder))
}

fn metered_exact_quotient<M: BudgetMeter>(
    value: &BigInt,
    divisor: &BigInt,
    invariant: &'static str,
    meter: &mut M,
) -> Result<BigInt, RationalArithmeticError> {
    let divisor = metered_require_nonzero(divisor, invariant, meter)?;
    let (quotient, remainder) = metered_div_rem_nonzero(value, divisor, meter)?;
    if !remainder.is_zero() {
        return rational_metered_error(
            RationalArithmeticError::InvariantViolation(invariant),
            meter,
        );
    }
    Ok(quotient)
}

fn metered_scaled_remainder<M: BudgetMeter>(
    value: &BigInt,
    divisor: &BigInt,
    meter: &mut M,
) -> Result<BigInt, RationalArithmeticError> {
    let divisor = metered_require_nonzero(divisor, "nonzero remainder divisor became zero", meter)?;
    let (_, remainder) = metered_div_rem_nonzero(value, divisor, meter)?;
    Ok(remainder)
}

fn metered_require_nonzero<'a, M: BudgetMeter>(
    value: &'a BigInt,
    invariant: &'static str,
    meter: &mut M,
) -> Result<NonZeroBigInt<'a>, RationalArithmeticError> {
    match NonZeroBigInt::new(value) {
        Some(value) => Ok(value),
        None => rational_metered_error(
            RationalArithmeticError::InvariantViolation(invariant),
            meter,
        ),
    }
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
fn continued_fraction_coefficient_bound(
    denominator_bits: u64,
) -> Result<usize, RationalArithmeticError> {
    let bound = denominator_bits
        .checked_mul(2)
        .and_then(|bits| bits.checked_add(1))
        .ok_or(RationalArithmeticError::SizeOverflow)?;
    usize::try_from(bound).map_err(|_| RationalArithmeticError::SizeOverflow)
}

fn reserve_next_continued_fraction_slot<M: BudgetMeter>(
    coefficients: &mut Vec<BigInt>,
    coefficient_limit: usize,
    meter: &mut M,
) -> Result<(), RationalArithmeticError> {
    if coefficients.len() >= coefficient_limit {
        return rational_metered_error(
            RationalArithmeticError::InvariantViolation(
                "continued-fraction coefficient bound exceeded",
            ),
            meter,
        );
    }
    if coefficients.len() < coefficients.capacity() {
        return Ok(());
    }

    let target_capacity = if coefficients.capacity() == 0 {
        1
    } else {
        match coefficients.capacity().checked_mul(2) {
            Some(capacity) => capacity,
            None => return rational_metered_error(RationalArithmeticError::SizeOverflow, meter),
        }
    }
    .min(coefficient_limit);
    let additional = match target_capacity.checked_sub(coefficients.len()) {
        Some(additional) => additional,
        None => return rational_metered_error(RationalArithmeticError::SizeOverflow, meter),
    };
    reserve_continued_fraction_slots(coefficients, additional, meter)
}

fn reserve_continued_fraction_slots<M: BudgetMeter>(
    coefficients: &mut Vec<BigInt>,
    additional: usize,
    meter: &mut M,
) -> Result<(), RationalArithmeticError> {
    let Some(slot_bytes) = u64::try_from(std::mem::size_of::<BigInt>()).ok() else {
        return rational_metered_error(RationalArithmeticError::SizeOverflow, meter);
    };
    let Some(additional_slots) = u64::try_from(additional).ok() else {
        return rational_metered_error(RationalArithmeticError::SizeOverflow, meter);
    };
    let Some(additional_bytes) = additional_slots.checked_mul(slot_bytes) else {
        return rational_metered_error(RationalArithmeticError::SizeOverflow, meter);
    };

    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::MemoryBytes, additional_bytes),
        (Dimension::AllocationCount, 1),
    ])?;
    if coefficients.try_reserve_exact(additional).is_err() {
        return rational_metered_error(RationalArithmeticError::AllocationFailure, meter);
    }
    meter.checkpoint()?;
    Ok(())
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

fn rounded_binary_quotient(
    numerator: &BigInt,
    denominator: &BigInt,
    binary_shift: i32,
) -> Option<u64> {
    let (scaled_numerator, scaled_denominator) = if binary_shift >= 0 {
        (
            numerator << u32::try_from(binary_shift).ok()?,
            denominator.clone(),
        )
    } else {
        (
            numerator.clone(),
            denominator << binary_shift.unsigned_abs(),
        )
    };
    let (quotient, remainder) = scaled_numerator.div_rem(&scaled_denominator);
    let twice_remainder = &remainder << 1u32;
    let round_up = twice_remainder > scaled_denominator
        || (twice_remainder == scaled_denominator && (&quotient % 2i64) == BigInt::one());
    let quotient = quotient.to_u64()?;
    if round_up {
        quotient.checked_add(1)
    } else {
        Some(quotient)
    }
}

/// Converts a canonical exact rational to binary64 with round-to-nearest, ties-to-even.
///
/// The coefficient magnitudes are never converted independently: doing so can form
/// `infinity / infinity` and turn an ordinary finite rational into `NaN`. Exact integer scaling
/// instead computes the final normal or subnormal significand before constructing the float bits.
fn rational_to_f64(value: &BigRational) -> Option<f64> {
    const FRACTION_BITS: u64 = 52;
    const SIGNIFICAND_FLOOR: u64 = 1u64 << FRACTION_BITS;
    const SIGNIFICAND_CARRY: u64 = SIGNIFICAND_FLOOR << 1;
    const MAX_NORMAL_EXPONENT: i128 = 1023;
    const MIN_NORMAL_EXPONENT: i128 = -1022;
    const HALF_MIN_SUBNORMAL_EXPONENT: i128 = -1075;
    const EXPONENT_BIAS: i128 = 1023;
    const SIGN_BIT: u64 = 1u64 << 63;
    const INFINITY_BITS: u64 = 0x7ff0_0000_0000_0000;

    let negative = value.numer().is_negative();
    let numerator = value.numer().abs();
    if numerator.is_zero() {
        return Some(0.0);
    }
    let denominator = value.denom();

    let mut exponent = i128::from(numerator.bits()) - i128::from(denominator.bits());
    if exponent > MAX_NORMAL_EXPONENT + 1 {
        return Some(f64::from_bits(
            INFINITY_BITS | if negative { SIGN_BIT } else { 0 },
        ));
    }
    if exponent < HALF_MIN_SUBNORMAL_EXPONENT {
        return Some(f64::from_bits(if negative { SIGN_BIT } else { 0 }));
    }

    let below_estimated_power = if exponent >= 0 {
        numerator < (denominator << u32::try_from(exponent).ok()?)
    } else {
        (&numerator << u32::try_from(-exponent).ok()?) < *denominator
    };
    if below_estimated_power {
        exponent -= 1;
    }

    let magnitude_bits = if exponent > MAX_NORMAL_EXPONENT {
        INFINITY_BITS
    } else if exponent < HALF_MIN_SUBNORMAL_EXPONENT {
        0
    } else if exponent < MIN_NORMAL_EXPONENT {
        let significand = rounded_binary_quotient(&numerator, denominator, 1074)?;
        if significand > SIGNIFICAND_FLOOR {
            return None;
        }
        significand
    } else {
        let binary_shift = i32::try_from(i128::from(FRACTION_BITS) - exponent).ok()?;
        let mut significand = rounded_binary_quotient(&numerator, denominator, binary_shift)?;
        if significand == SIGNIFICAND_CARRY {
            significand = SIGNIFICAND_FLOOR;
            exponent += 1;
        }
        if exponent > MAX_NORMAL_EXPONENT {
            INFINITY_BITS
        } else {
            if !(SIGNIFICAND_FLOOR..SIGNIFICAND_CARRY).contains(&significand) {
                return None;
            }
            let biased_exponent = u64::try_from(exponent + EXPONENT_BIAS).ok()?;
            (biased_exponent << FRACTION_BITS) | (significand - SIGNIFICAND_FLOOR)
        }
    };

    Some(f64::from_bits(
        magnitude_bits | if negative { SIGN_BIT } else { 0 },
    ))
}

impl ToPrimitive for BigRational {
    fn to_i64(&self) -> Option<i64> {
        self.to_integer().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.to_integer().to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        rational_to_f64(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetError, BudgetLimits, DIMENSION_COUNT, Unbounded};
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

    #[derive(Debug, Default)]
    struct CountingMeter {
        dimensions: [u64; DIMENSION_COUNT],
        checkpoints: usize,
    }

    impl BudgetMeter for CountingMeter {
        fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
            self.dimensions[dimension.index()] =
                self.dimensions[dimension.index()].saturating_add(amount);
            Ok(())
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            for &(dimension, amount) in charges {
                self.charge(dimension, amount)?;
            }
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints = self.checkpoints.saturating_add(1);
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct TerminalProbe {
        checkpoints: usize,
        trailing_uncharged_checkpoints: usize,
    }

    impl BudgetMeter for TerminalProbe {
        fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
            self.trailing_uncharged_checkpoints = 0;
            Ok(())
        }

        fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            self.trailing_uncharged_checkpoints = 0;
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints = self.checkpoints.saturating_add(1);
            self.trailing_uncharged_checkpoints =
                self.trailing_uncharged_checkpoints.saturating_add(1);
            Ok(())
        }
    }

    fn scalar_normalize(numerator: i64, denominator: i64) -> (i128, i128) {
        assert_ne!(denominator, 0);
        let mut left = i128::from(numerator).abs();
        let mut right = i128::from(denominator).abs();
        while right != 0 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        let mut numerator = i128::from(numerator) / left;
        let mut denominator = i128::from(denominator) / left;
        if denominator < 0 {
            numerator = -numerator;
            denominator = -denominator;
        }
        (numerator, denominator)
    }

    fn bigint_from_i128(value: i128) -> BigInt {
        BigInt::from_str_radix(&value.to_string(), 10).expect("an i128 decimal is a valid BigInt")
    }

    #[derive(Debug, Default)]
    struct GrowthMeter {
        reservations: Vec<(u64, u64)>,
    }

    impl BudgetMeter for GrowthMeter {
        fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
            Ok(())
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            let memory_bytes = charges
                .iter()
                .find_map(|(dimension, amount)| {
                    (*dimension == Dimension::MemoryBytes).then_some(*amount)
                })
                .unwrap_or(0);
            let allocation_count = charges
                .iter()
                .find_map(|(dimension, amount)| {
                    (*dimension == Dimension::AllocationCount).then_some(*amount)
                })
                .unwrap_or(0);
            self.reservations.push((memory_bytes, allocation_count));
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            Ok(())
        }
    }

    fn cross_product_order(lhs: &BigRational, rhs: &BigRational) -> Ordering {
        (lhs.numer() * rhs.denom()).cmp(&(rhs.numer() * lhs.denom()))
    }

    fn fibonacci_pair(steps: usize) -> (BigInt, BigInt) {
        let mut previous = BigInt::zero();
        let mut current = BigInt::one();
        for _ in 0..steps {
            let next = &previous + &current;
            previous = current;
            current = next;
        }
        (previous, current)
    }

    #[test]
    fn radix_parser_inherits_bigint_admission_without_unwinding() {
        for radix in [0, 1, 37, u32::MAX] {
            let parsed = std::panic::catch_unwind(|| BigRational::from_str_radix("1/2", radix))
                .expect("fallible rational parsing must not unwind for an unsupported radix");
            assert!(parsed.is_err(), "radix {radix}");
        }

        assert_eq!(
            BigRational::from_str_radix("101/11", 2),
            Ok(BigRational::new(BigInt::from(5), BigInt::from(3)))
        );
        assert_eq!(
            BigRational::from_str_radix("z/10", 36),
            Ok(BigRational::new(BigInt::from(35), BigInt::from(36)))
        );
        for malformed in ["1", "1/2/3", "2/1", "/1", "1/", "1/0"] {
            assert!(
                BigRational::from_str_radix(malformed, 2).is_err(),
                "malformed rational was accepted: {malformed}"
            );
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
    fn ordering_equality_and_hash_are_iterative_on_deep_euclidean_chains() {
        let (f_n, f_n_plus_one) = fibonacci_pair(4_096);
        let f_n_plus_two = &f_n + &f_n_plus_one;
        let lhs = BigRational::new(f_n_plus_one.clone(), f_n);
        let rhs = BigRational::new(f_n_plus_two, f_n_plus_one);
        let expected = cross_product_order(&lhs, &rhs);

        let worker = std::thread::Builder::new()
            .name("stack-safe-rational-order".to_string())
            .stack_size(128 * 1024)
            .spawn(move || {
                let ordering = lhs.cmp(&rhs);
                let unequal = lhs != rhs;
                let mut lhs_hasher = std::collections::hash_map::DefaultHasher::new();
                lhs.hash(&mut lhs_hasher);
                let mut lhs_clone_hasher = std::collections::hash_map::DefaultHasher::new();
                lhs.clone().hash(&mut lhs_clone_hasher);
                let metered_ordering = lhs.metered_cmp(&rhs, &mut Unbounded);
                let mut cancelled = CheckpointMeter::cancelling_at(32);
                let metered_cancellation = lhs.metered_cmp(&rhs, &mut cancelled);
                (
                    ordering,
                    unequal,
                    lhs_hasher.finish(),
                    lhs_clone_hasher.finish(),
                    metered_ordering,
                    metered_cancellation,
                )
            })
            .expect("comparison worker thread must start");
        let (ordering, unequal, hash, clone_hash, metered_ordering, metered_cancellation) = worker
            .join()
            .expect("iterative comparison must not exhaust stack");
        assert_eq!(ordering, expected);
        assert!(unequal);
        assert_eq!(hash, clone_hash);
        assert_eq!(metered_ordering, Ok(expected));
        assert_eq!(
            metered_cancellation,
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );

        let canonical = BigRational::new(BigInt::from(3), BigInt::from(2));
        let independently_reduced = BigRational::new(BigInt::from(9), BigInt::from(6));
        let mut canonical_hasher = std::collections::hash_map::DefaultHasher::new();
        canonical.hash(&mut canonical_hasher);
        let mut reduced_hasher = std::collections::hash_map::DefaultHasher::new();
        independently_reduced.hash(&mut reduced_hasher);
        assert_eq!(canonical, independently_reduced);
        assert_eq!(canonical_hasher.finish(), reduced_hasher.finish());
    }

    #[test]
    fn governed_comparison_matches_order_and_enforces_resources_and_cancellation() {
        for (left, right) in [
            ((-13, 8), (-21, 13)),
            ((-2, 1), (-7, 3)),
            ((0, 1), (1, 97)),
            ((3, 2), (9, 6)),
            ((233, 144), (377, 233)),
        ] {
            let lhs = BigRational::new(left.0.into(), left.1.into());
            let rhs = BigRational::new(right.0.into(), right.1.into());
            assert_eq!(lhs.cmp(&rhs), cross_product_order(&lhs, &rhs));
            assert_eq!(lhs.metered_cmp(&rhs, &mut Unbounded), Ok(lhs.cmp(&rhs)));
        }

        let lhs = BigRational::new(BigInt::from(-13), BigInt::from(8));
        let rhs = BigRational::new(BigInt::from(-21), BigInt::from(13));
        let mut measured = CountingMeter::default();
        assert_eq!(lhs.metered_cmp(&rhs, &mut measured), Ok(lhs.cmp(&rhs)));
        assert_eq!(measured.dimensions, [353, 272, 50, 0, 0]);
        assert_eq!(measured.checkpoints, 407);
        assert!(measured.dimensions[Dimension::ComputeSteps.index()] > 0);
        assert!(measured.dimensions[Dimension::MemoryBytes.index()] > 0);
        assert!(measured.dimensions[Dimension::AllocationCount.index()] > 0);
        assert_eq!(measured.dimensions[Dimension::DepthLimit.index()], 0);
        assert_eq!(measured.dimensions[Dimension::RandomDraws.index()], 0);
        assert!(measured.checkpoints > 1);

        let mut terminal = TerminalProbe::default();
        assert_eq!(lhs.metered_cmp(&rhs, &mut terminal), Ok(lhs.cmp(&rhs)));
        assert_eq!(terminal.trailing_uncharged_checkpoints, 2);

        for (terminal_lhs, terminal_rhs, expected) in [
            (
                BigRational::new(BigInt::from(3), BigInt::from(2)),
                BigRational::new(BigInt::from(9), BigInt::from(6)),
                Ordering::Equal,
            ),
            (
                BigRational::from_integer(BigInt::from(2)),
                BigRational::new(BigInt::from(7), BigInt::from(3)),
                Ordering::Less,
            ),
        ] {
            let mut shape = TerminalProbe::default();
            assert_eq!(
                terminal_lhs.metered_cmp(&terminal_rhs, &mut shape),
                Ok(expected)
            );
            assert_eq!(shape.trailing_uncharged_checkpoints, 2);
            let mut counted = CountingMeter::default();
            assert_eq!(
                terminal_lhs.metered_cmp(&terminal_rhs, &mut counted),
                Ok(expected)
            );
            let mut cancelled = CheckpointMeter::cancelling_at(counted.checkpoints);
            assert_eq!(
                terminal_lhs.metered_cmp(&terminal_rhs, &mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled))
            );
        }

        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let admitted = measured.dimensions[dimension.index()];
            let mut limits = BudgetLimits::uniform(u64::MAX, 0);
            limits.dimensions[dimension.index()] = admitted - 1;
            let mut budget = Budget::new(limits);
            assert!(matches!(
                lhs.metered_cmp(&rhs, &mut budget),
                Err(RationalArithmeticError::Meter(MeterError::Budget(
                    BudgetError::Exhausted {
                        dimension: exhausted,
                        ..
                    }
                ))) if exhausted == dimension
            ));
        }

        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                lhs.metered_cmp(&rhs, &mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled)),
                "checkpoint {checkpoint} must prevent ordering publication"
            );
            assert_eq!(cancelled.checkpoints, checkpoint);
        }
        assert_eq!(lhs.metered_cmp(&rhs, &mut Unbounded), Ok(lhs.cmp(&rhs)));
    }

    #[test]
    fn governed_comparison_topology_distinguishes_deep_chains_from_early_decisions() {
        let (f_n, f_n_plus_one) = fibonacci_pair(64);
        let f_n_plus_two = &f_n + &f_n_plus_one;
        let deep_lhs = BigRational::new(f_n_plus_one.clone(), f_n.clone());
        let deep_rhs = BigRational::new(f_n_plus_two, f_n_plus_one.clone());
        let early_rhs = BigRational::new(&f_n_plus_one + (&f_n * 2i64), f_n);

        assert_eq!(
            deep_lhs.height().total_limbs(),
            early_rhs.height().total_limbs()
        );
        let mut deep = CountingMeter::default();
        assert_eq!(
            deep_lhs.metered_cmp(&deep_rhs, &mut deep),
            Ok(deep_lhs.cmp(&deep_rhs))
        );
        let mut early = CountingMeter::default();
        assert_eq!(
            deep_lhs.metered_cmp(&early_rhs, &mut early),
            Ok(deep_lhs.cmp(&early_rhs))
        );
        assert!(
            deep.dimensions[Dimension::ComputeSteps.index()]
                > early.dimensions[Dimension::ComputeSteps.index()] * 8
        );
        assert!(deep.checkpoints > early.checkpoints * 8);

        let late_checkpoint = deep.checkpoints * 3 / 4;
        let mut cancelled = CheckpointMeter::cancelling_at(late_checkpoint);
        assert_eq!(
            deep_lhs.metered_cmp(&deep_rhs, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.checkpoints, late_checkpoint);
        assert_eq!(
            deep_lhs.metered_cmp(&deep_rhs, &mut Unbounded),
            Ok(deep_lhs.cmp(&deep_rhs))
        );
    }

    proptest! {
        #[test]
        fn rational_order_matches_independent_i128_cross_products(
            left_numerator in -1_000_000i64..=1_000_000,
            left_denominator in 1i64..=1_000_000,
            right_numerator in -1_000_000i64..=1_000_000,
            right_denominator in 1i64..=1_000_000,
            third_numerator in -1_000_000i64..=1_000_000,
            third_denominator in 1i64..=1_000_000,
        ) {
            let lhs = BigRational::new(left_numerator.into(), left_denominator.into());
            let rhs = BigRational::new(right_numerator.into(), right_denominator.into());
            let third = BigRational::new(third_numerator.into(), third_denominator.into());
            let expected = (i128::from(left_numerator) * i128::from(right_denominator))
                .cmp(&(i128::from(right_numerator) * i128::from(left_denominator)));
            prop_assert_eq!(lhs.cmp(&rhs), expected);
            prop_assert_eq!(lhs.partial_cmp(&rhs), Some(expected));
            prop_assert_eq!(lhs == rhs, expected == Ordering::Equal);
            prop_assert_eq!(lhs.metered_cmp(&rhs, &mut Unbounded), Ok(expected));
            prop_assert_eq!(rhs.cmp(&lhs), expected.reverse());
            if lhs <= rhs && rhs <= third {
                prop_assert!(lhs <= third);
            }
        }
    }

    #[test]
    fn governed_constructor_normalizes_sign_reduction_and_large_coefficients() {
        for (numerator, denominator, expected_numerator, expected_denominator) in [
            (0i64, 17i64, 0i64, 1i64),
            (0, -17, 0, 1),
            (7, 1, 7, 1),
            (7, -1, -7, 1),
            (-42, -30, 7, 5),
            (-42, 30, -7, 5),
        ] {
            let numerator = BigInt::from(numerator);
            let denominator = BigInt::from(denominator);
            let actual = BigRational::metered_new(&numerator, &denominator, &mut Unbounded)
                .expect("valid coefficients normalize under the unbounded meter");
            assert_eq!(actual.numer(), &BigInt::from(expected_numerator));
            assert_eq!(actual.denom(), &BigInt::from(expected_denominator));
            assert_eq!(
                actual,
                BigRational::new(numerator.clone(), denominator.clone())
            );
            assert!(actual.denom().is_positive());
            assert_eq!(gcd(actual.numer(), actual.denom()), BigInt::one());
            assert_eq!(actual.numer() * &denominator, actual.denom() * &numerator);
        }

        let shared_factor = (BigInt::one() << 4_096u32) - 1i64;
        let numerator = BigInt::from(-42) * &shared_factor;
        let denominator = BigInt::from(-30) * &shared_factor;
        assert_eq!(
            BigRational::metered_new(&numerator, &denominator, &mut Unbounded),
            Ok(BigRational::new(BigInt::from(7), BigInt::from(5)))
        );
    }

    #[test]
    fn governed_constructor_refuses_zero_and_pins_resource_topology() {
        let numerator = BigInt::from(-42);
        let denominator = BigInt::from(-30);
        let mut measured = CountingMeter::default();
        assert_eq!(
            BigRational::metered_new(&numerator, &denominator, &mut measured),
            Ok(BigRational::new(BigInt::from(7), BigInt::from(5)))
        );
        assert_eq!(measured.dimensions, [228, 116, 22, 0, 0]);
        assert_eq!(measured.checkpoints, 253);

        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let admitted = measured.dimensions[dimension.index()];
            assert!(admitted > 0);
            let mut limits = BudgetLimits::uniform(u64::MAX, 0);
            limits.dimensions[dimension.index()] = admitted - 1;
            let mut budget = Budget::new(limits);
            assert!(matches!(
                BigRational::metered_new(&numerator, &denominator, &mut budget),
                Err(RationalArithmeticError::Meter(MeterError::Budget(
                    BudgetError::Exhausted {
                        dimension: exhausted,
                        ..
                    }
                ))) if exhausted == dimension
            ));
        }

        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                BigRational::metered_new(&numerator, &denominator, &mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled))
            );
            assert_eq!(cancelled.checkpoints, checkpoint);
        }
        assert_eq!(
            BigRational::metered_new(&numerator, &denominator, &mut Unbounded),
            Ok(BigRational::new(BigInt::from(7), BigInt::from(5)))
        );

        let zero = BigInt::zero();
        let mut refused = CountingMeter::default();
        assert_eq!(
            BigRational::metered_new(&numerator, &zero, &mut refused),
            Err(RationalArithmeticError::DivisionByZero)
        );
        assert_eq!(refused.dimensions, [0; DIMENSION_COUNT]);
        assert_eq!(refused.checkpoints, 2);
        for checkpoint in 1..=refused.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                BigRational::metered_new(&numerator, &zero, &mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled))
            );
        }
    }

    #[test]
    fn governed_signed_power_handles_reciprocals_refusal_budget_and_cancellation() {
        let value = BigRational::new(BigInt::from(2), BigInt::from(3));
        assert_eq!(
            value.metered_pow(-5, &mut Unbounded),
            Ok(BigRational::new(BigInt::from(243), BigInt::from(32)))
        );
        assert_eq!(
            (-value.clone()).metered_pow(-5, &mut Unbounded),
            Ok(BigRational::new(BigInt::from(-243), BigInt::from(32)))
        );
        assert_eq!(value.metered_pow(0, &mut Unbounded), Ok(BigRational::one()));

        let zero = BigRational::zero();
        assert_eq!(
            zero.metered_pow(-1, &mut Unbounded),
            Err(RationalArithmeticError::DivisionByZero)
        );
        for checkpoint in 1..=2 {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                zero.metered_pow(-1, &mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled))
            );
            assert_eq!(cancelled.checkpoints, checkpoint);
        }

        let mut budget = Budget::new(BudgetLimits::uniform(31, 0));
        let before = budget.snapshot();
        assert_eq!(
            value.metered_pow(i32::MIN, &mut budget),
            Err(RationalArithmeticError::Meter(MeterError::Budget(
                BudgetError::Exhausted {
                    dimension: Dimension::ComputeSteps,
                    requested: 32,
                    remaining: 31,
                }
            )))
        );
        assert_eq!(budget.snapshot(), before);

        let mut baseline = CheckpointMeter::default();
        let expected = value
            .metered_pow(-5, &mut baseline)
            .expect("baseline power succeeds");
        for checkpoint in 1..=baseline.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                value.metered_pow(-5, &mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled)),
                "checkpoint {checkpoint} did not stop publication"
            );
            assert_eq!(cancelled.checkpoints, checkpoint);
        }
        assert_eq!(expected, BigRational::new(243.into(), 32.into()));
    }

    #[test]
    fn binary64_conversion_balances_large_coefficients_and_rounds_boundaries() {
        let scale = &BigInt::one() << 2000u32;
        let near_one = BigRational::new(&scale + 1i64, &scale - 1i64);
        let negative_near_one = -near_one.clone();
        assert_eq!(near_one.to_f64(), Some(1.0));
        assert_eq!(negative_near_one.to_f64(), Some(-1.0));

        let tie_denominator = &BigInt::one() << 53u32;
        let tie_above_one = BigRational::new(&tie_denominator + 1i64, tie_denominator.clone());
        assert_eq!(tie_above_one.to_f64(), Some(1.0));
        let odd_tie_above_one = BigRational::new(&tie_denominator + 3i64, tie_denominator);
        assert_eq!(
            odd_tie_above_one.to_f64().map(f64::to_bits),
            Some(1.0f64.to_bits() + 2)
        );
        let above_tie_denominator = &BigInt::one() << 54u32;
        let above_tie = BigRational::new(&above_tie_denominator + 3i64, above_tie_denominator);
        assert_eq!(
            above_tie.to_f64().map(f64::to_bits),
            Some(1.0f64.to_bits() + 1)
        );

        let min_subnormal_denominator = &BigInt::one() << 1074u32;
        let min_subnormal = BigRational::new(BigInt::one(), min_subnormal_denominator);
        assert_eq!(min_subnormal.to_f64().map(f64::to_bits), Some(1));

        let transition_denominator = &BigInt::one() << 1075u32;
        let transition_midpoint =
            BigRational::new((&BigInt::one() << 53u32) - 1i64, transition_denominator);
        assert_eq!(
            transition_midpoint.to_f64().map(f64::to_bits),
            Some(f64::MIN_POSITIVE.to_bits())
        );

        let half_subnormal_denominator = &BigInt::one() << 1075u32;
        let positive_half = BigRational::new(BigInt::one(), half_subnormal_denominator.clone());
        let negative_half = BigRational::new(BigInt::from(-1), half_subnormal_denominator);
        assert_eq!(positive_half.to_f64().map(f64::to_bits), Some(0));
        assert_eq!(negative_half.to_f64().map(f64::to_bits), Some(1u64 << 63));
        let below_half = BigRational::new(BigInt::from(-1), &BigInt::one() << 1076u32);
        assert_eq!(below_half.to_f64().map(f64::to_bits), Some(1u64 << 63));

        let max_finite = ((&BigInt::one() << 53u32) - 1i64) << 971u32;
        assert_eq!(
            BigRational::from_integer(max_finite).to_f64(),
            Some(f64::MAX)
        );
        assert_eq!(
            BigRational::from_integer(&BigInt::one() << 1024u32).to_f64(),
            Some(f64::INFINITY)
        );
        assert_eq!(
            BigRational::from_integer(-(&BigInt::one() << 1025u32)).to_f64(),
            Some(f64::NEG_INFINITY)
        );
        let overflow_midpoint = ((&BigInt::one() << 54u32) - 1i64) << 970u32;
        assert_eq!(
            BigRational::from_integer(&overflow_midpoint - 1i64).to_f64(),
            Some(f64::MAX)
        );
        assert_eq!(
            BigRational::from_integer(overflow_midpoint).to_f64(),
            Some(f64::INFINITY)
        );
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
    fn invariant_refusals_observe_one_terminal_checkpoint() {
        let fixture = "exact-quotient fixture";
        let mut zero_divisor = CheckpointMeter::default();
        assert_eq!(
            metered_exact_quotient(&BigInt::one(), &BigInt::zero(), fixture, &mut zero_divisor,),
            Err(RationalArithmeticError::InvariantViolation(fixture))
        );
        assert_eq!(zero_divisor.checkpoints, 1);
        assert!(!zero_divisor.charged);

        let mut cancelled = CheckpointMeter::cancelling_at(1);
        assert_eq!(
            metered_exact_quotient(&BigInt::one(), &BigInt::zero(), fixture, &mut cancelled,),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.checkpoints, 1);
        assert!(!cancelled.charged);

        let mut zero_remainder_divisor = CheckpointMeter::default();
        assert_eq!(
            metered_scaled_remainder(&BigInt::one(), &BigInt::zero(), &mut zero_remainder_divisor,),
            Err(RationalArithmeticError::InvariantViolation(
                "nonzero remainder divisor became zero"
            ))
        );
        assert_eq!(zero_remainder_divisor.checkpoints, 1);
        assert!(!zero_remainder_divisor.charged);

        let value = BigInt::from(5);
        let divisor = BigInt::from(2);
        let mut division = CheckpointMeter::default();
        let (_, remainder) = metered_div_rem_nonzero(
            &value,
            NonZeroBigInt::new(&divisor).expect("fixture divisor is nonzero"),
            &mut division,
        )
        .expect("governed division succeeds");
        assert_eq!(remainder, BigInt::one());
        assert!(division.charged);

        let mut inexact = CheckpointMeter::default();
        assert_eq!(
            metered_exact_quotient(&value, &divisor, fixture, &mut inexact),
            Err(RationalArithmeticError::InvariantViolation(fixture))
        );
        assert_eq!(inexact.checkpoints, division.checkpoints + 1);
        assert!(inexact.charged);

        let mut cancelled = CheckpointMeter::cancelling_at(inexact.checkpoints);
        assert_eq!(
            metered_exact_quotient(&value, &divisor, fixture, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.checkpoints, inexact.checkpoints);
        assert!(cancelled.charged);

        let exact_value = BigInt::from(6);
        let mut exact_division = CheckpointMeter::default();
        metered_div_rem_nonzero(
            &exact_value,
            NonZeroBigInt::new(&divisor).expect("fixture divisor is nonzero"),
            &mut exact_division,
        )
        .expect("governed division succeeds");
        let mut exact = CheckpointMeter::default();
        assert_eq!(
            metered_exact_quotient(&exact_value, &divisor, fixture, &mut exact),
            Ok(BigInt::from(3))
        );
        assert_eq!(exact.checkpoints, exact_division.checkpoints);
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

        for malformed_coefficient in [
            "[[1,[]],[1,[1]]]",
            "[[1,[1]],[1,[1,0]]]",
            "[[0,[1]],[1,[1]]]",
        ] {
            assert!(
                serde_json::from_str::<BigRational>(malformed_coefficient).is_err(),
                "rational admitted a noncanonical integer coefficient: {malformed_coefficient}"
            );
        }
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
            Err(RationalArithmeticError::Meter(MeterError::Budget(_)))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            BigRational::metered_from_continued_fraction(&actual, &mut budget),
            Err(MeterError::Budget(_))
        ));
    }

    #[test]
    fn metered_expansion_allocates_for_actual_sparse_output() {
        let denominator = BigInt::one() << 4_096u32;
        let value = BigRational::new(BigInt::one(), denominator.clone());
        let mut meter = Unbounded;
        let coefficients = value
            .metered_continued_fraction(&mut meter)
            .expect("sparse expansion uses bounded incremental storage");

        assert_eq!(coefficients, [BigInt::zero(), denominator]);
        assert!(coefficients.capacity() <= 4);
        let theorem_limit = continued_fraction_coefficient_bound(value.denom().bits())
            .expect("4097 denominator bits have a representable theorem bound");
        assert!(theorem_limit > coefficients.capacity().saturating_mul(1_000));
    }

    #[test]
    fn continued_fraction_growth_refuses_before_reserve_and_reports_allocator_failure() {
        let slot_bytes =
            u64::try_from(std::mem::size_of::<BigInt>()).expect("BigInt header size fits u64");

        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::MemoryBytes.index()] = slot_bytes - 1;
        let mut budget = Budget::new(limits);
        let before = budget.snapshot();
        let mut coefficients = Vec::new();
        assert_eq!(
            reserve_continued_fraction_slots(&mut coefficients, 1, &mut budget),
            Err(RationalArithmeticError::Meter(MeterError::Budget(
                BudgetError::Exhausted {
                    dimension: Dimension::MemoryBytes,
                    requested: slot_bytes,
                    remaining: slot_bytes - 1,
                }
            )))
        );
        assert_eq!(coefficients.capacity(), 0);
        assert_eq!(budget.snapshot(), before);

        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::AllocationCount.index()] = 0;
        let mut budget = Budget::new(limits);
        let before = budget.snapshot();
        assert_eq!(
            reserve_continued_fraction_slots(&mut coefficients, 1, &mut budget),
            Err(RationalArithmeticError::Meter(MeterError::Budget(
                BudgetError::Exhausted {
                    dimension: Dimension::AllocationCount,
                    requested: 1,
                    remaining: 0,
                }
            )))
        );
        assert_eq!(coefficients.capacity(), 0);
        assert_eq!(budget.snapshot(), before);

        let impossible_slots = (isize::MAX as usize)
            .checked_div(std::mem::size_of::<BigInt>())
            .and_then(|slots| slots.checked_add(1))
            .expect("an impossible Vec<BigInt> capacity is representable");
        assert_eq!(
            reserve_continued_fraction_slots(&mut coefficients, impossible_slots, &mut Unbounded,),
            Err(RationalArithmeticError::AllocationFailure)
        );
        assert_eq!(coefficients.capacity(), 0);

        let mut measured = CheckpointMeter::default();
        assert_eq!(
            reserve_continued_fraction_slots(&mut coefficients, impossible_slots, &mut measured,),
            Err(RationalArithmeticError::AllocationFailure)
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            reserve_continued_fraction_slots(&mut coefficients, impossible_slots, &mut cancelled,),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.checkpoints, measured.checkpoints);
        assert_eq!(coefficients.capacity(), 0);
        assert_eq!(
            continued_fraction_coefficient_bound(u64::MAX),
            Err(RationalArithmeticError::SizeOverflow)
        );
    }

    #[test]
    fn continued_fraction_growth_topology_is_geometric_and_capped() {
        let logical_limit = 7;
        let slot_bytes =
            u64::try_from(std::mem::size_of::<BigInt>()).expect("BigInt header size fits u64");
        let mut coefficients = Vec::new();
        let mut meter = GrowthMeter::default();

        for value in 0i64..6 {
            reserve_next_continued_fraction_slot(&mut coefficients, logical_limit, &mut meter)
                .expect("slot growth stays within its logical cap");
            assert!(coefficients.capacity() <= logical_limit);
            coefficients.push(BigInt::from(value));
        }

        assert_eq!(
            meter.reservations,
            [
                (slot_bytes, 1),
                (slot_bytes, 1),
                (slot_bytes * 2, 1),
                (slot_bytes * 3, 1),
            ]
        );
        assert_eq!(coefficients.capacity(), logical_limit);
    }

    #[test]
    fn metered_expansion_enforces_the_logical_coefficient_bound() {
        let value = BigRational::new(BigInt::from(415), BigInt::from(93));
        assert_eq!(value.continued_fraction().len(), 4);
        assert_eq!(
            value.metered_continued_fraction_with_limit(3, &mut Unbounded),
            Err(RationalArithmeticError::InvariantViolation(
                "continued-fraction coefficient bound exceeded"
            ))
        );

        let mut measured = CheckpointMeter::default();
        assert_eq!(
            value.metered_continued_fraction_with_limit(3, &mut measured),
            Err(RationalArithmeticError::InvariantViolation(
                "continued-fraction coefficient bound exceeded"
            ))
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            value.metered_continued_fraction_with_limit(3, &mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.checkpoints, measured.checkpoints);
    }

    #[test]
    fn metered_expansion_cancels_at_every_observed_safe_point() {
        let value = BigRational::new(BigInt::from(-415), BigInt::from(93));
        let mut measured = CheckpointMeter::default();
        let expected = value
            .metered_continued_fraction(&mut measured)
            .expect("baseline expansion succeeds");
        assert_eq!(expected, value.continued_fraction());
        assert!(measured.checkpoints > 0);

        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                value.metered_continued_fraction(&mut cancelled),
                Err(RationalArithmeticError::Meter(MeterError::Cancelled)),
                "checkpoint {checkpoint} did not stop publication"
            );
            assert_eq!(cancelled.checkpoints, checkpoint);
        }
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
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
        );

        let late_checkpoint = measured.checkpoints.saturating_mul(3) / 4;
        let mut cancelled = CheckpointMeter::cancelling_at(late_checkpoint);
        assert_eq!(
            value.metered_continued_fraction(&mut cancelled),
            Err(RationalArithmeticError::Meter(MeterError::Cancelled))
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
        fn binary64_conversion_matches_exact_small_coefficient_division(
            numerator in -(1i64 << 52)..(1i64 << 52),
            denominator in 1u64..(1u64 << 52),
        ) {
            let rational = BigRational::new(
                BigInt::from(numerator),
                BigInt::from(denominator),
            );
            let expected = (numerator as f64) / (denominator as f64);
            let actual = rational.to_f64().expect("finite bounded ratio converts");
            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }

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
        fn governed_constructor_matches_independent_i128_normalization(
            numerator in any::<i64>(),
            denominator in any::<i64>().prop_filter("nonzero denominator", |value| *value != 0),
        ) {
            let original_numerator = BigInt::from(numerator);
            let original_denominator = BigInt::from(denominator);
            let actual = BigRational::metered_new(
                &original_numerator,
                &original_denominator,
                &mut Unbounded,
            ).expect("a nonzero denominator is admitted");
            let (expected_numerator, expected_denominator) =
                scalar_normalize(numerator, denominator);

            prop_assert_eq!(actual.numer(), &bigint_from_i128(expected_numerator));
            prop_assert_eq!(actual.denom(), &bigint_from_i128(expected_denominator));
            prop_assert_eq!(
                actual.clone(),
                BigRational::new(original_numerator.clone(), original_denominator.clone())
            );
            prop_assert!(actual.denom().is_positive());
            prop_assert_eq!(gcd(actual.numer(), actual.denom()), BigInt::one());
            prop_assert_eq!(
                actual.numer() * &original_denominator,
                actual.denom() * &original_numerator,
            );
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
