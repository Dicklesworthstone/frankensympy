//! Canonical arbitrary-precision rational arithmetic (WS03 / architecture doc §5.4).
//!
//! The provisional `num-rational` substrate is private to this crate. Higher layers use only
//! [`BigRational`] and [`fsym_bigint::BigInt`], keeping rational representation and normalization
//! policy independently replaceable from the integer substrate.
//!
//! This initial boundary owns the canonical rational value type and its scalar operations.
//! Rational reconstruction remains in the modular arithmetic lane until `fsym-modular` is split;
//! this crate does not yet claim every deliverable in architecture §5.4.

#![forbid(unsafe_code)]

use fsym_bigint::BigInt;
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
    ($trait:ident, $method:ident, $operator:tt) => {
        impl $trait for BigRational {
            type Output = BigRational;

            fn $method(self, rhs: BigRational) -> Self::Output {
                BigRational(self.0 $operator rhs.0)
            }
        }

        impl $trait<&BigRational> for BigRational {
            type Output = BigRational;

            fn $method(self, rhs: &BigRational) -> Self::Output {
                BigRational(self.0 $operator &rhs.0)
            }
        }

        impl $trait<BigRational> for &BigRational {
            type Output = BigRational;

            fn $method(self, rhs: BigRational) -> Self::Output {
                BigRational(&self.0 $operator rhs.0)
            }
        }

        impl $trait<&BigRational> for &BigRational {
            type Output = BigRational;

            fn $method(self, rhs: &BigRational) -> Self::Output {
                BigRational(&self.0 $operator &rhs.0)
            }
        }
    };
}

impl_rational_binary_op!(Add, add, +);
impl_rational_binary_op!(Sub, sub, -);
impl_rational_binary_op!(Mul, mul, *);
impl_rational_binary_op!(Div, div, /);
impl_rational_binary_op!(Rem, rem, %);

macro_rules! impl_rational_assign_op {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl $trait for BigRational {
            fn $method(&mut self, rhs: BigRational) {
                self.0 $operator rhs.0;
            }
        }

        impl $trait<&BigRational> for BigRational {
            fn $method(&mut self, rhs: &BigRational) {
                self.0 $operator &rhs.0;
            }
        }
    };
}

impl_rational_assign_op!(AddAssign, add_assign, +=);
impl_rational_assign_op!(SubAssign, sub_assign, -=);
impl_rational_assign_op!(MulAssign, mul_assign, *=);
impl_rational_assign_op!(DivAssign, div_assign, /=);
impl_rational_assign_op!(RemAssign, rem_assign, %=);

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
    use proptest::prelude::*;

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

    proptest! {
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
    }
}
