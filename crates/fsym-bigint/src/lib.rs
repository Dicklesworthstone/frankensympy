//! Owned arbitrary-precision integer arithmetic behind a strict containment
//! boundary (WS03 / architecture doc §5.3).
//!
//! `num-bigint` is an audited temporary substrate: its types and symbols appear ONLY
//! inside this crate and are NOT publicly re-exported or leaked. Higher layers depend
//! exclusively on [`BigInt`] and [`BigRational`] as defined here, so the substrate can be evolved
//! or replaced without touching term/domain schemas or API surfaces.
//!
//! # Strategy selection
//!
//! Multiplication offers three strategies with an explicit threshold:
//!
//! - [`Strategy::SchoolbookReference`] — repeated-addition scalar reference lane,
//!   O(log₂|min|) doublings; used below threshold for simple formal cross-checking.
//! - [`Strategy::Karatsuba`] — pure-Rust recursive divide-and-conquer multiplication ($O(n^{1.585})$).
//! - [`Strategy::NativeSubstrate`] — delegates to the contained substrate multiplication.
//!
//! [`select_strategy`] applies [`DEFAULT_STRATEGY_THRESHOLD_BITS`]; every
//! strategy pair is proptest-differential-tested across boundary thresholds.
//!
//! # Limb accounting and cooperative cancellation
//!
//! [`BigInt::limb_count`] exposes u64-limb height (`LIMB_BITS = 64`) so callers can charge
//! budget per unit of work. [`metered_multiply`] and [`metered_div_rem`] use deliberately
//! simple base-$2^{32}$ reference algorithms with safe points inside their limb loops.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use num_bigint::{BigInt as Substrate, BigUint, Sign};
use num_integer::Integer;
use num_traits::{FromPrimitive, Num, One, Pow, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign,
    Shl, Shr, Sub, SubAssign,
};
use std::str::FromStr;

/// Bits per u64 limb.
pub const LIMB_BITS: u64 = 64;

/// Magnitude size in u64 limbs (rounded up); zero for zero.
#[inline]
pub fn limb_count_u64(magnitude_bits: u64) -> u64 {
    magnitude_bits.div_ceil(LIMB_BITS)
}

/// Multiplication strategy for [`multiply_with_strategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Repeated-addition oracle reference lane.
    SchoolbookReference,
    /// Pure-Rust recursive Karatsuba multiplication.
    Karatsuba,
    /// Contained substrate multiplication.
    NativeSubstrate,
}

/// Bit-size at or above which multiplication uses [`Strategy::Karatsuba`].
pub const DEFAULT_STRATEGY_THRESHOLD_BITS: u64 = 256;

/// Pure strategy policy: visible and unit-testable on its own.
pub fn select_strategy(max_magnitude_bits: u64) -> Strategy {
    if max_magnitude_bits >= DEFAULT_STRATEGY_THRESHOLD_BITS {
        Strategy::Karatsuba
    } else {
        Strategy::SchoolbookReference
    }
}

/// Owned arbitrary-precision integer. The ONLY bigint type visible above this crate.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BigInt(Substrate);

/// Borrowed proof that a [`BigInt`] divisor is nonzero.
///
/// The field is private: callers can obtain this capability only through [`NonZeroBigInt::new`].
/// Algorithms admitted through this type cannot silently reinterpret division by zero as a
/// normal arithmetic result.
#[derive(Debug, Clone, Copy)]
pub struct NonZeroBigInt<'a>(&'a BigInt);

impl<'a> NonZeroBigInt<'a> {
    /// Returns a nonzero-divisor capability, or `None` for zero.
    pub fn new(value: &'a BigInt) -> Option<Self> {
        if value.is_zero() {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Returns the admitted nonzero integer.
    pub fn get(self) -> &'a BigInt {
        self.0
    }
}

/// Owned, canonical arbitrary-precision rational.
///
/// The provisional `num-rational` substrate is deliberately private so replacing it cannot
/// change higher-layer term or domain schemas.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BigRational(num_rational::Ratio<BigInt>);

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

impl BigInt {
    pub fn zero() -> Self {
        Self(Substrate::zero())
    }

    pub fn one() -> Self {
        Self(Substrate::one())
    }

    pub fn from_i64(v: i64) -> Self {
        Self(Substrate::from(v))
    }

    pub fn from_u64(v: u64) -> Self {
        Self(Substrate::from(v))
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn is_one(&self) -> bool {
        self.0.is_one()
    }

    pub fn is_positive(&self) -> bool {
        self.0.sign() == Sign::Plus
    }

    pub fn is_negative(&self) -> bool {
        self.0.sign() == Sign::Minus
    }

    pub fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    pub fn gcd(&self, other: &Self) -> Self {
        Self(self.0.gcd(&other.0))
    }

    pub fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (q, r) = self.0.div_rem(&other.0);
        (Self(q), Self(r))
    }

    pub fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    pub fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    pub fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }

    pub fn parse_bytes(buf: &[u8], radix: u32) -> Option<Self> {
        Substrate::parse_bytes(buf, radix).map(Self)
    }

    /// Magnitude size in bits (0 for zero).
    pub fn bits(&self) -> u64 {
        self.0.bits()
    }

    /// Height in u64 limbs — the charging unit for limb-operation budgets.
    pub fn limb_count(&self) -> u64 {
        limb_count_u64(self.bits())
    }

    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let res = self.0.extended_gcd(&other.0);
        (Self(res.gcd), Self(res.x), Self(res.y))
    }

    pub fn pow(&self, exp: u32) -> Self {
        Self(self.0.clone().pow(exp))
    }

    pub fn sqrt(&self) -> Self {
        Self(self.0.sqrt())
    }

    pub fn to_bytes_le(&self) -> Vec<u8> {
        self.0.magnitude().to_bytes_le()
    }

    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        Self(Substrate::from(BigUint::from_bytes_le(bytes)))
    }

    pub fn to_signed_bytes_be(&self) -> Vec<u8> {
        self.0.to_signed_bytes_be()
    }

    pub fn from_signed_bytes_be(bytes: &[u8]) -> Self {
        Self(Substrate::from_signed_bytes_be(bytes))
    }

    pub fn to_str_radix(&self, radix: u32) -> String {
        self.0.to_str_radix(radix)
    }

    pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, String> {
        Substrate::from_str_radix(src, radix)
            .map(Self)
            .map_err(|e| e.to_string())
    }
}

impl Zero for BigInt {
    fn zero() -> Self {
        Self(Substrate::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for BigInt {
    fn one() -> Self {
        Self(Substrate::one())
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl ToPrimitive for BigInt {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

impl FromPrimitive for BigInt {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(Substrate::from(n)))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(Substrate::from(n)))
    }
}

impl Signed for BigInt {
    fn abs(&self) -> Self {
        Self(self.0.abs())
    }
    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other {
            Self::zero()
        } else {
            self - other
        }
    }
    fn signum(&self) -> Self {
        Self(self.0.signum())
    }
    fn is_positive(&self) -> bool {
        self.0.sign() == Sign::Plus
    }
    fn is_negative(&self) -> bool {
        self.0.sign() == Sign::Minus
    }
}

impl Integer for BigInt {
    fn div_floor(&self, other: &Self) -> Self {
        Self(self.0.div_floor(&other.0))
    }
    fn mod_floor(&self, other: &Self) -> Self {
        Self(self.0.mod_floor(&other.0))
    }
    fn gcd(&self, other: &Self) -> Self {
        Self(self.0.gcd(&other.0))
    }
    fn lcm(&self, other: &Self) -> Self {
        Self(self.0.lcm(&other.0))
    }
    fn is_multiple_of(&self, other: &Self) -> bool {
        self.0.is_multiple_of(&other.0)
    }
    fn is_even(&self) -> bool {
        self.0.is_even()
    }
    fn is_odd(&self) -> bool {
        self.0.is_odd()
    }
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (q, r) = self.0.div_rem(&other.0);
        (Self(q), Self(r))
    }
}

impl Num for BigInt {
    type FromStrRadixErr = String;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Substrate::from_str_radix(str, radix)
            .map(Self)
            .map_err(|e| e.to_string())
    }
}

impl Pow<u32> for BigInt {
    type Output = BigInt;
    fn pow(self, exp: u32) -> BigInt {
        BigInt(self.0.pow(exp))
    }
}

impl Pow<u32> for &BigInt {
    type Output = BigInt;
    fn pow(self, exp: u32) -> BigInt {
        BigInt(self.0.clone().pow(exp))
    }
}

impl fmt::Debug for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigInt({})", self.0)
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for BigInt {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Substrate::from_str(s).map(Self).map_err(|e| e.to_string())
    }
}

impl Serialize for BigInt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BigInt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Substrate::deserialize(deserializer).map(Self)
    }
}

impl From<i64> for BigInt {
    fn from(v: i64) -> Self {
        Self(Substrate::from(v))
    }
}

impl From<u64> for BigInt {
    fn from(v: u64) -> Self {
        Self(Substrate::from(v))
    }
}

impl From<i32> for BigInt {
    fn from(v: i32) -> Self {
        Self(Substrate::from(v))
    }
}

impl From<u32> for BigInt {
    fn from(v: u32) -> Self {
        Self(Substrate::from(v))
    }
}

impl TryFrom<BigInt> for usize {
    type Error = String;
    fn try_from(value: BigInt) -> Result<Self, Self::Error> {
        value
            .0
            .to_usize()
            .ok_or_else(|| "BigInt out of usize range".to_string())
    }
}

impl TryFrom<&BigInt> for usize {
    type Error = String;
    fn try_from(value: &BigInt) -> Result<Self, Self::Error> {
        value
            .0
            .to_usize()
            .ok_or_else(|| "BigInt out of usize range".to_string())
    }
}

impl TryFrom<BigInt> for i64 {
    type Error = String;
    fn try_from(value: BigInt) -> Result<Self, Self::Error> {
        value
            .0
            .to_i64()
            .ok_or_else(|| "BigInt out of i64 range".to_string())
    }
}

impl TryFrom<&BigInt> for i64 {
    type Error = String;
    fn try_from(value: &BigInt) -> Result<Self, Self::Error> {
        value
            .0
            .to_i64()
            .ok_or_else(|| "BigInt out of i64 range".to_string())
    }
}

impl TryFrom<BigInt> for u64 {
    type Error = String;
    fn try_from(value: BigInt) -> Result<Self, Self::Error> {
        value
            .0
            .to_u64()
            .ok_or_else(|| "BigInt out of u64 range".to_string())
    }
}

impl TryFrom<&BigInt> for u64 {
    type Error = String;
    fn try_from(value: &BigInt) -> Result<Self, Self::Error> {
        value
            .0
            .to_u64()
            .ok_or_else(|| "BigInt out of u64 range".to_string())
    }
}

// ---------------- Add ----------------
impl Add for BigInt {
    type Output = BigInt;
    fn add(self, other: BigInt) -> BigInt {
        BigInt(self.0 + other.0)
    }
}

impl Add<&BigInt> for BigInt {
    type Output = BigInt;
    fn add(self, other: &BigInt) -> BigInt {
        BigInt(self.0 + &other.0)
    }
}

impl Add<BigInt> for &BigInt {
    type Output = BigInt;
    fn add(self, other: BigInt) -> BigInt {
        BigInt(&self.0 + other.0)
    }
}

impl Add<&BigInt> for &BigInt {
    type Output = BigInt;
    fn add(self, other: &BigInt) -> BigInt {
        BigInt(&self.0 + &other.0)
    }
}

impl Add<i64> for BigInt {
    type Output = BigInt;
    fn add(self, other: i64) -> BigInt {
        BigInt(self.0 + Substrate::from(other))
    }
}

impl Add<i64> for &BigInt {
    type Output = BigInt;
    fn add(self, other: i64) -> BigInt {
        BigInt(&self.0 + Substrate::from(other))
    }
}

impl AddAssign for BigInt {
    fn add_assign(&mut self, rhs: BigInt) {
        self.0 += rhs.0;
    }
}

impl AddAssign<&BigInt> for BigInt {
    fn add_assign(&mut self, rhs: &BigInt) {
        self.0 += &rhs.0;
    }
}

impl AddAssign<i64> for BigInt {
    fn add_assign(&mut self, rhs: i64) {
        self.0 += Substrate::from(rhs);
    }
}

// ---------------- Sub ----------------
impl Sub for BigInt {
    type Output = BigInt;
    fn sub(self, other: BigInt) -> BigInt {
        BigInt(self.0 - other.0)
    }
}

impl Sub<&BigInt> for BigInt {
    type Output = BigInt;
    fn sub(self, other: &BigInt) -> BigInt {
        BigInt(self.0 - &other.0)
    }
}

impl Sub<BigInt> for &BigInt {
    type Output = BigInt;
    fn sub(self, other: BigInt) -> BigInt {
        BigInt(&self.0 - other.0)
    }
}

impl Sub<&BigInt> for &BigInt {
    type Output = BigInt;
    fn sub(self, other: &BigInt) -> BigInt {
        BigInt(&self.0 - &other.0)
    }
}

impl Sub<i64> for BigInt {
    type Output = BigInt;
    fn sub(self, other: i64) -> BigInt {
        BigInt(self.0 - Substrate::from(other))
    }
}

impl Sub<i64> for &BigInt {
    type Output = BigInt;
    fn sub(self, other: i64) -> BigInt {
        BigInt(&self.0 - Substrate::from(other))
    }
}

impl SubAssign for BigInt {
    fn sub_assign(&mut self, rhs: BigInt) {
        self.0 -= rhs.0;
    }
}

impl SubAssign<&BigInt> for BigInt {
    fn sub_assign(&mut self, rhs: &BigInt) {
        self.0 -= &rhs.0;
    }
}

impl SubAssign<i64> for BigInt {
    fn sub_assign(&mut self, rhs: i64) {
        self.0 -= Substrate::from(rhs);
    }
}

// ---------------- Mul ----------------
impl Mul for BigInt {
    type Output = BigInt;
    fn mul(self, other: BigInt) -> BigInt {
        multiply(&self, &other)
    }
}

impl Mul<&BigInt> for BigInt {
    type Output = BigInt;
    fn mul(self, other: &BigInt) -> BigInt {
        multiply(&self, other)
    }
}

impl Mul<BigInt> for &BigInt {
    type Output = BigInt;
    fn mul(self, other: BigInt) -> BigInt {
        multiply(self, &other)
    }
}

impl Mul<&BigInt> for &BigInt {
    type Output = BigInt;
    fn mul(self, other: &BigInt) -> BigInt {
        multiply(self, other)
    }
}

impl Mul<i64> for BigInt {
    type Output = BigInt;
    fn mul(self, other: i64) -> BigInt {
        multiply(&self, &BigInt::from(other))
    }
}

impl Mul<i64> for &BigInt {
    type Output = BigInt;
    fn mul(self, other: i64) -> BigInt {
        multiply(self, &BigInt::from(other))
    }
}

impl MulAssign for BigInt {
    fn mul_assign(&mut self, rhs: BigInt) {
        *self = multiply(self, &rhs);
    }
}

impl MulAssign<&BigInt> for BigInt {
    fn mul_assign(&mut self, rhs: &BigInt) {
        *self = multiply(self, rhs);
    }
}

impl MulAssign<i64> for BigInt {
    fn mul_assign(&mut self, rhs: i64) {
        *self = multiply(self, &BigInt::from(rhs));
    }
}

// ---------------- Div ----------------
impl Div for BigInt {
    type Output = BigInt;
    fn div(self, other: BigInt) -> BigInt {
        BigInt(self.0 / other.0)
    }
}

impl Div<&BigInt> for BigInt {
    type Output = BigInt;
    fn div(self, other: &BigInt) -> BigInt {
        BigInt(self.0 / &other.0)
    }
}

impl Div<BigInt> for &BigInt {
    type Output = BigInt;
    fn div(self, other: BigInt) -> BigInt {
        BigInt(&self.0 / other.0)
    }
}

impl Div<&BigInt> for &BigInt {
    type Output = BigInt;
    fn div(self, other: &BigInt) -> BigInt {
        BigInt(&self.0 / &other.0)
    }
}

impl Div<i64> for BigInt {
    type Output = BigInt;
    fn div(self, other: i64) -> BigInt {
        BigInt(self.0 / Substrate::from(other))
    }
}

impl Div<i64> for &BigInt {
    type Output = BigInt;
    fn div(self, other: i64) -> BigInt {
        BigInt(&self.0 / Substrate::from(other))
    }
}

impl DivAssign for BigInt {
    fn div_assign(&mut self, rhs: BigInt) {
        self.0 /= rhs.0;
    }
}

impl DivAssign<&BigInt> for BigInt {
    fn div_assign(&mut self, rhs: &BigInt) {
        self.0 /= &rhs.0;
    }
}

impl DivAssign<i64> for BigInt {
    fn div_assign(&mut self, rhs: i64) {
        self.0 /= Substrate::from(rhs);
    }
}

// ---------------- Rem ----------------
impl Rem for BigInt {
    type Output = BigInt;
    fn rem(self, other: BigInt) -> BigInt {
        BigInt(self.0 % other.0)
    }
}

impl Rem<&BigInt> for BigInt {
    type Output = BigInt;
    fn rem(self, other: &BigInt) -> BigInt {
        BigInt(self.0 % &other.0)
    }
}

impl Rem<BigInt> for &BigInt {
    type Output = BigInt;
    fn rem(self, other: BigInt) -> BigInt {
        BigInt(&self.0 % other.0)
    }
}

impl Rem<&BigInt> for &BigInt {
    type Output = BigInt;
    fn rem(self, other: &BigInt) -> BigInt {
        BigInt(&self.0 % &other.0)
    }
}

impl Rem<i64> for BigInt {
    type Output = BigInt;
    fn rem(self, other: i64) -> BigInt {
        BigInt(self.0 % Substrate::from(other))
    }
}

impl Rem<i64> for &BigInt {
    type Output = BigInt;
    fn rem(self, other: i64) -> BigInt {
        BigInt(&self.0 % Substrate::from(other))
    }
}

impl RemAssign for BigInt {
    fn rem_assign(&mut self, rhs: BigInt) {
        self.0 %= rhs.0;
    }
}

impl RemAssign<&BigInt> for BigInt {
    fn rem_assign(&mut self, rhs: &BigInt) {
        self.0 %= &rhs.0;
    }
}

impl RemAssign<i64> for BigInt {
    fn rem_assign(&mut self, rhs: i64) {
        self.0 %= Substrate::from(rhs);
    }
}

// ---------------- Neg ----------------
impl Neg for BigInt {
    type Output = BigInt;
    fn neg(self) -> BigInt {
        BigInt(-self.0)
    }
}

impl Neg for &BigInt {
    type Output = BigInt;
    fn neg(self) -> BigInt {
        BigInt(-&self.0)
    }
}

// ---------------- Bitwise / Shifts ----------------
impl Shl<u32> for BigInt {
    type Output = BigInt;
    fn shl(self, rhs: u32) -> BigInt {
        BigInt(self.0 << rhs)
    }
}

impl Shl<u32> for &BigInt {
    type Output = BigInt;
    fn shl(self, rhs: u32) -> BigInt {
        BigInt(&self.0 << rhs)
    }
}

impl Shr<u32> for BigInt {
    type Output = BigInt;
    fn shr(self, rhs: u32) -> BigInt {
        BigInt(self.0 >> rhs)
    }
}

impl Shr<u32> for &BigInt {
    type Output = BigInt;
    fn shr(self, rhs: u32) -> BigInt {
        BigInt(&self.0 >> rhs)
    }
}

impl BitAnd for &BigInt {
    type Output = BigInt;
    fn bitand(self, rhs: Self) -> BigInt {
        BigInt(&self.0 & &rhs.0)
    }
}

impl BitOr for &BigInt {
    type Output = BigInt;
    fn bitor(self, rhs: Self) -> BigInt {
        BigInt(&self.0 | &rhs.0)
    }
}

impl BitXor for &BigInt {
    type Output = BigInt;
    fn bitxor(self, rhs: Self) -> BigInt {
        BigInt(&self.0 ^ &rhs.0)
    }
}

impl PartialEq<i64> for BigInt {
    fn eq(&self, other: &i64) -> bool {
        self.0 == Substrate::from(*other)
    }
}

impl PartialOrd<i64> for BigInt {
    fn partial_cmp(&self, other: &i64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&Substrate::from(*other))
    }
}

/// Multiplies via the explicitly chosen strategy.
pub fn multiply_with_strategy(a: &BigInt, b: &BigInt, strategy: Strategy) -> BigInt {
    match strategy {
        Strategy::SchoolbookReference => schoolbook_reference(&a.0, &b.0),
        Strategy::Karatsuba => karatsuba_mul_internal(&a.0, &b.0),
        Strategy::NativeSubstrate => BigInt(a.0.clone() * b.0.clone()),
    }
}

/// Applies [`select_strategy`] over the operands' larger bit height, then multiplies.
pub fn multiply(a: &BigInt, b: &BigInt) -> BigInt {
    let strategy = select_strategy(std::cmp::max(a.bits(), b.bits()));
    multiply_with_strategy(a, b, strategy)
}

/// Metered multiplication with safe points inside the limb-product loop.
///
/// This cancellation-first lane deliberately uses a simple base-$2^{32}$ reference algorithm.
/// Each input-copy and limb-product unit is charged and preceded by a checkpoint, so cancellation
/// latency does not depend on an opaque substrate multiplication call.
pub fn metered_multiply<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;

    if a.is_zero() || b.is_zero() {
        return Ok(BigInt::zero());
    }

    let a_len = a.0.iter_u32_digits().len();
    let b_len = b.0.iter_u32_digits().len();
    let output_len = a_len.saturating_add(b_len);
    let transient_digits = a_len.saturating_add(b_len).saturating_add(output_len);
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            u64::try_from(transient_digits)
                .unwrap_or(u64::MAX)
                .saturating_mul(4),
        ),
        (Dimension::AllocationCount, 3),
    ])?;

    let mut a_digits = Vec::with_capacity(a_len);
    for digit in a.0.iter_u32_digits() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        a_digits.push(digit);
    }

    let mut b_digits = Vec::with_capacity(b_len);
    for digit in b.0.iter_u32_digits() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        b_digits.push(digit);
    }

    let mut product = vec![0u32; output_len];
    for (i, &a_digit) in a_digits.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &b_digit) in b_digits.iter().enumerate() {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;

            let index = i + j;
            let value = u64::from(product[index]) + u64::from(a_digit) * u64::from(b_digit) + carry;
            product[index] = value as u32;
            carry = value >> 32;
        }

        let mut index = i + b_digits.len();
        while carry != 0 {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;

            let value = u64::from(product[index]) + carry;
            product[index] = value as u32;
            carry = value >> 32;
            index += 1;
        }
    }

    meter.checkpoint()?;
    while product.last() == Some(&0) {
        product.pop();
    }
    let magnitude = BigUint::new(product);
    let sign = if a.0.sign() == b.0.sign() {
        Sign::Plus
    } else {
        Sign::Minus
    };
    Ok(BigInt(Substrate::from_biguint(sign, magnitude)))
}

/// Cancellation-first truncating division with remainder.
///
/// Returns `Ok(None)` when `divisor` is zero. Otherwise the result agrees with Rust/`num-bigint`
/// truncation semantics: the quotient truncates toward zero, the remainder has the dividend's
/// sign, and `dividend == quotient * divisor + remainder`.
///
/// This is a scalar reference lane, not the ordinary performance path. It copies magnitudes into
/// owned base-$2^{32}$ digits, then performs binary long division. Every digit copy, remainder
/// shift, comparison digit, and subtraction digit is charged and preceded by a cancellation
/// checkpoint. All four transient digit buffers are charged before allocation.
pub fn metered_div_rem<M: BudgetMeter>(
    dividend: &BigInt,
    divisor: &BigInt,
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    let Some(divisor) = NonZeroBigInt::new(divisor) else {
        return Ok(None);
    };
    metered_div_rem_nonzero(dividend, divisor, meter).map(Some)
}

/// Cancellation-first truncating division after typed nonzero-divisor admission.
pub fn metered_div_rem_nonzero<M: BudgetMeter>(
    dividend: &BigInt,
    divisor: NonZeroBigInt<'_>,
    meter: &mut M,
) -> Result<(BigInt, BigInt), MeterError> {
    meter.checkpoint()?;
    let divisor = divisor.get();
    if dividend.is_zero() {
        return Ok((BigInt::zero(), BigInt::zero()));
    }
    if dividend.bits() < divisor.bits() {
        meter.charge_batch(&[
            (
                Dimension::MemoryBytes,
                dividend.limb_count().saturating_mul(8),
            ),
            (Dimension::AllocationCount, 1),
        ])?;
        meter.checkpoint()?;
        return Ok((BigInt::zero(), dividend.clone()));
    }

    let dividend_len = dividend.0.iter_u32_digits().len();
    let divisor_len = divisor.0.iter_u32_digits().len();
    let remainder_capacity = divisor_len.min(dividend_len).saturating_add(1);
    let transient_digits = dividend_len
        .saturating_mul(2)
        .saturating_add(divisor_len)
        .saturating_add(remainder_capacity);
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            u64::try_from(transient_digits)
                .unwrap_or(u64::MAX)
                .saturating_mul(4),
        ),
        (Dimension::AllocationCount, 4),
    ])?;

    let dividend_digits = copy_u32_digits(&dividend.0, dividend_len, meter)?;
    let divisor_digits = copy_u32_digits(&divisor.0, divisor_len, meter)?;
    let mut quotient_digits = vec![0u32; dividend_len];
    let mut remainder_digits = Vec::with_capacity(remainder_capacity);

    for (&dividend_digit, quotient_digit) in
        dividend_digits.iter().zip(quotient_digits.iter_mut()).rev()
    {
        for bit in (0..32).rev() {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            shift_left_one(&mut remainder_digits, meter)?;
            if dividend_digit & (1u32 << bit) != 0 {
                if remainder_digits.is_empty() {
                    remainder_digits.push(1);
                } else {
                    remainder_digits[0] |= 1;
                }
            }

            if compare_digits(&remainder_digits, &divisor_digits, meter)?
                != std::cmp::Ordering::Less
            {
                subtract_digits(&mut remainder_digits, &divisor_digits, meter)?;
                *quotient_digit |= 1u32 << bit;
            }
        }
    }

    meter.checkpoint()?;
    trim_digits(&mut quotient_digits, meter)?;
    trim_digits(&mut remainder_digits, meter)?;
    let quotient_magnitude = BigUint::new(quotient_digits);
    let remainder_magnitude = BigUint::new(remainder_digits);
    let quotient_sign = if quotient_magnitude.is_zero() {
        Sign::NoSign
    } else if dividend.0.sign() == divisor.0.sign() {
        Sign::Plus
    } else {
        Sign::Minus
    };
    let remainder_sign = if remainder_magnitude.is_zero() {
        Sign::NoSign
    } else {
        dividend.0.sign()
    };
    Ok((
        BigInt(Substrate::from_biguint(quotient_sign, quotient_magnitude)),
        BigInt(Substrate::from_biguint(remainder_sign, remainder_magnitude)),
    ))
}

fn copy_u32_digits<M: BudgetMeter>(
    value: &Substrate,
    capacity: usize,
    meter: &mut M,
) -> Result<Vec<u32>, MeterError> {
    let mut digits = Vec::with_capacity(capacity);
    for digit in value.iter_u32_digits() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        digits.push(digit);
    }
    Ok(digits)
}

fn shift_left_one<M: BudgetMeter>(digits: &mut Vec<u32>, meter: &mut M) -> Result<(), MeterError> {
    let mut carry = 0u32;
    for digit in digits.iter_mut() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let next_carry = *digit >> 31;
        *digit = (*digit << 1) | carry;
        carry = next_carry;
    }
    if carry != 0 {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        digits.push(carry);
    }
    Ok(())
}

fn compare_digits<M: BudgetMeter>(
    lhs: &[u32],
    rhs: &[u32],
    meter: &mut M,
) -> Result<std::cmp::Ordering, MeterError> {
    match lhs.len().cmp(&rhs.len()) {
        std::cmp::Ordering::Equal => {}
        ordering => return Ok(ordering),
    }
    for (lhs_digit, rhs_digit) in lhs.iter().rev().zip(rhs.iter().rev()) {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        match lhs_digit.cmp(rhs_digit) {
            std::cmp::Ordering::Equal => {}
            ordering => return Ok(ordering),
        }
    }
    Ok(std::cmp::Ordering::Equal)
}

fn subtract_digits<M: BudgetMeter>(
    lhs: &mut Vec<u32>,
    rhs: &[u32],
    meter: &mut M,
) -> Result<(), MeterError> {
    let mut borrow = 0u64;
    for (index, lhs_digit) in lhs.iter_mut().enumerate() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let rhs_digit = u64::from(rhs.get(index).copied().unwrap_or(0));
        let subtrahend = rhs_digit + borrow;
        let lhs_value = u64::from(*lhs_digit);
        if lhs_value >= subtrahend {
            *lhs_digit = u32::try_from(lhs_value - subtrahend).unwrap_or(0);
            borrow = 0;
        } else {
            *lhs_digit = u32::try_from((1u64 << 32) + lhs_value - subtrahend).unwrap_or(0);
            borrow = 1;
        }
    }
    debug_assert_eq!(borrow, 0, "subtraction caller must prove lhs >= rhs");
    trim_digits(lhs, meter)?;
    Ok(())
}

fn trim_digits<M: BudgetMeter>(digits: &mut Vec<u32>, meter: &mut M) -> Result<(), MeterError> {
    while digits.last() == Some(&0) {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        digits.pop();
    }
    Ok(())
}

fn schoolbook_reference(a: &Substrate, b: &Substrate) -> BigInt {
    let neg = a.sign() != b.sign();
    let steps = a.magnitude().min(b.magnitude());
    let unit: Substrate = if a.magnitude() <= b.magnitude() {
        Substrate::from(b.magnitude().clone())
    } else {
        Substrate::from(a.magnitude().clone())
    };

    let mut acc = Substrate::zero();
    let mut power = unit;
    let mut bits = steps.clone();

    while !bits.is_zero() {
        if &bits % 2u32 != BigUint::zero() {
            acc += &power;
        }
        power = &power + &power;
        bits /= 2u32;
    }

    if neg && !acc.is_zero() {
        BigInt(-acc)
    } else {
        BigInt(acc)
    }
}

fn karatsuba_mul_internal(a: &Substrate, b: &Substrate) -> BigInt {
    let is_neg = a.sign() != b.sign();
    let res_mag = karatsuba_mag_internal(a.magnitude(), b.magnitude());
    if is_neg && !res_mag.is_zero() {
        BigInt(-Substrate::from(res_mag))
    } else {
        BigInt(Substrate::from(res_mag))
    }
}

fn karatsuba_mag_internal(a: &BigUint, b: &BigUint) -> BigUint {
    let max_bits = std::cmp::max(a.bits(), b.bits());
    if max_bits <= 128 {
        return a * b;
    }
    let m = max_bits / 2;
    let mask = (BigUint::one() << m) - 1u32;
    let a0 = a & &mask;
    let a1 = a >> m;
    let b0 = b & &mask;
    let b1 = b >> m;

    let z0 = karatsuba_mag_internal(&a0, &b0);
    let z2 = karatsuba_mag_internal(&a1, &b1);
    let sum_a = &a0 + &a1;
    let sum_b = &b0 + &b1;
    let z1 = karatsuba_mag_internal(&sum_a, &sum_b) - &z0 - &z2;

    (z2 << (2 * m)) + (z1 << m) + z0
}

#[cfg(test)]
mod tests {
    use super::Strategy;
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, Unbounded};
    use proptest::prelude::*;

    #[derive(Debug)]
    struct CancelAfter {
        cancel_at_checkpoint: usize,
        checkpoints: usize,
        compute_steps: u64,
    }

    impl BudgetMeter for CancelAfter {
        fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
            if dimension == Dimension::ComputeSteps {
                self.compute_steps = self.compute_steps.saturating_add(amount);
            }
            Ok(())
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            for &(dimension, amount) in charges {
                if dimension == Dimension::ComputeSteps {
                    self.compute_steps = self.compute_steps.saturating_add(amount);
                }
            }
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints += 1;
            if self.checkpoints >= self.cancel_at_checkpoint {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn select_strategy_switches_at_the_documented_threshold() {
        assert_eq!(
            select_strategy(DEFAULT_STRATEGY_THRESHOLD_BITS - 1),
            Strategy::SchoolbookReference
        );
        assert_eq!(
            select_strategy(DEFAULT_STRATEGY_THRESHOLD_BITS),
            Strategy::Karatsuba
        );
        assert_eq!(
            select_strategy(DEFAULT_STRATEGY_THRESHOLD_BITS + 1),
            Strategy::Karatsuba
        );
    }

    #[test]
    fn zero_identities_hold_for_all_strategies() {
        let x = BigInt::from(123456789i64);
        let zero = BigInt::zero();
        for strategy in [
            Strategy::SchoolbookReference,
            Strategy::Karatsuba,
            Strategy::NativeSubstrate,
        ] {
            assert_eq!(multiply_with_strategy(&x, &zero, strategy), zero);
            assert_eq!(multiply_with_strategy(&zero, &x, strategy), zero);
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
    fn metered_multiplication_cancels_inside_limb_products() {
        let a = (BigInt::one() << 1_024) + 123_456_789i64;
        let b = (BigInt::one() << 1_024) + 987_654_321i64;
        let mut meter = CancelAfter {
            cancel_at_checkpoint: 80,
            checkpoints: 0,
            compute_steps: 0,
        };

        assert_eq!(
            metered_multiply(&a, &b, &mut meter),
            Err(MeterError::Cancelled)
        );
        assert!(
            meter.compute_steps > 64,
            "input limbs must have been copied"
        );
        assert!(
            meter.compute_steps < 32 * 32 + 64,
            "cancellation must stop before the full product"
        );
    }

    #[test]
    fn metered_balanced_multiplication_matches_native_lane() {
        let a = (BigInt::one() << 2_047) + (BigInt::one() << 1_023) + 17i64;
        let b = -((BigInt::one() << 2_031) + (BigInt::one() << 997) + 29i64);
        let mut meter = Unbounded;
        let metered = metered_multiply(&a, &b, &mut meter).unwrap();
        let native = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
        assert_eq!(metered, native);
    }

    #[test]
    fn refused_arithmetic_batch_does_not_burn_memory_allowance() {
        let mut limits = BudgetLimits::uniform(1_000_000, 0);
        limits.dimensions[Dimension::AllocationCount.index()] = 2;
        let mut budget = Budget::new(limits);
        let before = budget.snapshot();
        let a = (BigInt::one() << 1_024) + 17i64;
        let b = (BigInt::one() << 1_024) + 29i64;

        assert_eq!(
            metered_multiply(&a, &b, &mut budget),
            Err(MeterError::Budget(fsym_budget::BudgetError::Exhausted {
                dimension: Dimension::AllocationCount,
                requested: 3,
                remaining: 2,
            }))
        );
        assert_eq!(budget.snapshot(), before);
    }

    #[test]
    fn metered_division_preserves_truncating_sign_rules() {
        let mut meter = Unbounded;
        for (dividend, divisor, quotient, remainder) in [
            (17, 5, 3, 2),
            (-17, 5, -3, -2),
            (17, -5, -3, 2),
            (-17, -5, 3, -2),
            (3, 8, 0, 3),
            (0, 8, 0, 0),
        ] {
            assert_eq!(
                metered_div_rem(&dividend.into(), &divisor.into(), &mut meter),
                Ok(Some((quotient.into(), remainder.into())))
            );
        }
        assert_eq!(
            metered_div_rem(&17.into(), &BigInt::zero(), &mut meter),
            Ok(None)
        );
    }

    #[test]
    fn metered_division_cancels_inside_digit_batches() {
        let dividend = (BigInt::one() << 4_096) - 1i64;
        let divisor = (BigInt::one() << 2_047) + 65_537i64;
        let mut baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        let completed = metered_div_rem(&dividend, &divisor, &mut baseline)
            .unwrap()
            .unwrap();
        assert_eq!(&completed.0 * &divisor + &completed.1, dividend);

        let cancel_at_checkpoint = baseline.checkpoints * 3 / 4;
        let mut meter = CancelAfter {
            cancel_at_checkpoint,
            checkpoints: 0,
            compute_steps: 0,
        };

        assert_eq!(
            metered_div_rem(&dividend, &divisor, &mut meter),
            Err(MeterError::Cancelled)
        );
        assert_eq!(meter.checkpoints, cancel_at_checkpoint);
        assert!(meter.compute_steps > baseline.compute_steps / 2);
        assert!(meter.compute_steps < baseline.compute_steps);
    }

    proptest! {
        #[test]
        fn strategies_agree_across_the_threshold_boundary(
            shift in 254u32..258u32,
            sign_a in proptest::bool::ANY,
            sign_b in proptest::bool::ANY,
        ) {
            let base = BigInt::one() << shift;
            for delta in [-1i64, 0, 1] {
                let a = match sign_a {
                    true => &base + delta,
                    false => -(&base + delta),
                };
                for b_raw in [1i64, 2, 3, 5, 255, 65537] {
                    let b = match sign_b {
                        true => BigInt::from(b_raw),
                        false => BigInt::from(-b_raw),
                    };
                    let ref_res = multiply_with_strategy(&a, &b, Strategy::SchoolbookReference);
                    let kar_res = multiply_with_strategy(&a, &b, Strategy::Karatsuba);
                    let nat_res = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
                    prop_assert_eq!(&ref_res, &kar_res);
                    prop_assert_eq!(&ref_res, &nat_res);
                }
            }
        }

        #[test]
        fn all_lanes_agree_for_broad_balanced_operands(
            a_bytes in proptest::collection::vec(any::<u8>(), 0..129),
            b_bytes in proptest::collection::vec(any::<u8>(), 0..129),
        ) {
            let a = BigInt::from_signed_bytes_be(&a_bytes);
            let b = BigInt::from_signed_bytes_be(&b_bytes);
            let reference = multiply_with_strategy(&a, &b, Strategy::SchoolbookReference);
            let karatsuba = multiply_with_strategy(&a, &b, Strategy::Karatsuba);
            let native = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
            let mut meter = Unbounded;
            let metered = metered_multiply(&a, &b, &mut meter).unwrap();
            prop_assert_eq!(&reference, &karatsuba);
            prop_assert_eq!(&reference, &native);
            prop_assert_eq!(&reference, &metered);
            prop_assert_eq!(&a * &b, native);
        }


        #[test]
        fn metered_division_matches_native_lane_for_broad_signed_operands(
            dividend_bytes in proptest::collection::vec(any::<u8>(), 0..97),
            divisor_bytes in proptest::collection::vec(any::<u8>(), 0..65),
        ) {
            let dividend = BigInt::from_signed_bytes_be(&dividend_bytes);
            let mut divisor = BigInt::from_signed_bytes_be(&divisor_bytes);
            if divisor.is_zero() {
                divisor = BigInt::one();
            }
            let mut meter = Unbounded;
            let (quotient, remainder) = metered_div_rem(&dividend, &divisor, &mut meter)
                .unwrap()
                .unwrap();
            let (native_quotient, native_remainder) = dividend.div_rem(&divisor);
            prop_assert_eq!(&quotient, &native_quotient);
            prop_assert_eq!(&remainder, &native_remainder);
            prop_assert_eq!(&quotient * &divisor + &remainder, dividend);
        }
    }
}
