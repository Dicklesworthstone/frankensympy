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
//! budget per unit of work. [`metered_multiply`] checks safe points and charges budget.

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

/// Exact rational number parameterized over [`BigInt`].
pub type BigRational = num_rational::Ratio<BigInt>;

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
        BigInt(self.0 * other.0)
    }
}

impl Mul<&BigInt> for BigInt {
    type Output = BigInt;
    fn mul(self, other: &BigInt) -> BigInt {
        BigInt(self.0 * &other.0)
    }
}

impl Mul<BigInt> for &BigInt {
    type Output = BigInt;
    fn mul(self, other: BigInt) -> BigInt {
        BigInt(&self.0 * other.0)
    }
}

impl Mul<&BigInt> for &BigInt {
    type Output = BigInt;
    fn mul(self, other: &BigInt) -> BigInt {
        BigInt(&self.0 * &other.0)
    }
}

impl Mul<i64> for BigInt {
    type Output = BigInt;
    fn mul(self, other: i64) -> BigInt {
        BigInt(self.0 * Substrate::from(other))
    }
}

impl Mul<i64> for &BigInt {
    type Output = BigInt;
    fn mul(self, other: i64) -> BigInt {
        BigInt(&self.0 * Substrate::from(other))
    }
}

impl MulAssign for BigInt {
    fn mul_assign(&mut self, rhs: BigInt) {
        self.0 *= rhs.0;
    }
}

impl MulAssign<&BigInt> for BigInt {
    fn mul_assign(&mut self, rhs: &BigInt) {
        self.0 *= &rhs.0;
    }
}

impl MulAssign<i64> for BigInt {
    fn mul_assign(&mut self, rhs: i64) {
        self.0 *= Substrate::from(rhs);
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

/// Metered multiplication with safe-point checkpoints and resource charging.
pub fn metered_multiply<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    let a_limbs = a.limb_count().max(1);
    let b_limbs = b.limb_count().max(1);
    meter.charge(Dimension::ComputeSteps, a_limbs.saturating_mul(b_limbs))?;
    meter.charge(
        Dimension::MemoryBytes,
        (a_limbs + b_limbs).saturating_mul(8),
    )?;
    meter.charge(Dimension::AllocationCount, 1)?;
    meter.checkpoint()?;
    Ok(multiply(a, b))
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
    use proptest::prelude::*;

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
    }
}
