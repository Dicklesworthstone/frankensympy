//! Owned arbitrary-precision integer arithmetic behind a strict containment
//! boundary (WS03 / architecture doc §5.3).
//!
//! `num-bigint` is an audited temporary substrate: its types and symbols appear ONLY
//! inside this crate and are NOT publicly re-exported or leaked. Higher layers depend
//! exclusively on [`BigInt`] as defined here, so the substrate can be evolved or replaced without
//! touching term/domain schemas or API surfaces. Canonical rationals live in the separate
//! `fsym-rational` ownership boundary.
//!
//! # Strategy selection
//!
//! Multiplication offers four explicit strategies:
//!
//! - [`Strategy::SchoolbookReference`] — repeated-addition scalar reference lane,
//!   O(log₂|min|) doublings; used below threshold for simple formal cross-checking.
//! - [`Strategy::Karatsuba`] — pure-Rust recursive divide-and-conquer multiplication ($O(n^{1.585})$).
//! - [`Strategy::Toom3`] — explicit, non-default Toom-3 evaluation/interpolation lane.
//! - [`Strategy::NativeSubstrate`] — delegates to the contained substrate multiplication.
//!
//! [`select_strategy`] applies [`DEFAULT_STRATEGY_THRESHOLD_BITS`] only to the existing
//! schoolbook/Karatsuba policy. Toom-3 remains opt-in until a pinned architecture/profile
//! benchmark establishes crossover evidence. All explicit lanes are differential-tested against
//! the scalar reference and contained substrate.
//!
//! # Limb accounting and cooperative cancellation
//!
//! [`BigInt::limb_count`] exposes u64-limb height (`LIMB_BITS = 64`) so callers can charge
//! budget per unit of work. [`metered_add`], [`metered_subtract`], [`metered_multiply`], and
//! [`metered_div_rem`] use deliberately simple base-$2^{32}$ reference algorithms with safe points
//! inside their limb loops. [`metered_karatsuba_candidate`] and [`metered_toom3_candidate`] add
//! controlled recursive digit kernels, while [`metered_ntt_crt_candidate`] adds an exact
//! transform/CRT lane with reconstruction bounds and independent modular self-checks. Their output
//! has no production lift into the provisional substrate.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use num_bigint::{BigInt as Substrate, BigUint, Sign};
use num_integer::Integer;
use num_traits::{FromPrimitive, Num, One, Pow, Signed, ToPrimitive, Zero};
use serde::de::{Error as DeError, SeqAccess, Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign,
    Shl, Shr, Sub, SubAssign,
};
use std::str::FromStr;

mod ntt;

/// Bits per u64 limb.
pub const LIMB_BITS: u64 = 64;

// Serde is only a convenience adapter; the authoritative bounded numeric wire lives in
// `fsym-core::canonical`. Keep the adapter's temporary magnitude storage within the same 1 MiB
// order of magnitude without treating this implementation-local limit as an identity, schema, or
// durable-format decision.
const MAX_SERDE_U32_DIGITS: usize = 1024 * 1024 / std::mem::size_of::<u32>();
const INITIAL_SERDE_U32_DIGIT_RESERVE: usize = 4_096;

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
    /// Recursive Karatsuba root with contained substrate leaves and skew fallback.
    Karatsuba,
    /// Toom-3 root with Karatsuba/substrate structural fallbacks; not selected by default.
    Toom3,
    /// Contained substrate multiplication.
    NativeSubstrate,
}

/// Canonical signed base-$2^{32}$ output from the recursively metered multiplication kernel.
///
/// Candidate publication deliberately exposes no `BigInt` lift because the provisional
/// `num-bigint` substrate has no safe public fallible constructor that adopts this digit buffer.
#[derive(Debug, PartialEq, Eq)]
pub struct MeteredProductCandidate {
    negative: bool,
    digits: Vec<u32>,
}

impl MeteredProductCandidate {
    /// Whether the canonical nonzero candidate is negative.
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Canonical little-endian base-$2^{32}$ magnitude digits.
    pub fn digits_le(&self) -> &[u32] {
        &self.digits
    }

    /// Whether this candidate is canonical zero.
    pub fn is_zero(&self) -> bool {
        self.digits.is_empty()
    }

    #[cfg(test)]
    fn materialize_unmetered(self) -> BigInt {
        let magnitude = BigUint::new(self.digits);
        let sign = if magnitude.is_zero() {
            Sign::NoSign
        } else if self.negative {
            Sign::Minus
        } else {
            Sign::Plus
        };
        BigInt(Substrate::from_biguint(sign, magnitude))
    }
}

/// Failure from the recursively metered candidate kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeteredMultiplyError {
    /// Cancellation or budget refusal from the caller-provided meter.
    Meter(MeterError),
    /// Checked buffer or shift-size arithmetic could not be represented.
    SizeOverflow,
    /// A preflighted kernel-owned digit-buffer reservation failed.
    AllocationFailure,
    /// The requested exact transform exceeds the fixed roots or CRT reconstruction domain.
    TransformDomainUnsupported,
    /// A mathematically required internal invariant did not hold.
    InvariantViolation,
}

impl fmt::Display for MeteredMultiplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meter(error) => write!(f, "{error}"),
            Self::SizeOverflow => f.write_str("metered multiplication size overflow"),
            Self::AllocationFailure => {
                f.write_str("metered multiplication backing-buffer allocation refused")
            }
            Self::TransformDomainUnsupported => {
                f.write_str("metered multiplication transform domain unsupported")
            }
            Self::InvariantViolation => {
                f.write_str("metered multiplication internal invariant violated")
            }
        }
    }
}

impl std::error::Error for MeteredMultiplyError {}

impl From<MeterError> for MeteredMultiplyError {
    fn from(error: MeterError) -> Self {
        Self::Meter(error)
    }
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

/// Greatest common divisor; always non-negative. `gcd(0, 0) == 0`.
pub fn gcd(a: &BigInt, b: &BigInt) -> BigInt {
    a.gcd(b)
}

/// Extended gcd: returns `(g, x, y)` with `a·x + b·y == g` and
/// `g == gcd(a, b)` (non-negative).
pub fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    a.extended_gcd(b)
}

/// Divides `a` by `b` when the division is exact; `None` otherwise.
pub fn exact_div(a: &BigInt, b: &BigInt) -> Option<BigInt> {
    if b.is_zero() {
        return None;
    }
    let (quotient, remainder) = a.div_rem(b);
    remainder.is_zero().then_some(quotient)
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

    /// Returns the floor of the real square root, or `None` for a negative value.
    pub fn sqrt(&self) -> Option<Self> {
        sqrt_floor(self)
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
        let max_magnitude_bits = u64::try_from(MAX_SERDE_U32_DIGITS)
            .ok()
            .and_then(|digits| digits.checked_mul(u64::from(u32::BITS)))
            .ok_or_else(|| serde::ser::Error::custom("big integer serde limit overflow"))?;
        if self.bits() > max_magnitude_bits {
            return Err(serde::ser::Error::custom(format_args!(
                "big integer magnitude exceeds {MAX_SERDE_U32_DIGITS} u32 digits"
            )));
        }
        self.0.serialize(serializer)
    }
}

struct BoundedU32Digits<const LIMIT: usize>(Vec<u32>);

struct BoundedU32DigitsVisitor<const LIMIT: usize>;

fn try_reserve_serde_digits<E>(digits: &mut Vec<u32>, additional: usize) -> Result<(), E>
where
    E: DeError,
{
    digits
        .try_reserve_exact(additional)
        .map_err(|_| E::custom("big integer magnitude digit allocation refused"))
}

fn next_serde_digit_capacity(current: usize, limit: usize) -> Option<usize> {
    if current >= limit {
        return None;
    }
    Some(current.checked_mul(2).unwrap_or(limit).max(1).min(limit))
}

impl<'de, const LIMIT: usize> Deserialize<'de> for BoundedU32Digits<LIMIT> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_seq(BoundedU32DigitsVisitor::<LIMIT>)
            .map(Self)
    }
}

impl<'de, const LIMIT: usize> Visitor<'de> for BoundedU32DigitsVisitor<LIMIT> {
    type Value = Vec<u32>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "at most {LIMIT} little-endian unsigned 32-bit magnitude digits"
        )
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let hint = seq.size_hint();
        if let Some(hint) = hint
            && hint > LIMIT
        {
            return Err(A::Error::invalid_length(hint, &self));
        }

        let mut digits = Vec::new();
        let initial_reserve = hint
            .unwrap_or(0)
            .min(INITIAL_SERDE_U32_DIGIT_RESERVE)
            .min(LIMIT);
        if initial_reserve != 0 {
            try_reserve_serde_digits::<A::Error>(&mut digits, initial_reserve)?;
        }

        while digits.len() < LIMIT {
            let Some(digit) = seq.next_element::<u32>()? else {
                return Ok(digits);
            };
            if digits.len() == digits.capacity() {
                let target_capacity = next_serde_digit_capacity(digits.capacity(), LIMIT)
                    .ok_or_else(|| {
                        A::Error::custom("big integer magnitude digit capacity invariant violated")
                    })?;
                let additional = target_capacity.checked_sub(digits.len()).ok_or_else(|| {
                    A::Error::custom("big integer magnitude digit capacity accounting overflow")
                })?;
                if additional == 0 {
                    return Err(A::Error::custom(
                        "big integer magnitude digit capacity invariant violated",
                    ));
                }
                try_reserve_serde_digits::<A::Error>(&mut digits, additional)?;
            }
            digits.push(digit);
        }

        if seq.next_element::<u32>()?.is_some() {
            return Err(A::Error::invalid_length(LIMIT.saturating_add(1), &self));
        }
        Ok(digits)
    }
}

struct BigIntWireVisitor;

impl<'de> Visitor<'de> for BigIntWireVisitor {
    type Value = BigInt;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "a canonical big integer tuple containing a sign and little-endian u32 magnitude",
        )
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let raw_sign = seq
            .next_element::<i8>()?
            .ok_or_else(|| A::Error::invalid_length(0, &self))?;
        let sign = match raw_sign {
            -1 => Sign::Minus,
            0 => Sign::NoSign,
            1 => Sign::Plus,
            other => {
                return Err(A::Error::invalid_value(
                    Unexpected::Signed(i64::from(other)),
                    &"a sign of -1, 0, or 1",
                ));
            }
        };
        let digits = match sign {
            Sign::NoSign => seq
                .next_element::<BoundedU32Digits<0>>()?
                .map(|digits| digits.0),
            Sign::Minus | Sign::Plus => seq
                .next_element::<BoundedU32Digits<MAX_SERDE_U32_DIGITS>>()?
                .map(|digits| digits.0),
        }
        .ok_or_else(|| A::Error::invalid_length(1, &self))?;
        if digits.last() == Some(&0) {
            return Err(A::Error::custom(
                "big integer magnitude has a redundant most-significant zero digit",
            ));
        }
        match (sign, digits.is_empty()) {
            (Sign::NoSign, false) => {
                return Err(A::Error::custom(
                    "zero sign requires an empty big integer magnitude",
                ));
            }
            (Sign::Minus | Sign::Plus, true) => {
                return Err(A::Error::custom(
                    "nonzero sign requires a nonempty big integer magnitude",
                ));
            }
            _ => {}
        }

        let magnitude = BigUint::new(digits);
        Ok(BigInt(Substrate::from_biguint(sign, magnitude)))
    }
}

impl<'de> Deserialize<'de> for BigInt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(2, BigIntWireVisitor)
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

/// Multiplies via the explicitly requested root strategy.
///
/// This API is unmetered. Recursive strategies use documented structural leaf and skew fallbacks,
/// so the requested enum alone is not benchmark execution evidence. [`metered_karatsuba_candidate`]
/// and [`metered_toom3_candidate`] control recursive digit kernels but expose no production
/// provisional-substrate materialization; no recursive strategy-to-`BigInt` path is presently
/// fully controlled. The explicit `BigInt`-returning Toom-3 lane remains unmetered.
pub fn multiply_with_strategy(a: &BigInt, b: &BigInt, strategy: Strategy) -> BigInt {
    match strategy {
        Strategy::SchoolbookReference => schoolbook_reference(&a.0, &b.0),
        Strategy::Karatsuba => karatsuba_mul_internal(&a.0, &b.0),
        Strategy::Toom3 => toom3_mul_internal(&a.0, &b.0),
        Strategy::NativeSubstrate => BigInt(a.0.clone() * b.0.clone()),
    }
}

/// Applies [`select_strategy`] over the operands' larger bit height, then multiplies.
pub fn multiply(a: &BigInt, b: &BigInt) -> BigInt {
    let strategy = select_strategy(std::cmp::max(a.bits(), b.bits()));
    multiply_with_strategy(a, b, strategy)
}

/// Returns the floor of the real square root, or `None` for a negative radicand.
///
/// Zero is admitted and maps to `Some(0)`. Keeping the negative-domain refusal distinct prevents
/// callers from confusing an invalid integer-to-real operation with the exact root of zero.
pub fn sqrt_floor(value: &BigInt) -> Option<BigInt> {
    if value.is_negative() {
        None
    } else {
        Some(BigInt(value.0.sqrt()))
    }
}

/// Cancellation-first binary exponentiation for a nonnegative machine exponent.
///
/// The complete, deterministic control-loop cost is charged before cloning or multiplying, so a
/// tiny compute budget can refuse an extreme exponent before result growth begins. Every multiply
/// and square then uses the cancellation-first scalar multiplication lane. A final checkpoint
/// occurs after the exact result exists and before publication.
pub fn metered_pow<M: BudgetMeter>(
    base: &BigInt,
    exponent: u32,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    if exponent == 0 {
        return metered_finish(BigInt::one(), meter);
    }
    if base.is_zero() {
        return metered_finish(BigInt::zero(), meter);
    }
    if base.is_one() {
        return metered_finish(BigInt::one(), meter);
    }
    if *base == -BigInt::one() {
        let result = if exponent.is_multiple_of(2) {
            BigInt::one()
        } else {
            -BigInt::one()
        };
        return metered_finish(result, meter);
    }

    let control_steps = u64::from(u32::BITS - exponent.leading_zeros());
    meter.charge(Dimension::ComputeSteps, control_steps)?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            base.limb_count().max(1).saturating_add(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 2),
    ])?;
    meter.checkpoint()?;
    let mut factor = base.clone();
    let mut result = BigInt::one();
    meter.checkpoint()?;

    let mut remaining = exponent;
    while remaining != 0 {
        meter.checkpoint()?;
        if remaining & 1 == 1 {
            result = metered_multiply(&result, &factor, meter)?;
        }
        remaining >>= 1;
        if remaining != 0 {
            factor = metered_multiply(&factor, &factor, meter)?;
        }
    }
    metered_finish(result, meter)
}

/// Cancellation-first floor square root with safe points in every Newton iteration and in the
/// scalar add/divide lanes it consumes.
///
/// Negative inputs publish `None`; zero publishes `Some(0)`. A final checkpoint occurs after the
/// exact root exists and before publication.
pub fn metered_sqrt_floor<M: BudgetMeter>(
    value: &BigInt,
    meter: &mut M,
) -> Result<Option<BigInt>, MeterError> {
    meter.checkpoint()?;
    if value.is_negative() {
        return metered_finish(None, meter);
    }
    if value.is_zero() {
        return metered_finish(Some(BigInt::zero()), meter);
    }
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            value
                .limb_count()
                .max(1)
                .saturating_add(1)
                .saturating_mul(8),
        ),
        (Dimension::AllocationCount, 2),
    ])?;
    meter.checkpoint()?;
    let mut current = value.clone();
    let two = BigInt::from(2i64);
    meter.checkpoint()?;
    let two_divisor = NonZeroBigInt::new(&two).expect("the constant two is nonzero");

    loop {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let current_divisor = NonZeroBigInt::new(&current).expect("Newton iterates stay positive");
        let (quotient, _) = metered_div_rem_nonzero(value, current_divisor, meter)?;
        let sum = metered_add(&current, &quotient, meter)?;
        let (next, _) = metered_div_rem_nonzero(&sum, two_divisor, meter)?;
        if metered_greater_or_equal(&next, &current, meter)? {
            return metered_finish(Some(current), meter);
        }
        current = next;
    }
}

fn metered_greater_or_equal<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    meter.checkpoint()?;
    meter.charge(
        Dimension::ComputeSteps,
        lhs.limb_count().max(rhs.limb_count()).max(1),
    )?;
    let result = lhs >= rhs;
    meter.checkpoint()?;
    Ok(result)
}

/// Metered multiplication with safe points inside the limb-product loop.
///
/// This cancellation-first lane deliberately uses a simple base-$2^{32}$ reference algorithm.
/// Each input-copy and limb-product unit is charged and preceded by a checkpoint, so cancellation
/// latency does not depend on an opaque substrate multiplication call. A final checkpoint occurs
/// after the complete result exists and before it is published.
pub fn metered_multiply<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;

    if a.is_zero() || b.is_zero() {
        return metered_finish(BigInt::zero(), meter);
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
    let result = BigInt(Substrate::from_biguint(sign, magnitude));
    metered_finish(result, meter)
}

/// Recursively computes a canonical Karatsuba product candidate under caller-owned metering.
///
/// The kernel uses normalized base-$2^{32}$ digits throughout: it does not construct provisional
/// substrate bigints inside recursion. Every nonempty kernel-owned buffer capacity is checked and
/// charged before a fallible reservation, every digit loop has safe points, and `DepthLimit` equals
/// the deepest recursion level actually entered. `MemoryBytes` records cumulative requested
/// `Vec<u32>` capacity, while `AllocationCount` records reservation attempts; neither reports
/// actual allocator capacity, allocator overhead, or peak live memory. `DepthLimit` records logical
/// recursion depth rather than stack bytes. A failed physical reservation retains its admitted
/// charges. A final checkpoint occurs after the complete digit result exists and before candidate
/// publication.
///
/// The returned candidate is not a controlled `BigInt`; pinned `num-bigint` cannot adopt the
/// buffer through a safe fallible public API.
pub fn metered_karatsuba_candidate<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<MeteredProductCandidate, MeteredMultiplyError> {
    let mut recursion = RecursiveMeter {
        meter,
        max_depth: 0,
        #[cfg(test)]
        toom_nodes: 0,
    };
    recursion.enter(1)?;
    if a.is_zero() || b.is_zero() {
        recursion.meter.checkpoint()?;
        return Ok(MeteredProductCandidate {
            negative: false,
            digits: Vec::new(),
        });
    }

    let negative = a.is_negative() != b.is_negative();
    let a_digits = metered_copy_magnitude(a, recursion.meter)?;
    let b_digits = metered_copy_magnitude(b, recursion.meter)?;
    let digits = metered_karatsuba_digits(&a_digits, &b_digits, 1, &mut recursion)?;
    if digits.is_empty() || digits.last() == Some(&0) {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    recursion.meter.checkpoint()?;
    Ok(MeteredProductCandidate { negative, digits })
}

/// Recursively computes an opt-in Toom-3 product candidate under caller-owned metering.
///
/// Evaluation at `0`, `1`, `-1`, `2`, and infinity, all five recursive products, exact signed
/// interpolation, and digit-aligned recombination stay in canonical base-$2^{32}$ buffers. Every
/// nonempty kernel-owned buffer follows the same checked capacity charge and fallible reservation
/// contract as [`metered_karatsuba_candidate`]. Structural cutoff and skew fallbacks remain inside
/// the metered Karatsuba/schoolbook digit kernels. Non-exact interpolation and negative final
/// coefficients fail with [`MeteredMultiplyError::InvariantViolation`] rather than truncating or
/// panicking. A final checkpoint separates the complete canonical digit result from publication.
/// `DepthLimit` is the maximum logical depth of the combined Toom/Karatsuba multiplication tree;
/// buffer accounting remains cumulative requested capacity rather than peak live memory.
///
/// This API does not select Toom-3 by default, establish a crossover threshold, claim a performance
/// win, or provide a controlled `BigInt` lift.
pub fn metered_toom3_candidate<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<MeteredProductCandidate, MeteredMultiplyError> {
    metered_toom3_candidate_inner(a, b, meter).map(|(candidate, _)| candidate)
}

/// Computes an opt-in exact NTT/CRT product candidate under caller-owned metering.
///
/// Magnitudes are split into base-$2^{16}$ coefficients and convolved under two fixed prime
/// moduli. Admission proves that the CRT modulus product strictly exceeds every possible integer
/// convolution coefficient before allocating transform buffers. Iterative bit reversal,
/// butterflies, pointwise multiplication, inverse normalization, CRT reconstruction, carry
/// propagation, and both modular self-checks contain cancellation safe points and compute charges.
/// Every nonempty owned buffer is checked, charged, and fallibly reserved before use. The
/// self-checks are internal fault detectors, not mathematical evidence. A final checkpoint occurs
/// after canonical output exists and before publication.
///
/// This API does not alter [`Strategy`] or [`select_strategy`], select a default transform
/// threshold, establish crossover/performance evidence, support transform lengths beyond the
/// fixed exact domain, or provide a controlled [`BigInt`] lift.
pub fn metered_ntt_crt_candidate<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<MeteredProductCandidate, MeteredMultiplyError> {
    meter.checkpoint()?;
    if a.is_zero() || b.is_zero() {
        meter.checkpoint()?;
        return Ok(MeteredProductCandidate {
            negative: false,
            digits: Vec::new(),
        });
    }

    let negative = a.is_negative() != b.is_negative();
    ntt::preflight_u32_lengths(a.0.iter_u32_digits().len(), b.0.iter_u32_digits().len())?;
    let a_digits = metered_copy_magnitude(a, meter)?;
    let b_digits = metered_copy_magnitude(b, meter)?;
    let digits = ntt::multiply_u32_digits(&a_digits, &b_digits, meter)?;
    if digits.is_empty() || digits.last() == Some(&0) {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    meter.checkpoint()?;
    Ok(MeteredProductCandidate { negative, digits })
}

fn metered_toom3_candidate_inner<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<(MeteredProductCandidate, u64), MeteredMultiplyError> {
    let mut recursion = RecursiveMeter {
        meter,
        max_depth: 0,
        #[cfg(test)]
        toom_nodes: 0,
    };
    recursion.enter(1)?;
    if a.is_zero() || b.is_zero() {
        recursion.meter.checkpoint()?;
        return Ok((
            MeteredProductCandidate {
                negative: false,
                digits: Vec::new(),
            },
            recursion.observed_toom_nodes(),
        ));
    }

    let negative = a.is_negative() != b.is_negative();
    let a_digits = metered_copy_magnitude(a, recursion.meter)?;
    let b_digits = metered_copy_magnitude(b, recursion.meter)?;
    let digits = metered_toom3_digits(&a_digits, &b_digits, 1, &mut recursion)?;
    if digits.is_empty() || digits.last() == Some(&0) {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    recursion.meter.checkpoint()?;
    Ok((
        MeteredProductCandidate { negative, digits },
        recursion.observed_toom_nodes(),
    ))
}

// Structural recursion bound only; this is not a calibrated strategy crossover.
const METERED_KARATSUBA_LEAF_DIGITS: usize = 4;
// Structural Toom recursion bound only; this does not authorize a selector threshold.
const METERED_TOOM3_RECURSION_CUTOFF_DIGITS: usize = 12;

struct RecursiveMeter<'a, M> {
    meter: &'a mut M,
    max_depth: u64,
    #[cfg(test)]
    toom_nodes: u64,
}

impl<M: BudgetMeter> RecursiveMeter<'_, M> {
    fn enter(&mut self, depth: usize) -> Result<(), MeteredMultiplyError> {
        self.meter.checkpoint()?;
        let depth = u64::try_from(depth).map_err(|_| MeteredMultiplyError::SizeOverflow)?;
        if depth > self.max_depth {
            let delta = depth - self.max_depth;
            self.meter
                .charge_batch(&[(Dimension::ComputeSteps, 1), (Dimension::DepthLimit, delta)])?;
            self.max_depth = depth;
        } else {
            self.meter.charge(Dimension::ComputeSteps, 1)?;
        }
        Ok(())
    }

    fn mark_toom_node(&mut self) -> Result<(), MeteredMultiplyError> {
        #[cfg(test)]
        {
            self.toom_nodes = self
                .toom_nodes
                .checked_add(1)
                .ok_or(MeteredMultiplyError::SizeOverflow)?;
        }
        Ok(())
    }

    fn observed_toom_nodes(&self) -> u64 {
        #[cfg(test)]
        {
            self.toom_nodes
        }
        #[cfg(not(test))]
        {
            0
        }
    }
}

fn metered_karatsuba_digits<M: BudgetMeter>(
    a: &[u32],
    b: &[u32],
    depth: usize,
    recursion: &mut RecursiveMeter<'_, M>,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    if depth != 1 {
        recursion.enter(depth)?;
    }
    metered_karatsuba_digits_entered(a, b, depth, recursion)
}

fn metered_karatsuba_digits_entered<M: BudgetMeter>(
    a: &[u32],
    b: &[u32],
    depth: usize,
    recursion: &mut RecursiveMeter<'_, M>,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }

    let output_capacity = checked_product_capacity(a.len(), b.len())?;
    let max_digits = a.len().max(b.len());
    let min_digits = a.len().min(b.len());
    let split_digits = max_digits.div_ceil(2);
    if max_digits <= METERED_KARATSUBA_LEAF_DIGITS || min_digits <= split_digits {
        return metered_schoolbook_digits(a, b, output_capacity, recursion.meter);
    }
    let z2_shift_digits = split_digits
        .checked_mul(2)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;

    let (a0, a1) = metered_split_slices(a, split_digits, recursion.meter)?;
    let (b0, b1) = metered_split_slices(b, split_digits, recursion.meter)?;
    let child_depth = depth
        .checked_add(1)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;

    let z0 = metered_karatsuba_digits(a0, b0, child_depth, recursion)?;
    let z2 = metered_karatsuba_digits(a1, b1, child_depth, recursion)?;
    let sum_a = metered_add_digit_slices(a0, a1, recursion.meter)?;
    let sum_b = metered_add_digit_slices(b0, b1, recursion.meter)?;
    let combined = metered_karatsuba_digits(&sum_a, &sum_b, child_depth, recursion)?;
    let without_z0 = metered_subtract_digit_slices(&combined, &z0, recursion.meter)?;
    let z1 = metered_subtract_digit_slices(&without_z0, &z2, recursion.meter)?;

    let z1_end = split_digits
        .checked_add(z1.len())
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    let z2_end = z2_shift_digits
        .checked_add(z2.len())
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    if z0.len() > output_capacity || z1_end > output_capacity || z2_end > output_capacity {
        return Err(MeteredMultiplyError::InvariantViolation);
    }

    let mut output = metered_zeroed_digits(output_capacity, recursion.meter)?;
    metered_add_shifted_digits(&mut output, &z0, 0, recursion.meter)?;
    metered_add_shifted_digits(&mut output, &z1, split_digits, recursion.meter)?;
    metered_add_shifted_digits(&mut output, &z2, z2_shift_digits, recursion.meter)?;
    metered_trim_owned_digits(&mut output, recursion.meter)?;
    Ok(output)
}

struct SignedDigits {
    negative: bool,
    digits: Vec<u32>,
}

impl SignedDigits {
    fn new(negative: bool, digits: Vec<u32>) -> Self {
        Self {
            negative: negative && !digits.is_empty(),
            digits,
        }
    }
}

struct ToomEvaluations {
    at_one: Vec<u32>,
    at_minus_one: SignedDigits,
    at_two: Vec<u32>,
}

struct ToomChunks<'a> {
    low: &'a [u32],
    middle: &'a [u32],
    high: &'a [u32],
}

struct ToomCoefficients {
    c0: Vec<u32>,
    c1: Vec<u32>,
    c2: Vec<u32>,
    c3: Vec<u32>,
    c4: Vec<u32>,
}

fn metered_toom3_digits<M: BudgetMeter>(
    a: &[u32],
    b: &[u32],
    depth: usize,
    recursion: &mut RecursiveMeter<'_, M>,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    if depth != 1 {
        recursion.enter(depth)?;
    }
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }

    let output_capacity = checked_product_capacity(a.len(), b.len())?;
    let max_digits = a.len().max(b.len());
    let min_digits = a.len().min(b.len());
    let chunk_digits = max_digits.div_ceil(3);
    let chunk_digits_x2 = chunk_digits
        .checked_mul(2)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    if max_digits <= METERED_TOOM3_RECURSION_CUTOFF_DIGITS || min_digits <= chunk_digits_x2 {
        return metered_karatsuba_digits_entered(a, b, depth, recursion);
    }
    let chunk_digits_x3 = chunk_digits
        .checked_mul(3)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    let chunk_digits_x4 = chunk_digits
        .checked_mul(4)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    let child_depth = depth
        .checked_add(1)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    recursion.mark_toom_node()?;

    let a_chunks = metered_split_three_slices(a, chunk_digits, chunk_digits_x2, recursion.meter)?;
    let b_chunks = metered_split_three_slices(b, chunk_digits, chunk_digits_x2, recursion.meter)?;
    let (a0, a1, a2) = (a_chunks.low, a_chunks.middle, a_chunks.high);
    let (b0, b1, b2) = (b_chunks.low, b_chunks.middle, b_chunks.high);
    let a_values = metered_evaluate_toom3_chunks(a0, a1, a2, recursion.meter)?;
    let b_values = metered_evaluate_toom3_chunks(b0, b1, b2, recursion.meter)?;

    let w0 = metered_toom3_signed_product(false, a0, false, b0, child_depth, recursion)?;
    let w1 = metered_toom3_signed_product(
        false,
        &a_values.at_one,
        false,
        &b_values.at_one,
        child_depth,
        recursion,
    )?;
    let w_minus_one = metered_toom3_signed_product(
        a_values.at_minus_one.negative,
        &a_values.at_minus_one.digits,
        b_values.at_minus_one.negative,
        &b_values.at_minus_one.digits,
        child_depth,
        recursion,
    )?;
    let w2 = metered_toom3_signed_product(
        false,
        &a_values.at_two,
        false,
        &b_values.at_two,
        child_depth,
        recursion,
    )?;
    let w4 = metered_toom3_signed_product(false, a2, false, b2, child_depth, recursion)?;
    let coefficients = metered_interpolate_toom3(w0, w1, w_minus_one, w2, w4, recursion.meter)?;
    metered_recombine_toom3(
        coefficients,
        [
            0,
            chunk_digits,
            chunk_digits_x2,
            chunk_digits_x3,
            chunk_digits_x4,
        ],
        output_capacity,
        recursion.meter,
    )
}

fn metered_toom3_signed_product<M: BudgetMeter>(
    a_negative: bool,
    a: &[u32],
    b_negative: bool,
    b: &[u32],
    depth: usize,
    recursion: &mut RecursiveMeter<'_, M>,
) -> Result<SignedDigits, MeteredMultiplyError> {
    let digits = metered_toom3_digits(a, b, depth, recursion)?;
    Ok(SignedDigits::new(a_negative != b_negative, digits))
}

fn metered_split_three_slices<'a, M: BudgetMeter>(
    digits: &'a [u32],
    chunk_digits: usize,
    chunk_digits_x2: usize,
    meter: &mut M,
) -> Result<ToomChunks<'a>, MeteredMultiplyError> {
    let first = chunk_digits.min(digits.len());
    let second = chunk_digits_x2.min(digits.len());
    let low = metered_trimmed_slice(&digits[..first], meter)?;
    let middle = metered_trimmed_slice(&digits[first..second], meter)?;
    let high = metered_trimmed_slice(&digits[second..], meter)?;
    Ok(ToomChunks { low, middle, high })
}

fn metered_evaluate_toom3_chunks<M: BudgetMeter>(
    c0: &[u32],
    c1: &[u32],
    c2: &[u32],
    meter: &mut M,
) -> Result<ToomEvaluations, MeteredMultiplyError> {
    let c0_plus_c2 = metered_add_digit_slices(c0, c2, meter)?;
    let at_one = metered_add_digit_slices(&c0_plus_c2, c1, meter)?;
    let at_minus_one = metered_signed_add_parts(false, &c0_plus_c2, true, c1, meter)?;

    let twice_c1 = metered_multiply_digit_slice_by_small(c1, 2, meter)?;
    let four_c2 = metered_multiply_digit_slice_by_small(c2, 4, meter)?;
    let c0_plus_twice_c1 = metered_add_digit_slices(c0, &twice_c1, meter)?;
    let at_two = metered_add_digit_slices(&c0_plus_twice_c1, &four_c2, meter)?;
    Ok(ToomEvaluations {
        at_one,
        at_minus_one,
        at_two,
    })
}

fn metered_interpolate_toom3<M: BudgetMeter>(
    w0: SignedDigits,
    w1: SignedDigits,
    w_minus_one: SignedDigits,
    w2: SignedDigits,
    w4: SignedDigits,
    meter: &mut M,
) -> Result<ToomCoefficients, MeteredMultiplyError> {
    if w0.negative || w1.negative || w2.negative || w4.negative {
        return Err(MeteredMultiplyError::InvariantViolation);
    }

    let w1_plus_minus_one = metered_signed_add(&w1, &w_minus_one, meter)?;
    let half_sum = metered_exact_divide_signed_small(w1_plus_minus_one, 2, meter)?;
    let without_w0 = metered_signed_subtract(&half_sum, &w0, meter)?;
    let c2 = metered_signed_subtract(&without_w0, &w4, meter)?;

    let w1_minus_minus_one = metered_signed_subtract(&w1, &w_minus_one, meter)?;
    let sum_c1_c3 = metered_exact_divide_signed_small(w1_minus_minus_one, 2, meter)?;

    let four_c2 = metered_multiply_signed_by_small(&c2, 4, meter)?;
    let sixteen_w4 = metered_multiply_signed_by_small(&w4, 16, meter)?;
    let w2_without_w0 = metered_signed_subtract(&w2, &w0, meter)?;
    let without_four_c2 = metered_signed_subtract(&w2_without_w0, &four_c2, meter)?;
    let without_sixteen_w4 = metered_signed_subtract(&without_four_c2, &sixteen_w4, meter)?;
    let c1_plus_four_c3 = metered_exact_divide_signed_small(without_sixteen_w4, 2, meter)?;
    let three_c3 = metered_signed_subtract(&c1_plus_four_c3, &sum_c1_c3, meter)?;
    let c3 = metered_exact_divide_signed_small(three_c3, 3, meter)?;
    let c1 = metered_signed_subtract(&sum_c1_c3, &c3, meter)?;

    if c1.negative || c2.negative || c3.negative {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    for digits in [&w0.digits, &c1.digits, &c2.digits, &c3.digits, &w4.digits] {
        if digits.last() == Some(&0) {
            return Err(MeteredMultiplyError::InvariantViolation);
        }
    }
    Ok(ToomCoefficients {
        c0: w0.digits,
        c1: c1.digits,
        c2: c2.digits,
        c3: c3.digits,
        c4: w4.digits,
    })
}

fn metered_recombine_toom3<M: BudgetMeter>(
    coefficients: ToomCoefficients,
    offsets: [usize; 5],
    output_capacity: usize,
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    let coefficient_slices = [
        coefficients.c0.as_slice(),
        coefficients.c1.as_slice(),
        coefficients.c2.as_slice(),
        coefficients.c3.as_slice(),
        coefficients.c4.as_slice(),
    ];
    for (digits, offset) in coefficient_slices.iter().zip(offsets) {
        let end = offset
            .checked_add(digits.len())
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
        if end > output_capacity {
            return Err(MeteredMultiplyError::InvariantViolation);
        }
    }

    let mut output = metered_zeroed_digits(output_capacity, meter)?;
    for (digits, offset) in coefficient_slices.iter().zip(offsets) {
        metered_add_shifted_digits(&mut output, digits, offset, meter)?;
    }
    metered_trim_owned_digits(&mut output, meter)?;
    Ok(output)
}

fn metered_signed_add<M: BudgetMeter>(
    lhs: &SignedDigits,
    rhs: &SignedDigits,
    meter: &mut M,
) -> Result<SignedDigits, MeteredMultiplyError> {
    metered_signed_add_parts(lhs.negative, &lhs.digits, rhs.negative, &rhs.digits, meter)
}

fn metered_signed_subtract<M: BudgetMeter>(
    lhs: &SignedDigits,
    rhs: &SignedDigits,
    meter: &mut M,
) -> Result<SignedDigits, MeteredMultiplyError> {
    metered_signed_add_parts(lhs.negative, &lhs.digits, !rhs.negative, &rhs.digits, meter)
}

fn metered_signed_add_parts<M: BudgetMeter>(
    lhs_negative: bool,
    lhs: &[u32],
    rhs_negative: bool,
    rhs: &[u32],
    meter: &mut M,
) -> Result<SignedDigits, MeteredMultiplyError> {
    if lhs.is_empty() {
        return Ok(SignedDigits::new(
            rhs_negative,
            metered_copy_digit_slice(rhs, meter)?,
        ));
    }
    if rhs.is_empty() {
        return Ok(SignedDigits::new(
            lhs_negative,
            metered_copy_digit_slice(lhs, meter)?,
        ));
    }
    if lhs_negative == rhs_negative {
        return Ok(SignedDigits::new(
            lhs_negative,
            metered_add_digit_slices(lhs, rhs, meter)?,
        ));
    }

    match metered_compare_digit_slices(lhs, rhs, meter)? {
        std::cmp::Ordering::Less => Ok(SignedDigits::new(
            rhs_negative,
            metered_subtract_digit_slices(rhs, lhs, meter)?,
        )),
        std::cmp::Ordering::Equal => Ok(SignedDigits::new(false, Vec::new())),
        std::cmp::Ordering::Greater => Ok(SignedDigits::new(
            lhs_negative,
            metered_subtract_digit_slices(lhs, rhs, meter)?,
        )),
    }
}

fn metered_copy_digit_slice<M: BudgetMeter>(
    digits: &[u32],
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    charge_vec_capacities(&[digits.len()], meter)?;
    let mut output = try_u32_vec(digits.len())?;
    for &digit in digits {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        output.push(digit);
    }
    Ok(output)
}

fn metered_multiply_digit_slice_by_small<M: BudgetMeter>(
    digits: &[u32],
    factor: u32,
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    if digits.is_empty() || factor == 0 {
        return Ok(Vec::new());
    }
    let capacity = digits
        .len()
        .checked_add(1)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    charge_vec_capacities(&[capacity], meter)?;
    let mut output = try_u32_vec(capacity)?;
    let mut carry = 0u64;
    for &digit in digits {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let product = u64::from(digit) * u64::from(factor) + carry;
        output.push(
            u32::try_from(product & u64::from(u32::MAX))
                .map_err(|_| MeteredMultiplyError::InvariantViolation)?,
        );
        carry = product >> 32;
    }
    if carry != 0 {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        output.push(u32::try_from(carry).map_err(|_| MeteredMultiplyError::InvariantViolation)?);
    }
    Ok(output)
}

fn metered_multiply_signed_by_small<M: BudgetMeter>(
    value: &SignedDigits,
    factor: u32,
    meter: &mut M,
) -> Result<SignedDigits, MeteredMultiplyError> {
    Ok(SignedDigits::new(
        value.negative,
        metered_multiply_digit_slice_by_small(&value.digits, factor, meter)?,
    ))
}

fn metered_exact_divide_signed_small<M: BudgetMeter>(
    mut value: SignedDigits,
    divisor: u32,
    meter: &mut M,
) -> Result<SignedDigits, MeteredMultiplyError> {
    if divisor == 0 {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    let divisor = u64::from(divisor);
    let mut remainder = 0u64;
    for digit in value.digits.iter_mut().rev() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let wide = (remainder << 32) | u64::from(*digit);
        *digit =
            u32::try_from(wide / divisor).map_err(|_| MeteredMultiplyError::InvariantViolation)?;
        remainder = wide % divisor;
    }
    if remainder != 0 {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    metered_trim_owned_digits(&mut value.digits, meter)?;
    value.negative = value.negative && !value.digits.is_empty();
    Ok(value)
}

fn checked_product_capacity(
    lhs_digits: usize,
    rhs_digits: usize,
) -> Result<usize, MeteredMultiplyError> {
    lhs_digits
        .checked_add(rhs_digits)
        .ok_or(MeteredMultiplyError::SizeOverflow)
}

fn metered_copy_magnitude<M: BudgetMeter>(
    value: &BigInt,
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    meter.checkpoint()?;
    let capacity = value.0.iter_u32_digits().len();
    charge_vec_capacities(&[capacity], meter)?;
    let mut digits = try_u32_vec(capacity)?;
    for digit in value.0.iter_u32_digits() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        digits.push(digit);
    }
    Ok(digits)
}

fn metered_split_slices<'a, M: BudgetMeter>(
    digits: &'a [u32],
    split_digits: usize,
    meter: &mut M,
) -> Result<(&'a [u32], &'a [u32]), MeteredMultiplyError> {
    let midpoint = split_digits.min(digits.len());
    let (low, high) = digits.split_at(midpoint);
    let low = metered_trimmed_slice(low, meter)?;
    let high = metered_trimmed_slice(high, meter)?;
    Ok((low, high))
}

fn metered_trimmed_slice<'a, M: BudgetMeter>(
    mut digits: &'a [u32],
    meter: &mut M,
) -> Result<&'a [u32], MeteredMultiplyError> {
    while let Some((&last, rest)) = digits.split_last() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        if last != 0 {
            break;
        }
        digits = rest;
    }
    Ok(digits)
}

fn metered_add_digit_slices<M: BudgetMeter>(
    lhs: &[u32],
    rhs: &[u32],
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    if lhs.is_empty() && rhs.is_empty() {
        return Ok(Vec::new());
    }
    let capacity = lhs
        .len()
        .max(rhs.len())
        .checked_add(1)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    charge_vec_capacities(&[capacity], meter)?;
    let mut output = try_u32_vec(capacity)?;
    let mut carry = 0u64;
    for index in 0..lhs.len().max(rhs.len()) {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let sum = u64::from(lhs.get(index).copied().unwrap_or(0))
            + u64::from(rhs.get(index).copied().unwrap_or(0))
            + carry;
        output.push(sum as u32);
        carry = sum >> 32;
    }
    if carry != 0 {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        output.push(carry as u32);
    }
    Ok(output)
}

fn metered_subtract_digit_slices<M: BudgetMeter>(
    lhs: &[u32],
    rhs: &[u32],
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    match metered_compare_digit_slices(lhs, rhs, meter)? {
        std::cmp::Ordering::Less => return Err(MeteredMultiplyError::InvariantViolation),
        std::cmp::Ordering::Equal => return Ok(Vec::new()),
        std::cmp::Ordering::Greater => {}
    }
    charge_vec_capacities(&[lhs.len()], meter)?;
    let mut output = try_u32_vec(lhs.len())?;
    let mut borrow = 0u64;
    for (index, &lhs_digit) in lhs.iter().enumerate() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let rhs_digit = u64::from(rhs.get(index).copied().unwrap_or(0));
        let subtrahend = rhs_digit + borrow;
        let lhs_value = u64::from(lhs_digit);
        if lhs_value >= subtrahend {
            output.push((lhs_value - subtrahend) as u32);
            borrow = 0;
        } else {
            output.push(((1u64 << 32) + lhs_value - subtrahend) as u32);
            borrow = 1;
        }
    }
    if borrow != 0 {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    metered_trim_owned_digits(&mut output, meter)?;
    Ok(output)
}

fn metered_compare_digit_slices<M: BudgetMeter>(
    lhs: &[u32],
    rhs: &[u32],
    meter: &mut M,
) -> Result<std::cmp::Ordering, MeteredMultiplyError> {
    match lhs.len().cmp(&rhs.len()) {
        std::cmp::Ordering::Equal => {}
        ordering => return Ok(ordering),
    }
    for (&lhs_digit, &rhs_digit) in lhs.iter().rev().zip(rhs.iter().rev()) {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        match lhs_digit.cmp(&rhs_digit) {
            std::cmp::Ordering::Equal => {}
            ordering => return Ok(ordering),
        }
    }
    Ok(std::cmp::Ordering::Equal)
}

fn metered_schoolbook_digits<M: BudgetMeter>(
    lhs: &[u32],
    rhs: &[u32],
    output_capacity: usize,
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    let mut product = metered_zeroed_digits(output_capacity, meter)?;
    for (i, &lhs_digit) in lhs.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &rhs_digit) in rhs.iter().enumerate() {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let index = i.checked_add(j).ok_or(MeteredMultiplyError::SizeOverflow)?;
            let slot = product
                .get_mut(index)
                .ok_or(MeteredMultiplyError::InvariantViolation)?;
            let value = u64::from(*slot) + u64::from(lhs_digit) * u64::from(rhs_digit) + carry;
            *slot = value as u32;
            carry = value >> 32;
        }
        let mut index = i
            .checked_add(rhs.len())
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
        while carry != 0 {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let slot = product
                .get_mut(index)
                .ok_or(MeteredMultiplyError::InvariantViolation)?;
            let value = u64::from(*slot) + carry;
            *slot = value as u32;
            carry = value >> 32;
            index = index
                .checked_add(1)
                .ok_or(MeteredMultiplyError::SizeOverflow)?;
        }
    }
    metered_trim_owned_digits(&mut product, meter)?;
    Ok(product)
}

fn metered_zeroed_digits<M: BudgetMeter>(
    capacity: usize,
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    charge_vec_capacities(&[capacity], meter)?;
    let mut digits = try_u32_vec(capacity)?;
    for _ in 0..capacity {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        digits.push(0);
    }
    Ok(digits)
}

fn metered_add_shifted_digits<M: BudgetMeter>(
    target: &mut [u32],
    source: &[u32],
    offset: usize,
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    let mut carry = 0u64;
    for (source_index, &source_digit) in source.iter().enumerate() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let index = offset
            .checked_add(source_index)
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
        let slot = target
            .get_mut(index)
            .ok_or(MeteredMultiplyError::InvariantViolation)?;
        let sum = u64::from(*slot) + u64::from(source_digit) + carry;
        *slot = sum as u32;
        carry = sum >> 32;
    }
    let mut index = offset
        .checked_add(source.len())
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    while carry != 0 {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let slot = target
            .get_mut(index)
            .ok_or(MeteredMultiplyError::InvariantViolation)?;
        let sum = u64::from(*slot) + carry;
        *slot = sum as u32;
        carry = sum >> 32;
        index = index
            .checked_add(1)
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
    }
    Ok(())
}

fn metered_trim_owned_digits<M: BudgetMeter>(
    digits: &mut Vec<u32>,
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    while let Some(&last) = digits.last() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        if last != 0 {
            break;
        }
        digits.pop();
    }
    Ok(())
}

fn charge_vec_capacities<M: BudgetMeter>(
    capacities: &[usize],
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    let mut total_elements = 0usize;
    let mut allocations = 0u64;
    for &capacity in capacities {
        if capacity == 0 {
            continue;
        }
        total_elements = total_elements
            .checked_add(capacity)
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
        allocations = allocations
            .checked_add(1)
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
    }
    if allocations == 0 {
        return Ok(());
    }
    let memory_bytes = u64::try_from(total_elements)
        .map_err(|_| MeteredMultiplyError::SizeOverflow)?
        .checked_mul(4)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::MemoryBytes, memory_bytes),
        (Dimension::AllocationCount, allocations),
    ])?;
    Ok(())
}

fn try_u32_vec(capacity: usize) -> Result<Vec<u32>, MeteredMultiplyError> {
    let mut digits = Vec::new();
    if capacity != 0 {
        digits
            .try_reserve_exact(capacity)
            .map_err(|_| MeteredMultiplyError::AllocationFailure)?;
    }
    Ok(digits)
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
/// checkpoint. All four transient digit buffers are charged before allocation. A final checkpoint
/// occurs after the complete optional result exists and before it is published.
pub fn metered_div_rem<M: BudgetMeter>(
    dividend: &BigInt,
    divisor: &BigInt,
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    let Some(divisor) = NonZeroBigInt::new(divisor) else {
        return metered_finish(None, meter);
    };
    let result = metered_div_rem_nonzero(dividend, divisor, meter)?;
    metered_finish(Some(result), meter)
}

/// Cancellation-first truncating division after typed nonzero-divisor admission.
///
/// Every terminal class is staged before a final cancellation checkpoint and publication.
pub fn metered_div_rem_nonzero<M: BudgetMeter>(
    dividend: &BigInt,
    divisor: NonZeroBigInt<'_>,
    meter: &mut M,
) -> Result<(BigInt, BigInt), MeterError> {
    meter.checkpoint()?;
    let divisor = divisor.get();
    if dividend.is_zero() {
        let result = (BigInt::zero(), BigInt::zero());
        return metered_finish(result, meter);
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
        let result = (BigInt::zero(), dividend.clone());
        return metered_finish(result, meter);
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
    let result = (
        BigInt(Substrate::from_biguint(quotient_sign, quotient_magnitude)),
        BigInt(Substrate::from_biguint(remainder_sign, remainder_magnitude)),
    );
    metered_finish(result, meter)
}

/// Cancellation-first exact division using the metered scalar division lane.
///
/// The exact quotient, inexact refusal, and zero-divisor refusal each pass a final cancellation
/// checkpoint after the complete optional result exists and before publication.
pub fn metered_exact_div<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<Option<BigInt>, MeterError> {
    let Some((quotient, remainder)) = metered_div_rem(a, b, meter)? else {
        return metered_finish(None, meter);
    };
    let result = remainder.is_zero().then_some(quotient);
    metered_finish(result, meter)
}

/// Metered greatest common divisor with step accounting and cancellation checkpoints.
pub fn metered_gcd<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            a.limb_count()
                .max(1)
                .saturating_add(b.limb_count().max(1))
                .saturating_mul(8),
        ),
        (Dimension::AllocationCount, 2),
    ])?;
    let mut a = a.clone();
    let mut b = b.clone();
    while let Some(divisor) = NonZeroBigInt::new(&b) {
        meter.checkpoint()?;
        let (_, remainder) = metered_div_rem_nonzero(&a, divisor, meter)?;
        a = b;
        b = remainder;
    }
    meter.checkpoint()?;
    let result = if a.is_negative() { -a } else { a };
    metered_finish(result, meter)
}

/// Metered extended gcd with step accounting and cancellation checkpoints.
pub fn metered_extended_gcd<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<(BigInt, BigInt, BigInt), MeterError> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            a.limb_count()
                .saturating_add(b.limb_count())
                .saturating_add(4)
                .saturating_mul(8),
        ),
        (Dimension::AllocationCount, 6),
    ])?;
    let (mut old_r, mut r) = (a.clone(), b.clone());
    let (mut old_s, mut s) = (BigInt::one(), BigInt::zero());
    let (mut old_t, mut t) = (BigInt::zero(), BigInt::one());
    while let Some(divisor) = NonZeroBigInt::new(&r) {
        meter.checkpoint()?;
        let (quotient, remainder) = metered_div_rem_nonzero(&old_r, divisor, meter)?;
        old_r = r;
        r = remainder;
        let quotient_times_s = metered_multiply(&quotient, &s, meter)?;
        let next_s = metered_subtract(&old_s, &quotient_times_s, meter)?;
        old_s = s;
        s = next_s;
        let quotient_times_t = metered_multiply(&quotient, &t, meter)?;
        let next_t = metered_subtract(&old_t, &quotient_times_t, meter)?;
        old_t = t;
        t = next_t;
    }
    let result = if old_r.is_negative() {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    };
    metered_finish(result, meter)
}

fn metered_finish<T, M: BudgetMeter>(value: T, meter: &mut M) -> Result<T, MeterError> {
    meter.checkpoint()?;
    Ok(value)
}

/// Cancellation-first signed addition with safe points inside the base-$2^{32}$ limb loop.
pub fn metered_add<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    metered_signed_sum(lhs, rhs, false, meter)
}

/// Cancellation-first signed subtraction with safe points inside the base-$2^{32}$ limb loop.
pub fn metered_subtract<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    metered_signed_sum(lhs, rhs, true, meter)
}

fn metered_signed_sum<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    subtract: bool,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    if lhs.is_zero() && rhs.is_zero() {
        return metered_finish(BigInt::zero(), meter);
    }

    let lhs_len = lhs.0.iter_u32_digits().len();
    let rhs_len = rhs.0.iter_u32_digits().len();
    let output_capacity = lhs_len.max(rhs_len).saturating_add(1);
    let transient_digits = lhs_len
        .saturating_add(rhs_len)
        .saturating_add(output_capacity);
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            u64::try_from(transient_digits)
                .unwrap_or(u64::MAX)
                .saturating_mul(4),
        ),
        (Dimension::AllocationCount, 3),
    ])?;

    let mut lhs_digits = copy_u32_digits(&lhs.0, lhs_len, meter)?;
    let mut rhs_digits = copy_u32_digits(&rhs.0, rhs_len, meter)?;
    let lhs_sign = lhs.0.sign();
    let rhs_sign = if subtract {
        opposite_sign(rhs.0.sign())
    } else {
        rhs.0.sign()
    };

    let (digits, sign) = if lhs_sign == rhs_sign {
        (
            add_digits(&lhs_digits, &rhs_digits, output_capacity, meter)?,
            lhs_sign,
        )
    } else {
        match compare_digits(&lhs_digits, &rhs_digits, meter)? {
            std::cmp::Ordering::Greater => {
                subtract_digits(&mut lhs_digits, &rhs_digits, meter)?;
                (lhs_digits, lhs_sign)
            }
            std::cmp::Ordering::Less => {
                subtract_digits(&mut rhs_digits, &lhs_digits, meter)?;
                (rhs_digits, rhs_sign)
            }
            std::cmp::Ordering::Equal => (Vec::new(), Sign::NoSign),
        }
    };

    meter.checkpoint()?;
    let magnitude = BigUint::new(digits);
    let sign = if magnitude.is_zero() {
        Sign::NoSign
    } else {
        sign
    };
    metered_finish(BigInt(Substrate::from_biguint(sign, magnitude)), meter)
}

fn opposite_sign(sign: Sign) -> Sign {
    match sign {
        Sign::Minus => Sign::Plus,
        Sign::NoSign => Sign::NoSign,
        Sign::Plus => Sign::Minus,
    }
}

fn add_digits<M: BudgetMeter>(
    lhs: &[u32],
    rhs: &[u32],
    capacity: usize,
    meter: &mut M,
) -> Result<Vec<u32>, MeterError> {
    let mut output = Vec::with_capacity(capacity);
    let mut carry = 0u64;
    for index in 0..lhs.len().max(rhs.len()) {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let sum = u64::from(lhs.get(index).copied().unwrap_or(0))
            + u64::from(rhs.get(index).copied().unwrap_or(0))
            + carry;
        output.push(sum as u32);
        carry = sum >> 32;
    }
    if carry != 0 {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        output.push(carry as u32);
    }
    Ok(output)
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

const KARATSUBA_LEAF_BITS: u64 = 128;
// A structural recursion guard, not a calibrated strategy-selection threshold.
const TOOM3_RECURSION_CUTOFF_BITS: u64 = 3 * KARATSUBA_LEAF_BITS;

fn karatsuba_mag_internal(a: &BigUint, b: &BigUint) -> BigUint {
    if a.is_zero() || b.is_zero() {
        return BigUint::zero();
    }
    let max_bits = std::cmp::max(a.bits(), b.bits());
    let min_bits = std::cmp::min(a.bits(), b.bits());
    if max_bits <= KARATSUBA_LEAF_BITS || min_bits <= max_bits / 2 {
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

fn toom3_mul_internal(a: &Substrate, b: &Substrate) -> BigInt {
    BigInt(toom3_signed_internal(a, b))
}

fn toom3_signed_internal(a: &Substrate, b: &Substrate) -> Substrate {
    if a.is_zero() || b.is_zero() {
        return Substrate::zero();
    }

    let negative = a.sign() != b.sign();
    let magnitude = toom3_mag_internal(a.magnitude(), b.magnitude());
    let result = Substrate::from(magnitude);
    if negative { -result } else { result }
}

fn toom3_mag_internal(a: &BigUint, b: &BigUint) -> BigUint {
    if a.is_zero() || b.is_zero() {
        return BigUint::zero();
    }

    let max_bits = std::cmp::max(a.bits(), b.bits());
    let min_bits = std::cmp::min(a.bits(), b.bits());
    if max_bits <= TOOM3_RECURSION_CUTOFF_BITS {
        return karatsuba_mag_internal(a, b);
    }

    let chunk_bits = max_bits.div_ceil(3);
    let (Some(chunk_bits_x2), Some(chunk_bits_x3), Some(chunk_bits_x4)) = (
        chunk_bits.checked_mul(2),
        chunk_bits.checked_mul(3),
        chunk_bits.checked_mul(4),
    ) else {
        return a * b;
    };
    // Both operands need a nonzero top third. Skewed or cancellation-shortened recursive values
    // stay on the hardened Karatsuba/native lane instead of expanding zero-heavy Toom branches.
    if min_bits <= chunk_bits_x2 {
        return karatsuba_mag_internal(a, b);
    }

    let mask = (BigUint::one() << chunk_bits) - 1u32;
    let split = |value: &BigUint| {
        let low = value & &mask;
        let middle = (value >> chunk_bits) & &mask;
        let high = value >> chunk_bits_x2;
        (
            Substrate::from(low),
            Substrate::from(middle),
            Substrate::from(high),
        )
    };
    let (a0, a1, a2) = split(a);
    let (b0, b1, b2) = split(b);

    let evaluate = |c0: &Substrate, c1: &Substrate, c2: &Substrate| {
        let at_zero = c0.clone();
        let at_one = c0 + c1 + c2;
        let at_minus_one = c0 - c1 + c2;
        let at_two = c0 + (c1 << 1u32) + (c2 << 2u32);
        let at_infinity = c2.clone();
        (at_zero, at_one, at_minus_one, at_two, at_infinity)
    };
    let (a_at_zero, a_at_one, a_at_minus_one, a_at_two, a_at_infinity) = evaluate(&a0, &a1, &a2);
    let (b_at_zero, b_at_one, b_at_minus_one, b_at_two, b_at_infinity) = evaluate(&b0, &b1, &b2);

    let w0 = toom3_signed_internal(&a_at_zero, &b_at_zero);
    let w1 = toom3_signed_internal(&a_at_one, &b_at_one);
    let w_minus_one = toom3_signed_internal(&a_at_minus_one, &b_at_minus_one);
    let w2 = toom3_signed_internal(&a_at_two, &b_at_two);
    let w4 = toom3_signed_internal(&a_at_infinity, &b_at_infinity);

    let c2 = exact_div_small(&w1 + &w_minus_one, 2) - &w0 - &w4;
    let sum_c1_c3 = exact_div_small(&w1 - &w_minus_one, 2);
    let c1_plus_four_c3 = exact_div_small(w2 - &w0 - (&c2 << 2u32) - (&w4 << 4u32), 2);
    let c3 = exact_div_small(&c1_plus_four_c3 - &sum_c1_c3, 3);
    let c1 = sum_c1_c3 - &c3;

    let result = w0
        + (c1 << chunk_bits)
        + (c2 << chunk_bits_x2)
        + (c3 << chunk_bits_x3)
        + (w4 << chunk_bits_x4);
    assert!(
        result.sign() != Sign::Minus,
        "internal Toom-3 magnitude interpolation must be non-negative"
    );
    result.magnitude().clone()
}

fn exact_div_small(value: Substrate, divisor: u32) -> Substrate {
    let divisor = Substrate::from(divisor);
    let (quotient, remainder) = value.div_rem(&divisor);
    assert!(
        remainder.is_zero(),
        "internal Toom-3 interpolation division must be exact"
    );
    quotient
}

#[cfg(test)]
mod tests {
    use super::Strategy;
    use super::*;
    use fsym_budget::{Budget, BudgetError, BudgetLimits, DIMENSION_COUNT, Unbounded};
    use proptest::prelude::*;
    use serde::de::DeserializeSeed;
    use serde::de::value::{
        Error as ValueError, I8Deserializer, SeqAccessDeserializer, U32Deserializer,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    struct HintSeq {
        values: std::vec::IntoIter<u32>,
        claimed: Option<usize>,
        next_calls: Rc<Cell<usize>>,
    }

    impl HintSeq {
        fn new(values: Vec<u32>, claimed: Option<usize>, next_calls: Rc<Cell<usize>>) -> Self {
            Self {
                values: values.into_iter(),
                claimed,
                next_calls,
            }
        }
    }

    impl<'de> serde::de::SeqAccess<'de> for HintSeq {
        type Error = ValueError;

        fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where
            T: DeserializeSeed<'de>,
        {
            let Some(value) = self.values.next() else {
                return Ok(None);
            };
            self.next_calls.set(self.next_calls.get() + 1);
            seed.deserialize(U32Deserializer::<ValueError>::new(value))
                .map(Some)
        }

        fn size_hint(&self) -> Option<usize> {
            self.claimed
        }
    }

    struct ObservedBigIntWireSeq {
        raw_sign: i8,
        digits: Option<HintSeq>,
        member_calls: Rc<Cell<usize>>,
    }

    impl<'de> serde::de::SeqAccess<'de> for ObservedBigIntWireSeq {
        type Error = ValueError;

        fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where
            T: DeserializeSeed<'de>,
        {
            let member = self.member_calls.get();
            self.member_calls.set(member + 1);
            match member {
                0 => seed
                    .deserialize(I8Deserializer::<ValueError>::new(self.raw_sign))
                    .map(Some),
                1 => {
                    let digits = self
                        .digits
                        .take()
                        .expect("magnitude requested at most once");
                    seed.deserialize(SeqAccessDeserializer::new(digits))
                        .map(Some)
                }
                _ => Ok(None),
            }
        }
    }

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

    #[derive(Debug, Default)]
    struct CountingMeter {
        dimensions: [u64; DIMENSION_COUNT],
        checkpoints: usize,
    }

    impl BudgetMeter for CountingMeter {
        fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
            self.charge_batch(&[(dimension, amount)])
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            let mut updated = self.dimensions;
            for &(dimension, amount) in charges {
                let slot = &mut updated[dimension.index()];
                *slot = slot.checked_add(amount).ok_or(MeterError::Budget(
                    BudgetError::ChargeOverflow { dimension },
                ))?;
            }
            self.dimensions = updated;
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints = self
                .checkpoints
                .checked_add(1)
                .expect("test checkpoint count must fit usize");
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
            self.checkpoints = self
                .checkpoints
                .checked_add(1)
                .expect("test checkpoint count must fit usize");
            self.trailing_uncharged_checkpoints = self
                .trailing_uncharged_checkpoints
                .checked_add(1)
                .expect("test trailing checkpoint count must fit usize");
            Ok(())
        }
    }

    fn assert_terminal_cancellation<T>(
        expected: &T,
        mut run: impl FnMut(&mut CancelAfter) -> Result<T, MeterError>,
    ) where
        T: std::fmt::Debug + PartialEq,
    {
        let mut baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        let baseline_result = run(&mut baseline);
        assert_eq!(baseline_result.as_ref(), Ok(expected));
        assert!(baseline.checkpoints > 0);

        let mut cancelled = CancelAfter {
            cancel_at_checkpoint: baseline.checkpoints,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(run(&mut cancelled), Err(MeterError::Cancelled));
        assert_eq!(cancelled.checkpoints, baseline.checkpoints);
        assert_eq!(cancelled.compute_steps, baseline.compute_steps);
    }

    fn assert_cancels_at_every_checkpoint<T>(
        mut run: impl FnMut(&mut CancelAfter) -> Result<T, MeterError>,
    ) where
        T: std::fmt::Debug + PartialEq,
    {
        let mut baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        run(&mut baseline).expect("baseline operation succeeds");
        assert!(baseline.checkpoints > 0);

        for cancel_at_checkpoint in 1..=baseline.checkpoints {
            let mut cancelled = CancelAfter {
                cancel_at_checkpoint,
                checkpoints: 0,
                compute_steps: 0,
            };
            assert_eq!(
                run(&mut cancelled),
                Err(MeterError::Cancelled),
                "checkpoint {cancel_at_checkpoint} did not stop publication"
            );
            assert_eq!(cancelled.checkpoints, cancel_at_checkpoint);
        }
    }

    fn assert_terminal_shape<T>(
        expected: &T,
        expected_trailing_checkpoints: usize,
        run: impl FnOnce(&mut TerminalProbe) -> Result<T, MeterError>,
    ) where
        T: std::fmt::Debug + PartialEq,
    {
        let mut meter = TerminalProbe::default();
        let result = run(&mut meter);
        assert_eq!(result.as_ref(), Ok(expected));
        assert!(meter.checkpoints >= expected_trailing_checkpoints);
        assert_eq!(
            meter.trailing_uncharged_checkpoints,
            expected_trailing_checkpoints
        );
    }

    #[test]
    fn serde_preserves_the_existing_canonical_sign_and_u32_digit_shape() {
        let negative_two_to_32_plus_one = -((&BigInt::one() << 32u32) + 1i64);
        for (value, expected) in [
            (BigInt::zero(), "[0,[]]"),
            (BigInt::one(), "[1,[1]]"),
            (negative_two_to_32_plus_one, "[-1,[1,1]]"),
        ] {
            let encoded = serde_json::to_string(&value).expect("canonical integer serializes");
            assert_eq!(encoded, expected);
            let decoded: BigInt =
                serde_json::from_str(&encoded).expect("canonical integer deserializes");
            assert_eq!(decoded, value);
            assert_eq!(
                serde_json::to_string(&decoded).expect("decoded integer reserializes"),
                encoded
            );
        }

        let low_zero_is_significant = "[1,[0,1]]";
        let decoded: BigInt = serde_json::from_str(low_zero_is_significant)
            .expect("a low zero digit is canonical and significant");
        assert_eq!(decoded, &BigInt::one() << 32u32);
        assert_eq!(
            serde_json::to_string(&decoded).expect("canonical integer reserializes"),
            low_zero_is_significant
        );
    }

    #[test]
    fn serde_rejects_noncanonical_or_malformed_sign_and_magnitude_tuples() {
        for malformed in [
            "[1,[]]",
            "[-1,[]]",
            "[0,[1]]",
            "[1,[0]]",
            "[-1,[1,0]]",
            "[2,[1]]",
            "[1,[4294967296]]",
            "[]",
            "[1]",
            "[1,[1],0]",
        ] {
            assert!(
                serde_json::from_str::<BigInt>(malformed).is_err(),
                "malformed wire was accepted: {malformed}"
            );
        }

        #[derive(serde::Deserialize)]
        struct WrappedBigInt {
            value: BigInt,
        }

        let wrapped: WrappedBigInt = serde_json::from_str(r#"{"value":[1,[7]]}"#)
            .expect("a nested canonical integer is admitted");
        assert_eq!(wrapped.value, BigInt::from(7));
        assert!(
            serde_json::from_str::<WrappedBigInt>(r#"{"value":[0,[7]]}"#).is_err(),
            "derived containers must inherit canonical integer admission"
        );
    }

    #[test]
    fn serde_digit_limit_handles_absent_underreported_and_hostile_hints() {
        for hostile_hint in [Some(4), Some(usize::MAX)] {
            let next_calls = Rc::new(Cell::new(0));
            let error = BoundedU32DigitsVisitor::<3>
                .visit_seq(HintSeq::new(
                    vec![1, 2, 3, 4],
                    hostile_hint,
                    Rc::clone(&next_calls),
                ))
                .expect_err("over-limit hint must fail before reading elements");
            assert!(error.to_string().contains("at most 3"));
            assert_eq!(next_calls.get(), 0);
        }

        for underreported_hint in [None, Some(1), Some(3)] {
            let next_calls = Rc::new(Cell::new(0));
            let error = BoundedU32DigitsVisitor::<3>
                .visit_seq(HintSeq::new(
                    vec![1, 2, 3, 4, 5, 6, 7],
                    underreported_hint,
                    Rc::clone(&next_calls),
                ))
                .expect_err("an underreported sequence must not bypass the logical limit");
            assert!(error.to_string().contains("at most 3"));
            assert_eq!(next_calls.get(), 4, "decoder must stop at limit plus one");
        }

        let next_calls = Rc::new(Cell::new(0));
        let digits = BoundedU32DigitsVisitor::<3>
            .visit_seq(HintSeq::new(vec![1, 2, 3], Some(3), Rc::clone(&next_calls)))
            .expect("the exact limit is admitted");
        assert_eq!(digits, [1, 2, 3]);
        assert_eq!(next_calls.get(), 3);
    }

    #[test]
    fn serde_validates_sign_before_reading_or_allocating_the_magnitude() {
        let member_calls = Rc::new(Cell::new(0));
        let digit_calls = Rc::new(Cell::new(0));
        let error = BigIntWireVisitor
            .visit_seq(ObservedBigIntWireSeq {
                raw_sign: 2,
                digits: Some(HintSeq::new(
                    vec![1, 2, 3, 4],
                    Some(usize::MAX),
                    Rc::clone(&digit_calls),
                )),
                member_calls: Rc::clone(&member_calls),
            })
            .expect_err("an invalid sign must be refused immediately");
        assert!(error.to_string().contains("-1, 0, or 1"));
        assert_eq!(member_calls.get(), 1);
        assert_eq!(digit_calls.get(), 0);

        let member_calls = Rc::new(Cell::new(0));
        let digit_calls = Rc::new(Cell::new(0));
        let error = BigIntWireVisitor
            .visit_seq(ObservedBigIntWireSeq {
                raw_sign: 0,
                digits: Some(HintSeq::new(
                    vec![1, 2, 3, 4],
                    None,
                    Rc::clone(&digit_calls),
                )),
                member_calls: Rc::clone(&member_calls),
            })
            .expect_err("a zero sign must refuse a nonempty magnitude immediately");
        assert!(error.to_string().contains("at most 0"));
        assert_eq!(member_calls.get(), 2);
        assert_eq!(digit_calls.get(), 1);
    }

    #[test]
    fn serde_absent_hint_capacity_growth_starts_small_and_doubles() {
        assert_eq!(next_serde_digit_capacity(0, 10), Some(1));
        assert_eq!(next_serde_digit_capacity(1, 10), Some(2));
        assert_eq!(next_serde_digit_capacity(2, 10), Some(4));
        assert_eq!(next_serde_digit_capacity(8, 10), Some(10));
        assert_eq!(next_serde_digit_capacity(10, 10), None);
    }

    #[test]
    fn serde_digit_reservation_failure_is_typed_instead_of_panicking() {
        let mut digits = Vec::<u32>::new();
        let error = try_reserve_serde_digits::<ValueError>(&mut digits, usize::MAX)
            .expect_err("an impossible reservation must fail");
        assert!(error.to_string().contains("allocation refused"));
        assert!(digits.is_empty());
    }

    fn repeated_digit_wire(count: usize) -> String {
        let mut encoded = String::with_capacity(count.saturating_mul(2).saturating_add(6));
        encoded.push_str("[1,[");
        for index in 0..count {
            if index != 0 {
                encoded.push(',');
            }
            encoded.push('1');
        }
        encoded.push_str("]]");
        encoded
    }

    #[test]
    fn public_serde_digit_limit_is_an_exact_boundary_without_a_json_hint() {
        let at_limit_wire = repeated_digit_wire(MAX_SERDE_U32_DIGITS);
        let at_limit: BigInt =
            serde_json::from_str(&at_limit_wire).expect("the exact digit limit is admitted");
        assert_eq!(
            serde_json::to_string(&at_limit).expect("limit value reserializes"),
            at_limit_wire
        );

        let over_limit_wire = repeated_digit_wire(MAX_SERDE_U32_DIGITS + 1);
        let error = serde_json::from_str::<BigInt>(&over_limit_wire)
            .expect_err("limit plus one must be refused");
        assert!(
            error
                .to_string()
                .contains(&MAX_SERDE_U32_DIGITS.to_string())
        );

        let first_over_limit_bit = u32::try_from(
            MAX_SERDE_U32_DIGITS
                .checked_mul(u32::BITS as usize)
                .expect("test limit product fits usize"),
        )
        .expect("test shift fits u32");
        let over_limit_value = BigInt::one() << first_over_limit_bit;
        let error = serde_json::to_string(&over_limit_value)
            .expect_err("serializer must not emit values its paired decoder refuses");
        assert!(
            error
                .to_string()
                .contains(&MAX_SERDE_U32_DIGITS.to_string())
        );
    }

    #[test]
    fn governed_power_matches_exact_signed_boundaries_and_preflights_extreme_control() {
        for base in -3i64..=3 {
            let base = BigInt::from(base);
            for exponent in 0u32..=12 {
                let expected = BigInt::pow(&base, exponent);
                let mut meter = CountingMeter::default();
                assert_eq!(metered_pow(&base, exponent, &mut meter), Ok(expected));
            }
        }
        assert_eq!(
            metered_pow(&BigInt::from(2), 10, &mut Unbounded),
            Ok(BigInt::from(1_024))
        );
        assert_eq!(
            metered_pow(&BigInt::from(-2), 11, &mut Unbounded),
            Ok(BigInt::from(-2_048))
        );

        let mut budget = Budget::new(BudgetLimits::uniform(31, 0));
        let before = budget.snapshot();
        assert_eq!(
            metered_pow(&BigInt::from(2), u32::MAX, &mut budget),
            Err(MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                requested: 32,
                remaining: 31,
            }))
        );
        assert_eq!(budget.snapshot(), before);

        // 3^13 follows four binary-control iterations and six one-limb multiply/square calls.
        // These exact charges kill an opaque substrate-pow mutant hidden between safe points.
        let mut topology_meter = CountingMeter::default();
        assert_eq!(
            metered_pow(&BigInt::from(3), 13, &mut topology_meter),
            Ok(BigInt::from(1_594_323))
        );
        assert_eq!(
            topology_meter.dimensions[Dimension::ComputeSteps.index()],
            22
        );
        assert_eq!(
            topology_meter.dimensions[Dimension::AllocationCount.index()],
            20
        );
    }

    #[test]
    fn floor_square_root_pins_exact_square_boundaries_and_governed_lane() {
        let negative = BigInt::from(-1);
        assert_eq!(negative.sqrt(), None);
        assert_eq!(sqrt_floor(&negative), None);
        assert_eq!(metered_sqrt_floor(&negative, &mut Unbounded), Ok(None));

        for (value, expected) in [
            (0i64, 0i64),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (15, 3),
            (16, 4),
            (17, 4),
        ] {
            let value = BigInt::from(value);
            let expected = BigInt::from(expected);
            assert_eq!(value.sqrt(), Some(expected.clone()));
            assert_eq!(sqrt_floor(&value), Some(expected.clone()));
            assert_eq!(
                metered_sqrt_floor(&value, &mut Unbounded),
                Ok(Some(expected))
            );
        }

        let root = BigInt::one() << 256u32;
        let square = &root * &root;
        assert_eq!(sqrt_floor(&(&square - 1i64)), Some(&root - 1i64));
        assert_eq!(sqrt_floor(&square), Some(root.clone()));
        assert_eq!(sqrt_floor(&(&square + 1i64)), Some(root));
    }

    #[test]
    fn governed_power_and_root_cancel_at_every_observed_safe_point() {
        let base = BigInt::from(3);
        assert_cancels_at_every_checkpoint(|meter| metered_pow(&base, 13, meter));

        let radicand = BigInt::from(15);
        assert_cancels_at_every_checkpoint(|meter| metered_sqrt_floor(&radicand, meter));
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
            Strategy::Toom3,
            Strategy::NativeSubstrate,
        ] {
            assert_eq!(multiply_with_strategy(&x, &zero, strategy), zero);
            assert_eq!(multiply_with_strategy(&zero, &x, strategy), zero);
        }
    }

    fn from_base_chunks(c0: &BigInt, c1: &BigInt, c2: &BigInt, chunk_bits: u32) -> BigInt {
        c0 + (c1 << chunk_bits) + (c2 << (2 * chunk_bits))
    }

    fn signed_digits_from_i128(value: i128) -> SignedDigits {
        let negative = value.is_negative();
        let mut magnitude = value.unsigned_abs();
        let mut digits = Vec::new();
        while magnitude != 0 {
            digits.push(
                u32::try_from(magnitude & u128::from(u32::MAX))
                    .expect("masked test digit must fit u32"),
            );
            magnitude >>= 32;
        }
        SignedDigits::new(negative, digits)
    }

    fn signed_digits_to_i128(value: &SignedDigits) -> i128 {
        let mut magnitude = 0u128;
        for &digit in value.digits.iter().rev() {
            magnitude = (magnitude << 32) | u128::from(digit);
        }
        let magnitude = i128::try_from(magnitude).expect("test magnitude must fit i128");
        if value.negative {
            -magnitude
        } else {
            magnitude
        }
    }

    fn actual_metered_toom_operands() -> (BigInt, BigInt) {
        let chunk_bits = 160;
        let high = (BigInt::one() << 64) + 5i64;
        let twice_high = &high + &high;
        let three_high = &twice_high + &high;
        // a(-1) is negative while b(-1) is positive, exercising a signed recursive product.
        let a = from_base_chunks(&high, &three_high, &high, chunk_bits);
        let b = from_base_chunks(
            &(&high + 11i64),
            &(&twice_high + 1i64),
            &(&high + 17i64),
            chunk_bits,
        );
        (a, b)
    }

    fn actual_metered_ntt_operands() -> (BigInt, BigInt) {
        let a = (BigInt::one() << 255)
            + (BigInt::one() << 193)
            + (BigInt::one() << 127)
            + (BigInt::one() << 65)
            + 65_537i64;
        let b = (BigInt::one() << 247)
            + (BigInt::one() << 181)
            + (BigInt::one() << 113)
            + (BigInt::one() << 47)
            + 17i64;
        (a, b)
    }

    fn assert_explicit_multiplication_lanes_agree(a: &BigInt, b: &BigInt) {
        let reference = multiply_with_strategy(a, b, Strategy::SchoolbookReference);
        assert_eq!(multiply_with_strategy(a, b, Strategy::Karatsuba), reference);
        assert_eq!(multiply_with_strategy(a, b, Strategy::Toom3), reference);
        assert_eq!(
            multiply_with_strategy(a, b, Strategy::NativeSubstrate),
            reference
        );
        assert_eq!(multiply_with_strategy(b, a, Strategy::Toom3), reference);
    }

    #[test]
    fn toom3_interpolation_handles_signed_evaluations_and_carries() {
        let chunk_bits = 129;
        let top = BigInt::one() << 126;
        let chunk_max = (BigInt::one() << chunk_bits) - 1i64;

        // a(-1) == 0 while both operands retain nonzero top thirds.
        let minus_one_zero = from_base_chunks(&3.into(), &(&top + 3i64), &top, chunk_bits);
        let ordinary = from_base_chunks(&11.into(), &17.into(), &(&top + 5i64), chunk_bits);

        // a(-1) < 0 exercises signed evaluation products and exact interpolation.
        let minus_one_negative =
            from_base_chunks(&1.into(), &chunk_max, &(&top + 7i64), chunk_bits);

        // Every chunk is dense, forcing cross-chunk carries and a nonzero cubic coefficient.
        let all_ones = from_base_chunks(&chunk_max, &chunk_max, &chunk_max, chunk_bits);

        for (a, b) in [
            (&minus_one_zero, &ordinary),
            (&minus_one_negative, &ordinary),
            (&all_ones, &minus_one_negative),
        ] {
            for (neg_a, neg_b) in [(false, false), (true, false), (false, true), (true, true)] {
                let signed_a = if neg_a { -a } else { a.clone() };
                let signed_b = if neg_b { -b } else { b.clone() };
                assert_explicit_multiplication_lanes_agree(&signed_a, &signed_b);
            }
        }
    }

    #[test]
    fn toom3_structural_cutoff_boundaries_match_reference() {
        for bits in [
            TOOM3_RECURSION_CUTOFF_BITS - 1,
            TOOM3_RECURSION_CUTOFF_BITS,
            TOOM3_RECURSION_CUTOFF_BITS + 1,
        ] {
            let shift = u32::try_from(bits - 1).unwrap();
            let a = (BigInt::one() << shift) + (BigInt::one() << (shift / 2)) + 65_537i64;
            let b = (BigInt::one() << shift) + (BigInt::one() << (shift / 3)) + 17i64;
            assert_explicit_multiplication_lanes_agree(&a, &b);
        }
    }

    #[test]
    fn toom3_nested_recursion_matches_reference() {
        // Dense 1,281-bit operands make each at-two evaluation exceed the 384-bit structural
        // cutoff, so the product exercises a second Toom level rather than only leaf fallback.
        let dense = (BigInt::one() << 1_281) - 1i64;
        let companion = &dense - (BigInt::one() << 637) - 65_537i64;
        assert_explicit_multiplication_lanes_agree(&dense, &companion);
    }

    #[test]
    fn toom3_skew_boundary_falls_back_only_without_a_top_third() {
        let wide = (BigInt::one() << 599) + (BigInt::one() << 311) + 17i64;
        // max_bits=600 gives 200-bit chunks. The 400-bit operand has no top-third chunk,
        // while the 401-bit operand has the smallest possible nonzero top third.
        let at_fallback_boundary = (BigInt::one() << 399) + 65_537i64;
        let above_fallback_boundary = (BigInt::one() << 400) + 65_537i64;
        assert_explicit_multiplication_lanes_agree(&wide, &at_fallback_boundary);
        assert_explicit_multiplication_lanes_agree(&wide, &above_fallback_boundary);
    }

    #[test]
    fn karatsuba_refuses_zero_heavy_recursion_for_skewed_operands() {
        let huge = (BigInt::one() << 32_768) + (BigInt::one() << 16_383) + 1i64;
        let tiny = BigInt::from(-65_537);
        let expected = multiply_with_strategy(&huge, &tiny, Strategy::NativeSubstrate);
        assert_eq!(select_strategy(huge.bits()), Strategy::Karatsuba);
        assert_eq!(
            multiply_with_strategy(&huge, &tiny, Strategy::Karatsuba),
            expected
        );
        assert_eq!(
            multiply_with_strategy(&huge, &tiny, Strategy::Toom3),
            expected
        );
        assert_eq!(&huge * &tiny, expected);
    }

    #[test]
    #[should_panic(expected = "interpolation division must be exact")]
    fn toom3_exact_division_helper_rejects_truncation() {
        let _ = exact_div_small(Substrate::from(5), 3);
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
    fn metered_karatsuba_candidate_covers_sign_cutoff_skew_and_nested_boundaries() {
        let five_digits = (BigInt::one() << 160) - 1i64;
        let cases = [
            (BigInt::zero(), five_digits.clone()),
            (BigInt::from(-17), BigInt::from(29)),
            (
                (BigInt::one() << 127) + 65_537i64,
                -((BigInt::one() << 126) + 17i64),
            ),
            (
                five_digits.clone(),
                -((BigInt::one() << 159) + (BigInt::one() << 65) + 29i64),
            ),
            (
                (BigInt::one() << 640) + (BigInt::one() << 319) + 1i64,
                BigInt::from(-65_537),
            ),
            (
                (BigInt::one() << 288) - 1i64,
                -((BigInt::one() << 287) + (BigInt::one() << 129) + 1i64),
            ),
        ];

        for (a, b) in cases {
            let mut meter = CountingMeter::default();
            let candidate = metered_karatsuba_candidate(&a, &b, &mut meter).unwrap();
            assert_eq!(candidate.is_zero(), a.is_zero() || b.is_zero());
            assert_eq!(
                candidate.is_negative(),
                !candidate.is_zero() && (a.is_negative() != b.is_negative())
            );
            assert_ne!(candidate.digits_le().last(), Some(&0));
            let result = candidate.materialize_unmetered();
            assert_eq!(
                result,
                multiply_with_strategy(&a, &b, Strategy::NativeSubstrate)
            );
        }

        let four_digits = (BigInt::one() << 128) - 1i64;
        let mut cutoff_meter = CountingMeter::default();
        metered_karatsuba_candidate(&four_digits, &four_digits, &mut cutoff_meter).unwrap();
        assert_eq!(cutoff_meter.dimensions[Dimension::DepthLimit.index()], 1);

        let mut one_level_meter = CountingMeter::default();
        metered_karatsuba_candidate(&five_digits, &five_digits, &mut one_level_meter).unwrap();
        assert_eq!(
            one_level_meter.dimensions[Dimension::DepthLimit.index()],
            2,
            "five base-2^32 digits must enter exactly one recursive level"
        );

        let skewed = (BigInt::one() << 640) + 1i64;
        let mut skew_meter = CountingMeter::default();
        metered_karatsuba_candidate(&skewed, &BigInt::from(65_537), &mut skew_meter).unwrap();
        assert_eq!(skew_meter.dimensions[Dimension::DepthLimit.index()], 1);

        let nested = (BigInt::one() << 288) - 1i64;
        let mut nested_meter = CountingMeter::default();
        metered_karatsuba_candidate(&nested, &nested, &mut nested_meter).unwrap();
        assert_eq!(
            nested_meter.dimensions[Dimension::DepthLimit.index()],
            3,
            "nested fixture must kill both all-leaf and depth-overcharge mutants"
        );
    }

    #[test]
    fn metered_karatsuba_candidate_exact_budgets_and_one_short_refusals() {
        let a = (BigInt::one() << 160) - 1i64;
        let b = -((BigInt::one() << 159) + (BigInt::one() << 97) + 65_537i64);
        let expected = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
        let mut count = CountingMeter::default();
        assert_eq!(
            metered_karatsuba_candidate(&a, &b, &mut count)
                .unwrap()
                .materialize_unmetered(),
            expected
        );
        assert_eq!(count.dimensions[Dimension::RandomDraws.index()], 0);

        let exact_limits = BudgetLimits {
            dimensions: count.dimensions,
            verifier_pool: 0,
        };
        let mut exact = Budget::new(exact_limits);
        assert_eq!(
            metered_karatsuba_candidate(&a, &b, &mut exact)
                .unwrap()
                .materialize_unmetered(),
            expected
        );
        for dimension in Dimension::ALL {
            assert_eq!(exact.remaining(dimension), 0);
        }

        for dimension in Dimension::ALL {
            let used = count.dimensions[dimension.index()];
            if used == 0 {
                continue;
            }
            let mut short_limits = exact_limits;
            short_limits.dimensions[dimension.index()] = used - 1;
            let mut short = Budget::new(short_limits);
            assert!(
                matches!(
                    metered_karatsuba_candidate(&a, &b, &mut short),
                    Err(MeteredMultiplyError::Meter(MeterError::Budget(
                        BudgetError::Exhausted {
                            dimension: observed,
                            ..
                        }
                    ))) if observed == dimension
                ),
                "one-unit-short {dimension} allowance must refuse without a value"
            );
        }
    }

    #[test]
    fn metered_karatsuba_candidate_cancels_at_every_safe_point() {
        let a = (BigInt::one() << 288) - 1i64;
        let b = (BigInt::one() << 287) + (BigInt::one() << 129) + 29i64;
        let expected = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
        let mut baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(
            metered_karatsuba_candidate(&a, &b, &mut baseline)
                .unwrap()
                .materialize_unmetered(),
            expected
        );
        assert!(baseline.checkpoints > 1);

        for cancel_at_checkpoint in 1..=baseline.checkpoints {
            let mut cancelled = CancelAfter {
                cancel_at_checkpoint,
                checkpoints: 0,
                compute_steps: 0,
            };
            assert_eq!(
                metered_karatsuba_candidate(&a, &b, &mut cancelled),
                Err(MeteredMultiplyError::Meter(MeterError::Cancelled)),
                "checkpoint {cancel_at_checkpoint} must not publish a partial value"
            );
            assert_eq!(cancelled.checkpoints, cancel_at_checkpoint);
        }
    }

    #[test]
    fn metered_karatsuba_capacity_preflight_reports_size_overflow() {
        assert_eq!(
            checked_product_capacity(usize::MAX, 1),
            Err(MeteredMultiplyError::SizeOverflow)
        );
    }

    #[test]
    fn candidate_digit_buffers_charge_independently_derived_capacities() {
        let mut schoolbook_meter = CountingMeter::default();
        let product = metered_schoolbook_digits(
            &[u32::MAX, u32::MAX],
            &[u32::MAX, u32::MAX],
            4,
            &mut schoolbook_meter,
        )
        .unwrap();
        assert_eq!(product.len(), 4);
        assert_eq!(
            schoolbook_meter.dimensions[Dimension::MemoryBytes.index()],
            4 * 4
        );
        assert_eq!(
            schoolbook_meter.dimensions[Dimension::AllocationCount.index()],
            1
        );

        let mut addition_meter = CountingMeter::default();
        assert_eq!(
            metered_add_digit_slices(&[u32::MAX, 1], &[1, 1], &mut addition_meter).unwrap(),
            vec![0, 3]
        );
        assert_eq!(
            addition_meter.dimensions[Dimension::MemoryBytes.index()],
            3 * 4
        );
        assert_eq!(
            addition_meter.dimensions[Dimension::AllocationCount.index()],
            1
        );
    }

    #[test]
    fn metered_toom3_evaluation_and_interpolation_match_direct_convolution() {
        let mut meter = CountingMeter::default();
        let values = metered_evaluate_toom3_chunks(&[1], &[9], &[2], &mut meter).unwrap();
        assert_eq!(values.at_one, vec![12]);
        assert_eq!(signed_digits_to_i128(&values.at_minus_one), -6);
        assert_eq!(values.at_two, vec![27]);

        let zero_at_minus_one =
            metered_evaluate_toom3_chunks(&[1], &[2], &[1], &mut meter).unwrap();
        assert_eq!(signed_digits_to_i128(&zero_at_minus_one.at_minus_one), 0);

        let a = [1i128, 9, 2];
        let b = [3i128, 4, 5];
        let evaluate = |coefficients: [i128; 3], point: i128| {
            coefficients[0] + coefficients[1] * point + coefficients[2] * point * point
        };
        let coefficients = metered_interpolate_toom3(
            signed_digits_from_i128(a[0] * b[0]),
            signed_digits_from_i128(evaluate(a, 1) * evaluate(b, 1)),
            signed_digits_from_i128(evaluate(a, -1) * evaluate(b, -1)),
            signed_digits_from_i128(evaluate(a, 2) * evaluate(b, 2)),
            signed_digits_from_i128(a[2] * b[2]),
            &mut meter,
        )
        .unwrap();
        assert_eq!(coefficients.c0, vec![3]);
        assert_eq!(coefficients.c1, vec![31]);
        assert_eq!(coefficients.c2, vec![47]);
        assert_eq!(coefficients.c3, vec![53]);
        assert_eq!(coefficients.c4, vec![10]);

        assert!(matches!(
            metered_interpolate_toom3(
                signed_digits_from_i128(10),
                signed_digits_from_i128(9),
                signed_digits_from_i128(11),
                signed_digits_from_i128(8),
                signed_digits_from_i128(0),
                &mut meter,
            ),
            Err(MeteredMultiplyError::InvariantViolation)
        ));
    }

    #[test]
    fn metered_toom3_exact_small_division_is_signed_and_fail_closed() {
        let mut meter = CountingMeter::default();
        let quotient =
            metered_exact_divide_signed_small(signed_digits_from_i128(-9), 3, &mut meter).unwrap();
        assert_eq!(signed_digits_to_i128(&quotient), -3);
        assert!(matches!(
            metered_exact_divide_signed_small(signed_digits_from_i128(5), 3, &mut meter,),
            Err(MeteredMultiplyError::InvariantViolation)
        ));
        assert!(matches!(
            metered_exact_divide_signed_small(signed_digits_from_i128(-5), 2, &mut meter,),
            Err(MeteredMultiplyError::InvariantViolation)
        ));
        assert!(matches!(
            metered_exact_divide_signed_small(signed_digits_from_i128(0), 0, &mut meter,),
            Err(MeteredMultiplyError::InvariantViolation)
        ));
    }

    #[test]
    fn metered_toom3_candidate_covers_real_branch_signs_and_canonicality() {
        let (a, b) = actual_metered_toom_operands();
        for (negative_a, negative_b) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let signed_a = if negative_a { -&a } else { a.clone() };
            let signed_b = if negative_b { -&b } else { b.clone() };
            let expected = multiply_with_strategy(&signed_a, &signed_b, Strategy::NativeSubstrate);
            let mut meter = CountingMeter::default();
            let (candidate, toom_nodes) =
                metered_toom3_candidate_inner(&signed_a, &signed_b, &mut meter).unwrap();
            assert!(
                toom_nodes > 0,
                "fixture must enter the Toom interpolation body"
            );
            assert!(!candidate.is_zero());
            assert_eq!(candidate.is_negative(), negative_a != negative_b);
            assert_ne!(candidate.digits_le().last(), Some(&0));
            assert_eq!(candidate.materialize_unmetered(), expected);
            assert_eq!(meter.dimensions[Dimension::RandomDraws.index()], 0);
        }

        let mut meter = CountingMeter::default();
        let (zero, toom_nodes) =
            metered_toom3_candidate_inner(&BigInt::zero(), &b, &mut meter).unwrap();
        assert!(zero.is_zero());
        assert!(!zero.is_negative());
        assert_eq!(toom_nodes, 0);
    }

    #[test]
    fn metered_toom3_structural_cutoff_skew_and_nested_boundaries_are_exercised() {
        let cutoff = (BigInt::one() << 384) - 1i64;
        let mut cutoff_meter = CountingMeter::default();
        let (_, cutoff_nodes) =
            metered_toom3_candidate_inner(&cutoff, &cutoff, &mut cutoff_meter).unwrap();
        assert_eq!(cutoff_nodes, 0);

        let thirteen_digits = (BigInt::one() << 416) - 1i64;
        let ten_digits = (BigInt::one() << 320) - 1i64;
        let eleven_digits = (BigInt::one() << 321) - 1i64;
        let mut skew_meter = CountingMeter::default();
        let (_, skew_nodes) =
            metered_toom3_candidate_inner(&thirteen_digits, &ten_digits, &mut skew_meter).unwrap();
        assert_eq!(skew_nodes, 0, "ten digits are exactly the 2k skew boundary");

        let mut above_skew_meter = CountingMeter::default();
        let (_, above_skew_nodes) =
            metered_toom3_candidate_inner(&thirteen_digits, &eleven_digits, &mut above_skew_meter)
                .unwrap();
        assert!(
            above_skew_nodes > 0,
            "eleven digits provide a nonzero top third"
        );

        let nested = (BigInt::one() << 1_280) - 1i64;
        let mut nested_meter = CountingMeter::default();
        let (nested_candidate, nested_nodes) =
            metered_toom3_candidate_inner(&nested, &nested, &mut nested_meter).unwrap();
        assert_eq!(
            nested_candidate.materialize_unmetered(),
            multiply_with_strategy(&nested, &nested, Strategy::NativeSubstrate)
        );
        assert!(
            nested_nodes > 1,
            "dense forty-digit input must recurse through Toom"
        );
        assert_eq!(
            nested_meter.dimensions[Dimension::DepthLimit.index()],
            5,
            "nested fixture must count its Toom and metered Karatsuba multiplication depths"
        );

        let mut nested_baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        metered_toom3_candidate(&nested, &nested, &mut nested_baseline).unwrap();
        let cancel_at_checkpoint = nested_baseline.checkpoints * 3 / 4;
        let mut nested_cancelled = CancelAfter {
            cancel_at_checkpoint,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(
            metered_toom3_candidate(&nested, &nested, &mut nested_cancelled),
            Err(MeteredMultiplyError::Meter(MeterError::Cancelled))
        );
        assert_eq!(nested_cancelled.checkpoints, cancel_at_checkpoint);
        assert!(nested_cancelled.compute_steps < nested_baseline.compute_steps);
    }

    #[test]
    fn metered_toom3_candidate_exact_budgets_and_one_short_refusals() {
        let (a, b) = actual_metered_toom_operands();
        let expected = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
        let mut count = CountingMeter::default();
        let (candidate, toom_nodes) = metered_toom3_candidate_inner(&a, &b, &mut count).unwrap();
        assert!(toom_nodes > 0);
        assert_eq!(candidate.materialize_unmetered(), expected);
        assert_eq!(count.dimensions[Dimension::RandomDraws.index()], 0);

        let exact_limits = BudgetLimits {
            dimensions: count.dimensions,
            verifier_pool: 0,
        };
        let mut exact = Budget::new(exact_limits);
        assert_eq!(
            metered_toom3_candidate(&a, &b, &mut exact)
                .unwrap()
                .materialize_unmetered(),
            expected
        );
        for dimension in Dimension::ALL {
            assert_eq!(exact.remaining(dimension), 0);
        }

        for dimension in Dimension::ALL {
            let used = count.dimensions[dimension.index()];
            if used == 0 {
                continue;
            }
            let mut short_limits = exact_limits;
            short_limits.dimensions[dimension.index()] = used - 1;
            let mut short = Budget::new(short_limits);
            assert!(
                matches!(
                    metered_toom3_candidate(&a, &b, &mut short),
                    Err(MeteredMultiplyError::Meter(MeterError::Budget(
                        BudgetError::Exhausted {
                            dimension: observed,
                            ..
                        }
                    ))) if observed == dimension
                ),
                "one-unit-short {dimension} allowance must refuse without a value"
            );
        }
    }

    #[test]
    fn metered_toom3_candidate_cancels_at_every_safe_point() {
        let (a, b) = actual_metered_toom_operands();
        let expected = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
        let mut baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(
            metered_toom3_candidate(&a, &b, &mut baseline)
                .unwrap()
                .materialize_unmetered(),
            expected
        );
        assert!(baseline.checkpoints > 1);

        for cancel_at_checkpoint in 1..=baseline.checkpoints {
            let mut cancelled = CancelAfter {
                cancel_at_checkpoint,
                checkpoints: 0,
                compute_steps: 0,
            };
            assert_eq!(
                metered_toom3_candidate(&a, &b, &mut cancelled),
                Err(MeteredMultiplyError::Meter(MeterError::Cancelled)),
                "checkpoint {cancel_at_checkpoint} must not publish a partial Toom candidate"
            );
            assert_eq!(cancelled.checkpoints, cancel_at_checkpoint);
        }
    }

    #[test]
    fn metered_toom3_helper_allocations_and_recombination_are_explicit() {
        let mut scale_meter = CountingMeter::default();
        assert_eq!(
            metered_multiply_digit_slice_by_small(&[u32::MAX, u32::MAX], 16, &mut scale_meter,)
                .unwrap(),
            vec![u32::MAX - 15, u32::MAX, 15]
        );
        assert_eq!(
            scale_meter.dimensions[Dimension::MemoryBytes.index()],
            3 * 4
        );
        assert_eq!(
            scale_meter.dimensions[Dimension::AllocationCount.index()],
            1
        );

        let mut division_meter = CountingMeter::default();
        let quotient = metered_exact_divide_signed_small(
            SignedDigits::new(false, vec![0, 3]),
            3,
            &mut division_meter,
        )
        .unwrap();
        assert_eq!(quotient.digits, vec![0, 1]);
        assert_eq!(division_meter.dimensions[Dimension::MemoryBytes.index()], 0);
        assert_eq!(
            division_meter.dimensions[Dimension::AllocationCount.index()],
            0
        );

        let mut recombination_meter = CountingMeter::default();
        let recombined = metered_recombine_toom3(
            ToomCoefficients {
                c0: vec![1],
                c1: vec![1],
                c2: vec![1],
                c3: vec![1],
                c4: vec![1],
            },
            [0, 1, 2, 3, 4],
            5,
            &mut recombination_meter,
        )
        .unwrap();
        assert_eq!(recombined, vec![1, 1, 1, 1, 1]);
        assert_eq!(
            recombination_meter.dimensions[Dimension::MemoryBytes.index()],
            5 * 4
        );
        assert_eq!(
            recombination_meter.dimensions[Dimension::AllocationCount.index()],
            1
        );

        let mut carry_meter = CountingMeter::default();
        assert_eq!(
            metered_recombine_toom3(
                ToomCoefficients {
                    c0: vec![u32::MAX],
                    c1: vec![1],
                    c2: Vec::new(),
                    c3: Vec::new(),
                    c4: Vec::new(),
                },
                [0, 0, 0, 0, 0],
                2,
                &mut carry_meter,
            )
            .unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            metered_recombine_toom3(
                ToomCoefficients {
                    c0: vec![1],
                    c1: Vec::new(),
                    c2: Vec::new(),
                    c3: Vec::new(),
                    c4: vec![1],
                },
                [0, 1, 2, 3, usize::MAX],
                5,
                &mut CountingMeter::default(),
            ),
            Err(MeteredMultiplyError::SizeOverflow)
        );
    }

    #[test]
    fn metered_ntt_crt_candidate_executes_exact_transform_for_all_signs() {
        let (a, b) = actual_metered_ntt_operands();
        for (negative_a, negative_b) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let signed_a = if negative_a { -&a } else { a.clone() };
            let signed_b = if negative_b { -&b } else { b.clone() };
            let expected = multiply_with_strategy(&signed_a, &signed_b, Strategy::NativeSubstrate);
            let mut meter = CountingMeter::default();
            let candidate = metered_ntt_crt_candidate(&signed_a, &signed_b, &mut meter).unwrap();
            assert!(!candidate.is_zero());
            assert_eq!(candidate.is_negative(), negative_a != negative_b);
            assert_ne!(candidate.digits_le().last(), Some(&0));
            assert_eq!(candidate.materialize_unmetered(), expected);
            assert!(meter.dimensions[Dimension::ComputeSteps.index()] > 0);
            assert!(meter.dimensions[Dimension::MemoryBytes.index()] > 0);
            assert!(meter.dimensions[Dimension::AllocationCount.index()] > 0);
            assert_eq!(meter.dimensions[Dimension::DepthLimit.index()], 0);
            assert_eq!(meter.dimensions[Dimension::RandomDraws.index()], 0);
        }
    }

    #[test]
    fn metered_ntt_crt_candidate_handles_dense_carry_and_canonical_zero() {
        let dense = (BigInt::one() << 2_048) - 1i64;
        let companion = &dense - (BigInt::one() << 1_023) - 65_537i64;
        let expected = multiply_with_strategy(&dense, &companion, Strategy::NativeSubstrate);
        let candidate = metered_ntt_crt_candidate(&dense, &companion, &mut Unbounded).unwrap();
        assert_ne!(candidate.digits_le().last(), Some(&0));
        assert_eq!(candidate.materialize_unmetered(), expected);

        let zero = metered_ntt_crt_candidate(&BigInt::zero(), &dense, &mut Unbounded).unwrap();
        assert!(zero.is_zero());
        assert!(!zero.is_negative());
    }

    #[test]
    fn metered_ntt_crt_preflight_refuses_unsupported_domain_without_allocation() {
        assert_eq!(
            ntt::preflight_u32_lengths(ntt::MAX_TRANSFORM_LENGTH / 2, 1),
            Err(MeteredMultiplyError::TransformDomainUnsupported)
        );
        assert_eq!(
            ntt::preflight_u32_lengths(usize::MAX, 1),
            Err(MeteredMultiplyError::SizeOverflow)
        );
    }

    #[test]
    fn metered_ntt_crt_candidate_exact_budgets_and_one_short_refusals() {
        let (a, b) = actual_metered_ntt_operands();
        let expected = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
        let mut count = CountingMeter::default();
        assert_eq!(
            metered_ntt_crt_candidate(&a, &b, &mut count)
                .unwrap()
                .materialize_unmetered(),
            expected
        );

        let exact_limits = BudgetLimits {
            dimensions: count.dimensions,
            verifier_pool: 0,
        };
        let mut exact = Budget::new(exact_limits);
        assert_eq!(
            metered_ntt_crt_candidate(&a, &b, &mut exact)
                .unwrap()
                .materialize_unmetered(),
            expected
        );
        for dimension in Dimension::ALL {
            assert_eq!(exact.remaining(dimension), 0);
        }

        for dimension in Dimension::ALL {
            let used = count.dimensions[dimension.index()];
            if used == 0 {
                continue;
            }
            let mut short_limits = exact_limits;
            short_limits.dimensions[dimension.index()] = used - 1;
            let mut short = Budget::new(short_limits);
            assert!(
                matches!(
                    metered_ntt_crt_candidate(&a, &b, &mut short),
                    Err(MeteredMultiplyError::Meter(MeterError::Budget(
                        BudgetError::Exhausted {
                            dimension: observed,
                            ..
                        }
                    ))) if observed == dimension
                ),
                "one-unit-short {dimension} allowance must refuse without a value"
            );
        }
    }

    #[test]
    fn metered_ntt_crt_candidate_cancels_at_every_observed_safe_point() {
        let (a, b) = actual_metered_ntt_operands();
        let expected = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
        let mut baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(
            metered_ntt_crt_candidate(&a, &b, &mut baseline)
                .unwrap()
                .materialize_unmetered(),
            expected
        );
        assert!(baseline.checkpoints > 1);

        for cancel_at_checkpoint in 1..=baseline.checkpoints {
            let mut cancelled = CancelAfter {
                cancel_at_checkpoint,
                checkpoints: 0,
                compute_steps: 0,
            };
            assert_eq!(
                metered_ntt_crt_candidate(&a, &b, &mut cancelled),
                Err(MeteredMultiplyError::Meter(MeterError::Cancelled)),
                "checkpoint {cancel_at_checkpoint} must not publish a partial NTT candidate"
            );
            assert_eq!(cancelled.checkpoints, cancel_at_checkpoint);
        }
    }

    #[test]
    fn metered_addition_and_subtraction_match_signed_native_lanes() {
        let values = [
            BigInt::zero(),
            BigInt::one(),
            BigInt::from(-1),
            (BigInt::one() << 257) + 65_537i64,
            -((BigInt::one() << 263) + 17i64),
        ];
        for lhs in &values {
            for rhs in &values {
                let mut meter = Unbounded;
                assert_eq!(metered_add(lhs, rhs, &mut meter).unwrap(), lhs + rhs);
                let mut meter = Unbounded;
                assert_eq!(metered_subtract(lhs, rhs, &mut meter).unwrap(), lhs - rhs);
            }
        }
    }

    #[test]
    fn metered_addition_cancels_inside_limb_work_and_before_publication() {
        let lhs = (BigInt::one() << 32_768) - 1i64;
        let rhs = (BigInt::one() << 32_767) + 65_537i64;
        let mut baseline = CancelAfter {
            cancel_at_checkpoint: usize::MAX,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(metered_add(&lhs, &rhs, &mut baseline).unwrap(), &lhs + &rhs);
        assert!(baseline.checkpoints > 2_000);

        let cancel_at_checkpoint = baseline.checkpoints * 3 / 4;
        let mut meter = CancelAfter {
            cancel_at_checkpoint,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(
            metered_add(&lhs, &rhs, &mut meter),
            Err(MeterError::Cancelled)
        );
        assert_eq!(meter.checkpoints, cancel_at_checkpoint);
        assert!(meter.compute_steps < baseline.compute_steps);

        let mut meter = CancelAfter {
            cancel_at_checkpoint: baseline.checkpoints,
            checkpoints: 0,
            compute_steps: 0,
        };
        assert_eq!(
            metered_add(&lhs, &rhs, &mut meter),
            Err(MeterError::Cancelled)
        );
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
    fn exact_division_and_gcd_apis_cover_signs_and_zero_boundaries() {
        assert_eq!(gcd(&BigInt::from(-54), &BigInt::from(24)), BigInt::from(6));
        assert_eq!(gcd(&BigInt::zero(), &BigInt::zero()), BigInt::zero());

        let a = BigInt::from(-54);
        let b = BigInt::from(24);
        let (g, x, y) = extended_gcd(&a, &b);
        assert_eq!(g, BigInt::from(6));
        assert_eq!(&a * x + &b * y, g);

        assert_eq!(
            exact_div(&BigInt::from(-42), &BigInt::from(7)),
            Some(BigInt::from(-6))
        );
        assert_eq!(exact_div(&BigInt::from(42), &BigInt::from(8)), None);
        assert_eq!(exact_div(&BigInt::from(42), &BigInt::zero()), None);
    }

    #[test]
    fn metered_exact_division_and_gcd_match_unmetered_apis() {
        let a = (BigInt::one() << 257) - 1i64;
        let b = (BigInt::one() << 129) - 1i64;

        let mut meter = Unbounded;
        assert_eq!(
            metered_exact_div(&a, &b, &mut meter).unwrap(),
            exact_div(&a, &b)
        );

        let mut meter = Unbounded;
        assert_eq!(metered_gcd(&a, &b, &mut meter).unwrap(), gcd(&a, &b));

        let mut meter = Unbounded;
        let (metered_g, x, y) = metered_extended_gcd(&a, &b, &mut meter).unwrap();
        assert_eq!(metered_g, gcd(&a, &b));
        assert_eq!(&a * x + &b * y, metered_g);
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

    #[test]
    fn metered_value_lanes_cancel_at_each_terminal_publication_class() {
        macro_rules! assert_terminal_boundary {
            ($expected:expr, $trailing_checkpoints:expr, $run:expr) => {{
                let expected = $expected;
                assert_terminal_shape(&expected, $trailing_checkpoints, $run);
                assert_terminal_cancellation(&expected, $run);
            }};
        }

        let zero = BigInt::zero();
        let multiplier = (BigInt::one() << 129) + 65_537i64;
        let multiplicand = -((BigInt::one() << 131) + 17i64);
        assert_terminal_boundary!(zero.clone(), 2, |meter| {
            metered_multiply(&zero, &multiplier, meter)
        });
        assert_terminal_boundary!(&multiplier * &multiplicand, 2, |meter| {
            metered_multiply(&multiplier, &multiplicand, meter)
        });

        let zero_divisor = BigInt::zero();
        let divisor = BigInt::from(97);
        let small_dividend = BigInt::from(7);
        let general_dividend = BigInt::from(12_345);
        assert_terminal_boundary!(None, 2, |meter| {
            metered_div_rem(&general_dividend, &zero_divisor, meter)
        });
        assert_terminal_boundary!((zero.clone(), zero.clone()), 2, |meter| {
            let divisor = NonZeroBigInt::new(&divisor).expect("fixture divisor is nonzero");
            metered_div_rem_nonzero(&zero, divisor, meter)
        });
        assert_terminal_boundary!((BigInt::zero(), small_dividend.clone()), 2, |meter| {
            let divisor = NonZeroBigInt::new(&divisor).expect("fixture divisor is nonzero");
            metered_div_rem_nonzero(&small_dividend, divisor, meter)
        });
        let general_result = (BigInt::from(127), BigInt::from(26));
        assert_terminal_boundary!(general_result.clone(), 2, |meter| {
            let divisor = NonZeroBigInt::new(&divisor).expect("fixture divisor is nonzero");
            metered_div_rem_nonzero(&general_dividend, divisor, meter)
        });
        assert_terminal_boundary!(Some(general_result), 3, |meter| {
            metered_div_rem(&general_dividend, &divisor, meter)
        });
        assert_terminal_boundary!(Some((zero.clone(), zero.clone())), 4, |meter| {
            metered_div_rem(&zero, &divisor, meter)
        });

        let exact_dividend = BigInt::from(12_222);
        let inexact_dividend = BigInt::from(12_223);
        assert_terminal_boundary!(None, 3, |meter| {
            metered_exact_div(&exact_dividend, &zero_divisor, meter)
        });
        assert_terminal_boundary!(Some(BigInt::from(126)), 4, |meter| {
            metered_exact_div(&exact_dividend, &divisor, meter)
        });
        assert_terminal_boundary!(None, 4, |meter| {
            metered_exact_div(&inexact_dividend, &divisor, meter)
        });
        assert_terminal_boundary!(Some(zero.clone()), 5, |meter| {
            metered_exact_div(&zero, &divisor, meter)
        });
    }

    proptest! {
        #[test]
        fn strategies_agree_across_the_threshold_boundary(
            shift in 254u32..258u32,
            sign_a in proptest::bool::ANY,
            sign_b in proptest::bool::ANY,
        ) {
            let base = BigInt::one() << shift;
            for delta_a in [-1i64, 0, 1] {
                let a = match sign_a {
                    true => &base + delta_a,
                    false => -(&base + delta_a),
                };
                for delta_b in [-1i64, 0, 1] {
                    let b = match sign_b {
                        true => &base + delta_b,
                        false => -(&base + delta_b),
                    };
                    let ref_res = multiply_with_strategy(&a, &b, Strategy::SchoolbookReference);
                    let kar_res = multiply_with_strategy(&a, &b, Strategy::Karatsuba);
                    let toom_res = multiply_with_strategy(&a, &b, Strategy::Toom3);
                    let nat_res = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
                    let selected_res = multiply(&a, &b);
                    prop_assert_eq!(&ref_res, &kar_res);
                    prop_assert_eq!(&ref_res, &toom_res);
                    prop_assert_eq!(&ref_res, &nat_res);
                    prop_assert_eq!(&ref_res, &selected_res);
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
            let toom3 = multiply_with_strategy(&a, &b, Strategy::Toom3);
            let native = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
            let mut meter = Unbounded;
            let metered = metered_multiply(&a, &b, &mut meter).unwrap();
            let mut recursive_meter = Unbounded;
            let recursively_metered =
                metered_karatsuba_candidate(&a, &b, &mut recursive_meter)
                    .unwrap()
                    .materialize_unmetered();
            let mut metered_toom_meter = Unbounded;
            let metered_toom = metered_toom3_candidate(&a, &b, &mut metered_toom_meter)
                .unwrap()
                .materialize_unmetered();
            let mut metered_ntt_meter = Unbounded;
            let metered_ntt = metered_ntt_crt_candidate(&a, &b, &mut metered_ntt_meter)
                .unwrap()
                .materialize_unmetered();
            prop_assert_eq!(&reference, &karatsuba);
            prop_assert_eq!(&reference, &toom3);
            prop_assert_eq!(&reference, &native);
            prop_assert_eq!(&reference, &metered);
            prop_assert_eq!(&reference, &recursively_metered);
            prop_assert_eq!(&reference, &metered_toom);
            prop_assert_eq!(&reference, &metered_ntt);
            prop_assert_eq!(multiply_with_strategy(&b, &a, Strategy::Toom3), native.clone());
            prop_assert_eq!(&a * &b, native);
        }

        #[test]
        fn metered_ntt_crt_candidate_matches_broad_balanced_operands(
            a_bytes in proptest::collection::vec(any::<u8>(), 0..65),
            b_bytes in proptest::collection::vec(any::<u8>(), 0..65),
        ) {
            let a = BigInt::from_signed_bytes_be(&a_bytes);
            let b = BigInt::from_signed_bytes_be(&b_bytes);
            let expected = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
            let actual = metered_ntt_crt_candidate(&a, &b, &mut Unbounded)
                .unwrap()
                .materialize_unmetered();
            prop_assert_eq!(&actual, &expected);
            let commuted = metered_ntt_crt_candidate(&b, &a, &mut Unbounded)
                .unwrap()
                .materialize_unmetered();
            prop_assert_eq!(commuted, expected);
        }

        #[test]
        fn metered_add_sub_match_broad_signed_operands(
            a_bytes in proptest::collection::vec(any::<u8>(), 0..257),
            b_bytes in proptest::collection::vec(any::<u8>(), 0..257),
        ) {
            let a = BigInt::from_signed_bytes_be(&a_bytes);
            let b = BigInt::from_signed_bytes_be(&b_bytes);
            let mut meter = Unbounded;
            prop_assert_eq!(metered_add(&a, &b, &mut meter).unwrap(), &a + &b);
            let mut meter = Unbounded;
            prop_assert_eq!(metered_subtract(&a, &b, &mut meter).unwrap(), &a - &b);
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

        #[test]
        fn owned_gcd_lanes_agree_for_broad_signed_operands(
            a_bytes in proptest::collection::vec(any::<u8>(), 0..97),
            b_bytes in proptest::collection::vec(any::<u8>(), 0..97),
        ) {
            let a = BigInt::from_signed_bytes_be(&a_bytes);
            let b = BigInt::from_signed_bytes_be(&b_bytes);

            let mut meter = Unbounded;
            prop_assert_eq!(metered_gcd(&a, &b, &mut meter).unwrap(), gcd(&a, &b));

            let mut meter = Unbounded;
            let (metered_g, x, y) = metered_extended_gcd(&a, &b, &mut meter).unwrap();
            prop_assert_eq!(&metered_g, &gcd(&a, &b));
            prop_assert_eq!(&a * x + &b * y, metered_g);
        }

        #[test]
        fn floor_square_root_satisfies_the_defining_invariant(
            bytes in proptest::collection::vec(any::<u8>(), 0..80),
        ) {
            let value = BigInt::from_bytes_le(&bytes);
            let root = sqrt_floor(&value).expect("byte magnitudes are nonnegative");
            let root_squared = &root * &root;
            let successor = &root + 1i64;
            let successor_squared = &successor * &successor;
            prop_assert!(root_squared <= value);
            prop_assert!(value < successor_squared);
        }
    }
}
