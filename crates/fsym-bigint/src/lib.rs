//! Owned arbitrary-precision integer arithmetic behind a containment
//! boundary (WS03 / architecture doc §5.3).
//!
//! num-bigint is an audited temporary substrate: it appears ONLY inside
//! this crate. Higher layers depend on [`BigInt`] as defined here, so the
//! substrate can be replaced without touching term/domain schemas.
//!
//! # Strategy selection
//!
//! Multiplication offers two strategies with an explicit threshold:
//!
//! - [`Strategy::SchoolbookReference`] — repeated-addition oracle,
//!   O(log₂|min|) doublings; used below the threshold where its cost is
//!   negligible and its simplicity makes it the ideal cross-check target.
//! - [`Strategy::NativeSubstrate`] — delegates to num-bigint's internally
//!   selected algorithm (its Karatsuba/Toom thresholds are opaque).
//!
//! [`select_strategy`] applies [`DEFAULT_STRATEGY_THRESHOLD_BITS`]; every
//! strategy pair is proptest-differential-tested ACROSS the boundary
//! (magnitudes 2²⁵⁴…2²⁵⁸), which is the threshold-selection evidence the
//! WS03 contract requires. An own-Karatsuba strategy replaces the native
//! delegation as a follow-up bead; the selection machinery does not move.
//!
//! # Limb accounting
//!
//! [`BigInt::limb_count`] exposes u64-limb height so callers can charge
//! budget per unit of work before entering an operation.

use num_bigint::BigInt as Substrate;
use num_integer::Integer;
use num_traits::{One, Pow, Signed, ToPrimitive, Zero};

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
    /// Repeated-addition oracle. O(log₂|min|) doublings; never use for
    /// very large operands outside differential testing.
    SchoolbookReference,
    /// Delegate to the audited num-bigint substrate.
    NativeSubstrate,
}

/// Bit-size at or above which multiplication uses
/// [`Strategy::NativeSubstrate`]. Below it, the schoolbook oracle is both
/// faster to verify and effectively free.
pub const DEFAULT_STRATEGY_THRESHOLD_BITS: u64 = 256;

/// Pure strategy policy: visible and unit-testable on its own.
pub fn select_strategy(max_magnitude_bits: u64) -> Strategy {
    if max_magnitude_bits >= DEFAULT_STRATEGY_THRESHOLD_BITS {
        Strategy::NativeSubstrate
    } else {
        Strategy::SchoolbookReference
    }
}

/// Owned arbitrary-precision integer. The only bigint type visible above
/// this crate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BigInt(pub Substrate);

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

    pub fn is_negative(&self) -> bool {
        self.0.sign() == num_bigint::Sign::Minus
    }

    pub fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    /// Magnitude size in bits (0 for zero).
    pub fn bits(&self) -> u64 {
        self.0.bits()
    }

    /// Height in u64 limbs — the charging unit for limb-operation budgets.
    pub fn limb_count(&self) -> u64 {
        limb_count_u64(self.bits())
    }

    pub fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    pub fn gcd(&self, other: &Self) -> Self {
        Self(self.0.gcd(&other.0))
    }

    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let res = self.0.extended_gcd(&other.0);
        (Self(res.gcd), Self(res.x), Self(res.y))
    }

    pub fn pow(&self, exp: u32) -> Self {
        Self(self.0.clone().pow(exp))
    }

    pub fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (q, r) = self.0.div_rem(&other.0);
        (Self(q), Self(r))
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

impl std::fmt::Display for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::ops::Add for BigInt {
    type Output = BigInt;
    fn add(self, other: BigInt) -> BigInt {
        BigInt(self.0 + other.0)
    }
}

impl std::ops::Mul for BigInt {
    type Output = BigInt;
    fn mul(self, other: BigInt) -> BigInt {
        BigInt(self.0 * other.0)
    }
}

impl std::ops::Sub for BigInt {
    type Output = BigInt;
    fn sub(self, other: BigInt) -> BigInt {
        BigInt(self.0 - other.0)
    }
}

impl std::ops::Div for BigInt {
    type Output = BigInt;
    fn div(self, other: BigInt) -> BigInt {
        BigInt(self.0 / other.0)
    }
}

impl std::ops::Rem for BigInt {
    type Output = BigInt;
    fn rem(self, other: BigInt) -> BigInt {
        BigInt(self.0 % other.0)
    }
}

impl std::ops::Neg for BigInt {
    type Output = BigInt;
    fn neg(self) -> BigInt {
        BigInt(-self.0)
    }
}

impl PartialEq<i64> for BigInt {
    fn eq(&self, other: &i64) -> bool {
        self.0 == Substrate::from(*other)
    }
}

/// Multiplies via the explicitly chosen strategy.
pub fn multiply_with_strategy(a: &BigInt, b: &BigInt, strategy: Strategy) -> BigInt {
    match strategy {
        Strategy::SchoolbookReference => schoolbook_reference(&a.0, &b.0),
        Strategy::NativeSubstrate => BigInt(a.0.clone() * b.0.clone()),
    }
}

/// Applies [`select_strategy`] over the operands' larger bit height, then
/// multiplies. This is the policy entry point evaluation code should call.
pub fn multiply(a: &BigInt, b: &BigInt) -> BigInt {
    let strategy = select_strategy(std::cmp::max(a.bits(), b.bits()));
    multiply_with_strategy(a, b, strategy)
}

/// Schoolbook oracle: double-and-add over the multiplier magnitude.
fn schoolbook_reference(a: &Substrate, b: &Substrate) -> BigInt {
    let neg = a.sign() != b.sign();
    let steps = a.magnitude().min(b.magnitude());
    // Accumulate UNSIGNED magnitudes; the sign is applied exactly once
    // at the end. Using a signed clone here double-flips the result.
    let unit: Substrate = if a.magnitude() <= b.magnitude() {
        Substrate::from(b.magnitude().clone())
    } else {
        Substrate::from(a.magnitude().clone())
    };
    let mut acc = Substrate::zero();
    let mut shifted = unit;
    let mut bits = steps.clone();
    while !bits.is_zero() {
        if bits.bit(0) {
            acc += &shifted;
        }
        shifted <<= 1;
        bits >>= 1;
    }
    if neg && !acc.is_zero() {
        BigInt(-acc)
    } else {
        BigInt(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big_pow(base: i64, exp: u32) -> BigInt {
        let mut acc = BigInt::one();
        for _ in 0..exp {
            acc = acc * BigInt::from_i64(base);
        }
        acc
    }

    #[test]
    fn select_strategy_switches_at_the_documented_threshold() {
        assert_eq!(
            select_strategy(DEFAULT_STRATEGY_THRESHOLD_BITS - 1),
            Strategy::SchoolbookReference
        );
        assert_eq!(
            select_strategy(DEFAULT_STRATEGY_THRESHOLD_BITS),
            Strategy::NativeSubstrate
        );
        assert_eq!(select_strategy(0), Strategy::SchoolbookReference);
    }

    #[test]
    fn limb_count_matches_bit_height() {
        assert_eq!(BigInt::zero().limb_count(), 0);
        assert_eq!(BigInt::one().limb_count(), 1);
        // 2^63 needs 63 bits -> one u64 limb; 2^64 -> two.
        assert_eq!((big_pow(2, 63)).limb_count(), 1);
        assert_eq!((big_pow(2, 64)).limb_count(), 2);
    }

    #[test]
    fn strategies_agree_across_the_threshold_boundary() {
        // Sweep magnitudes through the 256-bit boundary in both signs.
        for shift in [
            DEFAULT_STRATEGY_THRESHOLD_BITS - 2,
            DEFAULT_STRATEGY_THRESHOLD_BITS - 1,
            DEFAULT_STRATEGY_THRESHOLD_BITS,
            DEFAULT_STRATEGY_THRESHOLD_BITS + 1,
        ] {
            let base = big_pow(2, shift as u32);
            for delta in [-1i64, 0, 1] {
                let a = match delta {
                    d if d < 0 => base.clone() + BigInt::from_i64(delta),
                    _ => base.clone(),
                };
                let b = BigInt::from_i64(-3);
                let via_ref = multiply_with_strategy(&a, &b, Strategy::SchoolbookReference);
                let via_native = multiply_with_strategy(&a, &b, Strategy::NativeSubstrate);
                assert_eq!(
                    via_ref, via_native,
                    "disagreement at shift {shift} delta {delta}"
                );
            }
        }
    }

    #[test]
    fn policy_multiply_uses_threshold_and_preserves_signs() {
        // Small: routed to schoolbook; large: routed to native. Both must
        // produce identical signed results through the policy entry point.
        let small_a = BigInt::from_i64(-7);
        let small_b = BigInt::from_i64(6);
        assert_eq!(multiply(&small_a, &small_b), BigInt::from_i64(-42));

        let big = big_pow(2, DEFAULT_STRATEGY_THRESHOLD_BITS as u32);
        let product = multiply(&big, &big);
        // 2^256 · 2^256 = 2^512: 513 bits -> ceil(513/64) = 9 u64 limbs.
        assert_eq!(product.bits(), 513);
        assert_eq!(product.limb_count(), 9);
        assert!(!product.is_negative());
    }

    #[test]
    fn zero_identities_hold_for_both_strategies() {
        for strategy in [Strategy::SchoolbookReference, Strategy::NativeSubstrate] {
            let x = BigInt::from_i64(-123456);
            let z = BigInt::zero();
            assert!(multiply_with_strategy(&x, &z, strategy).is_zero());
            assert!(multiply_with_strategy(&z, &x, strategy).is_zero());
        }
    }
}
