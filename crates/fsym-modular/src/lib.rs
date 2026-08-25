//! Exact modular arithmetic primitives (WS03).
//!
//! Everything here is pure modular big-integer math with no FFI and no
//! machine-float intermediates behind the [`fsym_bigint::BigInt`] containment boundary.
//! Determinism: identical inputs always produce identical outputs; the prime stream is
//! a fixed deterministic sequence, and primality testing uses a fixed base set.
//!
//! # Primality honesty
//!
//! [`is_probable_prime`] is **deterministic** for `n < 3.317·10²⁴` (the
//! first 13 prime bases are a proven certificate for that range) and only
//! probabilistic beyond it. Callers needing certainty above that bound
//! must supply their own proof (e.g. ECPP later in WS11).

#![forbid(unsafe_code)]

use fsym_bigint::{
    BigInt, NonZeroBigInt, extended_gcd, gcd, metered_add as metered_bigint_add,
    metered_div_rem_nonzero, metered_extended_gcd, metered_gcd, metered_multiply as metered_mul,
    metered_subtract as metered_bigint_subtract,
};
#[cfg(test)]
use fsym_bigint::{exact_div, metered_exact_div};
use fsym_budget::{BudgetMeter, Dimension, MeterError};

/// Multiplicative inverse of `a` modulo `m` (`m > 0`); `None` when
/// `gcd(a, m) != 1`.
pub fn mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    if !m.is_positive() {
        return None;
    }
    let (g, x, _) = extended_gcd(&(a % m), m);
    if !g.is_one() {
        return None;
    }
    Some(((x % m) + m) % m)
}

/// Cancellation-first multiplicative inverse using metered division and Bézout lanes.
pub fn metered_mod_inverse<M: BudgetMeter>(
    a: &BigInt,
    m: &BigInt,
    meter: &mut M,
) -> Result<Option<BigInt>, MeterError> {
    meter.checkpoint()?;
    if !m.is_positive() {
        return Ok(None);
    }
    let Some(modulus) = NonZeroBigInt::new(m) else {
        return Ok(None);
    };
    let residue = metered_normalized_remainder(a, modulus, meter)?;
    let (g, x, _) = metered_extended_gcd(&residue, m, meter)?;
    if !g.is_one() {
        return metered_finish(None, meter);
    }
    let inverse = metered_normalized_remainder(&x, modulus, meter)?;
    metered_finish(Some(inverse), meter)
}

/// Solves the two-congruence system `x ≡ rem_i (mod mod_i)`. Returns
/// `(x, lcm(mod_1, mod_2))` with `0 <= x < lcm`, or `None` when the
/// congruences are inconsistent.
pub fn crt_pair(
    rem1: &BigInt,
    mod1: &BigInt,
    rem2: &BigInt,
    mod2: &BigInt,
) -> Option<(BigInt, BigInt)> {
    if !mod1.is_positive() || !mod2.is_positive() {
        return None;
    }
    let g = gcd(mod1, mod2);
    let diff = rem2 - rem1;
    if (&diff % &g) != BigInt::zero() {
        return None;
    }
    let lcm = (mod1 / &g) * mod2;
    let m1_div_g = mod1 / &g;
    let m2_div_g = mod2 / &g;
    let (_, u, _) = extended_gcd(&m1_div_g, &m2_div_g);
    let shift = (diff / &g) * u * mod1;
    let mut x = (rem1 + shift) % &lcm;
    if x.is_negative() {
        x += &lcm;
    }
    Some((x, lcm))
}

/// Cancellation-first two-congruence CRT using only metered arithmetic lanes.
pub fn metered_crt_pair<M: BudgetMeter>(
    rem1: &BigInt,
    mod1: &BigInt,
    rem2: &BigInt,
    mod2: &BigInt,
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    if !mod1.is_positive() || !mod2.is_positive() {
        return Ok(None);
    }

    let g = metered_gcd(mod1, mod2, meter)?;
    let Some(g_divisor) = NonZeroBigInt::new(&g) else {
        return Ok(None);
    };
    let diff = metered_subtract(rem2, rem1, meter)?;
    let (diff_over_g, diff_remainder) = metered_div_rem_nonzero(&diff, g_divisor, meter)?;
    if !diff_remainder.is_zero() {
        return metered_finish(None, meter);
    }
    let (m1_div_g, m1_remainder) = metered_div_rem_nonzero(mod1, g_divisor, meter)?;
    let (m2_div_g, m2_remainder) = metered_div_rem_nonzero(mod2, g_divisor, meter)?;
    if !m1_remainder.is_zero() || !m2_remainder.is_zero() {
        return metered_finish(None, meter);
    }

    let lcm = metered_mul(&m1_div_g, mod2, meter)?;
    let (_, u, _) = metered_extended_gcd(&m1_div_g, &m2_div_g, meter)?;
    let scaled_diff = metered_mul(&diff_over_g, &u, meter)?;
    let shift = metered_mul(&scaled_diff, mod1, meter)?;
    let shifted_remainder = metered_add(rem1, &shift, meter)?;
    let Some(lcm_divisor) = NonZeroBigInt::new(&lcm) else {
        return metered_finish(None, meter);
    };
    let x = metered_normalized_remainder(&shifted_remainder, lcm_divisor, meter)?;
    metered_finish(Some((x, lcm)), meter)
}

/// Solves an arbitrary system of simultaneous congruences.
pub fn crt(congruences: &[(BigInt, BigInt)]) -> Option<(BigInt, BigInt)> {
    if congruences.is_empty() {
        return Some((BigInt::zero(), BigInt::one()));
    }
    let mut congruence_iter = congruences.iter();
    let (mut x, mut m) = congruence_iter.next()?.clone();
    if !m.is_positive() {
        return None;
    }
    x %= &m;
    if x.is_negative() {
        x += &m;
    }
    for (r_i, m_i) in congruence_iter {
        let (next_x, next_m) = crt_pair(&x, &m, r_i, m_i)?;
        x = next_x;
        m = next_m;
    }
    Some((x, m))
}

/// Cancellation-first arbitrary CRT fold.
pub fn metered_crt<M: BudgetMeter>(
    congruences: &[(BigInt, BigInt)],
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    let mut congruence_iter = congruences.iter();
    let Some((first_remainder, first_modulus)) = congruence_iter.next() else {
        return Ok(Some((BigInt::zero(), BigInt::one())));
    };
    if !first_modulus.is_positive() {
        return Ok(None);
    }
    let Some(first_modulus_divisor) = NonZeroBigInt::new(first_modulus) else {
        return Ok(None);
    };
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            first_modulus.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut x = metered_normalized_remainder(first_remainder, first_modulus_divisor, meter)?;
    let mut modulus = first_modulus.clone();
    for (remainder, next_modulus) in congruence_iter {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let Some((next_x, combined_modulus)) =
            metered_crt_pair(&x, &modulus, remainder, next_modulus, meter)?
        else {
            return metered_finish(None, meter);
        };
        x = next_x;
        modulus = combined_modulus;
    }
    metered_finish(Some((x, modulus)), meter)
}

/// Symmetric rational reconstruction: recovers `(r, s)` with `gcd(r, s) == 1`,
/// `s > 0`, and `r · s⁻¹ ≡ n (mod m)`, where `|r|` and `s` do not exceed
/// `floor(sqrt((m - 1) / 2))`. The strict `2 * bound^2 < m` inequality makes the
/// representative unique.
pub fn rational_reconstruct(n: &BigInt, m: &BigInt) -> Option<(BigInt, BigInt)> {
    if *m <= BigInt::one() {
        return None;
    }
    let residue = (n % m + m) % m;
    if residue.is_zero() {
        return Some((BigInt::zero(), BigInt::one()));
    }

    // The symmetric uniqueness condition is 2 * bound^2 < m. Using sqrt(m)
    // admits multiple representatives and can make the result depend on the Euclidean path.
    let bound = sqrt_floor(&((m - 1i64) / 2i64));
    let (mut r_prev, mut r_cur) = (m.clone(), residue.clone());
    let (mut t_prev, mut t_cur) = (BigInt::zero(), BigInt::one());

    while r_cur.abs() > bound {
        let (q, r_next) = r_prev.div_rem(&r_cur);
        r_prev = r_cur;
        r_cur = r_next;

        let t_next = t_prev - q * &t_cur;
        t_prev = t_cur;
        t_cur = t_next;
    }

    let mut r_out = r_cur;
    if t_cur.is_negative() {
        r_out = -r_out;
        t_cur = -t_cur;
    }
    if !t_cur.is_positive() {
        return None;
    }
    if r_out.abs() > bound || t_cur > bound {
        return None;
    }
    if gcd(&r_out, &t_cur) != BigInt::one() {
        return None;
    }
    if (&r_out - &residue * &t_cur) % m != BigInt::zero() {
        return None;
    }
    Some((r_out, t_cur))
}

/// Cancellation-first symmetric rational reconstruction.
pub fn metered_rational_reconstruct<M: BudgetMeter>(
    n: &BigInt,
    m: &BigInt,
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    if *m <= BigInt::one() {
        return Ok(None);
    }
    let Some(modulus) = NonZeroBigInt::new(m) else {
        return Ok(None);
    };
    let residue = metered_normalized_remainder(n, modulus, meter)?;
    if residue.is_zero() {
        return metered_finish(Some((BigInt::zero(), BigInt::one())), meter);
    }

    let one = BigInt::one();
    let two = BigInt::from(2i64);
    let m_minus_one = metered_subtract(m, &one, meter)?;
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return Ok(None);
    };
    let (half, _) = metered_div_rem_nonzero(&m_minus_one, two_divisor, meter)?;
    let bound = metered_sqrt_floor(&half, meter)?;

    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            m.limb_count()
                .max(1)
                .saturating_add(residue.limb_count().max(1))
                .saturating_mul(8),
        ),
        (Dimension::AllocationCount, 2),
    ])?;
    let (mut r_prev, mut r_cur) = (m.clone(), residue.clone());
    let (mut t_prev, mut t_cur) = (BigInt::zero(), BigInt::one());

    while metered_greater(&r_cur, &bound, meter)? {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let Some(r_cur_divisor) = NonZeroBigInt::new(&r_cur) else {
            return Ok(None);
        };
        let (q, r_next) = metered_div_rem_nonzero(&r_prev, r_cur_divisor, meter)?;
        r_prev = r_cur;
        r_cur = r_next;

        let q_times_t = metered_mul(&q, &t_cur, meter)?;
        let t_next = metered_subtract(&t_prev, &q_times_t, meter)?;
        t_prev = t_cur;
        t_cur = t_next;
    }

    let mut r_out = r_cur;
    if t_cur.is_negative() {
        r_out = metered_negate(r_out, meter)?;
        t_cur = metered_negate(t_cur, meter)?;
    }
    if !t_cur.is_positive() || metered_greater(&t_cur, &bound, meter)? {
        return metered_finish(None, meter);
    }
    if metered_gcd(&r_out, &t_cur, meter)? != BigInt::one() {
        return metered_finish(None, meter);
    }
    let residue_times_denominator = metered_mul(&residue, &t_cur, meter)?;
    let congruence_delta = metered_subtract(&r_out, &residue_times_denominator, meter)?;
    if !metered_normalized_remainder(&congruence_delta, modulus, meter)?.is_zero() {
        return metered_finish(None, meter);
    }
    metered_finish(Some((r_out, t_cur)), meter)
}

/// Deterministic increasing stream of primes: 2, 3, 5, 7, ...
pub struct PrimeStream {
    emitted: Vec<BigInt>,
    current: BigInt,
}

impl PrimeStream {
    pub fn new() -> Self {
        Self {
            emitted: Vec::new(),
            current: BigInt::from(2i64),
        }
    }

    /// Returns the next prime through cancellation-first metered arithmetic.
    ///
    /// Refusal leaves the stream cursor and emitted-prime table unchanged, so retrying cannot
    /// silently skip a candidate. Consumed budget is not refunded.
    pub fn next_metered<M: BudgetMeter>(&mut self, meter: &mut M) -> Result<BigInt, MeterError> {
        meter.checkpoint()?;
        meter.charge_batch(&[
            (
                Dimension::MemoryBytes,
                self.current.limb_count().max(1).saturating_mul(8),
            ),
            (Dimension::AllocationCount, 1),
        ])?;
        let mut current = self.current.clone();
        let one = BigInt::one();
        loop {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let candidate = current;
            let next_current = metered_add(&candidate, &one, meter)?;
            let root = metered_sqrt_floor(&candidate, meter)?;
            let mut divides = false;
            for prime in &self.emitted {
                meter.checkpoint()?;
                meter.charge(Dimension::ComputeSteps, 1)?;
                if metered_greater(prime, &root, meter)? {
                    break;
                }
                let Some(prime_divisor) = NonZeroBigInt::new(prime) else {
                    continue;
                };
                let (_, remainder) = metered_div_rem_nonzero(&candidate, prime_divisor, meter)?;
                if remainder.is_zero() {
                    divides = true;
                    break;
                }
            }
            if !divides {
                meter.charge_batch(&[
                    (
                        Dimension::MemoryBytes,
                        candidate
                            .limb_count()
                            .max(1)
                            .saturating_mul(8)
                            .saturating_add(
                                u64::try_from(std::mem::size_of::<BigInt>()).unwrap_or(u64::MAX),
                            ),
                    ),
                    (Dimension::AllocationCount, 2),
                ])?;
                let stored_candidate = candidate.clone();
                meter.checkpoint()?;
                self.emitted.push(stored_candidate);
                self.current = next_current;
                return Ok(candidate);
            }
            current = next_current;
        }
    }
}

impl Default for PrimeStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for PrimeStream {
    type Item = BigInt;

    fn next(&mut self) -> Option<BigInt> {
        loop {
            let cand = self.current.clone();
            self.current = &self.current + 1i64;
            let root = sqrt_floor(&cand);
            let divides = self.emitted.iter().take_while(|p| **p <= root).any(|p| {
                let r = &cand % p;
                r.is_zero()
            });
            if !divides {
                self.emitted.push(cand.clone());
                return Some(cand);
            }
        }
    }
}

fn sqrt_floor(n: &BigInt) -> BigInt {
    if !n.is_positive() {
        return BigInt::zero();
    }
    let two = BigInt::from(2i64);
    let mut x = n.clone();
    loop {
        let next = (&x + n / &x) / &two;
        if next >= x {
            return x;
        }
        x = next;
    }
}

const MR_BASES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Miller-Rabin primality with fixed deterministic bases.
pub fn is_probable_prime(n: &BigInt) -> bool {
    if *n < 2i64 {
        return false;
    }
    for base in MR_BASES {
        let b = BigInt::from(i64::from(base));
        if *n == b {
            return true;
        }
        if n % &b == BigInt::zero() {
            return false;
        }
    }

    let n_minus_1 = n - BigInt::one();
    let mut d = n_minus_1.clone();
    let mut s: u32 = 0;
    while (&d % 2i64).is_zero() {
        d /= 2i64;
        s += 1;
    }

    for base in MR_BASES {
        let a = BigInt::from(i64::from(base));
        if &a >= n {
            continue;
        }
        let mut x = mod_pow(&a, &d, n);
        if x.is_one() || x == n_minus_1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            x = mod_pow(&x, &BigInt::from(2i64), n);
            if x == n_minus_1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

/// Cancellation-first Miller-Rabin primality with the same fixed base set.
pub fn metered_is_probable_prime<M: BudgetMeter>(
    n: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    meter.checkpoint()?;
    if *n < 2i64 {
        return Ok(false);
    }
    let Some(modulus) = NonZeroBigInt::new(n) else {
        return Ok(false);
    };
    for base in MR_BASES {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let divisor = BigInt::from(i64::from(base));
        if *n == divisor {
            return metered_finish(true, meter);
        }
        let Some(divisor) = NonZeroBigInt::new(&divisor) else {
            return Ok(false);
        };
        let (_, remainder) = metered_div_rem_nonzero(n, divisor, meter)?;
        if remainder.is_zero() {
            return metered_finish(false, meter);
        }
    }

    let n_minus_one = metered_subtract(n, &BigInt::one(), meter)?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            n_minus_one.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut d = n_minus_one.clone();
    let two = BigInt::from(2i64);
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return Ok(false);
    };
    let mut s: u32 = 0;
    loop {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let (quotient, remainder) = metered_div_rem_nonzero(&d, two_divisor, meter)?;
        if !remainder.is_zero() {
            break;
        }
        d = quotient;
        s = s.saturating_add(1);
    }

    for base in MR_BASES {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let a = BigInt::from(i64::from(base));
        if metered_greater_or_equal(&a, n, meter)? {
            continue;
        }
        let mut x = metered_mod_pow(&a, &d, modulus, meter)?;
        if x.is_one() || metered_equal(&x, &n_minus_one, meter)? {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let square = metered_mul(&x, &x, meter)?;
            x = metered_normalized_remainder(&square, modulus, meter)?;
            if metered_equal(&x, &n_minus_one, meter)? {
                composite = false;
                break;
            }
        }
        if composite {
            return metered_finish(false, meter);
        }
    }
    metered_finish(true, meter)
}

fn mod_pow(base: &BigInt, exp: &BigInt, modulus: &BigInt) -> BigInt {
    if modulus.is_one() {
        return BigInt::zero();
    }
    let mut res = BigInt::one();
    let mut b = base % modulus;
    let mut e = exp.clone();
    let two = BigInt::from(2i64);

    while e.is_positive() {
        if !(&e % &two).is_zero() {
            res = &(&res * &b) % modulus;
        }
        b = &(&b * &b) % modulus;
        e = &e / &two;
    }
    res
}

fn metered_mod_pow<M: BudgetMeter>(
    base: &BigInt,
    exp: &BigInt,
    modulus: NonZeroBigInt<'_>,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    if modulus.get().is_one() {
        return Ok(BigInt::zero());
    }
    let mut result = BigInt::one();
    let mut base = metered_normalized_remainder(base, modulus, meter)?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            exp.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut exponent = exp.clone();
    let two = BigInt::from(2i64);
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return Ok(BigInt::zero());
    };

    while exponent.is_positive() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let (next_exponent, parity) = metered_div_rem_nonzero(&exponent, two_divisor, meter)?;
        if !parity.is_zero() {
            let product = metered_mul(&result, &base, meter)?;
            result = metered_normalized_remainder(&product, modulus, meter)?;
        }
        exponent = next_exponent;
        if exponent.is_positive() {
            let square = metered_mul(&base, &base, meter)?;
            base = metered_normalized_remainder(&square, modulus, meter)?;
        }
    }
    meter.checkpoint()?;
    Ok(result)
}

fn metered_sqrt_floor<M: BudgetMeter>(n: &BigInt, meter: &mut M) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    if !n.is_positive() {
        return Ok(BigInt::zero());
    }
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            n.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut x = n.clone();
    let two = BigInt::from(2i64);
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return Ok(BigInt::zero());
    };
    loop {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let Some(x_divisor) = NonZeroBigInt::new(&x) else {
            return Ok(BigInt::zero());
        };
        let (quotient, _) = metered_div_rem_nonzero(n, x_divisor, meter)?;
        let sum = metered_add(&x, &quotient, meter)?;
        let (next, _) = metered_div_rem_nonzero(&sum, two_divisor, meter)?;
        if metered_greater_or_equal(&next, &x, meter)? {
            return metered_finish(x, meter);
        }
        x = next;
    }
}

fn metered_normalized_remainder<M: BudgetMeter>(
    value: &BigInt,
    modulus: NonZeroBigInt<'_>,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    let (_, remainder) = metered_div_rem_nonzero(value, modulus, meter)?;
    if remainder.is_negative() {
        metered_add(&remainder, modulus.get(), meter)
    } else {
        Ok(remainder)
    }
}

fn metered_finish<T, M: BudgetMeter>(value: T, meter: &mut M) -> Result<T, MeterError> {
    meter.checkpoint()?;
    Ok(value)
}

fn metered_equal<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    metered_compare(lhs, rhs, |ordering| ordering.is_eq(), meter)
}

fn metered_greater<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    metered_compare(lhs, rhs, |ordering| ordering.is_gt(), meter)
}

fn metered_greater_or_equal<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    metered_compare(lhs, rhs, |ordering| ordering.is_ge(), meter)
}

fn metered_compare<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    predicate: impl FnOnce(std::cmp::Ordering) -> bool,
    meter: &mut M,
) -> Result<bool, MeterError> {
    meter.checkpoint()?;
    meter.charge(
        Dimension::ComputeSteps,
        lhs.limb_count().max(rhs.limb_count()).max(1),
    )?;
    let result = predicate(lhs.cmp(rhs));
    meter.checkpoint()?;
    Ok(result)
}

fn metered_add<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    metered_bigint_add(lhs, rhs, meter)
}

fn metered_negate<M: BudgetMeter>(value: BigInt, meter: &mut M) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge(Dimension::ComputeSteps, value.limb_count().max(1))?;
    let result = -value;
    meter.checkpoint()?;
    Ok(result)
}

fn metered_subtract<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    metered_bigint_subtract(lhs, rhs, meter)
}

/// Exclusive upper bound for the fixed-base Miller-Rabin theorem used by this crate.
/// Values at or above this boundary remain probable-prime candidates, not exact field evidence.
fn deterministic_primality_bound() -> BigInt {
    BigInt::from(3_317_044_064_679_887_385u64) * BigInt::from(1_000_000u64)
        + BigInt::from(961_981u64)
}

fn is_certified_prime(characteristic: &BigInt) -> bool {
    characteristic > &BigInt::one()
        && characteristic < &deterministic_primality_bound()
        && is_probable_prime(characteristic)
}

fn is_canonical_residue(value: &BigInt, modulus: &BigInt) -> bool {
    !value.is_negative() && value < modulus
}

fn normalized_remainder(value: &BigInt, modulus: &BigInt) -> BigInt {
    let remainder = value % modulus;
    if remainder.is_negative() {
        remainder + modulus
    } else {
        remainder
    }
}

fn metered_clone_bigint<M: BudgetMeter>(
    value: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
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

fn metered_power_of_two<M: BudgetMeter>(
    exponent: u32,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    let mut value = BigInt::one();
    for _ in 0..exponent {
        value = metered_add(&value, &value, meter)?;
    }
    metered_finish(value, meter)
}

/// Typed representation of a modular arithmetic residue ring $\mathbb{Z} / m\mathbb{Z}$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModularRing {
    modulus: BigInt,
}

impl ModularRing {
    /// Creates a new modular ring $\mathbb{Z} / m\mathbb{Z}$ for $m > 1$.
    pub fn new(modulus: BigInt) -> Option<Self> {
        if modulus > BigInt::one() {
            Some(Self { modulus })
        } else {
            None
        }
    }

    /// Access the modulus $m$.
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// Constructs a canonical element in $\mathbb{Z} / m\mathbb{Z}$ from an arbitrary integer.
    pub fn element(&self, value: BigInt) -> ModularRingElement {
        let residue = normalized_remainder(&value, &self.modulus);
        ModularRingElement {
            ring: self.clone(),
            value: residue,
        }
    }

    /// Cancellation-first construction of a canonical element from an arbitrary integer.
    pub fn metered_element<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<ModularRingElement, MeterError> {
        let Some(modulus) = NonZeroBigInt::new(&self.modulus) else {
            return metered_finish(self.zero(), meter);
        };
        let residue = metered_normalized_remainder(value, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.modulus, meter)?,
        };
        metered_finish(
            ModularRingElement {
                ring,
                value: residue,
            },
            meter,
        )
    }

    /// The additive identity $0 \pmod m$.
    pub fn zero(&self) -> ModularRingElement {
        self.element(BigInt::zero())
    }

    /// The multiplicative identity $1 \pmod m$.
    pub fn one(&self) -> ModularRingElement {
        self.element(BigInt::one())
    }
}

/// A typed canonical element in a modular ring $\mathbb{Z} / m\mathbb{Z}$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModularRingElement {
    ring: ModularRing,
    value: BigInt,
}

impl ModularRingElement {
    /// The canonical integer value in the range $[0, m)$.
    pub fn value(&self) -> &BigInt {
        &self.value
    }

    /// Reference to the parent modular ring.
    pub fn ring(&self) -> &ModularRing {
        &self.ring
    }

    /// Whether this element is $0 \pmod m$.
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Whether this element is $1 \pmod m$.
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Modular addition: $(a + b) \pmod m$.
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let m = &self.ring.modulus;
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: (&self.value + &other.value) % m,
        })
    }

    /// Modular subtraction: $(a - b) \pmod m$.
    pub fn sub(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let m = &self.ring.modulus;
        let mut diff = (&self.value - &other.value) % m;
        if diff.is_negative() {
            diff += m;
        }
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: diff,
        })
    }

    /// Modular multiplication: $(a \cdot b) \pmod m$.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let m = &self.ring.modulus;
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: (&self.value * &other.value) % m,
        })
    }

    /// Modular negation: $-a \pmod m$.
    pub fn neg(&self) -> Self {
        if self.value.is_zero() {
            self.clone()
        } else {
            ModularRingElement {
                ring: self.ring.clone(),
                value: &self.ring.modulus - &self.value,
            }
        }
    }

    /// Cancellation-first modular negation.
    pub fn metered_neg<M: BudgetMeter>(&self, meter: &mut M) -> Result<Self, MeterError> {
        meter.checkpoint()?;
        let value = if self.value.is_zero() {
            metered_clone_bigint(&self.value, meter)?
        } else {
            metered_subtract(&self.ring.modulus, &self.value, meter)?
        };
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Self { ring, value }, meter)
    }

    /// Multiplicative inverse $a^{-1} \pmod m$; returns `None` when $\gcd(a, m) \neq 1$.
    pub fn inv(&self) -> Option<Self> {
        let inv_val = mod_inverse(&self.value, &self.ring.modulus)?;
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: inv_val,
        })
    }

    /// Exact modular division: $a / b \pmod m \iff a \cdot b^{-1} \pmod m$.
    pub fn div(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let b_inv = other.inv()?;
        self.mul(&b_inv)
    }

    /// Modular exponentiation $a^e \pmod m$; negative exponents are refused because a ring
    /// element need not be a unit.
    pub fn pow(&self, exp: &BigInt) -> Option<Self> {
        if exp.is_negative() {
            return None;
        }
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: mod_pow(&self.value, exp, &self.ring.modulus),
        })
    }

    /// Cancellation-first modular addition. A different parent ring is a computed refusal.
    pub fn metered_add<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let sum = metered_add(&self.value, &other.value, meter)?;
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_normalized_remainder(&sum, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular subtraction. A different parent ring is refused.
    pub fn metered_sub<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let difference = metered_subtract(&self.value, &other.value, meter)?;
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_normalized_remainder(&difference, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular multiplication. A different parent ring is refused.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let product = metered_mul(&self.value, &other.value, meter)?;
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_normalized_remainder(&product, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular inverse; non-units produce `Ok(None)`.
    pub fn metered_inv<M: BudgetMeter>(&self, meter: &mut M) -> Result<Option<Self>, MeterError> {
        let Some(value) = metered_mod_inverse(&self.value, &self.ring.modulus, meter)? else {
            return metered_finish(None, meter);
        };
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular division; mismatched rings and non-unit divisors are refused.
    pub fn metered_div<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let Some(inverse) = other.metered_inv(meter)? else {
            return metered_finish(None, meter);
        };
        self.metered_mul(&inverse, meter)
    }

    /// Cancellation-first nonnegative modular exponentiation.
    pub fn metered_pow<M: BudgetMeter>(
        &self,
        exponent: &BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if exponent.is_negative() {
            return metered_finish(None, meter);
        }
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_mod_pow(&self.value, exponent, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }
}

/// Typed representation of a prime Galois field $\mathbb{F}_p = \mathbb{Z} / p\mathbb{Z}$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FiniteField {
    characteristic: BigInt,
}

impl FiniteField {
    /// Creates a prime finite field $\mathbb{F}_p$ only when the fixed-base primality theorem
    /// certifies `p` exactly. Larger probable primes are refused rather than promoted to fields.
    pub fn new(characteristic: BigInt) -> Option<Self> {
        if is_certified_prime(&characteristic) {
            Some(Self { characteristic })
        } else {
            None
        }
    }

    /// Cancellation-first exact field admission under the same deterministic theorem bound.
    pub fn metered_new<M: BudgetMeter>(
        characteristic: BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        let bound = deterministic_primality_bound();
        if characteristic <= BigInt::one()
            || metered_greater_or_equal(&characteristic, &bound, meter)?
        {
            return metered_finish(None, meter);
        }
        if !metered_is_probable_prime(&characteristic, meter)? {
            return metered_finish(None, meter);
        }
        metered_finish(Some(Self { characteristic }), meter)
    }

    /// Access the prime characteristic $p$.
    pub fn characteristic(&self) -> &BigInt {
        &self.characteristic
    }

    /// Constructs a canonical element in $\mathbb{F}_p$.
    pub fn element(&self, value: BigInt) -> FiniteFieldElement {
        let residue = normalized_remainder(&value, &self.characteristic);
        FiniteFieldElement {
            field: self.clone(),
            value: residue,
        }
    }

    /// Cancellation-first construction of a canonical field element.
    pub fn metered_element<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<FiniteFieldElement, MeterError> {
        let modulus =
            NonZeroBigInt::new(&self.characteristic).expect("FiniteField characteristic invariant");
        let residue = metered_normalized_remainder(value, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.characteristic, meter)?,
        };
        metered_finish(
            FiniteFieldElement {
                field,
                value: residue,
            },
            meter,
        )
    }

    /// The field additive identity $0$.
    pub fn zero(&self) -> FiniteFieldElement {
        self.element(BigInt::zero())
    }

    /// The field multiplicative identity $1$.
    pub fn one(&self) -> FiniteFieldElement {
        self.element(BigInt::one())
    }
}

/// A typed canonical element in a prime finite field $\mathbb{F}_p$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FiniteFieldElement {
    field: FiniteField,
    value: BigInt,
}

impl FiniteFieldElement {
    /// The canonical integer representative in $[0, p)$.
    pub fn value(&self) -> &BigInt {
        &self.value
    }

    /// Reference to the underlying finite field.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Whether this element is $0$.
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Whether this element is $1$.
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Field addition $(a + b) \pmod p$.
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.field != other.field {
            return None;
        }
        let p = &self.field.characteristic;
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: (&self.value + &other.value) % p,
        })
    }

    /// Field subtraction $(a - b) \pmod p$.
    pub fn sub(&self, other: &Self) -> Option<Self> {
        if self.field != other.field {
            return None;
        }
        let p = &self.field.characteristic;
        let mut diff = (&self.value - &other.value) % p;
        if diff.is_negative() {
            diff += p;
        }
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: diff,
        })
    }

    /// Field multiplication $(a \cdot b) \pmod p$.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.field != other.field {
            return None;
        }
        let p = &self.field.characteristic;
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: (&self.value * &other.value) % p,
        })
    }

    /// Field negation $-a \pmod p$.
    pub fn neg(&self) -> Self {
        if self.value.is_zero() {
            self.clone()
        } else {
            FiniteFieldElement {
                field: self.field.clone(),
                value: &self.field.characteristic - &self.value,
            }
        }
    }

    /// Cancellation-first field negation.
    pub fn metered_neg<M: BudgetMeter>(&self, meter: &mut M) -> Result<Self, MeterError> {
        meter.checkpoint()?;
        let value = if self.value.is_zero() {
            metered_clone_bigint(&self.value, meter)?
        } else {
            metered_subtract(&self.field.characteristic, &self.value, meter)?
        };
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Self { field, value }, meter)
    }

    /// Multiplicative inverse $a^{-1} \pmod p$; returns `None` only for $0$.
    pub fn inv(&self) -> Option<Self> {
        if self.value.is_zero() {
            return None;
        }
        let inv_val = mod_inverse(&self.value, &self.field.characteristic)?;
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: inv_val,
        })
    }

    /// Field division $a / b \pmod p$; returns `None` when $b = 0$ or fields mismatch.
    pub fn div(&self, other: &Self) -> Option<Self> {
        if self.field != other.field || other.is_zero() {
            return None;
        }
        let b_inv = other.inv()?;
        self.mul(&b_inv)
    }

    /// Nonnegative exponentiation $a^e \pmod p$. Negative exponents are refused explicitly;
    /// callers may invert a nonzero value and then exponentiate.
    pub fn pow(&self, exp: &BigInt) -> Option<Self> {
        if exp.is_negative() {
            return None;
        }
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: mod_pow(&self.value, exp, &self.field.characteristic),
        })
    }

    /// Cancellation-first field addition. A different parent field is refused.
    pub fn metered_add<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? {
            return metered_finish(None, meter);
        }
        let sum = metered_add(&self.value, &other.value, meter)?;
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_normalized_remainder(&sum, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first field subtraction. A different parent field is refused.
    pub fn metered_sub<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? {
            return metered_finish(None, meter);
        }
        let difference = metered_subtract(&self.value, &other.value, meter)?;
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_normalized_remainder(&difference, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first field multiplication. A different parent field is refused.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? {
            return metered_finish(None, meter);
        }
        let product = metered_mul(&self.value, &other.value, meter)?;
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_normalized_remainder(&product, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first multiplicative inverse. Zero is a computed refusal.
    pub fn metered_inv<M: BudgetMeter>(&self, meter: &mut M) -> Result<Option<Self>, MeterError> {
        if self.value.is_zero() {
            meter.checkpoint()?;
            return metered_finish(None, meter);
        }
        let Some(value) = metered_mod_inverse(&self.value, &self.field.characteristic, meter)?
        else {
            return metered_finish(None, meter);
        };
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first field division. Mismatched fields and zero divisors are refused.
    pub fn metered_div<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? || other.value.is_zero()
        {
            return metered_finish(None, meter);
        }
        let Some(inverse) = other.metered_inv(meter)? else {
            return metered_finish(None, meter);
        };
        self.metered_mul(&inverse, meter)
    }

    /// Cancellation-first nonnegative field exponentiation.
    pub fn metered_pow<M: BudgetMeter>(
        &self,
        exponent: &BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if exponent.is_negative() {
            return metered_finish(None, meter);
        }
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_mod_pow(&self.value, exponent, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }
}

/// Montgomery representation reducer for fast modular arithmetic modulo an odd integer $M$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryReducer {
    modulus: BigInt,
    r: BigInt,
    reduction_bound: BigInt,
    r_shift: u32,
    r2_mod_m: BigInt,
    m_prime: BigInt,
}

impl MontgomeryReducer {
    /// Creates a Montgomery reducer for an odd modulus $M > 1$.
    pub fn new(modulus: BigInt) -> Option<Self> {
        if modulus <= BigInt::one() || (&modulus % &BigInt::from(2i64)).is_zero() {
            return None;
        }
        let bit_len = u32::try_from(modulus.bits()).ok()?;
        let r_shift = bit_len.checked_add(1)?;
        let r = BigInt::one() << r_shift;
        let r_mod_m = &r % &modulus;
        let r2_mod_m = (&r_mod_m * &r_mod_m) % &modulus;
        let m_inv = mod_inverse(&modulus, &r)?;
        let m_prime = &r - m_inv;
        let reduction_bound = &modulus * &r;
        Some(Self {
            modulus,
            r,
            reduction_bound,
            r_shift,
            r2_mod_m,
            m_prime,
        })
    }

    /// Cancellation-first reducer construction. Power-of-two setup is deliberately performed by
    /// metered doublings so cancellation remains observable during large precomputation.
    pub fn metered_new<M: BudgetMeter>(
        modulus: BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if modulus <= BigInt::one() {
            return metered_finish(None, meter);
        }
        let two = BigInt::from(2i64);
        let two_divisor = NonZeroBigInt::new(&two).expect("two is nonzero");
        let (_, parity) = metered_div_rem_nonzero(&modulus, two_divisor, meter)?;
        if parity.is_zero() {
            return metered_finish(None, meter);
        }
        let Some(r_shift) = u32::try_from(modulus.bits())
            .ok()
            .and_then(|bits| bits.checked_add(1))
        else {
            return metered_finish(None, meter);
        };
        let r = metered_power_of_two(r_shift, meter)?;
        let (_, r_mod_m) = {
            let modulus_divisor =
                NonZeroBigInt::new(&modulus).expect("admitted Montgomery modulus is nonzero");
            metered_div_rem_nonzero(&r, modulus_divisor, meter)?
        };
        let r_squared = metered_mul(&r_mod_m, &r_mod_m, meter)?;
        let r2_mod_m = {
            let modulus_divisor =
                NonZeroBigInt::new(&modulus).expect("admitted Montgomery modulus is nonzero");
            metered_normalized_remainder(&r_squared, modulus_divisor, meter)?
        };
        let Some(m_inv) = metered_mod_inverse(&modulus, &r, meter)? else {
            return metered_finish(None, meter);
        };
        let m_prime = metered_subtract(&r, &m_inv, meter)?;
        let reduction_bound = metered_mul(&modulus, &r, meter)?;
        let reducer = Self {
            modulus,
            r,
            reduction_bound,
            r_shift,
            r2_mod_m,
            m_prime,
        };
        metered_finish(Some(reducer), meter)
    }

    /// Access the odd modulus.
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// Converts any signed integer into canonical Montgomery form $a \cdot R \pmod M$.
    pub fn to_montgomery(&self, a: &BigInt) -> BigInt {
        let canonical = normalized_remainder(a, &self.modulus);
        self.reduce_admitted(&(&canonical * &self.r2_mod_m))
    }

    /// Cancellation-first conversion into canonical Montgomery form.
    pub fn metered_to_montgomery<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        let modulus = NonZeroBigInt::new(&self.modulus).expect("Montgomery modulus invariant");
        let canonical = metered_normalized_remainder(value, modulus, meter)?;
        let product = metered_mul(&canonical, &self.r2_mod_m, meter)?;
        self.metered_reduce(&product, meter)
    }

    /// Converts a canonical Montgomery residue back to a standard representative.
    pub fn from_montgomery(&self, value: &BigInt) -> Option<BigInt> {
        is_canonical_residue(value, &self.modulus).then(|| self.reduce_admitted(value))
    }

    /// Cancellation-first conversion back from a canonical Montgomery residue.
    pub fn metered_from_montgomery<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if value.is_negative() || metered_greater_or_equal(value, &self.modulus, meter)? {
            return metered_finish(None, meter);
        }
        self.metered_reduce(value, meter)
    }

    /// Montgomery reduction for the required domain `0 <= T < M*R`. Inputs outside that domain
    /// are refused instead of producing a noncanonical or mathematically invalid value.
    pub fn reduce(&self, value: &BigInt) -> Option<BigInt> {
        if value.is_negative() || value >= &self.reduction_bound {
            return None;
        }
        Some(self.reduce_admitted(value))
    }

    fn reduce_admitted(&self, value: &BigInt) -> BigInt {
        let prod = value * &self.m_prime;
        let r_minus_1 = (&self.r) - BigInt::one();
        let m = &prod & &r_minus_1;
        let u = (value + &m * &self.modulus) >> self.r_shift;
        if u >= self.modulus {
            u - &self.modulus
        } else {
            u
        }
    }

    /// Cancellation-first Montgomery reduction over the same admitted domain.
    pub fn metered_reduce<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if value.is_negative() || metered_greater_or_equal(value, &self.reduction_bound, meter)? {
            return metered_finish(None, meter);
        }
        let product = metered_mul(value, &self.m_prime, meter)?;
        let r_divisor = NonZeroBigInt::new(&self.r).expect("Montgomery R invariant");
        let m = metered_normalized_remainder(&product, r_divisor, meter)?;
        let correction = metered_mul(&m, &self.modulus, meter)?;
        let numerator = metered_add(value, &correction, meter)?;
        let (mut reduced, remainder) = metered_div_rem_nonzero(&numerator, r_divisor, meter)?;
        if !remainder.is_zero() {
            return metered_finish(None, meter);
        }
        if metered_greater_or_equal(&reduced, &self.modulus, meter)? {
            reduced = metered_subtract(&reduced, &self.modulus, meter)?;
        }
        metered_finish(Some(reduced), meter)
    }

    /// Montgomery multiplication over two canonical Montgomery residues.
    pub fn mul(&self, lhs: &BigInt, rhs: &BigInt) -> Option<BigInt> {
        if !is_canonical_residue(lhs, &self.modulus) || !is_canonical_residue(rhs, &self.modulus) {
            return None;
        }
        self.reduce(&(lhs * rhs))
    }

    /// Cancellation-first Montgomery multiplication over canonical operands.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        lhs: &BigInt,
        rhs: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if lhs.is_negative()
            || rhs.is_negative()
            || metered_greater_or_equal(lhs, &self.modulus, meter)?
            || metered_greater_or_equal(rhs, &self.modulus, meter)?
        {
            return metered_finish(None, meter);
        }
        let product = metered_mul(lhs, rhs, meter)?;
        self.metered_reduce(&product, meter)
    }
}

/// Barrett reducer for division-free reduction modulo $M$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarrettReducer {
    modulus: BigInt,
    k: u32,
    mu: BigInt,
    modulus_squared: BigInt,
    b_k_minus_one: BigInt,
    b_k_plus_one: BigInt,
}

impl BarrettReducer {
    /// Creates a Barrett reducer for modulus $M > 1$.
    pub fn new(modulus: BigInt) -> Option<Self> {
        if modulus <= BigInt::one() {
            return None;
        }
        let k = u32::try_from(modulus.bits()).ok()?;
        let two_k = k.checked_mul(2)?;
        let k_minus_one = k.checked_sub(1)?;
        let k_plus_one = k.checked_add(1)?;
        let num = BigInt::one() << two_k;
        let mu = num / &modulus;
        let modulus_squared = &modulus * &modulus;
        let b_k_minus_one = BigInt::one() << k_minus_one;
        let b_k_plus_one = BigInt::one() << k_plus_one;
        Some(Self {
            modulus,
            k,
            mu,
            modulus_squared,
            b_k_minus_one,
            b_k_plus_one,
        })
    }

    /// Cancellation-first reducer construction with metered power-of-two precomputation.
    pub fn metered_new<M: BudgetMeter>(
        modulus: BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if modulus <= BigInt::one() {
            return metered_finish(None, meter);
        }
        let Some(k) = u32::try_from(modulus.bits()).ok() else {
            return metered_finish(None, meter);
        };
        let Some(two_k) = k.checked_mul(2) else {
            return metered_finish(None, meter);
        };
        let Some(k_minus_one) = k.checked_sub(1) else {
            return metered_finish(None, meter);
        };
        let Some(k_plus_one) = k.checked_add(1) else {
            return metered_finish(None, meter);
        };
        let numerator = metered_power_of_two(two_k, meter)?;
        let modulus_divisor =
            NonZeroBigInt::new(&modulus).expect("admitted Barrett modulus is nonzero");
        let (mu, _) = metered_div_rem_nonzero(&numerator, modulus_divisor, meter)?;
        let modulus_squared = metered_mul(&modulus, &modulus, meter)?;
        let b_k_minus_one = metered_power_of_two(k_minus_one, meter)?;
        let b_k_plus_one = metered_power_of_two(k_plus_one, meter)?;
        metered_finish(
            Some(Self {
                modulus,
                k,
                mu,
                modulus_squared,
                b_k_minus_one,
                b_k_plus_one,
            }),
            meter,
        )
    }

    /// Access the positive modulus.
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// Reduces `value` modulo $M$ for the admitted Barrett domain `0 <= value < M^2`.
    /// Negative and out-of-range inputs are refused, making the correction loop bounded.
    pub fn reduce(&self, value: &BigInt) -> Option<BigInt> {
        if value.is_negative() || value >= &self.modulus_squared {
            return None;
        }
        if value < &self.modulus {
            return Some(value.clone());
        }
        let q1 = value >> (self.k - 1);
        let q2 = &q1 * &self.mu;
        let q3 = q2 >> (self.k + 1);
        let mut r = value - &q3 * &self.modulus;
        if r.is_negative() {
            return None;
        }
        for _ in 0..3 {
            if r < self.modulus {
                return Some(r);
            }
            r -= &self.modulus;
        }
        (r < self.modulus).then_some(r)
    }

    /// Cancellation-first Barrett reduction over the same bounded domain.
    pub fn metered_reduce<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if value.is_negative() || metered_greater_or_equal(value, &self.modulus_squared, meter)? {
            return metered_finish(None, meter);
        }
        if !metered_greater_or_equal(value, &self.modulus, meter)? {
            let cloned = metered_clone_bigint(value, meter)?;
            return metered_finish(Some(cloned), meter);
        }
        let k_minus_divisor =
            NonZeroBigInt::new(&self.b_k_minus_one).expect("Barrett power invariant");
        let (q1, _) = metered_div_rem_nonzero(value, k_minus_divisor, meter)?;
        let q2 = metered_mul(&q1, &self.mu, meter)?;
        let k_plus_divisor =
            NonZeroBigInt::new(&self.b_k_plus_one).expect("Barrett power invariant");
        let (q3, _) = metered_div_rem_nonzero(&q2, k_plus_divisor, meter)?;
        let product = metered_mul(&q3, &self.modulus, meter)?;
        let mut reduced = metered_subtract(value, &product, meter)?;
        if reduced.is_negative() {
            return metered_finish(None, meter);
        }
        for _ in 0..3 {
            meter.checkpoint()?;
            if !metered_greater_or_equal(&reduced, &self.modulus, meter)? {
                return metered_finish(Some(reduced), meter);
            }
            reduced = metered_subtract(&reduced, &self.modulus, meter)?;
        }
        if metered_greater_or_equal(&reduced, &self.modulus, meter)? {
            metered_finish(None, meter)
        } else {
            metered_finish(Some(reduced), meter)
        }
    }
}

/// Classification of unlucky prime failures during modular algorithms (e.g. modular GCD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnluckyPrimeReason {
    /// Candidate is nonpositive, composite, or above the exact fixed-base theorem range.
    InvalidPrimeCandidate,
    /// Prime divides the leading coefficient of one or more input polynomials.
    DividesLeadingCoefficient,
    /// Modular reduction causes degree collapse or degenerate structures.
    DegenerateReduction,
    /// Inconsistent modular residues during CRT combination.
    InconsistentResidues,
    /// Prime characteristic is smaller than algorithm coefficient bound.
    ModulusTooSmall,
}

/// Diagnostic record explaining why a chosen prime is unusable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnluckyPrimeDiagnostic {
    /// Rejected candidate; this is diagnostic data, never prime evidence.
    pub prime: BigInt,
    /// Structured refusal class.
    pub reason: UnluckyPrimeReason,
    /// Index of the offending leading coefficient when the reason is coefficient-specific.
    pub coefficient_index: Option<usize>,
}

fn unlucky_prime_diagnostic(
    prime: &BigInt,
    reason: UnluckyPrimeReason,
    coefficient_index: Option<usize>,
) -> UnluckyPrimeDiagnostic {
    UnluckyPrimeDiagnostic {
        prime: prime.clone(),
        reason,
        coefficient_index,
    }
}

/// Verifies an exactly admitted prime against polynomial leading coefficients.
///
/// This bounded diagnostic check does not establish that a prime will be lucky for every later
/// modular-algorithm phase; it only rejects the declared leading-coefficient obstruction.
pub fn check_lucky_prime(
    prime: &BigInt,
    leading_coefficients: &[BigInt],
) -> Result<(), UnluckyPrimeDiagnostic> {
    if !is_certified_prime(prime) {
        return Err(unlucky_prime_diagnostic(
            prime,
            UnluckyPrimeReason::InvalidPrimeCandidate,
            None,
        ));
    }
    for (idx, coeff) in leading_coefficients.iter().enumerate() {
        if (coeff % prime).is_zero() {
            return Err(unlucky_prime_diagnostic(
                prime,
                UnluckyPrimeReason::DividesLeadingCoefficient,
                Some(idx),
            ));
        }
    }
    Ok(())
}

/// Cancellation-first form of [`check_lucky_prime`].
pub fn metered_check_lucky_prime<M: BudgetMeter>(
    prime: &BigInt,
    leading_coefficients: &[BigInt],
    meter: &mut M,
) -> Result<Result<(), UnluckyPrimeDiagnostic>, MeterError> {
    meter.checkpoint()?;
    let bound = deterministic_primality_bound();
    let invalid = prime <= &BigInt::one()
        || metered_greater_or_equal(prime, &bound, meter)?
        || !metered_is_probable_prime(prime, meter)?;
    if invalid {
        let diagnostic = UnluckyPrimeDiagnostic {
            prime: metered_clone_bigint(prime, meter)?,
            reason: UnluckyPrimeReason::InvalidPrimeCandidate,
            coefficient_index: None,
        };
        return metered_finish(Err(diagnostic), meter);
    }
    let divisor = NonZeroBigInt::new(prime).expect("certified prime is nonzero");
    for (index, coefficient) in leading_coefficients.iter().enumerate() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let (_, remainder) = metered_div_rem_nonzero(coefficient, divisor, meter)?;
        if remainder.is_zero() {
            let diagnostic = UnluckyPrimeDiagnostic {
                prime: metered_clone_bigint(prime, meter)?,
                reason: UnluckyPrimeReason::DividesLeadingCoefficient,
                coefficient_index: Some(index),
            };
            return metered_finish(Err(diagnostic), meter);
        }
    }
    metered_finish(Ok(()), meter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, Unbounded};
    use proptest::prelude::*;

    #[derive(Debug, Default)]
    struct CheckpointMeter {
        checkpoints: usize,
        cancel_at: Option<usize>,
        arm_after: Option<usize>,
        armed: bool,
    }

    impl CheckpointMeter {
        fn cancelling_at(checkpoint: usize) -> Self {
            Self {
                checkpoints: 0,
                cancel_at: Some(checkpoint.max(1)),
                arm_after: None,
                armed: false,
            }
        }

        fn arming_after(checkpoint: usize) -> Self {
            Self {
                checkpoints: 0,
                cancel_at: None,
                arm_after: Some(checkpoint),
                armed: false,
            }
        }
    }

    impl BudgetMeter for CheckpointMeter {
        fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
            Ok(())
        }

        fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.armed {
                return Err(MeterError::Cancelled);
            }
            self.checkpoints = self.checkpoints.saturating_add(1);
            if self.cancel_at == Some(self.checkpoints) {
                Err(MeterError::Cancelled)
            } else {
                if self.arm_after == Some(self.checkpoints) {
                    self.armed = true;
                }
                Ok(())
            }
        }
    }

    fn scalar_gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    fn scalar_is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        let mut divisor = 2u64;
        while divisor <= n / divisor {
            if n.is_multiple_of(divisor) {
                return false;
            }
            divisor += 1;
        }
        true
    }

    fn assert_terminal_checkpoint<T: std::fmt::Debug + PartialEq>(
        expected: T,
        expected_checkpoints: usize,
        mut operation: impl FnMut(&mut CheckpointMeter) -> Result<T, MeterError>,
    ) {
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            operation(&mut measured).expect("measurement run completes"),
            expected
        );
        assert_eq!(measured.checkpoints, expected_checkpoints);

        let mut cancelled = CheckpointMeter::arming_after(expected_checkpoints - 1);
        assert!(matches!(
            operation(&mut cancelled),
            Err(MeterError::Cancelled)
        ));
    }

    #[test]
    fn known_gcd_and_bezout_identity() {
        let (g, x, y) = extended_gcd(&BigInt::from(240i64), &BigInt::from(46i64));
        assert_eq!(g, BigInt::from(2i64));
        let lhs = BigInt::from(240i64) * x + BigInt::from(46i64) * y;
        assert_eq!(lhs, g);
        assert_eq!(
            gcd(&BigInt::from(0i64), &BigInt::from(7i64)),
            BigInt::from(7i64)
        );
        assert_eq!(
            gcd(&BigInt::from(0i64), &BigInt::from(0i64)),
            BigInt::from(0i64)
        );
    }

    #[test]
    fn invalid_first_crt_modulus_is_refused_without_division() {
        assert_eq!(crt(&[(1.into(), 0.into())]), None);
        assert_eq!(crt(&[(1.into(), (-7).into())]), None);
    }

    #[test]
    fn rational_reconstruction_handles_zero_and_refuses_degenerate_moduli() {
        assert_eq!(
            rational_reconstruct(&BigInt::zero(), &BigInt::from(101)),
            Some((BigInt::zero(), BigInt::one()))
        );
        assert_eq!(
            rational_reconstruct(&BigInt::from(17), &BigInt::one()),
            None
        );
        assert_eq!(
            rational_reconstruct(&BigInt::from(17), &BigInt::zero()),
            None
        );
    }

    #[test]
    fn prime_stream_and_miller_rabin_match_independent_trial_division() {
        let expected: Vec<u64> = (2..)
            .filter(|value| scalar_is_prime(*value))
            .take(100)
            .collect();
        let actual: Vec<u64> = PrimeStream::new()
            .take(100)
            .map(|value| value.to_u64().unwrap())
            .collect();
        assert_eq!(actual, expected);

        let mut metered_stream = PrimeStream::new();
        let mut meter = Unbounded;
        let metered: Vec<u64> = (0..100)
            .map(|_| {
                metered_stream
                    .next_metered(&mut meter)
                    .unwrap()
                    .to_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(metered, expected);

        for value in 0..10_000u64 {
            assert_eq!(
                is_probable_prime(&BigInt::from(value)),
                scalar_is_prime(value),
                "primality mismatch for {value}"
            );
            if value < 1_000 {
                let mut meter = Unbounded;
                assert_eq!(
                    metered_is_probable_prime(&BigInt::from(value), &mut meter).unwrap(),
                    scalar_is_prime(value),
                    "metered primality mismatch for {value}"
                );
            }
        }
        for carmichael in [561u64, 1_105, 1_729, 2_465, 2_821, 6_601] {
            assert!(!is_probable_prime(&BigInt::from(carmichael)));
            let mut meter = Unbounded;
            assert!(!metered_is_probable_prime(&BigInt::from(carmichael), &mut meter).unwrap());
        }
    }

    proptest! {
        #[test]
        fn exact_division_round_trips_broad_operands(
            dividend_bytes in proptest::collection::vec(any::<u8>(), 0..129),
            divisor_bytes in proptest::collection::vec(any::<u8>(), 0..129),
        ) {
            let dividend = BigInt::from_signed_bytes_be(&dividend_bytes);
            let mut divisor = BigInt::from_signed_bytes_be(&divisor_bytes);
            if divisor.is_zero() {
                divisor = BigInt::one();
            }
            let product = &dividend * &divisor;
            prop_assert_eq!(exact_div(&product, &divisor), Some(dividend));

            if divisor.abs() > BigInt::one() {
                prop_assert_eq!(exact_div(&(product + 1i64), &divisor), None);
            }
        }

        #[test]
        fn metered_exact_division_matches_unmetered_lane(
            dividend_bytes in proptest::collection::vec(any::<u8>(), 0..97),
            divisor_bytes in proptest::collection::vec(any::<u8>(), 0..65),
        ) {
            let dividend = BigInt::from_signed_bytes_be(&dividend_bytes);
            let divisor = BigInt::from_signed_bytes_be(&divisor_bytes);
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_exact_div(&dividend, &divisor, &mut meter).unwrap(),
                exact_div(&dividend, &divisor)
            );
        }

        #[test]
        fn modular_inverse_matches_bounded_scalar_oracle(a in -500i64..500, modulus in 1i64..200) {
            let normalized = a.rem_euclid(modulus);
            let expected = (0..modulus)
                .find(|candidate| (normalized * candidate).rem_euclid(modulus) == 1i64.rem_euclid(modulus));
            let actual = mod_inverse(&BigInt::from(a), &BigInt::from(modulus))
                .map(|value| value.to_i64().unwrap());
            prop_assert_eq!(actual, expected);

            let mut meter = Unbounded;
            let metered = metered_mod_inverse(
                &BigInt::from(a),
                &BigInt::from(modulus),
                &mut meter,
            )
            .unwrap()
            .map(|value| value.to_i64().unwrap());
            prop_assert_eq!(metered, expected);
        }

        #[test]
        fn crt_pair_matches_bounded_exhaustive_oracle(
            remainder_a in -500i64..500,
            modulus_a in 1u64..200,
            remainder_b in -500i64..500,
            modulus_b in 1u64..200,
        ) {
            let gcd = scalar_gcd(modulus_a, modulus_b);
            let lcm = (modulus_a / gcd) * modulus_b;
            let normalized_a = remainder_a.rem_euclid(modulus_a as i64) as u64;
            let normalized_b = remainder_b.rem_euclid(modulus_b as i64) as u64;
            let expected = (0..lcm).find(|candidate| {
                candidate % modulus_a == normalized_a && candidate % modulus_b == normalized_b
            });
            let actual = crt_pair(
                &BigInt::from(remainder_a),
                &BigInt::from(modulus_a),
                &BigInt::from(remainder_b),
                &BigInt::from(modulus_b),
            );
            let mut meter = Unbounded;
            let metered = metered_crt_pair(
                &BigInt::from(remainder_a),
                &BigInt::from(modulus_a),
                &BigInt::from(remainder_b),
                &BigInt::from(modulus_b),
                &mut meter,
            )
            .unwrap();
            prop_assert_eq!(&metered, &actual);
            match (actual, expected) {
                (Some((value, combined_modulus)), Some(expected_value)) => {
                    prop_assert_eq!(value.to_u64(), Some(expected_value));
                    prop_assert_eq!(combined_modulus.to_u64(), Some(lcm));
                }
                (None, None) => {}
                (actual, expected) => prop_assert!(false, "CRT mismatch: {actual:?} vs {expected:?}"),
            }
        }

        #[test]
        fn rational_reconstruction_recovers_unique_small_fraction(
            numerator in -50i64..51,
            denominator in 1u64..51,
        ) {
            prop_assume!(scalar_gcd(numerator.unsigned_abs(), denominator) == 1);
            const MODULUS: i64 = 1_000_003;
            let inverse = mod_inverse(&BigInt::from(denominator), &BigInt::from(MODULUS)).unwrap();
            let residue = ((BigInt::from(numerator) * inverse) % BigInt::from(MODULUS)
                + BigInt::from(MODULUS)) % BigInt::from(MODULUS);
            prop_assert_eq!(
                rational_reconstruct(&residue, &BigInt::from(MODULUS)),
                Some((BigInt::from(numerator), BigInt::from(denominator)))
            );
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_rational_reconstruct(
                    &residue,
                    &BigInt::from(MODULUS),
                    &mut meter,
                )
                .unwrap(),
                Some((BigInt::from(numerator), BigInt::from(denominator)))
            );
        }

        #[test]
        fn metered_rational_reconstruction_matches_unmetered_refusals(
            residue in -20_000i64..20_001,
            modulus in 2i64..10_000,
        ) {
            let residue = BigInt::from(residue);
            let modulus = BigInt::from(modulus);
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_rational_reconstruct(&residue, &modulus, &mut meter).unwrap(),
                rational_reconstruct(&residue, &modulus)
            );
        }

        #[test]
        fn metered_crt_fold_matches_unmetered_lane(
            congruences in proptest::collection::vec((-500i64..500, 1u64..80), 0..6),
        ) {
            let congruences: Vec<(BigInt, BigInt)> = congruences
                .into_iter()
                .map(|(remainder, modulus)| (remainder.into(), modulus.into()))
                .collect();
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_crt(&congruences, &mut meter).unwrap(),
                crt(&congruences)
            );
        }

        #[test]
        fn metered_gcd_and_bezout_match_unmetered_lanes(
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
    }

    #[test]
    fn metered_mul_halts_on_budget_exhaustion() {
        let a = (BigInt::one() << 300) + 12345i64;
        let b = (BigInt::one() << 300) + 67890i64;

        let limits = BudgetLimits::uniform(1, 0);
        let mut budget = Budget::new(limits);
        let err = metered_mul(&a, &b, &mut budget).unwrap_err();
        assert!(matches!(err, MeterError::Budget(_)));

        let limits = BudgetLimits::uniform(1_000_000, 0);
        let mut budget = Budget::new(limits);
        let res = metered_mul(&a, &b, &mut budget).expect("computes within budget");
        assert_eq!(res, &a * &b);
    }

    #[test]
    fn metered_gcd_halts_on_budget_exhaustion() {
        let a = (BigInt::one() << 200) + 1i64;
        let b = (BigInt::one() << 150) + 1i64;

        let limits = BudgetLimits::uniform(2, 0);
        let mut budget = Budget::new(limits);
        let err = metered_gcd(&a, &b, &mut budget).unwrap_err();
        assert!(matches!(err, MeterError::Budget(_)));

        let limits = BudgetLimits::uniform(100, 0);
        let mut budget = Budget::new(limits);
        assert_eq!(
            metered_gcd(&BigInt::zero(), &BigInt::zero(), &mut budget),
            Ok(BigInt::zero())
        );
    }

    #[test]
    fn metered_modular_lanes_refuse_invalid_inputs_without_zero_charges() {
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert_eq!(
            metered_mod_inverse(&BigInt::one(), &BigInt::zero(), &mut budget),
            Ok(None)
        );
        assert_eq!(
            metered_crt_pair(
                &BigInt::zero(),
                &BigInt::zero(),
                &BigInt::zero(),
                &BigInt::one(),
                &mut budget,
            ),
            Ok(None)
        );
        assert_eq!(
            metered_crt(&[(BigInt::zero(), BigInt::zero())], &mut budget),
            Ok(None)
        );
        assert_eq!(
            metered_rational_reconstruct(&BigInt::one(), &BigInt::one(), &mut budget),
            Ok(None)
        );
        assert_eq!(
            metered_is_probable_prime(&BigInt::one(), &mut budget),
            Ok(false)
        );
    }

    #[test]
    fn metered_modular_lanes_halt_on_budget_exhaustion() {
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_mod_inverse(&BigInt::from(17), &BigInt::from(101), &mut budget),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_crt_pair(
                &BigInt::from(3),
                &BigInt::from(5),
                &BigInt::from(4),
                &BigInt::from(7),
                &mut budget,
            ),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_rational_reconstruct(&BigInt::from(27), &BigInt::from(1_000_003), &mut budget,),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_is_probable_prime(&BigInt::from(1_000_003), &mut budget),
            Err(MeterError::Budget(_))
        ));
    }

    #[test]
    fn prime_stream_cancellation_is_retry_safe_and_interleavable() {
        let mut measured_stream = PrimeStream::new();
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            measured_stream.next_metered(&mut measured).unwrap(),
            BigInt::from(2)
        );
        assert!(measured.checkpoints > 1);

        let mut stream = PrimeStream::new();
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            stream.next_metered(&mut cancelled),
            Err(MeterError::Cancelled)
        );
        assert_eq!(stream.current, BigInt::from(2));
        assert!(stream.emitted.is_empty());

        let mut meter = Unbounded;
        assert_eq!(stream.next_metered(&mut meter).unwrap(), BigInt::from(2));
        assert_eq!(stream.next(), Some(BigInt::from(3)));
        assert_eq!(stream.next_metered(&mut meter).unwrap(), BigInt::from(5));
        assert_eq!(stream.next(), Some(BigInt::from(7)));
    }

    #[test]
    fn primality_supports_late_in_algorithm_cancellation() {
        let candidate = (BigInt::one() << 127) - 1i64;
        let mut measured = CheckpointMeter::default();
        let expected = metered_is_probable_prime(&candidate, &mut measured).unwrap();
        assert!(expected);
        assert!(measured.checkpoints > 100);

        let late_checkpoint = measured.checkpoints.saturating_mul(3) / 4;
        let mut cancelled = CheckpointMeter::cancelling_at(late_checkpoint);
        assert_eq!(
            metered_is_probable_prime(&candidate, &mut cancelled),
            Err(MeterError::Cancelled)
        );
        assert!(cancelled.checkpoints > 75);
    }

    #[test]
    fn every_computed_terminal_class_observes_final_cancellation() {
        assert_terminal_checkpoint(None, 53, |meter| {
            metered_exact_div(&BigInt::from(35), &BigInt::from(6), meter)
        });
        assert_terminal_checkpoint(None, 152, |meter| {
            metered_mod_inverse(&BigInt::from(6), &BigInt::from(9), meter)
        });
        assert_terminal_checkpoint(Some(BigInt::from(6)), 232, |meter| {
            metered_mod_inverse(&BigInt::from(17), &BigInt::from(101), meter)
        });
        assert_terminal_checkpoint(None, 56, |meter| {
            metered_crt_pair(
                &BigInt::zero(),
                &BigInt::from(2),
                &BigInt::one(),
                &BigInt::from(2),
                meter,
            )
        });
        assert_terminal_checkpoint(Some((BigInt::zero(), BigInt::one())), 4, |meter| {
            metered_rational_reconstruct(&BigInt::zero(), &BigInt::from(101), meter)
        });
        assert_terminal_checkpoint(None, 668, |meter| {
            metered_rational_reconstruct(&BigInt::from(8), &BigInt::from(101), meter)
        });
        assert_terminal_checkpoint(true, 603, |meter| {
            metered_is_probable_prime(&BigInt::from(41), meter)
        });
        assert_terminal_checkpoint(false, 2_399, |meter| {
            metered_is_probable_prime(&BigInt::from(2_021), meter)
        });
    }

    #[test]
    fn crt_and_reconstruction_support_late_in_algorithm_cancellation() {
        let modulus_a = (BigInt::one() << 127) - 1i64;
        let modulus_b = (BigInt::one() << 89) - 1i64;
        let mut measured_crt = CheckpointMeter::default();
        assert!(
            metered_crt_pair(
                &BigInt::zero(),
                &modulus_a,
                &BigInt::zero(),
                &modulus_b,
                &mut measured_crt,
            )
            .unwrap()
            .is_some()
        );
        assert!(measured_crt.checkpoints > 100);
        let mut cancelled_crt =
            CheckpointMeter::cancelling_at(measured_crt.checkpoints.saturating_mul(3) / 4);
        assert_eq!(
            metered_crt_pair(
                &BigInt::zero(),
                &modulus_a,
                &BigInt::zero(),
                &modulus_b,
                &mut cancelled_crt,
            ),
            Err(MeterError::Cancelled)
        );

        let denominator = BigInt::from(37);
        let inverse = mod_inverse(&denominator, &modulus_a).unwrap();
        let residue = (&BigInt::from(-23) * inverse) % &modulus_a;
        let mut measured_reconstruction = CheckpointMeter::default();
        assert_eq!(
            metered_rational_reconstruct(&residue, &modulus_a, &mut measured_reconstruction,)
                .unwrap(),
            Some((BigInt::from(-23), denominator))
        );
        assert!(measured_reconstruction.checkpoints > 100);
        let mut cancelled_reconstruction = CheckpointMeter::cancelling_at(
            measured_reconstruction.checkpoints.saturating_mul(3) / 4,
        );
        assert_eq!(
            metered_rational_reconstruct(&residue, &modulus_a, &mut cancelled_reconstruction,),
            Err(MeterError::Cancelled)
        );
    }

    #[test]
    fn modular_ring_and_exact_finite_field_preserve_parent_invariants() {
        let ring = ModularRing::new(BigInt::from(12)).expect("modulus > 1");
        let a = ring.element(BigInt::from(7));
        let b = ring.element(BigInt::from(8));
        assert_eq!(a.add(&b).unwrap().value(), &BigInt::from(3));
        assert_eq!(a.sub(&b).unwrap().value(), &BigInt::from(11));
        assert_eq!(a.mul(&b).unwrap().value(), &BigInt::from(8));
        assert_eq!(a.inv().unwrap().value(), &BigInt::from(7));
        assert!(b.inv().is_none());
        assert!(a.pow(&BigInt::from(-1)).is_none());
        let other_ring = ModularRing::new(BigInt::from(13)).unwrap();
        assert!(a.add(&other_ring.one()).is_none());

        let ff = FiniteField::new(BigInt::from(17)).expect("17 is prime");
        assert!(FiniteField::new(BigInt::from(18)).is_none());
        assert!(FiniteField::new(deterministic_primality_bound()).is_none());
        let x = ff.element(BigInt::from(5));
        let y = ff.element(BigInt::from(11));
        assert_eq!(x.add(&y).unwrap().value(), &BigInt::from(16));
        assert_eq!(x.sub(&y).unwrap().value(), &BigInt::from(11));
        assert_eq!(x.mul(&y).unwrap().value(), &BigInt::from(4));
        let x_inv = x.inv().expect("5 is invertible mod 17");
        assert_eq!(x_inv.value(), &BigInt::from(7));
        assert_eq!(y.div(&x).unwrap().value(), y.mul(&x_inv).unwrap().value());
        assert_eq!(x.pow(&BigInt::from(16)).unwrap().value(), &BigInt::one());
        assert!(x.pow(&BigInt::from(-1)).is_none());
        let other_field = FiniteField::new(BigInt::from(19)).unwrap();
        assert!(x.mul(&other_field.one()).is_none());

        let mut meter = Unbounded;
        assert_eq!(
            ring.metered_element(&BigInt::from(-5), &mut meter)
                .unwrap()
                .value(),
            &BigInt::from(7)
        );
        let mut meter = Unbounded;
        assert_eq!(
            a.metered_mul(&b, &mut meter).unwrap().unwrap().value(),
            a.mul(&b).unwrap().value()
        );
        let mut meter = Unbounded;
        assert_eq!(
            FiniteField::metered_new(BigInt::from(17), &mut meter).unwrap(),
            Some(ff.clone())
        );
        let mut meter = Unbounded;
        assert_eq!(
            x.metered_pow(&BigInt::from(16), &mut meter)
                .unwrap()
                .unwrap(),
            x.pow(&BigInt::from(16)).unwrap()
        );
    }

    #[test]
    fn montgomery_and_barrett_reducers_enforce_their_input_domains() {
        let m = BigInt::from(97);
        let mont = MontgomeryReducer::new(m.clone()).expect("valid odd modulus");
        let a = BigInt::from(35);
        let b = BigInt::from(42);
        let expected_prod = (&a * &b) % &m;

        let a_r = mont.to_montgomery(&a);
        let b_r = mont.to_montgomery(&b);
        let prod_r = mont.mul(&a_r, &b_r).unwrap();
        let actual_prod = mont.from_montgomery(&prod_r).unwrap();
        assert_eq!(actual_prod, expected_prod);
        assert_eq!(
            mont.from_montgomery(&mont.to_montgomery(&BigInt::from(-3))),
            Some(BigInt::from(94))
        );
        assert_eq!(mont.reduce(&BigInt::from(-1)), None);
        assert_eq!(mont.reduce(&mont.reduction_bound), None);
        assert_eq!(mont.mul(&BigInt::from(-1), &b_r), None);

        let barrett = BarrettReducer::new(m.clone()).expect("valid modulus");
        for v in [0i64, 1, 35, 96, 97, 100, 500, 9000] {
            let x = BigInt::from(v);
            assert_eq!(barrett.reduce(&x), Some(&x % &m));
        }
        assert_eq!(barrett.reduce(&BigInt::from(-1)), None);
        assert_eq!(barrett.reduce(&barrett.modulus_squared), None);

        let mut meter = Unbounded;
        let metered_mont = MontgomeryReducer::metered_new(m.clone(), &mut meter)
            .unwrap()
            .unwrap();
        assert_eq!(metered_mont, mont);
        let mut meter = Unbounded;
        let a_r_metered = metered_mont
            .metered_to_montgomery(&a, &mut meter)
            .unwrap()
            .unwrap();
        assert_eq!(a_r_metered, a_r);
        let mut meter = Unbounded;
        assert_eq!(
            metered_mont.metered_mul(&a_r, &b_r, &mut meter).unwrap(),
            Some(prod_r)
        );

        let mut meter = Unbounded;
        let metered_barrett = BarrettReducer::metered_new(m.clone(), &mut meter)
            .unwrap()
            .unwrap();
        assert_eq!(metered_barrett, barrett);
        let mut meter = Unbounded;
        assert_eq!(
            metered_barrett
                .metered_reduce(&BigInt::from(9000), &mut meter)
                .unwrap(),
            Some(BigInt::from(76))
        );
    }

    #[test]
    fn unlucky_prime_diagnostics_are_structured_bounded_and_fail_closed() {
        let p_lucky = BigInt::from(17);
        let p_unlucky = BigInt::from(5);
        let leading_coeffs = vec![BigInt::from(15), BigInt::from(28)];

        assert!(check_lucky_prime(&p_lucky, &leading_coeffs).is_ok());
        let diag = check_lucky_prime(&p_unlucky, &leading_coeffs).unwrap_err();
        assert_eq!(diag.reason, UnluckyPrimeReason::DividesLeadingCoefficient);
        assert_eq!(diag.prime, p_unlucky);
        assert_eq!(diag.coefficient_index, Some(0));

        for invalid in [
            BigInt::from(-3),
            BigInt::zero(),
            BigInt::one(),
            BigInt::from(9),
            deterministic_primality_bound(),
        ] {
            let diag = check_lucky_prime(&invalid, &leading_coeffs).unwrap_err();
            assert_eq!(diag.reason, UnluckyPrimeReason::InvalidPrimeCandidate);
            assert_eq!(diag.coefficient_index, None);
        }

        let mut meter = Unbounded;
        assert_eq!(
            metered_check_lucky_prime(&p_lucky, &leading_coeffs, &mut meter).unwrap(),
            Ok(())
        );
        let mut meter = Unbounded;
        assert_eq!(
            metered_check_lucky_prime(&BigInt::zero(), &leading_coeffs, &mut meter)
                .unwrap()
                .unwrap_err()
                .reason,
            UnluckyPrimeReason::InvalidPrimeCandidate
        );
    }

    #[test]
    fn new_metered_types_check_cancellation_before_terminal_publication() {
        let ring = ModularRing::new(BigInt::from(97)).unwrap();
        let lhs = ring.element(BigInt::from(35));
        let rhs = ring.element(BigInt::from(42));
        let mut measured = CheckpointMeter::default();
        assert!(lhs.metered_mul(&rhs, &mut measured).unwrap().is_some());
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            lhs.metered_mul(&rhs, &mut cancelled),
            Err(MeterError::Cancelled)
        );
        let other_ring = ModularRing::new(BigInt::from(101)).unwrap();
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            lhs.metered_add(&other_ring.one(), &mut measured).unwrap(),
            None
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            lhs.metered_add(&other_ring.one(), &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let field = FiniteField::new(BigInt::from(97)).unwrap();
        let value = field.element(BigInt::from(35));
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            value.metered_pow(&BigInt::from(-1), &mut measured).unwrap(),
            None
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            value.metered_pow(&BigInt::from(-1), &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let mut measured = CheckpointMeter::default();
        let reducer = MontgomeryReducer::metered_new(BigInt::from(97), &mut measured)
            .unwrap()
            .unwrap();
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            MontgomeryReducer::metered_new(BigInt::from(97), &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let value = reducer.to_montgomery(&BigInt::from(35));
        let mut measured = CheckpointMeter::default();
        assert!(
            reducer
                .metered_from_montgomery(&value, &mut measured)
                .unwrap()
                .is_some()
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            reducer.metered_from_montgomery(&value, &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let coefficients = [BigInt::from(15), BigInt::from(28)];
        let mut measured = CheckpointMeter::default();
        assert!(
            metered_check_lucky_prime(&BigInt::from(5), &coefficients, &mut measured)
                .unwrap()
                .is_err()
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            metered_check_lucky_prime(&BigInt::from(5), &coefficients, &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            MontgomeryReducer::metered_new(BigInt::from(97), &mut budget),
            Err(MeterError::Budget(_))
        ));
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            BarrettReducer::metered_new(BigInt::from(97), &mut budget),
            Err(MeterError::Budget(_))
        ));
    }

    proptest! {
        #[test]
        fn reducers_match_scalar_remainders_over_their_full_admitted_ranges(
            modulus in 2u64..500,
            value_seed in any::<u64>(),
            lhs in -10_000i64..10_001,
            rhs in -10_000i64..10_001,
        ) {
            let modulus_big = BigInt::from(modulus);
            let modulus_squared = modulus.saturating_mul(modulus);
            let value = value_seed % modulus_squared;
            let barrett = BarrettReducer::new(modulus_big.clone()).unwrap();
            let expected = BigInt::from(value % modulus);
            prop_assert_eq!(barrett.reduce(&BigInt::from(value)), Some(expected.clone()));
            let mut meter = Unbounded;
            prop_assert_eq!(
                barrett.metered_reduce(&BigInt::from(value), &mut meter).unwrap(),
                Some(expected)
            );

            if modulus > 2 && modulus % 2 == 1 {
                let montgomery = MontgomeryReducer::new(modulus_big.clone()).unwrap();
                let lhs_mont = montgomery.to_montgomery(&BigInt::from(lhs));
                let rhs_mont = montgomery.to_montgomery(&BigInt::from(rhs));
                let product_mont = montgomery.mul(&lhs_mont, &rhs_mont).unwrap();
                let product = montgomery.from_montgomery(&product_mont).unwrap();
                let expected_product =
                    BigInt::from(lhs.rem_euclid(modulus as i64))
                        * BigInt::from(rhs.rem_euclid(modulus as i64))
                        % &modulus_big;
                prop_assert_eq!(product, expected_product);
            }
        }
    }
}
