//! Exact integer arithmetic primitives (WS03).
//!
//! Everything here is pure big-integer math with no FFI and no
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

pub use fsym_bigint::{
    BigInt, DEFAULT_STRATEGY_THRESHOLD_BITS, LIMB_BITS, NonZeroBigInt, Strategy as MulStrategy,
    limb_count_u64, metered_div_rem, metered_div_rem_nonzero, metered_multiply as metered_mul,
    multiply, multiply_with_strategy as mul_with_strategy, select_strategy,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};

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
    let (q, r) = a.div_rem(b);
    if r.is_zero() { Some(q) } else { None }
}

/// Cancellation-first exact division using the metered scalar division lane.
pub fn metered_exact_div<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<Option<BigInt>, MeterError> {
    let Some((quotient, remainder)) = metered_div_rem(a, b, meter)? else {
        return Ok(None);
    };
    if remainder.is_zero() {
        Ok(Some(quotient))
    } else {
        Ok(None)
    }
}

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
        return Ok(None);
    }
    Ok(Some(metered_normalized_remainder(&x, modulus, meter)?))
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
        return Ok(None);
    }
    let (m1_div_g, m1_remainder) = metered_div_rem_nonzero(mod1, g_divisor, meter)?;
    let (m2_div_g, m2_remainder) = metered_div_rem_nonzero(mod2, g_divisor, meter)?;
    if !m1_remainder.is_zero() || !m2_remainder.is_zero() {
        return Ok(None);
    }

    let lcm = metered_mul(&m1_div_g, mod2, meter)?;
    let (_, u, _) = metered_extended_gcd(&m1_div_g, &m2_div_g, meter)?;
    let scaled_diff = metered_mul(&diff_over_g, &u, meter)?;
    let shift = metered_mul(&scaled_diff, mod1, meter)?;
    let shifted_remainder = metered_add(rem1, &shift, meter)?;
    let Some(lcm_divisor) = NonZeroBigInt::new(&lcm) else {
        return Ok(None);
    };
    let x = metered_normalized_remainder(&shifted_remainder, lcm_divisor, meter)?;
    meter.checkpoint()?;
    Ok(Some((x, lcm)))
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
    if congruences.is_empty() {
        return Ok(Some((BigInt::zero(), BigInt::one())));
    }
    let (first_remainder, first_modulus) = &congruences[0];
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
    for (remainder, next_modulus) in &congruences[1..] {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let Some((next_x, combined_modulus)) =
            metered_crt_pair(&x, &modulus, remainder, next_modulus, meter)?
        else {
            return Ok(None);
        };
        x = next_x;
        modulus = combined_modulus;
    }
    meter.checkpoint()?;
    Ok(Some((x, modulus)))
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
        return Ok(Some((BigInt::zero(), BigInt::one())));
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

    while r_cur > bound {
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
    if !t_cur.is_positive() || t_cur > bound {
        return Ok(None);
    }
    if metered_gcd(&r_out, &t_cur, meter)? != BigInt::one() {
        return Ok(None);
    }
    let residue_times_denominator = metered_mul(&residue, &t_cur, meter)?;
    let congruence_delta = metered_subtract(&r_out, &residue_times_denominator, meter)?;
    if !metered_normalized_remainder(&congruence_delta, modulus, meter)?.is_zero() {
        return Ok(None);
    }
    meter.checkpoint()?;
    Ok(Some((r_out, t_cur)))
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
                if *prime > root {
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
                        candidate.limb_count().max(1).saturating_mul(8),
                    ),
                    (Dimension::AllocationCount, 1),
                ])?;
                meter.checkpoint()?;
                self.current = next_current;
                self.emitted.push(candidate.clone());
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
            return Ok(true);
        }
        let Some(divisor) = NonZeroBigInt::new(&divisor) else {
            return Ok(false);
        };
        let (_, remainder) = metered_div_rem_nonzero(n, divisor, meter)?;
        if remainder.is_zero() {
            return Ok(false);
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
        if &a >= n {
            continue;
        }
        let mut x = metered_mod_pow(&a, &d, modulus, meter)?;
        if x.is_one() || x == n_minus_one {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let square = metered_mul(&x, &x, meter)?;
            x = metered_normalized_remainder(&square, modulus, meter)?;
            if x == n_minus_one {
                composite = false;
                break;
            }
        }
        if composite {
            return Ok(false);
        }
    }
    meter.checkpoint()?;
    Ok(true)
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
        if next >= x {
            return Ok(x);
        }
        x = next;
    }
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
        let (_, r) = metered_div_rem_nonzero(&a, divisor, meter)?;
        a = b;
        b = r;
    }
    meter.checkpoint()?;
    Ok(if a.is_negative() { -a } else { a })
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
        let (q, tmp_r) = metered_div_rem_nonzero(&old_r, divisor, meter)?;
        old_r = r;
        r = tmp_r;
        let q_times_s = metered_mul(&q, &s, meter)?;
        let tmp_s = metered_subtract(&old_s, &q_times_s, meter)?;
        old_s = s;
        s = tmp_s;
        let q_times_t = metered_mul(&q, &t, meter)?;
        let tmp_t = metered_subtract(&old_t, &q_times_t, meter)?;
        old_t = t;
        t = tmp_t;
    }
    meter.checkpoint()?;
    if old_r.is_negative() {
        Ok((-old_r, -old_s, -old_t))
    } else {
        Ok((old_r, old_s, old_t))
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

fn metered_add<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    let output_limbs = lhs.limb_count().max(rhs.limb_count()).saturating_add(1);
    meter.charge_batch(&[
        (Dimension::ComputeSteps, output_limbs.max(1)),
        (Dimension::MemoryBytes, output_limbs.saturating_mul(8)),
        (Dimension::AllocationCount, 1),
    ])?;
    Ok(lhs + rhs)
}

fn metered_negate<M: BudgetMeter>(value: BigInt, meter: &mut M) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge(Dimension::ComputeSteps, value.limb_count().max(1))?;
    Ok(-value)
}

fn metered_subtract<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    let output_limbs = lhs.limb_count().max(rhs.limb_count()).saturating_add(1);
    meter.charge_batch(&[
        (Dimension::ComputeSteps, output_limbs.max(1)),
        (Dimension::MemoryBytes, output_limbs.saturating_mul(8)),
        (Dimension::AllocationCount, 1),
    ])?;
    Ok(lhs - rhs)
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
    }

    impl CheckpointMeter {
        fn cancelling_at(checkpoint: usize) -> Self {
            Self {
                checkpoints: 0,
                cancel_at: Some(checkpoint.max(1)),
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
            self.checkpoints = self.checkpoints.saturating_add(1);
            if self.cancel_at == Some(self.checkpoints) {
                Err(MeterError::Cancelled)
            } else {
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
}
