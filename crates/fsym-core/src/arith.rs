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
    BigInt, DEFAULT_STRATEGY_THRESHOLD_BITS, LIMB_BITS, Strategy as MulStrategy, limb_count_u64,
    metered_multiply as metered_mul, multiply, multiply_with_strategy as mul_with_strategy,
    select_strategy,
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

/// Solves an arbitrary system of simultaneous congruences.
pub fn crt(congruences: &[(BigInt, BigInt)]) -> Option<(BigInt, BigInt)> {
    if congruences.is_empty() {
        return Some((BigInt::zero(), BigInt::one()));
    }
    let (mut x, mut m) = congruences[0].clone();
    x %= &m;
    if x.is_negative() {
        x += &m;
    }
    for (r_i, m_i) in &congruences[1..] {
        let (next_x, next_m) = crt_pair(&x, &m, r_i, m_i)?;
        x = next_x;
        m = next_m;
    }
    Some((x, m))
}

/// Wang's rational reconstruction: recovers `(r, s)` with `gcd(r, s) == 1`,
/// `s > 0`, and `r · s⁻¹ ≡ n (mod m)`, bounded by `2·r_max·s_max < m`.
pub fn rational_reconstruct(n: &BigInt, m: &BigInt) -> Option<(BigInt, BigInt)> {
    if !m.is_positive() {
        return None;
    }
    let sq = sqrt_floor(m);
    let (mut r_prev, mut r_cur) = (m.clone(), (n % m + m) % m);
    let (mut t_prev, mut t_cur) = (BigInt::zero(), BigInt::one());

    while r_cur.abs() > sq {
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
    if r_out.is_zero() || !t_cur.is_positive() {
        return None;
    }
    if r_out.abs() > sq || t_cur > sq {
        return None;
    }
    if gcd(&r_out, &t_cur) != BigInt::one() {
        return None;
    }
    Some((r_out, t_cur))
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
        let b = BigInt::from(base as i64);
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
        let a = BigInt::from(base as i64);
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

/// Metered greatest common divisor with step accounting and cancellation checkpoints.
pub fn metered_gcd<M: BudgetMeter>(
    a: &BigInt,
    b: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        meter.checkpoint()?;
        let b_limbs = b.limb_count().max(1);
        meter.charge(Dimension::ComputeSteps, b_limbs)?;
        let r = &a % &b;
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
    let (mut old_r, mut r) = (a.clone(), b.clone());
    let (mut old_s, mut s) = (BigInt::one(), BigInt::zero());
    let (mut old_t, mut t) = (BigInt::zero(), BigInt::one());
    while !r.is_zero() {
        meter.checkpoint()?;
        let r_limbs = r.limb_count().max(1);
        meter.charge(Dimension::ComputeSteps, r_limbs)?;
        let q = &old_r / &r;
        let tmp_r = &old_r - (&q * &r);
        old_r = r;
        r = tmp_r;
        let tmp_s = &old_s - (&q * &s);
        old_s = s;
        s = tmp_s;
        let tmp_t = &old_t - (&q * &t);
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

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits};

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
    }
}
