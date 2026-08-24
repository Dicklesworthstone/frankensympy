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

        for value in 0..10_000u64 {
            assert_eq!(
                is_probable_prime(&BigInt::from(value)),
                scalar_is_prime(value),
                "primality mismatch for {value}"
            );
        }
        for carmichael in [561u64, 1_105, 1_729, 2_465, 2_821, 6_601] {
            assert!(!is_probable_prime(&BigInt::from(carmichael)));
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
}
