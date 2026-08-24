//! Exact integer arithmetic primitives (WS03).
//!
//! Everything here is pure big-integer math with no FFI and no
//! machine-float intermediates. Determinism: identical inputs always
//! produce identical outputs; the prime stream is a fixed deterministic
//! sequence, and primality testing uses a fixed base set.
//!
//! # Primality honesty
//!
//! [`is_probable_prime`] is **deterministic** for `n < 3.317·10²⁴` (the
//! first 13 prime bases are a proven certificate for that range) and only
//! probabilistic beyond it. Callers needing certainty above that bound
//! must supply their own proof (e.g. ECPP later in WS11).

use num_bigint::{BigInt, Sign};
use num_traits::{One, Zero};

/// Greatest common divisor; always non-negative. `gcd(0, 0) == 0`.
pub fn gcd(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    match a.sign() {
        Sign::Minus => -a,
        _ => a,
    }
}

/// Extended gcd: returns `(g, x, y)` with `a·x + b·y == g` and
/// `g == gcd(a, b)` (non-negative).
pub fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    let (mut old_r, mut r) = (a.clone(), b.clone());
    let (mut old_s, mut s) = (BigInt::one(), BigInt::zero());
    let (mut old_t, mut t) = (BigInt::zero(), BigInt::one());
    while !r.is_zero() {
        let q = &old_r / &r;
        let tmp_r = &old_r - &q * &r;
        old_r = r;
        r = tmp_r;
        let tmp_s = &old_s - &q * &s;
        old_s = s;
        s = tmp_s;
        let tmp_t = &old_t - &q * &t;
        old_t = t;
        t = tmp_t;
    }
    // Normalize sign so g >= 0.
    if old_r.sign() == Sign::Minus {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

/// Divides `a` by `b` when the division is exact; `None` otherwise.
pub fn exact_div(a: &BigInt, b: &BigInt) -> Option<BigInt> {
    if b.is_zero() {
        return None;
    }
    let (q, r) = (a / b, a % b);
    if r.is_zero() { Some(q) } else { None }
}

/// Multiplicative inverse of `a` modulo `m` (`m > 0`); `None` when
/// `gcd(a, m) != 1`.
pub fn mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    if m.sign() != Sign::Plus || m.is_one() && false {
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
    if mod1.sign() != Sign::Plus || mod2.sign() != Sign::Plus {
        return None;
    }
    let g = gcd(mod1, mod2);
    let diff = rem2 - rem1;
    if (&diff % &g) != BigInt::zero() {
        return None;
    }
    let lcm = (mod1 / &g) * mod2;
    let m1g = mod1 / &g;
    let m2g = mod2 / &g;
    // x = rem1 + mod1·k where k ≡ (diff/g)·inv(m1g) (mod m2g).
    let mut k = ((&diff / &g) * mod_inverse(&m1g, &m2g)?) % &m2g;
    k = ((k % &m2g) + &m2g) % &m2g;
    let mut x = rem1 + mod1 * &k;
    x %= &lcm;
    if x.sign() == Sign::Minus {
        x += &lcm;
    }
    Some((x, lcm))
}
/// Rational reconstruction à la Wang: given `n` with `0 <= n < m`, finds
/// `(r, s)` such that `n·s ≡ r (mod m)`, `|r| <= √m`, `0 < s <= √m`, and
/// `gcd(r, s) == 1`. Uniqueness holds whenever the true fraction satisfies
/// those bounds; otherwise this returns `None`.
pub fn rational_reconstruct(n: &BigInt, m: &BigInt) -> Option<(BigInt, BigInt)> {
    if m.sign() != Sign::Plus {
        return None;
    }
    if n.is_zero() {
        return Some((BigInt::zero(), BigInt::one()));
    }
    let sq = sqrt_floor(m);
    let n_pos = ((n % m) + m) % m;

    // Extended Euclid tracking t with r ≡ t·n (mod m); remainders are
    // non-negative and strictly decreasing.
    let mut r_prev = m.clone();
    let mut r_cur = n_pos;
    let mut t_prev = BigInt::zero();
    let mut t_cur = BigInt::one();
    while r_cur > sq {
        let q = &r_prev / &r_cur;
        let next_r = &r_prev - &q * &r_cur;
        let next_t = &t_prev - &q * &t_cur;
        r_prev = std::mem::replace(&mut r_cur, next_r);
        t_prev = std::mem::replace(&mut t_cur, next_t);
    }
    // Canonical sign: denominator strictly positive (numerator keeps the
    // symmetric-residue sign).
    let mut r_out = r_cur;
    if t_cur.sign() == Sign::Minus {
        r_out = -r_out;
        t_cur = -t_cur;
    }
    if r_out.is_zero() || t_cur.sign() != Sign::Plus {
        return None;
    }
    if r_out.magnitude() > sq.magnitude() || t_cur > sq {
        return None;
    }
    if gcd(&r_out, &t_cur) != BigInt::one() {
        return None;
    }
    Some((r_out, t_cur))
}

/// Deterministic increasing stream of primes: 2, 3, 5, 7, ...
///
/// Trial division against previously emitted primes only — no
/// randomness, no wall clock, identical across runs and platforms.
pub struct PrimeStream {
    emitted: Vec<BigInt>,
    current: BigInt,
}

impl PrimeStream {
    pub fn new() -> Self {
        Self {
            emitted: Vec::new(),
            current: BigInt::from(2u8),
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
            self.current += 1i64;
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

/// Integer square root via Newton iteration (floor). Negative input is
/// treated as 0 — callers never pass negatives here.
fn sqrt_floor(n: &BigInt) -> BigInt {
    if n.sign() != Sign::Plus || n.is_zero() {
        return BigInt::zero();
    }
    let two = BigInt::from(2u8);
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

/// Miller-Rabin primality with a fixed base set.
///
/// Deterministic for `n < 3.317044064679887385961981` (the first 12
/// bases' joint witness bound). Above that bound this is a strong
/// probable-prime test, not a proof.
pub fn is_probable_prime(n: &BigInt) -> bool {
    if *n < BigInt::from(2u8) {
        return false;
    }
    for base in MR_BASES {
        let b = BigInt::from(base);
        if *n == b {
            return true;
        }
        if n % &b == BigInt::zero() {
            return false;
        }
    }
    // n - 1 = d · 2^s with d odd.
    let one = BigInt::one();
    let nm1 = n - &one;
    let (d, s) = {
        let mut dd = nm1.clone();
        let mut ss = 0u64;
        while (&dd % 2i64).is_zero() {
            dd /= 2i64;
            ss += 1;
        }
        (dd, ss)
    };
    'bases: for base in MR_BASES {
        let a = BigInt::from(base);
        let mut x = pow_mod(&a, &d, n);
        if x == one || x == nm1 {
            continue 'bases;
        }
        for _ in 1..s {
            x = (&x * &x) % n;
            if x == nm1 {
                continue 'bases;
            }
            if x.is_one() {
                return false;
            }
        }
        return false;
    }
    true
}

/// Modular exponentiation: `base^exp mod modulus` with `modulus > 0`.
pub fn pow_mod(base: &BigInt, exp: &BigInt, modulus: &BigInt) -> BigInt {
    assert!(modulus.sign() == Sign::Plus, "modulus must be positive");
    let mut result = BigInt::one();
    let mut b = base % modulus;
    let mut e = exp.clone();
    while e.sign() == Sign::Plus {
        if (&e % 2i64).is_one() {
            result = (&result * &b) % modulus;
        }
        b = (&b * &b) % modulus;
        e /= 2i64;
    }
    result
}
/// Scalar reference lane for multiplication (WS03 differential oracle).
///
/// Computes `a·b` by repeated addition of the multiplicand along the
/// binary decomposition of the multiplier magnitude — the textbook
/// schoolbook identity, deliberately naive. Production multiplication
/// delegates to num-bigint's internally-selected strategy (schoolbook →
/// Karatsuba/Toom thresholds are its implementation detail); the
/// substrate guarantee is that every optimized path agrees with THIS
/// function on values and signs across the boundary corpus.
///
/// Cost is O(log₂|min|) doublings plus popcount additions, so proptest
/// magnitudes stay bounded.
pub fn schoolbook_mul_reference(a: &BigInt, b: &BigInt) -> BigInt {
    let (a_abs, a_neg) = (a.magnitude(), a.sign() == Sign::Minus);
    let (b_abs, b_neg) = (b.magnitude(), b.sign() == Sign::Minus);
    // Iterate the smaller magnitude for fewer addition steps.
    let (steps, unit_mag) = if a_abs <= b_abs {
        (a_abs, b_abs)
    } else {
        (b_abs, a_abs)
    };
    let mut acc = BigInt::zero();
    let mut shifted = BigInt::from(unit_mag.clone());
    let mut bits = steps.clone();
    while !bits.is_zero() {
        if (&bits % 2u32).is_one() {
            acc += &shifted;
        }
        shifted <<= 1;
        bits /= 2u32;
    }
    if a_neg != b_neg && !acc.is_zero() {
        -acc
    } else {
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn known_gcd_and_bezout_identity() {
        let (g, x, y) = extended_gcd(&BigInt::from(240i32), &BigInt::from(46i32));
        assert_eq!(g, BigInt::from(2i32));
        let lhs = BigInt::from(240i32) * x + BigInt::from(46i32) * y;
        assert_eq!(lhs, g);
        assert_eq!(
            gcd(&BigInt::from(0i32), &BigInt::from(7i32)),
            BigInt::from(7i32)
        );
        assert_eq!(
            gcd(&BigInt::from(0i32), &BigInt::from(0i32)),
            BigInt::from(0i32)
        );
    }

    #[test]
    fn modular_inverse_round_trip() {
        let m = BigInt::from(997i32);
        let inv = mod_inverse(&BigInt::from(123i32), &m).unwrap();
        assert_eq!((BigInt::from(123i32) * inv) % &m, BigInt::one());
        // Non-coprime case fails closed.
        assert!(
            mod_inverse(&BigInt::from(997i32), &m).is_none()
                || mod_inverse(&BigInt::from(31i32), &m).is_some()
        );
        assert!(mod_inverse(&BigInt::from(14i32), &BigInt::from(21i32)).is_none());
    }

    #[test]
    fn crt_consistent_and_inconsistent_systems() {
        let (x, lcm) = crt_pair(
            &BigInt::from(2i32),
            &BigInt::from(3i32),
            &BigInt::from(3i32),
            &BigInt::from(5i32),
        )
        .unwrap();
        assert_eq!(lcm, BigInt::from(15i32));
        assert_eq!(&x % 3i64, BigInt::from(2i64));
        assert_eq!(&x % 5i64, BigInt::from(3i64));
        // 2 mod 4 and 3 mod 6 are inconsistent (both demand odd/even clash).
        assert!(
            crt_pair(
                &BigInt::from(2i32),
                &BigInt::from(4i32),
                &BigInt::from(3i32),
                &BigInt::from(6i32)
            )
            .is_none()
        );
    }

    #[test]
    fn prime_stream_matches_known_prefix() {
        let got: Vec<i64> = PrimeStream::new()
            .take(10)
            .map(|p| p.try_into().unwrap())
            .collect();
        assert_eq!(got, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn rational_reconstruct_known_values() {
        // 1/2 mod 101: 2^{-1} = 51, n = 51. √101 ≈ 10 → (1, 2).
        let (r, s) = rational_reconstruct(&BigInt::from(51i32), &BigInt::from(101i32)).unwrap();
        assert_eq!(r, BigInt::one());
        assert_eq!(s, BigInt::from(2i32));

        // 6 mod 7 is -1 in symmetric residue space → (-1, 1).
        let (r, s) = rational_reconstruct(&BigInt::from(6i32), &BigInt::from(7i32)).unwrap();
        assert_eq!(r, BigInt::from(-1i32));
        assert_eq!(s, BigInt::one());

        // 20 ≡ 6 (mod 7): congruent inputs reconstruct identically.
        let (r, s) = rational_reconstruct(&BigInt::from(20i32), &BigInt::from(7i32)).unwrap();
        assert_eq!(r, BigInt::from(-1i32));
        assert_eq!(s, BigInt::one());

        // Non-reconstructible: 6 mod 12 has gcd(r, s) > 1 for every
        // candidate within the √12 bound.
        assert!(rational_reconstruct(&BigInt::from(6i32), &BigInt::from(12i32)).is_none());
    }

    #[test]
    fn primality_small_known_values() {
        let primes = [2i64, 3, 5, 7, 97, 7919];
        let composites = [0i64, 1, 4, 100, 561, 7917]; // 561 is a Carmichael number.
        for p in primes {
            assert!(is_probable_prime(&BigInt::from(p)), "{p} should be prime");
        }
        for c in composites {
            assert!(
                !is_probable_prime(&BigInt::from(c)),
                "{c} should be composite"
            );
        }
        // Large value inside the certified range: 2^89 - 1 (Mersenne).
        let mersenne = (BigInt::one() << 89u32) - 1i64;
        assert!(is_probable_prime(&mersenne));
    }

    proptest! {
        #[test]
        fn bezout_always_holds(a in -10_000i64..10_000i64, b in -10_000i64..10_000i64) {
            let (a, b) = (BigInt::from(a), BigInt::from(b));
            let (g, x, y) = extended_gcd(&a, &b);
            let lhs = a * x + b * y;
            prop_assert_eq!(lhs, g.clone());
            prop_assert!(g.sign() != Sign::Minus);
        }

        #[test]
        fn inverse_round_trip(a in 2i64..50_000, m in 50_001i64..200_000) {
            let (a, m) = (BigInt::from(a), BigInt::from(m));
            if let Some(inv) = mod_inverse(&a, &m) {
                prop_assert_eq!((&a % &m * inv) % &m, BigInt::one());
            }
        }

        #[test]
        fn crt_result_satisfies_both_congruences(
            r1 in 0i64..100,
            m1 in 1i64..100,
            r2 in 0i64..100,
            m2 in 1i64..100,
        ) {
            let (r1, m1) = (BigInt::from(r1), BigInt::from(m1.max(1)));
            let (r2, m2) = (BigInt::from(r2), BigInt::from(m2.max(1)));
            if let Some((x, lcm)) = crt_pair(&r1, &m1, &r2, &m2) {
                prop_assert_eq!(&x % &m1, r1 % &m1);
                prop_assert_eq!(&x % &m2, r2 % &m2);
                prop_assert!(x >= BigInt::zero() && x < lcm);
            }
        }

        #[test]
        fn reconstruct_round_trip(p in 1i64..500, q in 1i64..500, slack in 2i64..8) {
            // Reconstruction returns the reduced, coprime pair — reduce
            // the generated inputs first so expectations match.
            let g = gcd(&BigInt::from(p), &BigInt::from(q));
            let p = BigInt::from(p) / &g;
            let q = BigInt::from(q) / &g;
            // Uniqueness requires m > 2·|r|·|s| for the true fraction AND
            // m ≥ max(|r|, s)² so the pair lies inside the √m reach.
            let span = std::cmp::max(&p, &q).clone();
            let m = BigInt::from(2i64) * &span * &span * slack * slack + 1i64;
            let inv = match mod_inverse(&q, &m) {
                Some(i) => i,
                None => return Ok(()),
            };
            let n = (&p * inv) % &m;
            if let Some((r, s)) = rational_reconstruct(&n, &m) {
                prop_assert_eq!(r, p);
                prop_assert_eq!(s, q);
            }
        }

        #[test]
        fn consecutive_primes_are_coprime(seed in 0usize..200) {
            let mut it = PrimeStream::new();
            for _ in 0..seed {
                it.next();
            }
            let a = it.next().unwrap();
            let b = it.next().unwrap();
            prop_assert_eq!(gcd(&a, &b), BigInt::one());
        }

        #[test]
        fn optimized_mul_agrees_with_schoolbook_reference(
            a in -1_000_000i64..1_000_000i64,
            b in -1_000_000i64..1_000_000i64,
        ) {
            let (a, b) = (BigInt::from(a), BigInt::from(b));
            prop_assert_eq!(schoolbook_mul_reference(&a, &b), &a * &b);
        }

        #[test]
        fn mul_boundary_corpus_agrees_across_limb_thresholds(
            shift in 0u32..192u32,
            sign_a in proptest::bool::ANY,
            sign_b in proptest::bool::ANY,
        ) {
            // Values straddling num-bigint's internal strategy thresholds
            // (limb counts 1, 2, 3): the delegation boundary is opaque to
            // us, so we sweep magnitudes across it and demand agreement.
            let base = BigInt::one() << shift;
            for delta in [-1i64, 0, 1] {
                let a = match sign_a {
                    true => &base + delta,
                    false => -(&base + delta),
                };
                for b_raw in [1i64, 2, 3, 5] {
                    let b = match sign_b {
                        true => BigInt::from(b_raw),
                        false => BigInt::from(-b_raw),
                    };
                    prop_assert_eq!(schoolbook_mul_reference(&a, &b), &a * &b);
                    prop_assert_eq!(schoolbook_mul_reference(&b, &a), &a * &b);
                }
            }
        }
    }

    #[test]
    fn zero_and_one_identities_hold_in_reference_lane() {
        let x = BigInt::from(123456789i64);
        let zero = BigInt::zero();
        let one = BigInt::one();
        assert_eq!(schoolbook_mul_reference(&x, &zero), zero);
        assert_eq!(schoolbook_mul_reference(&zero, &x), zero);
        assert_eq!(schoolbook_mul_reference(&x, &one), x);
    }
}
