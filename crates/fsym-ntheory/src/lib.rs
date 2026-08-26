//! # fsym-ntheory
//!
//! Number-theoretic functions: deterministic `u64` Miller-Rabin primality,
//! bounded trial-division factorization, Euler's totient and divisor functions,
//! the exact-arithmetic owner's Jacobi symbol, and arbitrary-precision extended GCD.

#![forbid(unsafe_code)]

use fsym_core::{BigInt, arith::jacobi_symbol as exact_jacobi_symbol};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NTheoryError {
    #[error("Cannot factor zero")]
    ZeroFactorization,
    #[error("Trial-division factorization limit reached with unresolved cofactor {0}")]
    FactorizationLimitExceeded(u64),
    #[error("Exact result does not fit u64 while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("n should be an odd positive integer")]
    InvalidJacobiDenominator,
    #[error("Invalid system for Chinese Remainder Theorem")]
    InvalidCRTSystem,
    #[error("Moduli in Chinese Remainder Theorem must be pairwise coprime")]
    NonCoprimeModuli,
}

/// Computes modular inverse of `a` modulo `m` such that `(a * x) % m == 1`.
pub fn mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    if m <= &BigInt::from(1) {
        return None;
    }
    let (g, x, _) = egcd(a, m);
    if g == BigInt::from(1) {
        let res = (x % m + m) % m;
        Some(res)
    } else {
        None
    }
}

/// Solves a system of modular congruences using the Chinese Remainder Theorem:
/// $x \equiv r_i \pmod{m_i}$ for pairwise coprime moduli $m_i > 1$.
/// Returns the unique solution $0 \le x < \prod m_i$.
pub fn crt(remainders: &[BigInt], moduli: &[BigInt]) -> Result<BigInt, NTheoryError> {
    if remainders.len() != moduli.len() || remainders.is_empty() {
        return Err(NTheoryError::InvalidCRTSystem);
    }
    for m in moduli {
        if m <= &BigInt::from(1) {
            return Err(NTheoryError::InvalidCRTSystem);
        }
    }
    let mut total_mod = BigInt::from(1);
    for m in moduli {
        total_mod *= m;
    }

    let mut result = BigInt::from(0);
    for (r, m) in remainders.iter().zip(moduli.iter()) {
        let m_i = &total_mod / m;
        let z_i = mod_inverse(&m_i, m).ok_or(NTheoryError::NonCoprimeModuli)?;
        let term = ((r % m + m) % m) * &m_i * z_i;
        result += term;
    }
    Ok(result % total_mod)
}

/// Deterministic primality test for small numbers (<= 2^64) using Miller-Rabin with optimal bases.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 || n == 5 || n == 7 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) || n.is_multiple_of(5) || n.is_multiple_of(7) {
        return false;
    }
    if n < 121 {
        return true;
    }

    // Factor n-1 as 2^s * d
    let mut d = n - 1;
    let mut s = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    // Deterministic witness set for u64
    let bases: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &a in bases {
        if a >= n {
            break;
        }
        if !miller_rabin_test(n, a, d, s) {
            return false;
        }
    }
    true
}

fn mod_pow(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp /= 2;
    }
    result
}

fn miller_rabin_test(n: u64, a: u64, d: u64, s: u32) -> bool {
    let n_128 = n as u128;
    let mut x = mod_pow(a as u128, d as u128, n_128);
    if x == 1 || x == n_128 - 1 {
        return true;
    }
    for _ in 1..s {
        x = (x * x) % n_128;
        if x == n_128 - 1 {
            return true;
        }
    }
    false
}

/// Compute prime factorization of an integer: n -> {p_1: e_1, p_2: e_2, ...}.
///
/// Prime cofactors are recognized with deterministic Miller-Rabin. Composite
/// cofactors that survive trial divisors through one million receive a typed
/// refusal rather than running without a work bound.
pub fn factorint(mut n: u64) -> Result<BTreeMap<u64, u32>, NTheoryError> {
    const MAX_TRIAL_DIVISOR: u64 = 1_000_000;

    if n == 0 {
        return Err(NTheoryError::ZeroFactorization);
    }
    let mut factors = BTreeMap::new();
    // Factor out 2s
    while n.is_multiple_of(2) {
        *factors.entry(2).or_insert(0) += 1;
        n /= 2;
    }
    if n > 1 && is_prime(n) {
        *factors.entry(n).or_insert(0) += 1;
        return Ok(factors);
    }
    // Trial division by odds
    let mut p = 3;
    while p <= n / p {
        if p > MAX_TRIAL_DIVISOR {
            return Err(NTheoryError::FactorizationLimitExceeded(n));
        }
        let mut divided = false;
        while n.is_multiple_of(p) {
            *factors.entry(p).or_insert(0) += 1;
            n /= p;
            divided = true;
        }
        if divided && n > 1 && is_prime(n) {
            *factors.entry(n).or_insert(0) += 1;
            return Ok(factors);
        }
        p += 2;
    }
    if n > 1 {
        *factors.entry(n).or_insert(0) += 1;
    }
    Ok(factors)
}

/// Euler's totient function: φ(n) = count of integers 1 <= k <= n coprime to n.
pub fn totient(n: u64) -> Result<u64, NTheoryError> {
    if n == 0 {
        return Ok(0);
    }
    let factors = factorint(n)?;
    let mut result = n;
    for (p, _) in factors {
        result = (result / p) * (p - 1);
    }
    Ok(result)
}

/// Mobius function: μ(n) = 1 if n is square-free with even number of prime factors, -1 if odd, 0 if n has a squared prime factor.
pub fn mobius(n: u64) -> Result<i64, NTheoryError> {
    if n == 0 {
        return Err(NTheoryError::ZeroFactorization);
    }
    if n == 1 {
        return Ok(1);
    }
    let factors = factorint(n)?;
    for exp in factors.values() {
        if *exp > 1 {
            return Ok(0);
        }
    }
    if factors.len() % 2 == 1 {
        Ok(-1)
    } else {
        Ok(1)
    }
}

/// Number of divisors function: d(n) = \prod (e_i + 1).
pub fn divisor_count(n: u64) -> Result<u64, NTheoryError> {
    if n == 0 {
        return Err(NTheoryError::ZeroFactorization);
    }
    let factors = factorint(n)?;
    let mut count = 1u64;
    for exp in factors.values() {
        count *= (*exp as u64) + 1;
    }
    Ok(count)
}

/// Sum of k-th powers of divisors: \sigma_k(n) = \prod \frac{p^{k(e+1)} - 1}{p^k - 1}.
pub fn divisor_sum(n: u64, k: u32) -> Result<u64, NTheoryError> {
    if n == 0 {
        return Err(NTheoryError::ZeroFactorization);
    }
    if k == 0 {
        return divisor_count(n);
    }
    let factors = factorint(n)?;
    let mut total = 1u64;
    for (p, exp) in factors {
        let pk = p
            .checked_pow(k)
            .ok_or(NTheoryError::ArithmeticOverflow("divisor sum prime power"))?;
        let mut term = 1u64;
        let mut cur_pk = 1u64;
        for _ in 0..exp {
            cur_pk = cur_pk
                .checked_mul(pk)
                .ok_or(NTheoryError::ArithmeticOverflow("divisor sum factor power"))?;
            term = term
                .checked_add(cur_pk)
                .ok_or(NTheoryError::ArithmeticOverflow("divisor sum factor"))?;
        }
        total = total
            .checked_mul(term)
            .ok_or(NTheoryError::ArithmeticOverflow("divisor sum product"))?;
    }
    Ok(total)
}

/// Jacobi symbol (a / n) for integer `a` and odd positive integer `n`.
///
/// The arithmetic is delegated to the arbitrary-precision WS03 owner. Invalid
/// denominators are a typed refusal, distinct from the legitimate symbol `0`.
pub fn jacobi_symbol(a: i64, n: u64) -> Result<i64, NTheoryError> {
    exact_jacobi_symbol(&BigInt::from(a), &BigInt::from(n))
        .map(i64::from)
        .ok_or(NTheoryError::InvalidJacobiDenominator)
}

/// Extended Euclidean Algorithm returning (gcd, x, y) such that a*x + b*y = gcd(a, b).
pub fn egcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    a.extended_gcd(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(997));
        assert!(!is_prime(1000));
        assert!(is_prime(1_000_000_007));
    }

    #[test]
    fn test_factorint_and_totient() {
        let factors = factorint(360).unwrap();
        // 360 = 2^3 * 3^2 * 5^1
        assert_eq!(factors.get(&2), Some(&3));
        assert_eq!(factors.get(&3), Some(&2));
        assert_eq!(factors.get(&5), Some(&1));

        // totient(360) = 360 * (1/2) * (2/3) * (4/5) = 96
        assert_eq!(totient(360).unwrap(), 96);

        let hard_semiprime = 1_000_003u64 * 1_000_003;
        assert!(is_prime(1_000_003));
        assert_eq!(
            factorint(hard_semiprime),
            Err(NTheoryError::FactorizationLimitExceeded(hard_semiprime))
        );
    }

    #[test]
    fn test_mobius_and_divisors() {
        // Mobius: mu(1) = 1, mu(2) = -1, mu(6) = 1, mu(12) = 0
        assert_eq!(mobius(1).unwrap(), 1);
        assert_eq!(mobius(2).unwrap(), -1);
        assert_eq!(mobius(6).unwrap(), 1);
        assert_eq!(mobius(12).unwrap(), 0);

        // Divisor count: d(12) = 6 (1, 2, 3, 4, 6, 12)
        assert_eq!(divisor_count(12).unwrap(), 6);

        // Divisor sum: sigma_1(12) = 1+2+3+4+6+12 = 28
        assert_eq!(divisor_sum(12, 1).unwrap(), 28);
        assert_eq!(
            divisor_sum(2, 64),
            Err(NTheoryError::ArithmeticOverflow("divisor sum prime power"))
        );

        // Jacobi symbol: (2 / 7) = 1, (3 / 7) = -1
        assert_eq!(jacobi_symbol(2, 7), Ok(1));
        assert_eq!(jacobi_symbol(3, 7), Ok(-1));
        assert_eq!(jacobi_symbol(-1, (1u64 << 63) + 1), Ok(1));
        assert_eq!(jacobi_symbol(i64::MIN, 3), Ok(1));
        assert_eq!(jacobi_symbol(3, 9), Ok(0));
        assert_eq!(
            jacobi_symbol(1, 0),
            Err(NTheoryError::InvalidJacobiDenominator)
        );
        assert_eq!(
            jacobi_symbol(1, 2),
            Err(NTheoryError::InvalidJacobiDenominator)
        );
    }

    #[test]
    fn extended_gcd_normalizes_negative_inputs() {
        let a = BigInt::from(-30);
        let b = BigInt::from(12);
        let (gcd, x, y) = egcd(&a, &b);
        assert_eq!(gcd, BigInt::from(6));
        assert_eq!(a * x + b * y, gcd);
    }

    #[test]
    fn arithmetic_functions_match_small_reference_lanes() {
        fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
            while right != 0 {
                (left, right) = (right, left % right);
            }
            left
        }

        for n in 1..=200u64 {
            let factors = factorint(n).unwrap();
            let reconstructed = factors.iter().fold(1u64, |product, (&prime, &exponent)| {
                assert!(is_prime(prime));
                product * prime.pow(exponent)
            });
            assert_eq!(reconstructed, n);

            let reference_totient = (1..=n).filter(|&value| gcd_u64(value, n) == 1).count() as u64;
            assert_eq!(totient(n), Ok(reference_totient));

            for power in 0..=2u32 {
                let reference_sum = (1..=n)
                    .filter(|divisor| n.is_multiple_of(*divisor))
                    .map(|divisor| divisor.pow(power))
                    .sum();
                assert_eq!(divisor_sum(n, power), Ok(reference_sum));
            }
        }

        for prime in (3..100u64).filter(|value| is_prime(*value)) {
            for value in -50..=50i64 {
                let reduced = i128::from(value).rem_euclid(i128::from(prime)) as u128;
                let residue = mod_pow(reduced, u128::from((prime - 1) / 2), u128::from(prime));
                assert!(matches!(residue, 0 | 1) || residue == u128::from(prime - 1));
                let reference = match residue {
                    0 => 0,
                    1 => 1,
                    _ => -1,
                };
                assert_eq!(jacobi_symbol(value, prime), Ok(reference));
            }
        }

        for denominator in (1..=199u64).step_by(2) {
            for value in -200..=200i64 {
                let expected = fsym_core::arith::jacobi_symbol(
                    &BigInt::from(value),
                    &BigInt::from(denominator),
                )
                .map(i64::from)
                .ok_or(NTheoryError::InvalidJacobiDenominator);
                assert_eq!(jacobi_symbol(value, denominator), expected);
            }
        }
    }

    #[test]
    fn test_chinese_remainder_theorem_and_modular_inverse() {
        let a = BigInt::from(3);
        let m = BigInt::from(7);
        let inv = mod_inverse(&a, &m).unwrap();
        assert_eq!((&a * &inv) % &m, BigInt::from(1));

        // System:
        // x = 2 mod 3
        // x = 3 mod 5
        // x = 2 mod 7
        // Solution: x = 23 mod 105
        let remainders = vec![BigInt::from(2), BigInt::from(3), BigInt::from(2)];
        let moduli = vec![BigInt::from(3), BigInt::from(5), BigInt::from(7)];
        let sol = crt(&remainders, &moduli).unwrap();
        assert_eq!(sol, BigInt::from(23));

        // Refusal on non-coprime moduli:
        let non_coprime = vec![BigInt::from(4), BigInt::from(6)];
        let rem = vec![BigInt::from(1), BigInt::from(3)];
        assert_eq!(crt(&rem, &non_coprime), Err(NTheoryError::NonCoprimeModuli));
    }
}
