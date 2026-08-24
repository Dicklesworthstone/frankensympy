//! # fsym-ntheory
//!
//! Number-theoretic functions: prime generation, Miller-Rabin primality test,
//! prime factorization, Euler's totient, divisor sigma, modular arithmetic,
//! and continued fractions.

#![forbid(unsafe_code)]

use fsym_core::BigInt;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NTheoryError {
    #[error("Cannot factor negative integer or zero")]
    NonPositiveFactorization,
    #[error("Modular inverse does not exist for {0} mod {1}")]
    NoModularInverse(String, String),
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
pub fn factorint(mut n: u64) -> Result<BTreeMap<u64, u32>, NTheoryError> {
    if n == 0 {
        return Err(NTheoryError::NonPositiveFactorization);
    }
    let mut factors = BTreeMap::new();
    // Factor out 2s
    while n.is_multiple_of(2) {
        *factors.entry(2).or_insert(0) += 1;
        n /= 2;
    }
    // Trial division by odds
    let mut p = 3;
    while p * p <= n {
        while n.is_multiple_of(p) {
            *factors.entry(p).or_insert(0) += 1;
            n /= p;
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

/// Extended Euclidean Algorithm returning (gcd, x, y) such that a*x + b*y = gcd(a, b).
pub fn egcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        (a.clone(), BigInt::one(), BigInt::zero())
    } else {
        let (q, r) = a.div_rem(b);
        let (g, x, y) = egcd(b, &r);
        (g, y.clone(), x - q * y)
    }
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
    }
}
