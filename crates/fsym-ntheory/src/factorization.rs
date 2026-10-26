//! Bounded deterministic integer factorization and a separate reference checker.
//!
//! Brent's cycle search only proposes divisors. Accepted factorizations pass a
//! second, non-searching implementation that checks every prime and the exact
//! product. Both lanes are finite synchronous `u64` operations; this module does
//! not claim structured cancellation or protected runtime verifier resources.

use crate::{NTheoryError, is_prime};
use std::collections::BTreeMap;

/// Hard ceiling on generator work units, including all restarts and cofactors.
///
/// Each trial remainder, modular product and Euclidean remainder is charged;
/// each exact-owner primality query reserves 4,096 units for its fixed 64-bit
/// input and thirteen witnesses. Verification has its own fixed input bound.
pub const MAX_FACTORIZATION_WORK: u64 = 8_000_000;

const SMALL_PRIMES: [u64; 25] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
];

struct Work {
    remaining: u64,
}

impl Work {
    fn charge(&mut self, amount: u64, cofactor: u64) -> Result<(), NTheoryError> {
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or(NTheoryError::FactorizationLimitExceeded(cofactor))?;
        Ok(())
    }

    fn step(&mut self, value: u64, constant: u64, modulus: u64) -> Result<u64, NTheoryError> {
        self.charge(2, modulus)?;
        Ok(
            ((u128::from(value) * u128::from(value) + u128::from(constant)) % u128::from(modulus))
                as u64,
        )
    }

    fn gcd(&mut self, mut left: u64, mut right: u64, n: u64) -> Result<u64, NTheoryError> {
        while right != 0 {
            self.charge(1, n)?;
            (left, right) = (right, left % right);
        }
        Ok(left)
    }
}

fn brent_divisor(n: u64, work: &mut Work) -> Result<u64, NTheoryError> {
    // Restarts and seed choice are deterministic. A failed cycle cannot reset
    // accounting, and no random seed or external state affects the result.
    for attempt in 1..=64u64 {
        let constant = (2 * attempt + 1) % n;
        let mut y = (attempt * attempt + 2) % n;
        let mut radius = 1usize;
        while radius <= (1 << 20) {
            let x = y;
            for _ in 0..radius {
                y = work.step(y, constant, n)?;
            }
            let mut traversed = 0;
            let mut failed_cycle = false;
            while traversed < radius {
                let saved_y = y;
                let block = (radius - traversed).min(128);
                let mut product = 1u64;
                for _ in 0..block {
                    y = work.step(y, constant, n)?;
                    work.charge(1, n)?;
                    product =
                        ((u128::from(product) * u128::from(x.abs_diff(y))) % u128::from(n)) as u64;
                }
                let factor = work.gcd(product, n, n)?;
                if factor > 1 && factor < n {
                    return Ok(factor);
                }
                if factor == n {
                    // A batch may contain both prime divisors. Replay only
                    // this batch, taking separate gcds to recover either one.
                    y = saved_y;
                    for _ in 0..block {
                        y = work.step(y, constant, n)?;
                        let factor = work.gcd(x.abs_diff(y), n, n)?;
                        if factor > 1 && factor < n {
                            return Ok(factor);
                        }
                        if factor == n {
                            break;
                        }
                    }
                    failed_cycle = true;
                    break;
                }
                traversed += block;
            }
            if failed_cycle {
                break;
            }
            radius *= 2;
        }
    }
    Err(NTheoryError::FactorizationLimitExceeded(n))
}

/// Compute the complete prime factorization of a positive `u64` integer.
///
/// Small trial divisors are followed by deterministic Pollard–Brent splitting.
/// A work refusal returns no partial factor map. The accepted answer is checked
/// independently, including primality and every multiplicity, before return.
pub fn factorint(n: u64) -> Result<BTreeMap<u64, u32>, NTheoryError> {
    factorint_with_work_limit(n, MAX_FACTORIZATION_WORK)
}

/// Factor with a caller-selected work limit capped at [`MAX_FACTORIZATION_WORK`].
///
/// This is a deterministic operation bound, not a wall-clock timeout or an
/// asynchronous cancellation interface. All restart attempts share the limit.
pub fn factorint_with_work_limit(n: u64, limit: u64) -> Result<BTreeMap<u64, u32>, NTheoryError> {
    if n == 0 {
        return Err(NTheoryError::ZeroFactorization);
    }
    let mut work = Work {
        remaining: limit.min(MAX_FACTORIZATION_WORK),
    };
    let mut factors = BTreeMap::new();
    let mut remaining = n;
    for prime in SMALL_PRIMES {
        if remaining == 1 {
            break;
        }
        loop {
            work.charge(1, remaining)?;
            if !remaining.is_multiple_of(prime) {
                break;
            }
            remaining /= prime;
            *factors.entry(prime).or_insert(0) += 1;
        }
    }

    // There can be at most 63 nontrivial factors, counted with multiplicity.
    // A fixed array avoids input-dependent stack allocation or recursion.
    let mut pending = [0u64; 64];
    let mut count = usize::from(remaining > 1);
    pending[0] = remaining;
    while count > 0 {
        count -= 1;
        let cofactor = pending[count];
        work.charge(4_096, cofactor)?;
        if is_prime(cofactor) {
            *factors.entry(cofactor).or_insert(0) += 1;
            continue;
        }
        let divisor = brent_divisor(cofactor, &mut work)?;
        // Validate the proposed split even though Brent's gcd constructs it.
        if divisor <= 1 || divisor >= cofactor || !cofactor.is_multiple_of(divisor) {
            return Err(NTheoryError::FactorizationVerificationFailed(
                "invalid divisor",
            ));
        }
        if count + 2 > pending.len() {
            return Err(NTheoryError::FactorizationVerificationFailed(
                "factor stack overflow",
            ));
        }
        pending[count] = divisor;
        pending[count + 1] = cofactor / divisor;
        count += 2;
    }
    verify_prime_factorization(n, &factors)?;
    Ok(factors)
}

// Deliberately separate from the generator's exact-owner BigInt primality
// operation and Brent products. Left-to-right exponentiation uses only u128
// arithmetic, whose product is exact for two residues below any u64 modulus.
fn reference_power(base: u64, exponent: u64, modulus: u64) -> u64 {
    let modulus = u128::from(modulus);
    let base = u128::from(base) % modulus;
    let mut result = 1u128;
    for bit in (0..64).rev() {
        result = result * result % modulus;
        if exponent & (1u64 << bit) != 0 {
            result = result * base % modulus;
        }
    }
    result as u64
}

pub(crate) fn reference_prime(n: u64) -> bool {
    // These are the same thirteen mathematical witnesses as the WS03 exact
    // owner's theorem range (<3.317e24), containing the entire u64 universe.
    // This independent scalar checker is not the public primality engine.
    const BASES: [u64; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];
    if n < 2 {
        return false;
    }
    for prime in BASES {
        if n == prime {
            return true;
        }
        if n.is_multiple_of(prime) {
            return false;
        }
    }
    let twos = (n - 1).trailing_zeros();
    let odd_part = (n - 1) >> twos;
    'witness: for base in BASES {
        let mut value = reference_power(base, odd_part, n);
        if value == 1 || value == n - 1 {
            continue;
        }
        for _ in 1..twos {
            value = (u128::from(value) * u128::from(value) % u128::from(n)) as u64;
            if value == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Independently verify a complete prime factorization, including multiplicities.
///
/// The map is treated as untrusted. Input cardinality and exponents are checked
/// before arithmetic. Prime witnesses use a separate fixed-width reference
/// implementation; checked multiplication binds the whole result to `n`.
pub fn verify_prime_factorization(
    n: u64,
    factors: &BTreeMap<u64, u32>,
) -> Result<(), NTheoryError> {
    let fail = |message| NTheoryError::FactorizationVerificationFailed(message);
    if n == 0 {
        return Err(NTheoryError::ZeroFactorization);
    }
    if factors.len() > 63 {
        return Err(fail("too many prime factors"));
    }
    let mut product = 1u64;
    for (&prime, &multiplicity) in factors {
        if !(1..=63).contains(&multiplicity) || !reference_prime(prime) {
            return Err(fail("nonprime factor or invalid multiplicity"));
        }
        for _ in 0..multiplicity {
            product = product
                .checked_mul(prime)
                .ok_or_else(|| fail("factor product overflow"))?;
            if product > n {
                return Err(fail("factor product exceeds the input"));
            }
        }
    }
    if product != n {
        return Err(fail("factor product does not equal the input"));
    }
    Ok(())
}
