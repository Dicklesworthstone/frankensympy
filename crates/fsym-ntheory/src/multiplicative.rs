//! Bounded exact arithmetic in the unit groups of `u64` residue rings.
//!
//! Generation uses the existing independently checked factorization owner.
//! Acceptance of an order uses a separate scalar power implementation and the
//! prime-divisor minimality criterion, without calling the order generator.
//! These are synchronous in-process checks, not portable proof-kernel receipts.

use crate::{NTheoryError, factorint, verify_prime_factorization};
use std::collections::BTreeMap;

/// Maximum modular products and Euclidean steps in one group operation.
///
/// At most two factorizations are also requested, each retaining the existing
/// `MAX_FACTORIZATION_WORK` bound. These operations do not enumerate a group.
pub const MAX_MODULAR_GROUP_WORK: u64 = 8_000_000;

pub(crate) struct Work {
    remaining: u64,
}

impl Work {
    pub(crate) fn new(remaining: u64) -> Self {
        Self { remaining }
    }

    pub(crate) fn charge(&mut self) -> Result<(), NTheoryError> {
        self.remaining = self
            .remaining
            .checked_sub(1)
            .ok_or(NTheoryError::ModularGroupWorkLimitExceeded)?;
        Ok(())
    }

    pub(crate) fn gcd(&mut self, mut a: u64, mut b: u64) -> Result<u64, NTheoryError> {
        while b != 0 {
            self.charge()?;
            (a, b) = (b, a % b);
        }
        Ok(a)
    }

    pub(crate) fn multiply(&mut self, a: u64, b: u64, n: u64) -> Result<u64, NTheoryError> {
        self.charge()?;
        Ok((u128::from(a) * u128::from(b) % u128::from(n)) as u64)
    }

    pub(crate) fn power(&mut self, mut a: u64, mut e: u64, n: u64) -> Result<u64, NTheoryError> {
        self.charge()?;
        a %= n;
        let mut value = 1 % n;
        while e != 0 {
            if e & 1 != 0 {
                value = self.multiply(value, a, n)?;
            }
            e >>= 1;
            if e != 0 {
                a = self.multiply(a, a, n)?;
            }
        }
        Ok(value)
    }
}

fn admit_unit(a: u64, n: u64, work: &mut Work) -> Result<u64, NTheoryError> {
    if n <= 1 {
        return Err(NTheoryError::InvalidGroupModulus);
    }
    let a = a % n;
    if work.gcd(a, n)? != 1 {
        return Err(NTheoryError::NotCoprime);
    }
    Ok(a)
}

fn totient_from_factors(n: u64, factors: &BTreeMap<u64, u32>) -> u64 {
    factors.keys().fold(n, |phi, p| (phi / p) * (p - 1))
}

// The checker deliberately does not call Work::power or the generator's public
// modular-power helper. All intermediates fit u128 for every u64 modulus.
fn reference_power(a: u64, e: u64, n: u64) -> u64 {
    let modulus = u128::from(n);
    let base = u128::from(a % n);
    let mut result = 1u128;
    for bit in (0..64).rev() {
        result = result * result % modulus;
        if (e >> bit) & 1 != 0 {
            result = result * base % modulus;
        }
    }
    result as u64
}

/// Check that `order` is the least positive exponent with `a^order = 1 (mod n)`.
///
/// The supplied factor map is untrusted and must be a complete prime
/// factorization of `order`. For each prime divisor q, `a^(order/q) != 1`
/// establishes minimality once `a^order = 1` is checked. No factorization or
/// multiplicative-order search is performed by this checker.
pub fn verify_multiplicative_order(
    a: u64,
    n: u64,
    order: u64,
    factors: &BTreeMap<u64, u32>,
) -> Result<(), NTheoryError> {
    let fail = |reason| NTheoryError::MultiplicativeOrderVerificationFailed(reason);
    if n <= 1 || order == 0 {
        return Err(fail("invalid modulus or zero order"));
    }
    verify_prime_factorization(order, factors)
        .map_err(|_| fail("invalid complete factorization of the order"))?;
    if reference_power(a, order, n) != 1 {
        return Err(fail("the proposed exponent does not give one"));
    }
    for prime in factors.keys() {
        if reference_power(a, order / prime, n) == 1 {
            return Err(fail("a smaller prime-divisor exponent already gives one"));
        }
    }
    Ok(())
}

/// Return the multiplicative order of a unit modulo any `u64` modulus > 1.
pub fn multiplicative_order(a: u64, n: u64) -> Result<u64, NTheoryError> {
    multiplicative_order_with_work_limit(a, n, MAX_MODULAR_GROUP_WORK)
}

/// As `multiplicative_order`, with a caller-selected modular-operation limit.
/// The existing independent factorization limits are not changed.
pub fn multiplicative_order_with_work_limit(
    a: u64,
    n: u64,
    work_limit: u64,
) -> Result<u64, NTheoryError> {
    let mut work = Work::new(work_limit);
    order_with_work(a, n, &mut work)
}

pub(crate) fn order_with_work(a: u64, n: u64, work: &mut Work) -> Result<u64, NTheoryError> {
    let a = admit_unit(a, n, work)?;
    if a == 1 {
        return Ok(1);
    }
    let phi = totient_from_factors(n, &factorint(n)?);
    let mut factors = factorint(phi)?;
    let mut order = phi;
    for (&prime, exponent) in &mut factors {
        while *exponent != 0 && work.power(a, order / prime, n)? == 1 {
            order /= prime;
            *exponent -= 1;
        }
    }
    factors.retain(|_, exponent| *exponent != 0);
    verify_multiplicative_order(a, n, order, &factors)?;
    Ok(order)
}

fn cyclic_odd_part(n: u64, factors: &BTreeMap<u64, u32>) -> Option<(u64, u32)> {
    let odd = if n.is_multiple_of(2) {
        if factors.get(&2) != Some(&1) || factors.len() != 2 {
            return None;
        }
        factors.iter().find(|(prime, _)| **prime != 2)?
    } else {
        if factors.len() != 1 {
            return None;
        }
        factors.first_key_value()?
    };
    Some((*odd.0, *odd.1))
}

fn is_generator(
    a: u64,
    n: u64,
    phi: u64,
    factors: &BTreeMap<u64, u32>,
    work: &mut Work,
) -> Result<bool, NTheoryError> {
    if work.gcd(a, n)? != 1 {
        return Ok(false);
    }
    for prime in factors.keys() {
        if work.power(a, phi / prime, n)? == 1 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Decide whether a unit generates the complete multiplicative group modulo n.
///
/// Composite moduli are supported. Noncyclic unit groups return false, while
/// nonunits and moduli <= 1 are domain errors rather than negative decisions.
pub fn is_primitive_root(a: u64, n: u64) -> Result<bool, NTheoryError> {
    let mut work = Work::new(MAX_MODULAR_GROUP_WORK);
    let a = admit_unit(a, n, &mut work)?;
    if n <= 4 {
        return Ok(a == n - 1);
    }
    let modulus_factors = factorint(n)?;
    if cyclic_odd_part(n, &modulus_factors).is_none() {
        return Ok(false);
    }
    let phi = totient_from_factors(n, &modulus_factors);
    let factors = factorint(phi)?;
    if !is_generator(a, n, phi, &factors, &mut work)? {
        return Ok(false);
    }
    verify_multiplicative_order(a, n, phi, &factors)?;
    Ok(true)
}

/// Return a primitive root, or None precisely when the unit group is noncyclic.
///
/// Cyclic groups occur at 2, 4, odd prime powers, and twice odd prime powers.
/// With `smallest=true` candidates are checked in increasing order. Otherwise
/// a prime-field generator is lifted and adjusted to an odd residue if needed.
/// Exhaustion is an error, never None or an unproved claim of minimality.
pub fn primitive_root(n: u64, smallest: bool) -> Result<Option<u64>, NTheoryError> {
    primitive_root_with_work_limit(n, smallest, MAX_MODULAR_GROUP_WORK)
}

/// Primitive-root search with a bounded number of modular operations.
pub fn primitive_root_with_work_limit(
    n: u64,
    smallest: bool,
    work_limit: u64,
) -> Result<Option<u64>, NTheoryError> {
    if n <= 1 {
        return Err(NTheoryError::InvalidGroupModulus);
    }
    let mut work = Work::new(work_limit);
    work.charge()?;
    if n <= 4 {
        return Ok(Some(n - 1));
    }
    let modulus_factors = factorint(n)?;
    let Some((prime, exponent)) = cyclic_odd_part(n, &modulus_factors) else {
        return Ok(None);
    };
    let phi = totient_from_factors(n, &modulus_factors);
    let factors = factorint(phi)?;
    let root = if smallest {
        let mut candidate = 2;
        while !is_generator(candidate, n, phi, &factors, &mut work)? {
            work.charge()?;
            if candidate == n - 1 {
                return Err(NTheoryError::MultiplicativeOrderVerificationFailed(
                    "cyclic group search found no generator",
                ));
            }
            candidate += 1;
        }
        candidate
    } else {
        // The factors of phi(p^k) other than p are exactly those of p-1.
        let mut prime_factors = factors.clone();
        prime_factors.remove(&prime);
        let mut candidate = 2;
        while !is_generator(candidate, prime, prime - 1, &prime_factors, &mut work)? {
            work.charge()?;
            if candidate == prime - 1 {
                return Err(NTheoryError::MultiplicativeOrderVerificationFailed(
                    "prime-field search found no generator",
                ));
            }
            candidate += 1;
        }
        if exponent > 1 {
            let prime_square = prime
                .checked_mul(prime)
                .ok_or(NTheoryError::ArithmeticOverflow(
                    "primitive-root lifting modulus",
                ))?;
            if work.power(candidate, prime - 1, prime_square)? == 1 {
                candidate += prime;
            }
        }
        if n.is_multiple_of(2) && candidate.is_multiple_of(2) {
            candidate += n / 2;
        }
        candidate
    };
    verify_multiplicative_order(root, n, phi, &factors)?;
    Ok(Some(root))
}
