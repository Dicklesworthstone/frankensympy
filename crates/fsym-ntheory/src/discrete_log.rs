//! Bounded least nonnegative discrete logarithms, including nonunit bases.
//!
//! A gcd reduction checks the transient powers before entering a unit group.
//! A complete baby-step/giant-step search then visits exponent blocks in order
//! and preserves the first index for each baby value. Exact subgroup order and
//! an original-input congruence check guard successful publication. This is a
//! synchronous exact algorithm, not a portable minimality certificate.

use crate::multiplicative::{Work, order_with_work};
use crate::{
    MAX_MODULAR_GROUP_WORK, NTheoryError, factorint, integer_nth_root, verify_multiplicative_order,
};
use std::collections::BTreeMap;

/// Maximum baby-step entries; the limit is admitted before allocating a table.
pub const MAX_DISCRETE_LOG_TABLE: u64 = 65_536;

fn inverse(a: u64, n: u64, work: &mut Work) -> Result<u64, NTheoryError> {
    // Keep Bezout coefficients modulo n, so every product fits u128 even at
    // the u64 boundary. No signed coefficient or intermediate can overflow.
    let (mut old_r, mut r) = (n, a % n);
    let (mut old_t, mut t) = (0, 1);
    while r != 0 {
        work.charge()?;
        let quotient = old_r / r;
        (old_r, r) = (r, old_r % r);
        let product = work.multiply(quotient, t, n)?;
        let next_t =
            ((u128::from(old_t) + u128::from(n) - u128::from(product)) % u128::from(n)) as u64;
        (old_t, t) = (t, next_t);
    }
    if old_r != 1 {
        return Err(NTheoryError::DiscreteLogVerificationFailed(
            "unit reduction produced a noninvertible factor",
        ));
    }
    Ok(old_t)
}

fn accept(n: u64, a: u64, b: u64, exponent: u64) -> Result<u64, NTheoryError> {
    // Separate left-to-right scalar replay; do not reuse the generator's
    // modular power or the reduced congruence when checking original inputs.
    let modulus = u128::from(n);
    let base = u128::from(b % n);
    let mut value = 1 % modulus;
    for bit in (0..64).rev() {
        value = value * value % modulus;
        if (exponent >> bit) & 1 != 0 {
            value = value * base % modulus;
        }
    }
    if value != u128::from(a % n) {
        return Err(NTheoryError::DiscreteLogVerificationFailed(
            "candidate fails the original congruence",
        ));
    }
    Ok(exponent)
}

fn reduce_order_hint(b: u64, n: u64, mut order: u64, work: &mut Work) -> Result<u64, NTheoryError> {
    if order == 0 || work.gcd(b, n)? != 1 || work.power(b, order, n)? != 1 {
        return Err(NTheoryError::InvalidDiscreteLogOrder);
    }
    let mut factors = factorint(order)?;
    for (&prime, exponent) in &mut factors {
        while *exponent != 0 && work.power(b, order / prime, n)? == 1 {
            order /= prime;
            *exponent -= 1;
        }
    }
    factors.retain(|_, exponent| *exponent != 0);
    verify_multiplicative_order(b, n, order, &factors)
        .map_err(|_| NTheoryError::InvalidDiscreteLogOrder)?;
    Ok(order)
}

fn unit_log(n: u64, a: u64, b: u64, order: u64, work: &mut Work) -> Result<u64, NTheoryError> {
    if a == 1 {
        return Ok(0);
    }
    if order == 1 || work.gcd(a, n)? != 1 {
        return Err(NTheoryError::DiscreteLogDoesNotExist);
    }
    // Necessary membership only: in a noncyclic unit group this condition is
    // not sufficient, so a passing value still receives the complete search.
    if work.power(a, order, n)? != 1 {
        return Err(NTheoryError::DiscreteLogDoesNotExist);
    }
    let root = integer_nth_root(order, 2)?;
    let width = root + u64::from(u128::from(root) * u128::from(root) < u128::from(order));
    if width > MAX_DISCRETE_LOG_TABLE {
        return Err(NTheoryError::DiscreteLogTableLimitExceeded(width));
    }
    let mut babies = BTreeMap::new();
    let mut value = 1;
    for j in 0..width {
        work.charge()?;
        babies.entry(value).or_insert(j);
        value = work.multiply(value, b, n)?;
    }
    // `value = b^width`; no additional large exponentiation is needed.
    let giant_factor = inverse(value, n, work)?;
    let mut giant = a;
    for i in 0..order.div_ceil(width) {
        work.charge()?;
        if let Some(&j) = babies.get(&giant) {
            let candidate = i * width + j;
            if candidate < order {
                return Ok(candidate);
            }
        }
        giant = work.multiply(giant, giant_factor, n)?;
    }
    // Every exponent in [0, order) was covered. A partial/exhausted search
    // cannot reach this mathematical nonexistence outcome.
    Err(NTheoryError::DiscreteLogDoesNotExist)
}

/// Return the least nonnegative x with `b^x = a (mod n)` for positive n.
///
/// Nonunit bases are supported by exact gcd reduction. An optional `order`
/// must, when used, be a positive annihilating exponent for the reduced unit
/// base. Its prime factors reduce it to an independently checked exact order.
/// Decisive exponents zero/one and transient solutions do not use an order
/// hint. Nonexistence, work exhaustion, and table admission failures are
/// distinct errors. No probabilistic search is used.
pub fn discrete_log(n: u64, a: u64, b: u64, order: Option<u64>) -> Result<u64, NTheoryError> {
    discrete_log_with_work_limit(n, a, b, order, MAX_MODULAR_GROUP_WORK)
}

/// As `discrete_log`, sharing one modular-operation limit across reductions,
/// order generation, and complete search. The existing factorization bounds
/// apply separately, with at most two factorizations (one for a supplied order).
pub fn discrete_log_with_work_limit(
    n: u64,
    a: u64,
    b: u64,
    order: Option<u64>,
    work_limit: u64,
) -> Result<u64, NTheoryError> {
    if n == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    let mut work = Work::new(work_limit);
    work.charge()?;
    if order == Some(0) {
        return Err(NTheoryError::InvalidDiscreteLogOrder);
    }
    if n == 1 {
        return Ok(0);
    }
    let (a, b) = (a % n, b % n);
    if a == 1 {
        return accept(n, a, b, 0);
    }
    if a == b {
        return accept(n, a, b, 1);
    }
    let (mut modulus, mut target, mut multiplier, mut offset) = (n, a, 1, 0u64);
    loop {
        let common = work.gcd(b, modulus)?;
        if common == 1 {
            break;
        }
        // This is exactly the next transient exponent. All smaller offsets
        // have already been checked before the corresponding gcd division.
        if target == multiplier {
            return accept(n, a, b, offset);
        }
        if !target.is_multiple_of(common) {
            return Err(NTheoryError::DiscreteLogDoesNotExist);
        }
        target /= common;
        modulus /= common;
        offset += 1;
        if modulus == 1 {
            return accept(n, a, b, offset);
        }
        multiplier = work.multiply(multiplier, b / common, modulus)?;
    }
    let multiplier_inverse = inverse(multiplier, modulus, &mut work)?;
    let adjusted = work.multiply(target, multiplier_inverse, modulus)?;
    if adjusted == 1 {
        return accept(n, a, b, offset);
    }
    let group_order = match order {
        Some(value) => reduce_order_hint(b % modulus, modulus, value, &mut work)?,
        None => order_with_work(b % modulus, modulus, &mut work)?,
    };
    let tail = unit_log(modulus, adjusted, b % modulus, group_order, &mut work)?;
    let answer = tail
        .checked_add(offset)
        .ok_or(NTheoryError::ArithmeticOverflow(
            "discrete logarithm exponent",
        ))?;
    accept(n, a, b, answer)
}
