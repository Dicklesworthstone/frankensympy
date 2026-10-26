//! Complete square roots modulo composite `u64` moduli.
//!
//! Generation uses Tonelli–Shanks, prime-power lifting and CRT. Acceptance uses
//! prime/product checking, a separate valuation-based root-count theorem, and
//! direct congruence checks. Counting and finding one root never enumerate the
//! full answer. These are bounded synchronous reference checks, not portable
//! proof-kernel certificates or runtime cancellation guarantees.

use crate::{NTheoryError, factorint, verify_prime_factorization};
use fsym_core::{BigInt, arith::mod_inverse};
use std::collections::BTreeMap;

/// Maximum number of roots admitted by the materializing all-roots API.
pub const MAX_MODULAR_ROOTS: usize = 65_536;
const MAX_ROOT_WORK: u64 = 8_000_000;

struct Work {
    remaining: u64,
}

impl Work {
    fn new() -> Self {
        Self {
            remaining: MAX_ROOT_WORK,
        }
    }

    fn charge(&mut self, amount: u64) -> Result<(), NTheoryError> {
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or(NTheoryError::ModularRootWorkLimitExceeded)?;
        Ok(())
    }

    fn product(&mut self, left: u64, right: u64, modulus: u64) -> Result<u64, NTheoryError> {
        self.charge(1)?;
        Ok((u128::from(left) * u128::from(right) % u128::from(modulus)) as u64)
    }

    fn power(
        &mut self,
        mut base: u64,
        mut exponent: u64,
        modulus: u64,
    ) -> Result<u64, NTheoryError> {
        let mut result = 1;
        base %= modulus;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = self.product(result, base, modulus)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = self.product(base, base, modulus)?;
            }
        }
        Ok(result)
    }
}

fn invalid(message: &'static str) -> NTheoryError {
    NTheoryError::ModularRootVerificationFailed(message)
}

fn prime_power(prime: u64, exponent: u32) -> Result<u64, NTheoryError> {
    prime
        .checked_pow(exponent)
        .ok_or_else(|| invalid("prime-power overflow"))
}

fn tonelli_shanks(a: u64, prime: u64, work: &mut Work) -> Result<Option<u64>, NTheoryError> {
    if a == 0 {
        return Ok(Some(0));
    }
    if work.power(a, (prime - 1) / 2, prime)? != 1 {
        return Ok(None);
    }
    if prime % 4 == 3 {
        return Ok(Some(work.power(a, prime / 4 + 1, prime)?));
    }
    let mut twos = (prime - 1).trailing_zeros();
    let odd_part = (prime - 1) >> twos;
    let mut nonresidue = 2;
    while work.power(nonresidue, (prime - 1) / 2, prime)? != prime - 1 {
        work.charge(1)?;
        nonresidue += 1;
        if nonresidue == prime {
            return Err(invalid("prime has no quadratic nonresidue"));
        }
    }
    let mut c = work.power(nonresidue, odd_part, prime)?;
    let mut root = work.power(a, odd_part.div_ceil(2), prime)?;
    let mut residual = work.power(a, odd_part, prime)?;
    while residual != 1 {
        let mut squared = residual;
        let mut index = 0;
        while squared != 1 && index < twos {
            squared = work.product(squared, squared, prime)?;
            index += 1;
        }
        if index == twos || index == 0 {
            return Err(invalid("Tonelli-Shanks order invariant failed"));
        }
        let b = work.power(c, 1u64 << (twos - index - 1), prime)?;
        root = work.product(root, b, prime)?;
        c = work.product(b, b, prime)?;
        residual = work.product(residual, c, prime)?;
        twos = index;
    }
    Ok(Some(root))
}

fn odd_unit_roots(
    a: u64,
    prime: u64,
    exponent: u32,
    work: &mut Work,
) -> Result<Vec<u64>, NTheoryError> {
    let Some(mut root) = tonelli_shanks(a % prime, prime, work)? else {
        return Ok(Vec::new());
    };
    let mut modulus = prime;
    let derivative = (2 * u128::from(root) % u128::from(prime)) as u64;
    let inverse = work.power(derivative, prime - 2, prime)?;
    for _ in 1..exponent {
        let next = modulus
            .checked_mul(prime)
            .ok_or_else(|| invalid("lifting modulus overflow"))?;
        let square = work.product(root, root, next)?;
        let difference =
            (u128::from(a % next) + u128::from(next) - u128::from(square)) % u128::from(next);
        if difference % u128::from(modulus) != 0 {
            return Err(invalid(
                "Hensel residual is not divisible by the prior modulus",
            ));
        }
        let correction = work.product((difference / u128::from(modulus)) as u64, inverse, prime)?;
        root = (u128::from(root) + u128::from(correction) * u128::from(modulus)) as u64;
        modulus = next;
    }
    let opposite = modulus - root;
    Ok(vec![root.min(opposite), root.max(opposite)])
}

fn two_adic_unit_roots(a: u64, exponent: u32, work: &mut Work) -> Result<Vec<u64>, NTheoryError> {
    let mut roots = vec![1];
    let mut modulus = 2u64;
    for _ in 1..exponent {
        let next = modulus
            .checked_mul(2)
            .ok_or_else(|| invalid("two-adic modulus overflow"))?;
        let mut lifted = Vec::with_capacity(8);
        for root in roots {
            for candidate in [root, root + modulus] {
                if work.product(candidate, candidate, next)? == a % next {
                    lifted.push(candidate);
                }
            }
        }
        roots = lifted;
        modulus = next;
        if roots.is_empty() {
            break;
        }
    }
    roots.sort_unstable();
    Ok(roots)
}

fn local_roots(
    a: u64,
    prime: u64,
    exponent: u32,
    all_roots: bool,
    work: &mut Work,
) -> Result<Vec<u64>, NTheoryError> {
    let modulus = prime_power(prime, exponent)?;
    let residue = a % modulus;
    if residue == 0 {
        if !all_roots {
            return Ok(vec![0]);
        }
        let count = prime_power(prime, exponent / 2)?;
        if count > MAX_MODULAR_ROOTS as u64 {
            return Err(NTheoryError::ModularRootCountExceeded(count));
        }
        let stride = prime_power(prime, exponent.div_ceil(2))?;
        work.charge(count)?;
        return Ok((0..count).map(|multiple| multiple * stride).collect());
    }
    let mut unit = residue;
    let mut valuation = 0u32;
    while unit.is_multiple_of(prime) {
        work.charge(1)?;
        valuation += 1;
        unit /= prime;
    }
    if !valuation.is_multiple_of(2) {
        return Ok(Vec::new());
    }
    let unit_exponent = exponent - valuation;
    let roots = if prime == 2 {
        two_adic_unit_roots(unit, unit_exponent, work)?
    } else {
        odd_unit_roots(unit, prime, unit_exponent, work)?
    };
    if roots.is_empty() {
        return Ok(roots);
    }
    let scale = prime_power(prime, valuation / 2)?;
    if !all_roots {
        return Ok(vec![roots[0] * scale]);
    }
    let count = scale
        .checked_mul(roots.len() as u64)
        .ok_or_else(|| invalid("root count overflow"))?;
    if count > MAX_MODULAR_ROOTS as u64 {
        return Err(NTheoryError::ModularRootCountExceeded(count));
    }
    let unit_modulus = prime_power(prime, unit_exponent)?;
    work.charge(count)?;
    let mut expanded = Vec::with_capacity(count as usize);
    for root in roots {
        for shift in 0..scale {
            expanded.push((root + shift * unit_modulus) * scale);
        }
    }
    expanded.sort_unstable();
    Ok(expanded)
}

fn crt_inverse(
    left_modulus: u64,
    right_modulus: u64,
    work: &mut Work,
) -> Result<u64, NTheoryError> {
    // The exact arithmetic owner receives only two positive 64-bit inputs.
    work.charge(4_096)?;
    mod_inverse(&BigInt::from(left_modulus), &BigInt::from(right_modulus))
        .and_then(|inverse| inverse.to_u64())
        .ok_or_else(|| invalid("CRT moduli are not coprime"))
}

fn crt_join(
    left: u64,
    left_modulus: u64,
    right: u64,
    right_modulus: u64,
    inverse: u64,
    work: &mut Work,
) -> Result<u64, NTheoryError> {
    work.charge(1)?;
    let difference = (u128::from(right) + u128::from(right_modulus)
        - u128::from(left % right_modulus))
        % u128::from(right_modulus);
    let correction = work.product(difference as u64, inverse, right_modulus)?;
    let result = u128::from(left) + u128::from(left_modulus) * u128::from(correction);
    u64::try_from(result).map_err(|_| invalid("CRT result overflow"))
}

// This left-to-right scalar power belongs to the cardinality checker. It does
// not invoke the generator's modular powering, lifting, root search or CRT.
fn reference_power(base: u64, exponent: u64, modulus: u64) -> u64 {
    let modulus = u128::from(modulus);
    let base = u128::from(base) % modulus;
    let mut value = 1u128;
    for bit in (0..64).rev() {
        value = value * value % modulus;
        if exponent & (1u64 << bit) != 0 {
            value = value * base % modulus;
        }
    }
    value as u64
}

fn verified_root_count(
    a: u64,
    modulus: u64,
    factors: &BTreeMap<u64, u32>,
) -> Result<u64, NTheoryError> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    verify_prime_factorization(modulus, factors)?;
    let mut total = 1u64;
    for (&prime, &exponent) in factors {
        let power = prime
            .checked_pow(exponent)
            .ok_or_else(|| invalid("reference prime-power overflow"))?;
        let mut residue = a % power;
        let mut valuation = 0;
        while valuation < exponent && residue.is_multiple_of(prime) {
            residue /= prime;
            valuation += 1;
        }
        let count = if valuation == exponent {
            prime.checked_pow(exponent / 2)
        } else if !valuation.is_multiple_of(2) {
            return Ok(0);
        } else {
            let unit_exponent = exponent - valuation;
            let unit_count = if prime != 2 {
                if reference_power(residue, (prime - 1) / 2, prime) != 1 {
                    return Ok(0);
                }
                2
            } else if unit_exponent == 1 {
                1
            } else if unit_exponent == 2 && residue % 4 == 1 {
                2
            } else if unit_exponent >= 3 && residue % 8 == 1 {
                4
            } else {
                return Ok(0);
            };
            prime
                .checked_pow(valuation / 2)
                .and_then(|scale| scale.checked_mul(unit_count))
        }
        .ok_or_else(|| invalid("reference root count overflow"))?;
        total = total
            .checked_mul(count)
            .ok_or_else(|| invalid("combined root count overflow"))?;
    }
    Ok(total)
}

/// Check the entire sorted root set against an untrusted prime factorization.
///
/// Each entry must lie in `[0, modulus)` and satisfy the congruence. Strict
/// ordering excludes duplicates. The separate prime-power root-count theorem
/// proves completeness; merely checking the supplied roots is insufficient.
pub fn verify_modular_square_roots(
    a: u64,
    modulus: u64,
    factors: &BTreeMap<u64, u32>,
    roots: &[u64],
) -> Result<(), NTheoryError> {
    if roots.len() > MAX_MODULAR_ROOTS {
        return Err(NTheoryError::ModularRootCountExceeded(roots.len() as u64));
    }
    let expected = verified_root_count(a, modulus, factors)?;
    if expected != roots.len() as u64 {
        return Err(invalid("root list is not complete"));
    }
    let mut previous = None;
    for &root in roots {
        if root >= modulus || previous.is_some_and(|value| value >= root) {
            return Err(invalid("root list is out of range or not strictly sorted"));
        }
        if u128::from(root) * u128::from(root) % u128::from(modulus) != u128::from(a % modulus) {
            return Err(invalid("candidate fails the square congruence"));
        }
        previous = Some(root);
    }
    Ok(())
}

/// Return every square root of `a` modulo a positive `u64` modulus, in order.
///
/// The root count is checked before any root-list allocation. Answers exceeding
/// [`MAX_MODULAR_ROOTS`] refuse explicitly; no truncated answer is returned.
pub fn modular_square_roots(a: u64, modulus: u64) -> Result<Vec<u64>, NTheoryError> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    let factors = factorint(modulus)?;
    let count = verified_root_count(a, modulus, &factors)?;
    if count > MAX_MODULAR_ROOTS as u64 {
        return Err(NTheoryError::ModularRootCountExceeded(count));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut work = Work::new();
    let mut roots = vec![0];
    let mut combined_modulus = 1u64;
    for (&prime, &exponent) in &factors {
        let local_modulus = prime_power(prime, exponent)?;
        let local = local_roots(a, prime, exponent, true, &mut work)?;
        let next_modulus = combined_modulus
            .checked_mul(local_modulus)
            .ok_or_else(|| invalid("CRT modulus overflow"))?;
        let count = roots
            .len()
            .checked_mul(local.len())
            .ok_or_else(|| invalid("CRT root count overflow"))?;
        if count > MAX_MODULAR_ROOTS {
            return Err(NTheoryError::ModularRootCountExceeded(count as u64));
        }
        let inverse = crt_inverse(combined_modulus, local_modulus, &mut work)?;
        let mut joined = Vec::with_capacity(count);
        for &left in &roots {
            for &right in &local {
                joined.push(crt_join(
                    left,
                    combined_modulus,
                    right,
                    local_modulus,
                    inverse,
                    &mut work,
                )?);
            }
        }
        roots = joined;
        combined_modulus = next_modulus;
    }
    roots.sort_unstable();
    verify_modular_square_roots(a, modulus, &factors, &roots)?;
    Ok(roots)
}

/// Return one root in `[0, modulus/2]`, without allocating the full root set.
///
/// A returned root is checked exactly. `None` requires a zero root count under
/// the independently checked prime-power classification. The selected root is
/// deterministic; it need not be the smallest member of the complete root set.
pub fn modular_square_root(a: u64, modulus: u64) -> Result<Option<u64>, NTheoryError> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    if a.is_multiple_of(modulus) {
        return Ok(Some(0));
    }
    let factors = factorint(modulus)?;
    if verified_root_count(a, modulus, &factors)? == 0 {
        return Ok(None);
    }
    let mut work = Work::new();
    let mut root = 0;
    let mut combined_modulus = 1u64;
    for (&prime, &exponent) in &factors {
        let local_modulus = prime_power(prime, exponent)?;
        let local = local_roots(a, prime, exponent, false, &mut work)?;
        if local.len() != 1 {
            return Err(invalid("positive root count produced no local root"));
        }
        let inverse = crt_inverse(combined_modulus, local_modulus, &mut work)?;
        root = crt_join(
            root,
            combined_modulus,
            local[0],
            local_modulus,
            inverse,
            &mut work,
        )?;
        combined_modulus = combined_modulus
            .checked_mul(local_modulus)
            .ok_or_else(|| invalid("CRT modulus overflow"))?;
    }
    root = root.min(modulus - root);
    if u128::from(root) * u128::from(root) % u128::from(modulus) != u128::from(a % modulus) {
        return Err(invalid("single root fails the square congruence"));
    }
    Ok(Some(root))
}

/// Decide quadratic residuosity for every positive `u64` modulus.
///
/// Prime powers and nonunits are included. This operation counts roots without
/// constructing them, so the all-roots output limit does not limit the answer.
pub fn is_quadratic_residue(a: u64, modulus: u64) -> Result<bool, NTheoryError> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    if a.is_multiple_of(modulus) {
        return Ok(true);
    }
    let factors = factorint(modulus)?;
    Ok(verified_root_count(a, modulus, &factors)? != 0)
}
