//! Complete bounded roots of x^n = a in composite u64 residue rings.
//!
//! Generation reduces prime-power valuations, solves unit-group exponent
//! congruences and combines the complete local sets by CRT. Odd prime-power
//! groups use bounded baby-step/giant-step search only when the exponent is
//! not invertible. The noncyclic group at two is split into its sign and
//! 5-generated factors; its logarithm is recovered one bit at a time.
//!
//! An independent checker uses a prime-power cardinality theorem and a
//! separate scalar power loop. Sorted distinct valid roots with that exact
//! cardinality establish completeness; a valid subset never does.
//!
//! All materialized answers contain at most 65536 roots. Generation and
//! verification have separate cumulative work/allocation bounds and caller
//! meters. The existing bounded synchronous factorization and square-root
//! helpers are reserved before invocation and checked for cancellation on
//! both sides; cancellation inside those helpers is not claimed. No portable
//! proof-kernel receipt or asynchronous cancellation guarantee is introduced.

use crate::{
    MAX_FACTORIZATION_WORK, MAX_MODULAR_ROOTS, NTheoryError, factorint,
    modular_square_roots, verify_prime_factorization,
};
use fsym_budget::{BudgetMeter, Dimension, Unbounded};
use std::collections::BTreeMap;

const MAX_WORK: u64 = 256_000_000;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const FACTOR_CHECK_WORK: u64 = 63 * 4096;
const MAX_LOG_TABLE: u64 = 65_536;

type Result<T> = std::result::Result<T, NTheoryError>;

fn invalid(reason: &'static str) -> NTheoryError {
    NTheoryError::ModularNthRootVerificationFailed(reason)
}

struct Work<'a, M> {
    meter: &'a mut M,
    remaining: u64,
    bytes: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            remaining: MAX_WORK,
            bytes: MAX_BYTES,
        }
    }

    fn checkpoint(&mut self) -> Result<()> {
        self.meter.checkpoint().map_err(NTheoryError::Meter)
    }

    fn tick(&mut self, amount: u64) -> Result<()> {
        self.checkpoint()?;
        let amount = amount.max(1);
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or(NTheoryError::ModularNthRootWorkLimitExceeded)?;
        self.meter
            .charge(Dimension::ComputeSteps, amount)
            .map_err(NTheoryError::Meter)
    }

    fn storage(&mut self, bytes: u64) -> Result<()> {
        self.checkpoint()?;
        if bytes == 0 {
            return Ok(());
        }
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .ok_or(NTheoryError::ModularNthRootAllocationLimitExceeded)?;
        self.meter
            .charge_batch(&[
                (Dimension::MemoryBytes, bytes),
                (Dimension::AllocationCount, 1),
            ])
            .map_err(NTheoryError::Meter)
    }

    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(NTheoryError::ModularNthRootAllocationLimitExceeded)?;
        self.storage(bytes)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| NTheoryError::ModularNthRootAllocationLimitExceeded)?;
        Ok(values)
    }

    fn multiply(&mut self, left: u64, right: u64, modulus: u64) -> Result<u64> {
        self.tick(1)?;
        Ok((u128::from(left) * u128::from(right) % u128::from(modulus)) as u64)
    }

    fn power(&mut self, mut base: u64, mut exponent: u64, modulus: u64) -> Result<u64> {
        self.tick(1)?;
        base %= modulus;
        let mut result = 1 % modulus;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = self.multiply(result, base, modulus)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = self.multiply(base, base, modulus)?;
            }
        }
        Ok(result)
    }

    fn gcd(&mut self, mut left: u64, mut right: u64) -> Result<u64> {
        while right != 0 {
            self.tick(1)?;
            (left, right) = (right, left % right);
        }
        Ok(left)
    }

    fn inverse(&mut self, value: u64, modulus: u64) -> Result<u64> {
        if modulus == 1 {
            return Ok(0);
        }
        let (mut old_r, mut r) = (modulus, value % modulus);
        let (mut old_t, mut t) = (0, 1);
        while r != 0 {
            self.tick(1)?;
            let quotient = old_r / r;
            (old_r, r) = (r, old_r % r);
            let product = self.multiply(quotient, t, modulus)?;
            let next =
                ((u128::from(old_t) + u128::from(modulus) - u128::from(product))
                    % u128::from(modulus)) as u64;
            (old_t, t) = (t, next);
        }
        if old_r != 1 {
            return Err(invalid("noninvertible scalar in unit-group or CRT solve"));
        }
        Ok(old_t)
    }

    fn factor(&mut self, value: u64) -> Result<BTreeMap<u64, u32>> {
        // This helper has its own finite synchronous cap. Reserve its entire
        // cap, including the independent factor checker, without resetting
        // this operation's or the caller's accounting.
        self.tick(MAX_FACTORIZATION_WORK + FACTOR_CHECK_WORK)?;
        self.storage(64 * 128)?;
        let factors = factorint(value)?;
        self.checkpoint()?;
        Ok(factors)
    }

    fn sort(&mut self, roots: &mut [u64]) -> Result<()> {
        let comparisons = (roots.len() as u64)
            .saturating_mul(u64::from(usize::BITS - roots.len().max(1).leading_zeros()));
        self.tick(comparisons)?;
        roots.sort_unstable();
        self.checkpoint()
    }
}

fn prime_power(prime: u64, exponent: u32) -> Result<u64> {
    prime
        .checked_pow(exponent)
        .ok_or_else(|| invalid("prime-power overflow"))
}

// Separate left-to-right power loop used only for reference acceptance. It
// never calls the generating power, logarithm, root lifting or CRT routines.
fn reference_power(
    base: u64,
    exponent: u64,
    modulus: u64,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<u64> {
    let modulus = u128::from(modulus);
    let base = u128::from(base) % modulus;
    let mut value = 1 % modulus;
    for bit in (0..64).rev() {
        work.tick(2)?;
        value = value * value % modulus;
        if (exponent >> bit) & 1 != 0 {
            value = value * base % modulus;
        }
    }
    Ok(value as u64)
}

fn reference_gcd(
    mut left: u64,
    mut right: u64,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<u64> {
    while right != 0 {
        work.tick(1)?;
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    Ok(left)
}

fn root_count(
    a: u64,
    degree: u64,
    modulus: u64,
    factors: &BTreeMap<u64, u32>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<u64> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    if degree == 0 {
        return Err(NTheoryError::InvalidRootDegree);
    }
    if factors.len() > 63 {
        return Err(invalid("too many prime-power components"));
    }
    work.tick(FACTOR_CHECK_WORK)?;
    verify_prime_factorization(modulus, factors)?;
    work.checkpoint()?;
    let mut total = 1u64;
    for (&prime, &exponent) in factors {
        work.tick(1)?;
        let component = prime_power(prime, exponent)?;
        let mut unit = a % component;
        let mut valuation = 0u32;
        while valuation < exponent && unit.is_multiple_of(prime) {
            work.tick(1)?;
            unit /= prime;
            valuation += 1;
        }
        let local = if valuation == exponent {
            let minimum = u64::from(exponent).div_ceil(degree) as u32;
            prime_power(prime, exponent - minimum)?
        } else {
            if !u64::from(valuation).is_multiple_of(degree) {
                return Ok(0);
            }
            let root_valuation = (u64::from(valuation) / degree) as u32;
            let unit_exponent = exponent - valuation;
            let unit_modulus = prime_power(prime, unit_exponent)?;
            let units = if prime != 2 {
                let order = (unit_modulus / prime) * (prime - 1);
                let common = reference_gcd(degree, order, work)?;
                if reference_power(unit, order / common, unit_modulus, work)? != 1 {
                    return Ok(0);
                }
                common
            } else if unit_exponent == 1 || !degree.is_multiple_of(2) {
                1
            } else {
                // C2 x C(2^(k-2)): an even n-th power is exactly 1 modulo
                // 2^min(v2(n)+2,k). Its kernel has 2*gcd(n,2^(k-2))
                // elements for k>=3; the k=2 group has two elements.
                let required = (degree.trailing_zeros() + 2).min(unit_exponent);
                let congruence_modulus = 1u64 << required;
                if unit % congruence_modulus != 1 {
                    return Ok(0);
                }
                if unit_exponent == 2 {
                    2
                } else {
                    let cyclic = 1u64 << (unit_exponent - 2);
                    2 * reference_gcd(degree, cyclic, work)?
                }
            };
            units
                .checked_mul(prime_power(prime, valuation - root_valuation)?)
                .ok_or_else(|| invalid("local root-count overflow"))?
        };
        // A late zero component must still produce an empty set even if an
        // earlier component by itself would exceed the materialization cap.
        total = total
            .checked_mul(local)
            .ok_or_else(|| invalid("combined root-count overflow"))?;
    }
    Ok(total)
}

fn verify_roots(
    a: u64,
    degree: u64,
    modulus: u64,
    factors: &BTreeMap<u64, u32>,
    roots: &[u64],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    if roots.len() > MAX_MODULAR_ROOTS {
        return Err(NTheoryError::ModularNthRootCountExceeded(roots.len() as u64));
    }
    let expected = root_count(a, degree, modulus, factors, work)?;
    if roots.len() as u64 != expected {
        return Err(invalid("root list does not have the complete cardinality"));
    }
    let mut previous = None;
    for &root in roots {
        work.tick(1)?;
        if root >= modulus || previous.is_some_and(|value| value >= root) {
            return Err(invalid("roots are out of range or not strictly sorted"));
        }
        if reference_power(root, degree, modulus, work)? != a % modulus {
            return Err(invalid("candidate fails the original power congruence"));
        }
        previous = Some(root);
    }
    work.checkpoint()
}

/// Independently check a complete sorted root set and its untrusted factor map.
pub fn verify_modular_nth_roots(
    a: u64,
    degree: u64,
    modulus: u64,
    factors: &BTreeMap<u64, u32>,
    roots: &[u64],
) -> Result<()> {
    verify_modular_nth_roots_with_meter(a, degree, modulus, factors, roots, &mut Unbounded)
}

/// Metered reference checker; no root search or factorization is performed.
pub fn verify_modular_nth_roots_with_meter(
    a: u64,
    degree: u64,
    modulus: u64,
    factors: &BTreeMap<u64, u32>,
    roots: &[u64],
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    verify_roots(a, degree, modulus, factors, roots, &mut Work::new(meter))
}

fn primitive_generator(
    modulus: u64,
    order: u64,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<u64> {
    let factors = work.factor(order)?;
    for candidate in 2..modulus {
        work.tick(1)?;
        if work.gcd(candidate, modulus)? != 1 {
            continue;
        }
        let mut generator = true;
        for &prime in factors.keys() {
            if work.power(candidate, order / prime, modulus)? == 1 {
                generator = false;
                break;
            }
        }
        if generator {
            return Ok(candidate);
        }
    }
    Err(invalid("odd prime-power group has no primitive generator"))
}

fn discrete_logarithm(
    target: u64,
    base: u64,
    modulus: u64,
    order: u64,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<u64> {
    if target == 1 {
        return Ok(0);
    }
    // Exact ceil(sqrt(order)), including order near u64::MAX.
    let (mut lower, mut upper) = (1u64, 1u64 << 32);
    while lower < upper {
        work.tick(1)?;
        let middle = lower + (upper - lower) / 2;
        if u128::from(middle) * u128::from(middle) >= u128::from(order) {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    let width = lower;
    if width > MAX_LOG_TABLE {
        return Err(NTheoryError::DiscreteLogTableLimitExceeded(width));
    }
    let mut babies = work.vector(width as usize)?;
    let mut value = 1u64;
    for index in 0..width {
        babies.push((value, index));
        value = work.multiply(value, base, modulus)?;
    }
    work.tick(width * u64::from(u64::BITS - width.leading_zeros()))?;
    babies.sort_unstable();
    work.checkpoint()?;
    let inverse = work.inverse(value, modulus)?;
    let mut giant = target;
    for block in 0..order.div_ceil(width) {
        work.tick(u64::from(u64::BITS - width.leading_zeros()))?;
        if let Ok(index) = babies.binary_search_by_key(&giant, |entry| entry.0) {
            let exponent = u128::from(block) * u128::from(width) + u128::from(babies[index].1);
            if exponent < u128::from(order) {
                return Ok(exponent as u64);
            }
        }
        giant = work.multiply(giant, inverse, modulus)?;
    }
    Err(invalid("unit-group logarithm search did not cover an admitted unit"))
}

fn two_adic_logarithm(
    target: u64,
    exponent: u32,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<u64> {
    let modulus = 1u64 << exponent;
    let mut remaining = target;
    let mut factor = 5 % modulus;
    let mut logarithm = 0;
    for bit in 0..exponent - 2 {
        let probe = work.power(remaining, 1u64 << (exponent - 3 - bit), modulus)?;
        if probe != 1 {
            if probe != 1 + modulus / 2 {
                return Err(invalid("two-adic logarithm left the principal unit group"));
            }
            logarithm |= 1u64 << bit;
            let inverse = work.inverse(factor, modulus)?;
            remaining = work.multiply(remaining, inverse, modulus)?;
        }
        factor = work.multiply(factor, factor, modulus)?;
    }
    if remaining != 1 {
        return Err(invalid("two-adic logarithm has a nontrivial remainder"));
    }
    Ok(logarithm)
}

fn cyclic_roots(
    unit: u64,
    degree: u64,
    modulus: u64,
    order: u64,
    base: u64,
    logarithm: u64,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<u64>> {
    let common = work.gcd(degree, order)?;
    if !logarithm.is_multiple_of(common) {
        return Err(invalid("admitted power residue has an incompatible logarithm"));
    }
    let reduced_order = order / common;
    let inverse = work.inverse(degree / common, reduced_order)?;
    let first_exponent = work.multiply(logarithm / common, inverse, reduced_order)?;
    let mut value = work.power(base, first_exponent, modulus)?;
    let step = work.power(base, reduced_order, modulus)?;
    if common > MAX_MODULAR_ROOTS as u64 {
        return Err(NTheoryError::ModularNthRootCountExceeded(common));
    }
    let mut roots = work.vector(common as usize)?;
    for _ in 0..common {
        roots.push(value);
        value = work.multiply(value, step, modulus)?;
    }
    if work.power(roots[0], degree, modulus)? != unit {
        return Err(invalid("unit root failed generation's congruence invariant"));
    }
    Ok(roots)
}

fn unit_roots(
    unit: u64,
    degree: u64,
    prime: u64,
    exponent: u32,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<u64>> {
    let modulus = prime_power(prime, exponent)?;
    let order = if prime == 2 {
        match exponent {
            1 => 1,
            2 => 2,
            _ => 1u64 << (exponent - 2),
        }
    } else {
        (modulus / prime) * (prime - 1)
    };
    if work.gcd(degree, order)? == 1 {
        let inverse = work.inverse(degree, order)?;
        let mut roots = work.vector(1)?;
        roots.push(work.power(unit, inverse, modulus)?);
        return Ok(roots);
    }
    if prime == 2 && exponent == 2 {
        let mut roots = work.vector(2)?;
        roots.extend([1, 3]);
        return Ok(roots);
    }
    let (base, logarithm) = if prime == 2 {
        (5, two_adic_logarithm(unit, exponent, work)?)
    } else {
        let base = primitive_generator(modulus, order, work)?;
        (base, discrete_logarithm(unit, base, modulus, order, work)?)
    };
    let roots = cyclic_roots(unit, degree, modulus, order, base, logarithm, work)?;
    if prime != 2 {
        return Ok(roots);
    }
    let mut signed = work.vector(2 * roots.len())?;
    for root in roots {
        work.tick(1)?;
        signed.push(root);
        signed.push(modulus - root);
    }
    Ok(signed)
}

fn local_roots(
    a: u64,
    degree: u64,
    prime: u64,
    exponent: u32,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<u64>> {
    let modulus = prime_power(prime, exponent)?;
    let mut unit = a % modulus;
    if unit == 0 {
        let minimum = u64::from(exponent).div_ceil(degree) as u32;
        let stride = prime_power(prime, minimum)?;
        let count = modulus / stride;
        if count > MAX_MODULAR_ROOTS as u64 {
            return Err(NTheoryError::ModularNthRootCountExceeded(count));
        }
        let mut roots = work.vector(count as usize)?;
        for index in 0..count {
            work.tick(1)?;
            roots.push(index * stride);
        }
        return Ok(roots);
    }
    let mut valuation = 0u32;
    while unit.is_multiple_of(prime) {
        work.tick(1)?;
        unit /= prime;
        valuation += 1;
    }
    if !u64::from(valuation).is_multiple_of(degree) {
        return Err(invalid("generation received a nondivisible valuation"));
    }
    let root_valuation = (u64::from(valuation) / degree) as u32;
    let unit_exponent = exponent - valuation;
    let roots = unit_roots(unit, degree, prime, unit_exponent, work)?;
    let scale = prime_power(prime, root_valuation)?;
    let stride = prime_power(prime, unit_exponent)?;
    let lifts = prime_power(prime, valuation - root_valuation)?;
    let count = (roots.len() as u64)
        .checked_mul(lifts)
        .ok_or_else(|| invalid("lifted root-count overflow"))?;
    if count > MAX_MODULAR_ROOTS as u64 {
        return Err(NTheoryError::ModularNthRootCountExceeded(count));
    }
    let mut lifted = work.vector(count as usize)?;
    for root in roots {
        for index in 0..lifts {
            work.tick(1)?;
            let value = (u128::from(root) + u128::from(index) * u128::from(stride))
                * u128::from(scale);
            if value >= u128::from(modulus) {
                return Err(invalid("lifted root escaped its prime-power component"));
            }
            lifted.push(value as u64);
        }
    }
    Ok(lifted)
}

/// Every root in increasing order, with explicit refusal above 65536 roots.
pub fn modular_nth_roots(a: u64, degree: u64, modulus: u64) -> Result<Vec<u64>> {
    metered_modular_nth_roots(a, degree, modulus, &mut Unbounded, &mut Unbounded)
}

/// Complete roots with distinct cumulative generator and verifier meters.
///
/// Existing factorization and the specialized square-root helper have finite
/// synchronous work caps; cancellation is observed before/after those calls.
pub fn metered_modular_nth_roots(
    a: u64,
    degree: u64,
    modulus: u64,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Vec<u64>> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    if degree == 0 {
        return Err(NTheoryError::InvalidRootDegree);
    }
    let mut generation = Work::new(generator);
    let mut verification = Work::new(verifier);
    generation.checkpoint()?;
    verification.checkpoint()?;
    let factors = generation.factor(modulus)?;
    let count = root_count(a, degree, modulus, &factors, &mut verification)?;
    if count > MAX_MODULAR_ROOTS as u64 {
        return Err(NTheoryError::ModularNthRootCountExceeded(count));
    }
    let mut roots = if count == 0 {
        generation.vector(0)?
    } else if degree == 2 {
        // Keep Tonelli-Shanks/Hensel for large prime square roots instead of
        // requiring a potentially enormous discrete-log table.
        generation.tick(4 * MAX_FACTORIZATION_WORK)?;
        generation.storage(64 * MAX_MODULAR_ROOTS as u64 * 8)?;
        let roots = modular_square_roots(a, modulus)?;
        generation.checkpoint()?;
        roots
    } else {
        let mut roots = generation.vector(1)?;
        roots.push(0);
        let mut prefix = 1u64;
        for (&prime, &exponent) in &factors {
            let component = prime_power(prime, exponent)?;
            let local = local_roots(a, degree, prime, exponent, &mut generation)?;
            let inverse = generation.inverse(prefix, component)?;
            let length = roots
                .len()
                .checked_mul(local.len())
                .ok_or_else(|| invalid("CRT root-count overflow"))?;
            if length > MAX_MODULAR_ROOTS {
                return Err(NTheoryError::ModularNthRootCountExceeded(length as u64));
            }
            let mut next = generation.vector(length)?;
            for &left in &roots {
                for &right in &local {
                    generation.tick(1)?;
                    let difference = (u128::from(right) + u128::from(component)
                        - u128::from(left % component))
                        % u128::from(component);
                    let correction =
                        generation.multiply(difference as u64, inverse, component)?;
                    let value = u128::from(left) + u128::from(prefix) * u128::from(correction);
                    next.push(u64::try_from(value).map_err(|_| invalid("CRT root overflow"))?);
                }
            }
            prefix = prefix
                .checked_mul(component)
                .ok_or_else(|| invalid("CRT modulus overflow"))?;
            roots = next;
        }
        roots
    };
    generation.sort(&mut roots)?;
    verify_roots(a, degree, modulus, &factors, &roots, &mut verification)?;
    generation.checkpoint()?;
    verification.checkpoint()?;
    Ok(roots)
}

/// Least root, or None after complete nonexistence verification.
///
/// Nonzero requests share the all-roots materialization bound. Zero and linear
/// sources have an immediate proved least root and require no enumeration.
pub fn modular_nth_root(a: u64, degree: u64, modulus: u64) -> Result<Option<u64>> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    if degree == 0 {
        return Err(NTheoryError::InvalidRootDegree);
    }
    if a % modulus == 0 || degree == 1 {
        return Ok(Some(a % modulus));
    }
    Ok(modular_nth_roots(a, degree, modulus)?.first().copied())
}
