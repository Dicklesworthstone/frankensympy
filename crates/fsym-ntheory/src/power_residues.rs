//! Exact power-residue decisions without enumerating roots or residue classes.
//!
//! For each checked prime-power factor, a nonzero residue has a valuation
//! divisible by the requested power. The remaining unit problem uses the
//! cyclic odd-prime group, or the explicit C2 x C(2^(e-2)) structure at two.
//! The Chinese remainder theorem combines existence, not candidate root sets.

use crate::multiplicative::Work;
use crate::{MAX_MODULAR_GROUP_WORK, NTheoryError, factorint};

/// Decide whether `x^degree = a (mod modulus)` has an integer solution.
///
/// All `u64` degrees and positive `u64` moduli are admitted. Prime powers,
/// nonunits, and mixed composite moduli use exact finite-group criteria.
/// No roots or complete residue intervals are allocated. Exponent zero uses
/// the ordinary power convention x^0=1, including the trivial residue ring.
pub fn is_nth_power_residue(a: u64, degree: u64, modulus: u64) -> Result<bool, NTheoryError> {
    is_nth_power_residue_with_work_limit(a, degree, modulus, MAX_MODULAR_GROUP_WORK)
}

/// Power-residue decision with bounded modular arithmetic and valuation work.
/// One factorization additionally retains its existing independent work bound.
pub fn is_nth_power_residue_with_work_limit(
    a: u64,
    degree: u64,
    modulus: u64,
    work_limit: u64,
) -> Result<bool, NTheoryError> {
    if modulus == 0 {
        return Err(NTheoryError::InvalidModulus);
    }
    let mut work = Work::new(work_limit);
    work.charge()?;
    let a = a % modulus;
    if degree == 0 {
        return Ok(a == 1 % modulus);
    }
    if modulus == 1 || a == 0 || degree == 1 {
        return Ok(true);
    }
    for (prime, exponent) in factorint(modulus)? {
        work.charge()?;
        let prime_power = prime
            .checked_pow(exponent)
            .ok_or(NTheoryError::ArithmeticOverflow(
                "power-residue prime power",
            ))?;
        let mut unit = a % prime_power;
        if unit == 0 {
            continue;
        }
        let mut valuation = 0u32;
        while unit.is_multiple_of(prime) {
            work.charge()?;
            unit /= prime;
            valuation += 1;
        }
        if !u64::from(valuation).is_multiple_of(degree) {
            return Ok(false);
        }
        let unit_exponent = exponent - valuation;
        if prime == 2 {
            if !degree.is_multiple_of(2) {
                // Odd powers are automorphisms of every finite 2-group.
                continue;
            }
            // Even d-th powers of odd units are exactly the residues equal to
            // 1 modulo 2^min(v2(d)+2, unit_exponent), including exponents1/2.
            let required_bits = (degree.trailing_zeros() + 2).min(unit_exponent);
            let mask = (1u64 << required_bits) - 1;
            if unit & mask != 1 {
                return Ok(false);
            }
        } else {
            let unit_modulus =
                prime
                    .checked_pow(unit_exponent)
                    .ok_or(NTheoryError::ArithmeticOverflow(
                        "power-residue unit modulus",
                    ))?;
            let phi = (unit_modulus / prime) * (prime - 1);
            let common = work.gcd(degree, phi)?;
            if work.power(unit, phi / common, unit_modulus)? != 1 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
