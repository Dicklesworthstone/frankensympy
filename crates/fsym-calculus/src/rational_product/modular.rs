//! Bounded proposal search for integer-root-free higher-degree factors.
//!
//! Only a proposed prime and the exact rational factor cross into the proof
//! kernel. The independent checker reconstructs the modular image and checks
//! all residues itself; no search label or modular sample is trusted.

use super::{MAX_WORK_BITS, RationalProductError, Result, reserve};
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::arith::metered_div_rem;
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::rational_product::RationalProductModularFactor;

const PRIMES: [u32; 25] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
];

fn residue(value: &BigInt, prime: u32, meter: &mut impl BudgetMeter) -> Result<u64> {
    if value.bits() > MAX_WORK_BITS {
        return Err(RationalProductError::ResourceLimit(
            "modular product coefficient height",
        ));
    }
    let (_, remainder) = metered_div_rem(value, &BigInt::from(prime), meter)?.ok_or(
        RationalProductError::ResourceLimit("invalid product domain modulus"),
    )?;
    let remainder = remainder
        .to_i64()
        .ok_or(RationalProductError::ResourceLimit(
            "modular product residue width",
        ))?;
    Ok(remainder.rem_euclid(i64::from(prime)) as u64)
}

fn power(
    mut base: u64,
    mut exponent: u32,
    prime: u64,
    meter: &mut impl BudgetMeter,
) -> Result<u64> {
    let mut result = 1;
    while exponent != 0 {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 2)?;
        if exponent & 1 != 0 {
            result = result * base % prime;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base * base % prime;
        }
    }
    Ok(result)
}

pub(super) fn nonzero_factor(
    coefficients: &[BigRational],
    multiplicity: u32,
    meter: &mut impl BudgetMeter,
) -> Result<Option<RationalProductModularFactor>> {
    'prime: for prime in PRIMES {
        reserve(meter, coefficients.len())?;
        let mut residues = Vec::with_capacity(coefficients.len());
        for coefficient in coefficients {
            let denominator = residue(coefficient.denom(), prime, meter)?;
            if denominator == 0 {
                continue 'prime;
            }
            let inverse = power(denominator, prime - 2, u64::from(prime), meter)?;
            let numerator = residue(coefficient.numer(), prime, meter)?;
            residues.push(numerator * inverse % u64::from(prime));
        }
        for point in 0..u64::from(prime) {
            // Ascending scalar powers are separate from the verifier's Horner
            // evaluation. This is only candidate search, never acceptance.
            let mut value = 0;
            let mut power = 1;
            for coefficient in &residues {
                meter.checkpoint()?;
                meter.charge(Dimension::ComputeSteps, 2)?;
                value = (value + coefficient * power) % u64::from(prime);
                power = power * point % u64::from(prime);
            }
            if value == 0 {
                continue 'prime;
            }
        }
        reserve(meter, coefficients.len())?;
        return Ok(Some(RationalProductModularFactor {
            coefficients: coefficients.to_vec(),
            multiplicity,
            prime,
        }));
    }
    Ok(None)
}
