//! Bounded two-prime NTT/CRT candidate support for exact integer multiplication.

#![forbid(unsafe_code)]

use crate::MeteredMultiplyError;
use fsym_budget::{BudgetMeter, Dimension};
use std::mem::size_of;

const COEFFICIENT_BITS: u32 = 16;
const COEFFICIENT_MASK: u32 = 0xffff;

// Both primes have primitive root 3. Their common power-of-two transform domain is 2^21.
const PRIME_1: u64 = 998_244_353; // 119 * 2^23 + 1
const PRIME_2: u64 = 1_004_535_809; // 479 * 2^21 + 1
const PRIMITIVE_ROOT: u64 = 3;
const CRT_MODULUS: u128 = PRIME_1 as u128 * PRIME_2 as u128;

/// Maximum exact transform length supported by the fixed two-prime configuration.
pub(crate) const MAX_TRANSFORM_LENGTH: usize = 1 << 21;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransformDomain {
    a_coefficient_len: usize,
    b_coefficient_len: usize,
    coefficient_len: usize,
    transform_len: usize,
    output_u32_capacity: usize,
    coefficient_bound: u128,
}

fn transform_domain(
    a_len: usize,
    b_len: usize,
) -> Result<Option<TransformDomain>, MeteredMultiplyError> {
    if a_len == 0 || b_len == 0 {
        return Ok(None);
    }

    let a_coefficient_len = a_len
        .checked_mul(2)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    let b_coefficient_len = b_len
        .checked_mul(2)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    let coefficient_len = a_coefficient_len
        .checked_add(b_coefficient_len)
        .and_then(|len| len.checked_sub(1))
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    let transform_len = coefficient_len
        .checked_next_power_of_two()
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    if transform_len > MAX_TRANSFORM_LENGTH {
        return Err(MeteredMultiplyError::TransformDomainUnsupported);
    }

    let maximum_terms = u128::try_from(a_coefficient_len.min(b_coefficient_len))
        .map_err(|_| MeteredMultiplyError::SizeOverflow)?;
    let maximum_coefficient = u128::from(COEFFICIENT_MASK);
    let coefficient_bound = maximum_terms
        .checked_mul(maximum_coefficient)
        .and_then(|bound| bound.checked_mul(maximum_coefficient))
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    if coefficient_bound >= CRT_MODULUS {
        return Err(MeteredMultiplyError::TransformDomainUnsupported);
    }

    let output_u32_capacity = a_len
        .checked_add(b_len)
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    Ok(Some(TransformDomain {
        a_coefficient_len,
        b_coefficient_len,
        coefficient_len,
        transform_len,
        output_u32_capacity,
        coefficient_bound,
    }))
}

fn checkpoint_and_charge<M: BudgetMeter>(
    meter: &mut M,
    amount: u64,
) -> Result<(), MeteredMultiplyError> {
    meter.checkpoint()?;
    meter.charge(Dimension::ComputeSteps, amount)?;
    Ok(())
}

fn charge_allocation<M: BudgetMeter>(
    capacity: usize,
    element_size: usize,
    initialization_steps: usize,
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    let memory_bytes = capacity
        .checked_mul(element_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MeteredMultiplyError::SizeOverflow)?;
    let initialization_steps =
        u64::try_from(initialization_steps).map_err(|_| MeteredMultiplyError::SizeOverflow)?;
    meter.checkpoint()?;
    if initialization_steps == 0 {
        meter.charge_batch(&[
            (Dimension::MemoryBytes, memory_bytes),
            (Dimension::AllocationCount, 1),
        ])?;
    } else {
        meter.charge_batch(&[
            (Dimension::MemoryBytes, memory_bytes),
            (Dimension::AllocationCount, 1),
            (Dimension::ComputeSteps, initialization_steps),
        ])?;
    }
    Ok(())
}

fn try_zeroed_u64_vec<M: BudgetMeter>(
    capacity: usize,
    meter: &mut M,
) -> Result<Vec<u64>, MeteredMultiplyError> {
    charge_allocation(capacity, size_of::<u64>(), capacity, meter)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| MeteredMultiplyError::AllocationFailure)?;
    values.resize(capacity, 0);
    Ok(values)
}

fn try_u32_vec<M: BudgetMeter>(
    capacity: usize,
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    charge_allocation(capacity, size_of::<u32>(), 0, meter)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| MeteredMultiplyError::AllocationFailure)?;
    Ok(values)
}

fn multiply_mod(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((u128::from(lhs) * u128::from(rhs)) % u128::from(modulus)) as u64
}

fn mod_pow<M: BudgetMeter>(
    mut base: u64,
    mut exponent: u64,
    modulus: u64,
    meter: &mut M,
) -> Result<u64, MeteredMultiplyError> {
    let mut result = 1;
    base %= modulus;
    while exponent != 0 {
        checkpoint_and_charge(meter, 1)?;
        if exponent & 1 != 0 {
            result = multiply_mod(result, base, modulus);
        }
        base = multiply_mod(base, base, modulus);
        exponent >>= 1;
    }
    Ok(result)
}

fn ntt_transform<M: BudgetMeter>(
    values: &mut [u64],
    prime: u64,
    invert: bool,
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    let len = values.len();
    if !len.is_power_of_two() || len > MAX_TRANSFORM_LENGTH {
        return Err(MeteredMultiplyError::InvariantViolation);
    }

    let mut reversed = 0;
    for index in 1..len {
        checkpoint_and_charge(meter, 1)?;
        let mut bit = len >> 1;
        while reversed & bit != 0 {
            checkpoint_and_charge(meter, 1)?;
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut block_len = 2;
    while block_len <= len {
        let block_len_u64 =
            u64::try_from(block_len).map_err(|_| MeteredMultiplyError::SizeOverflow)?;
        let mut root = mod_pow(PRIMITIVE_ROOT, (prime - 1) / block_len_u64, prime, meter)?;
        if invert {
            root = mod_pow(root, prime - 2, prime, meter)?;
        }

        for block_start in (0..len).step_by(block_len) {
            let mut twiddle = 1;
            for offset in 0..block_len / 2 {
                checkpoint_and_charge(meter, 1)?;
                let left = values[block_start + offset];
                let right =
                    multiply_mod(values[block_start + offset + block_len / 2], twiddle, prime);
                let sum = left + right;
                values[block_start + offset] = if sum >= prime { sum - prime } else { sum };
                values[block_start + offset + block_len / 2] = if left >= right {
                    left - right
                } else {
                    left + prime - right
                };
                twiddle = multiply_mod(twiddle, root, prime);
            }
        }

        if block_len == len {
            break;
        }
        block_len = block_len
            .checked_mul(2)
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
    }

    if invert {
        let inverse_len = mod_pow(
            u64::try_from(len).map_err(|_| MeteredMultiplyError::SizeOverflow)?,
            prime - 2,
            prime,
            meter,
        )?;
        for value in values {
            checkpoint_and_charge(meter, 1)?;
            *value = multiply_mod(*value, inverse_len, prime);
        }
    }
    Ok(())
}

/// Validates input slice lengths before allocation.
pub fn preflight_u32_lengths(a_len: usize, b_len: usize) -> Result<(), MeteredMultiplyError> {
    transform_domain(a_len, b_len).map(|_| ())
}

/// Computes the exact polynomial product of two u32 digit slices using 2-prime NTT and CRT.
pub fn multiply_u32_digits<M: BudgetMeter>(
    a: &[u32],
    b: &[u32],
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    meter.checkpoint()?;
    let domain = match transform_domain(a.len(), b.len())? {
        Some(domain) => domain,
        None => return Ok(Vec::new()),
    };

    let mut a_poly_1 = try_zeroed_u64_vec(domain.transform_len, meter)?;
    let mut b_poly_1 = try_zeroed_u64_vec(domain.transform_len, meter)?;
    let mut a_poly_2 = try_zeroed_u64_vec(domain.transform_len, meter)?;
    let mut b_poly_2 = try_zeroed_u64_vec(domain.transform_len, meter)?;

    for (idx, &digit) in a.iter().enumerate() {
        let low = (digit & COEFFICIENT_MASK) as u64;
        let high = (digit >> COEFFICIENT_BITS) as u64;
        a_poly_1[idx * 2] = low;
        a_poly_1[idx * 2 + 1] = high;
        a_poly_2[idx * 2] = low;
        a_poly_2[idx * 2 + 1] = high;
    }
    for (idx, &digit) in b.iter().enumerate() {
        let low = (digit & COEFFICIENT_MASK) as u64;
        let high = (digit >> COEFFICIENT_BITS) as u64;
        b_poly_1[idx * 2] = low;
        b_poly_1[idx * 2 + 1] = high;
        b_poly_2[idx * 2] = low;
        b_poly_2[idx * 2 + 1] = high;
    }

    ntt_transform(&mut a_poly_1, PRIME_1, false, meter)?;
    ntt_transform(&mut b_poly_1, PRIME_1, false, meter)?;
    for i in 0..domain.transform_len {
        checkpoint_and_charge(meter, 1)?;
        a_poly_1[i] = multiply_mod(a_poly_1[i], b_poly_1[i], PRIME_1);
    }
    ntt_transform(&mut a_poly_1, PRIME_1, true, meter)?;

    ntt_transform(&mut a_poly_2, PRIME_2, false, meter)?;
    ntt_transform(&mut b_poly_2, PRIME_2, false, meter)?;
    for i in 0..domain.transform_len {
        checkpoint_and_charge(meter, 1)?;
        a_poly_2[i] = multiply_mod(a_poly_2[i], b_poly_2[i], PRIME_2);
    }
    ntt_transform(&mut a_poly_2, PRIME_2, true, meter)?;

    let inv_p1_mod_p2 = mod_pow(PRIME_1 % PRIME_2, PRIME_2 - 2, PRIME_2, meter)? as u128;

    let mut carry = 0u128;
    let mut u16_coeffs = Vec::new();
    u16_coeffs
        .try_reserve(domain.coefficient_len + 8)
        .map_err(|_| MeteredMultiplyError::AllocationFailure)?;

    for i in 0..domain.coefficient_len {
        checkpoint_and_charge(meter, 1)?;
        let r1 = a_poly_1[i] as u128;
        let r2 = a_poly_2[i] as u128;
        let p2 = PRIME_2 as u128;
        let diff = if r2 >= (r1 % p2) {
            r2 - (r1 % p2)
        } else {
            r2 + p2 - (r1 % p2)
        };
        let h = (diff * inv_p1_mod_p2) % p2;
        let coeff = r1 + (PRIME_1 as u128) * h;
        let total = coeff + carry;
        u16_coeffs.push((total & (COEFFICIENT_MASK as u128)) as u16);
        carry = total >> COEFFICIENT_BITS;
    }
    while carry > 0 {
        checkpoint_and_charge(meter, 1)?;
        u16_coeffs.push((carry & (COEFFICIENT_MASK as u128)) as u16);
        carry >>= COEFFICIENT_BITS;
    }

    let mut result_digits = try_u32_vec(domain.output_u32_capacity, meter)?;
    for chunk in u16_coeffs.chunks(2) {
        checkpoint_and_charge(meter, 1)?;
        let low = chunk[0] as u32;
        let high = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        result_digits.push(low | (high << 16));
    }

    while result_digits.len() > 1 && result_digits.last() == Some(&0) {
        result_digits.pop();
    }
    if result_digits.last() == Some(&0) {
        result_digits.pop();
    }

    meter.checkpoint()?;
    Ok(result_digits)
}
