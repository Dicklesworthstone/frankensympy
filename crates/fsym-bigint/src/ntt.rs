//! Bounded two-prime NTT/CRT candidate support for exact integer multiplication.

#![forbid(unsafe_code)]

use crate::MeteredMultiplyError;
use fsym_budget::{BudgetMeter, Dimension};
use std::mem::size_of;

const COEFFICIENT_BITS: u32 = 16;
const COEFFICIENT_MASK: u32 = 0xffff;

// Keep zero-initialization cancellation latency bounded to at most 64 KiB of writes between
// safe points. The total initialization work is still charged before allocation begins.
const ZERO_INITIALIZATION_CHUNK_ELEMENTS: usize = 8 * 1024;

// Both primes have primitive root 3. Their common power-of-two transform domain is 2^21.
const PRIME_1: u64 = 998_244_353; // 119 * 2^23 + 1
const PRIME_2: u64 = 1_004_535_809; // 479 * 2^21 + 1
const PRIMITIVE_ROOT: u64 = 3;
const CRT_MODULUS: u128 = PRIME_1 as u128 * PRIME_2 as u128;

// Independent of both transform primes. This is an internal fault detector, not evidence.
const CHECK_MODULUS: u64 = 18_446_744_073_709_551_557; // 2^64 - 59
const CHECK_POINT: u64 = 65_537;

/// Maximum exact transform length supported by the fixed two-prime configuration.
pub(crate) const MAX_TRANSFORM_LENGTH: usize = 1 << 21;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransformDomain {
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
    meter.checkpoint()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| MeteredMultiplyError::AllocationFailure)?;
    while values.len() < capacity {
        meter.checkpoint()?;
        let initialized = (capacity - values.len()).min(ZERO_INITIALIZATION_CHUNK_ELEMENTS);
        values.resize(values.len() + initialized, 0);
    }
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

fn multiply_add_mod(lhs: u64, rhs: u64, addend: u64, modulus: u64) -> u64 {
    ((u128::from(lhs) * u128::from(rhs) + u128::from(addend)) % u128::from(modulus)) as u64
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
pub(crate) fn preflight_u32_lengths(
    a_len: usize,
    b_len: usize,
) -> Result<(), MeteredMultiplyError> {
    transform_domain(a_len, b_len).map(|_| ())
}

fn fill_transform_inputs<M: BudgetMeter>(
    digits: &[u32],
    prime_1_values: &mut [u64],
    prime_2_values: &mut [u64],
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    for (index, &digit) in digits.iter().enumerate() {
        let coefficient_index = index
            .checked_mul(2)
            .ok_or(MeteredMultiplyError::SizeOverflow)?;
        let low = u64::from(digit & COEFFICIENT_MASK);
        let high = u64::from(digit >> COEFFICIENT_BITS);
        checkpoint_and_charge(meter, 2)?;
        prime_1_values[coefficient_index] = low;
        prime_1_values[coefficient_index + 1] = high;
        prime_2_values[coefficient_index] = low;
        prime_2_values[coefficient_index + 1] = high;
    }
    Ok(())
}

fn pointwise_multiply<M: BudgetMeter>(
    lhs: &mut [u64],
    rhs: &[u64],
    prime: u64,
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    if lhs.len() != rhs.len() {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    for (lhs_value, &rhs_value) in lhs.iter_mut().zip(rhs) {
        checkpoint_and_charge(meter, 1)?;
        *lhs_value = multiply_mod(*lhs_value, rhs_value, prime);
    }
    Ok(())
}

fn evaluate_split_coefficients<M: BudgetMeter>(
    digits: &[u32],
    point: u64,
    meter: &mut M,
) -> Result<u64, MeteredMultiplyError> {
    let mut value = 0;
    for &digit in digits.iter().rev() {
        let high = u64::from(digit >> COEFFICIENT_BITS);
        let low = u64::from(digit & COEFFICIENT_MASK);
        checkpoint_and_charge(meter, 1)?;
        value = multiply_add_mod(value, point, high, CHECK_MODULUS);
        checkpoint_and_charge(meter, 1)?;
        value = multiply_add_mod(value, point, low, CHECK_MODULUS);
    }
    Ok(value)
}

fn evaluate_u32_digits<M: BudgetMeter>(
    digits: &[u32],
    meter: &mut M,
) -> Result<u64, MeteredMultiplyError> {
    let radix = (1u64 << 32) % CHECK_MODULUS;
    let mut value = 0;
    for &digit in digits.iter().rev() {
        checkpoint_and_charge(meter, 1)?;
        value = multiply_add_mod(value, radix, u64::from(digit), CHECK_MODULUS);
    }
    Ok(value)
}

fn reconstruct_coefficient(residue_1: u64, residue_2: u64, inverse: u64) -> u128 {
    let residue_1_mod_prime_2 = residue_1 % PRIME_2;
    let difference = if residue_2 >= residue_1_mod_prime_2 {
        residue_2 - residue_1_mod_prime_2
    } else {
        residue_2 + PRIME_2 - residue_1_mod_prime_2
    };
    let multiplier = multiply_mod(difference, inverse, PRIME_2);
    u128::from(residue_1) + u128::from(PRIME_1) * u128::from(multiplier)
}

fn push_base16_limb<M: BudgetMeter>(
    limb: u32,
    pending_low: &mut Option<u32>,
    output: &mut Vec<u32>,
    output_capacity: usize,
    meter: &mut M,
) -> Result<(), MeteredMultiplyError> {
    if limb > COEFFICIENT_MASK {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    checkpoint_and_charge(meter, 1)?;
    if let Some(low) = pending_low.take() {
        if output.len() >= output_capacity {
            return Err(MeteredMultiplyError::InvariantViolation);
        }
        output.push(low | (limb << COEFFICIENT_BITS));
    } else {
        *pending_low = Some(limb);
    }
    Ok(())
}

/// Computes the exact product of two canonical base-2^32 digit slices using two-prime NTT/CRT.
pub(crate) fn multiply_u32_digits<M: BudgetMeter>(
    a: &[u32],
    b: &[u32],
    meter: &mut M,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    multiply_u32_digits_inner(a, b, meter, None, None)
}

fn multiply_u32_digits_inner<M: BudgetMeter>(
    a: &[u32],
    b: &[u32],
    meter: &mut M,
    corrupt_coefficient: Option<usize>,
    corrupt_output_digit: Option<usize>,
) -> Result<Vec<u32>, MeteredMultiplyError> {
    meter.checkpoint()?;
    let domain = match transform_domain(a.len(), b.len())? {
        Some(domain) => domain,
        None => return Ok(Vec::new()),
    };

    let mut a_prime_1 = try_zeroed_u64_vec(domain.transform_len, meter)?;
    let mut b_prime_1 = try_zeroed_u64_vec(domain.transform_len, meter)?;
    let mut a_prime_2 = try_zeroed_u64_vec(domain.transform_len, meter)?;
    let mut b_prime_2 = try_zeroed_u64_vec(domain.transform_len, meter)?;
    fill_transform_inputs(a, &mut a_prime_1, &mut a_prime_2, meter)?;
    fill_transform_inputs(b, &mut b_prime_1, &mut b_prime_2, meter)?;

    let expected_raw_evaluation = multiply_mod(
        evaluate_split_coefficients(a, CHECK_POINT, meter)?,
        evaluate_split_coefficients(b, CHECK_POINT, meter)?,
        CHECK_MODULUS,
    );

    ntt_transform(&mut a_prime_1, PRIME_1, false, meter)?;
    ntt_transform(&mut b_prime_1, PRIME_1, false, meter)?;
    pointwise_multiply(&mut a_prime_1, &b_prime_1, PRIME_1, meter)?;
    ntt_transform(&mut a_prime_1, PRIME_1, true, meter)?;

    ntt_transform(&mut a_prime_2, PRIME_2, false, meter)?;
    ntt_transform(&mut b_prime_2, PRIME_2, false, meter)?;
    pointwise_multiply(&mut a_prime_2, &b_prime_2, PRIME_2, meter)?;
    ntt_transform(&mut a_prime_2, PRIME_2, true, meter)?;

    if let Some(index) = corrupt_coefficient {
        if index >= domain.coefficient_len {
            return Err(MeteredMultiplyError::InvariantViolation);
        }
        checkpoint_and_charge(meter, 1)?;
        a_prime_1[index] = (a_prime_1[index] + 1) % PRIME_1;
        a_prime_2[index] = (a_prime_2[index] + 1) % PRIME_2;
    }

    let inverse_prime_1_mod_prime_2 = mod_pow(PRIME_1 % PRIME_2, PRIME_2 - 2, PRIME_2, meter)?;

    let mut result = try_u32_vec(domain.output_u32_capacity, meter)?;
    let mut pending_low = None;
    let mut carry = 0u128;
    let mut reconstructed_evaluation = 0;
    let mut evaluation_power = 1;
    for index in 0..domain.coefficient_len {
        checkpoint_and_charge(meter, 1)?;
        let coefficient = reconstruct_coefficient(
            a_prime_1[index],
            a_prime_2[index],
            inverse_prime_1_mod_prime_2,
        );
        if coefficient > domain.coefficient_bound {
            return Err(MeteredMultiplyError::InvariantViolation);
        }
        let coefficient_mod_check = u64::try_from(coefficient % u128::from(CHECK_MODULUS))
            .map_err(|_| MeteredMultiplyError::InvariantViolation)?;
        reconstructed_evaluation = multiply_add_mod(
            coefficient_mod_check,
            evaluation_power,
            reconstructed_evaluation,
            CHECK_MODULUS,
        );
        evaluation_power = multiply_mod(evaluation_power, CHECK_POINT, CHECK_MODULUS);

        let total = coefficient
            .checked_add(carry)
            .ok_or(MeteredMultiplyError::InvariantViolation)?;
        let limb = u32::try_from(total & u128::from(COEFFICIENT_MASK))
            .map_err(|_| MeteredMultiplyError::InvariantViolation)?;
        carry = total >> COEFFICIENT_BITS;
        push_base16_limb(
            limb,
            &mut pending_low,
            &mut result,
            domain.output_u32_capacity,
            meter,
        )?;
    }
    if reconstructed_evaluation != expected_raw_evaluation {
        return Err(MeteredMultiplyError::InvariantViolation);
    }

    while carry != 0 {
        checkpoint_and_charge(meter, 1)?;
        let limb = u32::try_from(carry & u128::from(COEFFICIENT_MASK))
            .map_err(|_| MeteredMultiplyError::InvariantViolation)?;
        carry >>= COEFFICIENT_BITS;
        push_base16_limb(
            limb,
            &mut pending_low,
            &mut result,
            domain.output_u32_capacity,
            meter,
        )?;
    }
    if let Some(low) = pending_low {
        checkpoint_and_charge(meter, 1)?;
        if result.len() >= domain.output_u32_capacity {
            return Err(MeteredMultiplyError::InvariantViolation);
        }
        result.push(low);
    }
    while result.last() == Some(&0) {
        checkpoint_and_charge(meter, 1)?;
        result.pop();
    }
    if result.is_empty() {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    if let Some(index) = corrupt_output_digit {
        checkpoint_and_charge(meter, 1)?;
        let digit = result
            .get_mut(index)
            .ok_or(MeteredMultiplyError::InvariantViolation)?;
        *digit ^= 1;
    }

    let expected_integer_evaluation = multiply_mod(
        evaluate_u32_digits(a, meter)?,
        evaluate_u32_digits(b, meter)?,
        CHECK_MODULUS,
    );
    if evaluate_u32_digits(&result, meter)? != expected_integer_evaluation {
        return Err(MeteredMultiplyError::InvariantViolation);
    }
    meter.checkpoint()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{MeterError, Unbounded};

    #[derive(Debug)]
    struct CancelAtCheckpoint {
        cancel_at: usize,
        checkpoints: usize,
    }

    impl BudgetMeter for CancelAtCheckpoint {
        fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
            Ok(())
        }

        fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints += 1;
            if self.checkpoints >= self.cancel_at {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    fn power(base: u64, exponent: u64, modulus: u64) -> u64 {
        mod_pow(base, exponent, modulus, &mut Unbounded).unwrap()
    }

    fn scalar_product(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
        let mut coefficients = vec![0u128; lhs.len() + rhs.len()];
        for (lhs_index, &lhs_digit) in lhs.iter().enumerate() {
            for (rhs_index, &rhs_digit) in rhs.iter().enumerate() {
                coefficients[lhs_index + rhs_index] +=
                    u128::from(lhs_digit) * u128::from(rhs_digit);
            }
        }
        let mut carry = 0u128;
        let mut output = Vec::with_capacity(coefficients.len());
        for coefficient in coefficients {
            let total = coefficient + carry;
            output.push(total as u32);
            carry = total >> 32;
        }
        while carry != 0 {
            output.push(carry as u32);
            carry >>= 32;
        }
        while output.last() == Some(&0) {
            output.pop();
        }
        output
    }

    #[test]
    fn fixed_roots_have_the_required_exact_orders() {
        for prime in [PRIME_1, PRIME_2] {
            let transform_len = MAX_TRANSFORM_LENGTH as u64;
            let root = power(PRIMITIVE_ROOT, (prime - 1) / transform_len, prime);
            assert_eq!(power(root, transform_len, prime), 1);
            assert_ne!(power(root, transform_len / 2, prime), 1);
        }
    }

    #[test]
    fn zeroed_transform_buffer_observes_cancellation_after_allocation_charge() {
        let mut meter = CancelAtCheckpoint {
            cancel_at: 2,
            checkpoints: 0,
        };
        assert_eq!(
            try_zeroed_u64_vec(1, &mut meter),
            Err(MeteredMultiplyError::Meter(MeterError::Cancelled))
        );
        assert_eq!(meter.checkpoints, 2);
    }

    #[test]
    fn zeroed_transform_buffer_checks_between_initialization_chunks() {
        let mut meter = CancelAtCheckpoint {
            cancel_at: 4,
            checkpoints: 0,
        };
        assert_eq!(
            try_zeroed_u64_vec(ZERO_INITIALIZATION_CHUNK_ELEMENTS + 1, &mut meter),
            Err(MeteredMultiplyError::Meter(MeterError::Cancelled))
        );
        assert_eq!(meter.checkpoints, 4);
    }

    #[test]
    fn inverse_round_trip_and_crt_product_match_independent_scalar_lane() {
        let original = vec![1, 65_535, 17, 42, 999, 0, 12_345, 7];
        for prime in [PRIME_1, PRIME_2] {
            let mut transformed = original.clone();
            ntt_transform(&mut transformed, prime, false, &mut Unbounded).unwrap();
            assert_ne!(transformed, original);
            ntt_transform(&mut transformed, prime, true, &mut Unbounded).unwrap();
            assert_eq!(transformed, original);
        }

        let lhs = [u32::MAX, 0x8000_0001, 0, 17];
        let rhs = [0xffff_0001, u32::MAX, 65_537];
        let actual = multiply_u32_digits(&lhs, &rhs, &mut Unbounded).unwrap();
        assert_eq!(actual, scalar_product(&lhs, &rhs));
    }

    #[test]
    fn admission_proves_reconstruction_bound_before_allocation() {
        let domain = transform_domain(MAX_TRANSFORM_LENGTH / 4, 1)
            .unwrap()
            .unwrap();
        assert!(domain.coefficient_bound < CRT_MODULUS);
        assert_eq!(domain.transform_len, MAX_TRANSFORM_LENGTH);
        assert_eq!(
            transform_domain(MAX_TRANSFORM_LENGTH / 2, 1),
            Err(MeteredMultiplyError::TransformDomainUnsupported)
        );
    }

    #[test]
    fn independent_checks_reject_planted_residue_and_output_corruption() {
        let lhs = [u32::MAX, 0x1234_5678, 0x9abc_def0, 7];
        let rhs = [0x8765_4321, u32::MAX, 11];
        assert_eq!(
            multiply_u32_digits_inner(&lhs, &rhs, &mut Unbounded, Some(1), None),
            Err(MeteredMultiplyError::InvariantViolation)
        );
        assert_eq!(
            multiply_u32_digits_inner(&lhs, &rhs, &mut Unbounded, None, Some(0)),
            Err(MeteredMultiplyError::InvariantViolation)
        );
    }
}
