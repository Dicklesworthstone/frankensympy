//! Round the least-squares cost and each gradient component once, after all products.
//! Rounding each multiply and add separately can turn a nonzero gradient into
//! an apparent stationary point. A verified sparse column has at most 4096
//! distinct rows. Its exact signed dot product therefore fits in 4208 bits
//! in units of 2^-2148, including products outside the binary64 range.

use super::{SystemConsumerError, buffer, charge};
use fsym_budget::BudgetMeter;

const WORDS: usize = 132;
const MASK: i64 = u32::MAX as i64;
const HIDDEN: u64 = 1 << 52;
// Exact products are stored in units of 2^-2148. A rounded normal f64
// has quantum 2^(exponent - 52), for unbiased exponent -1022 through 1023.
const MIN_DISCARD: usize = 2148 - 1022 - 52;
const MAX_DISCARD: usize = 2148 + 1023 - 52;

struct ProductSum {
    // Signed base-2^32 digits. Carries can be deferred until all products are
    // present: each term adds at most two 32-bit pieces to any one digit.
    digits: [i64; WORDS],
}

fn parts(bits: u64) -> (u64, usize) {
    let exponent = ((bits >> 52) & 0x7ff) as usize;
    let fraction = bits & (HIDDEN - 1);
    if exponent == 0 {
        (fraction, 0)
    } else {
        (fraction | HIDDEN, exponent - 1)
    }
}

impl ProductSum {
    fn add(&mut self, left: f64, right: f64) {
        let left = left.to_bits();
        let right = right.to_bits();
        let (a, a_shift) = parts(left);
        let (b, b_shift) = parts(right);
        let mut product = u128::from(a) * u128::from(b);
        let sign = if (left ^ right) >> 63 == 0 { 1 } else { -1 };
        let shift = (a_shift + b_shift) % 32;
        let mut index = (a_shift + b_shift) / 32;
        while product != 0 {
            let piece = u64::from(product as u32) << shift;
            self.digits[index] += sign * i64::from(piece as u32);
            self.digits[index + 1] += sign * ((piece >> 32) as i64);
            product >>= 32;
            index += 1;
        }
    }

    fn carry(&mut self) {
        for index in 0..WORDS - 1 {
            self.digits[index + 1] += self.digits[index] >> 32;
            self.digits[index] &= MASK;
        }
    }

    // Dividing the exact sum by two raises its discard thresholds by one bit.
    fn rounded(mut self, halve: bool, label: &'static str) -> Result<f64, SystemConsumerError> {
        let min_discard = MIN_DISCARD + usize::from(halve);
        let max_discard = MAX_DISCARD + usize::from(halve);
        self.carry();
        let negative = self.digits[WORDS - 1] < 0;
        if negative {
            for digit in &mut self.digits {
                *digit = -*digit;
            }
            self.carry();
        }
        let Some(last) = self.digits.iter().rposition(|&digit| digit != 0) else {
            return Ok(0.0);
        };
        let top = 32 * last + (31 - (self.digits[last] as u32).leading_zeros()) as usize;
        // Discard enough low bits to fit a 53-bit significand, or to reach
        // the least subnormal quantum. Use guard, sticky and parity for ties.
        let mut discard = top.saturating_sub(52).max(min_discard);
        let overflow = SystemConsumerError::NonFinite(label);
        if discard > max_discard {
            return Err(overflow);
        }
        let word = discard / 32;
        let window = self.digits[word] as u128
            | ((self.digits[word + 1] as u128) << 32)
            | ((self.digits[word + 2] as u128) << 64);
        let mut significand = (window >> (discard % 32)) as u64;
        let guard = discard - 1;
        let mask = 1_i64 << (guard % 32);
        let digit = self.digits[guard / 32];
        let sticky = digit & (mask - 1) != 0 || self.digits[..guard / 32].iter().any(|&d| d != 0);
        if digit & mask != 0 && (sticky || significand & 1 != 0) {
            significand += 1;
        }
        if significand == 2 * HIDDEN {
            significand >>= 1;
            discard += 1;
        }
        if discard > max_discard {
            return Err(overflow);
        }
        let magnitude = if significand < HIDDEN {
            significand
        } else {
            (((discard - min_discard + 1) as u64) << 52) | (significand - HIDDEN)
        };
        Ok(f64::from_bits((u64::from(negative) << 63) | magnitude))
    }
}

/// Inputs have passed the original-source finite checks. Charge each fixed
/// accumulator, sparse pass and bounded carry/rounding pass to the caller.
pub(super) fn cost_and_gradient<M: BudgetMeter>(
    pattern: &[(usize, usize)],
    jacobian: &[f64],
    residuals: &[f64],
    columns: usize,
    meter: &mut M,
) -> Result<(f64, Vec<f64>), SystemConsumerError> {
    if pattern.len() != jacobian.len() {
        return Err(SystemConsumerError::ConsumerContract(
            "least-squares Jacobian value count",
        ));
    }
    let mut squares = ProductSum { digits: [0; WORDS] };
    charge(meter, (9 * WORDS) as u64, 0, 0)?;
    for chunk in residuals.chunks(256) {
        charge(meter, (64 * chunk.len()) as u64, 0, 0)?;
        for &residual in chunk {
            squares.add(residual, residual);
        }
    }
    let cost = squares.rounded(true, "least-squares squared residual cost")?;
    // Index each sparse column with one link per stored derivative. Only one
    // wide accumulator is live at a time, regardless of the number of axes.
    let end = pattern.len();
    let mut heads = buffer(columns, meter)?;
    charge(meter, columns as u64, 0, 0)?;
    heads.resize(columns, end);
    let mut next = buffer(end, meter)?;
    for chunk in pattern.chunks(256) {
        charge(meter, (4 * chunk.len()) as u64, 0, 0)?;
        for &(row, col) in chunk {
            if row >= residuals.len() || col >= columns {
                return Err(SystemConsumerError::ConsumerContract(
                    "least-squares Jacobian coordinate",
                ));
            }
            let index = next.len();
            next.push(heads[col]);
            heads[col] = index;
        }
    }
    let mut result = buffer(columns, meter)?;
    for mut index in heads {
        charge(meter, 4, 0, 0)?;
        // One hardware multiply already rounds the exact single-term dot
        // product once. Empty and diagonal columns need no wide arithmetic.
        let value = if index == end {
            0.0
        } else if next[index] == end {
            let a = jacobian[index];
            let b = residuals[pattern[index].0];
            let product = if a == 0.0 || b == 0.0 { 0.0 } else { a * b };
            if !product.is_finite() {
                return Err(SystemConsumerError::NonFinite("least-squares gradient sum"));
            }
            product
        } else {
            let mut sum = ProductSum { digits: [0; WORDS] };
            charge(meter, (9 * WORDS) as u64, 0, 0)?;
            while index != end {
                charge(meter, 64, 0, 0)?;
                sum.add(jacobian[index], residuals[pattern[index].0]);
                index = next[index];
            }
            sum.rounded(false, "least-squares gradient sum")?
        };
        result.push(value);
    }
    Ok((cost, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(pairs: &[(f64, f64)]) -> Result<f64, SystemConsumerError> {
        let mut acc = ProductSum { digits: [0; WORDS] };
        for &(a, b) in pairs {
            acc.add(a, b);
        }
        acc.rounded(false, "test dot product")
    }

    #[test]
    fn retains_low_product_bits_through_signed_cancellation() {
        let a = 134_217_729.0; // 2^27 + 1
        let rounded_square = a * a;
        for sign in [1.0, -1.0] {
            assert_eq!(sum(&[(sign * a, a), (-sign, rounded_square)]), Ok(sign));
            assert_eq!(
                sum(&[(f64::MAX, f64::MAX), (sign, 1.0), (-f64::MAX, f64::MAX)]),
                Ok(sign)
            );
        }
        assert!(sum(&[(f64::MAX, 2.0)]).is_err());
    }

    #[test]
    fn gradual_underflow_rounds_once_with_even_ties() {
        let tiny = f64::from_bits(1);
        for sign in [1.0_f64, -1.0] {
            let half = (tiny * sign, 0.5);
            assert_eq!(sum(&[half]).unwrap().to_bits(), (sign * 0.0).to_bits());
            assert_eq!(sum(&[half, half]), Ok(sign * tiny));
            assert_eq!(sum(&[half; 3]), Ok(sign * 2.0 * tiny));
            assert_eq!(sum(&[half; 5]), Ok(sign * 2.0 * tiny));
        }
    }

    #[test]
    fn rounds_across_binades_and_refuses_only_final_overflow() {
        let half_gap = 2.0_f64.powi(-54);
        let below_one = f64::from_bits(1.0_f64.to_bits() - 1);
        assert_eq!(sum(&[(below_one, 1.0), (half_gap, 1.0)]), Ok(1.0));
        assert_eq!(
            sum(&[(f64::MAX, 1.0), (2.0_f64.powi(969), 1.0)]),
            Ok(f64::MAX)
        );
        assert!(sum(&[(f64::MAX, 1.0), (2.0_f64.powi(970), 1.0)]).is_err());
    }

    #[test]
    fn half_squared_norm_rounds_in_the_range_of_the_final_cost() {
        let cost = |values: &[f64]| {
            let mut squares = ProductSum { digits: [0; WORDS] };
            for &value in values {
                squares.add(value, value);
            }
            squares.rounded(true, "test cost")
        };
        let large = 2.0_f64.powi(512);
        assert_eq!(cost(&[large]), Ok(2.0_f64.powi(1023)));
        assert!(cost(&[large, large]).is_err());
        let small = 2.0_f64.powi(-538);
        assert_eq!(cost(&[small; 4]), Ok(0.0));
        assert_eq!(cost(&[small; 8]), Ok(f64::from_bits(1)));
        assert_eq!(cost(&[small; 12]), Ok(f64::from_bits(2)));
    }

    #[test]
    fn sparse_columns_keep_independent_sums_and_empty_coordinates() {
        let large = 2.0_f64.powi(53);
        let (_, gradient) = cost_and_gradient(
            &[(0, 1), (1, 0), (1, 1), (2, 1), (3, 3)],
            &[1.0, 7.0, 1.0, 1.0, -0.0],
            &[large, 1.0, -large, -2.0],
            4,
            &mut fsym_budget::Unbounded,
        )
        .unwrap();
        assert_eq!(gradient, [7.0, 1.0, 0.0, 0.0]);
        assert_eq!(gradient[3].to_bits(), 0);
    }

    #[test]
    fn cancelled_large_products_preserve_single_rounding_across_binary64() {
        // The two extreme products cancel exactly. A single hardware
        // multiplication independently supplies the required rounded result.
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let mut sample = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Keep the sign and significand; fold the nonfinite exponent to
            // the largest finite exponent without narrowing the other cases.
            let bits = if state & f64::INFINITY.to_bits() == f64::INFINITY.to_bits() {
                state ^ (1_u64 << 52)
            } else {
                state
            };
            f64::from_bits(bits)
        };
        for _ in 0..1024 {
            let a = sample();
            let b = sample();
            let expected = a * b;
            let found = sum(&[(f64::MAX, f64::MAX), (a, b), (-f64::MAX, f64::MAX)]);
            if expected.is_finite() {
                let expected = if a == 0.0 || b == 0.0 { 0.0 } else { expected };
                assert_eq!(found.unwrap().to_bits(), expected.to_bits());
            } else {
                assert!(found.is_err());
            }
        }
    }
}
