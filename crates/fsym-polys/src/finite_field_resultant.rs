//! Independent resultant and discriminant algorithms over a bounded prime field.
//!
//! Generation follows Euclidean polynomial remainders. Verification constructs
//! the full Sylvester matrix and computes its determinant by modular row
//! elimination, with integer extended Euclid for pivot inverses. It never
//! invokes polynomial division, the generator's scalar powers, or its remainder
//! recurrence. Matrices have at most 128 rows and share the operation's complete
//! generator/verifier allocation and computation ledgers.
//!
//! The discriminant is the universal polynomial in the original coefficients.
//! Its verifier retains the derivative's formal degree n-1 even when the
//! characteristic cancels leading derivative coefficients. Trimming that formal
//! degree and blindly dividing the resultant by the leading coefficient would
//! change the invariant in positive characteristic.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field::{copy, derivative, divide, scalar_power, zeros};
use crate::{MAX_PRIME_FIELD_DEGREE, PolyError};
use fsym_budget::BudgetMeter;

fn scalar(
    value: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u64>, PolyError> {
    if value == 0 {
        copy(&[], work)
    } else {
        copy(&[value], work)
    }
}

fn generated_value(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    work.tick(1)?;
    if left.is_empty() || right.is_empty() {
        return Ok(0);
    }
    let (mut a, mut b) = (copy(left, work)?, copy(right, work)?);
    let mut value = 1u64;
    while b.len() > 1 {
        let (m, n) = (a.len() - 1, b.len() - 1);
        let remainder = divide(&a, &b, prime, work)?.1;
        if remainder.is_empty() {
            return Ok(0);
        }
        // Res(a,b)=(-1)^(mn) lc(b)^(m-deg(r)) Res(b,r).
        // This also handles m<n: the first remainder is a and the power is 0.
        let factor = scalar_power(b[n], (m + 1 - remainder.len()) as u64, prime, work)?;
        work.tick(2)?;
        value = value * factor % prime;
        if !(m * n).is_multiple_of(2) && value != 0 {
            value = prime - value;
        }
        (a, b) = (b, remainder);
    }
    let factor = scalar_power(b[0], (a.len() - 1) as u64, prime, work)?;
    work.tick(1)?;
    Ok(value * factor % prime)
}

pub(super) fn generated_resultant(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u64>, PolyError> {
    let value = generated_value(left, right, prime, work)?;
    scalar(value, work)
}

pub(super) fn generated_discriminant(
    source: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u64>, PolyError> {
    work.tick(1)?;
    if source.len() <= 1 {
        return scalar(0, work);
    }
    let n = source.len() - 1;
    if n == 1 {
        return scalar(1, work);
    }
    let differentiated = derivative(source, prime, work)?;
    if differentiated.is_empty() {
        return scalar(0, work);
    }
    let m = differentiated.len() - 1;
    let mut value = generated_value(source, &differentiated, prime, work)?;
    // The trimmed derivative has degree m. Restore the leading-coefficient
    // factor removed by that trim: Disc(f)=sign*lc(f)^(n-2-m)*Res(f,f').
    let exponent = if m == n - 1 {
        prime - 2 // inverse leading coefficient; n-2-m=-1
    } else {
        (n - 2 - m) as u64
    };
    let factor = scalar_power(source[n], exponent, prime, work)?;
    work.tick(2)?;
    value = value * factor % prime;
    if !(n * (n - 1) / 2).is_multiple_of(2) && value != 0 {
        value = prime - value;
    }
    scalar(value, work)
}

fn reference_inverse(
    value: u64,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    let (mut r, mut next_r) = (i128::from(prime), i128::from(value));
    let (mut t, mut next_t) = (0i128, 1i128);
    while next_r != 0 {
        work.tick(1)?;
        let quotient = r / next_r;
        (r, next_r) = (next_r, r - quotient * next_r);
        (t, next_t) = (next_t, t - quotient * next_t);
    }
    if r != 1 {
        return Err(refused("prime-field Sylvester pivot is not invertible"));
    }
    Ok(t.rem_euclid(i128::from(prime)) as u64)
}

fn sylvester_determinant(
    left: &[u64],
    m: usize,
    right: &[u64],
    n: usize,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    work.tick(1)?;
    if m > MAX_PRIME_FIELD_DEGREE || n > MAX_PRIME_FIELD_DEGREE {
        return Err(refused("prime-field Sylvester matrix exceeds 128 rows"));
    }
    let size = m + n;
    let mut matrix = zeros(size * size, work)?;
    work.tick(size * size)?;
    // Coefficients are stored ascending; Sylvester rows are descending.
    // `m` and `n` are formal degrees, never inferred by trimming these rows.
    for row in 0..n {
        for index in 0..=m {
            matrix[row * size + row + index] = left.get(m - index).copied().unwrap_or(0);
        }
    }
    for row in 0..m {
        for index in 0..=n {
            matrix[(n + row) * size + row + index] = right.get(n - index).copied().unwrap_or(0);
        }
    }
    let modulus = i128::from(prime);
    let mut determinant = 1u64;
    for column in 0..size {
        work.tick(size - column)?;
        let Some(pivot_row) = (column..size).find(|&row| matrix[row * size + column] != 0) else {
            return Ok(0);
        };
        if pivot_row != column {
            work.tick(size)?;
            for index in 0..size {
                matrix.swap(column * size + index, pivot_row * size + index);
            }
            determinant = prime - determinant;
        }
        let pivot = matrix[column * size + column];
        determinant = (u128::from(determinant) * u128::from(pivot) % u128::from(prime)) as u64;
        let inverse = reference_inverse(pivot, prime, work)?;
        for row in column + 1..size {
            work.tick(1)?;
            if matrix[row * size + column] == 0 {
                continue;
            }
            let factor = (u128::from(matrix[row * size + column]) * u128::from(inverse)
                % u128::from(prime)) as i128;
            matrix[row * size + column] = 0;
            work.tick(size - column)?;
            for index in column + 1..size {
                let updated = i128::from(matrix[row * size + index])
                    - factor * i128::from(matrix[column * size + index]);
                matrix[row * size + index] = updated.rem_euclid(modulus) as u64;
            }
        }
    }
    Ok(determinant)
}

fn proposed_scalar(values: &[u64]) -> Result<u64, PolyError> {
    match values {
        [] => Ok(0),
        [value] => Ok(*value),
        _ => Err(refused(
            "prime-field resultant or discriminant must be a scalar",
        )),
    }
}

fn equal(actual: u64, expected: u64) -> Result<(), PolyError> {
    if actual != expected {
        return Err(PolyError::IdentityCheckFailed(
            "prime-field Sylvester determinant identity failed".into(),
        ));
    }
    Ok(())
}

pub(super) fn verify_resultant(
    left: &[u64],
    right: &[u64],
    result: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let actual = proposed_scalar(result)?;
    work.tick(1)?;
    let expected = if left.is_empty() || right.is_empty() {
        0
    } else {
        sylvester_determinant(left, left.len() - 1, right, right.len() - 1, prime, work)?
    };
    equal(actual, expected)
}

pub(super) fn verify_discriminant(
    source: &[u64],
    result: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let actual = proposed_scalar(result)?;
    work.tick(1)?;
    if source.len() <= 1 {
        return equal(actual, 0);
    }
    let n = source.len() - 1;
    if n == 1 {
        return equal(actual, 1);
    }
    // Keep all n slots, including leading zeros when p divides n. Using the
    // actual derivative degree here would certify a different invariant.
    let mut differentiated = zeros(n, work)?;
    work.tick(n)?;
    for (power, &coefficient) in source.iter().enumerate().skip(1) {
        differentiated[power - 1] =
            (u128::from(coefficient) * power as u128 % u128::from(prime)) as u64;
    }
    let mut determinant = sylvester_determinant(source, n, &differentiated, n - 1, prime, work)?;
    if !(n * (n - 1) / 2).is_multiple_of(2) && determinant != 0 {
        determinant = prime - determinant;
    }
    work.tick(1)?;
    let scaled = (u128::from(actual) * u128::from(source[n]) % u128::from(prime)) as u64;
    equal(scaled, determinant)
}
