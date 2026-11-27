//! Retry a donor pivot refusal in power-of-two equation and variable units.
//! Every transformed coefficient and RHS entry must be exactly reversible.
//! The same pinned pivot policy then applies to the equilibrated matrix.

use super::{MAX_CONSUMER_VARIABLES, SystemConsumerError, charge, donor_solve, finite, zeros};
use fsym_budget::BudgetMeter;

fn exponent(value: f64) -> i32 {
    debug_assert!(value.is_finite() && value != 0.0);
    let bits = value.abs().to_bits();
    let field = (bits >> 52) as i32;
    if field == 0 {
        -1074 + (63 - bits.leading_zeros()) as i32
    } else {
        field - 1023
    }
}

// Scale directly, even when the factor itself is outside binary64. Each
// intermediate multiplication is exact whenever the final value is nonzero
// and finite; the final operation supplies gradual underflow rounding.
fn shift(mut value: f64, mut power: i32) -> f64 {
    while power > 1023 {
        value *= f64::from_bits(1535_u64 << 52);
        power -= 512;
    }
    while power < -1022 {
        value *= f64::from_bits(511_u64 << 52);
        power += 512;
    }
    value * f64::from_bits(((power + 1023) as u64) << 52)
}

fn shift_exact(value: f64, power: i32) -> Option<f64> {
    let scaled = shift(value, power);
    (scaled.is_finite() && shift(scaled, -power) == value).then_some(scaled)
}

pub(super) fn retry<M: BudgetMeter + ?Sized>(
    matrix: &[f64],
    rhs: &[f64],
    n: usize,
    columns: usize,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    let singular =
        || SystemConsumerError::LinearSolve(fnp_linalg::LinAlgError::SolverSingularity.to_string());
    // Stack bounds come from all consumer admission paths. This allowance
    // includes exponent discovery, exact round trips, and solution rescaling.
    charge(
        meter,
        (80 * (matrix.len() + rhs.len()) + 16 * n) as u64,
        0,
        0,
    )?;
    let mut rows = [0_i32; MAX_CONSUMER_VARIABLES];
    let mut vars = [i32::MIN; MAX_CONSUMER_VARIABLES];
    for (i, row) in matrix.chunks_exact(n).enumerate() {
        let max = row.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max == 0.0 {
            return Err(singular());
        }
        rows[i] = exponent(max);
        for (j, &a) in row.iter().enumerate() {
            if a != 0.0 {
                vars[j] = vars[j].max(exponent(a) - rows[i]);
            }
        }
    }
    if vars[..n].contains(&i32::MIN) {
        return Err(singular());
    }
    let mut scaled_a = zeros(matrix.len(), meter)?;
    for (i, row) in matrix.chunks_exact(n).enumerate() {
        for (j, &a) in row.iter().enumerate() {
            scaled_a[i * n + j] = shift_exact(a, -rows[i] - vars[j]).ok_or_else(singular)?;
        }
    }

    // Each RHS may choose its own power of two. Prefer a largest magnitude
    // around one while retaining the smallest component in the normal range
    // when possible. The exact round trip below also admits subnormals.
    let mut rhs_powers = [0_i32; MAX_CONSUMER_VARIABLES];
    let mut scaled_b = zeros(rhs.len(), meter)?;
    for j in 0..columns {
        let mut largest = i32::MIN;
        let mut smallest = i32::MAX;
        for i in 0..n {
            let b = rhs[i * columns + j];
            if b != 0.0 {
                let e = exponent(b) - rows[i];
                largest = largest.max(e);
                smallest = smallest.min(e);
            }
        }
        if largest != i32::MIN {
            rhs_powers[j] = largest.min(smallest + 1022).max(largest - 1023);
        }
        for i in 0..n {
            let k = i * columns + j;
            scaled_b[k] = shift_exact(rhs[k], -rows[i] - rhs_powers[j]).ok_or_else(singular)?;
        }
    }
    let mut solution = donor_solve(&scaled_a, &scaled_b, n, columns, meter)?
        .map_err(|error| SystemConsumerError::LinearSolve(error.to_string()))?;
    super::point_dimension(&solution, rhs.len(), "scaled linear solver result")?;
    finite(&solution, "scaled linear solver result", meter)?;
    for (i, row) in solution.chunks_exact_mut(columns).enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = shift(*x, rhs_powers[j] - vars[i]);
        }
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

    #[test]
    fn exponent_and_reversible_scaling_cover_the_binary64_range() {
        for power in -1074..=1023 {
            let x = shift(-1.0, power);
            assert_eq!(exponent(x), power);
            assert_eq!(shift_exact(x, -power), Some(-1.0));
        }
        assert_eq!(shift_exact(f64::from_bits(3), -1), None);
        assert_eq!(shift_exact(f64::MAX, 1), None);
        assert_eq!(shift(-0.0, 2000).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(shift(1.5, -1074).to_bits(), 2);
    }

    #[test]
    fn relative_column_exponents_need_not_fit_in_a_binary64_scale() {
        let large = shift(1.0, 1023);
        let tiny = f64::from_bits(1);
        // The relative column scale is 2^-2097. A row-normalized
        // intermediate would erase the entire second column.
        let matrix = [large, tiny, large, -tiny];
        let solution =
            super::super::solve_rhs(&matrix, &[tiny, -tiny], 2, 1, &mut Unbounded).unwrap();
        assert_eq!(solution, [0.0, 1.0]);
    }

    #[test]
    fn no_coefficient_can_be_silently_discarded_to_retry_a_pivot() {
        let large = shift(1.0, 1023);
        let tiny = f64::from_bits(1);
        // The off-diagonal entry cannot survive the selected units.
        let matrix = [large, tiny, 0.0, 1.0];
        assert!(matches!(
            super::super::solve_rhs(&matrix, &[large, 1.0], 2, 1, &mut Unbounded),
            Err(SystemConsumerError::LinearSolve(_))
        ));
    }
}
