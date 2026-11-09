//! Bounded sparse root proposals from independently admitted residual programs.
//!
//! The shared LSQR kernel proposes column-equilibrated Newton directions,
//! including for consistent singular systems. Only a fresh original-source
//! residual infinity norm at or below the requested tolerance admits a root
//! proposal. A stationary least-squares cost, a short step, or a linear solver
//! stopping condition cannot authorize success. No existence, uniqueness,
//! conditioning, or certification claim is made.
//!
//! Each outer iteration fixes a residual scale and uses the corresponding
//! normalized squared-residual merit for bounded Armijo backtracking. This
//! avoids squaring very small or very large original residuals. The source,
//! Jacobian, and final stopping test remain unscaled. Scratch is O(nnz + n),
//! and every rejected direction/trial retains its cumulative meter charges.

use super::sparse_linear::{
    DAMPING_ATTEMPTS, LinearModel, column_scales, dot, finite_scalar, lsqr_scaled_direction,
    trial_domain_failure,
};
use super::{
    RootProposal, SystemConsumerError, charge, copy_values, finite, infinity_norm, point_dimension,
    publication_checkpoint, validate_guards, zeros,
};
use crate::NewtonGuards;
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_calculus::VerifiedResidualSystem;

pub const MAX_SPARSE_NEWTON_VARIABLES: usize = 256;
const MAX_BACKTRACKS: usize = 32;
const ARMIJO: f64 = 1e-4;

struct Evaluation {
    residuals: Vec<f64>,
    jacobian: Vec<f64>,
    residual_norm: f64,
}

pub fn sparse_newton_refine(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    guards: &NewtonGuards,
) -> Result<RootProposal, SystemConsumerError> {
    sparse_newton_refine_with_meter(system, initial, guards, &mut Unbounded)
}

/// Propose a root of a square system with 1..=256 declared variables.
///
/// Admission bounds nonlinear iterations by 1024. Each iteration admits at
/// most six damping attempts, 512 LSQR steps per direction, and 32 line-search
/// trials. The direction is clipped to `guards.max_step_norm` before searching.
/// Initial/final source refusals and operational errors are terminal; only
/// finite-arithmetic or mathematical-domain failures of trial points backtrack.
pub fn sparse_newton_refine_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    guards: &NewtonGuards,
    meter: &mut M,
) -> Result<RootProposal, SystemConsumerError> {
    meter.checkpoint()?;
    let n = system.num_vars();
    if !(1..=MAX_SPARSE_NEWTON_VARIABLES).contains(&n) || system.num_residuals() != n {
        return Err(SystemConsumerError::UnsupportedSparseRootShape {
            variables: n,
            residuals: system.num_residuals(),
        });
    }
    validate_guards(guards)?;
    point_dimension(initial, n, "sparse Newton initial point")?;
    finite(initial, "sparse Newton initial point", meter)?;
    let mut point = copy_values(initial, meter)?;
    let mut state = evaluate(system, &point, n, meter)?;
    let mut trial_point = zeros(n, meter)?;

    for iteration in 0..=guards.max_iterations {
        if state.residual_norm <= guards.residual_tolerance {
            let checked = evaluate(system, &point, n, meter)?;
            if checked.residual_norm > guards.residual_tolerance {
                return Err(SystemConsumerError::ConsumerContract(
                    "sparse Newton final original residual agreement",
                ));
            }
            publication_checkpoint(n, meter)?;
            return Ok(RootProposal {
                root: point,
                residual_norm: checked.residual_norm,
                iterations: iteration,
            });
        }
        if iteration == guards.max_iterations {
            return Err(stopped(iteration, &state));
        }

        // This strictly positive scale is held fixed for this entire line
        // search. It changes neither its minimizer nor the stopping criterion.
        let residual_scale = state.residual_norm;
        let mut normalized_residuals = zeros(n, meter)?;
        charge(meter, (2 * n) as u64, 0, 0)?;
        for (normalized, &residual) in normalized_residuals.iter_mut().zip(&state.residuals) {
            *normalized = residual / residual_scale;
        }
        let cost = normalized_cost(&state.residuals, residual_scale, meter)?;
        let scales = column_scales(system.jacobian_pattern(), &state.jacobian, n, meter)?;
        let gradient = normalized_gradient(
            system.jacobian_pattern(),
            &state.jacobian,
            &normalized_residuals,
            &scales,
            meter,
        )?;
        let model = LinearModel {
            pattern: system.jacobian_pattern(),
            jacobian: &state.jacobian,
            residuals: &normalized_residuals,
            scales: &scales,
        };
        let mut accepted = None;
        for damping in DAMPING_ATTEMPTS {
            let mut scaled_direction = match lsqr_scaled_direction(&model, damping, meter) {
                Ok(direction) => direction,
                Err(SystemConsumerError::NonFinite(_)) => continue,
                Err(error) => return Err(error),
            };
            let mut direction =
                match physical_direction(&scaled_direction, residual_scale, &scales, meter) {
                    Ok(direction) => direction,
                    Err(SystemConsumerError::NonFinite(_)) => continue,
                    Err(error) => return Err(error),
                };
            let norm = infinity_norm(&direction, meter)?;
            if norm > guards.max_step_norm {
                charge(meter, (2 * n + 1) as u64, 0, 0)?;
                let shrink = guards.max_step_norm / norm;
                for (physical, scaled) in direction.iter_mut().zip(&mut scaled_direction) {
                    *physical *= shrink;
                    *scaled *= shrink;
                }
            }
            // Both operands are in scaled coordinates. Forming J^T F first
            // would underflow for small but nonzero original residuals.
            let slope = match dot(&gradient, &scaled_direction, meter) {
                Ok(slope) => slope,
                Err(SystemConsumerError::NonFinite(_)) => continue,
                Err(error) => return Err(error),
            };
            if slope >= 0.0 {
                continue;
            }

            let mut alpha = 1.0;
            for _ in 0..MAX_BACKTRACKS {
                charge(meter, (4 * n + 16) as u64, 0, 0)?;
                for ((trial, original), delta) in trial_point.iter_mut().zip(&point).zip(&direction)
                {
                    *trial = original + alpha * delta;
                }
                if trial_point == point {
                    break;
                }
                if trial_point.iter().any(|value| !value.is_finite()) {
                    alpha *= 0.5;
                    continue;
                }
                let trial = match evaluate(system, &trial_point, n, meter) {
                    Ok(trial) => trial,
                    Err(error) if trial_domain_failure(&error) => {
                        alpha *= 0.5;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                // A satisfied numerical tolerance is useful even if an
                // approximate linear model predicted the merit inaccurately.
                // Publication still performs another original-source check.
                if trial.residual_norm <= guards.residual_tolerance {
                    accepted = Some(trial);
                    break;
                }
                let trial_cost = match normalized_cost(&trial.residuals, residual_scale, meter) {
                    Ok(cost) => cost,
                    Err(SystemConsumerError::NonFinite(_)) => {
                        alpha *= 0.5;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                let decrease =
                    finite_scalar(ARMIJO * alpha * slope, "sparse Newton Armijo decrease")?;
                let threshold = finite_scalar(cost + decrease, "sparse Newton Armijo threshold")?;
                let predicted = finite_scalar(-alpha * slope, "sparse Newton predicted decrease")?;
                let roundoff = 8.0 * f64::EPSILON * cost;
                let refinement = predicted <= roundoff
                    && trial_cost <= cost + roundoff
                    && trial.residual_norm < state.residual_norm;
                if (trial_cost < cost && trial_cost <= threshold) || refinement {
                    accepted = Some(trial);
                    break;
                }
                alpha *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }
        let Some(next) = accepted else {
            return Err(stopped(iteration, &state));
        };
        charge(meter, n as u64, 0, 0)?;
        point.copy_from_slice(&trial_point);
        state = next;
    }
    Err(stopped(guards.max_iterations, &state))
}

fn stopped(iterations: usize, state: &Evaluation) -> SystemConsumerError {
    SystemConsumerError::NoConvergence {
        iterations,
        residual_norm: state.residual_norm,
    }
}

fn evaluate<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    n: usize,
    meter: &mut M,
) -> Result<Evaluation, SystemConsumerError> {
    // Even an empty Jacobian pattern or a later zero direction cannot erase
    // any original residual or derivative domain check.
    let values = system.evaluate_sparse_with_meter(point, meter)?;
    if values.residuals.len() != n || values.jacobian.len() != system.jacobian_pattern().len() {
        return Err(SystemConsumerError::ConsumerContract(
            "sparse Newton evaluation dimensions",
        ));
    }
    finite(&values.residuals, "sparse Newton residuals", meter)?;
    finite(&values.jacobian, "sparse Newton Jacobian", meter)?;
    let residual_norm = infinity_norm(&values.residuals, meter)?;
    Ok(Evaluation {
        residuals: values.residuals,
        jacobian: values.jacobian,
        residual_norm,
    })
}

fn normalized_cost<M: BudgetMeter>(
    residuals: &[f64],
    scale: f64,
    meter: &mut M,
) -> Result<f64, SystemConsumerError> {
    charge(meter, (4 * residuals.len() + 1) as u64, 0, 0)?;
    let mut squared = 0.0;
    for &residual in residuals {
        let value = finite_scalar(residual / scale, "sparse Newton normalized residual")?;
        let term = finite_scalar(value * value, "sparse Newton normalized squared residual")?;
        squared = finite_scalar(squared + term, "sparse Newton normalized merit sum")?;
    }
    Ok(0.5 * squared)
}

fn normalized_gradient<M: BudgetMeter>(
    pattern: &[(usize, usize)],
    jacobian: &[f64],
    residuals: &[f64],
    scales: &[f64],
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    let mut result = zeros(scales.len(), meter)?;
    charge(meter, (6 * pattern.len()) as u64, 0, 0)?;
    for (&(row, column), &value) in pattern.iter().zip(jacobian) {
        let product = finite_scalar(
            (value / scales[column]) * residuals[row],
            "sparse Newton normalized gradient product",
        )?;
        result[column] = finite_scalar(
            result[column] + product,
            "sparse Newton normalized gradient",
        )?;
    }
    Ok(result)
}

fn physical_direction<M: BudgetMeter>(
    scaled: &[f64],
    residual_scale: f64,
    column_scales: &[f64],
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    let mut result = zeros(scaled.len(), meter)?;
    charge(meter, (64 * scaled.len()) as u64, 0, 0)?;
    for ((out, &value), &column_scale) in result.iter_mut().zip(scaled).zip(column_scales) {
        *out = scaled_product_quotient(value, residual_scale, column_scale)?;
    }
    Ok(result)
}

/// Decompose a positive, finite, nonzero binary64 value as mantissa * 2^exponent,
/// with mantissa in [1,2). Subnormal normalization is one exact power-of-two step.
fn binary_parts(value: f64) -> (f64, i32) {
    let (value, adjustment) = if value < f64::MIN_POSITIVE {
        (value * 4_503_599_627_370_496.0, -52)
    } else {
        (value, 0)
    };
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + adjustment;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    (mantissa, exponent)
}

/// Restore `value * numerator / denominator` without an intermediate overflowing
/// ratio or underflowing product. Final binary64 overflow still refuses; genuine
/// subnormal results receive ordinary gradual-underflow rounding.
fn scaled_product_quotient(
    value: f64,
    numerator: f64,
    denominator: f64,
) -> Result<f64, SystemConsumerError> {
    finite_scalar(value, "sparse Newton scaled direction")?;
    if !numerator.is_finite() || !denominator.is_finite() || numerator <= 0.0 || denominator <= 0.0
    {
        return Err(SystemConsumerError::ConsumerContract(
            "sparse Newton positive finite scales",
        ));
    }
    if value == 0.0 || numerator == denominator {
        return Ok(value);
    }
    let (left, left_exponent) = binary_parts(value.abs());
    let (top, top_exponent) = binary_parts(numerator);
    let (bottom, bottom_exponent) = binary_parts(denominator);
    let (mantissa, adjustment) = binary_parts((left * top) / bottom);
    let exponent = left_exponent + top_exponent - bottom_exponent + adjustment;
    let magnitude = if exponent > 1023 {
        return Err(SystemConsumerError::NonFinite(
            "sparse Newton physical direction",
        ));
    } else if exponent >= -1022 {
        mantissa * f64::from_bits(((exponent + 1023) as u64) << 52)
    } else if exponent >= -1075 {
        // Two exact normal powers keep the scale representable until the
        // final multiplication performs the one subnormal rounding.
        (mantissa * f64::MIN_POSITIVE) * f64::from_bits(((exponent + 2045) as u64) << 52)
    } else {
        0.0
    };
    finite_scalar(
        magnitude.copysign(value),
        "sparse Newton physical direction",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn power(exponent: i32) -> f64 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }

    #[test]
    fn physical_unscaling_cancels_extreme_ratios_before_rounding() {
        assert_eq!(
            scaled_product_quotient(power(-800), power(800), power(-800)).unwrap(),
            power(800)
        );
        assert_eq!(
            scaled_product_quotient(power(800), power(-800), power(800)).unwrap(),
            power(-800)
        );
        assert_eq!(
            scaled_product_quotient(-1.0, f64::MAX, f64::MAX).unwrap(),
            -1.0
        );
        assert_eq!(
            scaled_product_quotient(1.0, f64::from_bits(1), f64::from_bits(1)).unwrap(),
            1.0
        );
        assert_eq!(
            scaled_product_quotient(0.0, f64::MAX, f64::from_bits(1)).unwrap(),
            0.0
        );
        assert!(matches!(
            scaled_product_quotient(f64::MAX, 2.0, 1.0),
            Err(SystemConsumerError::NonFinite(_))
        ));
    }

    #[test]
    fn physical_unscaling_preserves_gradual_underflow_and_signed_zero() {
        assert_eq!(
            scaled_product_quotient(1.0, f64::from_bits(1), 2.0)
                .unwrap()
                .to_bits(),
            0
        );
        assert_eq!(
            scaled_product_quotient(1.5, f64::from_bits(1), 2.0)
                .unwrap()
                .to_bits(),
            1
        );
        assert_eq!(
            scaled_product_quotient(1.5, f64::MIN_POSITIVE, power(52))
                .unwrap()
                .to_bits(),
            2
        );
        assert_eq!(
            scaled_product_quotient(-power(-800), power(-800), 1.0)
                .unwrap()
                .to_bits(),
            (-0.0_f64).to_bits()
        );
    }
}
