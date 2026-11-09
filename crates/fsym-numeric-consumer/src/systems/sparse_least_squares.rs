//! Sparse approximate least squares over an admitted residual artifact.
//!
//! Gauss--Newton directions use the Golub--Kahan/plane-rotation LSQR recurrence
//! of Paige and Saunders, *ACM TOMS* 8(1), 1982, section 4.1:
//! <https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf>.
//! Bounded regularization applies that recurrence to `[J D^-1; damping I]`,
//! following the augmented formulation and right preconditioning in Algorithm
//! 583, sections 1 and 5:
//! <https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82b.pdf>.
//! `D` contains column maximum magnitudes; zero columns
//! have scale one. This changes the direction, never the residual objective.
//!
//! Solver scratch is O(nnz + m + n); no dense Jacobian, normal matrix or Q
//! factor is built. The admitted artifact stores only original residual and
//! nonzero derivative programs. Rectangular and rank-deficient systems
//! are allowed. A truncated inner iteration is only a proposed direction, not
//! evidence of convergence. Only a freshly evaluated original J-transpose-r
//! authorizes a stationary floating proposal, with no root, minimum, uniqueness
//! or minimum-norm claim.
//!
//! All original residual and stored derivative domains are checked at every
//! source evaluation. Only mathematical trial refusals backtrack; operational
//! errors remain terminal. Each sparse product, vector stage, allocation,
//! rejected trial and publication uses the same cumulative caller meter.

use super::sparse_linear::{
    DAMPING_ATTEMPTS, LinearModel, column_scales, dot, finite_scalar, lsqr_direction,
    trial_domain_failure,
};
use super::{
    LeastSquaresOptions, LeastSquaresProposal, LeastSquaresTermination,
    MAX_CONSUMER_NEWTON_ITERATIONS, SystemConsumerError, charge, copy_values, finite,
    infinity_norm, point_dimension, publication_checkpoint, zeros,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_calculus::VerifiedResidualSystem;

pub const MAX_SPARSE_LEAST_SQUARES_VARIABLES: usize = 256;
pub const MAX_SPARSE_LEAST_SQUARES_RESIDUALS: usize = 4096;
const MAX_BACKTRACKS: usize = 32;
const ARMIJO: f64 = 1e-4;

struct Evaluation {
    residuals: Vec<f64>,
    jacobian: Vec<f64>,
    gradient: Vec<f64>,
    cost: f64,
    gradient_norm: f64,
}

pub fn sparse_least_squares(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    options: &LeastSquaresOptions,
) -> Result<LeastSquaresProposal, SystemConsumerError> {
    sparse_least_squares_with_meter(system, initial, options, &mut Unbounded)
}

/// Find an approximate stationary point of `sum(residual_i^2) / 2`.
///
/// At most 256 variables, 4096 residuals, 1024 nonlinear iterations, six
/// damping attempts per iteration, 512 LSQR steps per attempt and 32 Armijo
/// trials per direction are admitted. Initial/final source failures and every
/// operational failure are terminal. Work is never refunded between attempts.
pub fn sparse_least_squares_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    options: &LeastSquaresOptions,
    meter: &mut M,
) -> Result<LeastSquaresProposal, SystemConsumerError> {
    meter.checkpoint()?;
    let n = system.vars().len();
    let m = system.residuals().len();
    if !(1..=MAX_SPARSE_LEAST_SQUARES_VARIABLES).contains(&n)
        || !(1..=MAX_SPARSE_LEAST_SQUARES_RESIDUALS).contains(&m)
    {
        return Err(SystemConsumerError::InvalidRequest(
            "sparse least squares requires 1..=256 variables and 1..=4096 residuals",
        ));
    }
    if options.max_iterations > MAX_CONSUMER_NEWTON_ITERATIONS {
        return Err(SystemConsumerError::InvalidRequest(
            "maximum sparse least-squares iterations exceeds 1024",
        ));
    }
    if !options.gradient_tolerance.is_finite() || options.gradient_tolerance < 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "sparse least-squares gradient tolerance must be finite and nonnegative",
        ));
    }
    if !options.max_step_norm.is_finite() || options.max_step_norm <= 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "sparse least-squares maximum step norm must be finite and positive",
        ));
    }
    point_dimension(initial, n, "sparse least-squares initial point")?;
    finite(initial, "sparse least-squares initial point", meter)?;
    let mut point = copy_values(initial, meter)?;
    let mut state = evaluate(system, &point, m, n, meter)?;
    let mut evaluations = 1;
    let mut trial_point = zeros(n, meter)?;

    for iteration in 0..=options.max_iterations {
        if state.gradient_norm <= options.gradient_tolerance {
            let checked = evaluate(system, &point, m, n, meter)?;
            evaluations += 1;
            if checked.gradient_norm > options.gradient_tolerance {
                return Err(SystemConsumerError::ConsumerContract(
                    "sparse least-squares final source gradient agreement",
                ));
            }
            publication_checkpoint(n + m, meter)?;
            return Ok(LeastSquaresProposal {
                point,
                residuals: checked.residuals,
                cost: checked.cost,
                gradient_norm: checked.gradient_norm,
                iterations: iteration,
                evaluations,
                termination: LeastSquaresTermination::GradientTolerance,
            });
        }
        if iteration == options.max_iterations {
            return Err(stopped(iteration, &state));
        }

        let scales = column_scales(system.jacobian_pattern(), &state.jacobian, n, meter)?;
        let model = LinearModel {
            pattern: system.jacobian_pattern(),
            jacobian: &state.jacobian,
            residuals: &state.residuals,
            scales: &scales,
        };
        let mut accepted = None;
        for damping in DAMPING_ATTEMPTS {
            let mut direction = match lsqr_direction(&model, damping, meter) {
                Ok(direction) => direction,
                // A floating linear proposal may break down. Larger damping
                // can repair it; no source or operational refusal is hidden.
                Err(SystemConsumerError::NonFinite(_)) => continue,
                Err(error) => return Err(error),
            };
            let norm = infinity_norm(&direction, meter)?;
            if norm > options.max_step_norm {
                charge(meter, n as u64 + 1, 0, 0)?;
                let scale = options.max_step_norm / norm;
                for value in &mut direction {
                    *value *= scale;
                }
            }
            let slope = match dot(&state.gradient, &direction, meter) {
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
                for ((target, original), delta) in
                    trial_point.iter_mut().zip(&point).zip(&direction)
                {
                    *target = original + alpha * delta;
                }
                if trial_point == point {
                    break;
                }
                if trial_point.iter().any(|value| !value.is_finite()) {
                    alpha *= 0.5;
                    continue;
                }
                evaluations += 1;
                let trial = match evaluate(system, &trial_point, m, n, meter) {
                    Ok(trial) => trial,
                    Err(error) if trial_domain_failure(&error) => {
                        alpha *= 0.5;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                let decrease = finite_scalar(ARMIJO * alpha * slope, "sparse Armijo decrease")?;
                let threshold = finite_scalar(state.cost + decrease, "sparse Armijo threshold")?;
                let predicted = finite_scalar(-alpha * slope, "sparse predicted decrease")?;
                // A representational cost plateau is not convergence. Permit
                // rounding-scale refinement only with a strictly better fresh
                // original gradient; a tiny step alone never admits success.
                let roundoff = 8.0 * f64::EPSILON * state.cost;
                let refinement = predicted <= roundoff
                    && trial.cost <= state.cost + roundoff
                    && trial.gradient_norm < state.gradient_norm;
                let stationary = trial.gradient_norm <= options.gradient_tolerance;
                if (trial.cost < state.cost && trial.cost <= threshold)
                    || (stationary && trial.cost <= state.cost)
                    || refinement
                {
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
    Err(stopped(options.max_iterations, &state))
}

fn stopped(iterations: usize, state: &Evaluation) -> SystemConsumerError {
    SystemConsumerError::SparseLeastSquaresNoConvergence {
        iterations,
        gradient_norm: state.gradient_norm,
    }
}

fn evaluate<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    m: usize,
    n: usize,
    meter: &mut M,
) -> Result<Evaluation, SystemConsumerError> {
    let values = system.evaluate_sparse_with_meter(point, meter)?;
    let pattern = system.jacobian_pattern();
    if values.residuals.len() != m || values.jacobian.len() != pattern.len() {
        return Err(SystemConsumerError::ConsumerContract(
            "sparse least-squares evaluation dimensions",
        ));
    }
    finite(&values.residuals, "sparse least-squares residuals", meter)?;
    finite(&values.jacobian, "sparse least-squares Jacobian", meter)?;
    let mut gradient = zeros(n, meter)?;
    charge(meter, (5 * pattern.len() + 3 * m + 1) as u64, 0, 0)?;
    let mut sum_squared = 0.0;
    for residual in &values.residuals {
        let square = finite_scalar(residual * residual, "sparse least-squares squared residual")?;
        sum_squared = finite_scalar(sum_squared + square, "sparse least-squares cost sum")?;
    }
    for (&(row, column), &value) in pattern.iter().zip(&values.jacobian) {
        if row >= m || column >= n {
            return Err(SystemConsumerError::ConsumerContract(
                "sparse least-squares Jacobian coordinate",
            ));
        }
        let product = finite_scalar(value * values.residuals[row], "sparse gradient product")?;
        gradient[column] = finite_scalar(gradient[column] + product, "sparse gradient sum")?;
    }
    let gradient_norm = infinity_norm(&gradient, meter)?;
    Ok(Evaluation {
        residuals: values.residuals,
        jacobian: values.jacobian,
        gradient,
        cost: 0.5 * sum_squared,
        gradient_norm,
    })
}
