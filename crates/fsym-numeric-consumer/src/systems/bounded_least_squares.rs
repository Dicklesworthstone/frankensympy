//! Sparse least squares with explicit coordinate bounds.
//!
//! Each linear proposal applies the existing bounded LSQR recurrence to the
//! checked Jacobian with currently blocked columns set to zero. The projected
//! line search uses the actual feasible displacement in the Armijo condition.
//! A projected steepest-descent direction is tried if the linear proposals do
//! not produce an admitted step. The projected-gradient step and sufficient
//! decrease condition follow the basic projection method in Bertsekas,
//! *SIAM J. Control Optim.* 20(2), 1982, equations (4) and (7):
//! <https://web.mit.edu/people/dimitrib/ProjectedNewton.pdf>.
//!
//! This bounded numerical implementation makes no convergence-rate, minimum,
//! root, uniqueness or completeness claim. A fresh original-source KKT residual
//! alone admits an approximate constrained-stationarity proposal. In particular,
//! a stationary maximum or saddle can satisfy that numerical condition.
//!
//! Every original residual and derivative is evaluated, including fixed and
//! active coordinates. One cumulative meter covers source evaluation, sparse
//! products, all trial directions, allocation, rejected trials and publication.
//! No dense Jacobian or normal matrix is constructed.

use super::sparse_linear::{
    DAMPING_ATTEMPTS, LinearModel, column_scales, dot, finite_scalar, lsqr_direction,
    trial_domain_failure,
};
use super::{
    LeastSquaresOptions, MAX_CONSUMER_NEWTON_ITERATIONS, MAX_SPARSE_LEAST_SQUARES_RESIDUALS,
    MAX_SPARSE_LEAST_SQUARES_VARIABLES, SystemConsumerError, charge, copy_values, finite,
    infinity_norm, point_dimension, publication_checkpoint, zeros,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_calculus::VerifiedResidualSystem;

const MAX_BACKTRACKS: usize = 32;
const ARMIJO: f64 = 1e-4;

/// An approximate box-constrained stationary point, without a minimum claim.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedLeastSquaresProposal {
    pub point: Vec<f64>,
    /// Original, unscaled residuals in source order, including constant rows.
    pub residuals: Vec<f64>,
    /// One half the original sum of squared residuals.
    pub cost: f64,
    /// Infinity norm of the original full gradient `J.transpose() * r`.
    pub gradient_norm: f64,
    /// Infinity norm after suppressing fixed coordinates and outward gradients
    /// at exact active bounds. Interior coordinates retain the full gradient.
    pub projected_gradient_norm: f64,
    pub iterations: usize,
    /// Original-source evaluations, including rejected trials and final replay.
    pub evaluations: usize,
}

struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Bounds {
    fn blocked(&self, column: usize, point: f64, gradient: f64) -> bool {
        self.lower[column] == self.upper[column]
            || (point == self.lower[column] && gradient >= 0.0)
            || (point == self.upper[column] && gradient <= 0.0)
    }
}

struct Evaluation {
    residuals: Vec<f64>,
    jacobian: Vec<f64>,
    gradient: Vec<f64>,
    projected_gradient: Vec<f64>,
    cost: f64,
    gradient_norm: f64,
    projected_gradient_norm: f64,
}

pub fn sparse_bounded_least_squares(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: &LeastSquaresOptions,
) -> Result<BoundedLeastSquaresProposal, SystemConsumerError> {
    sparse_bounded_least_squares_with_meter(system, initial, lower, upper, options, &mut Unbounded)
}

/// Find an approximate stationary point of `sum(residual_i^2) / 2` on a box.
///
/// Each endpoint vector has one entry per ordered source variable. A lower
/// endpoint may be negative infinity and an upper endpoint positive infinity;
/// NaN, the opposite infinities and reversed endpoints refuse. Equal finite
/// endpoints fix a coordinate. The finite initial point must already be feasible.
/// Neither admission nor a failure silently changes it.
///
/// Shapes are bounded to 1..=256 variables and 1..=4096 residuals, with at most
/// 1024 nonlinear iterations. Each iteration tries at most six damped LSQR
/// directions (512 inner iterations each), one projected-gradient direction,
/// and 32 backtracking trials per direction. All work remains charged after
/// rejection; operational errors are terminal. Cancellation is cooperative at
/// the existing sparse-product, vector, source-evaluation and publication stages.
pub fn sparse_bounded_least_squares_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: &LeastSquaresOptions,
    meter: &mut M,
) -> Result<BoundedLeastSquaresProposal, SystemConsumerError> {
    meter.checkpoint()?;
    let n = system.vars().len();
    let m = system.residuals().len();
    if !(1..=MAX_SPARSE_LEAST_SQUARES_VARIABLES).contains(&n)
        || !(1..=MAX_SPARSE_LEAST_SQUARES_RESIDUALS).contains(&m)
    {
        return Err(SystemConsumerError::InvalidRequest(
            "bounded sparse least squares requires 1..=256 variables and 1..=4096 residuals",
        ));
    }
    if options.max_iterations > MAX_CONSUMER_NEWTON_ITERATIONS {
        return Err(SystemConsumerError::InvalidRequest(
            "maximum bounded least-squares iterations exceeds 1024",
        ));
    }
    if !options.gradient_tolerance.is_finite() || options.gradient_tolerance < 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "bounded least-squares gradient tolerance must be finite and nonnegative",
        ));
    }
    if !options.max_step_norm.is_finite() || options.max_step_norm <= 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "bounded least-squares maximum step norm must be finite and positive",
        ));
    }
    point_dimension(initial, n, "bounded least-squares initial point")?;
    point_dimension(lower, n, "bounded least-squares lower bounds")?;
    point_dimension(upper, n, "bounded least-squares upper bounds")?;
    finite(initial, "bounded least-squares initial point", meter)?;
    charge(meter, (10 * n + 1) as u64, 0, 0)?;
    for ((&point, &lo), &hi) in initial.iter().zip(lower).zip(upper) {
        if lo.is_nan() || hi.is_nan() || lo == f64::INFINITY || hi == f64::NEG_INFINITY || lo > hi {
            return Err(SystemConsumerError::InvalidRequest(
                "bounded least-squares endpoints must be ordered, non-NaN and contain finite points",
            ));
        }
        if point < lo || point > hi {
            return Err(SystemConsumerError::InvalidRequest(
                "bounded least-squares initial point is outside its bounds",
            ));
        }
    }
    let bounds = Bounds {
        lower: copy_values(lower, meter)?,
        upper: copy_values(upper, meter)?,
    };
    let mut point = copy_values(initial, meter)?;
    let mut state = evaluate(system, &point, &bounds, meter)?;
    let mut evaluations = 1;
    let mut trial_point = zeros(n, meter)?;
    let mut displacement = zeros(n, meter)?;

    for iteration in 0..=options.max_iterations {
        if state.projected_gradient_norm <= options.gradient_tolerance {
            let checked = evaluate(system, &point, &bounds, meter)?;
            evaluations += 1;
            if checked.projected_gradient_norm > options.gradient_tolerance {
                return Err(SystemConsumerError::ConsumerContract(
                    "bounded least-squares final source gradient agreement",
                ));
            }
            publication_checkpoint(n + m, meter)?;
            return Ok(BoundedLeastSquaresProposal {
                point,
                residuals: checked.residuals,
                cost: checked.cost,
                gradient_norm: checked.gradient_norm,
                projected_gradient_norm: checked.projected_gradient_norm,
                iterations: iteration,
                evaluations,
            });
        }
        if iteration == options.max_iterations {
            return Err(stopped(iteration, &state));
        }

        // Mask only the proposed linear model. The original Jacobian, gradient
        // and every derivative domain remain present in source evaluation.
        let mut free_jacobian = copy_values(&state.jacobian, meter)?;
        charge(meter, (8 * free_jacobian.len() + 1) as u64, 0, 0)?;
        for (&(_, column), value) in system.jacobian_pattern().iter().zip(&mut free_jacobian) {
            if bounds.blocked(column, point[column], state.gradient[column]) {
                *value = 0.0;
            }
        }
        let scales = column_scales(system.jacobian_pattern(), &free_jacobian, n, meter)?;
        let model = LinearModel {
            pattern: system.jacobian_pattern(),
            jacobian: &free_jacobian,
            residuals: &state.residuals,
            scales: &scales,
        };
        let mut search = LineSearch {
            system,
            point: &point,
            bounds: &bounds,
            state: &state,
            options,
            trial_point: &mut trial_point,
            displacement: &mut displacement,
            evaluations: &mut evaluations,
        };
        let mut accepted = None;
        for damping in DAMPING_ATTEMPTS {
            let mut direction = match lsqr_direction(&model, damping, meter) {
                Ok(direction) => direction,
                Err(SystemConsumerError::NonFinite(_)) => continue,
                Err(error) => return Err(error),
            };
            accepted = search.try_direction(&mut direction, meter)?;
            if accepted.is_some() {
                break;
            }
        }
        if accepted.is_none() {
            // Projection of a coupled Gauss--Newton direction need not descend.
            // A separately chosen projected gradient direction keeps feasible
            // inward motion available, including arbitrarily near a boundary.
            let mut direction = copy_values(&state.projected_gradient, meter)?;
            charge(meter, n as u64, 0, 0)?;
            for value in &mut direction {
                *value = -*value;
            }
            accepted = search.try_direction(&mut direction, meter)?;
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

struct LineSearch<'a> {
    system: &'a VerifiedResidualSystem,
    point: &'a [f64],
    bounds: &'a Bounds,
    state: &'a Evaluation,
    options: &'a LeastSquaresOptions,
    trial_point: &'a mut [f64],
    displacement: &'a mut [f64],
    evaluations: &'a mut usize,
}

impl LineSearch<'_> {
    fn try_direction<M: BudgetMeter>(
        &mut self,
        direction: &mut [f64],
        meter: &mut M,
    ) -> Result<Option<Evaluation>, SystemConsumerError> {
        let n = self.point.len();
        let norm = infinity_norm(direction, meter)?;
        if norm > self.options.max_step_norm {
            charge(meter, (2 * n + 1) as u64, 0, 0)?;
            for value in direction.iter_mut() {
                // Dividing each component first avoids underflow in a shared
                // scale factor for very large gradients and small step limits.
                *value = (*value / norm) * self.options.max_step_norm;
            }
        }
        let mut alpha = 1.0;
        for _ in 0..MAX_BACKTRACKS {
            charge(meter, (12 * n + 32) as u64, 0, 0)?;
            for (column, ((target, original), delta)) in self
                .trial_point
                .iter_mut()
                .zip(self.point)
                .zip(direction.iter())
                .enumerate()
            {
                *target = (original + alpha * delta)
                    .clamp(self.bounds.lower[column], self.bounds.upper[column]);
                self.displacement[column] = *target - original;
            }
            if self.trial_point == self.point {
                break;
            }
            if self.trial_point.iter().any(|value| !value.is_finite())
                || self.displacement.iter().any(|value| !value.is_finite())
            {
                alpha *= 0.5;
                continue;
            }
            // Clipping changes direction. The unprojected directional
            // derivative cannot authorize a step along the feasible path.
            let slope = match dot(&self.state.gradient, self.displacement, meter) {
                Ok(slope) => slope,
                Err(SystemConsumerError::NonFinite(_)) => {
                    alpha *= 0.5;
                    continue;
                }
                Err(error) => return Err(error),
            };
            if slope >= 0.0 {
                alpha *= 0.5;
                continue;
            }
            *self.evaluations += 1;
            let trial = match evaluate(self.system, self.trial_point, self.bounds, meter) {
                Ok(trial) => trial,
                Err(error) if trial_domain_failure(&error) => {
                    alpha *= 0.5;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let threshold = finite_scalar(
                self.state.cost + ARMIJO * slope,
                "bounded least-squares Armijo threshold",
            )?;
            let roundoff = 8.0 * f64::EPSILON * self.state.cost;
            let refinement = -slope <= roundoff
                && trial.cost <= self.state.cost + roundoff
                && trial.projected_gradient_norm < self.state.projected_gradient_norm;
            let stationary = trial.projected_gradient_norm <= self.options.gradient_tolerance;
            if (trial.cost < self.state.cost && trial.cost <= threshold)
                || (stationary && trial.cost <= self.state.cost)
                || refinement
            {
                return Ok(Some(trial));
            }
            alpha *= 0.5;
        }
        Ok(None)
    }
}

fn stopped(iterations: usize, state: &Evaluation) -> SystemConsumerError {
    SystemConsumerError::BoundedLeastSquaresNoConvergence {
        iterations,
        projected_gradient_norm: state.projected_gradient_norm,
    }
}

fn evaluate<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    bounds: &Bounds,
    meter: &mut M,
) -> Result<Evaluation, SystemConsumerError> {
    let values = system.evaluate_sparse_with_meter(point, meter)?;
    let pattern = system.jacobian_pattern();
    let n = point.len();
    let m = system.residuals().len();
    if values.residuals.len() != m || values.jacobian.len() != pattern.len() {
        return Err(SystemConsumerError::ConsumerContract(
            "bounded least-squares evaluation dimensions",
        ));
    }
    finite(&values.residuals, "bounded least-squares residuals", meter)?;
    finite(&values.jacobian, "bounded least-squares Jacobian", meter)?;
    let mut gradient = zeros(n, meter)?;
    charge(meter, (5 * pattern.len() + 3 * m + 1) as u64, 0, 0)?;
    let mut sum_squared = 0.0;
    for residual in &values.residuals {
        let square = finite_scalar(
            residual * residual,
            "bounded least-squares squared residual",
        )?;
        sum_squared = finite_scalar(sum_squared + square, "bounded least-squares cost sum")?;
    }
    for (&(row, column), &value) in pattern.iter().zip(&values.jacobian) {
        if row >= m || column >= n {
            return Err(SystemConsumerError::ConsumerContract(
                "bounded least-squares Jacobian coordinate",
            ));
        }
        let product = finite_scalar(value * values.residuals[row], "bounded gradient product")?;
        gradient[column] = finite_scalar(gradient[column] + product, "bounded gradient sum")?;
    }
    let gradient_norm = infinity_norm(&gradient, meter)?;
    let mut projected_gradient = copy_values(&gradient, meter)?;
    charge(meter, (8 * n + 1) as u64, 0, 0)?;
    for (column, value) in projected_gradient.iter_mut().enumerate() {
        if bounds.blocked(column, point[column], *value) {
            *value = 0.0;
        }
    }
    let projected_gradient_norm = infinity_norm(&projected_gradient, meter)?;
    Ok(Evaluation {
        residuals: values.residuals,
        jacobian: values.jacobian,
        gradient,
        projected_gradient,
        cost: 0.5 * sum_squared,
        gradient_norm,
        projected_gradient_norm,
    })
}
