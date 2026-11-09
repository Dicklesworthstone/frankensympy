//! Approximate overdetermined least squares over a checked residual artifact.
//!
//! A stationary floating point proposal is neither a root certificate nor a
//! local/global minimum claim. Every original residual and stored derivative
//! is evaluated, including rows after the variable count and zero gradients.
//! Mathematical point-domain failures reject a line-search trial. Invalid
//! initial/final values, operational failures and malformed artifacts remain
//! terminal; there is no alternate solver or ledger reset.
//!
//! The bounded Gauss-Newton direction uses the pinned NumPy Householder QR
//! after column equilibration. Rank-deficient nonstationary systems refuse.
//! Armijo backtracking admits at most 32 trials per iteration. A rounding-scale
//! cost increase is allowed only when the fresh gradient strictly improves.
//! Cancellation is checked around the synchronous donor call and each bounded native stage;
//! it does not interrupt a donor kernel or promise a wall-time bound.

use super::{
    MAX_CONSUMER_NEWTON_ITERATIONS, MAX_CONSUMER_VARIABLES, SystemConsumerError, charge,
    copy_values, finite, infinity_norm, point_dimension, publication_checkpoint, zeros,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;
use std::mem::size_of;

/// Maximum residual rows admitted by the dense numerical least-squares lane.
pub const MAX_LEAST_SQUARES_RESIDUALS: usize = 64;
const MAX_BACKTRACKS: usize = 32;
const ARMIJO: f64 = 1e-4;

/// Controls for approximate stationarity of one half the squared residual norm.
#[derive(Debug, Clone)]
pub struct LeastSquaresOptions {
    pub max_iterations: usize,
    /// Absolute infinity-norm bound on the freshly evaluated J-transpose-r.
    pub gradient_tolerance: f64,
    /// Clip a proposed direction to this infinity norm before backtracking.
    pub max_step_norm: f64,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            gradient_tolerance: 1e-10,
            max_step_norm: 1e6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeastSquaresTermination {
    /// The floating point gradient met the requested absolute tolerance.
    GradientTolerance,
}

/// An approximate stationary point, without a minimum or root assertion.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresProposal {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    pub cost: f64,
    pub gradient_norm: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub termination: LeastSquaresTermination,
}

struct Evaluation {
    residuals: Vec<f64>,
    jacobian: Vec<f64>,
    gradient: Vec<f64>,
    cost: f64,
    gradient_norm: f64,
}

pub fn least_squares(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    options: &LeastSquaresOptions,
) -> Result<LeastSquaresProposal, SystemConsumerError> {
    least_squares_with_meter(system, initial, options, &mut Unbounded)
}

/// One cumulative meter covers admission, rejected trials, donor buffers and
/// final publication. Only mathematical trial-point refusals backtrack; they
/// cannot authorize success or discard the original source/derivative domains.
pub fn least_squares_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    options: &LeastSquaresOptions,
    meter: &mut M,
) -> Result<LeastSquaresProposal, SystemConsumerError> {
    meter.checkpoint()?;
    let n = system.vars().len();
    let m = system.residuals().len();
    if !(1..=MAX_CONSUMER_VARIABLES).contains(&n) || !(n..=MAX_LEAST_SQUARES_RESIDUALS).contains(&m)
    {
        return Err(SystemConsumerError::InvalidRequest(
            "least squares requires 1..=16 variables and variable_count..=64 residuals",
        ));
    }
    if options.max_iterations > MAX_CONSUMER_NEWTON_ITERATIONS {
        return Err(SystemConsumerError::InvalidRequest(
            "maximum least-squares iterations exceeds 1024",
        ));
    }
    if !options.gradient_tolerance.is_finite() || options.gradient_tolerance < 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "least-squares gradient tolerance must be finite and nonnegative",
        ));
    }
    if !options.max_step_norm.is_finite() || options.max_step_norm <= 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "least-squares maximum step norm must be finite and positive",
        ));
    }
    point_dimension(initial, n, "least-squares initial point")?;
    finite(initial, "least-squares initial point", meter)?;
    let mut point = copy_values(initial, meter)?;
    let mut state = evaluate(system, &point, m, n, meter)?;
    let mut evaluations = 1;
    let mut trial_point = zeros(n, meter)?;
    for iteration in 0..=options.max_iterations {
        if state.gradient_norm <= options.gradient_tolerance {
            // Re-evaluate the published coordinates, even after an accepted
            // trial. No donor metric or stale trial state authorizes output.
            let checked = evaluate(system, &point, m, n, meter)?;
            evaluations += 1;
            if checked.gradient_norm > options.gradient_tolerance {
                return Err(SystemConsumerError::ConsumerContract(
                    "least-squares final source gradient agreement",
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

        let mut direction = qr_direction(&state, m, n, meter)?;
        let norm = infinity_norm(&direction, meter)?;
        if norm > options.max_step_norm {
            charge(meter, n as u64 + 1, 0, 0)?;
            let scale = options.max_step_norm / norm;
            for value in &mut direction {
                *value *= scale;
            }
        }
        charge(meter, (2 * n + 1) as u64, 0, 0)?;
        let mut slope = 0.0;
        for (gradient, delta) in state.gradient.iter().zip(&direction) {
            let product = checked_finite(gradient * delta, "least-squares descent product")?;
            slope = checked_finite(slope + product, "least-squares descent sum")?;
        }
        if slope >= 0.0 {
            return Err(stopped(iteration, &state));
        }

        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..MAX_BACKTRACKS {
            charge(meter, (4 * n + 16) as u64, 0, 0)?;
            for ((target, original), delta) in trial_point.iter_mut().zip(&point).zip(&direction) {
                *target = original + alpha * delta;
            }
            if trial_point == point {
                return Err(stopped(iteration, &state));
            }
            // Both the accepted point and direction are finite. Their sum can
            // still overflow, so shrink this proposal before trying the source.
            // A finite trial must pass every original residual and derivative.
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
            let decrease = checked_finite(ARMIJO * alpha * slope, "least-squares Armijo decrease")?;
            let target_cost = state.cost + decrease;
            let threshold = checked_finite(target_cost, "least-squares Armijo threshold")?;
            // Near stationarity, squared residuals may round a few ulps
            // uphill. Permit refinement only at the objective's own
            // rounding scale and with a strictly improving fresh gradient.
            // A small step or cost plateau alone never establishes success.
            let roundoff = 8.0 * f64::EPSILON * state.cost;
            let predicted = checked_finite(-alpha * slope, "least-squares predicted decrease")?;
            let rounded_cost = state.cost + roundoff;
            let refinement = predicted <= roundoff
                && trial.cost <= rounded_cost
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
        let Some(next) = accepted else {
            return Err(stopped(iteration, &state));
        };
        charge(meter, n as u64, 0, 0)?;
        point.copy_from_slice(&trial_point);
        state = next;
    }
    Err(stopped(options.max_iterations, &state))
}

/// A failed proposal may lie outside a function's real domain or overflow even
/// though a shorter step is admissible. This is a deliberately narrow boundary:
/// nested meter errors, allocation failures, dimensions and invalid bytecode
/// stay terminal with their original typed cause and all prior work retained.
fn trial_domain_failure(error: &SystemConsumerError) -> bool {
    matches!(
        error,
        SystemConsumerError::NonFinite(_)
            | SystemConsumerError::Evaluation(
                EvalError::NonFiniteValue
                    | EvalError::NonPositivePowerBase
                    | EvalError::NoMatchingPiecewiseBranch
            )
            | SystemConsumerError::SparseEvaluation(ResidualEvaluationError::Evaluation(
                EvalError::NonFiniteValue
                    | EvalError::NonPositivePowerBase
                    | EvalError::NoMatchingPiecewiseBranch
            ))
    )
}

fn checked_finite(value: f64, label: &'static str) -> Result<f64, SystemConsumerError> {
    if !value.is_finite() {
        return Err(SystemConsumerError::NonFinite(label));
    }
    Ok(value)
}

fn stopped(iterations: usize, state: &Evaluation) -> SystemConsumerError {
    SystemConsumerError::NonlinearSolve(format!(
        "least squares did not reach stationarity after {iterations} iterations; gradient infinity-norm {}",
        state.gradient_norm
    ))
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
            "least-squares sparse evaluation dimensions",
        ));
    }
    finite(&values.residuals, "least-squares residuals", meter)?;
    finite(&values.jacobian, "least-squares Jacobian", meter)?;
    let mut jacobian = zeros(m * n, meter)?;
    let mut gradient = zeros(n, meter)?;
    charge(meter, (5 * pattern.len() + 3 * m + 1) as u64, 0, 0)?;
    let mut sum_squared = 0.0;
    for residual in &values.residuals {
        let square = checked_finite(residual * residual, "least-squares squared residual cost")?;
        sum_squared = checked_finite(sum_squared + square, "least-squares squared residual cost")?;
    }
    for (&(row, column), &value) in pattern.iter().zip(&values.jacobian) {
        if row >= m || column >= n {
            return Err(SystemConsumerError::ConsumerContract(
                "least-squares sparse Jacobian coordinate",
            ));
        }
        jacobian[row * n + column] = value;
        let product = value * values.residuals[row];
        let product = checked_finite(product, "least-squares gradient product")?;
        gradient[column] = checked_finite(
            gradient[column] + product,
            "least-squares gradient accumulation",
        )?;
    }
    let gradient_norm = infinity_norm(&gradient, meter)?;
    Ok(Evaluation {
        residuals: values.residuals,
        jacobian,
        gradient,
        cost: 0.5 * sum_squared,
        gradient_norm,
    })
}

fn qr_direction<M: BudgetMeter>(
    state: &Evaluation,
    m: usize,
    n: usize,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    let mut scales = zeros(n, meter)?;
    let mut matrix = zeros(m * n, meter)?;
    let mut rhs = zeros(m, meter)?;
    let mut step = zeros(n, meter)?;
    charge(meter, (4 * m * n + 2 * m + n) as u64, 0, 0)?;
    for row in state.jacobian.chunks_exact(n) {
        for (scale, value) in scales.iter_mut().zip(row) {
            *scale = scale.max(value.abs());
        }
    }
    if scales.contains(&0.0) {
        return Err(SystemConsumerError::LinearSolve(
            "least-squares Jacobian has a zero column".into(),
        ));
    }
    for row in 0..m {
        for column in 0..n {
            matrix[row * n + column] = state.jacobian[row * n + column] / scales[column];
        }
    }
    let residual_scale = infinity_norm(&state.residuals, meter)?;
    if residual_scale == 0.0 {
        return Err(SystemConsumerError::ConsumerContract(
            "nonzero least-squares gradient with zero residual",
        ));
    }
    for (target, residual) in rhs.iter_mut().zip(&state.residuals) {
        *target = -residual / residual_scale;
    }

    // The exact pinned qr_mxn path is synchronous, with bounded Householder
    // loops and three buffers (Q, R, reflector). This allowance also covers
    // its triangular fast path and bounded refusal formatting.
    charge(
        meter,
        (8 * m * m * n + 8 * m * n * n + 16 * m * n + 64) as u64,
        ((m * m + m * n + 2 * m) * size_of::<f64>() + 512) as u64,
        4,
    )?;
    let (q, r) = fnp_linalg::qr_mxn(&matrix, m, n)
        .map_err(|error| SystemConsumerError::LinearSolve(error.to_string()))?;
    meter.checkpoint()?;
    if q.len() != m * m || r.len() != m * n {
        return Err(SystemConsumerError::ConsumerContract(
            "least-squares QR result dimensions",
        ));
    }
    finite(&q, "least-squares Q factor", meter)?;
    finite(&r, "least-squares R factor", meter)?;
    charge(meter, (2 * m * n + 3 * n * n + 5 * n) as u64, 0, 0)?;
    let rank_threshold = 8.0 * f64::EPSILON * m.max(n) as f64;
    for column in (0..n).rev() {
        if r[column * n + column].abs() <= rank_threshold {
            return Err(SystemConsumerError::LinearSolve(
                "least-squares Jacobian is numerically rank deficient".into(),
            ));
        }
        let mut value = 0.0;
        for row in 0..m {
            let product = q[row * m + column] * rhs[row];
            let product = checked_finite(product, "least-squares Q-transpose-r")?;
            value = checked_finite(value + product, "least-squares Q-transpose-r sum")?;
        }
        for next in column + 1..n {
            let product = r[column * n + next] * step[next];
            let product = checked_finite(product, "least-squares triangular product")?;
            value = checked_finite(value - product, "least-squares triangular sum")?;
        }
        let value = value / r[column * n + column];
        step[column] = checked_finite(value, "least-squares scaled direction")?;
    }
    for (value, scale) in step.iter_mut().zip(scales) {
        *value = checked_finite((*value * residual_scale) / scale, "least-squares direction")?;
    }
    Ok(step)
}
