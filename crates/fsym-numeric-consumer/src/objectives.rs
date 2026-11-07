//! Bounded objective descent using checked symbolic gradients and Hessians.
//!
//! The pinned FrankenNumPy solver receives an equilibrated Hessian and gradient,
//! never symbolic objects. Singular or non-descent Newton systems trigger a
//! declared diagonal-shift strategy, followed by bounded Armijo backtracking.
//! Domain-invalid trial points may be rejected; operational failures, invalid
//! artifacts and allocation failures are terminal and retain all consumed work.
//!
//! Termination establishes only approximate gradient stationarity. In particular,
//! an initially stationary maximum or saddle is still just a numerical proposal.
//! Minimum, uniqueness and global claims require separate original-source
//! verification. No donor convergence flag or floating Hessian establishes them.
//!
//! One caller-owned meter covers admission, every rejected shift/trial, donor
//! buffers, fresh final evaluation and publication. Cancellation is cooperative
//! around the synchronous donor solve; it does not interrupt that kernel or
//! promise a wall-time bound. Compilation and proof admission precede this API.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::VerifiedObjective;
use fsym_calculus::compile::EvalError;
use fsym_calculus::sparse_hessian::{ObjectiveEvaluation, ObjectiveEvaluationError};
use std::mem::size_of;

pub const MAX_MINIMIZE_VARIABLES: usize = 16;
pub const MAX_MINIMIZE_ITERATIONS: usize = 1024;
const MAX_SHIFT_ATTEMPTS: usize = 8;
const MAX_BACKTRACKS: usize = 64;
const ARMIJO: f64 = 1e-4;

/// Controls for a finite-real, approximate stationary-point proposal.
#[derive(Debug, Clone)]
pub struct MinimizeOptions {
    /// Zero checks only whether the initial point already meets the tolerance.
    pub max_iterations: usize,
    /// Absolute infinity-norm tolerance on the freshly evaluated gradient.
    pub gradient_tolerance: f64,
    /// Clip each descent direction to this infinity norm before line search.
    pub max_step_norm: f64,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            gradient_tolerance: 1e-10,
            max_step_norm: 1e6,
        }
    }
}

/// Approximate stationarity only, with no minimum or existence certificate.
#[derive(Debug, Clone, PartialEq)]
pub struct MinimizationProposal {
    pub point: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub gradient_norm: f64,
    pub iterations: usize,
    /// Includes unsuccessful domain trials and the fresh final evaluation.
    pub evaluations: usize,
    /// Number of shifted donor systems attempted, including rejected attempts.
    pub regularizations: usize,
}

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum ObjectiveConsumerError {
    #[error("invalid objective consumer request: {0}")]
    InvalidRequest(&'static str),
    #[error("{input} has {actual} coordinates; expected {expected}")]
    Dimension {
        input: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{0} contains non-finite arithmetic")]
    NonFinite(&'static str),
    #[error("the pinned objective linear solver refused: {0}")]
    LinearSolve(String),
    #[error("the objective consumer exceeded its declared contract: {0}")]
    ConsumerContract(&'static str),
    #[error("no convergence after {iterations} iterations; gradient infinity-norm {gradient_norm}")]
    NoConvergence {
        iterations: usize,
        gradient_norm: f64,
    },
    #[error("checked objective evaluation refused: {0}")]
    Evaluation(#[from] ObjectiveEvaluationError),
    #[error("objective arithmetic refused: {0}")]
    Arithmetic(#[from] EvalError),
    #[error("objective consumer work stopped: {0}")]
    Meter(#[from] MeterError),
}

struct Evaluation {
    values: ObjectiveEvaluation,
    gradient_norm: f64,
}

pub fn minimize(
    objective: &VerifiedObjective,
    initial: &[f64],
    options: &MinimizeOptions,
) -> Result<MinimizationProposal, ObjectiveConsumerError> {
    minimize_with_meter(objective, initial, options, &mut Unbounded)
}

/// Minimize by checked objective descent and publish only a numerical proposal.
/// Every accepted step evaluates the original source and both derivative orders.
pub fn minimize_with_meter<M: BudgetMeter + ?Sized>(
    objective: &VerifiedObjective,
    initial: &[f64],
    options: &MinimizeOptions,
    meter: &mut M,
) -> Result<MinimizationProposal, ObjectiveConsumerError> {
    meter.checkpoint()?;
    let n = objective.variables().len();
    if !(1..=MAX_MINIMIZE_VARIABLES).contains(&n) {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "minimization requires 1..=16 variables",
        ));
    }
    if initial.len() != n {
        return Err(ObjectiveConsumerError::Dimension {
            input: "initial point",
            expected: n,
            actual: initial.len(),
        });
    }
    if options.max_iterations > MAX_MINIMIZE_ITERATIONS {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "maximum minimization iterations exceeds 1024",
        ));
    }
    if !options.gradient_tolerance.is_finite() || options.gradient_tolerance < 0.0 {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "gradient tolerance must be finite and nonnegative",
        ));
    }
    if !options.max_step_norm.is_finite() || options.max_step_norm <= 0.0 {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "maximum step norm must be finite and positive",
        ));
    }
    finite(initial, "initial point", meter)?;
    let mut point = copy_values(initial, meter)?;
    let mut state = evaluate(objective, &point, meter)?;
    let mut evaluations = 1;
    let mut regularizations = 0;
    let mut trial_point = zeros(n, meter)?;

    for iteration in 0..=options.max_iterations {
        if state.gradient_norm <= options.gradient_tolerance {
            // Acceptance of a floating proposal depends on the precise returned
            // coordinates, not stale trial state or a donor's success status.
            let checked = evaluate(objective, &point, meter)?;
            evaluations += 1;
            if checked.gradient_norm > options.gradient_tolerance {
                return Err(ObjectiveConsumerError::ConsumerContract(
                    "final source gradient agreement",
                ));
            }
            charge(meter, (2 * n + 8) as u64, 0, 0)?;
            meter.checkpoint()?;
            return Ok(MinimizationProposal {
                point,
                value: checked.values.value,
                gradient: checked.values.gradient,
                gradient_norm: checked.gradient_norm,
                iterations: iteration,
                evaluations,
                regularizations,
            });
        }
        if iteration == options.max_iterations {
            return Err(stopped(iteration, &state));
        }

        let Some(direction) = descent_direction(
            objective,
            &state,
            options.max_step_norm,
            &mut regularizations,
            meter,
        )?
        else {
            return Err(stopped(iteration, &state));
        };
        let slope = dot(&state.values.gradient, &direction, meter)?;
        if slope >= 0.0 {
            return Err(stopped(iteration, &state));
        }

        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..MAX_BACKTRACKS {
            charge(meter, (3 * n + 16) as u64, 0, 0)?;
            for ((target, original), delta) in trial_point.iter_mut().zip(&point).zip(&direction) {
                *target = original + alpha * delta;
            }
            if trial_point == point {
                break;
            }
            // Overflow in a proposed coordinate is a rejected trial, not a
            // reason to skip source-domain checks on any admitted coordinate.
            if trial_point.iter().any(|value| !value.is_finite()) {
                alpha *= 0.5;
                continue;
            }
            evaluations += 1;
            let trial = match evaluate(objective, &trial_point, meter) {
                Ok(trial) => trial,
                Err(error) if trial_domain_failure(&error) => {
                    alpha *= 0.5;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let decrease = finite_scalar(ARMIJO * alpha * slope, "Armijo decrease")?;
            let threshold = finite_scalar(state.values.value + decrease, "Armijo threshold")?;
            let predicted = finite_scalar(-alpha * slope, "predicted decrease")?;
            // An additive constant can conceal the last few objective changes
            // in binary64. Only a rounding-sized predicted change and a strictly
            // improving fresh gradient admit such a refinement. A plateau or
            // tiny step by itself never authorizes convergence.
            let value_scale = state.values.value.abs().max(trial.values.value.abs());
            let roundoff = 8.0 * f64::EPSILON * value_scale;
            let rounded_change = trial.values.value - state.values.value;
            let refinement = predicted <= roundoff
                && rounded_change.is_finite()
                && rounded_change <= roundoff
                && trial.gradient_norm < state.gradient_norm;
            let stationary = trial.gradient_norm <= options.gradient_tolerance;
            if (trial.values.value < state.values.value && trial.values.value <= threshold)
                || (stationary && trial.values.value <= state.values.value)
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

fn stopped(iterations: usize, state: &Evaluation) -> ObjectiveConsumerError {
    ObjectiveConsumerError::NoConvergence {
        iterations,
        gradient_norm: state.gradient_norm,
    }
}

fn evaluate<M: BudgetMeter + ?Sized>(
    objective: &VerifiedObjective,
    point: &[f64],
    meter: &mut M,
) -> Result<Evaluation, ObjectiveConsumerError> {
    let values = objective.evaluate_with_meter(point, meter)?;
    let n = objective.variables().len();
    if values.gradient.len() != n || values.hessian.len() != objective.hessian_pattern().len() {
        return Err(ObjectiveConsumerError::ConsumerContract(
            "checked objective output dimensions",
        ));
    }
    charge(
        meter,
        (1 + values.gradient.len() + values.hessian.len()) as u64,
        0,
        0,
    )?;
    finite_scalar(values.value, "objective value")?;
    if values
        .gradient
        .iter()
        .chain(&values.hessian)
        .any(|value| !value.is_finite())
    {
        return Err(ObjectiveConsumerError::NonFinite("objective derivatives"));
    }
    let gradient_norm = infinity_norm(&values.gradient, meter)?;
    Ok(Evaluation {
        values,
        gradient_norm,
    })
}

/// Reject only mathematical point-domain failures during line search. A nested
/// meter error, malformed program, dimension error or failed allocation must
/// propagate unchanged, even when a smaller step might otherwise succeed.
fn trial_domain_failure(error: &ObjectiveConsumerError) -> bool {
    matches!(
        error,
        ObjectiveConsumerError::Evaluation(ObjectiveEvaluationError::NonFinite)
            | ObjectiveConsumerError::Evaluation(ObjectiveEvaluationError::Evaluation(
                EvalError::NonFiniteValue
                    | EvalError::NonPositivePowerBase
                    | EvalError::NoMatchingPiecewiseBranch
            ))
            | ObjectiveConsumerError::NonFinite(_)
    )
}

fn descent_direction<M: BudgetMeter + ?Sized>(
    objective: &VerifiedObjective,
    state: &Evaluation,
    maximum_step: f64,
    regularizations: &mut usize,
    meter: &mut M,
) -> Result<Option<Vec<f64>>, ObjectiveConsumerError> {
    let n = objective.variables().len();
    let mut hessian = zeros(n * n, meter)?;
    charge(meter, objective.hessian_pattern().len() as u64, 0, 0)?;
    for (&(row, column), value) in objective
        .hessian_pattern()
        .iter()
        .zip(&state.values.hessian)
    {
        if row >= n || column >= n {
            return Err(ObjectiveConsumerError::ConsumerContract(
                "Hessian coordinates",
            ));
        }
        hessian[row * n + column] = *value;
    }
    // Independently checked mixed partials may round differently. Symmetrizing
    // here is only numerical candidate generation, never a new symbolic proof.
    charge(meter, (3 * n * n) as u64, 0, 0)?;
    for row in 0..n {
        for column in 0..row {
            let average = 0.5 * hessian[row * n + column] + 0.5 * hessian[column * n + row];
            hessian[row * n + column] = average;
            hessian[column * n + row] = average;
        }
    }
    let mut scales = zeros(n, meter)?;
    let mut rhs = zeros(n, meter)?;
    for row in 0..n {
        charge(meter, (n + 3) as u64, 0, 0)?;
        let magnitude = hessian[row * n..(row + 1) * n]
            .iter()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()));
        scales[row] = if magnitude == 0.0 {
            1.0
        } else {
            magnitude.sqrt()
        };
        rhs[row] = finite_scalar(-state.values.gradient[row] / scales[row], "scaled gradient")?;
    }
    let mut lower_bound = f64::INFINITY;
    charge(meter, (4 * n * n) as u64, 0, 0)?;
    for row in 0..n {
        let mut off_diagonal = 0.0;
        for column in 0..n {
            let value = finite_scalar(
                (hessian[row * n + column] / scales[row]) / scales[column],
                "scaled Hessian",
            )?;
            hessian[row * n + column] = value;
            if row != column {
                off_diagonal += value.abs();
            }
        }
        lower_bound = lower_bound.min(hessian[row * n + row] - off_diagonal);
    }
    let mut matrix = zeros(n * n, meter)?;
    let mut shift = 0.0;
    for attempt in 0..MAX_SHIFT_ATTEMPTS {
        charge(meter, (n * n + n) as u64, 0, 0)?;
        matrix.copy_from_slice(&hessian);
        for index in 0..n {
            matrix[index * n + index] += shift;
        }
        if attempt != 0 {
            *regularizations += 1;
        }
        let proposed = linear_solve(&matrix, &rhs, n, meter)?;
        if let Some(mut direction) = proposed {
            charge(meter, n as u64, 0, 0)?;
            for (value, scale) in direction.iter_mut().zip(&scales) {
                *value = finite_scalar(*value / scale, "Newton direction")?;
            }
            let norm = infinity_norm(&direction, meter)?;
            if norm > maximum_step {
                charge(meter, (n + 1) as u64, 0, 0)?;
                let shrink = maximum_step / norm;
                for value in &mut direction {
                    *value *= shrink;
                }
            }
            if dot(&state.values.gradient, &direction, meter)? < 0.0 {
                return Ok(Some(direction));
            }
        }
        // A positive Gershgorin lower bound makes the symmetric shifted matrix
        // strictly positive definite in exact arithmetic. Here it only proposes
        // a numerical system; the actual returned direction must still descend.
        shift = if attempt == 0 {
            (1e-6 - lower_bound).max(1e-6)
        } else {
            shift * 10.0
        };
    }
    Ok(None)
}

fn linear_solve<M: BudgetMeter + ?Sized>(
    matrix: &[f64],
    rhs: &[f64],
    n: usize,
    meter: &mut M,
) -> Result<Option<Vec<f64>>, ObjectiveConsumerError> {
    // This cap stays below the donor's parallel LU threshold. Account for LU,
    // permutation, solution, and bounded error formatting before entering it.
    charge(
        meter,
        (4 * n * n * n + 16 * n * n + 8 * n + 32) as u64,
        ((n * n + n) * size_of::<f64>() + n * size_of::<usize>() + 256) as u64,
        4,
    )?;
    let proposed = fnp_linalg::solve_nxn(matrix, rhs, n);
    meter.checkpoint()?;
    let solution = match proposed {
        Ok(solution) => solution,
        Err(fnp_linalg::LinAlgError::SolverSingularity) => return Ok(None),
        Err(error) => return Err(ObjectiveConsumerError::LinearSolve(error.to_string())),
    };
    if solution.len() != n {
        return Err(ObjectiveConsumerError::ConsumerContract(
            "linear solver result dimensions",
        ));
    }
    finite(&solution, "linear solver result", meter)?;
    Ok(Some(solution))
}

fn dot<M: BudgetMeter + ?Sized>(
    left: &[f64],
    right: &[f64],
    meter: &mut M,
) -> Result<f64, ObjectiveConsumerError> {
    charge(meter, (2 * left.len()) as u64, 0, 0)?;
    let mut result = 0.0;
    for (left, right) in left.iter().zip(right) {
        let product = finite_scalar(left * right, "descent product")?;
        result = finite_scalar(result + product, "descent sum")?;
    }
    Ok(result)
}

fn finite_scalar(value: f64, label: &'static str) -> Result<f64, ObjectiveConsumerError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ObjectiveConsumerError::NonFinite(label))
    }
}

fn finite<M: BudgetMeter + ?Sized>(
    values: &[f64],
    label: &'static str,
    meter: &mut M,
) -> Result<(), ObjectiveConsumerError> {
    charge(meter, values.len() as u64, 0, 0)?;
    for value in values {
        finite_scalar(*value, label)?;
    }
    Ok(())
}

fn infinity_norm<M: BudgetMeter + ?Sized>(
    values: &[f64],
    meter: &mut M,
) -> Result<f64, ObjectiveConsumerError> {
    charge(meter, values.len() as u64, 0, 0)?;
    Ok(values
        .iter()
        .fold(0.0_f64, |norm, value| norm.max(value.abs())))
}

fn zeros<M: BudgetMeter + ?Sized>(
    count: usize,
    meter: &mut M,
) -> Result<Vec<f64>, ObjectiveConsumerError> {
    let bytes = count
        .checked_mul(size_of::<f64>())
        .ok_or(EvalError::DimensionOverflow)?;
    charge(meter, count as u64, bytes as u64, u64::from(bytes != 0))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| EvalError::AllocationFailure)?;
    values.resize(count, 0.0);
    Ok(values)
}

fn copy_values<M: BudgetMeter + ?Sized>(
    values: &[f64],
    meter: &mut M,
) -> Result<Vec<f64>, ObjectiveConsumerError> {
    let mut copied = zeros(values.len(), meter)?;
    charge(meter, values.len() as u64, 0, 0)?;
    copied.copy_from_slice(values);
    Ok(copied)
}

fn charge<M: BudgetMeter + ?Sized>(
    meter: &mut M,
    steps: u64,
    bytes: u64,
    allocations: u64,
) -> Result<(), ObjectiveConsumerError> {
    meter.checkpoint()?;
    let mut charges = [(Dimension::ComputeSteps, 0); 3];
    let mut count = 0;
    for entry in [
        (Dimension::ComputeSteps, steps),
        (Dimension::MemoryBytes, bytes),
        (Dimension::AllocationCount, allocations),
    ] {
        if entry.1 != 0 {
            charges[count] = entry;
            count += 1;
        }
    }
    if count != 0 {
        meter.charge_batch(&charges[..count])?;
    }
    Ok(())
}
