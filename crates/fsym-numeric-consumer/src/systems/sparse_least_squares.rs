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

use super::{
    LeastSquaresOptions, LeastSquaresProposal, LeastSquaresTermination,
    MAX_CONSUMER_NEWTON_ITERATIONS, SystemConsumerError, charge, copy_values, finite,
    infinity_norm, point_dimension, publication_checkpoint, zeros,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;

pub const MAX_SPARSE_LEAST_SQUARES_VARIABLES: usize = 256;
pub const MAX_SPARSE_LEAST_SQUARES_RESIDUALS: usize = 4096;
const MAX_LSQR_ITERATIONS: usize = 512;
const MAX_BACKTRACKS: usize = 32;
const DAMPING_ATTEMPTS: [f64; 6] = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0];
const LINEAR_RELATIVE_TOLERANCE: f64 = 1e-12;
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

fn finite_scalar(value: f64, label: &'static str) -> Result<f64, SystemConsumerError> {
    if !value.is_finite() {
        return Err(SystemConsumerError::NonFinite(label));
    }
    Ok(value)
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

fn column_scales<M: BudgetMeter>(
    pattern: &[(usize, usize)],
    jacobian: &[f64],
    n: usize,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    let mut scales = zeros(n, meter)?;
    charge(meter, (2 * pattern.len() + n) as u64, 0, 0)?;
    for (&(_, column), &value) in pattern.iter().zip(jacobian) {
        scales[column] = scales[column].max(value.abs());
    }
    for scale in &mut scales {
        if *scale == 0.0 {
            *scale = 1.0;
        }
    }
    Ok(scales)
}

/// One immutable linearization. Both products use the same ordered sparse
/// values and scaling. An unused column remains zero, rather than a rank error.
struct LinearModel<'a> {
    pattern: &'a [(usize, usize)],
    jacobian: &'a [f64],
    residuals: &'a [f64],
    scales: &'a [f64],
}

impl LinearModel<'_> {
    fn apply<M: BudgetMeter>(
        &self,
        x: &[f64],
        damping: f64,
        out: &mut [f64],
        meter: &mut M,
    ) -> Result<(), SystemConsumerError> {
        charge(
            meter,
            (6 * self.pattern.len() + out.len() + 2 * x.len() + 1) as u64,
            0,
            0,
        )?;
        out.fill(0.0);
        for (&(row, column), &value) in self.pattern.iter().zip(self.jacobian) {
            let product = finite_scalar(
                (value / self.scales[column]) * x[column],
                "sparse LSQR forward product",
            )?;
            out[row] = finite_scalar(out[row] + product, "sparse LSQR forward sum")?;
        }
        for (out, &value) in out[self.residuals.len()..].iter_mut().zip(x) {
            *out = finite_scalar(damping * value, "sparse LSQR damping product")?;
        }
        Ok(())
    }

    fn transpose<M: BudgetMeter>(
        &self,
        y: &[f64],
        damping: f64,
        out: &mut [f64],
        meter: &mut M,
    ) -> Result<(), SystemConsumerError> {
        charge(
            meter,
            (6 * self.pattern.len() + 2 * out.len() + 1) as u64,
            0,
            0,
        )?;
        for (out, &value) in out.iter_mut().zip(&y[self.residuals.len()..]) {
            *out = finite_scalar(damping * value, "sparse LSQR transposed damping")?;
        }
        for (&(row, column), &value) in self.pattern.iter().zip(self.jacobian) {
            let product = finite_scalar(
                (value / self.scales[column]) * y[row],
                "sparse LSQR transposed product",
            )?;
            out[column] = finite_scalar(out[column] + product, "sparse LSQR transposed sum")?;
        }
        Ok(())
    }
}

fn norm2<M: BudgetMeter>(values: &[f64], meter: &mut M) -> Result<f64, SystemConsumerError> {
    // hypot retains tiny magnitudes and avoids overflowing intermediate squares.
    charge(meter, (8 * values.len() + 1) as u64, 0, 0)?;
    let mut norm = 0.0_f64;
    for &value in values {
        norm = finite_scalar(norm.hypot(value), "sparse LSQR vector norm")?;
    }
    Ok(norm)
}

fn normalize<M: BudgetMeter>(
    values: &mut [f64],
    norm: f64,
    meter: &mut M,
) -> Result<(), SystemConsumerError> {
    charge(meter, values.len() as u64 + 1, 0, 0)?;
    if norm > 0.0 {
        for value in values {
            // Divide directly: reciprocal formation can overflow at a tiny norm.
            *value /= norm;
        }
    }
    Ok(())
}

fn dot<M: BudgetMeter>(
    left: &[f64],
    right: &[f64],
    meter: &mut M,
) -> Result<f64, SystemConsumerError> {
    charge(meter, (3 * left.len() + 1) as u64, 0, 0)?;
    let mut sum = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let product = finite_scalar(left * right, "sparse least-squares descent product")?;
        sum = finite_scalar(sum + product, "sparse least-squares descent sum")?;
    }
    Ok(sum)
}

fn lsqr_direction<M: BudgetMeter>(
    model: &LinearModel<'_>,
    damping: f64,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    let n = model.scales.len();
    let m = model.residuals.len();
    let mut u = zeros(m + n, meter)?;
    charge(meter, m as u64, 0, 0)?;
    for (out, &residual) in u.iter_mut().zip(model.residuals) {
        *out = -residual;
    }
    let beta = norm2(&u, meter)?;
    normalize(&mut u, beta, meter)?;
    let mut v = zeros(n, meter)?;
    model.transpose(&u, damping, &mut v, meter)?;
    let mut alpha = norm2(&v, meter)?;
    normalize(&mut v, alpha, meter)?;
    let mut w = copy_values(&v, meter)?;
    let mut step = zeros(n, meter)?;
    let mut next_u = zeros(m + n, meter)?;
    let mut next_v = zeros(n, meter)?;
    let mut rho_bar = alpha;
    let mut phi_bar = beta;
    let initial_normal = finite_scalar(alpha * beta, "sparse LSQR initial gradient")?;
    if initial_normal == 0.0 {
        return Ok(step);
    }

    for _ in 0..MAX_LSQR_ITERATIONS {
        // Golub--Kahan bidiagonalization of the augmented sparse operator.
        model.apply(&v, damping, &mut next_u, meter)?;
        charge(meter, (3 * (m + n)) as u64, 0, 0)?;
        for (next, &previous) in next_u.iter_mut().zip(&u) {
            *next = finite_scalar(*next - alpha * previous, "sparse LSQR left recurrence")?;
        }
        let beta = norm2(&next_u, meter)?;
        normalize(&mut next_u, beta, meter)?;
        std::mem::swap(&mut u, &mut next_u);
        model.transpose(&u, damping, &mut next_v, meter)?;
        charge(meter, (3 * n) as u64, 0, 0)?;
        for (next, &previous) in next_v.iter_mut().zip(&v) {
            *next = finite_scalar(*next - beta * previous, "sparse LSQR right recurrence")?;
        }
        alpha = norm2(&next_v, meter)?;
        normalize(&mut next_v, alpha, meter)?;
        std::mem::swap(&mut v, &mut next_v);

        // Eliminate the next bidiagonal entry by a stable plane rotation, then
        // update only the solution and the latest search vector (no basis/Q).
        charge(meter, (8 * n + 24) as u64, 0, 0)?;
        let rho = finite_scalar(rho_bar.hypot(beta), "sparse LSQR rotation norm")?;
        if rho == 0.0 {
            break;
        }
        let cosine = rho_bar / rho;
        let sine = beta / rho;
        let theta = sine * alpha;
        rho_bar = -cosine * alpha;
        let phi = cosine * phi_bar;
        phi_bar *= sine;
        let update = finite_scalar(phi / rho, "sparse LSQR solution coefficient")?;
        let recurrence = finite_scalar(theta / rho, "sparse LSQR search coefficient")?;
        for ((value, search), &next) in step.iter_mut().zip(&mut w).zip(&v) {
            let change = finite_scalar(update * *search, "sparse LSQR solution product")?;
            *value = finite_scalar(*value + change, "sparse LSQR solution sum")?;
            *search = finite_scalar(next - recurrence * *search, "sparse LSQR search recurrence")?;
        }

        // Recompute the linear normal residual instead of relying on estimated
        // recurrence norms after loss of orthogonality. This stops only the
        // inner proposal; the original nonlinear gradient is checked separately.
        model.apply(&step, damping, &mut next_u, meter)?;
        charge(meter, (2 * m + 1) as u64, 0, 0)?;
        for (value, &residual) in next_u.iter_mut().zip(model.residuals) {
            *value = finite_scalar(*value + residual, "sparse LSQR linear residual")?;
        }
        model.transpose(&next_u, damping, &mut next_v, meter)?;
        let normal = norm2(&next_v, meter)?;
        if normal <= LINEAR_RELATIVE_TOLERANCE * initial_normal || alpha == 0.0 || beta == 0.0 {
            break;
        }
    }
    charge(meter, (2 * n) as u64, 0, 0)?;
    for (value, &scale) in step.iter_mut().zip(model.scales) {
        *value = finite_scalar(*value / scale, "sparse LSQR unscaled direction")?;
    }
    Ok(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected}"
        );
    }

    #[test]
    fn augmented_products_share_exact_coordinates_scaling_and_damping() {
        let model = LinearModel {
            pattern: &[(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
            jacobian: &[2.0, -4.0, 3.0, 1.0, 2.0],
            residuals: &[0.0; 3],
            scales: &[2.0, 3.0, 4.0],
        };
        let mut product = [0.0; 6];
        model
            .apply(&[2.0, -1.0, 3.0], 0.25, &mut product, &mut Unbounded)
            .unwrap();
        assert_eq!(product, [-1.0, -1.0, 2.5, 0.5, -0.25, 0.75]);
        let mut transpose = [0.0; 3];
        model
            .transpose(
                &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                0.25,
                &mut transpose,
                &mut Unbounded,
            )
            .unwrap();
        assert_eq!(transpose, [5.25, 4.5, 1.75]);
    }

    #[test]
    fn augmented_lsqr_damping_matches_the_exact_two_column_solution() {
        let model = LinearModel {
            pattern: &[(0, 0), (0, 1)],
            jacobian: &[1.0, 2.0],
            residuals: &[-3.0],
            scales: &[1.0, 2.0],
        };
        // A = [1,1]. The scaled solution is (3,3)/(2 + damping^2),
        // and the physical second coordinate is divided by its scale two.
        for damping in [0.0, 1.0, 100.0] {
            let step = lsqr_direction(&model, damping, &mut Unbounded).unwrap();
            near(step[0], 3.0 / (2.0 + damping * damping), 1e-12);
            near(step[1], 1.5 / (2.0 + damping * damping), 1e-12);
        }
    }

    #[test]
    fn inconsistent_linear_model_reaches_the_full_normal_equations() {
        let model = LinearModel {
            pattern: &[(0, 0), (1, 1), (2, 0), (2, 1)],
            jacobian: &[1.0; 4],
            residuals: &[-1.0, -1.0, -4.0],
            scales: &[1.0; 2],
        };
        let step = lsqr_direction(&model, 0.0, &mut Unbounded).unwrap();
        near(step[0], 5.0 / 3.0, 1e-12);
        near(step[1], 5.0 / 3.0, 1e-12);
        near(2.0 * step[0] + step[1], 5.0, 1e-12);
        near(step[0] + 2.0 * step[1], 5.0, 1e-12);
    }

    #[test]
    fn norm_normalization_does_not_square_tiny_values_to_zero() {
        let mut values = [1e-200, -1e-200];
        let norm = norm2(&values, &mut Unbounded).unwrap();
        assert!(norm > 0.0);
        normalize(&mut values, norm, &mut Unbounded).unwrap();
        near(values[0], 1.0 / 2.0_f64.sqrt(), 1e-15);
        near(values[1], -1.0 / 2.0_f64.sqrt(), 1e-15);
    }
}
