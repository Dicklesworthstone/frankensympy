//! Shared bounded LSQR proposal kernel for checked sparse linearizations.
//!
//! The Golub--Kahan/plane-rotation recurrence follows Paige and Saunders,
//! ACM TOMS 8(1), 1982, section 4.1:
//! <https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf>.
//! The augmented operator `[J D^-1; damping I]` and right scaling follow
//! Algorithm 583, sections 1 and 5:
//! <https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82b.pdf>.
//! A truncated inner solve proposes a direction only. The nonlinear caller
//! must independently check its own original-source stopping criterion.

use super::{SystemConsumerError, charge, copy_values, zeros};
use fsym_budget::BudgetMeter;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;

const MAX_LSQR_ITERATIONS: usize = 512;
const LINEAR_RELATIVE_TOLERANCE: f64 = 1e-12;
pub(super) const DAMPING_ATTEMPTS: [f64; 6] = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0];

pub(super) fn trial_domain_failure(error: &SystemConsumerError) -> bool {
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

pub(super) fn finite_scalar(value: f64, label: &'static str) -> Result<f64, SystemConsumerError> {
    if !value.is_finite() {
        return Err(SystemConsumerError::NonFinite(label));
    }
    Ok(value)
}

pub(super) fn column_scales<M: BudgetMeter>(
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
pub(super) struct LinearModel<'a> {
    pub(super) pattern: &'a [(usize, usize)],
    pub(super) jacobian: &'a [f64],
    pub(super) residuals: &'a [f64],
    pub(super) scales: &'a [f64],
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

pub(super) fn dot<M: BudgetMeter>(
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

/// Return the physical direction, preserving the established least-squares
/// arithmetic and ledger. Original source values remain the caller's concern.
pub(super) fn lsqr_direction<M: BudgetMeter>(
    model: &LinearModel<'_>,
    damping: f64,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    lsqr_direction_impl(model, damping, true, meter)
}

/// Return the column-scaled solution before physical-coordinate division.
/// A root caller may normalize the RHS first and restore its scale without
/// forming a large norm or underflowing a squared residual.
pub(super) fn lsqr_scaled_direction<M: BudgetMeter>(
    model: &LinearModel<'_>,
    damping: f64,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    lsqr_direction_impl(model, damping, false, meter)
}

fn lsqr_direction_impl<M: BudgetMeter>(
    model: &LinearModel<'_>,
    damping: f64,
    physical_coordinates: bool,
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
    if physical_coordinates {
        charge(meter, (2 * n) as u64, 0, 0)?;
        for (value, &scale) in step.iter_mut().zip(model.scales) {
            *value = finite_scalar(*value / scale, "sparse LSQR unscaled direction")?;
        }
    }
    Ok(step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

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
