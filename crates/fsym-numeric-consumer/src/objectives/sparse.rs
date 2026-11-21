//! Bounded diagonally preconditioned CG for sparse Newton proposals.
//!
//! The operator is the symmetric part of the checked Hessian, equilibrated
//! by coordinate scales. It is applied directly from ordered sparse entries;
//! even a truncated solve is only a candidate direction. The caller checks
//! descent, original-source trial values and fresh final stationarity.

use super::{
    Evaluation, MAX_SHIFT_ATTEMPTS, ObjectiveConsumerError, charge, copy_values, dot,
    finite_scalar, infinity_norm, zeros,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::VerifiedObjective;

const MAX_CG_ITERATIONS: usize = 512;
const LINEAR_RELATIVE_TOLERANCE: f64 = 1e-12;

struct Model<'a> {
    pattern: &'a [(usize, usize)],
    values: &'a [f64],
    scales: Vec<f64>,
    diagonal: Vec<f64>,
    lower_bound: f64,
}

impl<'a> Model<'a> {
    fn new<M: BudgetMeter + ?Sized>(
        objective: &'a VerifiedObjective,
        state: &'a Evaluation,
        meter: &mut M,
    ) -> Result<Self, ObjectiveConsumerError> {
        let n = objective.variables().len();
        let pattern = objective.hessian_pattern();
        let values = &state.values.hessian;
        let mut scales = zeros(n, meter)?;
        charge(meter, (5 * pattern.len() + n + 1) as u64, 0, 0)?;
        for (&(row, column), &value) in pattern.iter().zip(values) {
            if row >= n || column >= n {
                return Err(ObjectiveConsumerError::ConsumerContract(
                    "sparse Hessian coordinates",
                ));
            }
            // Either orientation contributes to both rows of (H + H^T)/2.
            // Using the full entry magnitude avoids prematurely underflowing
            // a tiny mixed partial while constructing the equilibration.
            scales[row] = scales[row].max(value.abs());
            scales[column] = scales[column].max(value.abs());
        }
        for scale in &mut scales {
            *scale = if *scale == 0.0 { 1.0 } else { scale.sqrt() };
        }
        let mut diagonal = zeros(n, meter)?;
        let mut off_diagonal = zeros(n, meter)?;
        charge(meter, (8 * pattern.len() + n + 1) as u64, 0, 0)?;
        for (&(row, column), &value) in pattern.iter().zip(values) {
            let value = finite_scalar(
                (value / scales[row]) / scales[column],
                "equilibrated sparse Hessian",
            )?;
            if row == column {
                diagonal[row] = value;
            } else {
                let magnitude = 0.5 * value.abs();
                off_diagonal[row] =
                    finite_scalar(off_diagonal[row] + magnitude, "sparse Hessian row bound")?;
                off_diagonal[column] = finite_scalar(
                    off_diagonal[column] + magnitude,
                    "sparse Hessian column bound",
                )?;
            }
        }
        let mut lower_bound = f64::INFINITY;
        for (&diagonal, off_diagonal) in diagonal.iter().zip(off_diagonal) {
            lower_bound = lower_bound.min(finite_scalar(
                diagonal - off_diagonal,
                "sparse Hessian lower bound",
            )?);
        }
        Ok(Self {
            pattern,
            values,
            scales,
            diagonal,
            lower_bound,
        })
    }

    fn apply<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        shift: f64,
        output: &mut [f64],
        meter: &mut M,
    ) -> Result<(), ObjectiveConsumerError> {
        charge(
            meter,
            (12 * self.pattern.len() + 2 * point.len() + 1) as u64,
            0,
            0,
        )?;
        for (out, &coordinate) in output.iter_mut().zip(point) {
            *out = finite_scalar(shift * coordinate, "sparse Hessian shift")?;
        }
        for (&(row, column), &value) in self.pattern.iter().zip(self.values) {
            let scaled = (value / self.scales[row]) / self.scales[column];
            if row == column {
                output[row] = finite_scalar(
                    output[row] + scaled * point[row],
                    "sparse Hessian diagonal product",
                )?;
            } else {
                let half = 0.5 * scaled;
                output[row] = finite_scalar(
                    output[row] + half * point[column],
                    "sparse Hessian forward product",
                )?;
                output[column] = finite_scalar(
                    output[column] + half * point[row],
                    "sparse Hessian transposed product",
                )?;
            }
        }
        Ok(())
    }
}

fn conjugate_gradient<M: BudgetMeter + ?Sized>(
    model: &Model<'_>,
    rhs: &[f64],
    shift: f64,
    meter: &mut M,
) -> Result<Option<Vec<f64>>, ObjectiveConsumerError> {
    let n = rhs.len();
    let mut diagonal = zeros(n, meter)?;
    charge(meter, (2 * n + 1) as u64, 0, 0)?;
    for (entry, &original) in diagonal.iter_mut().zip(&model.diagonal) {
        let value = finite_scalar(original + shift, "sparse CG diagonal")?;
        if value < 0.0 {
            return Ok(None);
        }
        // An unused coordinate may have zero curvature and zero right-hand
        // side. Its unit preconditioner preserves that coordinate. A truly
        // singular nonzero direction is rejected by the curvature check.
        *entry = if value == 0.0 { 1.0 } else { value };
    }
    let rhs_norm = infinity_norm(rhs, meter)?;
    if rhs_norm == 0.0 {
        return Ok(None);
    }
    let mut normalized = copy_values(rhs, meter)?;
    charge(meter, n as u64, 0, 0)?;
    for value in &mut normalized {
        *value /= rhs_norm;
    }
    let mut solution = zeros(n, meter)?;
    let mut residual = copy_values(&normalized, meter)?;
    let mut preconditioned = zeros(n, meter)?;
    let mut direction = zeros(n, meter)?;
    let mut product = zeros(n, meter)?;
    let mut previous_rho = 0.0;
    let mut restart = true;
    let mut advanced = false;

    for iteration in 0..(8 * n).min(MAX_CG_ITERATIONS) {
        charge(meter, (6 * n + 1) as u64, 0, 0)?;
        for ((out, &value), &diagonal) in preconditioned.iter_mut().zip(&residual).zip(&diagonal) {
            *out = finite_scalar(value / diagonal, "sparse CG preconditioning")?;
        }
        let rho = dot(&residual, &preconditioned, meter)?;
        if rho <= 0.0 {
            break;
        }
        let beta = if restart { 0.0 } else { rho / previous_rho };
        for (direction, &value) in direction.iter_mut().zip(&preconditioned) {
            *direction = finite_scalar(value + beta * *direction, "sparse CG direction")?;
        }
        model.apply(&direction, shift, &mut product, meter)?;
        let curvature = dot(&direction, &product, meter)?;
        if curvature <= 0.0 {
            break;
        }
        let alpha = finite_scalar(rho / curvature, "sparse CG step")?;
        for index in 0..n {
            solution[index] = finite_scalar(
                solution[index] + alpha * direction[index],
                "sparse CG solution",
            )?;
            residual[index] = finite_scalar(
                residual[index] - alpha * product[index],
                "sparse CG residual",
            )?;
        }
        advanced = true;
        previous_rho = rho;
        restart = false;
        // Reconstruct the linear residual periodically and before accepting
        // inner convergence. Recurrence drift never supplies a stopping flag
        // for the outer nonlinear problem.
        if infinity_norm(&residual, meter)? <= LINEAR_RELATIVE_TOLERANCE
            || (iteration + 1) % 16 == 0
        {
            model.apply(&solution, shift, &mut product, meter)?;
            charge(meter, n as u64, 0, 0)?;
            for index in 0..n {
                residual[index] = finite_scalar(
                    normalized[index] - product[index],
                    "fresh sparse CG residual",
                )?;
            }
            if infinity_norm(&residual, meter)? <= LINEAR_RELATIVE_TOLERANCE {
                break;
            }
            restart = true;
        }
    }
    if !advanced {
        return Ok(None);
    }
    charge(meter, (2 * n + 1) as u64, 0, 0)?;
    for (value, &scale) in solution.iter_mut().zip(&model.scales) {
        *value = finite_scalar((*value * rhs_norm) / scale, "sparse Newton direction")?;
    }
    meter.checkpoint()?;
    Ok(Some(solution))
}

pub(super) fn descent_direction<M: BudgetMeter + ?Sized>(
    objective: &VerifiedObjective,
    state: &Evaluation,
    maximum_step: f64,
    regularizations: &mut usize,
    meter: &mut M,
) -> Result<Option<Vec<f64>>, ObjectiveConsumerError> {
    let model = Model::new(objective, state, meter)?;
    let n = objective.variables().len();
    let mut rhs = zeros(n, meter)?;
    charge(meter, (2 * n + 1) as u64, 0, 0)?;
    for ((target, &gradient), &scale) in rhs
        .iter_mut()
        .zip(&state.values.gradient)
        .zip(&model.scales)
    {
        *target = finite_scalar(-gradient / scale, "scaled sparse gradient")?;
    }
    let mut shift = 0.0;
    for attempt in 0..MAX_SHIFT_ATTEMPTS {
        charge(meter, 1, 0, 0)?;
        if attempt != 0 {
            *regularizations += 1;
        }
        if let Some(mut direction) = conjugate_gradient(&model, &rhs, shift, meter)? {
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
        shift = finite_scalar(
            if attempt == 0 {
                (1e-6 - model.lower_bound).max(1e-6)
            } else {
                shift * 10.0
            },
            "sparse Hessian regularization",
        )?;
    }
    Ok(None)
}
