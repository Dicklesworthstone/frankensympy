//! Bounded Newton proposals with an immutable suffix of fixed parameters.
//!
//! For m residuals, the first m ordered coordinates are unknowns and every
//! remaining coordinate is a fixed parameter. Evaluation keeps the original
//! checked program and all of its derivative-domain checks, including columns
//! belonging to parameters. Only the first m Jacobian columns enter the donor
//! solve. No source substitution, recompilation or parameter update occurs.
//!
//! Returned roots are approximate proposals at the supplied binary64 parameter
//! values. Neither a small residual nor a proposed parameter box establishes
//! existence, uniqueness or a root family. The independent original-source
//! parameter checker is the only acceptance boundary for a family certificate.

use super::{
    MAX_CONSUMER_VARIABLES, NewtonGuards, RootProposal, SystemConsumerError, charge,
    evaluate_system, finite, infinity_norm, linear_solve, point_dimension, publication_checkpoint,
    validate_guards, zeros,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_calculus::VerifiedResidualSystem;

fn shape<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    meter: &mut M,
) -> Result<(usize, usize), SystemConsumerError> {
    meter.checkpoint()?;
    let total = system.vars().len();
    let unknowns = system.residuals().len();
    if unknowns == 0 || unknowns >= total || total > MAX_CONSUMER_VARIABLES {
        return Err(SystemConsumerError::InvalidRequest(
            "parametric roots require 1 <= residual_count < total_variables <= 16",
        ));
    }
    Ok((unknowns, total))
}

fn point<M: BudgetMeter>(
    initial: &[f64],
    parameters: &[f64],
    unknowns: usize,
    total: usize,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    point_dimension(initial, unknowns, "unknown point")?;
    point_dimension(parameters, total - unknowns, "parameters")?;
    finite(initial, "unknown point", meter)?;
    finite(parameters, "parameters", meter)?;
    let mut point = zeros(total, meter)?;
    charge(meter, total as u64, 0, 0)?;
    point[..unknowns].copy_from_slice(initial);
    point[unknowns..].copy_from_slice(parameters);
    Ok(point)
}

fn evaluate<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    residuals: &mut [f64],
    full_jacobian: &mut [f64],
    unknown_jacobian: &mut [f64],
    meter: &mut M,
) -> Result<(), SystemConsumerError> {
    // This evaluator checks every stored derivative before the unknown-column
    // copy; a parameter-domain failure cannot disappear from the linear solve.
    evaluate_system(system, point, residuals, full_jacobian, meter)?;
    let unknowns = residuals.len();
    let total = point.len();
    charge(meter, (unknowns * unknowns) as u64, 0, 0)?;
    for row in 0..unknowns {
        unknown_jacobian[row * unknowns..(row + 1) * unknowns]
            .copy_from_slice(&full_jacobian[row * total..row * total + unknowns]);
    }
    Ok(())
}

/// Refine the unknown prefix while retaining the supplied parameter suffix.
pub fn newton_refine_parametric(
    system: &VerifiedResidualSystem,
    initial_unknowns: &[f64],
    parameters: &[f64],
    guards: &NewtonGuards,
) -> Result<RootProposal, SystemConsumerError> {
    newton_refine_parametric_with_meter(
        system,
        initial_unknowns,
        parameters,
        guards,
        &mut Unbounded,
    )
}

/// One cumulative ledger covers input ownership, full-source evaluation,
/// unknown-column extraction, every bounded donor solve and publication.
/// The returned vector contains only unknown coordinates, in their original
/// order. The final original residual is checked at those precise coordinates
/// and the unchanged fixed parameters. Singular Newton steps, invalid domains and
/// exhausted iteration/step/resource limits return no numerical proposal.
pub fn newton_refine_parametric_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial_unknowns: &[f64],
    parameters: &[f64],
    guards: &NewtonGuards,
    meter: &mut M,
) -> Result<RootProposal, SystemConsumerError> {
    let (unknowns, total) = shape(system, meter)?;
    validate_guards(guards)?;
    let mut point = point(initial_unknowns, parameters, unknowns, total, meter)?;
    let mut residuals = zeros(unknowns, meter)?;
    let mut full_jacobian = zeros(unknowns * total, meter)?;
    let mut unknown_jacobian = zeros(unknowns * unknowns, meter)?;
    let mut rhs = zeros(unknowns, meter)?;
    let mut residual_norm = f64::INFINITY;
    for iteration in 0..=guards.max_iterations {
        evaluate(
            system,
            &point,
            &mut residuals,
            &mut full_jacobian,
            &mut unknown_jacobian,
            meter,
        )?;
        residual_norm = infinity_norm(&residuals, meter)?;
        if residual_norm <= guards.residual_tolerance {
            publication_checkpoint(total, meter)?;
            point.truncate(unknowns);
            return Ok(RootProposal {
                root: point,
                residual_norm,
                iterations: iteration,
            });
        }
        if iteration == guards.max_iterations {
            break;
        }
        charge(meter, unknowns as u64, 0, 0)?;
        for (target, residual) in rhs.iter_mut().zip(&residuals) {
            *target = -*residual;
        }
        let step = linear_solve(&unknown_jacobian, &rhs, unknowns, meter)?;
        if infinity_norm(&step, meter)? > guards.max_step_norm {
            return Err(SystemConsumerError::NoConvergence {
                iterations: iteration,
                residual_norm,
            });
        }
        charge(meter, unknowns as u64, 0, 0)?;
        for (coordinate, delta) in point[..unknowns].iter_mut().zip(step) {
            *coordinate += delta;
        }
        finite(&point, "updated parametric iterate", meter)?;
    }
    Err(SystemConsumerError::NoConvergence {
        iterations: guards.max_iterations,
        residual_norm,
    })
}
