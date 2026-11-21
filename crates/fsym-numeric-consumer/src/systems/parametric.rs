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
    MAX_CONSUMER_VARIABLES, NewtonGuards, RootProposal, SystemConsumerError, buffer, charge,
    evaluate_system, exact_coordinate, finite, infinity_norm, linear_solve, point_dimension,
    publication_checkpoint, validate_guards, zeros,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{BigRational, ParametricRootBoxWitness, RealBall, RootBoxError};

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

fn rational_height<M: BudgetMeter>(
    value: &BigRational,
    meter: &mut M,
) -> Result<(), SystemConsumerError> {
    charge(meter, 1, 0, 0)?;
    if value.numer().limb_count() > fsym_core::MAX_ROOT_BOX_LIMBS
        || value.denom().limb_count() > fsym_core::MAX_ROOT_BOX_LIMBS
    {
        return Err(RootBoxError::ResourceLimit("parametric box rational height").into());
    }
    Ok(())
}

/// Propose a uniform root-family witness without granting a family claim.
///
/// The original program is evaluated at the supplied unknown approximation and
/// fixed binary64 parameters solely to propose an inverse of the unknown
/// Jacobian block. The parameter values need not equal or belong to the exact
/// parameter domain: even an inexact binary64 image of a fixed rational point
/// may propose a useful inverse. Acceptance must independently check the full
/// original source on the exact product of the returned unknown and parameter
/// boxes. No parameter midpoint, sample or compiled derivative supplies proof.
///
/// Unknown centers and inverse entries retain their exact binary64 values.
/// Unknown radii must be strictly positive exact rationals; parameter intervals
/// may have zero width. The complete exact parameter domain is copied unchanged.
/// Dimensions, rational heights and precision are bounded before solver work.
/// A coordinator can charge construction to ordinary resources and reserve its
/// protected verifier meter for independent acceptance. No refusal resets the
/// supplied ledger or requests protected verifier authority.
pub fn propose_parametric_root_box_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    approximation: &[f64],
    parameters: &[f64],
    radii: &[BigRational],
    parameter_domain: &[RealBall],
    precision_digits: u32,
    meter: &mut M,
) -> Result<ParametricRootBoxWitness, SystemConsumerError> {
    let (unknowns, total) = shape(system, meter)?;
    for (input, expected, actual) in [
        ("unknown radii", unknowns, radii.len()),
        ("parameter domain", total - unknowns, parameter_domain.len()),
    ] {
        if actual != expected {
            return Err(SystemConsumerError::Dimension {
                input,
                expected,
                actual,
            });
        }
    }
    if !(8..=fsym_core::MAX_ROOT_BOX_PRECISION).contains(&precision_digits) {
        return Err(SystemConsumerError::InvalidRequest(
            "precision must be 8..=128",
        ));
    }
    for radius in radii {
        rational_height(radius, meter)?;
        if radius.numer().is_zero() || radius.numer().is_negative() {
            return Err(SystemConsumerError::InvalidRequest(
                "unknown radii must be positive",
            ));
        }
    }
    for coordinate in parameter_domain {
        rational_height(coordinate.midpoint(), meter)?;
        rational_height(coordinate.radius(), meter)?;
    }
    let point = point(approximation, parameters, unknowns, total, meter)?;
    let mut residuals = zeros(unknowns, meter)?;
    let mut full_jacobian = zeros(unknowns * total, meter)?;
    let mut unknown_jacobian = zeros(unknowns * unknowns, meter)?;
    evaluate(
        system,
        &point,
        &mut residuals,
        &mut full_jacobian,
        &mut unknown_jacobian,
        meter,
    )?;
    let mut inverse = zeros(unknowns * unknowns, meter)?;
    let mut unit = zeros(unknowns, meter)?;
    for column in 0..unknowns {
        charge(meter, (2 * unknowns + 1) as u64, 0, 0)?;
        unit.fill(0.0);
        unit[column] = 1.0;
        let solution = linear_solve(&unknown_jacobian, &unit, unknowns, meter)?;
        for row in 0..unknowns {
            inverse[row * unknowns + column] = solution[row];
        }
    }

    let mut center = buffer(unknowns, meter)?;
    let mut unknown_domain = buffer(unknowns, meter)?;
    let mut preconditioner = buffer(unknowns, meter)?;
    for coordinate in 0..unknowns {
        center.push(exact_coordinate(approximation[coordinate], meter)?);
        let midpoint = exact_coordinate(approximation[coordinate], meter)?;
        let limbs = radii[coordinate].numer().limb_count() + radii[coordinate].denom().limb_count();
        charge(meter, limbs + 1, limbs * 8 + 128, 2)?;
        unknown_domain
            .push(RealBall::new(midpoint, radii[coordinate].clone()).map_err(RootBoxError::from)?);
        let mut row = buffer(unknowns, meter)?;
        for column in 0..unknowns {
            row.push(exact_coordinate(
                inverse[coordinate * unknowns + column],
                meter,
            )?);
        }
        preconditioner.push(row);
    }
    let mut exact_parameters = buffer(parameter_domain.len(), meter)?;
    for coordinate in parameter_domain {
        let limbs = coordinate.midpoint().numer().limb_count()
            + coordinate.midpoint().denom().limb_count()
            + coordinate.radius().numer().limb_count()
            + coordinate.radius().denom().limb_count();
        charge(meter, limbs + 1, limbs * 8 + 256, 4)?;
        exact_parameters.push(coordinate.clone());
    }
    publication_checkpoint(total, meter)?;
    Ok(ParametricRootBoxWitness {
        unknown_domain,
        parameter_domain: exact_parameters,
        center,
        preconditioner,
        precision_digits,
    })
}
