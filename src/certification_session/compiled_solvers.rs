//! Bounded numerical proposals from resident, independently admitted programs.
//!
//! These drivers select an existing consumer without recompiling the source or
//! replacing a refused backend. Controls, input conversion, unsuccessful trials
//! and native consumer work share the session's ordinary cumulative budget.
//! Returned proposals remain approximate; root and extremum certification are
//! separate original-source operations.

use super::compiled::numeric_values;
use super::compiled_objective;
use super::{CertificationSessionError, charge, parse_scalar};
use fsym_budget::{Budget, BudgetMeter};
use fsym_calculus::{VerifiedObjective, VerifiedResidualSystem};
use fsym_core::{BigRational, RootBoxWitness};
use fsym_numeric_consumer::{
    LeastSquaresOptions, LeastSquaresProposal, MAX_CONSUMER_NEWTON_ITERATIONS,
    MAX_CONSUMER_VARIABLES, MAX_LEAST_SQUARES_RESIDUALS, MAX_MINIMIZE_ITERATIONS,
    MAX_MINIMIZE_VARIABLES, MAX_SPARSE_LEAST_SQUARES_RESIDUALS, MAX_SPARSE_LEAST_SQUARES_VARIABLES,
    MAX_SPARSE_NEWTON_VARIABLES, MinimizationProposal, MinimizeOptions, NewtonGuards,
    ObjectiveConsumerError, RootProposal, SystemConsumerError,
    frankenscipy_analytic_system_with_meter, frankenscipy_fsolve_system_with_meter,
    least_squares_with_meter, minimize_with_meter, newton_refine_with_meter,
    propose_root_box_with_meter, propose_stationary_box_with_meter,
    sparse_least_squares_with_meter, sparse_newton_refine_with_meter,
};
use serde::{Deserialize, Serialize};

/// An explicit root-proposal backend; failure never selects another backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompiledRootBackend {
    #[serde(rename = "frankennumpy")]
    FrankenNumpy,
    #[serde(rename = "frankenscipy")]
    FrankenScipy,
    #[serde(rename = "frankenscipy_analytic")]
    FrankenScipyAnalytic,
    #[serde(rename = "sparse")]
    Sparse,
}

impl CompiledRootBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FrankenNumpy => "frankennumpy",
            Self::FrankenScipy => "frankenscipy",
            Self::FrankenScipyAnalytic => "frankenscipy_analytic",
            Self::Sparse => "sparse",
        }
    }
}

/// An explicit dense or sparse least-squares proposal backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompiledLeastSquaresBackend {
    #[serde(rename = "dense")]
    Dense,
    #[serde(rename = "sparse")]
    Sparse,
}

impl CompiledLeastSquaresBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
        }
    }
}

/// Optional controls for approximate residual convergence.
///
/// Omitted fields use the selected native consumer's defaults. The pinned
/// FrankenSciPy backends have fixed controls and refuse every supplied field.
/// Numerical strings use the same bounded finite-binary64 parser as points.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompiledRootSolveOptions {
    pub max_iterations: Option<usize>,
    pub residual_tolerance: Option<String>,
    pub max_step_norm: Option<String>,
}

/// Optional controls for approximate gradient stationarity.
///
/// These controls apply to least squares and objective descent. Reaching the
/// gradient tolerance is not a root, local minimum, or global minimum proof.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompiledStationarityOptions {
    pub max_iterations: Option<usize>,
    pub gradient_tolerance: Option<String>,
    pub max_step_norm: Option<String>,
}

/// Construct a candidate using ordinary resources. Only the caller's separate
/// original-source verification may turn this witness into an accepted root.
pub(super) fn propose_root_witness(
    system: &VerifiedResidualSystem,
    initial: &[String],
    backend: CompiledRootBackend,
    options: &CompiledRootSolveOptions,
    radii: &[String],
    precision_digits: u32,
    budget: &mut Budget,
) -> Result<RootBoxWitness, CertificationSessionError> {
    budget.checkpoint()?;
    let variables = system.num_vars();
    let residuals = system.num_residuals();
    // The exact checker retains its own bound even when the selected numerical
    // backend can solve a larger sparse system.
    if !(1..=MAX_CONSUMER_VARIABLES).contains(&variables) || residuals != variables {
        return Err(SystemConsumerError::UnsupportedShape {
            variables,
            residuals,
        }
        .into());
    }
    if radii.len() != variables {
        return Err(SystemConsumerError::Dimension {
            input: "radii",
            expected: variables,
            actual: radii.len(),
        }
        .into());
    }
    if !(8..=fsym_core::MAX_ROOT_BOX_PRECISION).contains(&precision_digits) {
        return Err(SystemConsumerError::InvalidRequest("precision must be 8..=128").into());
    }
    charge(
        budget,
        variables,
        variables * std::mem::size_of::<BigRational>(),
    )?;
    let mut exact_radii = Vec::new();
    exact_radii
        .try_reserve_exact(variables)
        .map_err(|_| crate::certification::CertificationError::AllocationFailure)?;
    for radius in radii {
        let radius = parse_scalar(radius, budget)?;
        if radius.numer().is_negative() || radius.numer().is_zero() {
            return Err(SystemConsumerError::InvalidRequest("radii must be positive").into());
        }
        exact_radii.push(radius);
    }
    let proposal = solve_root(system, initial, backend, options, budget)?;
    Ok(propose_root_box_with_meter(
        system,
        &proposal.root,
        &exact_radii,
        precision_digits,
        budget,
    )?)
}

pub(super) fn solve_root(
    system: &VerifiedResidualSystem,
    initial: &[String],
    backend: CompiledRootBackend,
    options: &CompiledRootSolveOptions,
    budget: &mut Budget,
) -> Result<RootProposal, CertificationSessionError> {
    budget.checkpoint()?;
    if matches!(
        backend,
        CompiledRootBackend::FrankenScipy | CompiledRootBackend::FrankenScipyAnalytic
    ) && (options.max_iterations.is_some()
        || options.residual_tolerance.is_some()
        || options.max_step_norm.is_some())
    {
        return Err(SystemConsumerError::InvalidRequest(
            "FrankenSciPy has fixed solver controls; omit max_iterations, residual_tolerance and max_step_norm",
        )
        .into());
    }
    let variables = system.num_vars();
    let residuals = system.num_residuals();
    if backend == CompiledRootBackend::Sparse {
        if !(1..=MAX_SPARSE_NEWTON_VARIABLES).contains(&variables) || residuals != variables {
            return Err(SystemConsumerError::UnsupportedSparseRootShape {
                variables,
                residuals,
            }
            .into());
        }
    } else if !(1..=MAX_CONSUMER_VARIABLES).contains(&variables) || residuals != variables {
        return Err(SystemConsumerError::UnsupportedShape {
            variables,
            residuals,
        }
        .into());
    }
    let mut guards = NewtonGuards::default();
    if let Some(iterations) = options.max_iterations {
        if iterations > MAX_CONSUMER_NEWTON_ITERATIONS {
            return Err(SystemConsumerError::InvalidRequest(
                "maximum Newton iterations exceeds 1024",
            )
            .into());
        }
        guards.max_iterations = iterations;
    }
    guards.residual_tolerance = numeric_option(
        options.residual_tolerance.as_ref(),
        guards.residual_tolerance,
        "residual_tolerance",
        budget,
    )?;
    if guards.residual_tolerance < 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "residual tolerance must be finite and nonnegative",
        )
        .into());
    }
    guards.max_step_norm = numeric_option(
        options.max_step_norm.as_ref(),
        guards.max_step_norm,
        "max_step_norm",
        budget,
    )?;
    if guards.max_step_norm <= 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "maximum step norm must be finite and positive",
        )
        .into());
    }
    let initial = numeric_values(initial, variables, "initial", budget)?;
    let proposal = match backend {
        CompiledRootBackend::FrankenNumpy => {
            newton_refine_with_meter(system, &initial, &guards, budget)
        }
        CompiledRootBackend::FrankenScipy => {
            frankenscipy_fsolve_system_with_meter(system, &initial, budget)
        }
        CompiledRootBackend::FrankenScipyAnalytic => {
            frankenscipy_analytic_system_with_meter(system, &initial, budget)
        }
        CompiledRootBackend::Sparse => {
            sparse_newton_refine_with_meter(system, &initial, &guards, budget)
        }
    }?;
    budget.checkpoint()?;
    Ok(proposal)
}

pub(super) fn solve_least_squares(
    system: &VerifiedResidualSystem,
    initial: &[String],
    backend: CompiledLeastSquaresBackend,
    options: &CompiledStationarityOptions,
    budget: &mut Budget,
) -> Result<LeastSquaresProposal, CertificationSessionError> {
    budget.checkpoint()?;
    let variables = system.num_vars();
    let residuals = system.num_residuals();
    match backend {
        CompiledLeastSquaresBackend::Dense => {
            if !(1..=MAX_CONSUMER_VARIABLES).contains(&variables)
                || !(variables..=MAX_LEAST_SQUARES_RESIDUALS).contains(&residuals)
            {
                return Err(SystemConsumerError::InvalidRequest(
                    "least squares requires 1..=16 variables and variable_count..=64 residuals",
                )
                .into());
            }
        }
        CompiledLeastSquaresBackend::Sparse => {
            if !(1..=MAX_SPARSE_LEAST_SQUARES_VARIABLES).contains(&variables)
                || !(1..=MAX_SPARSE_LEAST_SQUARES_RESIDUALS).contains(&residuals)
            {
                return Err(SystemConsumerError::InvalidRequest(
                    "sparse least squares requires 1..=256 variables and 1..=4096 residuals",
                )
                .into());
            }
        }
    }
    let mut controls = LeastSquaresOptions::default();
    if let Some(iterations) = options.max_iterations {
        if iterations > MAX_CONSUMER_NEWTON_ITERATIONS {
            return Err(SystemConsumerError::InvalidRequest(
                "maximum least-squares iterations exceeds 1024",
            )
            .into());
        }
        controls.max_iterations = iterations;
    }
    controls.gradient_tolerance = numeric_option(
        options.gradient_tolerance.as_ref(),
        controls.gradient_tolerance,
        "gradient_tolerance",
        budget,
    )?;
    if controls.gradient_tolerance < 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "least-squares gradient tolerance must be finite and nonnegative",
        )
        .into());
    }
    controls.max_step_norm = numeric_option(
        options.max_step_norm.as_ref(),
        controls.max_step_norm,
        "max_step_norm",
        budget,
    )?;
    if controls.max_step_norm <= 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "least-squares maximum step norm must be finite and positive",
        )
        .into());
    }
    let initial = numeric_values(initial, variables, "initial", budget)?;
    let proposal = match backend {
        CompiledLeastSquaresBackend::Dense => {
            least_squares_with_meter(system, &initial, &controls, budget)
        }
        CompiledLeastSquaresBackend::Sparse => {
            sparse_least_squares_with_meter(system, &initial, &controls, budget)
        }
    }?;
    budget.checkpoint()?;
    Ok(proposal)
}

pub(super) fn minimize(
    objective: &VerifiedObjective,
    initial: &[String],
    options: &CompiledStationarityOptions,
    budget: &mut Budget,
) -> Result<MinimizationProposal, CertificationSessionError> {
    budget.checkpoint()?;
    let variables = objective.variables().len();
    if !(1..=MAX_MINIMIZE_VARIABLES).contains(&variables) {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "minimization requires 1..=16 variables",
        )
        .into());
    }
    let mut controls = MinimizeOptions::default();
    if let Some(iterations) = options.max_iterations {
        if iterations > MAX_MINIMIZE_ITERATIONS {
            return Err(ObjectiveConsumerError::InvalidRequest(
                "maximum minimization iterations exceeds 1024",
            )
            .into());
        }
        controls.max_iterations = iterations;
    }
    controls.gradient_tolerance = objective_numeric_option(
        options.gradient_tolerance.as_ref(),
        controls.gradient_tolerance,
        "gradient_tolerance",
        budget,
    )?;
    if controls.gradient_tolerance < 0.0 {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "gradient tolerance must be finite and nonnegative",
        )
        .into());
    }
    controls.max_step_norm = objective_numeric_option(
        options.max_step_norm.as_ref(),
        controls.max_step_norm,
        "max_step_norm",
        budget,
    )?;
    if controls.max_step_norm <= 0.0 {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "maximum step norm must be finite and positive",
        )
        .into());
    }
    let initial = compiled_objective::numeric_values(initial, variables, "initial", budget)?;
    let proposal = minimize_with_meter(objective, &initial, &controls, budget)?;
    budget.checkpoint()?;
    Ok(proposal)
}

/// Numerical descent and a Hessian inverse propose an untrusted exact witness.
/// Protected original-source verification is the caller's separate next step.
pub(super) fn propose_stationary_witness(
    objective: &VerifiedObjective,
    initial: &[String],
    options: &CompiledStationarityOptions,
    radii: &[String],
    precision_digits: u32,
    budget: &mut Budget,
) -> Result<RootBoxWitness, CertificationSessionError> {
    budget.checkpoint()?;
    let variables = objective.variables().len();
    if !(1..=MAX_MINIMIZE_VARIABLES).contains(&variables) {
        return Err(ObjectiveConsumerError::InvalidRequest(
            "stationary boxes require 1..=16 variables",
        )
        .into());
    }
    if radii.len() != variables {
        return Err(ObjectiveConsumerError::Dimension {
            input: "radii",
            expected: variables,
            actual: radii.len(),
        }
        .into());
    }
    if !(8..=fsym_core::MAX_ROOT_BOX_PRECISION).contains(&precision_digits) {
        return Err(ObjectiveConsumerError::InvalidRequest("precision must be 8..=128").into());
    }
    charge(
        budget,
        variables,
        variables * std::mem::size_of::<BigRational>(),
    )?;
    let mut exact_radii = Vec::new();
    exact_radii
        .try_reserve_exact(variables)
        .map_err(|_| crate::certification::CertificationError::AllocationFailure)?;
    for radius in radii {
        let radius = parse_scalar(radius, budget)?;
        if radius.numer().is_negative() || radius.numer().is_zero() {
            return Err(ObjectiveConsumerError::InvalidRequest("radii must be positive").into());
        }
        exact_radii.push(radius);
    }
    let proposal = minimize(objective, initial, options, budget)?;
    Ok(propose_stationary_box_with_meter(
        objective,
        &proposal.point,
        &exact_radii,
        precision_digits,
        budget,
    )?)
}

fn numeric_option(
    value: Option<&String>,
    default: f64,
    name: &'static str,
    budget: &mut Budget,
) -> Result<f64, CertificationSessionError> {
    match value {
        Some(value) => Ok(numeric_values(std::slice::from_ref(value), 1, name, budget)?[0]),
        None => Ok(default),
    }
}

fn objective_numeric_option(
    value: Option<&String>,
    default: f64,
    name: &'static str,
    budget: &mut Budget,
) -> Result<f64, CertificationSessionError> {
    match value {
        Some(value) => {
            Ok(
                compiled_objective::numeric_values(std::slice::from_ref(value), 1, name, budget)?
                    [0],
            )
        }
        None => Ok(default),
    }
}
