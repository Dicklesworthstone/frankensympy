//! Bounded square-system consumers for checked symbolic artifacts.
//!
//! These adapters accept 1..=16 variables and keep one caller-owned ledger
//! through numerical proposal generation and original-source root verification.
//! The pinned donors are synchronous. Cancellation is checked before and after
//! each NumPy solve, at every SciPy residual callback, and before publication;
//! it does not interrupt an individual donor kernel or promise a wall-time bound.
//!
//! Donor work is charged conservatively before execution. NumPy's small-matrix
//! LU uses three buffers. SciPy receives an initialization allowance and an
//! allowance for the bounded work between successive residual callbacks, covering
//! its dense solve, possible damped solve, and update/search buffers. Unused
//! allowances are not refunded. Native expression and independent interval work
//! use their existing metered evaluators. Counts describe logical work and
//! cumulative requested buffer storage, not RSS or allocator overhead.

use crate::{NewtonGuards, exact_binary64_rational};
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_core::{BigRational, RealBall, RootBoxError, RootBoxReceipt, RootBoxWitness};
use std::cell::RefCell;
use std::mem::size_of;

/// The independent source checker and these numerical consumers share this cap.
pub const MAX_CONSUMER_VARIABLES: usize = fsym_core::MAX_ROOT_BOX_VARIABLES;
/// A finite ceiling also applies when the caller chooses the unbounded ledger.
pub const MAX_CONSUMER_NEWTON_ITERATIONS: usize = 1024;
const FSCIPY_ITERATIONS: usize = 200;
const FSCIPY_TOLERANCE: f64 = 1e-10;

/// An approximate numerical proposal, without a root-existence claim.
#[derive(Debug, Clone, PartialEq)]
pub struct RootProposal {
    pub root: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: usize,
}

/// A proposal accompanied by an immutable, independently checked root box.
#[derive(Debug, Clone)]
pub struct CertifiedRootOutcome {
    approximation: Vec<f64>,
    certificate: RootBoxReceipt,
}

impl CertifiedRootOutcome {
    pub fn approximation(&self) -> &[f64] {
        &self.approximation
    }

    pub fn certificate(&self) -> &RootBoxReceipt {
        &self.certificate
    }
}

/// Operational refusals retain their original typed causes across donor calls.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SystemConsumerError {
    #[error(
        "expected a square system with 1..=16 variables; got {variables} variables and {residuals} residuals"
    )]
    UnsupportedShape { variables: usize, residuals: usize },
    #[error("{input} has {actual} coordinates; expected {expected}")]
    Dimension {
        input: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{0} contains a non-finite value")]
    NonFinite(&'static str),
    #[error("invalid consumer request: {0}")]
    InvalidRequest(&'static str),
    #[error("the pinned linear solver refused: {0}")]
    LinearSolve(String),
    #[error("the pinned nonlinear solver refused: {0}")]
    NonlinearSolve(String),
    #[error("the pinned consumer exceeded its declared contract: {0}")]
    ConsumerContract(&'static str),
    #[error("no convergence after {iterations} iterations; residual infinity-norm {residual_norm}")]
    NoConvergence {
        iterations: usize,
        residual_norm: f64,
    },
    #[error("compiled evaluation refused: {0}")]
    Evaluation(#[from] EvalError),
    #[error("consumer work stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("original-source root verification refused: {0}")]
    Certification(#[from] RootBoxError),
}

/// Newton refinement using the pinned `fnp_linalg::solve_nxn` for every step.
/// A small floating residual is only a proposal, even at a singular root.
pub fn newton_refine(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    guards: &NewtonGuards,
) -> Result<RootProposal, SystemConsumerError> {
    newton_refine_with_meter(system, initial, guards, &mut Unbounded)
}

pub fn newton_refine_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    guards: &NewtonGuards,
    meter: &mut M,
) -> Result<RootProposal, SystemConsumerError> {
    let n = admit_point(system, initial, meter)?;
    validate_guards(guards)?;
    let mut point = copy_values(initial, meter)?;
    let mut residuals = zeros(n, meter)?;
    let mut jacobian = zeros(n * n, meter)?;
    let mut rhs = zeros(n, meter)?;
    let mut residual_norm = f64::INFINITY;
    for iteration in 0..=guards.max_iterations {
        evaluate_system(system, &point, &mut residuals, &mut jacobian, meter)?;
        residual_norm = infinity_norm(&residuals, meter)?;
        if residual_norm <= guards.residual_tolerance {
            publication_checkpoint(n, meter)?;
            return Ok(RootProposal {
                root: point,
                residual_norm,
                iterations: iteration,
            });
        }
        if iteration == guards.max_iterations {
            break;
        }
        charge(meter, n as u64, 0, 0)?;
        for (target, value) in rhs.iter_mut().zip(&residuals) {
            *target = -*value;
        }
        let step = linear_solve(&jacobian, &rhs, n, meter)?;
        if infinity_norm(&step, meter)? > guards.max_step_norm {
            return Err(SystemConsumerError::NoConvergence {
                iterations: iteration,
                residual_norm,
            });
        }
        charge(meter, n as u64, 0, 0)?;
        for (coordinate, delta) in point.iter_mut().zip(step) {
            *coordinate += delta;
        }
        finite(&point, "updated iterate", meter)?;
    }
    Err(SystemConsumerError::NoConvergence {
        iterations: guards.max_iterations,
        residual_norm,
    })
}

/// Run the pinned synchronous `fsci_opt::fsolve` on the same checked source.
/// Its finite-difference Jacobian is a donor algorithm, not symbolic evidence.
pub fn frankenscipy_fsolve_system(
    system: &VerifiedResidualSystem,
    initial: &[f64],
) -> Result<RootProposal, SystemConsumerError> {
    frankenscipy_fsolve_system_with_meter(system, initial, &mut Unbounded)
}

pub fn frankenscipy_fsolve_system_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    meter: &mut M,
) -> Result<RootProposal, SystemConsumerError> {
    let n = admit_point(system, initial, meter)?;
    // x, the nested Jacobian, and bounded terminal/error formatting. Callback
    // intervals are admitted separately, so a quick solve need not reserve all
    // 200 iterations. The donor is pinned to this exact synchronous interface.
    charge(
        meter,
        (4 * n * n + 8 * n + 64) as u64,
        ((n * n + n) * size_of::<f64>() + n * size_of::<Vec<f64>>() + 512) as u64,
        (n + 6) as u64,
    )?;
    let maximum_calls = 1 + FSCIPY_ITERATIONS * (n + 11);
    let (result, failure, calls) = {
        let state = RefCell::new(CallbackState {
            meter: &mut *meter,
            error: None,
            calls: 0,
        });
        let result = fsci_opt::fsolve(
            |point| {
                let mut state = state.borrow_mut();
                if state.error.is_some() {
                    return Vec::new();
                }
                state.calls += 1;
                let evaluated = if state.calls > maximum_calls {
                    Err(SystemConsumerError::ConsumerContract("callback count"))
                } else {
                    scipy_interval(n, state.meter)
                        .and_then(|()| evaluate_residuals(system, point, n, state.meter))
                };
                match evaluated {
                    Ok(values) => values,
                    Err(error) => {
                        state.error = Some(error);
                        // The donor rejects a wrong output length immediately.
                        // This sentinel allocates nothing after a refusal, and
                        // the original typed error wins over the donor wrapper.
                        Vec::new()
                    }
                }
            },
            initial,
        );
        let state = state.into_inner();
        (result, state.error, state.calls)
    };
    if let Some(error) = failure {
        return Err(error);
    }
    meter.checkpoint()?;
    let result = result.map_err(|error| SystemConsumerError::NonlinearSolve(error.to_string()))?;
    if result.x.len() != n || result.fun.len() != n {
        return Err(SystemConsumerError::ConsumerContract("result dimensions"));
    }
    if result.iterations > FSCIPY_ITERATIONS || result.function_calls != calls {
        return Err(SystemConsumerError::ConsumerContract(
            "iteration accounting",
        ));
    }
    finite(&result.x, "consumer root", meter)?;
    finite(&result.fun, "consumer reported residual", meter)?;
    // Neither a cached donor residual nor its convergence flag establishes
    // even our approximate acceptance condition. Recheck the actual source.
    let residuals = evaluate_residuals(system, &result.x, n, meter)?;
    let residual_norm = infinity_norm(&residuals, meter)?;
    if !result.converged || residual_norm > FSCIPY_TOLERANCE {
        return Err(SystemConsumerError::NoConvergence {
            iterations: result.iterations,
            residual_norm,
        });
    }
    publication_checkpoint(n, meter)?;
    Ok(RootProposal {
        root: result.x,
        residual_norm,
        iterations: result.iterations,
    })
}

struct CallbackState<'a, M> {
    meter: &'a mut M,
    error: Option<SystemConsumerError>,
    calls: usize,
}

/// Check a caller-proposed box around an approximate root. The donor inverse
/// is only an untrusted preconditioner; original-source interval AD determines
/// whether the box contains one unique nonsingular real root.
pub fn certify_root(
    system: &VerifiedResidualSystem,
    approximation: &[f64],
    radii: &[f64],
    precision_digits: u32,
) -> Result<CertifiedRootOutcome, SystemConsumerError> {
    certify_root_with_meter(
        system,
        approximation,
        radii,
        precision_digits,
        &mut Unbounded,
    )
}

pub fn certify_root_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    approximation: &[f64],
    radii: &[f64],
    precision_digits: u32,
    meter: &mut M,
) -> Result<CertifiedRootOutcome, SystemConsumerError> {
    let n = admit_root_request(system, approximation, radii, precision_digits, meter)?;
    let mut values = zeros(n, meter)?;
    let mut jacobian = zeros(n * n, meter)?;
    evaluate_system(system, approximation, &mut values, &mut jacobian, meter)?;
    let mut inverse = zeros(n * n, meter)?;
    let mut unit = zeros(n, meter)?;
    for column in 0..n {
        charge(meter, (2 * n + 1) as u64, 0, 0)?;
        unit.fill(0.0);
        unit[column] = 1.0;
        let solution = linear_solve(&jacobian, &unit, n, meter)?;
        for row in 0..n {
            inverse[row * n + column] = solution[row];
        }
    }
    let mut center = buffer(n, meter)?;
    let mut domain = buffer(n, meter)?;
    let mut preconditioner = buffer(n, meter)?;
    for coordinate in 0..n {
        center.push(exact_coordinate(approximation[coordinate], meter)?);
        // Both exact conversions are charged: the ball owns its midpoint,
        // while the witness separately retains its proposed center.
        let midpoint = exact_coordinate(approximation[coordinate], meter)?;
        let radius = exact_coordinate(radii[coordinate], meter)?;
        domain.push(RealBall::new(midpoint, radius).map_err(RootBoxError::from)?);
        let mut row = buffer(n, meter)?;
        for column in 0..n {
            row.push(exact_coordinate(inverse[coordinate * n + column], meter)?);
        }
        preconditioner.push(row);
    }
    let approximation = copy_values(approximation, meter)?;
    let witness = RootBoxWitness {
        domain,
        center,
        preconditioner,
        precision_digits,
    };
    meter.checkpoint()?;
    let certificate =
        fsym_core::verify_root_box(system.residuals(), system.vars(), &witness, meter)?;
    publication_checkpoint(n, meter)?;
    Ok(CertifiedRootOutcome {
        approximation,
        certificate,
    })
}

pub fn newton_refine_root_box(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    guards: &NewtonGuards,
    radii: &[f64],
    precision_digits: u32,
) -> Result<CertifiedRootOutcome, SystemConsumerError> {
    newton_refine_root_box_with_meter(
        system,
        initial,
        guards,
        radii,
        precision_digits,
        &mut Unbounded,
    )
}

/// Use one cumulative ledger for the complete proposal/verification operation.
pub fn newton_refine_root_box_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    guards: &NewtonGuards,
    radii: &[f64],
    precision_digits: u32,
    meter: &mut M,
) -> Result<CertifiedRootOutcome, SystemConsumerError> {
    admit_root_request(system, initial, radii, precision_digits, meter)?;
    let proposal = newton_refine_with_meter(system, initial, guards, meter)?;
    certify_root_with_meter(system, &proposal.root, radii, precision_digits, meter)
}

pub fn frankenscipy_fsolve_system_root_box(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    radii: &[f64],
    precision_digits: u32,
) -> Result<CertifiedRootOutcome, SystemConsumerError> {
    frankenscipy_fsolve_system_root_box_with_meter(
        system,
        initial,
        radii,
        precision_digits,
        &mut Unbounded,
    )
}

pub fn frankenscipy_fsolve_system_root_box_with_meter<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    initial: &[f64],
    radii: &[f64],
    precision_digits: u32,
    meter: &mut M,
) -> Result<CertifiedRootOutcome, SystemConsumerError> {
    admit_root_request(system, initial, radii, precision_digits, meter)?;
    let proposal = frankenscipy_fsolve_system_with_meter(system, initial, meter)?;
    certify_root_with_meter(system, &proposal.root, radii, precision_digits, meter)
}

fn admit_point<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    meter: &mut M,
) -> Result<usize, SystemConsumerError> {
    meter.checkpoint()?;
    let n = system.vars().len();
    if !(1..=MAX_CONSUMER_VARIABLES).contains(&n)
        || system.residuals().len() != n
        || system.compiled().num_vars != n
        || system.compiled().num_residuals != n
    {
        return Err(SystemConsumerError::UnsupportedShape {
            variables: n,
            residuals: system.residuals().len(),
        });
    }
    point_dimension(point, n, "point")?;
    finite(point, "point", meter)?;
    Ok(n)
}

fn admit_root_request<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    radii: &[f64],
    precision: u32,
    meter: &mut M,
) -> Result<usize, SystemConsumerError> {
    let n = admit_point(system, point, meter)?;
    point_dimension(radii, n, "radii")?;
    finite(radii, "radii", meter)?;
    charge(meter, n as u64, 0, 0)?;
    if radii.iter().any(|radius| *radius <= 0.0) {
        return Err(SystemConsumerError::InvalidRequest(
            "radii must be positive",
        ));
    }
    if !(8..=fsym_core::MAX_ROOT_BOX_PRECISION).contains(&precision) {
        return Err(SystemConsumerError::InvalidRequest(
            "precision must be 8..=128",
        ));
    }
    Ok(n)
}

fn point_dimension(
    values: &[f64],
    n: usize,
    input: &'static str,
) -> Result<(), SystemConsumerError> {
    if values.len() != n {
        return Err(SystemConsumerError::Dimension {
            input,
            expected: n,
            actual: values.len(),
        });
    }
    Ok(())
}

fn validate_guards(guards: &NewtonGuards) -> Result<(), SystemConsumerError> {
    if guards.max_iterations > MAX_CONSUMER_NEWTON_ITERATIONS {
        return Err(SystemConsumerError::InvalidRequest(
            "maximum Newton iterations exceeds 1024",
        ));
    }
    if !guards.residual_tolerance.is_finite() || guards.residual_tolerance < 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "residual tolerance must be finite and nonnegative",
        ));
    }
    if !guards.max_step_norm.is_finite() || guards.max_step_norm <= 0.0 {
        return Err(SystemConsumerError::InvalidRequest(
            "maximum step norm must be finite and positive",
        ));
    }
    Ok(())
}

fn evaluate_system<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    residuals: &mut [f64],
    jacobian: &mut [f64],
    meter: &mut M,
) -> Result<(), SystemConsumerError> {
    // The dense system's legacy evaluator permits nonfinite intermediates.
    // A numerical consumer must retain every original source-domain check,
    // even when a later zero, reciprocal, or zero power conceals that value.
    charge(meter, (residuals.len() + jacobian.len()) as u64, 0, 0)?;
    for (value, program) in residuals
        .iter_mut()
        .zip(&system.compiled().compiled_residuals)
    {
        *value = program.try_eval_finite_with_meter(point, meter)?;
    }
    for (value, program) in jacobian
        .iter_mut()
        .zip(&system.compiled().compiled_jacobian)
    {
        *value = program.try_eval_finite_with_meter(point, meter)?;
    }
    finite(residuals, "residual", meter)?;
    finite(jacobian, "Jacobian", meter)
}

fn evaluate_residuals<M: BudgetMeter>(
    system: &VerifiedResidualSystem,
    point: &[f64],
    n: usize,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    point_dimension(point, n, "consumer iterate")?;
    finite(point, "consumer iterate", meter)?;
    let mut values = zeros(n, meter)?;
    charge(meter, n as u64, 0, 0)?;
    for (value, program) in values.iter_mut().zip(&system.compiled().compiled_residuals) {
        *value = program.try_eval_finite_with_meter(point, meter)?;
    }
    finite(&values, "residual", meter)?;
    Ok(values)
}

fn linear_solve<M: BudgetMeter>(
    matrix: &[f64],
    rhs: &[f64],
    n: usize,
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    // n<=16 stays below the pinned donor's blocked/parallel LU threshold (512).
    // Its finite-input path allocates LU, permutation, and solution vectors.
    charge(
        meter,
        (4 * n * n * n + 16 * n * n + 8 * n + 32) as u64,
        ((n * n + n) * size_of::<f64>() + n * size_of::<usize>() + 256) as u64,
        4,
    )?;
    let solution = fnp_linalg::solve_nxn(matrix, rhs, n)
        .map_err(|error| SystemConsumerError::LinearSolve(error.to_string()))?;
    meter.checkpoint()?;
    point_dimension(&solution, n, "linear solver result")?;
    finite(&solution, "linear solver result", meter)?;
    Ok(solution)
}

fn scipy_interval<M: BudgetMeter>(n: usize, meter: &mut M) -> Result<(), SystemConsumerError> {
    // Bounds include two augmented dense solves, J^T J, Broyden bookkeeping,
    // and all temporary vectors until the next synchronous residual callback.
    // Some callbacks precede only one finite-difference column, so this is an
    // intentional conservative charge; it never refunds unused work.
    charge(
        meter,
        (8 * n * n * n + 64 * n * n + 128 * n + 64) as u64,
        ((8 * n * n + 64 * n) * size_of::<f64>() + (4 * n + 8) * size_of::<Vec<f64>>() + 512)
            as u64,
        (8 * n + 32) as u64,
    )
}

fn exact_coordinate<M: BudgetMeter>(
    value: f64,
    meter: &mut M,
) -> Result<BigRational, SystemConsumerError> {
    // A finite binary64 rational has at most 1075 bits per component. Reserve
    // a conservative allowance for powers of two, normalization, and storage.
    charge(meter, 8192, 16 * 1024, 32)?;
    exact_binary64_rational(value).ok_or(SystemConsumerError::NonFinite("root-box proposal"))
}

fn finite<M: BudgetMeter>(
    values: &[f64],
    label: &'static str,
    meter: &mut M,
) -> Result<(), SystemConsumerError> {
    charge(meter, values.len() as u64, 0, 0)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SystemConsumerError::NonFinite(label));
    }
    Ok(())
}

fn infinity_norm<M: BudgetMeter>(
    values: &[f64],
    meter: &mut M,
) -> Result<f64, SystemConsumerError> {
    charge(meter, values.len() as u64, 0, 0)?;
    Ok(values
        .iter()
        .fold(0.0_f64, |norm, value| norm.max(value.abs())))
}

fn buffer<T, M: BudgetMeter>(count: usize, meter: &mut M) -> Result<Vec<T>, SystemConsumerError> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(EvalError::DimensionOverflow)?;
    charge(meter, 0, bytes as u64, u64::from(bytes != 0))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| EvalError::AllocationFailure)?;
    Ok(result)
}

fn zeros<M: BudgetMeter>(count: usize, meter: &mut M) -> Result<Vec<f64>, SystemConsumerError> {
    let mut values = buffer(count, meter)?;
    charge(meter, count as u64, 0, 0)?;
    values.resize(count, 0.0);
    Ok(values)
}

fn copy_values<M: BudgetMeter>(
    source: &[f64],
    meter: &mut M,
) -> Result<Vec<f64>, SystemConsumerError> {
    let mut values = buffer(source.len(), meter)?;
    charge(meter, source.len() as u64, 0, 0)?;
    values.extend_from_slice(source);
    Ok(values)
}

fn publication_checkpoint<M: BudgetMeter>(
    n: usize,
    meter: &mut M,
) -> Result<(), SystemConsumerError> {
    charge(meter, n as u64 + 1, 0, 0)?;
    meter.checkpoint()?;
    Ok(())
}

fn charge<M: BudgetMeter>(
    meter: &mut M,
    steps: u64,
    bytes: u64,
    allocations: u64,
) -> Result<(), SystemConsumerError> {
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
