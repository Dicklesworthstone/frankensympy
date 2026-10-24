//! Narrow admitted consumer boundary for pinned FrankenNumPy numeric
//! kernels (`fra-rc-adapters-iym`, WS12).
//!
//! The campaign's compiled residual system and symbolic Jacobian
//! ([`fsym_calculus::CompiledResidualSystem`]) define the artifact; this
//! boundary executes the LINEAR SOLVE of each Newton step through the
//! pinned `fnp-linalg` consumer (`franken_numpy` @ d5d29c9d, admitted in
//! registries/dependencies.toml) instead of an in-house solver, so the
//! consumer's real API is exercised against the same canonical artifact
//! the native lane evaluates.
//!
//! Boundary rules (registries/dependencies.toml, franken_numpy entry):
//! - only `fnp-linalg` is imported; no pyo3/CPython bridge, no RNG, no
//!   I/O crates from the consumer tree;
//! - only plain `f64` slices cross the boundary — no consumer types
//!   leak into symbolic/native representations;
//! - the consumer's rayon data-parallelism stays internal to its own
//!   crate and never schedules frankensympy work;
//! - every consumer refusal is typed and never reported as a native
//!   mathematical result;
//! - finite differences remain diagnostics only: the Jacobian fed to the
//!   consumer is the SYMBOLIC one compiled by fsym-calculus.

#![forbid(unsafe_code)]

use fsym_calculus::{CompiledResidualSystem, VerifiedResidualSystem};
use fsym_core::{BigInt, BigRational, Expr, RealBall, Symbol};
use std::collections::HashMap;

/// Identity of the pinned consumer this boundary was compiled against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumerProvenance {
    pub crate_name: &'static str,
    pub crate_version: &'static str,
    pub source_commit: &'static str,
    /// Path of the pinned worktree the path dependency resolves into.
    pub pinned_worktree: &'static str,
    /// The function this boundary executes from the consumer's API.
    pub consumer_entry_point: &'static str,
}

pub fn provenance() -> ConsumerProvenance {
    ConsumerProvenance {
        crate_name: "fnp-linalg",
        crate_version: "0.3.0",
        source_commit: "d5d29c9da36141f611d1414b4c6ef434ab31ea24",
        pinned_worktree: "/data/projects/franken_numpy-pin",
        consumer_entry_point: "fnp_linalg::solve_2x2",
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConsumerError {
    #[error("Jacobian is singular at the current iterate (consumer refused: {0})")]
    SingularJacobian(String),
    #[error(
        "Newton failed to converge in {max_iterations} iterations; last residual norm {last_residual_norm}"
    )]
    NoConvergence {
        max_iterations: usize,
        last_residual_norm: f64,
    },
    #[error("compiled system evaluation failed: {0}")]
    Evaluation(String),
    #[error("system is not 2x2; this boundary executes the declared 2x2 consumer ABI only")]
    UnsupportedShape,
    #[error("invalid Newton guard: {0}")]
    InvalidGuards(&'static str),
    #[error("certified residual check failed: {0}")]
    Certification(String),
}

/// Result of a converged Newton refinement driven by the consumer solver.
#[derive(Debug, Clone)]
pub struct NewtonOutcome {
    /// Final approximate iterate. A residual certificate is a separate check.
    pub root: [f64; 2],
    /// Residual infinity-norm at the final iterate (native compiled lane).
    pub residual_norm: f64,
    /// Determinant of the Jacobian at the final iterate, as computed by
    /// boundary as a floating-point diagnostic, not a nonsingularity proof.
    pub final_jacobian_determinant: f64,
    pub iterations: usize,
}

/// Guards for one refinement run. All fields are caller-declared.
#[derive(Debug, Clone)]
pub struct NewtonGuards {
    pub max_iterations: usize,
    /// Converge when the residual infinity-norm drops below this bound.
    pub residual_tolerance: f64,
    /// Refuse when the step infinity-norm exceeds this (divergence guard).
    pub max_step_norm: f64,
}

impl Default for NewtonGuards {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            residual_tolerance: 1e-12,
            max_step_norm: 1e6,
        }
    }
}

/// Newton-refines a root of the compiled 2x2 system, solving each linear
/// step through the pinned `fnp-linalg` consumer.
///
/// Residuals and the Jacobian come from the compiled system (the exact
/// same artifact the native lane evaluates); the consumer only ever sees
/// plain `f64` rows of `J` and `-f`. A consumer refusal of the solve is a
/// typed singular-Jacobian condition — never a silent fallback to any
/// in-house path.
pub fn newton_refine_2x2(
    system: &CompiledResidualSystem,
    initial: [f64; 2],
    guards: &NewtonGuards,
) -> Result<NewtonOutcome, ConsumerError> {
    check_system_shape(system)?;
    require_finite(&initial, "initial iterate")?;
    if !guards.residual_tolerance.is_finite() || guards.residual_tolerance < 0.0 {
        return Err(ConsumerError::InvalidGuards(
            "residual tolerance must be finite and nonnegative",
        ));
    }
    if !guards.max_step_norm.is_finite() || guards.max_step_norm <= 0.0 {
        return Err(ConsumerError::InvalidGuards(
            "maximum step norm must be finite and positive",
        ));
    }
    let mut x = initial;
    let mut residuals = [0.0_f64; 2];
    let mut jacobian_flat = [0.0_f64; 4];
    let mut last_norm = f64::INFINITY;
    for iteration in 0..=guards.max_iterations {
        system
            .try_eval_system(&x, &mut residuals, &mut jacobian_flat)
            .map_err(|error| ConsumerError::Evaluation(error.to_string()))?;
        require_finite(&residuals, "residual")?;
        require_finite(&jacobian_flat, "Jacobian")?;
        last_norm = residuals
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let jacobian = [
            [jacobian_flat[0], jacobian_flat[1]],
            [jacobian_flat[2], jacobian_flat[3]],
        ];
        let determinant = jacobian_det(&jacobian);
        require_finite(&[determinant], "Jacobian determinant")?;
        if last_norm <= guards.residual_tolerance {
            return Ok(NewtonOutcome {
                root: x,
                residual_norm: last_norm,
                final_jacobian_determinant: determinant,
                iterations: iteration,
            });
        }
        if iteration == guards.max_iterations {
            break;
        }
        let negated = [-residuals[0], -residuals[1]];
        // Only the admitted consumer solves the linear step. An exact root
        // with a singular Jacobian may satisfy the residual criterion above;
        // that does not establish root uniqueness or nonsingularity.
        let step = fnp_linalg::solve_2x2(jacobian, negated)
            .map_err(|error| ConsumerError::SingularJacobian(error.to_string()))?;
        require_finite(&step, "Newton step")?;
        let step_norm = step[0].abs().max(step[1].abs());
        if step_norm > guards.max_step_norm {
            return Err(ConsumerError::NoConvergence {
                max_iterations: guards.max_iterations,
                last_residual_norm: last_norm,
            });
        }
        x[0] += step[0];
        x[1] += step[1];
        require_finite(&x, "updated iterate")?;
    }
    Err(ConsumerError::NoConvergence {
        max_iterations: guards.max_iterations,
        last_residual_norm: last_norm,
    })
}

/// Executes the checked derivative artifact and certifies the residual of the
/// returned binary64 point against that artifact's own symbolic residuals.
/// This proves the declared residual bound, not existence or uniqueness of an
/// exact root near the point. Failed certification never returns an outcome.
pub fn newton_refine_verified_2x2(
    system: &VerifiedResidualSystem,
    initial: [f64; 2],
    guards: &NewtonGuards,
    bound_decimal_places: u32,
) -> Result<NewtonOutcome, ConsumerError> {
    validate_bound_precision(bound_decimal_places).map_err(ConsumerError::Certification)?;
    let outcome = newton_refine_2x2(system.compiled(), initial, guards)?;
    certified_residual_bound(
        system.residuals(),
        system.vars(),
        &outcome.root,
        bound_decimal_places,
    )
    .map_err(ConsumerError::Certification)?;
    Ok(outcome)
}

fn check_system_shape(system: &CompiledResidualSystem) -> Result<(), ConsumerError> {
    if system.num_vars != 2
        || system.num_residuals != 2
        || system.vars.len() != 2
        || system.vars[0] == system.vars[1]
    {
        return Err(ConsumerError::UnsupportedShape);
    }
    Ok(())
}

fn require_finite(values: &[f64], lane: &str) -> Result<(), ConsumerError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ConsumerError::Evaluation(format!(
            "{lane} contains non-finite values"
        )));
    }
    Ok(())
}

/// FrankenSciPy consumer execution: `fsci-opt::fsolve` drives the SAME
/// compiled residual artifact (plain `f64` vector in, residual vector
/// out). The consumer's internal finite-difference Jacobian is its own
/// algorithm; trust comes from the certified ball bound at its returned
/// root, not from the solver's self-report.
pub fn frankenscipy_fsolve(
    system: &CompiledResidualSystem,
    initial: &[f64; 2],
) -> Result<[f64; 2], ConsumerError> {
    check_system_shape(system)?;
    require_finite(initial, "initial iterate")?;
    let evaluation_error: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    let residuals = |point: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0_f64; 2];
        let evaluated = if point.len() != 2 {
            Err(ConsumerError::UnsupportedShape)
        } else {
            require_finite(point, "consumer iterate")
                .and_then(|()| {
                    system
                        .try_eval_residuals(point, &mut out)
                        .map_err(|error| ConsumerError::Evaluation(error.to_string()))
                })
                .and_then(|()| require_finite(&out, "consumer residual"))
        };
        if let Err(error) = evaluated {
            let mut slot = evaluation_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if slot.is_none() {
                *slot = Some(error.to_string());
            }
            out.fill(f64::NAN);
        }
        out
    };
    let result = fsci_opt::fsolve(residuals, initial);
    if let Some(message) = evaluation_error
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(ConsumerError::Evaluation(format!(
            "compiled residual evaluation failed during the consumer solve: {message}"
        )));
    }
    let result = result.map_err(|error| ConsumerError::Evaluation(error.to_string()))?;
    if result.x.len() != 2 || result.fun.len() != 2 {
        return Err(ConsumerError::UnsupportedShape);
    }
    require_finite(&result.x, "consumer root")?;
    require_finite(&result.fun, "consumer reported residual")?;
    // Re-evaluate at the returned point rather than trusting the consumer's
    // cached residual vector. This is still an approximate diagnostic.
    let mut final_residual = [0.0_f64; 2];
    system
        .try_eval_residuals(&result.x, &mut final_residual)
        .map_err(|error| ConsumerError::Evaluation(error.to_string()))?;
    require_finite(&final_residual, "final residual")?;
    if !result.converged {
        return Err(ConsumerError::NoConvergence {
            max_iterations: 200,
            last_residual_norm: final_residual
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        });
    }
    Ok([result.x[0], result.x[1]])
}

/// Runs FrankenSciPy on the same verified artifact and checks the returned
/// point using its retained symbolic sources and a declared enclosure bound.
pub fn frankenscipy_fsolve_verified(
    system: &VerifiedResidualSystem,
    initial: &[f64; 2],
    bound_decimal_places: u32,
) -> Result<[f64; 2], ConsumerError> {
    validate_bound_precision(bound_decimal_places).map_err(ConsumerError::Certification)?;
    let root = frankenscipy_fsolve(system.compiled(), initial)?;
    certified_residual_bound(
        system.residuals(),
        system.vars(),
        &root,
        bound_decimal_places,
    )
    .map_err(ConsumerError::Certification)?;
    Ok(root)
}

fn jacobian_det(jacobian: &[[f64; 2]; 2]) -> f64 {
    jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
}

/// The exact binary64 value of a finite `f64` as a rational
/// (mantissa × 2^exponent), for certified ball evaluation at accepted
/// iterates. Subnormals and zero included; non-finite refuses.
pub fn exact_binary64_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(BigRational::new(BigInt::from(0), BigInt::from(1)));
    }
    let bits = value.to_bits();
    let negative = bits >> 63 == 1;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;
    let (mantissa, exponent) = if exponent_bits == 0 {
        (BigInt::from(mantissa_bits), -1074_i32)
    } else {
        (
            BigInt::from(mantissa_bits | (1u64 << 52)),
            exponent_bits - 1075,
        )
    };
    let two = BigInt::from(2);
    let magnitude = if exponent >= 0 {
        BigRational::new(mantissa * two.pow(exponent as u32), BigInt::from(1))
    } else {
        BigRational::new(mantissa, two.pow((-exponent) as u32))
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// Certified residual bound at the supplied binary64 point, evaluated as exact
/// rational coordinates. Every full enclosure must lie in `[-bound, bound]`:
/// `abs(midpoint) + radius <= 10^(-bound_decimal_places)`.
///
/// This boundary has exactly two distinct variables. Residuals cannot be empty.
/// The statement is about residual size at this point, not exact-root existence,
/// uniqueness, or a neighborhood around the point.
pub fn certified_residual_bound(
    residual_exprs: &[Expr],
    vars: &[Symbol],
    root: &[f64; 2],
    bound_decimal_places: u32,
) -> Result<(), String> {
    let precision = validate_bound_precision(bound_decimal_places)?;
    if vars.len() != root.len() || vars[0] == vars[1] {
        return Err("the residual certificate requires exactly two distinct variables".into());
    }
    if residual_exprs.is_empty() {
        return Err("the residual certificate requires at least one residual".into());
    }
    let bound = BigRational::new(BigInt::from(1), BigInt::from(10).pow(bound_decimal_places));
    let mut substitution = HashMap::new();
    for (variable, &value) in vars.iter().zip(root) {
        let exact = exact_binary64_rational(value)
            .ok_or_else(|| "root contains a non-finite component".to_string())?;
        substitution.insert(variable.clone(), Expr::Rational(exact));
    }
    for (index, residual) in residual_exprs.iter().enumerate() {
        let specialized = specialize_residual(residual, &substitution, 0, &mut 0)?;
        let ball = specialized
            .evalf_ball(precision)
            .map_err(|error| format!("residual {index} ball evaluation failed: {error}"))?;
        if !enclosure_within_bound(&ball, &bound) {
            return Err(format!(
                "residual {index} full certified enclosure {ball} exceeds the declared bound"
            ));
        }
    }
    Ok(())
}

/// Substitute coordinates without simplifying away a source domain check.
/// Smart constructors could turn e.g. log(x)^0 into 1 at x=0 before the ball
/// evaluator gets the chance to reject the original undefined logarithm.
fn specialize_residual(
    expr: &Expr,
    substitution: &HashMap<Symbol, Expr>,
    depth: usize,
    visited: &mut usize,
) -> Result<Expr, String> {
    if depth > fsym_calculus::MAX_COMPILE_DEPTH || *visited >= fsym_calculus::MAX_COMPILE_OPS {
        return Err("residual exceeds the bounded certified-evaluation envelope".into());
    }
    *visited += 1;
    match expr {
        Expr::Sym(variable) => Ok(substitution.get(variable).unwrap_or(expr).clone()),
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => Ok(expr.clone()),
        Expr::Pow(base, exponent) => Ok(Expr::Pow(
            std::sync::Arc::new(specialize_residual(base, substitution, depth + 1, visited)?),
            std::sync::Arc::new(specialize_residual(
                exponent,
                substitution,
                depth + 1,
                visited,
            )?),
        )),
        Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
            if args.len() > fsym_calculus::MAX_COMPILE_OPS.saturating_sub(*visited) {
                return Err("residual exceeds the bounded certified-evaluation envelope".into());
            }
            let mut specialized = Vec::new();
            specialized
                .try_reserve_exact(args.len())
                .map_err(|_| "residual specialization allocation refused".to_string())?;
            for arg in args {
                specialized.push(specialize_residual(arg, substitution, depth + 1, visited)?);
            }
            Ok(match expr {
                Expr::Add(_) => Expr::Add(specialized),
                Expr::Mul(_) => Expr::Mul(specialized),
                Expr::Function(name, _) => Expr::Function(name.clone(), specialized),
                _ => unreachable!("the enclosing match admits only variadic nodes"),
            })
        }
    }
}

fn validate_bound_precision(bound_decimal_places: u32) -> Result<u32, String> {
    // Reserve guard digits while refusing hostile exponents before constructing
    // 10^digits or starting any numerical solve.
    let precision = bound_decimal_places
        .checked_add(8)
        .ok_or_else(|| "requested residual precision exceeds the certified envelope".to_string())?
        .max(30);
    if precision > RealBall::MAX_TRANSCENDENTAL_PRECISION_DIGITS {
        return Err("requested residual precision exceeds the certified envelope".into());
    }
    Ok(precision)
}

fn enclosure_within_bound(ball: &RealBall, bound: &BigRational) -> bool {
    let midpoint = ball.midpoint();
    let magnitude = if midpoint < &BigRational::from_integer(BigInt::from(0)) {
        -midpoint.clone()
    } else {
        midpoint.clone()
    };
    magnitude + ball.radius().clone() <= *bound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_bound_contains_the_entire_enclosure() {
        let q = |n, d| BigRational::new(BigInt::from(n), BigInt::from(d));
        let bound = q(1_i64, 1_i64);
        for sign in [-1_i64, 1] {
            // Each component is below one, but the enclosure reaches ±1.5.
            let too_wide = RealBall::new(q(sign * 3, 4), q(3, 4)).unwrap();
            assert!(!enclosure_within_bound(&too_wide, &bound));
            let exact_boundary = RealBall::new(q(sign * 3, 4), q(1, 4)).unwrap();
            assert!(enclosure_within_bound(&exact_boundary, &bound));
        }
    }
}

/// A numerical proposal together with a checked box containing exactly one
/// nonsingular real root of the retained original equations. The proposal
/// itself remains approximate; the immutable receipt states the exact scope.
#[derive(Debug, Clone)]
pub struct CertifiedRootOutcome2 {
    approximation: [f64; 2],
    certificate: fsym_core::RootBoxReceipt2,
}

impl CertifiedRootOutcome2 {
    pub fn approximation(&self) -> &[f64; 2] {
        &self.approximation
    }

    pub fn certificate(&self) -> &fsym_core::RootBoxReceipt2 {
        &self.certificate
    }
}

/// Certify a unique nonsingular real root in a caller-declared box around the
/// binary64 proposal. Both center and positive radii are interpreted as exact
/// binary64 rationals. A compiled Jacobian only proposes a preconditioner;
/// fsym-core independently recomputes interval values and derivatives from
/// the original expressions on the whole box, checking all C1 obligations.
pub fn certify_root_2x2(
    system: &VerifiedResidualSystem,
    approximation: [f64; 2],
    radii: [f64; 2],
    precision_digits: u32,
) -> Result<CertifiedRootOutcome2, ConsumerError> {
    validate_root_box_request(&radii, precision_digits)?;
    check_system_shape(system.compiled())?;
    require_finite(&approximation, "root proposal")?;
    let mut values = [0.0; 2];
    let mut jacobian = [0.0; 4];
    system
        .compiled()
        .try_eval_system(&approximation, &mut values, &mut jacobian)
        .map_err(|error| ConsumerError::Evaluation(error.to_string()))?;
    require_finite(&values, "root proposal residual")?;
    require_finite(&jacobian, "root proposal Jacobian")?;
    let exact = |value| {
        exact_binary64_rational(value)
            .ok_or_else(|| ConsumerError::Evaluation("nonfinite root-box input".into()))
    };
    let center = [exact(approximation[0])?, exact(approximation[1])?];
    let domain = [
        RealBall::new(center[0].clone(), exact(radii[0])?)
            .map_err(|error| ConsumerError::Certification(error.to_string()))?,
        RealBall::new(center[1].clone(), exact(radii[1])?)
            .map_err(|error| ConsumerError::Certification(error.to_string()))?,
    ];
    // Invert the rounded Jacobian exactly as a rational matrix. This avoids
    // floating determinant overflow/underflow and rounding the inverse again.
    // No accuracy claim about this proposal is needed by the verifier.
    let j = [
        [exact(jacobian[0])?, exact(jacobian[1])?],
        [exact(jacobian[2])?, exact(jacobian[3])?],
    ];
    let determinant = &j[0][0] * &j[1][1] - &j[0][1] * &j[1][0];
    if determinant == BigRational::from_integer(BigInt::from(0)) {
        return Err(ConsumerError::SingularJacobian(
            "root-box preconditioner proposal is singular".into(),
        ));
    }
    let inverse = BigRational::from_integer(BigInt::from(1)) / determinant;
    let preconditioner = [
        [&j[1][1] * &inverse, -j[0][1].clone() * &inverse],
        [-j[1][0].clone() * &inverse, &j[0][0] * &inverse],
    ];
    let witness = fsym_core::RootBoxWitness2 {
        domain,
        center,
        preconditioner,
        precision_digits,
    };
    let residuals: &[Expr; 2] = system
        .residuals()
        .try_into()
        .map_err(|_| ConsumerError::UnsupportedShape)?;
    let variables: &[Symbol; 2] = system
        .vars()
        .try_into()
        .map_err(|_| ConsumerError::UnsupportedShape)?;
    let certificate = fsym_core::certify_root_box_2x2(residuals, variables, &witness)
        .map_err(|error| ConsumerError::Certification(error.to_string()))?;
    Ok(CertifiedRootOutcome2 {
        approximation,
        certificate,
    })
}

/// Execute the pinned Newton consumer, then check existence and uniqueness of
/// a root in the requested box. Floating convergence alone cannot publish an
/// accepted outcome; the independent core checker must succeed.
pub fn newton_refine_root_box_2x2(
    system: &VerifiedResidualSystem,
    initial: [f64; 2],
    guards: &NewtonGuards,
    radii: [f64; 2],
    precision_digits: u32,
) -> Result<CertifiedRootOutcome2, ConsumerError> {
    validate_root_box_request(&radii, precision_digits)?;
    let outcome = newton_refine_2x2(system.compiled(), initial, guards)?;
    certify_root_2x2(system, outcome.root, radii, precision_digits)
}

/// Execute the pinned FrankenSciPy consumer and independently verify a unique
/// nonsingular real root in the requested box using the same retained source.
pub fn frankenscipy_fsolve_root_box(
    system: &VerifiedResidualSystem,
    initial: &[f64; 2],
    radii: [f64; 2],
    precision_digits: u32,
) -> Result<CertifiedRootOutcome2, ConsumerError> {
    validate_root_box_request(&radii, precision_digits)?;
    let proposal = frankenscipy_fsolve(system.compiled(), initial)?;
    certify_root_2x2(system, proposal, radii, precision_digits)
}

fn validate_root_box_request(radii: &[f64; 2], precision: u32) -> Result<(), ConsumerError> {
    if radii
        .iter()
        .any(|radius| !radius.is_finite() || *radius <= 0.0)
    {
        return Err(ConsumerError::Certification(
            "root-box radii must be finite and strictly positive".into(),
        ));
    }
    if !(8..=fsym_core::MAX_ROOT_BOX_PRECISION).contains(&precision) {
        return Err(ConsumerError::Certification(
            "root-box precision is outside the bounded checker envelope".into(),
        ));
    }
    Ok(())
}
