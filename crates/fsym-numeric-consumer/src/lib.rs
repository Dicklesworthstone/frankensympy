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

use fsym_calculus::CompiledResidualSystem;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
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
}

/// Result of a converged Newton refinement driven by the consumer solver.
#[derive(Debug, Clone)]
pub struct NewtonOutcome {
    /// Final iterate (the accepted approximate root).
    pub root: [f64; 2],
    /// Residual infinity-norm at the final iterate (native compiled lane).
    pub residual_norm: f64,
    /// Determinant of the Jacobian at the final iterate, as computed by
    /// the boundary for the nonsingularity certificate.
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
    if system.num_vars != 2 || system.num_residuals != 2 {
        return Err(ConsumerError::UnsupportedShape);
    }
    let mut x = initial;
    let mut residuals = [0.0_f64; 2];
    let mut jacobian_flat = [0.0_f64; 4];
    let mut last_norm = f64::INFINITY;
    let mut determinant = 0.0_f64;
    for iteration in 0..=guards.max_iterations {
        system
            .try_eval_residuals(&x, &mut residuals)
            .map_err(|error| ConsumerError::Evaluation(error.to_string()))?;
        last_norm = residuals
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
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
        system
            .try_eval_jacobian(&x, &mut jacobian_flat)
            .map_err(|error| ConsumerError::Evaluation(error.to_string()))?;
        let jacobian = [
            [jacobian_flat[0], jacobian_flat[1]],
            [jacobian_flat[2], jacobian_flat[3]],
        ];
        determinant = jacobian_det(&jacobian);
        let negated = [-residuals[0], -residuals[1]];
        // The pinned consumer solves the linear step; its typed refusal is
        // the singular-Jacobian condition.
        let [dx, dy] = fnp_linalg::solve_2x2(jacobian, negated)
            .map_err(|error| ConsumerError::SingularJacobian(error.to_string()))?;
        let step_norm = dx.abs().max(dy.abs());
        if !step_norm.is_finite() || step_norm > guards.max_step_norm {
            return Err(ConsumerError::NoConvergence {
                max_iterations: guards.max_iterations,
                last_residual_norm: last_norm,
            });
        }
        x[0] += dx;
        x[1] += dy;
    }
    Err(ConsumerError::NoConvergence {
        max_iterations: guards.max_iterations,
        last_residual_norm: last_norm,
    })
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
    if system.num_vars != 2 || system.num_residuals != 2 {
        return Err(ConsumerError::UnsupportedShape);
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(ConsumerError::Evaluation(
            "initial iterate contains non-finite values".into(),
        ));
    }
    let evaluation_error: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    let residuals = |point: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0_f64; system.num_residuals];
        if let Err(error) = system.try_eval_residuals(point, &mut out) {
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
    let result = fsci_opt::fsolve(residuals, initial)
        .map_err(|error| ConsumerError::Evaluation(error.to_string()))?;
    if let Some(message) = evaluation_error
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(ConsumerError::Evaluation(format!(
            "compiled residual evaluation failed during the consumer solve: {message}"
        )));
    }
    if !result.converged {
        return Err(ConsumerError::NoConvergence {
            max_iterations: 200,
            last_residual_norm: result
                .fun
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        });
    }
    if result.x.iter().any(|value| !value.is_finite()) {
        return Err(ConsumerError::Evaluation(
            "consumer returned a non-finite root".into(),
        ));
    }
    Ok([result.x[0], result.x[1]])
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

/// Certified consistency statement for an accepted consumer outcome:
/// substitutes the accepted iterate (as exact binary64 rationals) into
/// the residual expressions, evaluates each under certified ball
/// arithmetic (`evalf_ball`, 30-digit request), and asserts — with exact
/// rational comparison — that the residual infinity-norm midpoint is
/// bounded by `10^(-bound_decimal_places)` with ball radii far below it.
/// The accepted iterate is thus a certified approximate root, not an
/// unchecked consumer claim. Declare the bound per consumer: exact-solve
/// lanes tolerate tighter bounds than finite-difference consumers whose
/// own convergence tolerance dominates the achievable residual.
pub fn certified_residual_bound(
    residual_exprs: &[Expr],
    vars: &[Symbol],
    root: &[f64; 2],
    bound_decimal_places: u32,
) -> Result<(), String> {
    let bound = BigRational::new(BigInt::from(1), BigInt::from(10).pow(bound_decimal_places));
    let mut substitution = HashMap::new();
    for (index, variable) in vars.iter().enumerate() {
        let exact = exact_binary64_rational(root[index])
            .ok_or_else(|| format!("root component {index} is not finite"))?;
        substitution.insert(variable.clone(), Expr::Rational(exact));
    }
    for (index, residual) in residual_exprs.iter().enumerate() {
        let specialized = residual.subs(&substitution);
        let ball = specialized
            .evalf_ball(30)
            .map_err(|error| format!("residual {index} ball evaluation failed: {error}"))?;
        if ball.radius() > &bound {
            return Err(format!(
                "residual {index} certified radius {ball} exceeds the declared bound"
            ));
        }
        let midpoint = ball.midpoint().clone();
        let zero = BigRational::new(BigInt::from(0), BigInt::from(1));
        let magnitude = if midpoint >= zero {
            midpoint
        } else {
            -midpoint.clone()
        };
        if magnitude > bound {
            return Err(format!(
                "residual {index} certified magnitude {magnitude} exceeds the declared bound"
            ));
        }
    }
    Ok(())
}
