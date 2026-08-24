//! Polynomial Identity Testing (PIT) and verified identity certificates (WS08).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::multivariate::MultivariatePoly;
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::BudgetMeter;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_evidence::{EvidenceEnvelope, VerificationReceipt};
use fsym_id::ReceiptId;
use fsym_outcome::EvidenceClass;
use fsym_proof_kernel::{Claim, ProofKernel, verify_derivation_independent};
use num_traits::Zero;
use std::sync::Arc;

/// Concrete witness point witnessing that a polynomial is non-zero: $P(\mathbf{w}) \neq 0$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonZeroWitness {
    pub point: Vec<BigRational>,
    pub evaluated_value: BigRational,
}

/// Exact polynomial identity check: returns true iff $P \equiv 0$ in canonical sparse representation.
pub fn is_identically_zero(poly: &MultivariatePoly) -> bool {
    poly.is_zero()
}

/// Checks if two multivariate polynomials are symbolically identical: $P \equiv Q$.
pub fn are_identically_equal(
    a: &MultivariatePoly,
    b: &MultivariatePoly,
) -> Result<bool, PolyError> {
    let diff = a.sub(b)?;
    Ok(diff.is_zero())
}

/// Schwartz-Zippel randomized polynomial identity test.
///
/// Evaluates $P$ at deterministic pseudorandom integer grid points.
/// If any point evaluates to non-zero, returns `Ok(Some(NonZeroWitness))` (certifying $P \not\equiv 0$).
/// If all points evaluate to zero and $P$ is identically zero, returns `Ok(None)`.
pub fn schwartz_zippel_test(
    poly: &MultivariatePoly,
    num_trials: usize,
    seed: u64,
) -> Result<Option<NonZeroWitness>, PolyError> {
    if poly.is_zero() {
        return Ok(None);
    }
    let n_vars = poly.generators.len();
    if n_vars == 0 {
        if poly.is_zero() {
            return Ok(None);
        } else {
            let val = poly
                .terms
                .get(&Vec::new())
                .cloned()
                .unwrap_or_else(BigRational::zero);
            return Ok(Some(NonZeroWitness {
                point: Vec::new(),
                evaluated_value: val,
            }));
        }
    }

    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    for _ in 0..num_trials.max(1) {
        let mut sample_point = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val_i64 = ((state >> 33) % 1000) as i64 + 1; // [1, 1000]
            sample_point.push(BigRational::from_integer(BigInt::from(val_i64)));
        }

        let evaluated = poly.eval(&sample_point)?;
        if !evaluated.is_zero() {
            return Ok(Some(NonZeroWitness {
                point: sample_point,
                evaluated_value: evaluated,
            }));
        }
    }

    // Double check exact representation if no witness found
    if poly.is_zero() {
        Ok(None)
    } else {
        // Fallback: evaluate at [2, 3, 5, 7, ...]
        let primes = [2i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        let mut deterministic_point = Vec::with_capacity(n_vars);
        for i in 0..n_vars {
            let p = primes[i % primes.len()];
            deterministic_point.push(BigRational::from_integer(BigInt::from(p)));
        }
        let val = poly.eval(&deterministic_point)?;
        if !val.is_zero() {
            Ok(Some(NonZeroWitness {
                point: deterministic_point,
                evaluated_value: val,
            }))
        } else {
            Ok(None)
        }
    }
}

/// Verifies a polynomial algebraic identity $A \equiv B$ and emits an independent evidence envelope.
pub fn verify_polynomial_identity<M: BudgetMeter>(
    lhs_expr: &Expr,
    rhs_expr: &Expr,
    generators: &[Symbol],
    context: &Arc<ImmutableAssumptionsSnapshot>,
    meter: &mut M,
) -> Result<EvidenceEnvelope, PolyError> {
    let poly_lhs = MultivariatePoly::from_expr(lhs_expr, generators)?;
    let poly_rhs = MultivariatePoly::from_expr(rhs_expr, generators)?;

    let diff = poly_lhs.sub(&poly_rhs)?;
    if !diff.is_zero() {
        return Err(PolyError::IdentityCheckFailed(format!(
            "Polynomial identity failed: LHS `{lhs_expr}` - RHS `{rhs_expr}` = `{diff}` != 0"
        )));
    }

    let claim = Claim::AlgebraicIdentity {
        lhs: lhs_expr.clone(),
        rhs: rhs_expr.clone(),
    };

    let mut kernel = ProofKernel::new((**context).clone());
    let step_id = kernel
        .prove_definitional_reduction(
            lhs_expr.clone(),
            rhs_expr.clone(),
            "polynomial_ring_equivalence",
            meter,
        )
        .map_err(|e| PolyError::General(e.to_string()))?;

    let derivation_tree = kernel
        .export_derivation(step_id)
        .map_err(|e| PolyError::General(e.to_string()))?;

    verify_derivation_independent(&derivation_tree, context).map_err(|e| {
        PolyError::General(format!("Independent verifier rejected derivation: {e}"))
    })?;

    let receipt_id = ReceiptId::new(1).map_err(|e| PolyError::General(e.to_string()))?;
    let receipt = VerificationReceipt::issue(
        receipt_id,
        &claim,
        EvidenceClass::KernelProved,
        "fsym-polys.v1",
        1,
        Some(derivation_tree.digest()),
    );

    Ok(EvidenceEnvelope::new(
        claim,
        EvidenceClass::KernelProved,
        receipt,
        Some(derivation_tree),
    ))
}
