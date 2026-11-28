//! Exact composition of an admitted residual system into a scalar objective.
//!
//! Every ordered residual occurs once in the held source `1/2 * sum(r_i^2)`.
//! The full Hessian therefore includes both J-transpose-J and the residual
//! curvature terms. No Gauss-Newton approximation enters derivative proofs or
//! the independent original-source stationary-point checker.

use super::{
    MAX_OBJECTIVE_SOURCE_DEPTH, MAX_OBJECTIVE_SOURCE_NODES, MAX_OBJECTIVE_VARIABLES,
    ObjectiveCandidate, ObjectiveCompilationError, VerifiedObjective, charge_buffer,
    charge_compilation_work, charge_expression_copy, finish_compilation, inspect_expression,
    resource,
};
use crate::{CompileError, VerifiedResidualSystem};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{BigInt, BigRational, Expr};
use num_traits::One;
use std::sync::Arc;

fn objective_source<M: BudgetMeter + ?Sized>(
    system: &VerifiedResidualSystem,
    weights: Option<&[BigRational]>,
    meter: &mut M,
) -> Result<Expr, ObjectiveCompilationError> {
    if system.vars().len() > MAX_OBJECTIVE_VARIABLES {
        charge_compilation_work(meter, 1, 0, 0)?;
        return Err(resource("variable count", MAX_OBJECTIVE_VARIABLES));
    }
    compose_source(system.residuals(), weights, meter)
}

/// Construct the held loss from the ordered original residuals without
/// compiling derivative programs. A zero factor or repeated row keeps all
/// its original domain obligations. An empty sum has exact value zero.
pub fn least_squares_objective_source<M: BudgetMeter + ?Sized>(
    residuals: &[Expr],
    meter: &mut M,
) -> Result<Expr, ObjectiveCompilationError> {
    compose_source(residuals, None, meter)
}

/// Construct one half of the ordered sum of `weight * residual^2`, using
/// exact nonnegative rational weights. A zero weight retains its row's domain.
/// Unit weights preserve the original unweighted source representation.
pub fn weighted_least_squares_objective_source<M: BudgetMeter + ?Sized>(
    residuals: &[Expr],
    weights: &[BigRational],
    meter: &mut M,
) -> Result<Expr, ObjectiveCompilationError> {
    compose_source(residuals, Some(weights), meter)
}

fn compose_source<M: BudgetMeter + ?Sized>(
    residuals: &[Expr],
    weights: Option<&[BigRational]>,
    meter: &mut M,
) -> Result<Expr, ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if weights.is_some_and(|values| values.len() != residuals.len()) {
        return Err(ObjectiveCompilationError::InvalidWeights(
            "one weight is required per residual",
        ));
    }
    if residuals.is_empty() {
        return Ok(Expr::from_i64(0));
    }
    // Three outer nodes (Mul, 1/2, Add) and two per residual (Pow, 2).
    // Preflight the complete transformed size before cloning any source.
    let mut total = residuals
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_add(3))
        .ok_or(CompileError::DimensionOverflow)?;
    if total > MAX_OBJECTIVE_SOURCE_NODES {
        return Err(resource("expression nodes", MAX_OBJECTIVE_SOURCE_NODES));
    }
    for (index, residual) in residuals.iter().enumerate() {
        let mut depth = MAX_OBJECTIVE_SOURCE_DEPTH - 3;
        if let Some(weights) = weights {
            let w = &weights[index];
            let bits = w.numer().bits().max(w.denom().bits());
            if bits > super::MAX_OBJECTIVE_PAYLOAD_BITS {
                return Err(resource(
                    "weight bits",
                    super::MAX_OBJECTIVE_PAYLOAD_BITS as usize,
                ));
            }
            charge_compilation_work(meter, 2 * bits.div_ceil(64) as usize + 1, 0, 0)?;
            if w.numer().is_negative() {
                return Err(ObjectiveCompilationError::InvalidWeights(
                    "weights must be nonnegative",
                ));
            }
            if !w.is_one() {
                total += 2; // Mul and the exact coefficient.
                depth -= 1;
            }
        }
        let remaining = MAX_OBJECTIVE_SOURCE_NODES.saturating_sub(total);
        inspect_expression(residual, depth, remaining, &mut total, meter)?;
    }
    charge_buffer::<Expr, _>(residuals.len(), meter)?;
    let mut squares = Vec::new();
    squares
        .try_reserve_exact(residuals.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    for (index, residual) in residuals.iter().enumerate() {
        charge_expression_copy(residual, meter)?;
        charge_compilation_work(meter, 4, 2 * std::mem::size_of::<Expr>() + 64, 3)?;
        // Raw nodes intentionally retain zero rows, repeated rows, held source
        // operations and original domain obligations. Smart constructors could
        // erase a pole behind a zero before certification sees it.
        let square = Expr::Pow(
            Arc::new(residual.clone()),
            Arc::new(Expr::Integer(BigInt::from(2))),
        );
        let term = match weights.map(|weights| &weights[index]) {
            Some(w) if !w.is_one() => {
                let bytes =
                    (w.numer().bits().div_ceil(64) + w.denom().bits().div_ceil(64)) as usize * 8;
                charge_buffer::<Expr, _>(2, meter)?;
                charge_compilation_work(meter, bytes / 8 + 2, bytes + 64, 2)?;
                let coefficient = if w.is_integer() {
                    Expr::Integer(w.numer().clone())
                } else {
                    Expr::Rational(w.clone())
                };
                Expr::Mul(vec![coefficient, square])
            }
            _ => square,
        };
        squares.push(term);
    }
    charge_buffer::<Expr, _>(2, meter)?;
    charge_compilation_work(meter, 8, 128, 4)?;
    Ok(Expr::Mul(vec![
        Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2))),
        Expr::Add(squares),
    ]))
}

impl VerifiedResidualSystem {
    /// Compile one half the sum of squares of every original residual.
    ///
    /// The returned immutable objective uses the same ordered typed variables
    /// and retains every original residual, including duplicate and zero rows.
    /// Both derivative orders are independently checked. Numerical evaluation
    /// remains approximate; a local minimum or stationary-point claim requires
    /// the separate original-source checker, just as for any other objective.
    pub fn least_squares_objective(&self) -> Result<VerifiedObjective, ObjectiveCompilationError> {
        self.least_squares_objective_with_meter(&mut Unbounded)
    }

    /// One cumulative ledger covers source composition, both derivative orders,
    /// independent admission and publication. Existing objective size/depth and
    /// work limits apply to the complete sum; they are not increased per row.
    /// Use the separate proposal and verification methods for protected pools.
    pub fn least_squares_objective_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        meter: &mut M,
    ) -> Result<VerifiedObjective, ObjectiveCompilationError> {
        self.compile_squared_loss(None, meter)
    }

    /// Compile and independently verify the full curvature of an exactly
    /// weighted fit, retaining zero-weight row domains and source order.
    pub fn weighted_least_squares_objective_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        weights: &[BigRational],
        meter: &mut M,
    ) -> Result<VerifiedObjective, ObjectiveCompilationError> {
        self.compile_squared_loss(Some(weights), meter)
    }

    fn compile_squared_loss<M: BudgetMeter + ?Sized>(
        &self,
        weights: Option<&[BigRational]>,
        meter: &mut M,
    ) -> Result<VerifiedObjective, ObjectiveCompilationError> {
        let result = (|| {
            let candidate = self.propose_squared_loss_with_meter(weights, meter)?;
            self.verify_squared_loss_with_meter(weights, candidate, meter)
        })();
        finish_compilation(result, meter)
    }

    /// Generate an untrusted second-order objective candidate. This does not
    /// grant proof authority to an imported residual's numerical values or to
    /// a least-squares solver's convergence flag.
    pub fn propose_least_squares_objective_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        meter: &mut M,
    ) -> Result<ObjectiveCandidate, ObjectiveCompilationError> {
        self.propose_squared_loss_with_meter(None, meter)
    }

    /// Construct a proposal using the supplied exact diagonal weights, or unit weights.
    pub fn propose_squared_loss_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        weights: Option<&[BigRational]>,
        meter: &mut M,
    ) -> Result<ObjectiveCandidate, ObjectiveCompilationError> {
        let result = (|| {
            let source = objective_source(self, weights, meter)?;
            ObjectiveCandidate::try_build_with_meter(&source, self.vars(), meter)
        })();
        finish_compilation(result, meter)
    }

    /// Independently bind and admit a proposal against this residual system.
    /// The exact held sum is reconstructed from the receiver's original rows;
    /// a candidate may not drop, reorder or rescale rows, change typed axes,
    /// substitute a Gauss-Newton Hessian, or replace retained source guards.
    pub fn verify_least_squares_objective_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        candidate: ObjectiveCandidate,
        meter: &mut M,
    ) -> Result<VerifiedObjective, ObjectiveCompilationError> {
        self.verify_squared_loss_with_meter(None, candidate, meter)
    }

    /// Reconstruct the ordered weighted loss before admitting its candidate.
    pub fn verify_squared_loss_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        weights: Option<&[BigRational]>,
        candidate: ObjectiveCandidate,
        meter: &mut M,
    ) -> Result<VerifiedObjective, ObjectiveCompilationError> {
        let result = (|| {
            let source = objective_source(self, weights, meter)?;
            VerifiedObjective::try_from_candidate_with_meter(&source, self.vars(), candidate, meter)
        })();
        finish_compilation(result, meter)
    }
}
