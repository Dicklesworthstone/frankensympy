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
use std::sync::Arc;

fn objective_source<M: BudgetMeter + ?Sized>(
    system: &VerifiedResidualSystem,
    meter: &mut M,
) -> Result<Expr, ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if system.vars().len() > MAX_OBJECTIVE_VARIABLES {
        return Err(resource("variable count", MAX_OBJECTIVE_VARIABLES));
    }
    let residuals = system.residuals();
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
    for residual in residuals {
        let remaining = MAX_OBJECTIVE_SOURCE_NODES.saturating_sub(total);
        inspect_expression(
            residual,
            MAX_OBJECTIVE_SOURCE_DEPTH - 3,
            remaining,
            &mut total,
            meter,
        )?;
    }
    charge_buffer::<Expr, _>(residuals.len(), meter)?;
    let mut squares = Vec::new();
    squares
        .try_reserve_exact(residuals.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    for residual in residuals {
        charge_expression_copy(residual, meter)?;
        charge_compilation_work(meter, 4, 2 * std::mem::size_of::<Expr>() + 64, 3)?;
        // Raw nodes intentionally retain zero rows, repeated rows, held source
        // operations and original domain obligations. Smart constructors could
        // erase a pole behind a zero before certification sees it.
        squares.push(Expr::Pow(
            Arc::new(residual.clone()),
            Arc::new(Expr::Integer(BigInt::from(2))),
        ));
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
        let result = (|| {
            let candidate = self.propose_least_squares_objective_with_meter(meter)?;
            self.verify_least_squares_objective_with_meter(candidate, meter)
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
        let result = (|| {
            let source = objective_source(self, meter)?;
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
        let result = (|| {
            let source = objective_source(self, meter)?;
            VerifiedObjective::try_from_candidate_with_meter(&source, self.vars(), candidate, meter)
        })();
        finish_compilation(result, meter)
    }
}
