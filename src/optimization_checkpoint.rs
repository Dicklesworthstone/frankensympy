//! Native facade for portable source-bound box-optimization continuations.
//!
//! The shared calculus codec preserves the existing schema and canonical bytes.
//! Restored state remains a candidate until independent complete-cover replay.

#![forbid(unsafe_code)]

use crate::objective::ObjectiveProblem;
use crate::optimization::OptimizationError;
use fsym_budget::BudgetMeter;
use fsym_calculus::box_optimization::checkpoint_transport::{self, OptimizationCheckpointError};
use fsym_calculus::box_optimization::{BoxOptimizationOptions, BoxOptimizationSearch};
use fsym_core::{BigRational, BoxExtremumKind, RealBall};

pub use checkpoint_transport::{
    BOX_OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION, MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES,
};

fn native_error(error: OptimizationCheckpointError) -> OptimizationError {
    match error {
        OptimizationCheckpointError::Certification(error) => error.into(),
        OptimizationCheckpointError::Optimization(error) => error.into(),
        OptimizationCheckpointError::StatementMismatch => OptimizationError::StatementMismatch,
    }
}

/// Export candidate progress, retaining the live search's consumed work.
/// A checkpoint supplies no accepted bound, cached arithmetic or receipt.
pub fn export_box_optimization_checkpoint(
    search: &mut BoxOptimizationSearch,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, OptimizationError> {
    checkpoint_transport::export_box_optimization_checkpoint(search, context_digest, meter)
        .map_err(native_error)
}

/// Restore against the receiving native problem's immutable exact statement.
#[allow(clippy::too_many_arguments)]
pub fn restore_box_optimization_checkpoint(
    bytes: &[u8],
    expected_problem: &ObjectiveProblem,
    expected_domain: &[RealBall],
    expected_kind: BoxExtremumKind,
    expected_tolerance: &BigRational,
    expected_options: BoxOptimizationOptions,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<BoxOptimizationSearch, OptimizationError> {
    checkpoint_transport::restore_box_optimization_checkpoint(
        bytes,
        expected_problem.objective(),
        expected_problem.variables(),
        expected_domain,
        expected_kind,
        expected_tolerance,
        expected_options,
        expected_context,
        meter,
    )
    .map_err(native_error)
}
