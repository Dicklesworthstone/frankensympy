//! Native facade for the shared source-bound candidate root-search transport.
//!
//! The calculus codec retains the existing schema and canonical bytes. A
//! restored checkpoint remains candidate state until independent verification.

#![forbid(unsafe_code)]

use crate::certification::{CertificationError, ResidualProblem};
use fsym_budget::BudgetMeter;
use fsym_calculus::root_isolation::checkpoint;
use fsym_calculus::root_isolation::{RootIsolationOptions, RootIsolationSearch};
use fsym_core::RealBall;

pub use checkpoint::{
    MAX_ROOT_SEARCH_CHECKPOINT_BYTES, ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION,
    export_root_search_checkpoint,
};

/// Restore against the receiving native problem's frozen exact statement.
pub fn restore_root_search_checkpoint(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_domain: &[RealBall],
    expected_options: RootIsolationOptions,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<RootIsolationSearch, CertificationError> {
    checkpoint::restore_root_search_checkpoint(
        bytes,
        expected_problem.sources(),
        expected_domain,
        expected_options,
        expected_context,
        meter,
    )
}
