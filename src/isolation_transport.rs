//! Native facade for the shared independent complete-root isolation codec.
//!
//! The core implementation preserves canonical bytes and verifier-only
//! dependencies while serving native sessions and host adapters alike.

#![forbid(unsafe_code)]

use crate::certification::{CertificationError, ResidualProblem};
use fsym_budget::BudgetMeter;
use fsym_core::root_certificate::isolation_artifact;
use fsym_core::{RealBall, RootIsolationReceipt};

pub(crate) use isolation_artifact::check_domain;
pub use isolation_artifact::{ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION, export_root_isolation};

/// Replay against the receiving native problem's frozen exact statement.
pub fn replay_root_isolation(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_domain: &[RealBall],
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<RootIsolationReceipt, CertificationError> {
    isolation_artifact::replay_root_isolation(
        bytes,
        expected_problem.sources(),
        expected_domain,
        expected_context,
        meter,
    )
}
