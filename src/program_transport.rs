//! Context-bound transport for untrusted compiled derivative program candidates.
//!
//! The payload contains original typed sources, ordered coordinates, derivative
//! proofs and bytecode. It never carries an authoritative verification receipt,
//! shared evaluation plan or numeric cache. Decoding only restores a candidate;
//! the receiver must independently admit it against its own registered source.

#![forbid(unsafe_code)]

use crate::certification::ResidualProblem;
use crate::objective::ObjectiveProblem;
use fsym_budget::BudgetMeter;
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::program_artifact::{admit_objective_candidate, admit_residual_candidate};
use fsym_calculus::sparse_hessian::VerifiedObjective;

// The same canonical context envelope serves Session and host adapters. These
// reexports retain the native API, exact wire bytes, errors and meter charges.
pub use fsym_calculus::program_artifact::context::{
    COMPILED_PROGRAM_ARTIFACT_SCHEMA_VERSION, MAX_COMPILED_PROGRAM_ARTIFACT_BYTES,
    ProgramReplayError, ProgramTransportError, decode_objective_program, decode_residual_program,
    export_objective_program, export_residual_program,
};

/// Restore a verified residual/Jacobian program against a receiving problem.
///
/// This complete facade checks the canonical context-bound transport, then
/// independently binds the original typed sources and ordered coordinates,
/// replays derivative evidence and rebuilds executable plans. A decoded
/// candidate or transported receipt alone cannot become a verified program.
///
/// Decoding and verification both consume the supplied cumulative meter. The
/// Session interface uses the same wire format with ordinary decoding and a
/// separate protected verification allowance.
pub fn replay_residual_program(
    bytes: &[u8],
    expected: &ResidualProblem,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedResidualSystem, ProgramReplayError> {
    let candidate = decode_residual_program(bytes, expected_context, meter)?;
    Ok(admit_residual_candidate(
        expected.residuals(),
        expected.variables(),
        candidate,
        meter,
    )?)
}

/// Restore a verified objective/gradient/Hessian program against its source.
///
/// Both derivative stages, sparse coverage and bytecode provenance are checked
/// independently. The resulting program retains original-domain guards for
/// every evaluation and product; its numerical outputs remain binary64
/// approximations. This operation grants no interval or optimization proof.
///
/// All transport and admission work consumes the same supplied meter. Use the
/// Session interface when decoding and protected verification must consume
/// different allowances on one cumulative ledger.
pub fn replay_objective_program(
    bytes: &[u8],
    expected: &ObjectiveProblem,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedObjective, ProgramReplayError> {
    let candidate = decode_objective_program(bytes, expected_context, meter)?;
    Ok(admit_objective_candidate(
        expected.objective(),
        expected.variables(),
        candidate,
        meter,
    )?)
}
