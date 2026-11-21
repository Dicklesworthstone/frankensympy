//! Context-bound transport shared by native sessions and host adapters.
//!
//! This is the existing compiled-program envelope and canonical candidate
//! format. Decoding checks byte/schema/context boundaries and returns an
//! untrusted candidate only. The receiver supplies its own exact source and
//! ordered typed coordinates to the independent admission helpers before use.
//! No accepted receipt, derived evaluation plan or numerical cache is imported.

#![forbid(unsafe_code)]

use super::{
    ProgramAdmissionError, ProgramArtifactError, decode_objective_candidate,
    decode_residual_candidate, encode_objective_candidate, encode_residual_candidate,
};
use crate::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use crate::sparse_jacobian::SparseJacobian;
use crate::verified::VerifiedResidualSystem;
use fsym_budget::BudgetMeter;
use fsym_core::root_certificate::artifact::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, charge, encode, encoding_error,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error;

pub const COMPILED_PROGRAM_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const MAX_COMPILED_PROGRAM_ARTIFACT_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

#[derive(Debug, Error)]
pub enum ProgramTransportError {
    #[error("compiled program envelope refused: {0}")]
    Envelope(#[from] CertificationError),
    #[error("compiled program candidate refused: {0}")]
    Candidate(#[from] ProgramArtifactError),
}

/// A complete replay can fail while decoding its transport or independently
/// checking its mathematical evidence against the receiving problem.
#[derive(Debug, Error)]
pub enum ProgramReplayError {
    #[error("compiled program transport refused: {0}")]
    Transport(#[from] ProgramTransportError),
    #[error("compiled program admission refused: {0}")]
    Admission(#[from] ProgramAdmissionError),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    #[serde(borrow)]
    candidate: &'a RawValue,
}

fn wrap_candidate(
    bytes: &[u8],
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, ProgramTransportError> {
    charge(meter, bytes.len() as u64 + 1, 1024, 1)?;
    let candidate = serde_json::from_slice(bytes).map_err(|error| encoding_error(&error))?;
    Ok(encode(
        &ProgramArtifact {
            schema_version: COMPILED_PROGRAM_ARTIFACT_SCHEMA_VERSION,
            context_digest,
            candidate,
        },
        meter,
    )?)
}

fn unwrap_candidate<'a>(
    bytes: &'a [u8],
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<&'a [u8], ProgramTransportError> {
    charge(meter, 1, 0, 0)?;
    if bytes.len() > MAX_COMPILED_PROGRAM_ARTIFACT_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_COMPILED_PROGRAM_ARTIFACT_BYTES,
        }
        .into());
    }
    // Bound the complete nested envelope before serde can traverse it. The
    // inner codec additionally checks sequence lengths and arithmetic payloads
    // before constructing any expressions, rationals, proofs or bytecode.
    let mut nesting = JsonNesting::default();
    for chunk in bytes.chunks(512) {
        charge(meter, chunk.len() as u64, 0, 0)?;
        nesting.scan(chunk)?;
    }
    charge(
        meter,
        bytes.len() as u64 + 1,
        bytes.len() as u64 * 4 + 1024,
        64,
    )?;
    let wire: ProgramArtifact<'_> =
        serde_json::from_slice(bytes).map_err(|error| encoding_error(&error))?;
    if wire.schema_version != COMPILED_PROGRAM_ARTIFACT_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: COMPILED_PROGRAM_ARTIFACT_SCHEMA_VERSION,
        }
        .into());
    }
    if wire.context_digest != expected_context {
        return Err(CertificationError::UniverseMismatch {
            found: wire.context_digest,
            expected: expected_context,
        }
        .into());
    }
    let canonical = encode(&wire, meter)?;
    charge(meter, (canonical.len() + bytes.len()) as u64 + 1, 0, 0)?;
    if canonical != bytes {
        return Err(CertificationError::NonCanonical.into());
    }
    Ok(wire.candidate.get().as_bytes())
}

/// Export a checked residual program as a context-bound untrusted proposal.
pub fn export_residual_program(
    program: &VerifiedResidualSystem,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, ProgramTransportError> {
    let candidate = encode_residual_candidate(program, meter)?;
    wrap_candidate(&candidate, context_digest, meter)
}

/// Decode a canonical residual proposal without granting mathematical evidence.
/// Use `admit_residual_candidate` with the receiver's own original expressions
/// and ordered typed variables before executing or publishing the program.
pub fn decode_residual_program(
    bytes: &[u8],
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<SparseJacobian, ProgramTransportError> {
    let candidate = unwrap_candidate(bytes, expected_context, meter)?;
    Ok(decode_residual_candidate(candidate, meter)?)
}

/// Export a checked objective program without receipts, plans or cached values.
pub fn export_objective_program(
    program: &VerifiedObjective,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, ProgramTransportError> {
    let candidate = encode_objective_candidate(program, meter)?;
    wrap_candidate(&candidate, context_digest, meter)
}

/// Decode a canonical objective proposal; independent source-bound gradient,
/// Hessian and bytecode admission remains mandatory before use.
pub fn decode_objective_program(
    bytes: &[u8],
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<ObjectiveCandidate, ProgramTransportError> {
    let candidate = unwrap_candidate(bytes, expected_context, meter)?;
    Ok(decode_objective_candidate(candidate, meter)?)
}
