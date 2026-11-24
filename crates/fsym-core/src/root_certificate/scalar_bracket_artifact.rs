//! Source-bound scalar root existence and portable exact sign brackets.
//!
//! The generator supplies only an exact subinterval. The independent core
//! checker establishes continuity on the complete original interval, containment,
//! requested width, and opposite endpoint signs or an exact endpoint zero.
//! Repeated and nonsmooth roots are allowed; uniqueness, multiplicity and complete
//! enumeration are separate claims that this receipt never grants.

#![forbid(unsafe_code)]

use super::artifact::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, charge, check_problem_sources,
    encode, encoding_error,
};
use crate::{
    BigRational, Expr, ScalarRootBracketError, ScalarRootBracketReceipt, ScalarRootBracketWitness,
    Symbol, validate_scalar_root_bracket_request, verify_scalar_root_bracket,
};
use fsym_budget::BudgetMeter;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error;

/// Version of the source, interval, tolerance and context-bound bracket format.
pub const SCALAR_ROOT_BRACKET_SCHEMA_VERSION: u32 = 1;
/// Maximum encoded bracket, including its original typed source.
pub const MAX_SCALAR_ROOT_BRACKET_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

/// A failed proposal or replay grants no root existence evidence.
#[derive(Debug, Error)]
pub enum ScalarBracketError {
    // Keep immediate causes visible to host operational-error mappers.
    #[error("{0}")]
    Certification(#[from] CertificationError),
    #[error("{0}")]
    Bracket(#[from] ScalarRootBracketError),
    #[error("scalar root brackets require exactly one original residual and one variable")]
    ProblemDimension,
    #[error("scalar root bracket bounds or tolerance differ from the expected statement")]
    StatementMismatch,
}

fn scalar_source<'a>(
    residuals: &'a [Expr],
    variables: &'a [Symbol],
) -> Result<(&'a Expr, &'a Symbol), ScalarBracketError> {
    if residuals.len() != 1 || variables.len() != 1 {
        return Err(ScalarBracketError::ProblemDimension);
    }
    Ok((&residuals[0], &variables[0]))
}

#[derive(Serialize)]
struct BracketArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    lower: &'a BigRational,
    upper: &'a BigRational,
    absolute_tolerance: &'a BigRational,
    witness: &'a ScalarRootBracketWitness,
}

// Borrow untrusted expression and rational payloads until they match the
// receiver's admitted statement. Deserializing an attacker's source merely to
// compare it would cross the allocation/arithmetic boundary too early.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    #[serde(borrow)]
    residuals: &'a RawValue,
    #[serde(borrow)]
    variables: &'a RawValue,
    #[serde(borrow)]
    lower: &'a RawValue,
    #[serde(borrow)]
    upper: &'a RawValue,
    #[serde(borrow)]
    absolute_tolerance: &'a RawValue,
    #[serde(borrow)]
    witness: &'a RawValue,
}

fn charge_number(
    value: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), CertificationError> {
    let limbs = value.numer().limb_count() + value.denom().limb_count();
    charge(meter, limbs + 1, 128 + 16 * limbs, 2)
}

fn charge_statement(
    lower: &BigRational,
    upper: &BigRational,
    tolerance: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), ScalarBracketError> {
    validate_scalar_root_bracket_request(lower, upper, tolerance)?;
    for value in [lower, upper, tolerance] {
        charge_number(value, meter)?;
    }
    Ok(())
}

fn charge_witness(
    witness: &ScalarRootBracketWitness,
    tolerance: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), ScalarBracketError> {
    validate_scalar_root_bracket_request(&witness.lower, &witness.upper, tolerance)?;
    if !(8..=128).contains(&witness.precision_digits) {
        return Err(ScalarRootBracketError::ResourceLimit("bracket precision").into());
    }
    charge_number(&witness.lower, meter)?;
    charge_number(&witness.upper, meter)?;
    Ok(())
}

/// Export an accepted bracket's original statement and untrusted witness.
///
/// The artifact contains no root value, enclosure, endpoint evaluations or
/// accepted flag. Raw source structure, typed variable identity, ordered request
/// endpoints, exact tolerance and context are bound. Receiving hosts independently
/// supply the expected statement and re-evaluate the source before accepting it.
pub fn export_scalar_root_bracket(
    expected_residuals: &[Expr],
    expected_variables: &[Symbol],
    receipt: &ScalarRootBracketReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, ScalarBracketError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    let (source, variable) = scalar_source(expected_residuals, expected_variables)?;
    check_problem_sources(expected_residuals, expected_variables, meter)?;
    if source != receipt.expression() || variable != receipt.variable() {
        return Err(CertificationError::SourceMismatch.into());
    }
    charge_statement(
        receipt.lower(),
        receipt.upper(),
        receipt.absolute_tolerance(),
        meter,
    )?;
    charge_witness(receipt.witness(), receipt.absolute_tolerance(), meter)?;
    Ok(encode(
        &BracketArtifact {
            schema_version: SCALAR_ROOT_BRACKET_SCHEMA_VERSION,
            context_digest,
            residuals: expected_residuals,
            variables: expected_variables,
            lower: receipt.lower(),
            upper: receipt.upper(),
            absolute_tolerance: receipt.absolute_tolerance(),
            witness: receipt.witness(),
        },
        meter,
    )?)
}

/// Replay a portable bracket against the receiver's exact original statement.
///
/// Byte/depth limits and cumulative decoder costs precede borrowed parsing.
/// Source, coordinate, interval, tolerance and context mismatches refuse before
/// witness construction. Canonical re-encoding rejects alternative encodings,
/// extra fields and cached evidence. Only the independent core continuity and
/// endpoint-sign checker can grant a receipt; replay never runs candidate search.
#[allow(clippy::too_many_arguments)]
pub fn replay_scalar_root_bracket(
    bytes: &[u8],
    expected_residuals: &[Expr],
    expected_variables: &[Symbol],
    expected_lower: &BigRational,
    expected_upper: &BigRational,
    expected_tolerance: &BigRational,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<ScalarRootBracketReceipt, ScalarBracketError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    let (source, variable) = scalar_source(expected_residuals, expected_variables)?;
    if bytes.len() > MAX_SCALAR_ROOT_BRACKET_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_SCALAR_ROOT_BRACKET_BYTES,
        }
        .into());
    }
    charge_statement(expected_lower, expected_upper, expected_tolerance, meter)?;
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
    let wire: UntrustedArtifact<'_> =
        serde_json::from_slice(bytes).map_err(|error| encoding_error(&error))?;
    if wire.schema_version != SCALAR_ROOT_BRACKET_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: SCALAR_ROOT_BRACKET_SCHEMA_VERSION,
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
    check_problem_sources(expected_residuals, expected_variables, meter)?;
    let sources = encode(expected_residuals, meter)?;
    let variables = encode(expected_variables, meter)?;
    charge(meter, (sources.len() + variables.len()) as u64 + 1, 0, 0)?;
    if sources.as_slice() != wire.residuals.get().as_bytes()
        || variables.as_slice() != wire.variables.get().as_bytes()
    {
        return Err(CertificationError::SourceMismatch.into());
    }
    for (expected, received) in [
        (expected_lower, wire.lower),
        (expected_upper, wire.upper),
        (expected_tolerance, wire.absolute_tolerance),
    ] {
        let canonical = encode(expected, meter)?;
        charge(meter, canonical.len() as u64 + 1, 0, 0)?;
        if canonical.as_slice() != received.get().as_bytes() {
            return Err(ScalarBracketError::StatementMismatch);
        }
    }
    // The core witness decoder bounds both rational heights and precision.
    // Precharge the bounded synchronous normalization unit, then use the core
    // checker's per-node/arithmetic safe points for independent replay.
    let witness_bytes = wire.witness.get().len() as u64;
    charge(
        meter,
        witness_bytes.saturating_mul(witness_bytes + 1),
        witness_bytes * 32 + 1024,
        witness_bytes + 1,
    )?;
    let witness: ScalarRootBracketWitness =
        serde_json::from_str(wire.witness.get()).map_err(|error| encoding_error(&error))?;
    meter.checkpoint().map_err(CertificationError::from)?;
    charge_witness(&witness, expected_tolerance, meter)?;
    let canonical = encode(
        &BracketArtifact {
            schema_version: SCALAR_ROOT_BRACKET_SCHEMA_VERSION,
            context_digest: expected_context,
            residuals: expected_residuals,
            variables: expected_variables,
            lower: expected_lower,
            upper: expected_upper,
            absolute_tolerance: expected_tolerance,
            witness: &witness,
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical.into());
    }
    // Preserve the facade's independent verification boundary. Neither
    // canonical bytes nor a supplied witness grant mathematical authority.
    meter.checkpoint().map_err(CertificationError::from)?;
    Ok(verify_scalar_root_bracket(
        source,
        variable,
        expected_lower,
        expected_upper,
        expected_tolerance,
        &witness,
        meter,
    )?)
}
