//! Portable proper-real integral partitions with independent source replay.
//!
//! Serialized partitions contain no accepted enclosure or cached arithmetic.
//! Receivers bind their own exact statement before decoding the witness, then
//! reevaluate the original source on every cell with the quadrature checker.

#![forbid(unsafe_code)]

use super::artifact::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, charge, check_problem_sources,
    encode, encoding_error,
};
use crate::{
    BigRational, Expr, IntegralEnclosureReceipt, MAX_QUADRATURE_INTERVALS, QuadratureError,
    QuadratureWitness, Symbol, validate_integral_request, verify_integral_enclosure,
};
use fsym_budget::BudgetMeter;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error;

/// Version of the exact source, interval and tolerance-bound partition format.
pub const INTEGRAL_CERTIFICATE_SCHEMA_VERSION: u32 = 1;
/// Maximum encoded certificate, sharing the bounded mathematical artifact envelope.
pub const MAX_INTEGRAL_CERTIFICATE_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

/// A refusal never supplies a partial integral or grants mathematical evidence.
#[derive(Debug, Error)]
pub enum IntegrationError {
    // Retain the wrapped error itself in the source chain. Transparent errors
    // would hide leaf allocation/resource failures from host error mappers.
    #[error("{0}")]
    Certification(#[from] CertificationError),
    #[error("{0}")]
    Quadrature(#[from] QuadratureError),
    #[error("definite integration requires exactly one original residual and one variable")]
    ProblemDimension,
    #[error("integral certificate bounds or absolute tolerance differ from the expected statement")]
    StatementMismatch,
}

fn integral_source<'a>(
    residuals: &'a [Expr],
    variables: &'a [Symbol],
) -> Result<(&'a Expr, &'a Symbol), IntegrationError> {
    if residuals.len() != 1 || variables.len() != 1 {
        return Err(IntegrationError::ProblemDimension);
    }
    Ok((&residuals[0], &variables[0]))
}

#[derive(Serialize)]
struct IntegralArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    lower: &'a BigRational,
    upper: &'a BigRational,
    absolute_tolerance: &'a BigRational,
    witness: &'a QuadratureWitness,
}

// Arithmetic-bearing payloads stay borrowed until they match the receiver's
// bounded statement. Foreign expressions and endpoint rationals are not built.
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
    absolute_tolerance: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), IntegrationError> {
    validate_integral_request(lower, upper, absolute_tolerance)?;
    for value in [lower, upper, absolute_tolerance] {
        charge_number(value, meter)?;
    }
    Ok(())
}

fn charge_partition(
    witness: &QuadratureWitness,
    tolerance: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), IntegrationError> {
    if witness.partition.is_empty() || witness.partition.len() > MAX_QUADRATURE_INTERVALS + 1 {
        return Err(QuadratureError::InvalidPartition("point count").into());
    }
    if !(8..=128).contains(&witness.precision_digits) {
        return Err(QuadratureError::ResourceLimit("quadrature precision").into());
    }
    for point in &witness.partition {
        validate_integral_request(point, point, tolerance)?;
        charge_number(point, meter)?;
    }
    Ok(())
}

/// Export an accepted integral's exact statement and untrusted partition.
///
/// The format contains no enclosure, cached value or accepted flag. The source
/// must match the expected statement exactly, including raw operand order and
/// symbol identity. Receivers supply their own ordered endpoints, tolerance
/// and assumption context before independent original-source witness replay.
pub fn export_integral_certificate(
    expected_residuals: &[Expr],
    expected_variables: &[Symbol],
    receipt: &IntegralEnclosureReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, IntegrationError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    let (source, variable) = integral_source(expected_residuals, expected_variables)?;
    check_problem_sources(expected_residuals, expected_variables, meter)?;
    if source != receipt.integrand() || variable != receipt.variable() {
        return Err(CertificationError::SourceMismatch.into());
    }
    charge_statement(
        receipt.lower(),
        receipt.upper(),
        receipt.absolute_tolerance(),
        meter,
    )?;
    charge_partition(receipt.witness(), receipt.absolute_tolerance(), meter)?;
    Ok(encode(
        &IntegralArtifact {
            schema_version: INTEGRAL_CERTIFICATE_SCHEMA_VERSION,
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

/// Replay a portable partition against the receiver's exact integral statement.
///
/// Byte length, JSON nesting and cumulative decoder costs are admitted before
/// the bounded witness decoder constructs partition rationals. Source, axis,
/// endpoint direction, tolerance and context substitutions refuse before witness
/// decoding. Canonical re-encoding rejects alternate encodings or ignored data.
/// Only the independent original-source checker grants a receipt; the adaptive
/// generator is never invoked during replay.
#[allow(clippy::too_many_arguments)]
pub fn replay_integral_certificate(
    bytes: &[u8],
    expected_residuals: &[Expr],
    expected_variables: &[Symbol],
    expected_lower: &BigRational,
    expected_upper: &BigRational,
    expected_tolerance: &BigRational,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<IntegralEnclosureReceipt, IntegrationError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    let (source, variable) = integral_source(expected_residuals, expected_variables)?;
    if bytes.len() > MAX_INTEGRAL_CERTIFICATE_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_INTEGRAL_CERTIFICATE_BYTES,
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
    if wire.schema_version != INTEGRAL_CERTIFICATE_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: INTEGRAL_CERTIFICATE_SCHEMA_VERSION,
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
            return Err(IntegrationError::StatementMismatch);
        }
    }
    // The witness decoder bounds partition length, scalar limbs and precision
    // before constructing rationals; precharge this bounded synchronous unit.
    let witness_bytes = wire.witness.get().len() as u64;
    charge(
        meter,
        witness_bytes.saturating_mul(witness_bytes + 1),
        witness_bytes * 32 + 1024,
        witness_bytes + 1,
    )?;
    let witness: QuadratureWitness =
        serde_json::from_str(wire.witness.get()).map_err(|error| encoding_error(&error))?;
    meter.checkpoint().map_err(CertificationError::from)?;
    charge_partition(&witness, expected_tolerance, meter)?;
    let canonical = encode(
        &IntegralArtifact {
            schema_version: INTEGRAL_CERTIFICATE_SCHEMA_VERSION,
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
    // Preserve the product facade's independent verification boundary. No
    // result of decoding or canonicalization has mathematical authority.
    meter.checkpoint().map_err(CertificationError::from)?;
    Ok(verify_integral_enclosure(
        source,
        variable,
        expected_lower,
        expected_upper,
        expected_tolerance,
        &witness,
        meter,
    )?)
}
