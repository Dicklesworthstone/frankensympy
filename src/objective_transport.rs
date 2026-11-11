//! Portable stationary-point witnesses bound to the receiver's objective and box.
//!
//! The format transports only exact typed inputs and an untrusted witness.
//! Classification, derivative bounds and acceptance are recomputed by the
//! dependency-separated original-source checker; no stored conclusion is read.

#![forbid(unsafe_code)]

use crate::certification::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, charge, charge_witness_encoding,
    check_problem_sources, encode, encoding_error,
};
use crate::isolation_transport::check_domain;
use crate::objective::ObjectiveProblem;
use fsym_budget::BudgetMeter;
use fsym_core::{Expr, RealBall, RootBoxWitness, StationaryPointReceipt, Symbol};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// Version of the canonical source-, coordinate-, context- and box-bound format.
pub const STATIONARY_CERTIFICATE_SCHEMA_VERSION: u32 = 1;
pub const MAX_STATIONARY_CERTIFICATE_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

#[derive(Serialize)]
struct StationaryArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    objective: &'a Expr,
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    witness: &'a RootBoxWitness,
}

// Retain arithmetic-bearing input as borrowed bytes until it matches the
// receiver's bounded typed statement. Never deserialize an attacker's Expr.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    #[serde(borrow)]
    objective: &'a RawValue,
    #[serde(borrow)]
    variables: &'a RawValue,
    #[serde(borrow)]
    domain: &'a RawValue,
    #[serde(borrow)]
    witness: &'a RawValue,
}

/// Export the exact statement and witness of an independently checked point.
///
/// No minimum/maximum/saddle label, enclosure or acceptance flag is serialized.
/// The receiver supplies its own objective, ordered typed coordinates, closed
/// box and assumptions context, and independently reclassifies the point.
pub fn export_stationary_certificate(
    receipt: &StationaryPointReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, CertificationError> {
    check_problem_sources(
        std::slice::from_ref(receipt.objective()),
        receipt.variables(),
        meter,
    )?;
    check_domain(&receipt.witness().domain, receipt.variables().len(), meter)?;
    charge_witness_encoding(receipt.witness(), meter)?;
    encode(
        &StationaryArtifact {
            schema_version: STATIONARY_CERTIFICATE_SCHEMA_VERSION,
            context_digest,
            objective: receipt.objective(),
            variables: receipt.variables(),
            domain: &receipt.witness().domain,
            witness: receipt.witness(),
        },
        meter,
    )
}

/// Recompute a unique stationary point and its local classification.
///
/// The expected box is mandatory: a valid point in another region cannot
/// satisfy the receiver's request. Complete source and symbol identities are
/// compared before bounded witness decoding. JSON alternatives, duplicated
/// fields, noncanonical scalar encodings and unknown stored conclusions fail
/// closed. Invalid source domains remain invalid even when algebraic terms
/// cancel. All parsing, canonicalization and verification use the same meter.
pub fn replay_stationary_certificate(
    bytes: &[u8],
    expected_problem: &ObjectiveProblem,
    expected_domain: &[RealBall],
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<StationaryPointReceipt, CertificationError> {
    meter.checkpoint()?;
    if bytes.len() > MAX_STATIONARY_CERTIFICATE_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_STATIONARY_CERTIFICATE_BYTES,
        });
    }
    check_domain(expected_domain, expected_problem.variables().len(), meter)?;
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
    if wire.schema_version != STATIONARY_CERTIFICATE_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: STATIONARY_CERTIFICATE_SCHEMA_VERSION,
        });
    }
    if wire.context_digest != expected_context {
        return Err(CertificationError::UniverseMismatch {
            found: wire.context_digest,
            expected: expected_context,
        });
    }
    check_problem_sources(
        std::slice::from_ref(expected_problem.objective()),
        expected_problem.variables(),
        meter,
    )?;
    let source = encode(expected_problem.objective(), meter)?;
    let variables = encode(expected_problem.variables(), meter)?;
    let domain = encode(expected_domain, meter)?;
    charge(
        meter,
        (source.len() + variables.len() + domain.len()) as u64 + 1,
        0,
        0,
    )?;
    if source.as_slice() != wire.objective.get().as_bytes()
        || variables.as_slice() != wire.variables.get().as_bytes()
    {
        return Err(CertificationError::SourceMismatch);
    }
    if domain.as_slice() != wire.domain.get().as_bytes() {
        return Err(CertificationError::DomainMismatch);
    }
    // The core's dedicated witness decoder preflights dimensions, vector
    // lengths and limb counts before integer/rational construction. Charge
    // that bounded synchronous unit before entering it.
    let witness_bytes = wire.witness.get().len() as u64;
    charge(
        meter,
        witness_bytes.saturating_mul(witness_bytes + 1),
        witness_bytes * 32 + 1024,
        witness_bytes + 1,
    )?;
    let witness: RootBoxWitness =
        serde_json::from_str(wire.witness.get()).map_err(|error| encoding_error(&error))?;
    meter.checkpoint()?;
    charge_witness_encoding(&witness, meter)?;
    if witness.domain != expected_domain {
        return Err(CertificationError::DomainMismatch);
    }
    let canonical = encode(
        &StationaryArtifact {
            schema_version: STATIONARY_CERTIFICATE_SCHEMA_VERSION,
            context_digest: expected_context,
            objective: expected_problem.objective(),
            variables: expected_problem.variables(),
            domain: expected_domain,
            witness: &witness,
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical);
    }
    expected_problem.verify_stationary(&witness, meter)
}
