//! Source-bound proper real definite integration and portable partition witnesses.
//!
//! The adaptive generator proposes only a partition. The independent core
//! checker re-evaluates every original source child on every closed cell,
//! establishes complete coverage, and checks the absolute enclosure width.
//! Neither a symbolic antiderivative nor an exported acceptance flag is trusted.

#![forbid(unsafe_code)]

use crate::certification::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, ResidualProblem, charge,
    check_problem_sources, encode, encoding_error,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::quadrature::{QuadratureOptions, propose_integral_enclosure};
use fsym_core::{
    BigRational, Expr, IntegralEnclosureReceipt, MAX_QUADRATURE_INTERVALS, QuadratureError,
    QuadratureWitness, Symbol, validate_integral_request, verify_integral_enclosure,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error;

/// Version of the exact source, interval and tolerance-bound partition format.
pub const INTEGRAL_CERTIFICATE_SCHEMA_VERSION: u32 = 1;
/// Maximum encoded certificate, sharing the bounded session artifact envelope.
pub const MAX_INTEGRAL_CERTIFICATE_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

/// A refusal never supplies a partial integral or grants mathematical evidence.
#[derive(Debug, Error)]
pub enum IntegrationError {
    #[error(transparent)]
    Certification(#[from] CertificationError),
    #[error(transparent)]
    Quadrature(#[from] QuadratureError),
    #[error("definite integration requires exactly one original residual and one variable")]
    ProblemDimension,
    #[error("integral certificate bounds or absolute tolerance differ from the expected statement")]
    StatementMismatch,
}

fn integral_source(problem: &ResidualProblem) -> Result<(&Expr, &Symbol), IntegrationError> {
    if problem.residuals().len() != 1 || problem.variables().len() != 1 {
        return Err(IntegrationError::ProblemDimension);
    }
    Ok((&problem.residuals()[0], &problem.variables()[0]))
}

impl ResidualProblem {
    /// Propose a proper-real integral partition using ordinary work resources.
    ///
    /// This bounded lane requires one original residual and one typed axis.
    /// Endpoints and tolerance are exact rationals; every original source
    /// domain obligation is retained. A returned partition is only a candidate.
    /// Improper integrals, complex contours and principal values are outside
    /// this claim. Unsupported sources or exhausted resources refuse.
    pub fn propose_integral(
        &self,
        lower: &BigRational,
        upper: &BigRational,
        absolute_tolerance: &BigRational,
        options: QuadratureOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<QuadratureWitness, IntegrationError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let (source, variable) = integral_source(self)?;
        Ok(propose_integral_enclosure(
            source,
            variable,
            lower,
            upper,
            absolute_tolerance,
            options,
            meter,
        )?)
    }

    /// Independently certify complete coverage and the exact enclosure width.
    ///
    /// The caller may supply a protected verifier meter from the same ledger
    /// used for proposal. Reversed limits negate the accepted enclosure; equal
    /// limits still require the original source to be admitted at the point.
    pub fn verify_integral(
        &self,
        lower: &BigRational,
        upper: &BigRational,
        absolute_tolerance: &BigRational,
        witness: &QuadratureWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<IntegralEnclosureReceipt, IntegrationError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let (source, variable) = integral_source(self)?;
        Ok(verify_integral_enclosure(
            source,
            variable,
            lower,
            upper,
            absolute_tolerance,
            witness,
            meter,
        )?)
    }
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

// Keep every large or arithmetic-bearing payload borrowed until it matches
// the receiver's bounded typed statement. In particular, do not construct an
// attacker's expression or normalize its bounds merely to compare them.
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
/// must match the supplied problem exactly, including raw operand order and
/// symbol identity. The receiver supplies its own expected ordered endpoints,
/// tolerance and assumption context, then independently replays the witness.
pub fn export_integral_certificate(
    problem: &ResidualProblem,
    receipt: &IntegralEnclosureReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, IntegrationError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    let (source, variable) = integral_source(problem)?;
    check_problem_sources(problem.residuals(), problem.variables(), meter)?;
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
            residuals: problem.residuals(),
            variables: problem.variables(),
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
/// the bounded core witness decoder constructs any partition rational. Source,
/// axis, endpoint direction, tolerance and context substitutions refuse before
/// witness decoding. Canonical re-encoding rejects alternate numeric encodings
/// and ignored data. Only the independent original-source checker grants a
/// receipt; this function never invokes the adaptive generator.
#[allow(clippy::too_many_arguments)]
pub fn replay_integral_certificate(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_lower: &BigRational,
    expected_upper: &BigRational,
    expected_tolerance: &BigRational,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<IntegralEnclosureReceipt, IntegrationError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    integral_source(expected_problem)?;
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
    // Bound decoder scratch growth and diagnostics before borrowed parsing.
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
    check_problem_sources(
        expected_problem.residuals(),
        expected_problem.variables(),
        meter,
    )?;
    let sources = encode(expected_problem.residuals(), meter)?;
    let variables = encode(expected_problem.variables(), meter)?;
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
    // The core decoder caps partition length, scalar limbs and precision
    // before constructing/normalizing its rationals. Precharge that bounded
    // synchronous unit; original-source checking below has finer safe points.
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
            residuals: expected_problem.residuals(),
            variables: expected_problem.variables(),
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
    expected_problem.verify_integral(
        expected_lower,
        expected_upper,
        expected_tolerance,
        &witness,
        meter,
    )
}
