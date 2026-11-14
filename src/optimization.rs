//! Source-bound global value bounds and portable box-optimization witnesses.
//!
//! A proposal contains an exact feasible point and a complete dyadic partition.
//! The independent core checker reconstructs every cell and reevaluates the
//! original objective. Acceptance encloses the global minimum or maximum and
//! bounds the point's absolute suboptimality; it does not prove a unique or
//! exact optimizer, or enumerate every optimizing point.

#![forbid(unsafe_code)]

use crate::certification::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, charge, check_problem_sources,
    encode, encoding_error,
};
use crate::objective::ObjectiveProblem;
use fsym_budget::BudgetMeter;
use fsym_calculus::box_optimization::{
    BoxOptimizationOptions, BoxOptimizationSearch, propose_box_optimization,
};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationReceipt,
    BoxOptimizationWitness, Expr, MAX_BOX_OPTIMIZATION_NODES, MAX_ROOT_BOX_LIMBS,
    MAX_ROOT_BOX_VARIABLES, RealBall, RootBoxError, Symbol, verify_box_optimization,
};
use num_traits::Signed;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error;

/// Version of the exact source, box, direction and tolerance-bound format.
pub const BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION: u32 = 1;
/// Maximum encoded witness, sharing the bounded session artifact envelope.
pub const MAX_BOX_OPTIMIZATION_ARTIFACT_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

/// A refusal never supplies a partial optimum or accepted mathematical result.
#[derive(Debug, Error)]
pub enum OptimizationError {
    #[error(transparent)]
    Certification(#[from] CertificationError),
    #[error(transparent)]
    Optimization(#[from] BoxOptimizationError),
    #[error("optimization direction or absolute tolerance differs from the expected statement")]
    StatementMismatch,
}

impl ObjectiveProblem {
    /// Begin a bounded search whose frontier and intrinsic work counters persist
    /// across advances. Completion supplies a candidate, never an accepted result.
    pub fn begin_box_optimization_search(
        &self,
        domain: &[RealBall],
        kind: BoxExtremumKind,
        absolute_tolerance: &BigRational,
        options: BoxOptimizationOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<BoxOptimizationSearch, OptimizationError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        Ok(BoxOptimizationSearch::new(
            self.objective(),
            self.variables(),
            domain,
            kind,
            absolute_tolerance,
            options,
            meter,
        )?)
    }

    /// Propose a feasible point and a complete partition using ordinary work.
    ///
    /// Every original source domain obligation is retained on the closed box.
    /// A returned witness is only a candidate; callers must independently
    /// verify it before exposing a global value or suboptimality claim.
    pub fn propose_box_optimization(
        &self,
        domain: &[RealBall],
        kind: BoxExtremumKind,
        absolute_tolerance: &BigRational,
        options: BoxOptimizationOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<BoxOptimizationWitness, OptimizationError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        Ok(propose_box_optimization(
            self.objective(),
            self.variables(),
            domain,
            kind,
            absolute_tolerance,
            options,
            meter,
        )?)
    }

    /// Independently establish complete coverage and a global value bound.
    ///
    /// All cells and the proposed point are recomputed from the original
    /// source. The caller may supply a protected verifier meter from the same
    /// ledger used for proposal. Resource failures and cancellation terminate
    /// without a partial receipt, and no search is retried during verification.
    pub fn verify_box_optimization(
        &self,
        domain: &[RealBall],
        kind: BoxExtremumKind,
        absolute_tolerance: &BigRational,
        witness: &BoxOptimizationWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<BoxOptimizationReceipt, OptimizationError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        Ok(verify_box_optimization(
            self.objective(),
            self.variables(),
            domain,
            kind,
            absolute_tolerance,
            witness,
            meter,
        )?)
    }
}

#[derive(Serialize)]
struct OptimizationArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    objective: &'a Expr,
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    kind: BoxExtremumKind,
    absolute_tolerance: &'a BigRational,
    witness: &'a BoxOptimizationWitness,
}

// Arithmetic-bearing fields stay borrowed until they match the receiver's
// already bounded statement. An attacker's Expr or domain is never decoded.
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
    kind: BoxExtremumKind,
    #[serde(borrow)]
    absolute_tolerance: &'a RawValue,
    #[serde(borrow)]
    witness: &'a RawValue,
}

// Partition decoding is linear in its bytes. Only the point's exact scalars
// can require quadratic reduction work; a large complete partition must not
// be billed as one enormous arbitrary-precision rational.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessCosts<'a> {
    #[serde(borrow)]
    partition: &'a RawValue,
    #[serde(borrow)]
    point: &'a RawValue,
    precision_digits: u32,
}

fn charge_number(
    value: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), CertificationError> {
    let numerator = value.numer().limb_count();
    let denominator = value.denom().limb_count();
    if numerator > MAX_ROOT_BOX_LIMBS || denominator > MAX_ROOT_BOX_LIMBS {
        return Err(RootBoxError::ResourceLimit("rational height").into());
    }
    charge(
        meter,
        numerator + denominator + 1,
        128 + 16 * (numerator + denominator),
        2,
    )
}

pub(crate) fn charge_statement(
    domain: &[RealBall],
    dimensions: usize,
    absolute_tolerance: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), OptimizationError> {
    if !(1..=MAX_ROOT_BOX_VARIABLES).contains(&dimensions) || domain.len() != dimensions {
        return Err(CertificationError::from(RootBoxError::DimensionMismatch).into());
    }
    // Unlike root-box existence certificates, global optimization admits fixed
    // coordinates. A zero-radius axis must survive export and replay exactly.
    for ball in domain {
        charge_number(ball.midpoint(), meter)?;
        charge_number(ball.radius(), meter)?;
    }
    charge_number(absolute_tolerance, meter)?;
    if !absolute_tolerance.is_positive() {
        return Err(BoxOptimizationError::InvalidTolerance.into());
    }
    Ok(())
}

fn charge_witness(
    witness: &BoxOptimizationWitness,
    dimensions: usize,
    meter: &mut impl BudgetMeter,
) -> Result<(), OptimizationError> {
    if witness.partition.is_empty() || witness.partition.len() > MAX_BOX_OPTIMIZATION_NODES {
        return Err(BoxOptimizationError::InvalidPartition("partition node count").into());
    }
    if witness.point.len() != dimensions {
        return Err(CertificationError::from(RootBoxError::DimensionMismatch).into());
    }
    if !(8..=128).contains(&witness.precision_digits) {
        return Err(CertificationError::from(RootBoxError::InvalidPrecision).into());
    }
    charge(meter, witness.partition.len() as u64 + 1, 0, 0)?;
    for coordinate in &witness.point {
        charge_number(coordinate, meter)?;
    }
    Ok(())
}

/// Export the accepted statement and an untrusted complete-partition witness.
///
/// No optimum value, point value, gap, leaf count or accepted flag is stored.
/// The receiver supplies its own original source, ordered typed coordinates,
/// closed box, direction, tolerance and assumptions context before replay.
pub fn export_box_optimization(
    receipt: &BoxOptimizationReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, OptimizationError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    check_problem_sources(
        std::slice::from_ref(receipt.objective()),
        receipt.variables(),
        meter,
    )?;
    charge_statement(
        receipt.domain(),
        receipt.variables().len(),
        receipt.absolute_tolerance(),
        meter,
    )?;
    charge_witness(receipt.witness(), receipt.variables().len(), meter)?;
    Ok(encode(
        &OptimizationArtifact {
            schema_version: BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION,
            context_digest,
            objective: receipt.objective(),
            variables: receipt.variables(),
            domain: receipt.domain(),
            kind: receipt.kind(),
            absolute_tolerance: receipt.absolute_tolerance(),
            witness: receipt.witness(),
        },
        meter,
    )?)
}

/// Independently replay a global bound against the receiver's exact statement.
///
/// Byte length, JSON nesting and decoder work are admitted before bounded
/// witness construction. Source, identity, domain, direction, tolerance and
/// context substitutions fail before witness decoding. Canonical reencoding
/// rejects alternate spellings and encodings. Only the independent core
/// checker may return an accepted result; this path never invokes search.
#[allow(clippy::too_many_arguments)]
pub fn replay_box_optimization(
    bytes: &[u8],
    expected_problem: &ObjectiveProblem,
    expected_domain: &[RealBall],
    expected_kind: BoxExtremumKind,
    expected_tolerance: &BigRational,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<BoxOptimizationReceipt, OptimizationError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    if bytes.len() > MAX_BOX_OPTIMIZATION_ARTIFACT_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_BOX_OPTIMIZATION_ARTIFACT_BYTES,
        }
        .into());
    }
    charge_statement(
        expected_domain,
        expected_problem.variables().len(),
        expected_tolerance,
        meter,
    )?;
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
    if wire.schema_version != BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION,
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
    if wire.kind != expected_kind {
        return Err(OptimizationError::StatementMismatch);
    }
    check_problem_sources(
        std::slice::from_ref(expected_problem.objective()),
        expected_problem.variables(),
        meter,
    )?;
    let objective = encode(expected_problem.objective(), meter)?;
    let variables = encode(expected_problem.variables(), meter)?;
    let domain = encode(expected_domain, meter)?;
    let tolerance = encode(expected_tolerance, meter)?;
    charge(
        meter,
        (objective.len() + variables.len() + domain.len() + tolerance.len()) as u64 + 1,
        0,
        0,
    )?;
    if objective.as_slice() != wire.objective.get().as_bytes()
        || variables.as_slice() != wire.variables.get().as_bytes()
    {
        return Err(CertificationError::SourceMismatch.into());
    }
    if domain.as_slice() != wire.domain.get().as_bytes() {
        return Err(CertificationError::DomainMismatch.into());
    }
    if tolerance.as_slice() != wire.absolute_tolerance.get().as_bytes() {
        return Err(OptimizationError::StatementMismatch);
    }
    // First borrow the witness's arithmetic-bearing point without constructing
    // its rationals. Admit linear parsing/bookkeeping before this preflight.
    let witness_bytes = wire.witness.get().len() as u64;
    charge(meter, 2 * witness_bytes + 1, witness_bytes * 4 + 1024, 64)?;
    let costs: WitnessCosts<'_> =
        serde_json::from_str(wire.witness.get()).map_err(|error| encoding_error(&error))?;
    if !(8..=128).contains(&costs.precision_digits) {
        return Err(CertificationError::from(RootBoxError::InvalidPrecision).into());
    }
    let point_bytes = costs.point.get().len() as u64;
    let partition_bytes = costs.partition.get().len() as u64;
    // The core decoder caps every sequence and scalar before arithmetic. The
    // point normalization allowance is independent of the partition length.
    charge(
        meter,
        point_bytes.saturating_mul(point_bytes + 1) + partition_bytes + 1,
        witness_bytes * 32 + 1024,
        witness_bytes + 1,
    )?;
    let witness: BoxOptimizationWitness =
        serde_json::from_str(wire.witness.get()).map_err(|error| encoding_error(&error))?;
    meter.checkpoint().map_err(CertificationError::from)?;
    charge_witness(&witness, expected_problem.variables().len(), meter)?;
    let canonical = encode(
        &OptimizationArtifact {
            schema_version: BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION,
            context_digest: expected_context,
            objective: expected_problem.objective(),
            variables: expected_problem.variables(),
            domain: expected_domain,
            kind: expected_kind,
            absolute_tolerance: expected_tolerance,
            witness: &witness,
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical.into());
    }
    expected_problem.verify_box_optimization(
        expected_domain,
        expected_kind,
        expected_tolerance,
        &witness,
        meter,
    )
}
