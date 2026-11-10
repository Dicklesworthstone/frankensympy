//! Canonical, source-bound continuations for unfinished root-isolation searches.
//!
//! A checkpoint contains a candidate traversal prefix, never a complete-root
//! receipt. Restoration independently reconstructs the pending cells and checks
//! every imported root/exclusion against the receiver's original source. The
//! eventual complete result still requires protected independent verification.

#![forbid(unsafe_code)]

use crate::certification::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, ResidualProblem, charge,
    charge_witness_encoding, check_problem_sources, encode, encoding_error,
};
use crate::isolation_transport::{WitnessView, check_domain, decode_witness};
use fsym_budget::BudgetMeter;
use fsym_calculus::root_isolation::{
    RootIsolationCounters, RootIsolationOptions, RootIsolationSearch, RootIsolationSearchCheckpoint,
};
use fsym_core::{Expr, RealBall, RootIsolationError, Symbol};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// Version of the portable, untrusted root-search continuation format.
pub const ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
/// Search checkpoints share the bounded mathematical artifact envelope.
pub const MAX_ROOT_SEARCH_CHECKPOINT_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

#[derive(Serialize)]
struct CheckpointArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    options: RootIsolationOptions,
    counters: RootIsolationCounters,
    witness: WitnessView<'a>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedCheckpoint<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    #[serde(borrow)]
    residuals: &'a RawValue,
    #[serde(borrow)]
    variables: &'a RawValue,
    #[serde(borrow)]
    domain: &'a RawValue,
    #[serde(borrow)]
    options: &'a RawValue,
    #[serde(borrow)]
    counters: &'a RawValue,
    #[serde(borrow)]
    witness: &'a RawValue,
}

/// Export normalized candidate progress without discarding its consumed work.
///
/// Snapshot copying charges the supplied ordinary ledger and the live search's
/// intrinsic counters. Repeated exports therefore preserve cumulative work
/// rather than reproducing stale counters. No pending-cell buffer, process
/// memory, accepted root count or accepted flag is serialized.
pub fn export_root_search_checkpoint(
    search: &mut RootIsolationSearch,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, CertificationError> {
    meter.checkpoint()?;
    check_problem_sources(search.residuals(), search.variables(), meter)?;
    check_domain(search.domain(), search.variables().len(), meter)?;
    let checkpoint = search.checkpoint(meter)?;
    for root in &checkpoint.witness.roots {
        charge_witness_encoding(root, meter)?;
    }
    encode(
        &CheckpointArtifact {
            schema_version: ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION,
            context_digest,
            residuals: search.residuals(),
            variables: search.variables(),
            domain: search.domain(),
            options: checkpoint.options,
            counters: checkpoint.counters,
            witness: (&checkpoint.witness).into(),
        },
        meter,
    )
}

/// Restore a candidate search for the receiver's exact immutable inputs.
///
/// Original raw typed sources, ordered axes, box, options and context must
/// match before counters or root payloads are decoded. The bounded decoder is
/// shared with complete-isolation transport, but restoration makes no complete
/// coverage claim: an empty or unfinished prefix remains a candidate.
///
/// The receiver pays for all decoding, reconstruction and independent prefix
/// checks on its own cumulative ordinary meter. Sender counters cannot enlarge
/// that allowance. Prior intrinsic consumption is retained and combined with
/// restoration work; impossible understatements or exhausted lifetime limits
/// refuse. Unauthenticated counters do not prove a sender's full work history.
pub fn restore_root_search_checkpoint(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_domain: &[RealBall],
    expected_options: RootIsolationOptions,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<RootIsolationSearch, CertificationError> {
    meter.checkpoint()?;
    if bytes.len() > MAX_ROOT_SEARCH_CHECKPOINT_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_ROOT_SEARCH_CHECKPOINT_BYTES,
        });
    }
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
    let wire: UntrustedCheckpoint<'_> =
        serde_json::from_slice(bytes).map_err(|error| encoding_error(&error))?;
    if wire.schema_version != ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION,
        });
    }
    if wire.context_digest != expected_context {
        return Err(CertificationError::UniverseMismatch {
            found: wire.context_digest,
            expected: expected_context,
        });
    }
    check_problem_sources(
        expected_problem.residuals(),
        expected_problem.variables(),
        meter,
    )?;
    check_domain(expected_domain, expected_problem.variables().len(), meter)?;
    let sources = encode(expected_problem.residuals(), meter)?;
    let variables = encode(expected_problem.variables(), meter)?;
    let domain = encode(expected_domain, meter)?;
    let options = encode(&expected_options, meter)?;
    charge(
        meter,
        (sources.len() + variables.len() + domain.len() + options.len()) as u64 + 1,
        0,
        0,
    )?;
    if sources.as_slice() != wire.residuals.get().as_bytes()
        || variables.as_slice() != wire.variables.get().as_bytes()
    {
        return Err(CertificationError::SourceMismatch);
    }
    if domain.as_slice() != wire.domain.get().as_bytes() {
        return Err(CertificationError::DomainMismatch);
    }
    if options.as_slice() != wire.options.get().as_bytes() {
        return Err(RootIsolationError::InvalidPartition(
            "checkpoint search options differ from the expected options",
        )
        .into());
    }
    // Four scalar counters are decoded only after immutable input binding.
    // Their values never define the receiver's budget or grant evidence.
    let counter_bytes = wire.counters.get().len() as u64;
    charge(meter, counter_bytes + 1, counter_bytes * 4 + 1024, 4)?;
    let counters: RootIsolationCounters =
        serde_json::from_str(wire.counters.get()).map_err(|error| encoding_error(&error))?;
    meter.checkpoint()?;
    let witness = decode_witness(wire.witness.get(), meter)?;
    let canonical = encode(
        &CheckpointArtifact {
            schema_version: ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION,
            context_digest: expected_context,
            residuals: expected_problem.residuals(),
            variables: expected_problem.variables(),
            domain: expected_domain,
            options: expected_options,
            counters,
            witness: (&witness).into(),
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical);
    }
    Ok(RootIsolationSearch::from_checkpoint(
        expected_problem.residuals(),
        expected_problem.variables(),
        expected_domain,
        &RootIsolationSearchCheckpoint {
            options: expected_options,
            counters,
            witness,
        },
        meter,
    )?)
}
