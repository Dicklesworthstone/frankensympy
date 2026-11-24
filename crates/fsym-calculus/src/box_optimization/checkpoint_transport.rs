//! Portable candidate continuations for bounded global box optimization.
//!
//! Imported bounds, cached derivatives and acceptance flags have no authority.
//! Restoration binds the receiver's original statement, reconstructs the full
//! dyadic partition and reevaluates its frontier. Independent protected finish
//! verification remains necessary before publishing a global optimum bound.

#![forbid(unsafe_code)]

use super::{
    BoxOptimizationCounters, BoxOptimizationOptions, BoxOptimizationSearch,
    BoxOptimizationSearchCheckpoint,
};
use fsym_budget::BudgetMeter;
use fsym_core::root_certificate::artifact::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, charge, check_problem_sources,
    encode, encoding_error,
};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationWitness, BoxPartitionStep,
    Expr, MAX_BOX_OPTIMIZATION_NODES, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_VARIABLES, RealBall,
    RootBoxError, Symbol,
};
use num_traits::Signed;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::value::RawValue;
use std::{fmt, marker::PhantomData};
use thiserror::Error;

/// Version of the source-bound candidate continuation format.
pub const BOX_OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
/// Portable searches share the bounded mathematical artifact envelope.
pub const MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

/// A continuation cannot confer acceptance or replace its expected statement.
#[derive(Debug, Error)]
pub enum OptimizationCheckpointError {
    #[error("{0}")]
    Certification(#[from] CertificationError),
    #[error("{0}")]
    Optimization(#[from] BoxOptimizationError),
    #[error("optimization direction or absolute tolerance differs from the expected statement")]
    StatementMismatch,
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

fn charge_statement(
    domain: &[RealBall],
    dimensions: usize,
    absolute_tolerance: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<(), OptimizationCheckpointError> {
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

#[derive(Serialize)]
struct CheckpointArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    objective: &'a Expr,
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    kind: BoxExtremumKind,
    absolute_tolerance: &'a BigRational,
    options: BoxOptimizationOptions,
    counters: BoxOptimizationCounters,
    partition: &'a [BoxPartitionStep],
    split_order: &'a [usize],
    incumbent_point: Option<&'a [BigRational]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedCheckpoint<'a> {
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
    options: &'a RawValue,
    #[serde(borrow)]
    counters: &'a RawValue,
    #[serde(borrow)]
    partition: &'a RawValue,
    #[serde(borrow)]
    split_order: &'a RawValue,
    #[serde(borrow)]
    incumbent_point: &'a RawValue,
}

// Refuse a sequence's first excess element before decoding or allocating it.
struct RejectExtra;

impl<'de> Deserialize<'de> for RejectExtra {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(de::Error::custom("optimization checkpoint sequence limit"))
    }
}

struct BoundedSequence<T, const N: usize>(Vec<T>);

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for BoundedSequence<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T, const N: usize>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>, const N: usize> de::Visitor<'de> for Visitor<T, N> {
            type Value = BoundedSequence<T, N>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "an optimization checkpoint sequence of at most {N} items"
                )
            }

            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                loop {
                    if values.len() == N {
                        sequence.next_element::<RejectExtra>()?;
                        return Ok(BoundedSequence(values));
                    }
                    let Some(value) = sequence.next_element()? else {
                        return Ok(BoundedSequence(values));
                    };
                    values.try_reserve(1).map_err(de::Error::custom)?;
                    values.push(value);
                }
            }
        }

        deserializer.deserialize_seq(Visitor::<T, N>(PhantomData))
    }
}

fn decode_point(
    raw: &RawValue,
    precision_digits: u32,
    meter: &mut impl BudgetMeter,
) -> Result<Option<Vec<BigRational>>, OptimizationCheckpointError> {
    if raw.get() == "null" {
        return Ok(None);
    }
    // Reuse the core witness decoder's pre-allocation coordinate and integer
    // limb checks. A single constant leaf is only a decoding envelope; this
    // function grants no feasibility or mathematical evidence.
    #[derive(Serialize)]
    struct PointEnvelope<'a> {
        partition: [BoxPartitionStep; 1],
        point: &'a RawValue,
        precision_digits: u32,
    }

    let bytes = raw.get().len() as u64;
    charge(
        meter,
        bytes.saturating_mul(bytes + 1) + 1,
        bytes * 32 + 1024,
        bytes + 1,
    )?;
    let encoded = encode(
        &PointEnvelope {
            partition: [BoxPartitionStep::Leaf],
            point: raw,
            precision_digits,
        },
        meter,
    )?;
    let witness: BoxOptimizationWitness =
        serde_json::from_slice(&encoded).map_err(|error| encoding_error(&error))?;
    meter.checkpoint().map_err(CertificationError::from)?;
    Ok(Some(witness.point))
}

/// Export candidate progress while retaining all intrinsic work consumption.
///
/// No cached values, heap priorities, derivative jets or completion flags are
/// serialized. Export itself consumes the live search's cumulative allowance;
/// failures cannot expose a partially copied checkpoint.
pub fn export_box_optimization_checkpoint(
    search: &mut BoxOptimizationSearch,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, OptimizationCheckpointError> {
    if let Some(error) = search.failure() {
        return Err(error.clone().into());
    }
    meter.checkpoint().map_err(CertificationError::from)?;
    check_problem_sources(
        std::slice::from_ref(search.objective()),
        search.variables(),
        meter,
    )?;
    charge_statement(
        search.domain(),
        search.variables().len(),
        search.absolute_tolerance(),
        meter,
    )?;
    let checkpoint = search.checkpoint(meter)?;
    Ok(encode(
        &CheckpointArtifact {
            schema_version: BOX_OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION,
            context_digest,
            objective: search.objective(),
            variables: search.variables(),
            domain: search.domain(),
            kind: search.kind(),
            absolute_tolerance: search.absolute_tolerance(),
            options: checkpoint.options,
            counters: checkpoint.counters,
            partition: &checkpoint.partition,
            split_order: &checkpoint.split_order,
            incumbent_point: checkpoint.incumbent_point.as_deref(),
        },
        meter,
    )?)
}

/// Restore a candidate for the receiver's exact immutable statement.
///
/// Typed source, ordered coordinates, domain, direction, tolerance, options
/// and context must match before any imported arithmetic is constructed.
/// Receiver-side decoding and reevaluation use its ordinary cumulative meter.
/// Imported intrinsic counters cannot enlarge that budget and are combined
/// with restoration work. They do not authenticate the sender's work history.
#[allow(clippy::too_many_arguments)]
pub fn restore_box_optimization_checkpoint(
    bytes: &[u8],
    expected_objective: &Expr,
    expected_variables: &[Symbol],
    expected_domain: &[RealBall],
    expected_kind: BoxExtremumKind,
    expected_tolerance: &BigRational,
    expected_options: BoxOptimizationOptions,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<BoxOptimizationSearch, OptimizationCheckpointError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    if bytes.len() > MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES,
        }
        .into());
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
    if wire.schema_version != BOX_OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: BOX_OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION,
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
        return Err(OptimizationCheckpointError::StatementMismatch);
    }
    check_problem_sources(
        std::slice::from_ref(expected_objective),
        expected_variables,
        meter,
    )?;
    charge_statement(
        expected_domain,
        expected_variables.len(),
        expected_tolerance,
        meter,
    )?;
    let objective = encode(expected_objective, meter)?;
    let variables = encode(expected_variables, meter)?;
    let domain = encode(expected_domain, meter)?;
    let tolerance = encode(expected_tolerance, meter)?;
    let options = encode(&expected_options, meter)?;
    charge(
        meter,
        (objective.len() + variables.len() + domain.len() + tolerance.len() + options.len()) as u64
            + 1,
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
    if tolerance.as_slice() != wire.absolute_tolerance.get().as_bytes()
        || options.as_slice() != wire.options.get().as_bytes()
    {
        return Err(OptimizationCheckpointError::StatementMismatch);
    }

    let counter_bytes = wire.counters.get().len() as u64;
    charge(meter, counter_bytes + 1, counter_bytes * 4 + 1024, 4)?;
    let counters: BoxOptimizationCounters =
        serde_json::from_str(wire.counters.get()).map_err(|error| encoding_error(&error))?;
    meter.checkpoint().map_err(CertificationError::from)?;
    let topology_bytes = (wire.partition.get().len() + wire.split_order.get().len()) as u64;
    charge(
        meter,
        topology_bytes + 1,
        topology_bytes * 16 + 1024,
        topology_bytes + 1,
    )?;
    let partition: BoundedSequence<BoxPartitionStep, MAX_BOX_OPTIMIZATION_NODES> =
        serde_json::from_str(wire.partition.get()).map_err(|error| encoding_error(&error))?;
    let split_order: BoundedSequence<usize, MAX_BOX_OPTIMIZATION_NODES> =
        serde_json::from_str(wire.split_order.get()).map_err(|error| encoding_error(&error))?;
    let incumbent_point = decode_point(
        wire.incumbent_point,
        expected_options.precision_digits,
        meter,
    )?;
    let checkpoint = BoxOptimizationSearchCheckpoint {
        options: expected_options,
        counters,
        partition: partition.0,
        split_order: split_order.0,
        incumbent_point,
    };
    let canonical = encode(
        &CheckpointArtifact {
            schema_version: BOX_OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION,
            context_digest: expected_context,
            objective: expected_objective,
            variables: expected_variables,
            domain: expected_domain,
            kind: expected_kind,
            absolute_tolerance: expected_tolerance,
            options: checkpoint.options,
            counters: checkpoint.counters,
            partition: &checkpoint.partition,
            split_order: &checkpoint.split_order,
            incumbent_point: checkpoint.incumbent_point.as_deref(),
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical.into());
    }
    Ok(BoxOptimizationSearch::from_checkpoint(
        expected_objective,
        expected_variables,
        expected_domain,
        expected_kind,
        expected_tolerance,
        &checkpoint,
        meter,
    )?)
}
