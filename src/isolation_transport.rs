//! Canonical complete-root isolation transport with receiver-bound statements.
//! Only independent cover replay grants acceptance; no Expr or receipt decoder is used.

#![forbid(unsafe_code)]

use crate::certification::{
    CertificationError, JsonNesting, MAX_ROOT_CERTIFICATE_BYTES, ResidualProblem, charge,
    charge_witness_encoding, check_problem_sources, encode, encoding_error,
};
use fsym_budget::BudgetMeter;
use fsym_core::{
    Expr, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_VARIABLES, MAX_ROOT_ISOLATION_NODES,
    MAX_ROOT_ISOLATION_ROOTS, RealBall, RootBoxError, RootBoxWitness, RootIsolationReceipt,
    RootIsolationStep, RootIsolationWitness, Symbol,
};
use num_traits::Zero;
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;
use std::{fmt, marker::PhantomData, mem::size_of};

/// Version of the complete-cover transport, independently replayed by receivers.
pub const ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
struct Artifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    witness: WitnessView<'a>,
}

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
    domain: &'a RawValue,
    #[serde(borrow)]
    witness: &'a RawValue,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum Step {
    Split { axis: usize },
    Exclude,
    Root { index: usize },
}

impl From<RootIsolationStep> for Step {
    fn from(value: RootIsolationStep) -> Self {
        match value {
            RootIsolationStep::Split { axis } => Self::Split { axis },
            RootIsolationStep::Exclude => Self::Exclude,
            RootIsolationStep::Root { index } => Self::Root { index },
        }
    }
}

impl From<Step> for RootIsolationStep {
    fn from(value: Step) -> Self {
        match value {
            Step::Split { axis } => Self::Split { axis },
            Step::Exclude => Self::Exclude,
            Step::Root { index } => Self::Root { index },
        }
    }
}

struct Steps<'a>(&'a [RootIsolationStep]);

impl Serialize for Steps<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for &step in self.0 {
            sequence.serialize_element(&Step::from(step))?;
        }
        sequence.end()
    }
}

#[derive(Serialize)]
struct WitnessView<'a> {
    roots: &'a [RootBoxWitness],
    partition: Steps<'a>,
    precision_digits: u32,
}

impl<'a> From<&'a RootIsolationWitness> for WitnessView<'a> {
    fn from(witness: &'a RootIsolationWitness) -> Self {
        Self {
            roots: &witness.roots,
            partition: Steps(&witness.partition),
            precision_digits: witness.precision_digits,
        }
    }
}

struct Limited<T, const N: usize>(Vec<T>);
struct RejectExtra;

impl<'de> Deserialize<'de> for RejectExtra {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom("isolation sequence exceeds its bound"))
    }
}

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Limited<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LimitedVisitor<T, const N: usize>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for LimitedVisitor<T, N> {
            type Value = Limited<T, N>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {N} isolation entries")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                if sequence.size_hint().is_some_and(|length| length > N) {
                    return Err(A::Error::custom("isolation sequence exceeds its bound"));
                }
                let mut values = Vec::new();
                while values.len() < N {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(Limited(values));
                    };
                    values
                        .try_reserve(1)
                        .map_err(|_| A::Error::custom("isolation sequence allocation refused"))?;
                    values.push(value);
                }
                // Reject before decoding an excess root or partition payload.
                let _ = sequence.next_element::<RejectExtra>()?;
                Ok(Limited(values))
            }
        }
        deserializer.deserialize_seq(LimitedVisitor::<T, N>(PhantomData))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessInput<'a> {
    #[serde(borrow)]
    roots: Limited<&'a RawValue, MAX_ROOT_ISOLATION_ROOTS>,
    partition: Limited<Step, MAX_ROOT_ISOLATION_NODES>,
    precision_digits: u32,
}

fn check_domain(
    domain: &[RealBall],
    dimensions: usize,
    meter: &mut impl BudgetMeter,
) -> Result<(), CertificationError> {
    if !(1..=MAX_ROOT_BOX_VARIABLES).contains(&dimensions) || domain.len() != dimensions {
        return Err(RootBoxError::DimensionMismatch.into());
    }
    for (coordinate, ball) in domain.iter().enumerate() {
        for value in [ball.midpoint(), ball.radius()] {
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
            )?;
        }
        if ball.radius().is_zero() {
            return Err(RootBoxError::DegenerateBox { coordinate }.into());
        }
    }
    Ok(())
}

/// Export a complete cover and its local witnesses, including a root-free proof.
/// No root count, acceptance flag or stored numerical conclusion supplies evidence.
pub fn export_root_isolation(
    receipt: &RootIsolationReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, CertificationError> {
    check_problem_sources(receipt.residuals(), receipt.variables(), meter)?;
    check_domain(receipt.domain(), receipt.variables().len(), meter)?;
    for root in &receipt.witness().roots {
        charge_witness_encoding(root, meter)?;
    }
    encode(
        &Artifact {
            schema_version: ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION,
            context_digest,
            residuals: receipt.residuals(),
            variables: receipt.variables(),
            domain: receipt.domain(),
            witness: receipt.witness().into(),
        },
        meter,
    )
}

/// Replay a complete proof against the receiver's exact source, axes and box.
/// The caller supplies one cumulative receiving meter; a Session uses its
/// protected verifier adapter. Canonical comparison precedes independent replay.
pub fn replay_root_isolation(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_domain: &[RealBall],
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<RootIsolationReceipt, CertificationError> {
    meter.checkpoint()?;
    if bytes.len() > MAX_ROOT_CERTIFICATE_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_ROOT_CERTIFICATE_BYTES,
        });
    }
    let mut nesting = JsonNesting::default();
    for chunk in bytes.chunks(512) {
        charge(meter, chunk.len() as u64, 0, 0)?;
        nesting.scan(chunk)?;
    }
    // Escaped keys can grow serde's reusable scratch buffer repeatedly.
    // Cover those bounded growth requests and decoder diagnostics.
    charge(
        meter,
        bytes.len() as u64 + 1,
        bytes.len() as u64 * 4 + 1024,
        64,
    )?;
    let wire: UntrustedArtifact<'_> =
        serde_json::from_slice(bytes).map_err(|error| encoding_error(&error))?;
    if wire.schema_version != ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION,
        });
    }
    if wire.context_digest != expected_context {
        return Err(CertificationError::UniverseMismatch {
            found: wire.context_digest,
            expected: expected_context,
        });
    }
    check_domain(expected_domain, expected_problem.variables().len(), meter)?;
    let sources = encode(expected_problem.residuals(), meter)?;
    let variables = encode(expected_problem.variables(), meter)?;
    let domain = encode(expected_domain, meter)?;
    charge(
        meter,
        (sources.len() + variables.len() + domain.len()) as u64 + 1,
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
    let length = wire.witness.get().len() as u64;
    let storage = MAX_ROOT_ISOLATION_ROOTS * (size_of::<&RawValue>() + size_of::<RootBoxWitness>())
        + MAX_ROOT_ISOLATION_NODES * (size_of::<Step>() + size_of::<RootIsolationStep>());
    charge(
        meter,
        length * 2 + 1,
        length * 4 + storage as u64 + 1024,
        (MAX_ROOT_ISOLATION_ROOTS + MAX_ROOT_ISOLATION_NODES + 8) as u64,
    )?;
    let input: WitnessInput<'_> =
        serde_json::from_str(wire.witness.get()).map_err(|error| encoding_error(&error))?;
    let mut roots = Vec::new();
    roots
        .try_reserve_exact(input.roots.0.len())
        .map_err(|_| CertificationError::AllocationFailure)?;
    for raw in input.roots.0 {
        let length = raw.get().len() as u64;
        // Each bounded scalar decoder is admitted separately; a multi-root
        // cover does not pay the square of its entire combined encoding.
        charge(meter, length * (length + 1), length * 32 + 1024, length + 1)?;
        let root: RootBoxWitness =
            serde_json::from_str(raw.get()).map_err(|error| encoding_error(&error))?;
        meter.checkpoint()?;
        charge_witness_encoding(&root, meter)?;
        roots.push(root);
    }
    let mut partition = Vec::new();
    partition
        .try_reserve_exact(input.partition.0.len())
        .map_err(|_| CertificationError::AllocationFailure)?;
    partition.extend(input.partition.0.into_iter().map(RootIsolationStep::from));
    let witness = RootIsolationWitness {
        roots,
        partition,
        precision_digits: input.precision_digits,
    };
    let canonical = encode(
        &Artifact {
            schema_version: ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION,
            context_digest: expected_context,
            residuals: expected_problem.residuals(),
            variables: expected_problem.variables(),
            domain: expected_domain,
            witness: (&witness).into(),
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical);
    }
    expected_problem.verify_isolation(expected_domain, &witness, meter)
}
