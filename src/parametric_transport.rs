//! Portable uniform and complete-cover parameter-root candidates.
//!
//! The receiver supplies the original residuals, two separately ordered typed
//! coordinate groups, both complete boxes and its context. Transport carries
//! no accepted enclosure or sensitivity. Only independent original-source
//! replay can reconstruct a family receipt, including every domain obligation.

#![forbid(unsafe_code)]

use crate::certification::{
    CertificationError, JsonNesting, MAX_CERTIFICATION_NAME_BYTES, MAX_ROOT_CERTIFICATE_BYTES,
    charge, check_problem_sources, encode, encoding_error,
};
use crate::parametric::{ParametricError, ParametricFamilyReceipt, ParametricProblem};
use fsym_budget::BudgetMeter;
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{
    BigInt, BigRational, Expr, MAX_PARAMETRIC_ROOT_COVER_DEPTH, MAX_PARAMETRIC_ROOT_COVER_LEAVES,
    MAX_PARAMETRIC_ROOT_COVER_NODES, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_PRECISION,
    MAX_ROOT_BOX_VARIABLES, ParametricRootBoxWitness, ParametricRootCoverError,
    ParametricRootCoverStep, ParametricRootCoverWitness, RealBall, RootBoxError, Symbol,
};
use num_traits::{Signed, Zero};
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;
use std::{fmt, marker::PhantomData, mem::size_of};

/// Version of the source-, role-, box- and context-bound family format.
pub const PARAMETRIC_FAMILY_ARTIFACT_SCHEMA_VERSION: u32 = 1;
/// Maximum canonical family artifact, including its entire candidate cover.
pub const MAX_PARAMETRIC_FAMILY_ARTIFACT_BYTES: usize = MAX_ROOT_CERTIFICATE_BYTES;

#[derive(Serialize)]
struct Artifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    unknowns: &'a [Symbol],
    parameters: &'a [Symbol],
    unknown_domain: &'a [RealBall],
    parameter_domain: &'a [RealBall],
    witness: WitnessView<'a>,
}

// Exact statement fields remain borrowed until their bytes match the receiver.
// In particular, an attacker's expressions and coordinate identities are never
// deserialized, simplified, substituted or inferred from printed names.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    #[serde(borrow)]
    residuals: &'a RawValue,
    #[serde(borrow)]
    unknowns: &'a RawValue,
    #[serde(borrow)]
    parameters: &'a RawValue,
    #[serde(borrow)]
    unknown_domain: &'a RawValue,
    #[serde(borrow)]
    parameter_domain: &'a RawValue,
    #[serde(borrow)]
    witness: &'a RawValue,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    RootBox,
    Cover,
}

#[derive(Serialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
enum WitnessView<'a> {
    RootBox(RootBoxView<'a>),
    Cover(CoverView<'a>),
}

#[derive(Serialize)]
struct RootBoxView<'a> {
    unknown_domain: &'a [RealBall],
    parameter_domain: &'a [RealBall],
    center: &'a [BigRational],
    preconditioner: &'a [Vec<BigRational>],
    precision_digits: u32,
}

impl<'a> From<&'a ParametricRootBoxWitness> for RootBoxView<'a> {
    fn from(witness: &'a ParametricRootBoxWitness) -> Self {
        Self {
            unknown_domain: &witness.unknown_domain,
            parameter_domain: &witness.parameter_domain,
            center: &witness.center,
            preconditioner: &witness.preconditioner,
            precision_digits: witness.precision_digits,
        }
    }
}

struct Leaves<'a>(&'a [ParametricRootBoxWitness]);

impl Serialize for Leaves<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for leaf in self.0 {
            sequence.serialize_element(&RootBoxView::from(leaf))?;
        }
        sequence.end()
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum Step {
    Split { axis: usize },
    Leaf { index: usize },
}

impl From<ParametricRootCoverStep> for Step {
    fn from(step: ParametricRootCoverStep) -> Self {
        match step {
            ParametricRootCoverStep::Split { axis } => Self::Split { axis },
            ParametricRootCoverStep::Leaf { index } => Self::Leaf { index },
        }
    }
}

impl From<Step> for ParametricRootCoverStep {
    fn from(step: Step) -> Self {
        match step {
            Step::Split { axis } => Self::Split { axis },
            Step::Leaf { index } => Self::Leaf { index },
        }
    }
}

struct Steps<'a>(&'a [ParametricRootCoverStep]);

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
struct CoverView<'a> {
    leaves: Leaves<'a>,
    partition: Steps<'a>,
    precision_digits: u32,
}

impl<'a> From<&'a ParametricRootCoverWitness> for CoverView<'a> {
    fn from(witness: &'a ParametricRootCoverWitness) -> Self {
        Self {
            leaves: Leaves(&witness.leaves),
            partition: Steps(&witness.partition),
            precision_digits: witness.precision_digits,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessInput<'a> {
    kind: Kind,
    #[serde(borrow)]
    data: &'a RawValue,
}

struct Limited<T, const N: usize>(Vec<T>);
struct RejectExtra;

impl<'de> Deserialize<'de> for RejectExtra {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "parameter-family sequence exceeds its bound",
        ))
    }
}

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Limited<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LimitedVisitor<T, const N: usize>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for LimitedVisitor<T, N> {
            type Value = Limited<T, N>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {N} parameter-family entries")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                if sequence.size_hint().is_some_and(|length| length > N) {
                    return Err(A::Error::custom(
                        "parameter-family sequence exceeds its bound",
                    ));
                }
                let mut values = Vec::new();
                while values.len() < N {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(Limited(values));
                    };
                    values.try_reserve(1).map_err(|_| {
                        A::Error::custom("parameter-family sequence allocation refused")
                    })?;
                    values.push(value);
                }
                // Refuse before decoding an excess scalar, matrix row or leaf.
                let _ = sequence.next_element::<RejectExtra>()?;
                Ok(Limited(values))
            }
        }
        deserializer.deserialize_seq(LimitedVisitor::<T, N>(PhantomData))
    }
}

// All array counts and matrix shapes are checked while scalars still borrow
// the bounded input buffer. Neither BigInt nor BigRational is constructed by
// this structural decoding pass.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BallInput<'a> {
    #[serde(borrow)]
    midpoint: &'a RawValue,
    #[serde(borrow)]
    radius: &'a RawValue,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RootBoxInput<'a> {
    #[serde(borrow)]
    unknown_domain: Limited<BallInput<'a>, MAX_ROOT_BOX_VARIABLES>,
    #[serde(borrow)]
    parameter_domain: Limited<BallInput<'a>, MAX_ROOT_BOX_VARIABLES>,
    #[serde(borrow)]
    center: Limited<&'a RawValue, MAX_ROOT_BOX_VARIABLES>,
    #[serde(borrow)]
    preconditioner: Limited<Limited<&'a RawValue, MAX_ROOT_BOX_VARIABLES>, MAX_ROOT_BOX_VARIABLES>,
    precision_digits: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverInput<'a> {
    #[serde(borrow)]
    leaves: Limited<RootBoxInput<'a>, MAX_PARAMETRIC_ROOT_COVER_LEAVES>,
    partition: Limited<Step, MAX_PARAMETRIC_ROOT_COVER_NODES>,
    precision_digits: u32,
}

// Integer limbs are bounded before allocation of either arbitrary-precision
// integer. Sign/magnitude canonicality is also checked before normalization.
struct IntegerInput {
    sign: i8,
    digits: Vec<u32>,
}

impl<'de> Deserialize<'de> for IntegerInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct IntegerVisitor;
        impl<'de> Visitor<'de> for IntegerVisitor {
            type Value = IntegerInput;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a canonical bounded sign/u32-magnitude integer")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let sign: i8 = sequence
                    .next_element()?
                    .ok_or_else(|| A::Error::custom("missing integer sign"))?;
                if !(-1..=1).contains(&sign) {
                    return Err(A::Error::custom("invalid integer sign"));
                }
                let digits = if sign == 0 {
                    sequence.next_element::<Limited<u32, 0>>()?.map(|x| x.0)
                } else {
                    sequence
                        .next_element::<Limited<u32, { MAX_ROOT_BOX_LIMBS as usize * 2 }>>()?
                        .map(|x| x.0)
                }
                .ok_or_else(|| A::Error::custom("missing integer magnitude"))?;
                if digits.last() == Some(&0) || (sign != 0 && digits.is_empty()) {
                    return Err(A::Error::custom("noncanonical integer magnitude"));
                }
                let _ = sequence.next_element::<RejectExtra>()?;
                Ok(IntegerInput { sign, digits })
            }
        }
        deserializer.deserialize_tuple(2, IntegerVisitor)
    }
}

impl IntegerInput {
    fn into_integer(self) -> Result<BigInt, CertificationError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.digits.len() * 4)
            .map_err(|_| CertificationError::AllocationFailure)?;
        for digit in self.digits {
            bytes.extend_from_slice(&digit.to_le_bytes());
        }
        let value = BigInt::from_bytes_le(&bytes);
        Ok(if self.sign < 0 { -value } else { value })
    }
}

fn scalar_error(error: RationalArithmeticError) -> CertificationError {
    match error {
        RationalArithmeticError::Meter(error) => error.into(),
        RationalArithmeticError::AllocationFailure => CertificationError::AllocationFailure,
        RationalArithmeticError::SizeOverflow => {
            RootBoxError::ResourceLimit("parameter-family scalar work size").into()
        }
        RationalArithmeticError::DivisionByZero
        | RationalArithmeticError::InvariantViolation(_) => {
            CertificationError::Encoding("parameter-family scalar normalization failed".into())
        }
    }
}

fn decode_number(
    raw: &RawValue,
    meter: &mut impl BudgetMeter,
) -> Result<BigRational, CertificationError> {
    let bytes = raw.get().len() as u64;
    // Structural parsing and bounded limb buffers are linear. A separate
    // per-scalar allowance bounds normalization; a whole matrix/cover is never
    // squared. Metered normalization also checks cancellation during its gcd.
    charge(meter, bytes + 1, bytes * 16 + 256, 32)?;
    let (numerator, denominator): (IntegerInput, IntegerInput) =
        serde_json::from_str(raw.get()).map_err(|error| encoding_error(&error))?;
    if denominator.sign != 1 {
        return Err(CertificationError::NonCanonical);
    }
    charge(meter, bytes.saturating_mul(bytes + 1), bytes * 16 + 256, 8)?;
    let numerator = numerator.into_integer()?;
    let denominator = denominator.into_integer()?;
    let value = BigRational::metered_new(&numerator, &denominator, meter).map_err(scalar_error)?;
    charge(
        meter,
        numerator.limb_count() + denominator.limb_count() + 1,
        0,
        0,
    )?;
    if value.numer() != &numerator || value.denom() != &denominator {
        return Err(CertificationError::NonCanonical);
    }
    Ok(value)
}

fn decode_domain(
    input: Limited<BallInput<'_>, MAX_ROOT_BOX_VARIABLES>,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<RealBall>, ParametricError> {
    let mut domain = reserve(input.0.len(), meter)?;
    for ball in input.0 {
        let midpoint = decode_number(ball.midpoint, meter)?;
        let radius = decode_number(ball.radius, meter)?;
        if radius.is_negative() {
            return Err(CertificationError::NonCanonical.into());
        }
        domain.push(RealBall::new(midpoint, radius).map_err(RootBoxError::from)?);
    }
    Ok(domain)
}

fn reserve<T>(count: usize, meter: &mut impl BudgetMeter) -> Result<Vec<T>, CertificationError> {
    charge(meter, count as u64 + 1, (count * size_of::<T>()) as u64, 1)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CertificationError::AllocationFailure)?;
    Ok(values)
}

impl RootBoxInput<'_> {
    fn check_shape(
        &self,
        unknowns: usize,
        parameters: usize,
        meter: &mut impl BudgetMeter,
    ) -> Result<(), ParametricError> {
        charge(meter, (unknowns * unknowns + parameters + 1) as u64, 0, 0)?;
        if self.unknown_domain.0.len() != unknowns
            || self.parameter_domain.0.len() != parameters
            || self.center.0.len() != unknowns
            || self.preconditioner.0.len() != unknowns
            || self
                .preconditioner
                .0
                .iter()
                .any(|row| row.0.len() != unknowns)
        {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        check_precision(self.precision_digits)
    }

    fn decode(
        self,
        meter: &mut impl BudgetMeter,
    ) -> Result<ParametricRootBoxWitness, ParametricError> {
        let unknown_domain = decode_domain(self.unknown_domain, meter)?;
        let parameter_domain = decode_domain(self.parameter_domain, meter)?;
        let mut center = reserve(self.center.0.len(), meter)?;
        for scalar in self.center.0 {
            center.push(decode_number(scalar, meter)?);
        }
        let mut preconditioner = reserve(self.preconditioner.0.len(), meter)?;
        for row in self.preconditioner.0 {
            let mut values = reserve(row.0.len(), meter)?;
            for scalar in row.0 {
                values.push(decode_number(scalar, meter)?);
            }
            preconditioner.push(values);
        }
        Ok(ParametricRootBoxWitness {
            unknown_domain,
            parameter_domain,
            center,
            preconditioner,
            precision_digits: self.precision_digits,
        })
    }
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

fn check_precision(precision: u32) -> Result<(), ParametricError> {
    if !(8..=MAX_ROOT_BOX_PRECISION).contains(&precision) {
        return Err(RootBoxError::InvalidPrecision.into());
    }
    Ok(())
}

fn check_statement(
    residuals: &[Expr],
    unknowns: &[Symbol],
    parameters: &[Symbol],
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
    meter: &mut impl BudgetMeter,
) -> Result<(), ParametricError> {
    let total = unknowns.len().saturating_add(parameters.len());
    if unknowns.is_empty()
        || parameters.is_empty()
        || total > MAX_ROOT_BOX_VARIABLES
        || residuals.len() != unknowns.len()
        || unknown_domain.len() != unknowns.len()
        || parameter_domain.len() != parameters.len()
    {
        return Err(RootBoxError::DimensionMismatch.into());
    }
    for coordinate in unknowns.iter().chain(parameters) {
        if coordinate.name.len() > MAX_CERTIFICATION_NAME_BYTES {
            return Err(RootBoxError::ResourceLimit("symbol length").into());
        }
        charge(
            meter,
            coordinate.name.len() as u64 + 1,
            coordinate.name.len() as u64,
            1,
        )?;
    }
    let mut coordinates = reserve(total, meter)?;
    coordinates.extend_from_slice(unknowns);
    coordinates.extend_from_slice(parameters);
    check_problem_sources(residuals, &coordinates, meter)?;
    for ball in unknown_domain.iter().chain(parameter_domain) {
        charge_number(ball.midpoint(), meter)?;
        charge_number(ball.radius(), meter)?;
    }
    for (coordinate, interval) in unknown_domain.iter().enumerate() {
        if interval.radius().is_zero() {
            return Err(RootBoxError::DegenerateBox { coordinate }.into());
        }
    }
    Ok(())
}

fn check_partition(
    partition: impl IntoIterator<Item = Step>,
    count: usize,
    leaf_count: usize,
    parameter_domain: &[RealBall],
    meter: &mut impl BudgetMeter,
) -> Result<(), ParametricError> {
    if count == 0
        || count > MAX_PARAMETRIC_ROOT_COVER_NODES
        || leaf_count == 0
        || leaf_count > MAX_PARAMETRIC_ROOT_COVER_LEAVES
    {
        return Err(
            ParametricRootCoverError::InvalidPartition("partition node or leaf count").into(),
        );
    }
    charge(
        meter,
        1,
        (MAX_PARAMETRIC_ROOT_COVER_LEAVES
            + (MAX_PARAMETRIC_ROOT_COVER_DEPTH + 1) * size_of::<usize>()) as u64,
        0,
    )?;
    let mut seen = [false; MAX_PARAMETRIC_ROOT_COVER_LEAVES];
    let mut depths = [0usize; MAX_PARAMETRIC_ROOT_COVER_DEPTH + 1];
    let mut pending = 1;
    let mut used = 0;
    for step in partition {
        charge(meter, 1, 0, 0)?;
        if pending == 0 {
            return Err(
                ParametricRootCoverError::InvalidPartition("trailing partition nodes").into(),
            );
        }
        pending -= 1;
        let depth = depths[pending];
        match step {
            Step::Split { axis } => {
                if depth >= MAX_PARAMETRIC_ROOT_COVER_DEPTH {
                    return Err(
                        ParametricRootCoverError::InvalidPartition("partition depth").into(),
                    );
                }
                if axis >= parameter_domain.len() || parameter_domain[axis].radius().is_zero() {
                    return Err(
                        ParametricRootCoverError::InvalidPartition("split coordinate").into(),
                    );
                }
                depths[pending] = depth + 1;
                depths[pending + 1] = depth + 1;
                pending += 2;
            }
            Step::Leaf { index } => {
                if index >= leaf_count || seen[index] {
                    return Err(ParametricRootCoverError::InvalidPartition(
                        "invalid or repeated leaf reference",
                    )
                    .into());
                }
                seen[index] = true;
                used += 1;
            }
        }
    }
    if pending != 0 || used != leaf_count {
        return Err(ParametricRootCoverError::InvalidPartition(
            "missing partition children or leaves",
        )
        .into());
    }
    Ok(())
}

fn charge_witness(
    witness: &ParametricRootBoxWitness,
    unknowns: usize,
    parameters: usize,
    meter: &mut impl BudgetMeter,
) -> Result<(), ParametricError> {
    charge(meter, (unknowns * unknowns + parameters + 1) as u64, 0, 0)?;
    if witness.unknown_domain.len() != unknowns
        || witness.parameter_domain.len() != parameters
        || witness.center.len() != unknowns
        || witness.preconditioner.len() != unknowns
        || witness
            .preconditioner
            .iter()
            .any(|row| row.len() != unknowns)
    {
        return Err(RootBoxError::DimensionMismatch.into());
    }
    check_precision(witness.precision_digits)?;
    for ball in witness
        .unknown_domain
        .iter()
        .chain(&witness.parameter_domain)
    {
        charge_number(ball.midpoint(), meter)?;
        charge_number(ball.radius(), meter)?;
    }
    for value in witness
        .center
        .iter()
        .chain(witness.preconditioner.iter().flatten())
    {
        charge_number(value, meter)?;
    }
    Ok(())
}

/// Export the exact statement and an untrusted uniform or complete-cover witness.
///
/// Accepted root enclosures, contraction bounds, sensitivities and acceptance
/// flags are absent. Receivers must supply their original source and full
/// statement, then independently recompute every accepted conclusion.
pub fn export_parametric_family(
    receipt: &ParametricFamilyReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, ParametricError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    check_statement(
        receipt.residuals(),
        receipt.unknowns(),
        receipt.parameters(),
        receipt.unknown_domain(),
        receipt.parameter_domain(),
        meter,
    )?;
    let witness = match receipt {
        ParametricFamilyReceipt::RootBox(receipt) => {
            charge_witness(
                receipt.witness(),
                receipt.unknowns().len(),
                receipt.parameters().len(),
                meter,
            )?;
            WitnessView::RootBox(receipt.witness().into())
        }
        ParametricFamilyReceipt::Cover(receipt) => {
            let witness = receipt.witness();
            check_precision(witness.precision_digits)?;
            check_partition(
                witness.partition.iter().copied().map(Step::from),
                witness.partition.len(),
                witness.leaves.len(),
                receipt.parameter_domain(),
                meter,
            )?;
            for leaf in &witness.leaves {
                charge_witness(
                    leaf,
                    receipt.unknowns().len(),
                    receipt.parameters().len(),
                    meter,
                )?;
            }
            WitnessView::Cover(witness.into())
        }
    };
    Ok(encode(
        &Artifact {
            schema_version: PARAMETRIC_FAMILY_ARTIFACT_SCHEMA_VERSION,
            context_digest,
            residuals: receipt.residuals(),
            unknowns: receipt.unknowns(),
            parameters: receipt.parameters(),
            unknown_domain: receipt.unknown_domain(),
            parameter_domain: receipt.parameter_domain(),
            witness,
        },
        meter,
    )?)
}

enum Candidate {
    RootBox(ParametricRootBoxWitness),
    Cover(ParametricRootCoverWitness),
}

impl Candidate {
    fn view(&self) -> WitnessView<'_> {
        match self {
            Self::RootBox(witness) => WitnessView::RootBox(witness.into()),
            Self::Cover(witness) => WitnessView::Cover(witness.into()),
        }
    }
}

fn decode_candidate(
    raw: &RawValue,
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
    meter: &mut impl BudgetMeter,
) -> Result<Candidate, ParametricError> {
    let bytes = raw.get().len() as u64;
    charge(meter, 2 * bytes + 1, bytes * 16 + 1024, bytes / 2 + 64)?;
    let wire: WitnessInput<'_> =
        serde_json::from_str(raw.get()).map_err(|error| encoding_error(&error))?;
    let unknowns = unknown_domain.len();
    let parameters = parameter_domain.len();
    match wire.kind {
        Kind::RootBox => {
            let input: RootBoxInput<'_> =
                serde_json::from_str(wire.data.get()).map_err(|error| encoding_error(&error))?;
            input.check_shape(unknowns, parameters, meter)?;
            let witness = input.decode(meter)?;
            charge_witness(&witness, unknowns, parameters, meter)?;
            if witness.unknown_domain != unknown_domain
                || witness.parameter_domain != parameter_domain
            {
                return Err(CertificationError::DomainMismatch.into());
            }
            Ok(Candidate::RootBox(witness))
        }
        Kind::Cover => {
            let input: CoverInput<'_> =
                serde_json::from_str(wire.data.get()).map_err(|error| encoding_error(&error))?;
            check_precision(input.precision_digits)?;
            check_partition(
                input.partition.0.iter().copied(),
                input.partition.0.len(),
                input.leaves.0.len(),
                parameter_domain,
                meter,
            )?;
            for leaf in &input.leaves.0 {
                leaf.check_shape(unknowns, parameters, meter)?;
                if leaf.precision_digits != input.precision_digits {
                    return Err(ParametricRootCoverError::InvalidPartition(
                        "leaf precision differs from cover",
                    )
                    .into());
                }
            }
            let mut leaves = reserve(input.leaves.0.len(), meter)?;
            for leaf in input.leaves.0 {
                let witness = leaf.decode(meter)?;
                charge_witness(&witness, unknowns, parameters, meter)?;
                if witness.unknown_domain != unknown_domain {
                    return Err(CertificationError::DomainMismatch.into());
                }
                leaves.push(witness);
            }
            let mut partition = reserve(input.partition.0.len(), meter)?;
            partition.extend(
                input
                    .partition
                    .0
                    .into_iter()
                    .map(ParametricRootCoverStep::from),
            );
            Ok(Candidate::Cover(ParametricRootCoverWitness {
                leaves,
                partition,
                precision_digits: input.precision_digits,
            }))
        }
    }
}

/// Replay a complete parameter-root family against the receiver's exact statement.
///
/// Source, both ordered typed coordinate groups, both complete boxes, schema
/// and context are bound before arithmetic-bearing witness decoding. Counts,
/// nesting and integer limbs are bounded before exact scalar construction;
/// canonical reencoding rejects alternative spellings and stored conclusions.
/// All decoding and verification share the supplied meter. Only the independent
/// core checker grants acceptance; this function never runs a search generator.
pub fn replay_parametric_family(
    bytes: &[u8],
    expected_problem: &ParametricProblem,
    expected_unknown_domain: &[RealBall],
    expected_parameter_domain: &[RealBall],
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<ParametricFamilyReceipt, ParametricError> {
    meter.checkpoint().map_err(CertificationError::from)?;
    if bytes.len() > MAX_PARAMETRIC_FAMILY_ARTIFACT_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_PARAMETRIC_FAMILY_ARTIFACT_BYTES,
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
    let wire: UntrustedArtifact<'_> =
        serde_json::from_slice(bytes).map_err(|error| encoding_error(&error))?;
    if wire.schema_version != PARAMETRIC_FAMILY_ARTIFACT_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: PARAMETRIC_FAMILY_ARTIFACT_SCHEMA_VERSION,
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
    check_statement(
        expected_problem.residuals(),
        expected_problem.unknowns(),
        expected_problem.parameters(),
        expected_unknown_domain,
        expected_parameter_domain,
        meter,
    )?;
    for (canonical, received, source) in [
        (
            encode(expected_problem.residuals(), meter)?,
            wire.residuals,
            true,
        ),
        (
            encode(expected_problem.unknowns(), meter)?,
            wire.unknowns,
            true,
        ),
        (
            encode(expected_problem.parameters(), meter)?,
            wire.parameters,
            true,
        ),
        (
            encode(expected_unknown_domain, meter)?,
            wire.unknown_domain,
            false,
        ),
        (
            encode(expected_parameter_domain, meter)?,
            wire.parameter_domain,
            false,
        ),
    ] {
        charge(meter, canonical.len() as u64 + 1, 0, 0)?;
        if canonical.as_slice() != received.get().as_bytes() {
            return Err(if source {
                CertificationError::SourceMismatch
            } else {
                CertificationError::DomainMismatch
            }
            .into());
        }
    }
    let candidate = decode_candidate(
        wire.witness,
        expected_unknown_domain,
        expected_parameter_domain,
        meter,
    )?;
    let canonical = encode(
        &Artifact {
            schema_version: PARAMETRIC_FAMILY_ARTIFACT_SCHEMA_VERSION,
            context_digest: expected_context,
            residuals: expected_problem.residuals(),
            unknowns: expected_problem.unknowns(),
            parameters: expected_problem.parameters(),
            unknown_domain: expected_unknown_domain,
            parameter_domain: expected_parameter_domain,
            witness: candidate.view(),
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical.into());
    }
    match candidate {
        Candidate::RootBox(witness) => expected_problem
            .verify_root_box(&witness, meter)
            .map(ParametricFamilyReceipt::RootBox),
        Candidate::Cover(witness) => expected_problem
            .verify_cover(
                expected_unknown_domain,
                expected_parameter_domain,
                &witness,
                meter,
            )
            .map(ParametricFamilyReceipt::Cover),
    }
}
