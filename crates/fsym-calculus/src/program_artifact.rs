//! Bounded, canonical transport for independently checkable program candidates.
//!
//! Decoding returns an untrusted candidate, never a verified program. The
//! admission helpers bind the caller's original source and ordered typed
//! variables before independent metered verification. Receipts, shared evaluation
//! plans and numeric caches are not part of this format. The portable proof profile
//! is the generator's single definitional equality, not arbitrary proof trees.
//! Parsing and encoding consume the supplied cumulative meter; these logical
//! charges do not describe allocator RSS or hard cancellation latency.

#![forbid(unsafe_code)]

pub mod context;

use crate::compile::{CompiledExpr, CompiledOp, MAX_COMPILE_OPS};
use crate::sparse_hessian::{ObjectiveCandidate, ObjectiveCompilationError, VerifiedObjective};
use crate::sparse_jacobian::{MAX_SPARSE_AXES, SparseJacobian};
use crate::verified::{
    MAX_VERIFIED_ARTIFACT_BYTES, MAX_VERIFIED_EXPRESSION_NODES, MAX_VERIFIED_NONZERO_ENTRIES,
    MAX_VERIFIED_SOURCE_DEPTH, MAX_VERIFIED_STORED_OPS, VERIFIED_ARTIFACT_SCHEMA_VERSION,
    VerifiedCompilationError, VerifiedResidualSystem,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_proof_kernel::{Claim, ProofRule, StepId};
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::fmt;
use std::io::{self, Write};
use thiserror::Error;

pub const PROGRAM_CANDIDATE_SCHEMA_VERSION: u32 = 1;
pub const MAX_PROGRAM_CANDIDATE_BYTES: usize = 1024 * 1024;
pub const MAX_PROGRAM_CANDIDATE_JSON_DEPTH: usize = 96;
pub const MAX_PROGRAM_CANDIDATE_INTEGER_BITS: usize = 4096;
pub const MAX_PROGRAM_CANDIDATE_NAME_BYTES: usize = 4096;
const MAX_RULE_NAME_BYTES: usize = 256;

#[derive(Debug, Error)]
pub enum ProgramArtifactError {
    #[error("program candidate transport stopped: {0}")]
    Budget(#[from] MeterError),
    #[error("program candidate has {actual} bytes, exceeding {max}")]
    ByteLimit { actual: usize, max: usize },
    #[error("program candidate JSON depth {depth} exceeds {max}")]
    DepthLimit { depth: usize, max: usize },
    #[error("program candidate {resource} exceeds {limit}")]
    ResourceLimit {
        resource: &'static str,
        limit: usize,
    },
    #[error("program candidate schema {found} is unsupported; expected {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("program candidate kind does not match {expected}")]
    WrongKind { expected: &'static str },
    #[error("invalid program candidate JSON: {0}")]
    Encoding(String),
    #[error("program candidate JSON is not canonical")]
    NonCanonical,
    #[error("non-finite bytecode cannot be represented losslessly as JSON")]
    NonFiniteBytecode,
    #[error("program candidate transport allocation refused")]
    AllocationFailure,
}

/// A failed source binding or independent check never produces an accepted
/// program. Error variants retain the underlying transport, budget and verifier
/// failures so callers can distinguish exhaustion/cancellation from rejection.
#[derive(Debug, Error)]
pub enum ProgramAdmissionError {
    #[error("program source admission stopped: {0}")]
    Transport(#[from] ProgramArtifactError),
    #[error("program candidate does not match the caller's {0}")]
    SourceMismatch(&'static str),
    #[error("residual program verification failed: {0}")]
    Residual(#[from] VerifiedCompilationError),
    #[error("objective program verification failed: {0}")]
    Objective(#[from] ObjectiveCompilationError),
}

fn charge<M: BudgetMeter + ?Sized>(
    meter: &mut M,
    steps: usize,
    bytes: usize,
    allocations: usize,
) -> Result<(), ProgramArtifactError> {
    meter.checkpoint()?;
    let values = [
        (Dimension::ComputeSteps, steps),
        (Dimension::MemoryBytes, bytes),
        (Dimension::AllocationCount, allocations),
    ];
    let mut charges = [(Dimension::ComputeSteps, 0u64); 3];
    let mut count = 0;
    for (dimension, amount) in values {
        if amount != 0 {
            charges[count] = (dimension, amount as u64);
            count += 1;
        }
    }
    if count != 0 {
        meter.charge_batch(&charges[..count])?;
    }
    Ok(())
}

fn invalid(message: &'static str) -> ProgramArtifactError {
    ProgramArtifactError::Encoding(message.to_owned())
}

fn encoding(error: serde_json::Error) -> ProgramArtifactError {
    // Diagnostics never copy an attacker-controlled unknown field or value.
    ProgramArtifactError::Encoding(format!(
        "invalid syntax or field at line {}, column {}",
        error.line(),
        error.column()
    ))
}

fn limit(
    actual: usize,
    maximum: usize,
    resource: &'static str,
) -> Result<(), ProgramArtifactError> {
    if actual > maximum {
        return Err(ProgramArtifactError::ResourceLimit {
            resource,
            limit: maximum,
        });
    }
    Ok(())
}

fn parse_charge<M: BudgetMeter + ?Sized>(
    bytes: usize,
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    // Borrowed RawValue parsing does not construct expression trees. Cover
    // serde's escaped-key scratch growth and bounded error bookkeeping.
    charge(meter, bytes + 1, bytes * 4 + 1024, 64)
}

fn read<'a, T: Deserialize<'a>, M: BudgetMeter + ?Sized>(
    raw: &'a RawValue,
    meter: &mut M,
) -> Result<T, ProgramArtifactError> {
    parse_charge(raw.get().len(), meter)?;
    let result = serde_json::from_str(raw.get()).map_err(encoding);
    meter.checkpoint()?;
    result
}

/// Walk a JSON array without collecting its elements. Every element stays a
/// borrowed RawValue until its own bounded shape has been inspected.
fn array<'a, F>(
    raw: &'a RawValue,
    maximum: usize,
    resource: &'static str,
    mut visit: F,
) -> Result<usize, ProgramArtifactError>
where
    F: FnMut(usize, &'a RawValue) -> Result<(), ProgramArtifactError>,
{
    struct ArrayVisitor<'b, F> {
        maximum: usize,
        resource: &'static str,
        visit: &'b mut F,
        failure: &'b mut Option<ProgramArtifactError>,
    }
    impl<'de, F> Visitor<'de> for ArrayVisitor<'_, F>
    where
        F: FnMut(usize, &'de RawValue) -> Result<(), ProgramArtifactError>,
    {
        type Value = usize;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded program candidate sequence")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<usize, A::Error> {
            let mut count = 0;
            while let Some(value) = sequence.next_element::<&'de RawValue>()? {
                let result = if count == self.maximum {
                    Err(ProgramArtifactError::ResourceLimit {
                        resource: self.resource,
                        limit: self.maximum,
                    })
                } else {
                    (self.visit)(count, value)
                };
                if let Err(error) = result {
                    *self.failure = Some(error);
                    return Err(A::Error::custom("program candidate sequence refused"));
                }
                count += 1;
            }
            Ok(count)
        }
    }
    let mut failure = None;
    let mut decoder = serde_json::Deserializer::from_str(raw.get());
    let result = serde::Deserializer::deserialize_seq(
        &mut decoder,
        ArrayVisitor {
            maximum,
            resource,
            visit: &mut visit,
            failure: &mut failure,
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    let result = result.map_err(encoding)?;
    decoder.end().map_err(encoding)?;
    Ok(result)
}

#[derive(Default)]
struct JsonNesting {
    depth: usize,
    quoted: bool,
    escaped: bool,
}

impl JsonNesting {
    fn scan(&mut self, bytes: &[u8]) -> Result<(), ProgramArtifactError> {
        for &byte in bytes {
            if self.quoted {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.quoted = false;
                }
            } else {
                match byte {
                    b'"' => self.quoted = true,
                    b'[' | b'{' => {
                        self.depth += 1;
                        if self.depth > MAX_PROGRAM_CANDIDATE_JSON_DEPTH {
                            return Err(ProgramArtifactError::DepthLimit {
                                depth: self.depth,
                                max: MAX_PROGRAM_CANDIDATE_JSON_DEPTH,
                            });
                        }
                    }
                    b']' | b'}' => self.depth = self.depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

struct Writer<'a, M: BudgetMeter + ?Sized> {
    bytes: Vec<u8>,
    nesting: JsonNesting,
    meter: &'a mut M,
    failure: Option<ProgramArtifactError>,
}

impl<M: BudgetMeter + ?Sized> Write for Writer<'_, M> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let result = (|| {
            if self.failure.is_some() {
                return Err(invalid("writer already refused"));
            }
            let required = self.bytes.len().saturating_add(bytes.len());
            if required > MAX_PROGRAM_CANDIDATE_BYTES {
                return Err(ProgramArtifactError::ByteLimit {
                    actual: required,
                    max: MAX_PROGRAM_CANDIDATE_BYTES,
                });
            }
            charge(self.meter, bytes.len() + 1, 0, 0)?;
            self.nesting.scan(bytes)?;
            if required > self.bytes.capacity() {
                let capacity = required
                    .max(self.bytes.capacity().saturating_mul(2))
                    .min(MAX_PROGRAM_CANDIDATE_BYTES);
                charge(self.meter, 1, capacity, 1)?;
                self.bytes
                    .try_reserve_exact(capacity - self.bytes.len())
                    .map_err(|_| ProgramArtifactError::AllocationFailure)?;
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        })();
        result.map_err(|error| {
            if self.failure.is_none() {
                self.failure = Some(error);
            }
            io::Error::other("program candidate writer refused")
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Serialize)]
struct EnvelopeRef<'a, T: ?Sized> {
    schema_version: u32,
    kind: &'static str,
    candidate: &'a T,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope<'a> {
    schema_version: u32,
    #[serde(borrow)]
    kind: &'a RawValue,
    #[serde(borrow)]
    candidate: &'a RawValue,
}

/// The old Python residual format predates the kind-tagged portable envelope.
/// Keep its import boundary separate: it is not a canonical native artifact.
/// Explicit nulls count as present fields, so they cannot disguise a mixed
/// legacy/portable envelope or an unknown acceptance flag.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompatibleResidualEnvelope<'a> {
    schema_version: u32,
    #[serde(default, borrow, deserialize_with = "present_raw")]
    kind: Option<&'a RawValue>,
    #[serde(default, borrow, deserialize_with = "present_raw")]
    candidate: Option<&'a RawValue>,
    #[serde(default, borrow, deserialize_with = "present_raw")]
    jacobian: Option<&'a RawValue>,
}

fn present_raw<'de, D>(deserializer: D) -> Result<Option<&'de RawValue>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    <&RawValue>::deserialize(deserializer).map(Some)
}

fn encode<T: Serialize + ?Sized, M: BudgetMeter + ?Sized>(
    candidate: &T,
    kind: &'static str,
    meter: &mut M,
) -> Result<Vec<u8>, ProgramArtifactError> {
    meter.checkpoint()?;
    let mut writer = Writer {
        bytes: Vec::new(),
        nesting: JsonNesting::default(),
        meter,
        failure: None,
    };
    let result = serde_json::to_writer(
        &mut writer,
        &EnvelopeRef {
            schema_version: PROGRAM_CANDIDATE_SCHEMA_VERSION,
            kind,
            candidate,
        },
    );
    if let Some(error) = writer.failure {
        return Err(error);
    }
    result.map_err(encoding)?;
    writer.meter.checkpoint()?;
    Ok(writer.bytes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SymbolWire<'a> {
    #[serde(borrow)]
    name: &'a RawValue,
    #[serde(borrow)]
    identity: Option<&'a RawValue>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityWire {
    assumptions: [u8; 32],
}

#[derive(Deserialize)]
enum ExpressionWire<'a> {
    #[serde(borrow)]
    Sym(&'a RawValue),
    #[serde(borrow)]
    Integer(&'a RawValue),
    #[serde(borrow)]
    Rational(&'a RawValue),
    Const(Constant),
    #[serde(borrow)]
    Add(&'a RawValue),
    #[serde(borrow)]
    Mul(&'a RawValue),
    #[serde(borrow)]
    Pow(&'a RawValue),
    #[serde(borrow)]
    Function(&'a RawValue),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JacobianWire<'a> {
    #[serde(borrow)]
    residuals: &'a RawValue,
    #[serde(borrow)]
    vars: &'a RawValue,
    #[serde(borrow)]
    pattern: PatternWire<'a>,
    #[serde(borrow)]
    entries: &'a RawValue,
    #[serde(borrow)]
    column_colors: &'a RawValue,
    num_colors: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PatternWire<'a> {
    num_rows: usize,
    num_cols: usize,
    #[serde(borrow)]
    nonzeros: &'a RawValue,
    nnz: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EntryWire<'a> {
    row: usize,
    col: usize,
    #[serde(borrow)]
    symbolic_deriv: &'a RawValue,
    #[serde(borrow)]
    compiled_deriv: &'a RawValue,
    #[serde(borrow)]
    derivation: &'a RawValue,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProofWire<'a> {
    #[serde(borrow)]
    steps: &'a RawValue,
    root: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepWire<'a> {
    id: u32,
    #[serde(borrow)]
    rule: RuleWire<'a>,
    #[serde(borrow)]
    claim: ClaimWire<'a>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
enum RuleWire<'a> {
    DefinitionalReduction {
        #[serde(borrow)]
        lhs: &'a RawValue,
        #[serde(borrow)]
        rhs: &'a RawValue,
        #[serde(borrow)]
        rule_name: &'a RawValue,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
enum ClaimWire<'a> {
    Equality {
        #[serde(borrow)]
        lhs: &'a RawValue,
        #[serde(borrow)]
        rhs: &'a RawValue,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BytecodeWire<'a> {
    #[serde(borrow)]
    ops: &'a RawValue,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObjectiveWire<'a> {
    #[serde(borrow)]
    compiled_objective: &'a RawValue,
    #[serde(borrow)]
    gradient: &'a RawValue,
    #[serde(borrow)]
    hessian: &'a RawValue,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedObjective {
    compiled_objective: CompiledExpr,
    gradient: SparseJacobian,
    hessian: SparseJacobian,
}

#[derive(Default)]
struct Preflight {
    nodes: usize,
    operations: usize,
    scalar_steps: usize,
    scalar_bytes: usize,
    scalar_allocations: usize,
}

fn string<M: BudgetMeter + ?Sized>(
    raw: &RawValue,
    maximum: usize,
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    // A Unicode escape needs at most six input bytes for each decoded byte.
    limit(raw.get().len(), maximum * 6 + 2, "encoded name bytes")?;
    let value: String = read(raw, meter)?;
    limit(value.len(), maximum, "name bytes")
}

fn symbol<M: BudgetMeter + ?Sized>(
    raw: &RawValue,
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    let wire: SymbolWire<'_> = read(raw, meter)?;
    string(wire.name, MAX_PROGRAM_CANDIDATE_NAME_BYTES, meter)?;
    if let Some(identity) = wire.identity {
        let identity: IdentityWire = read(identity, meter)?;
        let _ = identity.assumptions;
    }
    Ok(())
}

fn integer<M: BudgetMeter + ?Sized>(
    raw: &RawValue,
    meter: &mut M,
) -> Result<(i8, usize), ProgramArtifactError> {
    let (sign, digits): (i8, &RawValue) = read(raw, meter)?;
    if !(-1..=1).contains(&sign) {
        return Err(invalid("invalid integer sign"));
    }
    parse_charge(digits.get().len(), meter)?;
    let mut last = 0u32;
    let count = array(
        digits,
        MAX_PROGRAM_CANDIDATE_INTEGER_BITS.div_ceil(32),
        "integer limbs",
        |_, digit| {
            last = read(digit, meter)?;
            Ok(())
        },
    )?;
    if (count == 0 && sign != 0) || (count != 0 && (sign == 0 || last == 0)) {
        return Err(ProgramArtifactError::NonCanonical);
    }
    Ok((sign, count))
}

fn rational<M: BudgetMeter + ?Sized>(
    raw: &RawValue,
    meter: &mut M,
) -> Result<usize, ProgramArtifactError> {
    let [numerator, denominator]: [&RawValue; 2] = read(raw, meter)?;
    let (_, numerator_limbs) = integer(numerator, meter)?;
    let (denominator_sign, denominator_limbs) = integer(denominator, meter)?;
    if denominator_sign != 1 {
        return Err(ProgramArtifactError::NonCanonical);
    }
    // Scalar allocation and gcd work start only after both limb counts passed.
    // fsym-rational's decoder checks coprimality without normalizing an invalid
    // value. Its bounded gcd is prepaid here and again at owned materialization.
    let limbs = numerator_limbs + denominator_limbs + 1;
    charge(meter, limbs * limbs * 64, limbs * 512, limbs * 8)?;
    parse_charge(raw.get().len(), meter)?;
    let value = serde_json::from_str::<BigRational>(raw.get());
    meter.checkpoint()?;
    // The exact tuple shape, signs, nonzero denominator and canonical limbs
    // have already passed; rejection at this point means an unreduced scalar.
    value.map_err(|_| ProgramArtifactError::NonCanonical)?;
    Ok(limbs)
}

impl Preflight {
    fn expression<M: BudgetMeter + ?Sized>(
        &mut self,
        raw: &RawValue,
        depth: usize,
        maximum_depth: usize,
        local_nodes: &mut usize,
        maximum_nodes: usize,
        meter: &mut M,
    ) -> Result<(), ProgramArtifactError> {
        limit(depth, maximum_depth, "expression depth")?;
        *local_nodes += 1;
        self.nodes += 1;
        limit(*local_nodes, maximum_nodes, "expression nodes")?;
        limit(
            self.nodes,
            MAX_VERIFIED_EXPRESSION_NODES,
            "aggregate expression nodes",
        )?;
        match read::<ExpressionWire<'_>, _>(raw, meter)? {
            ExpressionWire::Sym(raw) => symbol(raw, meter),
            ExpressionWire::Integer(raw) => integer(raw, meter).map(|_| ()),
            ExpressionWire::Rational(raw) => {
                let limbs = rational(raw, meter)?;
                self.scalar_steps += limbs * limbs * 64;
                self.scalar_bytes += limbs * 512;
                self.scalar_allocations += limbs * 8;
                Ok(())
            }
            ExpressionWire::Const(constant) => {
                let _ = constant;
                Ok(())
            }
            ExpressionWire::Add(children) | ExpressionWire::Mul(children) => {
                parse_charge(children.get().len(), meter)?;
                array(
                    children,
                    maximum_nodes,
                    "expression children",
                    |_, child| {
                        self.expression(
                            child,
                            depth + 1,
                            maximum_depth,
                            local_nodes,
                            maximum_nodes,
                            meter,
                        )
                    },
                )?;
                Ok(())
            }
            ExpressionWire::Pow(raw) => {
                let children: [&RawValue; 2] = read(raw, meter)?;
                for child in children {
                    self.expression(
                        child,
                        depth + 1,
                        maximum_depth,
                        local_nodes,
                        maximum_nodes,
                        meter,
                    )?;
                }
                Ok(())
            }
            ExpressionWire::Function(raw) => {
                let [name, children]: [&RawValue; 2] = read(raw, meter)?;
                string(name, MAX_PROGRAM_CANDIDATE_NAME_BYTES, meter)?;
                parse_charge(children.get().len(), meter)?;
                array(children, maximum_nodes, "function arguments", |_, child| {
                    self.expression(
                        child,
                        depth + 1,
                        maximum_depth,
                        local_nodes,
                        maximum_nodes,
                        meter,
                    )
                })?;
                Ok(())
            }
        }
    }

    fn source<M: BudgetMeter + ?Sized>(
        &mut self,
        raw: &RawValue,
        proof: bool,
        meter: &mut M,
    ) -> Result<(), ProgramArtifactError> {
        self.expression(
            raw,
            0,
            MAX_VERIFIED_SOURCE_DEPTH + usize::from(proof),
            &mut 0,
            MAX_COMPILE_OPS + 2 * usize::from(proof),
            meter,
        )
    }

    fn bytecode<M: BudgetMeter + ?Sized>(
        &mut self,
        raw: &RawValue,
        meter: &mut M,
    ) -> Result<(), ProgramArtifactError> {
        let wire: BytecodeWire<'_> = read(raw, meter)?;
        parse_charge(wire.ops.get().len(), meter)?;
        array(
            wire.ops,
            MAX_COMPILE_OPS,
            "bytecode operations",
            |_, raw| {
                self.operations += 1;
                limit(
                    self.operations,
                    MAX_VERIFIED_STORED_OPS,
                    "aggregate bytecode operations",
                )?;
                let operation: CompiledOp = read(raw, meter)?;
                finite_operation(&operation)
            },
        )?;
        Ok(())
    }

    fn proof<M: BudgetMeter + ?Sized>(
        &mut self,
        raw: &RawValue,
        meter: &mut M,
    ) -> Result<(), ProgramArtifactError> {
        let proof: ProofWire<'_> = read(raw, meter)?;
        if proof.root != 1 {
            return Err(invalid("portable derivative proof root must be step one"));
        }
        parse_charge(proof.steps.get().len(), meter)?;
        let count = array(proof.steps, 1, "derivative proof steps", |_, raw| {
            let step: StepWire<'_> = read(raw, meter)?;
            if step.id != 1 {
                return Err(invalid("portable derivative proof id must be one"));
            }
            let RuleWire::DefinitionalReduction {
                lhs,
                rhs,
                rule_name,
            } = step.rule;
            string(rule_name, MAX_RULE_NAME_BYTES, meter)?;
            self.source(lhs, true, meter)?;
            self.source(rhs, true, meter)?;
            let ClaimWire::Equality { lhs, rhs } = step.claim;
            self.source(lhs, true, meter)?;
            self.source(rhs, true, meter)
        })?;
        if count != 1 {
            return Err(invalid("portable derivative proof must contain one step"));
        }
        Ok(())
    }

    fn jacobian<M: BudgetMeter + ?Sized>(
        &mut self,
        raw: &RawValue,
        meter: &mut M,
    ) -> Result<(), ProgramArtifactError> {
        let wire: JacobianWire<'_> = read(raw, meter)?;
        let rows = wire.pattern.num_rows;
        let columns = wire.pattern.num_cols;
        limit(rows, MAX_SPARSE_AXES, "residual axis")?;
        limit(columns, MAX_SPARSE_AXES, "variable axis")?;
        let maximum_entries = rows
            .saturating_mul(columns)
            .min(MAX_VERIFIED_NONZERO_ENTRIES);
        limit(wire.pattern.nnz, maximum_entries, "nonzero entries")?;
        limit(wire.num_colors, columns, "column colors")?;
        parse_charge(wire.residuals.get().len(), meter)?;
        let count = array(wire.residuals, rows, "residual count", |_, raw| {
            self.source(raw, false, meter)
        })?;
        if count != rows {
            return Err(invalid("residual count does not match sparse pattern"));
        }
        parse_charge(wire.vars.get().len(), meter)?;
        let count = array(wire.vars, columns, "variable count", |_, raw| {
            symbol(raw, meter)
        })?;
        if count != columns {
            return Err(invalid("variable count does not match sparse pattern"));
        }
        parse_charge(wire.pattern.nonzeros.get().len(), meter)?;
        let count = array(
            wire.pattern.nonzeros,
            maximum_entries,
            "nonzero coordinates",
            |_, raw| {
                let [row, column]: [usize; 2] = read(raw, meter)?;
                if row >= rows || column >= columns {
                    return Err(invalid("nonzero coordinate lies outside sparse shape"));
                }
                Ok(())
            },
        )?;
        if count != wire.pattern.nnz {
            return Err(invalid("nonzero count does not match sparse pattern"));
        }
        parse_charge(wire.entries.get().len(), meter)?;
        let count = array(
            wire.entries,
            maximum_entries,
            "derivative entries",
            |_, raw| {
                let entry: EntryWire<'_> = read(raw, meter)?;
                if entry.row >= rows || entry.col >= columns {
                    return Err(invalid("derivative entry lies outside sparse shape"));
                }
                self.source(entry.symbolic_deriv, false, meter)?;
                self.bytecode(entry.compiled_deriv, meter)?;
                self.proof(entry.derivation, meter)
            },
        )?;
        if count != wire.pattern.nnz {
            return Err(invalid(
                "derivative entry count does not match sparse pattern",
            ));
        }
        parse_charge(wire.column_colors.get().len(), meter)?;
        let count = array(
            wire.column_colors,
            columns,
            "column color count",
            |_, raw| {
                let color: usize = read(raw, meter)?;
                if color >= wire.num_colors {
                    return Err(invalid("column color lies outside declared colors"));
                }
                Ok(())
            },
        )?;
        if count != columns {
            return Err(invalid("column color count does not match variable axis"));
        }
        Ok(())
    }
}

fn preflight<'a, M: BudgetMeter + ?Sized>(
    bytes: &'a [u8],
    expected_kind: &'static str,
    meter: &mut M,
) -> Result<(&'a RawValue, Preflight), ProgramArtifactError> {
    meter.checkpoint()?;
    if bytes.len() > MAX_PROGRAM_CANDIDATE_BYTES {
        return Err(ProgramArtifactError::ByteLimit {
            actual: bytes.len(),
            max: MAX_PROGRAM_CANDIDATE_BYTES,
        });
    }
    let mut nesting = JsonNesting::default();
    for chunk in bytes.chunks(512) {
        charge(meter, chunk.len(), 0, 0)?;
        nesting.scan(chunk)?;
    }
    parse_charge(bytes.len(), meter)?;
    let envelope: Envelope<'_> = serde_json::from_slice(bytes).map_err(encoding)?;
    if envelope.schema_version != PROGRAM_CANDIDATE_SCHEMA_VERSION {
        return Err(ProgramArtifactError::UnsupportedVersion {
            found: envelope.schema_version,
            supported: PROGRAM_CANDIDATE_SCHEMA_VERSION,
        });
    }
    // A kind tag is a fixed ASCII token; escaped spelling is noncanonical.
    let kind: &str = read(envelope.kind, meter)?;
    if kind != expected_kind {
        return Err(ProgramArtifactError::WrongKind {
            expected: expected_kind,
        });
    }
    let mut inspection = Preflight::default();
    if expected_kind == "residual" {
        inspection.jacobian(envelope.candidate, meter)?;
    } else {
        let wire: ObjectiveWire<'_> = read(envelope.candidate, meter)?;
        inspection.bytecode(wire.compiled_objective, meter)?;
        inspection.jacobian(wire.gradient, meter)?;
        inspection.jacobian(wire.hessian, meter)?;
    }
    meter.checkpoint()?;
    Ok((envelope.candidate, inspection))
}

fn finite_operation(operation: &CompiledOp) -> Result<(), ProgramArtifactError> {
    if matches!(operation, CompiledOp::LoadConst(value) | CompiledOp::Pow(value) if !value.is_finite())
    {
        return Err(ProgramArtifactError::NonFiniteBytecode);
    }
    Ok(())
}

fn finite_program<M: BudgetMeter + ?Sized>(
    program: &CompiledExpr,
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    for operation in program.ops() {
        charge(meter, 1, 0, 0)?;
        finite_operation(operation)?;
    }
    Ok(())
}

fn retained_expression<M: BudgetMeter + ?Sized>(
    expression: &Expr,
    depth: usize,
    nodes: &mut usize,
    proof: bool,
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    charge(meter, 1, 0, 0)?;
    limit(
        depth,
        MAX_VERIFIED_SOURCE_DEPTH + usize::from(proof),
        "expression depth",
    )?;
    *nodes += 1;
    limit(
        *nodes,
        MAX_COMPILE_OPS + 2 * usize::from(proof),
        "expression nodes",
    )?;
    match expression {
        Expr::Integer(value) => limit(
            value.bits() as usize,
            MAX_PROGRAM_CANDIDATE_INTEGER_BITS,
            "integer bits",
        ),
        Expr::Rational(value) => limit(
            value.numer().bits().max(value.denom().bits()) as usize,
            MAX_PROGRAM_CANDIDATE_INTEGER_BITS,
            "rational bits",
        ),
        Expr::Sym(symbol) => limit(
            symbol.name.len(),
            MAX_PROGRAM_CANDIDATE_NAME_BYTES,
            "name bytes",
        ),
        Expr::Const(_) => Ok(()),
        Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
            if let Expr::Function(name, _) = expression {
                limit(name.len(), MAX_PROGRAM_CANDIDATE_NAME_BYTES, "name bytes")?;
            }
            for child in children {
                retained_expression(child, depth + 1, nodes, proof, meter)?;
            }
            Ok(())
        }
        Expr::Pow(base, exponent) => {
            retained_expression(base, depth + 1, nodes, proof, meter)?;
            retained_expression(exponent, depth + 1, nodes, proof, meter)
        }
    }
}

/// Inspect borrowed retained values before serde can traverse their numeric
/// payloads. This introduces no expression/proof clone or serialization buffer.
fn retained_jacobian<M: BudgetMeter + ?Sized>(
    jacobian: &SparseJacobian,
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    for variable in &jacobian.vars {
        charge(meter, 1, 0, 0)?;
        limit(
            variable.name.len(),
            MAX_PROGRAM_CANDIDATE_NAME_BYTES,
            "name bytes",
        )?;
    }
    for source in &jacobian.residuals {
        retained_expression(source, 0, &mut 0, false, meter)?;
    }
    for entry in &jacobian.entries {
        retained_expression(&entry.symbolic_deriv, 0, &mut 0, false, meter)?;
        finite_program(&entry.compiled_deriv, meter)?;
        let proof = &entry.derivation;
        if proof.steps.len() != 1 || proof.root != StepId(1) || proof.steps[0].id != StepId(1) {
            return Err(invalid("portable derivative proof must be one rooted step"));
        }
        let ProofRule::DefinitionalReduction {
            lhs,
            rhs,
            rule_name,
        } = &proof.steps[0].rule
        else {
            return Err(invalid("portable derivative proof must be definitional"));
        };
        limit(rule_name.len(), MAX_RULE_NAME_BYTES, "rule name bytes")?;
        retained_expression(lhs, 0, &mut 0, true, meter)?;
        retained_expression(rhs, 0, &mut 0, true, meter)?;
        let Claim::Equality { lhs, rhs } = &proof.steps[0].claim else {
            return Err(invalid("portable derivative proof must claim equality"));
        };
        retained_expression(lhs, 0, &mut 0, true, meter)?;
        retained_expression(rhs, 0, &mut 0, true, meter)?;
    }
    Ok(())
}

fn materialization<M: BudgetMeter + ?Sized>(
    raw: &RawValue,
    inspection: &Preflight,
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    // Only already-preflighted bounded Vec/Expr/proof shapes reach serde. The
    // CompiledExpr decoder also validates each bounded control-flow graph;
    // reserve its synchronous structural work and temporary storage here.
    let bytes = raw.get().len();
    charge(
        meter,
        bytes * 4 + inspection.nodes + inspection.operations * 8 + inspection.scalar_steps + 1,
        bytes * 64 + inspection.operations * 32 + inspection.scalar_bytes + 4096,
        bytes + inspection.operations + inspection.scalar_allocations + 64,
    )
}

fn canonical<T: Serialize + ?Sized, M: BudgetMeter + ?Sized>(
    candidate: &T,
    kind: &'static str,
    bytes: &[u8],
    meter: &mut M,
) -> Result<(), ProgramArtifactError> {
    let encoded = encode(candidate, kind, meter)?;
    charge(meter, encoded.len() + bytes.len() + 1, 0, 0)?;
    if encoded != bytes {
        return Err(ProgramArtifactError::NonCanonical);
    }
    meter.checkpoint()?;
    Ok(())
}

/// Export the retained sparse candidate without cloning expressions or proofs.
/// Unsupported portable envelopes refuse instead of dropping proof information.
pub fn encode_residual_candidate<M: BudgetMeter + ?Sized>(
    program: &VerifiedResidualSystem,
    meter: &mut M,
) -> Result<Vec<u8>, ProgramArtifactError> {
    retained_jacobian(program.sparse_jacobian(), meter)?;
    let bytes = encode(program.sparse_jacobian(), "residual", meter)?;
    preflight(&bytes, "residual", meter)?;
    Ok(bytes)
}

/// Decode a bounded proposal. This does not check mathematical derivatives,
/// source binding or bytecode provenance and returns no acceptance receipt.
pub fn decode_residual_candidate<M: BudgetMeter + ?Sized>(
    bytes: &[u8],
    meter: &mut M,
) -> Result<SparseJacobian, ProgramArtifactError> {
    let (raw, inspection) = preflight(bytes, "residual", meter)?;
    materialization(raw, &inspection, meter)?;
    let candidate: SparseJacobian = serde_json::from_str(raw.get()).map_err(encoding)?;
    meter.checkpoint()?;
    canonical(&candidate, "residual", bytes, meter)?;
    Ok(candidate)
}

/// Import a portable residual candidate or a previously saved Python bundle.
///
/// Portable input keeps the exact canonical encoding and 1 MiB limit of
/// [`decode_residual_candidate`]. Legacy `{schema_version,jacobian}` input
/// retains its original 16 MiB limit and accepts its historical JSON spelling.
/// Both paths preflight bounded shapes, typed payloads and all retained proof
/// and bytecode data before allocating the owned candidate. Neither path
/// verifies derivatives, regenerates evidence or trusts an acceptance flag;
/// the caller must subsequently use [`admit_residual_candidate`] with its own
/// expected source and variable order. Every phase charges the same meter.
pub fn decode_residual_candidate_with_legacy<M: BudgetMeter + ?Sized>(
    bytes: &[u8],
    meter: &mut M,
) -> Result<SparseJacobian, ProgramArtifactError> {
    meter.checkpoint()?;
    if bytes.len() > MAX_VERIFIED_ARTIFACT_BYTES {
        return Err(ProgramArtifactError::ByteLimit {
            actual: bytes.len(),
            max: MAX_VERIFIED_ARTIFACT_BYTES,
        });
    }
    let mut nesting = JsonNesting::default();
    for chunk in bytes.chunks(512) {
        charge(meter, chunk.len(), 0, 0)?;
        nesting.scan(chunk)?;
    }
    parse_charge(bytes.len(), meter)?;
    let envelope: CompatibleResidualEnvelope<'_> =
        serde_json::from_slice(bytes).map_err(encoding)?;
    let (raw, portable) = match (envelope.kind, envelope.candidate, envelope.jacobian) {
        (Some(kind), Some(candidate), None) => {
            if bytes.len() > MAX_PROGRAM_CANDIDATE_BYTES {
                return Err(ProgramArtifactError::ByteLimit {
                    actual: bytes.len(),
                    max: MAX_PROGRAM_CANDIDATE_BYTES,
                });
            }
            if envelope.schema_version != PROGRAM_CANDIDATE_SCHEMA_VERSION {
                return Err(ProgramArtifactError::UnsupportedVersion {
                    found: envelope.schema_version,
                    supported: PROGRAM_CANDIDATE_SCHEMA_VERSION,
                });
            }
            if read::<&str, _>(kind, meter)? != "residual" {
                return Err(ProgramArtifactError::WrongKind {
                    expected: "residual",
                });
            }
            (candidate, true)
        }
        (None, None, Some(jacobian)) => {
            if envelope.schema_version != VERIFIED_ARTIFACT_SCHEMA_VERSION {
                return Err(ProgramArtifactError::UnsupportedVersion {
                    found: envelope.schema_version,
                    supported: VERIFIED_ARTIFACT_SCHEMA_VERSION,
                });
            }
            (jacobian, false)
        }
        _ => return Err(invalid("expected one complete residual artifact envelope")),
    };
    let mut inspection = Preflight::default();
    inspection.jacobian(raw, meter)?;
    materialization(raw, &inspection, meter)?;
    let candidate: SparseJacobian = serde_json::from_str(raw.get()).map_err(encoding)?;
    meter.checkpoint()?;
    if portable {
        canonical(&candidate, "residual", bytes, meter)?;
    }
    meter.checkpoint()?;
    Ok(candidate)
}

/// Export the complete objective, gradient and Hessian candidate by reference.
pub fn encode_objective_candidate<M: BudgetMeter + ?Sized>(
    program: &VerifiedObjective,
    meter: &mut M,
) -> Result<Vec<u8>, ProgramArtifactError> {
    let candidate = program.candidate();
    finite_program(&candidate.compiled_objective, meter)?;
    retained_jacobian(&candidate.gradient, meter)?;
    retained_jacobian(&candidate.hessian, meter)?;
    let bytes = encode(candidate, "objective", meter)?;
    preflight(&bytes, "objective", meter)?;
    Ok(bytes)
}

/// Decode a bounded objective proposal; independent source-bound admission is
/// still required before any caller treats its derivatives as verified.
pub fn decode_objective_candidate<M: BudgetMeter + ?Sized>(
    bytes: &[u8],
    meter: &mut M,
) -> Result<ObjectiveCandidate, ProgramArtifactError> {
    let (raw, inspection) = preflight(bytes, "objective", meter)?;
    materialization(raw, &inspection, meter)?;
    let owned: OwnedObjective = serde_json::from_str(raw.get()).map_err(encoding)?;
    meter.checkpoint()?;
    let candidate = ObjectiveCandidate {
        compiled_objective: owned.compiled_objective,
        gradient: owned.gradient,
        hessian: owned.hessian,
    };
    canonical(&candidate, "objective", bytes, meter)?;
    Ok(candidate)
}

/// Compare already preflighted expressions without simplification, printing or
/// numeric sampling. In particular, held operand order/multiplicity and scalar
/// variants remain part of the source binding. Each visited node and payload is
/// charged before comparison; the preflight bounds recursion independently of
/// whether the candidate came through the portable decoder.
fn source_expressions_match<M: BudgetMeter + ?Sized>(
    expected: &Expr,
    candidate: &Expr,
    meter: &mut M,
) -> Result<bool, ProgramArtifactError> {
    charge(meter, 1, 0, 0)?;
    Ok(match (expected, candidate) {
        (Expr::Sym(expected), Expr::Sym(candidate)) => {
            charge(meter, expected.name.len() + candidate.name.len() + 65, 0, 0)?;
            expected == candidate
        }
        (Expr::Integer(expected), Expr::Integer(candidate)) => {
            let limbs = expected.bits().div_ceil(64) + candidate.bits().div_ceil(64);
            charge(meter, limbs as usize + 1, 0, 0)?;
            expected == candidate
        }
        (Expr::Rational(expected), Expr::Rational(candidate)) => {
            let limbs = expected.numer().bits().div_ceil(64)
                + expected.denom().bits().div_ceil(64)
                + candidate.numer().bits().div_ceil(64)
                + candidate.denom().bits().div_ceil(64);
            charge(meter, limbs as usize + 1, 0, 0)?;
            expected == candidate
        }
        (Expr::Const(expected), Expr::Const(candidate)) => expected == candidate,
        (Expr::Pow(expected_base, expected_exp), Expr::Pow(candidate_base, candidate_exp)) => {
            source_expressions_match(expected_base, candidate_base, meter)?
                && source_expressions_match(expected_exp, candidate_exp, meter)?
        }
        (Expr::Add(expected), Expr::Add(candidate))
        | (Expr::Mul(expected), Expr::Mul(candidate)) => {
            source_lists_match(expected, candidate, meter)?
        }
        (Expr::Function(expected_name, expected), Expr::Function(candidate_name, candidate)) => {
            charge(meter, expected_name.len() + candidate_name.len() + 1, 0, 0)?;
            expected_name == candidate_name && source_lists_match(expected, candidate, meter)?
        }
        _ => false,
    })
}

fn source_lists_match<M: BudgetMeter + ?Sized>(
    expected: &[Expr],
    candidate: &[Expr],
    meter: &mut M,
) -> Result<bool, ProgramArtifactError> {
    charge(meter, 1, 0, 0)?;
    if expected.len() != candidate.len() {
        return Ok(false);
    }
    for (expected, candidate) in expected.iter().zip(candidate) {
        if !source_expressions_match(expected, candidate, meter)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn bind_sources<M: BudgetMeter + ?Sized>(
    expected_sources: &[Expr],
    expected_variables: &[Symbol],
    candidate_sources: &[Expr],
    candidate_variables: &[Symbol],
    meter: &mut M,
) -> Result<(), ProgramAdmissionError> {
    charge(meter, 1, 0, 0)?;
    for (length, resource) in [
        (expected_sources.len(), "expected source count"),
        (candidate_sources.len(), "candidate source count"),
        (expected_variables.len(), "expected variable count"),
        (candidate_variables.len(), "candidate variable count"),
    ] {
        limit(length, MAX_SPARSE_AXES, resource)?;
    }
    if expected_variables.len() != candidate_variables.len() {
        return Err(ProgramAdmissionError::SourceMismatch(
            "ordered typed variables",
        ));
    }
    for (expected, candidate) in expected_variables.iter().zip(candidate_variables) {
        charge(meter, 1, 0, 0)?;
        limit(
            expected.name.len(),
            MAX_PROGRAM_CANDIDATE_NAME_BYTES,
            "name bytes",
        )?;
        limit(
            candidate.name.len(),
            MAX_PROGRAM_CANDIDATE_NAME_BYTES,
            "name bytes",
        )?;
        charge(meter, expected.name.len() + candidate.name.len() + 65, 0, 0)?;
        if expected != candidate {
            return Err(ProgramAdmissionError::SourceMismatch(
                "ordered typed variables",
            ));
        }
    }
    if expected_sources.len() != candidate_sources.len() {
        return Err(ProgramAdmissionError::SourceMismatch(
            "ordered residual sources",
        ));
    }
    // Source expectations are a separate trust boundary: the transport
    // decoder inspected only the candidate, never these borrowed caller inputs.
    for sources in [expected_sources, candidate_sources] {
        let mut total_nodes = 0;
        for source in sources {
            let mut nodes = 0;
            retained_expression(source, 0, &mut nodes, false, meter)?;
            total_nodes += nodes;
            limit(
                total_nodes,
                MAX_VERIFIED_EXPRESSION_NODES,
                "source expression nodes",
            )?;
        }
    }
    if !source_lists_match(expected_sources, candidate_sources, meter)? {
        return Err(ProgramAdmissionError::SourceMismatch(
            "ordered residual sources",
        ));
    }
    charge(meter, 1, 0, 0)?;
    Ok(())
}

/// Bind an untrusted derivative proposal to the caller's exact original
/// residuals and ordered typed axes, then independently verify its sparse
/// coverage, supplied derivative proofs and supplied derivative bytecode.
///
/// Decode with an ordinary transport meter, then call this function with a
/// protected verifier meter when the caller separates those budgets. This
/// function neither creates a budget nor generates replacement derivatives.
/// Original residual bytecode and the derived shared evaluation plan are
/// rebuilt only after the candidate's independent checks succeed. The caller
/// owns the surrounding assumptions/context universe; no context or accepted
/// status is imported here. Results are formal derivatives on their common
/// smooth domain, with approximate finite binary64 evaluation.
pub fn admit_residual_candidate<M: BudgetMeter + ?Sized>(
    residuals: &[Expr],
    variables: &[Symbol],
    candidate: SparseJacobian,
    meter: &mut M,
) -> Result<VerifiedResidualSystem, ProgramAdmissionError> {
    bind_sources(
        residuals,
        variables,
        &candidate.residuals,
        &candidate.vars,
        meter,
    )?;
    let verified =
        VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(candidate, meter)?;
    charge(meter, 1, 0, 0)?;
    Ok(verified)
}

/// Bind a portable objective proposal to the caller's raw source and ordered
/// typed axes, then independently replay both derivative stages and all
/// bytecode provenance. Supplied objective, gradient and Hessian programs are
/// checked rather than regenerated; derived plans are rebuilt locally.
///
/// The meter, context ownership and evidence boundaries are the same as for
/// [`admit_residual_candidate`]. Exhaustion or cancellation returns no partially
/// accepted objective and does not reset any caller accounting.
pub fn admit_objective_candidate<M: BudgetMeter + ?Sized>(
    source: &Expr,
    variables: &[Symbol],
    candidate: ObjectiveCandidate,
    meter: &mut M,
) -> Result<VerifiedObjective, ProgramAdmissionError> {
    bind_sources(
        std::slice::from_ref(source),
        variables,
        &candidate.gradient.residuals,
        &candidate.gradient.vars,
        meter,
    )?;
    let verified =
        VerifiedObjective::try_from_candidate_with_meter(source, variables, candidate, meter)?;
    charge(meter, 1, 0, 0)?;
    Ok(verified)
}
