//! Immutable bridge from checked symbolic Jacobians to numerical consumers.
//!
//! The artifact retains its complete sparse proof bundle and the exact bytecode
//! checked at admission. Versioned JSON transport replays that admission before
//! returning a verified value; serialized receipts are never trusted.
//! Binary64 evaluation remains approximate and does not establish a root.

#![forbid(unsafe_code)]

use crate::compile::{
    CompileError, CompiledExpr, CompiledOp, CompiledResidualSystem, EvalError, MAX_COMPILE_OPS,
    SharedEvaluationPlan, charge_compilation_work, charge_evaluation_allocation,
    charge_evaluation_steps,
};
use crate::sparse_jacobian::{
    JacobianVerificationError, MAX_SPARSE_AXES, MAX_SPARSE_SOURCE_WORK, SparseJacobian,
    SparseJacobianReceipt, compiled_expression_matches_with_meter, plan_sparse_sources_with_meter,
    verify_sparse_jacobian_certificate_with_meter,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, ProofRule, derivation_verification_units};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::HashMap;
use std::io::{self, Write};
use thiserror::Error;

/// Version of the proof-bearing residual-system JSON format.
pub const VERIFIED_ARTIFACT_SCHEMA_VERSION: u32 = 1;
/// Maximum encoded input or output, including whitespace and metadata.
pub const MAX_VERIFIED_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Transport depth stays below the JSON decoder's own recursion ceiling.
pub const MAX_VERIFIED_ARTIFACT_JSON_DEPTH: usize = 96;
/// Maximum number of positions in an explicitly requested dense Jacobian.
/// Sparse compilation, transport and evaluation do not materialize these zeros.
pub const MAX_VERIFIED_JACOBIAN_ENTRIES: usize = 65_536;
/// Maximum potential derivative coordinates and retained nonzero proofs.
/// Conservative source dependency counts are bounded before proof generation.
pub const MAX_VERIFIED_NONZERO_ENTRIES: usize = 65_536;
/// Aggregate original-source and derivative expression nodes at admission.
pub const MAX_VERIFIED_EXPRESSION_NODES: usize = 1_048_576;
/// Aggregate bytecode operations in original residuals and stored derivatives.
pub const MAX_VERIFIED_STORED_OPS: usize = 1_048_576;
/// Aggregate independent kernel preflight units across all derivative proofs.
pub const MAX_VERIFIED_PROOF_UNITS: u64 = 1_048_576;
/// Proof construction and replay use symbolic recursion. Refuse deeper raw
/// expression trees before entering those paths on an ordinary worker stack.
pub const MAX_VERIFIED_SOURCE_DEPTH: usize = 32;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum VerifiedCompilationError {
    #[error("Jacobian verification failed: {0}")]
    Jacobian(#[from] JacobianVerificationError),
    #[error("Residual compilation failed: {0}")]
    Compilation(#[from] CompileError),
    #[error("Compiled residual {residual} does not match its symbolic source")]
    ResidualProvenanceMismatch { residual: usize },
    #[error("Dense consumer Jacobian requires {entries} entries, exceeding maximum {max}")]
    DenseJacobianLimit { entries: usize, max: usize },
    #[error("Consumer system {axis} length {length} exceeds maximum {max}")]
    AxisLimit {
        axis: &'static str,
        length: usize,
        max: usize,
    },
    #[error("Verified source expression depth {depth} exceeds maximum {max}")]
    SourceDepthLimit { depth: usize, max: usize },
    #[error("Verified {resource} requires {actual}, exceeding maximum {max}")]
    AdmissionLimit {
        resource: &'static str,
        actual: u64,
        max: u64,
    },
    #[error("Proof payload exceeds its admission envelope: {0}")]
    ProofEnvelope(String),
}

/// Finite binary64 values from one admitted residual/Jacobian program.
/// Derivative values follow [`VerifiedResidualSystem::jacobian_pattern`].
/// These approximate values do not certify numerical rounding or a root.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseResidualEvaluation {
    pub residuals: Vec<f64>,
    pub jacobian: Vec<f64>,
}

/// Finite original residuals and an explicitly requested row-major Jacobian.
/// Dense output is bounded separately from the retained sparse program.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualEvaluation {
    pub residuals: Vec<f64>,
    pub jacobian: Vec<f64>,
}

/// Evaluation failures preserve the underlying budget/cancellation error chain.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResidualEvaluationError {
    #[error("residual evaluation failed: {0}")]
    Evaluation(#[from] EvalError),
    #[error("residual {lane} length must be {expected}, got {actual}")]
    Dimension {
        lane: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Dense consumer Jacobian requires {entries} entries, exceeding maximum {max}")]
    DenseJacobianLimit { entries: usize, max: usize },
}

/// A transport failure never publishes a partially decoded verified system.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum VerifiedArtifactError {
    #[error("Encoded artifact size {actual} exceeds maximum {max}")]
    ByteLimit { actual: usize, max: usize },
    #[error("Artifact JSON nesting depth {depth} exceeds maximum {max}")]
    DepthLimit { depth: usize, max: usize },
    #[error("Unsupported artifact schema version {found}; supported version is {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("Invalid artifact JSON: {0}")]
    Encoding(String),
    #[error("Artifact transport allocation refused")]
    AllocationFailure,
    #[error("Non-finite derivative bytecode cannot be represented losslessly as JSON")]
    NonFiniteBytecode,
    #[error("Artifact admission failed: {0}")]
    Admission(#[from] VerifiedCompilationError),
}

#[derive(Serialize)]
struct ArtifactWireRef<'a> {
    schema_version: u32,
    jacobian: &'a SparseJacobian,
}

// Parse the envelope without constructing expression/proof objects. This lets
// version rejection precede decoding a payload from an incompatible schema.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactWire<'a> {
    schema_version: u32,
    #[serde(borrow)]
    jacobian: &'a RawValue,
}

struct JsonNesting {
    depth: usize,
    in_string: bool,
    escaped: bool,
    maximum: usize,
}

impl JsonNesting {
    fn new(maximum: usize) -> Self {
        Self {
            depth: 0,
            in_string: false,
            escaped: false,
            maximum,
        }
    }

    fn scan(&mut self, bytes: &[u8]) -> Result<(), VerifiedArtifactError> {
        for &byte in bytes {
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.in_string = false;
                }
            } else {
                match byte {
                    b'"' => self.in_string = true,
                    b'[' | b'{' => {
                        self.depth += 1;
                        if self.depth > self.maximum {
                            return Err(VerifiedArtifactError::DepthLimit {
                                depth: self.depth,
                                max: self.maximum,
                            });
                        }
                    }
                    b']' | b'}' => self.depth = self.depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
        // Syntax, balanced delimiters, and UTF-8 are checked by serde_json.
        Ok(())
    }
}

struct BoundedArtifactWriter {
    bytes: Vec<u8>,
    maximum: usize,
    nesting: JsonNesting,
    failure: Option<VerifiedArtifactError>,
}

impl BoundedArtifactWriter {
    fn new(maximum: usize, depth: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
            nesting: JsonNesting::new(depth),
            failure: None,
        }
    }

    fn refuse(&mut self, error: VerifiedArtifactError) -> io::Error {
        self.failure = Some(error);
        io::Error::other("verified artifact transport refused")
    }
}

impl Write for BoundedArtifactWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::Error::other(
                "verified artifact transport already refused",
            ));
        }
        let required = self.bytes.len().saturating_add(bytes.len());
        if required > self.maximum {
            return Err(self.refuse(VerifiedArtifactError::ByteLimit {
                actual: required,
                max: self.maximum,
            }));
        }
        if let Err(error) = self.nesting.scan(bytes) {
            return Err(self.refuse(error));
        }
        if required > self.bytes.capacity() {
            let capacity = required
                .max(self.bytes.capacity().saturating_mul(2))
                .min(self.maximum);
            if self
                .bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .is_err()
            {
                return Err(self.refuse(VerifiedArtifactError::AllocationFailure));
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A compiled system bound to independently checked derivative proofs and
/// symbolic sources. Private fields and the absence of an unchecked serde
/// decoder ensure that construction always includes proof and provenance checks.
#[derive(Debug, Clone)]
pub struct VerifiedResidualSystem {
    residual_programs: Vec<CompiledExpr>,
    stored_operation_count: usize,
    sparse_jacobian: SparseJacobian,
    receipt: SparseJacobianReceipt,
    shared_evaluation: SharedEvaluationPlan,
}

impl VerifiedResidualSystem {
    /// Builds derivatives and carries those same checked programs to consumers.
    pub fn try_compile(
        residuals: &[Expr],
        vars: &[Symbol],
    ) -> Result<Self, VerifiedCompilationError> {
        Self::try_compile_with_meter(residuals, vars, &mut Unbounded)
    }

    /// Compile and independently admit a complete program on a cumulative
    /// ledger. Callers with a reserved verifier pool can instead generate with
    /// `SparseJacobian::try_build_with_meter`, then pass its owned candidate to
    /// [`Self::try_from_owned_sparse_jacobian_with_meter`] using the verifier meter.
    pub fn try_compile_with_meter<M: BudgetMeter + ?Sized>(
        residuals: &[Expr],
        vars: &[Symbol],
        meter: &mut M,
    ) -> Result<Self, VerifiedCompilationError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        Self::check_axes(residuals.len(), vars.len())?;
        let mut expression_nodes = 0;
        for residual in residuals {
            let nodes = Self::check_expression_shape(residual, meter)?;
            Self::add_admission_count(
                &mut expression_nodes,
                nodes,
                MAX_SPARSE_SOURCE_WORK,
                "source nodes",
            )?;
        }
        // Bound row-local differentiation work before constructing any proofs.
        Self::check_source_plan(residuals, vars, meter)?;
        Self::try_from_owned_sparse_jacobian_with_meter(
            SparseJacobian::try_build_with_meter(residuals, vars, meter)?,
            meter,
        )
    }

    /// Admits a possibly untrusted sparse artifact and retains its full proof.
    /// No numerical sample is required for symbolic or bytecode acceptance.
    pub fn try_from_sparse_jacobian(
        jacobian: &SparseJacobian,
    ) -> Result<Self, VerifiedCompilationError> {
        let (residual_programs, stored_operation_count, receipt) =
            Self::check_and_compile(jacobian, &mut Unbounded)?;
        let shared_evaluation = Self::shared_plan(&residual_programs, jacobian, &mut Unbounded)?;
        Ok(Self {
            residual_programs,
            stored_operation_count,
            sparse_jacobian: jacobian.clone(),
            receipt,
            shared_evaluation,
        })
    }

    fn try_from_owned_sparse_jacobian(
        jacobian: SparseJacobian,
    ) -> Result<Self, VerifiedCompilationError> {
        Self::try_from_owned_sparse_jacobian_with_meter(jacobian, &mut Unbounded)
    }

    /// Consume and independently admit an untrusted sparse candidate. All
    /// structural, derivative and bytecode checks run under the supplied meter;
    /// a cancelled or exhausted check cannot return a partially verified value.
    /// Taking ownership avoids cloning the retained expression/proof forest.
    pub fn try_from_owned_sparse_jacobian_with_meter<M: BudgetMeter + ?Sized>(
        jacobian: SparseJacobian,
        meter: &mut M,
    ) -> Result<Self, VerifiedCompilationError> {
        let (residual_programs, stored_operation_count, receipt) =
            Self::check_and_compile(&jacobian, meter)?;
        let shared_evaluation = Self::shared_plan(&residual_programs, &jacobian, meter)?;
        charge_compilation_work(meter, 1, 0, 0)?;
        Ok(Self {
            residual_programs,
            stored_operation_count,
            sparse_jacobian: jacobian,
            receipt,
            shared_evaluation,
        })
    }

    fn shared_plan<M: BudgetMeter + ?Sized>(
        residuals: &[CompiledExpr],
        jacobian: &SparseJacobian,
        meter: &mut M,
    ) -> Result<SharedEvaluationPlan, CompileError> {
        // The programs are already independently source/proof checked and
        // remain private and immutable for the entire plan lifetime. Rebuild
        // this derived plan on artifact import; never trust serialized caches.
        SharedEvaluationPlan::build(
            (0..residuals.len() + jacobian.entries.len()).map(|index| {
                if index < residuals.len() {
                    &residuals[index]
                } else {
                    &jacobian.entries[index - residuals.len()].compiled_deriv
                }
            }),
            meter,
        )
    }

    fn check_axes(rows: usize, columns: usize) -> Result<(), VerifiedCompilationError> {
        for (axis, length) in [("rows", rows), ("columns", columns)] {
            if length > MAX_SPARSE_AXES {
                return Err(VerifiedCompilationError::AxisLimit {
                    axis,
                    length,
                    max: MAX_SPARSE_AXES,
                });
            }
        }
        Ok(())
    }

    fn check_dense_size(rows: usize, columns: usize) -> Result<usize, VerifiedCompilationError> {
        let entries = rows
            .checked_mul(columns)
            .ok_or(CompileError::DimensionOverflow)?;
        if entries > MAX_VERIFIED_JACOBIAN_ENTRIES {
            return Err(VerifiedCompilationError::DenseJacobianLimit {
                entries,
                max: MAX_VERIFIED_JACOBIAN_ENTRIES,
            });
        }
        Ok(entries)
    }

    fn add_admission_count(
        total: &mut usize,
        added: usize,
        maximum: usize,
        resource: &'static str,
    ) -> Result<(), VerifiedCompilationError> {
        let next = total
            .checked_add(added)
            .ok_or(VerifiedCompilationError::AdmissionLimit {
                resource,
                actual: u64::MAX,
                max: maximum as u64,
            })?;
        if next > maximum {
            return Err(VerifiedCompilationError::AdmissionLimit {
                resource,
                actual: next as u64,
                max: maximum as u64,
            });
        }
        *total = next;
        Ok(())
    }

    fn check_source_plan<M: BudgetMeter + ?Sized>(
        residuals: &[Expr],
        vars: &[Symbol],
        meter: &mut M,
    ) -> Result<(), VerifiedCompilationError> {
        let plan = plan_sparse_sources_with_meter(residuals, vars, meter)?;
        let mut coordinates = 0;
        for dependencies in plan.row_dependencies {
            charge_compilation_work(meter, 1, 0, 0)?;
            Self::add_admission_count(
                &mut coordinates,
                dependencies.len(),
                MAX_VERIFIED_NONZERO_ENTRIES,
                "potential derivative coordinates",
            )?;
        }
        Ok(())
    }

    fn check_expression_shape<M: BudgetMeter + ?Sized>(
        expr: &Expr,
        meter: &mut M,
    ) -> Result<usize, VerifiedCompilationError> {
        Self::check_expression_limits(expr, MAX_VERIFIED_SOURCE_DEPTH, MAX_COMPILE_OPS, meter)
    }

    fn check_proof_expression<M: BudgetMeter + ?Sized>(
        expr: &Expr,
        meter: &mut M,
    ) -> Result<(), VerifiedCompilationError> {
        // The independently checked differentiation claim adds a diff(source,
        // variable) wrapper: one level and two nodes beyond the source.
        Self::check_expression_limits(
            expr,
            MAX_VERIFIED_SOURCE_DEPTH + 1,
            MAX_COMPILE_OPS + 2,
            meter,
        )
        .map(|_| ())
    }

    fn check_expression_limits<M: BudgetMeter + ?Sized>(
        expr: &Expr,
        maximum_depth: usize,
        maximum_nodes: usize,
        meter: &mut M,
    ) -> Result<usize, VerifiedCompilationError> {
        charge_compilation_work(meter, 1, std::mem::size_of::<(&Expr, usize)>(), 1)?;
        let mut pending = Vec::new();
        pending
            .try_reserve(1)
            .map_err(|_| CompileError::AllocationFailure)?;
        pending.push((expr, 0usize));
        let mut visited = 0usize;
        while let Some((current, depth)) = pending.pop() {
            charge_compilation_work(meter, 1, 0, 0)?;
            if depth > maximum_depth {
                return Err(VerifiedCompilationError::SourceDepthLimit {
                    depth,
                    max: maximum_depth,
                });
            }
            visited += 1;
            let children = match current {
                Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => args.len(),
                Expr::Pow(_, _) => 2,
                Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) | Expr::Sym(_) => 0,
            };
            if children
                > maximum_nodes
                    .saturating_sub(visited)
                    .saturating_sub(pending.len())
            {
                return Err(CompileError::OpLimitExceeded {
                    ops: maximum_nodes + 1,
                    max: maximum_nodes,
                }
                .into());
            }
            charge_compilation_work(
                meter,
                children,
                children * std::mem::size_of::<(&Expr, usize)>(),
                usize::from(children != 0),
            )?;
            pending
                .try_reserve(children)
                .map_err(|_| CompileError::AllocationFailure)?;
            match current {
                Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                    pending.extend(args.iter().map(|arg| (arg, depth + 1)));
                }
                Expr::Pow(base, exponent) => {
                    pending.push((base.as_ref(), depth + 1));
                    pending.push((exponent.as_ref(), depth + 1));
                }
                _ => {}
            }
        }
        Ok(visited)
    }

    fn check_proof_claim<M: BudgetMeter + ?Sized>(
        claim: &Claim,
        meter: &mut M,
    ) -> Result<(), VerifiedCompilationError> {
        match claim {
            Claim::Equality { lhs, rhs } | Claim::AlgebraicIdentity { lhs, rhs } => {
                Self::check_proof_expression(lhs, meter)?;
                Self::check_proof_expression(rhs, meter)
            }
            Claim::PredicateHold { expr, .. }
            | Claim::DomainMembership { expr, .. }
            | Claim::NonZero(expr) => Self::check_proof_expression(expr, meter),
            Claim::IrreducibleModPrime { .. } => Err(VerifiedCompilationError::ProofEnvelope(
                "modular irreducibility does not certify a residual-system expression".into(),
            )),
        }
    }

    fn check_proof_rule<M: BudgetMeter + ?Sized>(
        rule: &ProofRule,
        meter: &mut M,
    ) -> Result<(), VerifiedCompilationError> {
        match rule {
            ProofRule::Reflexivity(expr)
            | ProofRule::ContextPredicate { expr, .. }
            | ProofRule::ContextDomain { expr, .. } => Self::check_proof_expression(expr, meter),
            ProofRule::Substitution { template, .. } => {
                Self::check_proof_expression(template, meter)
            }
            ProofRule::DefinitionalReduction { lhs, rhs, .. } => {
                Self::check_proof_expression(lhs, meter)?;
                Self::check_proof_expression(rhs, meter)
            }
            ProofRule::CertificateLemma { claim, .. } => Self::check_proof_claim(claim, meter),
            ProofRule::Symmetry(_)
            | ProofRule::Transitivity(_, _)
            | ProofRule::CongruenceAdd(_)
            | ProofRule::CongruenceMul(_)
            | ProofRule::CongruencePow { .. }
            | ProofRule::CongruenceFunction { .. } => Ok(()),
        }
    }

    fn check_and_compile<M: BudgetMeter + ?Sized>(
        jacobian: &SparseJacobian,
        meter: &mut M,
    ) -> Result<(Vec<CompiledExpr>, usize, SparseJacobianReceipt), VerifiedCompilationError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        let num_residuals = jacobian.residuals.len();
        let num_vars = jacobian.vars.len();
        Self::check_axes(num_residuals, num_vars)?;
        let mut nonzero_entries = 0;
        Self::add_admission_count(
            &mut nonzero_entries,
            jacobian.entries.len().max(jacobian.pattern.nonzeros.len()),
            MAX_VERIFIED_NONZERO_ENTRIES,
            "nonzero entries",
        )?;
        // Reject duplicate/extra coordinates and inconsistent counts before
        // traversing any untrusted symbolic or proof payload.
        jacobian.validate_structure_with_meter(meter)?;
        let mut expression_nodes = 0;
        for expression in jacobian
            .residuals
            .iter()
            .chain(jacobian.entries.iter().map(|entry| &entry.symbolic_deriv))
        {
            let nodes = Self::check_expression_shape(expression, meter)?;
            Self::add_admission_count(
                &mut expression_nodes,
                nodes,
                MAX_VERIFIED_EXPRESSION_NODES,
                "expression nodes",
            )?;
        }
        // This is an admission/work plan only. The independent checker below
        // reconstructs source dependencies and replays every retained proof.
        Self::check_source_plan(&jacobian.residuals, &jacobian.vars, meter)?;
        let mut proof_units = 0u64;
        let mut stored_operation_count = 0;
        for entry in &jacobian.entries {
            charge_compilation_work(meter, entry.derivation.steps.len() + 1, 0, 0)?;
            Self::add_admission_count(
                &mut stored_operation_count,
                entry.compiled_deriv.ops().len(),
                MAX_VERIFIED_STORED_OPS,
                "stored bytecode operations",
            )?;
            // Reuse the kernel's iterative limits for all step references,
            // claims, domains, text, and numeric payloads, including unused steps.
            let units = derivation_verification_units(&entry.derivation)
                .map_err(|error| VerifiedCompilationError::ProofEnvelope(error.to_string()))?;
            proof_units =
                proof_units
                    .checked_add(units)
                    .ok_or(VerifiedCompilationError::AdmissionLimit {
                        resource: "proof verification units",
                        actual: u64::MAX,
                        max: MAX_VERIFIED_PROOF_UNITS,
                    })?;
            if proof_units > MAX_VERIFIED_PROOF_UNITS {
                return Err(VerifiedCompilationError::AdmissionLimit {
                    resource: "proof verification units",
                    actual: proof_units,
                    max: MAX_VERIFIED_PROOF_UNITS,
                });
            }
            for step in &entry.derivation.steps {
                Self::check_proof_claim(&step.claim, meter)?;
                Self::check_proof_rule(&step.rule, meter)?;
            }
        }
        let receipt = verify_sparse_jacobian_certificate_with_meter(jacobian, &[], meter)?;
        let mut var_map = HashMap::new();
        charge_compilation_work(
            meter,
            num_vars,
            num_vars * std::mem::size_of::<(Symbol, usize)>(),
            usize::from(num_vars != 0),
        )?;
        var_map
            .try_reserve(num_vars)
            .map_err(|_| CompileError::AllocationFailure)?;
        for (index, variable) in jacobian.vars.iter().enumerate() {
            charge_compilation_work(meter, variable.name.len() + 1, variable.name.len(), 1)?;
            var_map.insert(variable.clone(), index);
        }

        let mut compiled_residuals = Vec::new();
        charge_compilation_work(
            meter,
            num_residuals,
            num_residuals * std::mem::size_of::<CompiledExpr>(),
            usize::from(num_residuals != 0),
        )?;
        compiled_residuals
            .try_reserve_exact(num_residuals)
            .map_err(|_| CompileError::AllocationFailure)?;
        for (residual, source) in jacobian.residuals.iter().enumerate() {
            let compiled = CompiledExpr::try_compile_with_meter(source, &var_map, meter)?;
            Self::add_admission_count(
                &mut stored_operation_count,
                compiled.ops().len(),
                MAX_VERIFIED_STORED_OPS,
                "stored bytecode operations",
            )?;
            if !compiled_expression_matches_with_meter(source, &var_map, &compiled, meter)? {
                return Err(VerifiedCompilationError::ResidualProvenanceMismatch { residual });
            }
            compiled_residuals.push(compiled);
        }
        charge_compilation_work(meter, 1, 0, 0)?;
        Ok((compiled_residuals, stored_operation_count, receipt))
    }

    /// Lossless, versioned JSON containing the original sparse proof bundle.
    /// The writer enforces size and nesting limits while encoding; a failed
    /// encoding does not return a partial artifact. Receipts are not serialized.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, VerifiedArtifactError> {
        // JSON's default non-finite float encoding is null. Refuse it explicitly
        // instead of silently emitting bytecode that cannot round-trip.
        if self.sparse_jacobian.entries.iter().any(|entry| {
            entry.compiled_deriv.ops().iter().any(|operation| {
                matches!(
                    operation,
                    CompiledOp::LoadConst(value) | CompiledOp::Pow(value)
                        if !value.is_finite()
                )
            })
        }) {
            return Err(VerifiedArtifactError::NonFiniteBytecode);
        }
        let wire = ArtifactWireRef {
            schema_version: VERIFIED_ARTIFACT_SCHEMA_VERSION,
            jacobian: &self.sparse_jacobian,
        };
        let mut writer = BoundedArtifactWriter::new(
            MAX_VERIFIED_ARTIFACT_BYTES,
            MAX_VERIFIED_ARTIFACT_JSON_DEPTH,
        );
        if let Err(error) = serde_json::to_writer(&mut writer, &wire) {
            return Err(writer
                .failure
                .unwrap_or_else(|| VerifiedArtifactError::Encoding(error.to_string())));
        }
        Ok(writer.bytes)
    }

    /// Decodes bounded JSON and independently rechecks its proof, sparse shape,
    /// omitted derivatives, variable order and bytecode. The verification
    /// receipt is freshly computed; no receipt or verified flag is accepted.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, VerifiedArtifactError> {
        if bytes.len() > MAX_VERIFIED_ARTIFACT_BYTES {
            return Err(VerifiedArtifactError::ByteLimit {
                actual: bytes.len(),
                max: MAX_VERIFIED_ARTIFACT_BYTES,
            });
        }
        JsonNesting::new(MAX_VERIFIED_ARTIFACT_JSON_DEPTH).scan(bytes)?;
        let wire: ArtifactWire<'_> = serde_json::from_slice(bytes)
            .map_err(|error| VerifiedArtifactError::Encoding(error.to_string()))?;
        if wire.schema_version != VERIFIED_ARTIFACT_SCHEMA_VERSION {
            return Err(VerifiedArtifactError::UnsupportedVersion {
                found: wire.schema_version,
                supported: VERIFIED_ARTIFACT_SCHEMA_VERSION,
            });
        }
        let jacobian: SparseJacobian = serde_json::from_str(wire.jacobian.get())
            .map_err(|error| VerifiedArtifactError::Encoding(error.to_string()))?;
        Self::try_from_owned_sparse_jacobian(jacobian).map_err(VerifiedArtifactError::Admission)
    }

    /// Original residual programs, evaluated even when all derivatives vanish.
    pub fn residual_programs(&self) -> &[CompiledExpr] {
        &self.residual_programs
    }

    pub fn num_residuals(&self) -> usize {
        self.sparse_jacobian.residuals.len()
    }

    pub fn num_vars(&self) -> usize {
        self.sparse_jacobian.vars.len()
    }

    /// Actual stored residual/derivative bytecode; implicit zeros have no program.
    pub fn stored_operation_count(&self) -> usize {
        self.stored_operation_count
    }

    /// Reusable pure bytecode subexpressions across original sources and
    /// sparse derivatives. This is a structural count, not a timing claim.
    pub fn shared_subexpression_count(&self) -> usize {
        self.shared_evaluation.slot_count()
    }

    /// Preflight explicit dense output before allocating or inspecting a point.
    pub fn dense_jacobian_len(&self) -> Result<usize, VerifiedCompilationError> {
        Self::check_dense_size(self.num_residuals(), self.num_vars())
    }

    fn copy_program(program: &CompiledExpr) -> Result<CompiledExpr, CompileError> {
        let mut ops = Vec::new();
        ops.try_reserve_exact(program.ops().len())
            .map_err(|_| CompileError::AllocationFailure)?;
        ops.extend_from_slice(program.ops());
        CompiledExpr::try_from_ops(ops).map_err(CompileError::InvalidBytecode)
    }

    /// Explicitly materialize the bounded legacy dense representation.
    /// The returned independently owned program is mutable and unverified.
    pub fn try_dense_program(&self) -> Result<CompiledResidualSystem, VerifiedCompilationError> {
        let dense_len = self.dense_jacobian_len()?;
        let mut compiled_residuals = Vec::new();
        compiled_residuals
            .try_reserve_exact(self.num_residuals())
            .map_err(|_| CompileError::AllocationFailure)?;
        for program in self.residual_programs() {
            compiled_residuals.push(Self::copy_program(program)?);
        }
        let mut compiled_jacobian = Vec::new();
        compiled_jacobian
            .try_reserve_exact(dense_len)
            .map_err(|_| CompileError::AllocationFailure)?;
        let zero = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)])
            .map_err(CompileError::InvalidBytecode)?;
        let mut entries = self.sparse_jacobian.entries.iter().peekable();
        for offset in 0..dense_len {
            let program = if let Some(entry) = entries.peek()
                && entry.row * self.num_vars() + entry.col == offset
            {
                &entries.next().unwrap().compiled_deriv
            } else {
                &zero
            };
            compiled_jacobian.push(Self::copy_program(program)?);
        }
        debug_assert!(entries.next().is_none());
        let mut vars = Vec::new();
        vars.try_reserve_exact(self.num_vars())
            .map_err(|_| CompileError::AllocationFailure)?;
        vars.extend_from_slice(self.vars());
        Ok(CompiledResidualSystem {
            compiled_residuals,
            compiled_jacobian,
            num_residuals: self.num_residuals(),
            num_vars: self.num_vars(),
            vars,
        })
    }

    /// Complete retained proof bundle for independent replay or checkpointing.
    /// Editing a clone cannot change the already admitted system.
    pub fn sparse_jacobian(&self) -> &SparseJacobian {
        &self.sparse_jacobian
    }

    /// Borrowed row-major coordinates of the independently checked nonzeros.
    /// Numerical zero values remain present at their structural coordinates.
    pub fn jacobian_pattern(&self) -> &[(usize, usize)] {
        &self.sparse_jacobian.pattern.nonzeros
    }

    fn input_with_meter<M: BudgetMeter + ?Sized>(
        values: &[f64],
        expected: usize,
        lane: &'static str,
        meter: &mut M,
    ) -> Result<(), ResidualEvaluationError> {
        charge_evaluation_steps(meter, 1)?;
        if values.len() != expected {
            return Err(ResidualEvaluationError::Dimension {
                lane,
                expected,
                actual: values.len(),
            });
        }
        for value in values {
            charge_evaluation_steps(meter, 1)?;
            if !value.is_finite() {
                return Err(EvalError::NonFiniteValue.into());
            }
        }
        Ok(())
    }

    fn evaluation_buffer<M: BudgetMeter + ?Sized>(
        length: usize,
        meter: &mut M,
    ) -> Result<Vec<f64>, ResidualEvaluationError> {
        charge_evaluation_allocation(meter, length)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(length)
            .map_err(|_| EvalError::AllocationFailure)?;
        Ok(result)
    }

    fn dense_evaluation_len(&self) -> Result<usize, ResidualEvaluationError> {
        let entries = self
            .num_residuals()
            .checked_mul(self.num_vars())
            .ok_or(EvalError::DimensionOverflow)?;
        if entries > MAX_VERIFIED_JACOBIAN_ENTRIES {
            return Err(ResidualEvaluationError::DenseJacobianLimit {
                entries,
                max: MAX_VERIFIED_JACOBIAN_ENTRIES,
            });
        }
        Ok(entries)
    }

    /// Evaluate finite original residuals and an explicitly bounded dense Jacobian.
    pub fn try_eval_dense(
        &self,
        point: &[f64],
    ) -> Result<ResidualEvaluation, ResidualEvaluationError> {
        self.try_eval_dense_with_meter(point, &mut Unbounded)
    }

    /// Dense size is checked before any meter activity or point inspection.
    /// Every original residual and stored derivative is evaluated with finite
    /// arithmetic; implicit zeros only initialize the bounded output buffer.
    pub fn try_eval_dense_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        meter: &mut M,
    ) -> Result<ResidualEvaluation, ResidualEvaluationError> {
        let dense_len = self.dense_evaluation_len()?;
        Self::input_with_meter(point, self.num_vars(), "point", meter)?;
        let mut memo = self.shared_evaluation.begin(meter)?;
        let mut residuals = Self::evaluation_buffer(self.num_residuals(), meter)?;
        for (index, program) in self.residual_programs().iter().enumerate() {
            charge_evaluation_steps(meter, 1)?;
            residuals.push(
                self.shared_evaluation
                    .evaluate(index, program, point, &mut memo, meter)?,
            );
        }
        let mut jacobian = Self::evaluation_buffer(dense_len, meter)?;
        charge_evaluation_steps(meter, dense_len)?;
        jacobian.resize(dense_len, 0.0);
        for (index, entry) in self.sparse_jacobian.entries.iter().enumerate() {
            let value = self.shared_evaluation.evaluate(
                self.num_residuals() + index,
                &entry.compiled_deriv,
                point,
                &mut memo,
                meter,
            )?;
            charge_evaluation_steps(meter, 1)?;
            jacobian[entry.row * self.num_vars() + entry.col] = value;
        }
        charge_evaluation_steps(meter, 1)?;
        Ok(ResidualEvaluation {
            residuals,
            jacobian,
        })
    }

    /// Publish both dense result buffers or leave them unchanged on refusal.
    pub fn try_eval_dense_into(
        &self,
        point: &[f64],
        residuals: &mut [f64],
        jacobian: &mut [f64],
    ) -> Result<(), ResidualEvaluationError> {
        self.try_eval_dense_into_with_meter(point, residuals, jacobian, &mut Unbounded)
    }

    /// Source checks, scratch storage, sparse evaluation and publication consume
    /// one cumulative caller ledger. No caller buffer is changed before the
    /// final cancellation checkpoint and all publication charges succeed.
    pub fn try_eval_dense_into_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        residuals: &mut [f64],
        jacobian: &mut [f64],
        meter: &mut M,
    ) -> Result<(), ResidualEvaluationError> {
        let dense_len = self.dense_evaluation_len()?;
        charge_evaluation_steps(meter, 1)?;
        for (lane, expected, actual) in [
            ("residual output", self.num_residuals(), residuals.len()),
            ("dense Jacobian output", dense_len, jacobian.len()),
        ] {
            if expected != actual {
                return Err(ResidualEvaluationError::Dimension {
                    lane,
                    expected,
                    actual,
                });
            }
        }
        let result = self.try_eval_dense_with_meter(point, meter)?;
        let published = residuals
            .len()
            .checked_add(jacobian.len())
            .ok_or(EvalError::DimensionOverflow)?;
        charge_evaluation_steps(meter, published)?;
        charge_evaluation_steps(meter, 0)?;
        residuals.copy_from_slice(&result.residuals);
        jacobian.copy_from_slice(&result.jacobian);
        Ok(())
    }

    /// Evaluate the original residuals and only stored Jacobian entries.
    pub fn evaluate_sparse(
        &self,
        point: &[f64],
    ) -> Result<SparseResidualEvaluation, ResidualEvaluationError> {
        self.evaluate_sparse_with_meter(point, &mut Unbounded)
    }

    /// Finite-real sparse evaluation under one cumulative caller ledger.
    ///
    /// Every original source is checked before derivative values, including
    /// domains hidden behind an identically zero derivative. Storage is
    /// proportional to residuals and structural nonzeros, without allocating
    /// dense Jacobian values. All scratch buffers and bytecode work are charged;
    /// failure or cancellation publishes no partial result.
    pub fn evaluate_sparse_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        meter: &mut M,
    ) -> Result<SparseResidualEvaluation, ResidualEvaluationError> {
        Self::input_with_meter(point, self.num_vars(), "point", meter)?;
        let mut memo = self.shared_evaluation.begin(meter)?;
        let mut residuals = Self::evaluation_buffer(self.num_residuals(), meter)?;
        for (index, program) in self.residual_programs().iter().enumerate() {
            charge_evaluation_steps(meter, 1)?;
            residuals.push(
                self.shared_evaluation
                    .evaluate(index, program, point, &mut memo, meter)?,
            );
        }
        let mut jacobian = Self::evaluation_buffer(self.sparse_jacobian.entries.len(), meter)?;
        for (index, entry) in self.sparse_jacobian.entries.iter().enumerate() {
            charge_evaluation_steps(meter, 1)?;
            jacobian.push(self.shared_evaluation.evaluate(
                self.num_residuals() + index,
                &entry.compiled_deriv,
                point,
                &mut memo,
                meter,
            )?);
        }
        charge_evaluation_steps(meter, 1)?;
        Ok(SparseResidualEvaluation {
            residuals,
            jacobian,
        })
    }

    /// Sparse evaluation into caller-owned buffers, unchanged on every refusal.
    pub fn try_eval_sparse_into(
        &self,
        point: &[f64],
        residuals: &mut [f64],
        jacobian: &mut [f64],
    ) -> Result<(), ResidualEvaluationError> {
        self.try_eval_sparse_into_with_meter(point, residuals, jacobian, &mut Unbounded)
    }

    /// Both result buffers are published together after all charges and checks.
    pub fn try_eval_sparse_into_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        residuals: &mut [f64],
        jacobian: &mut [f64],
        meter: &mut M,
    ) -> Result<(), ResidualEvaluationError> {
        charge_evaluation_steps(meter, 1)?;
        for (lane, expected, actual) in [
            ("residual output", self.num_residuals(), residuals.len()),
            (
                "sparse Jacobian output",
                self.sparse_jacobian.entries.len(),
                jacobian.len(),
            ),
        ] {
            if actual != expected {
                return Err(ResidualEvaluationError::Dimension {
                    lane,
                    expected,
                    actual,
                });
            }
        }
        let result = self.evaluate_sparse_with_meter(point, meter)?;
        let published = residuals
            .len()
            .checked_add(jacobian.len())
            .ok_or(EvalError::DimensionOverflow)?;
        charge_evaluation_steps(meter, published)?;
        charge_evaluation_steps(meter, 0)?;
        residuals.copy_from_slice(&result.residuals);
        jacobian.copy_from_slice(&result.jacobian);
        Ok(())
    }

    /// Evaluate J(point) times a direction without dense Jacobian values.
    pub fn jacobian_vector_product(
        &self,
        point: &[f64],
        direction: &[f64],
    ) -> Result<Vec<f64>, ResidualEvaluationError> {
        self.jacobian_vector_product_with_meter(point, direction, &mut Unbounded)
    }

    /// The original residuals and every derivative are checked even for a zero
    /// direction. Each multiplication and accumulated sum must remain finite.
    pub fn jacobian_vector_product_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        direction: &[f64],
        meter: &mut M,
    ) -> Result<Vec<f64>, ResidualEvaluationError> {
        self.product_with_meter(point, direction, false, meter)
    }

    /// Evaluate J(point)^T times residual weights without a dense transpose.
    pub fn jacobian_transpose_vector_product(
        &self,
        point: &[f64],
        weights: &[f64],
    ) -> Result<Vec<f64>, ResidualEvaluationError> {
        self.jacobian_transpose_vector_product_with_meter(point, weights, &mut Unbounded)
    }

    /// All source and derivative domain checks remain active for zero weights.
    /// Work, output storage and publication share the supplied cumulative meter.
    pub fn jacobian_transpose_vector_product_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        weights: &[f64],
        meter: &mut M,
    ) -> Result<Vec<f64>, ResidualEvaluationError> {
        self.product_with_meter(point, weights, true, meter)
    }

    fn product_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        vector: &[f64],
        transpose: bool,
        meter: &mut M,
    ) -> Result<Vec<f64>, ResidualEvaluationError> {
        let (input_length, output_length, lane) = if transpose {
            (self.num_residuals(), self.num_vars(), "weights")
        } else {
            (self.num_vars(), self.num_residuals(), "direction")
        };
        Self::input_with_meter(vector, input_length, lane, meter)?;
        Self::input_with_meter(point, self.num_vars(), "point", meter)?;
        let mut memo = self.shared_evaluation.begin(meter)?;
        // A derivative can erase an original source domain. Do not infer
        // eligibility from a zero vector or from the sparsity pattern alone.
        for (index, program) in self.residual_programs().iter().enumerate() {
            charge_evaluation_steps(meter, 1)?;
            self.shared_evaluation
                .evaluate(index, program, point, &mut memo, meter)?;
        }

        let mut result = Self::evaluation_buffer(output_length, meter)?;
        charge_evaluation_steps(meter, output_length)?;
        result.resize(output_length, 0.0);
        for (index, entry) in self.sparse_jacobian.entries.iter().enumerate() {
            let derivative = self.shared_evaluation.evaluate(
                self.num_residuals() + index,
                &entry.compiled_deriv,
                point,
                &mut memo,
                meter,
            )?;
            let (output_index, input_index) = if transpose {
                (entry.col, entry.row)
            } else {
                (entry.row, entry.col)
            };
            charge_evaluation_steps(meter, 2)?;
            let contribution = derivative * vector[input_index];
            let sum = result[output_index] + contribution;
            if !contribution.is_finite() || !sum.is_finite() {
                return Err(EvalError::NonFiniteValue.into());
            }
            result[output_index] = sum;
        }
        charge_evaluation_steps(meter, 1)?;
        Ok(result)
    }

    /// Publish a complete Jacobian-vector product or leave output unchanged.
    pub fn try_jacobian_vector_product_into(
        &self,
        point: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), ResidualEvaluationError> {
        self.try_jacobian_vector_product_into_with_meter(point, direction, output, &mut Unbounded)
    }

    pub fn try_jacobian_vector_product_into_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        direction: &[f64],
        output: &mut [f64],
        meter: &mut M,
    ) -> Result<(), ResidualEvaluationError> {
        self.product_into_with_meter(point, direction, output, false, meter)
    }

    /// Publish a complete transposed product or leave output unchanged.
    pub fn try_jacobian_transpose_vector_product_into(
        &self,
        point: &[f64],
        weights: &[f64],
        output: &mut [f64],
    ) -> Result<(), ResidualEvaluationError> {
        self.try_jacobian_transpose_vector_product_into_with_meter(
            point,
            weights,
            output,
            &mut Unbounded,
        )
    }

    pub fn try_jacobian_transpose_vector_product_into_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        weights: &[f64],
        output: &mut [f64],
        meter: &mut M,
    ) -> Result<(), ResidualEvaluationError> {
        self.product_into_with_meter(point, weights, output, true, meter)
    }

    fn product_into_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        vector: &[f64],
        output: &mut [f64],
        transpose: bool,
        meter: &mut M,
    ) -> Result<(), ResidualEvaluationError> {
        charge_evaluation_steps(meter, 1)?;
        let expected = if transpose {
            self.num_vars()
        } else {
            self.num_residuals()
        };
        if output.len() != expected {
            return Err(ResidualEvaluationError::Dimension {
                lane: if transpose {
                    "transposed Jacobian product output"
                } else {
                    "Jacobian product output"
                },
                expected,
                actual: output.len(),
            });
        }
        let result = self.product_with_meter(point, vector, transpose, meter)?;
        charge_evaluation_steps(meter, output.len())?;
        charge_evaluation_steps(meter, 0)?;
        output.copy_from_slice(&result);
        Ok(())
    }

    pub fn residuals(&self) -> &[Expr] {
        &self.sparse_jacobian.residuals
    }

    pub fn vars(&self) -> &[Symbol] {
        &self.sparse_jacobian.vars
    }

    /// Receipt for derivative proofs, sparsity and bytecode provenance only.
    pub fn receipt(&self) -> &SparseJacobianReceipt {
        &self.receipt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_sparse_program_is_the_program_executed_by_consumers() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let residuals = vec![
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
            Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
        ];
        let vars = vec![x, y];
        let sparse = SparseJacobian::try_build(&residuals, &vars).unwrap();
        let verified = VerifiedResidualSystem::try_from_sparse_jacobian(&sparse).unwrap();
        assert_eq!(verified.residuals(), residuals);
        assert_eq!(verified.vars(), vars);
        assert_eq!(verified.receipt().proofs_verified, sparse.entries.len());
        assert_eq!(verified.receipt().test_points_evaluated, 0);
        let dense_program = verified.try_dense_program().unwrap();
        for entry in &sparse.entries {
            assert_eq!(
                dense_program.compiled_jacobian[entry.row * vars.len() + entry.col],
                entry.compiled_deriv,
            );
        }
        let mut values = [0.0; 2];
        let mut jacobian = [0.0; 4];
        verified
            .try_eval_dense_into(&[3.0, 5.0], &mut values, &mut jacobian)
            .unwrap();
        assert_eq!(values, [9.0, 8.0]);
        assert_eq!(jacobian, [6.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn derivative_bytecode_cannot_be_swapped_after_its_proof_was_generated() {
        let x = Symbol::new("x");
        let mut sparse =
            SparseJacobian::try_build(&[Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2))], &[x])
                .unwrap();
        sparse.entries[0].compiled_deriv =
            CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
        assert!(matches!(
            VerifiedResidualSystem::try_from_sparse_jacobian(&sparse),
            Err(VerifiedCompilationError::Jacobian(
                JacobianVerificationError::CompiledProvenanceMismatch { .. }
            )),
        ));
    }

    #[test]
    fn empty_verified_system_is_well_formed() {
        let system = VerifiedResidualSystem::try_compile(&[], &[]).unwrap();
        system.try_eval_dense_into(&[], &mut [], &mut []).unwrap();
        assert_eq!(system.receipt().proofs_verified, 0);
    }

    #[test]
    fn streaming_encoding_refuses_before_crossing_the_byte_budget() {
        let mut writer = BoundedArtifactWriter::new(5, 96);
        serde_json::to_writer(&mut writer, &[1, 2]).unwrap();
        assert_eq!(writer.bytes, b"[1,2]");
        assert!(writer.write_all(b" ").is_err());
        assert_eq!(writer.bytes, b"[1,2]");
        assert!(writer.bytes.capacity() <= 5);
        assert_eq!(
            writer.failure,
            Some(VerifiedArtifactError::ByteLimit { actual: 6, max: 5 })
        );
    }

    #[test]
    fn streaming_encoding_refuses_nested_json_before_publication() {
        let mut accepted = BoundedArtifactWriter::new(1024, 2);
        serde_json::to_writer(&mut accepted, &serde_json::json!([[0]])).unwrap();
        assert_eq!(accepted.bytes, b"[[0]]");
        let mut rejected = BoundedArtifactWriter::new(1024, 2);
        assert!(serde_json::to_writer(&mut rejected, &serde_json::json!([[[0]]])).is_err());
        assert_eq!(
            rejected.failure,
            Some(VerifiedArtifactError::DepthLimit { depth: 3, max: 2 })
        );
    }
}
