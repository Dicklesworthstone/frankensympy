//! Immutable bridge from checked symbolic Jacobians to numerical consumers.
//!
//! The artifact retains its complete sparse proof bundle and the exact bytecode
//! checked at admission. Versioned JSON transport replays that admission before
//! returning a verified value; serialized receipts are never trusted.
//! Binary64 evaluation remains approximate and does not establish a root.

#![forbid(unsafe_code)]

use crate::compile::{
    CompileError, CompiledExpr, CompiledOp, CompiledResidualSystem, MAX_COMPILE_OPS,
};
use crate::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, SparseJacobianReceipt, compiled_expression_matches,
    verify_sparse_jacobian_certificate,
};
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
/// This bridge materializes a dense consumer program from a sparse certificate.
/// Bound that expansion before proof replay or allocation, including on decode.
pub const MAX_VERIFIED_JACOBIAN_ENTRIES: usize = 65_536;
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
    #[error("Proof payload exceeds its admission envelope: {0}")]
    ProofEnvelope(String),
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
    compiled: CompiledResidualSystem,
    sparse_jacobian: SparseJacobian,
    receipt: SparseJacobianReceipt,
}

impl VerifiedResidualSystem {
    /// Builds derivatives and carries those same checked programs to consumers.
    pub fn try_compile(
        residuals: &[Expr],
        vars: &[Symbol],
    ) -> Result<Self, VerifiedCompilationError> {
        Self::check_dense_size(residuals.len(), vars.len())?;
        for residual in residuals {
            Self::check_expression_shape(residual)?;
        }
        Self::try_from_owned_sparse_jacobian(SparseJacobian::try_build(residuals, vars)?)
    }

    /// Admits a possibly untrusted sparse artifact and retains its full proof.
    /// No numerical sample is required for symbolic or bytecode acceptance.
    pub fn try_from_sparse_jacobian(
        jacobian: &SparseJacobian,
    ) -> Result<Self, VerifiedCompilationError> {
        let (compiled, receipt) = Self::check_and_compile(jacobian)?;
        Ok(Self {
            compiled,
            sparse_jacobian: jacobian.clone(),
            receipt,
        })
    }

    fn try_from_owned_sparse_jacobian(
        jacobian: SparseJacobian,
    ) -> Result<Self, VerifiedCompilationError> {
        let (compiled, receipt) = Self::check_and_compile(&jacobian)?;
        Ok(Self {
            compiled,
            sparse_jacobian: jacobian,
            receipt,
        })
    }

    fn check_dense_size(rows: usize, columns: usize) -> Result<usize, VerifiedCompilationError> {
        for (axis, length) in [("rows", rows), ("columns", columns)] {
            if length > MAX_VERIFIED_JACOBIAN_ENTRIES {
                return Err(VerifiedCompilationError::AxisLimit {
                    axis,
                    length,
                    max: MAX_VERIFIED_JACOBIAN_ENTRIES,
                });
            }
        }
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

    fn check_expression_shape(expr: &Expr) -> Result<(), VerifiedCompilationError> {
        Self::check_expression_limits(expr, MAX_VERIFIED_SOURCE_DEPTH, MAX_COMPILE_OPS)
    }

    fn check_proof_expression(expr: &Expr) -> Result<(), VerifiedCompilationError> {
        // The independently checked differentiation claim adds a diff(source,
        // variable) wrapper: one level and two nodes beyond the source.
        Self::check_expression_limits(expr, MAX_VERIFIED_SOURCE_DEPTH + 1, MAX_COMPILE_OPS + 2)
    }

    fn check_expression_limits(
        expr: &Expr,
        maximum_depth: usize,
        maximum_nodes: usize,
    ) -> Result<(), VerifiedCompilationError> {
        let mut pending = Vec::new();
        pending
            .try_reserve(1)
            .map_err(|_| CompileError::AllocationFailure)?;
        pending.push((expr, 0usize));
        let mut visited = 0usize;
        while let Some((current, depth)) = pending.pop() {
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
        Ok(())
    }

    fn check_proof_claim(claim: &Claim) -> Result<(), VerifiedCompilationError> {
        match claim {
            Claim::Equality { lhs, rhs } | Claim::AlgebraicIdentity { lhs, rhs } => {
                Self::check_proof_expression(lhs)?;
                Self::check_proof_expression(rhs)
            }
            Claim::PredicateHold { expr, .. }
            | Claim::DomainMembership { expr, .. }
            | Claim::NonZero(expr) => Self::check_proof_expression(expr),
        }
    }

    fn check_proof_rule(rule: &ProofRule) -> Result<(), VerifiedCompilationError> {
        match rule {
            ProofRule::Reflexivity(expr)
            | ProofRule::ContextPredicate { expr, .. }
            | ProofRule::ContextDomain { expr, .. } => Self::check_proof_expression(expr),
            ProofRule::Substitution { template, .. } => Self::check_proof_expression(template),
            ProofRule::DefinitionalReduction { lhs, rhs, .. } => {
                Self::check_proof_expression(lhs)?;
                Self::check_proof_expression(rhs)
            }
            ProofRule::CertificateLemma { claim, .. } => Self::check_proof_claim(claim),
            ProofRule::Symmetry(_)
            | ProofRule::Transitivity(_, _)
            | ProofRule::CongruenceAdd(_)
            | ProofRule::CongruenceMul(_)
            | ProofRule::CongruencePow { .. }
            | ProofRule::CongruenceFunction { .. } => Ok(()),
        }
    }

    fn check_and_compile(
        jacobian: &SparseJacobian,
    ) -> Result<(CompiledResidualSystem, SparseJacobianReceipt), VerifiedCompilationError> {
        let num_residuals = jacobian.residuals.len();
        let num_vars = jacobian.vars.len();
        let dense_len = Self::check_dense_size(num_residuals, num_vars)?;
        // Reject duplicate/extra coordinates and inconsistent counts before
        // traversing any untrusted symbolic or proof payload.
        jacobian.validate_structure()?;
        for expression in jacobian
            .residuals
            .iter()
            .chain(jacobian.entries.iter().map(|entry| &entry.symbolic_deriv))
        {
            Self::check_expression_shape(expression)?;
        }
        for entry in &jacobian.entries {
            // Reuse the kernel's iterative limits for all step references,
            // claims, domains, text, and numeric payloads, including unused steps.
            derivation_verification_units(&entry.derivation)
                .map_err(|error| VerifiedCompilationError::ProofEnvelope(error.to_string()))?;
            for step in &entry.derivation.steps {
                Self::check_proof_claim(&step.claim)?;
                Self::check_proof_rule(&step.rule)?;
            }
        }
        let receipt = verify_sparse_jacobian_certificate(jacobian, &[])?;
        let mut var_map = HashMap::new();
        var_map
            .try_reserve(num_vars)
            .map_err(|_| CompileError::AllocationFailure)?;
        for (index, variable) in jacobian.vars.iter().enumerate() {
            var_map.insert(variable.clone(), index);
        }

        let mut compiled_residuals = Vec::new();
        compiled_residuals
            .try_reserve_exact(num_residuals)
            .map_err(|_| CompileError::AllocationFailure)?;
        for (residual, source) in jacobian.residuals.iter().enumerate() {
            let compiled = CompiledExpr::try_compile(source, &var_map)?;
            if !compiled_expression_matches(source, &var_map, &compiled)? {
                return Err(VerifiedCompilationError::ResidualProvenanceMismatch { residual });
            }
            compiled_residuals.push(compiled);
        }

        let mut compiled_jacobian = Vec::new();
        compiled_jacobian
            .try_reserve_exact(dense_len)
            .map_err(|_| CompileError::AllocationFailure)?;
        let mut entries = jacobian.entries.iter().peekable();
        for offset in 0..dense_len {
            let mut ops = Vec::new();
            if let Some(entry) = entries.peek()
                && entry.row * num_vars + entry.col == offset
            {
                ops.try_reserve_exact(entry.compiled_deriv.ops().len())
                    .map_err(|_| CompileError::AllocationFailure)?;
                ops.extend_from_slice(entry.compiled_deriv.ops());
                entries.next();
            } else {
                ops.try_reserve_exact(1)
                    .map_err(|_| CompileError::AllocationFailure)?;
                ops.push(CompiledOp::LoadConst(0.0));
            }
            compiled_jacobian
                .push(CompiledExpr::try_from_ops(ops).map_err(CompileError::InvalidBytecode)?);
        }
        debug_assert!(entries.next().is_none());

        let mut vars = Vec::new();
        vars.try_reserve_exact(num_vars)
            .map_err(|_| CompileError::AllocationFailure)?;
        vars.extend_from_slice(&jacobian.vars);
        Ok((
            CompiledResidualSystem {
                vars,
                num_residuals,
                num_vars,
                compiled_residuals,
                compiled_jacobian,
            },
            receipt,
        ))
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

    /// Read-only access for numerical consumers. An edited clone is an ordinary,
    /// unverified compiled system.
    pub fn compiled(&self) -> &CompiledResidualSystem {
        &self.compiled
    }

    /// Complete retained proof bundle for independent replay or checkpointing.
    /// Editing a clone cannot change the already admitted system.
    pub fn sparse_jacobian(&self) -> &SparseJacobian {
        &self.sparse_jacobian
    }

    pub fn residuals(&self) -> &[Expr] {
        &self.sparse_jacobian.residuals
    }

    pub fn vars(&self) -> &[Symbol] {
        &self.compiled.vars
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
        for entry in &sparse.entries {
            assert_eq!(
                verified.compiled().compiled_jacobian[entry.row * vars.len() + entry.col],
                entry.compiled_deriv,
            );
        }
        let mut values = [0.0; 2];
        let mut jacobian = [0.0; 4];
        verified
            .compiled()
            .try_eval_system(&[3.0, 5.0], &mut values, &mut jacobian)
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
        system
            .compiled()
            .try_eval_system(&[], &mut [], &mut [])
            .unwrap();
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
