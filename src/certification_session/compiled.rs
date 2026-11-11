//! Resident, independently admitted sparse residual/Jacobian programs.
//!
//! A compile receipt concerns symbolic derivatives and bytecode provenance.
//! Numerical execution remains approximate binary64 and evaluates the frozen
//! original residuals, including their domain guards, on every request.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError,
    ResidualSystemId, admit_response, charge, valid_label,
};
use crate::certification::ResidualProblem;
use fsym_budget::{Budget, BudgetError, BudgetMeter, MeterError, VerifierLease};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompileError, EvalError};
use fsym_calculus::sparse_jacobian::{JacobianVerificationError, SparseJacobian};
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedCompilationError};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::mem::{size_of, size_of_val};

/// Independently bounded resident programs; releasing one never refunds work.
pub const MAX_SESSION_COMPILED_SYSTEMS: usize = 32;
const MAX_POINT_SCALAR_BYTES: usize = 512;

/// A compiled immutable residual system, scoped to one session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct CompiledResidualSystemId(Box<str>);

impl CompiledResidualSystemId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "program:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CompiledResidualSystemId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

struct ResidentProgram {
    source: ResidualSystemId,
    context_digest: [u8; 32],
    program: VerifiedResidualSystem,
}

#[derive(Default)]
pub(super) struct CompiledState {
    programs: BTreeMap<CompiledResidualSystemId, ResidentProgram>,
}

impl CompiledState {
    pub(super) fn system(&self, id: &CompiledResidualSystemId) -> Option<&VerifiedResidualSystem> {
        self.programs.get(id).map(|resident| &resident.program)
    }

    pub(super) fn compile(
        &mut self,
        source: ResidualSystemId,
        id: CompiledResidualSystemId,
        problem: &ResidualProblem,
        context_digest: [u8; 32],
        budget: &mut Budget,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        if self.programs.contains_key(&id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.programs.len() >= MAX_SESSION_COMPILED_SYSTEMS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        // Reserve bounded map/identifier metadata before creating a candidate.
        charge(budget, 1, 4096)?;
        let candidate =
            SparseJacobian::try_build_with_meter(problem.residuals(), problem.variables(), budget)
                .map_err(VerifiedCompilationError::from)?;
        let program = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
            candidate,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;
        charge(
            budget,
            program.num_vars() + program.jacobian_pattern().len(),
            program
                .vars()
                .iter()
                .map(|variable| variable.name.len() + size_of::<fsym_core::Symbol>())
                .sum::<usize>()
                + size_of_val(program.jacobian_pattern()),
        )?;
        let response = CertificationResponse::ResidualSystemCompiled {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: source.clone(),
            program: id.clone(),
            evidence: "certificate_verified",
            claim: "formal_symbolic_jacobian_and_binary64_lowering",
            variables: program.vars().to_vec(),
            residual_count: program.num_residuals(),
            jacobian_pattern: program.jacobian_pattern().to_vec(),
            proofs_verified: program.receipt().proofs_verified,
            stored_operations: program.stored_operation_count(),
            context_digest,
        };
        // The response and final safe point must succeed before the ID becomes
        // observable. Neither failed generation nor admission creates a handle.
        admit_response(&response, budget)?;
        self.programs.insert(
            id,
            ResidentProgram {
                source,
                context_digest,
                program,
            },
        );
        Ok(response)
    }

    pub(super) fn evaluate(
        &self,
        id: CompiledResidualSystemId,
        point: &[String],
        budget: &mut Budget,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        let point = point_values(point, resident.program.num_vars(), budget)?;
        let evaluated = resident
            .program
            .evaluate_sparse_with_meter(&point, budget)?;
        charge(
            budget,
            resident.program.jacobian_pattern().len(),
            size_of_val(resident.program.jacobian_pattern()) + resident.source.as_str().len(),
        )?;
        let response = CertificationResponse::ResidualJacobianEvaluated {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: resident.source.clone(),
            program: id,
            outcome: "approximate",
            numeric_model: "binary64",
            point,
            residuals: evaluated.residuals,
            jacobian_pattern: resident.program.jacobian_pattern().to_vec(),
            jacobian: evaluated.jacobian,
            context_digest: resident.context_digest,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn release(
        &mut self,
        id: CompiledResidualSystemId,
        budget: &mut Budget,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        if !self.programs.contains_key(&id) {
            return Err(CertificationSessionError::UnknownCompiledSystem);
        }
        charge(budget, 1, id.as_str().len())?;
        let response = CertificationResponse::CompiledResidualSystemReleased {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            program: id.clone(),
        };
        admit_response(&response, budget)?;
        self.programs.remove(&id);
        Ok(response)
    }
}

fn point_values(
    values: &[String],
    expected: usize,
    budget: &mut Budget,
) -> Result<Vec<f64>, CertificationSessionError> {
    charge(budget, 1, 0)?;
    if values.len() != expected {
        return Err(ResidualEvaluationError::Dimension {
            lane: "point",
            expected,
            actual: values.len(),
        }
        .into());
    }
    charge(budget, expected, expected * size_of::<f64>())?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(expected)
        .map_err(|_| ResidualEvaluationError::Evaluation(EvalError::AllocationFailure))?;
    for value in values {
        if value.is_empty() || value.len() > MAX_POINT_SCALAR_BYTES {
            return Err(CertificationSessionError::InvalidNumericPoint);
        }
        charge(budget, value.len(), 0)?;
        let parsed: f64 = value
            .parse()
            .map_err(|_| CertificationSessionError::InvalidNumericPoint)?;
        if !parsed.is_finite() {
            return Err(CertificationSessionError::InvalidNumericPoint);
        }
        result.push(parsed);
    }
    budget.checkpoint()?;
    Ok(result)
}

fn compile_error_code(error: &CompileError) -> &'static str {
    match error {
        CompileError::Meter(MeterError::Cancelled) => "cancelled",
        CompileError::Meter(MeterError::Budget(_)) => "budget_exhausted",
        CompileError::DepthLimitExceeded { .. }
        | CompileError::OpLimitExceeded { .. }
        | CompileError::NameLimitExceeded { .. }
        | CompileError::DimensionOverflow
        | CompileError::AllocationFailure
        | CompileError::SharedEvaluationLimit(_) => "resource_limit_exceeded",
        CompileError::DuplicateVariable(_) | CompileError::UnmappedVariable(_) => "invalid_input",
        CompileError::UnsupportedExpression(_)
        | CompileError::UnsupportedExponent(_)
        | CompileError::UnsupportedFunction(_)
        | CompileError::NumericConversion(_) => "unsupported_operation",
        CompileError::InvalidBytecode(error) => eval_error_code(error, "verification_rejected"),
        CompileError::SourceProvenanceMismatch | CompileError::InvalidSharedEvaluationPlan => {
            "verification_rejected"
        }
    }
}

pub(super) fn compilation_error_code(error: &VerifiedCompilationError) -> &'static str {
    match error {
        VerifiedCompilationError::Compilation(error)
        | VerifiedCompilationError::Jacobian(JacobianVerificationError::Compile(error)) => {
            compile_error_code(error)
        }
        VerifiedCompilationError::Jacobian(JacobianVerificationError::Eval(error)) => {
            eval_error_code(error, "verification_rejected")
        }
        VerifiedCompilationError::DenseJacobianLimit { .. }
        | VerifiedCompilationError::AxisLimit { .. }
        | VerifiedCompilationError::SourceDepthLimit { .. }
        | VerifiedCompilationError::AdmissionLimit { .. } => "resource_limit_exceeded",
        VerifiedCompilationError::Jacobian(_)
        | VerifiedCompilationError::ResidualProvenanceMismatch { .. }
        | VerifiedCompilationError::ProofEnvelope(_) => "verification_rejected",
    }
}

fn eval_error_code(error: &EvalError, refused: &'static str) -> &'static str {
    match error {
        EvalError::Meter(MeterError::Cancelled) => "cancelled",
        EvalError::Meter(MeterError::Budget(_)) => "budget_exhausted",
        EvalError::OpLimitExceeded { .. }
        | EvalError::DimensionOverflow
        | EvalError::AllocationFailure => "resource_limit_exceeded",
        _ => refused,
    }
}

pub(super) fn evaluation_error_code(error: &ResidualEvaluationError) -> &'static str {
    match error {
        ResidualEvaluationError::Evaluation(error) => {
            eval_error_code(error, "numeric_evaluation_refused")
        }
        ResidualEvaluationError::Dimension { .. } => "invalid_input",
        ResidualEvaluationError::DenseJacobianLimit { .. } => "resource_limit_exceeded",
    }
}
