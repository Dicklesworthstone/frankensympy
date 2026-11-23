//! Resident, independently admitted scalar objective derivative programs.
//!
//! Symbolic gradient/Hessian proofs and bytecode provenance are checked once
//! before publication. Each execution still checks the frozen original source
//! and derivative domains. Binary64 results remain approximate; compilation
//! grants no interval, convexity, smoothness-on-a-box or optimality claim.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError,
    CompiledStationarityOptions, MAX_SESSION_ROOT_ARTIFACT_BYTES, ObjectiveId, admit_response,
    charge, valid_label,
};
use crate::objective::ObjectiveProblem;
use crate::program_transport::{decode_objective_program, export_objective_program};
use crate::request_budget::RequestBudget;
use fsym_budget::{BudgetError, BudgetMeter, MeterError, VerifierLease};
use fsym_calculus::compile::EvalError;
use fsym_calculus::program_artifact::admit_objective_candidate;
use fsym_calculus::sparse_hessian::{
    ObjectiveCandidate, ObjectiveCompilationError, ObjectiveEvaluationError, VerifiedObjective,
};
use fsym_calculus::sparse_jacobian::JacobianVerificationError;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::mem::{size_of, size_of_val};

/// Resident objective programs have a separate bound from residual systems.
/// Releasing a program never refunds computation or allocation already charged.
pub const MAX_SESSION_COMPILED_OBJECTIVES: usize = 32;
const MAX_POINT_SCALAR_BYTES: usize = 512;

/// An immutable checked objective program scoped to one session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct CompiledObjectiveId(Box<str>);

impl CompiledObjectiveId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "objective_program:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CompiledObjectiveId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

struct ResidentObjective {
    source: ObjectiveId,
    context_digest: [u8; 32],
    program: VerifiedObjective,
}

#[derive(Default)]
pub(super) struct CompiledObjectiveState {
    programs: BTreeMap<CompiledObjectiveId, ResidentObjective>,
}

pub(super) struct ObjectiveReplay<'a> {
    pub source: ObjectiveId,
    pub id: CompiledObjectiveId,
    pub problem: &'a ObjectiveProblem,
    pub context_digest: [u8; 32],
    pub bytes: &'a [u8],
}

impl CompiledObjectiveState {
    pub(super) fn objective(&self, id: &CompiledObjectiveId) -> Option<&VerifiedObjective> {
        self.programs.get(id).map(|resident| &resident.program)
    }

    /// Freeze the registered source and context alongside the checked program;
    /// numerical proposals cannot select a different verification statement.
    pub(super) fn source_record(
        &self,
        id: &CompiledObjectiveId,
    ) -> Option<(&ObjectiveId, [u8; 32], &VerifiedObjective)> {
        self.programs
            .get(id)
            .map(|resident| (&resident.source, resident.context_digest, &resident.program))
    }

    pub(super) fn compile(
        &mut self,
        source: ObjectiveId,
        id: CompiledObjectiveId,
        problem: &ObjectiveProblem,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_id(&id, budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        charge(budget, 1, 4096)?;
        let candidate = ObjectiveCandidate::try_build_with_meter(
            problem.objective(),
            problem.variables(),
            budget,
        )?;
        let program = VerifiedObjective::try_from_candidate_with_meter(
            problem.objective(),
            problem.variables(),
            candidate,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;
        self.publish(source, id, program, context_digest, budget)
    }

    pub(super) fn replay(
        &mut self,
        request: ObjectiveReplay<'_>,
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_id(&request.id, budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        charge(budget, 1, 4096)?;
        let candidate = decode_objective_program(request.bytes, request.context_digest, budget)?;
        let program = admit_objective_candidate(
            request.problem.objective(),
            request.problem.variables(),
            candidate,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;
        self.publish(
            request.source,
            request.id,
            program,
            request.context_digest,
            budget,
        )
    }

    fn admit_id(
        &self,
        id: &CompiledObjectiveId,
        budget: &mut RequestBudget<'_>,
    ) -> Result<(), CertificationSessionError> {
        budget.checkpoint()?;
        if self.programs.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.programs.len() >= MAX_SESSION_COMPILED_OBJECTIVES {
            return Err(CertificationSessionError::ObjectLimit);
        }
        Ok(())
    }

    fn publish(
        &mut self,
        source: ObjectiveId,
        id: CompiledObjectiveId,
        program: VerifiedObjective,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        charge(
            budget,
            program.variables().len() + program.hessian_pattern().len(),
            program
                .variables()
                .iter()
                .map(|variable| variable.name.len() + size_of::<fsym_core::Symbol>())
                .sum::<usize>()
                + size_of_val(program.hessian_pattern()),
        )?;
        let response = CertificationResponse::ObjectiveCompiled {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            objective: source.clone(),
            program: id.clone(),
            evidence: "certificate_verified",
            claim: "formal_symbolic_gradient_hessian_and_binary64_lowering",
            variables: program.variables().to_vec(),
            hessian_pattern: program.hessian_pattern().to_vec(),
            gradient_proofs_verified: program.receipt().gradient_proofs_verified,
            hessian_proofs_verified: program.receipt().hessian_proofs_verified,
            zero_entries_checked: program.receipt().zero_entries_checked,
            context_digest,
        };
        // No handle exists until response reservation and the final safe point
        // succeed. A rejected candidate cannot become a resident program.
        admit_response(&response, budget)?;
        self.programs.insert(
            id,
            ResidentObjective {
                source,
                context_digest,
                program,
            },
        );
        Ok(response)
    }

    pub(super) fn export_bytes(
        &self,
        id: &CompiledObjectiveId,
        budget: &mut RequestBudget<'_>,
    ) -> Result<Vec<u8>, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(id)
            .ok_or(CertificationSessionError::UnknownCompiledObjective)?;
        Ok(export_objective_program(
            &resident.program,
            resident.context_digest,
            budget,
        )?)
    }

    pub(super) fn export(
        &self,
        id: CompiledObjectiveId,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        let bytes = self.export_bytes(&id, budget)?;
        if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
            return Err(CertificationSessionError::ArtifactLimit);
        }
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledObjective)?;
        charge(budget, 1, resident.source.as_str().len())?;
        let artifact = String::from_utf8(bytes).map_err(|_| {
            CertificationSessionError::Serialization("invalid program artifact UTF-8".into())
        })?;
        let response = CertificationResponse::CompiledObjectiveExported {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            objective: resident.source.clone(),
            program: id,
            artifact,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    /// Numerical descent reports approximate stationarity only. The immutable
    /// source and derivative guards remain active on every trial evaluation.
    pub(super) fn minimize(
        &self,
        id: CompiledObjectiveId,
        initial: &[String],
        options: &CompiledStationarityOptions,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledObjective)?;
        let proposal =
            super::compiled_solvers::minimize(&resident.program, initial, options, budget)?;
        charge(budget, 1, resident.source.as_str().len())?;
        let response = CertificationResponse::CompiledObjectiveMinimized {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            objective: resident.source.clone(),
            program: id,
            outcome: "approximate",
            numeric_model: "binary64",
            point: proposal.point,
            value: proposal.value,
            gradient: proposal.gradient,
            gradient_norm: proposal.gradient_norm,
            iterations: proposal.iterations,
            evaluations: proposal.evaluations,
            regularizations: proposal.regularizations,
            termination: "gradient_tolerance",
            context_digest: resident.context_digest,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn evaluate(
        &self,
        id: CompiledObjectiveId,
        point: &[String],
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledObjective)?;
        let point = numeric_values(point, resident.program.variables().len(), "point", budget)?;
        let evaluated = resident.program.evaluate_with_meter(&point, budget)?;
        charge(
            budget,
            resident.program.hessian_pattern().len(),
            size_of_val(resident.program.hessian_pattern()) + resident.source.as_str().len(),
        )?;
        let response = CertificationResponse::ObjectiveEvaluated {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            objective: resident.source.clone(),
            program: id,
            outcome: "approximate",
            numeric_model: "binary64",
            point,
            value: evaluated.value,
            gradient: evaluated.gradient,
            hessian_pattern: resident.program.hessian_pattern().to_vec(),
            hessian: evaluated.hessian,
            context_digest: resident.context_digest,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn hessian_vector_product(
        &self,
        id: CompiledObjectiveId,
        point: &[String],
        direction: &[String],
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledObjective)?;
        let count = resident.program.variables().len();
        let point = numeric_values(point, count, "point", budget)?;
        let direction = numeric_values(direction, count, "direction", budget)?;
        // The native sparse product rechecks the original objective and both
        // derivative stages, including for a zero direction. No dense Hessian
        // or full vector of Hessian values is materialized for this operation.
        let product = resident
            .program
            .hessian_vector_product_with_meter(&point, &direction, budget)?;
        charge(budget, 1, resident.source.as_str().len())?;
        let response = CertificationResponse::ObjectiveHessianVectorProduct {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            objective: resident.source.clone(),
            program: id,
            outcome: "approximate",
            numeric_model: "binary64",
            point,
            direction,
            product,
            context_digest: resident.context_digest,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn release(
        &mut self,
        id: CompiledObjectiveId,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        if !self.programs.contains_key(&id) {
            return Err(CertificationSessionError::UnknownCompiledObjective);
        }
        charge(budget, 1, id.as_str().len())?;
        let response = CertificationResponse::CompiledObjectiveReleased {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            program: id.clone(),
        };
        admit_response(&response, budget)?;
        self.programs.remove(&id);
        Ok(response)
    }
}

pub(super) fn numeric_values(
    values: &[String],
    expected: usize,
    lane: &'static str,
    budget: &mut RequestBudget<'_>,
) -> Result<Vec<f64>, CertificationSessionError> {
    charge(budget, 1, 0)?;
    if values.len() != expected {
        return Err(ObjectiveEvaluationError::Dimension {
            lane,
            expected,
            actual: values.len(),
        }
        .into());
    }
    charge(budget, expected, expected * size_of::<f64>())?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(expected)
        .map_err(|_| ObjectiveEvaluationError::Evaluation(EvalError::AllocationFailure))?;
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

pub(super) fn compilation_error_code(error: &ObjectiveCompilationError) -> &'static str {
    match error {
        ObjectiveCompilationError::Compilation(error)
        | ObjectiveCompilationError::Verification(JacobianVerificationError::Compile(error)) => {
            super::compiled::compile_error_code(error)
        }
        ObjectiveCompilationError::Verification(JacobianVerificationError::Eval(error)) => {
            super::compiled::eval_error_code(error, "verification_rejected")
        }
        ObjectiveCompilationError::ResourceLimit { .. } => "resource_limit_exceeded",
        ObjectiveCompilationError::Verification(_)
        | ObjectiveCompilationError::SourceMismatch(_)
        | ObjectiveCompilationError::InvalidProofEnvelope => "verification_rejected",
    }
}

pub(super) fn evaluation_error_code(error: &ObjectiveEvaluationError) -> &'static str {
    match error {
        ObjectiveEvaluationError::Evaluation(error) => {
            super::compiled::eval_error_code(error, "numeric_evaluation_refused")
        }
        ObjectiveEvaluationError::Dimension { .. } => "invalid_input",
        ObjectiveEvaluationError::NonFinite => "numeric_evaluation_refused",
    }
}
