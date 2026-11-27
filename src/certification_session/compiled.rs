//! Resident, independently admitted sparse residual/Jacobian programs.
//!
//! A compile receipt concerns symbolic derivatives and bytecode provenance.
//! Numerical execution remains approximate binary64 and evaluates the frozen
//! original residuals, including their domain guards, on every request.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError,
    CompiledLeastSquaresBackend, CompiledRootBackend, CompiledRootSolveOptions,
    CompiledStationarityOptions, MAX_SESSION_ROOT_ARTIFACT_BYTES, ResidualSystemId, admit_response,
    charge, valid_label,
};
use crate::certification::ResidualProblem;
use crate::program_transport::{decode_residual_program, export_residual_program};
use crate::request_budget::RequestBudget;
use fsym_budget::{BudgetError, BudgetMeter, MeterError, VerifierLease};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompileError, EvalError};
use fsym_calculus::program_artifact::admit_residual_candidate;
use fsym_calculus::sparse_jacobian::{JacobianVerificationError, SparseJacobian};
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedCompilationError};
use fsym_core::{BallError, LambertWError};
use fsym_numeric_consumer::LeastSquaresTermination;
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

pub(super) struct ResidualReplay<'a> {
    pub source: ResidualSystemId,
    pub id: CompiledResidualSystemId,
    pub problem: &'a ResidualProblem,
    pub context_digest: [u8; 32],
    pub bytes: &'a [u8],
}

impl CompiledState {
    pub(super) fn system(&self, id: &CompiledResidualSystemId) -> Option<&VerifiedResidualSystem> {
        self.programs.get(id).map(|resident| &resident.program)
    }

    pub(super) fn source_record(
        &self,
        id: &CompiledResidualSystemId,
    ) -> Option<(&ResidualSystemId, [u8; 32], &VerifiedResidualSystem)> {
        self.programs
            .get(id)
            .map(|resident| (&resident.source, resident.context_digest, &resident.program))
    }

    pub(super) fn compile(
        &mut self,
        source: ResidualSystemId,
        id: CompiledResidualSystemId,
        problem: &ResidualProblem,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_id(&id, budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        charge(budget, 1, 4096)?;
        let candidate =
            SparseJacobian::try_build_with_meter(problem.residuals(), problem.variables(), budget)
                .map_err(VerifiedCompilationError::from)?;
        let program = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
            candidate,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;
        self.publish(source, id, program, context_digest, budget)
    }

    pub(super) fn replay(
        &mut self,
        request: ResidualReplay<'_>,
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_id(&request.id, budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        charge(budget, 1, 4096)?;
        // Untrusted byte parsing uses ordinary resources. Exact source binding,
        // derivative proof replay and bytecode admission have protected fuel.
        let candidate = decode_residual_program(request.bytes, request.context_digest, budget)?;
        let program = admit_residual_candidate(
            request.problem.residuals(),
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
        id: &CompiledResidualSystemId,
        budget: &mut RequestBudget<'_>,
    ) -> Result<(), CertificationSessionError> {
        budget.checkpoint()?;
        if self.programs.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.programs.len() >= MAX_SESSION_COMPILED_SYSTEMS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        Ok(())
    }

    fn publish(
        &mut self,
        source: ResidualSystemId,
        id: CompiledResidualSystemId,
        program: VerifiedResidualSystem,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
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

    pub(super) fn export_bytes(
        &self,
        id: &CompiledResidualSystemId,
        budget: &mut RequestBudget<'_>,
    ) -> Result<Vec<u8>, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        // Export the program's frozen context, even after session rebinding.
        Ok(export_residual_program(
            &resident.program,
            resident.context_digest,
            budget,
        )?)
    }

    pub(super) fn export(
        &self,
        id: CompiledResidualSystemId,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        let bytes = self.export_bytes(&id, budget)?;
        if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
            return Err(CertificationSessionError::ArtifactLimit);
        }
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        charge(budget, 1, resident.source.as_str().len())?;
        let artifact = String::from_utf8(bytes).map_err(|_| {
            CertificationSessionError::Serialization("invalid program artifact UTF-8".into())
        })?;
        let response = CertificationResponse::CompiledResidualSystemExported {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: resident.source.clone(),
            program: id,
            artifact,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    /// Reuse an immutable, already admitted artifact for an approximate solve.
    /// Solving consumes ordinary work and cannot create or strengthen a proof.
    pub(super) fn solve(
        &self,
        id: CompiledResidualSystemId,
        initial: &[String],
        backend: CompiledRootBackend,
        options: &CompiledRootSolveOptions,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        let proposal = super::compiled_solvers::solve_root(
            &resident.program,
            initial,
            backend,
            options,
            budget,
        )?;
        charge(budget, 1, resident.source.as_str().len())?;
        let response = CertificationResponse::CompiledResidualSystemSolved {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: resident.source.clone(),
            program: id,
            backend,
            outcome: "approximate",
            numeric_model: "binary64",
            root: proposal.root,
            residual_norm: proposal.residual_norm,
            iterations: proposal.iterations,
            termination: "residual_tolerance",
            context_digest: resident.context_digest,
        };
        // A refusal retains the immutable program and all consumed work.
        admit_response(&response, budget)?;
        Ok(response)
    }

    /// Solve only the leading unknown coordinates. The source and complete
    /// typed axis order remain attached to the reusable resident program.
    pub(super) fn solve_parametric(
        &self,
        id: CompiledResidualSystemId,
        initial: &[String],
        parameters: &[String],
        options: &CompiledRootSolveOptions,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        let (proposal, parameters) = super::compiled_solvers::solve_parametric_root(
            &resident.program,
            initial,
            parameters,
            options,
            budget,
        )?;
        let axes = resident.program.vars();
        charge(
            budget,
            axes.len() + 1,
            axes.iter().map(|axis| axis.name.len() + 128).sum::<usize>()
                + resident.source.as_str().len(),
        )?;
        let dimension = resident.program.num_residuals();
        let response = CertificationResponse::CompiledParametricSystemSolved {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: resident.source.clone(),
            program: id,
            backend: CompiledRootBackend::FrankenNumpy,
            outcome: "approximate",
            numeric_model: "binary64",
            unknowns: axes[..dimension].to_vec(),
            parameter_variables: axes[dimension..].to_vec(),
            parameters,
            root: proposal.root,
            residual_norm: proposal.residual_norm,
            iterations: proposal.iterations,
            termination: "residual_tolerance",
            context_digest: resident.context_digest,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn least_squares(
        &self,
        id: CompiledResidualSystemId,
        initial: &[String],
        backend: CompiledLeastSquaresBackend,
        options: &CompiledStationarityOptions,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        let proposal = super::compiled_solvers::solve_least_squares(
            &resident.program,
            initial,
            backend,
            options,
            budget,
        )?;
        charge(budget, 1, resident.source.as_str().len())?;
        let response = CertificationResponse::CompiledLeastSquaresSolved {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: resident.source.clone(),
            program: id,
            backend,
            outcome: "approximate",
            numeric_model: "binary64",
            point: proposal.point,
            residuals: proposal.residuals,
            cost: proposal.cost,
            gradient_norm: proposal.gradient_norm,
            iterations: proposal.iterations,
            evaluations: proposal.evaluations,
            termination: match proposal.termination {
                LeastSquaresTermination::GradientTolerance => "gradient_tolerance",
            },
            context_digest: resident.context_digest,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn bounded_least_squares(
        &self,
        id: CompiledResidualSystemId,
        initial: &[String],
        lower: &[Option<String>],
        upper: &[Option<String>],
        options: &CompiledStationarityOptions,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        let axes = resident.program.num_vars();
        let initial = numeric_values(initial, axes, "initial", budget)?;
        let lower = numeric_coordinates(
            lower.iter().map(Option::as_deref),
            axes,
            "lower",
            Some(f64::NEG_INFINITY),
            budget,
        )?;
        let upper = numeric_coordinates(
            upper.iter().map(Option::as_deref),
            axes,
            "upper",
            Some(f64::INFINITY),
            budget,
        )?;
        let controls = super::compiled_solvers::least_squares_controls(options, budget)?;
        let fit = fsym_numeric_consumer::sparse_bounded_least_squares_with_meter(
            &resident.program,
            &initial,
            &lower,
            &upper,
            &controls,
            budget,
        )?;
        charge(budget, 1, resident.source.as_str().len())?;
        let response = CertificationResponse::CompiledBoundedLeastSquaresSolved {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: resident.source.clone(),
            program: id,
            backend: CompiledLeastSquaresBackend::Sparse,
            outcome: "approximate",
            numeric_model: "binary64",
            point: fit.point,
            residuals: fit.residuals,
            cost: fit.cost,
            gradient_norm: fit.gradient_norm,
            projected_gradient_norm: fit.projected_gradient_norm,
            iterations: fit.iterations,
            evaluations: fit.evaluations,
            termination: "projected_gradient_tolerance",
            context_digest: resident.context_digest,
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn evaluate(
        &self,
        id: CompiledResidualSystemId,
        point: &[String],
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        let point = numeric_values(point, resident.program.num_vars(), "point", budget)?;
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

    /// Use the checked sparse programs directly, without materializing or
    /// returning a Jacobian. Even zero vectors retain all original source and
    /// derivative domain checks in the native product evaluator.
    pub(super) fn jacobian_product(
        &self,
        id: CompiledResidualSystemId,
        point: &[String],
        vector: &[String],
        transpose: bool,
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let resident = self
            .programs
            .get(&id)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        let point = numeric_values(point, resident.program.num_vars(), "point", budget)?;
        let (expected, lane) = if transpose {
            (resident.program.num_residuals(), "weights")
        } else {
            (resident.program.num_vars(), "direction")
        };
        let vector = numeric_values(vector, expected, lane, budget)?;
        let product = if transpose {
            resident
                .program
                .jacobian_transpose_vector_product_with_meter(&point, &vector, budget)?
        } else {
            resident
                .program
                .jacobian_vector_product_with_meter(&point, &vector, budget)?
        };
        charge(budget, 1, resident.source.as_str().len())?;
        let response = if transpose {
            CertificationResponse::JacobianTransposeVectorProductEvaluated {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system: resident.source.clone(),
                program: id,
                outcome: "approximate",
                numeric_model: "binary64",
                point,
                weights: vector,
                product,
                context_digest: resident.context_digest,
            }
        } else {
            CertificationResponse::JacobianVectorProductEvaluated {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system: resident.source.clone(),
                program: id,
                outcome: "approximate",
                numeric_model: "binary64",
                point,
                direction: vector,
                product,
                context_digest: resident.context_digest,
            }
        };
        admit_response(&response, budget)?;
        Ok(response)
    }

    pub(super) fn release(
        &mut self,
        id: CompiledResidualSystemId,
        budget: &mut RequestBudget<'_>,
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

pub(super) fn numeric_values(
    values: &[String],
    expected: usize,
    lane: &'static str,
    budget: &mut RequestBudget<'_>,
) -> Result<Vec<f64>, CertificationSessionError> {
    numeric_coordinates(
        values.iter().map(|s| Some(s.as_str())),
        expected,
        lane,
        None,
        budget,
    )
}

fn numeric_coordinates<'a>(
    values: impl ExactSizeIterator<Item = Option<&'a str>>,
    expected: usize,
    lane: &'static str,
    missing: Option<f64>,
    budget: &mut RequestBudget<'_>,
) -> Result<Vec<f64>, CertificationSessionError> {
    charge(budget, 1, 0)?;
    if values.len() != expected {
        return Err(ResidualEvaluationError::Dimension {
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
        .map_err(|_| ResidualEvaluationError::Evaluation(EvalError::AllocationFailure))?;
    for value in values {
        let Some(value) = value else {
            result.push(missing.ok_or(CertificationSessionError::InvalidNumericPoint)?);
            continue;
        };
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

pub(super) fn compile_error_code(error: &CompileError) -> &'static str {
    match error {
        CompileError::Meter(MeterError::Cancelled) => "cancelled",
        CompileError::Meter(MeterError::Budget(_)) => "budget_exhausted",
        CompileError::DerivativeNormalization(error) => match error {
            fsym_simplify::SimplifyError::Cancelled => "cancelled",
            fsym_simplify::SimplifyError::BudgetExhausted(_) => "budget_exhausted",
            fsym_simplify::SimplifyError::DepthLimitExceeded(_)
            | fsym_simplify::SimplifyError::NumericLimitExceeded { .. }
            | fsym_simplify::SimplifyError::AllocationFailure => "resource_limit_exceeded",
            fsym_simplify::SimplifyError::ArithmeticInvariant(_)
            | fsym_simplify::SimplifyError::General(_)
            | fsym_simplify::SimplifyError::ProofFailed(_) => "internal_error",
        },
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

pub(super) fn eval_error_code(error: &EvalError, refused: &'static str) -> &'static str {
    match error {
        EvalError::Meter(MeterError::Cancelled) => "cancelled",
        EvalError::Meter(MeterError::Budget(_)) => "budget_exhausted",
        EvalError::OpLimitExceeded { .. }
        | EvalError::DimensionOverflow
        | EvalError::AllocationFailure
        | EvalError::LambertW(LambertWError::Ball(BallError::ResourceLimitExceeded { .. })) => {
            "resource_limit_exceeded"
        }
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
