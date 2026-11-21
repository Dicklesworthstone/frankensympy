//! Shared native Session wire format for explicitly portable Python programs.
//!
//! The legacy Python residual and objective formats remain available through
//! their existing APIs. This format includes the ambient context and never
//! admits a stored proof without rechecking the receiver's own exact source.

use super::PyVerifiedResidualSystem;
use crate::numeric_budget::{
    PyEvaluationBudget, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_calculus::program_artifact::context::{
    self, MAX_COMPILED_PROGRAM_ARTIFACT_BYTES, ProgramReplayError, ProgramTransportError,
};
use fsym_calculus::program_artifact::{admit_objective_candidate, admit_residual_candidate};
use fsym_calculus::{VerifiedObjective, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::LazyLock;

// Captured Symbol identities already retain their declarations. Python's
// compiled-program APIs add no ambient assumptions, exactly as an empty Session.
static CONTEXT: LazyLock<[u8; 32]> =
    LazyLock::new(|| ImmutableAssumptionsSnapshot::empty().digest());

pub(crate) fn check_data(
    data: &Bound<'_, PyAny>,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<()> {
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    if !data.is_exact_instance(&data.py().get_type::<PyBytes>()) {
        return Err(PyTypeError::new_err(
            "portable program data must be exact built-in bytes",
        ));
    }
    if data.cast::<PyBytes>()?.as_bytes().len() > MAX_COMPILED_PROGRAM_ARTIFACT_BYTES {
        return Err(PyValueError::new_err(
            "portable program data exceeds the 1 MiB byte limit",
        ));
    }
    Ok(())
}

pub(crate) fn export_residual<'py>(
    py: Python<'py>,
    program: &VerifiedResidualSystem,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = with_certification_meter(py, budget, |meter| {
        let bytes = context::export_residual_program(program, *CONTEXT, meter)?;
        meter
            .charge_batch(&[
                (Dimension::ComputeSteps, bytes.len() as u64 + 1),
                (Dimension::MemoryBytes, bytes.len() as u64),
                (Dimension::AllocationCount, 1),
            ])
            .map_err(fsym_core::root_certificate::artifact::CertificationError::from)?;
        Ok::<_, ProgramTransportError>(bytes)
    })?;
    let result = PyBytes::new(py, &bytes);
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}

pub(crate) fn export_objective<'py>(
    py: Python<'py>,
    program: &VerifiedObjective,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = with_certification_meter(py, budget, |meter| {
        let bytes = context::export_objective_program(program, *CONTEXT, meter)?;
        meter
            .charge_batch(&[
                (Dimension::ComputeSteps, bytes.len() as u64 + 1),
                (Dimension::MemoryBytes, bytes.len() as u64),
                (Dimension::AllocationCount, 1),
            ])
            .map_err(fsym_core::root_certificate::artifact::CertificationError::from)?;
        Ok::<_, ProgramTransportError>(bytes)
    })?;
    let result = PyBytes::new(py, &bytes);
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}

pub(crate) fn replay_residual(
    data: &Bound<'_, PyAny>,
    residuals: &[Expr],
    variables: &[Symbol],
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<VerifiedResidualSystem> {
    check_data(data, budget)?;
    let bytes = data.cast::<PyBytes>()?.as_bytes();
    with_certification_meter(data.py(), budget, |meter| {
        PyVerifiedResidualSystem::check_source_plan_with_meter(residuals, variables, meter)
            .map_err(fsym_calculus::program_artifact::ProgramAdmissionError::from)?;
        let candidate = context::decode_residual_program(bytes, *CONTEXT, meter)?;
        Ok::<_, ProgramReplayError>(admit_residual_candidate(
            residuals, variables, candidate, meter,
        )?)
    })
}

pub(crate) fn replay_objective(
    data: &Bound<'_, PyAny>,
    source: &Expr,
    variables: &[Symbol],
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<VerifiedObjective> {
    check_data(data, budget)?;
    let bytes = data.cast::<PyBytes>()?.as_bytes();
    with_certification_meter(data.py(), budget, |meter| {
        let candidate = context::decode_objective_program(bytes, *CONTEXT, meter)?;
        Ok::<_, ProgramReplayError>(admit_objective_candidate(
            source, variables, candidate, meter,
        )?)
    })
}
