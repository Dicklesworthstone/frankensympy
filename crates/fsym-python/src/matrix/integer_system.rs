//! Native integer-system bridge with one active request and private phase caps.

use super::{
    NormalFormRequestMeter, PyMatrix, PySparseMatrix, PySupervisedCallbackSession,
    integer_dense_output, integer_sparse_output, matrix_err,
};
use crate::to_value_error;
use fsym_budget::BudgetMeter;
use fsym_matrices::{IntegerMatrixRef, Matrix, SparseMatrix, metered_solve_integer};
use fsym_proof_kernel::integer_linear_system::IntegerLinearOutcome;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

fn checked_output<'py, 'session, T>(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    lift: impl FnOnce(IntegerLinearOutcome, &mut NormalFormRequestMeter<'py, 'session>) -> PyResult<T>,
) -> PyResult<T> {
    let mut generator = NormalFormRequestMeter::new(py, session);
    let mut verifier = NormalFormRequestMeter::new(py, session);
    let result = (|| {
        let accepted = metered_solve_integer(source, rhs, &mut generator, &mut verifier)
            .map_err(matrix_err)?;
        let (_, _, witness, _) = accepted.into_parts();
        let output = lift(witness.outcome, &mut generator)?;
        generator.checkpoint().map_err(to_value_error)?;
        verifier.checkpoint().map_err(to_value_error)?;
        Ok(output)
    })();
    // Operational refusal cannot become the mathematical `None` outcome.
    match generator.request.failure.or(verifier.request.failure) {
        Some(error) => Err(error),
        None => result,
    }
}

fn dense_result(
    outcome: IntegerLinearOutcome,
    meter: &mut NormalFormRequestMeter<'_, '_>,
) -> PyResult<Option<(PyMatrix, PyMatrix)>> {
    match outcome {
        IntegerLinearOutcome::Inconsistent { .. } => Ok(None),
        IntegerLinearOutcome::Solvable { particular, kernel } => Ok(Some((
            integer_dense_output(particular, meter)?,
            integer_dense_output(kernel, meter)?,
        ))),
    }
}

fn sparse_result(
    outcome: IntegerLinearOutcome,
    meter: &mut NormalFormRequestMeter<'_, '_>,
) -> PyResult<Option<(PySparseMatrix, PySparseMatrix)>> {
    match outcome {
        IntegerLinearOutcome::Inconsistent { .. } => Ok(None),
        IntegerLinearOutcome::Solvable { particular, kernel } => Ok(Some((
            integer_sparse_output(particular, meter)?,
            integer_sparse_output(kernel, meter)?,
        ))),
    }
}

pub(super) fn solve_dense(
    source: &Matrix,
    rhs: &Bound<'_, PyAny>,
    py: Python<'_>,
    session: Option<&PySupervisedCallbackSession>,
) -> PyResult<Option<(PyMatrix, PyMatrix)>> {
    if rhs.is_exact_instance_of::<PyMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PyMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            py,
            session,
            dense_result,
        )
    } else if rhs.is_exact_instance_of::<PySparseMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PySparseMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            py,
            session,
            dense_result,
        )
    } else {
        Err(PyTypeError::new_err(
            "integer solve requires an exact native dense or sparse RHS",
        ))
    }
}

pub(super) fn solve_sparse(
    source: &SparseMatrix,
    rhs: &Bound<'_, PyAny>,
    py: Python<'_>,
    session: Option<&PySupervisedCallbackSession>,
) -> PyResult<Option<(PySparseMatrix, PySparseMatrix)>> {
    if rhs.is_exact_instance_of::<PyMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PyMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            py,
            session,
            sparse_result,
        )
    } else if rhs.is_exact_instance_of::<PySparseMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PySparseMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            py,
            session,
            sparse_result,
        )
    } else {
        Err(PyTypeError::new_err(
            "integer solve requires an exact native dense or sparse RHS",
        ))
    }
}
