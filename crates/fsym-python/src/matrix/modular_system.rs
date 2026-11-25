//! Complete modular families through the existing checked integer boundary.

use super::modular_inverse::modulus;
use super::{
    NormalFormRequestMeter, PyMatrix, PySparseMatrix, PySupervisedCallbackSession,
    integer_dense_output, integer_sparse_output, matrix_err,
};
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigInt, Expr};
use fsym_matrices::{IntegerMatrixRef, Matrix, SparseMatrix, metered_solve_mod};
use fsym_proof_kernel::integer_modular_system::IntegerModularSystemOutcome;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

pub(super) type DenseModularParts = (PyMatrix, PyMatrix, Vec<PyExpr>);
pub(super) type SparseModularParts = (PySparseMatrix, PySparseMatrix, Vec<PyExpr>);

fn checked_output<'py, 'session, T>(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    original_modulus: &Bound<'_, PyAny>,
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    lift: impl FnOnce(
        IntegerModularSystemOutcome,
        &mut NormalFormRequestMeter<'py, 'session>,
    ) -> PyResult<T>,
) -> PyResult<T> {
    let mut generator = NormalFormRequestMeter::new(py, session);
    let mut verifier = NormalFormRequestMeter::new(py, session);
    let result = (|| {
        let modulus = modulus(original_modulus, &mut generator, "solve_mod")?;
        let accepted = metered_solve_mod(source, rhs, &modulus, &mut generator, &mut verifier)
            .map_err(matrix_err)?;
        let (_, _, witness, _) = accepted.into_parts();
        let output = lift(witness.outcome, &mut generator)?;
        generator.checkpoint().map_err(to_value_error)?;
        verifier.checkpoint().map_err(to_value_error)?;
        Ok(output)
    })();
    // Inconsistency is a checked mathematical outcome. Operational failures
    // retain the original request exception and cannot become None.
    match generator.request.failure.or(verifier.request.failure) {
        Some(error) => Err(error),
        None => result,
    }
}

fn period_output(
    values: Vec<BigInt>,
    meter: &mut NormalFormRequestMeter<'_, '_>,
) -> PyResult<Vec<PyExpr>> {
    meter.checkpoint().map_err(to_value_error)?;
    if values.len() > 64 {
        return Err(PyValueError::new_err("solve_mod: too many finite periods"));
    }
    let mut steps = 1;
    // Reserve the Rust vector, Python list/scalar owners, both Python matrix
    // owners and the enclosing result tuple before the PyO3 conversion.
    // Matrix coefficient storage is charged by the existing output lifters.
    let mut bytes = (values.len() * (std::mem::size_of::<PyExpr>() + 128) + 512) as u64;
    for value in &values {
        if !value.is_positive() || value.is_one() || value.bits() > 4096 {
            return Err(PyValueError::new_err("solve_mod: invalid finite period"));
        }
        let limbs = value.limb_count().max(1);
        steps += limbs;
        bytes += limbs * 8;
    }
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, steps),
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, values.len() as u64 + 5),
        ])
        .map_err(to_value_error)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| PyValueError::new_err("solve_mod: period output allocation refused"))?;
    for value in values {
        meter.checkpoint().map_err(to_value_error)?;
        output.push(PyExpr::from_expr(Expr::Integer(value)));
    }
    Ok(output)
}

fn dense_result(
    outcome: IntegerModularSystemOutcome,
    meter: &mut NormalFormRequestMeter<'_, '_>,
) -> PyResult<Option<DenseModularParts>> {
    match outcome {
        IntegerModularSystemOutcome::Inconsistent { .. } => Ok(None),
        IntegerModularSystemOutcome::Solvable {
            particular,
            generators,
            periods,
        } => Ok(Some((
            integer_dense_output(particular, meter)?,
            integer_dense_output(generators, meter)?,
            period_output(periods, meter)?,
        ))),
    }
}

fn sparse_result(
    outcome: IntegerModularSystemOutcome,
    meter: &mut NormalFormRequestMeter<'_, '_>,
) -> PyResult<Option<SparseModularParts>> {
    match outcome {
        IntegerModularSystemOutcome::Inconsistent { .. } => Ok(None),
        IntegerModularSystemOutcome::Solvable {
            particular,
            generators,
            periods,
        } => Ok(Some((
            integer_sparse_output(particular, meter)?,
            integer_sparse_output(generators, meter)?,
            period_output(periods, meter)?,
        ))),
    }
}

pub(super) fn solve_dense(
    source: &Matrix,
    rhs: &Bound<'_, PyAny>,
    original_modulus: &Bound<'_, PyAny>,
    py: Python<'_>,
    session: Option<&PySupervisedCallbackSession>,
) -> PyResult<Option<DenseModularParts>> {
    if rhs.is_exact_instance_of::<PyMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PyMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            original_modulus,
            py,
            session,
            dense_result,
        )
    } else if rhs.is_exact_instance_of::<PySparseMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PySparseMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            original_modulus,
            py,
            session,
            dense_result,
        )
    } else {
        Err(PyTypeError::new_err(
            "solve_mod requires an exact native dense or sparse RHS",
        ))
    }
}

pub(super) fn solve_sparse(
    source: &SparseMatrix,
    rhs: &Bound<'_, PyAny>,
    original_modulus: &Bound<'_, PyAny>,
    py: Python<'_>,
    session: Option<&PySupervisedCallbackSession>,
) -> PyResult<Option<SparseModularParts>> {
    if rhs.is_exact_instance_of::<PyMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PyMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            original_modulus,
            py,
            session,
            sparse_result,
        )
    } else if rhs.is_exact_instance_of::<PySparseMatrix>() {
        let rhs = rhs.extract::<PyRef<'_, PySparseMatrix>>()?;
        checked_output(
            source.into(),
            (&rhs.inner).into(),
            original_modulus,
            py,
            session,
            sparse_result,
        )
    } else {
        Err(PyTypeError::new_err(
            "solve_mod requires an exact native dense or sparse RHS",
        ))
    }
}
