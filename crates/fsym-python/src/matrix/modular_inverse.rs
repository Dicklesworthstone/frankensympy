//! Source-bound modular matrix inversion with cumulative request accounting.

use super::{
    NormalFormRequestMeter, PyMatrix, PySupervisedCallbackSession, integer_dense_output, matrix_err,
};
use crate::expr::{PyExpr, exact_python_integer};
use crate::to_value_error;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigInt, Expr};
use fsym_matrices::{IntegerMatrixRef, metered_inverse_mod};
use fsym_proof_kernel::integer_modular_inverse::IntegerModularInverseOutcome;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyInt;

fn modulus(
    value: &Bound<'_, PyAny>,
    meter: &mut NormalFormRequestMeter<'_, '_>,
) -> PyResult<BigInt> {
    meter.checkpoint().map_err(to_value_error)?;
    if value.is_exact_instance_of::<PyExpr>() {
        let atom = value.extract::<PyRef<'_, PyExpr>>()?;
        let integer = match &atom.inner {
            Expr::Integer(integer) => integer,
            Expr::Rational(rational) if rational.is_integer() => rational.numer(),
            _ => {
                return Err(PyTypeError::new_err(
                    "inv_mod: modulus m must be an integer",
                ));
            }
        };
        if integer.bits() > 4096 {
            return Err(PyValueError::new_err("inv_mod: modulus exceeds 4096 bits"));
        }
        let limbs = integer.limb_count().max(1);
        meter
            .charge_batch(&[
                (Dimension::ComputeSteps, limbs + 1),
                (Dimension::MemoryBytes, limbs * 8),
                (Dimension::AllocationCount, 1),
            ])
            .map_err(to_value_error)?;
        return Ok(integer.clone());
    }
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "inv_mod: modulus m must be an integer",
        ));
    }
    let bits = value.call_method0("bit_length")?.extract::<u64>()?;
    if bits > 4096 {
        return Err(PyValueError::new_err("inv_mod: modulus exceeds 4096 bits"));
    }
    let limbs = bits.div_ceil(64).max(1);
    // Cover the exact built-in-int transport before its byte conversion.
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, limbs + 1),
            (Dimension::MemoryBytes, limbs * 16 + 256),
            (Dimension::AllocationCount, 4),
        ])
        .map_err(to_value_error)?;
    exact_python_integer(value, "modulus")
}

pub(super) fn inverse(
    source: IntegerMatrixRef<'_>,
    original_modulus: &Bound<'_, PyAny>,
    py: Python<'_>,
    session: Option<&PySupervisedCallbackSession>,
) -> PyResult<Option<PyMatrix>> {
    let mut generator = NormalFormRequestMeter::new(py, session);
    let mut verifier = NormalFormRequestMeter::new(py, session);
    let result = (|| {
        let modulus = modulus(original_modulus, &mut generator)?;
        let accepted = metered_inverse_mod(source, &modulus, &mut generator, &mut verifier)
            .map_err(matrix_err)?;
        let (_, witness, _) = accepted.into_parts();
        let result = match witness.outcome {
            IntegerModularInverseOutcome::Invertible { inverse } => {
                Some(integer_dense_output(inverse, &mut generator)?)
            }
            IntegerModularInverseOutcome::NonUnit { .. } => None,
        };
        generator
            .charge(Dimension::ComputeSteps, 1)
            .map_err(to_value_error)?;
        generator.checkpoint().map_err(to_value_error)?;
        verifier.checkpoint().map_err(to_value_error)?;
        Ok(result)
    })();
    // Budget, ownership and cancellation errors are operational refusals.
    // They can never be relabelled as mathematical noninvertibility.
    match generator.request.failure.or(verifier.request.failure) {
        Some(error) => Err(error),
        None => result,
    }
}
