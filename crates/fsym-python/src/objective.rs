//! Typed Python access to independently checked sparse second derivatives.

#![forbid(unsafe_code)]

use crate::expr::PyExpr;
use crate::numeric::{check_source, extract_variables, input_values};
use fsym_calculus::VerifiedObjective;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Checked symbolic gradient and Hessian with finite binary64 evaluation.
#[pyclass(name = "VerifiedObjective", module = "fsym_python", frozen)]
pub struct PyVerifiedObjective {
    objective: VerifiedObjective,
}

#[pymethods]
impl PyVerifiedObjective {
    #[new]
    fn new(source: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !source.is_exact_instance(&source.py().get_type::<PyExpr>()) {
            return Err(PyTypeError::new_err(
                "objective must be an exact native Expr object",
            ));
        }
        let source = source.extract::<PyRef<'_, PyExpr>>()?;
        let variables = extract_variables(variables)?;
        let mut nodes = 0;
        check_source(&source.inner, &mut nodes)?;
        let objective = VerifiedObjective::try_compile(&source.inner, &variables)
            .map_err(crate::to_value_error)?;
        Ok(Self { objective })
    }

    /// Return value, gradient and ordered sparse Hessian values atomically.
    fn evaluate(&self, values: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
        let values = input_values(values, self.objective.variables().len())?;
        let result = self
            .objective
            .evaluate(&values)
            .map_err(crate::to_value_error)?;
        Ok((result.value, result.gradient, result.hessian))
    }

    /// Multiply by the Hessian without allocating its dense representation.
    fn hessian_vector_product(
        &self,
        values: &Bound<'_, PyAny>,
        direction: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<f64>> {
        let size = self.objective.variables().len();
        let values = input_values(values, size)?;
        let direction = input_values(direction, size)?;
        self.objective
            .hessian_vector_product(&values, &direction)
            .map_err(crate::to_value_error)
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.objective.variables().len()
    }

    #[getter]
    fn hessian_pattern(&self) -> Vec<(usize, usize)> {
        self.objective.hessian_pattern().to_vec()
    }

    #[getter]
    fn hessian_nonzeros(&self) -> usize {
        self.objective.receipt().hessian_nonzeros
    }

    /// Formal derivative proofs; this does not certify numerical rounding.
    #[getter]
    fn proofs_verified(&self) -> usize {
        let receipt = self.objective.receipt();
        receipt.gradient_proofs_verified + receipt.hessian_proofs_verified
    }

    #[getter]
    fn zero_entries_checked(&self) -> usize {
        self.objective.receipt().zero_entries_checked
    }
}
