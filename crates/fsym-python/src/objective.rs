//! Typed Python access to independently checked sparse second derivatives.

#![forbid(unsafe_code)]

use crate::expr::PyExpr;
use crate::numeric::{check_source, extract_variables, input_values_metered};
use crate::numeric_budget::{PyEvaluationBudget, with_evaluation_meter};
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
    #[pyo3(signature = (values, *, budget=None))]
    fn evaluate(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
        let inputs =
            input_values_metered(values, self.objective.variables().len(), budget.as_deref())?;
        let result = with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            self.objective.evaluate_with_meter(&inputs, meter)
        })?;
        Ok((result.value, result.gradient, result.hessian))
    }

    /// Multiply by the Hessian without allocating its dense representation.
    #[pyo3(signature = (values, direction, *, budget=None))]
    fn hessian_vector_product(
        &self,
        values: &Bound<'_, PyAny>,
        direction: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<f64>> {
        let size = self.objective.variables().len();
        let inputs = input_values_metered(values, size, budget.as_deref())?;
        let direction = input_values_metered(direction, size, budget.as_deref())?;
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            self.objective
                .hessian_vector_product_with_meter(&inputs, &direction, meter)
        })
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
