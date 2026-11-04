//! Typed Python access to independently checked sparse second derivatives.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric::{
    ball_bounds, check_source, exact_box, exact_vector, extract_variables, input_values_metered,
    rational_expr,
};
use crate::numeric_budget::{
    PyEvaluationBudget, evaluation_buffer, with_certification_meter, with_evaluation_meter,
    with_evaluation_meter_attached,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::VerifiedObjective;
use fsym_core::{
    BigRational, ObjectiveEnclosure, RootBoxWitness, StationaryPointKind, StationaryPointReceipt,
    metered_enclose_objective, verify_stationary_point,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

fn exact_preconditioner(
    values: &Bound<'_, PyAny>,
    dimension: usize,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Vec<Vec<BigRational>>> {
    let py = values.py();
    if !values.is_exact_instance(&py.get_type::<PyList>())
        && !values.is_exact_instance(&py.get_type::<PyTuple>())
    {
        return Err(PyTypeError::new_err(
            "preconditioner must be an exact built-in list or tuple",
        ));
    }
    if dimension > 16 || values.len()? != dimension {
        return Err(PyValueError::new_err(
            "preconditioner has the wrong dimension",
        ));
    }
    let mut matrix =
        with_evaluation_meter_attached(budget, |meter| evaluation_buffer(dimension, meter))?;
    for index in 0..dimension {
        matrix.push(exact_vector(&values.get_item(index)?, dimension, budget)?);
    }
    Ok(matrix)
}

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

    /// The immutable original typed objective, before symbolic differentiation.
    #[getter]
    fn source(&self) -> PyExpr {
        PyExpr::from_expr(self.objective.objective().clone())
    }

    /// Independently enclose the original source and both derivative orders.
    #[pyo3(signature = (domain, precision_digits=32, *, budget=None))]
    fn enclose(
        &self,
        domain: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyObjectiveEnclosure> {
        let py = domain.py();
        let domain = exact_box(domain, self.objective.variables().len(), budget.as_deref())?;
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            metered_enclose_objective(
                self.objective.objective(),
                self.objective.variables(),
                &domain,
                precision_digits,
                meter,
            )
        })?;
        let result = PyObjectiveEnclosure { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Check an untrusted stationary-point witness against the original source.
    #[pyo3(signature = (domain, center, preconditioner, precision_digits=32, *, budget=None))]
    fn certify_stationary_point(
        &self,
        domain: &Bound<'_, PyAny>,
        center: &Bound<'_, PyAny>,
        preconditioner: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyStationaryPointReceipt> {
        let dimension = self.objective.variables().len();
        let py = domain.py();
        let domain = exact_box(domain, dimension, budget.as_deref())?;
        let center = exact_vector(center, dimension, budget.as_deref())?;
        let preconditioner = exact_preconditioner(preconditioner, dimension, budget.as_deref())?;
        let witness = RootBoxWitness {
            domain,
            center,
            preconditioner,
            precision_digits,
        };
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            verify_stationary_point(
                self.objective.objective(),
                self.objective.variables(),
                &witness,
                meter,
            )
        })?;
        let result = PyStationaryPointReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
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

/// Accepted second-order source enclosures, with no public constructor.
#[pyclass(name = "ObjectiveEnclosure", module = "fsym_python", frozen)]
pub struct PyObjectiveEnclosure {
    receipt: ObjectiveEnclosure,
}

#[pymethods]
impl PyObjectiveEnclosure {
    #[getter]
    fn objective(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.objective().clone())
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.domain())
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.precision_digits()
    }

    #[getter]
    fn value(&self) -> (PyExpr, PyExpr) {
        (
            rational_expr(&self.receipt.value().lower()),
            rational_expr(&self.receipt.value().upper()),
        )
    }

    #[getter]
    fn gradient(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.gradient())
    }

    #[getter]
    fn hessian(&self) -> Vec<Vec<(PyExpr, PyExpr)>> {
        self.receipt
            .hessian()
            .iter()
            .map(|row| ball_bounds(row))
            .collect()
    }
}

/// Accepted local stationary-point evidence, with no public constructor.
#[pyclass(name = "StationaryPointReceipt", module = "fsym_python", frozen)]
pub struct PyStationaryPointReceipt {
    receipt: StationaryPointReceipt,
}

#[pymethods]
impl PyStationaryPointReceipt {
    #[getter]
    fn objective(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.objective().clone())
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(&self.receipt.witness().domain)
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.witness().precision_digits
    }

    #[getter]
    fn center(&self) -> Vec<PyExpr> {
        self.receipt
            .witness()
            .center
            .iter()
            .map(rational_expr)
            .collect()
    }

    #[getter]
    fn preconditioner(&self) -> Vec<Vec<PyExpr>> {
        self.receipt
            .witness()
            .preconditioner
            .iter()
            .map(|row| row.iter().map(rational_expr).collect())
            .collect()
    }

    #[getter]
    fn stationary_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.stationary_enclosure())
    }

    #[getter]
    fn objective_enclosure(&self) -> (PyExpr, PyExpr) {
        (
            rational_expr(&self.receipt.objective_enclosure().lower()),
            rational_expr(&self.receipt.objective_enclosure().upper()),
        )
    }

    #[getter]
    fn gradient_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.gradient_enclosure())
    }

    #[getter]
    fn hessian_enclosure(&self) -> Vec<Vec<(PyExpr, PyExpr)>> {
        self.receipt
            .hessian_enclosure()
            .iter()
            .map(|row| ball_bounds(row))
            .collect()
    }

    #[getter]
    fn contraction_bound(&self) -> PyExpr {
        rational_expr(self.receipt.contraction_bound())
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.receipt.kind() {
            StationaryPointKind::StrictMinimum => "strict_minimum",
            StationaryPointKind::StrictMaximum => "strict_maximum",
            StationaryPointKind::Saddle => "saddle",
            StationaryPointKind::Unclassified => "unclassified",
        }
    }
}
