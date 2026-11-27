//! Source-bound approximate constrained fitting for the Python boundary.

use super::{
    PyEvaluationBudget, PyVerifiedResidualSystem, charge_input_storage, container_length,
    input_values_metered, with_evaluation_meter, with_evaluation_meter_attached,
};
use fsym_budget::BudgetMeter;
use fsym_numeric_consumer::{
    BoundedLeastSquaresProposal, LeastSquaresOptions, MAX_SPARSE_LEAST_SQUARES_RESIDUALS,
    MAX_SPARSE_LEAST_SQUARES_VARIABLES, sparse_bounded_least_squares_with_meter,
};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt, PyTuple};

/// The original program and actual binary64 bounds accompany the result.
/// Python cannot construct or replace the native proposal's data.
#[pyclass(
    name = "NumericBoundedLeastSquaresProposal",
    module = "fsym_python",
    frozen
)]
pub struct PyNumericBoundedLeastSquaresProposal {
    proposal: BoundedLeastSquaresProposal,
    system: Py<PyVerifiedResidualSystem>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[pymethods]
impl PyNumericBoundedLeastSquaresProposal {
    #[getter]
    fn system(&self, py: Python<'_>) -> Py<PyVerifiedResidualSystem> {
        self.system.clone_ref(py)
    }

    #[getter]
    fn point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.proposal.point.iter().copied())
    }

    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.proposal.residuals.iter().copied())
    }

    #[getter]
    fn lower_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.lower.iter().copied())
    }

    #[getter]
    fn upper_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.upper.iter().copied())
    }

    #[getter]
    fn cost(&self) -> f64 {
        self.proposal.cost
    }

    #[getter]
    fn gradient<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.proposal.gradient.iter().copied())
    }

    #[getter]
    fn projected_gradient<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.proposal.projected_gradient.iter().copied())
    }

    #[getter]
    fn gradient_norm(&self) -> f64 {
        self.proposal.gradient_norm
    }

    #[getter]
    fn projected_gradient_norm(&self) -> f64 {
        self.proposal.projected_gradient_norm
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.proposal.iterations
    }

    #[getter]
    fn evaluations(&self) -> usize {
        self.proposal.evaluations
    }

    #[getter]
    fn termination(&self) -> &'static str {
        "projected_gradient_tolerance"
    }

    #[getter]
    fn backend(&self) -> &'static str {
        "sparse_bounded_lsqr"
    }

    #[getter]
    fn evidence(&self) -> &'static str {
        "approximate"
    }
}

fn endpoints(
    source: &Bound<'_, PyAny>,
    expected: usize,
    lower: bool,
    budget: &PyEvaluationBudget,
) -> PyResult<Vec<f64>> {
    let count = container_length(source, MAX_SPARSE_LEAST_SQUARES_VARIABLES, "bounds")?;
    if count != expected {
        return Err(PyTypeError::new_err(format!(
            "expected {expected} coordinate bounds, got {count}"
        )));
    }
    charge_input_storage(Some(budget), expected, 1, false)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected)
        .map_err(|_| PyMemoryError::new_err("cannot allocate coordinate bounds"))?;
    let py = source.py();
    for index in 0..expected {
        let value = source.get_item(index)?;
        if !value.is_exact_instance(&py.get_type::<PyInt>())
            && !value.is_exact_instance(&py.get_type::<PyFloat>())
        {
            return Err(PyTypeError::new_err(
                "coordinate bounds must be exact built-in int or float values",
            ));
        }
        let value = value.extract::<f64>()?;
        if value.is_nan()
            || (lower && value == f64::INFINITY)
            || (!lower && value == f64::NEG_INFINITY)
        {
            return Err(PyValueError::new_err(
                "bounds allow only finite endpoints, negative-infinite lower bounds and positive-infinite upper bounds",
            ));
        }
        values.push(value);
    }
    with_evaluation_meter_attached(Some(budget), |meter| meter.checkpoint())?;
    Ok(values)
}

pub(super) fn solve<'py>(
    slf: &Bound<'py, PyVerifiedResidualSystem>,
    initial: &Bound<'py, PyAny>,
    lower: &Bound<'py, PyAny>,
    upper: &Bound<'py, PyAny>,
    options: &LeastSquaresOptions,
    supplied_budget: Option<&PyEvaluationBudget>,
) -> PyResult<Py<PyNumericBoundedLeastSquaresProposal>> {
    // One private ledger spans all native preparation and solver phases even
    // for direct bridge calls that omit a public EvaluationBudget.
    let fallback = supplied_budget
        .is_none()
        .then(PyEvaluationBudget::certification_default);
    let budget = supplied_budget
        .or(fallback.as_ref())
        .expect("owned fallback budget");
    with_evaluation_meter_attached(Some(budget), |meter| meter.checkpoint())?;
    let owner = slf.get();
    let n = owner.system.num_vars();
    let m = owner.system.num_residuals();
    if !(1..=MAX_SPARSE_LEAST_SQUARES_VARIABLES).contains(&n)
        || !(1..=MAX_SPARSE_LEAST_SQUARES_RESIDUALS).contains(&m)
    {
        return Err(PyValueError::new_err(
            "bounded sparse least squares requires 1..256 variables and 1..4096 residuals",
        ));
    }
    let initial = input_values_metered(initial, n, Some(budget))?;
    let lower = endpoints(lower, n, true, budget)?;
    let upper = endpoints(upper, n, false, budget)?;
    let py = slf.py();
    let proposal = with_evaluation_meter(py, Some(budget), |meter| {
        sparse_bounded_least_squares_with_meter(
            &owner.system,
            &initial,
            &lower,
            &upper,
            options,
            meter,
        )
    })?;
    let result = Py::new(
        py,
        PyNumericBoundedLeastSquaresProposal {
            proposal,
            system: slf.clone().unbind(),
            lower,
            upper,
        },
    )?;
    with_evaluation_meter_attached(Some(budget), |meter| meter.checkpoint())?;
    Ok(result)
}
