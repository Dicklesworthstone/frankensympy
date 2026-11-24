//! Typed exact-input bridge for certified proper real quadrature.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric::exact_rational;
use crate::numeric_budget::{
    PyEvaluationBudget, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::quadrature::{QuadratureOptions, propose_integral_enclosure};
use fsym_core::{
    BigRational, IntegralEnclosureReceipt, MAX_QUADRATURE_INTERVALS, verify_integral_enclosure,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

pub(crate) mod artifact;

fn number(value: &BigRational) -> PyExpr {
    PyExpr::from_expr(crate::rational_to_expr(value.clone()))
}

/// Acceptance is only produced by the independent core quadrature checker.
#[pyclass(name = "IntegralEnclosureReceipt", module = "fsym_python", frozen)]
pub struct PyIntegralEnclosureReceipt {
    receipt: IntegralEnclosureReceipt,
}

#[pymethods]
impl PyIntegralEnclosureReceipt {
    #[getter]
    fn integrand(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.integrand().clone())
    }

    #[getter]
    fn variable(&self) -> PySymbolBinding {
        PySymbolBinding::from_symbol(self.receipt.variable())
    }

    #[getter]
    fn interval(&self) -> (PyExpr, PyExpr) {
        (number(self.receipt.lower()), number(self.receipt.upper()))
    }

    #[getter]
    fn bounds(&self) -> (PyExpr, PyExpr) {
        (
            number(&self.receipt.enclosure().lower()),
            number(&self.receipt.enclosure().upper()),
        )
    }

    #[getter]
    fn midpoint(&self) -> PyExpr {
        number(self.receipt.enclosure().midpoint())
    }

    #[getter]
    fn error_bound(&self) -> PyExpr {
        number(self.receipt.enclosure().radius())
    }

    #[getter]
    fn absolute_tolerance(&self) -> PyExpr {
        number(self.receipt.absolute_tolerance())
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.witness().precision_digits
    }

    #[getter]
    fn partition(&self) -> Vec<PyExpr> {
        self.receipt
            .witness()
            .partition
            .iter()
            .map(number)
            .collect()
    }

    #[getter]
    fn interval_count(&self) -> usize {
        self.receipt.interval_count()
    }

    /// Export a canonical source-bound partition for independent replay.
    #[pyo3(signature = (*, budget=None))]
    fn to_bytes<'py>(
        &self,
        py: Python<'py>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        artifact::export(py, &self.receipt, budget.as_deref())
    }
}

#[pyfunction]
#[pyo3(signature = (source, variable, lower, upper, absolute_tolerance, precision_digits=32, max_intervals=512, *, budget=None))]
pub fn certified_integral_native(
    source: &Bound<'_, PyAny>,
    variable: &PySymbolBinding,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    absolute_tolerance: &Bound<'_, PyAny>,
    precision_digits: u32,
    max_intervals: usize,
    budget: Option<PyRef<'_, PyEvaluationBudget>>,
) -> PyResult<PyIntegralEnclosureReceipt> {
    let py = source.py();
    // Capture one private ledger before preparing inputs when the direct
    // native caller omits a budget. Proposal and verification share this
    // ledger; neither phase receives a refreshed fallback allowance.
    let fallback = budget
        .is_none()
        .then(PyEvaluationBudget::certification_default);
    let budget = budget.as_deref().or(fallback.as_ref());
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    if !source.is_exact_instance(&py.get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "certified integral source requires an exact native expression",
        ));
    }
    if !(8..=128).contains(&precision_digits) {
        return Err(PyValueError::new_err(
            "certified precision must be an integer from 8 through 128",
        ));
    }
    if !(1..=MAX_QUADRATURE_INTERVALS).contains(&max_intervals) {
        return Err(PyValueError::new_err(
            "max_intervals must be an integer from 1 through 1024",
        ));
    }
    let source_ref = source.extract::<PyRef<'_, PyExpr>>()?;
    let source = &source_ref.inner;
    let lower = exact_rational(lower, budget)?;
    let upper = exact_rational(upper, budget)?;
    let tolerance = exact_rational(absolute_tolerance, budget)?;
    let variable = variable.to_symbol();
    let witness = with_certification_meter(py, budget, |meter| {
        propose_integral_enclosure(
            source,
            &variable,
            &lower,
            &upper,
            &tolerance,
            QuadratureOptions {
                precision_digits,
                max_intervals,
            },
            meter,
        )
    })?;
    // The independent checker re-evaluates the original source and every
    // cell. Its intrinsic work limits remain separate, while all work is
    // also charged to the same cumulative caller ledger. Both arithmetic
    // phases release the GIL so another Python thread can request cancellation.
    let receipt = with_certification_meter(py, budget, |meter| {
        verify_integral_enclosure(
            source, &variable, &lower, &upper, &tolerance, &witness, meter,
        )
    })?;
    let result = PyIntegralEnclosureReceipt { receipt };
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}
