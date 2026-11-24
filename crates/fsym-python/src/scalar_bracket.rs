//! Typed source and exact-bound bridge for scalar root existence certificates.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric_budget::{
    PyEvaluationBudget, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::scalar_bracket::{ScalarRootBracketOptions, propose_scalar_root_bracket};
use fsym_core::{
    BigRational, Expr, MAX_SCALAR_ROOT_INPUT_BITS, ScalarRootBracketReceipt,
    verify_scalar_root_bracket,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

fn literal(value: &Bound<'_, PyAny>) -> PyResult<BigRational> {
    if !value.is_exact_instance(&value.py().get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "certified root bounds require exact native rational expressions",
        ));
    }
    let value = value.extract::<PyRef<'_, PyExpr>>()?;
    match &value.inner {
        Expr::Integer(n) if n.bits() <= MAX_SCALAR_ROOT_INPUT_BITS => {
            Ok(BigRational::from_integer(n.clone()))
        }
        Expr::Rational(q)
            if q.numer().bits().max(q.denom().bits()) <= MAX_SCALAR_ROOT_INPUT_BITS =>
        {
            Ok(q.clone())
        }
        _ => Err(PyValueError::new_err(
            "certified root bounds and tolerance require rational literals of at most 4096 bits",
        )),
    }
}

fn number(value: &BigRational) -> PyExpr {
    PyExpr::from_expr(crate::rational_to_expr(value.clone()))
}

type EndpointValueBounds = ((PyExpr, PyExpr), (PyExpr, PyExpr));

/// Only the independent core verifier can construct accepted native evidence.
#[pyclass(name = "ScalarRootBracketReceipt", module = "fsym_python", frozen)]
pub struct PyScalarRootBracketReceipt {
    receipt: ScalarRootBracketReceipt,
}

#[pymethods]
impl PyScalarRootBracketReceipt {
    #[getter]
    fn expression(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.expression().clone())
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
            number(&self.receipt.root_enclosure().lower()),
            number(&self.receipt.root_enclosure().upper()),
        )
    }

    #[getter]
    fn midpoint(&self) -> PyExpr {
        number(self.receipt.root_enclosure().midpoint())
    }

    #[getter]
    fn error_bound(&self) -> PyExpr {
        number(self.receipt.root_enclosure().radius())
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
    fn endpoint_values(&self) -> EndpointValueBounds {
        let [left, right] = self.receipt.endpoint_values();
        (
            (number(&left.lower()), number(&left.upper())),
            (number(&right.lower()), number(&right.upper())),
        )
    }
}

#[pyfunction]
#[pyo3(signature = (source, variable, lower, upper, absolute_tolerance, precision_digits=32, max_bisections=128, *, budget=None, verification_budget=None))]
#[allow(clippy::too_many_arguments)]
pub fn certified_scalar_root_bracket_native(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    variable: &PySymbolBinding,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    absolute_tolerance: &Bound<'_, PyAny>,
    precision_digits: u32,
    max_bisections: usize,
    budget: Option<PyRef<'_, PyEvaluationBudget>>,
    verification_budget: Option<PyRef<'_, PyEvaluationBudget>>,
) -> PyResult<PyScalarRootBracketReceipt> {
    let generator_budget = budget.as_deref();
    let checker_budget = verification_budget.as_deref();
    with_evaluation_meter_attached(generator_budget, |meter| meter.checkpoint())?;
    with_evaluation_meter_attached(checker_budget, |meter| meter.checkpoint())?;
    if !source.is_exact_instance(&py.get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "certified root source requires an exact native expression",
        ));
    }
    let source = source.extract::<PyRef<'_, PyExpr>>()?;
    let original = &source.inner;
    let symbol = variable.to_symbol();
    let lower = literal(lower)?;
    let upper = literal(upper)?;
    let tolerance = literal(absolute_tolerance)?;
    // These sequential phases borrow distinct optional caller ledgers. Passing
    // the same object explicitly shares one cumulative allowance; omitting
    // them keeps independent bounded default ledgers. The checker consumes
    // the original source and witness, never cached generator evaluations.
    let witness = with_certification_meter(py, generator_budget, |meter| {
        propose_scalar_root_bracket(
            original,
            &symbol,
            &lower,
            &upper,
            &tolerance,
            ScalarRootBracketOptions {
                precision_digits,
                max_bisections,
            },
            meter,
        )
    })?;
    let receipt = with_certification_meter(py, checker_budget, |meter| {
        verify_scalar_root_bracket(
            original, &symbol, &lower, &upper, &tolerance, &witness, meter,
        )
    })?;
    // Recheck both phases after GIL reentry, before publishing evidence.
    with_evaluation_meter_attached(generator_budget, |meter| meter.checkpoint())?;
    with_evaluation_meter_attached(checker_budget, |meter| meter.checkpoint())?;
    Ok(PyScalarRootBracketReceipt { receipt })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyScalarRootBracketReceipt>()?;
    module.add_function(wrap_pyfunction!(
        certified_scalar_root_bracket_native,
        module
    )?)?;
    Ok(())
}
