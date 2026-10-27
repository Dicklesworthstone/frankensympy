//! Typed exact-input bridge for certified proper real quadrature.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use fsym_calculus::quadrature::{QuadratureOptions, metered_certified_integral};
use fsym_core::{BigRational, Expr, IntegralEnclosureReceipt, MAX_QUADRATURE_INPUT_BITS};
use fsym_runtime::{Budget, BudgetLimits};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

fn literal(value: &Bound<'_, PyAny>) -> PyResult<BigRational> {
    if !value.is_exact_instance(&value.py().get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "certified integral bounds require exact native rational expressions",
        ));
    }
    let value = value.extract::<PyRef<'_, PyExpr>>()?;
    match &value.inner {
        Expr::Integer(n) if n.bits() <= MAX_QUADRATURE_INPUT_BITS => {
            Ok(BigRational::from_integer(n.clone()))
        }
        Expr::Rational(q)
            if q.numer().bits().max(q.denom().bits()) <= MAX_QUADRATURE_INPUT_BITS =>
        {
            Ok(q.clone())
        }
        _ => Err(PyValueError::new_err(
            "certified integral bounds and tolerance require rational literals of at most 4096 bits",
        )),
    }
}

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
}

#[pyfunction]
#[pyo3(signature = (source, variable, lower, upper, absolute_tolerance, precision_digits=32, max_intervals=512))]
pub fn certified_integral_native(
    source: &Bound<'_, PyAny>,
    variable: &PySymbolBinding,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    absolute_tolerance: &Bound<'_, PyAny>,
    precision_digits: u32,
    max_intervals: usize,
) -> PyResult<PyIntegralEnclosureReceipt> {
    if !source.is_exact_instance(&source.py().get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "certified integral source requires an exact native expression",
        ));
    }
    let source = source.extract::<PyRef<'_, PyExpr>>()?;
    let lower = literal(lower)?;
    let upper = literal(upper)?;
    let tolerance = literal(absolute_tolerance)?;
    // Separate bounded synchronous ledgers: candidate refinement cannot
    // spend the checker's allowance. This does not assert async supervision.
    let mut generator_meter = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let mut verifier_meter = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let receipt = metered_certified_integral(
        &source.inner,
        &variable.to_symbol(),
        &lower,
        &upper,
        &tolerance,
        QuadratureOptions {
            precision_digits,
            max_intervals,
        },
        &mut generator_meter,
        &mut verifier_meter,
    )
    .map_err(crate::to_value_error)?;
    Ok(PyIntegralEnclosureReceipt { receipt })
}
