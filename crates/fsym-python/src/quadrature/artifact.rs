//! Proper-real integral witnesses bound to the receiver's original statement.
//!
//! The shared core codec accepts only canonical bounded partitions and reruns
//! independent quadrature. No stored enclosure or acceptance flag is trusted.

use super::PyIntegralEnclosureReceipt;
use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric::exact_rational;
use crate::numeric_budget::{
    PyEvaluationBudget, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::BudgetMeter;
use fsym_core::IntegralEnclosureReceipt;
use fsym_core::root_certificate::artifact::charge;
use fsym_core::root_certificate::integral_artifact::{self, IntegrationError};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::LazyLock;

// Typed symbols retain their declared facts. This public API supplies no
// additional ambient assumptions, matching a default native Session.
static CONTEXT: LazyLock<[u8; 32]> =
    LazyLock::new(|| ImmutableAssumptionsSnapshot::empty().digest());

pub(super) fn export<'py>(
    py: Python<'py>,
    receipt: &IntegralEnclosureReceipt,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Bound<'py, PyBytes>> {
    // Keep serialization, the Python-owned byte copy and publication on one
    // ledger even when a direct native caller omits the optional budget.
    let fallback = budget
        .is_none()
        .then(PyEvaluationBudget::certification_default);
    let budget = budget.or(fallback.as_ref());
    let encoded = with_certification_meter(py, budget, |meter| {
        let encoded = integral_artifact::export_integral_certificate(
            std::slice::from_ref(receipt.integrand()),
            std::slice::from_ref(receipt.variable()),
            receipt,
            *CONTEXT,
            meter,
        )?;
        charge(meter, encoded.len() as u64 + 1, encoded.len() as u64, 1)?;
        Ok::<_, IntegrationError>(encoded)
    })?;
    let result = PyBytes::new(py, &encoded);
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}

/// Independently replay a partition against exact receiver-owned inputs.
#[pyfunction]
#[pyo3(signature = (data, source, variable, lower, upper, absolute_tolerance, *, budget=None))]
pub fn replay_integral_certificate_native(
    data: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    variable: &PySymbolBinding,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    absolute_tolerance: &Bound<'_, PyAny>,
    budget: Option<PyRef<'_, PyEvaluationBudget>>,
) -> PyResult<PyIntegralEnclosureReceipt> {
    let py = data.py();
    let fallback = budget
        .is_none()
        .then(PyEvaluationBudget::certification_default);
    let budget = budget.as_deref().or(fallback.as_ref());
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    // Refuse foreign byte-like objects and oversized inputs before touching
    // the expected source or exact endpoint representations.
    if !data.is_exact_instance(&py.get_type::<PyBytes>()) {
        return Err(PyTypeError::new_err(
            "integral certificate data must be exact built-in bytes",
        ));
    }
    let encoded = data.cast::<PyBytes>()?.as_bytes();
    if encoded.len() > integral_artifact::MAX_INTEGRAL_CERTIFICATE_BYTES {
        return Err(PyValueError::new_err(
            "integral certificate data exceeds the 1 MiB byte limit",
        ));
    }
    if !source.is_exact_instance(&py.get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "certified integral source requires an exact native expression",
        ));
    }
    let source_ref = source.extract::<PyRef<'_, PyExpr>>()?;
    let source = &source_ref.inner;
    let lower = exact_rational(lower, budget)?;
    let upper = exact_rational(upper, budget)?;
    let tolerance = exact_rational(absolute_tolerance, budget)?;
    let variable = variable.to_symbol();
    let receipt = with_certification_meter(py, budget, |meter| {
        integral_artifact::replay_integral_certificate(
            encoded,
            std::slice::from_ref(source),
            std::slice::from_ref(&variable),
            &lower,
            &upper,
            &tolerance,
            *CONTEXT,
            meter,
        )
    })?;
    let result = PyIntegralEnclosureReceipt { receipt };
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}
