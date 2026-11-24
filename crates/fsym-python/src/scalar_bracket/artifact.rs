//! Portable root-existence brackets bound to receiver-owned exact statements.
//!
//! Canonical bytes carry no acceptance, uniqueness or cached endpoint values.
//! Only the independent core continuity and sign checker grants a receipt.

use super::{PyScalarRootBracketReceipt, literal};
use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric_budget::{
    PyEvaluationBudget, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::BudgetMeter;
use fsym_core::ScalarRootBracketReceipt;
use fsym_core::root_certificate::artifact::charge;
use fsym_core::root_certificate::scalar_bracket_artifact::{self, ScalarBracketError};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::LazyLock;

// The public API supplies no ambient assumptions. Declared facts remain bound
// through typed symbols, exactly as in a default native Session.
static CONTEXT: LazyLock<[u8; 32]> =
    LazyLock::new(|| ImmutableAssumptionsSnapshot::empty().digest());

pub(super) fn export<'py>(
    py: Python<'py>,
    receipt: &ScalarRootBracketReceipt,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Bound<'py, PyBytes>> {
    let fallback = budget
        .is_none()
        .then(PyEvaluationBudget::certification_default);
    let budget = budget.or(fallback.as_ref());
    let encoded = with_certification_meter(py, budget, |meter| {
        let encoded = scalar_bracket_artifact::export_scalar_root_bracket(
            std::slice::from_ref(receipt.expression()),
            std::slice::from_ref(receipt.variable()),
            receipt,
            *CONTEXT,
            meter,
        )?;
        // The Python-owned byte copy and its final publication use this same
        // operation's ledger, including when a native caller omitted it.
        charge(meter, encoded.len() as u64 + 1, encoded.len() as u64, 1)?;
        Ok::<_, ScalarBracketError>(encoded)
    })?;
    let result = PyBytes::new(py, &encoded);
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}

/// Independently replay a scalar bracket without running candidate search.
#[pyfunction]
#[pyo3(signature = (data, source, variable, lower, upper, absolute_tolerance, *, budget=None))]
pub fn replay_scalar_root_bracket_native(
    data: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    variable: &PySymbolBinding,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
    absolute_tolerance: &Bound<'_, PyAny>,
    budget: Option<PyRef<'_, PyEvaluationBudget>>,
) -> PyResult<PyScalarRootBracketReceipt> {
    let py = data.py();
    let fallback = budget
        .is_none()
        .then(PyEvaluationBudget::certification_default);
    let budget = budget.as_deref().or(fallback.as_ref());
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    // Admit exact bounded bytes before touching the receiver's mathematical
    // source or exact rational representations. Foreign hooks never run.
    if !data.is_exact_instance(&py.get_type::<PyBytes>()) {
        return Err(PyTypeError::new_err(
            "scalar root bracket data must be exact built-in bytes",
        ));
    }
    let encoded = data.cast::<PyBytes>()?.as_bytes();
    if encoded.len() > scalar_bracket_artifact::MAX_SCALAR_ROOT_BRACKET_BYTES {
        return Err(PyValueError::new_err(
            "scalar root bracket data exceeds the 1 MiB byte limit",
        ));
    }
    if !source.is_exact_instance(&py.get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "certified root source requires an exact native expression",
        ));
    }
    let source_ref = source.extract::<PyRef<'_, PyExpr>>()?;
    let source = &source_ref.inner;
    let lower = literal(lower, budget)?;
    let upper = literal(upper, budget)?;
    let tolerance = literal(absolute_tolerance, budget)?;
    let variable = variable.to_symbol();
    let receipt = with_certification_meter(py, budget, |meter| {
        scalar_bracket_artifact::replay_scalar_root_bracket(
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
    let result = PyScalarRootBracketReceipt { receipt };
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}
