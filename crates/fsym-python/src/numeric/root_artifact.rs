//! Bounded root-proof transport for existing native Python numerical objects.
//!
//! The shared core codec checks a receiver-owned source and reruns the original
//! interval verifier. No accepted flag, enclosure, Python object or compiled
//! evaluator is deserialized as mathematical evidence.

use super::PyRootBoxReceipt;
use crate::numeric_budget::{
    PyEvaluationBudget, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::BudgetMeter;
use fsym_core::root_certificate::artifact::{self, RootCertificateSources};
use fsym_core::{Expr, RootBoxReceipt, Symbol};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::LazyLock;

// Python's captured symbol identities retain their declared facts. These APIs
// introduce no additional ambient assumptions, matching a default Session.
static CONTEXT: LazyLock<[u8; 32]> =
    LazyLock::new(|| ImmutableAssumptionsSnapshot::empty().digest());

pub(super) fn export<'py>(
    py: Python<'py>,
    receipt: &RootBoxReceipt,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = with_certification_meter(py, budget, |meter| {
        let encoded = artifact::export_root_certificate(receipt, *CONTEXT, meter)?;
        // PyBytes owns another copy after the GIL is reacquired. Reserve it
        // on this same ledger, including when it is the one private fallback.
        artifact::charge(meter, encoded.len() as u64 + 1, encoded.len() as u64, 1)?;
        Ok::<_, artifact::CertificationError>(encoded)
    })?;
    let result = PyBytes::new(py, &encoded);
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}

pub(super) fn replay(
    data: &Bound<'_, PyAny>,
    residuals: &[Expr],
    variables: &[Symbol],
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<PyRootBoxReceipt> {
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    let py = data.py();
    if !data.is_exact_instance(&py.get_type::<PyBytes>()) {
        return Err(PyTypeError::new_err(
            "root certificate data must be exact built-in bytes",
        ));
    }
    let encoded = data.cast::<PyBytes>()?.as_bytes();
    if encoded.len() > artifact::MAX_ROOT_CERTIFICATE_BYTES {
        return Err(PyValueError::new_err(
            "root certificate data exceeds the 1 MiB byte limit",
        ));
    }
    let receipt = with_certification_meter(py, budget, |meter| {
        let source = RootCertificateSources::copy_from(residuals, variables, meter)?;
        artifact::replay_root_certificate(encoded, &source, *CONTEXT, meter)
    })?;
    let result = PyRootBoxReceipt { receipt };
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}
