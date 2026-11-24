//! Yielding, source-bound root discovery for Python's captured native systems.
//!
//! A live handle owns its exact search and one strong reference to a cumulative
//! ledger. Progress and restored checkpoints are candidates; only `finish`
//! runs the independent complete-cover checker and returns accepted evidence.

#![forbid(unsafe_code)]

use super::{PyRootIsolationReceipt, ball_bounds, exact_box};
use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric_budget::{
    PyEvaluationBudget, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::BudgetMeter;
use fsym_calculus::root_isolation::checkpoint::{
    MAX_ROOT_SEARCH_CHECKPOINT_BYTES, export_root_search_checkpoint, restore_root_search_checkpoint,
};
use fsym_calculus::root_isolation::{
    MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationOptions, RootIsolationProgress,
    RootIsolationSearch,
};
use fsym_core::root_certificate::artifact::{CertificationError, RootCertificateSources, charge};
use fsym_core::{Expr, MAX_ROOT_ISOLATION_LEAVES, Symbol, verify_root_isolation};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};
use std::sync::LazyLock;

// Declared facts remain in the frozen typed symbol identities. Like the other
// Python certificate bridges, this introduces no ambient assumption context.
static CONTEXT: LazyLock<[u8; 32]> =
    LazyLock::new(|| ImmutableAssumptionsSnapshot::empty().digest());

/// An immutable snapshot of candidate traversal progress, never a root receipt.
#[pyclass(name = "RootIsolationProgress", module = "fsym_python", frozen)]
pub struct PyRootIsolationProgress {
    progress: RootIsolationProgress,
}

#[pymethods]
impl PyRootIsolationProgress {
    #[getter]
    fn processed_cells(&self) -> usize {
        self.progress.processed_cells
    }

    #[getter]
    fn pending_cells(&self) -> usize {
        self.progress.pending_cells
    }

    #[getter]
    fn candidate_root_count(&self) -> usize {
        self.progress.candidate_root_count
    }

    #[getter]
    fn excluded_cells(&self) -> usize {
        self.progress.excluded_cells
    }

    #[getter]
    fn leaf_count(&self) -> usize {
        self.progress.leaf_count
    }

    #[getter]
    fn ready_for_verification(&self) -> bool {
        self.progress.ready_for_verification
    }
}

/// An owned candidate search with bounded advances and a persistent budget.
///
/// There is no public constructor, source setter, budget replacement or receipt
/// deserializer. PyO3's exclusive mutable borrow serializes advances/checkpoints
/// while native arithmetic releases the GIL. Cancellation can still be requested
/// through the separately held budget. Read-only progress remains available after
/// a failed advance; exporting a checkpoint itself requires remaining work.
#[pyclass(name = "RootIsolationSearch", module = "fsym_python")]
pub struct PyRootIsolationSearch {
    search: RootIsolationSearch,
    budget: Py<PyEvaluationBudget>,
}

#[pymethods]
impl PyRootIsolationSearch {
    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.search
            .residuals()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.search
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.search.domain())
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.search.options().precision_digits
    }

    #[getter]
    fn max_leaves(&self) -> usize {
        self.search.options().max_leaves
    }

    #[getter]
    fn max_advance_cells(&self) -> usize {
        MAX_ROOT_ISOLATION_ADVANCE_CELLS
    }

    #[getter]
    fn max_checkpoint_bytes(&self) -> usize {
        MAX_ROOT_SEARCH_CHECKPOINT_BYTES
    }

    #[getter]
    fn budget(&self, py: Python<'_>) -> Py<PyEvaluationBudget> {
        self.budget.clone_ref(py)
    }

    #[getter]
    fn progress(&self) -> PyRootIsolationProgress {
        PyRootIsolationProgress {
            progress: self.search.progress(),
        }
    }

    /// Attempt a bounded batch without accepting any root or resetting work.
    #[pyo3(signature = (max_cells=None))]
    fn advance(
        &mut self,
        py: Python<'_>,
        max_cells: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRootIsolationProgress> {
        let max_cells = if let Some(value) = max_cells {
            if !value.is_exact_instance(&py.get_type::<PyInt>()) {
                return Err(PyTypeError::new_err(
                    "max_cells must be an exact built-in integer",
                ));
            }
            value
                .extract::<usize>()
                .map_err(|_| PyValueError::new_err("max_cells must be between 1 and 2047"))?
        } else {
            1
        };
        if !(1..=MAX_ROOT_ISOLATION_ADVANCE_CELLS).contains(&max_cells) {
            return Err(PyValueError::new_err(
                "max_cells must be between 1 and 2047",
            ));
        }
        let budget = self.budget.borrow(py);
        let search = &mut self.search;
        let progress =
            with_certification_meter(py, Some(&budget), |meter| search.advance(max_cells, meter))?;
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        Ok(PyRootIsolationProgress { progress })
    }

    /// Independently check the complete original-source cover on the same ledger.
    /// An unfinished candidate or an exhausted/cancelled budget returns no receipt.
    fn finish(&self, py: Python<'_>) -> PyResult<PyRootIsolationReceipt> {
        let budget = self.budget.borrow(py);
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        let witness = self.search.completed_witness().ok_or_else(|| {
            PyValueError::new_err("root isolation search is not ready for verification")
        })?;
        let receipt = with_certification_meter(py, Some(&budget), |meter| {
            verify_root_isolation(
                self.search.residuals(),
                self.search.variables(),
                self.search.domain(),
                witness,
                meter,
            )
        })?;
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        Ok(PyRootIsolationReceipt { receipt })
    }

    /// Export a canonical candidate prefix. Save before cancelling or exhausting
    /// the ledger: copying and encoding this checkpoint also consume its budget.
    fn checkpoint<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let budget = self.budget.borrow(py);
        let search = &mut self.search;
        let encoded = with_certification_meter(py, Some(&budget), |meter| {
            let encoded = export_root_search_checkpoint(search, *CONTEXT, meter)?;
            // Reserve the Python-owned copy on the same persistent ledger.
            charge(meter, encoded.len() as u64 + 1, encoded.len() as u64, 1)?;
            Ok::<_, CertificationError>(encoded)
        })?;
        let result = PyBytes::new(py, &encoded);
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        Ok(result)
    }
}

fn options(precision_digits: u32, max_leaves: usize) -> PyResult<RootIsolationOptions> {
    if !(8..=128).contains(&precision_digits) {
        return Err(PyValueError::new_err(
            "root isolation precision must be between 8 and 128",
        ));
    }
    if !(1..=MAX_ROOT_ISOLATION_LEAVES).contains(&max_leaves) {
        return Err(PyValueError::new_err(
            "root isolation max_leaves must be between 1 and 1024",
        ));
    }
    Ok(RootIsolationOptions {
        precision_digits,
        max_leaves,
    })
}

fn check_shape(residuals: &[Expr], variables: &[Symbol]) -> PyResult<()> {
    if !(1..=16).contains(&variables.len()) || residuals.len() != variables.len() {
        return Err(PyValueError::new_err(
            "root isolation requires a square system with 1..16 variables",
        ));
    }
    Ok(())
}

fn capture_budget(
    py: Python<'_>,
    budget: Option<Py<PyEvaluationBudget>>,
) -> PyResult<Py<PyEvaluationBudget>> {
    match budget {
        Some(budget) => Ok(budget),
        None => Py::new(py, PyEvaluationBudget::certification_default()),
    }
}

pub(super) fn begin(
    residuals: &[Expr],
    variables: &[Symbol],
    domain: &Bound<'_, PyAny>,
    precision_digits: u32,
    max_leaves: usize,
    budget: Option<Py<PyEvaluationBudget>>,
) -> PyResult<PyRootIsolationSearch> {
    let options = options(precision_digits, max_leaves)?;
    check_shape(residuals, variables)?;
    let py = domain.py();
    let budget = capture_budget(py, budget)?;
    let bound_budget = budget.borrow(py);
    let domain = exact_box(domain, variables.len(), Some(&bound_budget))?;
    let search = with_certification_meter(py, Some(&bound_budget), |meter| {
        RootIsolationSearch::new(residuals, variables, &domain, options, meter)
    })?;
    with_evaluation_meter_attached(Some(&bound_budget), |meter| meter.checkpoint())?;
    drop(bound_budget);
    Ok(PyRootIsolationSearch { search, budget })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn restore(
    residuals: &[Expr],
    variables: &[Symbol],
    data: &Bound<'_, PyAny>,
    domain: &Bound<'_, PyAny>,
    precision_digits: u32,
    max_leaves: usize,
    budget: Option<Py<PyEvaluationBudget>>,
) -> PyResult<PyRootIsolationSearch> {
    let options = options(precision_digits, max_leaves)?;
    check_shape(residuals, variables)?;
    let py = data.py();
    if !data.is_exact_instance(&py.get_type::<PyBytes>()) {
        return Err(PyTypeError::new_err(
            "root search checkpoint must be exact built-in bytes",
        ));
    }
    let encoded = data.cast::<PyBytes>()?.as_bytes();
    if encoded.len() > MAX_ROOT_SEARCH_CHECKPOINT_BYTES {
        return Err(PyValueError::new_err(
            "root search checkpoint exceeds the 1 MiB byte limit",
        ));
    }
    let budget = capture_budget(py, budget)?;
    let bound_budget = budget.borrow(py);
    let domain = exact_box(domain, variables.len(), Some(&bound_budget))?;
    let search = with_certification_meter(py, Some(&bound_budget), |meter| {
        let sources = RootCertificateSources::copy_from(residuals, variables, meter)?;
        restore_root_search_checkpoint(encoded, &sources, &domain, options, *CONTEXT, meter)
    })?;
    with_evaluation_meter_attached(Some(&bound_budget), |meter| meter.checkpoint())?;
    drop(bound_budget);
    Ok(PyRootIsolationSearch { search, budget })
}
