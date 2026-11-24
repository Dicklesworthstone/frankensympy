//! Yielding, source-bound global optimization on exact closed boxes.
//!
//! A handle owns one cumulative ledger and a native candidate frontier. Only
//! `finish` runs the independent complete-cover verifier and returns a receipt.

#![forbid(unsafe_code)]

use super::{PyBoxOptimizationReceipt, box_extremum_kind};
use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric::{ball_bounds, exact_box, exact_rational, rational_expr};
use crate::numeric_budget::{
    PyEvaluationBudget, evaluation_error, with_certification_meter, with_evaluation_meter_attached,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::BudgetMeter;
use fsym_calculus::box_optimization::checkpoint_transport::{
    MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES, OptimizationCheckpointError,
    export_box_optimization_checkpoint, restore_box_optimization_checkpoint,
};
use fsym_calculus::box_optimization::{
    BoxOptimizationOptions, BoxOptimizationProgress, BoxOptimizationSearch,
    BoxOptimizationSearchStatus, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS,
};
use fsym_core::root_certificate::artifact::charge;
use fsym_core::{
    BoxExtremumKind, Expr, MAX_BOX_OPTIMIZATION_LEAVES, MAX_ROOT_BOX_VARIABLES, Symbol,
    verify_box_optimization,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};
use std::sync::LazyLock;

// Declared facts stay in the frozen typed coordinates. The Python bridge does
// not introduce an ambient assumption context absent from the source snapshot.
static CONTEXT: LazyLock<[u8; 32]> =
    LazyLock::new(|| ImmutableAssumptionsSnapshot::empty().digest());

/// An immutable candidate snapshot, without accepted values or optimizer claims.
#[pyclass(name = "BoxOptimizationProgress", module = "fsym_python", frozen)]
pub struct PyBoxOptimizationProgress {
    progress: BoxOptimizationProgress,
}

#[pymethods]
impl PyBoxOptimizationProgress {
    #[getter]
    fn status(&self) -> &'static str {
        match self.progress.status {
            BoxOptimizationSearchStatus::Searching => "searching",
            BoxOptimizationSearchStatus::Complete => "ready_for_verification",
            BoxOptimizationSearchStatus::Failed => "failed",
        }
    }

    #[getter]
    fn leaf_count(&self) -> usize {
        self.progress.leaf_count
    }

    #[getter]
    fn split_count(&self) -> usize {
        self.progress.split_count
    }

    #[getter]
    fn evaluation_count(&self) -> usize {
        self.progress.evaluation_count
    }

    #[getter]
    fn ready_for_verification(&self) -> bool {
        self.progress.status == BoxOptimizationSearchStatus::Complete
    }
}

/// An owned continuation with bounded advances and a persistent caller ledger.
///
/// There is no public constructor, mutable statement, budget replacement, or
/// receipt deserializer. PyO3 serializes mutable advances/checkpoints while
/// native arithmetic releases the GIL. A failed native transition is terminal;
/// progress remains readable, but only an earlier saved checkpoint can resume.
#[pyclass(name = "BoxOptimizationSearch", module = "fsym_python")]
pub struct PyBoxOptimizationSearch {
    search: BoxOptimizationSearch,
    budget: Py<PyEvaluationBudget>,
}

#[pymethods]
impl PyBoxOptimizationSearch {
    #[getter]
    fn objective(&self) -> PyExpr {
        PyExpr::from_expr(self.search.objective().clone())
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
    fn kind(&self) -> &'static str {
        match self.search.kind() {
            BoxExtremumKind::Minimum => "minimum",
            BoxExtremumKind::Maximum => "maximum",
        }
    }

    #[getter]
    fn absolute_tolerance(&self) -> PyExpr {
        rational_expr(self.search.absolute_tolerance())
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
    fn max_advance_splits(&self) -> usize {
        MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS
    }

    #[getter]
    fn max_checkpoint_bytes(&self) -> usize {
        MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES
    }

    #[getter]
    fn budget(&self, py: Python<'_>) -> Py<PyEvaluationBudget> {
        self.budget.clone_ref(py)
    }

    #[getter]
    fn progress(&self) -> PyBoxOptimizationProgress {
        PyBoxOptimizationProgress {
            progress: self.search.progress(),
        }
    }

    /// Attempt a bounded number of completed bisections on the same allowance.
    #[pyo3(signature = (max_splits=None))]
    fn advance(
        &mut self,
        py: Python<'_>,
        max_splits: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBoxOptimizationProgress> {
        let max_splits = if let Some(value) = max_splits {
            if !value.is_exact_instance(&py.get_type::<PyInt>()) {
                return Err(PyTypeError::new_err(
                    "max_splits must be an exact built-in integer",
                ));
            }
            value
                .extract::<usize>()
                .map_err(|_| PyValueError::new_err("max_splits must be between 1 and 1024"))?
        } else {
            1
        };
        if !(1..=MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS).contains(&max_splits) {
            return Err(PyValueError::new_err(
                "max_splits must be between 1 and 1024",
            ));
        }
        let budget = self.budget.borrow(py);
        let search = &mut self.search;
        let progress =
            with_certification_meter(py, Some(&budget), |meter| search.advance(max_splits, meter))?;
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        Ok(PyBoxOptimizationProgress { progress })
    }

    /// Reevaluate the original objective on the complete cover and point.
    /// Unfinished/failed candidates and refused ledgers cannot publish a receipt.
    fn finish(&self, py: Python<'_>) -> PyResult<PyBoxOptimizationReceipt> {
        let budget = self.budget.borrow(py);
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        if let Some(error) = self.search.failure() {
            return Err(evaluation_error(error.clone()));
        }
        let witness = self.search.completed_witness().ok_or_else(|| {
            PyValueError::new_err("box optimization search is not ready for verification")
        })?;
        let receipt = with_certification_meter(py, Some(&budget), |meter| {
            verify_box_optimization(
                self.search.objective(),
                self.search.variables(),
                self.search.domain(),
                self.search.kind(),
                self.search.absolute_tolerance(),
                witness,
                meter,
            )
        })?;
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        Ok(PyBoxOptimizationReceipt { receipt })
    }

    /// Encode a canonical candidate, spending the same ledger as live search.
    /// Save before cancellation/exhaustion: encoding also requires work.
    fn checkpoint<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let budget = self.budget.borrow(py);
        let search = &mut self.search;
        let encoded = with_certification_meter(py, Some(&budget), |meter| {
            let encoded = export_box_optimization_checkpoint(search, *CONTEXT, meter)?;
            charge(meter, encoded.len() as u64 + 1, encoded.len() as u64, 1)?;
            Ok::<_, OptimizationCheckpointError>(encoded)
        })?;
        let result = PyBytes::new(py, &encoded);
        with_evaluation_meter_attached(Some(&budget), |meter| meter.checkpoint())?;
        Ok(result)
    }
}

fn options(precision_digits: u32, max_leaves: usize) -> PyResult<BoxOptimizationOptions> {
    if !(8..=128).contains(&precision_digits) {
        return Err(PyValueError::new_err(
            "box optimization precision must be between 8 and 128",
        ));
    }
    if !(1..=MAX_BOX_OPTIMIZATION_LEAVES).contains(&max_leaves) {
        return Err(PyValueError::new_err(
            "box optimization max_leaves must be between 1 and 1024",
        ));
    }
    Ok(BoxOptimizationOptions {
        precision_digits,
        max_leaves,
    })
}

fn check_shape(variables: &[Symbol]) -> PyResult<()> {
    if !(1..=MAX_ROOT_BOX_VARIABLES).contains(&variables.len()) {
        return Err(PyValueError::new_err(
            "box optimization requires 1..16 variables",
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

#[allow(clippy::too_many_arguments)]
pub(super) fn begin(
    objective: &Expr,
    variables: &[Symbol],
    domain: &Bound<'_, PyAny>,
    tolerance: &Bound<'_, PyAny>,
    kind: &str,
    precision_digits: u32,
    max_leaves: usize,
    budget: Option<Py<PyEvaluationBudget>>,
) -> PyResult<PyBoxOptimizationSearch> {
    let options = options(precision_digits, max_leaves)?;
    let kind = box_extremum_kind(kind)?;
    check_shape(variables)?;
    let py = domain.py();
    let budget = capture_budget(py, budget)?;
    let bound_budget = budget.borrow(py);
    let domain = exact_box(domain, variables.len(), Some(&bound_budget))?;
    let tolerance = exact_rational(tolerance, Some(&bound_budget))?;
    let search = with_certification_meter(py, Some(&bound_budget), |meter| {
        BoxOptimizationSearch::new(
            objective, variables, &domain, kind, &tolerance, options, meter,
        )
    })?;
    with_evaluation_meter_attached(Some(&bound_budget), |meter| meter.checkpoint())?;
    drop(bound_budget);
    Ok(PyBoxOptimizationSearch { search, budget })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn restore(
    objective: &Expr,
    variables: &[Symbol],
    data: &Bound<'_, PyAny>,
    domain: &Bound<'_, PyAny>,
    tolerance: &Bound<'_, PyAny>,
    kind: &str,
    precision_digits: u32,
    max_leaves: usize,
    budget: Option<Py<PyEvaluationBudget>>,
) -> PyResult<PyBoxOptimizationSearch> {
    let options = options(precision_digits, max_leaves)?;
    let kind = box_extremum_kind(kind)?;
    check_shape(variables)?;
    let py = data.py();
    if !data.is_exact_instance(&py.get_type::<PyBytes>()) {
        return Err(PyTypeError::new_err(
            "box optimization checkpoint must be exact built-in bytes",
        ));
    }
    let encoded = data.cast::<PyBytes>()?.as_bytes();
    if encoded.len() > MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES {
        return Err(PyValueError::new_err(
            "box optimization checkpoint exceeds the 1 MiB byte limit",
        ));
    }
    let budget = capture_budget(py, budget)?;
    let bound_budget = budget.borrow(py);
    let domain = exact_box(domain, variables.len(), Some(&bound_budget))?;
    let tolerance = exact_rational(tolerance, Some(&bound_budget))?;
    let search = with_certification_meter(py, Some(&bound_budget), |meter| {
        restore_box_optimization_checkpoint(
            encoded, objective, variables, &domain, kind, &tolerance, options, *CONTEXT, meter,
        )
    })?;
    with_evaluation_meter_attached(Some(&bound_budget), |meter| meter.checkpoint())?;
    drop(bound_budget);
    Ok(PyBoxOptimizationSearch { search, budget })
}
