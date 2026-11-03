//! Caller-owned ledgers for numerical evaluation and source-bound certification.
//!
//! No worker or executor is created here. A call borrows one ledger for its
//! complete native evaluation and releases the GIL while performing arithmetic.
//! A second caller may request cancellation without acquiring that ledger.

#![forbid(unsafe_code)]

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::EvalError;
use fsym_core::{BallError, RootBoxError};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyInt;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};

pyo3::create_exception!(fsym_python, EvaluationBudgetExceeded, PyRuntimeError);
pyo3::create_exception!(fsym_python, EvaluationCancelled, PyRuntimeError);

const DEFAULT_STEPS: u64 = 10_000_000;
const DEFAULT_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_ALLOCATIONS: u64 = 100_000;

fn limit(value: Option<&Bound<'_, PyAny>>, default: u64, name: &str) -> PyResult<u64> {
    let Some(value) = value else {
        return Ok(default);
    };
    if !value.is_exact_instance(&value.py().get_type::<PyInt>()) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an exact built-in integer"
        )));
    }
    value
        .extract::<u64>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be between 0 and 2**64 - 1")))
}

/// Cumulative fuel and native-buffer accounting, with irreversible cancellation.
///
/// Reuse this object to share limits across programs, outputs, derivative
/// stages and solver iterations. Defaults are 10,000,000 logical steps,
/// 256 MiB cumulative buffer bytes, and 100,000 allocations. Zero is a valid
/// limit; None selects the default. Consumption survives failures, including
/// rejected root boxes and precision retries. Compilation, symbolic proof
/// admission and CPython's own allocation overhead are outside this ledger.
/// Memory is cumulative charged buffer bytes and conservative interval-work
/// reservations, not RSS
/// or a peak-live-memory measurement. Omitting a budget preserves the existing
/// structurally bounded evaluation API without a cumulative caller ledger.
#[pyclass(name = "EvaluationBudget", module = "fsym_python", frozen)]
pub struct PyEvaluationBudget {
    ledger: Mutex<Budget>,
    cancelled: AtomicBool,
}

impl PyEvaluationBudget {
    fn ledger(&self) -> PyResult<MutexGuard<'_, Budget>> {
        match self.ledger.try_lock() {
            Ok(ledger) => Ok(ledger),
            Err(TryLockError::WouldBlock) => Err(PyRuntimeError::new_err(
                "evaluation budget is already in use; concurrent evaluations require distinct budgets",
            )),
            Err(TryLockError::Poisoned(_)) => Err(PyRuntimeError::new_err(
                "evaluation budget is unavailable after an internal fault",
            )),
        }
    }

    fn remaining(&self, dimension: Dimension) -> PyResult<u64> {
        Ok(self.ledger()?.remaining(dimension))
    }
}

#[pymethods]
impl PyEvaluationBudget {
    #[new]
    #[pyo3(signature = (*, max_steps=None, max_memory_bytes=None, max_allocations=None))]
    fn new(
        max_steps: Option<&Bound<'_, PyAny>>,
        max_memory_bytes: Option<&Bound<'_, PyAny>>,
        max_allocations: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] =
            limit(max_steps, DEFAULT_STEPS, "max_steps")?;
        limits.dimensions[Dimension::MemoryBytes.index()] =
            limit(max_memory_bytes, DEFAULT_MEMORY_BYTES, "max_memory_bytes")?;
        limits.dimensions[Dimension::AllocationCount.index()] =
            limit(max_allocations, DEFAULT_ALLOCATIONS, "max_allocations")?;
        Ok(Self {
            ledger: Mutex::new(Budget::new(limits)),
            cancelled: AtomicBool::new(false),
        })
    }

    /// Request cancellation at the next native or solver safe point.
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[getter]
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    #[getter]
    fn remaining_steps(&self) -> PyResult<u64> {
        self.remaining(Dimension::ComputeSteps)
    }

    #[getter]
    fn remaining_memory_bytes(&self) -> PyResult<u64> {
        self.remaining(Dimension::MemoryBytes)
    }

    #[getter]
    fn remaining_allocations(&self) -> PyResult<u64> {
        self.remaining(Dimension::AllocationCount)
    }

    // Python solver loops use this same private ledger seam before dense
    // linear work. The method only consumes: it cannot refund or reset fuel.
    #[pyo3(signature = (steps, *, memory_bytes=0, allocations=0))]
    fn _charge(
        &self,
        py: Python<'_>,
        steps: u64,
        memory_bytes: u64,
        allocations: u64,
    ) -> PyResult<()> {
        with_evaluation_meter(py, Some(self), |meter| {
            let requested = [
                (Dimension::ComputeSteps, steps),
                (Dimension::MemoryBytes, memory_bytes),
                (Dimension::AllocationCount, allocations),
            ];
            let mut charges = [(Dimension::ComputeSteps, 0); 3];
            let mut count = 0;
            for charge in requested {
                if charge.1 != 0 {
                    charges[count] = charge;
                    count += 1;
                }
            }
            meter.charge_batch(&charges[..count])
        })
    }

    fn _checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        with_evaluation_meter(py, Some(self), |meter| meter.checkpoint())
    }
}

pub(crate) struct EvaluationMeter<'a> {
    ledger: Option<MutexGuard<'a, Budget>>,
    cancellation: Option<&'a AtomicBool>,
    fallback: Option<Budget>,
}

impl BudgetMeter for EvaluationMeter<'_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.checkpoint()?;
        if let Some(ledger) = &mut self.ledger {
            ledger.charge(dimension, amount)?;
        } else if let Some(ledger) = &mut self.fallback {
            ledger.charge(dimension, amount)?;
        }
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()?;
        if let Some(ledger) = &mut self.ledger {
            ledger.charge_batch(charges)?;
        } else if let Some(ledger) = &mut self.fallback {
            ledger.charge_batch(charges)?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self
            .cancellation
            .is_some_and(|cancelled| cancelled.load(Ordering::Acquire))
        {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Preserve typed operational refusals, including nested objective errors.
pub(crate) fn evaluation_error<E: Error + 'static>(error: E) -> PyErr {
    let mut current: &(dyn Error + 'static) = &error;
    loop {
        if let Some(meter) = current.downcast_ref::<MeterError>() {
            return match meter {
                MeterError::Cancelled => EvaluationCancelled::new_err(error.to_string()),
                MeterError::Budget(_) => EvaluationBudgetExceeded::new_err(error.to_string()),
            };
        }
        if matches!(
            current.downcast_ref::<EvalError>(),
            Some(EvalError::AllocationFailure)
        ) {
            return PyMemoryError::new_err(error.to_string());
        }
        if matches!(
            current.downcast_ref::<RootBoxError>(),
            Some(RootBoxError::ResourceLimit(_))
        ) || matches!(
            current.downcast_ref::<BallError>(),
            Some(BallError::ResourceLimitExceeded { .. })
        ) {
            return EvaluationBudgetExceeded::new_err(error.to_string());
        }
        match current.source() {
            Some(source) => current = source,
            None => return PyValueError::new_err(error.to_string()),
        }
    }
}

pub(crate) fn with_evaluation_meter<T, E, F>(
    py: Python<'_>,
    budget: Option<&PyEvaluationBudget>,
    operation: F,
) -> PyResult<T>
where
    T: Send,
    E: Error + 'static,
    F: FnOnce(&mut EvaluationMeter<'_>) -> Result<T, E> + Send,
{
    py.detach(move || with_evaluation_meter_attached(budget, operation))
}

/// Interval work shares a supplied ledger. Without one, retain the previous
/// fixed 512 MiB/dimension envelope in addition to the core checker's own caps.
/// A supplied budget never creates a replacement ledger when a trial fails.
pub(crate) fn with_certification_meter<T, E, F>(
    py: Python<'_>,
    budget: Option<&PyEvaluationBudget>,
    operation: F,
) -> PyResult<T>
where
    T: Send,
    E: Error + 'static,
    F: FnOnce(&mut EvaluationMeter<'_>) -> Result<T, E> + Send,
{
    py.detach(move || {
        let fallback = budget
            .is_none()
            .then(|| Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0)));
        with_meter_attached(budget, fallback, operation)
    })
}

/// Trivial precharges must not release the GIL between reading a mutable
/// Python container's length and making its bounded native snapshot.
pub(crate) fn with_evaluation_meter_attached<T, E, F>(
    budget: Option<&PyEvaluationBudget>,
    operation: F,
) -> PyResult<T>
where
    E: Error + 'static,
    F: FnOnce(&mut EvaluationMeter<'_>) -> Result<T, E>,
{
    with_meter_attached(budget, None, operation)
}

fn with_meter_attached<T, E, F>(
    budget: Option<&PyEvaluationBudget>,
    fallback: Option<Budget>,
    operation: F,
) -> PyResult<T>
where
    E: Error + 'static,
    F: FnOnce(&mut EvaluationMeter<'_>) -> Result<T, E>,
{
    let mut meter = EvaluationMeter {
        ledger: budget.map(PyEvaluationBudget::ledger).transpose()?,
        cancellation: budget.map(|budget| &budget.cancelled),
        fallback,
    };
    meter.checkpoint().map_err(evaluation_error)?;
    let result = operation(&mut meter).map_err(evaluation_error)?;
    // Even an empty batch or zero-dimensional output has a publication
    // safe point. In the detached lane the guard drops before GIL reentry.
    meter.checkpoint().map_err(evaluation_error)?;
    Ok(result)
}

/// Charge bridge-owned native vectors before reserving their storage.
pub(crate) fn evaluation_buffer<T>(
    count: usize,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<T>, EvalError> {
    meter.checkpoint()?;
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(EvalError::DimensionOverflow)?;
    if bytes != 0 {
        meter.charge_batch(&[
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, 1),
        ])?;
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(count)
        .map_err(|_| EvalError::AllocationFailure)?;
    Ok(buffer)
}

#[cfg(test)]
#[path = "numeric_budget/tests.rs"]
mod tests;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEvaluationBudget>()?;
    module.add(
        "EvaluationBudgetExceeded",
        module.py().get_type::<EvaluationBudgetExceeded>(),
    )?;
    module.add(
        "EvaluationCancelled",
        module.py().get_type::<EvaluationCancelled>(),
    )?;
    Ok(())
}
