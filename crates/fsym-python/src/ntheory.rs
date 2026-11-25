//! Typed modular-root bindings. The native number-theory crate checks every
//! returned root and establishes completeness for the all-roots operation.

use crate::callbacks::PySupervisedCallbackSession;
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn group_error(error: fsym_ntheory::NTheoryError) -> PyErr {
    match error {
        fsym_ntheory::NTheoryError::ModularGroupWorkLimitExceeded
        | fsym_ntheory::NTheoryError::DiscreteLogTableLimitExceeded(_)
        | fsym_ntheory::NTheoryError::FactorizationLimitExceeded(_)
        | fsym_ntheory::NTheoryError::ModularNthRootCountExceeded(_)
        | fsym_ntheory::NTheoryError::ModularNthRootWorkLimitExceeded
        | fsym_ntheory::NTheoryError::ModularNthRootAllocationLimitExceeded
        | fsym_ntheory::NTheoryError::ModularNthRootVerificationFailed(_)
        | fsym_ntheory::NTheoryError::Meter(_) => {
            pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
        }
        _ => crate::to_value_error(error),
    }
}

/// Least positive exponent of a unit modulo any admitted composite or prime.
#[pyfunction]
pub fn multiplicative_order_fn(a: u64, modulus: u64) -> PyResult<u64> {
    fsym_ntheory::multiplicative_order(a, modulus).map_err(group_error)
}

/// Complete cyclic-group classification followed by a bounded generator search.
#[pyfunction]
pub fn primitive_root_fn(modulus: u64, smallest: bool) -> PyResult<Option<u64>> {
    fsym_ntheory::primitive_root(modulus, smallest).map_err(group_error)
}

/// Least nonnegative exponent, with exact order-hint validation and bounded work.
#[pyfunction]
#[pyo3(signature = (modulus, target, base, order=None))]
pub fn discrete_log_fn(modulus: u64, target: u64, base: u64, order: Option<u64>) -> PyResult<u64> {
    fsym_ntheory::discrete_log(modulus, target, base, order).map_err(group_error)
}

/// Power-residue existence over prime powers and composites without enumeration.
#[pyfunction]
pub fn is_nth_power_residue_fn(a: u64, degree: u64, modulus: u64) -> PyResult<bool> {
    fsym_ntheory::is_nth_power_residue(a, degree, modulus).map_err(group_error)
}

/// All distinct square roots modulo a positive `u64` modulus, in sorted order.
#[pyfunction]
pub fn modular_square_roots_fn(a: u64, modulus: u64) -> PyResult<Vec<u64>> {
    fsym_ntheory::modular_square_roots(a, modulus).map_err(crate::to_value_error)
}

/// One square root without allocating the potentially large complete root set.
#[pyfunction]
pub fn modular_square_root_fn(a: u64, modulus: u64) -> PyResult<Option<u64>> {
    fsym_ntheory::modular_square_root(a, modulus).map_err(crate::to_value_error)
}

/// Exact existence test, including nonunits and composite moduli.
#[pyfunction]
pub fn is_quadratic_residue_fn(a: u64, modulus: u64) -> PyResult<bool> {
    fsym_ntheory::is_quadratic_residue(a, modulus).map_err(crate::to_value_error)
}

// Both native phases retain their own finite work and storage bounds while
// debiting the same active Python request. Preserve the first Python refusal:
// MeterError is only the stop signal used by the native implementation.
struct NthRootRequestMeter<'py, 'session> {
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    failure: Option<PyErr>,
}

impl<'py, 'session> NthRootRequestMeter<'py, 'session> {
    fn new(py: Python<'py>, session: Option<&'session PySupervisedCallbackSession>) -> Self {
        Self {
            py,
            session,
            failure: None,
        }
    }

    fn forward(&mut self, steps: u64) -> Result<(), MeterError> {
        if self.failure.is_some() {
            return Err(MeterError::Cancelled);
        }
        if let Some(session) = self.session
            && let Err(error) = session.charge_nested_steps_atomic(self.py, steps)
        {
            self.failure = Some(error);
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

impl BudgetMeter for NthRootRequestMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.forward(if dimension == Dimension::ComputeSteps {
            amount
        } else {
            0
        })
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if self.failure.is_some() {
            return Err(MeterError::Cancelled);
        }
        let mut steps = 0u64;
        for &(dimension, amount) in charges {
            if dimension == Dimension::ComputeSteps {
                let Some(total) = steps.checked_add(amount) else {
                    self.failure = Some(PyValueError::new_err(
                        "modular nth-root request step charge overflow",
                    ));
                    if let Some(session) = self.session {
                        let _ = session.cancel(self.py);
                    }
                    return Err(MeterError::Cancelled);
                };
                steps = total;
            }
        }
        self.forward(steps)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.forward(0)
    }
}

fn checked_nth_root_output<T>(
    result: Result<T, fsym_ntheory::NTheoryError>,
    mut generator: NthRootRequestMeter<'_, '_>,
    mut verifier: NthRootRequestMeter<'_, '_>,
    output_work: impl FnOnce(&T) -> u64,
) -> PyResult<T> {
    let output = (|| {
        let value = result.map_err(group_error)?;
        // Publication remains on the captured request, including an empty
        // family or None. Native admission already bounds materialized roots.
        generator
            .charge(Dimension::ComputeSteps, output_work(&value).max(1))
            .map_err(crate::to_value_error)?;
        generator.checkpoint().map_err(crate::to_value_error)?;
        verifier.checkpoint().map_err(crate::to_value_error)?;
        Ok(value)
    })();
    match generator.failure.or(verifier.failure) {
        Some(error) => Err(error),
        None => output,
    }
}

/// Complete bounded roots for any positive degree and composite u64 modulus.
#[pyfunction]
#[pyo3(signature = (a, degree, modulus, session=None))]
pub fn modular_nth_roots_fn(
    py: Python<'_>,
    a: u64,
    degree: u64,
    modulus: u64,
    session: Option<&PySupervisedCallbackSession>,
) -> PyResult<Vec<u64>> {
    let mut generator = NthRootRequestMeter::new(py, session);
    let mut verifier = NthRootRequestMeter::new(py, session);
    let result =
        fsym_ntheory::metered_modular_nth_roots(a, degree, modulus, &mut generator, &mut verifier);
    checked_nth_root_output(result, generator, verifier, |roots| roots.len() as u64)
}

/// Least root, with absence and resource refusal kept distinct.
#[pyfunction]
#[pyo3(signature = (a, degree, modulus, session=None))]
pub fn modular_nth_root_fn(
    py: Python<'_>,
    a: u64,
    degree: u64,
    modulus: u64,
    session: Option<&PySupervisedCallbackSession>,
) -> PyResult<Option<u64>> {
    let mut generator = NthRootRequestMeter::new(py, session);
    let mut verifier = NthRootRequestMeter::new(py, session);
    let result =
        fsym_ntheory::metered_modular_nth_root(a, degree, modulus, &mut generator, &mut verifier);
    checked_nth_root_output(result, generator, verifier, |_| 1)
}
