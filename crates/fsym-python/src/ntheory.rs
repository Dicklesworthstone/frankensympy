//! Typed modular-root bindings. The native number-theory crate checks every
//! returned root and establishes completeness for the all-roots operation.

use pyo3::prelude::*;

pub(crate) fn group_error(error: fsym_ntheory::NTheoryError) -> PyErr {
    match error {
        fsym_ntheory::NTheoryError::ModularGroupWorkLimitExceeded
        | fsym_ntheory::NTheoryError::DiscreteLogTableLimitExceeded(_)
        | fsym_ntheory::NTheoryError::FactorizationLimitExceeded(_) => {
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
