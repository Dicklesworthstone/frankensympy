//! Typed modular-root bindings. The native number-theory crate checks every
//! returned root and establishes completeness for the all-roots operation.

use pyo3::prelude::*;

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
