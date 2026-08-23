//! # fsym-python
//!
//! PyO3 Python bindings exposing FrankenSymPy to the CPython runtime as a drop-in
//! drop-in module for SymPy workloads.

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn symbol(name: &str) -> String {
    let sym = fsym_core::Symbol::new(name);
    format!("{}", sym)
}

#[pyfunction]
fn is_prime(n: u64) -> bool {
    fsym_ntheory::is_prime(n)
}

#[pymodule]
fn fsym_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(symbol, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    Ok(())
}
