//! # fsym-python
//!
//! PyO3 bindings exposing FrankenSymPy to CPython as a native extension
//! module. Strings cross the boundary; everything inside is exact.

use fsym_calculus::{diff, integrate, limit, taylor};
use fsym_core::{Expr, Symbol, parse};
use fsym_ntheory::{factorint, totient};
use fsym_simplify::{expand, simplify};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_expr(src: &str) -> PyResult<Expr> {
    parse(src).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Parse-failures and typed domain refusals both surface as `ValueError`
/// with the original message.
fn to_value_error<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Library version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Render a symbol.
#[pyfunction]
fn symbol(name: &str) -> String {
    format!("{}", Symbol::new(name))
}

/// Miller-Rabin primality test.
#[pyfunction]
fn is_prime(n: u64) -> bool {
    fsym_ntheory::is_prime(n)
}

/// Prime factorization as a {prime: exponent} mapping.
#[pyfunction]
fn factorize(n: u64) -> PyResult<std::collections::BTreeMap<u64, u32>> {
    factorint(n).map_err(to_value_error)
}

/// Euler's totient φ(n).
#[pyfunction]
fn euler_totient(n: u64) -> PyResult<u64> {
    totient(n).map_err(to_value_error)
}

/// Simplify an expression string.
#[pyfunction]
fn simplify_expr(src: &str) -> PyResult<String> {
    Ok(simplify(&parse_expr(src)?).to_string())
}

/// Expand products of sums and bounded powers.
#[pyfunction]
fn expand_expr(src: &str) -> PyResult<String> {
    Ok(expand(&parse_expr(src)?).to_string())
}

/// Taylor polynomial of `src` around `var = at` through degree `order`.
#[pyfunction]
fn taylor_expr(src: &str, var: &str, at: i64, order: usize) -> PyResult<String> {
    let e = parse_expr(src)?;
    let at_expr = parse_expr(&at.to_string())?;
    taylor(&e, &Symbol::new(var), &at_expr, order)
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Differentiate `src` with respect to `var`.
#[pyfunction]
fn diff_expr(src: &str, var: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    Ok(diff(&e, &Symbol::new(var)).to_string())
}

/// Indefinite integral of `src` with respect to `var`.
///
/// Raises `ValueError` when no rule applies — refusals are explicit.
#[pyfunction]
fn integrate_expr(src: &str, var: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    integrate(&e, &Symbol::new(var))
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Limit of `src` as `var -> to` (`to` may be `"oo"` / `"-oo"`).
#[pyfunction]
fn limit_expr(src: &str, var: &str, to: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    let point = parse_expr(to)?;
    limit(&e, &Symbol::new(var), &point)
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Solve a linear equation `expr == 0` for `var`.
#[pyfunction]
fn solve_linear_expr(src: &str, var: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    fsym_solvers::solve_linear(&e, &Symbol::new(var))
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Numeric evaluation of an expression string.
#[pyfunction]
fn evalf_expr(src: &str) -> PyResult<f64> {
    parse_expr(src)?.evalf().map_err(to_value_error)
}

/// Native FrankenSymPy module.
#[pymodule]
fn fsym_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(symbol, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(euler_totient, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_expr, m)?)?;
    m.add_function(wrap_pyfunction!(expand_expr, m)?)?;
    m.add_function(wrap_pyfunction!(diff_expr, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_expr, m)?)?;
    m.add_function(wrap_pyfunction!(limit_expr, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_expr, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear_expr, m)?)?;
    m.add_function(wrap_pyfunction!(evalf_expr, m)?)?;
    Ok(())
}
