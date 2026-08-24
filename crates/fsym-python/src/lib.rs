//! # fsym-python
//!
//! PyO3 bindings exposing FrankenSymPy to CPython as a native extension
//! module. Strings cross the boundary; everything inside is exact.

use fsym_calculus::{diff, integrate, limit, taylor};
use fsym_core::{Expr, Symbol, parse};
use fsym_ntheory::{factorint, totient};
use fsym_runtime::{Budget, BudgetLimits, FsymCx, RuntimeBudget};
use fsym_simplify::{SimplifyError, expand_with, simplify_with};
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

/// Maps a metered evaluation refusal onto the Python surface. Resource and
/// cancellation refusals are explicit `ValueError`s; nothing silently
/// falls back to unbounded execution.
fn eval_error(e: SimplifyError) -> PyErr {
    match e {
        SimplifyError::BudgetExhausted(b) => {
            PyValueError::new_err(format!("evaluation budget exhausted: {b}"))
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Default per-request evaluation region: RuntimeBudget-default step
/// limits over a fresh detached asupersync cancel region. No verifier
/// pool: evaluation is generator-side work.
fn eval_region() -> (asupersync::Cx<asupersync::cx::cap::None>, BudgetLimits) {
    let steps =
        u64::try_from(RuntimeBudget::default().max_eval_steps).expect("step limit fits u64");
    let limits = BudgetLimits::uniform(steps, 0);
    (asupersync::Cx::detached_cancel_context(), limits)
}

/// Simplify an expression string under a budgeted evaluation region.
#[pyfunction]
fn simplify_expr(src: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    let (cx, limits) = eval_region();
    let mut region = FsymCx::new(&cx, Budget::new(limits), limits);
    simplify_with(&e, &mut region)
        .map(|v| v.to_string())
        .map_err(eval_error)
}

/// Expand an expression string under a budgeted evaluation region.
#[pyfunction]
fn expand_expr(src: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    let (cx, limits) = eval_region();
    let mut region = FsymCx::new(&cx, Budget::new(limits), limits);
    expand_with(&e, &mut region)
        .map(|v| v.to_string())
        .map_err(eval_error)
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

pub mod expr;
pub use expr::*;

/// Numeric evaluation of an expression string.
#[pyfunction]
fn evalf_expr(src: &str) -> PyResult<f64> {
    parse_expr(src)?.evalf().map_err(to_value_error)
}

/// Native FrankenSymPy module.
#[pymodule]
fn fsym_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_function(wrap_pyfunction!(py_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(py_integer, m)?)?;
    m.add_function(wrap_pyfunction!(py_rational, m)?)?;
    m.add_function(wrap_pyfunction!(py_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_mul, m)?)?;
    m.add_function(wrap_pyfunction!(py_pow, m)?)?;
    m.add_function(wrap_pyfunction!(py_derivative, m)?)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_expr_structural_args_and_properties() {
        let x = py_symbol("x");
        let two = py_integer(2);
        let expr = x.__mul__(&two);

        assert_eq!(expr.func_name(), "Mul");
        assert_eq!(expr.args().len(), 2);
        assert!(!expr.is_integer());
        assert!(!expr.is_symbol());
        assert!(expr.is_mul());

        let x_sym = py_symbol("x");
        assert!(x_sym.is_symbol());
        assert_eq!(x_sym.func_name(), "Symbol");
    }

    #[test]
    fn test_py_expr_differentiation_and_latex() {
        // d/dx (x^3) = 3*x^2
        let x = py_symbol("x");
        let three = py_integer(3);
        let pow_expr = py_pow(x, three);

        let d = pow_expr.diff("x");
        assert_eq!(d.__str__(), "3*(x**2)");
        assert!(pow_expr._repr_latex_().contains("x^{3}"));
    }

    #[test]
    fn test_py_expr_substitution() {
        // x + 5 where x -> 10
        let x = py_symbol("x");
        let five = py_integer(5);
        let expr = x.__add__(&five);

        let ten = py_integer(10);
        let res = expr.subs(&x, &ten).unwrap();
        assert_eq!(res.__str__(), "15");
    }
}
