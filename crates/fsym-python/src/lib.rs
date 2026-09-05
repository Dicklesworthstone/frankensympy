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

/// Definite integral of `src` from `a` to `b` with respect to `var`.
#[pyfunction]
fn integrate_definite_expr(src: &str, var: &str, a_src: &str, b_src: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    let a = parse_expr(a_src)?;
    let b = parse_expr(b_src)?;
    fsym_calculus::integrate_definite(&e, &Symbol::new(var), &a, &b)
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Laplace transform of `src(t)` to `s`.
#[pyfunction]
fn laplace_expr(src: &str, t_var: &str, s_var: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    fsym_calculus::laplace_transform(&e, &Symbol::new(t_var), &Symbol::new(s_var))
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Exact 1st-order linear ODE solver: dy/dx + P(x)*y = Q(x).
#[pyfunction]
fn dsolve_linear_first_order_expr(p_src: &str, q_src: &str, x_var: &str) -> PyResult<String> {
    let p = parse_expr(p_src)?;
    let q = parse_expr(q_src)?;
    let c1 = Symbol::new("C1");
    fsym_solvers::dsolve_linear_first_order(&p, &q, &Symbol::new(x_var), &c1)
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Exact 2nd-order constant-coefficient ODE solver: a*y'' + b*y' + c*y = 0.
#[pyfunction]
fn dsolve_const_coeff_second_order_expr(a: i64, b: i64, c: i64, x_var: &str) -> PyResult<String> {
    let c1 = Symbol::new("C1");
    let c2 = Symbol::new("C2");
    fsym_solvers::dsolve_const_coeff_second_order(a, b, c, &Symbol::new(x_var), &c1, &c2)
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Mobius function μ(n).
#[pyfunction]
fn mobius_fn(n: u64) -> PyResult<i64> {
    fsym_ntheory::mobius(n).map_err(to_value_error)
}

/// Divisor count d(n).
#[pyfunction]
fn divisor_count_fn(n: u64) -> PyResult<u64> {
    fsym_ntheory::divisor_count(n).map_err(to_value_error)
}

/// Divisor power sum σ_k(n).
#[pyfunction]
fn divisor_sum_fn(n: u64, k: u32) -> PyResult<u64> {
    fsym_ntheory::divisor_sum(n, k).map_err(to_value_error)
}

/// Jacobi symbol (a/n).
#[pyfunction]
fn jacobi_symbol_fn(a: i64, n: u64) -> PyResult<i64> {
    fsym_ntheory::jacobi_symbol(a, n).map_err(to_value_error)
}

pub mod expr;
pub mod matrix;
pub use expr::*;
pub use matrix::*;

/// Numeric evaluation of an expression string.
#[pyfunction]
fn evalf_expr(src: &str) -> PyResult<f64> {
    parse_expr(src)?.evalf().map_err(to_value_error)
}

/// Native FrankenSymPy module.
#[pymodule]
fn fsym_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PySymbol>()?;
    m.add_class::<PyInteger>()?;
    m.add_class::<PyRational>()?;
    m.add_class::<PyAdd>()?;
    m.add_class::<PyMul>()?;
    m.add_class::<PyPow>()?;
    m.add_class::<PyDerivative>()?;
    m.add_class::<PyMatrix>()?;
    m.add_function(wrap_pyfunction!(py_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(py_integer_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(py_rational_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(py_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_mul, m)?)?;
    m.add_function(wrap_pyfunction!(py_pow, m)?)?;
    m.add_function(wrap_pyfunction!(py_function, m)?)?;
    m.add_function(wrap_pyfunction!(py_sin, m)?)?;
    m.add_function(wrap_pyfunction!(py_cos, m)?)?;
    m.add_function(wrap_pyfunction!(py_exp, m)?)?;
    m.add_function(wrap_pyfunction!(py_log, m)?)?;
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
    m.add_function(wrap_pyfunction!(integrate_definite_expr, m)?)?;
    m.add_function(wrap_pyfunction!(laplace_expr, m)?)?;
    m.add_function(wrap_pyfunction!(limit_expr, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_expr, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear_expr, m)?)?;
    m.add_function(wrap_pyfunction!(dsolve_linear_first_order_expr, m)?)?;
    m.add_function(wrap_pyfunction!(dsolve_const_coeff_second_order_expr, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_fn, m)?)?;
    m.add_function(wrap_pyfunction!(divisor_count_fn, m)?)?;
    m.add_function(wrap_pyfunction!(divisor_sum_fn, m)?)?;
    m.add_function(wrap_pyfunction!(jacobi_symbol_fn, m)?)?;
    m.add_function(wrap_pyfunction!(evalf_expr, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{bigint_to_python_int, exact_python_integer};
    use fsym_core::BigInt;
    use pyo3::types::PyInt;

    #[test]
    fn test_py_expr_structural_args_and_properties() {
        let x = py_symbol("x");
        let two = py_integer(2);
        let expr = x.__mul__(&two).unwrap();

        assert_eq!(expr.func_name(), "Mul");
        assert_eq!(expr.raw_args().len(), 2);
        assert!(!expr.is_integer());
        assert!(!expr.is_symbol());
        assert!(expr.is_mul());
        assert_eq!(expr.free_symbols(), vec!["x".to_string()]);

        let x_sym = py_symbol("x");
        assert!(x_sym.is_symbol());
        assert_eq!(x_sym.func_name(), "Symbol");

        // Dedicated classes
        let sym_cls = PySymbol::new("y");
        assert_eq!(sym_cls.name(), "y");
        let int_cls = PyInteger::new(42);
        assert_eq!(int_cls.q(), 1);
        let rat_cls = PyRational::new(3, 4).unwrap();
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(int_cls.p(py).unwrap().extract::<i64>().unwrap(), 42);
            assert_eq!(rat_cls.p(py).unwrap().extract::<i64>().unwrap(), 3);
            assert_eq!(rat_cls.q(py).unwrap().extract::<i64>().unwrap(), 4);
        });

        // evaluate=False held forms
        let held_add = PyAdd::new(vec![x.clone(), x.clone()], false);
        assert_eq!(held_add.as_expr().raw_args().len(), 2);
        assert_eq!(held_add.as_expr().func_name(), "Add");

        // Evaluated n-ary constructors use identities only for empty input.
        let evaluated_add = PyAdd::new(vec![x.clone(), two.clone()], true);
        // Canonical args order: exact numbers precede symbols
        // (fra-add-args-canonical-order-o1i).
        assert_eq!(
            evaluated_add.as_expr().raw_args(),
            vec![two.clone(), x.clone()]
        );
        assert_eq!(PyAdd::new(Vec::new(), true).as_expr(), py_integer(0));
        let evaluated_mul = PyMul::new(vec![x.clone(), two.clone()], true);
        assert_eq!(
            evaluated_mul.as_expr().raw_args(),
            vec![two.clone(), x.clone()]
        );
        assert_eq!(PyMul::new(Vec::new(), true).as_expr(), py_integer(1));
    }

    #[test]
    fn python_integer_bridge_preserves_arbitrary_precision_and_preflights_size() {
        Python::initialize();
        Python::attach(|py| {
            let int_type = py.get_type::<PyInt>();
            let two_to_100 = int_type
                .call1(("1267650600228229401496703205376",))
                .unwrap();
            let three = PyInt::new(py, 3);

            let integer = py_integer_from_python(&two_to_100).unwrap();
            assert_eq!(integer.__str__(), "1267650600228229401496703205376");

            let rational = py_rational_from_python(&two_to_100, &three).unwrap();
            assert_eq!(rational.__str__(), "1267650600228229401496703205376/3");

            // This exceeds CPython's default integer-to-decimal-string digit limit. The bridge
            // must use bounded binary transit rather than depending on that interpreter setting.
            let beyond_decimal_limit = PyInt::new(py, 1)
                .call_method1("__lshift__", (20_000,))
                .unwrap();
            let bridged = py_integer_from_python(&beyond_decimal_limit).unwrap();
            assert!(matches!(
                bridged.inner,
                Expr::Integer(ref value) if value.bits() == 20_001
            ));

            let oversized = PyInt::new(py, 1)
                .call_method1("__lshift__", (MAX_PYTHON_INTEGER_BITS + 1,))
                .unwrap();
            let error = py_integer_from_python(&oversized).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("exceeds the Python integer bridge limit")
            );

            for expected in [
                -(1_i128 << 100),
                -129,
                -128,
                -127,
                -1,
                0,
                1,
                127,
                128,
                129,
                1_i128 << 100,
            ] {
                let native = BigInt::from_signed_bytes_be(&expected.to_be_bytes());
                let python = bigint_to_python_int(&native, py).unwrap();
                assert!(python.is_exact_instance(&int_type));
                assert_eq!(python.extract::<i128>().unwrap(), expected);
                assert_eq!(exact_python_integer(&python, "round_trip").unwrap(), native);
            }
        });
    }

    #[test]
    fn native_numeric_getters_do_not_launder_normalization_overflow() {
        Python::initialize();
        Python::attach(|py| {
            let integer = PyInteger {
                inner: PyExpr::from_expr(Expr::Integer(BigInt::from(i64::MAX) + BigInt::from(1))),
            };
            assert_eq!(
                integer.p(py).unwrap().str().unwrap().to_str().unwrap(),
                "9223372036854775808"
            );

            let wide_integer = PyInteger {
                inner: PyExpr::from_expr(Expr::Integer(BigInt::from(1) << 20_000u32)),
            };
            assert_eq!(
                wide_integer
                    .p(py)
                    .unwrap()
                    .call_method0("bit_length")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                20_001
            );

            let normalized_numerator = PyRational::new(i64::MIN, -1).unwrap();
            assert_eq!(
                normalized_numerator
                    .p(py)
                    .unwrap()
                    .str()
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "9223372036854775808"
            );

            let normalized_denominator = PyRational::new(-1, i64::MIN).unwrap();
            assert_eq!(
                normalized_denominator
                    .q(py)
                    .unwrap()
                    .str()
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "9223372036854775808"
            );
        });
    }

    #[test]
    fn test_py_expr_differentiation_and_latex() {
        // d/dx (x^3) = 3*x^2
        let x = py_symbol("x");
        let three = py_integer(3);
        let pow_expr = py_pow(x, three);

        let d = pow_expr.diff("x", vec![]);
        assert_eq!(d.__str__(), "3*x**2");
        assert!(pow_expr._repr_latex_().unwrap().contains("x^{3}"));
        assert_eq!(pow_expr.pretty().unwrap(), "x³");
        assert_eq!(py_sin(py_integer(0)).__str__(), "0");
        assert_eq!(py_cos(py_integer(0)).__str__(), "1");
        assert_eq!(py_exp(py_integer(0)).__str__(), "1");
        assert_eq!(py_log(py_integer(1)).__str__(), "0");
    }

    #[test]
    fn test_py_expr_substitution() {
        // x + 5 where x -> 10
        let x = py_symbol("x");
        let five = py_integer(5);
        let expr = x.__add__(&five).unwrap();

        let ten = py_integer(10);
        let res = expr.subs(&x, &ten).unwrap();
        assert_eq!(res.__str__(), "15");
    }

    #[test]
    fn test_py_definite_integral_and_laplace() {
        // \int_0^1 2*x dx = 1
        let def_int = integrate_definite_expr("2*x", "x", "0", "1").unwrap();
        assert_eq!(def_int, "1");

        // L{1}(s) = 1/s
        let lap = laplace_expr("1", "t", "s").unwrap();
        assert!(
            lap.contains("s**-1")
                || lap.contains("s^(-1)")
                || lap.contains("s**(-1)")
                || lap == "1/s"
        );
    }

    #[test]
    fn test_py_solvers_and_ntheory() {
        // dy/dx + 0*y = 2*x -> y = x^2 + C1
        let ode = dsolve_linear_first_order_expr("0", "2*x", "x").unwrap();
        assert!(ode.contains("x^2") || ode.contains("x**2") || ode.contains("C1"));

        // Mobius and divisors
        assert_eq!(mobius_fn(1).unwrap(), 1);
        assert_eq!(mobius_fn(6).unwrap(), 1);
        assert_eq!(mobius_fn(4).unwrap(), 0);
        assert_eq!(divisor_count_fn(12).unwrap(), 6);
        assert_eq!(divisor_sum_fn(6, 1).unwrap(), 12);
        assert_eq!(jacobi_symbol_fn(2, 7).unwrap(), 1);
        assert_eq!(jacobi_symbol_fn(3, 9).unwrap(), 0);
        Python::initialize();
        assert_eq!(
            jacobi_symbol_fn(1, 2).unwrap_err().to_string(),
            "ValueError: n should be an odd positive integer"
        );
    }

    #[test]
    fn test_py_matrix_operations() {
        let eye2 = PyMatrix::eye(2).unwrap();
        assert_eq!(eye2.shape(), (2, 2));
        assert!(eye2.is_square());
        assert!(eye2.is_symmetric());
        assert!(eye2.is_diagonal());
        assert_eq!(eye2.trace().unwrap().__str__(), "2");
        assert_eq!(eye2.det().unwrap().__str__(), "1");

        let m = PyMatrix::new(
            2,
            2,
            vec![
                PyExpr::from_expr(fsym_core::Expr::from_i64(1)),
                PyExpr::from_expr(fsym_core::Expr::from_i64(2)),
                PyExpr::from_expr(fsym_core::Expr::from_i64(3)),
                PyExpr::from_expr(fsym_core::Expr::from_i64(4)),
            ],
        )
        .unwrap();

        assert_eq!(m.det().unwrap().__str__(), "-2");
        assert_eq!(m.trace().unwrap().__str__(), "5");
        let inv = m.inv().unwrap();
        assert_eq!(inv.shape(), (2, 2));
        let prod = m.__matmul__(&inv).unwrap();
        assert_eq!(prod.flat()[0].__str__(), "1");
        assert_eq!(prod.flat()[1].__str__(), "0");
        assert_eq!(prod.flat()[2].__str__(), "0");
        assert_eq!(prod.flat()[3].__str__(), "1");
    }
}
