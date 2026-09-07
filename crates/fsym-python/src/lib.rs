//! # fsym-python
//!
//! PyO3 bindings exposing FrankenSymPy to CPython as a native extension
//! module. Strings cross the boundary; everything inside is exact.

use fsym_calculus::{diff, integrate, limit, taylor};
use fsym_core::{BigInt, BigRational, Expr, Symbol, parse};
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

/// Exact Cauchy-Euler 2nd-order ODE solver: a*x^2*y'' + b*x*y' + c*y = 0.
#[pyfunction]
fn dsolve_cauchy_euler_expr(a: i64, b: i64, c: i64, x_var: &str) -> PyResult<String> {
    let c1 = Symbol::new("C1");
    let c2 = Symbol::new("C2");
    fsym_solvers::dsolve_cauchy_euler(a, b, c, &Symbol::new(x_var), &c1, &c2)
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Exact 2nd-order nonhomogeneous constant-coefficient ODE solver: a*y'' + b*y' + c*y = f(x).
#[pyfunction]
fn dsolve_const_coeff_second_order_nonhomogeneous_expr(
    a: i64,
    b: i64,
    c: i64,
    f_src: &str,
    x_var: &str,
) -> PyResult<String> {
    let f = parse_expr(f_src)?;
    let c1 = Symbol::new("C1");
    let c2 = Symbol::new("C2");
    fsym_solvers::dsolve_const_coeff_second_order_nonhomogeneous(
        a,
        b,
        c,
        &f,
        &Symbol::new(x_var),
        &c1,
        &c2,
    )
    .map(|v| v.to_string())
    .map_err(to_value_error)
}

/// Exact separable linear ODE solver: y'(x) = f(x)*y(x).
#[pyfunction]
fn dsolve_separable_linear_expr(f_src: &str, x_var: &str) -> PyResult<String> {
    let f = parse_expr(f_src)?;
    let c1 = Symbol::new("C1");
    fsym_solvers::dsolve_separable_linear(&f, &Symbol::new(x_var), &c1)
        .map(|v| v.to_string())
        .map_err(to_value_error)
}

/// Independent verifier for first-order linear ODE: y'(x) + P(x)*y(x) = Q(x).
#[pyfunction]
fn verify_linear_first_order_solution_expr(
    sol_src: &str,
    p_src: &str,
    q_src: &str,
    x_var: &str,
) -> PyResult<bool> {
    let sol = parse_expr(sol_src)?;
    let p = parse_expr(p_src)?;
    let q = parse_expr(q_src)?;
    Ok(fsym_solvers::verify_linear_first_order_solution(
        &sol,
        &p,
        &q,
        &Symbol::new(x_var),
    ))
}

/// Independent verifier for 2nd-order constant coefficient ODE: a*y''(x) + b*y'(x) + c*y(x) = 0.
#[pyfunction]
fn verify_const_coeff_second_order_solution_expr(
    sol_src: &str,
    a: i64,
    b: i64,
    c: i64,
    x_var: &str,
) -> PyResult<bool> {
    let sol = parse_expr(sol_src)?;
    Ok(fsym_solvers::verify_const_coeff_second_order_solution(
        &sol,
        a,
        b,
        c,
        &Symbol::new(x_var),
    ))
}

/// Independent verifier for Cauchy-Euler ODE: a*x^2*y''(x) + b*x*y'(x) + c*y(x) = 0.
#[pyfunction]
fn verify_cauchy_euler_solution_expr(
    sol_src: &str,
    a: i64,
    b: i64,
    c: i64,
    x_var: &str,
) -> PyResult<bool> {
    let sol = parse_expr(sol_src)?;
    Ok(fsym_solvers::verify_cauchy_euler_solution(
        &sol,
        a,
        b,
        c,
        &Symbol::new(x_var),
    ))
}

/// Exact 2-variable polynomial system solver via Lex Groebner basis.
#[pyfunction]
fn solve_poly_system_expr(
    eq_sources: Vec<String>,
    var1: &str,
    var2: &str,
) -> PyResult<Vec<(String, String)>> {
    let x = Symbol::new(var1);
    let y = Symbol::new(var2);
    let gens = vec![x.clone(), y.clone()];
    let mut polys = Vec::with_capacity(eq_sources.len());
    for s in &eq_sources {
        let e = parse_expr(s)?;
        let p = fsym_polys::multivariate::MultivariatePoly::from_expr(&e, &gens)
            .map_err(to_value_error)?;
        polys.push(p);
    }
    let solutions = fsym_solvers::solve_2var_poly_system(&polys, &x, &y).map_err(to_value_error)?;
    let mut out = Vec::with_capacity(solutions.len());
    for sol in solutions {
        let x_str = sol
            .get(&x)
            .map(|e| e.to_string())
            .unwrap_or_else(|| "0".to_string());
        let y_str = sol
            .get(&y)
            .map(|e| e.to_string())
            .unwrap_or_else(|| "0".to_string());
        out.push((x_str, y_str));
    }
    Ok(out)
}

/// Univariate polynomial coefficients in descending degree order.
#[pyfunction]
fn poly_coeffs_expr(p_src: &str, var: &str) -> PyResult<Vec<String>> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let coeffs: Vec<String> = poly
        .coeffs
        .iter()
        .rev()
        .map(|r| {
            if r.is_integer() {
                Expr::Integer(r.to_integer()).to_string()
            } else {
                Expr::Rational(r.clone()).to_string()
            }
        })
        .collect();
    Ok(coeffs)
}

/// Univariate polynomial degree (None for zero polynomial).
#[pyfunction]
fn poly_degree_expr(p_src: &str, var: &str) -> PyResult<Option<usize>> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    Ok(poly.degree())
}

/// Degree of a multivariate polynomial with respect to a target generator, or total degree if none specified.
#[pyfunction]
fn poly_multivariate_degree_expr(
    p_src: &str,
    var_names: Vec<String>,
    target_var: Option<&str>,
) -> PyResult<Option<usize>> {
    let e = parse_expr(p_src)?;
    let gens: Vec<Symbol> = var_names.into_iter().map(Symbol::new).collect();
    let poly =
        fsym_polys::multivariate::MultivariatePoly::from_expr(&e, &gens).map_err(to_value_error)?;
    if let Some(target) = target_var {
        let sym = Symbol::new(target);
        if let Some(idx) = gens.iter().position(|g| g == &sym) {
            Ok(Some(
                usize::try_from(poly.degree_in(idx)).unwrap_or(usize::MAX),
            ))
        } else {
            Ok(Some(0))
        }
    } else {
        Ok(poly
            .total_degree()
            .map(|d| usize::try_from(d).unwrap_or(usize::MAX)))
    }
}

/// Leading coefficient of a univariate polynomial.
#[pyfunction]
fn poly_leading_coeff_expr(p_src: &str, var: &str) -> PyResult<String> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let lc = poly.leading_coeff();
    let lc_expr = if lc.is_integer() {
        Expr::Integer(lc.to_integer())
    } else {
        Expr::Rational(lc.clone())
    };
    Ok(lc_expr.to_string())
}

/// Monic normalization of a univariate polynomial.
#[pyfunction]
fn poly_monic_expr(p_src: &str, var: &str) -> PyResult<String> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let monic = poly.make_monic().map_err(to_value_error)?;
    Ok(monic.to_expr().to_string())
}

/// Polynomial division with remainder returning (quotient, remainder).
#[pyfunction]
fn poly_div_rem_expr(p1_src: &str, p2_src: &str, var: &str) -> PyResult<(String, String)> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let sym = Symbol::new(var);
    let poly1 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e1, &sym).map_err(to_value_error)?;
    let poly2 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e2, &sym).map_err(to_value_error)?;
    let (q, r) = poly1.div_rem(&poly2).map_err(to_value_error)?;
    Ok((q.to_expr().to_string(), r.to_expr().to_string()))
}

/// Univariate polynomial resultant.
#[pyfunction]
fn poly_resultant_expr(p1_src: &str, p2_src: &str, var: &str) -> PyResult<String> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let sym = Symbol::new(var);
    let poly1 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e1, &sym).map_err(to_value_error)?;
    let poly2 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e2, &sym).map_err(to_value_error)?;
    let res = poly1.resultant(&poly2).map_err(to_value_error)?;
    let res_expr = if res.is_integer() {
        Expr::Integer(res.to_integer())
    } else {
        Expr::Rational(res)
    };
    Ok(res_expr.to_string())
}

/// Univariate polynomial discriminant.
#[pyfunction]
fn poly_discriminant_expr(p_src: &str, var: &str) -> PyResult<String> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let disc = poly.discriminant().map_err(to_value_error)?;
    let disc_expr = if disc.is_integer() {
        Expr::Integer(disc.to_integer())
    } else {
        Expr::Rational(disc)
    };
    Ok(disc_expr.to_string())
}

/// Univariate polynomial greatest common divisor (monic).
#[pyfunction]
fn poly_gcd_expr(p1_src: &str, p2_src: &str, var: &str) -> PyResult<String> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let sym = Symbol::new(var);
    let poly1 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e1, &sym).map_err(to_value_error)?;
    let poly2 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e2, &sym).map_err(to_value_error)?;
    let g = poly1.gcd(&poly2).map_err(to_value_error)?;
    Ok(g.to_expr().to_string())
}

/// Univariate polynomial least common multiple (monic).
#[pyfunction]
fn poly_lcm_expr(p1_src: &str, p2_src: &str, var: &str) -> PyResult<String> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let sym = Symbol::new(var);
    let poly1 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e1, &sym).map_err(to_value_error)?;
    let poly2 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e2, &sym).map_err(to_value_error)?;
    let g = poly1.gcd(&poly2).map_err(to_value_error)?;
    let lcm = if g.is_zero() {
        fsym_polys::univariate::UnivariatePoly::zero(sym)
    } else {
        let prod = poly1.mul(&poly2).map_err(to_value_error)?;
        let (q, _) = prod.div_rem(&g).map_err(to_value_error)?;
        q.to_monic()
    };
    Ok(lcm.to_expr().to_string())
}

/// Univariate polynomial extended GCD: returns (u, v, gcd) such that u*p1 + v*p2 = gcd.
#[pyfunction]
fn poly_gcdex_expr(p1_src: &str, p2_src: &str, var: &str) -> PyResult<(String, String, String)> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let sym = Symbol::new(var);
    let poly1 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e1, &sym).map_err(to_value_error)?;
    let poly2 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e2, &sym).map_err(to_value_error)?;
    let cert = poly1.extended_gcd(&poly2).map_err(to_value_error)?;
    Ok((
        cert.u.to_expr().to_string(),
        cert.v.to_expr().to_string(),
        cert.gcd.to_expr().to_string(),
    ))
}

/// Univariate polynomial composition: computes p1(p2(var)).
#[pyfunction]
fn poly_compose_expr(p1_src: &str, p2_src: &str, var: &str) -> PyResult<String> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let sym = Symbol::new(var);
    let poly1 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e1, &sym).map_err(to_value_error)?;
    let poly2 =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e2, &sym).map_err(to_value_error)?;
    let comp = poly1.compose(&poly2).map_err(to_value_error)?;
    Ok(comp.to_expr().to_string())
}

/// Univariate polynomial shift: computes p(var + a) where a is rational.
#[pyfunction]
fn poly_shift_expr(p_src: &str, a_src: &str, var: &str) -> PyResult<String> {
    let e = parse_expr(p_src)?;
    let a_expr = parse_expr(a_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let a_rat = match a_expr {
        Expr::Integer(i) => BigRational::from_integer(i),
        Expr::Rational(r) => r,
        _ => {
            return Err(PyValueError::new_err(
                "shift amount must be an exact integer or rational",
            ));
        }
    };
    let shifted = poly.shift(&a_rat).map_err(to_value_error)?;
    Ok(shifted.to_expr().to_string())
}

/// Univariate polynomial square-free factorization returning (scale, [(factor, multiplicity), ...]).
#[pyfunction]
fn poly_sqf_list_expr(p_src: &str, var: &str) -> PyResult<(String, Vec<(String, usize)>)> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let result =
        fsym_polys::factorization::square_free_decomposition(&poly).map_err(to_value_error)?;
    let scale_expr = if result.scale.is_integer() {
        Expr::Integer(result.scale.to_integer())
    } else {
        Expr::Rational(result.scale)
    };
    let factors = result
        .factors
        .into_iter()
        .map(|f| (f.poly.to_expr().to_string(), f.multiplicity))
        .collect();
    Ok((scale_expr.to_string(), factors))
}

/// Univariate polynomial factorization into rational factors: (scale, [(factor, multiplicity), ...]).
#[pyfunction]
fn poly_factor_list_expr(p_src: &str, var: &str) -> PyResult<(String, Vec<(String, usize)>)> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let result = fsym_polys::factorization::bounded_rational_root_decomposition(&poly)
        .map_err(to_value_error)?;
    let scale_expr = if result.scale.is_integer() {
        Expr::Integer(result.scale.to_integer())
    } else {
        Expr::Rational(result.scale)
    };
    let factors = result
        .factors
        .into_iter()
        .map(|f| (f.poly.to_expr().to_string(), f.multiplicity))
        .collect();
    Ok((scale_expr.to_string(), factors))
}

/// Univariate polynomial roots with multiplicities: [(root, multiplicity), ...].
#[pyfunction]
fn poly_roots_expr(p_src: &str, var: &str) -> PyResult<Vec<(String, usize)>> {
    let e = parse_expr(p_src)?;
    let sym = Symbol::new(var);
    let poly =
        fsym_polys::univariate::UnivariatePoly::from_expr(&e, &sym).map_err(to_value_error)?;
    let decomp = fsym_polys::factorization::bounded_rational_root_decomposition(&poly)
        .map_err(to_value_error)?;
    let mut roots = Vec::new();
    for factor in decomp.factors {
        let deg = factor.poly.degree().unwrap_or(0);
        match (deg, factor.poly.coeffs.as_slice()) {
            (1, [c0, c1, ..]) => {
                let root = -c0 / c1;
                let root_expr = if root.is_integer() {
                    Expr::Integer(root.to_integer())
                } else {
                    Expr::Rational(root)
                };
                roots.push((root_expr.to_string(), factor.multiplicity));
            }
            (2, [c0, c1, c2, ..]) => {
                let four = fsym_core::BigRational::from_integer(fsym_core::BigInt::from(4));
                let discr = c1 * c1 - four * c2 * c0;
                let zero = fsym_core::BigRational::from_integer(fsym_core::BigInt::from(0));
                if discr >= zero {
                    let maybe_sqrt = match (discr.numer().sqrt(), discr.denom().sqrt()) {
                        (Some(num_s), Some(den_s))
                            if &num_s * &num_s == *discr.numer()
                                && &den_s * &den_s == *discr.denom() =>
                        {
                            Some(fsym_core::BigRational::new(num_s, den_s))
                        }
                        _ => None,
                    };
                    if let Some(sqrt_d) = maybe_sqrt {
                        let two = fsym_core::BigRational::from_integer(fsym_core::BigInt::from(2));
                        let two_a = two * c2;
                        let r1 = (-c1 + &sqrt_d) / &two_a;
                        let r2 = (-c1 - &sqrt_d) / &two_a;
                        let same = r1 == r2;
                        let r1_expr = if r1.is_integer() {
                            Expr::Integer(r1.to_integer())
                        } else {
                            Expr::Rational(r1)
                        };
                        roots.push((r1_expr.to_string(), factor.multiplicity));
                        if !same {
                            let r2_expr = if r2.is_integer() {
                                Expr::Integer(r2.to_integer())
                            } else {
                                Expr::Rational(r2)
                            };
                            roots.push((r2_expr.to_string(), factor.multiplicity));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(roots)
}

/// Multivariate Groebner basis under Lex order.
#[pyfunction]
fn groebner_basis_expr(eq_sources: Vec<String>, var_names: Vec<String>) -> PyResult<Vec<String>> {
    let gens: Vec<Symbol> = var_names.into_iter().map(Symbol::new).collect();
    let mut polys = Vec::with_capacity(eq_sources.len());
    for s in &eq_sources {
        let e = parse_expr(s)?;
        let p = fsym_polys::multivariate::MultivariatePoly::from_expr(&e, &gens)
            .map_err(to_value_error)?;
        polys.push(p);
    }
    let basis = fsym_polys::groebner::groebner_basis(&polys, fsym_polys::TermOrder::Lex)
        .map_err(to_value_error)?;
    let mut out = Vec::with_capacity(basis.len());
    for p in basis {
        let expr = p.to_expr().map_err(to_value_error)?;
        out.push(expr.to_string());
    }
    Ok(out)
}

/// Multivariate polynomial GCD (monic under Lex order).
#[pyfunction]
fn poly_multivariate_gcd_expr(
    p1_src: &str,
    p2_src: &str,
    var_names: Vec<String>,
) -> PyResult<String> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let gens: Vec<Symbol> = var_names.into_iter().map(Symbol::new).collect();
    let poly1 = fsym_polys::multivariate::MultivariatePoly::from_expr(&e1, &gens)
        .map_err(to_value_error)?;
    let poly2 = fsym_polys::multivariate::MultivariatePoly::from_expr(&e2, &gens)
        .map_err(to_value_error)?;
    let g = poly1.gcd(&poly2).map_err(to_value_error)?;
    let expr = g.to_expr().map_err(to_value_error)?;
    Ok(expr.to_string())
}

/// Multivariate polynomial LCM (monic under Lex order).
#[pyfunction]
fn poly_multivariate_lcm_expr(
    p1_src: &str,
    p2_src: &str,
    var_names: Vec<String>,
) -> PyResult<String> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let gens: Vec<Symbol> = var_names.into_iter().map(Symbol::new).collect();
    let poly1 = fsym_polys::multivariate::MultivariatePoly::from_expr(&e1, &gens)
        .map_err(to_value_error)?;
    let poly2 = fsym_polys::multivariate::MultivariatePoly::from_expr(&e2, &gens)
        .map_err(to_value_error)?;
    let lcm = poly1.lcm(&poly2).map_err(to_value_error)?;
    let expr = lcm.to_expr().map_err(to_value_error)?;
    Ok(expr.to_string())
}

/// Multivariate polynomial division with remainder returning (quotient, remainder).
#[pyfunction]
fn poly_multivariate_div_rem_expr(
    p1_src: &str,
    p2_src: &str,
    var_names: Vec<String>,
) -> PyResult<(String, String)> {
    let e1 = parse_expr(p1_src)?;
    let e2 = parse_expr(p2_src)?;
    let gens: Vec<Symbol> = var_names.into_iter().map(Symbol::new).collect();
    let poly1 = fsym_polys::multivariate::MultivariatePoly::from_expr(&e1, &gens)
        .map_err(to_value_error)?;
    let poly2 = fsym_polys::multivariate::MultivariatePoly::from_expr(&e2, &gens)
        .map_err(to_value_error)?;
    let (quotients, rem) = poly1
        .div_rem(&[poly2], fsym_polys::TermOrder::Lex)
        .map_err(to_value_error)?;
    let q_expr = quotients
        .first()
        .cloned()
        .unwrap_or_else(|| fsym_polys::multivariate::MultivariatePoly::zero(gens.clone()))
        .to_expr()
        .map_err(to_value_error)?;
    let r_expr = rem.to_expr().map_err(to_value_error)?;
    Ok((q_expr.to_string(), r_expr.to_string()))
}

/// Algebraic equation solver (linear, quadratic, factorable higher-degree polynomial).
#[pyfunction]
fn solve_expr(src: &str, var: &str) -> PyResult<Vec<String>> {
    let e = parse_expr(src)?;
    fsym_solvers::solve(&e, &Symbol::new(var))
        .map(|roots| roots.into_iter().map(|r| r.to_string()).collect())
        .map_err(to_value_error)
}

/// Fourier transform of `src(t)` to `omega`.
#[pyfunction]
fn fourier_expr(src: &str, t_var: &str, omega_var: &str) -> PyResult<String> {
    let e = parse_expr(src)?;
    fsym_calculus::fourier_transform(&e, &Symbol::new(t_var), &Symbol::new(omega_var))
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

/// Legendre symbol (a/p) for odd prime p.
#[pyfunction]
fn legendre_symbol_fn(a: i64, p: u64) -> PyResult<i64> {
    fsym_ntheory::legendre_symbol(a, p).map_err(to_value_error)
}

/// Square-free check for integer n.
#[pyfunction]
fn is_square_free_fn(n: u64) -> PyResult<bool> {
    fsym_ntheory::is_square_free(n).map_err(to_value_error)
}

/// Number of distinct prime factors of n: ω(n).
#[pyfunction]
fn prime_omega_fn(n: u64) -> PyResult<u32> {
    fsym_ntheory::prime_omega(n).map_err(to_value_error)
}

/// Total number of prime factors of n counted with multiplicity: Ω(n).
#[pyfunction]
fn prime_big_omega_fn(n: u64) -> PyResult<u32> {
    fsym_ntheory::prime_big_omega(n).map_err(to_value_error)
}

/// Perfect number check: sum of proper divisors equals n.
#[pyfunction]
fn is_perfect_number_fn(n: u64) -> PyResult<bool> {
    fsym_ntheory::is_perfect_number(n).map_err(to_value_error)
}

/// Carmichael lambda function λ(n).
#[pyfunction]
fn carmichael_fn(n: u64) -> PyResult<u64> {
    fsym_ntheory::carmichael(n).map_err(to_value_error)
}

/// Primitive root check: whether g is a primitive root modulo p.
#[pyfunction]
fn is_primitive_root_fn(g: u64, p: u64) -> PyResult<bool> {
    fsym_ntheory::is_primitive_root(g, p).map_err(to_value_error)
}

/// Integer floor of k-th root of n.
#[pyfunction]
fn integer_nth_root_fn(n: u64, k: u32) -> PyResult<u64> {
    fsym_ntheory::integer_nth_root(n, k).map_err(to_value_error)
}

/// Modular inverse of a modulo m using extended GCD.
#[pyfunction]
fn mod_inverse_fn(a_str: &str, m_str: &str) -> PyResult<String> {
    let a_big = a_str.trim().parse::<BigInt>().map_err(to_value_error)?;
    let m_big = m_str.trim().parse::<BigInt>().map_err(to_value_error)?;
    match fsym_ntheory::mod_inverse(&a_big, &m_big) {
        Some(inv) => Ok(inv.to_string()),
        None => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "inverse of {a_str} mod {m_str} does not exist"
        ))),
    }
}

/// Chinese Remainder Theorem: solves x = r_i (mod m_i).
#[pyfunction]
fn crt_fn(moduli: Vec<String>, remainders: Vec<String>) -> PyResult<Option<(String, String)>> {
    let mut r_bigs = Vec::with_capacity(remainders.len());
    for r in remainders {
        r_bigs.push(r.trim().parse::<BigInt>().map_err(to_value_error)?);
    }
    let mut m_bigs = Vec::with_capacity(moduli.len());
    let mut total_mod = BigInt::from(1);
    for m in moduli {
        let m_val = m.trim().parse::<BigInt>().map_err(to_value_error)?;
        total_mod = &total_mod * &m_val;
        m_bigs.push(m_val);
    }
    match fsym_ntheory::crt(&r_bigs, &m_bigs) {
        Ok(sol) => Ok(Some((sol.to_string(), total_mod.to_string()))),
        Err(_) => Ok(None),
    }
}

pub mod assumptions;
pub mod expr;
pub mod geometry;
pub mod logic;
pub mod matrix;
pub mod sets;
pub mod tensor;
pub use assumptions::*;
pub use expr::*;
pub use geometry::*;
pub use logic::*;
pub use matrix::*;
pub use sets::*;
pub use tensor::*;

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
    m.add_class::<PySparseMatrix>()?;
    m.add_class::<PyBoolExpr>()?;
    m.add_class::<PyPoint2D>()?;
    m.add_class::<PyPoint3D>()?;
    m.add_class::<PySegment2D>()?;
    m.add_class::<PySegment3D>()?;
    m.add_class::<PyLine2D>()?;
    m.add_class::<PyLine3D>()?;
    m.add_class::<PyRay2D>()?;
    m.add_class::<PyRay3D>()?;
    m.add_class::<PyCircle>()?;
    m.add_class::<PySphere>()?;
    m.add_class::<PyTriangle2D>()?;
    m.add_class::<PyPolygon2D>()?;
    m.add_class::<PyPlane3D>()?;
    m.add_class::<PySymSet>()?;
    m.add_class::<PyTensorIndex>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyAssumptionsContext>()?;
    m.add_function(wrap_pyfunction!(ask_expr, m)?)?;
    m.add_function(wrap_pyfunction!(inherent_facts_expr, m)?)?;
    m.add_function(wrap_pyfunction!(predicate_closure, m)?)?;
    m.add_function(wrap_pyfunction!(predicate_contradictions, m)?)?;
    m.add_function(wrap_pyfunction!(py_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(py_integer_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(py_rational_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(py_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_mul, m)?)?;
    m.add_function(wrap_pyfunction!(py_pow, m)?)?;
    m.add_function(wrap_pyfunction!(py_function, m)?)?;
    m.add_function(wrap_pyfunction!(py_abs, m)?)?;
    m.add_function(wrap_pyfunction!(py_sin, m)?)?;
    m.add_function(wrap_pyfunction!(py_cos, m)?)?;
    m.add_function(wrap_pyfunction!(py_tan, m)?)?;
    m.add_function(wrap_pyfunction!(py_asin, m)?)?;
    m.add_function(wrap_pyfunction!(py_atan, m)?)?;
    m.add_function(wrap_pyfunction!(py_sinh, m)?)?;
    m.add_function(wrap_pyfunction!(py_cosh, m)?)?;
    m.add_function(wrap_pyfunction!(py_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(py_floor, m)?)?;
    m.add_function(wrap_pyfunction!(py_ceiling, m)?)?;
    m.add_function(wrap_pyfunction!(py_factorial, m)?)?;
    m.add_function(wrap_pyfunction!(py_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(py_fibonacci, m)?)?;
    m.add_function(wrap_pyfunction!(py_cot, m)?)?;
    m.add_function(wrap_pyfunction!(py_sec, m)?)?;
    m.add_function(wrap_pyfunction!(py_csc, m)?)?;
    m.add_function(wrap_pyfunction!(py_acos, m)?)?;
    m.add_function(wrap_pyfunction!(py_acot, m)?)?;
    m.add_function(wrap_pyfunction!(py_asec, m)?)?;
    m.add_function(wrap_pyfunction!(py_acsc, m)?)?;
    m.add_function(wrap_pyfunction!(py_coth, m)?)?;
    m.add_function(wrap_pyfunction!(py_sech, m)?)?;
    m.add_function(wrap_pyfunction!(py_csch, m)?)?;
    m.add_function(wrap_pyfunction!(py_asinh, m)?)?;
    m.add_function(wrap_pyfunction!(py_acosh, m)?)?;
    m.add_function(wrap_pyfunction!(py_atanh, m)?)?;
    m.add_function(wrap_pyfunction!(py_acoth, m)?)?;
    m.add_function(wrap_pyfunction!(py_asech, m)?)?;
    m.add_function(wrap_pyfunction!(py_acsch, m)?)?;
    m.add_function(wrap_pyfunction!(py_sinc, m)?)?;
    m.add_function(wrap_pyfunction!(py_erf, m)?)?;
    m.add_function(wrap_pyfunction!(py_erfc, m)?)?;
    m.add_function(wrap_pyfunction!(py_sign, m)?)?;
    m.add_function(wrap_pyfunction!(py_binomial, m)?)?;
    m.add_function(wrap_pyfunction!(py_lucas, m)?)?;
    m.add_function(wrap_pyfunction!(py_harmonic, m)?)?;
    m.add_function(wrap_pyfunction!(py_catalan, m)?)?;
    m.add_function(wrap_pyfunction!(py_bernoulli, m)?)?;
    m.add_function(wrap_pyfunction!(py_bell, m)?)?;
    m.add_function(wrap_pyfunction!(py_subfactorial, m)?)?;
    m.add_function(wrap_pyfunction!(py_zeta, m)?)?;
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
    m.add_function(wrap_pyfunction!(fourier_expr, m)?)?;
    m.add_function(wrap_pyfunction!(limit_expr, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_expr, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear_expr, m)?)?;
    m.add_function(wrap_pyfunction!(solve_expr, m)?)?;
    m.add_function(wrap_pyfunction!(dsolve_linear_first_order_expr, m)?)?;
    m.add_function(wrap_pyfunction!(dsolve_const_coeff_second_order_expr, m)?)?;
    m.add_function(wrap_pyfunction!(dsolve_cauchy_euler_expr, m)?)?;
    m.add_function(wrap_pyfunction!(
        dsolve_const_coeff_second_order_nonhomogeneous_expr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dsolve_separable_linear_expr, m)?)?;
    m.add_function(wrap_pyfunction!(
        verify_linear_first_order_solution_expr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        verify_const_coeff_second_order_solution_expr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(verify_cauchy_euler_solution_expr, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poly_system_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_coeffs_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_degree_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_multivariate_degree_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_leading_coeff_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_monic_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_div_rem_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_resultant_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_discriminant_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_gcd_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_gcdex_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_lcm_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_compose_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_shift_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_sqf_list_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_factor_list_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_roots_expr, m)?)?;
    m.add_function(wrap_pyfunction!(groebner_basis_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_multivariate_gcd_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_multivariate_lcm_expr, m)?)?;
    m.add_function(wrap_pyfunction!(poly_multivariate_div_rem_expr, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_fn, m)?)?;
    m.add_function(wrap_pyfunction!(divisor_count_fn, m)?)?;
    m.add_function(wrap_pyfunction!(divisor_sum_fn, m)?)?;
    m.add_function(wrap_pyfunction!(jacobi_symbol_fn, m)?)?;
    m.add_function(wrap_pyfunction!(legendre_symbol_fn, m)?)?;
    m.add_function(wrap_pyfunction!(is_square_free_fn, m)?)?;
    m.add_function(wrap_pyfunction!(prime_omega_fn, m)?)?;
    m.add_function(wrap_pyfunction!(prime_big_omega_fn, m)?)?;
    m.add_function(wrap_pyfunction!(is_perfect_number_fn, m)?)?;
    m.add_function(wrap_pyfunction!(carmichael_fn, m)?)?;
    m.add_function(wrap_pyfunction!(is_primitive_root_fn, m)?)?;
    m.add_function(wrap_pyfunction!(integer_nth_root_fn, m)?)?;
    m.add_function(wrap_pyfunction!(mod_inverse_fn, m)?)?;
    m.add_function(wrap_pyfunction!(crt_fn, m)?)?;
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
        assert_eq!(py_sec(py_integer(0)).__str__(), "1");
        assert_eq!(py_sech(py_integer(0)).__str__(), "1");
        assert_eq!(py_erf(py_integer(0)).__str__(), "0");
        assert_eq!(py_erfc(py_integer(0)).__str__(), "1");
        assert_eq!(py_binomial(py_integer(5), py_integer(2)).__str__(), "10");
        assert_eq!(py_lucas(py_integer(4)).__str__(), "7");
        assert_eq!(py_catalan(py_integer(3)).__str__(), "5");
        assert_eq!(py_subfactorial(py_integer(4)).__str__(), "9");
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

        let neg = m.__neg__().unwrap();
        assert_eq!(neg.flat()[0].__str__(), "-1");
        assert_eq!(neg.flat()[3].__str__(), "-4");

        Python::initialize();
        Python::attach(|py| {
            let m_bound = Bound::new(py, m.clone()).unwrap();
            let eq_res = m
                .__richcmp__(m_bound.as_any(), pyo3::basic::CompareOp::Eq)
                .unwrap();
            assert!(eq_res);

            let two = PyInt::new(py, 2);
            let scaled = m.__mul__(two.as_any()).unwrap();
            assert_eq!(scaled.flat()[0].__str__(), "2");

            let r_scaled = m.__rmul__(two.as_any()).unwrap();
            assert_eq!(r_scaled.flat()[0].__str__(), "2");

            let div = m.__truediv__(two.as_any()).unwrap();
            assert_eq!(div.flat()[0].__str__(), "1/2");
            assert_eq!(div.flat()[1].__str__(), "1");
        });

        // Test to_sparse conversion and PySparseMatrix
        let sp = m.to_sparse().unwrap();
        assert_eq!(sp.shape(), (2, 2));
        assert_eq!(sp.nnz(), 4);
        assert_eq!(sp.__len__(), 4);
        let dense_roundtrip = sp.to_dense().unwrap();
        assert_eq!(dense_roundtrip.flat()[0].__str__(), "1");
        assert_eq!(dense_roundtrip.flat()[3].__str__(), "4");

        let sp_eye = PySparseMatrix::eye(3).unwrap();
        assert_eq!(sp_eye.shape(), (3, 3));
        assert_eq!(sp_eye.nnz(), 3);
        assert_eq!(sp_eye.trace().unwrap().__str__(), "3");

        // Test Jacobian
        let x2 = PyExpr::from_expr(fsym_core::Expr::pow(
            fsym_core::Expr::symbol("x"),
            fsym_core::Expr::from_i64(2),
        ));
        let xy = PyExpr::from_expr(fsym_core::Expr::Mul(vec![
            fsym_core::Expr::symbol("x"),
            fsym_core::Expr::symbol("y"),
        ]));
        let j = PyMatrix::jacobian(vec![x2, xy], vec!["x".to_string(), "y".to_string()]).unwrap();
        assert_eq!(j.shape(), (2, 2));
        assert_eq!(j.flat()[0].__str__(), "2*x");
        assert_eq!(j.flat()[1].__str__(), "0");
        assert_eq!(j.flat()[2].__str__(), "y");
        assert_eq!(j.flat()[3].__str__(), "x");
    }

    #[test]
    fn test_py_bool_expr() {
        let x = PyBoolExpr::bool_var("x");
        let y = PyBoolExpr::bool_var("y");
        let not_x = PyBoolExpr::bool_not(&x);
        let and_expr = PyBoolExpr::bool_and(vec![x.clone(), y.clone()]);
        let or_expr = PyBoolExpr::bool_or(vec![x.clone(), not_x.clone()]);

        assert_eq!(and_expr.kind(), "And");
        assert_eq!(and_expr.args().len(), 2);
        assert_eq!(x.var_name(), Some("x".to_string()));

        // Tautology: x | ~x simplifies to true
        let simplified = or_expr.simplify();
        assert_eq!(simplified.const_value(), Some(true));

        // Satisfiability: x & y is satisfiable
        assert!(and_expr.is_satisfiable().unwrap());
        let model = and_expr.satisfiable().unwrap().unwrap();
        assert_eq!(model.get("x"), Some(&true));
        assert_eq!(model.get("y"), Some(&true));

        // Contradiction: x & ~x
        let contra = PyBoolExpr::bool_and(vec![x.clone(), not_x.clone()]);
        assert!(!contra.is_satisfiable().unwrap());
        assert_eq!(contra.satisfiable().unwrap(), None);

        // CNF and DNF
        let implies = PyBoolExpr::bool_implies(&x, &y);
        let cnf = implies.to_cnf().unwrap();
        assert!(cnf.is_satisfiable().unwrap());
    }

    #[test]
    fn test_py_geometry() {
        let p1 = PyPoint2D::new(py_integer(0), py_integer(0));
        let p2 = PyPoint2D::new(py_integer(3), py_integer(4));
        assert_eq!(p1.distance_squared(&p2).to_string(), "25");

        let mid = p1.midpoint(&p2);
        assert_eq!(mid.x().to_string(), "3/2");
        assert_eq!(mid.y().to_string(), "2");

        let seg = PySegment2D::new(p1.clone(), p2.clone());
        assert_eq!(seg.length_squared().to_string(), "25");

        let p3 = PyPoint2D::new(py_integer(3), py_integer(0));
        let tri = PyTriangle2D::new(p1.clone(), p3.clone(), p2.clone());
        assert_eq!(tri.is_right(), Some(true));
        assert_eq!(tri.double_signed_area().to_string(), "12");

        let circle = PyCircle::new(p1.clone(), py_integer(5)).unwrap();
        assert_eq!(circle.area().to_string(), "25*pi");
        assert_eq!(circle.circumference().to_string(), "10*pi");

        let poly = PyPolygon2D::new(vec![
            PyPoint2D::new(py_integer(0), py_integer(0)),
            PyPoint2D::new(py_integer(4), py_integer(0)),
            PyPoint2D::new(py_integer(4), py_integer(3)),
            PyPoint2D::new(py_integer(0), py_integer(3)),
        ])
        .unwrap();
        assert_eq!(poly.double_signed_area().to_string(), "24");
        assert_eq!(poly.is_convex(), Some(true));

        // 3D, rays, spheres, planes
        let p3d1 = PyPoint3D::new(py_integer(1), py_integer(2), py_integer(3));
        let p3d2 = PyPoint3D::new(py_integer(4), py_integer(6), py_integer(3));
        assert_eq!(p3d1.distance_squared(&p3d2).to_string(), "25");

        let seg3d = PySegment3D::new(p3d1.clone(), p3d2.clone());
        assert_eq!(seg3d.length_squared().to_string(), "25");

        let sphere = PySphere::new(p3d1.clone(), py_integer(3)).unwrap();
        assert_eq!(sphere.surface_area().to_string(), "36*pi");
        assert_eq!(sphere.volume().to_string(), "36*pi");

        let plane = PyPlane3D::new(
            p3d1.clone(),
            PyPoint3D::new(py_integer(0), py_integer(0), py_integer(1)),
        )
        .unwrap();
        assert_eq!(plane.eval_at_point(&p3d1).to_string(), "0");
    }

    #[test]
    fn test_py_sets() {
        let empty = PySymSet::empty();
        assert_eq!(empty.kind(), "EmptySet");
        assert!(empty.is_empty_set().unwrap());

        let univ = PySymSet::universal();
        assert_eq!(univ.kind(), "UniversalSet");
        assert!(!univ.is_empty_set().unwrap());

        let iv = PySymSet::interval(py_integer(0), py_integer(5), false, false).unwrap();
        assert_eq!(iv.kind(), "Interval");
        assert_eq!(iv.start().unwrap().to_string(), "0");
        assert_eq!(iv.end().unwrap().to_string(), "5");
        assert!(!iv.left_open().unwrap());
        assert!(!iv.right_open().unwrap());
        assert_eq!(iv.measure().unwrap().to_string(), "5");
        assert!(iv.contains(&py_integer(3)).unwrap());
        assert!(!iv.contains(&py_integer(7)).unwrap());

        let finite = PySymSet::finite(vec![py_integer(1), py_integer(2), py_integer(3)]);
        assert_eq!(finite.kind(), "FiniteSet");
        assert!(finite.contains(&py_integer(2)).unwrap());
        assert!(!finite.contains(&py_integer(4)).unwrap());

        let u = iv.union(&finite);
        assert_eq!(u.kind(), "Union");

        let inter = iv.intersection(&finite);
        assert!(inter.contains(&py_integer(2)).unwrap());
    }

    #[test]
    fn test_ode_extended_and_poly_system() {
        // Nonhomogeneous ODE: y'' - 3*y' + 2*y = 4
        let ode_nonhom =
            dsolve_const_coeff_second_order_nonhomogeneous_expr(1, -3, 2, "4", "x").unwrap();
        assert!(ode_nonhom.contains("C1") && ode_nonhom.contains("C2"));

        // Separable ODE: y' = 2*x*y
        let ode_sep = dsolve_separable_linear_expr("2*x", "x").unwrap();
        assert!(ode_sep.contains("C1"));

        // 2-variable poly system: x + y - 5 = 0, x - y - 1 = 0
        let sols = solve_poly_system_expr(
            vec!["x + y - 5".to_string(), "x - y - 1".to_string()],
            "x",
            "y",
        )
        .unwrap();
        assert_eq!(sols, vec![("3".to_string(), "2".to_string())]);
    }

    #[test]
    fn test_poly_bindings() {
        // x^2 - 4
        let coeffs = poly_coeffs_expr("x**2 - 4", "x").unwrap();
        assert_eq!(coeffs, vec!["1", "0", "-4"]);

        let deg = poly_degree_expr("x**2 - 4", "x").unwrap();
        assert_eq!(deg, Some(2));

        let lc = poly_leading_coeff_expr("3*x**2 - 4", "x").unwrap();
        assert_eq!(lc, "3");

        let monic = poly_monic_expr("2*x**2 - 8", "x").unwrap();
        assert_eq!(parse(&monic).unwrap(), parse("x**2 - 4").unwrap());

        let (q, r) = poly_div_rem_expr("x**2 - 1", "x - 1", "x").unwrap();
        assert_eq!(parse(&q).unwrap(), parse("x + 1").unwrap());
        assert_eq!(r, "0");

        let disc = poly_discriminant_expr("x**2 - 4", "x").unwrap();
        assert_eq!(disc, "16");

        let res = poly_resultant_expr("x - 2", "x - 3", "x").unwrap();
        assert_eq!(res, "-1");

        let gcd_res = poly_gcd_expr("x**2 - 1", "x - 1", "x").unwrap();
        assert_eq!(parse(&gcd_res).unwrap(), parse("x - 1").unwrap());

        let lcm_res = poly_lcm_expr("x - 1", "x + 1", "x").unwrap();
        assert_eq!(parse(&lcm_res).unwrap(), parse("x**2 - 1").unwrap());

        let (scale, factors) = poly_sqf_list_expr("(x - 1)**2 * (x + 2)", "x").unwrap();
        assert_eq!(scale, "1");
        let factor_exprs: Vec<(Expr, usize)> = factors
            .into_iter()
            .map(|(s, m)| (parse(&s).unwrap(), m))
            .collect();
        assert!(factor_exprs.contains(&(parse("x - 1").unwrap(), 2)));
        assert!(factor_exprs.contains(&(parse("x + 2").unwrap(), 1)));

        let gb = groebner_basis_expr(
            vec!["x*y - 2*y".to_string(), "2*y**2 - x**2".to_string()],
            vec!["x".to_string(), "y".to_string()],
        )
        .unwrap();
        assert!(!gb.is_empty());

        // Factor list: (x - 2) * (x + 2)
        let (scale, factors) = poly_factor_list_expr("x**2 - 4", "x").unwrap();
        assert_eq!(scale, "1");
        assert_eq!(factors.len(), 2);

        // Roots: x^2 - 4 => roots 2, -2 with mult 1
        let roots = poly_roots_expr("x**2 - 4", "x").unwrap();
        assert_eq!(roots.len(), 2);
        let root_vals: Vec<String> = roots.into_iter().map(|(r, _)| r).collect();
        assert!(root_vals.contains(&"2".to_string()));
        assert!(root_vals.contains(&"-2".to_string()));
    }

    #[test]
    fn test_tensor_bindings() {
        let mu = PyTensorIndex::upper("mu");
        assert!(mu.is_up());
        assert_eq!(mu.name(), "mu");
        let mu_low = mu.flip();
        assert!(!mu_low.is_up());

        let eta = PyMetric::minkowski_4d("eta");
        assert_eq!(eta.dimension(), 4);
        assert_eq!(eta.matrix().len(), 16);

        let v = PyTensor::new(
            "v",
            4,
            vec![PyTensorIndex::upper("mu")],
            Some(vec![
                PyExpr::from_expr(Expr::from_i64(3)),
                PyExpr::from_expr(Expr::from_i64(0)),
                PyExpr::from_expr(Expr::from_i64(0)),
                PyExpr::from_expr(Expr::from_i64(4)),
            ]),
        )
        .unwrap();
        assert_eq!(v.rank(), 1);

        let v_low = eta.lower_vector(&v).unwrap();
        assert_eq!(v_low.rank(), 1);
        assert!(!v_low.indices()[0].is_up());

        let norm_sq = eta.norm_squared(&v).unwrap();
        // -3^2 + 0 + 0 + 4^2 = -9 + 16 = 7
        assert_eq!(norm_sq.inner, Expr::from_i64(7));
    }

    #[test]
    fn test_assumptions_bindings() {
        let x_expr = PyExpr::from_expr(Expr::Sym(Symbol::new("x")));
        let five_expr = PyExpr::from_expr(Expr::from_i64(5));

        // Inherent facts on number 5: positive, integer, real, odd, etc.
        let facts = inherent_facts_expr(&five_expr);
        assert!(facts.contains(&"positive".to_string()));
        assert!(facts.contains(&"integer".to_string()));
        assert!(facts.contains(&"odd".to_string()));

        // Inherent ask on 5
        let is_pos = ask_expr(&five_expr, "positive", None).unwrap();
        assert_eq!(is_pos, Some(true));
        let is_neg = ask_expr(&five_expr, "negative", None).unwrap();
        assert_eq!(is_neg, Some(false));

        // Query on symbol x with positive assumption:
        // positive => real (entailed true)
        let is_real = ask_expr(
            &x_expr,
            "real",
            Some(vec![("x".to_string(), "positive".to_string())]),
        )
        .unwrap();
        assert_eq!(is_real, Some(true));

        // positive => negative (contradicted false)
        let is_neg = ask_expr(
            &x_expr,
            "negative",
            Some(vec![("x".to_string(), "positive".to_string())]),
        )
        .unwrap();
        assert_eq!(is_neg, Some(false));

        // positive => integer (unknown None)
        let is_int = ask_expr(
            &x_expr,
            "integer",
            Some(vec![("x".to_string(), "positive".to_string())]),
        )
        .unwrap();
        assert_eq!(is_int, None);

        // AssumptionsContext
        let mut ctx = PyAssumptionsContext::new();
        ctx.assume("x", "positive").unwrap();
        assert_eq!(ctx.is_true(&x_expr, "real").unwrap(), Some(true));
        assert_eq!(ctx.is_true(&x_expr, "negative").unwrap(), Some(false));
        assert_eq!(ctx.is_true(&x_expr, "integer").unwrap(), None);
        assert_eq!(ctx.query(&x_expr, "real").unwrap(), "True");
        assert_eq!(ctx.query(&x_expr, "negative").unwrap(), "False");
        assert_eq!(ctx.query(&x_expr, "integer").unwrap(), "Unknown");

        // Closure and contradictions
        let closure = predicate_closure("positive").unwrap();
        assert!(closure.contains(&"real".to_string()));
        assert!(closure.contains(&"complex".to_string()));

        let contras = predicate_contradictions("positive").unwrap();
        assert!(contras.contains(&"negative".to_string()));
        assert!(contras.contains(&"zero".to_string()));
    }
}
