//! Typed scalar ODE boundaries. Symbol identities and retained expression
//! payloads never pass through the printer/parser on these routes.
//!
//! Candidate generation and exact residual admission remain in fsym-solvers;
//! this module adds no new mathematical claim or numerical fallback.

#![forbid(unsafe_code)]

use crate::expr::{PyExpr, ensure_admissible_depth};
use crate::to_value_error;
use fsym_core::{Expr, Symbol};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn symbol(value: &PyExpr) -> PyResult<&Symbol> {
    match &value.inner {
        Expr::Sym(symbol) => Ok(symbol),
        _ => Err(PyValueError::new_err(
            "ODE coordinates and integration constants must be native Symbols",
        )),
    }
}

fn admit(inputs: &[&PyExpr]) -> PyResult<()> {
    for input in inputs {
        ensure_admissible_depth(&input.inner)?;
    }
    Ok(())
}

fn lift(result: Result<Expr, fsym_solvers::SolverError>) -> PyResult<PyExpr> {
    let result = result.map_err(to_value_error)?;
    ensure_admissible_depth(&result)?;
    Ok(PyExpr::from_expr(result))
}

#[pyfunction]
fn dsolve_linear_first_order_native(
    p: &PyExpr,
    q: &PyExpr,
    x: &PyExpr,
    c1: &PyExpr,
) -> PyResult<PyExpr> {
    admit(&[p, q])?;
    lift(fsym_solvers::dsolve_linear_first_order(
        &p.inner,
        &q.inner,
        symbol(x)?,
        symbol(c1)?,
    ))
}

#[pyfunction]
fn dsolve_const_coeff_second_order_native(
    a: i64,
    b: i64,
    c: i64,
    x: &PyExpr,
    c1: &PyExpr,
    c2: &PyExpr,
) -> PyResult<PyExpr> {
    lift(fsym_solvers::dsolve_const_coeff_second_order(
        a,
        b,
        c,
        symbol(x)?,
        symbol(c1)?,
        symbol(c2)?,
    ))
}

#[pyfunction]
fn dsolve_const_coeff_second_order_nonhomogeneous_native(
    a: i64,
    b: i64,
    c: i64,
    forcing: &PyExpr,
    x: &PyExpr,
    c1: &PyExpr,
    c2: &PyExpr,
) -> PyResult<PyExpr> {
    admit(&[forcing])?;
    lift(fsym_solvers::dsolve_const_coeff_second_order_nonhomogeneous(
        a,
        b,
        c,
        &forcing.inner,
        symbol(x)?,
        symbol(c1)?,
        symbol(c2)?,
    ))
}

#[pyfunction]
fn dsolve_cauchy_euler_native(
    a: i64,
    b: i64,
    c: i64,
    x: &PyExpr,
    c1: &PyExpr,
    c2: &PyExpr,
) -> PyResult<PyExpr> {
    lift(fsym_solvers::dsolve_cauchy_euler(
        a,
        b,
        c,
        symbol(x)?,
        symbol(c1)?,
        symbol(c2)?,
    ))
}

#[pyfunction]
fn dsolve_separable_linear_native(
    coefficient: &PyExpr,
    x: &PyExpr,
    c1: &PyExpr,
) -> PyResult<PyExpr> {
    admit(&[coefficient])?;
    lift(fsym_solvers::dsolve_separable_linear(
        &coefficient.inner,
        symbol(x)?,
        symbol(c1)?,
    ))
}

#[pyfunction]
fn verify_linear_first_order_solution_native(
    solution: &PyExpr,
    p: &PyExpr,
    q: &PyExpr,
    x: &PyExpr,
) -> PyResult<bool> {
    admit(&[solution, p, q])?;
    Ok(fsym_solvers::verify_linear_first_order_solution(
        &solution.inner,
        &p.inner,
        &q.inner,
        symbol(x)?,
    ))
}

#[pyfunction]
fn verify_const_coeff_second_order_solution_native(
    solution: &PyExpr,
    a: i64,
    b: i64,
    c: i64,
    x: &PyExpr,
) -> PyResult<bool> {
    admit(&[solution])?;
    Ok(fsym_solvers::verify_const_coeff_second_order_solution(
        &solution.inner,
        a,
        b,
        c,
        symbol(x)?,
    ))
}

#[pyfunction]
fn verify_cauchy_euler_solution_native(
    solution: &PyExpr,
    a: i64,
    b: i64,
    c: i64,
    x: &PyExpr,
) -> PyResult<bool> {
    admit(&[solution])?;
    Ok(fsym_solvers::verify_cauchy_euler_solution(
        &solution.inner,
        a,
        b,
        c,
        symbol(x)?,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dsolve_linear_first_order_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        dsolve_const_coeff_second_order_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        dsolve_const_coeff_second_order_nonhomogeneous_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(dsolve_cauchy_euler_native, module)?)?;
    module.add_function(wrap_pyfunction!(dsolve_separable_linear_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        verify_linear_first_order_solution_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        verify_const_coeff_second_order_solution_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        verify_cauchy_euler_solution_native,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::SymbolIdentity;

    fn coordinate(name: &str, discriminator: u8) -> PyExpr {
        PyExpr::from_expr(Expr::Sym(Symbol::with_identity(
            name,
            SymbolIdentity::from_assumptions_digest([discriminator; 32]),
        )))
    }

    fn integer(value: i64) -> PyExpr {
        PyExpr::from_expr(Expr::from_i64(value))
    }

    #[test]
    fn typed_coordinate_and_same_name_parameter_are_not_merged() {
        let x = coordinate("x", 1);
        let parameter = coordinate("x", 2);
        let constant = coordinate("C1", 3);
        let solution =
            dsolve_linear_first_order_native(&integer(0), &parameter, &x, &constant).unwrap();
        assert!(
            verify_linear_first_order_solution_native(&solution, &integer(0), &parameter, &x)
                .unwrap()
        );
        let expected = parameter.inner.clone() * x.inner.clone() + constant.inner.clone();
        assert!(fsym_simplify::simplify(&(solution.inner.clone() - expected)).is_zero());
        let rebound = PyExpr::from_expr(x.inner.clone() * x.inner.clone() + constant.inner.clone());
        assert!(
            !verify_linear_first_order_solution_native(&rebound, &integer(0), &parameter, &x)
                .unwrap()
        );
    }

    #[test]
    fn every_scalar_generator_keeps_the_supplied_typed_coordinate() {
        let x = coordinate("x", 4);
        let c1 = coordinate("C1", 5);
        let c2 = coordinate("C2", 6);
        let homogeneous = dsolve_const_coeff_second_order_native(1, 0, -1, &x, &c1, &c2).unwrap();
        assert!(
            verify_const_coeff_second_order_solution_native(&homogeneous, 1, 0, -1, &x)
                .unwrap()
        );
        let forced = dsolve_const_coeff_second_order_nonhomogeneous_native(
            1,
            0,
            -1,
            &integer(2),
            &x,
            &c1,
            &c2,
        )
        .unwrap();
        assert!(fsym_solvers::verify_const_coeff_second_order_nonhomogeneous_solution(
            &forced.inner,
            1,
            0,
            -1,
            &Expr::from_i64(2),
            symbol(&x).unwrap(),
        ));
        let euler = dsolve_cauchy_euler_native(1, 0, 0, &x, &c1, &c2).unwrap();
        assert!(verify_cauchy_euler_solution_native(&euler, 1, 0, 0, &x).unwrap());
        let separable = dsolve_separable_linear_native(&integer(1), &x, &c1).unwrap();
        assert!(
            verify_linear_first_order_solution_native(&separable, &integer(-1), &integer(0), &x)
                .unwrap()
        );
    }

    #[test]
    fn original_undefined_terms_are_not_erased_by_typed_lowering() {
        use std::sync::Arc;

        let x = coordinate("x", 9);
        let c1 = coordinate("C1", 10);
        let hole = PyExpr::from_expr(Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
        ]));
        assert!(dsolve_linear_first_order_native(&integer(0), &hole, &x, &c1).is_err());
        assert!(
            !verify_linear_first_order_solution_native(&hole, &integer(0), &integer(0), &x)
                .unwrap()
        );
    }

    #[test]
    fn invalid_coordinates_constant_aliases_and_candidate_mutations_are_rejected() {
        let x = coordinate("x", 7);
        let c1 = coordinate("C1", 8);
        assert!(
            dsolve_linear_first_order_native(&integer(0), &integer(1), &integer(2), &c1)
                .is_err()
        );
        assert!(dsolve_linear_first_order_native(&integer(0), &integer(1), &x, &x).is_err());
        assert!(dsolve_linear_first_order_native(&integer(0), &c1, &x, &c1).is_err());
        assert!(dsolve_const_coeff_second_order_native(1, 0, 0, &x, &c1, &c1).is_err());
        let solution =
            dsolve_linear_first_order_native(&integer(0), &integer(1), &x, &c1).unwrap();
        let mutated = PyExpr::from_expr(solution.inner + x.inner.clone());
        assert!(
            !verify_linear_first_order_solution_native(&mutated, &integer(0), &integer(1), &x)
                .unwrap()
        );
    }
}
