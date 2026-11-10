//! Typed bounded bridge for checked first-order characteristic families.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_budget::{Budget, BudgetLimits};
use fsym_core::{BigRational, Expr};
use fsym_solvers::pde::metered_linear_first_order_pde;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn rational(expression: &Expr) -> PyResult<BigRational> {
    match expression {
        Expr::Integer(value) if value.bits() <= 1_024 => {
            Ok(BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value)
            if value.numer().bits() <= 1_024 && value.denom().bits() <= 1_024 =>
        {
            Ok(value.clone())
        }
        _ => Err(PyValueError::new_err(
            "PDE coefficients must be exact rational literals of at most 1024 bits",
        )),
    }
}

/// Borrow source arguments before bounded native admission. The receiving
/// solver independently checks the original polynomial and characteristic
/// identities before these four expressions can reach the Python shell.
#[pyfunction]
pub fn linear_first_order_pde_native(
    a: &PyExpr,
    b: &PyExpr,
    c: &PyExpr,
    forcing: &PyExpr,
    x: &PySymbolBinding,
    y: &PySymbolBinding,
) -> PyResult<(PyExpr, PyExpr, PyExpr, PyExpr)> {
    if x.name.len().saturating_add(y.name.len()) > 16_384 {
        return Err(PyValueError::new_err(
            "PDE coordinate identifiers exceed the text bound",
        ));
    }
    let coefficients = [rational(&a.inner)?, rational(&b.inner)?, rational(&c.inner)?];
    let variables = [x.to_symbol(), y.to_symbol()];
    let mut meter = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let checked = metered_linear_first_order_pde(
        &coefficients[0],
        &coefficients[1],
        &coefficients[2],
        &forcing.inner,
        &variables[0],
        &variables[1],
        &mut meter,
    )
    .map_err(to_value_error)?;
    let (time, invariant, particular, multiplier) = checked.into_parts();
    Ok((
        PyExpr::from_expr(time),
        PyExpr::from_expr(invariant),
        PyExpr::from_expr(particular),
        PyExpr::from_expr(multiplier),
    ))
}
