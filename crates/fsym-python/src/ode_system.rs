//! Typed fixed-width bridge for complete rational two-component ODE systems.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_core::{BigRational, Expr};
use fsym_runtime::{Budget, BudgetLimits};
use fsym_solvers::ode::system::metered_linear_ode_system;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn rational(expression: &Expr) -> PyResult<BigRational> {
    match expression {
        Expr::Integer(value) if value.bits() <= 1024 => {
            Ok(BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value) if value.numer().bits() <= 1024 && value.denom().bits() <= 1024 => {
            Ok(value.clone())
        }
        _ => Err(PyValueError::new_err(
            "ODE system coefficients and initial point must be exact rationals of at most 1024 bits",
        )),
    }
}

// Fixed borrowed arguments avoid extracting an unbounded Python matrix/list
// before the native width and scalar checks have run.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn linear_ode_system_native(
    a00: &PyExpr,
    a01: &PyExpr,
    a10: &PyExpr,
    a11: &PyExpr,
    time: &PySymbolBinding,
    initial_point: &PyExpr,
    first_value: &PyExpr,
    second_value: &PyExpr,
) -> PyResult<Vec<PyExpr>> {
    if time.name.len() > 16_384 {
        return Err(PyValueError::new_err(
            "ODE time symbol exceeds its text bound",
        ));
    }
    let coefficients = [
        [rational(&a00.inner)?, rational(&a01.inner)?],
        [rational(&a10.inner)?, rational(&a11.inner)?],
    ];
    let point = rational(&initial_point.inner)?;
    let mut generator = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let checked = metered_linear_ode_system(
        &coefficients,
        &time.to_symbol(),
        &point,
        [&first_value.inner, &second_value.inner],
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    Ok(checked
        .into_solutions()
        .into_iter()
        .map(PyExpr::from_expr)
        .collect())
}
