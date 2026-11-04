//! Typed fixed-width bridge for complete rational two-component ODE systems.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigRational, Expr};
use fsym_runtime::{Budget, BudgetLimits};
use fsym_solvers::ode::system::metered_linear_ode_system;
use fsym_solvers::ode::system::polynomial::metered_polynomial_ode_system;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

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

/// Bounded borrowed coefficient vectors are admitted before extracting a Vec.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn polynomial_ode_system_native(
    a00: &PyExpr,
    a01: &PyExpr,
    a10: &PyExpr,
    a11: &PyExpr,
    time: &PySymbolBinding,
    initial_point: &PyExpr,
    first_value: &PyExpr,
    second_value: &PyExpr,
    first_forcing: &Bound<'_, PyList>,
    second_forcing: &Bound<'_, PyList>,
) -> PyResult<Vec<PyExpr>> {
    if time.name.len() > 16_384 {
        return Err(PyValueError::new_err(
            "ODE time symbol exceeds its text bound",
        ));
    }
    let rows = [first_forcing, second_forcing];
    if rows.iter().any(|row| row.is_empty() || row.len() > 17) {
        return Err(PyValueError::new_err(
            "ODE forcing requires one to seventeen coefficients per coordinate",
        ));
    }
    for row in rows {
        for item in row.iter() {
            let value = item.extract::<PyRef<'_, PyExpr>>()?;
            let valid = match &value.inner {
                Expr::Integer(number) => number.bits() <= 1024,
                Expr::Rational(number) => {
                    number.numer().bits() <= 1024 && number.denom().bits() <= 1024
                }
                _ => false,
            };
            if !valid {
                return Err(PyValueError::new_err(
                    "ODE forcing coefficients must be exact rationals of at most 1024 bits",
                ));
            }
        }
    }
    let mut generator = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    generator
        .charge_batch(&[
            (Dimension::ComputeSteps, 64),
            (Dimension::MemoryBytes, 32_768 + time.name.len() as u64),
            (Dimension::AllocationCount, 64),
        ])
        .map_err(to_value_error)?;
    let coefficients = [
        [rational(&a00.inner)?, rational(&a01.inner)?],
        [rational(&a10.inner)?, rational(&a11.inner)?],
    ];
    let point = rational(&initial_point.inner)?;
    let mut forcing: [Vec<BigRational>; 2] =
        std::array::from_fn(|row| Vec::with_capacity(rows[row].len()));
    for (row, values) in rows.iter().enumerate() {
        for item in values.iter() {
            let value = item.extract::<PyRef<'_, PyExpr>>()?;
            forcing[row].push(rational(&value.inner)?);
        }
    }
    let checked = metered_polynomial_ode_system(
        &coefficients,
        &time.to_symbol(),
        &point,
        &forcing,
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
