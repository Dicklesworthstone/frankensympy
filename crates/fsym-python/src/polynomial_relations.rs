//! Complete QQ polynomial relations and inversion in polynomial quotient rings.
//!
//! Admission, checked generation, independent replay and bounded publication
//! share one generator budget and one verifier budget for each public call.

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, lift_basis, polynomials, symbols};
use crate::{groebner_term_order, to_value_error};
use fsym_core::Symbol;
use fsym_polys::{MultivariatePoly, bounded_polynomial_from_expr};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn ring_bytes(variables: &[Symbol]) -> usize {
    variables.iter().map(|symbol| symbol.name.len() + 128).sum()
}

/// Return generators of every polynomial relation on the exact input sequence.
#[pyfunction]
pub fn polynomial_syzygies_native(
    equations: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    order: &str,
) -> PyResult<Vec<Vec<PyExpr>>> {
    let order = groebner_term_order(order)?;
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    let inputs = polynomials(equations, &variables, &mut generator)?;
    let result = fsym_polys::checked_polynomial_syzygies(
        &inputs,
        &variables,
        order,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    let rows = result.into_syzygies();
    let row_count = rows.len();
    let width = inputs.len();
    let cells = row_count
        .checked_mul(width)
        .filter(|&count| count <= 16_384 && row_count <= 16_384)
        .ok_or_else(|| PyValueError::new_err("polynomial relation output exceeds 16384 cells"))?;
    if rows.iter().any(|row| row.len() != width) {
        return Err(PyValueError::new_err(
            "checked polynomial relation has an inconsistent input dimension",
        ));
    }
    let storage = cells
        .checked_mul(std::mem::size_of::<MultivariatePoly>())
        .and_then(|bytes| bytes.checked_add(row_count * std::mem::size_of::<Vec<PyExpr>>()))
        .ok_or_else(|| PyValueError::new_err("polynomial relation output allocation overflow"))?;
    charge(&mut generator, cells, storage)?;
    // One flattened lift enforces the aggregate expression and allocation
    // bounds across every row; no row restarts either cumulative budget.
    let flat = rows.into_iter().flatten().collect();
    let mut lifted =
        lift_basis(flat, ring_bytes(&variables), &mut generator, &mut verifier)?.into_iter();
    let output = (0..row_count)
        .map(|_| lifted.by_ref().take(width).collect())
        .collect();
    charge(&mut generator, 1, 1)?;
    charge(&mut verifier, 1, 1)?;
    Ok(output)
}

/// Return the unique reduced inverse, or None for an independently proved nonunit.
#[pyfunction]
pub fn ideal_inverse_native(
    equations: &Bound<'_, PyList>,
    element: &PyExpr,
    variables: &Bound<'_, PyList>,
    order: &str,
) -> PyResult<Option<PyExpr>> {
    if equations.len() > 63 {
        return Err(PyValueError::new_err(
            "checked ideal inversion supports at most 63 ideal generators",
        ));
    }
    let order = groebner_term_order(order)?;
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    let inputs = polynomials(equations, &variables, &mut generator)?;
    // The original element is admitted even when the ideal is the unit ideal.
    let element = bounded_polynomial_from_expr(&element.inner, &variables, &mut generator)
        .map_err(to_value_error)?;
    let result = fsym_polys::checked_ideal_inverse(
        &inputs,
        &element,
        &variables,
        order,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    let Some(inverse) = result.into_inverse() else {
        charge(&mut generator, 1, 1)?;
        charge(&mut verifier, 1, 1)?;
        return Ok(None);
    };
    let mut values = lift_basis(
        vec![inverse],
        ring_bytes(&variables),
        &mut generator,
        &mut verifier,
    )?;
    values
        .pop()
        .map(Some)
        .ok_or_else(|| PyValueError::new_err("checked ideal inverse output is missing"))
}
