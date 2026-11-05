//! Exact quotient-algebra operations with source-bound native verification.
//!
//! One pair of cumulative budgets covers admission, complete algebraic
//! generation, independent verification and every published expression.

use std::collections::BTreeMap;

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, lift_basis, polynomials, symbols};
use crate::{groebner_term_order, rational_to_expr, to_value_error};
use fsym_core::{BigRational, Symbol};
use fsym_polys::{MultivariatePoly, bounded_polynomial_from_expr};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

type QuotientMultiplicationParts = (Vec<PyExpr>, Vec<Vec<PyExpr>>);

fn ring_bytes(variables: &[Symbol]) -> usize {
    variables.iter().map(|symbol| symbol.name.len() + 128).sum()
}

/// Complete standard monomials and the exact multiplication-by-element matrix.
/// Column j is the coefficient vector of element times monomial j.
#[pyfunction]
pub fn quotient_multiplication_native(
    equations: &Bound<'_, PyList>,
    element: &PyExpr,
    variables: &Bound<'_, PyList>,
    order: &str,
) -> PyResult<QuotientMultiplicationParts> {
    let order = groebner_term_order(order)?;
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    let inputs = polynomials(equations, &variables, &mut generator)?;
    let element = bounded_polynomial_from_expr(&element.inner, &variables, &mut generator)
        .map_err(to_value_error)?;
    let result = fsym_polys::checked_quotient_multiplication(
        &inputs,
        &element,
        &variables,
        order,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    let dimension = result.standard_monomials().len();
    if dimension > 64
        || result.matrix().len() != dimension
        || result.matrix().iter().any(|row| row.len() != dimension)
    {
        return Err(PyValueError::new_err(
            "checked quotient multiplication has an invalid dimension",
        ));
    }
    let names = ring_bytes(&variables);
    // The verified result bounds all exponents and coefficient heights. Charge
    // the complete output before cloning coefficients or typed symbol names.
    charge(
        &mut generator,
        dimension * (variables.len() + dimension + 1),
        dimension * (names + 512 + variables.len() * 4) + dimension * dimension * 4096,
    )?;
    let monomials = result
        .standard_monomials()
        .iter()
        .map(|powers| {
            MultivariatePoly::new(
                variables.clone(),
                BTreeMap::from([(powers.clone(), BigRational::from_integer(1.into()))]),
            )
            .map_err(to_value_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let basis = lift_basis(monomials, names, &mut generator, &mut verifier)?;
    let matrix = result
        .matrix()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| PyExpr::from_expr(rational_to_expr(value.clone())))
                .collect()
        })
        .collect();
    charge(&mut generator, 1, 1)?;
    charge(&mut verifier, 1, 1)?;
    Ok((basis, matrix))
}

/// The monic generator of the kernel QQ[t] -> QQ[x]/I, t |-> element.
/// None is returned only when complete checked elimination proves that this
/// kernel is zero. Resource, domain and verification failures remain errors.
#[pyfunction]
pub fn quotient_minimal_polynomial_native(
    equations: &Bound<'_, PyList>,
    element: &PyExpr,
    variables: &Bound<'_, PyList>,
    parameter: &Bound<'_, PyAny>,
) -> PyResult<Option<PyExpr>> {
    if equations.len() > 63 || variables.len() > 7 {
        return Err(PyValueError::new_err(
            "quotient minimal polynomial supports at most seven variables and 63 ideal generators",
        ));
    }
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    let parameters = PyList::new(parameter.py(), [parameter])?;
    let parameter = symbols(&parameters, &mut generator)?
        .pop()
        .ok_or_else(|| PyValueError::new_err("missing quotient polynomial parameter"))?;
    let inputs = polynomials(equations, &variables, &mut generator)?;
    let element = bounded_polynomial_from_expr(&element.inner, &variables, &mut generator)
        .map_err(to_value_error)?;
    let result = fsym_polys::checked_quotient_minimal_polynomial(
        &inputs,
        &element,
        &variables,
        &parameter,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    let Some(coefficients) = result.into_coefficients() else {
        charge(&mut generator, 1, 1)?;
        charge(&mut verifier, 1, 1)?;
        return Ok(None);
    };
    if coefficients.is_empty() || coefficients.len() > 257 {
        return Err(PyValueError::new_err(
            "checked quotient minimal polynomial has an invalid degree",
        ));
    }
    let names = parameter.name.len() + 128;
    charge(
        &mut generator,
        coefficients.len(),
        coefficients.len() * 512 + names,
    )?;
    let terms = coefficients
        .into_iter()
        .enumerate()
        .filter(|(_, value)| value.numer().bits() != 0)
        .map(|(degree, value)| (vec![degree as u32], value))
        .collect();
    let polynomial = MultivariatePoly::new(vec![parameter], terms).map_err(to_value_error)?;
    let mut result = lift_basis(vec![polynomial], names, &mut generator, &mut verifier)?;
    let value = result
        .pop()
        .ok_or_else(|| PyValueError::new_err("missing checked quotient minimal polynomial"))?;
    charge(&mut generator, 1, 1)?;
    charge(&mut verifier, 1, 1)?;
    Ok(Some(value))
}
