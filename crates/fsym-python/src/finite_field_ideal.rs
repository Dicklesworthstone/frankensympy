//! Source-bound elimination, intersection and principal saturation over F_p.
//!
//! Original operands, declared Poly rings, checked ambient construction,
//! independent ideal verification and result lifting share two cumulative
//! ledgers. In particular, zero/unit shortcuts never erase an unadmitted source.

#![forbid(unsafe_code)]

use crate::expr::PyExpr;
use crate::finite_field_groebner::admit_sources;
use crate::finite_field_multivariate::{lift, symbols};
use crate::finite_field_polynomial::{budget, charge, polynomial_error};
use crate::to_value_error;
use fsym_budget::{Budget, BudgetMeter};
use fsym_core::Symbol;
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, VerifiedPrimeFieldIdealOperation,
    checked_prime_field_elimination_ideal, checked_prime_field_ideal_intersection,
    checked_prime_field_ideal_saturation,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::BTreeMap;

fn ring(variables: &Bound<'_, PyList>, prime: u64, generator: &mut Budget) -> PyResult<Vec<Symbol>> {
    let variables = symbols(variables, generator)?;
    // Empty source ideals still retain a validated characteristic and ring.
    Polynomial::metered_new(&variables, prime, &BTreeMap::new(), generator)
        .map_err(polynomial_error)?;
    Ok(variables)
}

fn publish(
    receipt: VerifiedPrimeFieldIdealOperation,
    symmetric: bool,
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<Vec<PyExpr>> {
    charge(
        generator,
        receipt.basis().len() + 1,
        (receipt.basis().len() + 1) * std::mem::size_of::<PyExpr>(),
    )?;
    let mut result = Vec::with_capacity(receipt.basis().len());
    for polynomial in receipt.basis() {
        result.push(lift(polynomial, symmetric, generator)?);
    }
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(result)
}

/// Complete elimination in the requested typed variables, including the
/// explicit constant field when every original variable is eliminated.
#[pyfunction]
#[pyo3(signature = (sources, variables, prime, eliminated, symmetric=true, source_variables=None))]
pub fn prime_field_ideal_eliminate_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    eliminated: &Bound<'_, PyList>,
    symmetric: bool,
    source_variables: Option<&Bound<'_, PyList>>,
) -> PyResult<Vec<PyExpr>> {
    if sources.len() > 32 {
        return Err(PyValueError::new_err(
            "prime-field elimination requires at most 32 source polynomials",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = ring(variables, prime, &mut generator)?;
    let eliminated = symbols(eliminated, &mut generator)?;
    let inputs = admit_sources(
        sources,
        &variables,
        prime,
        source_variables,
        &mut generator,
        &mut verifier,
    )?;
    let receipt = checked_prime_field_elimination_ideal(
        &variables,
        prime,
        &inputs,
        &eliminated,
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    publish(receipt, symmetric, &mut generator, &mut verifier)
}

/// Complete ideal intersection in the left operand's retained coordinate
/// order. Both original source lists are admitted before constructing tI+(1-t)J.
#[pyfunction]
#[pyo3(signature = (left, right, variables, prime, symmetric=true, left_source_variables=None, right_source_variables=None))]
pub fn prime_field_ideal_intersection_native(
    left: &Bound<'_, PyList>,
    right: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    symmetric: bool,
    left_source_variables: Option<&Bound<'_, PyList>>,
    right_source_variables: Option<&Bound<'_, PyList>>,
) -> PyResult<Vec<PyExpr>> {
    if left.len() > 32 || right.len() > 32 - left.len() {
        return Err(PyValueError::new_err(
            "prime-field intersection requires at most 32 combined source polynomials",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = ring(variables, prime, &mut generator)?;
    let left = admit_sources(
        left,
        &variables,
        prime,
        left_source_variables,
        &mut generator,
        &mut verifier,
    )?;
    let right = admit_sources(
        right,
        &variables,
        prime,
        right_source_variables,
        &mut generator,
        &mut verifier,
    )?;
    let receipt = checked_prime_field_ideal_intersection(
        &variables,
        prime,
        &left,
        &right,
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    publish(receipt, symmetric, &mut generator, &mut verifier)
}

/// Complete I:f^infinity. Sources are the original factor f followed by the
/// ideal generators, so every declared source ring uses the same admission path.
#[pyfunction]
#[pyo3(signature = (sources, variables, prime, symmetric=true, source_variables=None))]
pub fn prime_field_ideal_saturation_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    symmetric: bool,
    source_variables: Option<&Bound<'_, PyList>>,
) -> PyResult<Vec<PyExpr>> {
    if !(1..=32).contains(&sources.len()) {
        return Err(PyValueError::new_err(
            "prime-field saturation requires one factor and at most 31 ideal polynomials",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = ring(variables, prime, &mut generator)?;
    let inputs = admit_sources(
        sources,
        &variables,
        prime,
        source_variables,
        &mut generator,
        &mut verifier,
    )?;
    let receipt = checked_prime_field_ideal_saturation(
        &variables,
        prime,
        &inputs[1..],
        &inputs[0],
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    publish(receipt, symmetric, &mut generator, &mut verifier)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(prime_field_ideal_eliminate_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_ideal_intersection_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(prime_field_ideal_saturation_native, module)?)?;
    Ok(())
}
