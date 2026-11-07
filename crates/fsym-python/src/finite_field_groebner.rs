//! Checked prime-field Groebner operations at the original-source boundary.
//!
//! Every source is admitted before a unit or zero ideal can terminate work.
//! Source conversion, basis/reduction generation, independent verification and
//! result lifting use the same two cumulative ledgers for the entire native call.

#![forbid(unsafe_code)]

use crate::expr::PyExpr;
use crate::finite_field_multivariate::{lift, monomial_cmp, order, symbols};
use crate::finite_field_polynomial::{budget, charge, polynomial_error, source_error};
use crate::to_value_error;
use fsym_budget::{Budget, BudgetMeter};
use fsym_core::Symbol;
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, checked_prime_field_groebner_basis,
    checked_prime_field_groebner_reduction, checked_prime_field_multivariate_source,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::BTreeMap;

pub(crate) fn admit_sources(
    sources: &Bound<'_, PyList>,
    variables: &[Symbol],
    prime: u64,
    source_variables: Option<&Bound<'_, PyList>>,
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<Vec<Polynomial>> {
    if source_variables.is_some_and(|declared| declared.len() != sources.len()) {
        return Err(PyValueError::new_err(
            "each prime-field source requires its own declared ring or None",
        ));
    }
    charge(
        generator,
        sources.len() + 1,
        (sources.len() + 1) * std::mem::size_of::<Polynomial>(),
    )?;
    let mut result = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        let source: PyRef<'_, PyExpr> = source.extract()?;
        if let Some(declared) = source_variables {
            let own_variables = declared.get_item(index)?;
            if !own_variables.is_none() {
                let own_variables = symbols(own_variables.cast::<PyList>()?, generator)?;
                // A retained Poly must belong to its own declared ring before
                // promotion to the request's common ring. Use the same ledgers
                // and original expression, including held/cancelled children.
                checked_prime_field_multivariate_source(
                    &source.inner,
                    &own_variables,
                    prime,
                    generator,
                    verifier,
                )
                .map_err(source_error)?;
            }
        }
        result.push(
            checked_prime_field_multivariate_source(
                &source.inner,
                variables,
                prime,
                generator,
                verifier,
            )
            .map_err(source_error)?
            .into_polynomial(),
        );
    }
    Ok(result)
}

/// Construct a complete reduced basis, reduce against a checked current basis,
/// or project the current source polynomials without asserting basis validity.
/// Reduction sources are ordered as dividend followed by the current basis.
#[pyfunction]
#[pyo3(signature = (sources, variables, prime, operation, order="lex", symmetric=true, source_variables=None))]
pub fn prime_field_groebner_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    operation: &str,
    order: &str,
    symmetric: bool,
    source_variables: Option<&Bound<'_, PyList>>,
) -> PyResult<Vec<PyExpr>> {
    let term_order = self::order(order)?;
    let valid_count = match operation {
        "basis" | "polynomials" => sources.len() <= 32,
        "reduce" => (1..=33).contains(&sources.len()),
        _ => false,
    };
    if !valid_count {
        return Err(PyValueError::new_err(
            "prime-field Groebner operation requires at most 32 basis polynomials",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    // An empty polynomial list still has to denote a valid typed prime field.
    Polynomial::metered_new(&variables, prime, &BTreeMap::new(), &mut generator)
        .map_err(polynomial_error)?;
    let polynomials = admit_sources(
        sources,
        &variables,
        prime,
        source_variables,
        &mut generator,
        &mut verifier,
    )?;
    let output = match operation {
        "basis" => checked_prime_field_groebner_basis(
            &variables,
            prime,
            &polynomials,
            term_order,
            &mut generator,
            &mut verifier,
        )
        .map_err(polynomial_error)?
        .into_basis(),
        "reduce" => {
            let receipt = checked_prime_field_groebner_reduction(
                &polynomials[0],
                &polynomials[1..],
                term_order,
                &mut generator,
                &mut verifier,
            )
            .map_err(polynomial_error)?;
            let (mut quotients, remainder) = receipt.into_parts();
            charge(&mut generator, 1, std::mem::size_of::<Polynomial>())?;
            quotients.push(remainder);
            quotients
        }
        "polynomials" => polynomials,
        _ => unreachable!("operation and operand count were admitted"),
    };
    charge(
        &mut generator,
        output.len() + 1,
        (output.len() + 1) * std::mem::size_of::<PyExpr>(),
    )?;
    let mut lifted = Vec::with_capacity(output.len());
    for polynomial in &output {
        lifted.push(lift(polynomial, symmetric, &mut generator)?);
    }
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(lifted)
}

/// Apply the pure-leading-power criterion only after checking the complete
/// current reduced basis. Python's predicate excludes both zero and unit ideals.
#[pyfunction]
#[pyo3(signature = (sources, variables, prime, order="lex", source_variables=None))]
pub fn prime_field_groebner_zero_dimensional_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    order: &str,
    source_variables: Option<&Bound<'_, PyList>>,
) -> PyResult<bool> {
    if sources.len() > 32 {
        return Err(PyValueError::new_err(
            "prime-field Groebner basis exceeds 32 polynomials",
        ));
    }
    let term_order = self::order(order)?;
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    let zero = Polynomial::metered_new(&variables, prime, &BTreeMap::new(), &mut generator)
        .map_err(polynomial_error)?;
    let polynomials = admit_sources(
        sources,
        &variables,
        prime,
        source_variables,
        &mut generator,
        &mut verifier,
    )?;
    checked_prime_field_groebner_reduction(
        &zero,
        &polynomials,
        term_order,
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    charge(&mut generator, variables.len() + 1, variables.len() + 1)?;
    let mut covered = vec![false; variables.len()];
    let mut unit = false;
    for polynomial in &polynomials {
        charge(
            &mut generator,
            (polynomial.terms().len() + 1) * (variables.len() + 1),
            1,
        )?;
        let leading = polynomial
            .terms()
            .keys()
            .max_by(|left, right| monomial_cmp(left, right, term_order))
            .ok_or_else(|| PyValueError::new_err("zero polynomial in checked Groebner basis"))?;
        let mut axis = None;
        let mut nonzero = 0;
        for (index, power) in leading.iter().enumerate() {
            if *power != 0 {
                axis = Some(index);
                nonzero += 1;
            }
        }
        if nonzero == 0 {
            unit = true;
        } else if nonzero == 1 {
            covered[axis.expect("one nonzero leading power")] = true;
        }
    }
    let result = !unit && covered.into_iter().all(|value| value);
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(prime_field_groebner_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_groebner_zero_dimensional_native,
        module
    )?)?;
    Ok(())
}
