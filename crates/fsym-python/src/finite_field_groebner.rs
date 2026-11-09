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
use fsym_core::{Expr, Symbol};
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, checked_prime_field_fglm,
    checked_prime_field_groebner_basis, checked_prime_field_groebner_reduction,
    checked_prime_field_multivariate_source, checked_prime_field_quotient_multiplication,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::BTreeMap;

type QuotientMultiplicationParts = (Vec<PyExpr>, Vec<Vec<PyExpr>>);

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

/// Convert a checked original ideal through its finite quotient in F_p.
/// A same-order call still verifies that the current source is a complete
/// reduced basis before the shell may return the original public object.
#[pyfunction]
#[pyo3(signature = (sources, variables, prime, source_order, target_order, symmetric=true, source_variables=None))]
pub fn prime_field_groebner_fglm_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    source_order: &str,
    target_order: &str,
    symmetric: bool,
    source_variables: Option<&Bound<'_, PyList>>,
) -> PyResult<Vec<PyExpr>> {
    if sources.len() > 32 {
        return Err(PyValueError::new_err(
            "prime-field FGLM requires at most 32 source polynomials",
        ));
    }
    let source_order = order(source_order)?;
    let target_order = order(target_order)?;
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
    let output = if source_order == target_order {
        checked_prime_field_groebner_reduction(
            &zero,
            &polynomials,
            source_order,
            &mut generator,
            &mut verifier,
        )
        .map_err(polynomial_error)?;
        polynomials
    } else {
        checked_prime_field_fglm(
            &variables,
            prime,
            &polynomials,
            source_order,
            target_order,
            &mut generator,
            &mut verifier,
        )
        .map_err(polynomial_error)?
        .into_basis()
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

/// Return a complete quotient staircase and its multiplication operator.
/// Sources are the element followed by every original ideal generator. Matrix
/// entries are integer representatives of residues, never rational arithmetic.
#[pyfunction]
#[pyo3(signature = (sources, variables, prime, order="lex", symmetric=true, source_variables=None))]
pub fn prime_field_quotient_multiplication_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    order: &str,
    symmetric: bool,
    source_variables: Option<&Bound<'_, PyList>>,
) -> PyResult<QuotientMultiplicationParts> {
    if !(1..=33).contains(&sources.len()) {
        return Err(PyValueError::new_err(
            "prime-field quotient multiplication requires one element and at most 32 ideal polynomials",
        ));
    }
    let term_order = self::order(order)?;
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
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
    let accepted = checked_prime_field_quotient_multiplication(
        &variables,
        prime,
        &polynomials[1..],
        &polynomials[0],
        term_order,
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    let dimension = accepted.standard_monomials().len();
    if dimension > 64
        || accepted.matrix().len() != dimension
        || accepted.matrix().iter().any(|row| row.len() != dimension)
    {
        return Err(PyValueError::new_err(
            "checked prime-field quotient multiplication has an invalid dimension",
        ));
    }
    // Reserve every lifted row/cell before creating expression owners. All
    // source, kernel, verifier and output work stays on the same phase meters.
    charge(
        &mut generator,
        dimension * (variables.len() + dimension + 1) + 1,
        dimension * (std::mem::size_of::<PyExpr>() + std::mem::size_of::<Vec<PyExpr>>())
            + dimension * dimension * (std::mem::size_of::<PyExpr>() + 512),
    )?;
    let mut monomials = Vec::with_capacity(dimension);
    for powers in accepted.standard_monomials() {
        charge(
            &mut generator,
            variables.len() + 1,
            powers.len() * std::mem::size_of::<u32>() + 128,
        )?;
        let terms = BTreeMap::from([(powers.clone(), 1)]);
        let polynomial = Polynomial::metered_new(&variables, prime, &terms, &mut generator)
            .map_err(polynomial_error)?;
        monomials.push(lift(&polynomial, symmetric, &mut generator)?);
    }
    let mut matrix = Vec::with_capacity(dimension);
    for row in accepted.matrix() {
        generator.checkpoint().map_err(to_value_error)?;
        let mut values = Vec::with_capacity(dimension);
        for &value in row {
            let value = if symmetric && value > prime / 2 {
                value as i64 - prime as i64
            } else {
                value as i64
            };
            values.push(PyExpr::from_expr(Expr::from_i64(value)));
        }
        matrix.push(values);
    }
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok((monomials, matrix))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(prime_field_groebner_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_groebner_zero_dimensional_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(prime_field_groebner_fglm_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_quotient_multiplication_native,
        module
    )?)?;
    Ok(())
}
