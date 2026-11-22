//! Source-bound ordered polynomial reduction over QQ and ZZ.
//!
//! Arbitrary divisors need a checked division identity and a reduced remainder,
//! not a complete Groebner basis. The separate Groebner membership entry keeps
//! its stronger basis-completeness requirement. Every phase of this request,
//! including original-ring admission and output lifting, shares two ledgers.

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, symbols};
use crate::recursive_division::{checked_lift, integral, original_domain, polynomial_source};
use crate::to_value_error;
use fsym_budget::BudgetMeter;
use fsym_polys::{
    MultivariatePoly, TermOrder, checked_integer_polynomial_division, checked_polynomial_division,
};
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::mem::size_of;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (sources, variables, original_variables, original_domains,
                   source_coordinates, domain="auto", auto=true, order="lex"))]
pub fn poly_reduced_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    original_variables: &Bound<'_, PyList>,
    original_domains: &Bound<'_, PyList>,
    source_coordinates: &Bound<'_, PyList>,
    domain: &str,
    auto: bool,
    order: &str,
) -> PyResult<(Vec<PyExpr>, PyExpr, String)> {
    let count = sources.len();
    if !(1..=33).contains(&count)
        || original_variables.len() != count
        || original_domains.len() != count
        || source_coordinates.len() != count
    {
        return Err(PyValueError::new_err(
            "reduced requires one source and at most 32 divisors with matching declarations",
        ));
    }
    if !matches!(domain, "auto" | "ZZ" | "QQ") {
        return Err(PyValueError::new_err("reduced supports only ZZ and QQ"));
    }
    let order = match order {
        "lex" => TermOrder::Lex,
        "grlex" => TermOrder::DegLex,
        "grevlex" => TermOrder::DegRevLex,
        _ => return Err(PyValueError::new_err("unknown reduction monomial order")),
    };
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    if variables.is_empty() {
        return Err(PyValueError::new_err(
            "reduced requires at least one generator",
        ));
    }
    charge(&mut generator, count, count * size_of::<MultivariatePoly>())?;
    let mut polynomials = Vec::with_capacity(count);
    let mut inputs_integral = true;
    for index in 0..count {
        let source = sources.get_item(index)?;
        let source = source.extract::<PyRef<'_, PyExpr>>()?;
        let original_axes = original_variables.get_item(index)?;
        let original_axes = if original_axes.is_none() {
            None
        } else {
            Some(original_axes.cast::<PyList>()?)
        };
        let original_coefficient_domain = original_domains.get_item(index)?;
        let original_coefficient_domain = if original_coefficient_domain.is_none() {
            None
        } else {
            Some(original_coefficient_domain.extract::<&str>()?)
        };
        let coordinates = source_coordinates.get_item(index)?;
        let coordinates = if coordinates.is_none() {
            None
        } else {
            let coordinates = symbols(coordinates.cast::<PyList>()?, &mut generator)?;
            if coordinates.len() != variables.len() || original_axes.is_none() {
                return Err(PyValueError::new_err(
                    "positional reduction requires equal original and operation dimensions",
                ));
            }
            let original = symbols(
                original_axes.expect("checked original axes"),
                &mut generator,
            )?;
            if original != coordinates {
                return Err(PyValueError::new_err(
                    "positional reduction must retain the source's original typed axes",
                ));
            }
            Some(coordinates)
        };
        let source_axes = coordinates.as_deref().unwrap_or(&variables);
        let mut polynomial =
            polynomial_source(&source.inner, source_axes, &mut generator, &mut verifier)?;
        original_domain(
            &source.inner,
            Some(&polynomial),
            source_axes,
            original_axes,
            original_coefficient_domain,
            &mut generator,
            &mut verifier,
        )?;
        if coordinates.is_some() {
            let bytes = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
            charge(&mut generator, variables.len(), bytes)?;
            // Explicit ordinal ring isomorphism, after original-source
            // admission. No substitution or printed-name identity is used.
            polynomial.generators = variables.clone();
        }
        if index != 0 && polynomial.is_zero() {
            return Err(PyZeroDivisionError::new_err("polynomial division"));
        }
        inputs_integral &= integral(&polynomial, &mut generator)?;
        polynomials.push(polynomial);
    }
    if domain == "ZZ" && !inputs_integral {
        return Err(PyValueError::new_err(
            "noninteger coefficients cannot be represented in ZZ",
        ));
    }
    let input_qq = domain == "QQ" || (domain == "auto" && !inputs_integral);
    let (quotients, remainder) = if auto || input_qq {
        checked_polynomial_division(
            &polynomials[0],
            &polynomials[1..],
            order,
            &mut generator,
            &mut verifier,
        )
    } else {
        checked_integer_polynomial_division(
            &polynomials[0],
            &polynomials[1..],
            order,
            &mut generator,
            &mut verifier,
        )
    }
    .map_err(to_value_error)?;
    let mut outputs_integral = true;
    let mut nodes = 0usize;
    for polynomial in quotients.iter().chain(std::iter::once(&remainder)) {
        outputs_integral &= integral(polynomial, &mut generator)?;
        let count = 1 + polynomial
            .terms()
            .keys()
            .map(|powers| 2 + 4 * powers.iter().filter(|&&power| power != 0).count())
            .sum::<usize>();
        nodes += count;
        if count > 4096 || nodes > 16_384 {
            return Err(PyValueError::new_err(
                "reduced exceeds its bounded output expression envelope",
            ));
        }
    }
    let ring_bytes: usize = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
    // Prepay the complete output before constructing even its first element.
    charge(&mut generator, nodes, nodes * (2048 + ring_bytes))?;
    charge(
        &mut generator,
        quotients.len(),
        quotients.len() * size_of::<PyExpr>(),
    )?;
    let quotients = quotients
        .iter()
        .map(|polynomial| checked_lift(polynomial, &mut verifier))
        .collect::<PyResult<Vec<_>>>()?;
    let remainder = checked_lift(&remainder, &mut verifier)?;
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    let result_domain = if input_qq || !outputs_integral {
        "QQ"
    } else {
        "ZZ"
    };
    Ok((quotients, remainder, result_domain.into()))
}
