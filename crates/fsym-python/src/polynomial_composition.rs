//! Source-bound QQ/ZZ composition for the retained Poly/PurePoly surface.
//!
//! Admission of both complete original inputs, their declared coefficient
//! domains, optional explicit ordinal coordinate mapping, generation,
//! independent coefficient verification and output lifting share two ledgers.

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, symbols};
use crate::recursive_division::{checked_lift, integral, original_domain, polynomial_source};
use crate::to_value_error;
use fsym_budget::BudgetMeter;
use fsym_polys::checked_polynomial_composition;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (outer, inner, variables, outer_variables=None, inner_variables=None,
                   outer_domain=None, inner_domain=None, domain="auto", inner_coordinates=None))]
pub fn polynomial_compose_native(
    outer: &PyExpr,
    inner: &PyExpr,
    variables: &Bound<'_, PyList>,
    outer_variables: Option<&Bound<'_, PyList>>,
    inner_variables: Option<&Bound<'_, PyList>>,
    outer_domain: Option<&str>,
    inner_domain: Option<&str>,
    domain: &str,
    inner_coordinates: Option<&Bound<'_, PyList>>,
) -> PyResult<(PyExpr, String)> {
    if !matches!(domain, "auto" | "ZZ" | "QQ") {
        return Err(PyValueError::new_err(
            "checked polynomial composition supports ZZ and QQ",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    if variables.is_empty() {
        return Err(PyValueError::new_err(
            "polynomial composition requires a primary generator",
        ));
    }
    let left = polynomial_source(&outer.inner, &variables, &mut generator, &mut verifier)?;
    original_domain(
        &outer.inner,
        Some(&left),
        &variables,
        outer_variables,
        outer_domain,
        &mut generator,
        &mut verifier,
    )?;
    let coordinates = if let Some(coordinates) = inner_coordinates {
        let coordinates = symbols(coordinates, &mut generator)?;
        let original = inner_variables.ok_or_else(|| {
            PyValueError::new_err("positional composition requires the original inner ring")
        })?;
        let original = symbols(original, &mut generator)?;
        charge(
            &mut generator,
            original
                .iter()
                .chain(&coordinates)
                .map(|symbol| symbol.name.len() + 1)
                .sum(),
            0,
        )?;
        if coordinates.len() != variables.len() || coordinates != original {
            return Err(PyValueError::new_err(
                "positional composition must retain equal original and operation dimensions",
            ));
        }
        Some(coordinates)
    } else {
        None
    };
    let inner_axes = coordinates.as_deref().unwrap_or(&variables);
    let mut right = polynomial_source(&inner.inner, inner_axes, &mut generator, &mut verifier)?;
    original_domain(
        &inner.inner,
        Some(&right),
        inner_axes,
        inner_variables,
        inner_domain,
        &mut generator,
        &mut verifier,
    )?;
    if coordinates.is_some() {
        // PurePoly's explicit ordinal isomorphism is applied only after the
        // original right source and its original domain have been checked.
        let bytes = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
        charge(&mut generator, variables.len(), bytes)?;
        right.generators = variables.clone();
    }
    let inputs_integral = integral(&left, &mut generator)? & integral(&right, &mut generator)?;
    if domain == "ZZ" && !inputs_integral {
        return Err(PyValueError::new_err(
            "noninteger coefficients cannot be represented in ZZ",
        ));
    }
    // The domain depends on both admitted original inputs, even when their
    // fractional coefficients cancel or the outer source is constant.
    let result_domain = if domain == "QQ" || !inputs_integral {
        "QQ"
    } else {
        "ZZ"
    };
    let accepted = checked_polynomial_composition(&left, &right, &mut generator, &mut verifier)
        .map_err(to_value_error)?;
    let result = accepted.result();
    let nodes = 1
        + result
            .terms()
            .keys()
            .map(|powers| 2 + 4 * powers.iter().filter(|&&power| power != 0).count())
            .sum::<usize>();
    if nodes > 4096 {
        return Err(PyValueError::new_err(
            "composition exceeds its bounded output expression envelope",
        ));
    }
    let ring_bytes: usize = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
    charge(&mut generator, nodes, nodes * (2048 + ring_bytes))?;
    let output = checked_lift(result, &mut verifier)?;
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok((output, result_domain.into()))
}
