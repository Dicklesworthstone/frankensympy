//! Source-bound recursive polynomial division for the Python Poly surface.
//!
//! The main variable is the first declared axis. Whole coefficient-ring
//! divisibility and strict integer-domain obstruction are checked by the
//! polynomial engine; sparse distributed reduction remains a separate API.

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, symbols};
use crate::to_value_error;
use fsym_budget::BudgetMeter;
use fsym_core::{Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, PolyError, RecursiveDivisionDomain, TermOrder, bounded_polynomial_from_expr,
    checked_recursive_division, metered_rational_source, verify_bounded_division,
};
use fsym_runtime::Budget;
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::mem::size_of;

fn polynomial_error(error: PolyError) -> PyErr {
    if error == PolyError::DivisionByZero {
        PyZeroDivisionError::new_err("polynomial division")
    } else {
        to_value_error(error)
    }
}

/// Independent rational-source replay binds the generated polynomial to the
/// entire original tree. Its denominator is a literal coefficient expression
/// admitted by polynomial lowering; the full coefficient identity is checked
/// before the recursive division sees either operand.
fn polynomial_source(
    source: &Expr,
    variables: &[Symbol],
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<MultivariatePoly> {
    let polynomial =
        bounded_polynomial_from_expr(source, variables, generator).map_err(polynomial_error)?;
    let original = metered_rational_source(source, variables, generator, verifier)
        .map_err(polynomial_error)?;
    let ring_bytes = variables
        .iter()
        .map(|symbol| symbol.name.len() + 128)
        .sum::<usize>();
    charge(verifier, 1, ring_bytes + size_of::<MultivariatePoly>())?;
    let zero = MultivariatePoly::zero(variables.to_vec());
    verify_bounded_division(
        &original.fraction().numerator,
        std::slice::from_ref(&original.fraction().denominator),
        std::slice::from_ref(&polynomial),
        &zero,
        TermOrder::Lex,
        verifier,
    )
    .map_err(polynomial_error)?;
    Ok(polynomial)
}

fn integral(polynomial: &MultivariatePoly, meter: &mut Budget) -> PyResult<bool> {
    charge(meter, polynomial.terms().len(), 0)?;
    Ok(polynomial
        .terms()
        .values()
        .all(|coefficient| coefficient.is_integer()))
}

/// Operation-ring union cannot repair a false declaration on a retained Poly.
/// The original ring is admitted separately, including unused typed axes.
fn original_domain(
    source: &Expr,
    polynomial: Option<&MultivariatePoly>,
    variables: &[Symbol],
    original_variables: Option<&Bound<'_, PyList>>,
    domain: Option<&str>,
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<()> {
    let (original_variables, domain) = match (original_variables, domain) {
        (None, None) => return Ok(()),
        (Some(variables), Some(domain @ ("ZZ" | "QQ"))) => (variables, domain),
        _ => {
            return Err(PyValueError::new_err(
                "original polynomial declaration requires its exact ring and ZZ or QQ domain",
            ));
        }
    };
    let original_variables = symbols(original_variables, generator)?;
    let name_bytes = variables
        .iter()
        .chain(&original_variables)
        .map(|symbol| symbol.name.len() + 1)
        .sum();
    charge(generator, name_bytes, 0)?;
    let original;
    let polynomial = if let Some(polynomial) = polynomial
        && original_variables == variables
    {
        polynomial
    } else {
        original = polynomial_source(source, &original_variables, generator, verifier)?;
        &original
    };
    if domain == "ZZ" && !integral(polynomial, generator)? {
        return Err(PyValueError::new_err(
            "noninteger coefficients cannot be represented in ZZ",
        ));
    }
    Ok(())
}

fn checked_lift(polynomial: &MultivariatePoly, verifier: &mut Budget) -> PyResult<PyExpr> {
    // Both complete output allocations were prepaid before the first lift.
    let expression = polynomial.to_expr().map_err(polynomial_error)?;
    let recovered = bounded_polynomial_from_expr(&expression, polynomial.generators(), verifier)
        .map_err(polynomial_error)?;
    let steps = polynomial
        .terms()
        .iter()
        .map(|(powers, coefficient)| {
            powers.len()
                + coefficient.numer().limb_count() as usize
                + coefficient.denom().limb_count() as usize
                + 1
        })
        .sum();
    charge(verifier, steps, 0)?;
    if recovered.terms() != polynomial.terms() {
        return Err(PyValueError::new_err(
            "recursive division lifting changed an exact coefficient",
        ));
    }
    Ok(PyExpr::from_expr(expression))
}

/// One bounded request covers source admission, original declarations, forced
/// recursive steps, independent acceptance and both returned polynomials.
/// Separate optional declarations preserve each retained Poly's original ring
/// without conflating it with the requested operation ring.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (left, right, variables, left_variables=None, right_variables=None,
                   left_domain=None, right_domain=None, domain="auto", auto=true,
                   method_scalar=false))]
pub fn poly_recursive_div_native(
    left: &PyExpr,
    right: &PyExpr,
    variables: &Bound<'_, PyList>,
    left_variables: Option<&Bound<'_, PyList>>,
    right_variables: Option<&Bound<'_, PyList>>,
    left_domain: Option<&str>,
    right_domain: Option<&str>,
    domain: &str,
    auto: bool,
    method_scalar: bool,
) -> PyResult<(PyExpr, PyExpr, String)> {
    if !matches!(domain, "ZZ" | "QQ" | "auto") {
        return Err(PyValueError::new_err(
            "recursive division domain must be ZZ, QQ, or auto",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    if variables.is_empty() {
        return Err(PyValueError::new_err(
            "recursive division requires a primary generator",
        ));
    }
    let dividend = polynomial_source(&left.inner, &variables, &mut generator, &mut verifier)?;
    let divisor = polynomial_source(&right.inner, &variables, &mut generator, &mut verifier)?;
    original_domain(
        &left.inner,
        Some(&dividend),
        &variables,
        left_variables,
        left_domain,
        &mut generator,
        &mut verifier,
    )?;
    original_domain(
        &right.inner,
        Some(&divisor),
        &variables,
        right_variables,
        right_domain,
        &mut generator,
        &mut verifier,
    )?;
    let left_integral = integral(&dividend, &mut generator)?;
    let right_integral = integral(&divisor, &mut generator)?;
    if method_scalar {
        charge(
            &mut generator,
            divisor.terms().len() * (variables.len() + 1),
            0,
        )?;
        // Coercion for Poly methods applies to the original scalar class;
        // a held symbolic expression cannot become a coefficient merely
        // because polynomial lowering happens to cancel all of its symbols.
        let constant = matches!(&right.inner, Expr::Integer(_) | Expr::Rational(_));
        let integer_left = left_domain != Some("QQ") && left_integral;
        if !constant || (integer_left && !right_integral) {
            return Err(PyValueError::new_err(
                "scalar does not belong to the left polynomial coefficient domain",
            ));
        }
    }
    let inputs_integral = left_integral && right_integral;
    if domain == "ZZ" && !inputs_integral {
        return Err(PyValueError::new_err(
            "noninteger coefficients cannot be represented in ZZ",
        ));
    }
    let input_qq = domain == "QQ" || (domain == "auto" && !inputs_integral);
    let operation_domain = if auto || input_qq {
        RecursiveDivisionDomain::QQ
    } else {
        RecursiveDivisionDomain::ZZ
    };
    let accepted = checked_recursive_division(
        &dividend,
        &divisor,
        operation_domain,
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    let (quotient, remainder) = accepted.into_parts();
    let outputs_integral =
        integral(&quotient, &mut generator)? && integral(&remainder, &mut generator)?;
    let result_domain = if input_qq || !outputs_integral {
        "QQ"
    } else {
        "ZZ"
    };
    let mut nodes = 0usize;
    for polynomial in [&quotient, &remainder] {
        let count = 1 + polynomial
            .terms()
            .keys()
            .map(|powers| 2 + 4 * powers.iter().filter(|&&power| power != 0).count())
            .sum::<usize>();
        if count > 4096 {
            return Err(PyValueError::new_err(
                "recursive division exceeds 4096 output nodes",
            ));
        }
        nodes += count;
    }
    let ring_bytes: usize = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
    charge(&mut generator, nodes, nodes * (2048 + ring_bytes))?;
    let quotient = checked_lift(&quotient, &mut verifier)?;
    let remainder = checked_lift(&remainder, &mut verifier)?;
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok((quotient, remainder, result_domain.into()))
}

/// Bound original sources for the separate, existing one-variable parametric
/// compatibility route. Coefficients may be rational functions in the other
/// variables, but no source denominator/guard may involve the main variable.
/// This admits inputs only: it is not a checked result of Python division.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (left, right, variables, primary=0, left_variables=None,
                   right_variables=None, left_domain=None, right_domain=None))]
pub fn poly_recursive_sources_native(
    left: &PyExpr,
    right: &PyExpr,
    variables: &Bound<'_, PyList>,
    primary: usize,
    left_variables: Option<&Bound<'_, PyList>>,
    right_variables: Option<&Bound<'_, PyList>>,
    left_domain: Option<&str>,
    right_domain: Option<&str>,
) -> PyResult<()> {
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    if primary >= variables.len() {
        return Err(PyValueError::new_err(
            "parametric division requires its original primary axis",
        ));
    }
    for (index, (source, original_variables, original_coefficient_domain)) in [
        (&left.inner, left_variables, left_domain),
        (&right.inner, right_variables, right_domain),
    ]
    .into_iter()
    .enumerate()
    {
        let accepted = metered_rational_source(source, &variables, &mut generator, &mut verifier)
            .map_err(polynomial_error)?;
        // Moving an original generator into the parameter field cannot make
        // a false retained QQ/ZZ polynomial declaration true (e.g. x/y in
        // original QQ[x,y], later requested as a polynomial only in x).
        original_domain(
            source,
            None,
            &variables,
            original_variables,
            original_coefficient_domain,
            &mut generator,
            &mut verifier,
        )?;
        if index == 1 && accepted.fraction().numerator.is_zero() {
            return Err(PyZeroDivisionError::new_err("polynomial division"));
        }
        for denominator in
            std::iter::once(&accepted.fraction().denominator).chain(accepted.guards())
        {
            charge(
                &mut verifier,
                denominator.terms().len() * (variables.len() + 1),
                0,
            )?;
            if denominator
                .terms()
                .keys()
                .any(|powers| powers[primary] != 0)
            {
                return Err(PyValueError::new_err(
                    "polynomial division source has an original pole in its primary variable",
                ));
            }
        }
    }
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(())
}
