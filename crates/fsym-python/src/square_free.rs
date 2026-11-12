//! Source-bound square-free decomposition over QQ with public ZZ normalization.
//!
//! The native receipt proves complete multiplicities before normalization.
//! Source conversion, content arithmetic and every lifted output share one
//! generator ledger and a separate verifier ledger.

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, lift_basis, symbols};
use crate::{polynomial_content, rational_to_expr, to_value_error};
use fsym_core::BigRational;
use fsym_polys::{
    MultivariatePoly, bounded_polynomial_from_expr, checked_multivariate_square_free,
};
use fsym_runtime::Budget;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

type SquareFreeResult = (PyExpr, Vec<(PyExpr, usize)>, PyExpr, PyExpr, bool);

fn bound_coefficient(value: BigRational) -> PyResult<BigRational> {
    if value.numer().bits() > 4096 || value.denom().bits() > 4096 {
        return Err(PyValueError::new_err(
            "square-free normalization exceeds 4096 coefficient bits",
        ));
    }
    Ok(value)
}

/// Return the removed positive rational content. A monic QQ polynomial
/// becomes its primitive integer associate; zero remains zero.
fn primitive(polynomial: &mut MultivariatePoly, meter: &mut Budget) -> PyResult<BigRational> {
    if polynomial.is_zero() {
        return Ok(BigRational::from_integer(1.into()));
    }
    let content = polynomial_content(polynomial, meter)?;
    for coefficient in polynomial.terms.values_mut() {
        *coefficient = bound_coefficient(
            coefficient
                .metered_div(&content, meter)
                .map_err(to_value_error)?,
        )?;
    }
    Ok(content)
}

/// Return (scale, multiplicity groups, square-free part, expanded source,
/// source-has-integer-coefficients). ``normalization`` governs the factors,
/// not input coercion: expression-level sqf_list may place rational content
/// outside primitive integer factors. The shell separately binds retained
/// Poly domains to their original declared ring.
#[pyfunction]
#[pyo3(signature = (source, vars, normalization="QQ"))]
pub fn poly_multivariate_sqf_native(
    source: &PyExpr,
    vars: &Bound<'_, PyList>,
    normalization: &str,
) -> PyResult<SquareFreeResult> {
    if !matches!(normalization, "ZZ" | "QQ" | "auto") {
        return Err(PyValueError::new_err(
            "square-free normalization must be ZZ, QQ, or auto",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(vars, &mut generator)?;
    let input = bounded_polynomial_from_expr(&source.inner, &variables, &mut generator)
        .map_err(to_value_error)?;
    charge(&mut generator, input.terms().len(), 0)?;
    let integral = input
        .terms()
        .values()
        .all(|coefficient| coefficient.is_integer());
    let accepted = checked_multivariate_square_free(&input, &mut generator, &mut verifier)
        .map_err(to_value_error)?;
    let (mut scale, mut factors, mut part) = accepted.into_parts();
    if normalization == "ZZ" || (normalization == "auto" && integral) {
        for factor in &mut factors {
            let content = primitive(&mut factor.polynomial, &mut generator)?;
            // Multiply into the scale one step at a time so cross-cancellation
            // and the coefficient bound apply before a large power is built.
            for _ in 0..factor.multiplicity {
                scale = bound_coefficient(
                    scale
                        .metered_mul(&content, &mut generator)
                        .map_err(to_value_error)?,
                )?;
            }
        }
        primitive(&mut part, &mut generator)?;
    }

    let count = factors.len();
    charge(
        &mut generator,
        count + 2,
        (count + 2) * std::mem::size_of::<MultivariatePoly>()
            + count * (std::mem::size_of::<usize>() + std::mem::size_of::<(PyExpr, usize)>()),
    )?;
    let mut multiplicities = Vec::with_capacity(count);
    let mut polynomials = Vec::with_capacity(count + 2);
    for factor in factors {
        multiplicities.push(factor.multiplicity);
        polynomials.push(factor.polynomial);
    }
    polynomials.push(part);
    polynomials.push(input);
    let ring_bytes = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
    let mut lifted = lift_basis(polynomials, ring_bytes, &mut generator, &mut verifier)?;
    let expanded = lifted
        .pop()
        .ok_or_else(|| PyValueError::new_err("missing admitted square-free source"))?;
    let part = lifted
        .pop()
        .ok_or_else(|| PyValueError::new_err("missing checked square-free part"))?;
    let factors = lifted.into_iter().zip(multiplicities).collect();
    charge(&mut generator, 1, 2048)?;
    let scale = PyExpr::from_expr(rational_to_expr(bound_coefficient(scale)?));
    charge(&mut generator, 1, 1)?;
    charge(&mut verifier, 1, 1)?;
    Ok((scale, factors, part, expanded, integral))
}
