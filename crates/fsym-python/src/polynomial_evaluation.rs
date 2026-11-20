//! Source-bound exact rational polynomial evaluation for the Python shell.
//!
//! Original polynomial lowering, independent source replay, all assignments,
//! exact contraction, independent coefficient verification and the output lift
//! retain one generator ledger and one verifier ledger for the whole request.

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, lift_basis, symbols};
use crate::recursive_division::polynomial_source;
use crate::to_value_error;
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Expr};
use fsym_polys::{bounded_polynomial_from_expr, checked_polynomial_evaluation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

pyo3::create_exception!(_native, PolynomialEvaluationCoefficientError, PyValueError);
pyo3::create_exception!(_native, PolynomialEvaluationDomainError, PyValueError);

#[pyfunction]
#[pyo3(signature = (source, variables, values, domain="auto", auto=true))]
pub fn polynomial_evaluate_native(
    source: &PyExpr,
    variables: &Bound<'_, PyList>,
    values: &Bound<'_, PyList>,
    domain: &str,
    auto: bool,
) -> PyResult<(PyExpr, String)> {
    if !matches!(domain, "auto" | "ZZ" | "QQ") {
        return Err(PyValueError::new_err(
            "checked polynomial evaluation supports ZZ and QQ",
        ));
    }
    if values.len() != variables.len() || values.len() > 8 {
        return Err(PyValueError::new_err(
            "polynomial evaluation requires one assignment per original generator",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    let source = polynomial_source(&source.inner, &variables, &mut generator, &mut verifier)?;
    charge(&mut generator, source.terms().len(), 0)?;
    let integral = source.terms().values().all(BigRational::is_integer);
    if domain == "ZZ" && !integral {
        return Err(PolynomialEvaluationCoefficientError::new_err(
            "noninteger source coefficients cannot be represented in ZZ",
        ));
    }
    let source_qq = domain == "QQ" || (domain == "auto" && !integral);
    charge(
        &mut generator,
        values.len(),
        values.len() * std::mem::size_of::<Option<BigRational>>(),
    )?;
    let mut assignments = Vec::with_capacity(values.len());
    let mut fractional_assignment = false;
    for value in values.iter() {
        if value.is_none() {
            assignments.push(None);
            continue;
        }
        let value: PyRef<'_, PyExpr> = value.extract()?;
        let coefficient = match &value.inner {
            Expr::Integer(integer) if integer.bits() <= 4096 => {
                charge(&mut generator, 1, 64 + 8 * integer.limb_count() as usize)?;
                BigRational::from_integer(integer.clone())
            }
            Expr::Rational(rational)
                if rational.numer().bits() <= 4096 && rational.denom().bits() <= 4096 =>
            {
                charge(
                    &mut generator,
                    1,
                    64 + 8
                        * (rational.numer().limb_count() + rational.denom().limb_count()) as usize,
                )?;
                rational.clone()
            }
            _ => {
                return Err(PyValueError::new_err(
                    "polynomial evaluation values must be literal exact rationals of at most 4096 bits",
                ));
            }
        };
        fractional_assignment |= !coefficient.is_integer();
        assignments.push(Some(coefficient));
    }
    // Promotion depends on admitted values even if the source is constant or
    // zero. A later cancellation cannot make an inadmissible ZZ value legal.
    if !source_qq && fractional_assignment && !auto {
        return Err(PolynomialEvaluationDomainError::new_err(
            "cannot evaluate a ZZ polynomial at a noninteger rational with auto=False",
        ));
    }
    let result_domain = if source_qq || fractional_assignment {
        "QQ"
    } else {
        "ZZ"
    };
    let accepted =
        checked_polynomial_evaluation(&source, &assignments, &mut generator, &mut verifier)
            .map_err(to_value_error)?;
    let ring_bytes = accepted
        .result()
        .generators()
        .iter()
        .map(|symbol| symbol.name.len() + 128)
        .sum();
    let mut lifted = lift_basis(
        std::slice::from_ref(accepted.result()),
        ring_bytes,
        &mut generator,
        &mut verifier,
    )?;
    let output = lifted
        .pop()
        .ok_or_else(|| PyValueError::new_err("checked evaluation output is missing"))?;
    // The output encoding is itself bound to the independently accepted map.
    let recovered =
        bounded_polynomial_from_expr(&output.inner, accepted.result().generators(), &mut verifier)
            .map_err(to_value_error)?;
    let comparison_steps = accepted
        .result()
        .terms()
        .iter()
        .map(|(powers, coefficient)| {
            powers.len()
                + coefficient.numer().limb_count() as usize
                + coefficient.denom().limb_count() as usize
                + 1
        })
        .sum();
    charge(&mut verifier, comparison_steps, 0)?;
    if &recovered != accepted.result() {
        return Err(PyValueError::new_err(
            "polynomial evaluation output does not match its accepted coefficient identity",
        ));
    }
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok((output, result_domain.into()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "PolynomialEvaluationCoefficientError",
        module
            .py()
            .get_type::<PolynomialEvaluationCoefficientError>(),
    )?;
    module.add(
        "PolynomialEvaluationDomainError",
        module.py().get_type::<PolynomialEvaluationDomainError>(),
    )?;
    module.add_function(wrap_pyfunction!(polynomial_evaluate_native, module)?)?;
    Ok(())
}
