//! Typed public modular factorization with checked exact source composition.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{
    PrimeFieldSourceRequest, UnivariatePoly, admit_prime_field_source,
    checked_prime_field_source_factorization,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

type FactorParts = (PyExpr, Vec<(PyExpr, u32)>);

fn budget() -> Budget {
    let mut limits = BudgetLimits::uniform(128 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 20_000_000;
    Budget::new(limits)
}

fn charge(meter: &mut Budget, steps: usize, bytes: usize) -> PyResult<()> {
    meter.checkpoint().map_err(to_value_error)?;
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(to_value_error)
}

fn symbol(variable: &PySymbolBinding, meter: &mut Budget) -> PyResult<Symbol> {
    if variable.name.len() > 65_536 {
        return Err(PyValueError::new_err(
            "modular generator name exceeds 65536 bytes",
        ));
    }
    charge(
        meter,
        1,
        variable.name.len() + std::mem::size_of::<Symbol>(),
    )?;
    Ok(variable.to_symbol())
}

/// Borrow the original tree before the shell inspects `.args` or free symbols.
#[pyfunction]
pub fn prime_field_source_preflight_native(source: &PyExpr) -> PyResult<()> {
    crate::ideal_operations::ideal_source_preflight_native(source)
}

/// Admit the whole original before shell numerator/denominator transformations.
#[pyfunction]
pub fn prime_field_source_admit_native(
    source: &PyExpr,
    variable: &PySymbolBinding,
    prime: u64,
) -> PyResult<()> {
    let mut meter = budget();
    let variable = symbol(variable, &mut meter)?;
    admit_prime_field_source(&source.inner, &variable, prime, &mut meter).map_err(to_value_error)
}

/// Every normalization hint is independently checked against `source`.
#[pyfunction]
pub fn prime_field_factor_source_native(
    source: &PyExpr,
    variable: &PySymbolBinding,
    prime: u64,
    content: &PyExpr,
    pieces: &Bound<'_, PyList>,
) -> PyResult<FactorParts> {
    let (mut generator, mut verifier) = (budget(), budget());
    let variable = symbol(variable, &mut generator)?;
    // Direct bridge callers get the same original-source admission as the shell.
    admit_prime_field_source(&source.inner, &variable, prime, &mut generator)
        .map_err(to_value_error)?;
    if pieces.len() > 64 {
        return Err(PyValueError::new_err(
            "modular source has more than 64 components",
        ));
    }
    let value = match &content.inner {
        Expr::Integer(value) if value.bits() <= 4096 => {
            charge(&mut generator, 1, value.bits().div_ceil(8) as usize + 128)?;
            BigRational::from_integer(value.clone())
        }
        Expr::Rational(value) if value.numer().bits() <= 4096 && value.denom().bits() <= 4096 => {
            charge(
                &mut generator,
                1,
                ((value.numer().bits() + value.denom().bits()).div_ceil(8)) as usize + 128,
            )?;
            value.clone()
        }
        _ => {
            return Err(PyValueError::new_err(
                "modular exterior coefficient must be a bounded exact rational",
            ));
        }
    };
    charge(&mut generator, pieces.len(), pieces.len() * 256)?;
    let mut objects: Vec<(Py<PyExpr>, u32)> = Vec::with_capacity(pieces.len());
    for item in pieces.iter() {
        let tuple = item.cast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "modular components require (expression, positive integer power) pairs",
            ));
        }
        let expression = tuple.get_item(0)?.extract::<Py<PyExpr>>()?;
        prime_field_source_preflight_native(&expression.borrow(pieces.py()))?;
        let power = tuple.get_item(1)?.extract::<u32>()?;
        objects.push((expression, power));
    }
    let borrowed: Vec<_> = objects
        .iter()
        .map(|(expression, power)| (expression.borrow(pieces.py()), *power))
        .collect();
    let components: Vec<_> = borrowed
        .iter()
        .map(|(expression, power)| (&expression.inner, *power))
        .collect();
    let request = PrimeFieldSourceRequest {
        source: &source.inner,
        variable: &variable,
        prime,
        content: &value,
        components: &components,
    };
    let accepted =
        checked_prime_field_source_factorization(&request, &mut generator, &mut verifier)
            .map_err(to_value_error)?;
    charge(
        &mut generator,
        accepted.factors().len(),
        accepted.factors().len() * 256
            + ((accepted.unit().numer().bits() + accepted.unit().denom().bits()).div_ceil(8))
                as usize
            + 256,
    )?;
    let mut factors = Vec::with_capacity(accepted.factors().len());
    for factor in accepted.factors() {
        charge(
            &mut generator,
            factor.coefficients.len(),
            // to_expr owns a typed symbol in every nonconstant monomial, and
            // UnivariatePoly also owns its generator until lifting completes.
            factor.coefficients.len() * (512 + variable.name.len()),
        )?;
        let coefficients = factor
            .coefficients
            .iter()
            .map(|&coefficient| {
                BigRational::from(if coefficient > prime / 2 {
                    coefficient as i64 - prime as i64
                } else {
                    coefficient as i64
                })
            })
            .collect();
        let polynomial = UnivariatePoly::new(variable.clone(), coefficients);
        factors.push((PyExpr::from_expr(polynomial.to_expr()), factor.multiplicity));
    }
    let result = (
        PyExpr::from_expr(rational_expr(accepted.unit().clone())),
        factors,
    );
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(result)
}
