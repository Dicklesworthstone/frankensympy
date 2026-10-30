//! Typed bounded bridge for checked proper rational inverse Laplace transforms.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_calculus::rational_laplace::{RationalLaplaceError, metered_inverse_rational_laplace};
use fsym_core::{Expr, Symbol};
use fsym_polys::{PolyError, bounded_polynomial_from_expr};
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use std::mem::size_of;

const MAX_SOURCE_NODES: usize = 4096;
const MAX_SOURCE_DEPTH: usize = 64;
const MAX_SOURCE_TEXT: usize = 65_536;
// The legacy polynomial transform lane supports degree 4096. The exact
// rational inverse kernel keeps its own, smaller degree/power-32 envelope.
const MAX_SOURCE_POWER: u64 = 4096;
const MAX_NATIVE_DEGREE: usize = 32;

fn budget() -> Budget {
    let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 200_000_000;
    Budget::new(limits)
}

fn charge(meter: &mut Budget, nodes: usize, bytes: usize) -> PyResult<()> {
    meter
        .try_charge_batch(&[
            (Dimension::ComputeSteps, nodes.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(to_value_error)?;
    Ok(())
}

fn integer_power(expression: &Expr) -> Option<i64> {
    match expression {
        Expr::Integer(value) => value.to_i64(),
        Expr::Rational(value) if value.is_integer() => value.numer().to_i64(),
        _ => None,
    }
}

struct OriginalShape<'a> {
    rational: bool,
    exceeds_native_power: bool,
    unresolved_power: bool,
    nonpositive_bases: Vec<&'a Expr>,
}

/// Fully inspect the borrowed original before any route, zero shortcut or
/// legacy expansion. Retain nested power bases even below a held outer ^0.
fn original_shape<'a>(
    source: &'a Expr,
    variable: Option<&Symbol>,
    meter: &mut Budget,
) -> PyResult<OriginalShape<'a>> {
    charge(
        meter,
        1,
        MAX_SOURCE_NODES * (size_of::<(&Expr, usize)>() + size_of::<&Expr>()),
    )?;
    let mut pending = Vec::with_capacity(MAX_SOURCE_NODES);
    pending.push((source, 0usize));
    let mut shape = OriginalShape {
        rational: true,
        exceeds_native_power: false,
        unresolved_power: false,
        nonpositive_bases: Vec::with_capacity(MAX_SOURCE_NODES),
    };
    let mut visited = 0usize;
    let mut text_bytes = variable.map_or(0, |symbol| symbol.name.len());
    if text_bytes > MAX_SOURCE_TEXT {
        return Err(PyValueError::new_err(
            "transform source names exceed their bound",
        ));
    }
    while let Some((node, depth)) = pending.pop() {
        visited += 1;
        if visited > MAX_SOURCE_NODES || depth > MAX_SOURCE_DEPTH {
            return Err(PyValueError::new_err(
                "transform source exceeds its bounded node/depth envelope",
            ));
        }
        charge(meter, 1, 1)?;
        match node {
            Expr::Integer(value) => {
                if value.bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "transform coefficient exceeds 4096 bits",
                    ));
                }
            }
            Expr::Rational(value) => {
                if value.numer().bits() > 4096 || value.denom().bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "transform coefficient exceeds 4096 bits",
                    ));
                }
            }
            Expr::Sym(symbol) => {
                text_bytes = text_bytes.saturating_add(symbol.name.len());
                if text_bytes > MAX_SOURCE_TEXT {
                    return Err(PyValueError::new_err(
                        "transform source names exceed their bound",
                    ));
                }
                charge(meter, symbol.name.len().div_ceil(64), 1)?;
                if let Some(variable) = variable {
                    shape.rational &= symbol == variable;
                }
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = node {
                    shape.rational = false;
                    text_bytes = text_bytes.saturating_add(name.len());
                    if text_bytes > MAX_SOURCE_TEXT {
                        return Err(PyValueError::new_err(
                            "transform source names exceed their bound",
                        ));
                    }
                }
                if children.len() > MAX_SOURCE_NODES.saturating_sub(visited + pending.len()) {
                    return Err(PyValueError::new_err(
                        "transform source exceeds its bounded node envelope",
                    ));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                let literal_integer = matches!(exponent.as_ref(), Expr::Integer(_))
                    || matches!(exponent.as_ref(), Expr::Rational(value) if value.is_integer());
                match integer_power(exponent) {
                    Some(power) if power.unsigned_abs() <= MAX_SOURCE_POWER => {
                        shape.exceeds_native_power |=
                            power.unsigned_abs() > MAX_NATIVE_DEGREE as u64;
                        if power <= 0 {
                            shape.nonpositive_bases.push(base.as_ref());
                        }
                    }
                    Some(_) | None if literal_integer => {
                        return Err(PyValueError::new_err(
                            "transform integer power exceeds its bounded envelope",
                        ));
                    }
                    _ => shape.unresolved_power = true,
                }
                if 2 > MAX_SOURCE_NODES.saturating_sub(visited + pending.len()) {
                    return Err(PyValueError::new_err(
                        "transform source exceeds its bounded node envelope",
                    ));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
            }
            _ => shape.rational = false,
        }
    }
    Ok(shape)
}

/// Admit one power base as a polynomial in its complete typed symbol ring.
/// All nodes already passed original_shape; this bounded second walk preserves
/// identity and prepays every retained name before polynomial construction.
fn power_base_ring(base: &Expr, meter: &mut Budget) -> PyResult<Option<Vec<Symbol>>> {
    charge(
        meter,
        1,
        MAX_SOURCE_NODES * size_of::<&Expr>() + 8 * size_of::<&Symbol>(),
    )?;
    let mut pending = Vec::with_capacity(MAX_SOURCE_NODES);
    pending.push(base);
    let mut symbols: Vec<&Symbol> = Vec::with_capacity(8);
    let mut too_many = false;
    while let Some(node) = pending.pop() {
        charge(meter, symbols.len() + 1, 1)?;
        match node {
            Expr::Sym(symbol) if !symbols.contains(&symbol) => {
                if symbols.len() == 8 {
                    too_many = true;
                } else {
                    symbols.push(symbol);
                }
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                pending.extend(children);
            }
            Expr::Pow(base, exponent) => {
                pending.push(base.as_ref());
                pending.push(exponent.as_ref());
            }
            _ => {}
        }
    }
    if too_many {
        return Ok(None);
    }
    let bytes = symbols
        .iter()
        .map(|symbol| symbol.name.len())
        .sum::<usize>()
        + symbols.len() * size_of::<Symbol>();
    charge(meter, symbols.len(), bytes)?;
    Ok(Some(symbols.into_iter().cloned().collect()))
}

fn check_original_domains(shape: &OriginalShape<'_>, meter: &mut Budget) -> PyResult<()> {
    let mut unresolved = shape.unresolved_power;
    // Descendant bases precede their parents. An unsupported outer rational
    // base must not hide an inner, identically zero polynomial denominator.
    for base in shape.nonpositive_bases.iter().rev() {
        let Some(variables) = power_base_ring(base, meter)? else {
            unresolved = true;
            continue;
        };
        match bounded_polynomial_from_expr(base, &variables, meter) {
            Ok(polynomial) if polynomial.is_zero() => {
                return Err(PyValueError::new_err(
                    "held transform source has a zero base with a nonpositive exponent",
                ));
            }
            Ok(_) => {}
            Err(PolyError::NonPolynomialExpression(_)) => unresolved = true,
            // Resource failures and every other actual admission failure stay
            // terminal; they cannot authorize a legacy-table retry.
            Err(error) => return Err(to_value_error(error)),
        }
    }
    if unresolved {
        return Err(PyNotImplementedError::new_err(
            "the original held transform domains cannot be established for table fallback",
        ));
    }
    charge(meter, 1, 1)?;
    Ok(())
}

/// Shared source-domain gate for forward and inverse legacy transform tables.
/// This performs no expansion and never converts unknown domains to zero.
pub(crate) fn validate_held_transform_domains(source: &Expr, meter: &mut Budget) -> PyResult<()> {
    let shape = original_shape(source, None, meter)?;
    check_original_domains(&shape, meter)
}

/// Conservative degree bounds for the same uncancelled fraction operations
/// used by the native inverse. Values saturate at 33: no polynomial algebra or
/// large exponent arithmetic is needed to recognize a declared larger shape.
fn native_degree_envelope(
    source: &Expr,
    meter: &mut Budget,
    exceeds: &mut bool,
) -> PyResult<(usize, usize)> {
    charge(meter, 1, 1)?;
    let cap = MAX_NATIVE_DEGREE + 1;
    let add = |left: usize, right: usize| left.saturating_add(right).min(cap);
    let degree = match source {
        Expr::Sym(_) => (1, 0),
        Expr::Add(children) => {
            let mut accumulated: Option<(usize, usize)> = None;
            for child in children {
                let part = native_degree_envelope(child, meter, exceeds)?;
                accumulated = Some(match accumulated {
                    Some(previous) => (
                        add(previous.0, part.1).max(add(part.0, previous.1)),
                        add(previous.1, part.1),
                    ),
                    None => part,
                });
            }
            accumulated.unwrap_or((0, 0))
        }
        Expr::Mul(children) => {
            let mut accumulated = (0, 0);
            for child in children {
                let part = native_degree_envelope(child, meter, exceeds)?;
                accumulated = (add(accumulated.0, part.0), add(accumulated.1, part.1));
            }
            accumulated
        }
        Expr::Pow(base, exponent) => {
            let base = native_degree_envelope(base, meter, exceeds)?;
            let power = integer_power(exponent)
                .ok_or_else(|| PyNotImplementedError::new_err("unknown transform power degree"))?;
            let magnitude = power.unsigned_abs() as usize;
            let positive = (
                base.0.saturating_mul(magnitude).min(cap),
                base.1.saturating_mul(magnitude).min(cap),
            );
            if power < 0 {
                (positive.1, positive.0)
            } else {
                positive
            }
        }
        _ => (0, 0),
    };
    *exceeds |= degree.0 > MAX_NATIVE_DEGREE || degree.1 > MAX_NATIVE_DEGREE;
    Ok(degree)
}

/// Declared unsupported or larger shapes are routed before the checked kernel.
/// Every fallback still passes original held-domain admission first.
fn rational_shape(source: &Expr, variable: &Symbol, meter: &mut Budget) -> PyResult<bool> {
    let shape = original_shape(source, Some(variable), meter)?;
    let mut exceeds = shape.exceeds_native_power;
    if shape.rational && !shape.unresolved_power {
        native_degree_envelope(source, meter, &mut exceeds)?;
    }
    if shape.rational && !exceeds {
        return Ok(true);
    }
    validate_held_transform_domains(source, meter)?;
    Ok(false)
}

#[pyfunction]
pub fn inverse_rational_laplace_native(
    source: &PyExpr,
    transform_variable: &PySymbolBinding,
    time_variable: &PySymbolBinding,
) -> PyResult<Option<PyExpr>> {
    let names = transform_variable
        .name
        .len()
        .checked_add(time_variable.name.len())
        .filter(|&size| size <= MAX_SOURCE_TEXT)
        .ok_or_else(|| {
            PyValueError::new_err("Laplace variable names exceed their bounded envelope")
        })?;
    let mut generator = budget();
    let mut verifier = budget();
    charge(&mut generator, 2, names + 256)?;
    let transform_variable = transform_variable.to_symbol();
    let time_variable = time_variable.to_symbol();
    if transform_variable == time_variable {
        return Err(PyValueError::new_err(
            "Laplace input and output variables must be distinct",
        ));
    }
    if !rational_shape(&source.inner, &transform_variable, &mut generator)? {
        return Ok(None);
    }
    let checked = metered_inverse_rational_laplace(
        &source.inner,
        &transform_variable,
        &time_variable,
        &mut generator,
        &mut verifier,
    )
    .map_err(|error| match error {
        RationalLaplaceError::Unsupported(message) => PyNotImplementedError::new_err(message),
        error => to_value_error(error),
    })?;
    charge(&mut generator, 1, 1)?;
    charge(&mut verifier, 1, 1)?;
    Ok(Some(PyExpr::from_expr(checked.into_function())))
}
