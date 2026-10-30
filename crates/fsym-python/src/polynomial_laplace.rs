//! Typed dispatch for checked polynomial-weighted elementary Laplace transforms.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::rational_laplace::validate_held_transform_domains;
use crate::{rational_to_expr, to_value_error};
use fsym_calculus::rational_laplace::{RationalLaplaceError, metered_laplace};
use fsym_core::{Expr, Symbol};
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use num_traits::Zero;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

fn budget() -> Budget {
    let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 200_000_000;
    Budget::new(limits)
}

fn charge(meter: &mut Budget, steps: usize, bytes: usize) -> PyResult<()> {
    meter
        .try_charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(to_value_error)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct Shape {
    eligible: bool,
    degree: usize,
    analytic: bool,
    oscillatory: usize,
    blocks: usize,
    zero_constant: bool,
}

impl Shape {
    fn outside() -> Self {
        Self {
            eligible: false,
            degree: 0,
            analytic: false,
            oscillatory: 0,
            blocks: 1,
            zero_constant: false,
        }
    }

    fn constant(zero: bool) -> Self {
        Self {
            eligible: true,
            degree: 0,
            analytic: false,
            oscillatory: 0,
            blocks: 1,
            zero_constant: zero,
        }
    }
}

/// Recognition only: never accepts a source or consumes a fresh algorithm budget.
/// Conservative degree bounds preserve existing large-polynomial table cases.
struct Scan<'a> {
    variable: &'a Symbol,
    meter: &'a mut Budget,
    nodes: usize,
    text: usize,
}

impl Scan<'_> {
    fn visit(&mut self, source: &Expr, depth: usize) -> PyResult<Shape> {
        self.nodes += 1;
        if self.nodes > 4096 || depth > 64 {
            return Err(PyValueError::new_err(
                "Laplace source exceeds its bounded envelope",
            ));
        }
        charge(self.meter, 1, 32)?;
        let result = match source {
            Expr::Integer(value) => {
                if value.bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "Laplace coefficient exceeds 4096 bits",
                    ));
                }
                Shape::constant(value.is_zero())
            }
            Expr::Rational(value) => {
                if value.numer().bits() > 4096 || value.denom().bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "Laplace coefficient exceeds 4096 bits",
                    ));
                }
                Shape::constant(value.is_zero())
            }
            Expr::Sym(symbol) => {
                self.name(&symbol.name)?;
                Shape {
                    eligible: symbol == self.variable,
                    degree: 1,
                    analytic: false,
                    oscillatory: 0,
                    blocks: 1,
                    zero_constant: true,
                }
            }
            Expr::Add(children) => {
                self.fanout(children.len())?;
                let mut result = Shape::constant(true);
                let mut blocks = 0usize;
                for child in children {
                    let shape = self.visit(child, depth + 1)?;
                    result.eligible &= shape.eligible;
                    result.degree = result.degree.max(shape.degree);
                    result.analytic |= shape.analytic;
                    result.oscillatory = result.oscillatory.max(shape.oscillatory);
                    result.zero_constant &= shape.zero_constant;
                    blocks = blocks.saturating_add(shape.blocks);
                }
                result.blocks = if result.analytic { blocks.max(1) } else { 1 };
                result
            }
            Expr::Mul(children) => {
                self.fanout(children.len())?;
                let mut result = Shape::constant(false);
                for child in children {
                    let shape = self.visit(child, depth + 1)?;
                    result.eligible &= shape.eligible;
                    result.degree = result.degree.saturating_add(shape.degree);
                    result.analytic |= shape.analytic;
                    result.oscillatory = result.oscillatory.saturating_add(shape.oscillatory);
                    result.blocks = result.blocks.saturating_mul(shape.blocks);
                    result.zero_constant |= shape.zero_constant;
                }
                result
            }
            Expr::Pow(base, exponent) => {
                self.fanout(2)?;
                let shape = self.visit(base, depth + 1)?;
                let exponent_shape = self.visit(exponent, depth + 1)?;
                let integer = match exponent.as_ref() {
                    Expr::Integer(value) => value.to_i64(),
                    Expr::Rational(value) if value.is_integer() => value.numer().to_i64(),
                    _ => None,
                };
                if let Some(power) = integer {
                    if power.unsigned_abs() > 4096 {
                        return Err(PyValueError::new_err("Laplace source power exceeds 4096"));
                    }
                    if power < 0 {
                        Shape {
                            eligible: shape.eligible
                                && exponent_shape.eligible
                                && shape.degree == 0
                                && !shape.analytic,
                            ..Shape::constant(false)
                        }
                    } else {
                        let power = power as u32;
                        Shape {
                            eligible: shape.eligible && exponent_shape.eligible,
                            degree: shape.degree.saturating_mul(power as usize),
                            analytic: shape.analytic,
                            oscillatory: shape.oscillatory.saturating_mul(power as usize),
                            blocks: shape.blocks.saturating_pow(power),
                            zero_constant: power != 0 && shape.zero_constant,
                        }
                    }
                } else {
                    Shape::outside()
                }
            }
            Expr::Function(name, children) => {
                self.name(name)?;
                self.fanout(children.len())?;
                let mut argument = Shape::outside();
                for child in children {
                    argument = self.visit(child, depth + 1)?;
                }
                let supported = matches!(name.as_str(), "exp" | "sin" | "cos" | "sinh" | "cosh");
                Shape {
                    eligible: supported
                        && children.len() == 1
                        && argument.eligible
                        && !argument.analytic
                        && argument.degree <= 1
                        && argument.zero_constant,
                    degree: 0,
                    analytic: true,
                    oscillatory: usize::from(matches!(name.as_str(), "sin" | "cos")),
                    blocks: if matches!(name.as_str(), "sinh" | "cosh") {
                        2
                    } else {
                        1
                    },
                    zero_constant: false,
                }
            }
            _ => Shape::outside(),
        };
        Ok(result)
    }

    fn name(&mut self, name: &str) -> PyResult<()> {
        self.text = self.text.saturating_add(name.len());
        if self.text > 65_536 {
            return Err(PyValueError::new_err(
                "Laplace source names exceed their bounded envelope",
            ));
        }
        charge(self.meter, name.len(), name.len())
    }

    fn fanout(&self, count: usize) -> PyResult<()> {
        if count > 4096usize.saturating_sub(self.nodes) {
            return Err(PyValueError::new_err(
                "Laplace source exceeds its bounded envelope",
            ));
        }
        Ok(())
    }
}

#[pyfunction]
pub fn polynomial_laplace_native(
    source: &PyExpr,
    time_variable: &PySymbolBinding,
    transform_variable: &PySymbolBinding,
) -> PyResult<Option<(PyExpr, Option<PyExpr>)>> {
    let names = time_variable
        .name
        .len()
        .checked_add(transform_variable.name.len())
        .filter(|&size| size <= 65_536)
        .ok_or_else(|| {
            PyValueError::new_err("Laplace variable names exceed their bounded envelope")
        })?;
    let mut generator = budget();
    let mut verifier = budget();
    charge(&mut generator, 2, names + 256)?;
    let time_variable = time_variable.to_symbol();
    let transform_variable = transform_variable.to_symbol();
    if time_variable == transform_variable {
        return Err(PyValueError::new_err(
            "Laplace input and output variables must be distinct",
        ));
    }
    let shape = Scan {
        variable: &time_variable,
        meter: &mut generator,
        nodes: 0,
        text: 0,
    }
    .visit(&source.inner, 0)?;
    let denominator_bound = shape
        .blocks
        .saturating_mul(shape.degree.saturating_add(1))
        .saturating_mul(if shape.oscillatory == 0 { 1 } else { 2 });
    if !shape.eligible || !shape.analytic || shape.oscillatory > 1 || denominator_bound > 32 {
        validate_held_transform_domains(&source.inner, &mut generator)?;
        return Ok(None);
    }
    let checked = metered_laplace(
        &source.inner,
        &time_variable,
        &transform_variable,
        &mut generator,
        &mut verifier,
    )
    .map_err(|error| match error {
        RationalLaplaceError::Unsupported(message) => PyNotImplementedError::new_err(message),
        error => to_value_error(error),
    })?;
    charge(&mut generator, 1, 16_512)?;
    charge(&mut verifier, 1, 1)?;
    let (transform, abscissa) = checked.into_parts();
    Ok(Some((
        PyExpr::from_expr(transform),
        abscissa.map(rational_to_expr).map(PyExpr::from_expr),
    )))
}
