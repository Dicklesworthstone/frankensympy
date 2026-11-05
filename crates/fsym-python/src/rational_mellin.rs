//! Typed bounded bridge for strip-aware ordinary rational inverse Mellin transforms.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_calculus::rational_mellin::{
    MellinStrip, MellinStripBound, RationalMellinError, metered_inverse_rational_mellin,
};
use fsym_core::{BigRational, Constant, Expr};
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use std::mem::size_of;

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

fn bound(source: &Expr, meter: &mut Budget) -> PyResult<MellinStripBound> {
    match source {
        Expr::Const(Constant::NegativeInfinity) => Ok(MellinStripBound::NegativeInfinity),
        Expr::Const(Constant::Infinity) => Ok(MellinStripBound::PositiveInfinity),
        Expr::Integer(value) => {
            if value.bits() > 4096 {
                return Err(PyValueError::new_err(
                    "Mellin strip bound exceeds 4096 bits",
                ));
            }
            charge(meter, 1, 128 + value.bits().div_ceil(8) as usize)?;
            Ok(MellinStripBound::Finite(BigRational::from_integer(
                value.clone(),
            )))
        }
        Expr::Rational(value) => {
            let bits = value.numer().bits().max(value.denom().bits());
            if bits > 4096 {
                return Err(PyValueError::new_err(
                    "Mellin strip bound exceeds 4096 bits",
                ));
            }
            charge(meter, 1, 128 + 2 * bits.div_ceil(8) as usize)?;
            Ok(MellinStripBound::Finite(value.clone()))
        }
        _ => Err(PyNotImplementedError::new_err(
            "inverse Mellin strip bounds must be exact rational literals or real infinities",
        )),
    }
}

fn inverse_error(error: RationalMellinError) -> PyErr {
    if error.is_unsupported() {
        PyNotImplementedError::new_err(error.to_string())
    } else {
        to_value_error(error)
    }
}

#[pyfunction]
pub fn inverse_rational_mellin_native(
    source: &PyExpr,
    transform_variable: &PySymbolBinding,
    argument_variable: &PySymbolBinding,
    lower: &PyExpr,
    upper: &PyExpr,
) -> PyResult<PyExpr> {
    let names = transform_variable
        .name
        .len()
        .checked_add(argument_variable.name.len())
        .filter(|&size| size <= 65_536)
        .ok_or_else(|| PyValueError::new_err("Mellin variable names exceed their bound"))?;
    let mut generator = budget();
    let mut verifier = budget();
    charge(&mut generator, 2, names + 256)?;
    let transform_variable = transform_variable.to_symbol();
    let argument_variable = argument_variable.to_symbol();
    let strip = MellinStrip {
        lower: bound(&lower.inner, &mut generator)?,
        upper: bound(&upper.inner, &mut generator)?,
    };
    let receipt = metered_inverse_rational_mellin(
        &source.inner,
        &transform_variable,
        &argument_variable,
        &strip,
        &mut generator,
        &mut verifier,
    )
    .map_err(inverse_error)?;
    charge(&mut generator, 1, 1)?;
    charge(&mut verifier, 1, size_of::<PyExpr>() + 128)?;
    Ok(PyExpr::from_expr(receipt.into_function()))
}
