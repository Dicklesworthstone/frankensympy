//! Typed bridge for independently checked forward gamma/beta Mellin kernels.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::fourier::check_retained;
use crate::to_value_error;
use fsym_calculus::forward_mellin::{ForwardMellinError, metered_mellin_transform};
use fsym_calculus::rational_mellin::MellinStripBound;
use fsym_core::elementary::rational_expr;
use fsym_core::{Constant, Expr};
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

fn budget() -> Budget {
    let mut limits = BudgetLimits::uniform(256 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 40_000_000;
    Budget::new(limits)
}

fn bound(value: MellinStripBound) -> PyExpr {
    PyExpr::from_expr(match value {
        MellinStripBound::NegativeInfinity => Expr::Const(Constant::NegativeInfinity),
        MellinStripBound::PositiveInfinity => Expr::Const(Constant::Infinity),
        MellinStripBound::Finite(value) => rational_expr(value),
    })
}

#[pyfunction]
#[pyo3(signature = (source, variable, transform_variable, retained=None))]
pub fn mellin_transform_native(
    source: &PyExpr,
    variable: &PySymbolBinding,
    transform_variable: &PySymbolBinding,
    retained: Option<&Bound<'_, PyList>>,
) -> PyResult<(PyExpr, PyExpr, PyExpr)> {
    let names = variable
        .name
        .len()
        .saturating_add(transform_variable.name.len());
    if names > 65_536 {
        return Err(PyValueError::new_err("Mellin variable name bound"));
    }
    let mut generator = budget();
    let mut verifier = budget();
    generator
        .try_charge_batch(&[
            (Dimension::ComputeSteps, names.max(1) as u64),
            (Dimension::MemoryBytes, (names + 256) as u64),
            (Dimension::AllocationCount, 2),
        ])
        .map_err(to_value_error)?;
    let variable = variable.to_symbol();
    let transform_variable = transform_variable.to_symbol();
    if let Some(retained) = retained {
        check_retained(retained, &mut generator)?;
    }
    let verified = metered_mellin_transform(
        &source.inner,
        &variable,
        &transform_variable,
        &mut generator,
        &mut verifier,
    )
    .map_err(|error| match error {
        ForwardMellinError::Unsupported(reason) => PyNotImplementedError::new_err(reason),
        error => to_value_error(error),
    })?;
    // Both independent phases must still admit publication.
    for meter in [&mut generator, &mut verifier] {
        meter
            .try_charge_batch(&[
                (Dimension::ComputeSteps, 1),
                (Dimension::MemoryBytes, 3 * 128),
                (Dimension::AllocationCount, 3),
            ])
            .map_err(to_value_error)?;
    }
    let (transform, strip) = verified.into_parts();
    Ok((
        PyExpr::from_expr(transform),
        bound(strip.lower),
        bound(strip.upper),
    ))
}
