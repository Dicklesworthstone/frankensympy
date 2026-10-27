//! Typed bridge for complete, bounded rational sign charts over the real line.

use crate::expr::PyExpr;
use fsym_runtime::{Budget, BudgetLimits};
use fsym_solvers::inequality::IneqOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type SignChartTransport = (PyExpr, Vec<bool>, Vec<bool>);

/// The checked critical polynomial and inclusion masks in increasing root order.
/// Python lifts the roots through the independently checked real-root association
/// lane; no approximate root or printed expression crosses this boundary.
#[pyfunction]
pub fn solve_real_rational_inequality_native(
    expression: &PyExpr,
    variable: &Bound<'_, PyAny>,
    operation: &str,
) -> PyResult<Option<SignChartTransport>> {
    let symbol = crate::extract_symbol_arg(variable)?;
    let operation = IneqOp::from_str_op(operation)
        .ok_or_else(|| PyValueError::new_err("unsupported real inequality operator"))?;
    let limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    let result = fsym_solvers::metered_solve_real_inequality(
        &expression.inner,
        &symbol,
        operation,
        &mut Budget::new(limits),
        &mut Budget::new(limits),
    )
    .map_err(crate::to_value_error)?;
    Ok(result.map(|receipt| {
        (
            PyExpr::from_expr(receipt.polynomial().to_expr()),
            receipt.gap_inclusions().to_vec(),
            receipt.point_inclusions().to_vec(),
        )
    }))
}
