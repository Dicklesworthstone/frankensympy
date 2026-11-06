//! Structured bridge for the bounded ordinary Fourier integral catalog.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_calculus::fourier::{FourierError, metered_fourier_transform_l1};
use fsym_core::Expr;
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

fn charge_steps(budget: &mut Budget, steps: u64) -> PyResult<()> {
    budget
        .try_charge_batch(&[(Dimension::ComputeSteps, steps.max(1))])
        .map_err(to_value_error)?;
    Ok(())
}

/// Compare borrowed immutable payloads, with no recursively unbounded Eq call.
fn retained_equal(left: &Expr, right: &Expr, budget: &mut Budget) -> PyResult<bool> {
    budget
        .try_charge_batch(&[
            (Dimension::MemoryBytes, 4096 * 32),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(to_value_error)?;
    let mut pending = Vec::with_capacity(4096);
    pending.push((left, right, 0usize));
    let mut visited = 0usize;
    while let Some((left, right, depth)) = pending.pop() {
        visited += 1;
        if visited > 4096 || depth > 64 {
            return Err(PyValueError::new_err(
                "Fourier retained source exceeds its bounded envelope",
            ));
        }
        charge_steps(budget, 1)?;
        let equal = match (left, right) {
            (Expr::Integer(a), Expr::Integer(b)) => {
                if a.bits() > 4096 || b.bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "Fourier retained coefficient exceeds 4096 bits",
                    ));
                }
                charge_steps(budget, a.bits().max(b.bits()).div_ceil(64))?;
                a == b
            }
            (Expr::Rational(a), Expr::Rational(b)) => {
                let values = [a.numer(), a.denom(), b.numer(), b.denom()];
                if values.iter().any(|value| value.bits() > 4096) {
                    return Err(PyValueError::new_err(
                        "Fourier retained coefficient exceeds 4096 bits",
                    ));
                }
                let limbs = values.iter().map(|value| value.bits().div_ceil(64)).sum();
                charge_steps(budget, limbs)?;
                a == b
            }
            (Expr::Sym(a), Expr::Sym(b)) => {
                if a.name.len() > 65_536 || b.name.len() > 65_536 {
                    return Err(PyValueError::new_err(
                        "Fourier retained symbol name exceeds its bound",
                    ));
                }
                charge_steps(budget, a.name.len().max(b.name.len()).div_ceil(64) as u64)?;
                a == b
            }
            (Expr::Const(a), Expr::Const(b)) => a == b,
            (Expr::Add(a), Expr::Add(b))
            | (Expr::Mul(a), Expr::Mul(b))
            | (Expr::Function(_, a), Expr::Function(_, b)) => {
                if let (Expr::Function(left_name, _), Expr::Function(right_name, _)) = (left, right)
                {
                    if left_name.len() > 65_536 || right_name.len() > 65_536 {
                        return Err(PyValueError::new_err(
                            "Fourier retained function name exceeds its bound",
                        ));
                    }
                    charge_steps(
                        budget,
                        left_name.len().max(right_name.len()).div_ceil(64) as u64,
                    )?;
                    if left_name != right_name {
                        return Ok(false);
                    }
                }
                if a.len() != b.len() {
                    return Ok(false);
                }
                if a.len() > 4096usize.saturating_sub(visited + pending.len()) {
                    return Err(PyValueError::new_err(
                        "Fourier retained source exceeds its node bound",
                    ));
                }
                pending.extend(a.iter().zip(b).map(|(a, b)| (a, b, depth + 1)));
                true
            }
            (Expr::Pow(a, b), Expr::Pow(c, d)) => {
                if 2 > 4096usize.saturating_sub(visited + pending.len()) {
                    return Err(PyValueError::new_err(
                        "Fourier retained source exceeds its node bound",
                    ));
                }
                pending.push((a.as_ref(), c.as_ref(), depth + 1));
                pending.push((b.as_ref(), d.as_ref(), depth + 1));
                true
            }
            _ => false,
        };
        if !equal {
            return Ok(false);
        }
    }
    Ok(true)
}

fn check_retained(records: &Bound<'_, PyList>, budget: &mut Budget) -> PyResult<()> {
    if records.len() > 4096 {
        return Err(PyValueError::new_err(
            "Fourier retained source exceeds its node bound",
        ));
    }
    for record in records.iter() {
        charge_steps(budget, 1)?;
        let record = record.cast::<PyTuple>()?;
        if record.len() != 3 {
            return Err(PyValueError::new_err("invalid Fourier source record"));
        }
        let source_item = record.get_item(0)?;
        let source = source_item.extract::<PyRef<'_, PyExpr>>()?;
        let operator_item = record.get_item(1)?;
        // Bound exact Python characters before requesting a UTF-8 view. Even
        // an uncached Unicode representation can then allocate at most 256 B.
        if !operator_item.is_exact_instance(&operator_item.py().get_type::<PyString>())
            || operator_item.len()? > 64
        {
            return Err(PyValueError::new_err(
                "Fourier source operator requires a bounded exact string",
            ));
        }
        let operator = operator_item.cast::<PyString>()?.to_str()?;
        if operator.len() > 64 {
            return Err(PyValueError::new_err(
                "Fourier source operator exceeds its byte bound",
            ));
        }
        let children_item = record.get_item(2)?;
        let children = if children_item.is_none() {
            None
        } else {
            let children = children_item.cast::<PyTuple>()?;
            if children.len() > 4096 {
                return Err(PyValueError::new_err("Fourier retained child count bound"));
            }
            Some(children)
        };
        // Inspect arity before constructing any child-reference container.
        // All subsequent traversal borrows the original native slice.
        let expected_count = match (&source.inner, operator) {
            (Expr::Integer(_), "Integer") => 0,
            (Expr::Integer(_) | Expr::Rational(_), "Rational") => 0,
            (Expr::Add(arguments), "Add") | (Expr::Mul(arguments), "Mul") => arguments.len(),
            (Expr::Pow(_, _), "Pow") => 2,
            (Expr::Function(name, arguments), operator)
                if name == operator
                    || (name == "abs" && operator == "Abs")
                    || (name == "ln" && operator == "log") =>
            {
                arguments.len()
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Fourier retained operator differs from its native source",
                ));
            }
        };
        if expected_count > 4096
            || children.is_some_and(|children| expected_count != children.len())
        {
            return Err(PyValueError::new_err(
                "Fourier retained arguments differ from their native source",
            ));
        }
        let Some(children) = children else {
            continue;
        };
        for (index, child) in children.iter().enumerate() {
            let expected = match &source.inner {
                Expr::Add(arguments) | Expr::Mul(arguments) | Expr::Function(_, arguments) => {
                    &arguments[index]
                }
                Expr::Pow(base, _) if index == 0 => base.as_ref(),
                Expr::Pow(_, exponent) => exponent.as_ref(),
                _ => unreachable!("operator shape checked above"),
            };
            let child = child.extract::<PyRef<'_, PyExpr>>()?;
            if !retained_equal(expected, &child.inner, budget)? {
                return Err(PyValueError::new_err(
                    "Fourier retained arguments differ from their native source",
                ));
            }
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (source, variable, frequency, retained=None))]
pub fn fourier_l1_native(
    source: &PyExpr,
    variable: &PySymbolBinding,
    frequency: &PySymbolBinding,
    retained: Option<&Bound<'_, PyList>>,
) -> PyResult<(PyExpr, bool)> {
    let names = variable.name.len().saturating_add(frequency.name.len());
    if names > 65_536 {
        return Err(PyValueError::new_err(
            "Fourier variable names exceed their bound",
        ));
    }
    let mut limits = BudgetLimits::uniform(128 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 20_000_000;
    let mut budget = Budget::new(limits);
    budget
        .try_charge_batch(&[
            (Dimension::ComputeSteps, names.max(1) as u64),
            (Dimension::MemoryBytes, (names + 256) as u64),
            (Dimension::AllocationCount, 2),
        ])
        .map_err(to_value_error)?;
    let variable = variable.to_symbol();
    let frequency = frequency.to_symbol();
    if let Some(retained) = retained {
        check_retained(retained, &mut budget)?;
    }
    let value = metered_fourier_transform_l1(&source.inner, &variable, &frequency, &mut budget)
        .map_err(|error| match error {
            FourierError::Unsupported(reason) => PyNotImplementedError::new_err(reason),
            error => to_value_error(error),
        })?;
    Ok((PyExpr { inner: value.value }, value.used_bilateral_identity))
}
