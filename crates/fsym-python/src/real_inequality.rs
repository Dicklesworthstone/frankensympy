//! Typed bridge for complete, bounded rational sign charts over the real line.

use crate::expr::PyExpr;
use fsym_core::Expr;
use fsym_runtime::{Budget, BudgetLimits};
use fsym_solvers::inequality::IneqOp;
use fsym_solvers::{RealConstraint, RealFormula, RealRelation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

type SignChartTransport = (PyExpr, Vec<bool>, Vec<bool>);

/// Bound the borrowed native tree before any Python getter or owned constraint
/// copies it. Unsupported grammar is still traversed: a later oversized source
/// must not disappear behind an unsupported atom or a false conjunction.
fn source_storage(expression: &Expr, meter: &mut Budget) -> PyResult<bool> {
    let mut pending = vec![(expression, 0usize)];
    let mut nodes = 0usize;
    let mut supported = true;
    while let Some((node, depth)) = pending.pop() {
        nodes += 1;
        if nodes > 2048 || depth > 64 {
            return Err(PyValueError::new_err(
                "real constraint resource limit: source nodes/depth",
            ));
        }
        let mut bytes = 256usize;
        match node {
            Expr::Integer(value) => {
                if value.bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "real constraint resource limit: coefficient bits",
                    ));
                }
                bytes += value.bits().div_ceil(8) as usize;
            }
            Expr::Rational(value) => {
                if value.numer().bits().max(value.denom().bits()) > 4096 {
                    return Err(PyValueError::new_err(
                        "real constraint resource limit: coefficient bits",
                    ));
                }
                bytes += (value.numer().bits() + value.denom().bits()).div_ceil(8) as usize;
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > 4096 {
                    return Err(PyValueError::new_err(
                        "real constraint resource limit: symbol bytes",
                    ));
                }
                bytes += symbol.name.len();
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if children.len() > 256 || nodes + pending.len() + children.len() > 2048 {
                    return Err(PyValueError::new_err(
                        "real constraint resource limit: source fanout",
                    ));
                }
                if let Expr::Function(name, _) = node {
                    supported = false;
                    if name.len() > 4096 {
                        return Err(PyValueError::new_err(
                            "real constraint resource limit: function bytes",
                        ));
                    }
                    bytes += name.len();
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                supported &= matches!(exponent.as_ref(), Expr::Integer(_));
                if let Expr::Integer(power) = exponent.as_ref()
                    && (power < &(-64).into() || power > &64.into())
                {
                    return Err(PyValueError::new_err(
                        "real constraint resource limit: integer exponent",
                    ));
                }
                if nodes + pending.len() + 2 > 2048 {
                    return Err(PyValueError::new_err(
                        "real constraint resource limit: source nodes",
                    ));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
            }
            Expr::Const(_) => supported = false,
        }
        crate::ideal_operations::charge(meter, 1, bytes)?;
    }
    Ok(supported)
}

/// Borrowed admission before the Python facade reads retained scalar children.
#[pyfunction]
pub fn real_constraint_source_preflight_native(expression: &PyExpr) -> PyResult<bool> {
    source_storage(
        &expression.inner,
        &mut Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0)),
    )
}

fn variable_storage(variable: &Bound<'_, PyAny>, meter: &mut Budget) -> PyResult<()> {
    let length =
        if let Ok(binding) = variable.extract::<PyRef<'_, crate::binding::PySymbolBinding>>() {
            binding.name.len()
        } else if let Ok(symbol) = variable.extract::<PyRef<'_, crate::expr::PySymbol>>() {
            let Expr::Sym(symbol) = &symbol.inner.inner else {
                return Err(PyValueError::new_err(
                    "real constraints require a typed symbol",
                ));
            };
            symbol.name.len()
        } else if let Ok(expression) = variable.extract::<PyRef<'_, PyExpr>>() {
            let Expr::Sym(symbol) = &expression.inner else {
                return Err(PyValueError::new_err(
                    "real constraints require a typed symbol",
                ));
            };
            symbol.name.len()
        } else if variable.is_exact_instance(&variable.py().get_type::<PyString>()) {
            let name = variable.cast::<PyString>()?;
            if name.len()? > 4096 {
                return Err(PyValueError::new_err(
                    "real constraint resource limit: symbol bytes",
                ));
            }
            name.to_str()?.len()
        } else {
            return Err(PyValueError::new_err(
                "real constraints require a typed symbol",
            ));
        };
    if length > 4096 {
        return Err(PyValueError::new_err(
            "real constraint resource limit: symbol bytes",
        ));
    }
    crate::ideal_operations::charge(meter, 1, length + 128)
}

/// Solve one flat conjunction on a shared, independently checked real partition.
/// A false Boolean input remains in the formula until every atom is admitted.
#[pyfunction]
#[pyo3(signature = (expressions, variable, operations, conjunction_false=false))]
pub fn solve_real_rational_constraints_native(
    expressions: &Bound<'_, PyList>,
    variable: &Bound<'_, PyAny>,
    operations: &Bound<'_, PyList>,
    conjunction_false: bool,
) -> PyResult<Option<SignChartTransport>> {
    if expressions.len() > fsym_solvers::MAX_REAL_CONSTRAINTS
        || expressions.len() != operations.len()
    {
        return Err(PyValueError::new_err(
            "real constraints require at most 32 matched sources and relations",
        ));
    }
    let limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    let mut generator = Budget::new(limits);
    let mut verifier = Budget::new(limits);
    crate::ideal_operations::charge(&mut generator, 1, expressions.len() * 256 + 64)?;
    let expressions = expressions
        .iter()
        .map(|value| value.extract::<PyRef<'_, PyExpr>>().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    for expression in &expressions {
        source_storage(&expression.inner, &mut generator)?;
    }
    let mut relations = Vec::with_capacity(operations.len());
    for operation in operations.iter() {
        if !operation.is_exact_instance(&operation.py().get_type::<PyString>()) {
            return Err(PyValueError::new_err(
                "real constraint relation must be a literal operator string",
            ));
        }
        let operation = operation.cast::<PyString>()?;
        if operation.len()? > 2 {
            return Err(PyValueError::new_err(
                "unsupported real constraint operator",
            ));
        }
        relations.push(match operation.to_str()? {
            "==" => RealRelation::Eq,
            "!=" => RealRelation::Ne,
            "<" => RealRelation::Lt,
            "<=" => RealRelation::Le,
            ">" => RealRelation::Gt,
            ">=" => RealRelation::Ge,
            _ => {
                return Err(PyValueError::new_err(
                    "unsupported real constraint operator",
                ));
            }
        });
    }
    variable_storage(variable, &mut generator)?;
    let symbol = crate::extract_symbol_arg(variable)?;
    let constraints = expressions
        .iter()
        .zip(relations)
        .map(|(expression, relation)| RealConstraint {
            expression: expression.inner.clone(),
            relation,
        })
        .collect::<Vec<_>>();
    let mut atoms = (0..constraints.len())
        .map(RealFormula::Atom)
        .collect::<Vec<_>>();
    if conjunction_false {
        atoms.push(RealFormula::Any(vec![]));
    }
    let result = fsym_solvers::metered_solve_real_constraints(
        &constraints,
        &symbol,
        &RealFormula::All(atoms),
        &mut generator,
        &mut verifier,
    )
    .map_err(crate::to_value_error)?;
    result
        .map(|receipt| {
            crate::ideal_operations::charge(
                &mut generator,
                1,
                65 * (4096 + symbol.name.len() + 128),
            )?;
            Ok((
                PyExpr::from_expr(receipt.polynomial().to_expr()),
                receipt.gap_inclusions().to_vec(),
                receipt.point_inclusions().to_vec(),
            ))
        })
        .transpose()
}

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
