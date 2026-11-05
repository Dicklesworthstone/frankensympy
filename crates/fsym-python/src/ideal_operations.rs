//! Bounded QQ ideal operations with one cumulative pair of native budgets.
//!
//! Saturation returns an ideal (the algebraic closure after localization),
//! not a solution set with the saturating polynomial's zeros removed.

use crate::binding::PySymbolBinding;
use crate::expr::{PyExpr, PySymbol};
use crate::{extract_symbol_arg, to_value_error};
use fsym_core::{Expr, Symbol};
use fsym_polys::{MultivariatePoly, VerifiedIdealOperation, bounded_polynomial_from_expr};
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

fn budget() -> Budget {
    let mut limits = BudgetLimits::uniform(128 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 20_000_000;
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

/// Read names while their original Python/native owners remain borrowed. In
/// particular, do not call `to_symbol` before the whole ring has been bounded.
fn symbol_name_bytes(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(binding) = value.extract::<PyRef<'_, PySymbolBinding>>() {
        return Ok(binding.name.len());
    }
    if let Ok(symbol) = value.extract::<PyRef<'_, PySymbol>>()
        && let Expr::Sym(symbol) = &symbol.inner.inner
    {
        return Ok(symbol.name.len());
    }
    if let Ok(expression) = value.extract::<PyRef<'_, PyExpr>>()
        && let Expr::Sym(symbol) = &expression.inner
    {
        return Ok(symbol.name.len());
    }
    if value.is_exact_instance(&value.py().get_type::<PyString>()) {
        let name = value.cast::<PyString>()?;
        // Bound code points before Python may materialize a UTF-8 cache, then
        // use bytes for the native ring's actual storage bound.
        if name.len()? > 65_536 {
            return Err(PyValueError::new_err(
                "ideal generator names exceed 65536 bytes",
            ));
        }
        return Ok(value.extract::<&str>()?.len());
    }
    Err(PyValueError::new_err(
        "expected a SymbolBinding, native Symbol/Expr(Sym), or symbol name",
    ))
}

fn symbols(values: &Bound<'_, PyList>, meter: &mut Budget) -> PyResult<Vec<Symbol>> {
    if values.len() > 8 {
        return Err(PyValueError::new_err(
            "checked ideals support at most eight generators",
        ));
    }
    let mut name_bytes = 0usize;
    for value in values.iter() {
        name_bytes = name_bytes
            .checked_add(symbol_name_bytes(&value)?)
            .filter(|&bytes| bytes <= 65_536)
            .ok_or_else(|| PyValueError::new_err("ideal generator names exceed 65536 bytes"))?;
    }
    charge(meter, values.len(), values.len() * 128 + name_bytes)?;
    let symbols = values
        .iter()
        .map(|value| extract_symbol_arg(&value))
        .collect::<PyResult<Vec<_>>>()?;
    // Bind the complete typed ring, including duplicate and name-size checks,
    // before admitting an empty list of polynomial generators.
    bounded_polynomial_from_expr(&Expr::from_i64(0), &symbols, meter).map_err(to_value_error)?;
    Ok(symbols)
}

/// Borrowed shell admission before inspecting Python `.args` can copy native
/// children. This performs no normalization or polynomial construction; the
/// metered operation still admits every original source in its exact ring.
#[pyfunction]
pub fn ideal_source_preflight_native(source: &PyExpr) -> PyResult<()> {
    let mut pending = vec![(&source.inner, 0usize)];
    let mut visited = 0usize;
    while let Some((node, depth)) = pending.pop() {
        visited += 1;
        if visited > 4096 || depth > 64 {
            return Err(PyValueError::new_err(
                "ideal source exceeds its bounded expression envelope",
            ));
        }
        match node {
            Expr::Integer(value) if value.bits() <= 4096 => {}
            Expr::Rational(value)
                if value.numer().bits() <= 4096 && value.denom().bits() <= 4096 => {}
            Expr::Integer(_) | Expr::Rational(_) => {
                return Err(PyValueError::new_err("ideal coefficient exceeds 4096 bits"));
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > 65_536 {
                    return Err(PyValueError::new_err(
                        "ideal generator names exceed 65536 bytes",
                    ));
                }
            }
            Expr::Add(children) | Expr::Mul(children) => {
                if children.len() > 4096 - visited - pending.len() {
                    return Err(PyValueError::new_err(
                        "ideal source exceeds its bounded expression envelope",
                    ));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if 2 > 4096 - visited - pending.len() {
                    return Err(PyValueError::new_err(
                        "ideal source exceeds its bounded expression envelope",
                    ));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "ideal operations require exact rational polynomial expressions",
                ));
            }
        }
    }
    Ok(())
}

fn retained_source_matches<'a>(
    source: &Expr,
    operator: &str,
    mut children: impl ExactSizeIterator<Item = &'a Expr>,
) -> bool {
    match (source, operator) {
        (Expr::Add(expected), "add") | (Expr::Mul(expected), "mul") => {
            expected.len() == children.len() && expected.iter().zip(children).all(|(a, b)| a == b)
        }
        (Expr::Pow(base, exponent), "pow") => {
            children.len() == 2
                && children.next() == Some(base.as_ref())
                && children.next() == Some(exponent.as_ref())
        }
        _ => false,
    }
}

/// Check retained arithmetic arguments against their immutable native source.
/// PyRefs borrow the children; no expression tree is cloned or normalized.
#[pyfunction]
pub fn ideal_source_arguments_native(
    source: &PyExpr,
    operator: &str,
    children: &Bound<'_, PyList>,
) -> PyResult<()> {
    if children.len() > 4096 {
        return Err(PyValueError::new_err(
            "ideal source exceeds its bounded expression envelope",
        ));
    }
    ideal_source_preflight_native(source)?;
    let children = children
        .iter()
        .map(|value| value.extract::<PyRef<'_, PyExpr>>())
        .collect::<PyResult<Vec<_>>>()?;
    if !retained_source_matches(
        &source.inner,
        operator,
        children.iter().map(|item| &item.inner),
    ) {
        return Err(PyValueError::new_err(
            "ideal source does not match its retained surface arguments",
        ));
    }
    Ok(())
}

fn polynomials(
    values: &Bound<'_, PyList>,
    variables: &[Symbol],
    meter: &mut Budget,
) -> PyResult<Vec<MultivariatePoly>> {
    if values.len() > 64 {
        return Err(PyValueError::new_err(
            "checked ideals support at most 64 input polynomials",
        ));
    }
    charge(meter, values.len(), values.len() * 128)?;
    values
        .iter()
        .map(|value| {
            let expression = value.extract::<PyRef<'_, PyExpr>>()?;
            bounded_polynomial_from_expr(&expression.inner, variables, meter)
                .map_err(to_value_error)
        })
        .collect()
}

fn lift(
    result: VerifiedIdealOperation,
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<Vec<PyExpr>> {
    let ring_bytes: usize = result
        .generators()
        .iter()
        .map(|symbol| symbol.name.len() + 128)
        .sum();
    lift_basis(result.into_basis(), ring_bytes, generator, verifier)
}

fn lift_basis(
    basis: Vec<MultivariatePoly>,
    ring_bytes: usize,
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<Vec<PyExpr>> {
    let mut total = 0usize;
    for polynomial in &basis {
        let nodes = 1 + polynomial
            .terms()
            .keys()
            .map(|powers| 2 + 4 * powers.iter().filter(|&&power| power != 0).count())
            .sum::<usize>();
        total = total
            .checked_add(nodes)
            .ok_or_else(|| PyValueError::new_err("ideal output size overflow"))?;
        if nodes > 4096 || total > 65_536 {
            return Err(PyValueError::new_err(
                "ideal output exceeds its bounded expression envelope",
            ));
        }
    }
    let bytes = total
        .checked_mul(2048 + ring_bytes)
        .ok_or_else(|| PyValueError::new_err("ideal output allocation overflow"))?;
    charge(generator, total, bytes)?;
    let values = basis
        .into_iter()
        .map(|polynomial| {
            polynomial
                .to_expr()
                .map(PyExpr::from_expr)
                .map_err(to_value_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    charge(generator, 1, 1)?;
    charge(verifier, 1, 1)?;
    Ok(values)
}

/// Complete elimination ideal in the retained variables' original order.
#[pyfunction]
pub fn ideal_eliminate_native(
    equations: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    eliminated: &Bound<'_, PyList>,
) -> PyResult<Vec<PyExpr>> {
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    let eliminated = symbols(eliminated, &mut generator)?;
    let inputs = polynomials(equations, &variables, &mut generator)?;
    let result = fsym_polys::checked_elimination_ideal(
        &inputs,
        &variables,
        &eliminated,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    lift(result, &mut generator, &mut verifier)
}

/// Intersection of the complete ideals generated by both input lists.
#[pyfunction]
pub fn ideal_intersection_native(
    left: &Bound<'_, PyList>,
    right: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
) -> PyResult<Vec<PyExpr>> {
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    let left = polynomials(left, &variables, &mut generator)?;
    let right = polynomials(right, &variables, &mut generator)?;
    let result = fsym_polys::checked_ideal_intersection(
        &left,
        &right,
        &variables,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    lift(result, &mut generator, &mut verifier)
}

/// Finite ideal quotient I:J, requiring every divisor product to belong to I.
#[pyfunction]
pub fn ideal_quotient_native(
    equations: &Bound<'_, PyList>,
    divisors: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
) -> PyResult<Vec<PyExpr>> {
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    // Admit both original lists before zero/unit shortcuts or generation.
    let inputs = polynomials(equations, &variables, &mut generator)?;
    let divisors = polynomials(divisors, &variables, &mut generator)?;
    let result = fsym_polys::checked_ideal_quotient(
        &inputs,
        &divisors,
        &variables,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    let ring_bytes = result
        .generators()
        .iter()
        .map(|symbol| symbol.name.len() + 128)
        .sum();
    lift_basis(
        result.into_basis(),
        ring_bytes,
        &mut generator,
        &mut verifier,
    )
}

/// Ideal saturation I:f^infinity, retaining full original source admission.
#[pyfunction]
pub fn ideal_saturation_native(
    equations: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    factor: &PyExpr,
) -> PyResult<Vec<PyExpr>> {
    let mut generator = budget();
    let mut verifier = budget();
    let variables = symbols(variables, &mut generator)?;
    let inputs = polynomials(equations, &variables, &mut generator)?;
    let factor = bounded_polynomial_from_expr(&factor.inner, &variables, &mut generator)
        .map_err(to_value_error)?;
    let result = fsym_polys::checked_ideal_saturation(
        &inputs,
        &factor,
        &variables,
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    lift(result, &mut generator, &mut verifier)
}

#[cfg(test)]
mod tests {
    use super::retained_source_matches;
    use fsym_core::{Expr, Symbol};
    use std::sync::Arc;

    #[test]
    fn held_identity_and_duplicate_children_remain_accepted() {
        let x = Expr::Sym(Symbol::new("x"));
        for children in [
            vec![x.clone(), Expr::from_i64(0)],
            vec![x.clone(), x.clone()],
        ] {
            assert!(retained_source_matches(
                &Expr::Add(children.clone()),
                "add",
                children.iter(),
            ));
        }
        let children = [Expr::from_i64(1), x.clone()];
        assert!(retained_source_matches(
            &Expr::Mul(children.to_vec()),
            "mul",
            children.iter(),
        ));
        for exponent in [0, 1] {
            let children = [x.clone(), Expr::from_i64(exponent)];
            assert!(retained_source_matches(
                &Expr::Pow(Arc::new(children[0].clone()), Arc::new(children[1].clone())),
                "pow",
                children.iter(),
            ));
        }
    }

    #[test]
    fn altered_native_operator_arity_and_nested_arguments_are_rejected() {
        let x = Expr::Sym(Symbol::new("x"));
        let y = Expr::Sym(Symbol::new("y"));
        let held = Expr::Add(vec![x.clone(), y.clone()]);
        let children = [x.clone(), y.clone()];
        assert!(!retained_source_matches(&held, "mul", children.iter()));
        assert!(!retained_source_matches(&held, "add", children[..1].iter()));
        assert!(!retained_source_matches(
            &held,
            "add",
            [x.clone(), x.clone()].iter(),
        ));
        let nested = Expr::Mul(vec![held.clone(), Expr::from_i64(1)]);
        assert!(retained_source_matches(
            &nested,
            "mul",
            [held, Expr::from_i64(1)].iter(),
        ));
        assert!(!retained_source_matches(
            &nested,
            "mul",
            [Expr::Add(vec![x, Expr::from_i64(0)]), Expr::from_i64(1)].iter(),
        ));
    }
}
