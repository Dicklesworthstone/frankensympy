//! Typed bounded bridge for checked first-order characteristic families.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_budget::{Budget, BudgetLimits};
use fsym_core::{BigRational, Constant, Expr};
use fsym_solvers::pde::metered_linear_first_order_pde;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Borrow every original native node before Python recovers its children.
/// This is bounded shape admission, not a solution or affine-phase proof.
#[pyfunction]
pub fn pde_source_preflight_native(source: &PyExpr) -> PyResult<()> {
    let mut pending = vec![(&source.inner, 0usize)];
    let mut visited = 0usize;
    let mut text_bytes = 0usize;
    while let Some((node, depth)) = pending.pop() {
        visited += 1;
        if visited > 4096 || depth > 64 {
            return Err(PyValueError::new_err(
                "PDE source exceeds its node or depth bound",
            ));
        }
        match node {
            Expr::Const(Constant::E) => {}
            Expr::Integer(value) if value.bits() <= 4096 => {}
            Expr::Rational(value)
                if value.numer().bits() <= 4096 && value.denom().bits() <= 4096 => {}
            Expr::Integer(_) | Expr::Rational(_) => {
                return Err(PyValueError::new_err("PDE literal exceeds 4096 bits"));
            }
            Expr::Sym(symbol) => {
                text_bytes = text_bytes.saturating_add(symbol.name.len());
                if text_bytes > 16_384 {
                    return Err(PyValueError::new_err(
                        "PDE coordinate text exceeds its bound",
                    ));
                }
            }
            Expr::Add(children) | Expr::Mul(children) => {
                if children.len() > 256 || visited + pending.len() + children.len() > 4096 {
                    return Err(PyValueError::new_err(
                        "PDE source exceeds its arity or node bound",
                    ));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if visited + pending.len() + 2 > 4096 {
                    return Err(PyValueError::new_err("PDE source exceeds its node bound"));
                }
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Function(name, children)
                if matches!(name.as_str(), "exp" | "cos" | "sin") && children.len() == 1 =>
            {
                text_bytes = text_bytes.saturating_add(name.len());
                if text_bytes > 16_384 {
                    return Err(PyValueError::new_err("PDE source text exceeds its bound"));
                }
                if visited + pending.len() + 1 > 4096 {
                    return Err(PyValueError::new_err("PDE source exceeds its node bound"));
                }
                pending.push((&children[0], depth + 1));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "PDE source requires exact polynomial/elementary expressions",
                ));
            }
        }
    }
    Ok(())
}

/// Retained Python arguments must agree with the borrowed native source.
#[pyfunction]
pub fn pde_source_arguments_native(
    source: &PyExpr,
    operator: &str,
    children: &Bound<'_, PyList>,
) -> PyResult<()> {
    if children.len() > 256 {
        return Err(PyValueError::new_err(
            "PDE source arguments exceed their arity bound",
        ));
    }
    pde_source_preflight_native(source)?;
    let borrowed = children
        .iter()
        .map(|value| value.extract::<PyRef<'_, PyExpr>>().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    let same = match (&source.inner, operator) {
        (Expr::Const(Constant::E), "e") => borrowed.is_empty(),
        (Expr::Add(expected), "add") | (Expr::Mul(expected), "mul") => {
            expected.len() == borrowed.len()
                && expected.iter().zip(&borrowed).all(|(a, b)| a == &b.inner)
        }
        (Expr::Pow(base, exponent), "pow") => {
            borrowed.len() == 2
                && base.as_ref() == &borrowed[0].inner
                && exponent.as_ref() == &borrowed[1].inner
        }
        (Expr::Function(name, expected), "exp" | "cos" | "sin") => {
            name == operator
                && expected.len() == 1
                && borrowed.len() == 1
                && expected[0] == borrowed[0].inner
        }
        _ => false,
    };
    if !same {
        return Err(PyValueError::new_err(
            "PDE source disagrees with its retained arguments",
        ));
    }
    Ok(())
}

fn rational(expression: &Expr) -> PyResult<BigRational> {
    match expression {
        Expr::Integer(value) if value.bits() <= 1_024 => {
            Ok(BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value) if value.numer().bits() <= 1_024 && value.denom().bits() <= 1_024 => {
            Ok(value.clone())
        }
        _ => Err(PyValueError::new_err(
            "PDE coefficients must be exact rational literals of at most 1024 bits",
        )),
    }
}

/// Borrow source arguments before bounded native admission. The receiving
/// solver independently checks the original forcing and characteristic
/// identities before these four expressions can reach the Python shell.
#[pyfunction]
pub fn linear_first_order_pde_native(
    a: &PyExpr,
    b: &PyExpr,
    c: &PyExpr,
    forcing: &PyExpr,
    x: &PySymbolBinding,
    y: &PySymbolBinding,
) -> PyResult<(PyExpr, PyExpr, PyExpr, PyExpr)> {
    if x.name.len().saturating_add(y.name.len()) > 16_384 {
        return Err(PyValueError::new_err(
            "PDE coordinate identifiers exceed the text bound",
        ));
    }
    let coefficients = [
        rational(&a.inner)?,
        rational(&b.inner)?,
        rational(&c.inner)?,
    ];
    let variables = [x.to_symbol(), y.to_symbol()];
    let mut meter = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let checked = metered_linear_first_order_pde(
        &coefficients[0],
        &coefficients[1],
        &coefficients[2],
        &forcing.inner,
        &variables[0],
        &variables[1],
        &mut meter,
    )
    .map_err(to_value_error)?;
    let (time, invariant, particular, multiplier) = checked.into_parts();
    Ok((
        PyExpr::from_expr(time),
        PyExpr::from_expr(invariant),
        PyExpr::from_expr(particular),
        PyExpr::from_expr(multiplier),
    ))
}
