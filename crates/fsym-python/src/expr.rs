//! SymPy-compatible Expr, Basic, Atom surface over native kernel (WS05).

#![forbid(unsafe_code)]

use fsym_calculus::diff;
use fsym_core::{BigInt, BigRational, Expr, Symbol, parse};
use fsym_printing::latex;
use fsym_runtime::{Budget, BudgetLimits, FsymCx, RuntimeBudget};
use fsym_simplify::{expand_with, simplify_with};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

/// Python compatibility wrapper for native FrankenSymPy symbolic expressions.
#[pyclass(name = "Expr", module = "fsym_python", subclass)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyExpr {
    pub inner: Expr,
}

impl PyExpr {
    pub fn from_expr(inner: Expr) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyExpr {
    #[new]
    #[pyo3(signature = (src=None))]
    pub fn new(src: Option<&str>) -> PyResult<Self> {
        match src {
            Some(s) => parse(s)
                .map(Self::from_expr)
                .map_err(|e| PyValueError::new_err(e.to_string())),
            None => Ok(Self::from_expr(Expr::from_i64(0))),
        }
    }

    /// Tuple of direct structural subexpressions.
    #[getter]
    pub fn args(&self) -> Vec<PyExpr> {
        match &self.inner {
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => Vec::new(),
            Expr::Add(terms) | Expr::Mul(terms) => {
                terms.iter().map(|t| PyExpr::from_expr(t.clone())).collect()
            }
            Expr::Pow(base, exp) => vec![
                PyExpr::from_expr((**base).clone()),
                PyExpr::from_expr((**exp).clone()),
            ],
            Expr::Function(_, args) => args.iter().map(|a| PyExpr::from_expr(a.clone())).collect(),
        }
    }

    /// Operation or class identifier string.
    #[getter]
    pub fn func_name(&self) -> String {
        match &self.inner {
            Expr::Sym(_) => "Symbol".into(),
            Expr::Integer(_) => "Integer".into(),
            Expr::Rational(_) => "Rational".into(),
            Expr::Const(_) => "Constant".into(),
            Expr::Add(_) => "Add".into(),
            Expr::Mul(_) => "Mul".into(),
            Expr::Pow(_, _) => "Pow".into(),
            Expr::Function(name, _) => name.clone(),
        }
    }

    #[getter]
    pub fn is_integer(&self) -> bool {
        matches!(&self.inner, Expr::Integer(_))
    }

    #[getter]
    pub fn is_rational(&self) -> bool {
        matches!(&self.inner, Expr::Rational(_))
    }

    #[getter]
    pub fn is_symbol(&self) -> bool {
        matches!(&self.inner, Expr::Sym(_))
    }

    #[getter]
    pub fn is_add(&self) -> bool {
        matches!(&self.inner, Expr::Add(_))
    }

    #[getter]
    pub fn is_mul(&self) -> bool {
        matches!(&self.inner, Expr::Mul(_))
    }

    #[getter]
    pub fn is_pow(&self) -> bool {
        matches!(&self.inner, Expr::Pow(_, _))
    }

    #[getter]
    pub fn is_number(&self) -> bool {
        matches!(
            &self.inner,
            Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_)
        )
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn _repr_latex_(&self) -> String {
        format!("${}$", latex(&self.inner))
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    pub fn __richcmp__(&self, other: &PyExpr, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            CompareOp::Lt => Ok(self.inner < other.inner),
            CompareOp::Le => Ok(self.inner <= other.inner),
            CompareOp::Gt => Ok(self.inner > other.inner),
            CompareOp::Ge => Ok(self.inner >= other.inner),
        }
    }

    pub fn __add__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::from_expr(self.inner.clone() + other.inner.clone())
    }

    pub fn __radd__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::from_expr(other.inner.clone() + self.inner.clone())
    }

    pub fn __sub__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::from_expr(self.inner.clone() + (other.inner.clone() * Expr::from_i64(-1)))
    }

    pub fn __rsub__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::from_expr(other.inner.clone() + (self.inner.clone() * Expr::from_i64(-1)))
    }

    pub fn __mul__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::from_expr(self.inner.clone() * other.inner.clone())
    }

    pub fn __rmul__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::from_expr(other.inner.clone() * self.inner.clone())
    }

    pub fn __neg__(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.clone() * Expr::from_i64(-1))
    }

    pub fn __pow__(&self, other: &PyExpr, _modulo: Option<Py<PyAny>>) -> PyExpr {
        PyExpr::from_expr(Expr::Pow(
            Arc::new(self.inner.clone()),
            Arc::new(other.inner.clone()),
        ))
    }

    /// Substitute sub-expression: `expr.subs(old, new)`.
    pub fn subs(&self, old: &PyExpr, new: &PyExpr) -> PyResult<PyExpr> {
        match &old.inner {
            Expr::Sym(s) => {
                let mut map = HashMap::new();
                map.insert(s.clone(), new.inner.clone());
                Ok(PyExpr::from_expr(self.inner.subs(&map)))
            }
            _ => {
                // Syntactic tree replacement
                if self.inner == old.inner {
                    Ok(new.clone())
                } else {
                    Ok(self.clone())
                }
            }
        }
    }

    /// Exact differentiation ∂expr / ∂var.
    pub fn diff(&self, var: &str) -> PyExpr {
        let sym = Symbol::new(var);
        PyExpr::from_expr(diff(&self.inner, &sym))
    }

    /// Simplify expression under budgeted region.
    pub fn simplify(&self) -> PyResult<PyExpr> {
        let cx = asupersync::Cx::detached_cancel_context();
        let steps = u64::try_from(RuntimeBudget::default().max_eval_steps).expect("valid u64");
        let limits = BudgetLimits::uniform(steps, 0);
        let mut region = FsymCx::new(&cx, Budget::new(limits), limits);

        simplify_with(&self.inner, &mut region)
            .map(PyExpr::from_expr)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Expand expression under budgeted region.
    pub fn expand(&self) -> PyResult<PyExpr> {
        let cx = asupersync::Cx::detached_cancel_context();
        let steps = u64::try_from(RuntimeBudget::default().max_eval_steps).expect("valid u64");
        let limits = BudgetLimits::uniform(steps, 0);
        let mut region = FsymCx::new(&cx, Budget::new(limits), limits);

        expand_with(&self.inner, &mut region)
            .map(PyExpr::from_expr)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Evaluate expression to floating point number.
    pub fn evalf(&self) -> PyResult<f64> {
        self.inner
            .evalf()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Pickle support: `__reduce__` returns `(Expr, (string_representation,))`.
    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let expr_cls = py.get_type::<PyExpr>().into();
        Ok((expr_cls, (self.__str__(),)))
    }

    /// Deepcopy support.
    pub fn __deepcopy__(&self, _memo: Py<PyAny>) -> PyExpr {
        self.clone()
    }
}

/// Construct a Symbol expression.
#[pyfunction]
pub fn py_symbol(name: &str) -> PyExpr {
    PyExpr::from_expr(Expr::symbol(name))
}

/// Construct an Integer expression.
#[pyfunction]
pub fn py_integer(val: i64) -> PyExpr {
    PyExpr::from_expr(Expr::from_i64(val))
}

/// Construct a Rational expression.
#[pyfunction]
pub fn py_rational(p: i64, q: i64) -> PyResult<PyExpr> {
    if q == 0 {
        return Err(PyValueError::new_err("Denominator cannot be zero"));
    }
    let r = BigRational::new(BigInt::from(p), BigInt::from(q));
    Ok(PyExpr::from_expr(Expr::Rational(r)))
}

/// Construct an Add expression.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn py_add(args: Vec<PyExpr>) -> PyExpr {
    let exprs: Vec<Expr> = args.into_iter().map(|a| a.inner).collect();
    PyExpr::from_expr(Expr::Add(exprs))
}

/// Construct a Mul expression.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn py_mul(args: Vec<PyExpr>) -> PyExpr {
    let exprs: Vec<Expr> = args.into_iter().map(|a| a.inner).collect();
    PyExpr::from_expr(Expr::Mul(exprs))
}

/// Construct a Pow expression.
#[pyfunction]
pub fn py_pow(base: PyExpr, exp: PyExpr) -> PyExpr {
    PyExpr::from_expr(Expr::Pow(Arc::new(base.inner), Arc::new(exp.inner)))
}

/// Construct a Derivative expression representation.
#[pyfunction]
pub fn py_derivative(expr: PyExpr, var: PyExpr) -> PyExpr {
    match &var.inner {
        Expr::Sym(s) => PyExpr::from_expr(diff(&expr.inner, s)),
        _ => PyExpr::from_expr(Expr::Function(
            "Derivative".into(),
            vec![expr.inner, var.inner],
        )),
    }
}
