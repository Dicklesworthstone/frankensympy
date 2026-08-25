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
use pyo3::types::PyTuple;
use std::collections::{BTreeSet, HashMap};
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

    /// Tuple of direct structural subexpressions (`args`).
    #[getter]
    pub fn args<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.raw_args())
    }

    /// Direct list of subexpressions for Rust callers.
    pub fn raw_args(&self) -> Vec<PyExpr> {
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

    /// Callable constructor (`func`) for SymPy reconstruction invariant: `expr.func(*expr.args) == expr`.
    #[getter]
    pub fn func<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Expr::Sym(_) => Ok(py.get_type::<PySymbol>().into()),
            Expr::Integer(_) => Ok(py.get_type::<PyInteger>().into()),
            Expr::Rational(_) => Ok(py.get_type::<PyRational>().into()),
            Expr::Add(_) => Ok(py.get_type::<PyAdd>().into()),
            Expr::Mul(_) => Ok(py.get_type::<PyMul>().into()),
            Expr::Pow(_, _) => Ok(py.get_type::<PyPow>().into()),
            Expr::Function(name, _) if name == "Derivative" => {
                Ok(py.get_type::<PyDerivative>().into())
            }
            _ => Ok(py.get_type::<PyExpr>().into()),
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

    /// Set of free symbolic variable names in the expression.
    #[getter]
    pub fn free_symbols(&self) -> Vec<String> {
        let mut symbols = BTreeSet::new();
        collect_free_syms(&self.inner, &mut symbols);
        symbols.into_iter().collect()
    }

    /// Tests if a subexpression or pattern is contained within this expression.
    pub fn has(&self, pattern: &PyExpr) -> bool {
        expr_contains(&self.inner, &pattern.inner)
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

    pub fn _repr_latex_(&self) -> PyResult<String> {
        latex(&self.inner)
            .map(|rendered| format!("${rendered}$"))
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
                if self.inner == old.inner {
                    Ok(new.clone())
                } else {
                    Ok(self.clone())
                }
            }
        }
    }

    /// Exact differentiation ∂expr / ∂var.
    #[pyo3(signature = (var, *more_vars))]
    pub fn diff(&self, var: &str, more_vars: Vec<String>) -> PyExpr {
        let mut res = diff(&self.inner, &Symbol::new(var));
        for v in more_vars {
            res = diff(&res, &Symbol::new(&v));
        }
        PyExpr::from_expr(res)
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

fn collect_free_syms(expr: &Expr, set: &mut BTreeSet<String>) {
    match expr {
        Expr::Sym(s) => {
            set.insert(s.name.clone());
        }
        Expr::Add(terms) | Expr::Mul(terms) => {
            for t in terms {
                collect_free_syms(t, set);
            }
        }
        Expr::Pow(b, e) => {
            collect_free_syms(b, set);
            collect_free_syms(e, set);
        }
        Expr::Function(_, args) => {
            for a in args {
                collect_free_syms(a, set);
            }
        }
        _ => {}
    }
}

fn expr_contains(haystack: &Expr, needle: &Expr) -> bool {
    if haystack == needle {
        return true;
    }
    match haystack {
        Expr::Add(terms) | Expr::Mul(terms) => terms.iter().any(|t| expr_contains(t, needle)),
        Expr::Pow(b, e) => expr_contains(b, needle) || expr_contains(e, needle),
        Expr::Function(_, args) => args.iter().any(|a| expr_contains(a, needle)),
        _ => false,
    }
}

/// Dedicated Symbol class for Python SymPy drop-in compatibility.
#[pyclass(name = "Symbol", module = "fsym_python")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PySymbol {
    pub inner: PyExpr,
}

#[pymethods]
impl PySymbol {
    #[new]
    pub fn new(name: &str) -> Self {
        Self {
            inner: PyExpr::from_expr(Expr::symbol(name)),
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        match &self.inner.inner {
            Expr::Sym(s) => s.name.clone(),
            _ => String::new(),
        }
    }

    pub fn __str__(&self) -> String {
        self.inner.__str__()
    }

    pub fn __repr__(&self) -> String {
        format!("Symbol('{}')", self.name())
    }

    pub fn as_expr(&self) -> PyExpr {
        self.inner.clone()
    }
}

/// Dedicated Integer class for Python SymPy drop-in compatibility.
#[pyclass(name = "Integer", module = "fsym_python")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyInteger {
    pub inner: PyExpr,
}

#[pymethods]
impl PyInteger {
    #[new]
    pub fn new(val: i64) -> Self {
        Self {
            inner: PyExpr::from_expr(Expr::from_i64(val)),
        }
    }

    #[getter]
    pub fn p(&self) -> i64 {
        match &self.inner.inner {
            Expr::Integer(n) => n.to_i64().unwrap_or(0),
            _ => 0,
        }
    }

    #[getter]
    pub fn q(&self) -> i64 {
        1
    }

    pub fn __str__(&self) -> String {
        self.inner.__str__()
    }

    pub fn __repr__(&self) -> String {
        format!("Integer({})", self.p())
    }

    pub fn as_expr(&self) -> PyExpr {
        self.inner.clone()
    }
}

/// Dedicated Rational class for Python SymPy drop-in compatibility.
#[pyclass(name = "Rational", module = "fsym_python")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyRational {
    pub inner: PyExpr,
}

#[pymethods]
impl PyRational {
    #[new]
    pub fn new(p: i64, q: i64) -> PyResult<Self> {
        if q == 0 {
            return Err(PyValueError::new_err("Denominator cannot be zero"));
        }
        let r = BigRational::new(BigInt::from(p), BigInt::from(q));
        Ok(Self {
            inner: PyExpr::from_expr(Expr::Rational(r)),
        })
    }

    #[getter]
    pub fn p(&self) -> i64 {
        match &self.inner.inner {
            Expr::Rational(r) => r.numer().to_i64().unwrap_or(0),
            Expr::Integer(n) => n.to_i64().unwrap_or(0),
            _ => 0,
        }
    }

    #[getter]
    pub fn q(&self) -> i64 {
        match &self.inner.inner {
            Expr::Rational(r) => r.denom().to_i64().unwrap_or(1),
            Expr::Integer(_) => 1,
            _ => 1,
        }
    }

    pub fn __str__(&self) -> String {
        self.inner.__str__()
    }

    pub fn __repr__(&self) -> String {
        format!("Rational({}, {})", self.p(), self.q())
    }

    pub fn as_expr(&self) -> PyExpr {
        self.inner.clone()
    }
}

/// Dedicated Add class for Python SymPy drop-in compatibility with `evaluate=False` support.
#[pyclass(name = "Add", module = "fsym_python")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyAdd {
    pub inner: PyExpr,
}

#[pymethods]
impl PyAdd {
    #[new]
    #[pyo3(signature = (*args, evaluate=true))]
    pub fn new(args: Vec<PyExpr>, evaluate: bool) -> Self {
        let exprs: Vec<Expr> = args.into_iter().map(|a| a.inner).collect();
        let inner = if evaluate {
            let mut terms = exprs.into_iter();
            terms
                .next()
                .map(|first| terms.fold(first, |sum, term| sum + term))
                .unwrap_or_else(|| Expr::from_i64(0))
        } else {
            Expr::Add(exprs)
        };
        Self {
            inner: PyExpr::from_expr(inner),
        }
    }

    pub fn as_expr(&self) -> PyExpr {
        self.inner.clone()
    }
}

/// Dedicated Mul class for Python SymPy drop-in compatibility with `evaluate=False` support.
#[pyclass(name = "Mul", module = "fsym_python")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyMul {
    pub inner: PyExpr,
}

#[pymethods]
impl PyMul {
    #[new]
    #[pyo3(signature = (*args, evaluate=true))]
    pub fn new(args: Vec<PyExpr>, evaluate: bool) -> Self {
        let exprs: Vec<Expr> = args.into_iter().map(|a| a.inner).collect();
        let inner = if evaluate {
            let mut factors = exprs.into_iter();
            factors
                .next()
                .map(|first| factors.fold(first, |product, factor| product * factor))
                .unwrap_or_else(|| Expr::from_i64(1))
        } else {
            Expr::Mul(exprs)
        };
        Self {
            inner: PyExpr::from_expr(inner),
        }
    }

    pub fn as_expr(&self) -> PyExpr {
        self.inner.clone()
    }
}

/// Dedicated Pow class for Python SymPy drop-in compatibility.
#[pyclass(name = "Pow", module = "fsym_python")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyPow {
    pub inner: PyExpr,
}

#[pymethods]
impl PyPow {
    #[new]
    #[pyo3(signature = (base, exp, evaluate=true))]
    pub fn new(base: PyExpr, exp: PyExpr, evaluate: bool) -> Self {
        let _ = evaluate;
        Self {
            inner: PyExpr::from_expr(Expr::Pow(Arc::new(base.inner), Arc::new(exp.inner))),
        }
    }

    pub fn as_expr(&self) -> PyExpr {
        self.inner.clone()
    }
}

/// Dedicated Derivative class for Python SymPy drop-in compatibility.
#[pyclass(name = "Derivative", module = "fsym_python")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyDerivative {
    pub inner: PyExpr,
}

#[pymethods]
impl PyDerivative {
    #[new]
    #[pyo3(signature = (expr, *variables, evaluate=false))]
    pub fn new(expr: PyExpr, variables: Vec<PyExpr>, evaluate: bool) -> Self {
        if evaluate {
            let mut cur = expr.inner;
            for v in variables {
                if let Expr::Sym(s) = v.inner {
                    cur = diff(&cur, &s);
                }
            }
            Self {
                inner: PyExpr::from_expr(cur),
            }
        } else {
            let mut args = vec![expr.inner];
            args.extend(variables.into_iter().map(|v| v.inner));
            Self {
                inner: PyExpr::from_expr(Expr::Function("Derivative".into(), args)),
            }
        }
    }

    pub fn as_expr(&self) -> PyExpr {
        self.inner.clone()
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
