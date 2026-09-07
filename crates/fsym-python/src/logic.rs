//! Logic and Boolean algebra Python bindings.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fsym_logic::{BoolExpr, Cnf, Literal, dpll_satisfiable, is_satisfiable, simplify_logic};

#[pyclass(name = "BoolExpr", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyBoolExpr {
    pub inner: BoolExpr,
}

pub fn cnf_to_bool_expr(cnf: &Cnf) -> BoolExpr {
    if cnf.is_empty() {
        return BoolExpr::Const(true);
    }
    if cnf.len() == 1 && cnf.first().is_some_and(|c| c.is_empty()) {
        return BoolExpr::Const(false);
    }
    let clauses: Vec<BoolExpr> = cnf
        .iter()
        .map(|clause| {
            if clause.is_empty() {
                BoolExpr::Const(false)
            } else if clause.len() == 1
                && let Some(lit) = clause.first()
            {
                literal_to_bool_expr(lit)
            } else {
                BoolExpr::Or(clause.iter().map(literal_to_bool_expr).collect())
            }
        })
        .collect();
    if clauses.len() == 1 {
        clauses.into_iter().next().unwrap_or(BoolExpr::Const(true))
    } else {
        BoolExpr::And(clauses)
    }
}

pub fn dnf_to_bool_expr(dnf: &Cnf) -> BoolExpr {
    if dnf.is_empty() {
        return BoolExpr::Const(false);
    }
    let terms: Vec<BoolExpr> = dnf
        .iter()
        .map(|term| {
            if term.is_empty() {
                BoolExpr::Const(true)
            } else if term.len() == 1
                && let Some(lit) = term.first()
            {
                literal_to_bool_expr(lit)
            } else {
                BoolExpr::And(term.iter().map(literal_to_bool_expr).collect())
            }
        })
        .collect();
    if terms.len() == 1 {
        terms.into_iter().next().unwrap_or(BoolExpr::Const(false))
    } else {
        BoolExpr::Or(terms)
    }
}

fn literal_to_bool_expr(lit: &Literal) -> BoolExpr {
    match lit {
        Literal::Pos(sym) => BoolExpr::Var(sym.clone()),
        Literal::Neg(sym) => BoolExpr::Not(Box::new(BoolExpr::Var(sym.clone()))),
    }
}

#[pymethods]
impl PyBoolExpr {
    #[staticmethod]
    pub fn bool_const(val: bool) -> Self {
        Self {
            inner: BoolExpr::Const(val),
        }
    }

    #[staticmethod]
    pub fn bool_var(name: &str) -> Self {
        Self {
            inner: BoolExpr::var(name),
        }
    }

    #[staticmethod]
    pub fn bool_not(arg: &PyBoolExpr) -> Self {
        Self {
            inner: BoolExpr::Not(Box::new(arg.inner.clone())),
        }
    }

    #[staticmethod]
    pub fn bool_and(args: Vec<PyBoolExpr>) -> Self {
        Self {
            inner: BoolExpr::And(args.into_iter().map(|a| a.inner).collect()),
        }
    }

    #[staticmethod]
    pub fn bool_or(args: Vec<PyBoolExpr>) -> Self {
        Self {
            inner: BoolExpr::Or(args.into_iter().map(|a| a.inner).collect()),
        }
    }

    #[staticmethod]
    pub fn bool_implies(a: &PyBoolExpr, b: &PyBoolExpr) -> Self {
        Self {
            inner: BoolExpr::Implies(Box::new(a.inner.clone()), Box::new(b.inner.clone())),
        }
    }

    #[staticmethod]
    pub fn bool_equivalent(a: &PyBoolExpr, b: &PyBoolExpr) -> Self {
        Self {
            inner: BoolExpr::Equivalent(Box::new(a.inner.clone()), Box::new(b.inner.clone())),
        }
    }

    #[staticmethod]
    pub fn bool_xor(a: &PyBoolExpr, b: &PyBoolExpr) -> Self {
        Self {
            inner: a.inner.clone().xor(b.inner.clone()),
        }
    }

    pub fn simplify(&self) -> Self {
        Self {
            inner: simplify_logic(&self.inner),
        }
    }

    pub fn to_cnf(&self) -> PyResult<Self> {
        let cnf = self
            .inner
            .to_cnf()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: cnf_to_bool_expr(&cnf),
        })
    }

    pub fn to_dnf(&self) -> PyResult<Self> {
        let dnf = self
            .inner
            .to_dnf()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: dnf_to_bool_expr(&dnf),
        })
    }

    pub fn is_satisfiable(&self) -> PyResult<bool> {
        is_satisfiable(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn satisfiable(&self) -> PyResult<Option<HashMap<String, bool>>> {
        let res =
            dpll_satisfiable(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(res.map(|map| map.into_iter().map(|(k, v)| (k.name, v)).collect()))
    }

    pub fn kind(&self) -> &'static str {
        match &self.inner {
            BoolExpr::Const(_) => "Const",
            BoolExpr::Var(_) => "Var",
            BoolExpr::Not(_) => "Not",
            BoolExpr::And(_) => "And",
            BoolExpr::Or(_) => "Or",
            BoolExpr::Implies(_, _) => "Implies",
            BoolExpr::Equivalent(_, _) => "Equivalent",
        }
    }

    pub fn var_name(&self) -> Option<String> {
        match &self.inner {
            BoolExpr::Var(s) => Some(s.name.clone()),
            _ => None,
        }
    }

    pub fn const_value(&self) -> Option<bool> {
        match &self.inner {
            BoolExpr::Const(b) => Some(*b),
            _ => None,
        }
    }

    pub fn args(&self) -> Vec<PyBoolExpr> {
        match &self.inner {
            BoolExpr::Const(_) | BoolExpr::Var(_) => vec![],
            BoolExpr::Not(inner) => vec![PyBoolExpr {
                inner: *inner.clone(),
            }],
            BoolExpr::And(terms) | BoolExpr::Or(terms) => terms
                .iter()
                .map(|t| PyBoolExpr { inner: t.clone() })
                .collect(),
            BoolExpr::Implies(a, b) | BoolExpr::Equivalent(a, b) => vec![
                PyBoolExpr { inner: *a.clone() },
                PyBoolExpr { inner: *b.clone() },
            ],
        }
    }

    fn __eq__(&self, other: &PyBoolExpr) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}
