//! Assumptions and predicate query Python bindings (WS04).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fsym_assumptions::{AssumptionsContext, Domain, Predicate, TruthValue, inherent_facts};
use fsym_core::Symbol;

use crate::expr::PyExpr;

pub fn parse_predicate(name: &str) -> PyResult<Predicate> {
    match name.to_ascii_lowercase().replace('_', "").as_str() {
        "complex" => Ok(Predicate::Complex),
        "real" => Ok(Predicate::Real),
        "rational" => Ok(Predicate::Rational),
        "integer" => Ok(Predicate::Integer),
        "algebraic" => Ok(Predicate::Algebraic),
        "transcendental" => Ok(Predicate::Transcendental),
        "positive" => Ok(Predicate::Positive),
        "negative" => Ok(Predicate::Negative),
        "nonnegative" => Ok(Predicate::NonNegative),
        "nonpositive" => Ok(Predicate::NonPositive),
        "zero" => Ok(Predicate::Zero),
        "nonzero" => Ok(Predicate::NonZero),
        "prime" => Ok(Predicate::Prime),
        "even" => Ok(Predicate::Even),
        "odd" => Ok(Predicate::Odd),
        "finite" => Ok(Predicate::Finite),
        "infinite" => Ok(Predicate::Infinite),
        _ => Err(PyValueError::new_err(format!(
            "Unknown predicate: {}",
            name
        ))),
    }
}

pub fn predicate_name(pred: Predicate) -> &'static str {
    match pred {
        Predicate::Complex => "complex",
        Predicate::Real => "real",
        Predicate::Rational => "rational",
        Predicate::Integer => "integer",
        Predicate::Algebraic => "algebraic",
        Predicate::Transcendental => "transcendental",
        Predicate::Positive => "positive",
        Predicate::Negative => "negative",
        Predicate::NonNegative => "nonnegative",
        Predicate::NonPositive => "nonpositive",
        Predicate::Zero => "zero",
        Predicate::NonZero => "nonzero",
        Predicate::Prime => "prime",
        Predicate::Even => "even",
        Predicate::Odd => "odd",
        Predicate::Finite => "finite",
        Predicate::Infinite => "infinite",
    }
}

#[pyclass(name = "AssumptionsContext", module = "fsym_python")]
#[derive(Clone, Debug, Default)]
pub struct PyAssumptionsContext {
    pub inner: AssumptionsContext,
}

#[pymethods]
impl PyAssumptionsContext {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: AssumptionsContext::new(),
        }
    }

    pub fn assume(&mut self, sym_name: &str, pred_name: &str) -> PyResult<()> {
        let pred = parse_predicate(pred_name)?;
        self.inner
            .assume(Symbol::new(sym_name), pred)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn assume_domain(&mut self, sym_name: &str, domain_name: &str) -> PyResult<()> {
        let dom = match domain_name.to_ascii_uppercase().as_str() {
            "ZZ" | "Z" | "INTEGER" => Domain::ZZ,
            "QQ" | "Q" | "RATIONAL" => Domain::QQ,
            "RR" | "R" | "REAL" => Domain::RR,
            "CC" | "C" | "COMPLEX" => Domain::CC,
            "EX" | "EXPR" => Domain::ExpressionDomain,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown domain: {}",
                    domain_name
                )));
            }
        };
        self.inner
            .assume_domain(Symbol::new(sym_name), dom)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn deductions(&self, sym_name: &str) -> Vec<String> {
        self.inner
            .deductions(&Symbol::new(sym_name))
            .into_iter()
            .map(predicate_name)
            .map(String::from)
            .collect()
    }

    pub fn is_true(&self, expr: &PyExpr, pred_name: &str) -> PyResult<Option<bool>> {
        let pred = parse_predicate(pred_name)?;
        Ok(self.inner.is_true(&expr.inner, pred))
    }

    pub fn query(&self, expr: &PyExpr, pred_name: &str) -> PyResult<String> {
        let pred = parse_predicate(pred_name)?;
        let tv = self.inner.query(&expr.inner, pred);
        let s = match tv {
            TruthValue::EntailedTrue => "True",
            TruthValue::EntailedFalse => "False",
            TruthValue::Unknown => "Unknown",
            TruthValue::Contradictory => "Contradictory",
        };
        Ok(s.to_string())
    }
}

/// Ask if predicate holds for expression given optional symbol assumption facts.
#[pyfunction]
#[pyo3(signature = (expr, pred_name, facts=None))]
pub fn ask_expr(
    expr: &PyExpr,
    pred_name: &str,
    facts: Option<Vec<(String, String)>>,
) -> PyResult<Option<bool>> {
    let pred = parse_predicate(pred_name)?;
    let mut ctx = AssumptionsContext::new();
    if let Some(fact_list) = facts {
        for (sym_name, pred_str) in fact_list {
            let p = parse_predicate(&pred_str)?;
            ctx.assume(Symbol::new(sym_name), p)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    Ok(ctx.is_true(&expr.inner, pred))
}

/// Deduce inherent facts known from an exact expression (e.g. constant, number).
#[pyfunction]
pub fn inherent_facts_expr(expr: &PyExpr) -> Vec<String> {
    inherent_facts(&expr.inner)
        .unwrap_or_default()
        .into_iter()
        .map(predicate_name)
        .map(String::from)
        .collect()
}

/// Return all consequences entailed by a predicate.
#[pyfunction]
pub fn predicate_closure(pred_name: &str) -> PyResult<Vec<String>> {
    let pred = parse_predicate(pred_name)?;
    Ok(pred
        .closure()
        .into_iter()
        .map(predicate_name)
        .map(String::from)
        .collect())
}

/// Return all predicates contradicted by a predicate.
#[pyfunction]
pub fn predicate_contradictions(pred_name: &str) -> PyResult<Vec<String>> {
    let pred = parse_predicate(pred_name)?;
    let all_preds = [
        Predicate::Complex,
        Predicate::Real,
        Predicate::Rational,
        Predicate::Integer,
        Predicate::Algebraic,
        Predicate::Transcendental,
        Predicate::Positive,
        Predicate::Negative,
        Predicate::NonNegative,
        Predicate::NonPositive,
        Predicate::Zero,
        Predicate::NonZero,
        Predicate::Prime,
        Predicate::Even,
        Predicate::Odd,
        Predicate::Finite,
        Predicate::Infinite,
    ];
    let mut contradicted = Vec::new();
    for other in all_preds {
        if Predicate::contradicts(pred, other) {
            contradicted.push(predicate_name(other).to_string());
        }
    }
    Ok(contradicted)
}
