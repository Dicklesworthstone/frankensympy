//! Assumptions and predicate query Python bindings (WS04).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fsym_assumptions::{AssumptionsContext, Domain, Predicate, TruthValue, inherent_facts};
use fsym_core::Symbol;

use crate::binding::PySymbolBinding;
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

    /// Assert a predicate about the exact typed shell symbol. The legacy
    /// string-taking method remains a deliberately plain-symbol API.
    pub fn assume_binding(&mut self, binding: &PySymbolBinding, pred_name: &str) -> PyResult<()> {
        let pred = parse_predicate(pred_name)?;
        let symbol = binding.to_symbol();
        let mut candidate = self.inner.clone();
        candidate
            .assume(symbol.clone(), pred)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        // Some legacy lattice exclusions are stated in one direction only.
        // Checking the complete candidate in both directions makes insertion
        // order irrelevant without publishing a contradictory intermediate.
        let facts = candidate.deductions(&symbol);
        if facts
            .iter()
            .any(|a| facts.iter().any(|b| Predicate::contradicts(*a, *b)))
        {
            return Err(PyValueError::new_err("Contradictory assumptions inferred"));
        }
        self.inner = candidate;
        Ok(())
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

    /// Query consequences without collapsing same-named typed symbols.
    pub fn deductions_binding(&self, binding: &PySymbolBinding) -> Vec<String> {
        self.inner
            .deductions(&binding.to_symbol())
            .into_iter()
            .map(predicate_name)
            .map(String::from)
            .collect()
    }

    /// Keep domain assertions in the same typed identity universe as facts.
    pub fn assume_domain_binding(
        &mut self,
        binding: &PySymbolBinding,
        domain_name: &str,
    ) -> PyResult<()> {
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
        // A domain can contradict existing facts (e.g. ZZ and Infinite).
        // Validate a candidate first so a failed insertion changes nothing.
        let symbol = binding.to_symbol();
        let mut candidate = self.inner.clone();
        candidate
            .assume_domain(symbol.clone(), dom)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let facts = candidate.deductions(&symbol);
        if facts
            .iter()
            .any(|a| facts.iter().any(|b| Predicate::contradicts(*a, *b)))
        {
            return Err(PyValueError::new_err("Contradictory assumptions inferred"));
        }
        self.inner = candidate;
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::Expr;

    fn binding(name: &str, facts: &[(&str, &str)]) -> PySymbolBinding {
        PySymbolBinding::new(
            name.to_string(),
            Some(
                facts
                    .iter()
                    .map(|(key, value)| (key.to_string(), value.to_string()))
                    .collect(),
            ),
        )
        .unwrap()
    }

    #[test]
    fn typed_predicate_facts_do_not_leak_to_same_name_symbols() {
        let plain = binding("same", &[]);
        let typed = binding("same", &[("integer", "True")]);
        let other = binding("same", &[("real", "True")]);
        let mut context = PyAssumptionsContext::new();
        context.assume_binding(&typed, "positive").unwrap();
        for (symbol, expected) in [(&typed, Some(true)), (&plain, None), (&other, None)] {
            let value = PyExpr::from_expr(Expr::Sym(symbol.to_symbol()));
            assert_eq!(context.is_true(&value, "positive").unwrap(), expected);
        }
        assert!(context.deductions("same").is_empty());
        assert!(
            context
                .deductions_binding(&typed)
                .contains(&"real".to_string())
        );
    }

    #[test]
    fn rejected_typed_predicate_keeps_the_previous_context() {
        let typed = binding("x", &[("real", "True")]);
        for (first, second) in [
            ("positive", "negative"),
            ("nonnegative", "negative"),
            ("negative", "nonnegative"),
            ("nonpositive", "positive"),
            ("positive", "nonpositive"),
        ] {
            let mut context = PyAssumptionsContext::new();
            context.assume_binding(&typed, first).unwrap();
            let before = context.inner.clone();
            assert!(context.assume_binding(&typed, second).is_err());
            assert_eq!(context.inner, before);
        }
    }

    #[test]
    fn typed_domains_preserve_identity_and_reject_conflicting_facts_atomically() {
        let plain = binding("x", &[]);
        let typed = binding("x", &[("real", "True")]);
        let mut context = PyAssumptionsContext::new();
        context.assume_domain_binding(&typed, "ZZ").unwrap();
        assert!(
            context
                .deductions_binding(&typed)
                .contains(&"integer".to_string())
        );
        assert!(context.deductions_binding(&plain).is_empty());
        let before = context.inner.clone();
        assert!(context.assume_binding(&typed, "infinite").is_err());
        assert_eq!(context.inner, before);

        let mut reverse = PyAssumptionsContext::new();
        reverse.assume_binding(&typed, "infinite").unwrap();
        let before = reverse.inner.clone();
        assert!(reverse.assume_domain_binding(&typed, "ZZ").is_err());
        assert_eq!(reverse.inner, before);
        assert!(
            reverse
                .assume_domain_binding(&typed, "not-a-domain")
                .is_err()
        );
        assert_eq!(reverse.inner, before);
    }

    #[test]
    fn named_facts_remain_plain_without_modifying_typed_facts() {
        let plain = binding("x", &[]);
        let typed = binding("x", &[("real", "True")]);
        let mut context = PyAssumptionsContext::new();
        context.assume_binding(&typed, "negative").unwrap();
        context.assume("x", "positive").unwrap();
        assert_eq!(
            context
                .is_true(&PyExpr::from_expr(Expr::Sym(plain.to_symbol())), "positive")
                .unwrap(),
            Some(true)
        );
        assert_eq!(
            context
                .is_true(&PyExpr::from_expr(Expr::Sym(typed.to_symbol())), "negative")
                .unwrap(),
            Some(true)
        );
    }
}
