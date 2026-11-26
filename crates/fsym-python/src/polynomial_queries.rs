//! Source-bound exact QQ/ZZ polynomial coefficient queries.
//!
//! Projection uses the admitted native coefficient map in the original typed
//! ring. Independent rational-source replay checks the full input before any
//! ordered terms are returned; cancellation, resource refusal and false source
//! declarations never trigger an algebraic fallback.

use crate::callbacks::PySupervisedCallbackSession;
use crate::expr::PyExpr;
use crate::finite_field_multivariate::monomial_cmp;
use crate::ideal_operations::{budget, charge, symbols};
use crate::recursive_division::{integral, polynomial_source};
use crate::to_value_error;
use fsym_budget::{Budget, BudgetError, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError};
use fsym_core::Expr;
use fsym_polys::TermOrder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::mem::size_of;

type TermParts = Vec<(Vec<u32>, PyExpr)>;

pyo3::create_exception!(_native, PolynomialQueryCoefficientError, PyValueError);

/// Each native phase retains its fixed envelope while both debit the same
/// caller-owned request. Preflight all private dimensions before forwarding a
/// charge, so a rejected batch never partially spends either ledger.
struct QueryMeter<'py, 'session> {
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    private: Budget,
    failure: Option<PyErr>,
}

impl<'py, 'session> QueryMeter<'py, 'session> {
    fn new(py: Python<'py>, session: Option<&'session PySupervisedCallbackSession>) -> Self {
        Self {
            py,
            session,
            private: budget(),
            failure: None,
        }
    }

    fn forward(&mut self, steps: u64) -> Result<(), MeterError> {
        if self.failure.is_some() {
            return Err(MeterError::Cancelled);
        }
        if let Some(session) = self.session
            && let Err(error) = session.charge_nested_steps_atomic(self.py, steps)
        {
            // Stop native recursion with MeterError but return the original
            // Python ownership, budget or cancellation failure at the boundary.
            self.failure = Some(error);
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }

    fn refuse(&mut self, error: BudgetError) -> MeterError {
        self.failure = Some(match self.session {
            Some(session) => session
                .cancel(self.py)
                .err()
                .unwrap_or_else(|| to_value_error(error)),
            None => to_value_error(error),
        });
        error.into()
    }
}

impl BudgetMeter for QueryMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if self.failure.is_some() {
            return Err(MeterError::Cancelled);
        }
        let mut totals = [0u64; DIMENSION_COUNT];
        for &(dimension, amount) in charges {
            if amount == 0 {
                return Err(self.refuse(BudgetError::ZeroCharge));
            }
            let index = dimension.index();
            let Some(total) = totals[index].checked_add(amount) else {
                return Err(self.refuse(BudgetError::ChargeOverflow { dimension }));
            };
            totals[index] = total;
        }
        for dimension in Dimension::ALL {
            let amount = totals[dimension.index()];
            let remaining = self.private.remaining(dimension);
            if amount > remaining {
                return Err(self.refuse(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
        }
        self.forward(totals[Dimension::ComputeSteps.index()])?;
        self.private.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.forward(0)
    }
}

/// Classify the original borrowed grammar before selecting the existing
/// generic coefficient surface. This is a bounded syntax probe, not polynomial
/// or domain admission: an exact-source failure cannot select this lane later.
///
/// Inspect all children after finding generic syntax so a constant/function
/// cannot hide an oversized literal, name or remaining expression forest.
#[pyfunction]
pub fn polynomial_query_generic_source_native(source: &PyExpr) -> PyResult<bool> {
    let mut pending = vec![(&source.inner, 0usize)];
    let mut visited = 0usize;
    let mut name_bytes = 0usize;
    let mut generic = false;
    while let Some((node, depth)) = pending.pop() {
        visited += 1;
        if visited > 4096 || depth > 64 {
            return Err(PyValueError::new_err(
                "polynomial query source exceeds its node or depth bound",
            ));
        }
        let children = match node {
            Expr::Integer(value) if value.bits() <= 4096 => None,
            Expr::Rational(value)
                if value.numer().bits() <= 4096 && value.denom().bits() <= 4096 =>
            {
                None
            }
            Expr::Integer(_) | Expr::Rational(_) => {
                return Err(PyValueError::new_err(
                    "polynomial query coefficient exceeds 4096 bits",
                ));
            }
            Expr::Sym(symbol) => {
                name_bytes = name_bytes
                    .checked_add(symbol.name.len())
                    .filter(|&bytes| bytes <= 65_536)
                    .ok_or_else(|| {
                        PyValueError::new_err("polynomial query source names exceed 65536 bytes")
                    })?;
                None
            }
            Expr::Const(_) => {
                generic = true;
                None
            }
            Expr::Add(children) | Expr::Mul(children) => Some(children),
            Expr::Function(name, children) => {
                generic = true;
                name_bytes = name_bytes
                    .checked_add(name.len())
                    .filter(|&bytes| bytes <= 65_536)
                    .ok_or_else(|| {
                        PyValueError::new_err("polynomial query source names exceed 65536 bytes")
                    })?;
                Some(children)
            }
            Expr::Pow(base, exponent) => {
                generic |= !matches!(exponent.as_ref(), Expr::Integer(_));
                if 2 > 4096 - visited - pending.len() {
                    return Err(PyValueError::new_err(
                        "polynomial query source exceeds its node bound",
                    ));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
                None
            }
        };
        if let Some(children) = children {
            if children.len() > 4096 - visited - pending.len() {
                return Err(PyValueError::new_err(
                    "polynomial query source exceeds its node bound",
                ));
            }
            pending.extend(children.iter().map(|child| (child, depth + 1)));
        }
    }
    Ok(generic)
}

/// Return descending ordered support and literal exact coefficients. Zero has
/// empty support; public Poly methods apply their own conventional zero term.
#[pyfunction]
#[pyo3(signature = (source, variables, domain="auto", order="lex", *, session=None))]
pub fn polynomial_terms_native(
    py: Python<'_>,
    source: &PyExpr,
    variables: &Bound<'_, PyList>,
    domain: &str,
    order: &str,
    session: Option<PyRef<'_, PySupervisedCallbackSession>>,
) -> PyResult<TermParts> {
    if variables.len() > 8 {
        return Err(PyValueError::new_err(
            "checked polynomial queries support at most eight generators",
        ));
    }
    // Ownership checks can execute Python. Capture the bounded original
    // coordinates first; compute from private copies even if a callback swaps
    // and restores entries in the caller's mutable list.
    let retained: Vec<_> = variables.iter().collect();
    let mut generator = QueryMeter::new(py, session.as_deref());
    let mut verifier = QueryMeter::new(py, session.as_deref());
    let result = (|| {
        charge(
            &mut generator,
            retained.len() + 1,
            (retained.len() + 1) * 128,
        )?;
        let coordinates = PyList::new(py, &retained)?;
        let output = query(
            source,
            &coordinates,
            domain,
            order,
            &mut generator,
            &mut verifier,
        )?;
        // All native checkpoints precede the terminal identity guard. There
        // are no later callbacks through which a substituted ring can publish.
        if variables.len() != retained.len()
            || variables
                .iter()
                .zip(&retained)
                .any(|(current, previous)| !current.is(previous))
        {
            return Err(PyRuntimeError::new_err(
                "polynomial query coordinates mutated during exact evaluation",
            ));
        }
        Ok(output)
    })();
    match generator.failure.or(verifier.failure) {
        Some(error) => Err(error),
        None => result,
    }
}

fn query(
    source: &PyExpr,
    variables: &Bound<'_, PyList>,
    domain: &str,
    order: &str,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> PyResult<TermParts> {
    if !matches!(domain, "auto" | "ZZ" | "QQ") {
        return Err(PyValueError::new_err(
            "checked polynomial queries support ZZ and QQ",
        ));
    }
    let order = match order {
        "lex" => TermOrder::Lex,
        "grlex" => TermOrder::DegLex,
        "grevlex" => TermOrder::DegRevLex,
        _ => {
            return Err(PyValueError::new_err(
                "polynomial query order must be lex, grlex or grevlex",
            ));
        }
    };
    let variables = symbols(variables, generator)?;
    let polynomial = polynomial_source(&source.inner, &variables, generator, verifier)?;
    if domain == "ZZ" && !integral(&polynomial, generator)? {
        return Err(PolynomialQueryCoefficientError::new_err(
            "noninteger source coefficients cannot be represented in ZZ",
        ));
    }

    // The admitted map has at most 4096 terms, eight coordinates, total degree
    // 256 and 4096-bit coefficients. Reserve sorting, exponent copies, exact
    // integer/rational payloads and Python result containers before allocation.
    let count = polynomial.terms().len();
    let coefficient_limbs = polynomial
        .terms()
        .values()
        .map(|value| value.numer().limb_count() as usize + value.denom().limb_count() as usize)
        .sum::<usize>();
    let sorting_steps =
        4 * (count + 1) * (count + 1).ilog2().max(1) as usize * (variables.len() + 1);
    let output_bytes =
        (count + 1) * (variables.len() * (size_of::<u32>() + 64) + 512) + 16 * coefficient_limbs;
    charge(generator, sorting_steps + coefficient_limbs, output_bytes)?;
    let mut terms: Vec<_> = polynomial.terms().iter().collect();
    terms.sort_unstable_by(|(left, _), (right, _)| monomial_cmp(right, left, order));
    let output = terms
        .into_iter()
        .map(|(powers, coefficient)| {
            let literal = if coefficient.is_integer() {
                Expr::Integer(coefficient.numer().clone())
            } else {
                Expr::Rational(coefficient.clone())
            };
            (powers.to_vec(), PyExpr::from_expr(literal))
        })
        .collect();
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(output)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "PolynomialQueryCoefficientError",
        module.py().get_type::<PolynomialQueryCoefficientError>(),
    )?;
    module.add_function(wrap_pyfunction!(polynomial_terms_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        polynomial_query_generic_source_native,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::BudgetLimits;
    use fsym_core::{BigRational, Constant, Symbol, SymbolIdentity};
    use std::sync::Arc;

    fn power(base: Expr, exponent: i64) -> Expr {
        Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
    }

    fn rational(numerator: i64, denominator: i64) -> Expr {
        Expr::Rational(BigRational::new(numerator.into(), denominator.into()))
    }

    fn plain_terms(parts: TermParts) -> Vec<(Vec<u32>, Expr)> {
        parts
            .into_iter()
            .map(|(powers, coefficient)| (powers, coefficient.inner))
            .collect()
    }

    fn session(py: Python<'_>, steps: u64) -> PySupervisedCallbackSession {
        PySupervisedCallbackSession::new(py, 8, 128, steps, 2048, 2048, 60_000).unwrap()
    }

    #[test]
    fn exact_three_variable_terms_follow_each_requested_order() {
        Python::initialize();
        Python::attach(|py| {
            let x = Expr::symbol("x");
            let y = Expr::symbol("y");
            let z = Expr::symbol("z");
            // x^2*z + x*y^2 + y^4 + (2/3)*z^2 - 5.
            let source = PyExpr::from_expr(Expr::Add(vec![
                Expr::Mul(vec![power(x.clone(), 2), z.clone()]),
                Expr::Mul(vec![x, power(y.clone(), 2)]),
                power(y, 4),
                Expr::Mul(vec![rational(2, 3), power(z, 2)]),
                Expr::from_i64(-5),
            ]));
            let variables = PyList::new(py, ["x", "y", "z"]).unwrap();
            let first = (vec![2, 0, 1], Expr::from_i64(1));
            let second = (vec![1, 2, 0], Expr::from_i64(1));
            let third = (vec![0, 4, 0], Expr::from_i64(1));
            let fourth = (vec![0, 0, 2], rational(2, 3));
            let fifth = (vec![0, 0, 0], Expr::from_i64(-5));
            for (order, expected) in [
                (
                    "lex",
                    vec![
                        first.clone(),
                        second.clone(),
                        third.clone(),
                        fourth.clone(),
                        fifth.clone(),
                    ],
                ),
                (
                    "grlex",
                    vec![
                        third.clone(),
                        first.clone(),
                        second.clone(),
                        fourth.clone(),
                        fifth.clone(),
                    ],
                ),
                ("grevlex", vec![third, second, first, fourth, fifth]),
            ] {
                let output =
                    polynomial_terms_native(py, &source, &variables, "QQ", order, None).unwrap();
                assert_eq!(plain_terms(output), expected);
            }
        });
    }

    #[test]
    fn same_named_typed_generators_and_unused_axes_remain_distinct() {
        Python::initialize();
        Python::attach(|py| {
            let plain = Expr::symbol("x");
            let typed = Expr::Sym(Symbol::with_identity(
                "x",
                SymbolIdentity::from_assumptions_digest([23; 32]),
            ));
            let source = PyExpr::from_expr(Expr::Add(vec![
                power(plain.clone(), 2),
                Expr::Mul(vec![Expr::from_i64(3), typed.clone()]),
            ]));
            let bindings: Vec<_> = [typed, plain, Expr::symbol("unused")]
                .into_iter()
                .map(|expr| Py::new(py, PyExpr::from_expr(expr)).unwrap())
                .collect();
            let variables = PyList::new(py, &bindings).unwrap();
            let output =
                polynomial_terms_native(py, &source, &variables, "ZZ", "lex", None).unwrap();
            assert_eq!(
                plain_terms(output),
                vec![
                    (vec![1, 0, 0], Expr::from_i64(3)),
                    (vec![0, 2, 0], Expr::from_i64(1)),
                ]
            );
        });
    }

    #[test]
    fn zero_support_and_declared_integer_domain_are_exact() {
        Python::initialize();
        Python::attach(|py| {
            let variables = PyList::new(py, ["x"]).unwrap();
            let zero = PyExpr::from_expr(Expr::from_i64(0));
            assert!(
                polynomial_terms_native(py, &zero, &variables, "ZZ", "lex", None)
                    .unwrap()
                    .is_empty()
            );
            let fraction = PyExpr::from_expr(Expr::Mul(vec![rational(1, 7), Expr::symbol("x")]));
            let error =
                polynomial_terms_native(py, &fraction, &variables, "ZZ", "lex", None).unwrap_err();
            assert!(error.is_instance_of::<PolynomialQueryCoefficientError>(py));
            assert_eq!(
                plain_terms(
                    polynomial_terms_native(py, &fraction, &variables, "auto", "lex", None)
                        .unwrap()
                ),
                vec![(vec![1], rational(1, 7))]
            );
        });
    }

    #[test]
    fn hidden_poles_foreign_axes_and_resource_excess_do_not_publish() {
        Python::initialize();
        Python::attach(|py| {
            let variables = PyList::new(py, ["x"]).unwrap();
            let inputs = [
                Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
                Expr::Mul(vec![Expr::from_i64(0), power(Expr::symbol("x"), -1)]),
                Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("foreign")]),
                Expr::Mul(vec![
                    Expr::from_i64(0),
                    Expr::Function("sin".into(), vec![Expr::symbol("x")]),
                ]),
                power(Expr::symbol("x"), 257),
                Expr::Add(vec![Expr::from_i64(0); 4097]),
            ];
            for expression in inputs {
                let source = PyExpr::from_expr(expression);
                assert!(
                    polynomial_terms_native(py, &source, &variables, "QQ", "lex", None).is_err()
                );
            }
        });
    }

    #[test]
    fn both_native_phases_and_repeated_queries_share_one_caller_ledger() {
        Python::initialize();
        Python::attach(|py| {
            let source = PyExpr::from_expr(Expr::Mul(vec![
                Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]),
                Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(-1)]),
            ]));
            let variables = PyList::new(py, ["x"]).unwrap();
            let mut reference_generator = budget();
            let mut reference_verifier = budget();
            let reference = query(
                &source,
                &variables,
                "ZZ",
                "lex",
                &mut reference_generator,
                &mut reference_verifier,
            )
            .unwrap();
            assert_eq!(
                plain_terms(reference),
                vec![(vec![2], Expr::from_i64(1)), (vec![0], Expr::from_i64(-1))]
            );
            let generated = 20_000_000 - reference_generator.remaining(Dimension::ComputeSteps);
            let verified = 20_000_000 - reference_verifier.remaining(Dimension::ComputeSteps);
            assert!(generated > 0 && verified > 0);
            let request = session(py, generated + verified);
            let mut generator = QueryMeter::new(py, Some(&request));
            let mut verifier = QueryMeter::new(py, Some(&request));
            query(
                &source,
                &variables,
                "ZZ",
                "lex",
                &mut generator,
                &mut verifier,
            )
            .unwrap();
            assert_eq!(request.consumed_nested_steps(), generated + verified);
            assert!(!request.cancelled());

            let mut retry_generator = QueryMeter::new(py, Some(&request));
            let mut retry_verifier = QueryMeter::new(py, Some(&request));
            assert!(
                query(
                    &source,
                    &variables,
                    "ZZ",
                    "lex",
                    &mut retry_generator,
                    &mut retry_verifier,
                )
                .is_err()
            );
            assert!(request.cancelled());
            assert_eq!(request.consumed_nested_steps(), generated + verified);
            assert!(retry_generator.failure.is_some());
        });
    }

    #[test]
    fn a_private_refusal_is_atomic_and_cancellation_blocks_publication() {
        Python::initialize();
        Python::attach(|py| {
            let request = session(py, 100);
            let mut meter = QueryMeter::new(py, Some(&request));
            meter.private = Budget::new(BudgetLimits::uniform(1, 0));
            let before = meter.private.snapshot();
            assert!(
                meter
                    .charge_batch(&[(Dimension::ComputeSteps, 1), (Dimension::MemoryBytes, 2)])
                    .is_err()
            );
            assert_eq!(meter.private.snapshot(), before);
            assert_eq!(request.consumed_nested_steps(), 0);
            assert!(request.cancelled());
            assert!(meter.checkpoint().is_err());
            drop(meter);

            let source = PyExpr::from_expr(Expr::symbol("x"));
            let variables = PyList::new(py, ["x"]).unwrap();
            let request = Py::new(py, request).unwrap();
            let error = polynomial_terms_native(
                py,
                &source,
                &variables,
                "ZZ",
                "lex",
                Some(request.borrow(py)),
            )
            .unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
        });
    }

    #[test]
    fn generic_coefficient_selection_is_syntax_only_and_preserves_exact_refusals() {
        Python::initialize();
        Python::attach(|_| {
            let x = Expr::symbol("x");
            for source in [
                Expr::Mul(vec![Expr::Const(Constant::I), x.clone()]),
                Expr::Mul(vec![Expr::Const(Constant::Pi), x.clone()]),
                Expr::Mul(vec![
                    Expr::Pow(Arc::new(Expr::from_i64(2)), Arc::new(rational(1, 2))),
                    x.clone(),
                ]),
                Expr::Function("sin".into(), vec![x.clone()]),
            ] {
                assert!(
                    polynomial_query_generic_source_native(&PyExpr::from_expr(source)).unwrap()
                );
            }
            for source in [
                Expr::Mul(vec![rational(1, 3), power(x.clone(), 2)]),
                Expr::Mul(vec![Expr::from_i64(0), power(x, -1)]),
                Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
            ] {
                // Symbolic reciprocals and zero poles belong to the exact
                // grammar's terminal admission checks, never a generic retry.
                assert!(
                    !polynomial_query_generic_source_native(&PyExpr::from_expr(source)).unwrap()
                );
            }
        });
    }

    #[test]
    fn generic_selection_still_bounds_the_complete_original_forest() {
        Python::initialize();
        Python::attach(|_| {
            let flagged = Expr::Const(Constant::Pi);
            let mut deep = Expr::symbol("x");
            for _ in 0..66 {
                deep = power(deep, 1);
            }
            for other in [
                Expr::symbol("x".repeat(65_537)),
                Expr::Add(vec![Expr::from_i64(0); 4096]),
                deep,
            ] {
                let source = PyExpr::from_expr(Expr::Add(vec![other, flagged.clone()]));
                assert!(polynomial_query_generic_source_native(&source).is_err());
            }
        });
    }
}
