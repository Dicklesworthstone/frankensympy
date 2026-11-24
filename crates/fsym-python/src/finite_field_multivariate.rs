//! Source-bound public operations in a bounded multivariate prime-field ring.
//!
//! Original expressions, arithmetic, independent verification and publication
//! share one cumulative pair of budgets. Typed generators never round-trip
//! through a string representation.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::finite_field_polynomial::{budget, charge, polynomial_error, source_error};
use crate::to_value_error;
use fsym_budget::{Budget, BudgetMeter};
use fsym_core::{Expr, Symbol};
use fsym_polys::{
    PrimeFieldMultivariateOperation as Op, PrimeFieldMultivariatePolynomial as Polynomial,
    TermOrder, checked_prime_field_evaluation, checked_prime_field_multivariate_arithmetic,
    checked_prime_field_multivariate_division, checked_prime_field_multivariate_gcd,
    checked_prime_field_multivariate_source, checked_prime_field_multivariate_square_free,
    checked_prime_field_recursive_division,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::Arc;

type TermParts = Vec<(Vec<u32>, PyExpr)>;
type SquareFreeParts = (PyExpr, Vec<(PyExpr, u32)>, PyExpr);

pub(crate) fn order(value: &str) -> PyResult<TermOrder> {
    match value {
        "lex" => Ok(TermOrder::Lex),
        "grlex" => Ok(TermOrder::DegLex),
        "grevlex" => Ok(TermOrder::DegRevLex),
        _ => Err(PyValueError::new_err(
            "prime-field monomial order must be lex, grlex, or grevlex",
        )),
    }
}

pub(crate) fn symbols(values: &Bound<'_, PyList>, meter: &mut Budget) -> PyResult<Vec<Symbol>> {
    if values.len() > 8 {
        return Err(PyValueError::new_err(
            "prime-field polynomial requires at most eight typed generators",
        ));
    }
    let mut name_bytes = 0usize;
    for value in values.iter() {
        let variable: PyRef<'_, PySymbolBinding> = value.extract()?;
        name_bytes = name_bytes
            .checked_add(variable.name.len())
            .filter(|bytes| *bytes <= 65_536)
            .ok_or_else(|| PyValueError::new_err("generator names exceed 65536 bytes"))?;
    }
    charge(
        meter,
        name_bytes + values.len(),
        name_bytes + values.len() * std::mem::size_of::<Symbol>(),
    )?;
    values
        .iter()
        .map(|value| {
            let variable: PyRef<'_, PySymbolBinding> = value.extract()?;
            Ok(variable.to_symbol())
        })
        .collect()
}

fn coefficient(value: u64, prime: u64, symmetric: bool) -> i64 {
    if symmetric && value > prime / 2 {
        value as i64 - prime as i64
    } else {
        value as i64
    }
}

pub(crate) fn lift(
    polynomial: &Polynomial,
    symmetric: bool,
    meter: &mut Budget,
) -> PyResult<PyExpr> {
    charge(
        meter,
        polynomial.terms().len() + 1,
        (polynomial.terms().len() + 1) * std::mem::size_of::<Expr>(),
    )?;
    let mut terms = Vec::with_capacity(polynomial.terms().len());
    for (powers, &value) in polynomial.terms() {
        let names: usize = powers
            .iter()
            .zip(polynomial.generators())
            .filter(|(power, _)| **power != 0)
            .map(|(_, variable)| variable.name.len() + 512)
            .sum();
        charge(meter, powers.len() + 1, names + 512)?;
        let mut factors = Vec::with_capacity(powers.len() + 1);
        let value = coefficient(value, polynomial.prime(), symmetric);
        if value != 1 || powers.iter().all(|power| *power == 0) {
            factors.push(Expr::from_i64(value));
        }
        for (&power, variable) in powers.iter().zip(polynomial.generators()) {
            match power {
                0 => {}
                1 => factors.push(Expr::Sym(variable.clone())),
                _ => factors.push(Expr::Pow(
                    Arc::new(Expr::Sym(variable.clone())),
                    Arc::new(Expr::from_i64(i64::from(power))),
                )),
            }
        }
        terms.push(match factors.len() {
            0 => Expr::from_i64(1),
            1 => factors.pop().expect("one admitted factor"),
            _ => Expr::Mul(factors),
        });
    }
    let result = match terms.len() {
        0 => Expr::from_i64(0),
        1 => terms.pop().expect("one admitted term"),
        _ => Expr::Add(terms),
    };
    meter.checkpoint().map_err(to_value_error)?;
    Ok(PyExpr::from_expr(result))
}

/// Partial evaluation preserves the declared prime field and removes only the
/// assigned original typed coordinates. The original source, integer-to-field
/// reductions, exact contraction, independent verifier and output lift all
/// consume the same cumulative pair of ledgers.
#[pyfunction]
#[pyo3(signature = (source, variables, prime, values, symmetric=true))]
pub fn prime_field_evaluate_native(
    source: &PyExpr,
    variables: &Bound<'_, PyList>,
    prime: u64,
    values: &Bound<'_, PyList>,
    symmetric: bool,
) -> PyResult<PyExpr> {
    if values.len() != variables.len() || values.len() > 8 {
        return Err(PyValueError::new_err(
            "prime-field evaluation requires one assignment per original generator",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    let source = checked_prime_field_multivariate_source(
        &source.inner,
        &variables,
        prime,
        &mut generator,
        &mut verifier,
    )
    .map_err(source_error)?;
    charge(
        &mut generator,
        values.len(),
        values.len() * std::mem::size_of::<Option<u64>>(),
    )?;
    let mut assignments = Vec::with_capacity(values.len());
    for value in values.iter() {
        if value.is_none() {
            assignments.push(None);
            continue;
        }
        let value: PyRef<'_, PyExpr> = value.extract()?;
        let exact_integer = match &value.inner {
            Expr::Integer(integer) => integer.bits() <= 4096,
            Expr::Rational(rational) => rational.is_integer() && rational.numer().bits() <= 4096,
            _ => false,
        };
        if !exact_integer {
            return Err(PyValueError::new_err(
                "prime-field evaluation values must be exact integers of at most 4096 bits",
            ));
        }
        // The source checker independently verifies reduction of signed and
        // large integers; the shell cannot supply an unchecked modular image.
        let checked = checked_prime_field_multivariate_source(
            &value.inner,
            &[],
            prime,
            &mut generator,
            &mut verifier,
        )
        .map_err(source_error)?;
        assignments.push(Some(
            checked
                .polynomial()
                .terms()
                .values()
                .next()
                .copied()
                .unwrap_or(0),
        ));
    }
    let accepted = checked_prime_field_evaluation(
        source.polynomial(),
        &assignments,
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    let result = lift(accepted.result(), symmetric, &mut generator)?;
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(result)
}

/// One cumulative pair of ledgers covers every original source, generation,
/// independent verification, all result lifts and final publication.
#[pyfunction]
#[pyo3(signature = (sources, variables, prime, operation, parameter=0, derivative_orders=None, order="lex", symmetric=true))]
#[allow(clippy::too_many_arguments)]
pub fn prime_field_multivariate_native(
    sources: &Bound<'_, PyList>,
    variables: &Bound<'_, PyList>,
    prime: u64,
    operation: &str,
    parameter: u64,
    derivative_orders: Option<&Bound<'_, PyList>>,
    order: &str,
    symmetric: bool,
) -> PyResult<Vec<PyExpr>> {
    let term_order = self::order(order)?;
    let valid_count = match operation {
        "reduced" => (1..=33).contains(&sources.len()),
        "add" | "sub" | "mul" | "compose" | "divrem" | "gcd" | "lcm" => sources.len() == 2,
        "normalize" | "neg" | "monic" | "pow" | "diff" => sources.len() == 1,
        _ => false,
    };
    if !valid_count {
        return Err(PyValueError::new_err(
            "unsupported multivariate prime-field operation or operand count",
        ));
    }
    if matches!(operation, "divrem" | "gcd" | "lcm") && term_order != TermOrder::Lex {
        return Err(PyValueError::new_err(
            "polynomial division and GCD/LCM use the declared generator order",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    let arithmetic = match operation {
        "normalize" => Some(Op::Normalize),
        "neg" => Some(Op::Negate),
        "add" => Some(Op::Add),
        "sub" => Some(Op::Subtract),
        "mul" => Some(Op::Multiply),
        "compose" => Some(Op::Compose),
        "monic" => Some(Op::Monic(term_order)),
        "pow" => Some(Op::Power(parameter)),
        "diff" => {
            let orders = derivative_orders.ok_or_else(|| {
                PyValueError::new_err("mixed derivative requires an order per generator")
            })?;
            if orders.len() != variables.len() {
                return Err(PyValueError::new_err(
                    "mixed derivative requires an order per generator",
                ));
            }
            charge(
                &mut generator,
                orders.len(),
                orders.len() * std::mem::size_of::<u32>(),
            )?;
            Some(Op::Derivative(
                orders
                    .iter()
                    .map(|value| value.extract::<u32>())
                    .collect::<PyResult<Vec<_>>>()?,
            ))
        }
        "divrem" | "reduced" | "gcd" | "lcm" => None,
        _ => unreachable!("operation and operand count were admitted"),
    };
    if operation != "diff" && derivative_orders.is_some() {
        return Err(PyValueError::new_err(
            "derivative orders require the diff operation",
        ));
    }
    charge(
        &mut generator,
        sources.len(),
        sources.len() * std::mem::size_of::<Polynomial>(),
    )?;
    let mut polynomials = Vec::with_capacity(sources.len());
    for source in sources.iter() {
        let source: PyRef<'_, PyExpr> = source.extract()?;
        polynomials.push(
            checked_prime_field_multivariate_source(
                &source.inner,
                &variables,
                prime,
                &mut generator,
                &mut verifier,
            )
            .map_err(source_error)?
            .into_polynomial(),
        );
    }
    charge(
        &mut generator,
        polynomials.len(),
        (polynomials.len() + 1) * 128,
    )?;
    let mut output = Vec::with_capacity(polynomials.len() + 1);
    if let Some(operation) = arithmetic {
        let accepted = checked_prime_field_multivariate_arithmetic(
            &operation,
            &polynomials[0],
            polynomials.get(1),
            &mut generator,
            &mut verifier,
        )
        .map_err(polynomial_error)?;
        output.push(lift(accepted.result(), symmetric, &mut generator)?);
    } else if matches!(operation, "gcd" | "lcm") {
        let accepted = checked_prime_field_multivariate_gcd(
            &polynomials[0],
            &polynomials[1],
            &mut generator,
            &mut verifier,
        )
        .map_err(polynomial_error)?;
        let result = if operation == "gcd" {
            accepted.gcd()
        } else {
            accepted.lcm()
        };
        output.push(lift(result, symmetric, &mut generator)?);
    } else if operation == "divrem" && !variables.is_empty() {
        // Poly.div recursively divides in the first declared variable over
        // the polynomial ring in the rest. A coefficient obstruction stops
        // it before the complete sparse normal form used by reduced().
        let accepted = checked_prime_field_recursive_division(
            &polynomials[0],
            &polynomials[1],
            &mut generator,
            &mut verifier,
        )
        .map_err(polynomial_error)?;
        output.push(lift(accepted.quotient(), symmetric, &mut generator)?);
        output.push(lift(accepted.remainder(), symmetric, &mut generator)?);
    } else {
        // A constant field has no first variable for recursive division. Its
        // one-divisor quotient/remainder is the checked sparse normal form.
        let accepted = checked_prime_field_multivariate_division(
            &polynomials[0],
            &polynomials[1..],
            term_order,
            &mut generator,
            &mut verifier,
        )
        .map_err(polynomial_error)?;
        for quotient in accepted.quotients() {
            output.push(lift(quotient, symmetric, &mut generator)?);
        }
        output.push(lift(accepted.remainder(), symmetric, &mut generator)?);
    }
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(output)
}

/// Complete multiplicity groups in F_p[x_1, ..., x_n], including inseparable
/// powers. The normalized original source is returned alongside the checked
/// factors so a public Poly result needs no second arithmetic request.
#[pyfunction]
#[pyo3(signature = (source, variables, prime, symmetric=true, mode="sqf"))]
pub fn prime_field_multivariate_square_free_native(
    source: &PyExpr,
    variables: &Bound<'_, PyList>,
    prime: u64,
    symmetric: bool,
    mode: &str,
) -> PyResult<SquareFreeParts> {
    if !matches!(mode, "sqf" | "sqf_part") {
        return Err(PyValueError::new_err(
            "multivariate prime-field square-free mode must be sqf or sqf_part",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    if variables.len() == 1 {
        // Preserve the established degree64 univariate lane. General
        // multivariate GCDs require an auxiliary axis/degree and must not
        // narrow a module-level call that previously used this kernel.
        let (unit, mut factors) = crate::finite_field_polynomial::factor_with_budgets(
            source,
            &variables[0],
            prime,
            symmetric,
            mode,
            &mut generator,
            &mut verifier,
        )?;
        if mode == "sqf_part" && factors.is_empty() {
            charge(&mut generator, 1, 128)?;
            factors.push((PyExpr::from_expr(Expr::from_i64(1)), 1));
        }
        let normalized = checked_prime_field_multivariate_source(
            &source.inner,
            &variables,
            prime,
            &mut generator,
            &mut verifier,
        )
        .map_err(source_error)?
        .into_polynomial();
        let normalized = lift(&normalized, symmetric, &mut generator)?;
        generator.checkpoint().map_err(to_value_error)?;
        verifier.checkpoint().map_err(to_value_error)?;
        return Ok((unit, factors, normalized));
    }
    let source = checked_prime_field_multivariate_source(
        &source.inner,
        &variables,
        prime,
        &mut generator,
        &mut verifier,
    )
    .map_err(source_error)?
    .into_polynomial();
    let accepted =
        checked_prime_field_multivariate_square_free(&source, &mut generator, &mut verifier)
            .map_err(polynomial_error)?;
    let count = if mode == "sqf_part" {
        1
    } else {
        accepted.factors().len()
    };
    charge(&mut generator, count + 1, (count + 1) * 128)?;
    let mut factors = Vec::with_capacity(count);
    if mode == "sqf_part" {
        // Match the pinned public GF(p) convention sqf_part(0) == 1.
        // The zero source still passed the complete native decomposition
        // checker, whose algebraic radical convention is zero.
        let part = if accepted.input().terms().is_empty() {
            charge(&mut generator, 1, 128)?;
            PyExpr::from_expr(Expr::from_i64(1))
        } else {
            lift(accepted.square_free_part(), symmetric, &mut generator)?
        };
        factors.push((part, 1));
    } else {
        for factor in accepted.factors() {
            let multiplicity = u32::try_from(factor.multiplicity).map_err(|_| {
                PyValueError::new_err("square-free multiplicity exceeds its native bound")
            })?;
            factors.push((
                lift(&factor.polynomial, symmetric, &mut generator)?,
                multiplicity,
            ));
        }
    }
    let unit = if mode == "sqf_part" {
        1
    } else {
        accepted.unit()
    };
    charge(&mut generator, 1, 128)?;
    let unit = PyExpr::from_expr(Expr::from_i64(coefficient(unit, prime, symmetric)));
    let normalized = lift(accepted.input(), symmetric, &mut generator)?;
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok((unit, factors, normalized))
}

pub(crate) fn monomial_cmp(left: &[u32], right: &[u32], order: TermOrder) -> Ordering {
    let degree = || left.iter().sum::<u32>().cmp(&right.iter().sum::<u32>());
    match order {
        TermOrder::Lex => left.cmp(right),
        TermOrder::DegLex => degree().then_with(|| left.cmp(right)),
        TermOrder::DegRevLex => degree().then_with(|| right.iter().rev().cmp(left.iter().rev())),
    }
}

/// Project coefficients only after independently admitting the original source.
#[pyfunction]
#[pyo3(signature = (source, variables, prime, symmetric=true, order="lex"))]
pub fn prime_field_multivariate_terms_native(
    source: &PyExpr,
    variables: &Bound<'_, PyList>,
    prime: u64,
    symmetric: bool,
    order: &str,
) -> PyResult<TermParts> {
    let term_order = self::order(order)?;
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(variables, &mut generator)?;
    let accepted = checked_prime_field_multivariate_source(
        &source.inner,
        &variables,
        prime,
        &mut generator,
        &mut verifier,
    )
    .map_err(source_error)?;
    let polynomial = accepted.polynomial();
    let count = polynomial.terms().len();
    charge(
        &mut generator,
        (count + 1) * (count + 1).ilog2().max(1) as usize * (variables.len() + 1),
        (count + 1) * (variables.len() * std::mem::size_of::<u32>() + 512),
    )?;
    let mut terms: Vec<_> = polynomial.terms().iter().collect();
    terms.sort_by(|(left, _), (right, _)| monomial_cmp(right, left, term_order));
    let result = terms
        .into_iter()
        .map(|(powers, &value)| {
            (
                powers.to_vec(),
                PyExpr::from_expr(Expr::from_i64(coefficient(value, prime, symmetric))),
            )
        })
        .collect();
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(result)
}

/// Infer exact typed generators while borrowing, bounding and charging original
/// nodes before cloning or constructing a free-symbol result.
#[pyfunction]
#[pyo3(signature = (source, other=None))]
pub fn prime_field_polynomial_generators_native(
    source: &PyExpr,
    other: Option<&PyExpr>,
) -> PyResult<Vec<PyExpr>> {
    fn visit<'a>(
        expression: &'a Expr,
        depth: usize,
        nodes: &mut usize,
        found: &mut BTreeSet<&'a Symbol>,
        meter: &mut Budget,
    ) -> PyResult<()> {
        charge(meter, 1, 1)?;
        *nodes += 1;
        if *nodes > 4096 || depth > 64 {
            return Err(PyValueError::new_err(
                "prime-field source exceeds 4096 nodes or depth 64",
            ));
        }
        match expression {
            Expr::Integer(value) if value.bits() <= 4096 => {}
            Expr::Rational(value)
                if value.numer().bits() <= 4096 && value.denom().bits() <= 4096 => {}
            Expr::Sym(variable) if variable.name.len() <= 65_536 => {
                charge(meter, (variable.name.len() + 1) * 8, 128)?;
                if found.len() == 8 {
                    if !found.contains(variable) {
                        return Err(PyValueError::new_err(
                            "prime-field source exceeds eight generators",
                        ));
                    }
                } else {
                    found.insert(variable);
                }
            }
            Expr::Add(children) | Expr::Mul(children) => {
                if children.len() > 4096 - *nodes {
                    return Err(PyValueError::new_err(
                        "prime-field source arity exceeds its node envelope",
                    ));
                }
                for child in children {
                    visit(child, depth + 1, nodes, found, meter)?;
                }
            }
            Expr::Pow(base, exponent) => {
                visit(base, depth + 1, nodes, found, meter)?;
                visit(exponent, depth + 1, nodes, found, meter)?;
                if !matches!(exponent.as_ref(), Expr::Integer(value)
                    if value.to_i64().is_some_and(|power| (-256..=256).contains(&power)))
                {
                    return Err(PyValueError::new_err(
                        "prime-field source requires bounded integer powers",
                    ));
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "unsupported or oversized prime-field original source",
                ));
            }
        }
        Ok(())
    }
    let mut meter = budget();
    let (mut nodes, mut found) = (0, BTreeSet::new());
    visit(&source.inner, 0, &mut nodes, &mut found, &mut meter)?;
    if let Some(other) = other {
        visit(&other.inner, 0, &mut nodes, &mut found, &mut meter)?;
    }
    let name_bytes: usize = found.iter().map(|variable| variable.name.len()).sum();
    if name_bytes > 65_536 {
        return Err(PyValueError::new_err("generator names exceed 65536 bytes"));
    }
    charge(
        &mut meter,
        found.len() + 1,
        name_bytes + found.len() * 256 + 128,
    )?;
    let result = found
        .into_iter()
        .map(|variable| PyExpr::from_expr(Expr::Sym(variable.clone())))
        .collect();
    meter.checkpoint().map_err(to_value_error)?;
    Ok(result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(prime_field_evaluate_native, module)?)?;
    module.add_function(wrap_pyfunction!(prime_field_multivariate_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_multivariate_square_free_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_multivariate_terms_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_polynomial_generators_native,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{BudgetLimits, Dimension, Unbounded};
    use fsym_core::SymbolIdentity;
    use std::collections::BTreeMap;

    #[test]
    fn signed_and_unsigned_lifts_retain_every_typed_coordinate() {
        let variables = [
            Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32])),
            Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32])),
        ];
        let terms = BTreeMap::from([
            (vec![2, 1], 4),
            (vec![0, 3], 2),
            (vec![1, 0], 1),
            (vec![0, 0], 3),
        ]);
        let polynomial = Polynomial::new(&variables, 5, &terms).unwrap();
        for symmetric in [false, true] {
            let expression = lift(&polynomial, symmetric, &mut budget()).unwrap();
            let accepted = checked_prime_field_multivariate_source(
                &expression.inner,
                &variables,
                5,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap();
            assert_eq!(accepted.polynomial(), &polynomial);
            assert!(
                checked_prime_field_multivariate_source(
                    &expression.inner,
                    &[variables[0].clone(), Symbol::new("x")],
                    5,
                    &mut Unbounded,
                    &mut Unbounded,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn output_construction_charges_the_callers_existing_ledger() {
        let polynomial = Polynomial::new(
            &[Symbol::new("x")],
            7,
            &BTreeMap::from([(vec![2], 3), (vec![0], 1)]),
        )
        .unwrap();
        let mut limits = BudgetLimits::uniform(1_000_000, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = 3;
        assert!(lift(&polynomial, true, &mut Budget::new(limits)).is_err());
    }

    #[test]
    fn coefficient_projection_distinguishes_all_three_orders() {
        assert_eq!(
            monomial_cmp(&[1, 0, 0], &[0, 2, 0], TermOrder::Lex),
            Ordering::Greater,
        );
        assert_eq!(
            monomial_cmp(&[1, 0, 0], &[0, 2, 0], TermOrder::DegLex),
            Ordering::Less,
        );
        assert_eq!(
            monomial_cmp(&[1, 0, 1], &[0, 2, 0], TermOrder::DegLex),
            Ordering::Greater,
        );
        assert_eq!(
            monomial_cmp(&[1, 0, 1], &[0, 2, 0], TermOrder::DegRevLex),
            Ordering::Less,
        );
    }
}
