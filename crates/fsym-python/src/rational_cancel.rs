//! Original-source-bound rational cancellation with public integer presentation.
//!
//! Complete QQ cancellation, joint coefficient normalization and output lifting
//! retain one generator ledger and one independent verifier ledger. The source
//! receipt, including every removed pole, stays alive through publication.

use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, symbols};
use crate::{polynomial_integer_lcm, to_value_error};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, bounded_polynomial_from_expr, checked_rational_cancel,
    metered_rational_source,
};
use fsym_runtime::Budget;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Decide whether general simplify should try its optional cancellation
/// candidate. This bounded borrowed-tree scan performs no algebra and makes
/// no admission claim: a selected operation must still check its complete
/// retained source, and every failure of that operation remains terminal.
#[pyfunction]
pub fn rational_cancel_candidate_native(source: &PyExpr) -> bool {
    let mut pending = vec![(&source.inner, 0usize)];
    let mut variables: Vec<&Symbol> = Vec::with_capacity(8);
    let mut visited = 0usize;
    let mut name_bytes = 0usize;
    while let Some((node, depth)) = pending.pop() {
        visited += 1;
        if visited > 4096 || depth > 64 {
            return false;
        }
        let children = match node {
            Expr::Integer(value) => {
                if value.bits() > 4096 {
                    return false;
                }
                None
            }
            Expr::Rational(value) => {
                if value.numer().bits() > 4096 || value.denom().bits() > 4096 {
                    return false;
                }
                None
            }
            Expr::Sym(symbol) => {
                name_bytes = name_bytes.saturating_add(symbol.name.len());
                if name_bytes > 65_536 {
                    return false;
                }
                if !variables.contains(&symbol) {
                    if variables.len() == 8 {
                        return false;
                    }
                    variables.push(symbol);
                }
                None
            }
            Expr::Const(_) | Expr::Function(_, _) => return false,
            Expr::Add(children) | Expr::Mul(children) => Some(children),
            Expr::Pow(base, exponent) => {
                let Expr::Integer(power) = exponent.as_ref() else {
                    return false;
                };
                if power
                    .to_i64()
                    .is_none_or(|value| !(-256..=256).contains(&value))
                    || 2 > 4096 - visited - pending.len()
                {
                    return false;
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
                None
            }
        };
        if let Some(children) = children {
            if children.len() > 4096 - visited - pending.len() {
                return false;
            }
            pending.extend(children.iter().map(|child| (child, depth + 1)));
        }
    }
    cancellation_candidate_size(&source.inner).is_some()
}

/// Upper bounds only: no coefficients, monomials or cancellations are
/// computed. A conservative decline is safe for this optional rewrite.
#[derive(Clone, Copy)]
struct CandidatePolynomialSize {
    degree: u32,
    terms: usize,
}

impl CandidatePolynomialSize {
    const CONSTANT: Self = Self {
        degree: 0,
        terms: 1,
    };

    fn add(self, other: Self) -> Option<Self> {
        let degree = self.degree.max(other.degree);
        let terms = self.terms.checked_add(other.terms)?;
        (degree <= 256 && terms <= 4096).then_some(Self { degree, terms })
    }

    fn multiply(self, other: Self) -> Option<Self> {
        let degree = self.degree.checked_add(other.degree)?;
        let terms = self.terms.checked_mul(other.terms)?;
        (degree <= 256 && terms <= 4096).then_some(Self { degree, terms })
    }
}

#[derive(Clone, Copy)]
struct CandidateFractionSize {
    numerator: CandidatePolynomialSize,
    denominator: CandidatePolynomialSize,
}

impl CandidateFractionSize {
    const CONSTANT: Self = Self {
        numerator: CandidatePolynomialSize::CONSTANT,
        denominator: CandidatePolynomialSize::CONSTANT,
    };

    fn add(self, other: Self) -> Option<Self> {
        Some(Self {
            numerator: self
                .numerator
                .multiply(other.denominator)?
                .add(other.numerator.multiply(self.denominator)?)?,
            denominator: self.denominator.multiply(other.denominator)?,
        })
    }

    fn multiply(self, other: Self) -> Option<Self> {
        Some(Self {
            numerator: self.numerator.multiply(other.numerator)?,
            denominator: self.denominator.multiply(other.denominator)?,
        })
    }
}

/// Called only after the borrowed pass has bounded the entire original tree
/// to depth 64 and 4096 nodes. Recursion retains those original bounds even
/// below a zero power; it never simplifies an unsupported child away.
fn cancellation_candidate_size(source: &Expr) -> Option<CandidateFractionSize> {
    match source {
        Expr::Integer(_) | Expr::Rational(_) => Some(CandidateFractionSize::CONSTANT),
        Expr::Sym(_) => Some(CandidateFractionSize {
            numerator: CandidatePolynomialSize {
                degree: 1,
                terms: 1,
            },
            denominator: CandidatePolynomialSize::CONSTANT,
        }),
        Expr::Add(children) => {
            let mut children = children.iter();
            let Some(first) = children.next() else {
                return Some(CandidateFractionSize::CONSTANT);
            };
            children.try_fold(cancellation_candidate_size(first)?, |left, right| {
                left.add(cancellation_candidate_size(right)?)
            })
        }
        Expr::Mul(children) => children
            .iter()
            .try_fold(CandidateFractionSize::CONSTANT, |left, right| {
                left.multiply(cancellation_candidate_size(right)?)
            }),
        Expr::Pow(base, exponent) => {
            let Expr::Integer(power) = exponent.as_ref() else {
                return None;
            };
            let power = power.to_i64()?;
            let base = cancellation_candidate_size(base)?;
            let mut factor = if power < 0 {
                CandidateFractionSize {
                    numerator: base.denominator,
                    denominator: base.numerator,
                }
            } else {
                base
            };
            let mut remaining = power.unsigned_abs();
            let mut result = CandidateFractionSize::CONSTANT;
            while remaining != 0 {
                if remaining & 1 != 0 {
                    result = result.multiply(factor)?;
                }
                remaining >>= 1;
                if remaining != 0 {
                    factor = factor.multiply(factor)?;
                }
            }
            Some(result)
        }
        Expr::Const(_) | Expr::Function(_, _) => None,
    }
}

/// Route only the exact rational grammar. Unsupported closed forms decline
/// without evaluation; size failures remain errors even below those forms.
#[pyfunction]
pub fn rational_cancel_source_native(source: &PyExpr) -> PyResult<(bool, Vec<String>)> {
    let mut pending = vec![(&source.inner, 0usize)];
    let mut visited = 0usize;
    let mut name_bytes = 0usize;
    let mut eligible = true;
    let mut functions = BTreeSet::new();
    while let Some((node, depth)) = pending.pop() {
        visited += 1;
        if visited > 4096 || depth > 64 {
            return Err(PyValueError::new_err(
                "rational cancellation source exceeds its node/depth envelope",
            ));
        }
        let children = match node {
            Expr::Integer(value) => {
                if value.bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "rational cancellation integer exceeds 4096 bits",
                    ));
                }
                None
            }
            Expr::Rational(value) => {
                if value.numer().bits() > 4096 || value.denom().bits() > 4096 {
                    return Err(PyValueError::new_err(
                        "rational cancellation coefficient exceeds 4096 bits",
                    ));
                }
                None
            }
            Expr::Sym(symbol) => {
                name_bytes += symbol.name.len();
                None
            }
            Expr::Const(_) => {
                eligible = false;
                None
            }
            Expr::Function(name, children) => {
                name_bytes += name.len();
                if name_bytes > 65_536 {
                    return Err(PyValueError::new_err(
                        "rational cancellation source names exceed 65536 bytes",
                    ));
                }
                functions.insert(name.clone());
                eligible = false;
                Some(children)
            }
            Expr::Add(children) | Expr::Mul(children) => Some(children),
            Expr::Pow(base, exponent) => {
                if let Expr::Integer(power) = exponent.as_ref() {
                    if power
                        .to_i64()
                        .is_none_or(|value| !(-256..=256).contains(&value))
                    {
                        return Err(PyValueError::new_err(
                            "rational cancellation powers exceed the bound of 256",
                        ));
                    }
                } else {
                    eligible = false;
                }
                if 2 > 4096 - visited - pending.len() {
                    return Err(PyValueError::new_err(
                        "rational cancellation source exceeds its node envelope",
                    ));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
                None
            }
        };
        if name_bytes > 65_536 {
            return Err(PyValueError::new_err(
                "rational cancellation source names exceed 65536 bytes",
            ));
        }
        if let Some(children) = children {
            if children.len() > 4096 - visited - pending.len() {
                return Err(PyValueError::new_err(
                    "rational cancellation source exceeds its node envelope",
                ));
            }
            pending.extend(children.iter().map(|child| (child, depth + 1)));
        }
    }
    Ok((eligible, functions.into_iter().collect()))
}

/// Bind retained surface children to the borrowed native tree before routing
/// either a checked operation or an unchanged non-rational expression.
#[pyfunction]
pub fn rational_cancel_arguments_native(
    source: &PyExpr,
    operator: &str,
    children: &Bound<'_, PyList>,
) -> PyResult<()> {
    rational_cancel_source_native(source)?;
    if children.len() > 4096 {
        return Err(PyValueError::new_err(
            "rational cancellation has too many retained children",
        ));
    }
    let children = children
        .iter()
        .map(|value| value.extract::<PyRef<'_, PyExpr>>())
        .collect::<PyResult<Vec<_>>>()?;
    for child in &children {
        rational_cancel_source_native(child)?;
    }
    let equal_children = |expected: &[Expr]| {
        expected.len() == children.len()
            && expected
                .iter()
                .zip(&children)
                .all(|(left, right)| left == &right.inner)
    };
    let matches = match (&source.inner, operator) {
        (Expr::Add(expected), "add") | (Expr::Mul(expected), "mul") => equal_children(expected),
        (Expr::Pow(base, exponent), "pow") => {
            children.len() == 2
                && base.as_ref() == &children[0].inner
                && exponent.as_ref() == &children[1].inner
        }
        (Expr::Function(name, expected), operator) => {
            let expected_name = operator.strip_prefix("function:");
            (expected_name == Some(name.as_str())
                || (expected_name == Some("log") && name == "ln")
                || (expected_name == Some("Abs") && name == "abs"))
                && equal_children(expected)
        }
        _ => false,
    };
    if !matches {
        return Err(PyValueError::new_err(
            "rational cancellation source does not match its retained surface arguments",
        ));
    }
    Ok(())
}

fn bound_coefficient(value: BigRational) -> PyResult<BigRational> {
    if value.numer().bits() > 4096 || value.denom().bits() > 4096 {
        return Err(PyValueError::new_err(
            "rational cancellation exceeds 4096 coefficient bits",
        ));
    }
    Ok(value)
}

/// Apply the same positive rational scale to both members of the accepted
/// pair. Normalizing them independently would change the rational function.
fn primitive_pair(
    numerator: &mut MultivariatePoly,
    denominator: &mut MultivariatePoly,
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<()> {
    let mut common_numerator = BigInt::from(0);
    let mut common_denominator = BigInt::from(1);
    for coefficient in numerator
        .terms()
        .values()
        .chain(denominator.terms().values())
    {
        common_numerator =
            fsym_core::arith::metered_gcd(&common_numerator, coefficient.numer(), generator)
                .map_err(to_value_error)?;
        common_denominator =
            polynomial_integer_lcm(&common_denominator, coefficient.denom(), generator)?;
    }
    let content = BigRational::metered_new(&common_numerator, &common_denominator, generator)
        .map_err(to_value_error)?;
    for polynomial in [numerator, denominator] {
        for coefficient in polynomial.terms.values_mut() {
            let scaled = bound_coefficient(
                coefficient
                    .metered_div(&content, generator)
                    .map_err(to_value_error)?,
            )?;
            let recovered = scaled
                .metered_mul(&content, verifier)
                .map_err(to_value_error)?;
            charge(
                verifier,
                1 + coefficient.numer().limb_count() as usize
                    + coefficient.denom().limb_count() as usize,
                0,
            )?;
            if !scaled.is_integer() || recovered != *coefficient {
                return Err(PyValueError::new_err(
                    "rational cancellation coefficient normalization was not exact",
                ));
            }
            *coefficient = scaled;
        }
    }
    Ok(())
}

fn checked_lift(polynomial: &MultivariatePoly, verifier: &mut Budget) -> PyResult<Expr> {
    // Allocation was prepaid for both complete outputs before either copy.
    let expression = polynomial.to_expr().map_err(to_value_error)?;
    let recovered = bounded_polynomial_from_expr(&expression, polynomial.generators(), verifier)
        .map_err(to_value_error)?;
    let work = polynomial
        .terms()
        .iter()
        .map(|(powers, coefficient)| {
            powers.len()
                + coefficient.numer().limb_count() as usize
                + coefficient.denom().limb_count() as usize
                + 1
        })
        .sum();
    charge(verifier, work, 0)?;
    if recovered.terms() != polynomial.terms() {
        return Err(PyValueError::new_err(
            "rational cancellation output changed an exact coefficient",
        ));
    }
    Ok(expression)
}

/// Cancel one exact original arithmetic tree. A retained Poly supplies its
/// original ring and optional coefficient declaration; its expression must be
/// a polynomial in that ring before it can take the public identity return.
#[pyfunction]
#[pyo3(signature = (source, vars, poly_domain=None))]
pub fn rational_cancel_native(
    source: &PyExpr,
    vars: &Bound<'_, PyList>,
    poly_domain: Option<&str>,
) -> PyResult<PyExpr> {
    if poly_domain.is_some_and(|domain| !matches!(domain, "ZZ" | "QQ" | "auto")) {
        return Err(PyValueError::new_err(
            "rational cancellation Poly domain must be ZZ, QQ, or auto",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variables = symbols(vars, &mut generator)?;
    if let Some(domain) = poly_domain {
        let polynomial = bounded_polynomial_from_expr(&source.inner, &variables, &mut generator)
            .map_err(to_value_error)?;
        charge(&mut generator, polynomial.terms().len(), 0)?;
        if domain == "ZZ"
            && polynomial
                .terms()
                .values()
                .any(|coefficient| !coefficient.is_integer())
        {
            return Err(PyValueError::new_err(
                "noninteger coefficients cannot be represented in ZZ",
            ));
        }
    }
    let accepted =
        checked_rational_cancel(&source.inner, &variables, &mut generator, &mut verifier)
            .map_err(to_value_error)?;
    let (mut numerator, mut denominator, original) = accepted.into_parts();
    primitive_pair(
        &mut numerator,
        &mut denominator,
        &mut generator,
        &mut verifier,
    )?;

    let mut nodes = 4 + 4 * variables.len();
    for polynomial in [&numerator, &denominator] {
        nodes += 1;
        for powers in polynomial.terms().keys() {
            nodes += 2 + 4 * powers.iter().filter(|&&power| power != 0).count();
            if nodes > 4096 {
                return Err(PyValueError::new_err(
                    "rational cancellation exceeds 4096 output nodes",
                ));
            }
        }
    }
    let ring_bytes: usize = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
    charge(&mut generator, nodes, nodes * (2048 + ring_bytes))?;
    let numerator_expression = checked_lift(&numerator, &mut verifier)?;
    let denominator_expression = checked_lift(&denominator, &mut verifier)?;
    let expression = if numerator_expression.is_zero() || denominator_expression.is_one() {
        numerator_expression
    } else {
        let mut factors = match numerator_expression {
            Expr::Mul(factors) => factors,
            value if value.is_one() => Vec::new(),
            value => vec![value],
        };
        // A monomial denominator is presented as individual inverse factors,
        // matching the public arithmetic convention without factorization.
        let denominator_factors = match denominator_expression {
            Expr::Mul(factors) => factors,
            value => vec![value],
        };
        for factor in denominator_factors {
            factors.push(Expr::Pow(Arc::new(factor), Arc::new(Expr::from_i64(-1))));
        }
        if factors.len() == 1 {
            factors
                .pop()
                .ok_or_else(|| PyValueError::new_err("missing rational cancellation factor"))?
        } else {
            Expr::Mul(factors)
        }
    };
    // Check the complete assembled expression as well as its two components.
    // These output guards do not replace the original source's domain receipt.
    let lifted = metered_rational_source(&expression, &variables, &mut generator, &mut verifier)
        .map_err(to_value_error)?;
    charge(
        &mut verifier,
        2 * ring_bytes
            + (numerator.terms().len() + denominator.terms().len()) * (variables.len() + 129),
        0,
    )?;
    if lifted.fraction().numerator != numerator || lifted.fraction().denominator != denominator {
        return Err(PyValueError::new_err(
            "rational cancellation assembly changed its checked fraction",
        ));
    }
    charge(&mut generator, 1, 1)?;
    charge(&mut verifier, 1, 1)?;
    // The immutable original source and its guards outlive every output check.
    drop(original);
    Ok(PyExpr::from_expr(expression))
}
