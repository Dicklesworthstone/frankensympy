//! Source-bound F_p[x] operations for the public Poly coefficient domain.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{
    PolyError, PrimeFieldArithmeticOperation as Op, PrimeFieldElementSourceError,
    PrimeFieldPolynomial, UnivariatePoly, checked_prime_field_arithmetic,
    checked_prime_field_element_source, checked_prime_field_factorization,
};
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use std::collections::BTreeMap;

pyo3::create_exception!(fsym_python, PrimeFieldCoefficientError, PyValueError);

type FactorParts = (PyExpr, Vec<(PyExpr, u32)>);

fn budget() -> Budget {
    let mut limits = BudgetLimits::uniform(128 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 20_000_000;
    Budget::new(limits)
}

fn charge(meter: &mut Budget, steps: usize, bytes: usize) -> PyResult<()> {
    meter.checkpoint().map_err(to_value_error)?;
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(to_value_error)
}

fn polynomial_error(error: PolyError) -> PyErr {
    match error {
        PolyError::DivisionByZero => PyZeroDivisionError::new_err(error.to_string()),
        _ => to_value_error(error),
    }
}

fn source_error(error: PrimeFieldElementSourceError) -> PyErr {
    match error {
        PrimeFieldElementSourceError::NonIntegralCoefficient => {
            PrimeFieldCoefficientError::new_err(error.to_string())
        }
        PrimeFieldElementSourceError::Polynomial(error) => polynomial_error(error),
    }
}

fn symbol(variable: &PySymbolBinding, meter: &mut Budget) -> PyResult<Symbol> {
    if variable.name.len() > 65_536 {
        return Err(PyValueError::new_err(
            "modular generator name exceeds 65536 bytes",
        ));
    }
    charge(
        meter,
        1,
        variable.name.len() + std::mem::size_of::<Symbol>(),
    )?;
    Ok(variable.to_symbol())
}

fn lift(
    coefficients: &[u64],
    variable: &Symbol,
    prime: u64,
    symmetric: bool,
    meter: &mut Budget,
) -> PyResult<PyExpr> {
    charge(
        meter,
        coefficients.len() + 1,
        (coefficients.len() + 1) * (512 + variable.name.len()),
    )?;
    let values = coefficients
        .iter()
        .map(|&coefficient| {
            BigRational::from(if symmetric && coefficient > prime / 2 {
                coefficient as i64 - prime as i64
            } else {
                coefficient as i64
            })
        })
        .collect();
    Ok(PyExpr::from_expr(
        UnivariatePoly::new(variable.clone(), values).to_expr(),
    ))
}

/// Infer one exact typed generator while borrowing all original nodes.
/// No free-symbol set or source clone precedes the fixed admission limits.
#[pyfunction]
#[pyo3(signature = (source, other=None))]
pub fn prime_field_polynomial_generator_native(
    source: &PyExpr,
    other: Option<&PyExpr>,
) -> PyResult<Option<PyExpr>> {
    fn visit<'a>(
        expression: &'a Expr,
        depth: usize,
        nodes: &mut usize,
        found: &mut Option<&'a Symbol>,
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
                charge(meter, variable.name.len() + 1, 1)?;
                if found.is_some_and(|existing| existing != variable) {
                    return Err(PyValueError::new_err(
                        "prime-field Poly requires one typed generator",
                    ));
                }
                *found = Some(variable);
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
    let (mut nodes, mut found) = (0, None);
    visit(&source.inner, 0, &mut nodes, &mut found, &mut meter)?;
    if let Some(other) = other {
        visit(&other.inner, 0, &mut nodes, &mut found, &mut meter)?;
    }
    let result = if let Some(variable) = found {
        charge(
            &mut meter,
            1,
            variable.name.len() + std::mem::size_of::<Expr>() + 128,
        )?;
        Some(PyExpr::from_expr(Expr::Sym(variable.clone())))
    } else {
        None
    };
    meter.checkpoint().map_err(to_value_error)?;
    Ok(result)
}

/// A single cumulative pair of ledgers covers both source admissions,
/// generation, independent verification, all output lifts and publication.
#[pyfunction]
#[pyo3(signature = (source, other, variable, prime, operation, parameter=0, symmetric=true))]
pub fn prime_field_polynomial_native(
    source: &PyExpr,
    other: Option<&PyExpr>,
    variable: &PySymbolBinding,
    prime: u64,
    operation: &str,
    parameter: u64,
    symmetric: bool,
) -> PyResult<Option<Vec<PyExpr>>> {
    let operation = match operation {
        "normalize" => Op::Normalize,
        "neg" => Op::Negate,
        "add" => Op::Add,
        "sub" => Op::Subtract,
        "mul" => Op::Multiply,
        "divrem" => Op::Divide,
        "monic" => Op::Monic,
        "gcd" => Op::Gcd,
        "gcdex" => Op::ExtendedGcd,
        "lcm" => Op::Lcm,
        "invert" => Op::Invert,
        "diff" => Op::Derivative(
            u32::try_from(parameter)
                .map_err(|_| PyValueError::new_err("derivative order exceeds u32"))?,
        ),
        "pow" => Op::Power(parameter),
        "compose" => Op::Compose,
        "resultant" => Op::Resultant,
        "discriminant" => Op::Discriminant,
        _ => {
            return Err(PyValueError::new_err(
                "unknown prime-field polynomial operation",
            ));
        }
    };
    let (mut generator, mut verifier) = (budget(), budget());
    let variable = symbol(variable, &mut generator)?;
    let left = checked_prime_field_element_source(
        &source.inner,
        &variable,
        prime,
        &mut generator,
        &mut verifier,
    )
    .map_err(source_error)?;
    let right = other
        .map(|other| {
            checked_prime_field_element_source(
                &other.inner,
                &variable,
                prime,
                &mut generator,
                &mut verifier,
            )
        })
        .transpose()
        .map_err(source_error)?;
    let accepted = checked_prime_field_arithmetic(
        operation,
        left.polynomial(),
        right.as_ref().map(|source| source.polynomial()),
        &mut generator,
        &mut verifier,
    )
    .map_err(polynomial_error)?;
    charge(
        &mut generator,
        accepted.results().len() + 1,
        accepted.results().len() * std::mem::size_of::<PyExpr>() + 64,
    )?;
    let mut values = Vec::with_capacity(accepted.results().len());
    for coefficients in accepted.results() {
        values.push(lift(
            coefficients,
            &variable,
            prime,
            symmetric,
            &mut generator,
        )?);
    }
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok(if operation == Op::Invert && values.is_empty() {
        None
    } else {
        Some(values)
    })
}

fn checked_product(
    left: &PrimeFieldPolynomial,
    right: &PrimeFieldPolynomial,
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<PrimeFieldPolynomial> {
    let accepted =
        checked_prime_field_arithmetic(Op::Multiply, left, Some(right), generator, verifier)
            .map_err(polynomial_error)?;
    let coefficients = accepted
        .into_results()
        .pop()
        .ok_or_else(|| PyValueError::new_err("checked finite-field product omitted its result"))?;
    PrimeFieldPolynomial::metered_new(left.variable(), left.prime(), &coefficients, generator)
        .map_err(polynomial_error)
}

/// Factorization, square-free layers and the radical all retain the original
/// field source and one cumulative budget across the complete factor family.
#[pyfunction]
#[pyo3(signature = (source, variable, prime, symmetric=true, mode="factor"))]
pub fn prime_field_polynomial_factor_native(
    source: &PyExpr,
    variable: &PySymbolBinding,
    prime: u64,
    symmetric: bool,
    mode: &str,
) -> PyResult<FactorParts> {
    if !matches!(mode, "factor" | "sqf" | "sqf_part") {
        return Err(PyValueError::new_err(
            "unknown prime-field factorization mode",
        ));
    }
    let (mut generator, mut verifier) = (budget(), budget());
    let variable = symbol(variable, &mut generator)?;
    let source = checked_prime_field_element_source(
        &source.inner,
        &variable,
        prime,
        &mut generator,
        &mut verifier,
    )
    .map_err(source_error)?;
    let accepted =
        checked_prime_field_factorization(source.polynomial(), &mut generator, &mut verifier)
            .map_err(polynomial_error)?;
    charge(
        &mut generator,
        accepted.factors().len() + 1,
        accepted.factors().len() * 512 + 256,
    )?;
    let mut factors = Vec::with_capacity(accepted.factors().len());
    if mode == "factor" {
        // The public factor order uses descending canonical residues, before
        // applying the signed/unsigned display policy. In particular the
        // multiplicity precedes the coefficient tuple in this ordering.
        let count = accepted.factors().len();
        charge(&mut generator, count * count * 65 + 1, count * 8 + 64)?;
        let mut ordered: Vec<_> = accepted.factors().iter().collect();
        ordered.sort_unstable_by(|left, right| {
            left.coefficients
                .len()
                .cmp(&right.coefficients.len())
                .then_with(|| left.multiplicity.cmp(&right.multiplicity))
                .then_with(|| {
                    left.coefficients
                        .iter()
                        .rev()
                        .cmp(right.coefficients.iter().rev())
                })
        });
        for factor in ordered {
            factors.push((
                lift(
                    &factor.coefficients,
                    &variable,
                    prime,
                    symmetric,
                    &mut generator,
                )?,
                factor.multiplicity,
            ));
        }
    } else {
        let mut groups: BTreeMap<u32, PrimeFieldPolynomial> = BTreeMap::new();
        for factor in accepted.factors() {
            let multiplicity = if mode == "sqf_part" {
                1
            } else {
                factor.multiplicity
            };
            let polynomial = PrimeFieldPolynomial::metered_new(
                &variable,
                prime,
                &factor.coefficients,
                &mut generator,
            )
            .map_err(polynomial_error)?;
            let polynomial = if let Some(previous) = groups.get(&multiplicity) {
                checked_product(previous, &polynomial, &mut generator, &mut verifier)?
            } else {
                polynomial
            };
            groups.insert(multiplicity, polynomial);
        }
        for (multiplicity, polynomial) in groups {
            factors.push((
                lift(
                    polynomial.coefficients(),
                    &variable,
                    prime,
                    symmetric,
                    &mut generator,
                )?,
                multiplicity,
            ));
        }
    }
    let unit = if mode == "sqf_part" {
        1
    } else {
        accepted.unit()
    };
    let unit = lift(&[unit], &variable, prime, symmetric, &mut generator)?;
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok((unit, factors))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "PrimeFieldCoefficientError",
        module.py().get_type::<PrimeFieldCoefficientError>(),
    )?;
    module.add_function(wrap_pyfunction!(
        prime_field_polynomial_generator_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(prime_field_polynomial_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        prime_field_polynomial_factor_native,
        module
    )?)?;
    Ok(())
}
