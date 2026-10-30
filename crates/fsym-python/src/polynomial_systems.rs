//! Shared-budget dispatch for exact polynomial solution sets.
//!
//! Affine matrices, unions of polynomial graphs, finite tuples and real
//! univariate representations remain distinct results. Every attempt uses the
//! same generator/verifier meters. Algebraic data reaches a complex-domain
//! caller only after the complete root count proves that every root is real.

use crate::expr::PyExpr;
use crate::{extract_symbol_arg, rational_to_expr, to_value_error};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_matrices::{Matrix, RationalLinearSolution};
use fsym_polys::{MultivariatePoly, bounded_polynomial_from_expr};
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use fsym_solvers::SolverError;
use fsym_solvers::polynomial_components::{
    PolynomialComponentsOutcome, metered_polynomial_components_system,
};
use fsym_solvers::real_rur::metered_solve_real_rur_system;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Immutable bridge data constructed only by the checked dispatcher.
/// `finite` and `family` use solutions; `real_shape` and `real_rur` use ascending
/// rational coefficient vectors and the checked distinct-root multiplicities.
/// In `real_rur` every coordinate is a polynomial in a bound separator, which
/// need not equal any original solve variable.
#[pyclass(name = "PolynomialSystemResult", module = "fsym_python", frozen)]
pub struct PyPolynomialSystemResult {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    solutions: Vec<Vec<PyExpr>>,
    #[pyo3(get)]
    eliminant: Vec<PyExpr>,
    #[pyo3(get)]
    coordinates: Vec<Vec<PyExpr>>,
    #[pyo3(get)]
    multiplicities: Vec<usize>,
}

fn rows(kind: &str, values: Vec<Vec<Expr>>) -> PyPolynomialSystemResult {
    PyPolynomialSystemResult {
        kind: kind.into(),
        solutions: values
            .into_iter()
            .map(|row| row.into_iter().map(PyExpr::from_expr).collect())
            .collect(),
        eliminant: Vec::new(),
        coordinates: Vec::new(),
        multiplicities: Vec::new(),
    }
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0))
}

fn charge(meter: &mut Budget, cells: usize, bytes: usize) -> PyResult<()> {
    meter
        .try_charge_batch(&[
            (Dimension::ComputeSteps, cells.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(to_value_error)?;
    Ok(())
}

fn checked_affine(
    polys: &[MultivariatePoly],
    vars: &[Symbol],
    generator: &mut Budget,
    verifier: &mut Budget,
) -> PyResult<PyPolynomialSystemResult> {
    // The public empty-input convention is distinct from a present zero row.
    if polys.is_empty() {
        return Ok(rows("finite", Vec::new()));
    }
    let width = vars.len();
    charge(
        generator,
        polys.len() * (width + 1),
        polys.len() * (width + 1) * 2048,
    )?;
    let mut coefficients = Vec::with_capacity(polys.len() * width);
    let mut rhs = Vec::with_capacity(polys.len());
    let zero = BigRational::from(0_i64);
    for poly in polys {
        let mut row = vec![zero.clone(); width];
        let mut constant = zero.clone();
        for (powers, coefficient) in poly.terms() {
            if let Some(column) = powers.iter().position(|&power| power == 1) {
                row[column] = coefficient.clone();
            } else {
                constant = -coefficient.clone();
            }
        }
        coefficients.extend(row.into_iter().map(rational_to_expr));
        rhs.push(rational_to_expr(constant));
    }
    let matrix = Matrix::new(polys.len(), width, coefficients).map_err(to_value_error)?;
    let rhs = Matrix::new(polys.len(), 1, rhs).map_err(to_value_error)?;
    let checked = matrix
        .solve_affine_with_meters(&rhs, generator, verifier)
        .map_err(to_value_error)?;
    let values = match checked.solution() {
        RationalLinearSolution::Inconsistent { .. } => return Ok(rows("finite", Vec::new())),
        RationalLinearSolution::Unique { values } => {
            // The checked matrix lane admits up to 8192-bit numerators and
            // denominators. Reserve the owned exact copies before lifting.
            charge(generator, width, width * 4096)?;
            values.iter().cloned().map(rational_to_expr).collect()
        }
        RationalLinearSolution::Affine {
            particular,
            basis,
            free_columns,
        } => {
            let directions = basis.len() + 1;
            let ring_bytes: usize = vars.iter().map(|variable| variable.name.len() + 128).sum();
            // Reserve the full typed-name payload and exact coefficients,
            // including copies made while canonicalizing successive sums.
            charge(
                generator,
                width * directions * directions,
                width * directions * directions * (4096 + ring_bytes),
            )?;
            particular
                .iter()
                .enumerate()
                .map(|(column, value)| {
                    if free_columns.contains(&column) {
                        return Expr::Sym(vars[column].clone());
                    }
                    let mut coordinate = rational_to_expr(value.clone());
                    for (direction, &free) in basis.iter().zip(free_columns) {
                        if direction[column] != zero {
                            coordinate = coordinate
                                + rational_to_expr(direction[column].clone())
                                    * Expr::Sym(vars[free].clone());
                        }
                    }
                    coordinate
                })
                .collect()
        }
    };
    charge(generator, width, width * 128)?;
    charge(verifier, 1, 1)?;
    Ok(rows("family", vec![values]))
}

/// Complete supported solution sets, preserving separate cumulative budgets
/// across recognition, alternate variable orders and the exact-root fallback.
#[pyfunction]
#[pyo3(signature = (eqs, vars, allow_families=false))]
pub fn polynomial_system_solutions_native(
    eqs: Vec<PyExpr>,
    vars: Vec<Bound<'_, PyAny>>,
    allow_families: bool,
) -> PyResult<PyPolynomialSystemResult> {
    if vars.is_empty() || vars.len() > 8 || eqs.len() > 64 {
        return Err(PyValueError::new_err(
            "polynomial systems require 1..8 variables and at most 64 equations",
        ));
    }
    let variables = vars
        .iter()
        .map(extract_symbol_arg)
        .collect::<PyResult<Vec<_>>>()?;
    if variables
        .iter()
        .enumerate()
        .any(|(index, variable)| variables[..index].contains(variable))
    {
        return Err(PyValueError::new_err("duplicate solve variables"));
    }
    let expressions: Vec<_> = eqs.into_iter().map(|expression| expression.inner).collect();
    let mut generator = budget();
    let mut verifier = budget();
    // Bind and bound the ring even when the equation list is empty.
    bounded_polynomial_from_expr(&Expr::from_i64(0), &variables, &mut generator)
        .map_err(to_value_error)?;
    // Complete original admission precedes every empty/affine/family decision.
    let polys = expressions
        .iter()
        .map(|expression| {
            bounded_polynomial_from_expr(expression, &variables, &mut generator)
                .map_err(to_value_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    if allow_families
        && polys
            .iter()
            .all(|poly| poly.total_degree().is_none_or(|degree| degree <= 1))
    {
        return checked_affine(&polys, &variables, &mut generator, &mut verifier);
    }
    if allow_families {
        match metered_polynomial_components_system(
            &expressions,
            &variables,
            &mut generator,
            &mut verifier,
        )
        .map_err(to_value_error)?
        {
            PolynomialComponentsOutcome::Empty => return Ok(rows("finite", Vec::new())),
            PolynomialComponentsOutcome::Families(families) => {
                // Transfer already checked and charged coordinates without
                // duplicating exact coefficients or typed symbol names.
                return Ok(rows("family", families.into_values()));
            }
            PolynomialComponentsOutcome::NotApplicable => {}
        }
    }
    let explicit_error = match fsym_solvers::polysys::metered_solve_polynomial_system(
        &expressions,
        &variables,
        &mut generator,
        &mut verifier,
    ) {
        Ok(solutions) => {
            return Ok(rows(
                "finite",
                fsym_solvers::polysys::sort_solutions(solutions),
            ));
        }
        // Only an unsupported exact root formula triggers the algebraic route.
        // Resource, cancellation and undecided-verification errors propagate.
        Err(error @ SolverError::UnsupportedDegree(_)) => error,
        Err(error) => return Err(to_value_error(error)),
    };
    let Some(representation) =
        metered_solve_real_rur_system(&expressions, &variables, &mut generator, &mut verifier)
            .map_err(to_value_error)?
    else {
        return Err(to_value_error(explicit_error));
    };
    if !representation.all_complex_roots_real() {
        return Err(to_value_error(explicit_error));
    }
    let cells = representation.eliminant().len()
        + representation
            .coordinates()
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    charge(&mut generator, cells, cells * 2048)?;
    charge(&mut verifier, 1, 1)?;
    let zero = BigRational::from(0_i64);
    let original_shape = representation.shear().iter().all(|value| value == &zero);
    Ok(PyPolynomialSystemResult {
        kind: if original_shape {
            "real_shape"
        } else {
            "real_rur"
        }
        .into(),
        solutions: Vec::new(),
        eliminant: representation
            .eliminant()
            .iter()
            .cloned()
            .map(rational_to_expr)
            .map(PyExpr::from_expr)
            .collect(),
        coordinates: representation
            .coordinates()
            .iter()
            .map(|coordinate| {
                coordinate
                    .iter()
                    .cloned()
                    .map(rational_to_expr)
                    .map(PyExpr::from_expr)
                    .collect()
            })
            .collect(),
        multiplicities: representation
            .roots()
            .intervals()
            .iter()
            .map(|root| root.multiplicity)
            .collect(),
    })
}
