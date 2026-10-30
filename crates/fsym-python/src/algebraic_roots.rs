//! Typed exact real-root isolation and decimal refinement bridge.

use crate::expr::PyExpr;
use fsym_core::real_roots::{
    MAX_REAL_ROOT_DEGREE, MAX_REAL_ROOT_INPUT_BITS, identify_real_root_factors, isolate_real_roots,
    real_root_decimal,
};
use fsym_core::{BigRational, Expr};
use fsym_runtime::{Budget, BudgetLimits};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn coefficients(values: &Bound<'_, PyList>) -> PyResult<Vec<BigRational>> {
    if values.is_empty() || values.len() > MAX_REAL_ROOT_DEGREE + 1 {
        return Err(PyValueError::new_err(
            "real-root coefficient count must be between 1 and 65",
        ));
    }
    let mut result = Vec::with_capacity(values.len());
    for value in values.iter() {
        let expression = value.extract::<PyRef<'_, PyExpr>>()?;
        let q = match &expression.inner {
            Expr::Integer(n) if n.bits() <= MAX_REAL_ROOT_INPUT_BITS => {
                BigRational::from_integer(n.clone())
            }
            Expr::Rational(q)
                if q.numer().bits().max(q.denom().bits()) <= MAX_REAL_ROOT_INPUT_BITS =>
            {
                q.clone()
            }
            _ => {
                return Err(PyValueError::new_err(
                    "real-root coefficients must be literal exact rationals of at most 4096 bits",
                ));
            }
        };
        result.push(q);
    }
    Ok(result)
}

fn budget() -> Budget {
    // Separate bounded synchronous meters; no protected runtime reservation
    // or forced cancellation through arbitrary Python callbacks is claimed.
    Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0))
}

/// A checked algebraic value retains the original typed expression, defining
/// polynomial, root index, domain obligations and accepted exact enclosure.
/// Python cannot construct or modify the underlying native receipt.
#[pyclass(name = "AlgebraicValue", module = "fsym_python", frozen)]
pub struct PyAlgebraicValue {
    inner: fsym_solvers::real_equation::VerifiedAlgebraicValue,
}

#[pymethods]
impl PyAlgebraicValue {
    #[getter]
    fn decimal(&self) -> &str {
        self.inner.decimal()
    }

    #[getter]
    fn lower(&self) -> PyExpr {
        PyExpr::from_expr(crate::rational_to_expr(self.inner.lower().clone()))
    }

    #[getter]
    fn upper(&self) -> PyExpr {
        PyExpr::from_expr(crate::rational_to_expr(self.inner.upper().clone()))
    }

    #[getter]
    fn index(&self) -> usize {
        self.inner.index()
    }

    #[getter]
    fn precision(&self) -> u32 {
        self.inner.precision()
    }

    #[getter]
    fn expression(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.expression().clone())
    }
}

/// A checked value of an expression in several indexed real roots. The
/// underlying receipt binds the source, every chosen embedding and the final
/// rational enclosure; Python cannot construct a receipt from candidate data.
#[pyclass(name = "AlgebraicExpressionValue", module = "fsym_python", frozen)]
pub struct PyAlgebraicExpressionValue {
    inner: fsym_solvers::algebraic_expression::VerifiedAlgebraicExpression,
}

#[pymethods]
impl PyAlgebraicExpressionValue {
    #[getter]
    fn decimal(&self) -> &str {
        self.inner.decimal()
    }

    #[getter]
    fn lower(&self) -> PyExpr {
        PyExpr::from_expr(crate::rational_to_expr(self.inner.lower().clone()))
    }

    #[getter]
    fn upper(&self) -> PyExpr {
        PyExpr::from_expr(crate::rational_to_expr(self.inner.upper().clone()))
    }

    #[getter]
    fn precision(&self) -> u32 {
        self.inner.precision()
    }

    #[getter]
    fn expression(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.expression().clone())
    }

    #[getter]
    fn binding_count(&self) -> usize {
        self.inner.bindings().len()
    }
}

/// Resolve a bounded tuple of exact real embeddings, then evaluate the held
/// rational source with one cumulative pair of generator/verifier budgets.
#[pyfunction]
#[pyo3(signature = (expression, variables, values, indices, digits=15))]
pub fn evaluate_real_algebraic_expression_native(
    expression: &PyExpr,
    variables: &Bound<'_, PyList>,
    values: &Bound<'_, PyList>,
    indices: &Bound<'_, PyList>,
    digits: u32,
) -> PyResult<Option<PyAlgebraicExpressionValue>> {
    use fsym_solvers::algebraic_expression::{
        RealAlgebraicBinding, metered_evaluate_real_algebraic_expression,
    };

    if variables.is_empty()
        || variables.len() > 8
        || variables.len() != values.len()
        || variables.len() != indices.len()
        || !(1..=512).contains(&digits)
    {
        return Err(PyValueError::new_err(
            "algebraic evaluation requires 1..8 matched root bindings and 1..512 digits",
        ));
    }
    // List lengths and coefficient payloads are bounded before copying. The
    // native admission checks typed uniqueness, all root indices, product
    // degree and the complete original source before publishing any value.
    let mut bindings = Vec::with_capacity(variables.len());
    for ((variable, values), index) in variables.iter().zip(values.iter()).zip(indices.iter()) {
        bindings.push(RealAlgebraicBinding {
            variable: crate::extract_symbol_arg(&variable)?,
            coefficients: coefficients(values.cast::<PyList>()?)?,
            index: index.extract::<usize>()?,
        });
    }
    let value = metered_evaluate_real_algebraic_expression(
        &expression.inner,
        &bindings,
        digits,
        &mut budget(),
        &mut budget(),
    )
    .map_err(crate::to_value_error)?;
    Ok(value.map(|inner| PyAlgebraicExpressionValue { inner }))
}

/// Correctly rounded rational arithmetic on one exact real algebraic root.
/// Unsupported source grammar is distinct from invalid domains, exhausted
/// resources and failed verification; none of those triggers a float fallback.
#[pyfunction]
#[pyo3(signature = (expression, variable, values, index, digits=15))]
pub fn evaluate_real_algebraic_native(
    expression: &PyExpr,
    variable: &Bound<'_, PyAny>,
    values: &Bound<'_, PyList>,
    index: usize,
    digits: u32,
) -> PyResult<Option<PyAlgebraicValue>> {
    let variable = crate::extract_symbol_arg(variable)?;
    let coefficients = coefficients(values)?;
    let value = fsym_solvers::real_equation::metered_evaluate_real_algebraic(
        &expression.inner,
        &variable,
        &coefficients,
        index,
        digits,
        &mut budget(),
        &mut budget(),
    )
    .map_err(crate::to_value_error)?;
    Ok(value.map(|inner| PyAlgebraicValue { inner }))
}

/// Compare two exact indexed real roots only after independent acceptance of
/// both complete isolation certificates and their equality/order witness.
#[pyfunction]
pub fn compare_real_roots_native(
    left_values: &Bound<'_, PyList>,
    left_index: usize,
    right_values: &Bound<'_, PyList>,
    right_index: usize,
) -> PyResult<i8> {
    let left = coefficients(left_values)?;
    let right = coefficients(right_values)?;
    let comparison = fsym_core::real_roots::compare_real_roots(
        &left,
        left_index,
        &right,
        right_index,
        &mut budget(),
        &mut budget(),
    )
    .map_err(crate::to_value_error)?;
    Ok(match comparison.ordering() {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    })
}

/// Complete finite real solution roots, or their complement in the real line.
/// The accepted native receipt retains the original typed source and variable.
#[pyfunction]
pub fn solve_real_rational_native(
    expression: &PyExpr,
    variable: &Bound<'_, PyAny>,
) -> PyResult<Option<(bool, PyExpr, usize)>> {
    let symbol = crate::extract_symbol_arg(variable)?;
    let result = fsym_solvers::metered_solve_real_equation(
        &expression.inner,
        &symbol,
        &mut budget(),
        &mut budget(),
    )
    .map_err(crate::to_value_error)?;
    Ok(result.map(|result| {
        (
            result.is_complement(),
            PyExpr::from_expr(result.polynomial().to_expr()),
            result.roots().intervals().len(),
        )
    }))
}

/// Extract coefficients only after the existing bounded converter has
/// validated the complete original tree, including all held children.
#[pyfunction]
pub fn real_root_coefficients_native(
    expression: &PyExpr,
    variable: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyExpr>> {
    let symbol = crate::extract_symbol_arg(variable)?;
    let polynomial =
        fsym_polys::bounded_polynomial_from_expr(&expression.inner, &[symbol], &mut budget())
            .map_err(crate::to_value_error)?;
    if polynomial.is_zero() {
        return Err(PyValueError::new_err(
            "the zero polynomial does not have a finite root set",
        ));
    }
    let degree = polynomial.degree_in(0) as usize;
    if degree > MAX_REAL_ROOT_DEGREE {
        return Err(PyValueError::new_err(
            "exact real-root isolation supports degree at most 64",
        ));
    }
    let mut result = vec![PyExpr::from_expr(Expr::from_i64(0)); degree + 1];
    for (powers, coefficient) in polynomial.terms() {
        result[powers[0] as usize] = PyExpr::from_expr(Expr::Rational(coefficient.clone()));
    }
    Ok(result)
}

#[pyfunction]
pub fn real_roots_isolate_native(
    values: &Bound<'_, PyList>,
) -> PyResult<Vec<(PyExpr, PyExpr, usize)>> {
    let p = coefficients(values)?;
    let roots =
        isolate_real_roots(&p, &mut budget(), &mut budget()).map_err(crate::to_value_error)?;
    Ok(roots
        .intervals()
        .iter()
        .map(|root| {
            (
                PyExpr::from_expr(Expr::Rational(root.lower.clone())),
                PyExpr::from_expr(Expr::Rational(root.upper.clone())),
                root.multiplicity,
            )
        })
        .collect())
}

#[pyfunction]
pub fn indexed_real_roots_native(
    values: &Bound<'_, PyList>,
    factors: &Bound<'_, PyList>,
    multiplicities: &Bound<'_, PyList>,
) -> PyResult<Vec<(usize, usize, usize)>> {
    if factors.len() > MAX_REAL_ROOT_DEGREE || factors.len() != multiplicities.len() {
        return Err(PyValueError::new_err(
            "invalid bounded real-root factor list",
        ));
    }
    let p = coefficients(values)?;
    let mut factor_values = Vec::with_capacity(factors.len());
    let mut factor_multiplicities = Vec::with_capacity(factors.len());
    for (factor, multiplicity) in factors.iter().zip(multiplicities.iter()) {
        let multiplicity = multiplicity.extract::<usize>()?;
        if multiplicity == 0 || multiplicity > MAX_REAL_ROOT_DEGREE {
            return Err(PyValueError::new_err(
                "invalid real-root factor multiplicity",
            ));
        }
        factor_values.push(coefficients(factor.cast::<PyList>()?)?);
        factor_multiplicities.push(multiplicity);
    }
    let roots =
        isolate_real_roots(&p, &mut budget(), &mut budget()).map_err(crate::to_value_error)?;
    let indices = identify_real_root_factors(
        &roots,
        &factor_values,
        &factor_multiplicities,
        &mut budget(),
    )
    .map_err(crate::to_value_error)?;
    Ok(indices
        .into_iter()
        .zip(roots.intervals())
        .map(|((factor, local), root)| (factor, local, root.multiplicity))
        .collect())
}

#[pyfunction]
pub fn real_root_decimal_native(
    values: &Bound<'_, PyList>,
    index: usize,
    digits: u32,
) -> PyResult<String> {
    let p = coefficients(values)?;
    real_root_decimal(&p, index, digits, &mut budget(), &mut budget())
        .map_err(crate::to_value_error)
}
