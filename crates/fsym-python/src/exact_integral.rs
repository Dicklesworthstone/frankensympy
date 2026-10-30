//! Source-bound exact real rational integration at the typed Python boundary.
//!
//! Original source fractions and every held inverse guard are independently
//! reconstructed before the elementary integral checker is invoked. Each
//! guard receives its own exact Sturm domain check, so cancellation never
//! removes a hole. These are bounded in-process receipts, not release/profile
//! certification or a claim about principal-value integrals.

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::to_value_error;
use fsym_calculus::elementary_rational_definite::{
    ElementaryRationalDefiniteIntegral, ElementaryRationalDefiniteIntegralError,
    metered_elementary_rational_definite_integral,
};
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, UnivariatePoly, VerifiedRationalSource, metered_rational_source,
};
use fsym_proof_kernel::rational_definite::{
    MAX_DEFINITE_COEFFICIENTS, RationalIntegralEndpoint, verify_rational_integral_domain,
};
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;

fn budget() -> Budget {
    let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 200_000_000;
    Budget::new(limits)
}

fn charge(meter: &mut Budget, steps: usize, bytes: usize) -> PyResult<()> {
    meter
        .try_charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(to_value_error)?;
    Ok(())
}

fn expression<'py>(value: &Bound<'py, PyAny>) -> PyResult<PyRef<'py, PyExpr>> {
    if !value.is_exact_instance(&value.py().get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "exact rational integration requires an exact native expression",
        ));
    }
    Ok(value.extract::<PyRef<'py, PyExpr>>()?)
}

fn endpoint(value: &Expr, meter: &mut Budget) -> PyResult<RationalIntegralEndpoint> {
    charge(meter, 1, 1152)?;
    Ok(match value {
        Expr::Integer(integer) if integer.bits() <= 4096 => {
            RationalIntegralEndpoint::Finite(BigRational::from_integer(integer.clone()))
        }
        Expr::Rational(rational)
            if rational.numer().bits().max(rational.denom().bits()) <= 4096 =>
        {
            RationalIntegralEndpoint::Finite(rational.clone())
        }
        Expr::Const(Constant::Infinity) => RationalIntegralEndpoint::PositiveInfinity,
        Expr::Const(Constant::NegativeInfinity) => RationalIntegralEndpoint::NegativeInfinity,
        _ => {
            return Err(PyValueError::new_err(
                "exact integral endpoints require rational literals of at most 4096 bits or real infinities",
            ));
        }
    })
}

/// Exact sparse-to-dense transfer in the single original typed variable.
/// No expression printer, parser or generic expansion participates.
fn polynomial(
    source: &MultivariatePoly,
    variable: &Symbol,
    meter: &mut Budget,
) -> PyResult<UnivariatePoly> {
    if source.generators() != std::slice::from_ref(variable) {
        return Err(PyValueError::new_err(
            "exact integral polynomial ring mismatch",
        ));
    }
    let mut degree = 0usize;
    for (powers, coefficient) in source.terms() {
        charge(meter, 1, 1)?;
        if powers.len() != 1 || coefficient.numer().bits().max(coefficient.denom().bits()) > 4096 {
            return Err(PyValueError::new_err(
                "invalid exact integral polynomial coefficient",
            ));
        }
        degree = degree.max(powers[0] as usize);
    }
    if degree >= MAX_DEFINITE_COEFFICIENTS {
        return Err(PyValueError::new_err(
            "exact integral polynomial degree exceeds 128",
        ));
    }
    charge(
        meter,
        degree + 1,
        (degree + 1) * 1152 + variable.name.len() + 128,
    )?;
    let mut coefficients = vec![BigRational::from(0_i64); degree + 1];
    for (powers, coefficient) in source.terms() {
        coefficients[powers[0] as usize] = coefficient.clone();
    }
    Ok(UnivariatePoly::new(variable.clone(), coefficients))
}

fn endpoint_expression(value: &RationalIntegralEndpoint) -> PyExpr {
    PyExpr::from_expr(match value {
        RationalIntegralEndpoint::Finite(value) => crate::rational_to_expr(value.clone()),
        RationalIntegralEndpoint::NegativeInfinity => Expr::Const(Constant::NegativeInfinity),
        RationalIntegralEndpoint::PositiveInfinity => Expr::Const(Constant::Infinity),
    })
}

#[pyclass(name = "ExactRationalIntegralReceipt", module = "fsym_python", frozen)]
pub struct PyExactRationalIntegralReceipt {
    source: VerifiedRationalSource,
    lower: RationalIntegralEndpoint,
    upper: RationalIntegralEndpoint,
    integral: ElementaryRationalDefiniteIntegral,
}

#[pymethods]
impl PyExactRationalIntegralReceipt {
    #[getter]
    fn integrand(&self) -> PyExpr {
        PyExpr::from_expr(self.source.source().clone())
    }

    #[getter]
    fn variable(&self) -> PyExpr {
        PyExpr::from_expr(Expr::Sym(self.source.variables()[0].clone()))
    }

    #[getter]
    fn interval(&self) -> (PyExpr, PyExpr) {
        (
            endpoint_expression(&self.lower),
            endpoint_expression(&self.upper),
        )
    }

    #[getter]
    fn value(&self) -> PyExpr {
        PyExpr::from_expr(self.integral.value().clone())
    }

    #[getter]
    fn original_guard_count(&self) -> usize {
        self.source.guards().len()
    }
}

/// Produce an immutable receipt only after original-source, domain,
/// derivative and endpoint checks all complete under two cumulative budgets.
#[pyfunction]
pub fn exact_rational_integral_native(
    source: &Bound<'_, PyAny>,
    variable: &PySymbolBinding,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
) -> PyResult<PyExactRationalIntegralReceipt> {
    let source = expression(source)?;
    let lower = expression(lower)?;
    let upper = expression(upper)?;
    let mut generator = budget();
    let mut verifier = budget();
    if variable.name.len() > 65_536 {
        return Err(PyValueError::new_err(
            "exact integral variable name exceeds 65536 bytes",
        ));
    }
    charge(&mut generator, 1, variable.name.len() + 128)?;
    let variable = variable.to_symbol();
    let lower = endpoint(&lower.inner, &mut generator)?;
    let upper = endpoint(&upper.inner, &mut generator)?;
    let source = metered_rational_source(
        &source.inner,
        std::slice::from_ref(&variable),
        &mut generator,
        &mut verifier,
    )
    .map_err(to_value_error)?;
    // The reduced denominator alone is insufficient for nested reciprocals,
    // cancelled factors, held zero products and outer zeroth powers.
    for guard in source.guards() {
        let guard = polynomial(guard, &variable, &mut verifier)?;
        verify_rational_integral_domain(&guard.coeffs, &lower, &upper, &mut verifier)
            .map_err(to_value_error)?;
    }
    let numerator = polynomial(&source.fraction().numerator, &variable, &mut generator)?;
    let denominator = polynomial(&source.fraction().denominator, &variable, &mut generator)?;
    let integral = metered_elementary_rational_definite_integral(
        &numerator,
        &denominator,
        &lower,
        &upper,
        &mut generator,
        &mut verifier,
    )
    .map_err(|error| match error {
        ElementaryRationalDefiniteIntegralError::UnsupportedPrimitive => {
            PyNotImplementedError::new_err(error.to_string())
        }
        other => to_value_error(other),
    })?;
    charge(
        &mut generator,
        1,
        std::mem::size_of::<PyExactRationalIntegralReceipt>(),
    )?;
    charge(&mut verifier, 1, 1)?;
    Ok(PyExactRationalIntegralReceipt {
        source,
        lower,
        upper,
        integral,
    })
}
