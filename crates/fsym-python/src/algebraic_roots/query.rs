//! Source-bound public polynomial root queries with cumulative phase meters.

use crate::callbacks::PySupervisedCallbackSession;
use crate::expr::PyExpr;
use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
};
use fsym_core::real_roots::{
    MAX_REAL_ROOT_DEGREE, MAX_REAL_ROOT_INPUT_BITS, RealRootQuery, isolate_real_roots_in_interval,
    refine_real_root_interval,
};
use fsym_core::{BigRational, Expr};
use num_traits::{Signed, Zero};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct QueryMeter<'py, 'session> {
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    local: Budget,
    failure: Option<PyErr>,
}

impl<'py, 'session> QueryMeter<'py, 'session> {
    fn new(py: Python<'py>, session: Option<&'session PySupervisedCallbackSession>) -> Self {
        let mut limits = BudgetLimits::uniform(256 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = 100_000_000;
        Self {
            py,
            session,
            local: Budget::new(limits),
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
            self.failure = Some(error);
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

impl BudgetMeter for QueryMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        // Preflight the private envelope before debiting the shared request.
        // Failed batches leave both counters unchanged.
        let mut totals = [0u64; DIMENSION_COUNT];
        for &(dimension, amount) in charges {
            if amount == 0 {
                return Err(BudgetError::ZeroCharge.into());
            }
            let index = dimension.index();
            totals[index] = totals[index]
                .checked_add(amount)
                .ok_or(BudgetError::ChargeOverflow { dimension })?;
        }
        for dimension in Dimension::ALL {
            let requested = totals[dimension.index()];
            let remaining = self.local.remaining(dimension);
            if requested > remaining {
                return Err(BudgetError::Exhausted {
                    dimension,
                    requested,
                    remaining,
                }
                .into());
            }
        }
        self.forward(totals[Dimension::ComputeSteps.index()])?;
        self.local.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.forward(0)
    }
}

fn charge(meter: &mut impl BudgetMeter, steps: usize, bytes: usize) -> PyResult<()> {
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(crate::to_value_error)
}

fn rational(value: Option<&PyExpr>, meter: &mut impl BudgetMeter) -> PyResult<Option<BigRational>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let bits = match &value.inner {
        Expr::Integer(n) => n.bits(),
        Expr::Rational(q) => q.numer().bits().max(q.denom().bits()),
        _ => {
            return Err(PyValueError::new_err(
                "real-root bounds and precision must be literal exact rationals",
            ));
        }
    };
    if bits > MAX_REAL_ROOT_INPUT_BITS {
        return Err(PyValueError::new_err("real-root control exceeds 4096 bits"));
    }
    charge(meter, 1, 128 + bits.div_ceil(8) as usize * 2)?;
    Ok(Some(match &value.inner {
        Expr::Integer(n) => BigRational::from_integer(n.clone()),
        Expr::Rational(q) => q.clone(),
        _ => unreachable!(),
    }))
}

/// Convert the original typed polynomial once, preserve the public query and
/// use one cumulative pair of meters through all mathematics and publication.
/// The zero-polynomial empty-list convention is a shell compatibility result;
/// it does not construct a certificate claiming that zero has no real roots.
#[pyfunction]
#[pyo3(signature = (
    expression, variable, operation, lower=None, upper=None, eps=None, steps=None,
    require_square_free=false, integral_coefficients=false, session=None
))]
#[allow(clippy::too_many_arguments)]
pub fn real_root_analysis_native(
    py: Python<'_>,
    expression: &PyExpr,
    variable: &Bound<'_, PyAny>,
    operation: &str,
    lower: Option<&PyExpr>,
    upper: Option<&PyExpr>,
    eps: Option<&PyExpr>,
    steps: Option<usize>,
    require_square_free: bool,
    integral_coefficients: bool,
    session: Option<PyRef<'_, PySupervisedCallbackSession>>,
) -> PyResult<Vec<(PyExpr, PyExpr, usize)>> {
    let mut generator = QueryMeter::new(py, session.as_deref());
    let mut verifier = QueryMeter::new(py, session.as_deref());
    let output = (|| {
        generator.checkpoint().map_err(crate::to_value_error)?;
        if !matches!(operation, "count" | "intervals" | "refine") {
            return Err(PyValueError::new_err(
                "unknown real-root analysis operation",
            ));
        }
        if operation != "refine" && steps.is_some() {
            return Err(PyValueError::new_err("steps apply only to root refinement"));
        }
        let symbol = crate::extract_symbol_arg(variable)?;
        let query = RealRootQuery {
            lower: rational(lower, &mut generator)?,
            upper: rational(upper, &mut generator)?,
            max_width: rational(eps, &mut generator)?,
            require_square_free,
        };
        if let (Some(lower), Some(upper)) = (&query.lower, &query.upper)
            && lower > upper
        {
            return Err(PyValueError::new_err(
                "root interval endpoints must be ordered",
            ));
        }
        if query
            .max_width
            .as_ref()
            .is_some_and(|eps| !eps.is_positive())
        {
            return Err(PyValueError::new_err("'eps' must be a positive rational"));
        }
        let polynomial =
            fsym_polys::bounded_polynomial_from_expr(&expression.inner, &[symbol], &mut generator)
                .map_err(crate::to_value_error)?;
        let degree = polynomial.degree_in(0) as usize;
        if degree > MAX_REAL_ROOT_DEGREE {
            return Err(PyValueError::new_err(
                "exact root queries support degree at most 64",
            ));
        }
        if integral_coefficients
            && polynomial
                .terms()
                .values()
                .any(|coefficient| !coefficient.denom().is_one())
        {
            return Err(PyValueError::new_err(
                "polynomial coefficients are not in declared ZZ",
            ));
        }
        charge(&mut generator, degree + 1, (degree + 1) * 96)?;
        let mut coefficients = vec![BigRational::zero(); degree + 1];
        for (powers, coefficient) in polynomial.terms() {
            charge(
                &mut generator,
                1,
                64 + (coefficient.numer().bits() + coefficient.denom().bits()).div_ceil(8) as usize,
            )?;
            coefficients[powers[0] as usize] = coefficient.clone();
        }
        if polynomial.is_zero() && operation != "refine" {
            generator.checkpoint().map_err(crate::to_value_error)?;
            verifier.checkpoint().map_err(crate::to_value_error)?;
            return Ok(Vec::new());
        }
        let roots = if operation == "refine" {
            refine_real_root_interval(&coefficients, &query, steps, &mut generator, &mut verifier)
        } else {
            isolate_real_roots_in_interval(&coefficients, &query, &mut generator, &mut verifier)
        }
        .map_err(crate::to_value_error)?;
        charge(
            &mut generator,
            roots.distinct_count() + 1,
            (roots.distinct_count() + 1) * 128,
        )?;
        for root in roots.intervals() {
            for endpoint in [&root.lower, &root.upper] {
                let payload_bits = endpoint.numer().bits() + endpoint.denom().bits();
                charge(
                    &mut generator,
                    1 + payload_bits.div_ceil(32) as usize,
                    64 + payload_bits.div_ceil(8) as usize,
                )?;
            }
        }
        let result = roots
            .intervals()
            .iter()
            .map(|root| {
                (
                    PyExpr::from_expr(crate::rational_to_expr(root.lower.clone())),
                    PyExpr::from_expr(crate::rational_to_expr(root.upper.clone())),
                    root.multiplicity,
                )
            })
            .collect();
        generator.checkpoint().map_err(crate::to_value_error)?;
        verifier.checkpoint().map_err(crate::to_value_error)?;
        Ok(result)
    })();
    match generator.failure.or(verifier.failure) {
        Some(error) => Err(error),
        None => output,
    }
}
