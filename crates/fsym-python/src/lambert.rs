//! Source-bound real Lambert-W equation solving through the symbolic request.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::callbacks::PySupervisedCallbackSession;
use crate::expr::PyExpr;
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use pyo3::prelude::*;

struct RequestMeter<'py, 'session> {
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    failure: Option<PyErr>,
}

impl RequestMeter<'_, '_> {
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

impl BudgetMeter for RequestMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.forward(if dimension == Dimension::ComputeSteps {
            amount
        } else {
            0
        })
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let steps = charges
            .iter()
            .filter(|(dimension, _)| *dimension == Dimension::ComputeSteps)
            .fold(0u64, |sum, (_, amount)| sum.saturating_add(*amount));
        self.forward(steps)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.forward(0)
    }
}

/// Return every real root of the admitted affine/exponential source.
/// Grammar rejection is None; verification, budget and source errors escape.
#[pyfunction]
#[pyo3(signature = (source, variable, *, session=None))]
pub fn solve_real_lambert_native(
    source: &PyExpr,
    variable: &PySymbolBinding,
    py: Python<'_>,
    session: Option<PyRef<'_, PySupervisedCallbackSession>>,
) -> PyResult<Option<Vec<PyExpr>>> {
    let mut generator = RequestMeter {
        py,
        session: session.as_deref(),
        failure: None,
    };
    let mut verifier = RequestMeter {
        py,
        session: session.as_deref(),
        failure: None,
    };
    let result: PyResult<Option<Vec<PyExpr>>> = (|| {
        generator.checkpoint().map_err(crate::to_value_error)?;
        verifier.checkpoint().map_err(crate::to_value_error)?;
        let symbol = variable.to_symbol();
        let receipt = fsym_solvers::lambert::metered_solve_real_lambert(
            &source.inner,
            &symbol,
            &mut generator,
            &mut verifier,
        )
        .map_err(crate::to_value_error)?;
        let roots = receipt
            .map(|receipt| -> PyResult<Vec<PyExpr>> {
                generator
                    .charge(Dimension::ComputeSteps, receipt.roots().len() as u64 + 1)
                    .map_err(crate::to_value_error)?;
                Ok(receipt
                    .roots()
                    .iter()
                    .cloned()
                    .map(PyExpr::from_expr)
                    .collect())
            })
            .transpose()?;
        generator.checkpoint().map_err(crate::to_value_error)?;
        verifier.checkpoint().map_err(crate::to_value_error)?;
        Ok(roots)
    })();
    match generator.failure.or(verifier.failure) {
        Some(error) => Err(error),
        None => result,
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(solve_real_lambert_native, module)?)?;
    Ok(())
}
