//! Source-bound QQ/ZZ composition for the retained Poly/PurePoly surface.
//!
//! Admission of both complete original inputs, their declared coefficient
//! domains, optional explicit ordinal coordinate mapping, generation,
//! independent coefficient verification and output lifting share two private
//! envelopes. Both phases also debit the same optional caller-owned session.

use crate::callbacks::PySupervisedCallbackSession;
use crate::expr::PyExpr;
use crate::ideal_operations::{budget, charge, symbols};
use crate::recursive_division::{checked_lift, integral, original_domain, polynomial_source};
use crate::to_value_error;
use fsym_budget::{Budget, BudgetError, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError};
use fsym_polys::checked_polynomial_composition;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Preserve the existing phase limits while forwarding actual native work.
/// A private generator envelope cannot borrow the verifier's allowance, and
/// neither phase can restart the caller's cumulative request ledger.
struct CompositionMeter<'py, 'session> {
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    private: Budget,
    failure: Option<PyErr>,
}

impl<'py, 'session> CompositionMeter<'py, 'session> {
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
            // MeterError stops native recursion; the wrapper returns the
            // original budget, cancellation, or ownership exception.
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

impl BudgetMeter for CompositionMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if self.failure.is_some() {
            return Err(MeterError::Cancelled);
        }
        // Preflight all private dimensions before changing either ledger.
        // The private envelope is exclusively owned and cannot change while
        // the session's ownership check executes Python.
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (outer, inner, variables, outer_variables=None, inner_variables=None,
                   outer_domain=None, inner_domain=None, domain="auto", inner_coordinates=None,
                   *, session=None))]
pub fn polynomial_compose_native(
    py: Python<'_>,
    outer: &PyExpr,
    inner: &PyExpr,
    variables: &Bound<'_, PyList>,
    outer_variables: Option<&Bound<'_, PyList>>,
    inner_variables: Option<&Bound<'_, PyList>>,
    outer_domain: Option<&str>,
    inner_domain: Option<&str>,
    domain: &str,
    inner_coordinates: Option<&Bound<'_, PyList>>,
    session: Option<PyRef<'_, PySupervisedCallbackSession>>,
) -> PyResult<(PyExpr, String)> {
    // Shallow, bounded ingress snapshots precede ownership callbacks. Native
    // expression arguments stay borrowed and immutable throughout the call.
    let snapshots = [
        Some(variables),
        outer_variables,
        inner_variables,
        inner_coordinates,
    ]
    .into_iter()
    .map(|values| {
        values
            .map(|values| {
                if values.len() > 8 {
                    return Err(PyValueError::new_err(
                        "checked composition supports at most eight generators",
                    ));
                }
                Ok((values, values.iter().collect::<Vec<_>>()))
            })
            .transpose()
    })
    .collect::<PyResult<Vec<_>>>()?;
    let mut generator = CompositionMeter::new(py, session.as_deref());
    let mut verifier = CompositionMeter::new(py, session.as_deref());
    let result = (|| {
        let count = snapshots
            .iter()
            .flatten()
            .map(|(_, values)| values.len() + 1)
            .sum();
        charge(&mut generator, count, (count + snapshots.len()) * 128)?;
        // Never reread mutable caller lists for computation: a callback can
        // swap coordinates and restore them before the terminal guard. These
        // bounded private copies retain the exact ingress objects instead.
        let retained = snapshots
            .iter()
            .map(|snapshot| {
                snapshot
                    .as_ref()
                    .map(|(_, values)| PyList::new(py, values))
                    .transpose()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let output = compose(
            outer,
            inner,
            retained[0]
                .as_ref()
                .expect("operation coordinates retained"),
            retained[1].as_ref(),
            retained[2].as_ref(),
            outer_domain,
            inner_domain,
            domain,
            retained[3].as_ref(),
            &mut generator,
            &mut verifier,
        )?;
        for (original, values) in snapshots.iter().flatten() {
            if original.len() != values.len()
                || original
                    .iter()
                    .zip(values)
                    .any(|(current, previous)| !current.is(previous))
            {
                return Err(PyRuntimeError::new_err(
                    "composition coordinates mutated during exact evaluation",
                ));
            }
        }
        Ok(output)
    })();
    match generator.failure.or(verifier.failure) {
        Some(error) => Err(error),
        None => result,
    }
}

#[allow(clippy::too_many_arguments)]
fn compose(
    outer: &PyExpr,
    inner: &PyExpr,
    variables: &Bound<'_, PyList>,
    outer_variables: Option<&Bound<'_, PyList>>,
    inner_variables: Option<&Bound<'_, PyList>>,
    outer_domain: Option<&str>,
    inner_domain: Option<&str>,
    domain: &str,
    inner_coordinates: Option<&Bound<'_, PyList>>,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> PyResult<(PyExpr, String)> {
    if !matches!(domain, "auto" | "ZZ" | "QQ") {
        return Err(PyValueError::new_err(
            "checked polynomial composition supports ZZ and QQ",
        ));
    }
    let variables = symbols(variables, generator)?;
    if variables.is_empty() {
        return Err(PyValueError::new_err(
            "polynomial composition requires a primary generator",
        ));
    }
    let left = polynomial_source(&outer.inner, &variables, generator, verifier)?;
    original_domain(
        &outer.inner,
        Some(&left),
        &variables,
        outer_variables,
        outer_domain,
        generator,
        verifier,
    )?;
    let coordinates = if let Some(coordinates) = inner_coordinates {
        let coordinates = symbols(coordinates, generator)?;
        let original = inner_variables.ok_or_else(|| {
            PyValueError::new_err("positional composition requires the original inner ring")
        })?;
        let original = symbols(original, generator)?;
        charge(
            generator,
            original
                .iter()
                .chain(&coordinates)
                .map(|symbol| symbol.name.len() + 1)
                .sum(),
            0,
        )?;
        if coordinates.len() != variables.len() || coordinates != original {
            return Err(PyValueError::new_err(
                "positional composition must retain equal original and operation dimensions",
            ));
        }
        Some(coordinates)
    } else {
        None
    };
    let inner_axes = coordinates.as_deref().unwrap_or(&variables);
    let mut right = polynomial_source(&inner.inner, inner_axes, generator, verifier)?;
    original_domain(
        &inner.inner,
        Some(&right),
        inner_axes,
        inner_variables,
        inner_domain,
        generator,
        verifier,
    )?;
    if coordinates.is_some() {
        // PurePoly's explicit ordinal isomorphism is applied only after the
        // original right source and its original domain have been checked.
        let bytes = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
        charge(generator, variables.len(), bytes)?;
        right.generators = variables.clone();
    }
    let inputs_integral = integral(&left, generator)? & integral(&right, generator)?;
    if domain == "ZZ" && !inputs_integral {
        return Err(PyValueError::new_err(
            "noninteger coefficients cannot be represented in ZZ",
        ));
    }
    // The domain depends on both admitted original inputs, even when their
    // fractional coefficients cancel or the outer source is constant.
    let result_domain = if domain == "QQ" || !inputs_integral {
        "QQ"
    } else {
        "ZZ"
    };
    let accepted = checked_polynomial_composition(&left, &right, generator, verifier)
        .map_err(to_value_error)?;
    let result = accepted.result();
    let nodes = 1 + result
        .terms()
        .keys()
        .map(|powers| 2 + 4 * powers.iter().filter(|&&power| power != 0).count())
        .sum::<usize>();
    if nodes > 4096 {
        return Err(PyValueError::new_err(
            "composition exceeds its bounded output expression envelope",
        ));
    }
    let ring_bytes: usize = variables.iter().map(|symbol| symbol.name.len() + 128).sum();
    charge(generator, nodes, nodes * (2048 + ring_bytes))?;
    let output = checked_lift(result, verifier)?;
    generator.checkpoint().map_err(to_value_error)?;
    verifier.checkpoint().map_err(to_value_error)?;
    Ok((output, result_domain.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::BudgetLimits;
    use fsym_core::Expr;

    fn session(py: Python<'_>, steps: u64) -> PySupervisedCallbackSession {
        PySupervisedCallbackSession::new(py, 8, 128, steps, 2048, 2048, 60_000).unwrap()
    }

    #[test]
    fn actual_composition_forwards_both_phase_costs_to_one_request() {
        Python::initialize();
        Python::attach(|py| {
            let x = Expr::symbol("x");
            let outer = PyExpr::from_expr(x.clone() * x.clone());
            let inner = PyExpr::from_expr(x.clone() + Expr::from_i64(1));
            let variables = PyList::new(py, ["x"]).unwrap();
            let mut reference_generator = budget();
            let mut reference_verifier = budget();
            let reference = compose(
                &outer,
                &inner,
                &variables,
                None,
                None,
                None,
                None,
                "QQ",
                None,
                &mut reference_generator,
                &mut reference_verifier,
            )
            .unwrap();
            let generated = 20_000_000 - reference_generator.remaining(Dimension::ComputeSteps);
            let verified = 20_000_000 - reference_verifier.remaining(Dimension::ComputeSteps);
            assert!(generated > 0 && verified > 0);
            let measured = session(py, generated + verified);
            let mut generator = CompositionMeter::new(py, Some(&measured));
            let mut verifier = CompositionMeter::new(py, Some(&measured));
            let output = compose(
                &outer,
                &inner,
                &variables,
                None,
                None,
                None,
                None,
                "QQ",
                None,
                &mut generator,
                &mut verifier,
            )
            .unwrap();
            assert_eq!(output.0.inner, reference.0.inner);
            assert_eq!(output.1, "QQ");
            assert_eq!(measured.consumed_nested_steps(), generated + verified);
            let expected = x.clone() * x.clone() + Expr::from_i64(2) * x + Expr::from_i64(1);
            assert_eq!(output.0.inner, expected);
            assert!(!measured.cancelled());
            assert!(generator.charge(Dimension::ComputeSteps, 1).is_err());
            assert!(measured.cancelled());
            assert_eq!(measured.consumed_nested_steps(), generated + verified);
        });
    }

    #[test]
    fn private_memory_refusal_is_atomic_and_latches_the_request() {
        Python::initialize();
        Python::attach(|py| {
            let session = session(py, 100);
            let mut meter = CompositionMeter::new(py, Some(&session));
            meter.private = Budget::new(BudgetLimits::uniform(1, 0));
            let before = meter.private.snapshot();
            assert!(
                meter
                    .charge_batch(&[(Dimension::ComputeSteps, 1), (Dimension::MemoryBytes, 2),])
                    .is_err()
            );
            assert_eq!(meter.private.snapshot(), before);
            assert_eq!(session.consumed_nested_steps(), 0);
            assert!(session.cancelled());
            assert!(
                meter
                    .failure
                    .as_ref()
                    .unwrap()
                    .is_instance_of::<PyValueError>(py)
            );
            assert!(meter.checkpoint().is_err());
        });
    }

    #[test]
    fn caller_refusal_does_not_spend_either_private_dimension() {
        Python::initialize();
        Python::attach(|py| {
            let session = session(py, 4);
            let mut meter = CompositionMeter::new(py, Some(&session));
            let before = meter.private.snapshot();
            assert!(
                meter
                    .charge_batch(&[
                        (Dimension::ComputeSteps, 2),
                        (Dimension::MemoryBytes, 32),
                        (Dimension::ComputeSteps, 3),
                    ])
                    .is_err()
            );
            assert_eq!(meter.private.snapshot(), before);
            assert_eq!(session.consumed_nested_steps(), 0);
            assert!(session.cancelled());
            let failure = meter.failure.as_ref().unwrap();
            assert!(failure.is_instance_of::<PyValueError>(py));
            assert!(
                failure
                    .to_string()
                    .contains("nested native request budget exhausted")
            );
        });
    }

    #[test]
    fn cancellation_after_generation_blocks_verification_and_publication() {
        Python::initialize();
        Python::attach(|py| {
            let session = session(py, 100);
            let mut generator = CompositionMeter::new(py, Some(&session));
            let mut verifier = CompositionMeter::new(py, Some(&session));
            generator.charge(Dimension::ComputeSteps, 3).unwrap();
            session.cancel(py).unwrap();
            let before = verifier.private.snapshot();
            assert!(verifier.charge(Dimension::ComputeSteps, 2).is_err());
            assert_eq!(verifier.private.snapshot(), before);
            assert_eq!(session.consumed_nested_steps(), 3);
            assert!(generator.checkpoint().is_err());
            assert!(
                verifier
                    .failure
                    .as_ref()
                    .unwrap()
                    .is_instance_of::<PyRuntimeError>(py)
            );
        });
    }
}
