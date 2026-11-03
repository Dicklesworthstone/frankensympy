//! Tests of the production Python ledger around the real independent checker.
//!
//! These also run when this unchanged module is included by a small standalone
//! component crate. They do not stand in for the full extension or Python shell.

use super::*;
use fsym_core::{
    BigInt, BigRational, Expr, MAX_ROOT_BOX_LIMBS, RealBall, RootBoxWitness, Symbol,
    SymbolIdentity, metered_enclose_residual_jacobian, verify_root_box,
};
use pyo3::types::PyDict;

const AMPLE: [u64; 3] = [1_000_000_000, 4 * 1024 * 1024 * 1024, 1_000_000];

fn python_budget(py: Python<'_>, limits: [u64; 3]) -> Bound<'_, PyAny> {
    let module = PyModule::new(py, "numeric_budget_component_tests").unwrap();
    register(&module).unwrap();
    let kwargs = PyDict::new(py);
    for (name, value) in ["max_steps", "max_memory_bytes", "max_allocations"]
        .into_iter()
        .zip(limits)
    {
        kwargs.set_item(name, value).unwrap();
    }
    module
        .getattr("EvaluationBudget")
        .unwrap()
        .call((), Some(&kwargs))
        .unwrap()
}

fn remaining(instance: &Bound<'_, PyAny>) -> [u64; 3] {
    [
        "remaining_steps",
        "remaining_memory_bytes",
        "remaining_allocations",
    ]
    .map(|name| instance.getattr(name).unwrap().extract().unwrap())
}

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn linear_problem() -> (Vec<Expr>, Vec<Symbol>, RootBoxWitness) {
    // The declared identity and held source structure must survive admission.
    let variable = Symbol::with_identity(
        "certification_budget_x",
        SymbolIdentity::from_assumptions_digest([7; 32]),
    );
    let residual = Expr::Add(vec![Expr::Sym(variable.clone()), Expr::from_i64(-1)]);
    let witness = RootBoxWitness {
        domain: vec![RealBall::new(rational(1, 1), rational(1, 4)).unwrap()],
        center: vec![rational(1, 1)],
        preconditioner: vec![vec![rational(1, 1)]],
        precision_digits: 32,
    };
    (vec![residual], vec![variable], witness)
}

#[test]
fn repeated_actual_enclosures_exhaust_one_cumulative_python_ledger() {
    Python::initialize();
    Python::attach(|py| {
        let (residuals, variables, witness) = linear_problem();
        let measured = python_budget(py, AMPLE);
        let measured_budget = measured.extract::<PyRef<'_, PyEvaluationBudget>>().unwrap();
        let reference = with_certification_meter(py, Some(&measured_budget), |meter| {
            metered_enclose_residual_jacobian(
                &residuals,
                &variables,
                &witness.domain,
                witness.precision_digits,
                meter,
            )
        })
        .unwrap();
        let after = remaining(&measured);
        let cost = std::array::from_fn::<_, 3, _>(|index| AMPLE[index] - after[index]);
        assert!(cost.iter().all(|amount| *amount > 0));
        let twice = cost.map(|amount| amount.checked_mul(2).unwrap());
        let shared = python_budget(py, twice);
        let shared_budget = shared.extract::<PyRef<'_, PyEvaluationBudget>>().unwrap();
        for _ in 0..2 {
            let actual = with_certification_meter(py, Some(&shared_budget), |meter| {
                metered_enclose_residual_jacobian(
                    &residuals,
                    &variables,
                    &witness.domain,
                    witness.precision_digits,
                    meter,
                )
            })
            .unwrap();
            assert_eq!(actual, reference);
        }
        assert_eq!(remaining(&shared), [0; 3]);
        let exhausted = with_certification_meter(py, Some(&shared_budget), |meter| {
            metered_enclose_residual_jacobian(
                &residuals,
                &variables,
                &witness.domain,
                witness.precision_digits,
                meter,
            )
        })
        .unwrap_err();
        assert!(exhausted.is_instance_of::<EvaluationBudgetExceeded>(py));
        assert!(!exhausted.is_instance_of::<PyValueError>(py));
        assert_eq!(remaining(&shared), [0; 3]);
    });
}

#[test]
fn failed_actual_root_box_retains_cost_before_later_exact_acceptance() {
    Python::initialize();
    Python::attach(|py| {
        let (residuals, variables, witness) = linear_problem();
        let instance = python_budget(py, AMPLE);
        let budget = instance.extract::<PyRef<'_, PyEvaluationBudget>>().unwrap();
        let mut excludes_root = witness.clone();
        excludes_root.domain[0] = RealBall::new(rational(2, 1), rational(1, 4)).unwrap();
        excludes_root.center[0] = rational(2, 1);
        let rejected = with_certification_meter(py, Some(&budget), |meter| {
            verify_root_box(&residuals, &variables, &excludes_root, meter)
        })
        .unwrap_err();
        assert!(rejected.is_instance_of::<PyValueError>(py));
        assert!(!rejected.is_instance_of::<EvaluationBudgetExceeded>(py));
        let after_failure = remaining(&instance);
        assert!(
            after_failure
                .iter()
                .zip(AMPLE)
                .all(|(left, start)| *left < start)
        );

        let accepted = with_certification_meter(py, Some(&budget), |meter| {
            verify_root_box(&residuals, &variables, &witness, meter)
        })
        .unwrap();
        assert_eq!(accepted.residuals(), residuals);
        assert_eq!(accepted.variables(), variables);
        assert_eq!(accepted.witness(), &witness);
        assert_eq!(
            accepted.root_enclosure(),
            &[RealBall::exact(rational(1, 1))]
        );
        assert_eq!(accepted.contraction_bound(), &rational(0, 1));
        assert!(
            remaining(&instance)
                .iter()
                .zip(after_failure)
                .all(|(left, before)| *left < before)
        );
    });
}

#[test]
fn real_caller_and_checker_resource_refusals_keep_operational_types() {
    Python::initialize();
    Python::attach(|py| {
        let (residuals, variables, witness) = linear_problem();
        let empty_memory = python_budget(py, [AMPLE[0], 0, AMPLE[2]]);
        let empty_budget = empty_memory
            .extract::<PyRef<'_, PyEvaluationBudget>>()
            .unwrap();
        let exhausted = with_certification_meter(py, Some(&empty_budget), |meter| {
            verify_root_box(&residuals, &variables, &witness, meter)
        })
        .unwrap_err();
        assert!(exhausted.is_instance_of::<EvaluationBudgetExceeded>(py));
        assert!(exhausted.is_instance_of::<PyRuntimeError>(py));
        assert!(!exhausted.is_instance_of::<PyValueError>(py));
        assert_eq!(remaining(&empty_memory), [AMPLE[0], 0, AMPLE[2]]);

        // The independent checker's own height refusal is operational too,
        // even when the caller's cumulative ledger still has ample capacity.
        let oversized =
            Expr::Integer(BigInt::from(1) << u32::try_from(MAX_ROOT_BOX_LIMBS * 64).unwrap());
        let instance = python_budget(py, AMPLE);
        let budget = instance.extract::<PyRef<'_, PyEvaluationBudget>>().unwrap();
        let intrinsic = with_certification_meter(py, Some(&budget), |meter| {
            metered_enclose_residual_jacobian(
                &[oversized],
                &variables,
                &witness.domain,
                witness.precision_digits,
                meter,
            )
        })
        .unwrap_err();
        assert!(intrinsic.is_instance_of::<EvaluationBudgetExceeded>(py));
        assert!(!intrinsic.is_instance_of::<PyValueError>(py));
        assert!(remaining(&instance)[0] < AMPLE[0]);
    });
}

#[test]
fn cancellation_before_helper_and_inside_real_checker_stays_typed() {
    Python::initialize();
    Python::attach(|py| {
        let (residuals, variables, witness) = linear_problem();
        let instance = python_budget(py, AMPLE);
        instance.call_method0("cancel").unwrap();
        let budget = instance.extract::<PyRef<'_, PyEvaluationBudget>>().unwrap();
        let entered = AtomicBool::new(false);
        let cancelled = with_certification_meter(py, Some(&budget), |meter| {
            entered.store(true, Ordering::Release);
            verify_root_box(&residuals, &variables, &witness, meter)
        })
        .unwrap_err();
        assert!(cancelled.is_instance_of::<EvaluationCancelled>(py));
        assert!(!entered.load(Ordering::Acquire));
        assert_eq!(remaining(&instance), AMPLE);

        let at_entry = python_budget(py, AMPLE);
        let at_entry_ref = at_entry.extract::<PyRef<'_, PyEvaluationBudget>>().unwrap();
        let ledger: &PyEvaluationBudget = &at_entry_ref;
        let cancelled_in_checker = with_certification_meter(py, Some(ledger), |meter| {
            // Exercise the real RootBoxError::Meter source chain after the
            // helper's initial safe point has already passed.
            ledger.cancel();
            verify_root_box(&residuals, &variables, &witness, meter)
        })
        .unwrap_err();
        assert!(cancelled_in_checker.is_instance_of::<EvaluationCancelled>(py));
        assert!(!cancelled_in_checker.is_instance_of::<PyValueError>(py));
        assert_eq!(remaining(&at_entry), AMPLE);
    });
}

#[test]
fn cancellation_after_actual_acceptance_prevents_publication_and_keeps_cost() {
    Python::initialize();
    Python::attach(|py| {
        let (residuals, variables, witness) = linear_problem();
        let instance = python_budget(py, AMPLE);
        let budget = instance.extract::<PyRef<'_, PyEvaluationBudget>>().unwrap();
        let ledger: &PyEvaluationBudget = &budget;
        let completed = AtomicBool::new(false);
        let cancelled = with_certification_meter(py, Some(ledger), |meter| {
            let receipt = verify_root_box(&residuals, &variables, &witness, meter)?;
            assert_eq!(receipt.witness(), &witness);
            completed.store(true, Ordering::Release);
            ledger.cancel();
            Ok::<_, RootBoxError>(receipt)
        })
        .unwrap_err();
        assert!(completed.load(Ordering::Acquire));
        assert!(cancelled.is_instance_of::<EvaluationCancelled>(py));
        assert!(!cancelled.is_instance_of::<PyValueError>(py));
        assert!(
            remaining(&instance)
                .iter()
                .zip(AMPLE)
                .all(|(left, start)| *left < start)
        );
        let after = remaining(&instance);
        let again = with_certification_meter(py, Some(ledger), |meter| {
            verify_root_box(&residuals, &variables, &witness, meter)
        })
        .unwrap_err();
        assert!(again.is_instance_of::<EvaluationCancelled>(py));
        assert_eq!(remaining(&instance), after);
    });
}

#[test]
fn omitted_budget_keeps_legacy_fallback_cap_after_real_enclosure_work() {
    Python::initialize();
    Python::attach(|py| {
        let (residuals, variables, witness) = linear_problem();
        let accepted = with_certification_meter(py, None, |meter| {
            verify_root_box(&residuals, &variables, &witness, meter)
        })
        .unwrap();
        assert_eq!(accepted.witness(), &witness);

        let exhausted = with_certification_meter(py, None, |meter| {
            let enclosure = metered_enclose_residual_jacobian(
                &residuals,
                &variables,
                &witness.domain,
                witness.precision_digits,
                meter,
            )?;
            // Actual checker work has already consumed some memory allowance.
            // Requesting the entire legacy allowance now must fail, proving
            // that None did not silently select the unbounded evaluation lane.
            meter.charge(Dimension::MemoryBytes, 512 * 1024 * 1024)?;
            Ok::<_, RootBoxError>(enclosure)
        })
        .unwrap_err();
        assert!(exhausted.is_instance_of::<EvaluationBudgetExceeded>(py));
        assert!(!exhausted.is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn direct_and_nested_allocation_failures_remain_memory_errors() {
    use fsym_calculus::sparse_hessian::ObjectiveEvaluationError;
    use pyo3::exceptions::PyMemoryError;

    Python::initialize();
    Python::attach(|py| {
        let direct = evaluation_error(EvalError::AllocationFailure);
        let nested = evaluation_error(ObjectiveEvaluationError::Evaluation(
            EvalError::AllocationFailure,
        ));
        for error in [direct, nested] {
            assert!(error.is_instance_of::<PyMemoryError>(py));
            assert!(!error.is_instance_of::<PyValueError>(py));
            assert!(!error.is_instance_of::<EvaluationBudgetExceeded>(py));
            assert!(!error.is_instance_of::<EvaluationCancelled>(py));
        }
    });
}
