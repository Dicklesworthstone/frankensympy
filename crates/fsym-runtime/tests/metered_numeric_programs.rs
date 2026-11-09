//! Numerical consumers use the real region ledger and cancellation authority.
//!
//! The programs are independently checked before execution. These tests cover
//! their approximate evaluation boundary, not runtime-managed compilation or
//! a numerical accuracy certificate. No background work is spawned.

use asupersync::{CancelKind, Cx};
use fsym_budget::{BudgetError, MeterError};
use fsym_calculus::compile::EvalError;
use fsym_calculus::sparse_hessian::ObjectiveEvaluationError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_calculus::{VerifiedObjective, VerifiedResidualSystem};
use fsym_core::{Symbol, parse};
use fsym_runtime::{Budget, BudgetLimits, Dimension, FsymCx};

fn residual_system() -> VerifiedResidualSystem {
    VerifiedResidualSystem::try_compile(
        &[parse("x^2+y").unwrap(), parse("x-y").unwrap()],
        &[Symbol::new("x"), Symbol::new("y")],
    )
    .unwrap()
}

fn objective() -> VerifiedObjective {
    VerifiedObjective::try_compile(
        &parse("x^2+x*y+3*y^2").unwrap(),
        &[Symbol::new("x"), Symbol::new("y")],
    )
    .unwrap()
}

#[test]
fn checked_numeric_programs_share_the_region_and_observe_its_cancellation() {
    let system = residual_system();
    let objective = objective();
    let runtime = Cx::detached_cancel_context();
    let limits = BudgetLimits::uniform(1_000_000, 137);
    let mut region = FsymCx::new(&runtime, Budget::new(limits), limits);
    let ledger = region.ledger_identity();
    let mut residuals = [901.0; 2];
    let mut jacobian = [902.0; 4];

    system
        .try_eval_dense_into_with_meter(&[2.0, 3.0], &mut residuals, &mut jacobian, &mut region)
        .unwrap();
    assert_eq!(residuals, [7.0, -1.0]);
    assert_eq!(jacobian, [4.0, 1.0, 1.0, -1.0]);
    let after_system = region.remaining(Dimension::ComputeSteps);
    assert!(after_system < limits.dimensions[Dimension::ComputeSteps.index()]);

    let result = objective
        .evaluate_with_meter(&[2.0, 3.0], &mut region)
        .unwrap();
    assert_eq!(result.value, 37.0);
    assert_eq!(result.gradient, [7.0, 20.0]);
    assert_eq!(result.hessian, [2.0, 1.0, 1.0, 6.0]);
    let after_objective = region.remaining(Dimension::ComputeSteps);
    assert!(after_objective < after_system);
    assert_eq!(
        objective
            .hessian_vector_product_with_meter(&[2.0, 3.0], &[5.0, -2.0], &mut region)
            .unwrap(),
        [8.0, -7.0]
    );
    assert!(region.remaining(Dimension::ComputeSteps) < after_objective);
    assert_eq!(region.ledger_identity(), ledger);
    assert!(region.has_verifier_authority());
    assert_eq!(region.verifier_remaining(), 137);
    for dimension in [Dimension::MemoryBytes, Dimension::AllocationCount] {
        assert!(region.remaining(dimension) < limits.dimensions[dimension.index()]);
    }

    let before_cancel = Dimension::ALL.map(|dimension| region.remaining(dimension));
    runtime.cancel_with(CancelKind::User, Some("stop numerical consumers"));
    residuals.fill(901.0);
    jacobian.fill(902.0);
    assert!(matches!(
        system.try_eval_dense_into_with_meter(
            &[2.0, 3.0],
            &mut residuals,
            &mut jacobian,
            &mut region,
        ),
        Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Cancelled
        )))
    ));
    assert_eq!(residuals, [901.0; 2]);
    assert_eq!(jacobian, [902.0; 4]);

    let mut gradient = [903.0; 2];
    let mut hessian = [904.0; 4];
    assert!(matches!(
        objective.try_eval_into_with_meter(&[2.0, 3.0], &mut gradient, &mut hessian, &mut region,),
        Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Cancelled
        )))
    ));
    assert_eq!(gradient, [903.0; 2]);
    assert_eq!(hessian, [904.0; 4]);
    let mut product = [905.0; 2];
    assert!(matches!(
        objective.try_hessian_vector_product_into_with_meter(
            &[2.0, 3.0],
            &[5.0, -2.0],
            &mut product,
            &mut region,
        ),
        Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Cancelled
        )))
    ));
    assert_eq!(product, [905.0; 2]);
    assert_eq!(
        Dimension::ALL.map(|dimension| region.remaining(dimension)),
        before_cancel,
        "pre-cancelled consumers cannot consume more work or reset the ledger"
    );
    assert_eq!(region.verifier_remaining(), 137);
}

#[test]
fn child_numeric_consumption_is_reconciled_without_refunding_spent_work() {
    let system = residual_system();
    let objective = objective();
    let runtime = Cx::detached_cancel_context();
    let limits = BudgetLimits::uniform(1_000_000, 251);
    let child_limits = BudgetLimits::uniform(250_000, 0);
    let mut parent = FsymCx::new(&runtime, Budget::new(limits), limits);
    let parent_ledger = parent.ledger_identity();
    let mut child = parent.reserve_child(child_limits).unwrap();
    assert_ne!(child.ledger_identity(), parent_ledger);
    assert!(!child.has_verifier_authority());
    assert_eq!(child.verifier_remaining(), 0);
    for dimension in Dimension::ALL {
        assert_eq!(
            parent.remaining(dimension),
            limits.dimensions[dimension.index()] - child_limits.dimensions[dimension.index()]
        );
    }

    let mut residuals = [0.0; 2];
    let mut jacobian = [0.0; 4];
    system
        .try_eval_dense_into_with_meter(&[2.0, 3.0], &mut residuals, &mut jacobian, &mut child)
        .unwrap();
    assert_eq!(residuals, [7.0, -1.0]);
    assert_eq!(jacobian, [4.0, 1.0, 1.0, -1.0]);
    assert_eq!(
        objective
            .hessian_vector_product_with_meter(&[2.0, 3.0], &[5.0, -2.0], &mut child)
            .unwrap(),
        [8.0, -7.0]
    );
    let spent = Dimension::ALL
        .map(|dimension| child_limits.dimensions[dimension.index()] - child.remaining(dimension));
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(spent[dimension.index()] > 0);
    }
    parent.merge_child(child).unwrap();
    for dimension in Dimension::ALL {
        assert_eq!(
            parent.remaining(dimension),
            limits.dimensions[dimension.index()] - spent[dimension.index()],
            "child merge must retain exact consumed {dimension}"
        );
    }
    assert_eq!(parent.ledger_identity(), parent_ledger);
    assert!(parent.has_verifier_authority());
    assert_eq!(parent.verifier_remaining(), 251);
}

#[test]
fn region_exhaustion_at_publication_cannot_expose_either_computed_lane() {
    let system = residual_system();
    let objective = objective();
    let limits = BudgetLimits::uniform(1_000_000, 79);
    let mut measured = Budget::new(limits);
    system
        .try_eval_dense_into_with_meter(&[2.0, 3.0], &mut [0.0; 2], &mut [0.0; 4], &mut measured)
        .unwrap();
    let required = limits.dimensions[Dimension::ComputeSteps.index()]
        - measured.remaining(Dimension::ComputeSteps);
    assert!(required > 1);

    // This permits all the arithmetic from the measured successful call, but
    // misses one unit of its final combined publication charge. In particular,
    // failure must not leak the already computed residuals while withholding
    // the Jacobian.
    let runtime = Cx::detached_cancel_context();
    let mut tight = limits;
    tight.dimensions[Dimension::ComputeSteps.index()] = required - 1;
    let mut region = FsymCx::new(&runtime, Budget::new(tight), tight);
    let ledger = region.ledger_identity();
    let mut residuals = [901.0; 2];
    let mut jacobian = [902.0; 4];
    assert!(matches!(
        system.try_eval_dense_into_with_meter(
            &[2.0, 3.0],
            &mut residuals,
            &mut jacobian,
            &mut region,
        ),
        Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            })
        )))
    ));
    assert_eq!(residuals, [901.0; 2]);
    assert_eq!(jacobian, [902.0; 4]);
    let remaining = region.remaining(Dimension::ComputeSteps);
    assert!(remaining < required - 1);

    // A subsequent consumer must inherit that exhaustion rather than silently
    // resetting the region or borrowing from the protected verifier pool.
    let mut gradient = [903.0; 2];
    let mut hessian = [904.0; 4];
    assert!(matches!(
        objective.try_eval_into_with_meter(&[2.0, 3.0], &mut gradient, &mut hessian, &mut region,),
        Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            })
        )))
    ));
    assert_eq!(gradient, [903.0; 2]);
    assert_eq!(hessian, [904.0; 4]);
    assert!(region.remaining(Dimension::ComputeSteps) <= remaining);
    assert_eq!(region.ledger_identity(), ledger);
    assert_eq!(region.verifier_remaining(), 79);
}
