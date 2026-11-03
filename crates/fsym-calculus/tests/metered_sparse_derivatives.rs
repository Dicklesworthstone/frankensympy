//! Real budget ledgers, cancellation at every safe point and atomic publication.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
};
use fsym_calculus::compile::EvalError;
use fsym_calculus::sparse_hessian::{ObjectiveEvaluationError, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{SparseJacobian, verify_sparse_jacobian_certificate};
use fsym_core::{Expr, Symbol, parse};

#[derive(Default)]
struct Probe {
    checkpoints: usize,
    cancel_at: Option<usize>,
    stopped: bool,
    charged: [u64; DIMENSION_COUNT],
}

impl BudgetMeter for Probe {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "work retried after cancellation");
        let mut next = self.charged;
        for &(dimension, amount) in charges {
            assert_ne!(amount, 0, "zero charges are invalid in the real ledger");
            next[dimension.index()] = next[dimension.index()]
                .checked_add(amount)
                .ok_or(BudgetError::ChargeOverflow { dimension })?;
        }
        self.charged = next;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "work retried after cancellation");
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn sparse() -> SparseJacobian {
    let system = SparseJacobian::try_build(
        &[parse("x^2+y").unwrap(), parse("z^2+y").unwrap()],
        &[Symbol::new("x"), Symbol::new("y"), Symbol::new("z")],
    )
    .unwrap();
    verify_sparse_jacobian_certificate(&system, &[]).unwrap();
    system
}

fn objective() -> VerifiedObjective {
    VerifiedObjective::try_compile(
        &parse("x^2+x*y+3*y^2").unwrap(),
        &[Symbol::new("x"), Symbol::new("y")],
    )
    .unwrap()
}

#[derive(Clone, Copy)]
enum Layout {
    Sparse,
    Dense,
    Compressed,
}

fn sparse_evaluate<M: BudgetMeter + ?Sized>(
    system: &SparseJacobian,
    layout: Layout,
    output: &mut [f64],
    meter: &mut M,
) -> Result<(), EvalError> {
    match layout {
        Layout::Sparse => system.try_eval_sparse_with_meter(&[2.0, 4.0, 3.0], output, meter),
        Layout::Dense => system.try_eval_dense_with_meter(&[2.0, 4.0, 3.0], output, meter),
        Layout::Compressed => {
            system.try_eval_compressed_with_meter(&[2.0, 4.0, 3.0], output, meter)
        }
    }
}

fn budget_for(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(10_000_000, 137);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn sparse_dense_and_colored_values_match_independent_hand_derivatives() {
    let system = sparse();
    assert_eq!(system.column_colors, [0, 1, 0]);
    assert_eq!(system.num_colors, 2);
    for (layout, expected) in [
        (Layout::Sparse, vec![4.0, 1.0, 1.0, 6.0]),
        (Layout::Dense, vec![4.0, 1.0, 0.0, 0.0, 1.0, 6.0]),
        (Layout::Compressed, vec![4.0, 1.0, 6.0, 1.0]),
    ] {
        let mut budget = Budget::new(BudgetLimits::uniform(10_000_000, 137));
        let before = budget.snapshot();
        let mut output = vec![901.0; expected.len()];
        sparse_evaluate(&system, layout, &mut output, &mut budget).unwrap();
        assert_eq!(output, expected);
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            assert!(budget.remaining(dimension) < before.dimensions[dimension.index()]);
        }
        assert_eq!(budget.snapshot().verifier_pool, 137);
    }
}

#[test]
fn sparse_layouts_leave_outputs_unchanged_at_every_cancellation_safe_point() {
    let system = sparse();
    for (layout, length) in [
        (Layout::Sparse, 4),
        (Layout::Dense, 6),
        (Layout::Compressed, 4),
    ] {
        let mut measured = Probe::default();
        sparse_evaluate(&system, layout, &mut vec![0.0; length], &mut measured).unwrap();
        assert!(measured.checkpoints > system.entries.len());
        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = Probe {
                cancel_at: Some(checkpoint),
                ..Probe::default()
            };
            let mut output = vec![901.0; length];
            assert!(matches!(
                sparse_evaluate(&system, layout, &mut output, &mut cancelled),
                Err(EvalError::Meter(MeterError::Cancelled))
            ));
            assert_eq!(output, vec![901.0; length]);
        }
    }
}

#[test]
fn sparse_evaluation_cannot_reset_any_budget_dimension_between_calls() {
    let system = sparse();
    let mut measured = Probe::default();
    let mut output = [0.0; 4];
    sparse_evaluate(&system, Layout::Sparse, &mut output, &mut measured).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let required = measured.charged[dimension.index()];
        assert!(required > 0);
        let mut budget = budget_for(dimension, required);
        sparse_evaluate(&system, Layout::Sparse, &mut output, &mut budget).unwrap();
        assert_eq!(budget.remaining(dimension), 0);
        output.fill(901.0);
        assert!(matches!(
            sparse_evaluate(&system, Layout::Sparse, &mut output, &mut budget),
            Err(EvalError::Meter(MeterError::Budget(BudgetError::Exhausted {
                dimension: actual,
                ..
            }))) if actual == dimension
        ));
        assert_eq!(output, [901.0; 4]);
        assert_eq!(budget.snapshot().verifier_pool, 137);

        let mut insufficient = budget_for(dimension, required - 1);
        assert!(matches!(
            sparse_evaluate(&system, Layout::Sparse, &mut output, &mut insufficient),
            Err(EvalError::Meter(MeterError::Budget(BudgetError::Exhausted {
                dimension: actual,
                ..
            }))) if actual == dimension
        ));
        assert_eq!(output, [901.0; 4]);
    }
}

#[test]
fn sparse_structure_and_buffer_failures_stay_transactional() {
    let mut system = sparse();
    let mut output = [901.0; 6];
    system.entries[0].col = usize::MAX;
    let mut budget = Budget::new(BudgetLimits::uniform(10_000_000, 137));
    assert!(matches!(
        sparse_evaluate(&system, Layout::Dense, &mut output, &mut budget),
        Err(EvalError::InvalidSparseJacobian(_))
    ));
    assert_eq!(output, [901.0; 6]);
    let system = sparse();
    assert!(matches!(
        system.try_eval_sparse_with_meter(&[2.0, 4.0, 3.0], &mut output, &mut budget),
        Err(EvalError::JacobianBufferMismatch { .. })
    ));
    assert_eq!(output, [901.0; 6]);
}

#[test]
fn metered_objective_and_hessian_product_match_independent_closed_forms() {
    let program = objective();
    let mut budget = Budget::new(BudgetLimits::uniform(10_000_000, 137));
    let before = budget.snapshot();
    let result = program
        .evaluate_with_meter(&[2.0, 3.0], &mut budget)
        .unwrap();
    assert_eq!(result.value, 37.0);
    assert_eq!(result.gradient, [7.0, 20.0]);
    assert_eq!(result.hessian, [2.0, 1.0, 1.0, 6.0]);
    assert_eq!(
        program
            .hessian_vector_product_with_meter(&[2.0, 3.0], &[5.0, -2.0], &mut budget)
            .unwrap(),
        [8.0, -7.0]
    );
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(budget.remaining(dimension) < before.dimensions[dimension.index()]);
    }
    assert_eq!(budget.snapshot().verifier_pool, 137);
}

#[test]
fn every_objective_cancellation_keeps_both_output_buffers_unchanged() {
    let program = objective();
    let receipt = program.receipt().clone();
    let mut measured = Probe::default();
    program
        .try_eval_into_with_meter(&[2.0, 3.0], &mut [0.0; 2], &mut [0.0; 4], &mut measured)
        .unwrap();
    for checkpoint in 1..=measured.checkpoints {
        let mut cancelled = Probe {
            cancel_at: Some(checkpoint),
            ..Probe::default()
        };
        let mut gradient = [901.0; 2];
        let mut hessian = [902.0; 4];
        assert!(matches!(
            program.try_eval_into_with_meter(
                &[2.0, 3.0],
                &mut gradient,
                &mut hessian,
                &mut cancelled,
            ),
            Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
                MeterError::Cancelled
            )))
        ));
        assert_eq!(gradient, [901.0; 2]);
        assert_eq!(hessian, [902.0; 4]);
        assert_eq!(program.receipt(), &receipt);
    }
}

#[test]
fn every_product_cancellation_keeps_the_output_unchanged() {
    let program = objective();
    let mut measured = Probe::default();
    program
        .try_hessian_vector_product_into_with_meter(
            &[2.0, 3.0],
            &[5.0, -2.0],
            &mut [0.0; 2],
            &mut measured,
        )
        .unwrap();
    for checkpoint in 1..=measured.checkpoints {
        let mut cancelled = Probe {
            cancel_at: Some(checkpoint),
            ..Probe::default()
        };
        let mut output = [901.0; 2];
        assert!(matches!(
            program.try_hessian_vector_product_into_with_meter(
                &[2.0, 3.0],
                &[5.0, -2.0],
                &mut output,
                &mut cancelled,
            ),
            Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
                MeterError::Cancelled
            )))
        ));
        assert_eq!(output, [901.0; 2]);
    }
}

#[test]
fn objective_and_product_refuse_real_budget_exhaustion_before_publication() {
    let program = objective();
    for product in [false, true] {
        let mut measured = Probe::default();
        if product {
            program
                .try_hessian_vector_product_into_with_meter(
                    &[2.0, 3.0],
                    &[5.0, -2.0],
                    &mut [0.0; 2],
                    &mut measured,
                )
                .unwrap();
        } else {
            program
                .try_eval_into_with_meter(&[2.0, 3.0], &mut [0.0; 2], &mut [0.0; 4], &mut measured)
                .unwrap();
        }
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let required = measured.charged[dimension.index()];
            assert!(required > 0);
            let mut budget = budget_for(dimension, required - 1);
            let mut first = [901.0; 2];
            let mut second = [902.0; 4];
            let result = if product {
                program.try_hessian_vector_product_into_with_meter(
                    &[2.0, 3.0],
                    &[5.0, -2.0],
                    &mut first,
                    &mut budget,
                )
            } else {
                program
                    .try_eval_into_with_meter(&[2.0, 3.0], &mut first, &mut second, &mut budget)
                    .map(|_| ())
            };
            assert!(matches!(
                result,
                Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
                    MeterError::Budget(BudgetError::Exhausted { dimension: actual, .. })
                ))) if actual == dimension
            ));
            assert_eq!(first, [901.0; 2]);
            assert_eq!(second, [902.0; 4]);
            assert_eq!(budget.snapshot().verifier_pool, 137);
        }
    }
}

#[test]
fn metered_zero_direction_still_checks_original_and_derivative_domains() {
    for source in ["log(x)", "x^(1/2)", "x^(3/2)"] {
        let program =
            VerifiedObjective::try_compile(&parse(source).unwrap(), &[Symbol::new("x")]).unwrap();
        let mut budget = Budget::new(BudgetLimits::uniform(10_000_000, 137));
        let mut output = [901.0];
        assert!(
            program
                .try_hessian_vector_product_into_with_meter(
                    &[0.0],
                    &[0.0],
                    &mut output,
                    &mut budget,
                )
                .is_err(),
            "{source} lost a domain guard"
        );
        assert_eq!(output, [901.0]);
    }
}

#[test]
fn empty_sparse_and_constant_objectives_remain_cancellable() {
    let system = SparseJacobian::try_build(&[], &[]).unwrap();
    let constant = VerifiedObjective::try_compile(&Expr::from_i64(7), &[]).unwrap();
    let mut budget = Budget::new(BudgetLimits::uniform(100_000, 137));
    system
        .try_eval_sparse_with_meter(&[], &mut [], &mut budget)
        .unwrap();
    let result = constant.evaluate_with_meter(&[], &mut budget).unwrap();
    assert_eq!(result.value, 7.0);
    assert!(result.gradient.is_empty() && result.hessian.is_empty());
    assert!(
        constant
            .hessian_vector_product_with_meter(&[], &[], &mut budget)
            .unwrap()
            .is_empty()
    );
    let mut cancelled = Probe {
        cancel_at: Some(1),
        ..Probe::default()
    };
    assert!(matches!(
        system.try_eval_dense_with_meter(&[], &mut [], &mut cancelled),
        Err(EvalError::Meter(MeterError::Cancelled))
    ));
    let mut cancelled = Probe {
        cancel_at: Some(1),
        ..Probe::default()
    };
    assert!(matches!(
        constant.evaluate_with_meter(&[], &mut cancelled),
        Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Cancelled
        )))
    ));
}
