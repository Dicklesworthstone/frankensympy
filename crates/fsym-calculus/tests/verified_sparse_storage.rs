//! Sparse admission/storage with an explicitly bounded dense output boundary.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError, MAX_COMPILE_OPS};
use fsym_calculus::sparse_jacobian::MAX_SPARSE_SOURCE_WORK;
use fsym_calculus::verified::{
    MAX_VERIFIED_JACOBIAN_ENTRIES, MAX_VERIFIED_NONZERO_ENTRIES, MAX_VERIFIED_STORED_OPS,
    ResidualEvaluationError, VerifiedCompilationError, VerifiedResidualSystem,
};
use fsym_core::{Constant, Expr, Symbol, parse};

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
        assert!(!self.stopped, "work continued after cancellation");
        let mut next = self.charged;
        for &(dimension, amount) in charges {
            assert_ne!(amount, 0);
            next[dimension.index()] = next[dimension.index()]
                .checked_add(amount)
                .ok_or(BudgetError::ChargeOverflow { dimension })?;
        }
        self.charged = next;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "work continued after cancellation");
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn row_local(rows: usize, columns: usize) -> VerifiedResidualSystem {
    let variables: Vec<_> = (0..columns)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    let residuals: Vec<_> = (0..rows)
        .map(|row| Expr::Sym(variables[row % columns].clone()))
        .collect();
    VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap()
}

fn rectangular() -> VerifiedResidualSystem {
    VerifiedResidualSystem::try_compile(
        &[
            parse("x^2+y").unwrap(),
            parse("3*y").unwrap(),
            Expr::from_i64(7),
        ],
        &[Symbol::new("x"), Symbol::new("y")],
    )
    .unwrap()
}

#[test]
fn sparse_storage_transport_and_products_exceed_the_dense_shape_cap() {
    let rows = 257;
    let columns = 256;
    assert!(rows * columns > MAX_VERIFIED_JACOBIAN_ENTRIES);
    let system = row_local(rows, columns);
    assert_eq!(system.num_residuals(), rows);
    assert_eq!(system.num_vars(), columns);
    assert_eq!(system.residual_programs().len(), rows);
    assert_eq!(system.jacobian_pattern().len(), rows);
    assert_eq!(system.stored_operation_count(), 2 * rows);
    assert!(system.stored_operation_count() < rows * columns);

    let point: Vec<_> = (0..columns).map(|index| index as f64).collect();
    let values = system.evaluate_sparse(&point).unwrap();
    assert_eq!(&values.residuals[..columns], point);
    assert_eq!(values.residuals[columns], 0.0);
    assert_eq!(values.jacobian, vec![1.0; rows]);
    assert_eq!(
        system
            .jacobian_vector_product(&point, &vec![2.0; columns])
            .unwrap(),
        vec![2.0; rows]
    );
    let transposed = system
        .jacobian_transpose_vector_product(&point, &vec![3.0; rows])
        .unwrap();
    assert_eq!(transposed[0], 6.0);
    assert_eq!(&transposed[1..], vec![3.0; columns - 1]);

    let saved = system.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&saved).unwrap();
    assert_eq!(restored.stored_operation_count(), 2 * rows);
    assert_eq!(restored.sparse_jacobian(), system.sparse_jacobian());
    assert_eq!(restored.evaluate_sparse(&point).unwrap(), values);
    assert_eq!(restored.to_json_bytes().unwrap(), saved);
    assert!(matches!(
        restored.try_dense_program(),
        Err(VerifiedCompilationError::DenseJacobianLimit { .. })
    ));
}

#[test]
fn dense_shape_refusal_precedes_meter_activity_and_preserves_caller_buffers() {
    let system = row_local(257, 256);
    let expected = ResidualEvaluationError::DenseJacobianLimit {
        entries: 257 * 256,
        max: MAX_VERIFIED_JACOBIAN_ENTRIES,
    };
    let mut meter = Probe {
        cancel_at: Some(1),
        ..Probe::default()
    };
    let mut residuals = [901.0];
    let mut jacobian = [902.0];
    assert_eq!(
        system
            .try_eval_dense_with_meter(&[f64::NAN], &mut meter)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        system
            .try_eval_dense_into_with_meter(&[], &mut residuals, &mut jacobian, &mut meter)
            .unwrap_err(),
        expected
    );
    assert_eq!(meter.checkpoints, 0);
    assert_eq!(meter.charged, [0; DIMENSION_COUNT]);
    assert_eq!(residuals, [901.0]);
    assert_eq!(jacobian, [902.0]);
}

#[test]
fn explicit_dense_values_and_legacy_program_match_hand_derivatives() {
    let system = rectangular();
    let result = system.try_eval_dense(&[2.0, 5.0]).unwrap();
    assert_eq!(result.residuals, [9.0, 15.0, 7.0]);
    assert_eq!(result.jacobian, [4.0, 1.0, 0.0, 3.0, 0.0, 0.0]);
    assert_eq!(system.dense_jacobian_len().unwrap(), 6);
    let mut legacy = system.try_dense_program().unwrap();
    let mut residuals = [0.0; 3];
    let mut jacobian = [0.0; 6];
    legacy
        .try_eval_system(&[2.0, 5.0], &mut residuals, &mut jacobian)
        .unwrap();
    assert_eq!(residuals, result.residuals.as_slice());
    assert_eq!(jacobian, result.jacobian.as_slice());
    legacy.compiled_jacobian[0] =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(99.0)]).unwrap();
    assert_eq!(system.try_eval_dense(&[2.0, 5.0]).unwrap(), result);
}

#[test]
fn dense_original_and_derivative_domains_are_strict_and_atomic() {
    let x = Symbol::new("x");
    let sources = [
        (
            Expr::Mul(vec![
                Expr::from_i64(0),
                Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
            ]),
            -1.0,
        ),
        (Expr::Function("log".into(), vec![Expr::from_i64(-1)]), 2.0),
        (parse("x^(1/2)").unwrap(), 0.0),
        (parse("x^2").unwrap(), f64::MAX),
    ];
    for (source, point) in sources {
        let system =
            VerifiedResidualSystem::try_compile(&[source], std::slice::from_ref(&x)).unwrap();
        let mut residual = [901.0];
        let mut jacobian = [902.0];
        assert_eq!(
            system
                .try_eval_dense_into(&[point], &mut residual, &mut jacobian)
                .unwrap_err(),
            ResidualEvaluationError::Evaluation(EvalError::NonFiniteValue),
        );
        assert_eq!(residual, [901.0]);
        assert_eq!(jacobian, [902.0]);
    }
    let no_derivatives = VerifiedResidualSystem::try_compile(
        &[Expr::Function("log".into(), vec![Expr::from_i64(-1)])],
        &[x],
    )
    .unwrap();
    assert!(no_derivatives.jacobian_pattern().is_empty());
    assert!(no_derivatives.try_eval_dense(&[0.0]).is_err());
}

#[test]
fn dense_dimension_and_nonfinite_point_errors_preserve_both_outputs() {
    let system = rectangular();
    for point in [vec![2.0], vec![2.0, f64::NAN], vec![2.0, f64::INFINITY]] {
        let mut residuals = [901.0; 3];
        let mut jacobian = [902.0; 6];
        assert!(
            system
                .try_eval_dense_into(&point, &mut residuals, &mut jacobian)
                .is_err()
        );
        assert_eq!(residuals, [901.0; 3]);
        assert_eq!(jacobian, [902.0; 6]);
    }
    let mut residuals = [901.0; 3];
    let mut jacobian = [902.0; 6];
    assert!(matches!(
        system.try_eval_dense_into(&[2.0, 5.0], &mut residuals, &mut jacobian[..5]),
        Err(ResidualEvaluationError::Dimension {
            lane: "dense Jacobian output",
            expected: 6,
            actual: 5
        })
    ));
    assert_eq!(residuals, [901.0; 3]);
    assert_eq!(jacobian, [902.0; 6]);
}

#[test]
fn every_dense_cancellation_checkpoint_precedes_atomic_publication() {
    let system = rectangular();
    let mut measured = Probe::default();
    system
        .try_eval_dense_into_with_meter(&[2.0, 5.0], &mut [0.0; 3], &mut [0.0; 6], &mut measured)
        .unwrap();
    assert!(measured.checkpoints > 10);
    for checkpoint in 1..=measured.checkpoints {
        let mut cancelled = Probe {
            cancel_at: Some(checkpoint),
            ..Probe::default()
        };
        let mut residuals = [901.0; 3];
        let mut jacobian = [902.0; 6];
        assert_eq!(
            system
                .try_eval_dense_into_with_meter(
                    &[2.0, 5.0],
                    &mut residuals,
                    &mut jacobian,
                    &mut cancelled
                )
                .unwrap_err(),
            ResidualEvaluationError::Evaluation(EvalError::Meter(MeterError::Cancelled))
        );
        assert_eq!(residuals, [901.0; 3], "checkpoint {checkpoint}");
        assert_eq!(jacobian, [902.0; 6], "checkpoint {checkpoint}");
    }
}

#[test]
fn dense_budgets_are_cumulative_and_one_short_refuses_without_publication() {
    let system = rectangular();
    let mut measured = Probe::default();
    system
        .try_eval_dense_into_with_meter(&[2.0, 5.0], &mut [0.0; 3], &mut [0.0; 6], &mut measured)
        .unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let used = measured.charged[dimension.index()];
        assert!(used > 0);
        for allowance in [used, used - 1] {
            let mut limits = BudgetLimits::uniform(10_000_000, 137);
            limits.dimensions[dimension.index()] = allowance;
            let mut meter = Budget::new(limits);
            if allowance == used {
                system
                    .try_eval_dense_into_with_meter(
                        &[2.0, 5.0],
                        &mut [0.0; 3],
                        &mut [0.0; 6],
                        &mut meter,
                    )
                    .unwrap();
                assert_eq!(meter.remaining(dimension), 0);
            }
            let mut residuals = [901.0; 3];
            let mut jacobian = [902.0; 6];
            assert!(
                matches!(system.try_eval_dense_into_with_meter(&[2.0, 5.0], &mut residuals, &mut jacobian, &mut meter), Err(ResidualEvaluationError::Evaluation(EvalError::Meter(MeterError::Budget(BudgetError::Exhausted { dimension: actual, .. })))) if actual == dimension)
            );
            assert_eq!(residuals, [901.0; 3]);
            assert_eq!(jacobian, [902.0; 6]);
            assert_eq!(meter.snapshot().verifier_pool, 137);
        }
    }
}

#[test]
fn empty_axes_are_shape_correct_and_original_constant_sources_still_run() {
    let empty = VerifiedResidualSystem::try_compile(&[], &[]).unwrap();
    empty.try_eval_dense_into(&[], &mut [], &mut []).unwrap();
    let no_rows = VerifiedResidualSystem::try_compile(&[], &[Symbol::new("unused")]).unwrap();
    assert!(no_rows.try_eval_dense(&[f64::NAN]).is_err());
    assert!(no_rows.try_eval_dense(&[2.0]).unwrap().jacobian.is_empty());
    let no_columns = VerifiedResidualSystem::try_compile(&[Expr::from_i64(7)], &[]).unwrap();
    let mut residuals = [0.0];
    no_columns
        .try_eval_dense_into(&[], &mut residuals, &mut [])
        .unwrap();
    assert_eq!(residuals, [7.0]);
    assert_eq!(no_columns.stored_operation_count(), 1);
    let invalid = VerifiedResidualSystem::try_compile(&[parse("log(-1)").unwrap()], &[]).unwrap();
    residuals[0] = 901.0;
    assert!(
        invalid
            .try_eval_dense_into(&[], &mut residuals, &mut [])
            .is_err()
    );
    assert_eq!(residuals, [901.0]);
}

#[test]
fn aggregate_source_and_candidate_coordinate_limits_precede_derivative_generation() {
    let wide_source = Expr::Add(vec![Expr::Const(Constant::Pi); MAX_COMPILE_OPS - 1]);
    let excessive_sources = vec![wide_source; MAX_SPARSE_SOURCE_WORK / MAX_COMPILE_OPS + 1];
    assert!(matches!(
        VerifiedResidualSystem::try_compile(&excessive_sources, &[]),
        Err(VerifiedCompilationError::AdmissionLimit {
            resource: "source nodes",
            ..
        })
    ));
    drop(excessive_sources);

    let variables = [Symbol::new("x"), Symbol::new("y")];
    let source = Expr::Add(variables.iter().cloned().map(Expr::Sym).collect());
    let residuals = vec![source; MAX_VERIFIED_NONZERO_ENTRIES / 2 + 1];
    assert!(matches!(
        VerifiedResidualSystem::try_compile(&residuals, &variables),
        Err(VerifiedCompilationError::AdmissionLimit {
            resource: "potential derivative coordinates",
            ..
        })
    ));
}

#[test]
fn untrusted_retained_nonzeros_and_bytecode_are_bounded_before_proof_replay() {
    let original = row_local(MAX_VERIFIED_STORED_OPS / MAX_COMPILE_OPS + 1, 1);
    let mut oversized = original.sparse_jacobian().clone();
    let mut ops = vec![CompiledOp::Neg; MAX_COMPILE_OPS];
    ops[0] = CompiledOp::LoadConst(0.0);
    let filler = CompiledExpr::try_from_ops(ops).unwrap();
    for entry in &mut oversized.entries {
        entry.compiled_deriv = filler.clone();
    }
    assert!(matches!(
        VerifiedResidualSystem::try_from_sparse_jacobian(&oversized),
        Err(VerifiedCompilationError::AdmissionLimit {
            resource: "stored bytecode operations",
            ..
        })
    ));
    let mut malformed = original.sparse_jacobian().clone();
    malformed.pattern.nonzeros = vec![(0, 0); MAX_VERIFIED_NONZERO_ENTRIES + 1];
    assert!(matches!(
        VerifiedResidualSystem::try_from_sparse_jacobian(&malformed),
        Err(VerifiedCompilationError::AdmissionLimit {
            resource: "nonzero entries",
            ..
        })
    ));
}
