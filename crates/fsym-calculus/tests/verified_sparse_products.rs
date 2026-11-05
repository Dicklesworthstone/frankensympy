//! Checked sparse residual values, directional products and atomic budget use.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};

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
            assert_ne!(amount, 0, "the real ledger refuses zero charges");
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

fn rectangular() -> VerifiedResidualSystem {
    let residuals = ["x^2+y", "z^2+y", "7", "x-2*z"].map(|source| parse(source).unwrap());
    VerifiedResidualSystem::try_compile(
        &residuals,
        &[Symbol::new("x"), Symbol::new("y"), Symbol::new("z")],
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Sparse,
    Forward,
    Transpose,
}

fn evaluate<M: BudgetMeter + ?Sized>(
    program: &VerifiedResidualSystem,
    operation: Operation,
    first: &mut [f64],
    second: &mut [f64],
    meter: &mut M,
) -> Result<(), ResidualEvaluationError> {
    match operation {
        Operation::Sparse => {
            program.try_eval_sparse_into_with_meter(&[2.0, 4.0, 3.0], first, second, meter)
        }
        Operation::Forward => program.try_jacobian_vector_product_into_with_meter(
            &[2.0, 4.0, 3.0],
            &[5.0, -2.0, 7.0],
            first,
            meter,
        ),
        Operation::Transpose => program.try_jacobian_transpose_vector_product_into_with_meter(
            &[2.0, 4.0, 3.0],
            &[3.0, -2.0, 7.0, 5.0],
            first,
            meter,
        ),
    }
}

fn output_length(operation: Operation) -> usize {
    match operation {
        Operation::Sparse | Operation::Forward => 4,
        Operation::Transpose => 3,
    }
}

fn budget_for(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(10_000_000, 137);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn rectangular_values_and_both_products_match_independent_hand_derivatives() {
    let program = rectangular();
    assert_eq!(
        program.jacobian_pattern(),
        [(0, 0), (0, 1), (1, 1), (1, 2), (3, 0), (3, 2)]
    );
    let result = program.evaluate_sparse(&[2.0, 4.0, 3.0]).unwrap();
    assert_eq!(result.residuals, [8.0, 13.0, 7.0, -4.0]);
    assert_eq!(result.jacobian, [4.0, 1.0, 1.0, 6.0, 1.0, -2.0]);
    let forward = program
        .jacobian_vector_product(&[2.0, 4.0, 3.0], &[5.0, -2.0, 7.0])
        .unwrap();
    assert_eq!(forward, [18.0, 40.0, 0.0, -9.0]);
    let transpose = program
        .jacobian_transpose_vector_product(&[2.0, 4.0, 3.0], &[3.0, -2.0, 7.0, 5.0])
        .unwrap();
    assert_eq!(transpose, [17.0, 1.0, -22.0]);
    let left: f64 = forward
        .iter()
        .zip([3.0, -2.0, 7.0, 5.0])
        .map(|(a, b)| a * b)
        .sum();
    let right: f64 = transpose
        .iter()
        .zip([5.0, -2.0, 7.0])
        .map(|(a, b)| a * b)
        .sum();
    assert_eq!(left, -71.0);
    assert_eq!(left, right);

    // Structural entries stay present when their numerical value vanishes.
    let at_zero = program.evaluate_sparse(&[0.0, 4.0, 0.0]).unwrap();
    assert_eq!(at_zero.jacobian, [0.0, 1.0, 1.0, 0.0, 1.0, -2.0]);
    assert_eq!(at_zero.jacobian.len(), program.jacobian_pattern().len());
}

#[test]
fn typed_same_named_variables_keep_independent_product_coordinates() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let source = Expr::Add(vec![
        Expr::Sym(first.clone()),
        Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(second.clone())]),
    ]);
    let program = VerifiedResidualSystem::try_compile(&[source], &[first, second]).unwrap();
    let result = program.evaluate_sparse(&[3.0, 7.0]).unwrap();
    assert_eq!(result.residuals, [17.0]);
    assert_eq!(result.jacobian, [1.0, 2.0]);
    assert_eq!(
        program
            .jacobian_vector_product(&[3.0, 7.0], &[5.0, 11.0])
            .unwrap(),
        [27.0]
    );
    assert_eq!(
        program
            .jacobian_transpose_vector_product(&[3.0, 7.0], &[4.0])
            .unwrap(),
        [4.0, 8.0]
    );
}

#[test]
fn sparse_products_use_only_the_admitted_bytecode_and_survive_transport() {
    let program = rectangular();
    let restored =
        VerifiedResidualSystem::from_json_bytes(&program.to_json_bytes().unwrap()).unwrap();
    assert_eq!(restored.jacobian_pattern(), program.jacobian_pattern());
    assert_eq!(
        restored.evaluate_sparse(&[2.0, 4.0, 3.0]).unwrap(),
        program.evaluate_sparse(&[2.0, 4.0, 3.0]).unwrap()
    );
    let mut forged = program.sparse_jacobian().clone();
    forged.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(999.0)]).unwrap();
    assert!(VerifiedResidualSystem::try_from_sparse_jacobian(&forged).is_err());
    assert_eq!(
        restored
            .jacobian_vector_product(&[2.0, 4.0, 3.0], &[5.0, -2.0, 7.0])
            .unwrap(),
        [18.0, 40.0, 0.0, -9.0]
    );
    assert_eq!(
        program
            .jacobian_transpose_vector_product(&[2.0, 4.0, 3.0], &[3.0, -2.0, 7.0, 5.0])
            .unwrap(),
        [17.0, 1.0, -22.0]
    );
}

#[test]
fn held_zero_sources_keep_their_original_domain_under_zero_weight_products() {
    let x = Symbol::new("x");
    let source = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
    ]);
    let program = VerifiedResidualSystem::try_compile(&[source], &[x.clone()]).unwrap();
    // Held derivative expressions are structural entries even when their
    // valid-point value is zero; sparsity never proves algebraic cancellation.
    assert_eq!(program.jacobian_pattern(), &[(0, 0)]);
    let evaluated = program.evaluate_sparse(&[1.0]).unwrap();
    assert_eq!(evaluated.residuals, [0.0]);
    assert_eq!(evaluated.jacobian, [0.0]);
    for point in [-1.0, 0.0] {
        let mut residual = [901.0];
        let mut jacobian = [903.0];
        assert!(matches!(
            program.try_eval_sparse_into(&[point], &mut residual, &mut jacobian),
            Err(ResidualEvaluationError::Evaluation(
                EvalError::NonFiniteValue
            ))
        ));
        assert_eq!(residual, [901.0]);
        assert_eq!(jacobian, [903.0]);
        for transpose in [false, true] {
            let mut output = [902.0];
            let result = if transpose {
                program.try_jacobian_transpose_vector_product_into(&[point], &[0.0], &mut output)
            } else {
                program.try_jacobian_vector_product_into(&[point], &[0.0], &mut output)
            };
            assert!(matches!(
                result,
                Err(ResidualEvaluationError::Evaluation(
                    EvalError::NonFiniteValue
                ))
            ));
            assert_eq!(output, [902.0]);
        }
    }

    // An invalid constant source has no derivative coordinates at all. Its
    // original domain must still run before publishing any sparse or product
    // result, despite the unused finite point and zero direction/weight.
    let constant_hole = Expr::Function("log".into(), vec![Expr::from_i64(-1)]);
    let no_derivatives = VerifiedResidualSystem::try_compile(&[constant_hole], &[x]).unwrap();
    assert!(no_derivatives.jacobian_pattern().is_empty());
    let mut residual = [904.0];
    assert!(matches!(
        no_derivatives.try_eval_sparse_into(&[1.0], &mut residual, &mut []),
        Err(ResidualEvaluationError::Evaluation(
            EvalError::NonFiniteValue
        ))
    ));
    assert_eq!(residual, [904.0]);
    for transpose in [false, true] {
        let mut output = [905.0];
        let result = if transpose {
            no_derivatives.try_jacobian_transpose_vector_product_into(&[1.0], &[0.0], &mut output)
        } else {
            no_derivatives.try_jacobian_vector_product_into(&[1.0], &[0.0], &mut output)
        };
        assert!(matches!(
            result,
            Err(ResidualEvaluationError::Evaluation(
                EvalError::NonFiniteValue
            ))
        ));
        assert_eq!(output, [905.0]);
    }
}

#[test]
fn zero_vectors_cannot_hide_unweighted_source_or_derivative_domains() {
    for (source, point) in [("log(y)", [2.0, -1.0]), ("y^(1/2)", [2.0, 0.0])] {
        let program = VerifiedResidualSystem::try_compile(
            &[Expr::symbol("x"), parse(source).unwrap()],
            &[Symbol::new("x"), Symbol::new("y")],
        )
        .unwrap();
        assert!(program.evaluate_sparse(&point).is_err());
        assert!(
            program
                .jacobian_vector_product(&point, &[1.0, 0.0])
                .is_err()
        );
        assert!(
            program
                .jacobian_transpose_vector_product(&point, &[1.0, 0.0])
                .is_err()
        );
    }
}

#[test]
fn every_point_and_vector_coordinate_is_dimension_checked_and_finite() {
    let program = rectangular();
    assert!(matches!(
        program.evaluate_sparse(&[2.0, 4.0]),
        Err(ResidualEvaluationError::Dimension {
            lane: "point",
            expected: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        program.jacobian_vector_product(&[2.0, 4.0, 3.0], &[1.0, 2.0]),
        Err(ResidualEvaluationError::Dimension {
            lane: "direction",
            expected: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        program.jacobian_transpose_vector_product(&[2.0, 4.0, 3.0], &[1.0, 2.0, 3.0]),
        Err(ResidualEvaluationError::Dimension {
            lane: "weights",
            expected: 4,
            actual: 3
        })
    ));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(program.evaluate_sparse(&[2.0, 4.0, bad]).is_err());
        assert!(
            program
                .jacobian_vector_product(&[2.0, 4.0, 3.0], &[0.0, 0.0, bad])
                .is_err()
        );
        assert!(
            program
                .jacobian_transpose_vector_product(&[2.0, 4.0, 3.0], &[0.0, 0.0, bad, 0.0])
                .is_err()
        );
    }
    let empty = VerifiedResidualSystem::try_compile(&[], &[Symbol::new("unused")]).unwrap();
    assert!(empty.evaluate_sparse(&[f64::NAN]).is_err());
    assert!(
        empty
            .jacobian_vector_product(&[0.0], &[f64::INFINITY])
            .is_err()
    );
    assert!(
        empty
            .jacobian_transpose_vector_product(&[f64::NAN], &[])
            .is_err()
    );
}

#[test]
fn multiplication_and_partial_sum_overflow_are_refused_atomically() {
    let scalar =
        VerifiedResidualSystem::try_compile(&[parse("2*x").unwrap()], &[Symbol::new("x")]).unwrap();
    let mut output = [901.0];
    assert!(
        scalar
            .try_jacobian_vector_product_into(&[0.0], &[f64::MAX], &mut output)
            .is_err()
    );
    assert_eq!(output, [901.0]);
    assert!(
        scalar
            .try_jacobian_transpose_vector_product_into(&[0.0], &[f64::MAX], &mut output)
            .is_err()
    );
    assert_eq!(output, [901.0]);

    // The final exact expression could be finite, but a non-finite partial
    // sum is not allowed to be concealed by a later subtraction.
    let forward = VerifiedResidualSystem::try_compile(
        &[parse("x+y-z").unwrap()],
        &[Symbol::new("x"), Symbol::new("y"), Symbol::new("z")],
    )
    .unwrap();
    assert!(
        forward
            .try_jacobian_vector_product_into(&[0.0; 3], &[f64::MAX; 3], &mut output)
            .is_err()
    );
    assert_eq!(output, [901.0]);
    let transpose = VerifiedResidualSystem::try_compile(
        &[
            parse("x").unwrap(),
            parse("x").unwrap(),
            parse("-x").unwrap(),
        ],
        &[Symbol::new("x")],
    )
    .unwrap();
    assert!(
        transpose
            .try_jacobian_transpose_vector_product_into(&[0.0], &[f64::MAX; 3], &mut output)
            .is_err()
    );
    assert_eq!(output, [901.0]);
}

#[test]
fn invalid_output_lengths_never_publish_another_valid_buffer() {
    let program = rectangular();
    let mut residual = [901.0; 4];
    let mut jacobian = [902.0; 6];
    assert!(
        program
            .try_eval_sparse_into(&[2.0, 4.0, 3.0], &mut residual[..3], &mut jacobian)
            .is_err()
    );
    assert_eq!(residual, [901.0; 4]);
    assert_eq!(jacobian, [902.0; 6]);
    assert!(
        program
            .try_eval_sparse_into(&[2.0, 4.0, 3.0], &mut residual, &mut jacobian[..5])
            .is_err()
    );
    assert_eq!(residual, [901.0; 4]);
    assert_eq!(jacobian, [902.0; 6]);
    assert!(
        program
            .try_jacobian_vector_product_into(
                &[2.0, 4.0, 3.0],
                &[5.0, -2.0, 7.0],
                &mut residual[..3],
            )
            .is_err()
    );
    assert!(
        program
            .try_jacobian_transpose_vector_product_into(
                &[2.0, 4.0, 3.0],
                &[3.0, -2.0, 7.0, 5.0],
                &mut residual,
            )
            .is_err()
    );
    assert_eq!(residual, [901.0; 4]);
}

#[test]
fn every_cancellation_safe_point_preserves_all_caller_owned_outputs() {
    let program = rectangular();
    let receipt = program.receipt().clone();
    for operation in [Operation::Sparse, Operation::Forward, Operation::Transpose] {
        let length = output_length(operation);
        let mut measured = Probe::default();
        evaluate(
            &program,
            operation,
            &mut vec![0.0; length],
            &mut [0.0; 6],
            &mut measured,
        )
        .unwrap();
        assert!(measured.checkpoints > program.jacobian_pattern().len());
        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = Probe {
                cancel_at: Some(checkpoint),
                ..Probe::default()
            };
            let mut first = vec![901.0; length];
            let mut second = [902.0; 6];
            assert!(matches!(
                evaluate(&program, operation, &mut first, &mut second, &mut cancelled),
                Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
                    MeterError::Cancelled
                )))
            ));
            assert_eq!(first, vec![901.0; length], "{operation:?} at {checkpoint}");
            assert_eq!(second, [902.0; 6], "{operation:?} at {checkpoint}");
            assert_eq!(program.receipt(), &receipt);
        }
    }
}

#[test]
fn real_budgets_are_cumulative_and_never_consume_the_verifier_reservation() {
    let program = rectangular();
    for operation in [Operation::Sparse, Operation::Forward, Operation::Transpose] {
        let length = output_length(operation);
        let mut measured = Probe::default();
        evaluate(
            &program,
            operation,
            &mut vec![0.0; length],
            &mut [0.0; 6],
            &mut measured,
        )
        .unwrap();
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let required = measured.charged[dimension.index()];
            assert!(required > 0);
            let mut budget = budget_for(dimension, required);
            let mut first = vec![0.0; length];
            let mut second = [0.0; 6];
            evaluate(&program, operation, &mut first, &mut second, &mut budget).unwrap();
            assert_eq!(budget.remaining(dimension), 0);
            first.fill(901.0);
            second.fill(902.0);
            assert!(matches!(
                evaluate(&program, operation, &mut first, &mut second, &mut budget),
                Err(ResidualEvaluationError::Evaluation(EvalError::Meter(MeterError::Budget(
                    BudgetError::Exhausted { dimension: actual, .. }
                )))) if actual == dimension
            ));
            assert_eq!(first, vec![901.0; length]);
            assert_eq!(second, [902.0; 6]);
            assert_eq!(budget.snapshot().verifier_pool, 137);

            let mut short = budget_for(dimension, required - 1);
            assert!(matches!(
                evaluate(&program, operation, &mut first, &mut second, &mut short),
                Err(ResidualEvaluationError::Evaluation(EvalError::Meter(MeterError::Budget(
                    BudgetError::Exhausted { dimension: actual, .. }
                )))) if actual == dimension
            ));
            assert_eq!(first, vec![901.0; length]);
            assert_eq!(second, [902.0; 6]);
            assert_eq!(short.snapshot().verifier_pool, 137);
        }
    }
}

#[test]
fn sparse_evaluation_and_products_fit_a_linear_output_storage_budget() {
    let variables: Vec<_> = (0..128)
        .map(|index| Symbol::new(format!("x{index}").as_str()))
        .collect();
    let residuals: Vec<_> = variables.iter().cloned().map(Expr::Sym).collect();
    let program = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
    let point = vec![2.0; 128];
    let direction = vec![3.0; 128];
    // The dense output alone needs 128*128*8 bytes. This allowance covers
    // the actual sparse output plus every scalar evaluator's stack instead.
    let mut budget = budget_for(Dimension::MemoryBytes, 8192);
    let result = program
        .evaluate_sparse_with_meter(&point, &mut budget)
        .unwrap();
    assert_eq!(result.residuals, point);
    assert_eq!(result.jacobian, vec![1.0; 128]);
    assert_eq!(program.jacobian_pattern().len(), 128);
    for transpose in [false, true] {
        let mut budget = budget_for(Dimension::MemoryBytes, 8192);
        let result = if transpose {
            program.jacobian_transpose_vector_product_with_meter(&point, &direction, &mut budget)
        } else {
            program.jacobian_vector_product_with_meter(&point, &direction, &mut budget)
        }
        .unwrap();
        assert_eq!(result, direction);
    }
}

#[test]
fn empty_and_zero_variable_systems_remain_cancellable_and_shape_correct() {
    let empty = VerifiedResidualSystem::try_compile(&[], &[]).unwrap();
    let result = empty.evaluate_sparse(&[]).unwrap();
    assert!(result.residuals.is_empty() && result.jacobian.is_empty());
    assert!(empty.jacobian_vector_product(&[], &[]).unwrap().is_empty());
    assert!(
        empty
            .jacobian_transpose_vector_product(&[], &[])
            .unwrap()
            .is_empty()
    );
    for operation in [Operation::Sparse, Operation::Forward, Operation::Transpose] {
        let mut cancelled = Probe {
            cancel_at: Some(1),
            ..Probe::default()
        };
        let result = match operation {
            Operation::Sparse => {
                empty.try_eval_sparse_into_with_meter(&[], &mut [], &mut [], &mut cancelled)
            }
            Operation::Forward => {
                empty.try_jacobian_vector_product_into_with_meter(&[], &[], &mut [], &mut cancelled)
            }
            Operation::Transpose => empty.try_jacobian_transpose_vector_product_into_with_meter(
                &[],
                &[],
                &mut [],
                &mut cancelled,
            ),
        };
        assert!(matches!(
            result,
            Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
                MeterError::Cancelled
            )))
        ));
    }
    let constants =
        VerifiedResidualSystem::try_compile(&[Expr::from_i64(7), Expr::from_i64(0)], &[]).unwrap();
    assert_eq!(
        constants.evaluate_sparse(&[]).unwrap().residuals,
        [7.0, 0.0]
    );
    assert_eq!(
        constants.jacobian_vector_product(&[], &[]).unwrap(),
        [0.0, 0.0]
    );
    assert!(
        constants
            .jacobian_transpose_vector_product(&[], &[3.0, 4.0])
            .unwrap()
            .is_empty()
    );
    let no_residuals = VerifiedResidualSystem::try_compile(&[], &[Symbol::new("unused")]).unwrap();
    assert!(
        no_residuals
            .jacobian_vector_product(&[2.0], &[3.0])
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        no_residuals
            .jacobian_transpose_vector_product(&[2.0], &[])
            .unwrap(),
        [0.0]
    );
}
