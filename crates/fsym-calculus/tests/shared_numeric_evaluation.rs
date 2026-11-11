//! Shared evaluation preserves exact reference bytecode, lazy domains and fuel.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::{CompileError, CompiledBatch, CompiledExpr, CompiledOp, EvalError};
use fsym_core::{Expr, Symbol, SymbolIdentity};
use std::collections::HashMap;

fn program(operations: Vec<CompiledOp>) -> CompiledExpr {
    CompiledExpr::try_from_ops(operations).unwrap()
}

fn reference(programs: &[CompiledExpr], point: &[f64]) -> Result<Vec<f64>, EvalError> {
    programs
        .iter()
        .map(|program| program.try_eval_finite(point))
        .collect()
}

fn assert_reference(batch: &CompiledBatch, points: &[Vec<f64>]) {
    for point in points {
        let expected = reference(batch.programs(), point);
        let actual = batch.try_eval_finite(point);
        match (actual, expected) {
            (Ok(actual), Ok(expected)) => assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "point {point:?}",
            ),
            (Err(actual), Err(expected)) => assert_eq!(actual, expected, "point {point:?}"),
            (actual, expected) => {
                panic!("different outcomes at {point:?}: {actual:?} != {expected:?}")
            }
        }
    }
}

#[test]
fn shared_nested_work_executes_once_across_outputs_and_preserves_reference_bits() {
    let mut repeated = vec![CompiledOp::LoadVar(0)];
    for _ in 0..8 {
        repeated.extend([CompiledOp::Sin, CompiledOp::Cos]);
    }
    let scalar = program(repeated);
    let batch = CompiledBatch::try_from_programs(vec![scalar.clone(); 32]).unwrap();
    assert!(batch.shared_subexpression_count() >= 16);
    assert_reference(
        &batch,
        &[
            vec![0.0],
            vec![-0.0],
            vec![0.125],
            vec![-25.0],
            vec![f64::NAN],
        ],
    );

    let mut shared = Budget::new(BudgetLimits::uniform(10_000_000, 37));
    let mut separate = Budget::new(BudgetLimits::uniform(10_000_000, 37));
    batch
        .try_eval_finite_with_meter(&[0.125], &mut shared)
        .unwrap();
    for original in batch.programs() {
        original
            .try_eval_finite_with_meter(&[0.125], &mut separate)
            .unwrap();
    }
    assert!(
        shared.remaining(Dimension::ComputeSteps) > separate.remaining(Dimension::ComputeSteps)
    );
    assert_eq!(shared.verifier_remaining(), 37);
    assert_eq!(batch.programs()[0].ops(), scalar.ops());
}

#[test]
fn compound_reuse_preserves_reduction_order_and_signed_zero() {
    let positive = program(vec![CompiledOp::LoadVar(0), CompiledOp::Sin]);
    let mut left = positive.ops().to_vec();
    left.extend([
        CompiledOp::LoadConst(1e16),
        CompiledOp::LoadConst(-1e16),
        CompiledOp::Add(3),
    ]);
    let mut right = positive.ops().to_vec();
    right.extend([
        CompiledOp::LoadConst(-1e16),
        CompiledOp::Add(2),
        CompiledOp::LoadConst(1e16),
        CompiledOp::Add(2),
    ]);
    let plus_zero = program(vec![CompiledOp::LoadVar(0), CompiledOp::Pow(0.0)]);
    let minus_zero = program(vec![CompiledOp::LoadVar(0), CompiledOp::Pow(-0.0)]);
    let batch = CompiledBatch::try_from_programs(vec![
        positive.clone(),
        positive,
        program(left),
        program(right),
        plus_zero,
        minus_zero,
    ])
    .unwrap();
    assert_reference(
        &batch,
        &[
            vec![0.0],
            vec![-0.0],
            vec![1.0],
            vec![-1.0],
            vec![f64::INFINITY],
        ],
    );
    assert_eq!(
        batch.try_eval_finite(&[-0.0]).unwrap()[0].to_bits(),
        (-0.0f64).to_bits()
    );
}

fn guarded_log(positive_first: bool) -> CompiledExpr {
    // if x > 0 then log(x), else 7; reversed form keeps log inactive on x<=0.
    let comparison = if positive_first {
        CompiledOp::Greater
    } else {
        CompiledOp::LessEqual
    };
    let mut code = vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadConst(0.0),
        comparison,
        CompiledOp::JumpIfFalse(0),
    ];
    if positive_first {
        code.extend([CompiledOp::LoadVar(0), CompiledOp::Ln]);
    } else {
        code.push(CompiledOp::LoadConst(7.0));
    }
    let end = code.len();
    code.push(CompiledOp::Jump(0));
    code[3] = CompiledOp::JumpIfFalse(code.len());
    if positive_first {
        code.push(CompiledOp::LoadConst(7.0));
    } else {
        code.extend([CompiledOp::LoadVar(0), CompiledOp::Ln]);
    }
    code[end] = CompiledOp::Jump(code.len());
    program(code)
}

#[test]
fn inactive_domains_and_alternative_branch_entry_are_never_hoisted() {
    let left = guarded_log(true);
    let right = guarded_log(false);
    let batch =
        CompiledBatch::try_from_programs(vec![left.clone(), right.clone(), left, right]).unwrap();
    assert!(batch.shared_subexpression_count() > 0);
    assert_reference(&batch, &[vec![-3.0], vec![0.0], vec![2.0], vec![10.0]]);
    assert_eq!(batch.try_eval_finite(&[-3.0]).unwrap(), vec![7.0; 4]);
}

#[test]
fn cache_is_local_to_the_input_and_does_not_mask_later_source_failure() {
    let guarded = guarded_log(true);
    let raw = program(vec![CompiledOp::LoadVar(0), CompiledOp::Ln]);
    let batch = CompiledBatch::try_from_programs(vec![guarded, raw.clone(), raw]).unwrap();
    assert_reference(
        &batch,
        &[vec![2.0], vec![-1.0], vec![4.0], vec![0.0], vec![1.0]],
    );
    assert_eq!(
        batch.try_eval_finite(&[0.0]),
        Err(EvalError::NonFiniteValue)
    );
}

#[test]
fn nested_conditional_expression_preserves_the_enclosing_operand_stack() {
    let branch = guarded_log(true);
    let mut code = vec![CompiledOp::LoadConst(13.0)];
    for operation in branch.ops() {
        code.push(match operation {
            CompiledOp::Jump(target) => CompiledOp::Jump(target + 1),
            CompiledOp::JumpIfFalse(target) => CompiledOp::JumpIfFalse(target + 1),
            operation => operation.clone(),
        });
    }
    code.push(CompiledOp::Mul(2));
    let scalar = program(code);
    let batch = CompiledBatch::try_from_programs(vec![scalar.clone(), scalar]).unwrap();
    assert_reference(&batch, &[vec![-1.0], vec![0.0], vec![1.0], vec![5.0]]);
}

#[test]
fn source_checked_programs_keep_same_name_typed_axes_distinct() {
    let x = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let map = HashMap::from([(x.clone(), 0), (other.clone(), 1)]);
    let one = Expr::Function("sin".into(), vec![Expr::Sym(x)]);
    let two = Expr::Function("sin".into(), vec![Expr::Sym(other)]);
    let first = CompiledExpr::try_compile_checked_with_meter(&one, &map, &mut Unbounded).unwrap();
    let second = CompiledExpr::try_compile_checked_with_meter(&two, &map, &mut Unbounded).unwrap();
    let batch =
        CompiledBatch::try_from_programs(vec![first.clone(), second.clone(), first, second])
            .unwrap();
    assert_reference(&batch, &[vec![0.0, 1.0], vec![1.0, 0.0], vec![-0.0, 0.0]]);
}

struct Interrupt {
    steps: usize,
    stop: Option<usize>,
}
impl BudgetMeter for Interrupt {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.steps += 1;
        if self.stop == Some(self.steps) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_plan_and_evaluation_safe_point_publishes_no_value() {
    let scalar = program(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::Sin,
        CompiledOp::Cos,
    ]);
    let originals = vec![scalar.clone(), scalar];
    let mut count = Interrupt {
        steps: 0,
        stop: None,
    };
    let batch = CompiledBatch::try_from_programs_with_meter(originals.clone(), &mut count).unwrap();
    for stop in 1..=count.steps {
        assert!(
            matches!(
                CompiledBatch::try_from_programs_with_meter(
                    originals.clone(),
                    &mut Interrupt {
                        steps: 0,
                        stop: Some(stop)
                    }
                ),
                Err(CompileError::Meter(MeterError::Cancelled)),
            ),
            "planning safe point {stop}"
        );
    }
    let mut count = Interrupt {
        steps: 0,
        stop: None,
    };
    let expected = batch
        .try_eval_finite_with_meter(&[0.5], &mut count)
        .unwrap();
    for stop in 1..=count.steps {
        assert_eq!(
            batch.try_eval_finite_with_meter(
                &[0.5],
                &mut Interrupt {
                    steps: 0,
                    stop: Some(stop)
                }
            ),
            Err(EvalError::Meter(MeterError::Cancelled)),
            "evaluation safe point {stop}",
        );
        assert_eq!(batch.try_eval_finite(&[0.5]).unwrap(), expected);
    }
}

#[test]
fn planning_and_memo_storage_obey_cumulative_budget_without_verifier_borrowing() {
    let scalar = program(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::Sin,
        CompiledOp::Cos,
    ]);
    let originals = vec![scalar.clone(), scalar];
    let mut budget = Budget::new(BudgetLimits::uniform(0, 97));
    assert!(CompiledBatch::try_from_programs_with_meter(originals.clone(), &mut budget).is_err());
    assert_eq!(budget.verifier_remaining(), 97);
    let batch = CompiledBatch::try_from_programs(originals).unwrap();
    let mut limits = BudgetLimits::uniform(100_000, 97);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1;
    let mut budget = Budget::new(limits);
    assert!(matches!(
        batch.try_eval_finite_with_meter(&[0.5], &mut budget),
        Err(EvalError::Meter(_))
    ));
    assert_eq!(budget.verifier_remaining(), 97);
    let mut limits = BudgetLimits::uniform(100_000, 97);
    limits.dimensions[Dimension::ComputeSteps.index()] = 1;
    let mut budget = Budget::new(limits);
    assert!(matches!(
        batch.try_eval_finite_with_meter(&[0.5], &mut budget),
        Err(EvalError::Meter(_))
    ));
    assert!(
        batch
            .try_eval_finite_with_meter(&[0.5], &mut budget)
            .is_err()
    );
    assert_eq!(budget.verifier_remaining(), 97);
}

#[test]
fn empty_batch_and_bytecode_round_trip_keep_reference_behavior() {
    let empty = CompiledBatch::try_from_programs(vec![]).unwrap();
    assert_eq!(empty.try_eval_finite(&[]).unwrap(), Vec::<f64>::new());
    let scalar = guarded_log(true);
    let encoded = serde_json::to_vec(&scalar).unwrap();
    let decoded: CompiledExpr = serde_json::from_slice(&encoded).unwrap();
    let batch = CompiledBatch::try_from_programs(vec![scalar, decoded]).unwrap();
    assert_eq!(serde_json::to_vec(&batch.programs()[0]).unwrap(), encoded);
    assert_reference(&batch, &[vec![-1.0], vec![2.0]]);
}

#[test]
fn partial_batch_caches_never_hide_missing_inputs_or_undefined_selected_branches() {
    let first = program(vec![CompiledOp::LoadVar(0), CompiledOp::Sin]);
    let missing_input = program(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::Sin,
        CompiledOp::LoadVar(1),
        CompiledOp::Add(2),
    ]);
    let batch = CompiledBatch::try_from_programs(vec![first.clone(), missing_input]).unwrap();
    assert_eq!(
        batch.try_eval_finite(&[1.0]),
        Err(EvalError::VariableOutOfBounds {
            index: 1,
            slice_len: 1
        })
    );
    assert_reference(&batch, &[vec![1.0], vec![1.0, 2.0]]);

    let undefined = program(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::Sin,
        CompiledOp::LoadConst(0.0),
        CompiledOp::Greater,
        CompiledOp::JumpIfFalse(8),
        CompiledOp::LoadVar(0),
        CompiledOp::Sin,
        CompiledOp::Jump(9),
        CompiledOp::NoMatchingPiecewiseBranch,
    ]);
    let batch =
        CompiledBatch::try_from_programs(vec![first, undefined.clone(), undefined]).unwrap();
    assert_eq!(
        batch.try_eval_finite(&[-1.0]),
        Err(EvalError::NoMatchingPiecewiseBranch)
    );
    assert_reference(&batch, &[vec![-1.0], vec![0.0], vec![1.0]]);
}
