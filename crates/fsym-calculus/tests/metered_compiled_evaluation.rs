//! Caller-owned fuel and cancellation must never expose partial numeric output.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, CompiledResidualSystem, EvalError};
use fsym_core::{Symbol, parse};

fn budget_with(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(1_000_000, 37);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

struct InterruptAt {
    budget: Budget,
    checkpoints: usize,
    interrupt: Option<usize>,
}

impl InterruptAt {
    fn new(interrupt: Option<usize>) -> Self {
        Self {
            budget: Budget::new(BudgetLimits::uniform(1_000_000, 37)),
            checkpoints: 0,
            interrupt,
        }
    }
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.interrupt == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_reduction_operand_consumes_fuel_before_arithmetic() {
    let mut ops = vec![CompiledOp::LoadConst(1.0); 64];
    ops.push(CompiledOp::Add(64));
    let program = CompiledExpr::try_from_ops(ops).unwrap();
    let mut budget = budget_with(Dimension::ComputeSteps, 70);
    assert!(matches!(
        program.try_eval_with_meter(&[], &mut budget),
        Err(EvalError::Meter(MeterError::Budget(
            BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            }
        )))
    ));
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(budget.verifier_remaining(), 37);

    // 65 instruction dispatches and 64 additions, with no hidden per-call reset.
    let mut budget = budget_with(Dimension::ComputeSteps, 258);
    assert_eq!(program.try_eval_with_meter(&[], &mut budget), Ok(64.0));
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 129);
    assert_eq!(program.try_eval_with_meter(&[], &mut budget), Ok(64.0));
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert!(program.try_eval_with_meter(&[], &mut budget).is_err());
}

#[test]
fn coupled_stack_storage_charges_are_atomic_and_cumulative() {
    let program = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(7.0)]).unwrap();
    for (dimension, limit) in [(Dimension::MemoryBytes, 7), (Dimension::AllocationCount, 0)] {
        let mut budget = budget_with(dimension, limit);
        let before = budget.snapshot();
        assert!(matches!(
            program.try_eval_with_meter(&[], &mut budget),
            Err(EvalError::Meter(MeterError::Budget(BudgetError::Exhausted {
                dimension: refused,
                ..
            }))) if refused == dimension
        ));
        assert_eq!(budget.snapshot(), before);
    }
    let mut budget = budget_with(Dimension::MemoryBytes, 16);
    assert_eq!(program.try_eval_with_meter(&[], &mut budget), Ok(7.0));
    assert_eq!(program.try_eval_with_meter(&[], &mut budget), Ok(7.0));
    assert_eq!(budget.remaining(Dimension::MemoryBytes), 0);
    assert!(program.try_eval_with_meter(&[], &mut budget).is_err());
    assert_eq!(budget.verifier_remaining(), 37);
}

#[test]
fn scalar_cancellation_covers_input_validation_instructions_and_return() {
    let program = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadConst(3.0),
        CompiledOp::Mul(2),
        CompiledOp::LoadConst(2.0),
        CompiledOp::Add(2),
    ])
    .unwrap();
    let mut counting = InterruptAt::new(None);
    assert_eq!(
        program.try_eval_finite_with_meter(&[4.0], &mut counting),
        Ok(14.0)
    );
    assert!(counting.checkpoints > program.ops().len());
    for checkpoint in 1..=counting.checkpoints {
        let mut meter = InterruptAt::new(Some(checkpoint));
        assert_eq!(
            program.try_eval_finite_with_meter(&[4.0], &mut meter),
            Err(EvalError::Meter(MeterError::Cancelled)),
            "safe point {checkpoint}"
        );
        assert_eq!(meter.budget.verifier_remaining(), 37);
    }
}

fn system() -> CompiledResidualSystem {
    CompiledResidualSystem::try_compile(
        &[parse("x^2 + y").unwrap(), parse("x*y").unwrap()],
        &[Symbol::new("x"), Symbol::new("y")],
    )
    .unwrap()
}

#[test]
fn system_cancellation_never_publishes_either_partial_lane() {
    let system = system();
    let mut counting = InterruptAt::new(None);
    let mut residuals = [f64::NAN; 2];
    let mut jacobian = [f64::NAN; 4];
    system
        .try_eval_system_with_meter(&[3.0, 4.0], &mut residuals, &mut jacobian, &mut counting)
        .unwrap();
    assert_eq!(residuals, [13.0, 12.0]);
    assert_eq!(jacobian, [6.0, 1.0, 4.0, 3.0]);
    for checkpoint in 1..=counting.checkpoints {
        let mut meter = InterruptAt::new(Some(checkpoint));
        let mut residuals = [101.0, 102.0];
        let mut jacobian = [201.0, 202.0, 203.0, 204.0];
        assert_eq!(
            system.try_eval_system_with_meter(
                &[3.0, 4.0],
                &mut residuals,
                &mut jacobian,
                &mut meter,
            ),
            Err(EvalError::Meter(MeterError::Cancelled)),
            "safe point {checkpoint}"
        );
        assert_eq!(residuals, [101.0, 102.0]);
        assert_eq!(jacobian, [201.0, 202.0, 203.0, 204.0]);
    }
}

#[test]
fn every_insufficient_system_budget_retains_atomic_outputs() {
    let system = system();
    let limits = BudgetLimits::uniform(1_000_000, 37);
    let mut successful = Budget::new(limits);
    system
        .try_eval_system_with_meter(&[3.0, 4.0], &mut [0.0; 2], &mut [0.0; 4], &mut successful)
        .unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let consumed = limits.dimensions[dimension.index()] - successful.remaining(dimension);
        assert!(consumed > 0);
        for allowance in 0..consumed {
            let mut budget = budget_with(dimension, allowance);
            let mut residuals = [101.0, 102.0];
            let mut jacobian = [201.0, 202.0, 203.0, 204.0];
            assert!(matches!(
                system.try_eval_system_with_meter(
                    &[3.0, 4.0], &mut residuals, &mut jacobian, &mut budget,
                ),
                Err(EvalError::Meter(MeterError::Budget(BudgetError::Exhausted {
                    dimension: refused,
                    ..
                }))) if refused == dimension
            ));
            assert_eq!(residuals, [101.0, 102.0]);
            assert_eq!(jacobian, [201.0, 202.0, 203.0, 204.0]);
            assert_eq!(budget.verifier_remaining(), 37);
        }
    }
}

#[test]
fn finite_metered_lane_cannot_hide_nonfinite_intermediates() {
    let program = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::Ln,
        CompiledOp::Pow(-1.0),
    ])
    .unwrap();
    let mut budget = Budget::new(BudgetLimits::uniform(1000, 37));
    assert_eq!(
        program.try_eval_finite_with_meter(&[0.0], &mut budget),
        Err(EvalError::NonFiniteValue)
    );
    assert!(budget.remaining(Dimension::ComputeSteps) < 1000);
    assert_eq!(
        program.try_eval_finite_with_meter(&[f64::INFINITY], &mut budget),
        Err(EvalError::NonFiniteValue)
    );
}

#[test]
fn malformed_later_program_and_empty_system_keep_failure_semantics() {
    let mut system = system();
    system.compiled_jacobian[3] = CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(7)]).unwrap();
    let mut budget = Budget::new(BudgetLimits::uniform(1000, 37));
    let mut residuals = [11.0; 2];
    let mut jacobian = [22.0; 4];
    assert!(matches!(
        system.try_eval_system_with_meter(&[3.0, 4.0], &mut residuals, &mut jacobian, &mut budget),
        Err(EvalError::VariableOutOfBounds { index: 7, .. })
    ));
    assert_eq!(residuals, [11.0; 2]);
    assert_eq!(jacobian, [22.0; 4]);

    let empty = CompiledResidualSystem::try_compile(&[], &[]).unwrap();
    let mut budget = Budget::new(BudgetLimits::uniform(0, 0));
    empty
        .try_eval_system_with_meter(&[], &mut [], &mut [], &mut budget)
        .unwrap();
    let mut cancelled = InterruptAt::new(Some(1));
    assert_eq!(
        empty.try_eval_system_with_meter(&[], &mut [], &mut [], &mut cancelled),
        Err(EvalError::Meter(MeterError::Cancelled))
    );
}
