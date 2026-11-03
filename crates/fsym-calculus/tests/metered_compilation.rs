//! Scalar generation and independent admission share real work accounting.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{
    CompileError, CompiledExpr, CompiledOp, EvalError, MAX_COMPILE_DEPTH,
    MAX_COMPILE_NAME_BYTES, MAX_COMPILE_OPS,
};
use fsym_calculus::sparse_jacobian::{
    compiled_expression_matches, compiled_expression_matches_with_meter,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

const AMPLE: u64 = 1 << 40;
const VERIFIER_RESERVE: u64 = 37;

fn budget(limits: [u64; 3]) -> Budget {
    let mut specification = BudgetLimits::uniform(AMPLE, VERIFIER_RESERVE);
    for (dimension, amount) in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ]
    .into_iter()
    .zip(limits)
    {
        specification.dimensions[dimension.index()] = amount;
    }
    Budget::new(specification)
}

fn remaining(budget: &Budget) -> [u64; 3] {
    [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ]
    .map(|dimension| budget.remaining(dimension))
}

struct InterruptAt {
    ledger: Budget,
    checkpoints: usize,
    interrupt: Option<usize>,
}

impl InterruptAt {
    fn new(interrupt: Option<usize>) -> Self {
        Self {
            ledger: budget([AMPLE; 3]),
            checkpoints: 0,
            interrupt,
        }
    }
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.ledger, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.ledger, charges)
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

fn source() -> (Expr, HashMap<Symbol, usize>) {
    let variable = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([4; 32]));
    let x = Expr::Sym(variable.clone());
    (
        Expr::Add(vec![
            Expr::Function("sin".into(), vec![x.clone()]),
            Expr::Pow(Arc::new(x), Arc::new(Expr::from_i64(2))),
            Expr::from_i64(3),
        ]),
        HashMap::from([(variable, 0)]),
    )
}

fn is_exhausted(error: CompileError, expected: Dimension) -> bool {
    matches!(
        error,
        CompileError::Meter(MeterError::Budget(BudgetError::Exhausted { dimension, .. }))
            if dimension == expected
    )
}

#[test]
fn repeated_checked_compilation_consumes_one_exact_cumulative_ledger() {
    let (expression, variables) = source();
    let mut probe = budget([AMPLE; 3]);
    let expected =
        CompiledExpr::try_compile_checked_with_meter(&expression, &variables, &mut probe).unwrap();
    assert_eq!(
        expected.ops(),
        &[
            CompiledOp::LoadVar(0),
            CompiledOp::Sin,
            CompiledOp::LoadVar(0),
            CompiledOp::Pow(2.0),
            CompiledOp::LoadConst(3.0),
            CompiledOp::Add(3),
        ]
    );
    let cost = remaining(&probe).map(|left| AMPLE - left);
    assert!(cost.iter().all(|value| *value > 0));
    let mut shared = budget(cost.map(|value| value * 2));
    for _ in 0..2 {
        let actual =
            CompiledExpr::try_compile_checked_with_meter(&expression, &variables, &mut shared)
                .unwrap();
        assert_eq!(actual, expected);
    }
    assert_eq!(remaining(&shared), [0; 3]);
    assert!(is_exhausted(
        CompiledExpr::try_compile_checked_with_meter(&expression, &variables, &mut shared)
            .unwrap_err(),
        Dimension::ComputeSteps,
    ));
    assert_eq!(shared.verifier_remaining(), VERIFIER_RESERVE);

    // An existing caller ledger also composes later admission and execution.
    let before = probe.remaining(Dimension::ComputeSteps);
    assert!(
        compiled_expression_matches_with_meter(
            &expression,
            &variables,
            &expected,
            &mut probe,
        )
        .unwrap()
    );
    let after_check = probe.remaining(Dimension::ComputeSteps);
    assert_eq!(
        expected.try_eval_finite_with_meter(&[0.0], &mut probe),
        Ok(3.0)
    );
    assert!(probe.remaining(Dimension::ComputeSteps) < after_check && after_check < before);
}

#[test]
fn generation_budget_does_not_silently_pay_for_independent_checking() {
    let (expression, variables) = source();
    let mut probe = budget([AMPLE; 3]);
    CompiledExpr::try_compile_with_meter(&expression, &variables, &mut probe).unwrap();
    let generation_cost = AMPLE - probe.remaining(Dimension::ComputeSteps);
    let mut generation_only = budget([generation_cost, AMPLE, AMPLE]);
    CompiledExpr::try_compile_with_meter(&expression, &variables, &mut generation_only).unwrap();
    assert_eq!(generation_only.remaining(Dimension::ComputeSteps), 0);

    let mut complete = budget([generation_cost, AMPLE, AMPLE]);
    assert!(is_exhausted(
        CompiledExpr::try_compile_checked_with_meter(&expression, &variables, &mut complete)
            .unwrap_err(),
        Dimension::ComputeSteps,
    ));
    assert_eq!(complete.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(complete.verifier_remaining(), VERIFIER_RESERVE);
}

#[test]
fn actual_instruction_growth_is_atomic_and_cumulatively_charged() {
    let expression = Expr::Add(vec![Expr::from_i64(1); 5]);
    let variables = HashMap::new();
    let mut probe = budget([AMPLE; 3]);
    let program = CompiledExpr::try_compile_with_meter(&expression, &variables, &mut probe).unwrap();
    assert_eq!(program.ops().len(), 6);
    // Six instructions grow through requests for 1, 2, 4, 8 slots. Released old
    // buffers cannot refund their cost to a later compiler invocation.
    assert_eq!(AMPLE - probe.remaining(Dimension::AllocationCount), 4);
    let bytes = ((1 + 2 + 4 + 8) * size_of::<CompiledOp>()) as u64;
    assert_eq!(AMPLE - probe.remaining(Dimension::MemoryBytes), bytes);
    for (limits, refused) in [
        ([AMPLE, 0, AMPLE], Dimension::MemoryBytes),
        ([AMPLE, AMPLE, 0], Dimension::AllocationCount),
    ] {
        let mut ledger = budget(limits);
        assert!(is_exhausted(
            CompiledExpr::try_compile_with_meter(&expression, &variables, &mut ledger)
                .unwrap_err(),
            refused,
        ));
        assert_eq!(ledger.remaining(Dimension::MemoryBytes), limits[1]);
        assert_eq!(ledger.remaining(Dimension::AllocationCount), limits[2]);
        assert!(ledger.remaining(Dimension::ComputeSteps) < limits[0]);
    }
    let mut one_growth_short = budget([AMPLE, bytes - 1, AMPLE]);
    assert!(is_exhausted(
        CompiledExpr::try_compile_with_meter(&expression, &variables, &mut one_growth_short)
            .unwrap_err(),
        Dimension::MemoryBytes,
    ));
    assert!(one_growth_short.remaining(Dimension::MemoryBytes) < bytes - 1);
}

#[test]
fn every_generation_check_and_combined_safe_point_can_cancel() {
    let (expression, variables) = source();
    let program = CompiledExpr::try_compile(&expression, &variables).unwrap();
    for lane in 0..3 {
        let run = |meter: &mut InterruptAt| -> Result<(), CompileError> {
            match lane {
                0 => CompiledExpr::try_compile_with_meter(&expression, &variables, meter).map(drop),
                1 => compiled_expression_matches_with_meter(
                    &expression,
                    &variables,
                    &program,
                    meter,
                )
                .map(|matches| assert!(matches)),
                _ => CompiledExpr::try_compile_checked_with_meter(
                    &expression,
                    &variables,
                    meter,
                )
                .map(drop),
            }
        };
        let mut observed = InterruptAt::new(None);
        run(&mut observed).unwrap();
        assert!(observed.checkpoints > program.ops().len());
        for point in 1..=observed.checkpoints {
            let mut interrupted = InterruptAt::new(Some(point));
            assert_eq!(
                run(&mut interrupted),
                Err(CompileError::Meter(MeterError::Cancelled)),
                "lane {lane}, checkpoint {point}",
            );
            assert_eq!(interrupted.ledger.verifier_remaining(), VERIFIER_RESERVE);
        }
    }
}

#[test]
fn cancellation_also_wins_at_mismatch_and_refusal_publication() {
    let (expression, variables) = source();
    let wrong = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    let mut observed = InterruptAt::new(None);
    assert!(
        !compiled_expression_matches_with_meter(
            &expression,
            &variables,
            &wrong,
            &mut observed,
        )
        .unwrap()
    );
    let mut late = InterruptAt::new(Some(observed.checkpoints));
    assert_eq!(
        compiled_expression_matches_with_meter(&expression, &variables, &wrong, &mut late),
        Err(CompileError::Meter(MeterError::Cancelled)),
    );

    let unsupported = Expr::Function("unknown".into(), vec![Expr::from_i64(1)]);
    let mut observed = InterruptAt::new(None);
    assert!(matches!(
        CompiledExpr::try_compile_with_meter(&unsupported, &variables, &mut observed),
        Err(CompileError::UnsupportedFunction(_)),
    ));
    let mut late = InterruptAt::new(Some(observed.checkpoints));
    assert_eq!(
        CompiledExpr::try_compile_with_meter(&unsupported, &variables, &mut late),
        Err(CompileError::Meter(MeterError::Cancelled)),
    );
}

#[test]
fn independent_check_rejects_typed_slot_power_and_trailing_instruction_mutants() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let variables = HashMap::from([(x.clone(), 0), (y.clone(), 1)]);
    let expression = Expr::Pow(Arc::new(Expr::Sym(x)), Arc::new(Expr::Sym(y)));
    let mut ledger = budget([AMPLE; 3]);
    let correct =
        CompiledExpr::try_compile_checked_with_meter(&expression, &variables, &mut ledger).unwrap();
    assert_eq!(
        correct.ops(),
        &[CompiledOp::LoadVar(0), CompiledOp::LoadVar(1), CompiledOp::PowPositive]
    );
    for operations in [
        vec![
            CompiledOp::LoadVar(1),
            CompiledOp::LoadVar(0),
            CompiledOp::PowPositive,
        ],
        vec![CompiledOp::LoadVar(0), CompiledOp::Pow(2.0)],
        vec![
            CompiledOp::LoadVar(0),
            CompiledOp::LoadVar(1),
            CompiledOp::PowPositive,
            CompiledOp::Neg,
        ],
    ] {
        let mutant = CompiledExpr::try_from_ops(operations).unwrap();
        let before = ledger.remaining(Dimension::ComputeSteps);
        assert!(
            !compiled_expression_matches_with_meter(
                &expression,
                &variables,
                &mutant,
                &mut ledger,
            )
            .unwrap()
        );
        assert!(ledger.remaining(Dimension::ComputeSteps) < before);
    }
}

#[test]
fn original_held_domain_checks_survive_governed_compilation() {
    let x = Symbol::new("domain_x");
    let variables = HashMap::from([(x.clone(), 0)]);
    let logarithm = Expr::Function("log".into(), vec![Expr::Sym(x)]);
    for source in [
        Expr::Mul(vec![Expr::from_i64(0), logarithm.clone()]),
        Expr::Pow(Arc::new(logarithm.clone()), Arc::new(Expr::from_i64(0))),
        Expr::Pow(Arc::new(logarithm), Arc::new(Expr::from_i64(-1))),
    ] {
        let mut ledger = budget([AMPLE; 3]);
        let program =
            CompiledExpr::try_compile_checked_with_meter(&source, &variables, &mut ledger)
                .unwrap();
        assert_eq!(
            program.try_eval_finite_with_meter(&[0.0], &mut ledger),
            Err(EvalError::NonFiniteValue)
        );
        let constant = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
        assert!(
            !compiled_expression_matches_with_meter(
                &source,
                &variables,
                &constant,
                &mut ledger,
            )
            .unwrap()
        );
    }
}

#[test]
fn actual_rational_rounding_is_metered_and_can_be_interrupted_inside_conversion() {
    let denominator = BigInt::one() << 511_u32;
    let source = Expr::Rational(BigRational::new(&denominator + 1_i64, denominator));
    let variables = HashMap::new();
    let mut observed = InterruptAt::new(None);
    let correct =
        CompiledExpr::try_compile_with_meter(&source, &variables, &mut observed).unwrap();
    assert_eq!(correct.ops(), &[CompiledOp::LoadConst(1.0)]);
    assert!(observed.checkpoints > 100);
    let conversion_point = observed.checkpoints / 2;
    let mut cancelled = InterruptAt::new(Some(conversion_point));
    assert_eq!(
        CompiledExpr::try_compile_with_meter(&source, &variables, &mut cancelled),
        Err(CompileError::Meter(MeterError::Cancelled)),
    );
    assert!(cancelled.ledger.remaining(Dimension::ComputeSteps) < AMPLE);
    assert!(cancelled.ledger.remaining(Dimension::MemoryBytes) < AMPLE);
    let mut zero_storage = budget([AMPLE, 0, AMPLE]);
    assert!(is_exhausted(
        CompiledExpr::try_compile_with_meter(&source, &variables, &mut zero_storage)
            .unwrap_err(),
        Dimension::MemoryBytes,
    ));
}

#[test]
fn metered_exact_quotients_keep_rounding_cells_and_canonical_underflow() {
    let variables = HashMap::new();
    let base = BigInt::one() << 53_u32;
    for (numerator, denominator, expected) in [
        (base.clone(), &base + 1_i64, 0x3fef_ffff_ffff_ffff),
        (&base + 2_i64, &base + 1_i64, 1.0_f64.to_bits()),
        (BigInt::from(3), BigInt::one() << 1075_u32, 2),
        (BigInt::from(-1), BigInt::one() << 1075_u32, 0),
    ] {
        let source = Expr::Rational(BigRational::new(numerator, denominator));
        let mut ledger = budget([AMPLE; 3]);
        let program =
            CompiledExpr::try_compile_checked_with_meter(&source, &variables, &mut ledger)
                .unwrap();
        let [CompiledOp::LoadConst(value)] = program.ops() else {
            panic!("a rational literal must produce one constant load");
        };
        assert_eq!(value.to_bits(), expected);
        assert!(compiled_expression_matches(&source, &variables, &program).unwrap());
        let mutant = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(
            f64::from_bits(expected + 1),
        )])
        .unwrap();
        assert!(
            !compiled_expression_matches_with_meter(
                &source,
                &variables,
                &mutant,
                &mut ledger,
            )
            .unwrap()
        );
    }
}

#[test]
fn huge_refused_literals_and_names_do_not_create_unbounded_diagnostics() {
    let huge = BigInt::one() << 100_000_u32;
    let variables = HashMap::new();
    for source in [
        Expr::Integer(huge.clone()),
        Expr::Pow(Arc::new(Expr::from_i64(2)), Arc::new(Expr::Integer(huge))),
    ] {
        let mut ledger = budget([1_000, 512, 8]);
        let error =
            CompiledExpr::try_compile_with_meter(&source, &variables, &mut ledger).unwrap_err();
        match error {
            CompileError::NumericConversion(message)
            | CompileError::UnsupportedExponent(message) => {
                assert!(message.len() <= 256);
            }
            unexpected => panic!("unexpected refusal: {unexpected:?}"),
        }
        assert!(ledger.remaining(Dimension::MemoryBytes) > 0);
    }
    let valid = Symbol::new("é".repeat(MAX_COMPILE_NAME_BYTES / 2));
    let map = HashMap::from([(valid.clone(), 0)]);
    assert!(
        CompiledExpr::try_compile_checked_with_meter(
            &Expr::Sym(valid),
            &map,
            &mut budget([AMPLE; 3]),
        )
        .is_ok()
    );
    for source in [
        Expr::Sym(Symbol::new("x".repeat(MAX_COMPILE_NAME_BYTES + 1))),
        Expr::Function("f".repeat(MAX_COMPILE_NAME_BYTES + 1), vec![]),
    ] {
        let mut ledger = budget([100, 0, 0]);
        assert!(matches!(
            CompiledExpr::try_compile_with_meter(&source, &variables, &mut ledger),
            Err(CompileError::NameLimitExceeded { bytes, max })
                if bytes == MAX_COMPILE_NAME_BYTES + 1 && max == MAX_COMPILE_NAME_BYTES,
        ));
        assert_eq!(remaining(&ledger)[1..], [0, 0]);
    }
}

#[test]
fn source_preflight_and_final_bytecode_validation_each_consume_actual_fuel() {
    let source = Expr::Add(vec![Expr::from_i64(1); MAX_COMPILE_OPS]);
    let dummy = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    let mut ledger = budget([5, 0, 0]);
    assert!(is_exhausted(
        compiled_expression_matches_with_meter(
            &source,
            &HashMap::new(),
            &dummy,
            &mut ledger,
        )
        .unwrap_err(),
        Dimension::ComputeSteps,
    ));
    assert_eq!(remaining(&ledger), [0; 3]);

    let mut operations = vec![CompiledOp::LoadConst(1.0); 64];
    operations.push(CompiledOp::Add(64));
    let mut ledger = budget([64, 0, 0]);
    assert!(matches!(
        CompiledExpr::try_from_ops_with_meter(operations.clone(), &mut ledger),
        Err(EvalError::Meter(MeterError::Budget(BudgetError::Exhausted {
            dimension: Dimension::ComputeSteps,
            ..
        }))),
    ));
    let mut ledger = budget([65, 0, 0]);
    assert!(CompiledExpr::try_from_ops_with_meter(operations, &mut ledger).is_ok());
    assert_eq!(remaining(&ledger), [0; 3]);

    let mut deep = Expr::from_i64(1);
    for _ in 0..MAX_COMPILE_DEPTH + 2 {
        deep = Expr::Function("sin".into(), vec![deep]);
    }
    let mut ledger = budget([AMPLE; 3]);
    assert!(matches!(
        CompiledExpr::try_compile_with_meter(&deep, &HashMap::new(), &mut ledger),
        Err(CompileError::DepthLimitExceeded { .. }),
    ));
    assert!(ledger.remaining(Dimension::ComputeSteps) < AMPLE);
}
