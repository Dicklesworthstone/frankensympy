//! Lazy branch execution and independent source binding for finite numerics.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{
    CompileError, CompiledExpr, CompiledOp, EvalError, MAX_COMPILE_DEPTH, MAX_COMPILE_OPS,
};
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_core::{Expr, Symbol, SymbolIdentity};
use std::collections::HashMap;
use std::sync::Arc;

fn function(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Function(name.into(), args)
}

fn x() -> Expr {
    Expr::symbol("x")
}

fn integer(value: i64) -> Expr {
    Expr::from_i64(value)
}

fn truth(value: bool) -> Expr {
    Expr::symbol(if value { "True" } else { "False" })
}

fn relation(name: &str, left: Expr, right: Expr) -> Expr {
    function(name, vec![left, right])
}

fn piecewise(branches: Vec<(Expr, Expr)>) -> Expr {
    function(
        "Piecewise",
        branches
            .into_iter()
            .map(|(value, condition)| function("Tuple", vec![value, condition]))
            .collect(),
    )
}

fn variables() -> HashMap<Symbol, usize> {
    HashMap::from([(Symbol::new("x"), 0)])
}

fn checked(source: &Expr) -> CompiledExpr {
    CompiledExpr::try_compile_checked_with_meter(source, &variables(), &mut Unbounded).unwrap()
}

fn reciprocal() -> Expr {
    Expr::Pow(Arc::new(x()), Arc::new(integer(-1)))
}

fn guarded_reciprocal() -> Expr {
    piecewise(vec![
        (reciprocal(), relation("Unequality", x(), integer(0))),
        (integer(0), truth(true)),
    ])
}

#[test]
fn selected_branch_alone_executes_division_logarithm_and_square_root() {
    let source = piecewise(vec![
        (
            function("log", vec![Expr::Mul(vec![integer(-1), x()])]),
            relation("StrictLessThan", x(), integer(0)),
        ),
        (
            function("sqrt", vec![x()]),
            relation("GreaterThan", x(), integer(0)),
        ),
        (reciprocal(), truth(true)),
    ]);
    let program = checked(&source);
    assert_eq!(program.try_eval_finite(&[-2.0]), Ok(2.0f64.ln()));
    assert_eq!(program.try_eval_finite(&[0.0]), Ok(0.0));
    assert_eq!(program.try_eval_finite(&[4.0]), Ok(2.0));

    let reciprocal = checked(&guarded_reciprocal());
    assert_eq!(reciprocal.try_eval_finite(&[0.0]), Ok(0.0));
    assert_eq!(reciprocal.try_eval_finite(&[-4.0]), Ok(-0.25));
}

#[test]
fn branch_order_is_observable_even_when_conditions_overlap() {
    let source = piecewise(vec![
        (integer(11), relation("LessThan", x(), integer(2))),
        (integer(22), relation("GreaterThan", x(), integer(0))),
        (integer(33), truth(true)),
    ]);
    let program = checked(&source);
    assert_eq!(program.try_eval_finite(&[1.0]), Ok(11.0));
    assert_eq!(program.try_eval_finite(&[3.0]), Ok(22.0));
}

#[test]
fn nested_branch_programs_preserve_enclosing_stack_operands() {
    let inner = piecewise(vec![
        (integer(3), relation("StrictLessThan", x(), integer(1))),
        (reciprocal(), truth(true)),
    ]);
    let outer = piecewise(vec![
        (inner, relation("GreaterThan", x(), integer(0))),
        (integer(5), truth(true)),
    ]);
    let source = Expr::Add(vec![integer(7), Expr::Mul(vec![integer(2), outer])]);
    let program = checked(&source);
    for (input, expected) in [(-1.0, 17.0), (0.0, 13.0), (2.0, 8.0)] {
        assert_eq!(program.try_eval_finite(&[input]), Ok(expected));
    }
}

#[test]
fn conjunction_and_disjunction_short_circuit_invalid_later_conditions() {
    let later = relation(
        "StrictGreaterThan",
        function("log", vec![Expr::Mul(vec![integer(-1), x()])]),
        integer(0),
    );
    for (tag, first, expected) in [
        ("And", relation("StrictLessThan", x(), integer(0)), 9.0),
        ("Or", relation("GreaterThan", x(), integer(0)), 4.0),
    ] {
        let source = piecewise(vec![
            (integer(4), function(tag, vec![first, later.clone()])),
            (integer(9), truth(true)),
        ]);
        let program = checked(&source);
        assert_eq!(program.try_eval_finite(&[1.0]), Ok(expected));
        assert_eq!(program.try_eval_finite(&[-2.0]), Ok(4.0));
    }
}

#[test]
fn all_comparisons_respect_boundaries_and_operand_direction() {
    let cases = [
        ("Equality", [false, true, false]),
        ("Unequality", [true, false, true]),
        ("StrictLessThan", [true, false, false]),
        ("LessThan", [true, true, false]),
        ("StrictGreaterThan", [false, false, true]),
        ("GreaterThan", [false, true, true]),
    ];
    for (tag, answers) in cases {
        let source = piecewise(vec![
            (integer(1), relation(tag, x(), integer(0))),
            (integer(0), truth(true)),
        ]);
        let program = checked(&source);
        for (input, truth) in [-1.0, 0.0, 1.0].into_iter().zip(answers) {
            assert_eq!(
                program.try_eval_finite(&[input]),
                Ok(f64::from(truth)),
                "{tag}"
            );
        }
    }
}

#[test]
fn boolean_identities_not_and_nested_shortcuts_are_supported() {
    let conditions = [
        (function("And", vec![]), true),
        (function("Or", vec![]), false),
        (function("And", vec![truth(true)]), true),
        (function("Or", vec![truth(false)]), false),
        (function("Not", vec![truth(false)]), true),
        (
            function("Not", vec![function("Or", vec![truth(false), truth(true)])]),
            false,
        ),
    ];
    for (condition, expected) in conditions {
        let source = piecewise(vec![(integer(1), condition), (integer(0), truth(true))]);
        assert_eq!(
            checked(&source).try_eval_finite(&[]),
            Ok(f64::from(expected))
        );
    }
}

#[test]
fn missing_branch_is_an_explicit_domain_refusal() {
    let source = piecewise(vec![(
        integer(5),
        relation("StrictGreaterThan", x(), integer(0)),
    )]);
    let program = checked(&source);
    assert_eq!(program.try_eval_finite(&[1.0]), Ok(5.0));
    assert_eq!(
        program.try_eval_finite(&[0.0]),
        Err(EvalError::NoMatchingPiecewiseBranch)
    );
    assert_eq!(
        program.try_eval(&[-1.0]),
        Err(EvalError::NoMatchingPiecewiseBranch)
    );
}

#[test]
fn active_domain_errors_and_nonfinite_inputs_remain_errors() {
    let source = piecewise(vec![(function("log", vec![x()]), truth(true))]);
    let program = checked(&source);
    for value in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            program.try_eval_finite(&[value]),
            Err(EvalError::NonFiniteValue)
        );
    }
    // Even the non-finite legacy evaluator must not silently select a branch
    // by interpreting a NaN comparison as false.
    let bad_condition = piecewise(vec![
        (
            integer(1),
            relation("LessThan", function("sqrt", vec![x()]), integer(0)),
        ),
        (integer(2), truth(true)),
    ]);
    assert_eq!(
        checked(&bad_condition).try_eval(&[-1.0]),
        Err(EvalError::NonFiniteValue)
    );
}

#[test]
fn same_printing_typed_variables_remain_distinct_in_conditions_and_values() {
    let first = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = piecewise(vec![
        (
            Expr::Sym(first.clone()),
            relation(
                "StrictLessThan",
                Expr::Sym(first.clone()),
                Expr::Sym(second.clone()),
            ),
        ),
        (Expr::Sym(second.clone()), truth(true)),
    ]);
    let map = HashMap::from([(first, 1), (second, 0)]);
    let program =
        CompiledExpr::try_compile_checked_with_meter(&source, &map, &mut Unbounded).unwrap();
    assert_eq!(program.try_eval_finite(&[2.0, 5.0]), Ok(2.0));
    assert_eq!(program.try_eval_finite(&[5.0, 2.0]), Ok(2.0));
}

#[test]
fn malformed_conditions_and_unsupported_inactive_branches_fail_admission() {
    let tagged_truth = Expr::Sym(Symbol::with_identity(
        "True",
        SymbolIdentity::from_assumptions_digest([3; 32]),
    ));
    let malformed = [
        function("Piecewise", vec![]),
        function("Piecewise", vec![integer(1)]),
        function("Piecewise", vec![function("Tuple", vec![integer(1)])]),
        piecewise(vec![(integer(1), x())]),
        piecewise(vec![(integer(1), integer(1))]),
        piecewise(vec![(integer(1), tagged_truth)]),
        piecewise(vec![(integer(1), function("Not", vec![]))]),
        piecewise(vec![(integer(1), function("Equality", vec![x()]))]),
        piecewise(vec![
            (integer(1), truth(true)),
            (function("custom", vec![x()]), truth(false)),
        ]),
    ];
    for source in malformed {
        assert!(CompiledExpr::try_compile(&source, &variables()).is_err());
    }
}

#[test]
fn control_flow_decoder_rejects_loops_invalid_targets_and_stack_joins() {
    use CompiledOp::*;
    let invalid = [
        vec![LoadConst(1.0), Jump(0)],
        vec![LoadConst(1.0), Jump(1)],
        vec![LoadConst(1.0), Jump(usize::MAX)],
        vec![LoadConst(1.0), Jump(3), LoadConst(2.0)],
        vec![
            LoadConst(1.0),
            JumpIfFalse(4),
            LoadConst(2.0),
            Jump(5),
            Jump(5),
        ],
        vec![
            LoadConst(1.0),
            JumpIfFalse(4),
            LoadConst(2.0),
            Jump(5),
            Add(2),
        ],
        vec![JumpIfFalse(1)],
        vec![NoMatchingPiecewiseBranch],
        vec![
            LoadConst(0.0),
            JumpIfFalse(4),
            LoadConst(-0.0),
            Jump(5),
            LoadConst(2.0),
        ],
    ];
    for ops in invalid {
        assert!(CompiledExpr::try_from_ops(ops.clone()).is_err(), "{ops:?}");
        let wire = serde_json::json!({"ops": ops});
        assert!(serde_json::from_value::<CompiledExpr>(wire).is_err());
    }
}

#[test]
fn serialized_valid_branches_roundtrip_without_losing_lazy_execution() {
    let original = checked(&guarded_reciprocal());
    let bytes = serde_json::to_vec(&original).unwrap();
    let restored: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(restored, original);
    assert_eq!(restored.try_eval_finite(&[0.0]), Ok(0.0));
    assert!(compiled_expression_matches(&guarded_reciprocal(), &variables(), &restored).unwrap());
}

#[test]
fn independently_checked_source_rejects_condition_and_branch_value_mutants() {
    let source = guarded_reciprocal();
    let program = checked(&source);
    let mut altered_condition = program.ops().to_vec();
    let comparison = altered_condition
        .iter_mut()
        .find(|op| matches!(op, CompiledOp::NotEqual))
        .unwrap();
    *comparison = CompiledOp::Equal;
    let wrong = CompiledExpr::try_from_ops(altered_condition).unwrap();
    assert!(!compiled_expression_matches(&source, &variables(), &wrong).unwrap());

    let other_source = piecewise(vec![
        (integer(7), relation("Unequality", x(), integer(0))),
        (integer(0), truth(true)),
    ]);
    assert!(!compiled_expression_matches(&other_source, &variables(), &program).unwrap());
}

#[test]
fn branch_predicates_cannot_treat_arbitrary_numbers_as_booleans() {
    let program = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::JumpIfFalse(4),
        CompiledOp::LoadConst(7.0),
        CompiledOp::Jump(5),
        CompiledOp::LoadConst(8.0),
    ])
    .unwrap();
    for value in [-1.0, 2.0, -0.0] {
        assert_eq!(
            program.try_eval_finite(&[value]),
            Err(EvalError::InvalidConditionValue)
        );
    }
    assert_eq!(program.try_eval_finite(&[0.0]), Ok(8.0));
    assert_eq!(program.try_eval_finite(&[1.0]), Ok(7.0));
}

#[test]
fn branch_and_condition_growth_obey_existing_compiler_limits() {
    let many = piecewise(vec![(integer(1), truth(false)); MAX_COMPILE_OPS]);
    assert!(matches!(
        CompiledExpr::try_compile(&many, &variables()),
        Err(CompileError::OpLimitExceeded { .. })
    ));
    let mut condition = truth(true);
    for _ in 0..MAX_COMPILE_DEPTH {
        condition = function("Not", vec![condition]);
    }
    let deep = piecewise(vec![(integer(1), condition)]);
    assert!(matches!(
        CompiledExpr::try_compile(&deep, &variables()),
        Err(CompileError::DepthLimitExceeded { .. })
    ));
}

struct InterruptAt {
    checkpoints: usize,
    at: Option<usize>,
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_covers_branch_admission_and_both_execution_paths() {
    let source = guarded_reciprocal();
    let mut probe = InterruptAt {
        checkpoints: 0,
        at: None,
    };
    let program =
        CompiledExpr::try_compile_checked_with_meter(&source, &variables(), &mut probe).unwrap();
    for at in 1..=probe.checkpoints {
        let mut meter = InterruptAt {
            checkpoints: 0,
            at: Some(at),
        };
        assert_eq!(
            CompiledExpr::try_compile_checked_with_meter(&source, &variables(), &mut meter),
            Err(CompileError::Meter(MeterError::Cancelled))
        );
    }
    for input in [0.0, 2.0] {
        let mut probe = InterruptAt {
            checkpoints: 0,
            at: None,
        };
        program
            .try_eval_finite_with_meter(&[input], &mut probe)
            .unwrap();
        for at in 1..=probe.checkpoints {
            let mut meter = InterruptAt {
                checkpoints: 0,
                at: Some(at),
            };
            assert_eq!(
                program.try_eval_finite_with_meter(&[input], &mut meter),
                Err(EvalError::Meter(MeterError::Cancelled))
            );
        }
    }
}

#[test]
fn evaluation_budgets_charge_only_executed_branch_work_without_resetting() {
    let costly = Expr::Add(vec![x(); 100]);
    let source = piecewise(vec![
        (integer(4), relation("StrictLessThan", x(), integer(0))),
        (costly, truth(true)),
    ]);
    let program = checked(&source);
    let limits = BudgetLimits::uniform(1_000_000, 31);
    let mut short = Budget::new(limits);
    assert_eq!(
        program.try_eval_finite_with_meter(&[-1.0], &mut short),
        Ok(4.0)
    );
    let short_cost = 1_000_000 - short.remaining(Dimension::ComputeSteps);
    let mut long = Budget::new(limits);
    assert_eq!(
        program.try_eval_finite_with_meter(&[1.0], &mut long),
        Ok(100.0)
    );
    assert!(1_000_000 - long.remaining(Dimension::ComputeSteps) > short_cost + 100);
    let mut tight = limits;
    tight.dimensions[Dimension::ComputeSteps.index()] = short_cost * 2;
    let mut shared = Budget::new(tight);
    for _ in 0..2 {
        assert_eq!(
            program.try_eval_finite_with_meter(&[-1.0], &mut shared),
            Ok(4.0)
        );
    }
    assert!(matches!(
        program.try_eval_finite_with_meter(&[-1.0], &mut shared),
        Err(EvalError::Meter(_))
    ));
    assert_eq!(shared.verifier_remaining(), 31);
}

#[test]
fn scalar_branch_support_does_not_claim_derivatives_across_seams() {
    let source = piecewise(vec![
        (integer(0), relation("StrictLessThan", x(), integer(0))),
        (integer(1), truth(true)),
    ]);
    assert!(VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("x")]).is_err());
}
