//! Smooth real absolute-value branches compose with checked numeric programs.
//! A value at zero is defined; a derivative bundle at its cusp is refused.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{CompiledBatch, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::program_artifact::{
    decode_objective_candidate, decode_residual_candidate, encode_objective_candidate,
    encode_residual_candidate,
};
use fsym_calculus::proof::{
    RULE_DIFF_GENERAL, classify_diff_rule, make_diff_term, verified_diff, verify_diff_derivation,
};
use fsym_calculus::sparse_hessian::{ObjectiveEvaluationError, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{
    SparseJacobianEntry, compiled_expression_matches, verify_sparse_jacobian_certificate,
};
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_proof_kernel::{Claim, ProofRule};
use std::collections::HashMap;

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn source(name: &str) -> Expr {
    function(name, Expr::symbol("x"))
}

fn compile(expression: &Expr) -> CompiledExpr {
    CompiledExpr::try_compile(expression, &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
}

fn residual_cusp<T: std::fmt::Debug>(result: Result<T, ResidualEvaluationError>) {
    assert!(
        matches!(
            result,
            Err(ResidualEvaluationError::Evaluation(
                EvalError::NonsmoothSource("abs")
            ))
        ),
        "{result:?}"
    );
}

fn objective_cusp<T: std::fmt::Debug>(result: Result<T, ObjectiveEvaluationError>) {
    assert!(
        matches!(
            result,
            Err(ObjectiveEvaluationError::Evaluation(
                EvalError::NonsmoothSource("abs")
            ))
        ),
        "{result:?}"
    );
}

#[test]
fn jets_preserve_exact_sign_across_the_finite_range_and_refuse_zero() {
    let first = compile(&source("_fsym_abs_d1"));
    let second = compile(&source("_fsym_abs_d2"));
    for magnitude in [f64::from_bits(1), f64::MIN_POSITIVE, 1.0, 1e200, f64::MAX] {
        for sign in [-1.0, 1.0] {
            let point = sign * magnitude;
            assert_eq!(first.try_eval_finite(&[point]), Ok(sign));
            assert_eq!(second.try_eval_finite(&[point]), Ok(0.0));
            for tag in ["abs", "Abs"] {
                assert_eq!(
                    compile(&source(tag)).try_eval_finite(&[point]),
                    Ok(magnitude)
                );
            }
        }
    }
    for zero in [0.0, -0.0] {
        assert_eq!(compile(&source("abs")).try_eval_finite(&[zero]), Ok(0.0));
        for jet in [&first, &second] {
            assert_eq!(
                jet.try_eval_finite(&[zero]),
                Err(EvalError::NonsmoothSource("abs"))
            );
            assert_eq!(
                jet.try_eval(&[zero]),
                Err(EvalError::NonsmoothSource("abs"))
            );
        }
    }
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for jet in [&first, &second] {
            assert_eq!(
                jet.try_eval_finite(&[nonfinite]),
                Err(EvalError::NonFiniteValue)
            );
            assert_eq!(jet.try_eval(&[nonfinite]), Err(EvalError::NonFiniteValue));
        }
    }
}

#[test]
fn source_opcode_admission_distinguishes_value_first_and_second_derivatives() {
    let mapping = HashMap::from([(Symbol::new("x"), 0)]);
    let families = [
        ("abs", CompiledOp::Abs),
        ("Abs", CompiledOp::Abs),
        ("_fsym_abs_d1", CompiledOp::AbsDerivative),
        ("_fsym_abs_d2", CompiledOp::AbsSecondDerivative),
    ];
    for (tag, operation) in &families {
        let expression = source(tag);
        let program = compile(&expression);
        assert_eq!(program.ops(), [CompiledOp::LoadVar(0), operation.clone()]);
        let restored: CompiledExpr =
            serde_json::from_slice(&serde_json::to_vec(&program).unwrap()).unwrap();
        assert_eq!(program, restored);
        assert!(compiled_expression_matches(&expression, &mapping, &restored).unwrap());
        for (_, other) in &families {
            if operation == other {
                continue;
            }
            let forged =
                CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), other.clone()]).unwrap();
            assert!(!compiled_expression_matches(&expression, &mapping, &forged).unwrap());
        }
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation.clone()]),
            Err(EvalError::StackUnderflow)
        );
        for arguments in [vec![], vec![Expr::symbol("x"), Expr::from_i64(1)]] {
            assert!(
                CompiledExpr::try_compile(&Expr::Function((*tag).into(), arguments), &mapping)
                    .is_err()
            );
        }
    }
}

#[test]
fn nonlinear_chain_rule_retains_both_signs_and_all_sparse_products() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    for tag in ["abs", "Abs"] {
        let expression = function(tag, parse("x^2+2*y-3").unwrap());
        let system =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables)
                .unwrap();
        let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
        assert_eq!(system.jacobian_pattern(), [(0, 0), (0, 1)]);
        assert_eq!(
            objective.hessian_pattern(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        for (point, value, gradient, hessian, product) in [
            (
                [2.0, 1.0],
                3.0,
                [4.0, 2.0],
                [2.0, 0.0, 0.0, 0.0],
                [6.0, 0.0],
            ),
            (
                [1.0, -1.0],
                4.0,
                [-2.0, -2.0],
                [-2.0, 0.0, 0.0, 0.0],
                [-6.0, 0.0],
            ),
        ] {
            let first = system.evaluate_sparse(&point).unwrap();
            assert_eq!(first.residuals, [value]);
            assert_eq!(first.jacobian, gradient);
            assert_eq!(
                system
                    .jacobian_vector_product(&point, &[3.0, -2.0])
                    .unwrap(),
                [3.0 * gradient[0] - 2.0 * gradient[1]]
            );
            assert_eq!(
                system
                    .jacobian_transpose_vector_product(&point, &[5.0])
                    .unwrap(),
                gradient.map(|entry| 5.0 * entry)
            );
            let second = objective.evaluate(&point).unwrap();
            assert_eq!(second.value, value);
            assert_eq!(second.gradient, gradient);
            assert_eq!(second.hessian, hessian);
            assert_eq!(
                objective
                    .hessian_vector_product(&point, &[3.0, -2.0])
                    .unwrap(),
                product
            );
        }
    }
}

#[test]
fn squared_absolute_residual_has_two_smooth_minima_and_no_cusp_gradient() {
    let residual = source("abs") - Expr::from_i64(2);
    let objective = VerifiedObjective::try_compile(
        &Expr::pow(residual, Expr::from_i64(2)),
        &[Symbol::new("x")],
    )
    .unwrap();
    for point in [-3.0_f64, -2.0, -1.0, 1.0, 2.0, 3.0] {
        let result = objective.evaluate(&[point]).unwrap();
        let distance = point.abs() - 2.0;
        assert_eq!(result.value, distance * distance);
        assert_eq!(result.gradient, [2.0 * distance * point.signum()]);
        assert_eq!(result.hessian, [2.0]);
        assert_eq!(
            objective.hessian_vector_product(&[point], &[-3.0]).unwrap(),
            [-6.0]
        );
    }
    for zero in [0.0, -0.0] {
        objective_cusp(objective.evaluate(&[zero]));
        objective_cusp(objective.hessian_vector_product(&[zero], &[0.0]));
    }
}

#[test]
fn nested_absolute_values_keep_inner_and_outer_cusps_distinct() {
    let expression = function("abs", source("abs") - Expr::from_i64(2));
    let objective = VerifiedObjective::try_compile(&expression, &[Symbol::new("x")]).unwrap();
    for (point, derivative) in [(-3.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (3.0, 1.0)] {
        let result = objective.evaluate(&[point]).unwrap();
        assert_eq!(result.value, 1.0);
        assert_eq!(result.gradient, [derivative]);
        assert_eq!(result.hessian, [0.0]);
    }
    for point in [-2.0, 0.0, 2.0] {
        objective_cusp(objective.evaluate(&[point]));
        objective_cusp(objective.hessian_vector_product(&[point], &[0.0]));
    }
}

#[test]
fn source_cusps_survive_constant_zero_rules_masks_and_zero_directions() {
    let absolute = source("abs");
    let expressions = [
        absolute.clone(),
        Expr::Add(vec![
            Expr::from_i64(7),
            Expr::Mul(vec![Expr::from_i64(0), absolute.clone()]),
        ]),
        Expr::pow(absolute, Expr::from_i64(0)),
        function("abs", Expr::from_i64(0)),
        function("abs", Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("x")])),
        // Conservatively retain the primitive's smooth-domain obligation even
        // when a composition happens to have a differentiable continuation.
        function("abs", Expr::pow(Expr::symbol("x"), Expr::from_i64(2))),
    ];
    for expression in expressions {
        assert!(
            compile(&expression)
                .try_eval_finite(&[0.0])
                .unwrap()
                .is_finite()
        );
        let system = VerifiedResidualSystem::try_compile(
            std::slice::from_ref(&expression),
            &[Symbol::new("x")],
        )
        .unwrap();
        let objective = VerifiedObjective::try_compile(&expression, &[Symbol::new("x")]).unwrap();
        for point in [0.0, -0.0] {
            residual_cusp(system.evaluate_sparse(&[point]));
            residual_cusp(system.jacobian_vector_product(&[point], &[0.0]));
            residual_cusp(system.jacobian_transpose_vector_product(&[point], &[0.0]));
            objective_cusp(objective.evaluate(&[point]));
            objective_cusp(objective.hessian_vector_product(&[point], &[0.0]));
        }
    }
    // A held variable occurrence may have an identically zero derivative;
    // nonzero constant inner values must still compile and execute normally.
    for inner in [
        Expr::from_i64(-3),
        Expr::pow(Expr::symbol("x"), Expr::from_i64(0)),
    ] {
        let expression = function("abs", inner);
        let objective = VerifiedObjective::try_compile(&expression, &[Symbol::new("x")]).unwrap();
        for point in [0.0, 1.0] {
            let result = objective.evaluate(&[point]).unwrap();
            assert_eq!(result.gradient, [0.0]);
            assert!(result.hessian.iter().all(|value| *value == 0.0));
        }
    }
}

#[test]
fn original_inner_poles_cannot_be_hidden_by_absolute_value_or_zero_arithmetic() {
    let inverse = Expr::pow(Expr::symbol("x"), Expr::from_i64(-1));
    let expressions = [
        function("abs", inverse.clone()),
        Expr::Add(vec![
            Expr::from_i64(7),
            Expr::Mul(vec![Expr::from_i64(0), function("abs", inverse.clone())]),
        ]),
        function(
            "abs",
            Expr::Add(vec![
                Expr::from_i64(1),
                Expr::Mul(vec![Expr::from_i64(0), inverse]),
            ]),
        ),
        function("abs", function("log", Expr::symbol("x"))),
    ];
    for expression in expressions {
        let system = VerifiedResidualSystem::try_compile(
            std::slice::from_ref(&expression),
            &[Symbol::new("x")],
        )
        .unwrap();
        let objective = VerifiedObjective::try_compile(&expression, &[Symbol::new("x")]).unwrap();
        assert!(system.evaluate_sparse(&[2.0]).is_ok());
        assert!(objective.evaluate(&[2.0]).is_ok());
        assert!(system.evaluate_sparse(&[0.0]).is_err());
        assert!(system.jacobian_vector_product(&[0.0], &[0.0]).is_err());
        assert!(
            system
                .jacobian_transpose_vector_product(&[0.0], &[0.0])
                .is_err()
        );
        assert!(objective.hessian_vector_product(&[0.0], &[0.0]).is_err());
    }
}

#[test]
fn same_named_typed_coordinates_remain_independent_in_both_derivative_stages() {
    let x = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([119; 32]));
    let expression = function("abs", Expr::Sym(x.clone()) - Expr::Sym(other.clone()));
    let objective = VerifiedObjective::try_compile(&expression, &[x, other]).unwrap();
    let result = objective.evaluate(&[2.0, 5.0]).unwrap();
    assert_eq!(result.value, 3.0);
    assert_eq!(result.gradient, [-1.0, 1.0]);
    assert_eq!(result.hessian, [0.0; 4]);
    assert_eq!(
        objective
            .hessian_vector_product(&[2.0, 5.0], &[3.0, 7.0])
            .unwrap(),
        [0.0, 0.0]
    );
    objective_cusp(objective.evaluate(&[2.0, 2.0]));
}

fn replace_entry(entry: &mut SparseJacobianEntry, replacement: Expr) {
    entry.compiled_deriv = compile(&replacement);
    entry.symbolic_deriv = replacement.clone();
    let step = entry.derivation.steps.last_mut().unwrap();
    let ProofRule::DefinitionalReduction { lhs, rhs, .. } = &mut step.rule else {
        panic!("expected a differentiation proof");
    };
    *rhs = replacement.clone();
    step.claim = Claim::equality(lhs.clone(), replacement);
}

#[test]
fn portable_programs_replay_and_reject_forged_sign_curvature_and_omissions() {
    let expression = source("abs");
    let variables = [Symbol::new("x")];
    let system =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables).unwrap();
    let bytes = encode_residual_candidate(&system, &mut Unbounded).unwrap();
    let candidate = decode_residual_candidate(&bytes, &mut Unbounded).unwrap();
    let restored = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
        candidate.clone(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        encode_residual_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
    assert_eq!(restored.evaluate_sparse(&[-3.0]).unwrap().jacobian, [-1.0]);
    residual_cusp(restored.jacobian_vector_product(&[0.0], &[0.0]));
    for replacement in [
        Expr::from_i64(-1),
        Expr::from_i64(0),
        Expr::from_i64(1),
        source("_fsym_abs_d2"),
        source("abs"),
    ] {
        let mut forged = candidate.clone();
        replace_entry(&mut forged.entries[0], replacement);
        assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
    }
    let mut missing = candidate.clone();
    missing.entries.clear();
    missing.pattern.nonzeros.clear();
    missing.pattern.nnz = 0;
    missing.validate_structure().unwrap();
    assert!(verify_sparse_jacobian_certificate(&missing, &[]).is_err());

    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let bytes = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
    let candidate = decode_objective_candidate(&bytes, &mut Unbounded).unwrap();
    let restored =
        VerifiedObjective::try_from_candidate(&expression, &variables, candidate.clone()).unwrap();
    assert_eq!(
        encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
    assert_eq!(restored.evaluate(&[-3.0]).unwrap().hessian, [0.0]);
    objective_cusp(restored.hessian_vector_product(&[0.0], &[0.0]));
    for replacement in [
        Expr::from_i64(0),
        Expr::from_i64(1),
        source("_fsym_abs_d1"),
        source("abs"),
    ] {
        let mut forged = candidate.clone();
        replace_entry(&mut forged.hessian.entries[0], replacement);
        assert!(VerifiedObjective::try_from_candidate(&expression, &variables, forged).is_err());
    }
}

#[test]
fn numeric_branch_rules_do_not_change_public_symbolic_receipts() {
    let variable = Symbol::new("x");
    for tag in ["abs", "Abs", "_fsym_abs_d1"] {
        let expression = source(tag);
        let (derivative, proof) = verified_diff(&expression, &variable);
        assert_eq!(derivative, make_diff_term(&expression, &variable));
        assert_eq!(
            classify_diff_rule(&expression, &variable),
            RULE_DIFF_GENERAL
        );
        verify_diff_derivation(&proof, &expression, &variable, &derivative).unwrap();
    }
    let expression = source("abs");
    let system = VerifiedResidualSystem::try_compile(
        std::slice::from_ref(&expression),
        std::slice::from_ref(&variable),
    )
    .unwrap();
    let entry = &system.sparse_jacobian().entries[0];
    assert!(
        verify_diff_derivation(
            &entry.derivation,
            &expression,
            &variable,
            &entry.symbolic_deriv
        )
        .is_err()
    );
}

#[test]
fn shared_numeric_evaluation_keeps_jet_identity_and_lazy_inactive_cusps() {
    let guarded = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadConst(0.0),
        CompiledOp::Equal,
        CompiledOp::JumpIfFalse(6),
        CompiledOp::LoadConst(7.0),
        CompiledOp::Jump(8),
        CompiledOp::LoadVar(0),
        CompiledOp::AbsDerivative,
    ])
    .unwrap();
    let programs = [compile(&source("abs")), guarded.clone(), guarded];
    let batch = CompiledBatch::try_from_programs(programs.to_vec()).unwrap();
    for (point, expected) in [
        (-3.0, [3.0, -1.0, -1.0]),
        (0.0, [0.0, 7.0, 7.0]),
        (3.0, [3.0, 1.0, 1.0]),
    ] {
        assert_eq!(batch.try_eval_finite(&[point]).unwrap(), expected);
    }
    let jets = [
        compile(&source("abs")),
        compile(&source("_fsym_abs_d1")),
        compile(&source("_fsym_abs_d2")),
    ];
    let batch = CompiledBatch::try_from_programs(jets.to_vec()).unwrap();
    assert_eq!(batch.try_eval_finite(&[-3.0]).unwrap(), [3.0, -1.0, 0.0]);
    assert_eq!(
        batch.try_eval_finite(&[0.0]),
        Err(EvalError::NonsmoothSource("abs"))
    );
    assert_eq!(batch.try_eval_finite(&[3.0]).unwrap(), [3.0, 1.0, 0.0]);
}

struct InterruptAt {
    budget: Budget,
    checkpoints: usize,
    stop: Option<usize>,
}

impl InterruptAt {
    fn new(limits: BudgetLimits) -> Self {
        Self {
            budget: Budget::new(limits),
            checkpoints: 0,
            stop: None,
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
        if self.stop.is_some_and(|stop| self.checkpoints >= stop) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn failed_evaluations_consume_one_ledger_and_never_publish_partial_buffers() {
    let system = VerifiedResidualSystem::try_compile(
        &[parse("x+1").unwrap(), source("abs")],
        &[Symbol::new("x")],
    )
    .unwrap();
    let limits = BudgetLimits::uniform(1_000_000, 37);
    let mut measured = InterruptAt::new(limits);
    let mut values = [71.0, 73.0];
    let mut derivatives = [83.0, 89.0];
    system
        .try_eval_sparse_into_with_meter(&[-3.0], &mut values, &mut derivatives, &mut measured)
        .unwrap();
    let checkpoint_count = measured.checkpoints;
    assert_eq!(values, [-2.0, 3.0]);
    assert_eq!(derivatives, [1.0, -1.0]);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let used = limits.dimensions[dimension.index()] - measured.budget.remaining(dimension);
        assert!(used > 0);
        let mut exact_limits = limits;
        exact_limits.dimensions[dimension.index()] = used;
        let mut budget = Budget::new(exact_limits);
        system
            .try_eval_sparse_into_with_meter(&[-3.0], &mut values, &mut derivatives, &mut budget)
            .unwrap();
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);
        values = [71.0, 73.0];
        derivatives = [83.0, 89.0];
        assert!(
            system
                .try_eval_sparse_into_with_meter(
                    &[-3.0],
                    &mut values,
                    &mut derivatives,
                    &mut budget
                )
                .is_err()
        );
        assert_eq!(values, [71.0, 73.0]);
        assert_eq!(derivatives, [83.0, 89.0]);
    }
    for stop in [
        1,
        checkpoint_count / 2,
        checkpoint_count - 1,
        checkpoint_count,
    ] {
        let mut meter = InterruptAt::new(limits);
        meter.stop = Some(stop);
        values = [71.0, 73.0];
        derivatives = [83.0, 89.0];
        assert!(matches!(
            system.try_eval_sparse_into_with_meter(
                &[-3.0],
                &mut values,
                &mut derivatives,
                &mut meter
            ),
            Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
                MeterError::Cancelled
            )))
        ));
        assert_eq!(values, [71.0, 73.0]);
        assert_eq!(derivatives, [83.0, 89.0]);
    }
    values = [71.0, 73.0];
    derivatives = [83.0, 89.0];
    residual_cusp(system.try_eval_sparse_into(&[0.0], &mut values, &mut derivatives));
    assert_eq!(values, [71.0, 73.0]);
    assert_eq!(derivatives, [83.0, 89.0]);
}
