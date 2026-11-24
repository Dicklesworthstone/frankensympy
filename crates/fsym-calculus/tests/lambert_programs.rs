//! Fixed real Lambert-W branches compose with checked numeric programs.
//! Binary64 observations remain separate from original-source certificates.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{CompiledBatch, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::program_artifact::{
    decode_objective_candidate, decode_residual_candidate, encode_objective_candidate,
    encode_residual_candidate,
};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::{
    SparseJacobianEntry, compiled_expression_matches, verify_sparse_jacobian_certificate,
};
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol, parse};
use fsym_proof_kernel::{Claim, ProofRule};
use std::collections::HashMap;

fn lambert(argument: Expr, branch: Option<i64>) -> Expr {
    let mut arguments = vec![argument];
    if let Some(branch) = branch {
        arguments.push(Expr::from_i64(branch));
    }
    Expr::Function("LambertW".into(), arguments)
}

fn jet(branch: i64, order: usize) -> Expr {
    let name = match (branch, order) {
        (0, 1) => "_fsym_lambert_w0_d1",
        (0, 2) => "_fsym_lambert_w0_d2",
        (-1, 1) => "_fsym_lambert_wm1_d1",
        (-1, 2) => "_fsym_lambert_wm1_d2",
        _ => panic!("invalid test jet"),
    };
    Expr::Function(name.into(), vec![Expr::symbol("x")])
}

fn compile(source: &Expr) -> CompiledExpr {
    CompiledExpr::try_compile(source, &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    let tolerance = (2e-12 * expected.abs()).max(4.0 * f64::from_bits(1));
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}

#[test]
fn both_real_branches_and_jets_match_closed_form_inverse_anchors() {
    // If z=w*exp(w), the selected inverse equals w. These anchors choose
    // the branch before evaluation and derive derivatives independently by
    // implicit differentiation, without calling the inverse generator.
    for (branch, w) in [(0, 1.0_f64), (0, 2.0), (-1, -3.0)] {
        let exponential = w.exp();
        let argument = w * exponential;
        let slope = 1.0 / (exponential * (1.0 + w));
        let curvature = -(w + 2.0)
            / (exponential * exponential * (1.0 + w) * (1.0 + w) * (1.0 + w));
        close(
            compile(&lambert(Expr::symbol("x"), Some(branch)))
                .try_eval_finite(&[argument])
                .unwrap(),
            w,
        );
        close(
            compile(&jet(branch, 1))
                .try_eval_finite(&[argument])
                .unwrap(),
            slope,
        );
        close(
            compile(&jet(branch, 2))
                .try_eval_finite(&[argument])
                .unwrap(),
            curvature,
        );
    }
    close(
        compile(&lambert(Expr::symbol("x"), None))
            .try_eval_finite(&[1.0])
            .unwrap(),
        0.567_143_290_409_783_8,
    );
}

#[test]
fn principal_origin_and_tiny_inputs_keep_the_analytic_extension() {
    let value = compile(&lambert(Expr::symbol("x"), None));
    let first = compile(&jet(0, 1));
    let second = compile(&jet(0, 2));
    for point in [0.0_f64, -0.0] {
        assert_eq!(value.try_eval_finite(&[point]).unwrap().to_bits(), point.to_bits());
        assert_eq!(first.try_eval_finite(&[point]).unwrap(), 1.0);
        assert_eq!(second.try_eval_finite(&[point]).unwrap(), -2.0);
    }
    // W0(x)=x-x^2+O(x^3): the omitted relative corrections here are
    // far below a binary64 ulp, while the value must remain nonzero.
    for point in [1e-30_f64, -1e-30] {
        let actual = value.try_eval_finite(&[point]).unwrap();
        assert_ne!(actual, 0.0);
        close(actual, point);
        close(first.try_eval_finite(&[point]).unwrap(), 1.0);
        close(second.try_eval_finite(&[point]).unwrap(), -2.0);
    }
}

#[test]
fn branch_and_derivative_order_are_bound_to_distinct_serialized_opcodes() {
    let variables = HashMap::from([(Symbol::new("x"), 0)]);
    let cases = [
        (lambert(Expr::symbol("x"), None), CompiledOp::LambertW),
        (
            lambert(Expr::symbol("x"), Some(0)),
            CompiledOp::LambertW,
        ),
        (
            lambert(Expr::symbol("x"), Some(-1)),
            CompiledOp::LambertWMinusOne,
        ),
        (jet(0, 1), CompiledOp::LambertWDerivative),
        (jet(0, 2), CompiledOp::LambertWSecondDerivative),
        (jet(-1, 1), CompiledOp::LambertWMinusOneDerivative),
        (jet(-1, 2), CompiledOp::LambertWMinusOneSecondDerivative),
    ];
    for (source, operation) in &cases {
        let program = compile(source);
        assert_eq!(program.ops(), &[CompiledOp::LoadVar(0), operation.clone()]);
        let bytes = serde_json::to_vec(&program).unwrap();
        let decoded: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, program);
        assert!(compiled_expression_matches(source, &variables, &decoded).unwrap());
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation.clone()]),
            Err(EvalError::StackUnderflow)
        );
        for (_, other) in &cases {
            if other == operation {
                continue;
            }
            let forged =
                CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), other.clone()]).unwrap();
            assert!(!compiled_expression_matches(source, &variables, &forged).unwrap());
        }
    }
}

#[test]
fn nonliteral_nonreal_and_malformed_branch_arguments_refuse_compilation() {
    let x = Expr::symbol("x");
    let variables = HashMap::from([(Symbol::new("x"), 0), (Symbol::new("k"), 1)]);
    let valid = compile(&lambert(x.clone(), None));
    for arguments in [
        vec![],
        vec![x.clone(), Expr::from_i64(1)],
        vec![x.clone(), Expr::from_i64(-2)],
        vec![x.clone(), Expr::symbol("k")],
        vec![x.clone(), Expr::Add(vec![Expr::from_i64(-1), Expr::from_i64(0)])],
        vec![x.clone(), Expr::rational(-1, 2).unwrap()],
        vec![x.clone(), Expr::from_i64(-1), Expr::from_i64(0)],
    ] {
        let invalid = Expr::Function("LambertW".into(), arguments);
        assert!(CompiledExpr::try_compile(&invalid, &variables).is_err());
        assert!(compiled_expression_matches(&invalid, &variables, &valid).is_err());
    }
    for name in [
        "_fsym_lambert_w0_d1",
        "_fsym_lambert_w0_d2",
        "_fsym_lambert_wm1_d1",
        "_fsym_lambert_wm1_d2",
    ] {
        for arguments in [vec![], vec![x.clone(), Expr::from_i64(0)]] {
            assert!(
                CompiledExpr::try_compile(
                    &Expr::Function(name.into(), arguments),
                    &variables,
                )
                .is_err()
            );
        }
    }
}

#[test]
fn nonlinear_principal_zero_composes_with_sparse_jacobian_and_hessian_products() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let expression = lambert(parse("x^2+y").unwrap(), None);
    let point = [0.5, -0.25];
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables).unwrap();
    let result = residual.evaluate_sparse(&point).unwrap();
    assert_eq!(result.residuals, [0.0]);
    assert_eq!(result.jacobian, [1.0, 1.0]);
    assert_eq!(residual.jacobian_vector_product(&point, &[2.0, 3.0]).unwrap(), [5.0]);
    assert_eq!(
        residual.jacobian_transpose_vector_product(&point, &[7.0]).unwrap(),
        [7.0, 7.0]
    );
    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let result = objective.evaluate(&point).unwrap();
    assert_eq!(result.value, 0.0);
    assert_eq!(result.gradient, [1.0, 1.0]);
    assert_eq!(objective.hessian_pattern(), [(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(result.hessian, [0.0, -2.0, -2.0, -2.0]);
    assert_eq!(
        objective.hessian_vector_product(&point, &[2.0, 3.0]).unwrap(),
        [-6.0, -10.0]
    );
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
fn portable_program_replay_rejects_forged_branch_and_derivative_claims() {
    let expression = lambert(Expr::symbol("x"), None);
    let variables = [Symbol::new("x")];
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables).unwrap();
    let bytes = encode_residual_candidate(&residual, &mut Unbounded).unwrap();
    let candidate = decode_residual_candidate(&bytes, &mut Unbounded).unwrap();
    let restored = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
        candidate.clone(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(encode_residual_candidate(&restored, &mut Unbounded).unwrap(), bytes);
    assert_eq!(restored.evaluate_sparse(&[0.0]).unwrap().jacobian, [1.0]);
    for replacement in [Expr::from_i64(0), jet(-1, 1), jet(0, 2)] {
        let mut forged = candidate.clone();
        replace_entry(&mut forged.entries[0], replacement);
        assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
    }

    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let bytes = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
    let candidate = decode_objective_candidate(&bytes, &mut Unbounded).unwrap();
    let restored =
        VerifiedObjective::try_from_candidate(&expression, &variables, candidate.clone()).unwrap();
    assert_eq!(encode_objective_candidate(&restored, &mut Unbounded).unwrap(), bytes);
    assert_eq!(restored.evaluate(&[0.0]).unwrap().hessian, [-2.0]);
    for replacement in [Expr::from_i64(0), jet(-1, 2), jet(0, 1)] {
        let mut forged = candidate.clone();
        replace_entry(&mut forged.hessian.entries[0], replacement);
        assert!(VerifiedObjective::try_from_candidate(&expression, &variables, forged).is_err());
    }
}

#[test]
fn original_lambert_domains_survive_zero_multipliers_and_derivative_elision() {
    let variable = Symbol::new("x");
    for (inner, branch, point) in [
        (Expr::symbol("x"), 0, -0.5),
        (Expr::symbol("x"), -1, 0.0),
        (Expr::symbol("x"), -1, 0.1),
        (Expr::pow(Expr::symbol("x"), Expr::from_i64(-1)), 0, 0.0),
    ] {
        let expression = Expr::Add(vec![
            Expr::from_i64(7),
            Expr::Mul(vec![Expr::from_i64(0), lambert(inner, Some(branch))]),
        ]);
        let residual = VerifiedResidualSystem::try_compile(
            std::slice::from_ref(&expression),
            std::slice::from_ref(&variable),
        )
        .unwrap();
        let objective =
            VerifiedObjective::try_compile(&expression, std::slice::from_ref(&variable)).unwrap();
        assert!(residual.evaluate_sparse(&[point]).is_err());
        assert!(residual.jacobian_vector_product(&[point], &[0.0]).is_err());
        assert!(objective.evaluate(&[point]).is_err());
        assert!(objective.hessian_vector_product(&[point], &[0.0]).is_err());
    }
}

#[test]
fn lower_branch_failures_leave_all_destination_buffers_unchanged() {
    let residual = VerifiedResidualSystem::try_compile(
        &[parse("x+1").unwrap(), lambert(Expr::symbol("x"), Some(-1))],
        &[Symbol::new("x")],
    )
    .unwrap();
    let mut values = [71.0, 73.0];
    let mut derivatives = [83.0, 89.0];
    assert!(
        residual
            .try_eval_sparse_into(&[0.0], &mut values, &mut derivatives)
            .is_err()
    );
    assert_eq!(values, [71.0, 73.0]);
    assert_eq!(derivatives, [83.0, 89.0]);
}

struct CancelAfter {
    checkpoints: usize,
    limit: usize,
}

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
        Ok(())
    }

    fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints >= self.limit {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn inverse_work_shares_cumulative_caller_limits_and_propagates_cancellation() {
    let program = compile(&lambert(Expr::symbol("x"), None));
    let limits = BudgetLimits::uniform(u64::MAX, 37);
    let mut counted = Budget::new(limits);
    program.try_eval_finite_with_meter(&[1.0], &mut counted).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let used = limits.dimensions[dimension.index()] - counted.remaining(dimension);
        assert!(used > 0);
        let mut exact_limits = limits;
        exact_limits.dimensions[dimension.index()] = used;
        let mut exact = Budget::new(exact_limits);
        program.try_eval_finite_with_meter(&[1.0], &mut exact).unwrap();
        assert_eq!(exact.remaining(dimension), 0);
        assert_eq!(exact.verifier_remaining(), 37);
        assert!(matches!(
            program.try_eval_finite_with_meter(&[1.0], &mut exact),
            Err(EvalError::Meter(MeterError::Budget(_)))
        ));
        exact_limits.dimensions[dimension.index()] -= 1;
        assert!(matches!(
            program.try_eval_finite_with_meter(&[1.0], &mut Budget::new(exact_limits)),
            Err(EvalError::Meter(MeterError::Budget(_)))
        ));
    }
    for limit in [1, 20, 100] {
        assert_eq!(
            program.try_eval_finite_with_meter(
                &[1.0],
                &mut CancelAfter { checkpoints: 0, limit },
            ),
            Err(EvalError::Meter(MeterError::Cancelled))
        );
    }
}

#[test]
fn cancellation_after_a_completed_residual_cannot_publish_partial_lambert_results() {
    let residual = VerifiedResidualSystem::try_compile(
        &[parse("x+1").unwrap(), lambert(Expr::symbol("x"), None)],
        &[Symbol::new("x")],
    )
    .unwrap();
    let mut values = [71.0, 73.0];
    let mut derivatives = [83.0, 89.0];
    let mut meter = CancelAfter { checkpoints: 0, limit: 100 };
    assert!(matches!(
        residual.try_eval_sparse_into_with_meter(&[1.0], &mut values, &mut derivatives, &mut meter),
        Err(ResidualEvaluationError::Evaluation(EvalError::Meter(MeterError::Cancelled)))
    ));
    assert_eq!(values, [71.0, 73.0]);
    assert_eq!(derivatives, [83.0, 89.0]);
}

#[test]
fn shared_programs_keep_branch_identity_and_inactive_piecewise_domains() {
    let principal = compile(&lambert(Expr::symbol("x"), None));
    let guarded_lower = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadConst(0.0),
        CompiledOp::Equal,
        CompiledOp::JumpIfFalse(6),
        CompiledOp::LoadConst(0.0),
        CompiledOp::Jump(8),
        CompiledOp::LoadVar(0),
        CompiledOp::LambertWMinusOne,
    ])
    .unwrap();
    let programs = vec![principal.clone(), guarded_lower.clone(), principal, guarded_lower];
    let batch = CompiledBatch::try_from_programs(programs.clone()).unwrap();
    assert!(batch.shared_subexpression_count() > 0);
    for point in [0.0, -0.1] {
        let expected = programs
            .iter()
            .map(|program| program.try_eval_finite(&[point]).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(batch.try_eval_finite(&[point]).unwrap(), expected);
    }
}

#[test]
fn finite_evaluation_never_accepts_nonfinite_arguments_or_inner_results() {
    for branch in [None, Some(-1)] {
        let program = compile(&lambert(Expr::symbol("x"), branch));
        for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                program.try_eval_finite(&[point]),
                Err(EvalError::NonFiniteValue)
            );
        }
    }
}
