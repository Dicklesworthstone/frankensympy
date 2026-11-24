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
        let curvature =
            -(w + 2.0) / (exponential * exponential * (1.0 + w) * (1.0 + w) * (1.0 + w));
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
        assert_eq!(
            value.try_eval_finite(&[point]).unwrap().to_bits(),
            point.to_bits()
        );
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
        (lambert(Expr::symbol("x"), Some(0)), CompiledOp::LambertW),
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
        vec![
            x.clone(),
            Expr::Add(vec![Expr::from_i64(-1), Expr::from_i64(0)]),
        ],
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
                CompiledExpr::try_compile(&Expr::Function(name.into(), arguments), &variables,)
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
    assert_eq!(
        residual
            .jacobian_vector_product(&point, &[2.0, 3.0])
            .unwrap(),
        [5.0]
    );
    assert_eq!(
        residual
            .jacobian_transpose_vector_product(&point, &[7.0])
            .unwrap(),
        [7.0, 7.0]
    );
    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let result = objective.evaluate(&point).unwrap();
    assert_eq!(result.value, 0.0);
    assert_eq!(result.gradient, [1.0, 1.0]);
    assert_eq!(
        objective.hessian_pattern(),
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    assert_eq!(result.hessian, [0.0, -2.0, -2.0, -2.0]);
    assert_eq!(
        objective
            .hessian_vector_product(&point, &[2.0, 3.0])
            .unwrap(),
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
    assert_eq!(
        encode_residual_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
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
    assert_eq!(
        encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
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
    program
        .try_eval_finite_with_meter(&[1.0], &mut counted)
        .unwrap();
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
        program
            .try_eval_finite_with_meter(&[1.0], &mut exact)
            .unwrap();
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
                &mut CancelAfter {
                    checkpoints: 0,
                    limit
                },
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
    let mut meter = CancelAfter {
        checkpoints: 0,
        limit: 100,
    };
    assert!(matches!(
        residual.try_eval_sparse_into_with_meter(&[1.0], &mut values, &mut derivatives, &mut meter),
        Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Cancelled
        )))
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
    let programs = vec![
        principal.clone(),
        guarded_lower.clone(),
        principal,
        guarded_lower,
    ];
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

// Additional exact-dyadic, conditioning, and receipt regressions.
mod numeric_regressions {
    //! Real Lambert-W branches compose with independently checked programs.
    //! Binary64 expectations are numerical observations, not enclosure receipts.

    use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
    use fsym_calculus::VerifiedResidualSystem;
    use fsym_calculus::compile::{CompiledBatch, CompiledExpr, CompiledOp, EvalError};
    use fsym_calculus::program_artifact::{
        admit_objective_candidate, decode_objective_candidate, encode_objective_candidate,
    };
    use fsym_calculus::proof::{make_diff_term, verified_diff, verify_diff_derivation};
    use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
    use fsym_calculus::sparse_jacobian::{
        SparseJacobian, compiled_expression_matches, verify_sparse_jacobian_certificate,
    };
    use fsym_core::{BigInt, Expr, Symbol, parse};
    use fsym_proof_kernel::{Claim, ProofRule};
    use std::collections::HashMap;

    fn function(argument: Expr, branch: i64) -> Expr {
        Expr::Function("LambertW".into(), vec![argument, Expr::from_i64(branch)])
    }

    fn jet(argument: Expr, branch: i64, order: u32) -> Expr {
        let name = match (branch, order) {
            (0, 1) => "_fsym_lambert_w0_d1",
            (0, 2) => "_fsym_lambert_w0_d2",
            (-1, 1) => "_fsym_lambert_wm1_d1",
            (-1, 2) => "_fsym_lambert_wm1_d2",
            _ => panic!("invalid test jet"),
        };
        Expr::Function(name.into(), vec![argument])
    }

    fn compile(source: &Expr) -> CompiledExpr {
        CompiledExpr::try_compile(source, &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
    }

    fn close(actual: f64, expected: f64) {
        assert!(actual.is_finite(), "nonfinite result for {expected:e}");
        assert!(
            (actual - expected).abs() <= 3e-13 * expected.abs() + 4.0 * f64::from_bits(1),
            "actual={actual:e}, expected={expected:e}"
        );
    }

    #[test]
    fn values_and_jets_match_independent_high_precision_anchors() {
        // mpmath 160-decimal evaluation at the exact binary64 dyadic input,
        // with derivatives from exp(-W)/(1+W) and -(W+2)*exp(-2W)/(1+W)^3.
        // These references do not run the native rational-ball implementation.
        for (argument, branch, expected) in [
            (
                1.0,
                0,
                [
                    0.567_143_290_409_783_8,
                    0.361_896_256_634_889_2,
                    -0.214_540_646_282_143_7,
                ],
            ),
            (
                -0.25,
                0,
                [
                    -0.357_402_956_181_388_9,
                    2.224_740_743_017_017_4,
                    -12.651_765_402_613_098,
                ],
            ),
            (
                -0.25,
                -1,
                [
                    -2.153_292_364_110_35,
                    -7.468_331_296_102_53,
                    -7.413_584_731_766_361,
                ],
            ),
            (
                f64::from_bits(0xbfd7_8b56_362c_ef37),
                0,
                [
                    -0.999_999_984_695_745_9,
                    177_616_090.535_021_16,
                    -2.061_353_388_076_378e24,
                ],
            ),
            (
                f64::from_bits(0xbfd7_8b56_362c_ef37),
                -1,
                [
                    -1.000_000_015_304_254_3,
                    -177_616_094.159_396_92,
                    2.061_353_388_076_378e24,
                ],
            ),
        ] {
            for (order, source) in [
                function(Expr::symbol("x"), branch),
                jet(Expr::symbol("x"), branch, 1),
                jet(Expr::symbol("x"), branch, 2),
            ]
            .iter()
            .enumerate()
            {
                close(
                    compile(source).try_eval_finite(&[argument]).unwrap(),
                    expected[order],
                );
            }
        }
    }

    #[test]
    fn lower_curvature_retains_the_sign_and_scale_next_to_its_inflection() {
        // Independent 180-decimal mpmath evaluation at the exact dyadic inputs.
        // W+2 cancels to about 1e-16 here; rounding W before forming this factor
        // would lose the curvature, including its sign across -2/e^2.
        let program = compile(&jet(Expr::symbol("x"), -1, 2));
        for (bits, expected) in [
            (0xbfd1_52aa_a3bf_81cd, 3.080_533_036_575_087_6e-14),
            (0xbfd1_52aa_a3bf_81cc, 8.410_533_599_131_757e-15),
            (0xbfd1_52aa_a3bf_81cb, -1.398_426_316_748_734_4e-14),
        ] {
            close(
                program.try_eval_finite(&[f64::from_bits(bits)]).unwrap(),
                expected,
            );
        }
    }

    #[test]
    fn nested_principal_zero_preserves_the_complete_numeric_chain() {
        let source = function(function(Expr::symbol("x"), 0), 0);
        let objective = VerifiedObjective::try_compile(&source, &[Symbol::new("x")]).unwrap();
        let result = objective.evaluate(&[0.0]).unwrap();
        // For W0(W0(x)), f''(0)=W0''(0)*W0'(0)^2+W0'(0)*W0''(0).
        assert_eq!(result.value, 0.0);
        assert_eq!(result.gradient, [1.0]);
        assert_eq!(result.hessian, [-4.0]);
        assert_eq!(
            objective.hessian_vector_product(&[0.0], &[3.0]).unwrap(),
            [-12.0]
        );
    }

    #[test]
    fn principal_origin_and_subnormal_arguments_keep_the_analytic_extension() {
        let variables = [Symbol::new("x")];
        let source = function(Expr::symbol("x"), 0);
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let program = compile(&source);
        for argument in [
            0.0,
            -0.0,
            1e-200,
            -1e-200,
            f64::from_bits(1),
            -f64::from_bits(1),
        ] {
            let result = objective.evaluate(&[argument]).unwrap();
            close(result.value, argument);
            assert_eq!(result.gradient, [1.0]);
            assert_eq!(result.hessian, [-2.0]);
            if argument == 0.0 {
                assert_eq!(
                    program.try_eval_finite(&[argument]).unwrap().to_bits(),
                    argument.to_bits()
                );
            }
        }
    }

    #[test]
    fn finite_lower_tail_values_and_jets_match_independent_anchors() {
        for (source, expected) in [
            (function(Expr::symbol("x"), -1), -235.721_158_875_685_32),
            (jet(Expr::symbol("x"), -1, 1), -1.004_260_374_330_077_5e100),
            (jet(Expr::symbol("x"), -1, 2), -1.004_242_146_211_487_6e200),
        ] {
            close(
                compile(&source).try_eval_finite(&[-1e-100]).unwrap(),
                expected,
            );
        }
    }

    #[test]
    fn difficult_tail_arguments_stop_at_the_callers_budget_without_fallback() {
        // The inherited exact kernel has finite intrinsic work ceilings, even
        // for valid finite inputs. W-1(-1e-308) was observed to hit that ceiling.
        // Exercise prompt typed refusal with a smaller caller budget instead of
        // repeating the full expensive inverse solve on every test run.
        for (branch, point) in [(-1, -1e-308), (0, f64::MAX)] {
            for source in [
                function(Expr::symbol("x"), branch),
                jet(Expr::symbol("x"), branch, 1),
                jet(Expr::symbol("x"), branch, 2),
            ] {
                let mut limits = BudgetLimits::uniform(u64::MAX, 0);
                limits.dimensions[Dimension::ComputeSteps.index()] = 100_000;
                let mut budget = Budget::new(limits);
                assert!(matches!(
                    compile(&source).try_eval_finite_with_meter(&[point], &mut budget),
                    Err(EvalError::Meter(MeterError::Budget(_)))
                ));
            }
        }
    }

    #[test]
    fn exact_dyadic_branch_admission_never_clamps_the_rounded_branch_point() {
        let invalid_branch_point = f64::from_bits(0xbfd7_8b56_362c_ef38);
        assert_eq!(invalid_branch_point, -1.0 / std::f64::consts::E);
        for branch in [0, -1] {
            for source in [
                function(Expr::symbol("x"), branch),
                jet(Expr::symbol("x"), branch, 1),
                jet(Expr::symbol("x"), branch, 2),
            ] {
                let program = compile(&source);
                for point in [invalid_branch_point, -0.5, f64::NAN, f64::INFINITY] {
                    assert!(program.try_eval_finite(&[point]).is_err());
                }
                if branch == -1 {
                    for point in [0.0, -0.0, 1.0] {
                        assert!(program.try_eval_finite(&[point]).is_err());
                    }
                }
            }
        }
    }

    #[test]
    fn checked_residuals_and_hessians_preserve_branch_and_complete_chain_rules() {
        let variables = [Symbol::new("x"), Symbol::new("y")];
        for (branch, argument, w) in [
            (0, 1.0_f64, 0.567_143_290_409_783_8_f64),
            (-1, -0.25, -2.153_292_364_110_35),
        ] {
            let source = function(parse("3*x+y^2").unwrap(), branch);
            let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
            let residual =
                VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables)
                    .unwrap();
            let point = [(argument - 0.25) / 3.0, 0.5];
            let slope = (-w).exp() / (1.0 + w);
            let curvature = -(w + 2.0) * (-2.0 * w).exp() / (1.0 + w).powi(3);
            let evaluated = objective.evaluate(&point).unwrap();
            close(evaluated.value, w);
            for (actual, expected) in evaluated.gradient.iter().zip([3.0 * slope, slope]) {
                close(*actual, expected);
            }
            for (actual, expected) in evaluated.hessian.iter().zip([
                9.0 * curvature,
                3.0 * curvature,
                3.0 * curvature,
                curvature + 2.0 * slope,
            ]) {
                close(*actual, expected);
            }
            let sparse = residual.evaluate_sparse(&point).unwrap();
            assert_eq!(sparse.jacobian, evaluated.gradient);
            let product = objective
                .hessian_vector_product(&point, &[2.0, 3.0])
                .unwrap();
            close(product[0], 27.0 * curvature);
            close(product[1], 9.0 * curvature + 6.0 * slope);
        }
    }

    #[test]
    fn independent_source_replay_rejects_branch_and_jet_opcode_mutation() {
        let mapping = HashMap::from([(Symbol::new("x"), 0)]);
        for (source, operation, wrong) in [
            (
                function(Expr::symbol("x"), 0),
                CompiledOp::LambertW,
                CompiledOp::LambertWMinusOne,
            ),
            (
                function(Expr::symbol("x"), -1),
                CompiledOp::LambertWMinusOne,
                CompiledOp::LambertW,
            ),
            (
                jet(Expr::symbol("x"), 0, 1),
                CompiledOp::LambertWDerivative,
                CompiledOp::LambertWMinusOneDerivative,
            ),
            (
                jet(Expr::symbol("x"), -1, 2),
                CompiledOp::LambertWMinusOneSecondDerivative,
                CompiledOp::LambertWMinusOneDerivative,
            ),
        ] {
            assert_eq!(
                CompiledExpr::try_from_ops(vec![operation]),
                Err(EvalError::StackUnderflow)
            );
            let correct = compile(&source);
            assert!(compiled_expression_matches(&source, &mapping, &correct).unwrap());
            let forged = CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), wrong]).unwrap();
            assert!(!compiled_expression_matches(&source, &mapping, &forged).unwrap());
        }
        for branch in [
            Expr::from_i64(1),
            Expr::symbol("x"),
            Expr::rational(-1, 2).unwrap(),
            Expr::Integer(BigInt::from(1) << 4096u32),
        ] {
            let source = Expr::Function("LambertW".into(), vec![Expr::symbol("x"), branch]);
            assert!(CompiledExpr::try_compile(&source, &mapping).is_err());
        }
        for arguments in [
            vec![],
            vec![Expr::symbol("x"), Expr::from_i64(0), Expr::from_i64(0)],
        ] {
            assert!(
                CompiledExpr::try_compile(&Expr::Function("LambertW".into(), arguments), &mapping)
                    .is_err()
            );
        }
    }

    #[test]
    fn independent_derivative_rules_reject_forged_slopes_and_curvatures() {
        let variables = [Symbol::new("x")];
        for branch in [0, -1] {
            let source = function(parse("3*x").unwrap(), branch);
            let candidate =
                SparseJacobian::try_build(std::slice::from_ref(&source), &variables).unwrap();
            verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
            for wrong in [
                Expr::from_i64(0),
                -candidate.entries[0].symbolic_deriv.clone(),
                jet(parse("3*x").unwrap(), branch, 1),
                jet(parse("3*x").unwrap(), if branch == 0 { -1 } else { 0 }, 1) * Expr::from_i64(3),
            ] {
                let mut forged = candidate.clone();
                let entry = &mut forged.entries[0];
                entry.symbolic_deriv = wrong.clone();
                entry.derivation.steps[0].claim =
                    Claim::equality(make_diff_term(&source, &variables[0]), wrong.clone());
                let ProofRule::DefinitionalReduction { rhs, .. } =
                    &mut entry.derivation.steps[0].rule
                else {
                    panic!("expected derivative rule")
                };
                *rhs = wrong.clone();
                entry.compiled_deriv = compile(&wrong);
                assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
            }
            let source = function(Expr::symbol("x"), branch).pow(Expr::from_i64(1));
            let original = ObjectiveCandidate::try_build(&source, &variables).unwrap();
            VerifiedObjective::try_from_candidate(&source, &variables, original.clone()).unwrap();
            for wrong in [
                Expr::from_i64(0),
                jet(Expr::symbol("x"), if branch == 0 { -1 } else { 0 }, 2),
            ] {
                let mut forged = original.clone();
                let hessian_source = &forged.hessian.residuals[0];
                let entry = &mut forged.hessian.entries[0];
                entry.symbolic_deriv = wrong.clone();
                entry.derivation.steps[0].claim =
                    Claim::equality(make_diff_term(hessian_source, &variables[0]), wrong.clone());
                let ProofRule::DefinitionalReduction { rhs, .. } =
                    &mut entry.derivation.steps[0].rule
                else {
                    panic!("expected curvature rule")
                };
                *rhs = wrong.clone();
                entry.compiled_deriv = compile(&wrong);
                assert!(
                    VerifiedObjective::try_from_candidate(&source, &variables, forged).is_err()
                );
            }
        }
    }

    #[test]
    fn public_symbolic_derivative_retains_the_original_fixed_branch_formula() {
        let variable = Symbol::new("x");
        for branch in [None, Some(0), Some(-1), Some(2)] {
            let source = if let Some(branch) = branch {
                function(Expr::symbol("x"), branch)
            } else {
                Expr::Function("LambertW".into(), vec![Expr::symbol("x")])
            };
            let (derivative, receipt) = verified_diff(&source, &variable);
            verify_diff_derivation(&receipt, &source, &variable, &derivative).unwrap();
            assert!(!derivative.to_string().contains("_fsym_lambert"));
            let mut forged = receipt.clone();
            let ProofRule::DefinitionalReduction { rhs, .. } = &mut forged.steps[0].rule else {
                panic!("expected derivative rule")
            };
            *rhs = Expr::from_i64(0);
            forged.steps[0].claim =
                Claim::equality(make_diff_term(&source, &variable), Expr::from_i64(0));
            assert!(
                verify_diff_derivation(&forged, &source, &variable, &Expr::from_i64(0)).is_err()
            );
        }
    }

    #[test]
    fn branch_tagged_programs_roundtrip_and_cannot_alias_shared_values() {
        let sources = [
            function(Expr::symbol("x"), 0),
            function(Expr::symbol("x"), -1),
        ];
        let programs: Vec<_> = sources.iter().map(compile).collect();
        let batch = CompiledBatch::try_from_programs(programs.clone()).unwrap();
        let values = batch.try_eval_finite(&[-0.25]).unwrap();
        assert!(values[0] > -1.0 && values[1] < -1.0);
        for (program, value) in programs.iter().zip(&values) {
            assert_eq!(program.try_eval_finite(&[-0.25]).unwrap(), *value);
            let encoded = serde_json::to_string(program).unwrap();
            let restored: CompiledExpr = serde_json::from_str(&encoded).unwrap();
            assert_eq!(&restored, program);
        }
        let variables = [Symbol::new("x")];
        let objective = VerifiedObjective::try_compile(&sources[1], &variables).unwrap();
        let encoded = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
        let candidate = decode_objective_candidate(&encoded, &mut Unbounded).unwrap();
        assert!(
            admit_objective_candidate(&sources[0], &variables, candidate.clone(), &mut Unbounded)
                .is_err()
        );
        let restored =
            admit_objective_candidate(&sources[1], &variables, candidate, &mut Unbounded).unwrap();
        assert_eq!(
            encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
            encoded
        );
        assert_eq!(
            restored.evaluate(&[-0.25]).unwrap(),
            objective.evaluate(&[-0.25]).unwrap()
        );
    }

    #[derive(Default)]
    struct Checkpoints {
        remaining: Option<usize>,
        reached: usize,
    }

    impl BudgetMeter for Checkpoints {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.reached += 1;
            if let Some(remaining) = self.remaining.as_mut() {
                if *remaining == 0 {
                    return Err(MeterError::Cancelled);
                }
                *remaining -= 1;
            }
            Ok(())
        }
    }

    #[test]
    fn caller_budgets_cancellation_and_original_domains_survive_publication() {
        let variables = [Symbol::new("x")];
        let source = function(Expr::symbol("x"), 0);
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let mut counted = Checkpoints::default();
        objective
            .try_eval_into_with_meter(&[1.0], &mut [0.0], &mut [0.0], &mut counted)
            .unwrap();
        for remaining in [0, counted.reached / 2, counted.reached - 1] {
            let mut meter = Checkpoints {
                remaining: Some(remaining),
                reached: 0,
            };
            let (mut gradient, mut hessian) = ([13.0], [17.0]);
            assert!(
                objective
                    .try_eval_into_with_meter(&[1.0], &mut gradient, &mut hessian, &mut meter)
                    .is_err()
            );
            assert_eq!(gradient, [13.0]);
            assert_eq!(hessian, [17.0]);
        }
        let mut exhausted = Budget::new(BudgetLimits::uniform(0, 0));
        assert!(matches!(
            compile(&source).try_eval_finite_with_meter(&[1.0], &mut exhausted),
            Err(EvalError::Meter(_))
        ));
        for (branch, point) in [(0, -0.5), (-1, 0.0)] {
            let held = Expr::Mul(vec![Expr::from_i64(0), function(Expr::symbol("x"), branch)]);
            let guarded = VerifiedObjective::try_compile(&held, &variables).unwrap();
            assert!(guarded.evaluate(&[point]).is_err());
            assert!(guarded.hessian_vector_product(&[point], &[0.0]).is_err());
        }
        let held = function(
            Expr::Mul(vec![
                Expr::from_i64(0),
                Expr::symbol("x").pow(Expr::from_i64(-1)),
            ]),
            0,
        );
        assert!(
            VerifiedObjective::try_compile(&held, &variables)
                .unwrap()
                .evaluate(&[0.0])
                .is_err()
        );
    }
}
