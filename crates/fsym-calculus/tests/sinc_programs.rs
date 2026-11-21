//! Entire sinc jets compose with checked programs without inventing a pole.
//! Binary64 anchors are numerical observations, not certified enclosures.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{CompiledBatch, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::program_artifact::{
    decode_objective_candidate, decode_residual_candidate, encode_objective_candidate,
    encode_residual_candidate,
};
use fsym_calculus::proof::{verified_diff, verify_diff_derivation};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::{
    SparseJacobianEntry, compiled_expression_matches, verify_sparse_jacobian_certificate,
};
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_proof_kernel::{Claim, ProofRule};
use std::collections::HashMap;

const JETS: [&str; 3] = ["sinc", "_fsym_sinc_d1", "_fsym_sinc_d2"];

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn source(name: &str) -> Expr {
    function(name, Expr::symbol("x"))
}

fn compile(expression: &Expr) -> CompiledExpr {
    CompiledExpr::try_compile(expression, &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
}

fn scalar(name: &str) -> CompiledExpr {
    compile(&source(name))
}

fn close(actual: f64, expected: f64) {
    let tolerance = (2e-13 * expected.abs()).max(4.0 * f64::from_bits(1));
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}

#[test]
fn entire_value_and_two_jets_match_independent_decimal_anchors() {
    // 100-digit Decimal sums of (-1)^n*x^(2n-k)/((2n+1)*(2n-k)!),
    // evaluated with exact factorials through n=179, independently of the
    // binary64 recurrence and quotient evaluator. Rounded once into literals.
    let anchors = [
        (
            0.1,
            [
                0.998_334_166_468_281_5,
                -0.033_300_011_902_557_57,
                -0.332_333_928_417_130_1,
            ],
        ),
        (
            0.5,
            [
                0.958_851_077_208_406,
                -0.162_537_030_636_066_57,
                -0.308_702_954_664_139_7,
            ],
        ),
        (
            1.0,
            [
                0.841_470_984_807_896_5,
                -0.301_168_678_939_756_8,
                -0.239_133_626_928_382_93,
            ],
        ),
        (
            2.0,
            [
                0.454_648_713_412_840_85,
                -0.435_397_774_979_991_6,
                -0.019_250_938_432_849_23,
            ],
        ),
        (
            3.0,
            [
                0.047_040_002_686_622_41,
                -0.345_677_499_762_355_95,
                0.183_411_663_821_614_9,
            ],
        ),
        (
            10.0,
            [
                -0.054_402_111_088_936_98,
                -0.078_466_941_798_751_55,
                0.070_095_499_448_687_29,
            ],
        ),
    ];
    for (order, name) in JETS.iter().enumerate() {
        let program = scalar(name);
        for (point, expected) in anchors {
            close(program.try_eval_finite(&[point]).unwrap(), expected[order]);
            close(
                program.try_eval_finite(&[-point]).unwrap(),
                if order == 1 {
                    -expected[order]
                } else {
                    expected[order]
                },
            );
        }
    }
}

#[test]
fn origin_and_near_zero_keep_the_analytic_extension_and_tiny_slopes() {
    for point in [0.0_f64, -0.0] {
        assert_eq!(scalar("sinc").try_eval_finite(&[point]).unwrap(), 1.0);
        assert_eq!(
            scalar("_fsym_sinc_d1").try_eval_finite(&[point]).unwrap(),
            0.0
        );
        assert_eq!(
            scalar("_fsym_sinc_d2").try_eval_finite(&[point]).unwrap(),
            -1.0 / 3.0
        );
    }
    for point in [
        1e-8_f64,
        -1e-8,
        1e-300,
        -1e-300,
        f64::from_bits(3),
        -f64::from_bits(3),
    ] {
        close(scalar("sinc").try_eval_finite(&[point]).unwrap(), 1.0);
        let slope = scalar("_fsym_sinc_d1").try_eval_finite(&[point]).unwrap();
        assert_ne!(slope, 0.0, "tiny slope at {point:e}");
        close(slope, -point / 3.0);
        close(
            scalar("_fsym_sinc_d2").try_eval_finite(&[point]).unwrap(),
            -1.0 / 3.0,
        );
    }
}

#[test]
fn finite_large_arguments_do_not_create_square_overflow_or_wrong_parity() {
    for point in [1e150_f64, 1e200, f64::MAX] {
        for (order, name) in JETS.iter().enumerate() {
            let positive = scalar(name).try_eval_finite(&[point]).unwrap();
            let negative = scalar(name).try_eval_finite(&[-point]).unwrap();
            assert!(positive.is_finite());
            assert!(positive.abs() <= 2.0 / point);
            assert_eq!(negative, if order == 1 { -positive } else { positive });
        }
    }
    for name in JETS {
        for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                scalar(name).try_eval_finite(&[point]),
                Err(EvalError::NonFiniteValue)
            );
        }
    }
}

#[test]
fn jet_opcodes_have_exact_source_binding_and_lossless_transport() {
    let variables = HashMap::from([(Symbol::new("x"), 0)]);
    let operations = [
        CompiledOp::Sinc,
        CompiledOp::SincDerivative,
        CompiledOp::SincSecondDerivative,
    ];
    for (index, name) in JETS.iter().enumerate() {
        let program = scalar(name);
        assert_eq!(
            program.ops(),
            &[CompiledOp::LoadVar(0), operations[index].clone()]
        );
        let bytes = serde_json::to_vec(&program).unwrap();
        let restored: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, program);
        assert!(compiled_expression_matches(&source(name), &variables, &restored).unwrap());
        for (other, operation) in operations.iter().enumerate() {
            if other == index {
                continue;
            }
            let forged =
                CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), operation.clone()])
                    .unwrap();
            assert!(!compiled_expression_matches(&source(name), &variables, &forged).unwrap());
        }
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operations[index].clone()]),
            Err(EvalError::StackUnderflow)
        );
        for arguments in [vec![], vec![Expr::symbol("x"), Expr::from_i64(1)]] {
            assert!(
                CompiledExpr::try_compile(&Expr::Function((*name).into(), arguments), &variables)
                    .is_err()
            );
        }
    }
}

#[test]
fn checked_nonlinear_origin_has_the_exact_gradient_hessian_and_products() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let expression = parse("sinc(x^2+y)").unwrap();
    let point = [0.5, -0.25]; // Inner value zero, gradient (1,1), Hessian diag(2,0).
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables).unwrap();
    let first = residual.evaluate_sparse(&point).unwrap();
    assert_eq!(first.residuals, [1.0]);
    assert_eq!(first.jacobian, [0.0, 0.0]);
    assert_eq!(
        residual
            .jacobian_vector_product(&point, &[2.0, 3.0])
            .unwrap(),
        [0.0]
    );
    assert_eq!(
        residual
            .jacobian_transpose_vector_product(&point, &[5.0])
            .unwrap(),
        [0.0, 0.0]
    );
    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let second = objective.evaluate(&point).unwrap();
    assert_eq!(second.value, 1.0);
    assert_eq!(second.gradient, [0.0, 0.0]);
    assert_eq!(
        objective.hessian_pattern(),
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    for value in second.hessian {
        close(value, -1.0 / 3.0);
    }
    for value in objective
        .hessian_vector_product(&point, &[2.0, 3.0])
        .unwrap()
    {
        close(value, -5.0 / 3.0);
    }
}

#[test]
fn nested_sinc_preserves_the_second_chain_rule_at_an_inner_stationary_point() {
    let expression = parse("sinc(sinc(x))").unwrap();
    let objective = VerifiedObjective::try_compile(&expression, &[Symbol::new("x")]).unwrap();
    let value = objective.evaluate(&[0.0]).unwrap();
    close(value.value, 0.841_470_984_807_896_5);
    assert_eq!(value.gradient, [0.0]);
    close(value.hessian[0], 0.100_389_559_646_585_6);
}

#[test]
fn typed_coordinates_with_equal_names_remain_independent() {
    let x = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([117; 32]));
    let expression = Expr::Add(vec![
        function("sinc", Expr::Sym(x.clone())),
        function("sinc", Expr::Sym(other.clone())),
    ]);
    let objective = VerifiedObjective::try_compile(&expression, &[x, other]).unwrap();
    assert_eq!(objective.hessian_pattern(), [(0, 0), (1, 1)]);
    let result = objective.evaluate(&[0.0, 1.0]).unwrap();
    close(result.value, 1.841_470_984_807_896_5);
    assert_eq!(result.gradient[0], 0.0);
    close(result.gradient[1], -0.301_168_678_939_756_8);
    close(result.hessian[0], -1.0 / 3.0);
    close(result.hessian[1], -0.239_133_626_928_382_93);
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
fn portable_programs_replay_jets_and_reject_forged_first_and_second_derivatives() {
    let expression = source("sinc");
    let variables = [Symbol::new("x")];
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables).unwrap();
    let bytes = encode_residual_candidate(&residual, &mut Unbounded).unwrap();
    let candidate = decode_residual_candidate(&bytes, &mut Unbounded).unwrap();
    assert_eq!(&candidate, residual.sparse_jacobian());
    let restored = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
        candidate.clone(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        encode_residual_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
    assert_eq!(restored.evaluate_sparse(&[0.0]).unwrap().residuals, [1.0]);
    for replacement in [Expr::from_i64(0), source("_fsym_sinc_d2"), source("sinc")] {
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
    close(restored.evaluate(&[0.0]).unwrap().hessian[0], -1.0 / 3.0);
    for replacement in [Expr::from_i64(0), source("_fsym_sinc_d1"), source("sinc")] {
        let mut forged = candidate.clone();
        replace_entry(&mut forged.hessian.entries[0], replacement);
        assert!(VerifiedObjective::try_from_candidate(&expression, &variables, forged).is_err());
    }
}

#[test]
fn public_symbolic_receipts_retain_the_original_quotient_form() {
    let expression = source("sinc");
    let variable = Symbol::new("x");
    let (derivative, proof) = verified_diff(&expression, &variable);
    verify_diff_derivation(&proof, &expression, &variable, &derivative).unwrap();
    let expected = Expr::Mul(vec![
        Expr::Add(vec![
            Expr::Mul(vec![
                source("cos"),
                Expr::pow(Expr::symbol("x"), Expr::from_i64(-1)),
            ]),
            Expr::Mul(vec![
                Expr::from_i64(-1),
                source("sin"),
                Expr::pow(Expr::symbol("x"), Expr::from_i64(-2)),
            ]),
        ]),
        Expr::from_i64(1),
    ]);
    assert_eq!(derivative, expected);
    let original_quotient = compile(&derivative);
    close(
        original_quotient.try_eval_finite(&[1.0]).unwrap(),
        -0.301_168_678_939_756_8,
    );
    assert!(original_quotient.try_eval_finite(&[0.0]).is_err());
}

#[test]
fn entire_sinc_cannot_erase_an_original_inner_pole() {
    let variable = Symbol::new("x");
    let expression = Expr::Add(vec![
        Expr::from_i64(7),
        Expr::Mul(vec![
            Expr::from_i64(0),
            function("sinc", Expr::pow(Expr::symbol("x"), Expr::from_i64(-1))),
        ]),
    ]);
    let residual = VerifiedResidualSystem::try_compile(
        std::slice::from_ref(&expression),
        std::slice::from_ref(&variable),
    )
    .unwrap();
    let objective = VerifiedObjective::try_compile(&expression, &[variable]).unwrap();
    assert!(residual.evaluate_sparse(&[0.0]).is_err());
    assert!(residual.jacobian_vector_product(&[0.0], &[0.0]).is_err());
    assert!(objective.evaluate(&[0.0]).is_err());
    assert!(objective.hessian_vector_product(&[0.0], &[0.0]).is_err());
    assert_eq!(residual.evaluate_sparse(&[1.0]).unwrap().residuals, [7.0]);
}

#[test]
fn held_constant_inner_sources_keep_zero_derivatives_and_original_domains() {
    let x = Expr::symbol("x");
    let variable = Symbol::new("x");
    let cases = [
        (Expr::pow(x.clone(), Expr::from_i64(0)), false),
        (
            Expr::Mul(vec![Expr::from_i64(0), Expr::pow(x, Expr::from_i64(-1))]),
            true,
        ),
    ];
    for (inner, has_pole) in cases {
        let expression = function("sinc", inner);
        let residual = VerifiedResidualSystem::try_compile(
            std::slice::from_ref(&expression),
            std::slice::from_ref(&variable),
        )
        .unwrap();
        assert!(
            residual
                .evaluate_sparse(&[1.0])
                .unwrap()
                .jacobian
                .iter()
                .all(|value| *value == 0.0)
        );
        let objective =
            VerifiedObjective::try_compile(&expression, std::slice::from_ref(&variable)).unwrap();
        assert_eq!(objective.evaluate(&[1.0]).unwrap().gradient, [0.0]);
        if has_pole {
            assert!(residual.evaluate_sparse(&[0.0]).is_err());
            assert!(objective.evaluate(&[0.0]).is_err());
        }
    }
}

#[test]
fn shared_evaluation_distinguishes_jets_and_keeps_inactive_poles_lazy() {
    let guarded = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadConst(0.0),
        CompiledOp::Equal,
        CompiledOp::JumpIfFalse(6),
        CompiledOp::LoadConst(1.0),
        CompiledOp::Jump(9),
        CompiledOp::LoadVar(0),
        CompiledOp::Pow(-1.0),
        CompiledOp::Sinc,
    ])
    .unwrap();
    let programs = vec![
        scalar("sinc"),
        scalar("_fsym_sinc_d1"),
        scalar("_fsym_sinc_d2"),
        scalar("sinc"),
        guarded.clone(),
        guarded,
    ];
    let batch = CompiledBatch::try_from_programs(programs.clone()).unwrap();
    assert!(batch.shared_subexpression_count() >= 1);
    for point in [0.0, -0.0, 0.25, 2.0, -10.0] {
        let expected = programs
            .iter()
            .map(|program| program.try_eval_finite(&[point]).unwrap().to_bits())
            .collect::<Vec<_>>();
        let actual = batch
            .try_eval_finite(&[point])
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

struct InterruptAt {
    budget: Budget,
    iterations: usize,
    stop_iteration: Option<usize>,
    stopped: bool,
}

impl InterruptAt {
    fn new(limits: BudgetLimits) -> Self {
        Self {
            budget: Budget::new(limits),
            iterations: 0,
            stop_iteration: None,
            stopped: false,
        }
    }
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "work after terminal cancellation");
        if dimension == Dimension::ComputeSteps && amount == 12 {
            self.iterations += 1;
        }
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "work after terminal cancellation");
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "work after terminal cancellation");
        if self
            .stop_iteration
            .is_some_and(|stop| self.iterations >= stop)
        {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn sinc_jets_keep_cumulative_budgets_and_cancel_inside_taylor_work() {
    let limits = BudgetLimits::uniform(1_000_000, 37);
    for name in JETS {
        let program = scalar(name);
        for point in [0.5, 2.0] {
            let mut counted = InterruptAt::new(limits);
            program
                .try_eval_finite_with_meter(&[point], &mut counted)
                .unwrap();
            assert!(counted.iterations > 0);
            for dimension in [
                Dimension::ComputeSteps,
                Dimension::MemoryBytes,
                Dimension::AllocationCount,
            ] {
                let used =
                    limits.dimensions[dimension.index()] - counted.budget.remaining(dimension);
                assert!(used > 0);
                let mut exact_limits = limits;
                exact_limits.dimensions[dimension.index()] = used;
                let mut exact = Budget::new(exact_limits);
                program
                    .try_eval_finite_with_meter(&[point], &mut exact)
                    .unwrap();
                assert_eq!(exact.remaining(dimension), 0);
                assert_eq!(exact.verifier_remaining(), 37);
                assert!(matches!(
                    program.try_eval_finite_with_meter(&[point], &mut exact),
                    Err(EvalError::Meter(MeterError::Budget(_)))
                ));
                exact_limits.dimensions[dimension.index()] -= 1;
                assert!(matches!(
                    program.try_eval_finite_with_meter(&[point], &mut Budget::new(exact_limits)),
                    Err(EvalError::Meter(MeterError::Budget(_)))
                ));
            }
        }
        let mut cancelled = InterruptAt::new(limits);
        cancelled.stop_iteration = Some(3);
        assert_eq!(
            program.try_eval_finite_with_meter(&[0.5], &mut cancelled),
            Err(EvalError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.iterations, 3);
    }
}

#[test]
fn cancellation_in_a_later_sinc_residual_leaves_both_output_buffers_unchanged() {
    let residual = VerifiedResidualSystem::try_compile(
        &[parse("x+1").unwrap(), source("sinc")],
        &[Symbol::new("x")],
    )
    .unwrap();
    let mut values = [71.0, 73.0];
    let mut derivatives = [83.0, 89.0];
    let mut meter = InterruptAt::new(BudgetLimits::uniform(1_000_000, 37));
    meter.stop_iteration = Some(3);
    assert!(matches!(
        residual.try_eval_sparse_into_with_meter(&[0.5], &mut values, &mut derivatives, &mut meter),
        Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Cancelled
        )))
    ));
    assert_eq!(meter.iterations, 3);
    assert_eq!(values, [71.0, 73.0]);
    assert_eq!(derivatives, [83.0, 89.0]);
    residual
        .try_eval_sparse_into(&[0.0], &mut values, &mut derivatives)
        .unwrap();
    assert_eq!(values, [1.0, 1.0]);
    assert_eq!(derivatives, [1.0, 0.0]);
}
