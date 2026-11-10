//! Independent binary64 anchors, analytic derivatives and publication checks.
//! These observations test approximate bytecode, not certified real enclosures.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use std::collections::HashMap;
use std::sync::Arc;

fn source(name: &str) -> Expr {
    Expr::Function(name.into(), vec![Expr::symbol("x")])
}

fn scalar(name: &str) -> CompiledExpr {
    CompiledExpr::try_compile(&source(name), &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-13 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn relative_close(actual: f64, expected: f64) {
    let tolerance = (2e-13 * expected.abs()).max(4.0 * f64::from_bits(1));
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}

fn gaussian_slope(point: f64) -> f64 {
    2.0 * (-(point * point)).exp() / std::f64::consts::PI.sqrt()
}

#[test]
fn independent_value_anchors_keep_positive_tails_tiny_arguments_and_signed_zero() {
    // Conventional libm values, stored independently of the implementation.
    // The final subnormal anchor allows four representable binary64 steps.
    let anchors = [
        (0.5, 0.479_500_122_186_953_5),
        (1.0, 0.157_299_207_050_285_13),
        (2.0, 0.004_677_734_981_047_266),
        (4.0, 1.541_725_790_028_002e-8),
        (8.0, 1.122_429_717_298_292_8e-29),
        (20.0, 5.395_865_611_607_901e-176),
        (26.0, 5.663_192_408_856_143e-296),
        (27.0, 5.237_05e-319),
    ];
    let erf = scalar("erf");
    let erfc = scalar("erfc");
    for (point, expected) in anchors {
        let tail = erfc.try_eval_finite(&[point]).unwrap();
        assert!(tail > 0.0, "positive tail at {point}");
        relative_close(tail, expected);
        close(erf.try_eval_finite(&[point]).unwrap(), 1.0 - expected);
        close(erfc.try_eval_finite(&[-point]).unwrap(), 2.0 - expected);
        close(erf.try_eval_finite(&[-point]).unwrap(), expected - 1.0);
    }
    for point in [1e-300_f64, -1e-300] {
        let actual = erf.try_eval_finite(&[point]).unwrap();
        assert_ne!(actual, 0.0);
        relative_close(actual, 2.0 * point / std::f64::consts::PI.sqrt());
    }
    for zero in [0.0_f64, -0.0] {
        assert_eq!(
            erf.try_eval_finite(&[zero]).unwrap().to_bits(),
            zero.to_bits()
        );
        assert_eq!(erfc.try_eval_finite(&[zero]).unwrap(), 1.0);
    }
    for point in [28.0_f64, f64::MAX] {
        assert_eq!(erf.try_eval_finite(&[point]).unwrap(), 1.0);
        assert_eq!(erfc.try_eval_finite(&[point]).unwrap(), 0.0);
        assert_eq!(erf.try_eval_finite(&[-point]).unwrap(), -1.0);
        assert_eq!(erfc.try_eval_finite(&[-point]).unwrap(), 2.0);
    }
}

#[test]
fn serialized_opcodes_keep_source_binding_and_reject_erf_erfc_substitution() {
    let mapping = HashMap::from([(Symbol::new("x"), 0)]);
    for (name, operation, other) in [
        ("erf", CompiledOp::Erf, CompiledOp::Erfc),
        ("erfc", CompiledOp::Erfc, CompiledOp::Erf),
    ] {
        let program = scalar(name);
        let bytes = serde_json::to_vec(&program).unwrap();
        let restored: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored.ops(), program.ops());
        assert_eq!(
            restored.try_eval_finite(&[0.5]),
            program.try_eval_finite(&[0.5])
        );
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation]),
            Err(EvalError::StackUnderflow)
        );
        assert!(compiled_expression_matches(&source(name), &mapping, &restored).unwrap());
        let forged = CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), other]).unwrap();
        assert!(!compiled_expression_matches(&source(name), &mapping, &forged).unwrap());
    }
}

#[test]
fn nonlinear_checked_gradients_hessians_and_products_match_the_gaussian_chain_rule() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let point = [0.5, 0.25]; // x^2+y = 1/2; inner gradient = (1,1).
    for (name, expected_value, sign) in [
        ("erf", 0.520_499_877_813_046_5, 1.0),
        ("erfc", 0.479_500_122_186_953_5, -1.0),
    ] {
        let expression = parse(&format!("{name}(x^2+y)")).unwrap();
        let slope = sign * gaussian_slope(0.5);
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables)
                .unwrap();
        let first = residual.evaluate_sparse(&point).unwrap();
        close(first.residuals[0], expected_value);
        assert_eq!(first.jacobian.len(), 2);
        for derivative in first.jacobian {
            close(derivative, slope);
        }
        close(
            residual
                .jacobian_vector_product(&point, &[2.0, 3.0])
                .unwrap()[0],
            5.0 * slope,
        );
        let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
        assert_eq!(
            objective.hessian_pattern(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        let second = objective.evaluate(&point).unwrap();
        close(second.value, expected_value);
        for derivative in second.gradient {
            close(derivative, slope);
        }
        // f''(1/2)=-f'(1/2), and the inner xx derivative contributes 2*f'.
        for (actual, factor) in second.hessian.iter().zip([1.0, -1.0, -1.0, -1.0]) {
            close(*actual, factor * slope);
        }
        let product = objective
            .hessian_vector_product(&point, &[2.0, 3.0])
            .unwrap();
        close(product[0], -slope);
        close(product[1], -5.0 * slope);
        let restored =
            VerifiedResidualSystem::from_json_bytes(&residual.to_json_bytes().unwrap()).unwrap();
        assert_eq!(
            restored.evaluate_sparse(&point).unwrap(),
            residual.evaluate_sparse(&point).unwrap()
        );
    }
}

#[test]
fn held_zero_products_and_zero_directions_do_not_erase_original_domain_failures() {
    let variables = [Symbol::new("x")];
    for name in ["erf", "erfc"] {
        let inverse = Expr::Pow(
            Arc::new(Expr::Sym(variables[0].clone())),
            Arc::new(Expr::from_i64(-1)),
        );
        let held = Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function(name.into(), vec![inverse]),
        ]);
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&held), &variables).unwrap();
        let objective = VerifiedObjective::try_compile(&held, &variables).unwrap();
        assert!(residual.evaluate_sparse(&[0.0]).is_err());
        assert!(residual.jacobian_vector_product(&[0.0], &[0.0]).is_err());
        assert!(objective.evaluate(&[0.0]).is_err());
        assert!(objective.hessian_vector_product(&[0.0], &[0.0]).is_err());
        assert_eq!(residual.evaluate_sparse(&[1.0]).unwrap().residuals, [0.0]);
        for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                scalar(name).try_eval_finite(&[point]),
                Err(EvalError::NonFiniteValue)
            );
        }
    }
}

#[test]
fn equal_printed_names_remain_distinct_gaussian_coordinates() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([173; 32]));
    let expression = Expr::Add(vec![
        Expr::Function("erf".into(), vec![Expr::Sym(first.clone())]),
        Expr::Function("erfc".into(), vec![Expr::Sym(second.clone())]),
    ]);
    let objective = VerifiedObjective::try_compile(&expression, &[first, second]).unwrap();
    assert_eq!(objective.hessian_pattern(), [(0, 0), (1, 1)]);
    let result = objective.evaluate(&[0.5, 1.0]).unwrap();
    close(
        result.value,
        0.520_499_877_813_046_5 + 0.157_299_207_050_285_13,
    );
    close(result.gradient[0], gaussian_slope(0.5));
    close(result.gradient[1], -gaussian_slope(1.0));
    close(result.hessian[0], -gaussian_slope(0.5));
    close(result.hessian[1], 2.0 * gaussian_slope(1.0));
}

struct InterruptAt {
    budget: Budget,
    checkpoints: usize,
    iterations: usize,
    stop_checkpoint: Option<usize>,
    stop_iteration: Option<usize>,
    stopped: bool,
}

impl InterruptAt {
    fn new(limits: BudgetLimits) -> Self {
        Self {
            budget: Budget::new(limits),
            checkpoints: 0,
            iterations: 0,
            stop_checkpoint: None,
            stop_iteration: None,
            stopped: false,
        }
    }
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "charge after terminal cancellation");
        if matches!(dimension, Dimension::ComputeSteps) && matches!(amount, 12 | 20) {
            self.iterations += 1;
        }
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "charge after terminal cancellation");
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "checkpoint after terminal cancellation");
        self.checkpoints += 1;
        if self.stop_checkpoint == Some(self.checkpoints)
            || self
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
fn both_algorithms_keep_the_original_cumulative_ledger_and_iteration_cancellation() {
    let limits = BudgetLimits::uniform(1_000_000, 37);
    for name in ["erf", "erfc"] {
        for point in [0.5, 2.0, 20.0] {
            let program = scalar(name);
            let mut counted = InterruptAt::new(limits);
            program
                .try_eval_finite_with_meter(&[point], &mut counted)
                .unwrap();
            assert!(counted.iterations >= 3);
            for stop in [1, counted.checkpoints / 2, counted.checkpoints] {
                let mut interrupted = InterruptAt::new(limits);
                interrupted.stop_checkpoint = Some(stop);
                assert_eq!(
                    program.try_eval_finite_with_meter(&[point], &mut interrupted),
                    Err(EvalError::Meter(MeterError::Cancelled))
                );
                assert_eq!(interrupted.checkpoints, stop);
                assert!(interrupted.stopped);
            }
            let mut during_iteration = InterruptAt::new(limits);
            during_iteration.stop_iteration = Some(3);
            assert_eq!(
                program.try_eval_finite_with_meter(&[point], &mut during_iteration),
                Err(EvalError::Meter(MeterError::Cancelled))
            );
            assert_eq!(during_iteration.iterations, 3);
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
                let mut short_limits = exact_limits;
                short_limits.dimensions[dimension.index()] = used - 1;
                assert!(matches!(
                    program.try_eval_finite_with_meter(&[point], &mut Budget::new(short_limits)),
                    Err(EvalError::Meter(MeterError::Budget(_)))
                ));
            }
        }
    }
}

#[test]
fn cancellation_inside_a_later_residual_iteration_keeps_both_caller_buffers_atomic() {
    let variables = [Symbol::new("x")];
    let residual =
        VerifiedResidualSystem::try_compile(&[parse("x+1").unwrap(), source("erfc")], &variables)
            .unwrap();
    let mut values = [71.0, 73.0];
    let mut derivatives = [83.0, 89.0];
    let mut meter = InterruptAt::new(BudgetLimits::uniform(1_000_000, 37));
    meter.stop_iteration = Some(3);
    assert!(matches!(
        residual
            .try_eval_sparse_into_with_meter(&[2.0], &mut values, &mut derivatives, &mut meter,),
        Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
            MeterError::Cancelled
        )))
    ));
    assert_eq!(meter.iterations, 3);
    assert!(meter.stopped);
    assert_eq!(values, [71.0, 73.0]);
    assert_eq!(derivatives, [83.0, 89.0]);
    residual
        .try_eval_sparse_into(&[2.0], &mut values, &mut derivatives)
        .unwrap();
    close(values[0], 3.0);
    relative_close(values[1], 0.004_677_734_981_047_266);
    close(derivatives[0], 1.0);
    close(derivatives[1], -gaussian_slope(2.0));
}
