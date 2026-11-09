//! Inverse circular functions in the original-source numeric pipeline.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_calculus::verified::VerifiedResidualSystem;
use fsym_core::{Expr, Symbol, parse};
use std::collections::HashMap;

fn scalar(name: &str) -> CompiledExpr {
    CompiledExpr::try_compile(
        &Expr::Function(name.into(), vec![Expr::symbol("x")]),
        &HashMap::from([(Symbol::new("x"), 0)]),
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-13 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn principal_values_and_closed_value_endpoints_compile() {
    let functions: [(&str, fn(f64) -> f64); 3] =
        [("asin", f64::asin), ("acos", f64::acos), ("atan", f64::atan)];
    for (name, reference) in functions {
        let program = scalar(name);
        for point in [-1.0, -0.6, 0.0, 0.6, 1.0] {
            close(program.try_eval_finite(&[point]).unwrap(), reference(point));
        }
    }
    for point in [-1e100, -2.0, 2.0, 1e100] {
        close(scalar("atan").try_eval_finite(&[point]).unwrap(), point.atan());
    }
}

#[test]
fn finite_domain_errors_cannot_be_hidden_by_zero_or_later_arithmetic() {
    for name in ["asin", "acos"] {
        let program = scalar(name);
        for point in [-2.0, 2.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                program.try_eval_finite(&[point]),
                Err(EvalError::NonFiniteValue)
            );
        }
        let source = Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function(name.into(), vec![Expr::symbol("x")]),
        ]);
        let system = VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("x")])
            .unwrap();
        assert!(system.evaluate_sparse(&[2.0]).is_err());
        assert!(system.jacobian_vector_product(&[2.0], &[0.0]).is_err());
    }
}

#[test]
fn inverse_opcodes_roundtrip_and_require_a_real_operand_stack() {
    for name in ["asin", "acos", "atan"] {
        let program = scalar(name);
        let bytes = serde_json::to_vec(&program).unwrap();
        let restored: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored.ops(), program.ops());
        assert_eq!(
            restored.try_eval_finite(&[0.5]),
            program.try_eval_finite(&[0.5])
        );
    }
    for operation in [CompiledOp::Asin, CompiledOp::Acos, CompiledOp::Atan] {
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation]),
            Err(EvalError::StackUnderflow)
        );
    }
}

#[test]
fn source_checker_rejects_another_well_formed_inverse_function() {
    let mapping = HashMap::from([(Symbol::new("x"), 0)]);
    for (name, other) in [
        ("asin", CompiledOp::Acos),
        ("acos", CompiledOp::Atan),
        ("atan", CompiledOp::Asin),
    ] {
        let source = Expr::Function(name.into(), vec![Expr::symbol("x")]);
        assert!(compiled_expression_matches(&source, &mapping, &scalar(name)).unwrap());
        let forged = CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), other])
            .unwrap();
        assert!(!compiled_expression_matches(&source, &mapping, &forged).unwrap());
    }
}

#[test]
fn verified_jacobians_hessians_and_products_keep_all_chain_factors() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let u: f64 = 0.6;
    for (name, value, slope, curvature) in [
        ("asin", u.asin(), 5.0 / 4.0, 75.0 / 64.0),
        ("acos", u.acos(), -5.0 / 4.0, -75.0 / 64.0),
        ("atan", u.atan(), 25.0 / 34.0, -375.0 / 578.0),
    ] {
        let source = parse(&format!("{name}(2*x+y)")).unwrap();
        let system = VerifiedResidualSystem::try_compile(
            std::slice::from_ref(&source),
            &variables,
        )
        .unwrap();
        let result = system.evaluate_sparse(&[0.2, 0.2]).unwrap();
        close(result.residuals[0], value);
        close(result.jacobian[0], 2.0 * slope);
        close(result.jacobian[1], slope);
        close(
            system
                .jacobian_vector_product(&[0.2, 0.2], &[2.0, 3.0])
                .unwrap()[0],
            7.0 * slope,
        );
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        assert_eq!(objective.hessian_pattern(), &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        let result = objective.evaluate(&[0.2, 0.2]).unwrap();
        close(result.value, value);
        close(result.gradient[0], 2.0 * slope);
        close(result.gradient[1], slope);
        for (actual, factor) in result.hessian.iter().zip([4.0, 2.0, 2.0, 1.0]) {
            close(*actual, factor * curvature);
        }
        let product = objective
            .hessian_vector_product(&[0.2, 0.2], &[2.0, 3.0])
            .unwrap();
        close(product[0], 14.0 * curvature);
        close(product[1], 7.0 * curvature);
        let restored = VerifiedResidualSystem::from_json_bytes(
            &system.to_json_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(
            restored.evaluate_sparse(&[0.2, 0.2]).unwrap(),
            system.evaluate_sparse(&[0.2, 0.2]).unwrap()
        );
    }
}

#[test]
fn singular_endpoint_derivatives_do_not_publish_partial_outputs() {
    for name in ["asin", "acos"] {
        let source = Expr::Function(name.into(), vec![Expr::symbol("x")]);
        let system = VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("x")])
            .unwrap();
        for point in [-1.0, 1.0] {
            assert!(scalar(name).try_eval_finite(&[point]).is_ok());
            let mut values = [71.0];
            let mut derivatives = [83.0];
            assert!(
                system
                    .try_eval_sparse_into(&[point], &mut values, &mut derivatives)
                    .is_err()
            );
            assert_eq!(values, [71.0]);
            assert_eq!(derivatives, [83.0]);
        }
    }
}

#[test]
fn inverse_opcodes_obey_lazy_forward_branch_execution() {
    let program = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadConst(0.0),
        CompiledOp::Greater,
        CompiledOp::JumpIfFalse(7),
        CompiledOp::LoadVar(0),
        CompiledOp::Asin,
        CompiledOp::Jump(9),
        CompiledOp::LoadConst(2.0),
        CompiledOp::Acos,
    ])
    .unwrap();
    close(program.try_eval_finite(&[0.5]).unwrap(), 0.5_f64.asin());
    assert_eq!(
        program.try_eval_finite(&[-0.5]),
        Err(EvalError::NonFiniteValue)
    );
}

struct InterruptAt {
    budget: Budget,
    checkpoints: usize,
    interrupt: Option<usize>,
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
fn inverse_functions_keep_cancellation_and_cumulative_resource_accounting() {
    for name in ["asin", "acos", "atan"] {
        let program = scalar(name);
        let limits = BudgetLimits::uniform(1_000_000, 37);
        let mut meter = InterruptAt {
            budget: Budget::new(limits),
            checkpoints: 0,
            interrupt: None,
        };
        program.try_eval_finite_with_meter(&[0.5], &mut meter).unwrap();
        for interrupt in 1..=meter.checkpoints {
            let mut cancelled = InterruptAt {
                budget: Budget::new(limits),
                checkpoints: 0,
                interrupt: Some(interrupt),
            };
            assert_eq!(
                program.try_eval_finite_with_meter(&[0.5], &mut cancelled),
                Err(EvalError::Meter(MeterError::Cancelled))
            );
        }
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let consumed = limits.dimensions[dimension.index()]
                - meter.budget.remaining(dimension);
            assert!(consumed > 0);
            let mut exact_limits = limits;
            exact_limits.dimensions[dimension.index()] = consumed;
            let mut exact = Budget::new(exact_limits);
            program.try_eval_finite_with_meter(&[0.5], &mut exact).unwrap();
            assert_eq!(exact.remaining(dimension), 0);
            assert!(matches!(
                program.try_eval_finite_with_meter(&[0.5], &mut exact),
                Err(EvalError::Meter(MeterError::Budget(_)))
            ));
            assert_eq!(exact.verifier_remaining(), 37);
        }
    }
}
