//! Ordered real atan2 partials must not overflow their intermediate norm.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{CompiledBatch, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_core::{Expr, Symbol};
use std::collections::HashMap;

fn jets() -> [(&'static str, CompiledOp); 5] {
    [
        ("_fsym_atan2_dy", CompiledOp::Atan2DerivativeY),
        ("_fsym_atan2_dx", CompiledOp::Atan2DerivativeX),
        ("_fsym_atan2_dyy", CompiledOp::Atan2SecondDerivativeYY),
        ("_fsym_atan2_dyx", CompiledOp::Atan2SecondDerivativeYX),
        ("_fsym_atan2_dxx", CompiledOp::Atan2SecondDerivativeXX),
    ]
}

fn program(operation: CompiledOp) -> CompiledExpr {
    CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadVar(1),
        operation,
    ])
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "nonfinite result for {expected:e}");
    if expected != 0.0 {
        assert_ne!(actual, 0.0, "erased representable partial {expected:e}");
        assert_eq!(actual.is_sign_positive(), expected.is_sign_positive());
    }
    assert!(
        (actual - expected).abs() <= 8e-14 * expected.abs() + 2.0 * f64::from_bits(1),
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn partials_match_independent_exact_scaled_three_four_triangles() {
    // Closed forms at (y,x)=(3t,4t); t is an exact power of two.
    // No reference squares t or executes the scaled implementation.
    for exponent in [-500, -400, -1, 0, 1, 400, 500] {
        let scale = 2.0_f64.powi(exponent);
        let inverse = 2.0_f64.powi(-exponent);
        let expected = [
            (4.0 / 25.0) * inverse,
            (-3.0 / 25.0) * inverse,
            ((-24.0 / 625.0) * inverse) * inverse,
            ((-7.0 / 625.0) * inverse) * inverse,
            ((24.0 / 625.0) * inverse) * inverse,
        ];
        for (index, (_, operation)) in jets().into_iter().enumerate() {
            let code = program(operation);
            close(
                code.try_eval_finite(&[3.0 * scale, 4.0 * scale]).unwrap(),
                expected[index],
            );
            // Reflection changes the sign only of the partials odd in y.
            let sign = if matches!(index, 1 | 2 | 4) {
                -1.0
            } else {
                1.0
            };
            close(
                code.try_eval_finite(&[-3.0 * scale, 4.0 * scale]).unwrap(),
                sign * expected[index],
            );
        }
    }
}

#[test]
fn slopes_reach_subnormal_and_large_finite_scales_without_a_squared_norm() {
    for exponent in [-1020, -700, 700, 1020] {
        let scale = 2.0_f64.powi(exponent);
        let inverse = 2.0_f64.powi(-exponent);
        close(
            program(CompiledOp::Atan2DerivativeY)
                .try_eval_finite(&[3.0 * scale, 4.0 * scale])
                .unwrap(),
            (4.0 / 25.0) * inverse,
        );
        close(
            program(CompiledOp::Atan2DerivativeX)
                .try_eval_finite(&[3.0 * scale, 4.0 * scale])
                .unwrap(),
            (-3.0 / 25.0) * inverse,
        );
    }
}

#[test]
fn mixed_partial_preserves_near_diagonal_curvature_and_axis_limits() {
    let mixed = program(CompiledOp::Atan2SecondDerivativeYX);
    let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
    close(
        mixed.try_eval_finite(&[1.0, adjacent]).unwrap(),
        -f64::EPSILON / 2.0,
    );
    close(
        mixed.try_eval_finite(&[adjacent, 1.0]).unwrap(),
        f64::EPSILON / 2.0,
    );
    assert_eq!(mixed.try_eval_finite(&[1e150, 1e150]).unwrap(), 0.0);
    for scale in [1e-150, 1.0, 1e150] {
        close(
            mixed.try_eval_finite(&[0.0, scale]).unwrap(),
            (-1.0 / scale) / scale,
        );
        close(
            mixed.try_eval_finite(&[scale, 0.0]).unwrap(),
            (1.0 / scale) / scale,
        );
        assert_eq!(
            program(CompiledOp::Atan2SecondDerivativeYY)
                .try_eval_finite(&[0.0, scale])
                .unwrap(),
            0.0,
        );
        assert_eq!(
            program(CompiledOp::Atan2SecondDerivativeXX)
                .try_eval_finite(&[scale, 0.0])
                .unwrap(),
            0.0,
        );
    }
}

#[test]
fn every_partial_retains_original_angle_domain_and_output_finiteness() {
    for (_, operation) in jets() {
        let code = program(operation);
        for point in [[0.0, 0.0], [-0.0, 0.0], [0.0, -0.0], [-0.0, -0.0]] {
            assert_eq!(code.try_eval_finite(&point), Err(EvalError::UndefinedAngle));
        }
        for point in [[0.0, -1.0], [-0.0, -1.0]] {
            assert_eq!(code.try_eval_finite(&point), Err(EvalError::NonsmoothAngle));
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for point in [[bad, 1.0], [1.0, bad]] {
                assert_eq!(code.try_eval_finite(&point), Err(EvalError::NonFiniteValue));
            }
        }
    }
    assert_eq!(
        program(CompiledOp::Atan2SecondDerivativeYY).try_eval_finite(&[1e-200, 1e-200]),
        Err(EvalError::NonFiniteValue),
    );
    // Value-only principal-angle behavior is unchanged.
    close(
        program(CompiledOp::Atan2)
            .try_eval_finite(&[0.0, -1.0])
            .unwrap(),
        std::f64::consts::PI,
    );
}

#[test]
fn private_binary_sources_require_exact_arity_order_and_opcode() {
    let variables = HashMap::from([(Symbol::new("y"), 0), (Symbol::new("x"), 1)]);
    for (name, operation) in jets() {
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation.clone()]),
            Err(EvalError::StackUnderflow),
        );
        assert_eq!(
            CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), operation.clone()]),
            Err(EvalError::StackUnderflow),
        );
        let source = Expr::Function(name.into(), vec![Expr::symbol("y"), Expr::symbol("x")]);
        let code = CompiledExpr::try_compile(&source, &variables).unwrap();
        assert_eq!(code, program(operation.clone()));
        assert!(compiled_expression_matches(&source, &variables, &code).unwrap());
        let wrong_order = CompiledExpr::try_from_ops(vec![
            CompiledOp::LoadVar(1),
            CompiledOp::LoadVar(0),
            operation.clone(),
        ])
        .unwrap();
        assert!(!compiled_expression_matches(&source, &variables, &wrong_order).unwrap());
        for (_, wrong_operation) in jets() {
            if wrong_operation != operation {
                assert!(
                    !compiled_expression_matches(&source, &variables, &program(wrong_operation))
                        .unwrap()
                );
            }
        }
        for arguments in [
            vec![],
            vec![Expr::symbol("y")],
            vec![Expr::symbol("y"), Expr::symbol("x"), Expr::from_i64(1)],
        ] {
            assert!(
                CompiledExpr::try_compile(&Expr::Function(name.into(), arguments), &variables)
                    .is_err()
            );
        }
        let bytes = serde_json::to_vec(&code).unwrap();
        let restored: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, code);
        assert_eq!(serde_json::to_vec(&restored).unwrap(), bytes);
    }
}

#[test]
fn partial_work_is_charged_cumulatively_and_cancelled_without_a_value() {
    const AMPLE: u64 = 1 << 30;
    let code = program(CompiledOp::Atan2SecondDerivativeYX);
    let mut measured = Budget::new(BudgetLimits::uniform(AMPLE, 0));
    code.try_eval_finite_with_meter(&[3.0, 4.0], &mut measured)
        .unwrap();
    let used = AMPLE - measured.remaining(Dimension::ComputeSteps);
    assert!(used >= 20);
    for (steps, accepted) in [(used, true), (used - 1, false)] {
        let mut limits = BudgetLimits::uniform(AMPLE, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = steps;
        let mut meter = Budget::new(limits);
        assert_eq!(
            code.try_eval_finite_with_meter(&[3.0, 4.0], &mut meter)
                .is_ok(),
            accepted,
        );
    }
    struct Cancelled;
    impl BudgetMeter for Cancelled {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Err(MeterError::Cancelled)
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Err(MeterError::Cancelled)
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            Err(MeterError::Cancelled)
        }
    }
    assert_eq!(
        code.try_eval_finite_with_meter(&[3.0, 4.0], &mut Cancelled),
        Err(EvalError::Meter(MeterError::Cancelled)),
    );
}

#[test]
fn shared_evaluation_preserves_binary_partial_operands() {
    let codes: Vec<_> = jets()
        .into_iter()
        .map(|(_, operation)| program(operation))
        .collect();
    let mut repeated = codes.clone();
    repeated.extend(codes.clone());
    let batch = CompiledBatch::try_from_programs_with_meter(repeated, &mut Unbounded).unwrap();
    let expected: Vec<_> = codes
        .iter()
        .map(|code| code.try_eval_finite(&[3e150, 4e150]).unwrap())
        .collect();
    let mut expected_repeated = expected.clone();
    expected_repeated.extend(expected);
    assert_eq!(
        batch.try_eval_finite(&[3e150, 4e150]).unwrap(),
        expected_repeated,
    );
}
