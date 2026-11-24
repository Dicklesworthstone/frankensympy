//! Independently derived angle derivatives and original-domain admission.
//!
//! Numeric bytecode is approximate. These tests do not turn binary64 values
//! into certified enclosures or a branch-independent differentiability claim.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, CompiledResidualSystem, EvalError};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use std::collections::HashMap;

#[path = "atan2_compilation/numeric_jets.rs"]
mod numeric_jets;

fn angle(y: Expr, x: Expr) -> Expr {
    Expr::Function("atan2".into(), vec![y, x])
}

fn variables() -> [Symbol; 2] {
    [Symbol::new("y"), Symbol::new("x")]
}

fn source() -> Expr {
    angle(Expr::symbol("y"), Expr::symbol("x"))
}

fn scalar() -> CompiledExpr {
    CompiledExpr::try_compile(
        &source(),
        &HashMap::from([(Symbol::new("y"), 0), (Symbol::new("x"), 1)]),
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
fn scalar_argument_order_quadrants_and_axis_conventions_are_explicit() {
    let program = scalar();
    assert_eq!(
        program.ops(),
        [
            CompiledOp::LoadVar(0),
            CompiledOp::LoadVar(1),
            CompiledOp::Atan2,
        ]
    );
    let pi = std::f64::consts::PI;
    for (point, expected) in [
        ([1.0, 1.0], pi / 4.0),
        ([1.0, -1.0], 3.0 * pi / 4.0),
        ([-1.0, -1.0], -3.0 * pi / 4.0),
        ([-1.0, 1.0], -pi / 4.0),
        ([1.0, 0.0], pi / 2.0),
        ([-1.0, 0.0], -pi / 2.0),
        ([0.0, 1.0], 0.0),
        ([4.0, 3.0], 0.927_295_218_001_612_2),
    ] {
        let value = program.try_eval_finite(&point).unwrap();
        close(value, expected);
        assert!(value > -pi && value <= pi);
    }
    // Exact symbolic zero has no lower-side branch designation. Both f64
    // spellings on the negative axis use the principal +pi value.
    for zero in [0.0_f64, -0.0] {
        assert_eq!(program.try_eval_finite(&[zero, -1.0]).unwrap(), pi);
        assert!(program.try_eval_finite(&[zero, 0.0]).is_err());
        assert!(program.try_eval_finite(&[zero, -0.0]).is_err());
    }
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            program.try_eval_finite(&[nonfinite, 1.0]),
            Err(EvalError::NonFiniteValue)
        );
        assert_eq!(
            program.try_eval_finite(&[1.0, nonfinite]),
            Err(EvalError::NonFiniteValue)
        );
    }
}

#[test]
fn binary_opcode_transport_and_provenance_reject_reversed_arguments() {
    let program = scalar();
    let mapping = HashMap::from([(Symbol::new("y"), 0), (Symbol::new("x"), 1)]);
    let restored: CompiledExpr =
        serde_json::from_slice(&serde_json::to_vec(&program).unwrap()).unwrap();
    assert_eq!(restored.ops(), program.ops());
    close(
        restored.try_eval_finite(&[4.0, 3.0]).unwrap(),
        0.927_295_218_001_612_2,
    );
    assert!(compiled_expression_matches(&source(), &mapping, &restored).unwrap());
    let reversed = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(1),
        CompiledOp::LoadVar(0),
        CompiledOp::Atan2,
    ])
    .unwrap();
    assert!(!compiled_expression_matches(&source(), &mapping, &reversed).unwrap());
    for ops in [
        vec![CompiledOp::Atan2],
        vec![CompiledOp::LoadVar(0), CompiledOp::Atan2],
    ] {
        assert_eq!(
            CompiledExpr::try_from_ops(ops),
            Err(EvalError::StackUnderflow)
        );
    }
    for arguments in [vec![], vec![Expr::symbol("y")], vec![Expr::from_i64(1); 3]] {
        assert!(
            CompiledExpr::try_compile(&Expr::Function("atan2".into(), arguments), &mapping)
                .is_err()
        );
    }
}

#[test]
fn independently_derived_jacobian_and_hessian_keep_y_then_x_order() {
    let residual = VerifiedResidualSystem::try_compile(&[source()], &variables()).unwrap();
    assert_eq!(residual.jacobian_pattern(), [(0, 0), (0, 1)]);
    let first = residual.evaluate_sparse(&[4.0, 3.0]).unwrap();
    close(first.residuals[0], 0.927_295_218_001_612_2);
    close(first.jacobian[0], 3.0 / 25.0);
    close(first.jacobian[1], -4.0 / 25.0);
    close(
        residual
            .jacobian_vector_product(&[4.0, 3.0], &[2.0, 3.0])
            .unwrap()[0],
        -6.0 / 25.0,
    );
    let objective = VerifiedObjective::try_compile(&source(), &variables()).unwrap();
    assert_eq!(
        objective.hessian_pattern(),
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    let second = objective.evaluate(&[4.0, 3.0]).unwrap();
    for (actual, expected) in second.gradient.iter().zip([3.0 / 25.0, -4.0 / 25.0]) {
        close(*actual, expected);
    }
    for (actual, numerator) in second.hessian.iter().zip([-24.0, 7.0, 7.0, 24.0]) {
        close(*actual, numerator / 625.0);
    }
}

#[test]
fn both_nonlinear_arguments_contribute_to_every_chain_rule_stage() {
    let vars = [Symbol::new("u"), Symbol::new("v")];
    let expression = angle(parse("u^2+v").unwrap(), parse("u-v^2").unwrap());
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &vars).unwrap();
    let first = residual.evaluate_sparse(&[2.0, 1.0]).unwrap();
    // At this point (Y,X)=(5,1), dY=(4,1), dX=(1,-2).
    close(first.residuals[0], 1.373_400_766_945_016);
    for (actual, expected) in first.jacobian.iter().zip([-1.0 / 26.0, 11.0 / 26.0]) {
        close(*actual, expected);
    }
    close(
        residual
            .jacobian_vector_product(&[2.0, 1.0], &[2.0, -1.0])
            .unwrap()[0],
        -0.5,
    );
    let objective = VerifiedObjective::try_compile(&expression, &vars).unwrap();
    let second = objective.evaluate(&[2.0, 1.0]).unwrap();
    // Differentiating N_i/(Y^2+X^2) directly gives these rational values;
    // no symbolic differentiator or finite differences supply the oracle.
    let expected = [47.0 / 338.0, -57.0 / 169.0, -57.0 / 169.0, 97.0 / 338.0];
    assert_eq!(second.hessian.len(), expected.len());
    for (actual, expected) in second.hessian.iter().zip(expected) {
        close(*actual, expected);
    }
    let product = objective
        .hessian_vector_product(&[2.0, 1.0], &[2.0, -1.0])
        .unwrap();
    close(product[0], 104.0 / 169.0);
    close(product[1], -325.0 / 338.0);
}

#[test]
fn value_domain_does_not_imply_smoothness_at_the_principal_cut() {
    let residual = VerifiedResidualSystem::try_compile(&[source()], &variables()).unwrap();
    let objective = VerifiedObjective::try_compile(&source(), &variables()).unwrap();
    for zero in [0.0_f64, -0.0] {
        let cut = [zero, -1.0];
        assert_eq!(
            scalar().try_eval_finite(&cut).unwrap(),
            std::f64::consts::PI
        );
        assert!(residual.evaluate_sparse(&cut).is_err());
        assert!(residual.try_eval_dense(&cut).is_err());
        assert!(residual.jacobian_vector_product(&cut, &[0.0, 0.0]).is_err());
        assert!(objective.evaluate(&cut).is_err());
        assert!(objective.hessian_vector_product(&cut, &[0.0, 0.0]).is_err());
    }
    // Both open sides of the cut remain valid. Positive x is smooth even
    // when y is zero, as are the two nonzero points on the vertical axis.
    for point in [
        [1e-8, -1.0],
        [-1e-8, -1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
    ] {
        assert!(residual.evaluate_sparse(&point).is_ok());
        assert!(objective.evaluate(&point).is_ok());
    }
}

#[test]
fn held_zero_and_zero_products_keep_original_undefined_angles_and_cuts() {
    let held = Expr::Mul(vec![Expr::from_i64(0), source()]);
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&held), &variables()).unwrap();
    let objective = VerifiedObjective::try_compile(&held, &variables()).unwrap();
    for point in [[0.0, 0.0], [0.0, -1.0]] {
        assert!(residual.evaluate_sparse(&point).is_err());
        assert!(
            residual
                .jacobian_vector_product(&point, &[0.0, 0.0])
                .is_err()
        );
        assert!(objective.evaluate(&point).is_err());
        assert!(
            objective
                .hessian_vector_product(&point, &[0.0, 0.0])
                .is_err()
        );
    }
    assert_eq!(
        residual.evaluate_sparse(&[1.0, 1.0]).unwrap().residuals,
        [0.0]
    );
    assert_eq!(objective.evaluate(&[1.0, 1.0]).unwrap().value, 0.0);
}

#[test]
fn admitted_transport_rebuilds_smooth_domain_checks_and_rejects_changed_source_order() {
    let original = VerifiedResidualSystem::try_compile(&[source()], &variables()).unwrap();
    let bytes = original.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
    assert_eq!(restored.receipt().test_points_evaluated, 0);
    assert_eq!(
        restored.evaluate_sparse(&[4.0, 3.0]).unwrap(),
        original.evaluate_sparse(&[4.0, 3.0]).unwrap()
    );
    assert!(restored.evaluate_sparse(&[0.0, -1.0]).is_err());
    assert!(restored.evaluate_sparse(&[0.0, 0.0]).is_err());
    let mut changed = original.sparse_jacobian().clone();
    changed.residuals[0] = angle(Expr::symbol("x"), Expr::symbol("y"));
    assert!(VerifiedResidualSystem::try_from_sparse_jacobian(&changed).is_err());
    let forged = serde_json::json!({"schema_version": 1, "jacobian": changed});
    assert!(
        VerifiedResidualSystem::from_json_bytes(&serde_json::to_vec(&forged).unwrap()).is_err()
    );
}

#[test]
fn identical_printed_names_remain_distinct_ordered_angle_arguments() {
    let y = Symbol::new("coordinate");
    let x = Symbol::with_identity(
        "coordinate",
        SymbolIdentity::from_assumptions_digest([43; 32]),
    );
    let expression = angle(Expr::Sym(y.clone()), Expr::Sym(x.clone()));
    let residual = VerifiedResidualSystem::try_compile(&[expression], &[y, x]).unwrap();
    let evaluated = residual.evaluate_sparse(&[4.0, 3.0]).unwrap();
    close(evaluated.residuals[0], 0.927_295_218_001_612_2);
    close(evaluated.jacobian[0], 3.0 / 25.0);
    close(evaluated.jacobian[1], -4.0 / 25.0);
}

struct InterruptAt {
    budget: Budget,
    checkpoints: usize,
    stop: Option<usize>,
    stopped: bool,
}

impl InterruptAt {
    fn new(limits: BudgetLimits) -> Self {
        Self {
            budget: Budget::new(limits),
            checkpoints: 0,
            stop: None,
            stopped: false,
        }
    }
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "charge after terminal cancellation");
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "charge after terminal cancellation");
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "checkpoint after terminal cancellation");
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn scalar_cancellation_and_resource_costs_stay_on_one_cumulative_ledger() {
    let program = scalar();
    let limits = BudgetLimits::uniform(1_000_000, 37);
    let mut counted = InterruptAt::new(limits);
    program
        .try_eval_finite_with_meter(&[4.0, 3.0], &mut counted)
        .unwrap();
    for stop in [1, counted.checkpoints / 2, counted.checkpoints] {
        let mut interrupted = InterruptAt::new(limits);
        interrupted.stop = Some(stop);
        assert_eq!(
            program.try_eval_finite_with_meter(&[4.0, 3.0], &mut interrupted),
            Err(EvalError::Meter(MeterError::Cancelled))
        );
        assert_eq!(interrupted.checkpoints, stop);
        assert!(interrupted.stopped);
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let used = limits.dimensions[dimension.index()] - counted.budget.remaining(dimension);
        assert!(used > 0);
        let mut exact_limits = limits;
        exact_limits.dimensions[dimension.index()] = used;
        let mut exact = Budget::new(exact_limits);
        program
            .try_eval_finite_with_meter(&[4.0, 3.0], &mut exact)
            .unwrap();
        assert_eq!(exact.remaining(dimension), 0);
        assert_eq!(exact.verifier_remaining(), 37);
        assert!(matches!(
            program.try_eval_finite_with_meter(&[4.0, 3.0], &mut exact),
            Err(EvalError::Meter(MeterError::Budget(_)))
        ));
        exact_limits.dimensions[dimension.index()] = used - 1;
        assert!(matches!(
            program.try_eval_finite_with_meter(&[4.0, 3.0], &mut Budget::new(exact_limits)),
            Err(EvalError::Meter(MeterError::Budget(_)))
        ));
    }
}

#[test]
fn cancelled_shared_evaluation_never_partially_publishes_caller_buffers() {
    let residual =
        VerifiedResidualSystem::try_compile(&[Expr::symbol("y"), source()], &variables()).unwrap();
    let mut counted = InterruptAt::new(BudgetLimits::uniform(1_000_000, 37));
    residual
        .try_eval_sparse_into_with_meter(&[4.0, 3.0], &mut [0.0; 2], &mut [0.0; 3], &mut counted)
        .unwrap();
    for stop in [1, counted.checkpoints / 2, counted.checkpoints] {
        let mut values = [71.0, 73.0];
        let mut derivatives = [83.0, 89.0, 97.0];
        let mut meter = InterruptAt::new(BudgetLimits::uniform(1_000_000, 37));
        meter.stop = Some(stop);
        assert!(matches!(
            residual.try_eval_sparse_into_with_meter(
                &[4.0, 3.0],
                &mut values,
                &mut derivatives,
                &mut meter,
            ),
            Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
                MeterError::Cancelled
            )))
        ));
        assert_eq!(values, [71.0, 73.0]);
        assert_eq!(derivatives, [83.0, 89.0, 97.0]);
        assert!(meter.stopped);
    }
}

#[test]
fn legacy_and_exported_dense_systems_keep_angle_domains_before_publication() {
    let expressions = [source()];
    let vars = variables();
    let verified = VerifiedResidualSystem::try_compile(&expressions, &vars).unwrap();
    for program in [
        CompiledResidualSystem::try_compile(&expressions, &vars).unwrap(),
        verified.try_dense_program().unwrap(),
    ] {
        for zero in [0.0_f64, -0.0] {
            let mut values = [71.0];
            program
                .try_eval_residuals(&[zero, -1.0], &mut values)
                .unwrap();
            assert_eq!(values, [std::f64::consts::PI]);

            for (point, expected) in [
                ([zero, -1.0], EvalError::NonsmoothAngle),
                ([zero, 0.0], EvalError::UndefinedAngle),
            ] {
                let mut derivatives = [83.0, 89.0];
                assert_eq!(
                    program.try_eval_jacobian(&point, &mut derivatives),
                    Err(expected.clone())
                );
                assert_eq!(derivatives, [83.0, 89.0]);

                let mut values = [71.0];
                assert_eq!(
                    program.try_eval_system(&point, &mut values, &mut derivatives),
                    Err(expected)
                );
                assert_eq!(values, [71.0]);
                assert_eq!(derivatives, [83.0, 89.0]);
            }

            let mut values = [71.0];
            assert_eq!(
                program.try_eval_residuals(&[zero, 0.0], &mut values),
                Err(EvalError::UndefinedAngle)
            );
            assert_eq!(values, [71.0]);
            // A permissive diagnostic tolerance cannot admit a nonsmooth
            // original source before the finite-difference comparison.
            assert!(!program.check_with_finite_differences(&[zero, -1.0], 1e-6, f64::MAX));
        }

        let mut values = [71.0];
        let mut derivatives = [83.0, 89.0];
        program
            .try_eval_system(&[4.0, 3.0], &mut values, &mut derivatives)
            .unwrap();
        close(values[0], 0.927_295_218_001_612_2);
        close(derivatives[0], 3.0 / 25.0);
        close(derivatives[1], -4.0 / 25.0);
    }
}
