//! Saturated function values must not erase representable derivatives.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, CompiledResidualSystem};
use fsym_calculus::proof::{RULE_DIFF_TANH, make_diff_term, verified_diff, verify_diff_derivation};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{SparseJacobian, verify_sparse_jacobian_certificate};
use fsym_core::{Expr, Symbol, parse};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use std::collections::HashMap;

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

// Use the defining cosh quotient as an independent moderate/tail reference.
// Taking the reciprocal BEFORE squaring also covers subnormal derivatives
// without overflowing cosh(x)^2. An absolute tolerance near one would hide
// precisely the false-zero regression these tests are meant to detect.
fn reference(point: f64) -> (f64, f64) {
    let slope = point.cosh().recip().powi(2);
    (slope, -2.0 * point.tanh() * slope)
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() <= 3e-13 * expected.abs() + 8.0 * f64::from_bits(1),
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn saturated_tanh_keeps_nonzero_gradient_curvature_and_directional_products() {
    let variable = Symbol::new("x");
    let expression = function("tanh", Expr::Sym(variable.clone()));
    let variables = [variable];
    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables).unwrap();
    let dense = CompiledResidualSystem::try_compile(&[expression], &variables).unwrap();
    for point in [-360.0, -100.0, -20.0, -2.0, 0.0, 2.0, 20.0, 100.0, 360.0] {
        let (slope, curvature) = reference(point);
        assert!(slope > 0.0);
        let result = objective.evaluate(&[point]).unwrap();
        assert!(result.gradient[0] > 0.0, "lost gradient at {point}");
        close(result.gradient[0], slope);
        close(result.hessian[0], curvature);
        close(
            residual.evaluate_sparse(&[point]).unwrap().jacobian[0],
            slope,
        );
        close(
            residual.jacobian_vector_product(&[point], &[2.0]).unwrap()[0],
            2.0 * slope,
        );
        close(
            residual
                .jacobian_transpose_vector_product(&[point], &[3.0])
                .unwrap()[0],
            3.0 * slope,
        );
        close(
            objective.hessian_vector_product(&[point], &[2.0]).unwrap()[0],
            2.0 * curvature,
        );
        let mut jacobian = [0.0];
        dense.try_eval_jacobian(&[point], &mut jacobian).unwrap();
        close(jacobian[0], slope);
    }
}

#[test]
fn nonlinear_nested_arguments_retain_the_complete_mixed_hessian() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let expression = function("exp", function("tanh", parse("x^2+y").unwrap()));
    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables).unwrap();
    for sign in [-1.0, 1.0] {
        let point = [3.0, sign * 20.0 - 9.0];
        let inner = sign * 20.0;
        let (slope, curvature) = reference(inner);
        let value = inner.tanh().exp();
        let first = value * slope;
        let second = value * (curvature + slope * slope);
        let result = objective.evaluate(&point).unwrap();
        close(result.value, value);
        close(result.gradient[0], 6.0 * first);
        close(result.gradient[1], first);
        assert_eq!(
            objective.hessian_pattern(),
            &[(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        for (actual, expected) in result.hessian.iter().zip([
            36.0 * second + 2.0 * first,
            6.0 * second,
            6.0 * second,
            second,
        ]) {
            close(*actual, expected);
        }
        let product = objective
            .hessian_vector_product(&point, &[2.0, 3.0])
            .unwrap();
        close(product[0], 90.0 * second + 4.0 * first);
        close(product[1], 15.0 * second);
        close(
            residual
                .jacobian_vector_product(&point, &[2.0, 3.0])
                .unwrap()[0],
            15.0 * first,
        );
    }
}

#[test]
fn stable_tanh_products_preserve_the_piecewise_certification_boundary() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(variables[0].clone());
    let y = Expr::Sym(variables[1].clone());
    let body = x.clone() * function("tanh", y.clone()) + function("tanh", x.clone());
    let objective = VerifiedObjective::try_compile(&body, &variables).unwrap();
    let expression = Expr::Function(
        "Piecewise".into(),
        vec![
            Expr::Function(
                "Tuple".into(),
                vec![
                    body,
                    Expr::Function("StrictGreaterThan".into(), vec![y, Expr::from_i64(0)]),
                ],
            ),
            Expr::Function(
                "Tuple".into(),
                vec![function("log", x), Expr::symbol("True")],
            ),
        ],
    );
    // Formal branch derivatives are available to the raw dense adapter.
    // Certified derivatives across Piecewise seams remain unsupported.
    assert!(VerifiedObjective::try_compile(&expression, &variables).is_err());
    let conditional = CompiledResidualSystem::try_compile(&[expression], &variables).unwrap();
    let point = [-20.0, 20.0];
    let result = objective.evaluate(&point).unwrap();
    let (slope_x, curvature_x) = reference(point[0]);
    let (slope_y, curvature_y) = reference(point[1]);
    close(result.gradient[0], point[1].tanh() + slope_x);
    close(result.gradient[1], point[0] * slope_y);
    let mut branch_gradient = [0.0; 2];
    conditional
        .try_eval_jacobian(&point, &mut branch_gradient)
        .unwrap();
    close(branch_gradient[0], result.gradient[0]);
    close(branch_gradient[1], result.gradient[1]);
    for (actual, expected) in
        result
            .hessian
            .iter()
            .zip([curvature_x, slope_y, slope_y, point[0] * curvature_y])
    {
        close(*actual, expected);
    }
    // The raw adapter retains its existing formal/nonfinite contract. A
    // finite evaluation of its original selected source still rejects log
    // of a negative number; the stable derivative grants no certification.
    assert!(
        conditional.compiled_residuals[0]
            .try_eval_finite(&[-20.0, -20.0])
            .is_err()
    );
}

fn derivation(expression: &Expr, variable: &Symbol, rhs: &Expr) -> DerivationTree {
    let lhs = make_diff_term(expression, variable);
    DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(1),
            rule: ProofRule::DefinitionalReduction {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                rule_name: RULE_DIFF_TANH.into(),
            },
            claim: Claim::equality(lhs, rhs.clone()),
        }],
        root: StepId(1),
    }
}

#[test]
fn symbolic_receipts_keep_their_form_and_numeric_mutants_are_rejected() {
    let variables = [Symbol::new("x")];
    let expression = function("tanh", parse("3*x").unwrap());
    let (ordinary, receipt) = verified_diff(&expression, &variables[0]);
    let expected = Expr::Mul(vec![
        Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![
                Expr::from_i64(-1),
                expression.clone().pow(Expr::from_i64(2)),
            ]),
        ]),
        Expr::from_i64(3),
    ]);
    assert_eq!(ordinary, expected);
    verify_diff_derivation(&receipt, &expression, &variables[0], &ordinary).unwrap();
    let candidate =
        SparseJacobian::try_build(std::slice::from_ref(&expression), &variables).unwrap();
    verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
    let expected = candidate.entries[0].symbolic_deriv.clone();
    for wrong in [
        Expr::from_i64(0),
        -expected.clone(),
        expected.clone() * Expr::rational(1, 3).unwrap(),
        expected.subs(&HashMap::from([(
            variables[0].clone(),
            Expr::symbol("other"),
        )])),
    ] {
        let mut forged = candidate.clone();
        forged.entries[0].symbolic_deriv = wrong.clone();
        forged.entries[0].derivation = derivation(&expression, &variables[0], &wrong);
        // A forged bytecode can agree with its forged RHS and still prove no
        // derivative of the original source. The mathematical check runs first.
        let mapping = HashMap::from([(variables[0].clone(), 0), (Symbol::new("other"), 0)]);
        forged.entries[0].compiled_deriv = CompiledExpr::try_compile(&wrong, &mapping).unwrap();
        assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
    }
    let original = ObjectiveCandidate::try_build(&expression, &variables).unwrap();
    for second_order in [false, true] {
        let mut candidate = original.clone();
        let stage = if second_order {
            &mut candidate.hessian
        } else {
            &mut candidate.gradient
        };
        stage.entries.clear();
        stage.pattern.nonzeros.clear();
        stage.pattern.nnz = 0;
        stage.validate_structure().unwrap();
        assert!(VerifiedObjective::try_from_candidate(&expression, &variables, candidate).is_err());
    }
}

#[test]
fn stable_programs_roundtrip_and_original_domain_failures_remain_terminal() {
    let variables = [Symbol::new("x")];
    let expression = function("tanh", Expr::Sym(variables[0].clone()));
    let system = VerifiedResidualSystem::try_compile(&[expression], &variables).unwrap();
    let bytes = system.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
    let actual = restored.evaluate_sparse(&[20.0]).unwrap();
    assert!(actual.jacobian[0] > 0.0);
    assert_eq!(actual, system.evaluate_sparse(&[20.0]).unwrap());

    let source = Expr::Mul(vec![
        Expr::from_i64(0),
        function("tanh", function("log", Expr::symbol("x"))),
    ]);
    let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
    for point in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
        assert!(objective.evaluate(&[point]).is_err());
        assert!(objective.hessian_vector_product(&[point], &[0.0]).is_err());
    }
}

#[test]
fn caller_budget_and_cancellation_still_govern_stable_evaluation() {
    let variables = [Symbol::new("x")];
    let expression = function("tanh", Expr::symbol("x"));
    let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
    let mut exhausted = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(
        objective
            .evaluate_with_meter(&[20.0], &mut exhausted)
            .is_err()
    );
    struct Cancelled;
    impl BudgetMeter for Cancelled {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            Err(MeterError::Cancelled)
        }
    }
    assert!(
        objective
            .evaluate_with_meter(&[20.0], &mut Cancelled)
            .is_err()
    );
}
