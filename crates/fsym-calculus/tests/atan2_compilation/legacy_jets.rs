//! Historical atan2 programs retain their exact source, derivatives and bytecode.
//! These fixtures spell out the old quotient recipe instead of generating it
//! through today's numerical differentiator.

use fsym_budget::{Budget, BudgetLimits, Dimension};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, CompiledOp};
use fsym_calculus::proof::{classify_diff_rule, make_diff_term, verified_diff};
use fsym_calculus::sparse_hessian::{
    ObjectiveCandidate, ObjectiveCompilationError, VerifiedObjective,
};
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, SparseJacobianEntry, SparsityPattern,
    verify_sparse_jacobian_certificate, verify_sparse_jacobian_certificate_with_meter,
};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use std::collections::HashMap;

fn function(name: &str, arguments: Vec<Expr>) -> Expr {
    Expr::Function(name.to_string(), arguments)
}

fn old_angle_derivative(y: &Expr, x: &Expr, dy: Expr, dx: Expr) -> Expr {
    Expr::Mul(vec![
        Expr::Add(vec![
            Expr::Mul(vec![x.clone(), dy]),
            Expr::Mul(vec![Expr::from_i64(-1), y.clone(), dx]),
        ]),
        Expr::pow(
            Expr::Add(vec![
                Expr::pow(x.clone(), Expr::from_i64(2)),
                Expr::pow(y.clone(), Expr::from_i64(2)),
            ]),
            Expr::from_i64(-1),
        ),
    ])
}

fn entry(source: &Expr, variable: &Symbol, derivative: Expr) -> SparseJacobianEntry {
    let lhs = make_diff_term(source, variable);
    let derivation = DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(1),
            rule: ProofRule::DefinitionalReduction {
                lhs: lhs.clone(),
                rhs: derivative.clone(),
                rule_name: classify_diff_rule(source, variable).to_string(),
            },
            claim: Claim::equality(lhs, derivative.clone()),
        }],
        root: StepId(1),
    };
    let mapping = HashMap::from([(variable.clone(), 0)]);
    SparseJacobianEntry {
        row: 0,
        col: 0,
        compiled_deriv: CompiledExpr::try_compile(&derivative, &mapping).unwrap(),
        symbolic_deriv: derivative,
        derivation,
    }
}

fn sparse(source: Expr, variable: &Symbol, derivative: Expr) -> SparseJacobian {
    let stored = entry(&source, variable, derivative);
    SparseJacobian {
        residuals: vec![source],
        vars: vec![variable.clone()],
        pattern: SparsityPattern {
            num_rows: 1,
            num_cols: 1,
            nonzeros: vec![(0, 0)],
            nnz: 1,
        },
        entries: vec![stored],
        column_colors: vec![0],
        num_colors: 1,
    }
}

fn old_scalar() -> SparseJacobian {
    let variable = Symbol::new("t");
    let y = Expr::Sym(variable.clone());
    let x = Expr::from_i64(1);
    let source = function("atan2", vec![y.clone(), x.clone()]);
    let derivative = old_angle_derivative(&y, &x, Expr::from_i64(1), Expr::from_i64(0));
    sparse(source, &variable, derivative)
}

fn old_nested_inverse() -> SparseJacobian {
    let variable = Symbol::new("t");
    let argument = Expr::Sym(variable.clone());
    let y = function("atan", vec![argument.clone()]);
    let x = function("asinh", vec![argument.clone()]);
    let source = function("atan2", vec![y.clone(), x.clone()]);
    // The immediately preceding recipe already had the unary inverse jets.
    // Replaying the pre-inverse recipe instead cannot accept this derivative.
    let derivative = old_angle_derivative(
        &y,
        &x,
        function("_fsym_atan_d1", vec![argument.clone()]),
        function("_fsym_asinh_d1", vec![argument]),
    );
    sparse(source, &variable, derivative)
}

fn old_nested_lambert(branch: i64) -> SparseJacobian {
    let variable = Symbol::new("t");
    let argument = Expr::Sym(variable.clone());
    let y = function("LambertW", vec![argument.clone(), Expr::from_i64(branch)]);
    let x = Expr::from_i64(1);
    let source = function("atan2", vec![y.clone(), x.clone()]);
    let first = if branch == 0 {
        "_fsym_lambert_w0_d1"
    } else {
        "_fsym_lambert_wm1_d1"
    };
    // Fixed real Lambert-W jets preceded the scaled angle recipe. Replaying
    // this stored derivative must retain both the old angle quotient and the
    // already-admitted Lambert branch rather than mixing recipe generations.
    let derivative =
        old_angle_derivative(&y, &x, function(first, vec![argument]), Expr::from_i64(0));
    sparse(source, &variable, derivative)
}

fn old_omitted_zero() -> SparseJacobian {
    let variable = Symbol::new("t");
    let argument = Expr::Sym(variable.clone());
    SparseJacobian {
        residuals: vec![function("atan2", vec![argument.clone(), argument])],
        vars: vec![variable],
        pattern: SparsityPattern {
            num_rows: 1,
            num_cols: 1,
            nonzeros: vec![],
            nnz: 0,
        },
        entries: vec![],
        column_colors: vec![0],
        num_colors: 1,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "non-finite result: {actual}");
    assert!(
        (actual - expected).abs() <= 2.0e-13 * expected.abs().max(1.0),
        "expected {expected}, got {actual}"
    );
}

fn assert_no_angle_jet(program: &CompiledExpr) {
    assert!(!program.ops().iter().any(|operation| matches!(
        operation,
        CompiledOp::Atan2DerivativeY
            | CompiledOp::Atan2DerivativeX
            | CompiledOp::Atan2SecondDerivativeYY
            | CompiledOp::Atan2SecondDerivativeYX
            | CompiledOp::Atan2SecondDerivativeXX
    )));
}

#[test]
fn old_angle_recipe_survives_schema_one_replay_without_recompilation() {
    for candidate in [
        old_scalar(),
        old_nested_inverse(),
        old_nested_lambert(0),
        old_nested_lambert(-1),
    ] {
        verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
        let envelope = serde_json::json!({
            "schema_version": 1,
            "jacobian": candidate.clone(),
        });
        let bytes = serde_json::to_vec(&envelope).unwrap();
        let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
        let exported = restored.to_json_bytes().unwrap();
        let retained: serde_json::Value = serde_json::from_slice(&exported).unwrap();
        assert_eq!(
            retained["jacobian"],
            serde_json::to_value(&candidate).unwrap()
        );
        let replayed = VerifiedResidualSystem::from_json_bytes(&exported).unwrap();
        assert_eq!(replayed.to_json_bytes().unwrap(), exported);
        assert_no_angle_jet(&candidate.entries[0].compiled_deriv);
    }
}

#[test]
fn immediately_previous_recipe_retains_scaled_unary_inverse_children() {
    let candidate = old_nested_inverse();
    verify_sparse_jacobian_certificate(&candidate, &[vec![0.75]]).unwrap();
    let t = 0.75_f64;
    let y = t.atan();
    let x = t.asinh();
    let expected = (x / (1.0 + t * t) - y / (1.0 + t * t).sqrt()) / (x * x + y * y);
    let actual = candidate.entries[0]
        .compiled_deriv
        .try_eval_finite(&[t])
        .unwrap();
    assert_close(actual, expected);
    assert!(
        candidate.entries[0]
            .compiled_deriv
            .ops()
            .contains(&CompiledOp::AtanDerivative)
    );
    assert!(
        candidate.entries[0]
            .compiled_deriv
            .ops()
            .contains(&CompiledOp::AsinhDerivative)
    );
}

#[test]
fn immediately_previous_angle_recipe_retains_both_real_lambert_branches() {
    let e = std::f64::consts::E;
    for (branch, point, expected, operation, opposite) in [
        (
            0,
            e,
            1.0 / (4.0 * e),
            CompiledOp::LambertWDerivative,
            "_fsym_lambert_wm1_d1",
        ),
        (
            -1,
            -2.0 / (e * e),
            -(e * e) / 5.0,
            CompiledOp::LambertWMinusOneDerivative,
            "_fsym_lambert_w0_d1",
        ),
    ] {
        let mut candidate = old_nested_lambert(branch);
        verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
        let stored = &candidate.entries[0].compiled_deriv;
        assert_no_angle_jet(stored);
        assert!(stored.ops().contains(&operation));
        assert_close(stored.try_eval_finite(&[point]).unwrap(), expected);

        let Expr::Function(_, arguments) = &candidate.residuals[0] else {
            panic!("fixture must retain its original angle source");
        };
        let wrong = old_angle_derivative(
            &arguments[0],
            &arguments[1],
            function(opposite, vec![Expr::Sym(candidate.vars[0].clone())]),
            Expr::from_i64(0),
        );
        candidate.entries[0] = entry(&candidate.residuals[0], &candidate.vars[0], wrong);
        assert!(matches!(
            verify_sparse_jacobian_certificate(&candidate, &[]),
            Err(JacobianVerificationError::ProofFailed { .. }),
        ));
    }
}

#[test]
fn historical_replay_rejects_coherent_forged_formula_and_swapped_sources() {
    let candidate = old_nested_inverse();
    let source = candidate.residuals[0].clone();
    let variable = candidate.vars[0].clone();
    let original = candidate.entries[0].symbolic_deriv.clone();

    let mut forged = candidate.clone();
    let false_derivative = Expr::Add(vec![original.clone(), Expr::from_i64(1)]);
    // Change the value, proof claim, rule RHS and executable bytecode together.
    // Only the independently reconstructed mathematical rule can reject it.
    forged.entries[0] = entry(&source, &variable, false_derivative);
    assert!(matches!(
        verify_sparse_jacobian_certificate(&forged, &[]),
        Err(JacobianVerificationError::ProofFailed { .. })
    ));

    let Expr::Function(_, arguments) = source else {
        panic!("fixture must be an atan2 application");
    };
    let swapped = function("atan2", vec![arguments[1].clone(), arguments[0].clone()]);
    forged.residuals[0] = swapped.clone();
    forged.entries[0] = entry(&swapped, &variable, original);
    assert!(matches!(
        verify_sparse_jacobian_certificate(&forged, &[]),
        Err(JacobianVerificationError::ProofFailed { .. })
    ));
}

#[test]
fn historical_replay_rejects_bytecode_that_disagrees_with_its_valid_proof() {
    let mut candidate = old_scalar();
    candidate.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(1.0)]).unwrap();
    assert!(matches!(
        verify_sparse_jacobian_certificate(&candidate, &[]),
        Err(JacobianVerificationError::CompiledProvenanceMismatch { row: 0, col: 0 })
    ));
}

#[test]
fn historical_exact_zero_omission_remains_bound_to_its_original_source() {
    let mut candidate = old_omitted_zero();
    let receipt = verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
    assert_eq!(receipt.nnz, 0);
    assert_eq!(receipt.proofs_verified, 0);
    candidate.residuals[0] = function(
        "atan2",
        vec![Expr::Sym(candidate.vars[0].clone()), Expr::from_i64(1)],
    );
    assert!(matches!(
        verify_sparse_jacobian_certificate(&candidate, &[]),
        Err(JacobianVerificationError::OmittedNonzero { row: 0, col: 0, .. })
    ));
}

#[test]
fn historical_semantic_retries_share_an_exact_cumulative_verifier_budget() {
    const AMPLE: u64 = 1 << 40;
    for candidate in [old_nested_inverse(), old_omitted_zero()] {
        let mut measured = Budget::new(BudgetLimits::uniform(AMPLE, 0));
        verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut measured).unwrap();
        let spent = AMPLE - measured.remaining(Dimension::ComputeSteps);
        assert!(spent > 0);
        for (steps, succeeds) in [(spent, true), (spent - 1, false)] {
            let mut limits = BudgetLimits::uniform(AMPLE, 0);
            limits.dimensions[Dimension::ComputeSteps.index()] = steps;
            let mut budget = Budget::new(limits);
            assert_eq!(
                verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut budget).is_ok(),
                succeeds
            );
        }
    }
}

fn old_objective_candidate() -> (Expr, Vec<Symbol>, ObjectiveCandidate) {
    let gradient = old_scalar();
    let source = gradient.residuals[0].clone();
    let variable = gradient.vars[0].clone();
    let old_first = gradient.entries[0].symbolic_deriv.clone();
    // This source contains only the historical rational formula. Its unchanged
    // public symbolic derivative supplies the old second-stage product rule;
    // today's numerical atan2 generator is never invoked.
    let (old_second, _) = verified_diff(&old_first, &variable);
    let hessian = sparse(old_first, &variable, old_second);
    let compiled_objective =
        CompiledExpr::try_compile(&source, &HashMap::from([(variable.clone(), 0)])).unwrap();
    (
        source,
        vec![variable],
        ObjectiveCandidate {
            compiled_objective,
            gradient,
            hessian,
        },
    )
}

#[test]
fn historical_objective_replays_both_stages_against_the_stored_gradient() {
    let (source, variables, candidate) = old_objective_candidate();
    let retained = candidate.clone();
    let program = VerifiedObjective::try_from_candidate(&source, &variables, candidate).unwrap();
    assert_eq!(program.candidate(), &retained);
    let result = program.evaluate(&[1.0]).unwrap();
    assert_close(result.value, std::f64::consts::FRAC_PI_4);
    assert_eq!(result.gradient.len(), 1);
    assert_close(result.gradient[0], 0.5);
    assert_eq!(result.hessian.len(), 1);
    assert_close(result.hessian[0], -0.5);
    assert_no_angle_jet(&program.candidate().gradient.entries[0].compiled_deriv);
    assert_no_angle_jet(&program.candidate().hessian.entries[0].compiled_deriv);
}

#[test]
fn historical_objective_rejects_a_valid_hessian_for_a_different_gradient() {
    let (source, variables, mut candidate) = old_objective_candidate();
    let other_gradient = Expr::pow(Expr::Sym(variables[0].clone()), Expr::from_i64(3));
    let (other_second, _) = verified_diff(&other_gradient, &variables[0]);
    candidate.hessian = sparse(other_gradient, &variables[0], other_second);
    assert!(matches!(
        VerifiedObjective::try_from_candidate(&source, &variables, candidate),
        Err(ObjectiveCompilationError::SourceMismatch(
            "checked gradient source"
        ))
    ));
}
