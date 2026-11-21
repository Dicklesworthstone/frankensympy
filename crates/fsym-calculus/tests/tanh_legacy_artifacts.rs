//! Schema-1 derivative recipes remain replayable without rewriting their code.

use fsym_budget::{Budget, BudgetLimits, Dimension, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::CompiledExpr;
use fsym_calculus::program_artifact::{
    admit_objective_candidate, admit_residual_candidate, decode_objective_candidate,
    decode_residual_candidate_with_legacy, encode_objective_candidate,
};
use fsym_calculus::proof::{RULE_DIFF_TANH, make_diff_term, verified_diff};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{
    SparseJacobian, SparseJacobianEntry, SparsityPattern,
    verify_sparse_jacobian_certificate_with_meter,
};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use serde::Serialize;
use std::collections::HashMap;

const AMPLE: u64 = 1 << 40;

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn one_entry(
    source: &Expr,
    variable: &Symbol,
    derivative: Expr,
    derivation: DerivationTree,
) -> SparseJacobian {
    let compiled_deriv =
        CompiledExpr::try_compile(&derivative, &HashMap::from([(variable.clone(), 0)])).unwrap();
    SparseJacobian {
        residuals: vec![source.clone()],
        vars: vec![variable.clone()],
        pattern: SparsityPattern {
            num_rows: 1,
            num_cols: 1,
            nonzeros: vec![(0, 0)],
            nnz: 1,
        },
        entries: vec![SparseJacobianEntry {
            row: 0,
            col: 0,
            symbolic_deriv: derivative,
            compiled_deriv,
            derivation,
        }],
        column_colors: vec![0],
        num_colors: 1,
    }
}

fn old_tanh_candidate() -> SparseJacobian {
    let variable = Symbol::new("x");
    let source = function("tanh", Expr::Sym(variable.clone()));
    // The unchanged public symbolic recipe is exactly the pre-stabilization
    // numerical recipe for this source. Do not call the new numeric generator.
    let (derivative, proof) = verified_diff(&source, &variable);
    one_entry(&source, &variable, derivative, proof)
}

#[derive(Serialize)]
struct HistoricalEnvelope<'a> {
    schema_version: u32,
    jacobian: &'a SparseJacobian,
}

fn legacy_bytes(candidate: &SparseJacobian) -> Vec<u8> {
    serde_json::to_vec(&HistoricalEnvelope {
        schema_version: 1,
        jacobian: candidate,
    })
    .unwrap()
}

fn tanh_proof(source: &Expr, variable: &Symbol, derivative: &Expr) -> DerivationTree {
    let lhs = make_diff_term(source, variable);
    DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(1),
            rule: ProofRule::DefinitionalReduction {
                lhs: lhs.clone(),
                rhs: derivative.clone(),
                rule_name: RULE_DIFF_TANH.into(),
            },
            claim: Claim::equality(lhs, derivative.clone()),
        }],
        root: StepId(1),
    }
}

#[test]
fn historical_tanh_bundle_preserves_its_exact_derivative_program_and_bytes() {
    let candidate = old_tanh_candidate();
    let bytes = legacy_bytes(&candidate);
    let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
    assert_eq!(restored.sparse_jacobian(), &candidate);
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);

    let decoded = decode_residual_candidate_with_legacy(&bytes, &mut Unbounded).unwrap();
    let admitted = admit_residual_candidate(
        &candidate.residuals,
        &candidate.vars,
        decoded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(admitted.to_json_bytes().unwrap(), bytes);
    // Historical bytecode retains its old binary64 cancellation behavior.
    // Replaying a saved computation must not silently regenerate its code.
    assert_eq!(admitted.evaluate_sparse(&[20.0]).unwrap().jacobian, [0.0]);
    let regenerated =
        VerifiedResidualSystem::try_compile(&candidate.residuals, &candidate.vars).unwrap();
    assert!(regenerated.evaluate_sparse(&[20.0]).unwrap().jacobian[0] > 0.0);
}

#[test]
fn compound_legacy_recipe_retains_stable_coth_inside_the_old_tanh_rule() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    let source = function("tanh", function("coth", x.clone()));
    // Schema-1 used (1 - tanh(coth(x))^2) * (-csch(x)^2).
    // Its coth derivative was already stable; symbolic 1-coth(x)^2 is not
    // a valid substitute for the historical numeric rule table.
    let outer = Expr::Add(vec![
        Expr::from_i64(1),
        Expr::Mul(vec![
            Expr::from_i64(-1),
            source.clone().pow(Expr::from_i64(2)),
        ]),
    ]);
    let inner = Expr::Mul(vec![
        Expr::from_i64(-1),
        function("csch", x).pow(Expr::from_i64(2)),
    ]);
    let derivative = Expr::Mul(vec![outer, inner]);
    let proof = tanh_proof(&source, &variable, &derivative);
    let candidate = one_entry(&source, &variable, derivative, proof);
    let bytes = legacy_bytes(&candidate);
    let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
    let point = 20.0f64;
    let expected = -point.tanh().recip().cosh().recip().powi(2) * point.sinh().recip().powi(2);
    let actual = restored.evaluate_sparse(&[point]).unwrap().jacobian[0];
    assert!(actual < 0.0);
    assert!((actual - expected).abs() <= expected.abs() * 3e-13);
}

#[test]
fn old_objective_gradient_and_hessian_replay_as_one_unchanged_artifact() {
    let gradient = old_tanh_candidate();
    let source = gradient.residuals[0].clone();
    let variable = gradient.vars[0].clone();
    let gradient_source = gradient.entries[0].symbolic_deriv.clone();
    let (second, proof) = verified_diff(&gradient_source, &variable);
    let hessian = one_entry(&gradient_source, &variable, second, proof);
    let candidate = ObjectiveCandidate {
        compiled_objective: CompiledExpr::try_compile(
            &source,
            &HashMap::from([(variable.clone(), 0)]),
        )
        .unwrap(),
        gradient,
        hessian,
    };
    let objective =
        VerifiedObjective::try_from_candidate(&source, std::slice::from_ref(&variable), candidate)
            .unwrap();
    let bytes = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
    let decoded = decode_objective_candidate(&bytes, &mut Unbounded).unwrap();
    let restored =
        admit_objective_candidate(&source, &[variable], decoded, &mut Unbounded).unwrap();
    assert_eq!(
        encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
    assert_eq!(restored.evaluate(&[20.0]).unwrap().gradient, [0.0]);
    assert_eq!(restored.evaluate(&[20.0]).unwrap().hessian, [0.0]);
}

#[test]
fn historical_zero_omission_survives_the_changed_tanh_recipe() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    let difference = Expr::Add(vec![
        function("atanh", function("tanh", x.clone())),
        Expr::Mul(vec![Expr::from_i64(-1), x]),
    ]);
    let source = difference.pow(Expr::from_i64(2));
    // In the historical recipe the inner derivative cancels exactly as
    // (1-tanh(x)^2)/(1-tanh(x)^2)-1, so the outer power has derivative zero.
    // This source has no coth or literal identity powers; public symbolic
    // differentiation reconstructs the former numeric recipe exactly.
    let (historical, _) = verified_diff(&source, &variable);
    assert!(historical.is_zero());
    let candidate = SparseJacobian {
        residuals: vec![source],
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
    };
    let bytes = legacy_bytes(&candidate);
    let decoded = decode_residual_candidate_with_legacy(&bytes, &mut Unbounded).unwrap();
    let restored = admit_residual_candidate(
        &candidate.residuals,
        &candidate.vars,
        decoded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
    assert!(
        restored
            .evaluate_sparse(&[0.25])
            .unwrap()
            .jacobian
            .is_empty()
    );
    assert_eq!(
        restored.jacobian_vector_product(&[0.25], &[3.0]).unwrap(),
        [0.0]
    );
}

#[test]
fn self_consistent_but_false_legacy_rhs_is_rejected_after_both_rule_checks() {
    let original = old_tanh_candidate();
    let source = &original.residuals[0];
    let variable = &original.vars[0];
    for wrong in [
        Expr::from_i64(0),
        Expr::Add(vec![
            original.entries[0].symbolic_deriv.clone(),
            Expr::rational(1, 1_000_000_000_000_000).unwrap(),
        ]),
    ] {
        let proof = tanh_proof(source, variable, &wrong);
        let forged = one_entry(source, variable, wrong, proof);
        let bytes = legacy_bytes(&forged);
        assert!(VerifiedResidualSystem::from_json_bytes(&bytes).is_err());
        let decoded = decode_residual_candidate_with_legacy(&bytes, &mut Unbounded).unwrap();
        assert!(
            admit_residual_candidate(&original.residuals, &original.vars, decoded, &mut Unbounded,)
                .is_err()
        );
    }
}

#[test]
fn legacy_verification_uses_one_cumulative_budget_for_the_complete_replay() {
    let candidate = old_tanh_candidate();
    let mut measured = Budget::new(BudgetLimits::uniform(AMPLE, 0));
    verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut measured).unwrap();
    let steps = AMPLE - measured.remaining(Dimension::ComputeSteps);
    assert!(steps > 1);
    let mut limits = BudgetLimits::uniform(AMPLE, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = steps;
    let mut exact = Budget::new(limits);
    verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut exact).unwrap();
    assert_eq!(exact.remaining(Dimension::ComputeSteps), 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = steps - 1;
    let mut short = Budget::new(limits);
    assert!(verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut short).is_err());
    assert!(short.remaining(Dimension::ComputeSteps) < steps - 1);
}
