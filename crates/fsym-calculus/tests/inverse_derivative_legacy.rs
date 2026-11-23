//! Historical inverse-function recipes remain exact-structure replay inputs.

use fsym_budget::{Budget, BudgetLimits, Dimension, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::CompiledExpr;
use fsym_calculus::program_artifact::{
    admit_objective_candidate, decode_objective_candidate, encode_objective_candidate,
};
use fsym_calculus::proof::{classify_diff_rule, make_diff_term, verified_diff};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{
    SparseJacobian, SparseJacobianEntry, SparsityPattern,
    verify_sparse_jacobian_certificate_with_meter,
};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use serde::Serialize;
use std::collections::HashMap;

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn old_entry(source: &Expr, variable: &Symbol, derivative: Expr) -> SparseJacobian {
    let lhs = make_diff_term(source, variable);
    let derivation = DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(1),
            rule: ProofRule::DefinitionalReduction {
                lhs: lhs.clone(),
                rhs: derivative.clone(),
                rule_name: classify_diff_rule(source, variable).into(),
            },
            claim: Claim::equality(lhs, derivative.clone()),
        }],
        root: StepId(1),
    };
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
            compiled_deriv: CompiledExpr::try_compile(
                &derivative,
                &HashMap::from([(variable.clone(), 0)]),
            )
            .unwrap(),
            symbolic_deriv: derivative,
            derivation,
        }],
        column_colors: vec![0],
        num_colors: 1,
    }
}

#[derive(Serialize)]
struct HistoricalEnvelope<'a> {
    schema_version: u32,
    jacobian: &'a SparseJacobian,
}

fn encoded(candidate: &SparseJacobian) -> Vec<u8> {
    serde_json::to_vec(&HistoricalEnvelope {
        schema_version: 1,
        jacobian: candidate,
    })
    .unwrap()
}

fn inverse_slope(name: &str, argument: Expr) -> Expr {
    let exponent = if name == "atan" {
        Expr::from_i64(-1)
    } else {
        Expr::rational(-1, 2).unwrap()
    };
    Expr::Add(vec![Expr::from_i64(1), argument.pow(Expr::from_i64(2))]).pow(exponent)
}

#[test]
fn historical_inverse_residual_and_objective_programs_preserve_their_original_bytes() {
    let variable = Symbol::new("x");
    for name in ["atan", "asinh"] {
        let source = function(name, Expr::Sym(variable.clone()));
        // On this source the unchanged public rule is exactly the former
        // numerical rule; no new numerical generator reconstructs the past.
        let (first, _) = verified_diff(&source, &variable);
        let gradient = old_entry(&source, &variable, first.clone());
        let bytes = encoded(&gradient);
        let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored.to_json_bytes().unwrap(), bytes);
        assert_eq!(restored.sparse_jacobian(), &gradient);

        let (second, _) = verified_diff(&first, &variable);
        let hessian = old_entry(&first, &variable, second);
        let candidate = ObjectiveCandidate {
            compiled_objective: CompiledExpr::try_compile(
                &source,
                &HashMap::from([(variable.clone(), 0)]),
            )
            .unwrap(),
            gradient,
            hessian,
        };
        let old = VerifiedObjective::try_from_candidate(
            &source,
            std::slice::from_ref(&variable),
            candidate,
        )
        .unwrap();
        let bytes = encode_objective_candidate(&old, &mut Unbounded).unwrap();
        let decoded = decode_objective_candidate(&bytes, &mut Unbounded).unwrap();
        let replayed = admit_objective_candidate(
            &source,
            std::slice::from_ref(&variable),
            decoded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(
            encode_objective_candidate(&replayed, &mut Unbounded).unwrap(),
            bytes
        );
        // Numerical replay retains the historical behavior, even its loss
        // of representable tail curvature. Recompilation is an explicit act.
        let point = if name == "atan" { 1e100 } else { 1e120 };
        assert_eq!(replayed.evaluate(&[point]).unwrap().hessian, [0.0]);
        let fresh =
            VerifiedObjective::try_compile(&source, std::slice::from_ref(&variable)).unwrap();
        assert!(fresh.evaluate(&[point]).unwrap().hessian[0] < 0.0);
    }
}

#[test]
fn historical_inverse_recipes_keep_the_already_stable_tanh_sinc_and_abs_rules() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    for name in ["atan", "asinh"] {
        for (inner_name, inner_slope) in [
            ("tanh", function("sech", x.clone()).pow(Expr::from_i64(2))),
            ("sinc", function("_fsym_sinc_d1", x.clone())),
            ("Abs", function("_fsym_abs_d1", x.clone())),
        ] {
            let argument = function(inner_name, x.clone());
            let source = function(name, argument.clone());
            let derivative = Expr::Mul(vec![inverse_slope(name, argument), inner_slope]);
            let candidate = old_entry(&source, &variable, derivative);
            let bytes = encoded(&candidate);
            let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
            assert_eq!(restored.to_json_bytes().unwrap(), bytes);
            assert_eq!(restored.sparse_jacobian(), &candidate);
            let fresh = VerifiedResidualSystem::try_compile(
                std::slice::from_ref(&source),
                std::slice::from_ref(&variable),
            )
            .unwrap();
            for point in [-2.0, -0.5, 0.5, 2.0] {
                let expected = restored.evaluate_sparse(&[point]).unwrap().jacobian[0];
                let actual = fresh.evaluate_sparse(&[point]).unwrap().jacobian[0];
                assert!((actual - expected).abs() <= 3e-13 * expected.abs());
            }
        }
    }
}

#[test]
fn historical_exact_zero_omissions_survive_the_new_inverse_nodes() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    let source = Expr::Add(vec![
        function("atan", function("tan", x.clone())),
        Expr::Mul(vec![Expr::from_i64(-1), x]),
    ])
    .pow(Expr::from_i64(2));
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
    let bytes = encoded(&candidate);
    let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
    assert_eq!(
        restored.jacobian_vector_product(&[0.25], &[3.0]).unwrap(),
        [0.0]
    );
}

#[test]
fn historical_replay_checks_every_recipe_without_resetting_the_budget() {
    const AMPLE: u64 = 1 << 40;
    let variable = Symbol::new("x");
    let inner = function("tanh", Expr::Sym(variable.clone()));
    let source = function("atan", inner.clone());
    let correct = Expr::Mul(vec![
        inverse_slope("atan", inner),
        function("sech", Expr::Sym(variable.clone())).pow(Expr::from_i64(2)),
    ]);
    let candidate = old_entry(&source, &variable, correct.clone());
    let mut measured = Budget::new(BudgetLimits::uniform(AMPLE, 0));
    verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut measured).unwrap();
    let spent = AMPLE - measured.remaining(Dimension::ComputeSteps);
    assert!(spent > 1);
    for (steps, passes) in [(spent, true), (spent - 1, false)] {
        let mut limits = BudgetLimits::uniform(AMPLE, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = steps;
        let mut budget = Budget::new(limits);
        assert_eq!(
            verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut budget).is_ok(),
            passes
        );
    }
    for wrong in [
        Expr::from_i64(0),
        -correct.clone(),
        correct * Expr::from_i64(2),
    ] {
        // Every envelope, RHS and bytecode agrees internally; only the
        // independent old/current mathematical recipes detect this forgery.
        let forged = old_entry(&source, &variable, wrong);
        assert!(VerifiedResidualSystem::from_json_bytes(&encoded(&forged)).is_err());
    }
}
