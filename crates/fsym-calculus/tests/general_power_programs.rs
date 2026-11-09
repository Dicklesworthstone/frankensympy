//! Variable powers: independent proofs, bytecode provenance and real domains.

use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{
    CompileError, CompiledExpr, CompiledOp, CompiledResidualSystem, EvalError, MAX_COMPILE_DEPTH,
};
use fsym_calculus::proof::{
    RULE_DIFF_GENERAL, RULE_DIFF_POW_GENERAL, make_diff_term, verify_diff_derivation,
};
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, compiled_expression_matches,
    verify_sparse_jacobian_certificate,
};
use fsym_core::{BigRational, Expr, RealBall, RootBoxWitness, Symbol, certify_root_box, parse};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use std::collections::HashMap;
use std::sync::Arc;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn variables() -> (Vec<Symbol>, HashMap<Symbol, usize>) {
    let variables = vec![Symbol::new("x"), Symbol::new("y")];
    let map = variables
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect();
    (variables, map)
}

fn receipt(source: &Expr, variable: &Symbol, rhs: &Expr, rule: &str) -> DerivationTree {
    let lhs = make_diff_term(source, variable);
    DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(1),
            rule: ProofRule::DefinitionalReduction {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                rule_name: rule.into(),
            },
            claim: Claim::equality(lhs, rhs.clone()),
        }],
        root: StepId(1),
    }
}

#[test]
fn dynamic_powers_require_positive_finite_bases_even_at_integer_exponents() {
    let (_, map) = variables();
    let source = parse("x^y").unwrap();
    let program = CompiledExpr::try_compile(&source, &map).unwrap();
    assert_eq!(
        program.ops(),
        &[
            CompiledOp::LoadVar(0),
            CompiledOp::LoadVar(1),
            CompiledOp::PowPositive
        ]
    );
    assert_eq!(program.try_eval_finite(&[2.0, 3.0]), Ok(8.0));
    close(program.try_eval_finite(&[4.0, 0.5]).unwrap(), 2.0);
    for point in [[0.0, 3.0], [-2.0, 3.0], [-2.0, 0.0], [-0.0, 0.0]] {
        assert_eq!(
            program.try_eval(&point),
            Err(EvalError::NonPositivePowerBase)
        );
        assert_eq!(
            program.try_eval_finite(&point),
            Err(EvalError::NonPositivePowerBase)
        );
    }
    for point in [[f64::INFINITY, 0.0], [2.0, f64::INFINITY], [2.0, f64::NAN]] {
        assert_eq!(program.try_eval(&point), Err(EvalError::NonFiniteValue));
    }
    assert_eq!(
        program.try_eval_finite(&[2.0, 1024.0]),
        Err(EvalError::NonFiniteValue)
    );
    let integer = CompiledExpr::try_compile(&parse("x^3").unwrap(), &map).unwrap();
    assert_eq!(integer.try_eval_finite(&[-2.0, 0.0]), Ok(-8.0));
    assert_eq!(integer.try_eval_finite(&[0.0, 0.0]), Ok(0.0));
}

#[test]
fn provenance_rejects_pointwise_agreement_wrong_operand_order_and_literal_substitution() {
    let (_, map) = variables();
    let source = parse("x^y").unwrap();
    let correct = CompiledExpr::try_compile(&source, &map).unwrap();
    assert!(compiled_expression_matches(&source, &map, &correct).unwrap());
    for ops in [
        vec![
            CompiledOp::LoadVar(1),
            CompiledOp::LoadVar(0),
            CompiledOp::PowPositive,
        ],
        vec![CompiledOp::LoadVar(0), CompiledOp::Pow(2.0)],
        vec![CompiledOp::LoadConst(4.0)],
    ] {
        let forged = CompiledExpr::try_from_ops(ops).unwrap();
        assert_eq!(forged.try_eval_finite(&[2.0, 2.0]), Ok(4.0));
        assert!(!compiled_expression_matches(&source, &map, &forged).unwrap());
    }
}

#[test]
fn independent_general_power_rule_rejects_missing_chain_factors_and_wrong_rules() {
    let x = Symbol::new("x");
    let source = parse("2^(3*x)").unwrap();
    let valid = Expr::Mul(vec![
        source.clone(),
        Expr::Function("log".into(), vec![Expr::from_i64(2)]),
        Expr::from_i64(3),
    ]);
    verify_diff_derivation(
        &receipt(&source, &x, &valid, RULE_DIFF_POW_GENERAL),
        &source,
        &x,
        &valid,
    )
    .unwrap();
    for chain in [0, 1, -3] {
        let forged = Expr::Mul(vec![
            source.clone(),
            Expr::Function("log".into(), vec![Expr::from_i64(2)]),
            Expr::from_i64(chain),
        ]);
        assert!(
            verify_diff_derivation(
                &receipt(&source, &x, &forged, RULE_DIFF_POW_GENERAL),
                &source,
                &x,
                &forged
            )
            .is_err()
        );
    }
    assert!(
        verify_diff_derivation(
            &receipt(&source, &x, &valid, RULE_DIFF_GENERAL),
            &source,
            &x,
            &valid
        )
        .is_err()
    );
    let source = parse("x^x").unwrap();
    let first = Expr::Mul(vec![
        Expr::from_i64(1),
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
    ]);
    let second = Expr::Mul(vec![
        Expr::Sym(x.clone()),
        Expr::from_i64(1),
        Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(-1)),
    ]);
    let valid = Expr::Mul(vec![
        source.clone(),
        Expr::Add(vec![first.clone(), second.clone()]),
    ]);
    verify_diff_derivation(
        &receipt(&source, &x, &valid, RULE_DIFF_POW_GENERAL),
        &source,
        &x,
        &valid,
    )
    .unwrap();
    for missing_term in [first, second] {
        let forged = Expr::Mul(vec![source.clone(), missing_term]);
        assert!(
            verify_diff_derivation(
                &receipt(&source, &x, &forged, RULE_DIFF_POW_GENERAL),
                &source,
                &x,
                &forged
            )
            .is_err()
        );
    }
}

#[test]
fn checked_power_jacobians_match_independent_closed_form() {
    let (vars, _) = variables();
    let residuals = ["x^y", "2^x", "x^x", "(2*x+1)^(x^2)"].map(|source| parse(source).unwrap());
    let program = VerifiedResidualSystem::try_compile(&residuals, &vars).unwrap();
    assert_eq!(program.receipt().proofs_verified, 5);
    let mut values = [0.0; 4];
    let mut jacobian = [0.0; 8];
    program
        .try_dense_program()
        .unwrap()
        .try_eval_system(&[2.0, 3.0], &mut values, &mut jacobian)
        .unwrap();
    assert_eq!(values, [8.0, 4.0, 4.0, 625.0]);
    let expected = [
        12.0,
        8.0 * 2.0f64.ln(),
        4.0 * 2.0f64.ln(),
        0.0,
        4.0 * (2.0f64.ln() + 1.0),
        0.0,
        625.0 * (4.0 * 5.0f64.ln() + 8.0 / 5.0),
        0.0,
    ];
    for (actual, expected) in jacobian.into_iter().zip(expected) {
        close(actual, expected);
    }
}

#[test]
fn omission_and_compiled_derivative_mutants_fail_without_diagnostic_samples() {
    let (vars, _) = variables();
    let source = parse("x^y").unwrap();
    let original = SparseJacobian::try_build(&[source], &vars).unwrap();
    verify_sparse_jacobian_certificate(&original, &[]).unwrap();
    let mut omitted = original.clone();
    omitted.entries.remove(0);
    omitted.pattern.nonzeros.remove(0);
    omitted.pattern.nnz -= 1;
    assert!(matches!(
        verify_sparse_jacobian_certificate(&omitted, &[]),
        Err(JacobianVerificationError::OmittedNonzero { .. })
    ));
    let mut substituted = original;
    substituted.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(12.0)]).unwrap();
    assert!(matches!(
        verify_sparse_jacobian_certificate(&substituted, &[]),
        Err(JacobianVerificationError::CompiledProvenanceMismatch { .. })
    ));
}

#[test]
fn derivative_only_calls_cannot_erase_original_dynamic_power_guards() {
    let x = Symbol::new("x");
    // Keep the source spelling: 0^(0*x) is outside this dynamic-real branch,
    // while algebraic differentiation simplifies its derivative to zero.
    let source = Expr::Pow(
        Arc::new(Expr::from_i64(0)),
        Arc::new(Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(x.clone())])),
    );
    let program = CompiledResidualSystem::try_compile(&[source], &[x]).unwrap();
    assert_eq!(
        program.compiled_jacobian[0].try_eval_finite(&[3.0]),
        Ok(0.0)
    );
    let mut jacobian = [123.0];
    assert_eq!(
        program.try_eval_jacobian(&[3.0], &mut jacobian),
        Err(EvalError::NonPositivePowerBase)
    );
    assert_eq!(jacobian, [123.0]);
    let mut residual = [456.0];
    assert!(
        program
            .try_eval_system(&[3.0], &mut residual, &mut jacobian)
            .is_err()
    );
    assert_eq!(residual, [456.0]);
    assert_eq!(jacobian, [123.0]);
}

#[test]
fn dynamic_bytecode_and_nested_exponents_keep_existing_resource_bounds() {
    for ops in [
        vec![CompiledOp::PowPositive],
        vec![CompiledOp::LoadConst(2.0), CompiledOp::PowPositive],
    ] {
        assert_eq!(
            CompiledExpr::try_from_ops(ops),
            Err(EvalError::StackUnderflow)
        );
    }
    assert!(serde_json::from_str::<CompiledExpr>(r#"{"ops":["PowPositive"]}"#).is_err());
    let mut exponent = Expr::symbol("x");
    for _ in 0..=MAX_COMPILE_DEPTH {
        exponent = Expr::Pow(Arc::new(Expr::from_i64(2)), Arc::new(exponent));
    }
    let map = HashMap::from([(Symbol::new("x"), 0)]);
    assert!(matches!(
        CompiledExpr::try_compile(&exponent, &map),
        Err(CompileError::DepthLimitExceeded { .. })
    ));
}

#[test]
fn transport_replays_variable_power_proofs_and_guarded_bytecode() {
    let vars = [Symbol::new("x")];
    let system = VerifiedResidualSystem::try_compile(&[parse("x^x-4").unwrap()], &vars).unwrap();
    let restored =
        VerifiedResidualSystem::from_json_bytes(&system.to_json_bytes().unwrap()).unwrap();
    assert_eq!(restored.sparse_jacobian(), system.sparse_jacobian());
    let mut residual = [0.0];
    let mut jacobian = [0.0];
    let compiled = restored.try_dense_program().unwrap();
    compiled
        .try_eval_system(&[2.0], &mut residual, &mut jacobian)
        .unwrap();
    assert_eq!(residual, [0.0]);
    close(jacobian[0], 4.0 * (2.0f64.ln() + 1.0));
    assert!(compiled.try_eval_jacobian(&[-2.0], &mut jacobian).is_err());
}

#[test]
fn original_source_interval_ad_certifies_a_power_root_and_rejects_invalid_boxes() {
    let vars = [Symbol::new("x")];
    let system = VerifiedResidualSystem::try_compile(&[parse("x^x-4").unwrap()], &vars).unwrap();
    let q = |n: i64, d: i64| BigRational::new(n.into(), d.into());
    let witness = RootBoxWitness {
        domain: vec![RealBall::new(q(2, 1), q(1, 100)).unwrap()],
        center: vec![q(2, 1)],
        preconditioner: vec![vec![q(1, 7)]],
        precision_digits: 24,
    };
    // This checker differentiates the original source in interval arithmetic;
    // it does not consume or trust the compiled symbolic derivative programs.
    let accepted = certify_root_box(system.residuals(), system.vars(), &witness).unwrap();
    assert!(accepted.contraction_bound() < &q(1, 1));
    assert!(accepted.root_enclosure()[0].lower() > witness.domain[0].lower());
    assert!(accepted.root_enclosure()[0].upper() < witness.domain[0].upper());
    let invalid = RootBoxWitness {
        domain: vec![RealBall::new(q(0, 1), q(3, 1)).unwrap()],
        ..witness
    };
    assert!(certify_root_box(system.residuals(), system.vars(), &invalid).is_err());
}
