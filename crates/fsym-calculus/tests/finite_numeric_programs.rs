//! Native source/provenance/evaluation contracts used by the Python preview.

use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompileError, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::proof::{
    RULE_DIFF_GENERAL, RULE_DIFF_POW_RATIONAL, make_diff_term, verified_diff,
    verify_diff_derivation,
};
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn checked_finite_programs_preserve_typed_variable_slots() {
    let plain = Symbol::new("x");
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let source = Expr::Add(vec![
        Expr::Sym(plain.clone()),
        Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(typed.clone())]),
    ]);
    let variables = HashMap::from([(plain, 0), (typed, 1)]);
    let program = CompiledExpr::try_compile(&source, &variables).unwrap();
    assert!(compiled_expression_matches(&source, &variables, &program).unwrap());
    assert_eq!(program.try_eval_finite(&[3.0, 7.0]).unwrap(), 17.0);
    assert_eq!(program.try_eval_finite(&[7.0, 3.0]).unwrap(), 13.0);
}

#[test]
fn independently_checked_lowering_rejects_substituted_bytecode() {
    let source = parse("x^2 + sin(y)").unwrap();
    let variables = HashMap::from([(Symbol::new("x"), 0), (Symbol::new("y"), 1)]);
    let program = CompiledExpr::try_compile(&source, &variables).unwrap();
    assert!(compiled_expression_matches(&source, &variables, &program).unwrap());
    assert_eq!(program.try_eval_finite(&[3.0, 0.0]).unwrap(), 9.0);
    // Agreement at a particular point does not establish source provenance.
    let replacement = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(9.0)]).unwrap();
    assert_eq!(replacement.try_eval_finite(&[3.0, 0.0]).unwrap(), 9.0);
    assert!(!compiled_expression_matches(&source, &variables, &replacement).unwrap());
}

#[test]
fn finite_programs_cannot_conceal_invalid_intermediates() {
    let variable = Symbol::new("x");
    let variables = HashMap::from([(variable.clone(), 0)]);
    let reciprocal_log = Expr::Pow(
        Arc::new(Expr::Function("log".into(), vec![Expr::Sym(variable)])),
        Arc::new(Expr::from_i64(-1)),
    );
    let program = CompiledExpr::try_compile(&reciprocal_log, &variables).unwrap();
    assert!(compiled_expression_matches(&reciprocal_log, &variables, &program).unwrap());
    assert_eq!(
        program.try_eval(&[0.0]).unwrap().to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(
        program.try_eval_finite(&[0.0]),
        Err(EvalError::NonFiniteValue)
    );
    assert_eq!(
        program.try_eval_finite(&[1.0]),
        Err(EvalError::NonFiniteValue)
    );
    assert!((program.try_eval_finite(&[2.0]).unwrap() - 1.0 / 2.0f64.ln()).abs() < 1e-15);
    assert_eq!(
        program.try_eval_finite(&[f64::NAN]),
        Err(EvalError::NonFiniteValue)
    );
}

#[test]
fn verified_residual_and_jacobian_programs_execute_in_the_finite_lane() {
    let residuals = [parse("x^2 + y").unwrap(), parse("sin(x) - y").unwrap()];
    let system =
        VerifiedResidualSystem::try_compile(&residuals, &[Symbol::new("x"), Symbol::new("y")])
            .unwrap();
    assert_eq!(system.receipt().proofs_verified, 4);
    assert_eq!(system.receipt().test_points_evaluated, 0);
    let residuals: Vec<_> = system
        .compiled()
        .compiled_residuals
        .iter()
        .map(|program| program.try_eval_finite(&[3.0, 4.0]).unwrap())
        .collect();
    let jacobian: Vec<_> = system
        .compiled()
        .compiled_jacobian
        .iter()
        .map(|program| program.try_eval_finite(&[3.0, 4.0]).unwrap())
        .collect();
    assert_eq!(residuals, [13.0, 3.0f64.sin() - 4.0]);
    assert_eq!(jacobian, [6.0, 1.0, 3.0f64.cos(), -1.0]);
}

#[test]
fn zero_variable_systems_and_absolute_values_are_admitted() {
    let constant = VerifiedResidualSystem::try_compile(&[Expr::from_i64(7)], &[]).unwrap();
    assert_eq!(constant.receipt().proofs_verified, 0);
    assert!(constant.compiled().compiled_jacobian.is_empty());
    assert_eq!(
        constant.compiled().compiled_residuals[0]
            .try_eval_finite(&[])
            .unwrap(),
        7.0
    );
    let variables = HashMap::from([(Symbol::new("x"), 0)]);
    let source = parse("abs(x)").unwrap();
    let program = CompiledExpr::try_compile(&source, &variables).unwrap();
    assert!(compiled_expression_matches(&source, &variables, &program).unwrap());
    assert_eq!(program.try_eval_finite(&[-3.0]).unwrap(), 3.0);
}

fn rational_power(base: Expr, numerator: i64, denominator: i64) -> Expr {
    Expr::Pow(
        Arc::new(base),
        Arc::new(Expr::Rational(BigRational::new(
            numerator.into(),
            denominator.into(),
        ))),
    )
}

#[test]
fn rational_power_jacobians_retain_exact_exponents_and_chain_factors() {
    let x = Symbol::new("x");
    let source = rational_power(parse("2*x+1").unwrap(), 3, 2);
    let system = VerifiedResidualSystem::try_compile(&[source], std::slice::from_ref(&x)).unwrap();
    assert_eq!(system.receipt().proofs_verified, 1);
    assert_eq!(
        system.compiled().compiled_residuals[0]
            .try_eval_finite(&[4.0])
            .unwrap(),
        27.0
    );
    assert_eq!(
        system.compiled().compiled_jacobian[0]
            .try_eval_finite(&[4.0])
            .unwrap(),
        9.0
    );
    let source = Expr::Add(vec![
        rational_power(Expr::Sym(x.clone()), 1, 2),
        rational_power(Expr::Sym(x.clone()), -1, 2),
    ]);
    let system = VerifiedResidualSystem::try_compile(&[source], &[x]).unwrap();
    assert_eq!(
        system.compiled().compiled_jacobian[0]
            .try_eval_finite(&[4.0])
            .unwrap(),
        0.1875
    );
}

#[test]
fn square_root_rules_do_not_extend_their_real_domain_or_erase_chain_signs() {
    let x = Symbol::new("x");
    let square = Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2)));
    let nested = rational_power(square, 1, 2);
    let residuals = [
        nested.clone(),
        Expr::Add(vec![nested, Expr::symbol("y")]),
        rational_power(Expr::Sym(x.clone()), 1, 2),
    ];
    let system = VerifiedResidualSystem::try_compile(&residuals, &[x, Symbol::new("y")]).unwrap();
    for (at, expected) in [(-3.0, -1.0), (3.0, 1.0)] {
        for index in [0, 2] {
            assert_eq!(
                system.compiled().compiled_jacobian[index]
                    .try_eval_finite(&[at, 0.0])
                    .unwrap(),
                expected
            );
        }
    }
    for index in [0, 2, 4] {
        assert_eq!(
            system.compiled().compiled_jacobian[index].try_eval_finite(&[0.0, 0.0]),
            Err(EvalError::NonFiniteValue)
        );
    }
    assert_eq!(
        system.compiled().compiled_residuals[2]
            .try_eval_finite(&[0.0, 0.0])
            .unwrap(),
        0.0
    );
    assert_eq!(
        system.compiled().compiled_residuals[2].try_eval_finite(&[-1.0, 0.0]),
        Err(EvalError::NonFiniteValue)
    );
    assert_eq!(
        system.compiled().compiled_jacobian[4].try_eval_finite(&[-1.0, 0.0]),
        Err(EvalError::NonFiniteValue)
    );
}

#[test]
fn hand_authored_rational_power_proofs_reject_coefficient_exponent_and_chain_mutants() {
    let x = Symbol::new("x");
    let base = parse("2*x+1").unwrap();
    let source = rational_power(base.clone(), 3, 2);
    let valid = Expr::Mul(vec![
        Expr::Rational(BigRational::new(3.into(), 2.into())),
        rational_power(base.clone(), 1, 2),
        Expr::from_i64(2),
    ]);
    let receipt = |derivative: &Expr, rule: &str| {
        let lhs = make_diff_term(&source, &x);
        DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(1),
                rule: ProofRule::DefinitionalReduction {
                    lhs: lhs.clone(),
                    rhs: derivative.clone(),
                    rule_name: rule.into(),
                },
                claim: Claim::equality(lhs, derivative.clone()),
            }],
            root: StepId(1),
        }
    };
    verify_diff_derivation(
        &receipt(&valid, RULE_DIFF_POW_RATIONAL),
        &source,
        &x,
        &valid,
    )
    .unwrap();
    for (coefficient, power, chain) in [(1, 1, 2), (3, 3, 2), (3, 1, 1), (3, 1, -2)] {
        let forged = Expr::Mul(vec![
            Expr::Rational(BigRational::new(coefficient.into(), 2.into())),
            rational_power(base.clone(), power, 2),
            Expr::from_i64(chain),
        ]);
        assert!(
            verify_diff_derivation(
                &receipt(&forged, RULE_DIFF_POW_RATIONAL),
                &source,
                &x,
                &forged
            )
            .is_err()
        );
    }
    assert!(
        verify_diff_derivation(&receipt(&valid, RULE_DIFF_GENERAL), &source, &x, &valid).is_err()
    );
    assert!(
        verify_diff_derivation(
            &receipt(&valid, RULE_DIFF_POW_RATIONAL),
            &rational_power(base, 5, 2),
            &x,
            &valid
        )
        .is_err()
    );
    assert!(
        verify_diff_derivation(
            &receipt(&valid, RULE_DIFF_POW_RATIONAL),
            &source,
            &Symbol::new("y"),
            &valid
        )
        .is_err()
    );
}

#[test]
fn rational_integer_payloads_and_independent_bases_have_consistent_proofs() {
    let x = Symbol::new("x");
    for (base, numerator, denominator) in [
        (Expr::Sym(x.clone()), 0, 1),
        (Expr::Sym(x.clone()), 1, 1),
        (Expr::Sym(x.clone()), 2, 1),
        (Expr::symbol("y"), 1, 2),
    ] {
        let source = rational_power(base, numerator, denominator);
        let (derivative, tree) = verified_diff(&source, &x);
        verify_diff_derivation(&tree, &source, &x, &derivative).unwrap();
    }
}

#[test]
fn fractional_exponents_cannot_round_to_a_different_integer_power() {
    let x = Symbol::new("x");
    let denominator = BigInt::from(1) << 54u32;
    let exponent = BigRational::new(&denominator + BigInt::from(1), denominator);
    let source = Expr::Pow(
        Arc::new(Expr::Sym(x.clone())),
        Arc::new(Expr::Rational(exponent)),
    );
    assert!(matches!(
        CompiledExpr::try_compile(&source, &HashMap::from([(x, 0)])),
        Err(CompileError::UnsupportedExponent(_))
    ));
}
