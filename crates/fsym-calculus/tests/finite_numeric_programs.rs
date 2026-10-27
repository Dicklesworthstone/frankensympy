//! Native source/provenance/evaluation contracts used by the Python preview.

use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
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
