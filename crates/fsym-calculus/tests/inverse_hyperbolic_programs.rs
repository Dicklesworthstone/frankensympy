//! Independent value, derivative, publication and source-binding regressions.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::proof::{
    RULE_DIFF_ACOSH, RULE_DIFF_ASINH, RULE_DIFF_ATANH, make_diff_term, verify_diff_derivation,
};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_calculus::verified::VerifiedResidualSystem;
use fsym_core::{BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use std::collections::HashMap;

fn source(name: &str) -> Expr {
    Expr::Function(name.into(), vec![Expr::symbol("x")])
}

fn scalar(name: &str) -> CompiledExpr {
    CompiledExpr::try_compile(&source(name), &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-13 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn real_principal_values_cover_signed_inputs_and_the_closed_acosh_endpoint() {
    for point in [-1e100_f64, -2.0, -0.75, 0.0, 0.75, 2.0, 1e100] {
        close(
            scalar("asinh").try_eval_finite(&[point]).unwrap(),
            point.asinh(),
        );
    }
    for point in [1.0_f64, 1.25, 2.0, 1e100] {
        close(
            scalar("acosh").try_eval_finite(&[point]).unwrap(),
            point.acosh(),
        );
    }
    for point in [-0.99_f64, -0.6, 0.0, 0.6, 0.99] {
        close(
            scalar("atanh").try_eval_finite(&[point]).unwrap(),
            point.atanh(),
        );
    }
    assert_eq!(scalar("acosh").try_eval_finite(&[1.0]).unwrap(), 0.0);
}

#[test]
fn original_undefined_values_survive_zero_products_and_zero_directions() {
    for (name, points) in [
        ("acosh", vec![-2.0, 0.0, 0.99]),
        ("atanh", vec![-2.0, -1.0, 1.0, 2.0]),
    ] {
        let hidden = Expr::Mul(vec![Expr::from_i64(0), source(name)]);
        let variables = [Symbol::new("x")];
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&hidden), &variables).unwrap();
        let objective = VerifiedObjective::try_compile(&hidden, &variables).unwrap();
        for point in points {
            assert_eq!(
                scalar(name).try_eval_finite(&[point]),
                Err(EvalError::NonFiniteValue)
            );
            assert!(residual.evaluate_sparse(&[point]).is_err());
            assert!(residual.jacobian_vector_product(&[point], &[0.0]).is_err());
            assert!(objective.evaluate(&[point]).is_err());
            assert!(objective.hessian_vector_product(&[point], &[0.0]).is_err());
        }
    }
    for name in ["asinh", "acosh", "atanh"] {
        for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                scalar(name).try_eval_finite(&[point]),
                Err(EvalError::NonFiniteValue)
            );
        }
    }
}

#[test]
fn opcodes_roundtrip_and_source_checkers_reject_function_substitution() {
    let mapping = HashMap::from([(Symbol::new("x"), 0)]);
    for (name, operation, other) in [
        ("asinh", CompiledOp::Asinh, CompiledOp::Atanh),
        ("acosh", CompiledOp::Acosh, CompiledOp::Asinh),
        ("atanh", CompiledOp::Atanh, CompiledOp::Acosh),
    ] {
        let program = scalar(name);
        let bytes = serde_json::to_vec(&program).unwrap();
        let restored: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored.ops(), program.ops());
        let point = if name == "acosh" { 1.25 } else { 0.6 };
        assert_eq!(
            restored.try_eval_finite(&[point]),
            program.try_eval_finite(&[point])
        );
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation]),
            Err(EvalError::StackUnderflow)
        );
        assert!(compiled_expression_matches(&source(name), &mapping, &program).unwrap());
        let forged = CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), other]).unwrap();
        assert!(!compiled_expression_matches(&source(name), &mapping, &forged).unwrap());
    }
}

fn derivation(source: &Expr, variable: &Symbol, rhs: &Expr, rule: &str) -> DerivationTree {
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

fn principal_acosh_derivative(inner: &Expr, chain: Expr) -> Expr {
    let exponent = Expr::Rational(BigRational::new((-1).into(), 2.into()));
    Expr::Mul(vec![
        Expr::pow(
            Expr::Add(vec![inner.clone(), Expr::from_i64(-1)]),
            exponent.clone(),
        ),
        Expr::pow(Expr::Add(vec![inner.clone(), Expr::from_i64(1)]), exponent),
        chain,
    ])
}

#[test]
fn independent_rule_admission_rejects_wrong_denominators_powers_and_missing_chain_factors() {
    let variable = Symbol::new("x");
    let inner = parse("2*x").unwrap();
    let square = Expr::pow(inner.clone(), Expr::from_i64(2));
    for (name, rule, denominator, exponent) in [
        (
            "asinh",
            RULE_DIFF_ASINH,
            Expr::Add(vec![Expr::from_i64(1), square.clone()]),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
        (
            "acosh",
            RULE_DIFF_ACOSH,
            Expr::Add(vec![square.clone(), Expr::from_i64(-1)]),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
        (
            "atanh",
            RULE_DIFF_ATANH,
            Expr::Add(vec![
                Expr::from_i64(1),
                Expr::Mul(vec![Expr::from_i64(-1), square]),
            ]),
            Expr::from_i64(-1),
        ),
    ] {
        let source = Expr::Function(name.into(), vec![inner.clone()]);
        let factor = Expr::pow(denominator.clone(), exponent.clone());
        let correct = if name == "acosh" {
            principal_acosh_derivative(&inner, Expr::from_i64(2))
        } else {
            Expr::Mul(vec![factor.clone(), Expr::from_i64(2)])
        };
        verify_diff_derivation(
            &derivation(&source, &variable, &correct, rule),
            &source,
            &variable,
            &correct,
        )
        .unwrap();
        let mut mutations = vec![
            factor.clone(),
            Expr::Mul(vec![
                Expr::pow(denominator, Expr::from_i64(-2)),
                Expr::from_i64(2),
            ]),
            Expr::Mul(vec![
                Expr::pow(Expr::from_i64(1), exponent),
                Expr::from_i64(2),
            ]),
        ];
        if name == "acosh" {
            // This agrees on x > 1/2 but is not the general principal
            // complex derivative; the source has no real-domain condition.
            mutations.push(Expr::Mul(vec![factor, Expr::from_i64(2)]));
            mutations.push(principal_acosh_derivative(&inner, Expr::from_i64(1)));
        }
        for wrong in mutations {
            assert!(
                verify_diff_derivation(
                    &derivation(&source, &variable, &wrong, rule),
                    &source,
                    &variable,
                    &wrong
                )
                .is_err()
            );
        }
        let wrong_rule = if rule == RULE_DIFF_ASINH {
            RULE_DIFF_ATANH
        } else {
            RULE_DIFF_ASINH
        };
        assert!(
            verify_diff_derivation(
                &derivation(&source, &variable, &correct, wrong_rule),
                &source,
                &variable,
                &correct
            )
            .is_err()
        );
    }
}

#[test]
fn acosh_complex_chain_retains_the_principal_branch_in_its_derivative_receipt() {
    let variable = Symbol::new("x");
    let inner = Expr::Add(vec![
        Expr::Const(Constant::I),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(variable.clone())]),
    ]);
    let source = Expr::Function("acosh".into(), vec![inner.clone()]);
    let expected = principal_acosh_derivative(&inner, Expr::from_i64(-1));
    let (generated, tree) = fsym_calculus::proof::verified_diff(&source, &variable);
    assert_eq!(generated, expected);
    verify_diff_derivation(&tree, &source, &variable, &expected).unwrap();

    let exponent = Expr::Rational(BigRational::new((-1).into(), 2.into()));
    let old_formula = Expr::Mul(vec![
        Expr::pow(
            Expr::Add(vec![
                Expr::pow(inner, Expr::from_i64(2)),
                Expr::from_i64(-1),
            ]),
            exponent,
        ),
        Expr::from_i64(-1),
    ]);
    assert!(
        verify_diff_derivation(
            &derivation(&source, &variable, &old_formula, RULE_DIFF_ACOSH),
            &source,
            &variable,
            &old_formula,
        )
        .is_err()
    );

    // At z = -2+i the two principal square roots have first-quadrant
    // arguments whose sum is in quadrant II. Inverting and applying the
    // chain factor -1 puts the derivative in quadrant I. The old combined
    // square root chooses the opposite sign. This numerical observation is
    // a regression oracle, not evidence used by the receipt verifier.
    let simplified = fsym_calculus::diff(&source, &variable);
    let values = HashMap::from([(variable, Expr::from_i64(2))]);
    let actual = fsym_calculus::gruntz::complex_value(&expected.subs(&values)).unwrap();
    let old = fsym_calculus::gruntz::complex_value(&old_formula.subs(&values)).unwrap();
    let simplified_value = fsym_calculus::gruntz::complex_value(&simplified.subs(&values)).unwrap();
    assert!(actual.re > 0.1 && actual.im > 0.1);
    assert!(old.re < -0.1 && old.im < -0.1);
    close(actual.re, -old.re);
    close(actual.im, -old.im);
    close(simplified_value.re, actual.re);
    close(simplified_value.im, actual.im);
}

#[test]
fn rational_anchors_have_the_exact_chain_gradient_hessian_and_products() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    // At these independently chosen rational anchors every value is log(2),
    // and both derivative orders are rational numbers.
    for (name, point, slope, curvature) in [
        ("asinh", [0.25, 0.25], 4.0 / 5.0, -48.0 / 125.0),
        ("acosh", [0.5, 0.25], 4.0 / 3.0, -80.0 / 27.0),
        ("atanh", [0.2, 0.2], 25.0 / 16.0, 375.0 / 128.0),
    ] {
        let expression = parse(&format!("{name}(2*x+y)")).unwrap();
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables)
                .unwrap();
        let result = residual.evaluate_sparse(&point).unwrap();
        close(result.residuals[0], 2.0_f64.ln());
        close(result.jacobian[0], 2.0 * slope);
        close(result.jacobian[1], slope);
        close(
            residual
                .jacobian_vector_product(&point, &[2.0, 3.0])
                .unwrap()[0],
            7.0 * slope,
        );
        let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
        assert_eq!(
            objective.hessian_pattern(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        let result = objective.evaluate(&point).unwrap();
        close(result.value, 2.0_f64.ln());
        close(result.gradient[0], 2.0 * slope);
        close(result.gradient[1], slope);
        for (actual, factor) in result.hessian.iter().zip([4.0, 2.0, 2.0, 1.0]) {
            close(*actual, factor * curvature);
        }
        let product = objective
            .hessian_vector_product(&point, &[2.0, 3.0])
            .unwrap();
        close(product[0], 14.0 * curvature);
        close(product[1], 7.0 * curvature);
        let restored =
            VerifiedResidualSystem::from_json_bytes(&residual.to_json_bytes().unwrap()).unwrap();
        assert_eq!(
            restored.evaluate_sparse(&point).unwrap(),
            residual.evaluate_sparse(&point).unwrap()
        );
    }
}

#[test]
fn nonlinear_chain_keeps_inner_curvature_in_the_checked_objective() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let objective =
        VerifiedObjective::try_compile(&parse("asinh(x^2+y)").unwrap(), &variables).unwrap();
    let result = objective.evaluate(&[0.5, 0.5]).unwrap();
    close(result.value, 2.0_f64.ln());
    assert_eq!(
        objective.hessian_pattern(),
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    for actual in result.gradient {
        close(actual, 4.0 / 5.0);
    }
    for (actual, expected) in
        result
            .hessian
            .iter()
            .zip([152.0 / 125.0, -48.0 / 125.0, -48.0 / 125.0, -48.0 / 125.0])
    {
        close(*actual, expected);
    }
}

#[test]
fn acosh_derivative_endpoint_failure_is_atomic_and_the_program_remains_reusable() {
    let variables = [Symbol::new("x")];
    let residual = VerifiedResidualSystem::try_compile(&[source("acosh")], &variables).unwrap();
    let mut values = [71.0];
    let mut derivatives = [83.0];
    assert!(
        residual
            .try_eval_sparse_into(&[1.0], &mut values, &mut derivatives)
            .is_err()
    );
    assert_eq!(values, [71.0]);
    assert_eq!(derivatives, [83.0]);
    assert!(residual.jacobian_vector_product(&[1.0], &[0.0]).is_err());
    residual
        .try_eval_sparse_into(&[1.25], &mut values, &mut derivatives)
        .unwrap();
    close(values[0], 2.0_f64.ln());
    close(derivatives[0], 4.0 / 3.0);
    let objective = VerifiedObjective::try_compile(&source("acosh"), &variables).unwrap();
    assert!(objective.evaluate(&[1.0]).is_err());
    assert!(objective.hessian_vector_product(&[1.0], &[0.0]).is_err());
}

#[test]
fn equal_printed_names_keep_their_distinct_inverse_hyperbolic_coordinates() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([51; 32]));
    let expression = Expr::Add(vec![
        Expr::Function("asinh".into(), vec![Expr::Sym(first.clone())]),
        Expr::Function("atanh".into(), vec![Expr::Sym(second.clone())]),
    ]);
    let objective = VerifiedObjective::try_compile(&expression, &[first, second]).unwrap();
    assert_eq!(objective.hessian_pattern(), [(0, 0), (1, 1)]);
    let result = objective.evaluate(&[0.75, 0.6]).unwrap();
    close(result.value, 2.0 * 2.0_f64.ln());
    close(result.gradient[0], 4.0 / 5.0);
    close(result.gradient[1], 25.0 / 16.0);
    close(result.hessian[0], -48.0 / 125.0);
    close(result.hessian[1], 375.0 / 128.0);
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
fn new_operations_preserve_cumulative_accounting_and_cancellation() {
    for (name, point) in [("asinh", 0.75), ("acosh", 1.25), ("atanh", 0.6)] {
        let program = scalar(name);
        let limits = BudgetLimits::uniform(1_000_000, 37);
        let mut counted = InterruptAt {
            budget: Budget::new(limits),
            checkpoints: 0,
            interrupt: None,
        };
        program
            .try_eval_finite_with_meter(&[point], &mut counted)
            .unwrap();
        for interrupt in 1..=counted.checkpoints {
            let mut meter = InterruptAt {
                budget: Budget::new(limits),
                checkpoints: 0,
                interrupt: Some(interrupt),
            };
            assert_eq!(
                program.try_eval_finite_with_meter(&[point], &mut meter),
                Err(EvalError::Meter(MeterError::Cancelled))
            );
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
                .try_eval_finite_with_meter(&[point], &mut exact)
                .unwrap();
            assert_eq!(exact.remaining(dimension), 0);
            assert!(matches!(
                program.try_eval_finite_with_meter(&[point], &mut exact),
                Err(EvalError::Meter(MeterError::Budget(_)))
            ));
            assert_eq!(exact.verifier_remaining(), 37);
        }
    }
}
