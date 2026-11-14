//! Explicit principal roots through source-bound derivative and numeric paths.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::proof::{
    RULE_DIFF_POW_RATIONAL, RULE_DIFF_SQRT, make_diff_term, verified_diff, verify_diff_derivation,
    verify_diff_receipt,
};
use fsym_calculus::sparse_hessian::{ObjectiveEvaluationError, VerifiedObjective};
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use std::collections::HashMap;

fn sqrt(inner: Expr) -> Expr {
    // Keep the actual Function node: evaluating a sqrt constructor to Pow
    // would exercise the already supported rational-power lane instead.
    Expr::Function("sqrt".into(), vec![inner])
}

fn fraction(numerator: i64, denominator: i64) -> Expr {
    Expr::Rational(BigRational::new(numerator.into(), denominator.into()))
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-13 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
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

#[test]
fn explicit_root_chain_has_an_independently_reconstructed_derivative_and_receipt() {
    let variable = Symbol::new("x");
    let inner = parse("2*x+1").unwrap();
    let source = sqrt(inner.clone());
    let expected = Expr::Mul(vec![
        fraction(1, 2),
        Expr::pow(inner, fraction(-1, 2)),
        Expr::from_i64(2),
    ]);
    let (generated, tree) = verified_diff(&source, &variable);
    assert_eq!(generated, expected);
    verify_diff_derivation(&tree, &source, &variable, &expected).unwrap();
    let text = expected.to_string();
    verify_diff_receipt(&source, &variable, RULE_DIFF_SQRT, &text, &text).unwrap();
    assert!(verify_diff_receipt(&source, &variable, RULE_DIFF_SQRT, "0", &text).is_err());

    // Constancy with respect to the requested coordinate remains distinct
    // from a statement about the real domain of the source value.
    for constant in [Expr::from_i64(4), Expr::symbol("y")] {
        let source = sqrt(constant);
        let (derivative, tree) = verified_diff(&source, &variable);
        assert_eq!(derivative, Expr::from_i64(0));
        verify_diff_derivation(&tree, &source, &variable, &derivative).unwrap();
    }
}

#[test]
fn independent_admission_rejects_wrong_half_power_chain_and_source_identity() {
    let variable = Symbol::new("x");
    let inner = parse("3*x+1").unwrap();
    let source = sqrt(inner.clone());
    let inverse_root = Expr::pow(inner.clone(), fraction(-1, 2));
    let correct = Expr::Mul(vec![fraction(1, 2), inverse_root.clone(), Expr::from_i64(3)]);
    verify_diff_derivation(
        &derivation(&source, &variable, &correct, RULE_DIFF_SQRT),
        &source,
        &variable,
        &correct,
    )
    .unwrap();
    for wrong in [
        Expr::Mul(vec![inverse_root.clone(), Expr::from_i64(3)]),
        Expr::Mul(vec![fraction(1, 2), inverse_root]),
        Expr::Mul(vec![
            fraction(1, 2),
            Expr::pow(inner, fraction(1, 2)),
            Expr::from_i64(3),
        ]),
        Expr::Mul(vec![
            fraction(1, 2),
            Expr::pow(Expr::symbol("x"), fraction(-1, 2)),
            Expr::from_i64(3),
        ]),
        Expr::from_i64(0),
    ] {
        assert!(
            verify_diff_derivation(
                &derivation(&source, &variable, &wrong, RULE_DIFF_SQRT),
                &source,
                &variable,
                &wrong,
            )
            .is_err()
        );
    }
    assert!(
        verify_diff_derivation(
            &derivation(&source, &variable, &correct, RULE_DIFF_POW_RATIONAL),
            &source,
            &variable,
            &correct,
        )
        .is_err()
    );
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([91; 32]));
    let original_tree = derivation(&source, &variable, &correct, RULE_DIFF_SQRT);
    assert!(verify_diff_derivation(&original_tree, &source, &other, &correct).is_err());
    assert!(
        verify_diff_derivation(&original_tree, &sqrt(Expr::symbol("x")), &variable, &correct)
            .is_err()
    );
}

#[test]
fn nonlinear_square_root_drives_sparse_jacobians_hessians_and_vector_products() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let source = sqrt(parse("x^2+y^2+1").unwrap());
    let residual =
        VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
    let result = residual.evaluate_sparse(&[2.0, 2.0]).unwrap();
    close(result.residuals[0], 3.0);
    assert_eq!(residual.jacobian_pattern(), [(0, 0), (0, 1)]);
    for derivative in result.jacobian {
        close(derivative, 2.0 / 3.0);
    }
    close(
        residual
            .jacobian_vector_product(&[2.0, 2.0], &[2.0, 3.0])
            .unwrap()[0],
        10.0 / 3.0,
    );
    let transposed = residual
        .jacobian_transpose_vector_product(&[2.0, 2.0], &[3.0])
        .unwrap();
    for value in transposed {
        close(value, 2.0);
    }

    let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
    assert_eq!(objective.hessian_pattern(), [(0, 0), (0, 1), (1, 0), (1, 1)]);
    let result = objective.evaluate(&[2.0, 2.0]).unwrap();
    close(result.value, 3.0);
    for derivative in result.gradient {
        close(derivative, 2.0 / 3.0);
    }
    for (actual, expected) in result.hessian.iter().zip([
        5.0 / 27.0,
        -4.0 / 27.0,
        -4.0 / 27.0,
        5.0 / 27.0,
    ]) {
        close(*actual, expected);
    }
    let product = objective
        .hessian_vector_product(&[2.0, 2.0], &[2.0, 3.0])
        .unwrap();
    close(product[0], -2.0 / 27.0);
    close(product[1], 7.0 / 27.0);

    let restored =
        VerifiedResidualSystem::from_json_bytes(&residual.to_json_bytes().unwrap()).unwrap();
    assert_eq!(restored.residuals(), &[source]);
    assert_eq!(
        restored.evaluate_sparse(&[2.0, 2.0]).unwrap(),
        residual.evaluate_sparse(&[2.0, 2.0]).unwrap()
    );
}

#[test]
fn nested_roots_and_vanishing_inner_gradients_keep_the_second_chain_term() {
    let variable = Symbol::new("x");
    let source = sqrt(sqrt(Expr::Sym(variable.clone())));
    let objective = VerifiedObjective::try_compile(&source, std::slice::from_ref(&variable)).unwrap();
    let result = objective.evaluate(&[16.0]).unwrap();
    close(result.value, 2.0);
    close(result.gradient[0], 1.0 / 32.0);
    close(result.hessian[0], -3.0 / 2048.0);

    let source = sqrt(parse("x^2+1").unwrap());
    let objective = VerifiedObjective::try_compile(&source, &[variable]).unwrap();
    let result = objective.evaluate(&[0.0]).unwrap();
    close(result.value, 1.0);
    close(result.gradient[0], 0.0);
    close(result.hessian[0], 1.0);
}

#[test]
fn derivative_singularity_and_original_source_domain_refuse_before_output_publication() {
    let variable = Symbol::new("x");
    let source = sqrt(Expr::Sym(variable.clone()));
    let mapping = HashMap::from([(variable.clone(), 0)]);
    let scalar = CompiledExpr::try_compile(&source, &mapping).unwrap();
    // The value exists at zero; the derivative does not.
    assert_eq!(scalar.try_eval_finite(&[0.0]), Ok(0.0));
    let residual = VerifiedResidualSystem::try_compile(
        std::slice::from_ref(&source),
        std::slice::from_ref(&variable),
    )
    .unwrap();
    let objective =
        VerifiedObjective::try_compile(&source, std::slice::from_ref(&variable)).unwrap();
    for point in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut values = [71.0];
        let mut derivatives = [83.0];
        assert!(
            residual
                .try_eval_sparse_into(&[point], &mut values, &mut derivatives)
                .is_err()
        );
        assert_eq!(values, [71.0]);
        assert_eq!(derivatives, [83.0]);
        assert!(residual.jacobian_vector_product(&[point], &[0.0]).is_err());
        assert!(objective.evaluate(&[point]).is_err());
        assert!(objective.hessian_vector_product(&[point], &[0.0]).is_err());
    }
    let result = objective.evaluate(&[4.0]).unwrap();
    close(result.value, 2.0);
    close(result.gradient[0], 1.0 / 4.0);
    close(result.hessian[0], -1.0 / 32.0);

    // Zero multiplication cannot erase an undefined original square root.
    let hidden = Expr::Mul(vec![Expr::from_i64(0), source]);
    let residual = VerifiedResidualSystem::try_compile(
        std::slice::from_ref(&hidden),
        std::slice::from_ref(&variable),
    )
    .unwrap();
    let objective = VerifiedObjective::try_compile(&hidden, &[variable]).unwrap();
    assert!(residual.evaluate_sparse(&[-1.0]).is_err());
    assert!(objective.evaluate(&[-1.0]).is_err());
}

#[test]
fn malformed_roots_and_substituted_bytecode_never_gain_derivative_evidence() {
    let variable = Symbol::new("x");
    let mapping = HashMap::from([(variable.clone(), 0)]);
    for arguments in [vec![], vec![Expr::symbol("x"), Expr::from_i64(1)]] {
        let source = Expr::Function("sqrt".into(), arguments);
        let wrong = Expr::from_i64(0);
        assert!(CompiledExpr::try_compile(&source, &mapping).is_err());
        assert!(
            VerifiedResidualSystem::try_compile(
                std::slice::from_ref(&source),
                std::slice::from_ref(&variable),
            )
            .is_err()
        );
        assert!(
            verify_diff_derivation(
                &derivation(&source, &variable, &wrong, RULE_DIFF_SQRT),
                &source,
                &variable,
                &wrong,
            )
            .is_err()
        );
    }
    let source = sqrt(Expr::symbol("x"));
    let program =
        CompiledExpr::try_compile_checked_with_meter(&source, &mapping, &mut Unbounded).unwrap();
    assert!(compiled_expression_matches(&source, &mapping, &program).unwrap());
    let forged =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), CompiledOp::Exp]).unwrap();
    assert!(!compiled_expression_matches(&source, &mapping, &forged).unwrap());
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
fn verified_root_compilation_and_evaluation_keep_cancellation_and_budget_accounting() {
    let variables = [Symbol::new("x")];
    let sources = [sqrt(Expr::symbol("x"))];
    let limits = BudgetLimits::uniform(100_000_000, 37);
    let mut counted = InterruptAt {
        budget: Budget::new(limits),
        checkpoints: 0,
        interrupt: None,
    };
    let residual =
        VerifiedResidualSystem::try_compile_with_meter(&sources, &variables, &mut counted).unwrap();
    for checkpoint in [1, counted.checkpoints / 2, counted.checkpoints] {
        let mut meter = InterruptAt {
            budget: Budget::new(limits),
            checkpoints: 0,
            interrupt: Some(checkpoint),
        };
        assert!(
            VerifiedResidualSystem::try_compile_with_meter(&sources, &variables, &mut meter).is_err()
        );
        assert_eq!(meter.checkpoints, checkpoint);
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
        VerifiedResidualSystem::try_compile_with_meter(&sources, &variables, &mut exact).unwrap();
        assert_eq!(exact.remaining(dimension), 0);
        assert!(
            VerifiedResidualSystem::try_compile_with_meter(&sources, &variables, &mut exact).is_err()
        );
        assert_eq!(exact.verifier_remaining(), 37);
    }

    let objective = VerifiedObjective::try_compile(&sources[0], &variables).unwrap();
    for objective_lane in [false, true] {
        let mut probe = InterruptAt {
            budget: Budget::new(limits),
            checkpoints: 0,
            interrupt: None,
        };
        if objective_lane {
            objective.evaluate_with_meter(&[4.0], &mut probe).unwrap();
        } else {
            residual
                .evaluate_sparse_with_meter(&[4.0], &mut probe)
                .unwrap();
        }
        for checkpoint in 1..=probe.checkpoints {
            let mut meter = InterruptAt {
                budget: Budget::new(limits),
                checkpoints: 0,
                interrupt: Some(checkpoint),
            };
            if objective_lane {
                assert_eq!(
                    objective.evaluate_with_meter(&[4.0], &mut meter),
                    Err(ObjectiveEvaluationError::Evaluation(EvalError::Meter(
                        MeterError::Cancelled,
                    )))
                );
            } else {
                assert_eq!(
                    residual.evaluate_sparse_with_meter(&[4.0], &mut meter),
                    Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
                        MeterError::Cancelled,
                    )))
                );
            }
        }
    }
}
