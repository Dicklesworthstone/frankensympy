//! Direct reciprocal functions retain their source, poles and derivative claims.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompiledBatch, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::proof::{
    RULE_DIFF_COT, RULE_DIFF_COTH, RULE_DIFF_CSC, RULE_DIFF_CSCH, RULE_DIFF_SEC, RULE_DIFF_SECH,
    classify_diff_rule, make_diff_term, verified_diff, verify_diff_derivation,
};
use fsym_calculus::sparse_hessian::{
    ObjectiveCandidate, ObjectiveCompilationError, VerifiedObjective,
};
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, compiled_expression_matches,
    verify_sparse_jacobian_certificate,
};
use fsym_calculus::verified::VerifiedResidualSystem;
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use std::collections::HashMap;

const NAMES: [&str; 6] = ["cot", "sec", "csc", "coth", "sech", "csch"];

fn source(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn scalar(name: &str) -> CompiledExpr {
    CompiledExpr::try_compile(
        &source(name, Expr::symbol("x")),
        &HashMap::from([(Symbol::new("x"), 0)]),
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 3e-12 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

// Independent moderate-range reference from the defining sine/cosine and
// sinh/cosh quotients, including both analytic derivative orders.
fn reference(name: &str, point: f64) -> (f64, f64, f64) {
    let (sine, cosine) = point.sin_cos();
    let sinh = point.sinh();
    let cosh = point.cosh();
    match name {
        "cot" => (
            cosine / sine,
            -1.0 / sine.powi(2),
            2.0 * cosine / sine.powi(3),
        ),
        "sec" => (
            1.0 / cosine,
            sine / cosine.powi(2),
            (1.0 + sine.powi(2)) / cosine.powi(3),
        ),
        "csc" => (
            1.0 / sine,
            -cosine / sine.powi(2),
            (1.0 + cosine.powi(2)) / sine.powi(3),
        ),
        "coth" => (cosh / sinh, -1.0 / sinh.powi(2), 2.0 * cosh / sinh.powi(3)),
        "sech" => (
            1.0 / cosh,
            -sinh / cosh.powi(2),
            (sinh.powi(2) - 1.0) / cosh.powi(3),
        ),
        "csch" => (
            1.0 / sinh,
            -cosh / sinh.powi(2),
            (1.0 + cosh.powi(2)) / sinh.powi(3),
        ),
        _ => unreachable!(),
    }
}

#[test]
fn direct_values_follow_defining_quotients_on_positive_and_negative_branches() {
    for name in NAMES {
        let program = scalar(name);
        for point in [-3.0, -1.0, -0.25, 0.25, 1.0, 3.0] {
            close(
                program.try_eval_finite(&[point]).unwrap(),
                reference(name, point).0,
            );
        }
        for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                program.try_eval_finite(&[point]),
                Err(EvalError::NonFiniteValue)
            );
        }
    }
    for name in ["sec", "sech"] {
        assert_eq!(scalar(name).try_eval_finite(&[0.0]).unwrap(), 1.0);
        assert_eq!(scalar(name).try_eval_finite(&[-0.0]).unwrap(), 1.0);
    }
}

#[test]
fn hyperbolic_reciprocals_do_not_overflow_or_invent_tiny_argument_poles() {
    for name in ["coth", "csch"] {
        for point in [-1e-300_f64, -1e-100, 1e-100, 1e-300] {
            let value = scalar(name).try_eval_finite(&[point]).unwrap();
            close(value * point, 1.0);
        }
    }
    // sinh/cosh overflow here, but their reciprocals remain representable.
    // The independent logarithmic anchor gives sech(x) ~= csch(x) ~= 2^-1025.
    let point = 1026.0 * std::f64::consts::LN_2;
    let expected = f64::from_bits(1u64 << 49); // exactly 2^-1025, a subnormal
    assert!(expected > 0.0 && point.cosh().is_infinite());
    for sign in [-1.0, 1.0] {
        for name in ["sech", "csch"] {
            let value = scalar(name).try_eval_finite(&[sign * point]).unwrap();
            assert!(value.abs() > 0.0);
            close(value.abs() / expected, 1.0);
            assert_eq!(value.is_sign_negative(), name == "csch" && sign < 0.0);
            let underflow = scalar(name).try_eval_finite(&[sign * 1000.0]).unwrap();
            assert_eq!(underflow, 0.0);
            let objective = VerifiedObjective::try_compile(
                &source(name, Expr::symbol("x")),
                &[Symbol::new("x")],
            )
            .unwrap();
            assert!(objective.evaluate(&[sign * 1000.0]).is_ok());
        }
        assert_eq!(
            scalar("coth").try_eval_finite(&[sign * 1000.0]).unwrap(),
            sign
        );
        assert_eq!(
            scalar("coth").try_eval_finite(&[sign * f64::MAX]).unwrap(),
            sign
        );
        assert_eq!(
            scalar("sech").try_eval_finite(&[sign * f64::MAX]).unwrap(),
            0.0
        );
        assert_eq!(
            scalar("csch").try_eval_finite(&[sign * f64::MAX]).unwrap(),
            0.0
        );
    }
    // A mathematically finite value outside the binary64 range is still a
    // finite-real evaluation refusal, not saturation or a successful infinity.
    assert_eq!(
        scalar("csch").try_eval_finite(&[f64::from_bits(1)]),
        Err(EvalError::NonFiniteValue)
    );
}

#[test]
fn serialized_operations_are_unary_and_independently_bound_to_the_exact_source() {
    let operations = [
        CompiledOp::Cot,
        CompiledOp::Sec,
        CompiledOp::Csc,
        CompiledOp::Coth,
        CompiledOp::Sech,
        CompiledOp::Csch,
    ];
    let mapping = HashMap::from([(Symbol::new("x"), 0)]);
    for (name, operation) in NAMES.into_iter().zip(&operations) {
        let expression = source(name, Expr::symbol("x"));
        let program = scalar(name);
        assert_eq!(program.ops(), &[CompiledOp::LoadVar(0), operation.clone()]);
        let bytes = serde_json::to_vec(&program).unwrap();
        let restored: CompiledExpr = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, program);
        assert_eq!(
            restored.try_eval_finite(&[0.75]),
            program.try_eval_finite(&[0.75])
        );
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation.clone()]),
            Err(EvalError::StackUnderflow)
        );
        assert!(compiled_expression_matches(&expression, &mapping, &program).unwrap());
        for other in &operations {
            if operation != other {
                let forged =
                    CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), other.clone()])
                        .unwrap();
                assert!(!compiled_expression_matches(&expression, &mapping, &forged).unwrap());
            }
        }
        for arguments in [vec![], vec![Expr::symbol("x"), Expr::from_i64(1)]] {
            let malformed = Expr::Function(name.into(), arguments);
            assert!(CompiledExpr::try_compile(&malformed, &mapping).is_err());
            assert!(compiled_expression_matches(&malformed, &mapping, &program).is_err());
        }
    }
}

fn derivation(expression: &Expr, variable: &Symbol, rhs: &Expr, rule: &str) -> DerivationTree {
    let lhs = make_diff_term(expression, variable);
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
fn independent_derivative_rules_reject_changed_sign_chain_source_and_operator() {
    let rules = [
        RULE_DIFF_COT,
        RULE_DIFF_SEC,
        RULE_DIFF_CSC,
        RULE_DIFF_COTH,
        RULE_DIFF_SECH,
        RULE_DIFF_CSCH,
    ];
    let variable = Symbol::new("x");
    for (index, (name, rule)) in NAMES.into_iter().zip(rules).enumerate() {
        let expression = source(name, parse("3*x").unwrap());
        assert_eq!(classify_diff_rule(&expression, &variable), rule);
        let (correct, tree) = verified_diff(&expression, &variable);
        verify_diff_derivation(&tree, &expression, &variable, &correct).unwrap();
        let Expr::Mul(mut missing_chain) = correct.clone() else {
            panic!("a reciprocal chain derivative must retain its factors");
        };
        assert_eq!(missing_chain.pop(), Some(Expr::from_i64(3)));
        let changed_argument =
            correct.subs(&HashMap::from([(variable.clone(), Expr::symbol("y"))]));
        for wrong in [
            Expr::from_i64(0),
            Expr::from_i64(1),
            Expr::Mul(missing_chain),
            Expr::Mul(vec![Expr::from_i64(-1), correct.clone()]),
            changed_argument,
        ] {
            assert!(
                verify_diff_derivation(
                    &derivation(&expression, &variable, &wrong, rule),
                    &expression,
                    &variable,
                    &wrong,
                )
                .is_err()
            );
        }
        let other_rule = rules[(index + 1) % rules.len()];
        assert!(
            verify_diff_derivation(
                &derivation(&expression, &variable, &correct, other_rule),
                &expression,
                &variable,
                &correct,
            )
            .is_err()
        );
        let malformed = Expr::Function(name.into(), vec![Expr::symbol("x"), Expr::from_i64(1)]);
        assert!(
            verify_diff_derivation(
                &derivation(&malformed, &variable, &correct, rule),
                &malformed,
                &variable,
                &correct,
            )
            .is_err()
        );
    }
}

#[test]
fn checked_residuals_jacobians_hessians_and_products_match_analytic_quotient_rules() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let point = [0.25, 0.5];
    for name in NAMES {
        let expression = source(name, parse("2*x+y").unwrap());
        let (value, slope, curvature) = reference(name, 1.0);
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables)
                .unwrap();
        let evaluation = residual.evaluate_sparse(&point).unwrap();
        close(evaluation.residuals[0], value);
        close(evaluation.jacobian[0], 2.0 * slope);
        close(evaluation.jacobian[1], slope);
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
        let evaluation = objective.evaluate(&point).unwrap();
        close(evaluation.value, value);
        close(evaluation.gradient[0], 2.0 * slope);
        close(evaluation.gradient[1], slope);
        for (actual, factor) in evaluation.hessian.iter().zip([4.0, 2.0, 2.0, 1.0]) {
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
fn nonlinear_inner_curvature_is_preserved_in_every_reciprocal_hessian() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    for name in NAMES {
        let expression = source(name, parse("x^2+y").unwrap());
        let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
        let evaluation = objective.evaluate(&[0.5, 0.75]).unwrap();
        let (_, slope, curvature) = reference(name, 1.0);
        for actual in evaluation.gradient {
            close(actual, slope);
        }
        for (actual, expected) in evaluation.hessian.iter().zip([
            curvature + 2.0 * slope,
            curvature,
            curvature,
            curvature,
        ]) {
            close(*actual, expected);
        }
    }
}

#[test]
fn numerical_coth_gradients_do_not_cancel_to_zero_at_large_finite_arguments() {
    let variable = Symbol::new("x");
    let coth = source("coth", Expr::Sym(variable.clone()));
    for outer_exp in [false, true] {
        let expression = if outer_exp {
            source("exp", coth.clone())
        } else {
            coth.clone()
        };
        let objective =
            VerifiedObjective::try_compile(&expression, std::slice::from_ref(&variable)).unwrap();
        let residual = VerifiedResidualSystem::try_compile(
            std::slice::from_ref(&expression),
            std::slice::from_ref(&variable),
        )
        .unwrap();
        for point in [-100.0, -20.0, 20.0, 100.0] {
            let (value, slope, curvature) = reference("coth", point);
            let (expected_slope, expected_curvature) = if outer_exp {
                (
                    value.exp() * slope,
                    value.exp() * (curvature + slope * slope),
                )
            } else {
                (slope, curvature)
            };
            assert!(expected_slope != 0.0 && expected_curvature != 0.0);
            let result = objective.evaluate(&[point]).unwrap();
            assert!(result.gradient[0] < 0.0 && result.hessian[0] != 0.0);
            close(result.gradient[0] / expected_slope, 1.0);
            close(result.hessian[0] / expected_curvature, 1.0);
            close(
                residual.jacobian_vector_product(&[point], &[1.0]).unwrap()[0] / expected_slope,
                1.0,
            );
            close(
                objective.hessian_vector_product(&[point], &[1.0]).unwrap()[0] / expected_curvature,
                1.0,
            );
        }
        let restored =
            VerifiedResidualSystem::from_json_bytes(&residual.to_json_bytes().unwrap()).unwrap();
        assert_eq!(
            restored.evaluate_sparse(&[20.0]).unwrap(),
            residual.evaluate_sparse(&[20.0]).unwrap()
        );
    }
}

#[test]
fn ordinary_coth_receipts_keep_their_form_and_numeric_receipts_require_their_exact_formula() {
    let variable = Symbol::new("x");
    let expression = source("coth", Expr::Sym(variable.clone()));
    let expected_ordinary = Expr::Mul(vec![
        Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![
                Expr::from_i64(-1),
                Expr::pow(expression.clone(), Expr::from_i64(2)),
            ]),
        ]),
        Expr::from_i64(1),
    ]);
    let (ordinary, tree) = verified_diff(&expression, &variable);
    assert_eq!(ordinary, expected_ordinary);
    assert_eq!(
        fsym_calculus::diff_unsimplified(&expression, &variable),
        expected_ordinary
    );
    verify_diff_derivation(&tree, &expression, &variable, &ordinary).unwrap();

    let candidate = SparseJacobian::try_build(
        std::slice::from_ref(&expression),
        std::slice::from_ref(&variable),
    )
    .unwrap();
    verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
    assert!(
        candidate.entries[0]
            .compiled_deriv
            .ops()
            .contains(&CompiledOp::Csch)
    );
    assert!(
        !candidate.entries[0]
            .compiled_deriv
            .ops()
            .contains(&CompiledOp::Coth)
    );
    let mut forged = candidate.clone();
    let wrong = Expr::Mul(vec![
        Expr::from_i64(-1),
        forged.entries[0].symbolic_deriv.clone(),
    ]);
    forged.entries[0].derivation = derivation(&expression, &variable, &wrong, RULE_DIFF_COTH);
    forged.entries[0].compiled_deriv =
        CompiledExpr::try_compile(&wrong, &HashMap::from([(variable, 0)])).unwrap();
    forged.entries[0].symbolic_deriv = wrong;
    assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
}

#[test]
fn omitted_coth_gradient_and_hessian_entries_cannot_be_accepted_as_zeros() {
    let variables = [Symbol::new("x")];
    let expression = source("coth", Expr::Sym(variables[0].clone()));
    let original = ObjectiveCandidate::try_build(&expression, &variables).unwrap();
    for second_order in [false, true] {
        let mut candidate = original.clone();
        let stage = if second_order {
            &mut candidate.hessian
        } else {
            &mut candidate.gradient
        };
        assert_eq!(stage.entries.len(), 1);
        stage.entries.clear();
        stage.pattern.nonzeros.clear();
        stage.pattern.nnz = 0;
        stage.validate_structure().unwrap();
        assert!(matches!(
            verify_sparse_jacobian_certificate(stage, &[]),
            Err(JacobianVerificationError::OmittedNonzero { row: 0, col: 0, .. })
        ));
        assert!(matches!(
            VerifiedObjective::try_from_candidate(&expression, &variables, candidate),
            Err(ObjectiveCompilationError::Verification(
                JacobianVerificationError::OmittedNonzero { row: 0, col: 0, .. }
            ))
        ));
    }
}

#[test]
fn original_poles_survive_zero_products_zeroth_powers_and_zero_directions() {
    let variables = [Symbol::new("x")];
    for name in ["cot", "csc", "coth", "csch"] {
        let original = source(name, Expr::symbol("x"));
        for expression in [
            original.clone(),
            Expr::Mul(vec![Expr::from_i64(0), original.clone()]),
            Expr::Pow(original.clone().into(), Expr::from_i64(0).into()),
            source("sech", original),
        ] {
            let residual =
                VerifiedResidualSystem::try_compile(std::slice::from_ref(&expression), &variables)
                    .unwrap();
            let objective = VerifiedObjective::try_compile(&expression, &variables).unwrap();
            for point in [0.0, -0.0] {
                assert!(residual.evaluate_sparse(&[point]).is_err());
                assert!(residual.jacobian_vector_product(&[point], &[0.0]).is_err());
                assert!(objective.evaluate(&[point]).is_err());
                assert!(objective.hessian_vector_product(&[point], &[0.0]).is_err());
            }
        }
    }
}

#[test]
fn domain_failure_publishes_no_partial_arrays_and_does_not_poison_later_evaluations() {
    for name in ["cot", "csc", "coth", "csch"] {
        let system = VerifiedResidualSystem::try_compile(
            &[source(name, Expr::symbol("x"))],
            &[Symbol::new("x")],
        )
        .unwrap();
        let mut values = [71.0];
        let mut derivatives = [83.0];
        assert!(
            system
                .try_eval_sparse_into(&[0.0], &mut values, &mut derivatives)
                .is_err()
        );
        assert_eq!(values, [71.0]);
        assert_eq!(derivatives, [83.0]);
        system
            .try_eval_sparse_into(&[1.0], &mut values, &mut derivatives)
            .unwrap();
        let (value, slope, _) = reference(name, 1.0);
        close(values[0], value);
        close(derivatives[0], slope);
    }
}

#[test]
fn identical_spelling_keeps_the_two_reciprocal_coordinates_independent() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([67; 32]));
    let expression = Expr::Add(vec![
        source("sech", Expr::Sym(first.clone())),
        source("csch", Expr::Sym(second.clone())),
    ]);
    let objective = VerifiedObjective::try_compile(&expression, &[first, second]).unwrap();
    assert_eq!(objective.hessian_pattern(), [(0, 0), (1, 1)]);
    let result = objective.evaluate(&[1.0, 2.0]).unwrap();
    let first = reference("sech", 1.0);
    let second = reference("csch", 2.0);
    close(result.value, first.0 + second.0);
    close(result.gradient[0], first.1);
    close(result.gradient[1], second.1);
    close(result.hessian[0], first.2);
    close(result.hessian[1], second.2);
}

#[test]
fn shared_execution_preserves_every_opcode_and_inactive_reciprocal_domains() {
    let mut programs = Vec::new();
    for name in NAMES {
        programs.extend([scalar(name), scalar(name)]);
    }
    let batch = CompiledBatch::try_from_programs(programs.clone()).unwrap();
    assert!(batch.shared_subexpression_count() >= NAMES.len());
    for point in [-1.0, 0.25, 1.0] {
        let expected: Vec<_> = programs
            .iter()
            .map(|program| program.try_eval_finite(&[point]).unwrap().to_bits())
            .collect();
        let actual: Vec<_> = batch
            .try_eval_finite(&[point])
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        assert_eq!(actual, expected);
    }
    assert_eq!(
        batch.try_eval_finite(&[0.0]),
        Err(EvalError::NonFiniteValue)
    );
    let guarded = CompiledExpr::try_from_ops(vec![
        CompiledOp::LoadVar(0),
        CompiledOp::LoadConst(0.0),
        CompiledOp::NotEqual,
        CompiledOp::JumpIfFalse(7),
        CompiledOp::LoadVar(0),
        CompiledOp::Csch,
        CompiledOp::Jump(8),
        CompiledOp::LoadConst(7.0),
    ])
    .unwrap();
    let guarded_batch = CompiledBatch::try_from_programs(vec![guarded.clone(); 3]).unwrap();
    assert_eq!(guarded_batch.try_eval_finite(&[0.0]).unwrap(), vec![7.0; 3]);
    assert_eq!(
        guarded_batch.try_eval_finite(&[1.0]).unwrap(),
        vec![guarded.try_eval_finite(&[1.0]).unwrap(); 3]
    );
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
fn reciprocal_work_keeps_cumulative_limits_and_every_cancellation_checkpoint() {
    for name in NAMES {
        let program = scalar(name);
        let limits = BudgetLimits::uniform(1_000_000, 37);
        let mut counted = InterruptAt {
            budget: Budget::new(limits),
            checkpoints: 0,
            interrupt: None,
        };
        program
            .try_eval_finite_with_meter(&[1.0], &mut counted)
            .unwrap();
        for interrupt in 1..=counted.checkpoints {
            let mut meter = InterruptAt {
                budget: Budget::new(limits),
                checkpoints: 0,
                interrupt: Some(interrupt),
            };
            assert_eq!(
                program.try_eval_finite_with_meter(&[1.0], &mut meter),
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
                .try_eval_finite_with_meter(&[1.0], &mut exact)
                .unwrap();
            assert_eq!(exact.remaining(dimension), 0);
            assert!(matches!(
                program.try_eval_finite_with_meter(&[1.0], &mut exact),
                Err(EvalError::Meter(MeterError::Budget(_)))
            ));
            assert_eq!(exact.verifier_remaining(), 37);
        }
    }
}
