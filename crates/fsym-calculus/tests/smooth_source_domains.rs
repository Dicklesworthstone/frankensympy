//! Common smooth-domain obligations survive zero derivatives and shared reuse.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{
    CompileError, CompiledExpr, CompiledOp, CompiledResidualSystem, EvalError,
};
use fsym_calculus::program_artifact::{
    admit_residual_candidate, decode_residual_candidate, encode_residual_candidate,
};
use fsym_calculus::proof::{
    classify_diff_rule, make_diff_term, verified_diff, verify_diff_derivation, verify_diff_receipt,
};
use fsym_calculus::sparse_hessian::{ObjectiveEvaluationError, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, SparseJacobianEntry,
    verify_sparse_jacobian_certificate,
};
use fsym_calculus::verified::{ResidualEvaluationError, VerifiedResidualSystem};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_proof_kernel::{Claim, ProofRule};
use std::collections::HashMap;
use std::sync::Arc;

fn function(name: &str, value: Expr) -> Expr {
    Expr::Function(name.into(), vec![value])
}

fn power(base: Expr, numerator: i64, denominator: i64) -> Expr {
    Expr::Pow(
        Arc::new(base),
        Arc::new(Expr::Rational(BigRational::new(
            numerator.into(),
            denominator.into(),
        ))),
    )
}

fn cases() -> Vec<(&'static str, Expr, f64, f64)> {
    let x = Expr::symbol("x");
    vec![
        ("sqrt", function("sqrt", x.clone()), 0.0, 4.0),
        ("sqrt", function("sqrt", x.clone()), -0.0, 4.0),
        ("asin", function("asin", x.clone()), -1.0, 0.0),
        ("asin", function("asin", x.clone()), 1.0, 0.0),
        ("acos", function("acos", x.clone()), -1.0, 0.0),
        ("acos", function("acos", x.clone()), 1.0, 0.0),
        ("acosh", function("acosh", x.clone()), 1.0, 2.0),
        ("abs", function("abs", x.clone()), 0.0, -2.0),
        ("abs", function("abs", x.clone()), -0.0, -2.0),
        ("power", power(x.clone(), 1, 2), 0.0, 4.0),
        ("power", power(x.clone(), 1, 2), -0.0, 4.0),
        // Finite low-order derivatives cannot extend the original real branch.
        ("power", power(x, 5, 2), 0.0, 4.0),
    ]
}

fn compile(source: &Expr) -> CompiledExpr {
    CompiledExpr::try_compile(source, &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
}

fn residual_error<T: std::fmt::Debug>(
    result: Result<T, ResidualEvaluationError>,
    name: &'static str,
) {
    assert!(
        matches!(
            result,
            Err(ResidualEvaluationError::Evaluation(EvalError::NonsmoothSource(actual)))
                if actual == name
        ),
        "{name}: {result:?}"
    );
}

fn objective_error<T: std::fmt::Debug>(
    result: Result<T, ObjectiveEvaluationError>,
    name: &'static str,
) {
    assert!(
        matches!(
            result,
            Err(ObjectiveEvaluationError::Evaluation(EvalError::NonsmoothSource(actual)))
                if actual == name
        ),
        "{name}: {result:?}"
    );
}

#[test]
fn scalar_values_keep_closed_endpoints_while_derivative_bundles_require_open_domains() {
    for (name, source, endpoint, interior) in cases() {
        let scalar = compile(&source);
        assert!(scalar.try_eval_finite(&[endpoint]).unwrap().is_finite());
        assert!(scalar.try_eval_finite(&[interior]).unwrap().is_finite());
        let system = VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("x")]).unwrap();
        residual_error(system.evaluate_sparse(&[endpoint]), name);
        assert!(system.evaluate_sparse(&[interior]).is_ok(), "{name}");
    }
}

#[test]
fn zero_products_powers_and_vectors_cannot_publish_erased_source_boundaries() {
    for (name, primitive, endpoint, interior) in cases() {
        for hidden in [
            Expr::Mul(vec![Expr::from_i64(0), primitive.clone()]),
            Expr::Pow(Arc::new(primitive), Arc::new(Expr::from_i64(0))),
        ] {
            // A sum uses simplified child derivatives, so the singular
            // derivative disappears while the held original source survives.
            let source = Expr::Add(vec![Expr::from_i64(7), hidden]);
            let vars = [Symbol::new("x")];
            let scalar_value = compile(&source).try_eval_finite(&[endpoint]).unwrap();
            let system =
                VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &vars).unwrap();
            assert!(
                system
                    .evaluate_sparse(&[interior])
                    .unwrap()
                    .jacobian
                    .iter()
                    .all(|value| *value == 0.0)
            );
            for program in [
                CompiledResidualSystem::try_compile(std::slice::from_ref(&source), &vars).unwrap(),
                system.try_dense_program().unwrap(),
            ] {
                let mut values = [71.0];
                program
                    .try_eval_residuals(&[endpoint], &mut values)
                    .unwrap();
                assert_eq!(values, [scalar_value]);
                let mut derivatives = [83.0];
                assert_eq!(
                    program.try_eval_jacobian(&[endpoint], &mut derivatives),
                    Err(EvalError::NonsmoothSource(name))
                );
                assert_eq!(derivatives, [83.0]);
                values = [71.0];
                assert_eq!(
                    program.try_eval_system(&[endpoint], &mut values, &mut derivatives),
                    Err(EvalError::NonsmoothSource(name))
                );
                assert_eq!(values, [71.0]);
                assert_eq!(derivatives, [83.0]);
            }
            let mut residuals = [71.0];
            let mut dense = [83.0];
            residual_error(
                system.try_eval_dense_into(&[endpoint], &mut residuals, &mut dense),
                name,
            );
            assert_eq!(residuals, [71.0]);
            assert_eq!(dense, [83.0]);
            let mut sparse = vec![89.0; system.jacobian_pattern().len()];
            residual_error(
                system.try_eval_sparse_into(&[endpoint], &mut residuals, &mut sparse),
                name,
            );
            assert_eq!(residuals, [71.0]);
            assert!(sparse.iter().all(|value| *value == 89.0));
            let mut product = [97.0];
            residual_error(
                system.try_jacobian_vector_product_into(&[endpoint], &[0.0], &mut product),
                name,
            );
            assert_eq!(product, [97.0]);
            residual_error(
                system.try_jacobian_transpose_vector_product_into(
                    &[endpoint],
                    &[0.0],
                    &mut product,
                ),
                name,
            );
            assert_eq!(product, [97.0]);

            // Nonempty gradient and Hessian buffers must both remain atomic.
            let objective_source = Expr::Add(vec![power(Expr::symbol("x"), 2, 1), source.clone()]);
            let objective = VerifiedObjective::try_compile(&objective_source, &vars).unwrap();
            let mut gradient = [101.0];
            let mut hessian = vec![103.0; objective.hessian_pattern().len()];
            assert_eq!(hessian.len(), 1);
            objective_error(
                objective.try_eval_into(&[endpoint], &mut gradient, &mut hessian),
                name,
            );
            assert_eq!(gradient, [101.0]);
            assert_eq!(hessian, [103.0]);
            objective_error(
                objective.try_hessian_vector_product_into(&[endpoint], &[0.0], &mut product),
                name,
            );
            assert_eq!(product, [97.0]);
            assert!(system.evaluate_sparse(&[interior]).is_ok());
            assert!(objective.evaluate(&[interior]).is_ok());
        }
    }
}

#[test]
fn shared_source_memo_never_promotes_scalar_endpoints_or_leaks_between_calls() {
    for (name, primitive, endpoint, interior) in cases() {
        let hidden = Expr::Add(vec![
            Expr::from_i64(7),
            Expr::Mul(vec![Expr::from_i64(0), primitive]),
        ]);
        let system = VerifiedResidualSystem::try_compile(
            &[Expr::symbol("x"), hidden.clone(), hidden],
            &[Symbol::new("x")],
        )
        .unwrap();
        assert!(system.shared_subexpression_count() > 0);
        let valid = system.try_eval_dense(&[interior]).unwrap();
        for _ in 0..2 {
            // Scalar evaluation cannot populate a derivative bundle's fresh memo.
            for program in system.residual_programs() {
                program.try_eval_finite(&[endpoint]).unwrap();
            }
            let mut values = [71.0; 3];
            let mut derivatives = [83.0; 3];
            residual_error(
                system.try_eval_dense_into(&[endpoint], &mut values, &mut derivatives),
                name,
            );
            assert_eq!(values, [71.0; 3]);
            assert_eq!(derivatives, [83.0; 3]);
            assert_eq!(system.try_eval_dense(&[interior]).unwrap(), valid);
        }
    }
}

#[test]
fn inactive_piecewise_boundaries_are_not_hoisted_into_derivative_evaluation() {
    for (_, primitive, endpoint, _) in cases() {
        let piecewise = Expr::Function(
            "Piecewise".into(),
            vec![
                Expr::Function(
                    "Tuple".into(),
                    vec![
                        primitive,
                        Expr::Function(
                            "StrictGreaterThan".into(),
                            vec![Expr::symbol("x"), Expr::from_i64(4)],
                        ),
                    ],
                ),
                Expr::Function(
                    "Tuple".into(),
                    vec![power(Expr::symbol("x"), 2, 1), Expr::symbol("True")],
                ),
            ],
        );
        // Bare Piecewise differentiation is not admitted by this rule table.
        // A zero-weight child has a checked zero derivative; its original
        // evaluation must still preserve branch-local source obligations.
        let source = Expr::Add(vec![
            power(Expr::symbol("x"), 2, 1),
            Expr::Mul(vec![Expr::from_i64(0), piecewise]),
        ]);
        let vars = [Symbol::new("x")];
        let system =
            VerifiedResidualSystem::try_compile(&[source.clone(), source.clone()], &vars).unwrap();
        assert!(system.shared_subexpression_count() > 0);
        let evaluated = system.try_eval_dense(&[endpoint]).unwrap();
        assert_eq!(evaluated.residuals, [endpoint * endpoint; 2]);
        assert_eq!(evaluated.jacobian, [2.0 * endpoint; 2]);
        let objective = VerifiedObjective::try_compile(&source, &vars).unwrap();
        let evaluated = objective.evaluate(&[endpoint]).unwrap();
        assert_eq!(evaluated.value, endpoint * endpoint);
        assert_eq!(evaluated.gradient, [2.0 * endpoint]);
        assert_eq!(evaluated.hessian, [2.0]);
    }
}

#[test]
fn integer_powers_remain_smooth_at_zero_and_on_negative_bases() {
    for exponent in [0, 1, 2, 3, 4] {
        for source in [
            Expr::Pow(
                Arc::new(Expr::symbol("x")),
                Arc::new(Expr::from_i64(exponent)),
            ),
            power(Expr::symbol("x"), exponent, 1),
        ] {
            let vars = [Symbol::new("x"), Symbol::new("unused")];
            let residual =
                VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &vars).unwrap();
            let objective = VerifiedObjective::try_compile(&source, &vars).unwrap();
            for point in [0.0_f64, -0.0, -2.0, 2.0] {
                let expected = point.powi(exponent as i32);
                let first = if exponent == 0 {
                    0.0
                } else {
                    exponent as f64 * point.powi(exponent as i32 - 1)
                };
                let second = if exponent < 2 {
                    0.0
                } else {
                    (exponent * (exponent - 1)) as f64 * point.powi(exponent as i32 - 2)
                };
                let evaluated = residual.try_eval_dense(&[point, 3.0]).unwrap();
                assert_eq!(evaluated.residuals, [expected], "{source} at {point}");
                assert_eq!(evaluated.jacobian, [first, 0.0], "{source} at {point}");
                let evaluated = objective.evaluate(&[point, 3.0]).unwrap();
                assert_eq!(evaluated.value, expected, "{source} at {point}");
                assert_eq!(evaluated.gradient, [first, 0.0], "{source} at {point}");
                assert_eq!(
                    objective
                        .hessian_vector_product(&[point, 3.0], &[1.0, 1.0])
                        .unwrap(),
                    [second, 0.0],
                    "{source} at {point}"
                );
            }
        }
    }
}

#[test]
fn literal_identity_derivatives_do_not_erase_original_base_domains() {
    let x = Expr::symbol("x");
    for exponent in [0, 1] {
        for literal in [
            Expr::from_i64(exponent),
            Expr::Rational(BigRational::from_integer(exponent.into())),
        ] {
            for (base, expected) in [
                (
                    Expr::Pow(Arc::new(x.clone()), Arc::new(Expr::from_i64(-1))),
                    EvalError::NonFiniteValue,
                ),
                (
                    function("sqrt", x.clone()),
                    EvalError::NonsmoothSource("sqrt"),
                ),
            ] {
                let source = Expr::Pow(Arc::new(base), Arc::new(literal.clone()));
                let variables = [Symbol::new("x")];
                let residual =
                    VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables)
                        .unwrap();
                let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
                assert!(residual.evaluate_sparse(&[2.0]).is_ok());
                assert!(objective.evaluate(&[2.0]).is_ok());
                assert_eq!(
                    residual.evaluate_sparse(&[0.0]),
                    Err(ResidualEvaluationError::Evaluation(expected.clone()))
                );
                assert_eq!(
                    objective.evaluate(&[0.0]),
                    Err(ObjectiveEvaluationError::Evaluation(expected))
                );
            }
        }
    }
    // The exponent is not a literal even when its runtime value is zero.
    let dynamic = Expr::Pow(Arc::new(x), Arc::new(Expr::symbol("p")));
    let system =
        VerifiedResidualSystem::try_compile(&[dynamic], &[Symbol::new("x"), Symbol::new("p")])
            .unwrap();
    assert_eq!(
        system.evaluate_sparse(&[0.0, 0.0]),
        Err(ResidualEvaluationError::Evaluation(
            EvalError::NonPositivePowerBase
        ))
    );
}

#[test]
fn legacy_literal_power_proofs_and_artifacts_replay_but_coordinated_mutants_fail() {
    let variables = [Symbol::new("x"), Symbol::new("unused")];
    let variable_map = HashMap::from([(variables[0].clone(), 0), (variables[1].clone(), 1)]);
    for exponent in [0, 1] {
        for source in [
            Expr::Pow(
                Arc::new(Expr::symbol("x")),
                Arc::new(Expr::from_i64(exponent)),
            ),
            power(Expr::symbol("x"), exponent, 1),
        ] {
            // The public symbolic lane still emits exactly the old raw
            // power-rule tree, including the zero or one leading factor.
            let (legacy_derivative, tree) = verified_diff(&source, &variables[0]);
            let Expr::Mul(factors) = &legacy_derivative else {
                panic!("public raw power-rule representation changed")
            };
            assert_eq!(factors.len(), 3);
            assert!(
                verify_diff_derivation(&tree, &source, &variables[0], &legacy_derivative).is_ok()
            );
            let rendered = legacy_derivative.to_string();
            verify_diff_receipt(
                &source,
                &variables[0],
                classify_diff_rule(&source, &variables[0]),
                &rendered,
                &rendered,
            )
            .unwrap();

            let mut candidate =
                SparseJacobian::try_build(std::slice::from_ref(&source), &variables).unwrap();
            candidate.entries = vec![SparseJacobianEntry {
                row: 0,
                col: 0,
                symbolic_deriv: legacy_derivative.clone(),
                compiled_deriv: CompiledExpr::try_compile(&legacy_derivative, &variable_map)
                    .unwrap(),
                derivation: tree,
            }];
            candidate.pattern.nonzeros = vec![(0, 0)];
            candidate.pattern.nnz = 1;
            assert_eq!(
                verify_sparse_jacobian_certificate(&candidate, &[])
                    .unwrap()
                    .proofs_verified,
                1
            );
            let admitted = VerifiedResidualSystem::try_from_sparse_jacobian(&candidate).unwrap();
            let restored =
                VerifiedResidualSystem::from_json_bytes(&admitted.to_json_bytes().unwrap())
                    .unwrap();
            assert_eq!(restored.sparse_jacobian(), &candidate);
            assert_eq!(
                restored.try_eval_dense(&[2.0, 3.0]).unwrap().jacobian,
                [exponent as f64, 0.0]
            );
            let canonical = encode_residual_candidate(&admitted, &mut Unbounded).unwrap();
            let transported = admit_residual_candidate(
                std::slice::from_ref(&source),
                &variables,
                decode_residual_candidate(&canonical, &mut Unbounded).unwrap(),
                &mut Unbounded,
            )
            .unwrap();
            assert_eq!(transported.sparse_jacobian(), &candidate);

            let mut wrong_chain = factors.clone();
            wrong_chain[2] = Expr::from_i64(9);
            for forged in [
                Expr::from_i64(1 - exponent),
                Expr::Mul(wrong_chain),
                Expr::Mul(vec![Expr::from_i64(2), legacy_derivative.clone()]),
            ] {
                // The claim, rule RHS and executable bytecode all agree with
                // the mutation. Only independent rule semantics can reject it.
                let mut mutant = candidate.clone();
                let entry = &mut mutant.entries[0];
                entry.symbolic_deriv = forged.clone();
                entry.compiled_deriv = CompiledExpr::try_compile(&forged, &variable_map).unwrap();
                let lhs = make_diff_term(&source, &variables[0]);
                entry.derivation.steps[0].claim = Claim::equality(lhs.clone(), forged.clone());
                entry.derivation.steps[0].rule = ProofRule::DefinitionalReduction {
                    lhs,
                    rhs: forged,
                    rule_name: classify_diff_rule(&source, &variables[0]).into(),
                };
                assert!(matches!(
                    verify_sparse_jacobian_certificate(&mutant, &[]),
                    Err(JacobianVerificationError::ProofFailed { .. })
                ));
            }

            // A matching literal exponent cannot grant the derivative for a
            // different axis, even when every serialized claim is rewritten.
            let mut wrong_axis = candidate.clone();
            wrong_axis.pattern.nonzeros[0] = (0, 1);
            let entry = &mut wrong_axis.entries[0];
            entry.col = 1;
            let lhs = make_diff_term(&source, &variables[1]);
            entry.derivation.steps[0].claim =
                Claim::equality(lhs.clone(), legacy_derivative.clone());
            entry.derivation.steps[0].rule = ProofRule::DefinitionalReduction {
                lhs,
                rhs: legacy_derivative,
                rule_name: classify_diff_rule(&source, &variables[1]).into(),
            };
            assert!(matches!(
                verify_sparse_jacobian_certificate(&wrong_axis, &[]),
                Err(JacobianVerificationError::ProofFailed { .. })
            ));
        }
    }
}

#[test]
fn literal_exponent_bounds_preserve_nonintegral_source_status() {
    let maximum = 9_007_199_254_740_991_i64;
    for (numerator, denominator) in [
        (maximum, maximum - 1),
        (maximum - 1, maximum),
        (maximum, 2),
        (maximum, 3),
        (maximum, 4_503_599_627_370_495),
        (maximum, 4_503_599_627_370_496),
        (1, maximum),
    ] {
        for sign in [-1, 1] {
            let source = power(Expr::symbol("x"), sign * numerator, denominator);
            let scalar = compile(&source);
            let CompiledOp::Pow(exponent) = scalar.ops().last().unwrap() else {
                panic!("literal power")
            };
            assert_ne!(exponent.fract(), 0.0);
            let held = Expr::Mul(vec![Expr::from_i64(0), source]);
            // This legacy adapter isolates source admission from stricter
            // proof-arithmetic bounds in full derivative generation.
            let system = CompiledResidualSystem::try_compile(&[held], &[Symbol::new("x")]).unwrap();
            assert_eq!(
                system.try_eval_jacobian(&[0.0], &mut [71.0]),
                Err(EvalError::NonsmoothSource("power"))
            );
        }
    }
    for exponent in [
        BigRational::new(BigInt::from(9_007_199_254_740_993_i64), BigInt::from(1)),
        BigRational::new(
            BigInt::from(1_152_921_504_606_846_977_i64),
            BigInt::from(1_152_921_504_606_846_976_i64),
        ),
    ] {
        let source = Expr::Pow(
            Arc::new(Expr::symbol("x")),
            Arc::new(Expr::Rational(exponent)),
        );
        assert!(matches!(
            CompiledExpr::try_compile(&source, &HashMap::from([(Symbol::new("x"), 0)])),
            Err(CompileError::UnsupportedExponent(_))
        ));
    }
}

struct InterruptAt {
    budget: Budget,
    checkpoints: usize,
    stop: Option<usize>,
}

impl InterruptAt {
    fn new() -> Self {
        Self {
            budget: Budget::new(BudgetLimits::uniform(1_000_000, 37)),
            checkpoints: 0,
            stop: None,
        }
    }
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
        if self.stop == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn rejected_boundary_work_keeps_one_ledger_and_cancellation_remains_terminal() {
    let source = Expr::Add(vec![
        Expr::from_i64(7),
        Expr::Mul(vec![Expr::from_i64(0), function("sqrt", Expr::symbol("x"))]),
    ]);
    let system = VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("x")]).unwrap();
    let mut measured = InterruptAt::new();
    let mut values = [71.0];
    let mut derivatives = vec![83.0; system.jacobian_pattern().len()];
    residual_error(
        system.try_eval_sparse_into_with_meter(
            &[0.0],
            &mut values,
            &mut derivatives,
            &mut measured,
        ),
        "sqrt",
    );
    assert_eq!(values, [71.0]);
    let checkpoints = measured.checkpoints;
    let first = measured.budget.snapshot();
    residual_error(
        system.try_eval_sparse_into_with_meter(
            &[0.0],
            &mut values,
            &mut derivatives,
            &mut measured,
        ),
        "sqrt",
    );
    assert_eq!(values, [71.0]);
    assert!(derivatives.iter().all(|value| *value == 83.0));
    let second = measured.budget.snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(first.dimensions[dimension.index()] < 1_000_000);
        assert_eq!(
            1_000_000 - second.dimensions[dimension.index()],
            2 * (1_000_000 - first.dimensions[dimension.index()])
        );
        let mut limits = BudgetLimits::uniform(1_000_000, 37);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(
            system.evaluate_sparse_with_meter(&[0.0], &mut Budget::new(limits)),
            Err(ResidualEvaluationError::Evaluation(EvalError::Meter(_)))
        ));
    }
    assert_eq!(first.verifier_pool, 37);
    assert_eq!(second.verifier_pool, 37);
    for stop in 1..=checkpoints {
        let mut meter = InterruptAt::new();
        meter.stop = Some(stop);
        let mut values = [71.0];
        let mut derivatives = vec![83.0; system.jacobian_pattern().len()];
        assert!(matches!(
            system.try_eval_sparse_into_with_meter(
                &[0.0],
                &mut values,
                &mut derivatives,
                &mut meter,
            ),
            Err(ResidualEvaluationError::Evaluation(EvalError::Meter(
                MeterError::Cancelled
            )))
        ));
        assert_eq!(values, [71.0]);
        assert!(derivatives.iter().all(|value| *value == 83.0));
        assert_eq!(meter.budget.verifier_remaining(), 37);
    }
}
