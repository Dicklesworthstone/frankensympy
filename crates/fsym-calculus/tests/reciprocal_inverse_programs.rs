//! Principal reciprocal inverses through checked residuals and objectives.
//! Exact rational anchors and asymptotic tails are independent of the jets.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, CompiledResidualSystem, EvalError};
use fsym_calculus::program_artifact::{
    decode_objective_candidate, decode_residual_candidate, encode_objective_candidate,
    encode_residual_candidate,
};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{
    SparseJacobian, SparseJacobianEntry, compiled_expression_matches,
    verify_sparse_jacobian_certificate,
};
use fsym_calculus::verified::VerifiedResidualSystem;
use fsym_core::{Expr, Symbol, parse};
use fsym_proof_kernel::{Claim, ProofRule};
use std::collections::HashMap;
use std::sync::Arc;

const NAMES: [&str; 6] = ["acot", "acoth", "asec", "acsc", "asech", "acsch"];

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn jet(name: &str, order: usize, argument: Expr) -> Expr {
    function(&format!("_fsym_{name}_d{order}"), argument)
}

fn compile(source: &Expr) -> CompiledExpr {
    CompiledExpr::try_compile(source, &HashMap::from([(Symbol::new("x"), 0)])).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "nonfinite result for {expected:e}");
    if expected != 0.0 {
        assert_ne!(actual, 0.0, "erased representable value {expected:e}");
        assert_eq!(actual.is_sign_positive(), expected.is_sign_positive());
    }
    assert!(
        (actual - expected).abs() <= 4e-13 * expected.abs() + 4.0 * f64::from_bits(1),
        "actual={actual:e}, expected={expected:e}"
    );
}

// Rational arguments make every slope and curvature below an exact rational.
// The values reduce to elementary angles or log(2), without private jets.
fn anchors() -> [(&'static str, f64, f64, f64, f64); 6] {
    [
        (
            "acot",
            4.0 / 3.0,
            0.75f64.atan(),
            -9.0 / 25.0,
            216.0 / 625.0,
        ),
        ("acoth", 5.0 / 3.0, 2.0f64.ln(), -9.0 / 16.0, 135.0 / 128.0),
        (
            "asec",
            5.0 / 4.0,
            0.8f64.acos(),
            16.0 / 15.0,
            -2176.0 / 675.0,
        ),
        (
            "acsc",
            5.0 / 3.0,
            0.6f64.asin(),
            -9.0 / 20.0,
            1107.0 / 1600.0,
        ),
        (
            "asech",
            4.0 / 5.0,
            2.0f64.ln(),
            -25.0 / 12.0,
            -875.0 / 432.0,
        ),
        (
            "acsch",
            4.0 / 3.0,
            2.0f64.ln(),
            -9.0 / 20.0,
            1107.0 / 2000.0,
        ),
    ]
}

#[test]
fn exact_anchors_reach_values_sparse_dense_derivatives_and_both_real_sides() {
    let variables = [Symbol::new("x")];
    for (name, argument, value, slope, curvature) in anchors() {
        let source = function(name, Expr::symbol("x"));
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
        let dense = CompiledResidualSystem::try_compile(&[source], &variables).unwrap();
        for sign in [1.0, -1.0] {
            if name == "asech" && sign < 0.0 {
                continue;
            }
            let point = [sign * argument];
            let expected_value = if name == "asec" && sign < 0.0 {
                std::f64::consts::PI - value
            } else {
                sign * value
            };
            let result = objective.evaluate(&point).unwrap();
            close(result.value, expected_value);
            close(result.gradient[0], slope);
            close(result.hessian[0], sign * curvature);
            let result = residual.evaluate_sparse(&point).unwrap();
            close(result.residuals[0], expected_value);
            close(result.jacobian[0], slope);
            let mut derivative = [0.0];
            dense.try_eval_jacobian(&point, &mut derivative).unwrap();
            close(derivative[0], slope);
        }
    }
}

#[test]
fn nonlinear_chain_preserves_mixed_curvature_and_matrix_products() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    for (name, argument, value, slope, curvature) in anchors() {
        let source = function(name, parse("x^2+y").unwrap());
        let point = [0.5, argument - 0.25];
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
        let result = residual.evaluate_sparse(&point).unwrap();
        close(result.residuals[0], value);
        for actual in result.jacobian {
            close(actual, slope);
        }
        close(
            residual
                .jacobian_vector_product(&point, &[2.0, 3.0])
                .unwrap()[0],
            5.0 * slope,
        );
        for actual in residual
            .jacobian_transpose_vector_product(&point, &[7.0])
            .unwrap()
        {
            close(actual, 7.0 * slope);
        }
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let result = objective.evaluate(&point).unwrap();
        assert_eq!(
            objective.hessian_pattern(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        for (actual, expected) in
            result
                .hessian
                .iter()
                .zip([curvature + 2.0 * slope, curvature, curvature, curvature])
        {
            close(*actual, expected);
        }
        let product = objective
            .hessian_vector_product(&point, &[2.0, 3.0])
            .unwrap();
        close(product[0], 5.0 * curvature + 4.0 * slope);
        close(product[1], 5.0 * curvature);
    }
}

#[test]
fn every_value_and_jet_has_distinct_replayable_bytecode_and_checked_arity() {
    let variables = HashMap::from([(Symbol::new("x"), 0)]);
    let operations = [
        [
            CompiledOp::Acot,
            CompiledOp::AcotDerivative,
            CompiledOp::AcotSecondDerivative,
        ],
        [
            CompiledOp::Acoth,
            CompiledOp::AcothDerivative,
            CompiledOp::AcothSecondDerivative,
        ],
        [
            CompiledOp::Asec,
            CompiledOp::AsecDerivative,
            CompiledOp::AsecSecondDerivative,
        ],
        [
            CompiledOp::Acsc,
            CompiledOp::AcscDerivative,
            CompiledOp::AcscSecondDerivative,
        ],
        [
            CompiledOp::Asech,
            CompiledOp::AsechDerivative,
            CompiledOp::AsechSecondDerivative,
        ],
        [
            CompiledOp::Acsch,
            CompiledOp::AcschDerivative,
            CompiledOp::AcschSecondDerivative,
        ],
    ];
    for (name, row) in NAMES.iter().zip(&operations) {
        for (order, operation) in row.iter().enumerate() {
            let name = if order == 0 {
                name.to_string()
            } else {
                format!("_fsym_{name}_d{order}")
            };
            let source = function(&name, Expr::symbol("x"));
            let program = compile(&source);
            assert_eq!(program.ops(), &[CompiledOp::LoadVar(0), operation.clone()]);
            assert!(compiled_expression_matches(&source, &variables, &program).unwrap());
            let encoded = serde_json::to_vec(&program).unwrap();
            let restored: CompiledExpr = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(restored.ops(), program.ops());
            assert!(CompiledExpr::try_from_ops(vec![operation.clone()]).is_err());
            for wrong in operations
                .iter()
                .flatten()
                .filter(|other| *other != operation)
            {
                let forged =
                    CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), wrong.clone()])
                        .unwrap();
                assert!(!compiled_expression_matches(&source, &variables, &forged).unwrap());
            }
            for arguments in [vec![], vec![Expr::symbol("x"), Expr::from_i64(1)]] {
                let malformed = Expr::Function(name.clone(), arguments);
                assert!(CompiledExpr::try_compile(&malformed, &variables).is_err());
                assert!(compiled_expression_matches(&malformed, &variables, &program).is_err());
            }
        }
    }
}

fn replace_entry(entry: &mut SparseJacobianEntry, replacement: Expr) {
    entry.compiled_deriv = compile(&replacement);
    entry.symbolic_deriv = replacement.clone();
    let step = entry.derivation.steps.last_mut().unwrap();
    let ProofRule::DefinitionalReduction { lhs, rhs, .. } = &mut step.rule else {
        panic!("expected a differentiation proof");
    };
    *rhs = replacement.clone();
    step.claim = Claim::equality(lhs.clone(), replacement);
}

#[test]
fn independent_proofs_reject_self_consistent_kind_order_sign_and_chain_mutants() {
    let variables = [Symbol::new("x")];
    for name in NAMES {
        let source = function(name, parse("3*x").unwrap());
        let candidate =
            SparseJacobian::try_build(std::slice::from_ref(&source), &variables).unwrap();
        verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
        let derivative = candidate.entries[0].symbolic_deriv.clone();
        for wrong in [
            Expr::from_i64(0),
            -derivative.clone(),
            derivative * Expr::rational(1, 3).unwrap(),
            jet(
                if name == "acot" { "acoth" } else { "acot" },
                1,
                parse("3*x").unwrap(),
            ),
            jet(name, 2, parse("3*x").unwrap()),
            jet(name, 1, parse("3*x+1").unwrap()) * Expr::from_i64(3),
        ] {
            let mut forged = candidate.clone();
            replace_entry(&mut forged.entries[0], wrong);
            assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
        }
        // Literal identity powers expose the first jet as its own Hessian
        // source, exercising the separate second-derivative proof rule.
        let source = function(name, Expr::symbol("x")).pow(Expr::from_i64(1));
        let candidate = ObjectiveCandidate::try_build(&source, &variables).unwrap();
        assert!(matches!(
            &candidate.hessian.residuals[0],
            Expr::Function(function, arguments)
                if function == &format!("_fsym_{name}_d1") && arguments.len() == 1
        ));
        VerifiedObjective::try_from_candidate(&source, &variables, candidate.clone()).unwrap();
        for wrong in [
            Expr::from_i64(0),
            jet(name, 1, Expr::symbol("x")),
            -candidate.hessian.entries[0].symbolic_deriv.clone(),
        ] {
            let mut forged = candidate.clone();
            replace_entry(&mut forged.hessian.entries[0], wrong);
            assert!(VerifiedObjective::try_from_candidate(&source, &variables, forged).is_err());
        }
        for second in [false, true] {
            let mut forged = candidate.clone();
            let stage = if second {
                &mut forged.hessian
            } else {
                &mut forged.gradient
            };
            stage.entries.clear();
            stage.pattern.nonzeros.clear();
            stage.pattern.nnz = 0;
            assert!(VerifiedObjective::try_from_candidate(&source, &variables, forged).is_err());
        }
    }
}

#[test]
fn artifact_roundtrips_reverify_the_original_functions_and_their_curvatures() {
    let variables = [Symbol::new("x")];
    for (name, point, _, slope, curvature) in anchors() {
        let source = function(name, Expr::symbol("x"));
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
        let encoded = encode_residual_candidate(&residual, &mut Unbounded).unwrap();
        let candidate = decode_residual_candidate(&encoded, &mut Unbounded).unwrap();
        let restored = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
            candidate,
            &mut Unbounded,
        )
        .unwrap();
        close(
            restored.evaluate_sparse(&[point]).unwrap().jacobian[0],
            slope,
        );
        assert_eq!(
            encode_residual_candidate(&restored, &mut Unbounded).unwrap(),
            encoded
        );

        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let encoded = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
        let candidate = decode_objective_candidate(&encoded, &mut Unbounded).unwrap();
        let restored =
            VerifiedObjective::try_from_candidate(&source, &variables, candidate).unwrap();
        close(restored.evaluate(&[point]).unwrap().hessian[0], curvature);
        assert_eq!(
            encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
            encoded
        );
    }
}

#[test]
fn closed_values_do_not_extend_smooth_domains_even_after_derivative_elision() {
    let variables = [Symbol::new("x")];
    for (name, endpoint, expected, interior) in [
        ("acot", 0.0, std::f64::consts::FRAC_PI_2, 2.0),
        ("acot", -0.0, std::f64::consts::FRAC_PI_2, -2.0),
        ("asec", 1.0, 0.0, 2.0),
        ("asec", -1.0, std::f64::consts::PI, -2.0),
        ("acsc", 1.0, std::f64::consts::FRAC_PI_2, 2.0),
        ("acsc", -1.0, -std::f64::consts::FRAC_PI_2, -2.0),
        ("asech", 1.0, 0.0, 0.5),
    ] {
        let primitive = function(name, Expr::symbol("x"));
        close(
            compile(&primitive).try_eval_finite(&[endpoint]).unwrap(),
            expected,
        );
        for order in [1, 2] {
            assert!(
                compile(&jet(name, order, Expr::symbol("x")))
                    .try_eval_finite(&[endpoint])
                    .is_err()
            );
        }
        for source in [
            primitive.clone(),
            Expr::Add(vec![
                Expr::from_i64(7),
                Expr::Mul(vec![Expr::from_i64(0), primitive.clone()]),
            ]),
            Expr::Pow(Arc::new(primitive), Arc::new(Expr::from_i64(0))),
        ] {
            let residual =
                VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables)
                    .unwrap();
            let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
            assert!(residual.evaluate_sparse(&[interior]).is_ok());
            assert!(objective.evaluate(&[interior]).is_ok());
            assert!(residual.evaluate_sparse(&[endpoint]).is_err());
            assert!(
                residual
                    .jacobian_vector_product(&[endpoint], &[0.0])
                    .is_err()
            );
            assert!(objective.evaluate(&[endpoint]).is_err());
            assert!(
                objective
                    .hessian_vector_product(&[endpoint], &[0.0])
                    .is_err()
            );
        }
    }
}

#[test]
fn real_domain_refusals_cannot_publish_partial_results() {
    let variables = [Symbol::new("x")];
    for (name, bad) in [
        ("acoth", 1.0),
        ("acoth", -1.0),
        ("acoth", 0.5),
        ("asec", 0.5),
        ("acsc", -0.5),
        ("asech", 0.0),
        ("asech", -0.5),
        ("asech", 1.1),
        ("acsch", 0.0),
        ("acsch", -0.0),
    ] {
        let source = function(name, Expr::symbol("x"));
        for expression in [
            source.clone(),
            jet(name, 1, Expr::symbol("x")),
            jet(name, 2, Expr::symbol("x")),
        ] {
            assert!(compile(&expression).try_eval_finite(&[bad]).is_err());
        }
        let residual = VerifiedResidualSystem::try_compile(
            &[parse("x+1").unwrap(), source.clone()],
            &variables,
        )
        .unwrap();
        let (mut values, mut derivatives) = ([71.0, 73.0], [83.0, 89.0]);
        assert!(
            residual
                .try_eval_sparse_into(&[bad], &mut values, &mut derivatives)
                .is_err()
        );
        assert_eq!(values, [71.0, 73.0]);
        assert_eq!(derivatives, [83.0, 89.0]);
        let hidden = Expr::Add(vec![
            Expr::from_i64(7),
            Expr::Mul(vec![Expr::from_i64(0), source]),
        ]);
        let objective = VerifiedObjective::try_compile(&hidden, &variables).unwrap();
        assert!(objective.evaluate(&[bad]).is_err());
        assert!(objective.hessian_vector_product(&[bad], &[0.0]).is_err());
    }
    for name in NAMES {
        let program = compile(&function(name, Expr::symbol("x")));
        for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(program.try_eval_finite(&[point]).is_err());
        }
    }
}

#[test]
fn scaled_jets_retain_subnormal_tails_and_finite_tiny_argument_values() {
    let variables = [Symbol::new("x")];
    // All omitted tail corrections are relative O(x^-2), below one ulp.
    for name in ["acot", "acoth", "asec", "acsc", "acsch"] {
        let objective =
            VerifiedObjective::try_compile(&function(name, Expr::symbol("x")), &variables).unwrap();
        for magnitude in [1e100, 1e107, 1e155, 1e160, f64::MAX] {
            for sign in [-1.0, 1.0] {
                let inverse = magnitude.recip();
                let positive = name == "asec";
                let slope = if positive {
                    inverse * inverse
                } else {
                    -inverse * inverse
                };
                let curvature =
                    ((if positive { -2.0 } else { 2.0 }) * sign * inverse) * inverse * inverse;
                let result = objective.evaluate(&[sign * magnitude]).unwrap();
                close(result.gradient[0], slope);
                close(result.hessian[0], curvature);
            }
        }
    }
    let tiny = f64::from_bits(1);
    for name in ["asech", "acsch"] {
        let value = compile(&function(name, Expr::symbol("x")));
        close(
            value.try_eval_finite(&[tiny]).unwrap(),
            std::f64::consts::LN_2 - tiny.ln(),
        );
        let objective =
            VerifiedObjective::try_compile(&function(name, Expr::symbol("x")), &variables).unwrap();
        let result = objective.evaluate(&[1e-100]).unwrap();
        close(result.gradient[0], -1e100);
        close(result.hessian[0], 1e200);
        // The value is representable here, but the true slope is not.
        assert!(
            compile(&jet(name, 1, Expr::symbol("x")))
                .try_eval_finite(&[tiny])
                .is_err()
        );
    }
    let acot =
        VerifiedObjective::try_compile(&function("acot", Expr::symbol("x")), &variables).unwrap();
    for point in [tiny, -tiny, 1e-200, -1e-200] {
        let result = acot.evaluate(&[point]).unwrap();
        close(result.value, std::f64::consts::FRAC_PI_2.copysign(point));
        close(result.gradient[0], -1.0);
        close(result.hessian[0], 2.0 * point);
    }
    // The first interior floats have independent leading endpoint series.
    // Their omitted relative corrections are O(epsilon), below the tolerance.
    // In particular, acosh(1/x) at next_down(1) loses a factor sqrt(2)
    // by rounding the reciprocal before applying the inverse hyperbolic.
    for name in ["acoth", "asec", "acsc", "asech"] {
        let bits = if name == "asech" {
            1.0f64.to_bits() - 1
        } else {
            1.0f64.to_bits() + 1
        };
        let point = f64::from_bits(bits);
        let distance = (point - 1.0).abs();
        let result = VerifiedObjective::try_compile(&function(name, Expr::symbol("x")), &variables)
            .unwrap()
            .evaluate(&[point])
            .unwrap();
        if name == "acoth" {
            close(result.value, 0.5 * (std::f64::consts::LN_2 - distance.ln()));
            close(result.gradient[0], -(2.0 * distance).recip());
            close(result.hessian[0], (2.0 * distance * distance).recip());
        } else {
            let edge = (2.0 * distance).sqrt();
            let value = if name == "acsc" {
                std::f64::consts::FRAC_PI_2 - edge
            } else {
                edge
            };
            let slope = if name == "asec" {
                edge.recip()
            } else {
                -edge.recip()
            };
            let curvature = (2.0 * distance * edge).recip();
            close(result.value, value);
            close(result.gradient[0], slope);
            close(
                result.hessian[0],
                if name == "acsc" {
                    curvature
                } else {
                    -curvature
                },
            );
        }
    }
}

#[derive(Default)]
struct Checkpoints {
    remaining: Option<usize>,
    reached: usize,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.reached += 1;
        if let Some(remaining) = self.remaining.as_mut() {
            if *remaining == 0 {
                return Err(MeterError::Cancelled);
            }
            *remaining -= 1;
        }
        Ok(())
    }
}

#[test]
fn kernels_share_caller_limits_and_cancellation_keeps_destinations_atomic() {
    for (name, point, _, _, _) in anchors() {
        for order in 0..=2 {
            let source = if order == 0 {
                function(name, Expr::symbol("x"))
            } else {
                jet(name, order, Expr::symbol("x"))
            };
            let program = compile(&source);
            let limits = BudgetLimits::uniform(u64::MAX, 37);
            let mut counted = Budget::new(limits);
            program
                .try_eval_finite_with_meter(&[point], &mut counted)
                .unwrap();
            let used = limits.dimensions[Dimension::ComputeSteps.index()]
                - counted.remaining(Dimension::ComputeSteps);
            assert!(used >= 32);
            let mut exact_limits = limits;
            exact_limits.dimensions[Dimension::ComputeSteps.index()] = used;
            let mut exact = Budget::new(exact_limits);
            program
                .try_eval_finite_with_meter(&[point], &mut exact)
                .unwrap();
            assert_eq!(exact.remaining(Dimension::ComputeSteps), 0);
            assert_eq!(exact.verifier_remaining(), 37);
            assert!(matches!(
                program.try_eval_finite_with_meter(&[point], &mut exact),
                Err(EvalError::Meter(MeterError::Budget(_)))
            ));
        }
        let objective =
            VerifiedObjective::try_compile(&function(name, Expr::symbol("x")), &[Symbol::new("x")])
                .unwrap();
        let mut counted = Checkpoints::default();
        objective
            .try_eval_into_with_meter(&[point], &mut [0.0], &mut [0.0], &mut counted)
            .unwrap();
        for remaining in [0, counted.reached / 2, counted.reached - 1] {
            let mut meter = Checkpoints {
                remaining: Some(remaining),
                reached: 0,
            };
            let (mut gradient, mut hessian) = ([13.0], [17.0]);
            assert!(
                objective
                    .try_eval_into_with_meter(&[point], &mut gradient, &mut hessian, &mut meter)
                    .is_err()
            );
            assert_eq!(gradient, [13.0]);
            assert_eq!(hessian, [17.0]);
        }
    }
}

#[test]
fn values_near_reciprocal_branch_edges_do_not_amplify_reciprocal_roundoff() {
    // These literals denote exact binary64 inputs, not exact decimal rationals.
    // Independent 110-digit rational-series references used sqrt(x*x-1),
    // alternating atan, log via an atanh series, and Machin's pi formula.
    // They never evaluated the private jets or these binary64 value kernels.
    let cases = [
        (
            1.00000001,
            [
                9.556913967794891,
                0.00014142135521831177,
                1.5706549054396783,
            ],
        ),
        (
            1.000000000001,
            [
                14.162039699929037,
                0.000001414276423180071,
                1.5707949125184735,
            ],
        ),
        (
            1.0001,
            [4.95176877564314, 0.014141546406083378, 1.5566547803888133],
        ),
    ];
    let variables = [Symbol::new("x")];
    for (column, name) in ["acoth", "asec", "acsc"].iter().enumerate() {
        let source = function(name, Expr::symbol("x"));
        let scalar = compile(&source);
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        for (magnitude, values) in cases {
            for sign in [-1.0, 1.0] {
                let argument = sign * magnitude;
                let expected = if *name == "asec" && sign < 0.0 {
                    std::f64::consts::PI - values[column]
                } else {
                    sign * values[column]
                };
                for actual in [
                    scalar.try_eval_finite(&[argument]).unwrap(),
                    residual.evaluate_sparse(&[argument]).unwrap().residuals[0],
                    objective.evaluate(&[argument]).unwrap().value,
                ] {
                    assert!(
                        (actual - expected).abs() <= 2e-15 * expected.abs(),
                        "{name}({argument:e}): actual={actual:e}, expected={expected:e}"
                    );
                }
            }
        }
    }
}
