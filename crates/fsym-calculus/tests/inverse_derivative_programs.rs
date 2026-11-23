//! Finite real inverse-function derivatives must survive their intermediates.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, CompiledOp, CompiledResidualSystem, EvalError};
use fsym_calculus::program_artifact::{
    admit_objective_candidate, decode_objective_candidate, encode_objective_candidate,
};
use fsym_calculus::proof::{make_diff_term, verified_diff, verify_diff_derivation};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::{
    SparseJacobian, compiled_expression_matches, verify_sparse_jacobian_certificate,
};
use fsym_core::{Expr, Symbol, parse};
use fsym_proof_kernel::{Claim, ProofRule};
use std::collections::HashMap;

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

// Use direct closed forms where their intermediates are well scaled, and
// their asymptotic expansions elsewhere. At |x| > 1e8, omitting the next
// relative correction (O(1/x^2)) costs less than one binary64 ulp. These
// references do not execute the numerical jets or finite differences.
fn reference(name: &str, point: f64) -> (f64, f64) {
    if point.abs() > 1e8 {
        let inverse = point.abs().recip();
        if name == "asinh" {
            (inverse, (-point.signum() * inverse) * inverse)
        } else {
            (
                inverse * inverse,
                ((-2.0 * point.signum() * inverse) * inverse) * inverse,
            )
        }
    } else {
        let square_plus_one = 1.0 + point * point;
        if name == "asinh" {
            (
                square_plus_one.sqrt().recip(),
                -point / square_plus_one.powf(1.5),
            )
        } else {
            (
                square_plus_one.recip(),
                -2.0 * point / square_plus_one.powi(2),
            )
        }
    }
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "nonfinite result for {expected:e}");
    if expected != 0.0 {
        assert_ne!(actual, 0.0, "erased representable derivative {expected:e}");
        assert_eq!(actual.is_sign_positive(), expected.is_sign_positive());
    }
    assert!(
        (actual - expected).abs() <= 4e-13 * expected.abs() + 2.0 * f64::from_bits(1),
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn inverse_jets_keep_finite_gradients_and_curvatures_through_the_binary64_range() {
    let variables = [Symbol::new("x")];
    let tiny = f64::from_bits(1);
    for name in ["atan", "asinh"] {
        let source = function(name, Expr::Sym(variables[0].clone()));
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
        let dense = CompiledResidualSystem::try_compile(&[source], &variables).unwrap();
        for point in [
            -f64::MAX,
            -1e200,
            -1e160,
            -1e155,
            -1e120,
            -1e107,
            -1e100,
            -2.0,
            -f64::from_bits(1.0f64.to_bits() + 1),
            -1.0,
            -f64::from_bits(1.0f64.to_bits() - 1),
            -0.5,
            -1e-200,
            -tiny,
            -0.0,
            0.0,
            tiny,
            1e-200,
            0.5,
            f64::from_bits(1.0f64.to_bits() - 1),
            1.0,
            f64::from_bits(1.0f64.to_bits() + 1),
            2.0,
            1e100,
            1e107,
            1e120,
            1e155,
            1e160,
            1e200,
            f64::MAX,
        ] {
            let (slope, curvature) = reference(name, point);
            let result = objective.evaluate(&[point]).unwrap();
            close(
                result.value,
                if name == "atan" {
                    point.atan()
                } else {
                    point.asinh()
                },
            );
            close(result.gradient[0], slope);
            close(result.hessian[0], curvature);
            close(
                residual.evaluate_sparse(&[point]).unwrap().jacobian[0],
                slope,
            );
            close(
                residual.jacobian_vector_product(&[point], &[2.0]).unwrap()[0],
                2.0 * slope,
            );
            close(
                residual
                    .jacobian_transpose_vector_product(&[point], &[3.0])
                    .unwrap()[0],
                3.0 * slope,
            );
            close(
                objective.hessian_vector_product(&[point], &[2.0]).unwrap()[0],
                2.0 * curvature,
            );
            let mut gradient = [0.0];
            dense.try_eval_jacobian(&[point], &mut gradient).unwrap();
            close(gradient[0], slope);
        }
    }
}

#[test]
fn inverse_jets_retain_mixed_and_inner_second_derivatives() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    for (name, magnitude) in [("atan", 1e100), ("asinh", 1e120)] {
        let source = function(name, parse("x^2+y").unwrap());
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        for sign in [-1.0, 1.0] {
            let point = [3.0, sign * magnitude];
            let (slope, curvature) = reference(name, point[0] * point[0] + point[1]);
            let result = objective.evaluate(&point).unwrap();
            close(result.gradient[0], 6.0 * slope);
            close(result.gradient[1], slope);
            for (actual, expected) in result.hessian.iter().zip([
                36.0 * curvature + 2.0 * slope,
                6.0 * curvature,
                6.0 * curvature,
                curvature,
            ]) {
                close(*actual, expected);
            }
            let product = objective
                .hessian_vector_product(&point, &[2.0, 3.0])
                .unwrap();
            close(product[0], 90.0 * curvature + 4.0 * slope);
            close(product[1], 15.0 * curvature);
        }
    }
}

#[test]
fn public_symbolic_receipts_remain_ordinary_inverse_function_rules() {
    let variable = Symbol::new("x");
    for name in ["atan", "asinh"] {
        let source = function(name, Expr::Sym(variable.clone()));
        let denominator = parse("1+x^2").unwrap();
        let exponent = if name == "atan" {
            Expr::from_i64(-1)
        } else {
            Expr::rational(-1, 2).unwrap()
        };
        let expected = Expr::Mul(vec![denominator.pow(exponent), Expr::from_i64(1)]);
        let (ordinary, proof) = verified_diff(&source, &variable);
        assert_eq!(ordinary, expected);
        verify_diff_derivation(&proof, &source, &variable, &ordinary).unwrap();
        VerifiedObjective::try_compile(&source, std::slice::from_ref(&variable)).unwrap();
        assert_eq!(verified_diff(&source, &variable), (ordinary, proof));
    }
}

#[test]
fn independent_proof_and_opcode_checks_reject_inverse_jet_mutants() {
    let variables = [Symbol::new("x")];
    for name in ["atan", "asinh"] {
        let source = function(name, parse("3*x").unwrap());
        let candidate =
            SparseJacobian::try_build(std::slice::from_ref(&source), &variables).unwrap();
        verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
        let derivative = candidate.entries[0].symbolic_deriv.clone();
        for wrong in [
            Expr::from_i64(0),
            -derivative.clone(),
            derivative.clone() * Expr::rational(1, 3).unwrap(),
            function(
                if name == "atan" {
                    "_fsym_asinh_d1"
                } else {
                    "_fsym_atan_d1"
                },
                parse("3*x").unwrap(),
            ),
        ] {
            let mut forged = candidate.clone();
            let entry = &mut forged.entries[0];
            entry.symbolic_deriv = wrong.clone();
            entry.derivation.steps[0].claim =
                Claim::equality(make_diff_term(&source, &variables[0]), wrong.clone());
            let ProofRule::DefinitionalReduction { rhs, .. } = &mut entry.derivation.steps[0].rule
            else {
                panic!("expected derivative rule")
            };
            *rhs = wrong.clone();
            entry.compiled_deriv =
                CompiledExpr::try_compile(&wrong, &HashMap::from([(variables[0].clone(), 0)]))
                    .unwrap();
            assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
        }
        let original = ObjectiveCandidate::try_build(&source, &variables).unwrap();
        for second in [false, true] {
            let mut forged = original.clone();
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

        // A literal identity power leaves a bare first jet as the Hessian's
        // input. Exercise its own independent chain rule, not only a parent
        // product rule that happens to reconstruct the same second jet.
        let source = function(name, Expr::symbol("x")).pow(Expr::from_i64(1));
        let original = ObjectiveCandidate::try_build(&source, &variables).unwrap();
        assert!(matches!(
            &original.hessian.residuals[0],
            Expr::Function(jet, arguments)
                if jet == &format!("_fsym_{name}_d1") && arguments.len() == 1
        ));
        let accepted =
            VerifiedObjective::try_from_candidate(&source, &variables, original.clone()).unwrap();
        close(
            accepted.evaluate(&[1e100]).unwrap().hessian[0],
            reference(name, 1e100).1,
        );
        let second = original.hessian.entries[0].symbolic_deriv.clone();
        for wrong in [
            Expr::from_i64(0),
            -second.clone(),
            second * Expr::from_i64(2),
        ] {
            let mut forged = original.clone();
            let hessian_source = &forged.hessian.residuals[0];
            let entry = &mut forged.hessian.entries[0];
            entry.symbolic_deriv = wrong.clone();
            entry.derivation.steps[0].claim =
                Claim::equality(make_diff_term(hessian_source, &variables[0]), wrong.clone());
            let ProofRule::DefinitionalReduction { rhs, .. } = &mut entry.derivation.steps[0].rule
            else {
                panic!("expected inverse curvature rule")
            };
            *rhs = wrong.clone();
            entry.compiled_deriv =
                CompiledExpr::try_compile(&wrong, &HashMap::from([(variables[0].clone(), 0)]))
                    .unwrap();
            assert!(VerifiedObjective::try_from_candidate(&source, &variables, forged).is_err());
        }
    }
    let mapping = HashMap::from([(variables[0].clone(), 0)]);
    for (name, operation, wrong) in [
        (
            "_fsym_atan_d1",
            CompiledOp::AtanDerivative,
            CompiledOp::AtanSecondDerivative,
        ),
        (
            "_fsym_atan_d2",
            CompiledOp::AtanSecondDerivative,
            CompiledOp::AsinhSecondDerivative,
        ),
        (
            "_fsym_asinh_d1",
            CompiledOp::AsinhDerivative,
            CompiledOp::AtanDerivative,
        ),
        (
            "_fsym_asinh_d2",
            CompiledOp::AsinhSecondDerivative,
            CompiledOp::AsinhDerivative,
        ),
    ] {
        assert_eq!(
            CompiledExpr::try_from_ops(vec![operation]),
            Err(EvalError::StackUnderflow)
        );
        let source = function(name, Expr::symbol("x"));
        let program = CompiledExpr::try_compile(&source, &mapping).unwrap();
        assert!(compiled_expression_matches(&source, &mapping, &program).unwrap());
        let forged = CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0), wrong]).unwrap();
        assert!(!compiled_expression_matches(&source, &mapping, &forged).unwrap());
    }
}

#[test]
fn scaled_inverse_programs_replay_without_changing_bytes_or_derivatives() {
    let variables = [Symbol::new("x")];
    for name in ["atan", "asinh"] {
        let source = function(name, Expr::symbol("x"));
        let residual =
            VerifiedResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
        let encoded = residual.to_json_bytes().unwrap();
        let restored = VerifiedResidualSystem::from_json_bytes(&encoded).unwrap();
        assert_eq!(restored.to_json_bytes().unwrap(), encoded);
        assert_eq!(
            restored.evaluate_sparse(&[1e120]).unwrap(),
            residual.evaluate_sparse(&[1e120]).unwrap()
        );
        let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let encoded = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
        let candidate = decode_objective_candidate(&encoded, &mut Unbounded).unwrap();
        let restored =
            admit_objective_candidate(&source, &variables, candidate, &mut Unbounded).unwrap();
        assert_eq!(
            encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
            encoded
        );
        assert_eq!(
            restored.evaluate(&[1e120]).unwrap(),
            objective.evaluate(&[1e120]).unwrap()
        );
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
fn scaled_inverse_evaluation_keeps_atomic_outputs_and_the_caller_meter() {
    let variables = [Symbol::new("x")];
    for name in ["atan", "asinh"] {
        let objective =
            VerifiedObjective::try_compile(&function(name, Expr::symbol("x")), &variables).unwrap();
        let mut counted = Checkpoints::default();
        objective
            .try_eval_into_with_meter(&[1e100], &mut [0.0], &mut [0.0], &mut counted)
            .unwrap();
        for remaining in [0, counted.reached / 2, counted.reached - 1] {
            let mut meter = Checkpoints {
                remaining: Some(remaining),
                reached: 0,
            };
            let (mut gradient, mut hessian) = ([13.0], [17.0]);
            assert!(
                objective
                    .try_eval_into_with_meter(&[1e100], &mut gradient, &mut hessian, &mut meter)
                    .is_err()
            );
            assert_eq!(gradient, [13.0]);
            assert_eq!(hessian, [17.0]);
        }
        let mut exhausted = Budget::new(BudgetLimits::uniform(0, 0));
        assert!(
            objective
                .evaluate_with_meter(&[1e100], &mut exhausted)
                .is_err()
        );
        for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(objective.hessian_vector_product(&[point], &[0.0]).is_err());
        }
        let held = Expr::Mul(vec![
            Expr::from_i64(0),
            function(name, function("log", Expr::symbol("x"))),
        ]);
        let guarded = VerifiedObjective::try_compile(&held, &variables).unwrap();
        for point in [-1.0, 0.0] {
            assert!(guarded.evaluate(&[point]).is_err());
            assert!(guarded.hessian_vector_product(&[point], &[0.0]).is_err());
        }
    }
}
