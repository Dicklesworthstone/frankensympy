//! End-to-end angle derivatives through independent admission and replay.

use super::{InterruptAt, source, variables};
use fsym_budget::{BudgetLimits, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, CompiledResidualSystem};
use fsym_calculus::program_artifact::{
    admit_objective_candidate, decode_objective_candidate, encode_objective_candidate,
};
use fsym_calculus::proof::{make_diff_term, verified_diff, verify_diff_derivation};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::verify_sparse_jacobian_certificate;
use fsym_core::Expr;
use fsym_proof_kernel::{Claim, ProofRule};
use std::collections::HashMap;

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "nonfinite result for {expected:e}");
    if expected != 0.0 {
        assert_ne!(actual, 0.0, "erased representable derivative {expected:e}");
        assert_eq!(actual.is_sign_positive(), expected.is_sign_positive());
    }
    assert!(
        (actual - expected).abs() <= 8e-13 * expected.abs() + 2.0 * f64::from_bits(1),
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn extreme_slopes_reach_sparse_dense_and_both_directional_interfaces() {
    let vars = variables();
    let residual = VerifiedResidualSystem::try_compile(&[source()], &vars).unwrap();
    let dense = CompiledResidualSystem::try_compile(&[source()], &vars).unwrap();
    let exported = residual.try_dense_program().unwrap();
    for exponent in [-1020, -700, -500, 0, 500, 700, 1020] {
        let scale = 2.0_f64.powi(exponent);
        let inverse = 2.0_f64.powi(-exponent);
        let point = [3.0 * scale, 4.0 * scale];
        let slope = [(4.0 / 25.0) * inverse, (-3.0 / 25.0) * inverse];
        let evaluated = residual.evaluate_sparse(&point).unwrap();
        close(evaluated.residuals[0], 3.0_f64.atan2(4.0));
        for (actual, expected) in evaluated.jacobian.iter().zip(slope) {
            close(*actual, expected);
        }
        close(
            residual
                .jacobian_vector_product(&point, &[2.0, -1.0])
                .unwrap()[0],
            (11.0 / 25.0) * inverse,
        );
        let transpose = residual
            .jacobian_transpose_vector_product(&point, &[3.0])
            .unwrap();
        close(transpose[0], (12.0 / 25.0) * inverse);
        close(transpose[1], (-9.0 / 25.0) * inverse);
        for program in [&dense, &exported] {
            let mut gradient = [0.0; 2];
            program.try_eval_jacobian(&point, &mut gradient).unwrap();
            close(gradient[0], slope[0]);
            close(gradient[1], slope[1]);
        }
    }
}

#[test]
fn finite_and_subnormal_hessians_survive_norm_overflow_and_underflow() {
    let objective = VerifiedObjective::try_compile(&source(), &variables()).unwrap();
    for scale in [1e-150, 1e-100, 1.0, 1e100, 1e150, 1e155] {
        let inverse = 1.0 / scale;
        let point = [3.0 * scale, 4.0 * scale];
        let result = objective.evaluate(&point).unwrap();
        let yy = ((-24.0 / 625.0) * inverse) * inverse;
        let yx = ((-7.0 / 625.0) * inverse) * inverse;
        let xx = ((24.0 / 625.0) * inverse) * inverse;
        for (actual, expected) in result.hessian.iter().zip([yy, yx, yx, xx]) {
            close(*actual, expected);
        }
        let product = objective
            .hessian_vector_product(&point, &[2.0, -1.0])
            .unwrap();
        close(product[0], 2.0 * yy - yx);
        close(product[1], 2.0 * yx - xx);
    }
    // A finite angle and gradient do not make an overflowing true Hessian finite.
    assert!(objective.evaluate(&[3e-200, 4e-200]).is_err());
}

#[test]
fn current_angle_programs_replay_exact_bytes_with_scaled_derivatives() {
    let vars = variables();
    let residual = VerifiedResidualSystem::try_compile(&[source()], &vars).unwrap();
    let encoded = residual.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&encoded).unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), encoded);
    for point in [[3e-200, 4e-200], [3e200, 4e200]] {
        assert_eq!(
            restored.evaluate_sparse(&point).unwrap(),
            residual.evaluate_sparse(&point).unwrap(),
        );
    }
    let objective = VerifiedObjective::try_compile(&source(), &vars).unwrap();
    let encoded = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
    let candidate = decode_objective_candidate(&encoded, &mut Unbounded).unwrap();
    let restored = admit_objective_candidate(&source(), &vars, candidate, &mut Unbounded).unwrap();
    assert_eq!(
        encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
        encoded,
    );
    for point in [[3e-150, 4e-150], [3e155, 4e155]] {
        assert_eq!(
            restored.evaluate(&point).unwrap(),
            objective.evaluate(&point).unwrap(),
        );
    }
}

#[test]
fn public_symbolic_receipts_keep_the_original_quotient_rule() {
    let original = source();
    let vars = variables();
    for (index, variable) in vars.iter().enumerate() {
        let expected = Expr::Mul(vec![
            Expr::Add(vec![
                Expr::Mul(vec![
                    Expr::symbol("x"),
                    Expr::from_i64(i64::from(index == 0)),
                ]),
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::symbol("y"),
                    Expr::from_i64(i64::from(index == 1)),
                ]),
            ]),
            Expr::Add(vec![
                Expr::symbol("x").pow(Expr::from_i64(2)),
                Expr::symbol("y").pow(Expr::from_i64(2)),
            ])
            .pow(Expr::from_i64(-1)),
        ]);
        let (derivative, proof) = verified_diff(&original, variable);
        assert_eq!(derivative, expected);
        verify_diff_derivation(&proof, &original, variable, &derivative).unwrap();
    }
}

fn replace_partial(expression: &Expr, from: &str, to: &str) -> Expr {
    match expression {
        Expr::Function(name, arguments) => Expr::Function(
            if name == from {
                to.into()
            } else {
                name.clone()
            },
            arguments
                .iter()
                .map(|argument| replace_partial(argument, from, to))
                .collect(),
        ),
        Expr::Add(terms) => Expr::Add(
            terms
                .iter()
                .map(|term| replace_partial(term, from, to))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::Mul(
            factors
                .iter()
                .map(|factor| replace_partial(factor, from, to))
                .collect(),
        ),
        Expr::Pow(base, exponent) => Expr::Pow(
            std::sync::Arc::new(replace_partial(base, from, to)),
            std::sync::Arc::new(replace_partial(exponent, from, to)),
        ),
        _ => expression.clone(),
    }
}

#[test]
fn independent_angle_verification_rejects_consistent_first_and_second_partial_forgery() {
    let vars = variables();
    let mapping = HashMap::from([(vars[0].clone(), 0), (vars[1].clone(), 1)]);
    let original = ObjectiveCandidate::try_build(&source(), &vars).unwrap();
    for (second, from, to) in [
        (false, "_fsym_atan2_dy", "_fsym_atan2_dx"),
        (true, "_fsym_atan2_dyy", "_fsym_atan2_dyx"),
        (true, "_fsym_atan2_dyx", "_fsym_atan2_dxx"),
    ] {
        let mut forged = original.clone();
        let stage = if second {
            &mut forged.hessian
        } else {
            &mut forged.gradient
        };
        let index = stage
            .entries
            .iter()
            .position(|entry| {
                replace_partial(&entry.symbolic_deriv, from, to) != entry.symbolic_deriv
            })
            .expect("the requested partial must occur in this derivative stage");
        let entry = &mut stage.entries[index];
        let wrong = replace_partial(&entry.symbolic_deriv, from, to);
        let lhs = make_diff_term(&stage.residuals[entry.row], &vars[entry.col]);
        entry.symbolic_deriv = wrong.clone();
        entry.derivation.steps[0].claim = Claim::equality(lhs, wrong.clone());
        let ProofRule::DefinitionalReduction { rhs, .. } = &mut entry.derivation.steps[0].rule
        else {
            panic!("expected an exact differentiation rule")
        };
        *rhs = wrong.clone();
        entry.compiled_deriv = CompiledExpr::try_compile(&wrong, &mapping).unwrap();
        assert!(verify_sparse_jacobian_certificate(stage, &[]).is_err());
        assert!(VerifiedObjective::try_from_candidate(&source(), &vars, forged).is_err());
    }
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
        assert!(VerifiedObjective::try_from_candidate(&source(), &vars, forged).is_err());
    }
}

#[test]
fn cancelled_extreme_angle_evaluation_keeps_complete_outputs_unpublished() {
    let objective = VerifiedObjective::try_compile(&source(), &variables()).unwrap();
    let limits = BudgetLimits::uniform(1_000_000, 37);
    let point = [3e150, 4e150];
    let mut measured = InterruptAt::new(limits);
    objective
        .try_eval_into_with_meter(&point, &mut [0.0; 2], &mut [0.0; 4], &mut measured)
        .unwrap();
    for stop in [1, measured.checkpoints / 2, measured.checkpoints] {
        let mut meter = InterruptAt::new(limits);
        meter.stop = Some(stop);
        let mut gradient = [71.0, 73.0];
        let mut hessian = [79.0, 83.0, 89.0, 97.0];
        assert!(
            objective
                .try_eval_into_with_meter(&point, &mut gradient, &mut hessian, &mut meter)
                .is_err()
        );
        assert_eq!(gradient, [71.0, 73.0]);
        assert_eq!(hessian, [79.0, 83.0, 89.0, 97.0]);
        assert!(meter.stopped);
    }
}
