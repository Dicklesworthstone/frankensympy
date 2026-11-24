//! Exact radial cancellation must precede evaluation of potentially huge partials.

use fsym_budget::{Budget, BudgetLimits, Dimension};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::{
    SparseJacobian, verify_sparse_jacobian_certificate,
    verify_sparse_jacobian_certificate_with_meter,
};
use fsym_core::{Expr, Symbol};

fn angle(y: Expr, x: Expr) -> Expr {
    Expr::Function("atan2".into(), vec![y, x])
}

fn radial_sources() -> Vec<Expr> {
    let t = Expr::symbol("t");
    let equal = angle(t.clone(), t.clone());
    vec![
        equal.clone(),
        angle(Expr::Mul(vec![Expr::from_i64(2), t.clone()]), t.clone()),
        angle(
            Expr::Mul(vec![Expr::from_i64(3), t.clone()]),
            Expr::Mul(vec![Expr::from_i64(2), t]),
        ),
        // This composition previously obtained zero through recursive numeric
        // simplification, before partial jets were introduced.
        Expr::Function("sin".into(), vec![equal]),
    ]
}

#[test]
fn exact_radial_cancellation_keeps_subnormal_residuals_and_objectives_evaluable() {
    let vars = [Symbol::new("t")];
    let smallest = f64::from_bits(1);
    for source in radial_sources() {
        let sources = [source.clone()];
        let candidate = SparseJacobian::try_build(&sources, &vars).unwrap();
        assert!(candidate.entries.is_empty());
        assert_eq!(candidate.pattern.nnz, 0);
        verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
        let residual = VerifiedResidualSystem::try_compile(&sources, &vars).unwrap();
        let dense = residual.try_dense_program().unwrap();
        let objective = VerifiedObjective::try_compile(&source, &vars).unwrap();
        assert!(objective.candidate().gradient.entries.is_empty());
        assert!(objective.candidate().hessian.entries.is_empty());
        for t in [smallest, -smallest, f64::MIN_POSITIVE, -1.0, 1.0] {
            let evaluated = residual.evaluate_sparse(&[t]).unwrap();
            assert!(evaluated.residuals[0].is_finite());
            assert!(evaluated.jacobian.is_empty());
            assert_eq!(
                residual.jacobian_vector_product(&[t], &[2.0]).unwrap(),
                [0.0]
            );
            assert_eq!(
                residual
                    .jacobian_transpose_vector_product(&[t], &[3.0])
                    .unwrap(),
                [0.0],
            );
            let mut gradient = [17.0];
            dense.try_eval_jacobian(&[t], &mut gradient).unwrap();
            assert_eq!(gradient, [0.0]);
            let evaluated = objective.evaluate(&[t]).unwrap();
            assert!(evaluated.value.is_finite());
            assert_eq!(evaluated.gradient, [0.0]);
            assert!(evaluated.hessian.is_empty());
            assert_eq!(
                objective.hessian_vector_product(&[t], &[7.0]).unwrap(),
                [0.0]
            );
        }
        for zero in [0.0, -0.0] {
            // Omitting a mathematically zero derivative never removes the
            // original angle's undefined origin from either checked program.
            assert!(residual.evaluate_sparse(&[zero]).is_err());
            assert!(objective.evaluate(&[zero]).is_err());
        }
        let bytes = residual.to_json_bytes().unwrap();
        let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored.to_json_bytes().unwrap(), bytes);
        assert_eq!(
            restored.evaluate_sparse(&[smallest]).unwrap(),
            residual.evaluate_sparse(&[smallest]).unwrap(),
        );
        let mut forged = candidate;
        forged.residuals[0] = angle(Expr::symbol("t"), Expr::from_i64(1));
        assert!(verify_sparse_jacobian_certificate(&forged, &[]).is_err());
    }
}

fn build_and_admit(sources: &[Expr], vars: &[Symbol], meter: &mut Budget) -> bool {
    let Ok(candidate) = SparseJacobian::try_build_with_meter(sources, vars, meter) else {
        return false;
    };
    verify_sparse_jacobian_certificate_with_meter(&candidate, &[], meter).is_ok()
}

#[test]
fn exact_cancellation_generation_and_verification_share_the_caller_budget() {
    const AMPLE: u64 = 1 << 40;
    let vars = [Symbol::new("t")];
    let sources = radial_sources();
    let mut measured = Budget::new(BudgetLimits::uniform(AMPLE, 0));
    assert!(build_and_admit(&sources, &vars, &mut measured));
    let spent = AMPLE - measured.remaining(Dimension::ComputeSteps);
    assert!(spent > 0);
    for (steps, passes) in [(spent, true), (spent - 1, false)] {
        let mut limits = BudgetLimits::uniform(AMPLE, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = steps;
        assert_eq!(
            build_and_admit(&sources, &vars, &mut Budget::new(limits)),
            passes,
        );
    }
}

#[test]
fn raw_angle_admission_does_not_expand_held_constant_powers_to_probe_zero() {
    let variable = Symbol::new("t");
    let t = Expr::Sym(variable.clone());
    let mut held = Expr::from_i64(2);
    for _ in 0..3 {
        held = held.pow(Expr::from_i64(1000));
    }
    let source = angle(t.clone(), held.clone());
    let mut limits = BudgetLimits::uniform(16 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 1_000_000;
    let mut meter = Budget::new(limits);
    let candidate = SparseJacobian::try_build_with_meter(&[source], &[variable], &mut meter)
        .expect("opaque zero matching must not materialize the held integer power");
    verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut meter).unwrap();
    let expected = Expr::Add(vec![
        Expr::Mul(vec![
            Expr::Function("_fsym_atan2_dy".into(), vec![t.clone(), held.clone()]),
            Expr::from_i64(1),
        ]),
        Expr::Mul(vec![
            Expr::Function("_fsym_atan2_dx".into(), vec![t, held]),
            Expr::from_i64(0),
        ]),
    ]);
    assert_eq!(candidate.entries.len(), 1);
    assert_eq!(candidate.entries[0].symbolic_deriv, expected);
    assert!(candidate.entries[0].compiled_deriv.ops().len() < 64);
    assert!(meter.remaining(Dimension::ComputeSteps) > 0);
    // Only generation and exact raw admission are asserted here. Evaluating
    // this coordinate would overflow binary64; generic recursive symbolic
    // simplification has its own existing constant-folding behavior.
}
