//! Integration proof for the pinned-consumer boundary
//! (`fra-rc-adapters-iym`): a transcendental campaign-shaped residual
//! system is compiled by fsym-calculus, Newton-refined with every linear
//! solve executed through the pinned `fnp-linalg` consumer, and the
//! accepted iterate is certified under exact ball arithmetic against the
//! isolated-oracle-compatible declared bound. Singular systems refuse
//! through the consumer's typed error.

use fsym_core::{BigInt, Expr, Symbol};
use fsym_numeric_consumer::{
    ConsumerError, NewtonGuards, certified_residual_bound, frankenscipy_fsolve_verified,
    newton_refine_2x2, newton_refine_verified_2x2,
};

fn transcendental_system() -> (Vec<Expr>, Vec<Symbol>) {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    // f1 = exp(x/2) + y - 3 ; f2 = x + sin(y) - 1.2
    let half_x = Expr::Mul(vec![
        Expr::Rational(fsym_core::BigRational::new(
            BigInt::from(1),
            BigInt::from(2),
        )),
        Expr::Sym(x.clone()),
    ]);
    let f1 = Expr::Add(vec![
        Expr::Function("exp".into(), vec![half_x]),
        Expr::Sym(y.clone()),
        Expr::from_i64(-3),
    ]);
    let f2 = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Function("sin".into(), vec![Expr::Sym(y.clone())]),
        Expr::Rational(fsym_core::BigRational::new(
            BigInt::from(-6),
            BigInt::from(5),
        )),
    ]);
    (vec![f1, f2], vec![x, y])
}

#[test]
fn consumer_newton_converges_and_certifies_transcendental_system() {
    let (exprs, vars) = transcendental_system();
    let system = fsym_calculus::VerifiedResidualSystem::try_compile(&exprs, &vars)
        .expect("campaign system compiles");

    let outcome = newton_refine_verified_2x2(&system, [1.0, 1.0], &NewtonGuards::default(), 12)
        .expect("consumer-driven Newton converges");

    assert!(
        outcome.residual_norm <= 1e-12,
        "residual norm {} exceeds the declared tolerance",
        outcome.residual_norm
    );
    assert!(
        outcome.final_jacobian_determinant.abs() > 1e-6,
        "floating determinant diagnostic should stay away from zero on this fixture"
    );
    assert!(outcome.iterations <= 16, "Newton must converge quickly");

    // Certified consistency: the accepted iterate, substituted as exact
    // binary64 rationals, evaluates under certified ball arithmetic to a
    // residual infinity-norm bounded by 1e-12 with ball radii far below.
    certified_residual_bound(system.residuals(), system.vars(), &outcome.root, 12)
        .expect("certified residual bound at the accepted iterate");
}

#[test]
fn singular_system_refuses_through_the_consumer() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    // Both equations impose the same constraint: the Jacobian is singular
    // everywhere.
    let degenerate = Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]);
    let exprs = vec![
        Expr::Add(vec![degenerate.clone(), Expr::from_i64(-1)]),
        Expr::Add(vec![degenerate, Expr::from_i64(-1)]),
    ];
    let vars = vec![x, y];
    let system = fsym_calculus::CompiledResidualSystem::try_compile(&exprs, &vars)
        .expect("degenerate system compiles");

    let outcome = newton_refine_2x2(&system, [0.0, 0.0], &NewtonGuards::default());
    assert!(
        matches!(
            outcome,
            Err(fsym_numeric_consumer::ConsumerError::SingularJacobian(_))
        ),
        "degenerate system must refuse through the consumer's typed error, got {outcome:?}"
    );
}

#[test]
fn frankenscipy_fsolve_consumer_agrees_under_declared_comparator() {
    use fsym_numeric_consumer::certified_residual_bound;

    let (exprs, vars) = transcendental_system();
    let system = fsym_calculus::VerifiedResidualSystem::try_compile(&exprs, &vars)
        .expect("campaign system compiles");

    // Declared comparator: both consumers approximate the same root; their
    // iterates must agree in infinity-norm within 1e-6, and each accepted
    // root must carry the certified ball bound.
    let scipy_root =
        frankenscipy_fsolve_verified(&system, &[1.0, 1.0], 8).expect("fsolve consumer");
    let newton_root = newton_refine_verified_2x2(&system, [1.0, 1.0], &NewtonGuards::default(), 12)
        .expect("consumer Newton");
    let agreement = (scipy_root[0] - newton_root.root[0])
        .abs()
        .max((scipy_root[1] - newton_root.root[1]).abs());
    assert!(
        agreement <= 1e-6,
        "consumer roots diverge beyond the declared comparator: {agreement}"
    );
    // Declared fsolve-lane bound: the consumer converges by its own
    // finite-difference tolerance (1e-10), so 1e-8 is the honest
    // certified statement for this lane.
    certified_residual_bound(&exprs, &vars, &scipy_root, 8)
        .expect("certified bound at the fsolve root");

    // Guarded input: non-finite initial iterates refuse before any solve.
    let refused = frankenscipy_fsolve_verified(&system, &[f64::NAN, 0.0], 8);
    assert!(
        matches!(
            refused,
            Err(fsym_numeric_consumer::ConsumerError::Evaluation(_))
        ),
        "non-finite initial iterate must refuse through a typed error"
    );
}

fn identity_system() -> fsym_calculus::CompiledResidualSystem {
    let vars = vec![Symbol::new("x"), Symbol::new("y")];
    let residuals = vars.iter().cloned().map(Expr::Sym).collect::<Vec<_>>();
    fsym_calculus::CompiledResidualSystem::try_compile(&residuals, &vars).unwrap()
}

#[test]
fn nonfinite_residuals_and_jacobians_cannot_report_convergence() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let logarithm = Expr::Function("log".into(), vec![Expr::Sym(x.clone())]);
    let system = fsym_calculus::CompiledResidualSystem::try_compile(
        &[logarithm, Expr::Sym(y.clone())],
        &[x, y],
    )
    .unwrap();
    // Previously max(0, NaN) suppressed the invalid residual and accepted this.
    assert!(matches!(
        newton_refine_2x2(&system, [-1.0, 0.0], &NewtonGuards::default()),
        Err(ConsumerError::Evaluation(_)),
    ));
    assert!(matches!(
        fsym_numeric_consumer::frankenscipy_fsolve(&system, &[-1.0, 0.0]),
        Err(ConsumerError::Evaluation(_)),
    ));
    // Even at an exact residual zero, a nonfinite derivative is not a valid
    // final Jacobian diagnostic.
    let mut bad_jacobian = identity_system();
    bad_jacobian.compiled_jacobian[0] = fsym_calculus::CompiledExpr::try_from_ops(vec![
        fsym_calculus::CompiledOp::LoadConst(-1.0),
        fsym_calculus::CompiledOp::Sqrt,
    ])
    .unwrap();
    assert!(matches!(
        newton_refine_2x2(&bad_jacobian, [0.0, 0.0], &NewtonGuards::default()),
        Err(ConsumerError::Evaluation(_)),
    ));
}

#[test]
fn malformed_guards_and_nonfinite_initial_points_refuse_before_convergence() {
    let system = identity_system();
    for tolerance in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let guards = NewtonGuards {
            residual_tolerance: tolerance,
            ..NewtonGuards::default()
        };
        assert!(matches!(
            newton_refine_2x2(&system, [0.0, 0.0], &guards),
            Err(ConsumerError::InvalidGuards(_))
        ));
    }
    for step in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        let guards = NewtonGuards {
            max_step_norm: step,
            ..NewtonGuards::default()
        };
        assert!(matches!(
            newton_refine_2x2(&system, [0.0, 0.0], &guards),
            Err(ConsumerError::InvalidGuards(_))
        ));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            newton_refine_2x2(&system, [value, 0.0], &NewtonGuards::default()),
            Err(ConsumerError::Evaluation(_))
        ));
    }
}

#[test]
fn final_determinant_is_evaluated_at_the_returned_iterate() {
    let outcome =
        newton_refine_2x2(&identity_system(), [0.0, 0.0], &NewtonGuards::default()).unwrap();
    assert_eq!(outcome.iterations, 0);
    assert_eq!(outcome.final_jacobian_determinant, 1.0);

    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let residuals = vec![
        Expr::Add(vec![
            Expr::Sym(x.clone()).pow(Expr::from_i64(2)),
            Expr::from_i64(-2),
        ]),
        Expr::Add(vec![
            Expr::Sym(y.clone()).pow(Expr::from_i64(2)),
            Expr::from_i64(-3),
        ]),
    ];
    let system = fsym_calculus::VerifiedResidualSystem::try_compile(&residuals, &[x, y]).unwrap();
    let guards = NewtonGuards {
        residual_tolerance: 1e-7,
        ..NewtonGuards::default()
    };
    let outcome = newton_refine_verified_2x2(&system, [1.0, 1.0], &guards, 7).unwrap();
    let determinant_at_return = (2.0 * outcome.root[0]) * (2.0 * outcome.root[1]);
    assert_eq!(outcome.final_jacobian_determinant, determinant_at_return);
}

#[test]
fn certified_bound_rejects_bad_shapes_and_hostile_precision_without_panicking() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let z = Symbol::new("z");
    let residual = [Expr::from_i64(0)];
    for vars in [
        vec![],
        vec![x.clone()],
        vec![x.clone(), x.clone()],
        vec![x.clone(), y.clone(), z],
    ] {
        assert!(certified_residual_bound(&residual, &vars, &[0.0, 0.0], 12).is_err());
    }
    assert!(certified_residual_bound(&[], &[x.clone(), y.clone()], &[0.0, 0.0], 12).is_err());
    assert!(
        certified_residual_bound(&residual, &[x.clone(), y.clone()], &[f64::NAN, 0.0], 12).is_err()
    );
    assert!(certified_residual_bound(&residual, &[x, y], &[0.0, 0.0], u32::MAX).is_err());
}

#[test]
fn verified_consumer_does_not_return_an_uncertified_approximation() {
    let (residuals, vars) = transcendental_system();
    let system = fsym_calculus::VerifiedResidualSystem::try_compile(&residuals, &vars).unwrap();
    // Loose floating stopping tolerance intentionally admits the initial point;
    // the independent full-enclosure check must still reject publication.
    let guards = NewtonGuards {
        residual_tolerance: 100.0,
        ..NewtonGuards::default()
    };
    assert!(matches!(
        newton_refine_verified_2x2(&system, [1.0, 1.0], &guards, 12),
        Err(ConsumerError::Certification(_)),
    ));
}

#[test]
fn certification_preserves_original_domain_checks_during_substitution() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let log_x = Expr::Function("log".into(), vec![Expr::Sym(x.clone())]);
    let held_power = Expr::Pow(
        std::sync::Arc::new(log_x.clone()),
        std::sync::Arc::new(Expr::from_i64(0)),
    );
    let held_zero_product = Expr::Mul(vec![Expr::from_i64(0), log_x]);
    let residuals = vec![
        Expr::Add(vec![held_power, Expr::from_i64(-1)]),
        held_zero_product,
    ];
    for residual in &residuals {
        assert!(
            certified_residual_bound(
                std::slice::from_ref(residual),
                &[x.clone(), y.clone()],
                &[0.0, 0.0],
                12
            )
            .is_err()
        );
        certified_residual_bound(
            std::slice::from_ref(residual),
            &[x.clone(), y.clone()],
            &[2.0, 0.0],
            12,
        )
        .unwrap();
    }
    let mut deep = Expr::Sym(x.clone());
    for _ in 0..=fsym_calculus::MAX_COMPILE_DEPTH {
        deep = Expr::Add(vec![deep]);
    }
    assert!(certified_residual_bound(&[deep], &[x, y], &[0.0, 0.0], 12).is_err());
}
