use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_numeric_consumer::{
    ConsumerError, NewtonGuards, certify_root_2x2, frankenscipy_fsolve_root_box, newton_refine_2x2,
    newton_refine_root_box_2x2,
};

fn q(n: i64, d: i64) -> Expr {
    Expr::Rational(BigRational::new(BigInt::from(n), BigInt::from(d)))
}
fn transcendental_system() -> VerifiedResidualSystem {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let [x, y] = variables.clone().map(Expr::Sym);
    let equations = [
        Expr::Function("exp".into(), vec![q(1, 2) * x.clone()]) + y.clone() - Expr::from_i64(3),
        x + Expr::Function("sin".into(), vec![y]) - q(6, 5),
    ];
    VerifiedResidualSystem::try_compile(&equations, &variables).unwrap()
}

#[test]
fn restored_proof_bundle_drives_both_consumers_and_independent_root_certificates() {
    let original = transcendental_system();
    let saved = original.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&saved).unwrap();
    assert_eq!(original.sparse_jacobian(), restored.sparse_jacobian());
    assert_eq!(
        original.compiled().compiled_residuals,
        restored.compiled().compiled_residuals
    );
    assert_eq!(
        original.compiled().compiled_jacobian,
        restored.compiled().compiled_jacobian
    );

    let newton = newton_refine_root_box_2x2(
        &restored,
        [1.0, 1.0],
        &NewtonGuards::default(),
        [1e-6, 1e-6],
        24,
    )
    .unwrap();
    let scipy = frankenscipy_fsolve_root_box(&restored, &[1.0, 1.0], [1e-6, 1e-6], 24).unwrap();
    for accepted in [&newton, &scipy] {
        let receipt = accepted.certificate();
        assert_eq!(receipt.residuals().as_slice(), restored.residuals());
        assert_eq!(receipt.variables().as_slice(), restored.vars());
        assert!(receipt.contraction_bound() < &BigRational::from_integer(BigInt::from(1)));
        // Replay needs only the retained originals and witness. It uses no
        // compiled artifact or numerical-consumer result as proof evidence.
        let replay = fsym_core::certify_root_box_2x2(
            receipt.residuals(),
            receipt.variables(),
            receipt.witness(),
        )
        .unwrap();
        assert_eq!(&replay, receipt);
        for i in 0..2 {
            assert!(receipt.root_enclosure()[i].lower() > receipt.witness().domain[i].lower());
            assert!(receipt.root_enclosure()[i].upper() < receipt.witness().domain[i].upper());
        }
    }
    for i in 0..2 {
        assert!((newton.approximation()[i] - scipy.approximation()[i]).abs() < 1e-6);
        assert!(
            !newton.certificate().root_enclosure()[i]
                .is_disjoint(&scipy.certificate().root_enclosure()[i])
        );
    }
}

#[test]
fn tiny_residual_of_a_system_without_roots_does_not_become_a_root_proof() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let equations = [
        Expr::Function("exp".into(), vec![Expr::Sym(variables[0].clone())]),
        Expr::Sym(variables[1].clone()),
    ];
    let system = VerifiedResidualSystem::try_compile(&equations, &variables).unwrap();
    // exp(-40) is below the floating stopping tolerance, although exp(x)
    // has no real zero. The interval theorem must reject this proposal.
    let approximate =
        newton_refine_2x2(system.compiled(), [-40.0, 0.0], &NewtonGuards::default()).unwrap();
    assert_eq!(approximate.iterations, 0);
    assert!(matches!(
        newton_refine_root_box_2x2(
            &system,
            [-40.0, 0.0],
            &NewtonGuards::default(),
            [1e-6, 1e-6],
            24,
        ),
        Err(ConsumerError::Certification(_))
    ));
}

#[test]
fn whole_box_domain_failure_refuses_even_when_the_center_is_an_exact_root() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let equations = [
        Expr::Function("log".into(), vec![Expr::Sym(variables[0].clone())]),
        Expr::Sym(variables[1].clone()),
    ];
    let system = VerifiedResidualSystem::try_compile(&equations, &variables).unwrap();
    assert!(matches!(
        certify_root_2x2(&system, [1.0, 0.0], [2.0, 0.1], 24),
        Err(ConsumerError::Certification(_))
    ));
    let accepted = certify_root_2x2(&system, [1.0, 0.0], [0.01, 0.01], 24).unwrap();
    assert!(
        accepted.certificate().root_enclosure()[0]
            .contains(&BigRational::from_integer(BigInt::from(1)))
    );
}

#[test]
fn invalid_root_box_requests_refuse_before_any_consumer_execution() {
    let system = transcendental_system();
    for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            newton_refine_root_box_2x2(
                &system,
                [1.0, 1.0],
                &NewtonGuards::default(),
                [radius, 1e-6],
                24,
            ),
            Err(ConsumerError::Certification(_))
        ));
    }
    for precision in [0, 7, 129, u32::MAX] {
        assert!(matches!(
            frankenscipy_fsolve_root_box(&system, &[1.0, 1.0], [1e-6, 1e-6], precision),
            Err(ConsumerError::Certification(_))
        ));
    }
    assert!(matches!(
        certify_root_2x2(&system, [f64::NAN, 0.0], [1e-6, 1e-6], 24),
        Err(ConsumerError::Evaluation(_))
    ));
}

#[test]
fn singular_proposal_is_a_typed_refusal() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let [x, y] = variables.clone().map(Expr::Sym);
    let equations = [x.clone() * x, y];
    let system = VerifiedResidualSystem::try_compile(&equations, &variables).unwrap();
    assert!(matches!(
        certify_root_2x2(&system, [0.0, 0.0], [0.1, 0.1], 24),
        Err(ConsumerError::SingularJacobian(_))
    ));
}
