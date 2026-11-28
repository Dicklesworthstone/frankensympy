//! An anisotropic nonlinear system whose ordinary Jacobian defect has norm two.
use fsym_budget::Unbounded;
use fsym_core::root_certificate::artifact::{
    RootCertificateSources, export_root_certificate, replay_root_certificate,
};
use fsym_core::{
    BigRational, ContractionNorm, ParametricRootBoxWitness, RealBall, RootBoxError, RootBoxWitness,
    RootBoxWitness2, Symbol, certify_parametric_root_box, certify_root_box, certify_root_box_2x2,
    enclose_parametric_root_sensitivity, parse_raw,
};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn axes() -> [Symbol; 2] {
    [Symbol::new("x"), Symbol::new("y")]
}
fn box_at_zero(radius: BigRational) -> RealBall {
    RealBall::new(q(0, 1), radius).unwrap()
}
fn witness() -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![box_at_zero(q(2, 1)), box_at_zero(q(1, 4))],
        center: vec![q(0, 1); 2],
        preconditioner: vec![vec![q(1, 1), q(0, 1)], vec![q(0, 1), q(1, 1)]],
        precision_digits: 24,
    }
}

#[test]
fn coupled_nonlinear_root_and_artifact_replay_use_exact_box_metric() {
    let source = [
        parse_raw("x-4*y^2").unwrap(),
        parse_raw("y-x^2/128").unwrap(),
    ];
    let witness = witness();
    let receipt = certify_root_box(&source, &axes(), &witness).unwrap();
    // |D_12| <= 2, |D_21| <= 1/32. With the radius ratio 8:1,
    // the two scaled row sums are both exactly 1/4.
    assert_eq!(
        receipt.contraction_norm(),
        ContractionNorm::BoxRadiusScaledInfinity
    );
    assert_eq!(receipt.contraction_bound(), &q(1, 4));
    assert!(
        receipt
            .root_enclosure()
            .iter()
            .all(|ball| ball.contains(&q(0, 1)))
    );
    let legacy = RootBoxWitness2 {
        domain: witness.domain.clone().try_into().unwrap(),
        center: [q(0, 1), q(0, 1)],
        preconditioner: [[q(1, 1), q(0, 1)], [q(0, 1), q(1, 1)]],
        precision_digits: 24,
    };
    let two = certify_root_box_2x2(&source, &axes(), &legacy).unwrap();
    assert_eq!(two.contraction_norm(), receipt.contraction_norm());
    assert_eq!(two.contraction_bound(), receipt.contraction_bound());
    let owner =
        RootCertificateSources::new(source.to_vec(), axes().to_vec(), &mut Unbounded).unwrap();
    let bytes = export_root_certificate(&receipt, [2; 32], &mut Unbounded).unwrap();
    assert_eq!(
        replay_root_certificate(&bytes, &owner, [2; 32], &mut Unbounded).unwrap(),
        receipt
    );
}

#[test]
fn contraction_and_strict_inclusion_are_both_obligatory() {
    let box_ = witness();
    let source = [
        parse_raw("x-4*y^2").unwrap(),
        parse_raw("y-x^2/32").unwrap(),
    ];
    // The second weighted row is exactly one, not strictly contractive.
    assert_eq!(
        certify_root_box(&source, &axes(), &box_),
        Err(RootBoxError::NotContractive)
    );
    let source = [
        parse_raw("x-4*y^2-2").unwrap(),
        parse_raw("y-x^2/128").unwrap(),
    ];
    assert!(matches!(
        certify_root_box(&source, &axes(), &box_),
        Err(RootBoxError::NoInteriorInclusion { .. })
    ));
    let linear = [parse_raw("x").unwrap(), parse_raw("y").unwrap()];
    let ordinary = certify_root_box(&linear, &axes(), &box_).unwrap();
    assert_eq!(ordinary.contraction_norm(), ContractionNorm::Infinity);
    assert_eq!(ordinary.contraction_bound(), &q(0, 1));
}

#[test]
fn uniform_family_and_derivatives_in_physical_units() {
    let source = [
        parse_raw("x-4*y^2").unwrap(),
        parse_raw("y-x^2/128-p").unwrap(),
    ];
    let base = witness();
    let mut family = ParametricRootBoxWitness {
        unknown_domain: base.domain,
        parameter_domain: vec![box_at_zero(q(1, 8))],
        center: base.center,
        preconditioner: base.preconditioner,
        precision_digits: 24,
    };
    let p = [Symbol::new("p")];
    let receipt = certify_parametric_root_box(&source, &axes(), &p, &family).unwrap();
    assert_eq!(
        receipt.contraction_norm(),
        ContractionNorm::BoxRadiusScaledInfinity
    );
    assert_eq!(receipt.contraction_bound(), &q(1, 4));
    let sensitivity = enclose_parametric_root_sensitivity(&receipt).unwrap();
    assert_eq!(sensitivity.contraction_bound(), receipt.contraction_bound());
    // y=1/8, x=1/16 gives p=4095/32768 and dx/dp=dy/dp=1024/1023.
    // A spurious unweighted Neumann remainder of 1/3 would lose dx/dp.
    assert!(family.parameter_domain[0].contains(&q(4095, 32768)));
    let jac = sensitivity.solution_jacobian();
    assert_eq!(jac[0][0], box_at_zero(q(8, 3)));
    assert_eq!(jac[1][0], RealBall::new(q(1, 1), q(1, 3)).unwrap());
    assert!(jac.iter().all(|row| row[0].contains(&q(1024, 1023))));
    family.parameter_domain[0] = box_at_zero(q(1, 4));
    assert!(matches!(
        certify_parametric_root_box(&source, &axes(), &p, &family),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 1 })
    ));
}
