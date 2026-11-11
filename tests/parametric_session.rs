//! Uniform root families and implicit sensitivities through the typed session.

use frankensympy::{
    BigRational, BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval,
    ExactParametricRootWitness, ParametricFamilyId, ParametricProblemId, ParametricSensitivityId,
    Session,
};
use fsym_budget::Dimension;

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(20_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits
}

fn problem() -> ParametricProblemId {
    ParametricProblemId::new("parametric:source").unwrap()
}

fn family(name: &str) -> ParametricFamilyId {
    ParametricFamilyId::new(format!("family:{name}")).unwrap()
}

fn sensitivity(name: &str) -> ParametricSensitivityId {
    ParametricSensitivityId::new(format!("sensitivity:{name}")).unwrap()
}

fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}

fn register(
    session: &mut Session,
    residuals: &[&str],
    unknowns: &[&str],
    parameters: &[&str],
) -> CertificationResponse {
    session
        .certification_request(CertificationRequest::RegisterParametricProblem {
            schema_version: 1,
            problem: problem(),
            residuals: residuals.iter().map(|source| (*source).into()).collect(),
            unknowns: unknowns.iter().map(|name| (*name).into()).collect(),
            parameters: parameters.iter().map(|name| (*name).into()).collect(),
        })
        .unwrap()
}

fn registered(residuals: &[&str], unknowns: &[&str], parameters: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    register(&mut session, residuals, unknowns, parameters);
    session
}

fn witness() -> ExactParametricRootWitness {
    ExactParametricRootWitness {
        unknown_domain: vec![interval("-2", "2")],
        parameter_domain: vec![interval("-1", "1")],
        center: vec!["0".into()],
        preconditioner: vec![vec!["1".into()]],
        precision_digits: 16,
    }
}

fn certify(name: &str, witness: ExactParametricRootWitness) -> CertificationRequest {
    CertificationRequest::CertifyParametricRoot {
        schema_version: 1,
        problem: problem(),
        family: family(name),
        witness,
    }
}

fn differentiate(family_name: &str, sensitivity_name: &str) -> CertificationRequest {
    CertificationRequest::EncloseParametricSensitivity {
        schema_version: 1,
        family: family(family_name),
        sensitivity: sensitivity(sensitivity_name),
    }
}

fn adaptive(name: &str, max_leaves: usize) -> CertificationRequest {
    CertificationRequest::CertifyParametricCover {
        schema_version: 1,
        problem: problem(),
        family: family(name),
        unknown_domain: vec![interval("0", "2")],
        parameter_domain: vec![interval("0", "2")],
        precision_digits: 16,
        max_leaves,
    }
}

#[test]
fn affine_family_keeps_every_parameter_and_frozen_source_before_sensitivity() {
    let mut session = Session::with_budget_limits(limits(), 256);
    let CertificationResponse::ParametricProblemRegistered {
        residuals,
        unknowns,
        parameters,
        context_digest,
        ..
    } = register(&mut session, &["x-p"], &["x"], &["p"])
    else {
        panic!("registered parameter problem");
    };
    assert_eq!(residuals, 1);
    assert_eq!(
        unknowns
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>(),
        ["x"]
    );
    assert_eq!(
        parameters
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>(),
        ["p"]
    );
    session.construct("x", "100").unwrap();
    session.construct("p", "200").unwrap();
    let CertificationResponse::ParametricFamilyCertified {
        evidence,
        claim,
        unknown_domain,
        parameter_domain,
        root_enclosure,
        contraction_bound,
        leaf_count,
        precision_digits,
        context_digest: family_context,
        ..
    } = session
        .certification_request(certify("affine", witness()))
        .unwrap()
    else {
        panic!("uniform family");
    };
    assert_eq!(evidence, "certified_numeric");
    assert_eq!(
        claim,
        "for_every_parameter_exactly_one_nonsingular_root_in_unknown_box"
    );
    assert_eq!(unknown_domain, [interval("-2", "2")]);
    assert_eq!(parameter_domain, [interval("-1", "1")]);
    assert_eq!(root_enclosure, [interval("-1", "1")]);
    assert_eq!(contraction_bound, "0");
    assert_eq!(leaf_count, 1);
    assert_eq!(precision_digits, 16);
    assert_eq!(family_context, context_digest);
    let CertificationResponse::ParametricSensitivityEnclosed {
        evidence,
        claim,
        solution_jacobian,
        leaf_count,
        precision_digits,
        context_digest: sensitivity_context,
        ..
    } = session
        .certification_request(differentiate("affine", "affine"))
        .unwrap()
    else {
        panic!("implicit solution derivative");
    };
    assert_eq!(evidence, "certified_numeric");
    assert_eq!(claim, "implicit_solution_jacobian_enclosure");
    assert_eq!(solution_jacobian, [vec![interval("1", "1")]]);
    assert_eq!(leaf_count, 1);
    assert_eq!(precision_digits, 16);
    assert_eq!(sensitivity_context, context_digest);
    assert!(session.parametric_problem(&problem()).is_some());
    assert!(session.parametric_family(&family("affine")).is_some());
    assert!(
        session
            .parametric_sensitivity(&sensitivity("affine"))
            .is_some()
    );
}

#[test]
fn coupled_family_and_sensitivity_preserve_separate_unknown_and_parameter_order() {
    // The declared row order is (y,x), and the parameter-column order is
    // (q,p). The exact family is y=q, x=p-2q.
    let mut session = registered(&["x+2*y-p", "y-q"], &["y", "x"], &["q", "p"]);
    let valid = ExactParametricRootWitness {
        unknown_domain: vec![interval("-4", "4"); 2],
        parameter_domain: vec![interval("-1", "1"); 2],
        center: vec!["0".into(); 2],
        preconditioner: vec![vec!["0".into(), "1".into()], vec!["1".into(), "-2".into()]],
        precision_digits: 16,
    };
    let mut wrong_order = valid.clone();
    wrong_order.preconditioner.swap(0, 1);
    let before = session.budget_snapshot();
    assert_eq!(
        session
            .certification_request(certify("ordered", wrong_order))
            .unwrap_err()
            .code(),
        "refused"
    );
    assert!(session.parametric_family(&family("ordered")).is_none());
    assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    let CertificationResponse::ParametricFamilyCertified { root_enclosure, .. } = session
        .certification_request(certify("ordered", valid))
        .unwrap()
    else {
        panic!("ordered family");
    };
    assert_eq!(root_enclosure, [interval("-1", "1"), interval("-3", "3")]);
    let CertificationResponse::ParametricSensitivityEnclosed {
        solution_jacobian, ..
    } = session
        .certification_request(differentiate("ordered", "ordered"))
        .unwrap()
    else {
        panic!("ordered derivative matrix");
    };
    assert_eq!(
        solution_jacobian,
        [
            vec![interval("1", "1"), interval("0", "0")],
            vec![interval("-2", "-2"), interval("1", "1")],
        ]
    );
    assert_eq!(
        session
            .certification_request(differentiate("ordered", "ordered"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn sensitivity_retains_parameter_variation_and_nonzero_preconditioner_defect() {
    let mut nonlinear = registered(&["x-p^2"], &["x"], &["p"]);
    let mut proposal = witness();
    proposal.unknown_domain = vec![interval("-1", "2")];
    proposal.parameter_domain = vec![interval("0", "1")];
    nonlinear
        .certification_request(certify("nonlinear", proposal))
        .unwrap();
    let CertificationResponse::ParametricSensitivityEnclosed {
        solution_jacobian, ..
    } = nonlinear
        .certification_request(differentiate("nonlinear", "nonlinear"))
        .unwrap()
    else {
        panic!("parameter-dependent derivative");
    };
    assert_eq!(solution_jacobian, [vec![interval("0", "2")]]);

    // C=1 is deliberately not the inverse of J_x=1/2. The accepted family
    // remains contractive, but the sensitivity needs the Neumann remainder.
    let mut noninverse = registered(&["x/2-p"], &["x"], &["p"]);
    let mut proposal = witness();
    proposal.unknown_domain = vec![interval("-4", "4")];
    proposal.parameter_domain = vec![interval("-1/2", "1/2")];
    noninverse
        .certification_request(certify("noninverse", proposal))
        .unwrap();
    noninverse
        .certification_request(differentiate("noninverse", "noninverse"))
        .unwrap();
    let accepted = noninverse
        .parametric_sensitivity(&sensitivity("noninverse"))
        .unwrap();
    assert_eq!(
        accepted.contraction_bound(),
        &BigRational::new(1.into(), 2.into())
    );
    assert!(accepted.solution_jacobian()[0][0].contains(&BigRational::from_integer(2.into())));
}

#[test]
fn adaptive_cover_must_finish_all_parameters_before_publishing_or_differentiating() {
    let mut session = registered(&["(1+p^2)*x-1"], &["x"], &["p"]);
    let before = session.budget_snapshot();
    assert_eq!(
        session
            .certification_request(adaptive("adaptive", 1))
            .unwrap_err()
            .code(),
        "inconclusive"
    );
    assert!(session.parametric_family(&family("adaptive")).is_none());
    assert_eq!(
        session
            .certification_request(differentiate("adaptive", "early"))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    let failed = session.budget_snapshot();
    assert!(
        failed.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    assert_eq!(failed.verifier_pool, before.verifier_pool);
    let CertificationResponse::ParametricFamilyCertified { leaf_count, .. } = session
        .certification_request(adaptive("adaptive", 128))
        .unwrap()
    else {
        panic!("complete adaptive family");
    };
    assert!(leaf_count > 1);
    assert!(session.budget_snapshot().verifier_pool < failed.verifier_pool);
    let CertificationResponse::ParametricSensitivityEnclosed {
        leaf_count: derivative_leaves,
        ..
    } = session
        .certification_request(differentiate("adaptive", "adaptive"))
        .unwrap()
    else {
        panic!("complete-cover sensitivity");
    };
    assert_eq!(derivative_leaves, leaf_count);
    let accepted = session
        .parametric_sensitivity(&sensitivity("adaptive"))
        .unwrap();
    assert!(accepted.root_enclosure()[0].contains(&BigRational::new(1.into(), 5.into())));
    assert!(accepted.root_enclosure()[0].contains(&BigRational::from_integer(1.into())));
    // Analytic derivative -2p/(1+p^2)^2 must retain p=0,1,2, including the
    // endpoint that a midpoint-only derivative would discard.
    for value in [
        BigRational::from_integer(0.into()),
        BigRational::new((-1).into(), 2.into()),
        BigRational::new((-4).into(), 25.into()),
    ] {
        assert!(accepted.solution_jacobian()[0][0].contains(&value));
    }
    assert_eq!(
        session
            .certification_request(adaptive("adaptive", 128))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn midpoint_roots_singular_endpoints_and_original_parameter_poles_never_become_families() {
    let mut too_wide = witness();
    too_wide.unknown_domain = vec![interval("-1", "1")];
    too_wide.parameter_domain = vec![interval("-2", "2")];
    let mut singular = witness();
    singular.unknown_domain = vec![interval("-1", "5")];
    singular.parameter_domain = vec![interval("0", "1")];
    singular.center = vec!["2".into()];
    singular.preconditioner = vec![vec!["2".into()]];
    for (source, proposal) in [
        ("x-p", too_wide),
        ("p*x-1", singular),
        ("x-p+0*(1/p)", witness()),
        ("x-p+(1/p)^0", witness()),
    ] {
        let mut session = registered(&[source], &["x"], &["p"]);
        assert!(
            session
                .certification_request(certify("invalid", proposal))
                .is_err(),
            "{source}"
        );
        assert!(
            session.parametric_family(&family("invalid")).is_none(),
            "{source}"
        );
        assert_eq!(
            session
                .certification_request(differentiate("invalid", "invalid"))
                .unwrap_err()
                .code(),
            "unknown_object"
        );
        assert!(
            session
                .parametric_sensitivity(&sensitivity("invalid"))
                .is_none()
        );
    }
}

#[test]
fn fixed_parameter_boxes_are_valid_but_unknown_boundary_roots_are_not() {
    let mut session = registered(&["x-p"], &["x"], &["p"]);
    let mut proposal = witness();
    proposal.unknown_domain = vec![interval("1", "2")];
    proposal.parameter_domain = vec![interval("2", "2")];
    proposal.center = vec!["3/2".into()];
    assert_eq!(
        session
            .certification_request(certify("point", proposal.clone()))
            .unwrap_err()
            .code(),
        "refused"
    );
    assert!(session.parametric_family(&family("point")).is_none());
    proposal.unknown_domain = vec![interval("1", "3")];
    let CertificationResponse::ParametricFamilyCertified {
        root_enclosure,
        parameter_domain,
        ..
    } = session
        .certification_request(certify("point", proposal))
        .unwrap()
    else {
        panic!("point parameter family");
    };
    assert_eq!(root_enclosure, [interval("2", "2")]);
    assert_eq!(parameter_domain, [interval("2", "2")]);
}

#[test]
fn typed_groups_and_exact_witness_scalars_fail_before_any_acceptance() {
    assert!(ParametricProblemId::new("system:source").is_err());
    assert!(ParametricFamilyId::new("certificate:source").is_err());
    assert!(ParametricSensitivityId::new("family:source").is_err());
    assert!(ParametricFamilyId::new("family:").is_err());
    for (residuals, unknowns, parameters) in [
        (vec!["x-p"], vec!["x"], Vec::new()),
        (vec!["x-p"], vec!["x"], vec!["x"]),
        (vec!["x-p", "x+p"], vec!["x"], vec!["p"]),
        (vec!["x-p"], vec!["x"], vec!["p", "p"]),
    ] {
        let mut session = Session::with_budget_limits(limits(), 256);
        assert!(
            session
                .certification_request(CertificationRequest::RegisterParametricProblem {
                    schema_version: 1,
                    problem: problem(),
                    residuals: residuals.into_iter().map(str::to_owned).collect(),
                    unknowns: unknowns.into_iter().map(str::to_owned).collect(),
                    parameters: parameters.into_iter().map(str::to_owned).collect(),
                })
                .is_err()
        );
        assert!(session.parametric_problem(&problem()).is_none());
        register(&mut session, &["x-p"], &["x"], &["p"]);
    }
    let mut session = registered(&["x-p"], &["x"], &["p"]);
    for scalar in ["NaN", "1e0", "1/0", " 0", "0+0"] {
        let mut proposal = witness();
        proposal.center[0] = scalar.into();
        assert_eq!(
            session
                .certification_request(certify("scalar", proposal))
                .unwrap_err()
                .code(),
            "invalid_input"
        );
        assert!(session.parametric_family(&family("scalar")).is_none());
    }
    session
        .certification_request(certify("scalar", witness()))
        .unwrap();
}

#[test]
fn family_and_sensitivity_share_one_cumulative_protected_allowance() {
    let mut measured = registered(&["x-p"], &["x"], &["p"]);
    let before = measured.budget_snapshot();
    measured
        .certification_request(certify("first", witness()))
        .unwrap();
    let after_family = measured.budget_snapshot();
    measured
        .certification_request(differentiate("first", "first"))
        .unwrap();
    let after_sensitivity = measured.budget_snapshot();
    assert!(after_family.verifier_pool < before.verifier_pool);
    assert!(after_sensitivity.verifier_pool < after_family.verifier_pool);
    let mut bounded = limits();
    bounded.verifier_pool = before.verifier_pool - after_sensitivity.verifier_pool;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, &["x-p"], &["x"], &["p"]);
    session
        .certification_request(certify("first", witness()))
        .unwrap();
    session
        .certification_request(differentiate("first", "first"))
        .unwrap();
    assert_eq!(session.budget_snapshot().verifier_pool, 0);
    assert_eq!(
        session
            .certification_request(differentiate("first", "second"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(session.parametric_family(&family("first")).is_some());
    assert!(
        session
            .parametric_sensitivity(&sensitivity("first"))
            .is_some()
    );
    assert!(
        session
            .parametric_sensitivity(&sensitivity("second"))
            .is_none()
    );
    assert!(session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
}

#[test]
fn ordinary_cover_exhaustion_cannot_borrow_verifier_allowance() {
    let measured = registered(&["x-p"], &["x"], &["p"]);
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - measured.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let mut bounded = limits();
    bounded.dimensions[Dimension::ComputeSteps.index()] = spent + 20;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, &["x-p"], &["x"], &["p"]);
    let before = session.budget_snapshot();
    let request = CertificationRequest::CertifyParametricCover {
        schema_version: 1,
        problem: problem(),
        family: family("exhausted"),
        unknown_domain: vec![interval("-2", "2")],
        parameter_domain: vec![interval("-1", "1")],
        precision_digits: 16,
        max_leaves: 1,
    };
    assert_eq!(
        session.certification_request(request).unwrap_err().code(),
        "budget_exhausted"
    );
    assert_eq!(
        session.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
    assert!(session.parametric_family(&family("exhausted")).is_none());
}

#[test]
fn output_refusal_cannot_publish_a_completed_family_or_sensitivity() {
    for include_sensitivity in [false, true] {
        let mut measured = registered(&["x-p"], &["x"], &["p"]);
        measured
            .certification_request(certify("publication", witness()))
            .unwrap();
        if include_sensitivity {
            measured
                .certification_request(differentiate("publication", "publication"))
                .unwrap();
        }
        let after = measured.budget_snapshot();
        let spent = limits().dimensions[Dimension::ComputeSteps.index()]
            - after.dimensions[Dimension::ComputeSteps.index()];
        let mut bounded = limits();
        bounded.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
        let mut session = Session::with_budget_limits(bounded, 256);
        register(&mut session, &["x-p"], &["x"], &["p"]);
        let error = if include_sensitivity {
            session
                .certification_request(certify("publication", witness()))
                .unwrap();
            session
                .certification_request(differentiate("publication", "publication"))
                .unwrap_err()
        } else {
            session
                .certification_request(certify("publication", witness()))
                .unwrap_err()
        };
        assert_eq!(error.code(), "budget_exhausted");
        assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
        assert_eq!(
            session.parametric_family(&family("publication")).is_some(),
            include_sensitivity
        );
        assert!(
            session
                .parametric_sensitivity(&sensitivity("publication"))
                .is_none()
        );
    }
}
