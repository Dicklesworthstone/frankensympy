//! Session publication and budget boundaries for source-bound objectives.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval, ExactRootWitness,
    ObjectiveId, Session, StationaryPointClassification, StationaryPointId,
};
use fsym_budget::Dimension;

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(2_000_000, 10_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits
}

fn session() -> Session {
    Session::with_budget_limits(limits(), 256)
}

fn objective() -> ObjectiveId {
    ObjectiveId::new("objective:quadratic").unwrap()
}

fn certificate(name: &str) -> StationaryPointId {
    StationaryPointId::new(format!("stationary:{name}")).unwrap()
}

fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}

fn register(session: &mut Session, source: &str, variables: &[&str]) {
    session
        .certification_request(CertificationRequest::RegisterObjective {
            schema_version: 1,
            objective: objective(),
            source: source.into(),
            variables: variables.iter().map(|name| (*name).into()).collect(),
        })
        .unwrap();
}

fn witness() -> ExactRootWitness {
    ExactRootWitness {
        domain: vec![interval("-1", "1")],
        center: vec!["0".into()],
        preconditioner: vec![vec!["1/2".into()]],
        precision_digits: 24,
    }
}

fn certify(name: &str) -> CertificationRequest {
    CertificationRequest::CertifyStationaryPoint {
        schema_version: 1,
        objective: objective(),
        certificate: certificate(name),
        witness: witness(),
    }
}

#[test]
fn registered_objective_encloses_exact_derivatives_in_declared_order() {
    let mut session = session();
    register(&mut session, "x^2+3*x*y+2*y^2", &["y", "x"]);
    let response = session
        .certification_request(CertificationRequest::EncloseObjective {
            schema_version: 1,
            objective: objective(),
            domain: vec![interval("3", "3"), interval("2", "2")],
            precision_digits: 24,
        })
        .unwrap();
    let CertificationResponse::ObjectiveEnclosed {
        value,
        gradient,
        hessian,
        evidence,
        claim,
        ..
    } = response
    else {
        panic!("objective enclosure");
    };
    assert_eq!(value, interval("40", "40"));
    assert_eq!(gradient, vec![interval("18", "18"), interval("13", "13")]);
    assert_eq!(
        hessian,
        vec![
            vec![interval("4", "4"), interval("3", "3")],
            vec![interval("3", "3"), interval("2", "2")],
        ]
    );
    assert_eq!(evidence, "certified_numeric");
    assert_eq!(claim, "whole_box_c2_objective_gradient_hessian_enclosure");
}

#[test]
fn stationary_receipt_is_source_bound_and_never_a_global_or_residual_root_claim() {
    let mut session = session();
    register(&mut session, "x^2+7", &["x"]);
    session.construct("x", "5").unwrap();
    let response = session.certification_request(certify("minimum")).unwrap();
    let CertificationResponse::StationaryPointCertified {
        evidence,
        claim,
        classification,
        stationary_enclosure,
        ..
    } = response
    else {
        panic!("stationary receipt");
    };
    assert_eq!(evidence, "certificate_verified");
    assert_eq!(claim, "unique_nonsingular_stationary_point_in_box");
    assert_eq!(classification, StationaryPointClassification::StrictMinimum);
    assert_eq!(stationary_enclosure, vec![interval("0", "0")]);
    let accepted = session.stationary_point(&certificate("minimum")).unwrap();
    assert_eq!(
        accepted.objective(),
        session.objective_problem(&objective()).unwrap().objective()
    );
    assert!(ObjectiveId::new("system:quadratic").is_err());
    assert!(StationaryPointId::new("certificate:minimum").is_err());
}

#[test]
fn failed_witness_retains_work_but_leaves_id_free_for_a_valid_retry() {
    let mut session = session();
    register(&mut session, "x^2", &["x"]);
    let before = session.budget_snapshot();
    let mut invalid = witness();
    invalid.preconditioner[0][0] = "0".into();
    let error = session
        .certification_request(CertificationRequest::CertifyStationaryPoint {
            schema_version: 1,
            objective: objective(),
            certificate: certificate("retry"),
            witness: invalid,
        })
        .unwrap_err();
    assert_eq!(error.code(), "refused");
    assert!(session.stationary_point(&certificate("retry")).is_none());
    assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    session.certification_request(certify("retry")).unwrap();
    assert_eq!(
        session
            .certification_request(certify("retry"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn protected_stationary_allowance_cannot_borrow_or_reset_ordinary_work() {
    let mut measured = session();
    register(&mut measured, "x^2", &["x"]);
    let before = measured.budget_snapshot();
    measured.certification_request(certify("first")).unwrap();
    let spent = before.verifier_pool - measured.budget_snapshot().verifier_pool;
    assert!(spent > 0);
    let mut bounded = limits();
    bounded.verifier_pool = spent;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, "x^2", &["x"]);
    session.certification_request(certify("first")).unwrap();
    assert_eq!(session.budget_snapshot().verifier_pool, 0);
    assert_eq!(
        session
            .certification_request(certify("second"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(session.stationary_point(&certificate("first")).is_some());
    assert!(session.stationary_point(&certificate("second")).is_none());
    assert!(session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
}

#[test]
fn failed_output_admission_after_verification_cannot_publish_a_stationary_receipt() {
    let mut measured = session();
    register(&mut measured, "x^2", &["x"]);
    measured
        .certification_request(certify("publication"))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut bounded = limits();
    bounded.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, "x^2", &["x"]);
    assert_eq!(
        session
            .certification_request(certify("publication"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
    assert!(
        session
            .stationary_point(&certificate("publication"))
            .is_none()
    );
}

#[test]
fn unknown_sources_invalid_scalars_and_hidden_domains_fail_closed() {
    let mut session = session();
    assert_eq!(
        session
            .certification_request(certify("unknown"))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    register(&mut session, "x^2+0*(1/x)", &["x"]);
    assert_eq!(
        session
            .certification_request(certify("undefined"))
            .unwrap_err()
            .code(),
        "domain_not_certified"
    );
    assert!(
        session
            .stationary_point(&certificate("undefined"))
            .is_none()
    );
    for scalar in ["NaN", "1e0", "1/0", " 1", "1+1"] {
        let mut invalid = witness();
        invalid.center[0] = scalar.into();
        assert_eq!(
            session
                .certification_request(CertificationRequest::CertifyStationaryPoint {
                    schema_version: 1,
                    objective: objective(),
                    certificate: certificate("scalar"),
                    witness: invalid,
                })
                .unwrap_err()
                .code(),
            "invalid_input"
        );
    }
}
