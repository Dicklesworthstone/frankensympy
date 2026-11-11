//! Global optimization through the cumulative, atomically published session.

use frankensympy::{
    ArtifactKind, ArtifactSource, ArtifactTransferId, BigRational, BoxExtremumKind,
    BoxOptimizationId, BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval,
    ObjectiveId, Session,
};
use fsym_budget::Dimension;

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(20_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits
}

fn objective() -> ObjectiveId {
    ObjectiveId::new("objective:source").unwrap()
}

fn optimization(name: &str) -> BoxOptimizationId {
    BoxOptimizationId::new(format!("optimization:{name}")).unwrap()
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

fn registered(source: &str, variables: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    register(&mut session, source, variables);
    session
}

fn minimize(name: &str, lower: &str, upper: &str) -> CertificationRequest {
    CertificationRequest::OptimizeBox {
        schema_version: 1,
        objective: objective(),
        optimization: optimization(name),
        domain: vec![interval(lower, upper)],
        extremum: BoxExtremumKind::Minimum,
        absolute_tolerance: "1/1000".into(),
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn export(session: &mut Session, name: &str) -> String {
    let CertificationResponse::BoxOptimizationExported { artifact, .. } = session
        .certification_request(CertificationRequest::ExportBoxOptimization {
            schema_version: 1,
            optimization: optimization(name),
        })
        .unwrap()
    else {
        panic!("optimization artifact");
    };
    artifact
}

#[test]
fn interior_minimum_freezes_source_and_exposes_a_global_value_claim() {
    let mut session = registered("x^2+7", &["x"]);
    session.construct("x", "100").unwrap();
    let before = session.budget_snapshot();
    let CertificationResponse::BoxOptimized {
        evidence,
        claim,
        extremum,
        domain,
        point,
        value,
        point_value,
        gap,
        ..
    } = session
        .certification_request(minimize("interior", "-1", "1"))
        .unwrap()
    else {
        panic!("global optimization response");
    };
    assert_eq!(evidence, "certified_numeric");
    assert_eq!(claim, "global_box_optimum_with_absolute_value_gap_bound");
    assert_eq!(extremum, BoxExtremumKind::Minimum);
    assert_eq!(domain, vec![interval("-1", "1")]);
    assert_eq!(point, vec!["0"]);
    assert_eq!(value, interval("7", "7"));
    assert_eq!(point_value, value);
    assert_eq!(gap, "0");
    let receipt = session.box_optimization(&optimization("interior")).unwrap();
    assert_eq!(
        receipt.objective(),
        session.objective_problem(&objective()).unwrap().objective()
    );
    let after = session.budget_snapshot();
    assert!(after.verifier_pool < before.verifier_pool);
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
}

#[test]
fn boundary_maximum_and_fixed_coordinates_are_supported() {
    let mut session = registered("(x-3)^2+y", &["x", "y"]);
    let response = session
        .certification_request(CertificationRequest::OptimizeBox {
            schema_version: 1,
            objective: objective(),
            optimization: optimization("boundary"),
            domain: vec![interval("3", "3"), interval("-2", "5")],
            extremum: BoxExtremumKind::Maximum,
            absolute_tolerance: "1/1000".into(),
            precision_digits: 16,
            max_leaves: 64,
        })
        .unwrap();
    let CertificationResponse::BoxOptimized {
        extremum,
        point,
        value,
        gap,
        ..
    } = response
    else {
        panic!("maximum response");
    };
    assert_eq!(extremum, BoxExtremumKind::Maximum);
    assert_eq!(point, vec!["3", "5"]);
    assert_eq!(value, interval("5", "5"));
    assert_eq!(gap, "0");
    let artifact = export(&mut session, "boundary");
    let mut receiver = registered("(x-3)^2+y", &["x", "y"]);
    receiver
        .certification_request(CertificationRequest::ReplayBoxOptimization {
            schema_version: 1,
            objective: objective(),
            optimization: optimization("received"),
            domain: vec![interval("3", "3"), interval("-2", "5")],
            extremum: BoxExtremumKind::Maximum,
            absolute_tolerance: "1/1000".into(),
            artifact,
        })
        .unwrap();
    assert_eq!(
        receiver.box_optimization(&optimization("received")),
        session.box_optimization(&optimization("boundary"))
    );
}

#[test]
fn nonconvex_irrational_optimum_gets_a_proved_value_gap() {
    let mut session = registered("(x^2-2)^2", &["x"]);
    session
        .certification_request(minimize("nonconvex", "-2", "2"))
        .unwrap();
    let receipt = session
        .box_optimization(&optimization("nonconvex"))
        .unwrap();
    let zero = BigRational::from_integer(0.into());
    assert!(receipt.value().lower() <= zero && receipt.value().upper() >= zero);
    assert!(receipt.gap() <= &BigRational::new(1.into(), 1000.into()));
    assert!(receipt.leaf_count() > 1);
    assert!(receipt.point_value().lower() >= zero);
}

#[test]
fn incomplete_search_does_not_publish_and_retry_retains_consumed_work() {
    let mut session = registered("(x^2-2)^2", &["x"]);
    let mut request = minimize("retry", "1", "2");
    if let CertificationRequest::OptimizeBox { max_leaves, .. } = &mut request {
        *max_leaves = 1;
    }
    let before = session.budget_snapshot();
    assert_eq!(
        session.certification_request(request).unwrap_err().code(),
        "inconclusive"
    );
    assert!(session.box_optimization(&optimization("retry")).is_none());
    let failed = session.budget_snapshot();
    assert!(
        failed.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    assert_eq!(failed.verifier_pool, before.verifier_pool);
    session
        .certification_request(minimize("retry", "1", "2"))
        .unwrap();
    assert_eq!(
        session
            .certification_request(minimize("retry", "1", "2"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn original_undefined_domains_cannot_be_hidden_or_published() {
    for source in ["x^2+0*(1/x)", "log(x)-log(x)", "sqrt(x)"] {
        let mut session = registered(source, &["x"]);
        assert!(
            session
                .certification_request(minimize("invalid", "-1", "1"))
                .is_err(),
            "{source}"
        );
        assert!(session.box_optimization(&optimization("invalid")).is_none());
    }
}

#[test]
fn invalid_tolerances_boxes_and_typed_ids_refuse() {
    assert!(BoxOptimizationId::new("stationary:wrong").is_err());
    assert!(BoxOptimizationId::new("optimization:").is_err());
    for tolerance in ["0", "-1", "1/0", "NaN"] {
        let mut session = registered("x^2", &["x"]);
        let mut request = minimize("invalid", "-1", "1");
        if let CertificationRequest::OptimizeBox {
            absolute_tolerance, ..
        } = &mut request
        {
            *absolute_tolerance = tolerance.into();
        }
        assert_eq!(
            session.certification_request(request).unwrap_err().code(),
            "invalid_input"
        );
        assert!(session.box_optimization(&optimization("invalid")).is_none());
    }
    let mut session = registered("x^2", &["x"]);
    assert_eq!(
        session
            .certification_request(minimize("box", "1", "-1"))
            .unwrap_err()
            .code(),
        "invalid_input"
    );
}

#[test]
fn zero_verifier_pool_rejects_after_search_and_keeps_the_label_unpublished() {
    let mut bounded = limits();
    bounded.verifier_pool = 0;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, "x^2", &["x"]);
    let before = session.budget_snapshot();
    assert_eq!(
        session
            .certification_request(minimize("unverified", "-1", "1"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(
        session
            .box_optimization(&optimization("unverified"))
            .is_none()
    );
    assert!(
        session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
}

#[test]
fn protected_allowance_cannot_borrow_from_ordinary_work_or_reset_between_requests() {
    let mut measured = registered("x^2", &["x"]);
    let before = measured.budget_snapshot();
    measured
        .certification_request(minimize("first", "-1", "1"))
        .unwrap();
    let spent = before.verifier_pool - measured.budget_snapshot().verifier_pool;
    assert!(spent > 0);
    let mut bounded = limits();
    bounded.verifier_pool = spent;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, "x^2", &["x"]);
    session
        .certification_request(minimize("first", "-1", "1"))
        .unwrap();
    assert_eq!(session.budget_snapshot().verifier_pool, 0);
    assert_eq!(
        session
            .certification_request(minimize("second", "-1", "1"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(session.box_optimization(&optimization("first")).is_some());
    assert!(session.box_optimization(&optimization("second")).is_none());
}

#[test]
fn failed_final_response_admission_cannot_publish_a_verified_optimization() {
    let mut measured = registered("x^2", &["x"]);
    measured
        .certification_request(minimize("publication", "-1", "1"))
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
            .certification_request(minimize("publication", "-1", "1"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
    assert!(
        session
            .box_optimization(&optimization("publication"))
            .is_none()
    );
}

#[test]
fn chunk_transfer_grants_no_evidence_until_independent_statement_bound_replay() {
    let mut sender = registered("x^2", &["x"]);
    sender
        .certification_request(minimize("sent", "-1", "1"))
        .unwrap();
    let send = ArtifactTransferId::new("transfer:send").unwrap();
    let receive = ArtifactTransferId::new("transfer:receive").unwrap();
    let CertificationResponse::ArtifactExportStarted { manifest, .. } = sender
        .certification_request(CertificationRequest::BeginArtifactExport {
            schema_version: 1,
            transfer: send.clone(),
            source: ArtifactSource::BoxOptimization(optimization("sent")),
        })
        .unwrap()
    else {
        panic!("export manifest");
    };
    assert_eq!(manifest.kind, ArtifactKind::BoxOptimization);
    let chunks = manifest.chunk_count();
    let mut receiver = registered("x^2", &["x"]);
    receiver
        .certification_request(CertificationRequest::BeginArtifactImport {
            schema_version: 1,
            transfer: receive.clone(),
            manifest,
        })
        .unwrap();
    let replay = CertificationRequest::ReplayBoxOptimizationFromArtifact {
        schema_version: 1,
        objective: objective(),
        optimization: optimization("received"),
        domain: vec![interval("-1", "1")],
        extremum: BoxExtremumKind::Minimum,
        absolute_tolerance: "1/1000".into(),
        transfer: receive.clone(),
    };
    assert_eq!(
        receiver
            .certification_request(replay.clone())
            .unwrap_err()
            .code(),
        "incomplete_transfer"
    );
    for index in 0..chunks {
        let CertificationResponse::ArtifactChunkRead { chunk, .. } = sender
            .certification_request(CertificationRequest::ReadArtifactChunk {
                schema_version: 1,
                transfer: send.clone(),
                index,
            })
            .unwrap()
        else {
            panic!("chunk");
        };
        receiver
            .certification_request(CertificationRequest::AppendArtifactChunk {
                schema_version: 1,
                transfer: receive.clone(),
                chunk,
            })
            .unwrap();
    }
    assert!(
        receiver
            .box_optimization(&optimization("received"))
            .is_none()
    );
    let mut altered = replay.clone();
    if let CertificationRequest::ReplayBoxOptimizationFromArtifact {
        absolute_tolerance, ..
    } = &mut altered
    {
        *absolute_tolerance = "1/100".into();
    }
    assert_eq!(
        receiver.certification_request(altered).unwrap_err().code(),
        "statement_mismatch"
    );
    assert!(
        receiver
            .box_optimization(&optimization("received"))
            .is_none()
    );
    receiver.certification_request(replay).unwrap();
    assert_eq!(
        receiver.box_optimization(&optimization("received")),
        sender.box_optimization(&optimization("sent"))
    );
}
