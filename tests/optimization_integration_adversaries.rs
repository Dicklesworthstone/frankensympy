//! Additional adversaries for the public global-optimization acceptance boundary.
//!
//! These cases exercise the cumulative Session ledger and receiver-owned
//! statement around the independent checker, rather than repeating search tests.

use frankensympy::{
    ArtifactExport, ArtifactKind, ArtifactTransferId, BoxExtremumKind, BoxOptimizationId,
    BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval, ObjectiveId, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Dimension, Unbounded};
use std::collections::HashMap;

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(20_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits
}

fn objective() -> ObjectiveId {
    ObjectiveId::new("objective:adversary").unwrap()
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

fn register(session: &mut Session, source: &str, axes: &[&str]) {
    session
        .certification_request(CertificationRequest::RegisterObjective {
            schema_version: 1,
            objective: objective(),
            source: source.into(),
            variables: axes.iter().map(|axis| (*axis).into()).collect(),
        })
        .unwrap();
}

fn registered(source: &str, axes: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    register(&mut session, source, axes);
    session
}

fn optimize(name: &str, dimensions: usize) -> CertificationRequest {
    CertificationRequest::OptimizeBox {
        schema_version: 1,
        objective: objective(),
        optimization: optimization(name),
        domain: vec![interval("-1", "1"); dimensions],
        extremum: BoxExtremumKind::Minimum,
        absolute_tolerance: "1/1000".into(),
        precision_digits: 16,
        max_leaves: 1,
    }
}

fn artifact(source: &str, axes: &[&str]) -> String {
    let mut session = registered(source, axes);
    session
        .certification_request(optimize("sender", axes.len()))
        .unwrap();
    let CertificationResponse::BoxOptimizationExported { artifact, .. } = session
        .certification_request(CertificationRequest::ExportBoxOptimization {
            schema_version: 1,
            optimization: optimization("sender"),
        })
        .unwrap()
    else {
        panic!("optimization export response");
    };
    artifact
}

fn replay(name: &str, dimensions: usize, artifact: &str) -> CertificationRequest {
    CertificationRequest::ReplayBoxOptimization {
        schema_version: 1,
        objective: objective(),
        optimization: optimization(name),
        domain: vec![interval("-1", "1"); dimensions],
        extremum: BoxExtremumKind::Minimum,
        absolute_tolerance: "1/1000".into(),
        artifact: artifact.into(),
    }
}

#[test]
fn ordinary_exhaustion_cannot_spend_verifier_fuel_or_create_an_accepted_handle() {
    let measured = registered("x", &["x"]);
    let registration_cost = limits().dimensions[Dimension::ComputeSteps.index()]
        - measured.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let mut bounded = limits();
    bounded.dimensions[Dimension::ComputeSteps.index()] = registration_cost + 20;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, "x", &["x"]);
    let before = session.budget_snapshot();
    assert_eq!(
        session
            .certification_request(optimize("ordinary-exhaustion", 1))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    let after = session.budget_snapshot();
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    assert_eq!(after.verifier_pool, before.verifier_pool);
    assert!(
        session
            .box_optimization(&optimization("ordinary-exhaustion"))
            .is_none()
    );
}

#[test]
fn replay_output_refusal_occurs_after_verification_but_before_publication() {
    let artifact = artifact("x", &["x"]);
    let mut measured = registered("x", &["x"]);
    measured
        .certification_request(replay("publication", 1, &artifact))
        .unwrap();
    let after = measured.budget_snapshot();
    let ordinary_spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut bounded = limits();
    bounded.dimensions[Dimension::ComputeSteps.index()] = ordinary_spent - 1;
    let mut session = Session::with_budget_limits(bounded, 256);
    register(&mut session, "x", &["x"]);
    assert_eq!(
        session
            .certification_request(replay("publication", 1, &artifact))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    // The full independent verification cost was incurred, exactly as in the
    // successful run. Refused response admission must still prevent storage.
    assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
    assert!(
        session
            .box_optimization(&optimization("publication"))
            .is_none()
    );
}

#[test]
fn session_replay_binds_its_own_context_and_ordered_coordinates() {
    let artifact = artifact("x-2*y", &["x", "y"]);
    for (source, axes) in [("x-2*y+1", ["x", "y"]), ("x-2*y", ["y", "x"])] {
        let mut receiver = registered(source, &axes);
        assert_eq!(
            receiver
                .certification_request(replay("changed-statement", 2, &artifact))
                .unwrap_err()
                .code(),
            "source_mismatch"
        );
        assert!(
            receiver
                .box_optimization(&optimization("changed-statement"))
                .is_none()
        );
    }
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(
            HashMap::new(),
            HashMap::new(),
            "independent receiver universe",
        )
        .unwrap();
    let mut receiver = Session::with_context_and_budget_limits(context, limits(), 256);
    register(&mut receiver, "x-2*y", &["x", "y"]);
    assert_eq!(
        receiver
            .certification_request(replay("changed-context", 2, &artifact))
            .unwrap_err()
            .code(),
        "universe_mismatch"
    );
    assert!(
        receiver
            .box_optimization(&optimization("changed-context"))
            .is_none()
    );
    // An independently constructed matching receiver accepts without sharing
    // any sender state. Mutable bindings do not replace the registered source.
    let mut matching = registered("x-2*y", &["x", "y"]);
    matching.construct("x", "100").unwrap();
    let CertificationResponse::BoxOptimized { value, point, .. } = matching
        .certification_request(replay("matching", 2, &artifact))
        .unwrap()
    else {
        panic!("replayed optimization response");
    };
    assert_eq!(value, interval("-3", "-3"));
    assert_eq!(point, vec!["-1", "1"]);
}

#[test]
fn complete_correctly_hashed_wrong_kind_transfer_never_reaches_verification() {
    let bytes = artifact("x", &["x"]).into_bytes();
    // Content integrity is correct for a deliberately different proof kind.
    // This must not become authorization to interpret bytes as an optimum.
    let exported =
        ArtifactExport::new(ArtifactKind::RootCertificate, bytes, &mut Unbounded).unwrap();
    let transfer = ArtifactTransferId::new("transfer:wrong-proof-kind").unwrap();
    let mut receiver = registered("x", &["x"]);
    receiver
        .certification_request(CertificationRequest::BeginArtifactImport {
            schema_version: 1,
            transfer: transfer.clone(),
            manifest: exported.manifest().clone(),
        })
        .unwrap();
    for index in 0..exported.manifest().chunk_count() {
        receiver
            .certification_request(CertificationRequest::AppendArtifactChunk {
                schema_version: 1,
                transfer: transfer.clone(),
                chunk: exported.chunk(index, &mut Unbounded).unwrap(),
            })
            .unwrap();
    }
    let before = receiver.budget_snapshot();
    assert_eq!(
        receiver
            .certification_request(CertificationRequest::ReplayBoxOptimizationFromArtifact {
                schema_version: 1,
                objective: objective(),
                optimization: optimization("wrong-proof-kind"),
                domain: vec![interval("-1", "1")],
                extremum: BoxExtremumKind::Minimum,
                absolute_tolerance: "1/1000".into(),
                transfer,
            })
            .unwrap_err()
            .code(),
        "artifact_kind_mismatch"
    );
    assert_eq!(
        receiver.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
    assert!(
        receiver
            .box_optimization(&optimization("wrong-proof-kind"))
            .is_none()
    );
}
