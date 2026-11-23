//! Host cancellation reaches native generation, independent verification and publication.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, CompiledResidualSystemId, Dimension,
    ExactInterval, ExactRootWitness, MeterError, ResidualSystemId, RootCertificateId,
    RootIsolationId, RootIsolationSearchId, Session,
};

fn session() -> Session {
    Session::with_budget_limits(BudgetLimits::uniform(1 << 40, 1 << 40), 256)
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:cancel").unwrap()
}

fn certificate() -> RootCertificateId {
    RootCertificateId::new("certificate:cancel").unwrap()
}

fn program() -> CompiledResidualSystemId {
    CompiledResidualSystemId::new("program:cancel").unwrap()
}

fn search() -> RootIsolationSearchId {
    RootIsolationSearchId::new("search:cancel").unwrap()
}

fn isolation() -> RootIsolationId {
    RootIsolationId::new("isolation:cancel").unwrap()
}

fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}

fn registration() -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec!["x^2-2".into()],
        variables: vec!["x".into()],
    }
}

fn registered() -> Session {
    let mut session = session();
    session.certification_request(registration()).unwrap();
    session
}

fn certify() -> CertificationRequest {
    CertificationRequest::CertifyRootBox {
        schema_version: 1,
        system: system(),
        certificate: certificate(),
        witness: ExactRootWitness {
            domain: vec![interval("7/5", "3/2")],
            center: vec!["29/20".into()],
            preconditioner: vec![vec!["1/3".into()]],
            precision_digits: 16,
        },
    }
}

fn compile() -> CertificationRequest {
    CertificationRequest::CompileResidualSystem {
        schema_version: 1,
        system: system(),
        program: program(),
    }
}

fn measured_request(session: &mut Session, request: CertificationRequest) -> usize {
    let mut checkpoints = 0;
    session
        .certification_request_with_checkpoint(request, || {
            checkpoints += 1;
            Ok(())
        })
        .unwrap();
    assert!(checkpoints > 0);
    checkpoints
}

fn cancel_at(session: &mut Session, request: CertificationRequest, target: usize) {
    let mut checkpoints = 0;
    let error = session
        .certification_request_with_checkpoint(request, || {
            checkpoints += 1;
            if checkpoints == target {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(error.code(), "cancelled", "{error}");
    assert_eq!(
        checkpoints, target,
        "a terminal checkpoint is never retried"
    );
}

#[test]
fn no_op_host_checkpoint_preserves_results_context_and_cumulative_accounting() {
    let mut ordinary = session();
    let mut supervised = session();
    for request in [registration(), certify(), compile()] {
        let expected = ordinary.certification_request(request.clone()).unwrap();
        let actual = supervised
            .certification_request_with_checkpoint(request, || Ok(()))
            .unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        assert_eq!(ordinary.budget_snapshot(), supervised.budget_snapshot());
        assert_eq!(ordinary.context_digest(), supervised.context_digest());
    }
}

#[test]
fn cancellation_before_work_spends_nothing_and_publishes_nothing() {
    let mut session = session();
    let before = session.budget_snapshot();
    cancel_at(&mut session, registration(), 1);
    assert_eq!(session.budget_snapshot(), before);
    assert!(session.residual_problem(&system()).is_none());

    session.certification_request(registration()).unwrap();
    for request in [certify(), compile()] {
        let before = session.budget_snapshot();
        cancel_at(&mut session, request, 1);
        assert_eq!(session.budget_snapshot(), before);
        assert!(session.root_certificate(&certificate()).is_none());
        assert!(session.compiled_residual_system(&program()).is_none());
    }
}

#[test]
fn registration_remains_atomic_at_every_observed_safe_point() {
    let checkpoints = measured_request(&mut session(), registration());
    for target in 1..=checkpoints {
        let mut session = session();
        cancel_at(&mut session, registration(), target);
        assert!(session.residual_problem(&system()).is_none());
        let after_cancel = session.budget_snapshot();
        session.certification_request(registration()).unwrap();
        assert!(session.residual_problem(&system()).is_some());
        for dimension in Dimension::ALL {
            assert!(
                session.budget_snapshot().dimensions[dimension.index()]
                    <= after_cancel.dimensions[dimension.index()]
            );
        }
    }
}

#[test]
fn cancellation_reaches_protected_verification_and_final_certificate_admission() {
    let mut completed = registered();
    let full_before = completed.budget_snapshot();
    let checkpoints = measured_request(&mut completed, certify());
    let full_verifier_work = full_before.verifier_pool - completed.budget_snapshot().verifier_pool;
    assert!(full_verifier_work > 0);
    let mut stopped_during_verifier_work = false;
    for target in [
        1,
        checkpoints / 8,
        checkpoints / 4,
        checkpoints / 2,
        checkpoints * 3 / 4,
        checkpoints * 7 / 8,
        checkpoints - 1,
        checkpoints,
    ] {
        let mut session = registered();
        let before = session.budget_snapshot();
        cancel_at(&mut session, certify(), target.max(1));
        let after = session.budget_snapshot();
        let verifier_work = before.verifier_pool - after.verifier_pool;
        stopped_during_verifier_work |= verifier_work > 0 && verifier_work < full_verifier_work;
        assert!(session.root_certificate(&certificate()).is_none());
        assert!(after.verifier_pool <= before.verifier_pool);
        // The same ID can be retried under a fresh host request. Past work is
        // retained on the ledger and the independent verifier runs again.
        session.certification_request(certify()).unwrap();
        assert!(session.root_certificate(&certificate()).is_some());
        assert!(session.budget_snapshot().verifier_pool < after.verifier_pool);
    }
    assert!(
        stopped_during_verifier_work,
        "cancellation must interrupt protected computation, not only final response admission"
    );
}

#[test]
fn late_compilation_cancellation_keeps_program_unpublished_and_existing_proofs_intact() {
    let checkpoints = measured_request(&mut registered(), compile());
    for target in [checkpoints / 2, checkpoints - 1, checkpoints] {
        let mut session = registered();
        session.certification_request(certify()).unwrap();
        let old_certificate = session.root_certificate(&certificate()).unwrap().clone();
        let before = session.budget_snapshot();
        cancel_at(&mut session, compile(), target.max(1));
        assert!(session.compiled_residual_system(&program()).is_none());
        assert_eq!(
            session.root_certificate(&certificate()),
            Some(&old_certificate)
        );
        let after = session.budget_snapshot();
        assert!(
            after.dimensions[Dimension::ComputeSteps.index()]
                < before.dimensions[Dimension::ComputeSteps.index()]
        );
        session.certification_request(compile()).unwrap();
        assert!(session.compiled_residual_system(&program()).is_some());
        assert!(session.budget_snapshot().verifier_pool < after.verifier_pool);
    }
}

#[test]
fn interleaved_exact_solver_lanes_preserve_typed_cancellation_and_protected_accounting() {
    for request in [
        CertificationRequest::SolveRealEquation {
            schema_version: 1,
            system: system(),
        },
        CertificationRequest::SolveRealPolynomialSystem {
            schema_version: 1,
            system: system(),
        },
        CertificationRequest::SolveRealRationalSystem {
            schema_version: 1,
            system: system(),
        },
    ] {
        let checkpoints = measured_request(&mut registered(), request.clone());
        for target in [(checkpoints / 2).max(1), checkpoints] {
            let mut session = registered();
            let before = session.budget_snapshot();
            cancel_at(&mut session, request.clone(), target);
            let after = session.budget_snapshot();
            assert!(after.verifier_pool <= before.verifier_pool);
            assert!(session.residual_problem(&system()).is_some());
            // Solver implementations that translate errors through strings
            // must still return the original typed cancellation, never a
            // mathematical refusal or a successful partial solution set.
            let resumed = session.certification_request(request.clone()).unwrap();
            let resumed = serde_json::to_value(resumed).unwrap();
            assert_eq!(resumed["completeness"], "complete");
            assert!(session.budget_snapshot().verifier_pool < after.verifier_pool);
        }
    }
}

fn started_search() -> Session {
    let mut session = registered();
    session
        .certification_request(CertificationRequest::BeginRootIsolationSearch {
            schema_version: 1,
            system: system(),
            search: search(),
            domain: vec![interval("-2", "2")],
            precision_digits: 16,
            max_leaves: 128,
        })
        .unwrap();
    session.certification_request(advance(1)).unwrap();
    session
}

fn advance(max_cells: usize) -> CertificationRequest {
    CertificationRequest::AdvanceRootIsolationSearch {
        schema_version: 1,
        search: search(),
        max_cells,
    }
}

#[test]
fn cancelled_search_retains_prior_cells_and_resumes_without_budget_reset() {
    let checkpoints = measured_request(&mut started_search(), advance(2));
    let mut session = started_search();
    let before = session.budget_snapshot();
    let prior = session.root_isolation_search_progress(&search()).unwrap();
    assert!(prior.processed_cells > 0);
    assert!(!prior.ready_for_verification);
    cancel_at(&mut session, advance(2), (checkpoints / 2).max(1));
    let after = session.budget_snapshot();
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    assert_eq!(after.verifier_pool, before.verifier_pool);
    assert!(session.root_isolation(&isolation()).is_none());
    let retained = session.root_isolation_search_progress(&search()).unwrap();
    assert!(retained.processed_cells >= prior.processed_cells);

    for _ in 0..128 {
        if session
            .root_isolation_search_progress(&search())
            .unwrap()
            .ready_for_verification
        {
            break;
        }
        session
            .certification_request_with_checkpoint(advance(3), || Ok(()))
            .unwrap();
    }
    assert!(
        session
            .root_isolation_search_progress(&search())
            .unwrap()
            .ready_for_verification
    );
    assert!(session.root_isolation(&isolation()).is_none());
    assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
    let response = session
        .certification_request_with_checkpoint(
            CertificationRequest::FinishRootIsolationSearch {
                schema_version: 1,
                search: search(),
                isolation: isolation(),
            },
            || Ok(()),
        )
        .unwrap();
    assert!(matches!(
        response,
        CertificationResponse::RealRootsIsolated { root_count: 2, .. }
    ));
    assert!(session.root_isolation(&isolation()).is_some());
    assert!(session.budget_snapshot().verifier_pool < after.verifier_pool);
}
