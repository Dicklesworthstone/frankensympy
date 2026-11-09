//! Resumable candidate searches share one ledger and accept only on independent finish.

use frankensympy::artifact_stream::{ArtifactChunk, ArtifactManifest, ArtifactTransferId};
use frankensympy::certification_session::{
    ArtifactSource, CertificationSessionError, MAX_SESSION_ROOT_SEARCHES, RootIsolationSearchId,
};
use frankensympy::{
    BudgetLimits, CertificationError, CertificationRequest, CertificationResponse, ExactInterval,
    ExactRootWitness, ResidualSystemId, RootCertificateId, RootIsolationId, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Dimension;
use fsym_calculus::root_isolation::{MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationProgress};
use fsym_core::{BallError, RootBoxError, RootIsolationError};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits.dimensions[Dimension::DepthLimit.index()] = 256;
    limits
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:equations").unwrap()
}

fn search(name: &str) -> RootIsolationSearchId {
    RootIsolationSearchId::new(format!("search:{name}")).unwrap()
}

fn isolation(name: &str) -> RootIsolationId {
    RootIsolationId::new(format!("isolation:{name}")).unwrap()
}

fn child(name: &str, index: usize) -> RootCertificateId {
    RootCertificateId::new(format!("certificate:{name}.root-{index}")).unwrap()
}

fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}

fn domain() -> Vec<ExactInterval> {
    vec![interval("-2", "2")]
}

fn registration(source: &str) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec![source.into()],
        variables: vec!["x".into()],
    }
}

fn registered(source: &str, limits: BudgetLimits) -> Session {
    let mut session = Session::with_budget_limits(limits, 256);
    session.certification_request(registration(source)).unwrap();
    session
}

fn begin(name: &str, max_leaves: usize) -> CertificationRequest {
    CertificationRequest::BeginRootIsolationSearch {
        schema_version: 1,
        system: system(),
        search: search(name),
        domain: domain(),
        precision_digits: 16,
        max_leaves,
    }
}

fn advance(name: &str, max_cells: usize) -> CertificationRequest {
    CertificationRequest::AdvanceRootIsolationSearch {
        schema_version: 1,
        search: search(name),
        max_cells,
    }
}

fn finish(name: &str, output: &str) -> CertificationRequest {
    CertificationRequest::FinishRootIsolationSearch {
        schema_version: 1,
        search: search(name),
        isolation: isolation(output),
    }
}

fn release(name: &str) -> CertificationRequest {
    CertificationRequest::ReleaseRootIsolationSearch {
        schema_version: 1,
        search: search(name),
    }
}

fn progress(session: &Session, name: &str) -> RootIsolationProgress {
    session
        .root_isolation_search_progress(&search(name))
        .unwrap()
}

fn run_to_ready(session: &mut Session, name: &str, quantum: usize) -> RootIsolationProgress {
    for _ in 0..=MAX_ROOT_ISOLATION_ADVANCE_CELLS {
        let before = progress(session, name);
        if before.ready_for_verification {
            assert_eq!(before.pending_cells, 0);
            return before;
        }
        let reply = session
            .certification_request(advance(name, quantum))
            .unwrap();
        let CertificationResponse::RootIsolationSearchProgress {
            progress: after,
            evidence,
            ..
        } = reply
        else {
            panic!("candidate progress response");
        };
        assert_eq!(evidence, "heuristic_candidate");
        assert!(after.processed_cells > before.processed_cells);
        assert!(after.processed_cells - before.processed_cells <= quantum);
        assert_eq!(after, progress(session, name));
    }
    panic!("the bounded search did not finish");
}

fn ready_session(source: &str, name: &str, limits: BudgetLimits) -> Session {
    let mut session = registered(source, limits);
    session.certification_request(begin(name, 512)).unwrap();
    run_to_ready(&mut session, name, 3);
    session
}

fn unpublished(session: &Session, name: &str) {
    assert!(session.root_isolation(&isolation(name)).is_none());
    assert!(session.root_certificate(&child(name, 0)).is_none());
    assert!(session.root_certificate(&child(name, 1)).is_none());
}

fn progress_response_bytes(name: &str, session: &Session) -> u64 {
    serde_json::to_vec(&CertificationResponse::RootIsolationSearchProgress {
        schema_version: 1,
        system: system(),
        search: search(name),
        evidence: "heuristic_candidate",
        progress: RootIsolationProgress {
            processed_cells: usize::MAX,
            pending_cells: usize::MAX,
            candidate_root_count: usize::MAX,
            excluded_cells: usize::MAX,
            leaf_count: usize::MAX,
            ready_for_verification: false,
        },
        context_digest: session.context_digest(),
    })
    .unwrap()
    .len() as u64
}

#[test]
fn progress_is_candidate_only_and_search_never_borrows_protected_fuel() {
    for (source, candidate_roots) in [("x^2-2", 2), ("x^2+1", 0)] {
        let mut budget = limits();
        budget.verifier_pool = 0;
        let mut session = registered(source, budget);
        let started = session
            .certification_request(begin("pending", 512))
            .unwrap();
        let wire = serde_json::to_value(started).unwrap();
        assert_eq!(wire["kind"], "root_isolation_search_started");
        assert_eq!(wire["evidence"], "heuristic_candidate");
        for absent in ["roots", "root_count", "certificate", "isolation", "claim"] {
            assert!(
                wire.get(absent).is_none(),
                "unexpected acceptance field {absent}"
            );
        }
        let initial = progress(&session, "pending");
        assert_eq!(initial.processed_cells, 0);
        assert_eq!(initial.pending_cells, 1);
        assert!(!initial.ready_for_verification);
        assert_eq!(
            session
                .certification_request(finish("pending", "premature"))
                .unwrap_err()
                .code(),
            "search_not_ready"
        );
        let ready = run_to_ready(&mut session, "pending", 1);
        assert_eq!(ready.candidate_root_count, candidate_roots);
        assert_eq!(session.budget_snapshot().verifier_pool, 0);
        unpublished(&session, "premature");
        let inspected = session
            .certification_request(CertificationRequest::InspectRootIsolationSearch {
                schema_version: 1,
                search: search("pending"),
            })
            .unwrap();
        let inspected = serde_json::to_value(inspected).unwrap();
        assert_eq!(inspected["evidence"], "heuristic_candidate");
        assert_eq!(inspected["progress"], serde_json::to_value(ready).unwrap());
        assert_eq!(
            session
                .certification_request(finish("pending", "unverified"))
                .unwrap_err()
                .code(),
            "budget_exhausted"
        );
        unpublished(&session, "unverified");
        assert_eq!(progress(&session, "pending"), ready);
        session
            .certification_request(advance("pending", 1))
            .unwrap();
        assert_eq!(progress(&session, "pending"), ready);
    }
}

#[test]
fn search_sources_and_context_remain_frozen_through_workspace_changes_and_replay() {
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(HashMap::new(), HashMap::new(), "search-test-context")
        .unwrap();
    let mut sender = Session::with_context_and_budget_limits(context.clone(), limits(), 256);
    sender.certification_request(registration("x^2-2")).unwrap();
    let original = sender.residual_problem(&system()).unwrap().clone();
    sender.certification_request(begin("frozen", 512)).unwrap();
    sender.construct("x", "100").unwrap();
    sender.fork_branch("changed").unwrap();
    sender.checkout_branch("changed").unwrap();
    sender.construct("x", "0").unwrap();
    assert_eq!(
        sender
            .certification_request(registration("x^2+1"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    run_to_ready(&mut sender, "frozen", 2);
    let accepted = sender
        .certification_request(finish("frozen", "accepted"))
        .unwrap();
    let CertificationResponse::RealRootsIsolated {
        root_count,
        context_digest,
        domain: accepted_domain,
        ..
    } = accepted
    else {
        panic!("verified complete root set");
    };
    assert_eq!(root_count, 2);
    assert_eq!(context_digest, context.digest());
    assert_eq!(accepted_domain, domain());
    let receipt = sender.root_isolation(&isolation("accepted")).unwrap();
    assert_eq!(receipt.residuals(), original.residuals());
    assert_eq!(receipt.variables(), original.variables());
    let exported = sender
        .certification_request(CertificationRequest::ExportRootIsolation {
            schema_version: 1,
            isolation: isolation("accepted"),
        })
        .unwrap();
    let CertificationResponse::RootIsolationExported { artifact, .. } = exported else {
        panic!("complete artifact");
    };
    let replay = || CertificationRequest::ReplayRootIsolation {
        schema_version: 1,
        system: system(),
        isolation: isolation("received"),
        domain: domain(),
        artifact: artifact.clone(),
    };
    let mut foreign = registered("x^2-2", limits());
    assert_eq!(
        foreign.certification_request(replay()).unwrap_err().code(),
        "universe_mismatch"
    );
    unpublished(&foreign, "received");
    let mut receiver = Session::with_context_and_budget_limits(context, limits(), 256);
    receiver
        .certification_request(registration("x^2-2"))
        .unwrap();
    receiver.certification_request(replay()).unwrap();
    assert_eq!(
        receiver.root_isolation(&isolation("received")),
        sender.root_isolation(&isolation("accepted"))
    );
    sender.certification_request(release("frozen")).unwrap();
    assert!(
        sender
            .root_isolation_search_progress(&search("frozen"))
            .is_none()
    );
    assert!(sender.root_isolation(&isolation("accepted")).is_some());
    assert!(sender.root_certificate(&child("accepted", 1)).is_some());
}

#[test]
fn advance_reserves_response_before_work_and_refusals_retain_the_committed_prefix() {
    // A constant excluded cell needs less logical memory than its progress
    // response. Prove the cell could commit if response admission were moved
    // after search work, then leave one byte too little for that response.
    let mut measured = registered("1", limits());
    measured
        .certification_request(begin("bounded", 512))
        .unwrap();
    let before = measured.budget_snapshot();
    let started = limits().dimensions[Dimension::MemoryBytes.index()]
        - before.dimensions[Dimension::MemoryBytes.index()];
    let acknowledgement = progress_response_bytes("bounded", &measured) * 8 + 8192;
    measured
        .certification_request(advance("bounded", 1))
        .unwrap();
    let advance_cost = before.dimensions[Dimension::MemoryBytes.index()]
        - measured.budget_snapshot().dimensions[Dimension::MemoryBytes.index()];
    assert!(
        advance_cost - acknowledgement < acknowledgement,
        "cell memory must fit before a wrongly delayed response admission"
    );
    let mut budget = limits();
    budget.dimensions[Dimension::MemoryBytes.index()] = started + acknowledgement - 1;
    let mut limited = registered("1", budget);
    limited
        .certification_request(begin("bounded", 512))
        .unwrap();
    let before_progress = progress(&limited, "bounded");
    assert_eq!(
        limited
            .certification_request(advance("bounded", 1))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(progress(&limited, "bounded"), before_progress);
    assert_eq!(
        limited.budget_snapshot().verifier_pool,
        before.verifier_pool
    );

    let mut measured = registered("x^2-2", limits());
    measured
        .certification_request(begin("prefix", 512))
        .unwrap();
    measured
        .certification_request(advance("prefix", 1))
        .unwrap();
    let first = progress(&measured, "prefix");
    assert_eq!(first.processed_cells, 1);
    assert!(first.pending_cells > 1);
    let first_cost = limits().dimensions[Dimension::ComputeSteps.index()]
        - measured.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let mut budget = limits();
    budget.dimensions[Dimension::ComputeSteps.index()] = first_cost + 1;
    let mut limited = registered("x^2-2", budget);
    limited.certification_request(begin("prefix", 512)).unwrap();
    assert_eq!(
        limited
            .certification_request(advance("prefix", 2))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(progress(&limited, "prefix"), first);
    let exhausted = limited.budget_snapshot();
    assert_eq!(
        limited
            .certification_request(advance("prefix", 1))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(progress(&limited, "prefix"), first);
    assert_eq!(
        limited.budget_snapshot().verifier_pool,
        exhausted.verifier_pool
    );
    unpublished(&limited, "prefix");
}

#[test]
fn finish_reuses_one_cumulative_protected_pool_without_consuming_the_search() {
    let mut measured = ready_session("x^2-2", "ready", limits());
    let protected = measured.budget_snapshot().verifier_pool;
    measured
        .certification_request(finish("ready", "first"))
        .unwrap();
    let verification_cost = protected - measured.budget_snapshot().verifier_pool;
    assert!(verification_cost > 0);
    let mut budget = limits();
    budget.verifier_pool = verification_cost;
    let mut limited = ready_session("x^2-2", "ready", budget);
    let candidate = progress(&limited, "ready");
    limited
        .certification_request(finish("ready", "first"))
        .unwrap();
    assert_eq!(limited.budget_snapshot().verifier_pool, 0);
    assert_eq!(progress(&limited, "ready"), candidate);
    assert_eq!(
        limited
            .certification_request(finish("ready", "second"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    unpublished(&limited, "second");
    assert!(limited.root_isolation(&isolation("first")).is_some());
    assert_eq!(progress(&limited, "ready"), candidate);
    limited.certification_request(advance("ready", 1)).unwrap();
    assert_eq!(limited.budget_snapshot().verifier_pool, 0);
    limited.certification_request(release("ready")).unwrap();
    assert_eq!(limited.budget_snapshot().verifier_pool, 0);
    assert!(limited.root_isolation(&isolation("first")).is_some());
}

#[test]
fn verified_finish_and_late_child_collision_publish_atomically_and_keep_the_search() {
    let mut measured = ready_session("x^2-2", "ready", limits());
    measured
        .certification_request(finish("ready", "publication"))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut budget = limits();
    budget.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut limited = ready_session("x^2-2", "ready", budget);
    let candidate = progress(&limited, "ready");
    assert_eq!(
        limited
            .certification_request(finish("ready", "publication"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(limited.budget_snapshot().verifier_pool, after.verifier_pool);
    assert_eq!(progress(&limited, "ready"), candidate);
    unpublished(&limited, "publication");

    let mut collision = ready_session("x^2-2", "ready", limits());
    collision
        .certification_request(CertificationRequest::CertifyRootBox {
            schema_version: 1,
            system: system(),
            certificate: child("blocked", 1),
            witness: ExactRootWitness {
                domain: vec![interval("1.4", "3/2")],
                center: vec!["29/20".into()],
                preconditioner: vec![vec!["1/3".into()]],
                precision_digits: 24,
            },
        })
        .unwrap();
    let old = collision
        .root_certificate(&child("blocked", 1))
        .unwrap()
        .clone();
    let candidate = progress(&collision, "ready");
    assert_eq!(
        collision
            .certification_request(finish("ready", "blocked"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    assert!(collision.root_isolation(&isolation("blocked")).is_none());
    assert!(collision.root_certificate(&child("blocked", 0)).is_none());
    assert_eq!(collision.root_certificate(&child("blocked", 1)), Some(&old));
    assert_eq!(progress(&collision, "ready"), candidate);
    collision
        .certification_request(finish("ready", "retry"))
        .unwrap();
    assert_eq!(
        collision
            .root_isolation(&isolation("retry"))
            .unwrap()
            .root_count(),
        2
    );
}

#[test]
fn search_slots_and_release_are_bounded_and_failed_begin_or_release_is_atomic() {
    let mut session = registered("x", limits());
    for index in 0..MAX_SESSION_ROOT_SEARCHES {
        session
            .certification_request(begin(&format!("slot-{index}"), 1))
            .unwrap();
    }
    assert_eq!(
        session
            .certification_request(begin("overflow", 1))
            .unwrap_err()
            .code(),
        "resource_limit_exceeded"
    );
    assert_eq!(
        session
            .certification_request(begin("slot-0", 1))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    let before_release = session.budget_snapshot();
    session.certification_request(release("slot-0")).unwrap();
    assert!(
        session
            .root_isolation_search_progress(&search("slot-0"))
            .is_none()
    );
    let after_release = session.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(
            after_release.dimensions[dimension.index()]
                < before_release.dimensions[dimension.index()]
        );
    }
    assert_eq!(after_release.verifier_pool, before_release.verifier_pool);
    session.certification_request(begin("overflow", 1)).unwrap();

    let mut measured = registered("x", limits());
    measured.certification_request(begin("atomic", 1)).unwrap();
    let begin_cost = limits().dimensions[Dimension::ComputeSteps.index()]
        - measured.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    measured.certification_request(release("atomic")).unwrap();
    let release_cost = limits().dimensions[Dimension::ComputeSteps.index()]
        - measured.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let mut budget = limits();
    budget.dimensions[Dimension::ComputeSteps.index()] = begin_cost - 1;
    let mut failed_begin = registered("x", budget);
    assert_eq!(
        failed_begin
            .certification_request(begin("atomic", 1))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(
        failed_begin
            .root_isolation_search_progress(&search("atomic"))
            .is_none()
    );
    let mut budget = limits();
    budget.dimensions[Dimension::ComputeSteps.index()] = release_cost - 1;
    let mut failed_release = registered("x", budget);
    failed_release
        .certification_request(begin("atomic", 1))
        .unwrap();
    let retained = progress(&failed_release, "atomic");
    assert_eq!(
        failed_release
            .certification_request(release("atomic"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(progress(&failed_release, "atomic"), retained);
}

#[test]
fn unresolved_searches_retain_pending_cells_without_partial_root_acceptance() {
    for (source, candidate_roots) in [("x^2*(x+1)", 1), ("1+0*(1/x)", 0)] {
        let mut session = registered(source, limits());
        session
            .certification_request(begin("unresolved", 16))
            .unwrap();
        let protected = session.budget_snapshot().verifier_pool;
        let failure = session
            .certification_request(advance("unresolved", MAX_ROOT_ISOLATION_ADVANCE_CELLS))
            .unwrap_err();
        let CertificationSessionError::Certification(cause) = &failure else {
            panic!("expected a mathematical search refusal, got {failure:?}");
        };
        if candidate_roots == 1 {
            assert_eq!(failure.code(), "inconclusive");
            assert!(matches!(
                cause,
                CertificationError::Isolation(RootIsolationError::Incomplete(_))
            ));
        } else {
            // Ball division retains its existing typed arithmetic refusal;
            // the held zero multiplier must not hide the original pole.
            assert_eq!(failure.code(), "refused");
            assert!(matches!(
                cause,
                CertificationError::Root(RootBoxError::Ball(BallError::DivisionByZero(_)))
            ));
        }
        let retained = progress(&session, "unresolved");
        assert!(retained.pending_cells > 0);
        assert_eq!(retained.leaf_count, 16);
        assert_eq!(retained.candidate_root_count, candidate_roots);
        assert!(!retained.ready_for_verification);
        assert_eq!(session.budget_snapshot().verifier_pool, protected);
        assert_eq!(
            session
                .certification_request(finish("unresolved", "partial"))
                .unwrap_err()
                .code(),
            "search_not_ready"
        );
        unpublished(&session, "partial");
        let retry = session
            .certification_request(advance("unresolved", 1))
            .unwrap_err();
        assert_eq!(retry.code(), failure.code());
        let CertificationSessionError::Certification(retried_cause) = retry else {
            panic!("retry must retain the original mathematical refusal");
        };
        assert_eq!(&retried_cause, cause);
        assert_eq!(progress(&session, "unresolved"), retained);
    }
}

fn wire(request: CertificationRequest) -> Value {
    serde_json::to_value(request).unwrap()
}

#[test]
fn wire_schemas_and_quantums_are_strict_and_invalid_requests_do_not_advance() {
    let mut session = registered("x", limits());
    session.certification_request(begin("strict", 8)).unwrap();
    let original = progress(&session, "strict");
    for quantum in [0, MAX_ROOT_ISOLATION_ADVANCE_CELLS + 1, usize::MAX] {
        let mut request = wire(advance("strict", quantum));
        request["id"] = json!("invalid-quantum");
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
        assert_eq!(response["code"], "invalid_input");
        assert_eq!(response["id"], "invalid-quantum");
        assert_eq!(progress(&session, "strict"), original);
    }
    let mut unknown = wire(advance("strict", 1));
    unknown["payload"]["accepted"] = json!(true);
    let future = json!({"type":"AdvanceRootIsolationSearch","payload":{"schema_version":2,"search":"search:strict","max_cells":1}});
    let wrong_kind = json!({"type":"AdvanceRootIsolationSearch","payload":{"schema_version":1,"search":"isolation:strict","max_cells":1}});
    for (request, code) in [
        (unknown, "malformed_request"),
        (future, "unsupported_schema"),
        (wrong_kind, "malformed_request"),
    ] {
        let reply: Value =
            serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
        assert_eq!(reply["code"], code, "{reply}");
        assert_eq!(progress(&session, "strict"), original);
    }
    let duplicated = r#"{"type":"AdvanceRootIsolationSearch","payload":{"schema_version":1,"search":"search:strict","max_cells":1,"max_cells":2}}"#;
    let response: Value = serde_json::from_str(&session.handle_envelope(duplicated)).unwrap();
    assert_eq!(response["code"], "malformed_request");
    assert_eq!(progress(&session, "strict"), original);
    for label in ["search:", "search:contains space", "isolation:other"] {
        assert!(RootIsolationSearchId::new(label).is_err());
    }
    assert!(RootIsolationSearchId::new(format!("search:{}", "x".repeat(122))).is_err());
    let mut excessive_capacity = String::with_capacity(1024 * 1024);
    excessive_capacity.push_str("search:small");
    assert_eq!(
        RootIsolationSearchId::new(excessive_capacity)
            .unwrap()
            .as_str(),
        "search:small"
    );
    assert_eq!(
        session
            .certification_request(advance("missing", 1))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    assert_eq!(
        session
            .certification_request(release("missing"))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
}

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        Self {
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        }
    }

    fn send(&mut self, request: Value) -> Value {
        let request = request.to_string();
        assert!(request.len() < frankensympy::MAX_ENVELOPE_BYTES);
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "response before EOF"
        );
        assert!(line.len() < frankensympy::MAX_ENVELOPE_BYTES);
        serde_json::from_str(&line).unwrap()
    }

    fn finish(&mut self) {
        self.input.take();
        assert!(self.child.wait().unwrap().success());
    }
}

impl Drop for Cli {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn success(reply: &Value) {
    assert_eq!(reply["status"], "success", "{reply}");
}

#[test]
fn fresh_cli_polls_bounded_searches_then_finishes_and_replays_the_chunked_complete_proof() {
    for (source, root_count) in [("x^2-2", 2), ("x^2+1", 0)] {
        let mut sender = Cli::new();
        let mut receiver = Cli::new();
        success(&sender.send(wire(registration(source))));
        success(&receiver.send(wire(registration(source))));
        let started = sender.send(wire(begin("live", 512)));
        success(&started);
        assert_eq!(started["result"]["evidence"], "heuristic_candidate");
        assert_eq!(
            sender.send(wire(finish("live", "early")))["code"],
            "search_not_ready"
        );
        let mut last_processed = 0;
        let mut ready = false;
        for tick in 0..MAX_ROOT_ISOLATION_ADVANCE_CELLS {
            let quantum = tick % 3 + 1;
            let mut request = wire(advance("live", quantum));
            request["id"] = json!(format!("advance-{tick}"));
            let advanced = sender.send(request);
            success(&advanced);
            assert_eq!(advanced["id"], format!("advance-{tick}"));
            let result = &advanced["result"];
            assert_eq!(result["evidence"], "heuristic_candidate");
            assert!(result.get("roots").is_none());
            assert!(result.get("isolation").is_none());
            let processed = result["progress"]["processed_cells"].as_u64().unwrap();
            assert!(processed > last_processed && processed - last_processed <= quantum as u64);
            last_processed = processed;
            let inspected = sender.send(wire(CertificationRequest::InspectRootIsolationSearch {
                schema_version: 1,
                search: search("live"),
            }));
            success(&inspected);
            assert_eq!(inspected["result"]["progress"], result["progress"]);
            if result["progress"]["ready_for_verification"] == true {
                ready = true;
                break;
            }
        }
        assert!(
            ready,
            "bounded CLI search reached its completed candidate witness"
        );
        let accepted = sender.send(wire(finish("live", "roots")));
        success(&accepted);
        assert_eq!(accepted["result"]["evidence"], "certificate_verified");
        assert_eq!(accepted["result"]["root_count"], root_count);
        let transfer = ArtifactTransferId::new("transfer:proof").unwrap();
        let exported = sender.send(wire(CertificationRequest::BeginArtifactExport {
            schema_version: 1,
            transfer: transfer.clone(),
            source: ArtifactSource::RootIsolation(isolation("roots")),
        }));
        success(&exported);
        let manifest: ArtifactManifest =
            serde_json::from_value(exported["result"]["manifest"].clone()).unwrap();
        success(
            &receiver.send(wire(CertificationRequest::BeginArtifactImport {
                schema_version: 1,
                transfer: transfer.clone(),
                manifest: manifest.clone(),
            })),
        );
        for index in 0..manifest.chunk_count() {
            let read = sender.send(wire(CertificationRequest::ReadArtifactChunk {
                schema_version: 1,
                transfer: transfer.clone(),
                index,
            }));
            success(&read);
            let chunk: ArtifactChunk =
                serde_json::from_value(read["result"]["chunk"].clone()).unwrap();
            success(
                &receiver.send(wire(CertificationRequest::AppendArtifactChunk {
                    schema_version: 1,
                    transfer: transfer.clone(),
                    chunk,
                })),
            );
        }
        let replayed = receiver.send(wire(
            CertificationRequest::ReplayRootIsolationFromArtifact {
                schema_version: 1,
                system: system(),
                isolation: isolation("received"),
                domain: domain(),
                transfer,
            },
        ));
        success(&replayed);
        assert_eq!(replayed["result"]["root_count"], root_count);
        assert_eq!(replayed["result"]["domain"], accepted["result"]["domain"]);
        assert_eq!(
            replayed["result"]["leaf_count"],
            accepted["result"]["leaf_count"]
        );
        success(&sender.send(wire(release("live"))));
        assert_eq!(
            sender.send(wire(advance("live", 1)))["code"],
            "unknown_object"
        );
        sender.finish();
        receiver.finish();
    }
}
