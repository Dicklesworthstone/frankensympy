//! Interactive optimization stays a candidate until protected independent finish.

use frankensympy::{
    BigRational, BoxExtremumKind, BoxOptimizationId, BoxOptimizationSearchId, BudgetLimits,
    CertificationRequest, CertificationResponse, ExactInterval, ObjectiveId, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Dimension;
use fsym_calculus::box_optimization::{
    BoxOptimizationProgress, BoxOptimizationSearchStatus, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS,
};
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

fn objective() -> ObjectiveId {
    ObjectiveId::new("objective:source").unwrap()
}

fn search(name: &str) -> BoxOptimizationSearchId {
    BoxOptimizationSearchId::new(format!("optimization-search:{name}")).unwrap()
}

fn optimization(name: &str) -> BoxOptimizationId {
    BoxOptimizationId::new(format!("optimization:{name}")).unwrap()
}

fn domain() -> Vec<ExactInterval> {
    vec![ExactInterval {
        lower: "-2".into(),
        upper: "2".into(),
    }]
}

fn registration(source: &str) -> CertificationRequest {
    CertificationRequest::RegisterObjective {
        schema_version: 1,
        objective: objective(),
        source: source.into(),
        variables: vec!["x".into()],
    }
}

fn registered(source: &str, limits: BudgetLimits) -> Session {
    let mut session = Session::with_budget_limits(limits, 256);
    session.certification_request(registration(source)).unwrap();
    session
}

fn begin(name: &str) -> CertificationRequest {
    CertificationRequest::BeginBoxOptimization {
        schema_version: 1,
        objective: objective(),
        search: search(name),
        domain: domain(),
        extremum: BoxExtremumKind::Minimum,
        absolute_tolerance: "1/1000".into(),
        precision_digits: 16,
        max_leaves: 256,
    }
}

fn advance(name: &str, max_splits: usize) -> CertificationRequest {
    CertificationRequest::AdvanceBoxOptimization {
        schema_version: 1,
        search: search(name),
        max_splits,
    }
}

fn inspect(name: &str) -> CertificationRequest {
    CertificationRequest::InspectBoxOptimizationSearch {
        schema_version: 1,
        search: search(name),
    }
}

fn finish(name: &str, result: &str) -> CertificationRequest {
    CertificationRequest::FinishBoxOptimization {
        schema_version: 1,
        search: search(name),
        optimization: optimization(result),
    }
}

fn release(name: &str) -> CertificationRequest {
    CertificationRequest::ReleaseBoxOptimizationSearch {
        schema_version: 1,
        search: search(name),
    }
}

fn progress(session: &Session, name: &str) -> BoxOptimizationProgress {
    session
        .box_optimization_search_progress(&search(name))
        .unwrap()
}

fn run_to_ready(session: &mut Session, name: &str, quantum: usize) -> BoxOptimizationProgress {
    for _ in 0..MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS {
        let before = progress(session, name);
        if before.status == BoxOptimizationSearchStatus::Complete {
            return before;
        }
        let previous_counters = session
            .box_optimization_search_counters(&search(name))
            .unwrap();
        let CertificationResponse::BoxOptimizationSearchProgress {
            progress: after,
            evidence,
            counters,
            ..
        } = session
            .certification_request(advance(name, quantum))
            .unwrap()
        else {
            panic!("expected bounded candidate progress");
        };
        assert_eq!(evidence, "heuristic_candidate");
        assert_eq!(after, progress(session, name));
        assert!(after.split_count > before.split_count);
        assert!(after.split_count - before.split_count <= quantum);
        assert_eq!(after.leaf_count, after.split_count + 1);
        assert!(counters.steps >= previous_counters.steps);
        assert!(counters.allocation_bytes >= previous_counters.allocation_bytes);
        assert_eq!(counters.evaluations, after.evaluation_count);
        assert_eq!(
            session.box_optimization_search_counters(&search(name)),
            Some(counters)
        );
    }
    panic!("bounded optimization did not finish");
}

fn ready(limits: BudgetLimits) -> Session {
    let mut session = registered("(x^2-2)^2", limits);
    session.certification_request(begin("ready")).unwrap();
    run_to_ready(&mut session, "ready", 3);
    session
}

fn export(session: &mut Session, name: &str) -> String {
    let CertificationResponse::BoxOptimizationExported { artifact, .. } = session
        .certification_request(CertificationRequest::ExportBoxOptimization {
            schema_version: 1,
            optimization: optimization(name),
        })
        .unwrap()
    else {
        panic!("expected a portable optimization witness");
    };
    artifact
}

fn replay(artifact: String, result: &str) -> CertificationRequest {
    CertificationRequest::ReplayBoxOptimization {
        schema_version: 1,
        objective: objective(),
        optimization: optimization(result),
        domain: domain(),
        extremum: BoxExtremumKind::Minimum,
        absolute_tolerance: "1/1000".into(),
        artifact,
    }
}

#[test]
fn candidate_progress_cannot_publish_or_borrow_the_protected_verifier_pool() {
    let mut budget = limits();
    budget.verifier_pool = 0;
    let mut session = registered("(x^2-2)^2", budget);
    let started = session.certification_request(begin("pending")).unwrap();
    let wire = serde_json::to_value(started).unwrap();
    assert_eq!(wire["kind"], "box_optimization_search_started");
    assert_eq!(wire["evidence"], "heuristic_candidate");
    for absent in [
        "claim",
        "value",
        "gap",
        "optimization",
        "receipt",
        "accepted",
    ] {
        assert!(
            wire.get(absent).is_none(),
            "unexpected accepted field {absent}"
        );
    }
    assert_eq!(
        progress(&session, "pending").status,
        BoxOptimizationSearchStatus::Searching
    );
    assert_eq!(
        session
            .certification_request(finish("pending", "premature"))
            .unwrap_err()
            .code(),
        "search_not_ready"
    );
    assert!(
        session
            .box_optimization(&optimization("premature"))
            .is_none()
    );
    let completed = run_to_ready(&mut session, "pending", 1);
    assert_eq!(session.budget_snapshot().verifier_pool, 0);
    let inspected =
        serde_json::to_value(session.certification_request(inspect("pending")).unwrap()).unwrap();
    assert_eq!(inspected["evidence"], "heuristic_candidate");
    assert_eq!(inspected["progress"]["status"], "complete");
    assert!(inspected.get("value").is_none());
    assert_eq!(
        session
            .certification_request(finish("pending", "unverified"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(progress(&session, "pending"), completed);
    assert!(
        session
            .box_optimization(&optimization("unverified"))
            .is_none()
    );
    session.certification_request(release("pending")).unwrap();
    assert!(
        session
            .box_optimization_search_progress(&search("pending"))
            .is_none()
    );
}

#[test]
fn interleaved_searches_freeze_their_statement_and_finish_a_replayable_global_claim() {
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(
            HashMap::new(),
            HashMap::new(),
            "optimization-search-context",
        )
        .unwrap();
    let mut sender = Session::with_context_and_budget_limits(context.clone(), limits(), 256);
    sender
        .certification_request(registration("(x^2-2)^2"))
        .unwrap();
    let original = sender.objective_problem(&objective()).unwrap().clone();
    sender.certification_request(begin("slow")).unwrap();
    sender.certification_request(begin("fast")).unwrap();
    let untouched = progress(&sender, "fast");
    sender.certification_request(advance("slow", 1)).unwrap();
    assert_eq!(progress(&sender, "fast"), untouched);
    sender.construct("x", "100").unwrap();
    sender.fork_branch("different").unwrap();
    sender.checkout_branch("different").unwrap();
    sender.construct("x", "0").unwrap();
    assert_eq!(
        sender
            .certification_request(registration("x^2+9"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    run_to_ready(&mut sender, "slow", 1);
    run_to_ready(&mut sender, "fast", 7);
    assert_eq!(
        sender.box_optimization_search_counters(&search("slow")),
        sender.box_optimization_search_counters(&search("fast"))
    );
    let CertificationResponse::BoxOptimized {
        evidence,
        claim,
        context_digest,
        extremum,
        ..
    } = sender
        .certification_request(finish("slow", "accepted"))
        .unwrap()
    else {
        panic!("expected independent global acceptance");
    };
    assert_eq!(evidence, "certified_numeric");
    assert_eq!(claim, "global_box_optimum_with_absolute_value_gap_bound");
    assert_eq!(extremum, BoxExtremumKind::Minimum);
    assert_eq!(context_digest, context.digest());
    let accepted = sender.box_optimization(&optimization("accepted")).unwrap();
    assert_eq!(accepted.objective(), original.objective());
    assert_eq!(accepted.variables(), original.variables());
    assert!(
        accepted
            .value()
            .contains(&BigRational::from_integer(0.into()))
    );
    assert!(accepted.point_value().lower() > BigRational::from_integer(0.into()));
    let artifact = export(&mut sender, "accepted");
    let mut foreign = registered("(x^2-2)^2", limits());
    assert_eq!(
        foreign
            .certification_request(replay(artifact.clone(), "foreign"))
            .unwrap_err()
            .code(),
        "universe_mismatch"
    );
    assert!(foreign.box_optimization(&optimization("foreign")).is_none());
    let mut receiver = Session::with_context_and_budget_limits(context, limits(), 256);
    receiver
        .certification_request(registration("(x^2-2)^2"))
        .unwrap();
    receiver
        .certification_request(replay(artifact, "received"))
        .unwrap();
    assert_eq!(
        receiver.box_optimization(&optimization("received")),
        sender.box_optimization(&optimization("accepted"))
    );
    sender.certification_request(release("slow")).unwrap();
    assert!(sender.box_optimization(&optimization("accepted")).is_some());
    assert!(
        sender
            .box_optimization_search_progress(&search("fast"))
            .is_some()
    );
}

#[test]
fn repeated_finish_uses_one_protected_pool_and_retains_the_completed_candidate() {
    let mut measured = ready(limits());
    let protected = measured.budget_snapshot().verifier_pool;
    measured
        .certification_request(finish("ready", "first"))
        .unwrap();
    let verification_cost = protected - measured.budget_snapshot().verifier_pool;
    assert!(verification_cost > 0);
    let mut budget = limits();
    budget.verifier_pool = verification_cost;
    let mut limited = ready(budget);
    let candidate = progress(&limited, "ready");
    limited
        .certification_request(finish("ready", "first"))
        .unwrap();
    assert_eq!(limited.budget_snapshot().verifier_pool, 0);
    assert_eq!(
        limited
            .certification_request(finish("ready", "second"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(progress(&limited, "ready"), candidate);
    assert!(limited.box_optimization(&optimization("first")).is_some());
    assert!(limited.box_optimization(&optimization("second")).is_none());
    assert_eq!(
        limited
            .certification_request(finish("ready", "first"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    limited.certification_request(release("ready")).unwrap();
    assert_eq!(limited.budget_snapshot().verifier_pool, 0);
    assert!(limited.box_optimization(&optimization("first")).is_some());
}

#[test]
fn response_admission_precedes_frontier_work_and_work_exhaustion_becomes_terminal() {
    // Advancing an already-complete x^2 candidate performs no mathematical
    // work. Its charge measures the same maximum-width response reservation.
    let mut probe = registered("x^2", limits());
    probe.certification_request(begin("bounded")).unwrap();
    assert_eq!(
        progress(&probe, "bounded").status,
        BoxOptimizationSearchStatus::Complete
    );
    let before = probe.budget_snapshot();
    probe.certification_request(advance("bounded", 1)).unwrap();
    let after = probe.budget_snapshot();
    let acknowledgement_memory = before.dimensions[Dimension::MemoryBytes.index()]
        - after.dimensions[Dimension::MemoryBytes.index()];
    let acknowledgement_compute = before.dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut measured = registered("(x^2-2)^2", limits());
    measured.certification_request(begin("bounded")).unwrap();
    let started = measured.budget_snapshot();

    let mut budget = limits();
    budget.dimensions[Dimension::MemoryBytes.index()] = limits().dimensions
        [Dimension::MemoryBytes.index()]
        - started.dimensions[Dimension::MemoryBytes.index()]
        + acknowledgement_memory
        - 1;
    let mut limited = registered("(x^2-2)^2", budget);
    limited.certification_request(begin("bounded")).unwrap();
    let initial = progress(&limited, "bounded");
    let counters = limited.box_optimization_search_counters(&search("bounded"));
    assert_eq!(
        limited
            .certification_request(advance("bounded", 1))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(progress(&limited, "bounded"), initial);
    assert_eq!(
        limited.box_optimization_search_counters(&search("bounded")),
        counters
    );

    let mut budget = limits();
    budget.dimensions[Dimension::ComputeSteps.index()] = limits().dimensions
        [Dimension::ComputeSteps.index()]
        - started.dimensions[Dimension::ComputeSteps.index()]
        + acknowledgement_compute
        + 1;
    let mut exhausted = registered("(x^2-2)^2", budget);
    exhausted.certification_request(begin("bounded")).unwrap();
    let protected = exhausted.budget_snapshot().verifier_pool;
    assert_eq!(
        exhausted
            .certification_request(advance("bounded", 1))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(
        progress(&exhausted, "bounded").status,
        BoxOptimizationSearchStatus::Failed
    );
    assert_eq!(exhausted.budget_snapshot().verifier_pool, protected);
    assert!(
        exhausted
            .box_optimization(&optimization("bounded"))
            .is_none()
    );
    assert!(
        exhausted
            .certification_request(finish("bounded", "partial"))
            .is_err()
    );
    assert!(
        exhausted
            .box_optimization(&optimization("partial"))
            .is_none()
    );
}

#[test]
fn verified_finish_with_refused_output_does_not_publish_an_accepted_handle() {
    let mut measured = ready(limits());
    measured
        .certification_request(finish("ready", "publication"))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut budget = limits();
    budget.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut limited = ready(budget);
    let candidate = progress(&limited, "ready");
    assert_eq!(
        limited
            .certification_request(finish("ready", "publication"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    // Reaching the same protected balance proves verification finished before
    // ordinary output admission refused. The result slot remains absent.
    assert_eq!(limited.budget_snapshot().verifier_pool, after.verifier_pool);
    assert_eq!(progress(&limited, "ready"), candidate);
    assert!(
        limited
            .box_optimization(&optimization("publication"))
            .is_none()
    );
}

fn wire(request: CertificationRequest) -> Value {
    serde_json::to_value(request).unwrap()
}

#[test]
fn invalid_wire_quantum_schema_and_intrinsic_leaf_cap_cannot_publish_a_result() {
    let mut session = registered("(x^2-2)^2", limits());
    session.certification_request(begin("strict")).unwrap();
    let initial = progress(&session, "strict");
    for quantum in [0, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS + 1, usize::MAX] {
        let mut request = wire(advance("strict", quantum));
        request["id"] = json!("invalid-quantum");
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
        assert_eq!(response["code"], "invalid_input");
        assert_eq!(response["id"], "invalid-quantum");
        assert_eq!(progress(&session, "strict"), initial);
    }
    let mut unknown = wire(advance("strict", 1));
    unknown["payload"]["verified"] = json!(true);
    let future = json!({"type":"AdvanceBoxOptimization","payload":{"schema_version":2,"search":"optimization-search:strict","max_splits":1}});
    let wrong_kind = json!({"type":"AdvanceBoxOptimization","payload":{"schema_version":1,"search":"optimization:strict","max_splits":1}});
    for (request, code) in [
        (unknown, "malformed_request"),
        (future, "unsupported_schema"),
        (wrong_kind, "malformed_request"),
    ] {
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
        assert_eq!(response["code"], code, "{response}");
        assert_eq!(progress(&session, "strict"), initial);
    }
    let mut capped = begin("capped");
    if let CertificationRequest::BeginBoxOptimization { max_leaves, .. } = &mut capped {
        *max_leaves = 3;
    }
    session.certification_request(capped).unwrap();
    let mut refused = false;
    for _ in 0..3 {
        if session.certification_request(advance("capped", 1)).is_err() {
            refused = true;
            break;
        }
    }
    assert!(refused);
    assert_eq!(
        progress(&session, "capped").status,
        BoxOptimizationSearchStatus::Failed
    );
    assert!(
        session
            .certification_request(finish("capped", "incomplete"))
            .is_err()
    );
    assert!(
        session
            .box_optimization(&optimization("incomplete"))
            .is_none()
    );
    session.certification_request(release("capped")).unwrap();
    assert_eq!(progress(&session, "strict"), initial);
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

    fn send(&mut self, request: CertificationRequest) -> Value {
        let request = serde_json::to_string(&request).unwrap();
        assert!(request.len() < frankensympy::MAX_ENVELOPE_BYTES);
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "response before EOF"
        );
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
fn live_cli_yields_search_progress_then_independently_finishes_and_replays() {
    let mut sender = Cli::new();
    let mut receiver = Cli::new();
    success(&sender.send(registration("(x^2-2)^2")));
    success(&receiver.send(registration("(x^2-2)^2")));
    let started = sender.send(begin("live"));
    success(&started);
    assert_eq!(started["result"]["evidence"], "heuristic_candidate");
    assert_eq!(
        sender.send(finish("live", "early"))["code"],
        "search_not_ready"
    );
    let mut last_splits = 0;
    let mut completed = false;
    for step in 0..256 {
        let quantum = step % 3 + 1;
        let advanced = sender.send(advance("live", quantum));
        success(&advanced);
        let result = &advanced["result"];
        assert_eq!(result["evidence"], "heuristic_candidate");
        assert!(result.get("claim").is_none());
        let splits = result["progress"]["split_count"].as_u64().unwrap();
        assert!(splits > last_splits);
        assert!(splits - last_splits <= quantum as u64);
        last_splits = splits;
        if result["progress"]["status"] == "complete" {
            completed = true;
            break;
        }
    }
    assert!(completed);
    let inspected = sender.send(inspect("live"));
    success(&inspected);
    assert_eq!(inspected["result"]["progress"]["split_count"], last_splits);
    let accepted = sender.send(finish("live", "accepted"));
    success(&accepted);
    assert_eq!(accepted["result"]["evidence"], "certified_numeric");
    assert_eq!(
        accepted["result"]["claim"],
        "global_box_optimum_with_absolute_value_gap_bound"
    );
    let exported = sender.send(CertificationRequest::ExportBoxOptimization {
        schema_version: 1,
        optimization: optimization("accepted"),
    });
    success(&exported);
    let received = receiver.send(replay(
        exported["result"]["artifact"].as_str().unwrap().into(),
        "received",
    ));
    success(&received);
    for field in [
        "claim",
        "domain",
        "point",
        "value",
        "point_value",
        "gap",
        "leaf_count",
        "context_digest",
    ] {
        assert_eq!(
            received["result"][field], accepted["result"][field],
            "{field}"
        );
    }
    success(&sender.send(release("live")));
    assert_eq!(sender.send(inspect("live"))["code"], "unknown_object");
    sender.finish();
    receiver.finish();
}
