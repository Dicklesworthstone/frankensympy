//! Complete root-set transport through native sessions and real NDJSON clients.

use frankensympy::certification_session::MAX_SESSION_ROOT_ARTIFACT_BYTES;
use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval, ExactRootWitness,
    ResidualSystemId, RootCertificateId, RootIsolationId, Session,
};
use fsym_budget::Dimension;
use serde_json::{Value, json};
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
    ResidualSystemId::new("system:proof").unwrap()
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

fn session(source: &str) -> Session {
    registered(source, limits())
}

fn isolate(name: &str) -> CertificationRequest {
    CertificationRequest::IsolateRealRootsInBox {
        schema_version: 1,
        system: system(),
        isolation: isolation(name),
        domain: domain(),
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn export(name: &str) -> CertificationRequest {
    CertificationRequest::ExportRootIsolation {
        schema_version: 1,
        isolation: isolation(name),
    }
}

fn replay(name: &str, artifact: &str, domain: Vec<ExactInterval>) -> CertificationRequest {
    CertificationRequest::ReplayRootIsolation {
        schema_version: 1,
        system: system(),
        isolation: isolation(name),
        domain,
        artifact: artifact.into(),
    }
}

fn artifact(source: &str) -> String {
    let mut sender = session(source);
    sender.certification_request(isolate("sender")).unwrap();
    let protected = sender.budget_snapshot().verifier_pool;
    let response = sender.certification_request(export("sender")).unwrap();
    assert_eq!(sender.budget_snapshot().verifier_pool, protected);
    let CertificationResponse::RootIsolationExported {
        artifact,
        system: source_system,
        isolation: source_isolation,
        ..
    } = response
    else {
        panic!("complete isolation export response")
    };
    assert_eq!(source_system, system());
    assert_eq!(source_isolation, isolation("sender"));
    artifact
}

fn unpublished(session: &Session, name: &str) {
    assert!(session.root_isolation(&isolation(name)).is_none());
    assert!(session.root_certificate(&child(name, 0)).is_none());
    assert!(session.root_certificate(&child(name, 1)).is_none());
}

fn wire(request: CertificationRequest) -> Value {
    serde_json::to_value(request).unwrap()
}

fn success(response: &Value) {
    assert_eq!(response["status"], "success", "{response}");
}

#[test]
fn receiving_domain_and_source_must_match_before_any_object_is_published() {
    let artifact = artifact("x^2-2");
    let mut receiver = session("x^2-2");
    let refusal = receiver
        .certification_request(replay("received", &artifact, vec![interval("-3", "3")]))
        .unwrap_err();
    assert_eq!(refusal.code(), "domain_mismatch");
    unpublished(&receiver, "received");
    receiver
        .certification_request(replay("received", &artifact, domain()))
        .unwrap();
    assert_eq!(
        receiver
            .root_isolation(&isolation("received"))
            .unwrap()
            .root_count(),
        2
    );

    let mut different = session("x^2-3");
    let refusal = different
        .certification_request(replay("changed", &artifact, domain()))
        .unwrap_err();
    assert_eq!(refusal.code(), "source_mismatch");
    unpublished(&different, "changed");
}

#[test]
fn canonical_forged_zero_root_cover_reaches_and_fails_independent_verification() {
    let original = artifact("x^2+1");
    let old = format!(
        "\"residuals\":{},\"variables\":",
        serde_json::to_string(&[fsym_core::parse_raw("x^2+1").unwrap()]).unwrap()
    );
    let new = format!(
        "\"residuals\":{},\"variables\":",
        serde_json::to_string(&[fsym_core::parse_raw("x^2-2").unwrap()]).unwrap()
    );
    assert_eq!(original.matches(&old).count(), 1);
    // Change only the typed source payload. Preserve every canonical byte of
    // the now-false exclusion cover so this reaches the mathematical checker.
    let forged = original.replacen(&old, &new, 1);
    let mut receiver = session("x^2-2");
    let protected = receiver.budget_snapshot().verifier_pool;
    let refusal = receiver
        .certification_request(replay("forged", &forged, domain()))
        .unwrap_err();
    assert_eq!(refusal.code(), "inconclusive");
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    unpublished(&receiver, "forged");
}

#[test]
fn replay_verifier_fuel_is_cumulative_and_cannot_borrow_ordinary_steps() {
    let artifact = artifact("x^2-2");
    let mut measured = session("x^2-2");
    let before = measured.budget_snapshot().verifier_pool;
    measured
        .certification_request(replay("first", &artifact, domain()))
        .unwrap();
    let spent = before - measured.budget_snapshot().verifier_pool;
    assert!(spent > 0);

    let mut exact = limits();
    exact.verifier_pool = spent;
    let mut receiver = registered("x^2-2", exact);
    receiver
        .certification_request(replay("first", &artifact, domain()))
        .unwrap();
    assert_eq!(receiver.budget_snapshot().verifier_pool, 0);
    let refusal = receiver
        .certification_request(replay("second", &artifact, domain()))
        .unwrap_err();
    assert_eq!(refusal.code(), "budget_exhausted");
    assert!(receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    unpublished(&receiver, "second");
    assert_eq!(
        receiver
            .root_isolation(&isolation("first"))
            .unwrap()
            .root_count(),
        2
    );
    assert!(receiver.root_certificate(&child("first", 0)).is_some());
    assert!(receiver.root_certificate(&child("first", 1)).is_some());
}

#[test]
fn response_exhaustion_after_replay_verification_publishes_no_parent_or_children() {
    let artifact = artifact("x^2-2");
    let mut measured = session("x^2-2");
    measured
        .certification_request(replay("publication", &artifact, domain()))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    assert!(spent > 1);
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut receiver = registered("x^2-2", limited);
    let refusal = receiver
        .certification_request(replay("publication", &artifact, domain()))
        .unwrap_err();
    assert_eq!(refusal.code(), "budget_exhausted");
    assert_eq!(
        receiver.budget_snapshot().verifier_pool,
        after.verifier_pool,
        "the same complete independent check ran before response admission refused",
    );
    unpublished(&receiver, "publication");
}

#[test]
fn a_second_child_collision_during_replay_preserves_all_existing_state() {
    let artifact = artifact("x^2-2");
    let mut receiver = session("x^2-2");
    receiver
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
    let existing = receiver
        .root_certificate(&child("blocked", 1))
        .unwrap()
        .clone();
    let refusal = receiver
        .certification_request(replay("blocked", &artifact, domain()))
        .unwrap_err();
    assert_eq!(refusal.code(), "duplicate_object");
    assert!(receiver.root_isolation(&isolation("blocked")).is_none());
    assert!(receiver.root_certificate(&child("blocked", 0)).is_none());
    assert_eq!(
        receiver.root_certificate(&child("blocked", 1)),
        Some(&existing),
    );
    receiver
        .certification_request(replay("retry", &artifact, domain()))
        .unwrap();
    assert_eq!(
        receiver
            .root_isolation(&isolation("retry"))
            .unwrap()
            .root_count(),
        2
    );
    assert!(receiver.root_certificate(&child("retry", 0)).is_some());
    assert!(receiver.root_certificate(&child("retry", 1)).is_some());
}

#[test]
fn transport_requests_refuse_ambiguous_missing_future_and_oversized_inputs() {
    let artifact = artifact("x^2-2");
    let mut receiver = session("x^2-2");
    assert_eq!(
        receiver
            .certification_request(export("missing"))
            .unwrap_err()
            .code(),
        "unknown_object",
    );

    let mut future = wire(replay("wire", &artifact, domain()));
    future["payload"]["schema_version"] = json!(2);
    let refusal: Value =
        serde_json::from_str(&receiver.handle_envelope(&future.to_string())).unwrap();
    assert_eq!(refusal["code"], "unsupported_schema");

    let mut unknown = wire(export("missing"));
    unknown["payload"]["accepted"] = json!(true);
    let refusal: Value =
        serde_json::from_str(&receiver.handle_envelope(&unknown.to_string())).unwrap();
    assert_eq!(refusal["code"], "malformed_request");

    let mut missing = wire(replay("wire", &artifact, domain()));
    missing["payload"].as_object_mut().unwrap().remove("domain");
    let refusal: Value =
        serde_json::from_str(&receiver.handle_envelope(&missing.to_string())).unwrap();
    assert_eq!(refusal["code"], "malformed_request");

    let encoded = serde_json::to_string(&artifact).unwrap();
    let duplicate = [
        r#"{"type":"ReplayRootIsolation","payload":{"schema_version":1,"system":"system:proof","isolation":"isolation:wire","domain":[{"lower":"-2","upper":"2"}],"artifact":"","artifact":"#,
        encoded.as_str(),
        "}}",
    ].concat();
    let refusal: Value = serde_json::from_str(&receiver.handle_envelope(&duplicate)).unwrap();
    assert_eq!(refusal["code"], "malformed_request");
    unpublished(&receiver, "wire");

    let protected = receiver.budget_snapshot().verifier_pool;
    let oversized = " ".repeat(MAX_SESSION_ROOT_ARTIFACT_BYTES + 1);
    assert_eq!(
        receiver
            .certification_request(replay("wire", &oversized, domain()))
            .unwrap_err()
            .code(),
        "resource_limit_exceeded",
    );
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    unpublished(&receiver, "wire");
    let accepted: Value = serde_json::from_str(
        &receiver.handle_envelope(&wire(replay("wire", &artifact, domain())).to_string()),
    )
    .unwrap();
    success(&accepted);
    assert!(receiver.root_isolation(&isolation("wire")).is_some());
}

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new(arguments: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take();
        let output = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            input,
            output,
        }
    }

    fn send(&mut self, request: Value) -> Value {
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "one response before stdin closes",
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

#[test]
fn default_cli_round_trips_complete_two_root_and_zero_root_proofs() {
    for (source, count) in [("x^2-2", 2_usize), ("x^2+1", 0)] {
        let mut sender = Cli::new(&[]);
        success(&sender.send(wire(registration(source))));
        let original = sender.send(wire(isolate("sender")));
        success(&original);
        assert_eq!(original["result"]["root_count"], count);
        let exported = sender.send(wire(export("sender")));
        success(&exported);
        let artifact = exported["result"]["artifact"].as_str().unwrap().to_owned();
        assert!(artifact.len() <= MAX_SESSION_ROOT_ARTIFACT_BYTES);
        sender.finish();

        let mut receiver = Cli::new(&[]);
        success(&receiver.send(wire(registration(source))));
        let mut request = wire(replay("received", &artifact, domain()));
        request["id"] = json!("complete-replay");
        let replayed = receiver.send(request);
        success(&replayed);
        assert_eq!(replayed["id"], "complete-replay");
        for field in [
            "root_count",
            "domain",
            "context_digest",
            "claim",
            "evidence",
            "leaf_count",
            "excluded_leaf_count",
            "precision_digits",
        ] {
            assert_eq!(
                replayed["result"][field], original["result"][field],
                "{field}"
            );
        }
        assert_eq!(replayed["result"]["roots"].as_array().unwrap().len(), count);
        for index in 0..count {
            let exported_child = receiver.send(wire(CertificationRequest::ExportRootCertificate {
                schema_version: 1,
                certificate: child("received", index),
            }));
            success(&exported_child);
            for field in ["domain", "root_enclosure", "contraction_bound"] {
                assert_eq!(
                    replayed["result"]["roots"][index][field],
                    original["result"]["roots"][index][field],
                    "{field}",
                );
            }
        }
        if count == 0 {
            let absent = receiver.send(wire(CertificationRequest::ExportRootCertificate {
                schema_version: 1,
                certificate: child("received", 0),
            }));
            assert_eq!(absent["code"], "unknown_object");
        }
        let reexported = receiver.send(wire(export("received")));
        success(&reexported);
        assert_eq!(reexported["result"]["artifact"], artifact);
        receiver.finish();
    }
}
