//! Complete-box search through native sessions and real interactive NDJSON.
use frankensympy::{
    BigRational, BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval,
    ExactRootWitness, ResidualSystemId, RootCertificateId, RootIsolationId, Session,
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
fn session() -> Session {
    Session::with_budget_limits(limits(), 256)
}
fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:equations").unwrap()
}
fn isolation(name: &str) -> RootIsolationId {
    RootIsolationId::new(format!("isolation:{name}")).unwrap()
}
fn certificate(name: &str) -> RootCertificateId {
    RootCertificateId::new(format!("certificate:{name}")).unwrap()
}
fn child(name: &str, index: usize) -> RootCertificateId {
    certificate(&format!("{name}.root-{index}"))
}
fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}
fn register(s: &mut Session, source: &str) {
    s.certification_request(CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec![source.into()],
        variables: vec!["x".into()],
    })
    .unwrap();
}
fn request(name: &str, lower: &str, upper: &str, max_leaves: usize) -> CertificationRequest {
    CertificationRequest::IsolateRealRootsInBox {
        schema_version: 1,
        system: system(),
        isolation: isolation(name),
        domain: vec![interval(lower, upper)],
        precision_digits: 16,
        max_leaves,
    }
}
fn export(s: &mut Session, certificate: RootCertificateId) -> String {
    let response = s
        .certification_request(CertificationRequest::ExportRootCertificate {
            schema_version: 1,
            certificate,
        })
        .unwrap();
    let CertificationResponse::RootCertificateExported { artifact, .. } = response else {
        panic!("certificate export response")
    };
    artifact
}
fn assert_unpublished(s: &Session, name: &str) {
    assert!(s.root_isolation(&isolation(name)).is_none());
    assert!(s.root_certificate(&child(name, 0)).is_none());
    assert!(s.root_certificate(&child(name, 1)).is_none());
}

#[test]
fn two_roots_have_atomic_source_bound_handles_and_fresh_session_replay() {
    let mut sender = session();
    register(&mut sender, "x^2-2");
    sender.construct("x", "0").unwrap();
    let before = sender.budget_snapshot();
    let response = sender
        .certification_request(request("roots", "-2", "2", 512))
        .unwrap();
    let after = sender.budget_snapshot();
    eprintln!(
        "two-root request: {} ordinary steps, {} verifier steps",
        before.dimensions[Dimension::ComputeSteps.index()]
            - after.dimensions[Dimension::ComputeSteps.index()],
        before.verifier_pool - after.verifier_pool,
    );
    let CertificationResponse::RealRootsIsolated {
        root_count,
        roots,
        leaf_count,
        excluded_leaf_count,
        domain,
        context_digest,
        ..
    } = response
    else {
        panic!("root isolation response")
    };
    assert_eq!(root_count, 2);
    assert_eq!(roots.len(), 2);
    assert!(leaf_count > excluded_leaf_count);
    assert!(excluded_leaf_count > 0);
    assert_eq!(domain, [interval("-2", "2")]);
    assert_eq!(context_digest, sender.context_digest());
    let accepted = sender.root_isolation(&isolation("roots")).unwrap();
    assert_eq!(accepted.root_count(), 2);
    assert_eq!(
        accepted.residuals(),
        sender.residual_problem(&system()).unwrap().residuals()
    );
    assert!(accepted.roots()[0].root_enclosure()[0].upper() < BigRational::from_integer(0.into()));
    assert!(accepted.roots()[1].root_enclosure()[0].lower() > BigRational::from_integer(0.into()));
    for (index, summary) in roots.iter().enumerate() {
        assert_eq!(summary.certificate, child("roots", index));
        assert_eq!(
            sender.root_certificate(&summary.certificate),
            Some(&accepted.roots()[index])
        );
    }
    for (index, summary) in roots.iter().enumerate() {
        let artifact = export(&mut sender, summary.certificate.clone());
        let mut receiver = session();
        register(&mut receiver, "x^2-2");
        let replay_id = certificate(&format!("replayed-{index}"));
        let replayed = receiver
            .certification_request(CertificationRequest::ReplayRootCertificate {
                schema_version: 1,
                system: system(),
                certificate: replay_id.clone(),
                artifact,
            })
            .unwrap();
        let CertificationResponse::RootBoxCertified {
            domain,
            root_enclosure,
            contraction_bound,
            ..
        } = replayed
        else {
            panic!("replayed child certificate")
        };
        assert_eq!(domain, summary.domain);
        assert_eq!(root_enclosure, summary.root_enclosure);
        assert_eq!(contraction_bound, summary.contraction_bound);
        assert_eq!(
            sender.root_certificate(&summary.certificate),
            receiver.root_certificate(&replay_id)
        );
    }
}

#[test]
fn verified_empty_root_sets_are_distinct_from_failed_searches() {
    let mut s = session();
    register(&mut s, "x^2+1");
    let response = s
        .certification_request(request("empty", "-2", "2", 512))
        .unwrap();
    let CertificationResponse::RealRootsIsolated {
        root_count,
        roots,
        leaf_count,
        excluded_leaf_count,
        ..
    } = response
    else {
        panic!("root-free isolation response")
    };
    assert_eq!(root_count, 0);
    assert!(roots.is_empty());
    assert!(leaf_count > 0);
    assert_eq!(leaf_count, excluded_leaf_count);
    assert!(
        s.root_isolation(&isolation("empty"))
            .unwrap()
            .is_root_free()
    );
    assert!(s.root_certificate(&child("empty", 0)).is_none());
    assert_eq!(
        s.certification_request(request("empty", "-2", "2", 512))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn singular_boundary_and_unresolved_domains_publish_no_partial_handles() {
    for (source, lower, upper) in [
        ("x^2", "-1", "1"),
        ("x", "0", "1"),
        ("x^2*(x+1)", "-2", "2"),
    ] {
        let mut s = session();
        register(&mut s, source);
        let protected = s.budget_snapshot().verifier_pool;
        assert_eq!(
            s.certification_request(request("incomplete", lower, upper, 16))
                .unwrap_err()
                .code(),
            "inconclusive"
        );
        assert_eq!(s.budget_snapshot().verifier_pool, protected);
        assert_unpublished(&s, "incomplete");
    }
    let mut s = session();
    register(&mut s, "1+0*(1/x)");
    let protected = s.budget_snapshot().verifier_pool;
    assert!(
        s.certification_request(request("pole", "-1", "1", 8))
            .is_err()
    );
    assert_eq!(s.budget_snapshot().verifier_pool, protected);
    assert_unpublished(&s, "pole");
}

#[test]
fn a_late_child_name_collision_preserves_existing_state_and_publishes_nothing_else() {
    let mut s = session();
    register(&mut s, "x^2-2");
    s.certification_request(CertificationRequest::CertifyRootBox {
        schema_version: 1,
        system: system(),
        certificate: child("roots", 1),
        witness: ExactRootWitness {
            domain: vec![interval("1.4", "3/2")],
            center: vec!["29/20".into()],
            preconditioner: vec![vec!["1/3".into()]],
            precision_digits: 24,
        },
    })
    .unwrap();
    let existing = s.root_certificate(&child("roots", 1)).unwrap().clone();
    assert_eq!(
        s.certification_request(request("roots", "-2", "2", 512))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    assert!(s.root_isolation(&isolation("roots")).is_none());
    assert!(s.root_certificate(&child("roots", 0)).is_none());
    assert_eq!(s.root_certificate(&child("roots", 1)), Some(&existing));
    s.certification_request(request("retry", "-2", "2", 512))
        .unwrap();
    assert_eq!(
        s.root_isolation(&isolation("retry")).unwrap().root_count(),
        2
    );
    assert!(s.root_certificate(&child("retry", 0)).is_some());
    assert!(s.root_certificate(&child("retry", 1)).is_some());
}

#[test]
fn response_exhaustion_after_complete_verification_publishes_no_parent_or_children() {
    let mut measured = session();
    register(&mut measured, "x^2-2");
    measured
        .certification_request(request("publication", "-2", "2", 512))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    assert!(spent > 1);
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut s = Session::with_budget_limits(limited, 256);
    register(&mut s, "x^2-2");
    assert_eq!(
        s.certification_request(request("publication", "-2", "2", 512))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(
        s.budget_snapshot().verifier_pool,
        after.verifier_pool,
        "the same complete independent check ran before output admission refused"
    );
    assert_unpublished(&s, "publication");
}

#[test]
fn exhausted_ordinary_fuel_cannot_borrow_the_protected_reserve() {
    let mut measured = session();
    register(&mut measured, "x^2-2");
    let registration_spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - measured.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] = registration_spent;
    let mut s = Session::with_budget_limits(limited, 256);
    register(&mut s, "x^2-2");
    assert_eq!(
        s.budget_snapshot().dimensions[Dimension::ComputeSteps.index()],
        0
    );
    let protected = s.budget_snapshot().verifier_pool;
    assert_eq!(
        s.certification_request(request("exhausted", "-2", "2", 512))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(s.budget_snapshot().verifier_pool, protected);
    assert_unpublished(&s, "exhausted");
}

#[test]
fn no_verifier_fuel_refuses_even_with_ordinary_resources_available() {
    let mut limited = limits();
    limited.verifier_pool = 0;
    let mut s = Session::with_budget_limits(limited, 256);
    register(&mut s, "x^2-2");
    assert_eq!(
        s.certification_request(request("unverified", "-2", "2", 512))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(s.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    assert_unpublished(&s, "unverified");
}

#[test]
fn isolation_wire_refuses_unknown_duplicate_nonexact_and_unbounded_inputs() {
    assert!(RootIsolationId::new("certificate:wrong-kind").is_err());
    assert!(RootIsolationId::new("isolation:").is_err());
    assert!(RootIsolationId::new(format!("isolation:{}", "x".repeat(94))).is_ok());
    assert!(RootIsolationId::new(format!("isolation:{}", "x".repeat(95))).is_err());
    let mut s = session();
    register(&mut s, "x^2-2");
    let original = serde_json::to_value(request("wire", "-2", "2", 512)).unwrap();
    let mut future = original.clone();
    future["payload"]["schema_version"] = json!(2);
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&future.to_string())).unwrap()["code"],
        "unsupported_schema"
    );
    let mut unknown = original.clone();
    unknown["payload"]["accepted"] = json!(true);
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&unknown.to_string())).unwrap()["code"],
        "malformed_request"
    );
    let mut nonexact = original.clone();
    nonexact["payload"]["domain"][0]["lower"] = json!(-2);
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&nonexact.to_string())).unwrap()["code"],
        "malformed_request"
    );
    let mut unbounded = original;
    unbounded["payload"]["max_leaves"] = json!(1025);
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&unbounded.to_string())).unwrap()["code"],
        "resource_limit_exceeded"
    );
    let duplicate = r#"{"type":"IsolateRealRootsInBox","payload":{"schema_version":1,"system":"system:equations","isolation":"isolation:wire","domain":[{"lower":"-2","upper":"2"}],"precision_digits":16,"max_leaves":0,"max_leaves":512}}"#;
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(duplicate)).unwrap()["code"],
        "malformed_request"
    );
    assert_unpublished(&s, "wire");
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
            "one response before stdin closes"
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
fn registration() -> Value {
    json!({"id":"register","type":"RegisterResidualSystem","payload":{
        "schema_version":1,"system":"system:equations","residuals":["x^2-2"],"variables":["x"]
    }})
}

#[test]
fn default_cli_finds_both_roots_and_exports_each_for_fresh_process_replay() {
    let mut sender = Cli::new(&[]);
    assert_eq!(sender.send(registration())["status"], "success");
    let isolated = sender.send(serde_json::to_value(request("roots", "-2", "2", 512)).unwrap());
    assert_eq!(isolated["status"], "success", "{isolated}");
    assert_eq!(isolated["result"]["root_count"], 2);
    let roots = isolated["result"]["roots"].as_array().unwrap();
    assert_eq!(roots.len(), 2);
    let mut artifacts = Vec::new();
    for (index, summary) in roots.iter().enumerate() {
        assert_eq!(summary["certificate"], child("roots", index).as_str());
        let exported = sender.send(json!({"type":"ExportRootCertificate","payload":{
            "schema_version":1,"certificate":summary["certificate"]
        }}));
        assert_eq!(exported["status"], "success", "{exported}");
        let artifact = exported["result"]["artifact"].as_str().unwrap().to_owned();
        assert!(artifact.len() < frankensympy::MAX_ENVELOPE_BYTES);
        artifacts.push(artifact);
    }
    sender.finish();
    let mut receiver = Cli::new(&[]);
    assert_eq!(receiver.send(registration())["status"], "success");
    for (index, artifact) in artifacts.into_iter().enumerate() {
        let replayed = receiver.send(json!({"id":index,"type":"ReplayRootCertificate","payload":{
            "schema_version":1,"system":"system:equations",
            "certificate":format!("certificate:replayed-{index}"),"artifact":artifact
        }}));
        assert_eq!(replayed["status"], "success", "{replayed}");
        assert_eq!(replayed["id"], index);
        for field in ["domain", "root_enclosure", "contraction_bound"] {
            assert_eq!(replayed["result"][field], roots[index][field], "{field}");
        }
        assert_eq!(
            replayed["result"]["context_digest"],
            isolated["result"]["context_digest"]
        );
    }
    receiver.finish();
}

#[test]
fn cli_without_verifier_fuel_never_exports_search_candidates_as_certificates() {
    let mut cli = Cli::new(&["--verifier-steps", "0"]);
    assert_eq!(cli.send(registration())["status"], "success");
    let refused = cli.send(serde_json::to_value(request("unverified", "-2", "2", 512)).unwrap());
    assert_eq!(refused["code"], "budget_exhausted", "{refused}");
    assert_eq!(
        cli.send(json!({"type":"ExportRootCertificate","payload":{
            "schema_version":1,"certificate":"certificate:unverified.root-0"
        }}))["code"],
        "unknown_object"
    );
    assert_eq!(
        cli.send(json!({"type":"Eval","payload":{"expr":"1+1"}}))["result"],
        "2"
    );
    cli.finish();
}
