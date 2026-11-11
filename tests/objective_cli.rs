//! Real interactive NDJSON and cross-process stationary certificate replay.

use frankensympy::{ArtifactKind, Session, artifact_content_id};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

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
        Self {
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        }
    }

    fn send(&mut self, request: Value) -> Value {
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
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
    json!({"id":"register","type":"RegisterObjective","payload":{
        "schema_version":1,"objective":"objective:quadratic",
        "source":"(x-1)^2+2*(y+2)^2","variables":["x","y"]
    }})
}

fn domain() -> Value {
    json!([{"lower":"0","upper":"2"},{"lower":"-3","upper":"-1"}])
}

fn certificate() -> Value {
    json!({"id":"check","type":"CertifyStationaryPoint","payload":{
        "schema_version":1,"objective":"objective:quadratic","certificate":"stationary:minimum",
        "witness":{"domain":domain(),"center":["1","-2"],
        "preconditioner":[["1/2","0"],["0","1/4"]],"precision_digits":24}
    }})
}

#[test]
fn interactive_cli_checks_derivatives_then_replays_the_minimum_in_a_fresh_process() {
    let mut sender = Cli::new(&[]);
    assert_eq!(sender.send(registration())["status"], "success");
    let bounds = sender.send(json!({"id":"bounds","type":"EncloseObjective","payload":{
        "schema_version":1,"objective":"objective:quadratic","domain":domain(),"precision_digits":24
    }}));
    assert_eq!(bounds["status"], "success", "{bounds}");
    assert_eq!(bounds["result"]["kind"], "objective_enclosed");
    assert_eq!(
        bounds["result"]["hessian"],
        json!([
            [{"lower":"2","upper":"2"},{"lower":"0","upper":"0"}],
            [{"lower":"0","upper":"0"},{"lower":"4","upper":"4"}]
        ])
    );
    let checked = sender.send(certificate());
    assert_eq!(checked["status"], "success", "{checked}");
    assert_eq!(checked["result"]["classification"], "strict_minimum");
    assert_eq!(
        checked["result"]["stationary_enclosure"],
        json!([
            {"lower":"1","upper":"1"},{"lower":"-2","upper":"-2"}
        ])
    );
    let exported = sender.send(json!({"type":"ExportStationaryCertificate","payload":{
        "schema_version":1,"certificate":"stationary:minimum"
    }}));
    assert_eq!(exported["status"], "success", "{exported}");
    let artifact = exported["result"]["artifact"].as_str().unwrap().to_owned();
    sender.finish();

    let mut receiver = Cli::new(&[]);
    assert_eq!(receiver.send(registration())["status"], "success");
    let replayed = receiver.send(json!({"id":"replay","type":"ReplayStationaryCertificate","payload":{
        "schema_version":1,"objective":"objective:quadratic","certificate":"stationary:received",
        "domain":domain(),"artifact":artifact
    }}));
    assert_eq!(replayed["status"], "success", "{replayed}");
    assert_eq!(replayed["id"], "replay");
    for field in [
        "claim",
        "classification",
        "domain",
        "stationary_enclosure",
        "hessian_enclosure",
        "contraction_bound",
        "context_digest",
    ] {
        assert_eq!(
            replayed["result"][field], checked["result"][field],
            "{field}"
        );
    }
    receiver.finish();
}

#[test]
fn chunked_stationary_artifact_replays_after_integrity_checks_in_another_process() {
    let mut sender = Cli::new(&[]);
    assert_eq!(sender.send(registration())["status"], "success");
    assert_eq!(sender.send(certificate())["status"], "success");
    let export = sender.send(json!({"type":"BeginArtifactExport","payload":{
        "schema_version":1,"transfer":"transfer:send",
        "source":{"stationary_certificate":"stationary:minimum"}
    }}));
    assert_eq!(export["status"], "success", "{export}");
    let manifest = export["result"]["manifest"].clone();
    assert_eq!(manifest["kind"], "stationary_certificate");
    let chunks = (manifest["total_bytes"].as_u64().unwrap() as usize)
        .div_ceil(frankensympy::ARTIFACT_CHUNK_BYTES);
    let mut receiver = Cli::new(&[]);
    assert_eq!(receiver.send(registration())["status"], "success");
    assert_eq!(
        receiver.send(json!({"type":"BeginArtifactImport","payload":{
            "schema_version":1,"transfer":"transfer:receive","manifest":manifest
        }}))["status"],
        "success"
    );
    let replay = json!({"type":"ReplayStationaryCertificateFromArtifact","payload":{
        "schema_version":1,"objective":"objective:quadratic","certificate":"stationary:received",
        "domain":domain(),"transfer":"transfer:receive"
    }});
    assert_eq!(receiver.send(replay.clone())["code"], "incomplete_transfer");
    for index in 0..chunks {
        let read = sender.send(json!({"type":"ReadArtifactChunk","payload":{
            "schema_version":1,"transfer":"transfer:send","index":index
        }}));
        assert_eq!(read["status"], "success", "{read}");
        let append = json!({"type":"AppendArtifactChunk","payload":{
            "schema_version":1,"transfer":"transfer:receive","chunk":read["result"]["chunk"]
        }});
        assert_eq!(receiver.send(append.clone())["status"], "success");
        assert_eq!(
            receiver.send(append)["status"],
            "success",
            "exact retries are idempotent"
        );
    }
    sender.finish();
    let checked = receiver.send(replay);
    assert_eq!(checked["status"], "success", "{checked}");
    assert_eq!(checked["result"]["classification"], "strict_minimum");
    assert_eq!(
        receiver.send(json!({"type":"RegisterResidualSystem","payload":{
            "schema_version":1,"system":"system:roots","residuals":["x-1"],"variables":["x"]
        }}))["status"],
        "success"
    );
    let wrong_kind = receiver.send(
        json!({"type":"ReplayRootCertificateFromArtifact","payload":{
            "schema_version":1,"system":"system:roots","certificate":"certificate:wrong-kind",
            "transfer":"transfer:receive"
        }}),
    );
    assert_eq!(wrong_kind["code"], "artifact_kind_mismatch", "{wrong_kind}");
    receiver.finish();
}

#[test]
fn no_verifier_resources_never_publishes_an_accepted_stationary_handle() {
    let mut cli = Cli::new(&["--verifier-steps", "0"]);
    assert_eq!(cli.send(registration())["status"], "success");
    let refused = cli.send(certificate());
    assert_eq!(refused["code"], "budget_exhausted", "{refused}");
    assert_eq!(
        cli.send(json!({"type":"ExportStationaryCertificate","payload":{
            "schema_version":1,"certificate":"stationary:minimum"
        }}))["code"],
        "unknown_object"
    );
    cli.finish();
}

#[test]
fn objective_wire_rejects_duplicate_fields_forged_evidence_and_unknown_schema() {
    let mut session = Session::with_budget_limits(
        frankensympy::BudgetLimits::uniform(100_000_000, 100_000_000),
        256,
    );
    let original = registration();
    let mut future = original.clone();
    future["payload"]["schema_version"] = json!(2);
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&future.to_string())).unwrap();
    assert_eq!(response["code"], "unsupported_schema");
    let mut forged = original.clone();
    forged["payload"]["verified"] = json!(true);
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&forged.to_string())).unwrap();
    assert_eq!(response["code"], "malformed_request");
    let duplicate = r#"{"type":"RegisterObjective","payload":{"schema_version":1,"objective":"objective:quadratic","source":"0","source":"x^2","variables":["x"]}}"#;
    let response: Value = serde_json::from_str(&session.handle_envelope(duplicate)).unwrap();
    assert_eq!(response["code"], "malformed_request");
    let accepted: Value =
        serde_json::from_str(&session.handle_envelope(&original.to_string())).unwrap();
    assert_eq!(accepted["status"], "success", "{accepted}");
}

#[test]
fn stationary_artifact_integrity_ids_have_a_distinct_kind_domain() {
    let bytes = b"same untrusted bytes";
    let stationary = artifact_content_id(
        ArtifactKind::StationaryCertificate,
        bytes,
        &mut fsym_budget::Unbounded,
    )
    .unwrap();
    for kind in [
        ArtifactKind::RootCertificate,
        ArtifactKind::RootIsolation,
        ArtifactKind::IntegralCertificate,
        ArtifactKind::RootSearchCheckpoint,
    ] {
        assert_ne!(
            stationary,
            artifact_content_id(kind, bytes, &mut fsym_budget::Unbounded).unwrap()
        );
    }
}
