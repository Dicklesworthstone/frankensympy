//! Interactive global box optimization and fresh-process proof exchange.

use frankensympy::{ARTIFACT_CHUNK_BYTES, ArtifactKind, Session, artifact_content_id};
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

fn registration(source: &str) -> Value {
    json!({"id":"register","type":"RegisterObjective","payload":{
        "schema_version":1,"objective":"objective:global",
        "source":source,"variables":["x"]
    }})
}

fn domain() -> Value {
    json!([{"lower":"-1","upper":"1"}])
}

fn optimize(extremum: &str) -> Value {
    json!({"id":"optimize","type":"OptimizeBox","payload":{
        "schema_version":1,"objective":"objective:global","optimization":"optimization:sent",
        "domain":domain(),"extremum":extremum,"absolute_tolerance":"1/1000",
        "precision_digits":24,"max_leaves":256
    }})
}

fn export() -> Value {
    json!({"type":"ExportBoxOptimization","payload":{
        "schema_version":1,"optimization":"optimization:sent"
    }})
}

#[test]
fn default_cli_certifies_minimum_and_maximum_then_replays_in_fresh_processes() {
    for (source, extremum, point, value) in
        [("x^2", "minimum", "0", "0"), ("x", "maximum", "1", "1")]
    {
        let mut sender = Cli::new(&[]);
        assert_eq!(sender.send(registration(source))["status"], "success");
        let checked = sender.send(optimize(extremum));
        assert_eq!(checked["status"], "success", "{checked}");
        assert_eq!(checked["id"], "optimize");
        assert_eq!(checked["result"]["kind"], "box_optimized");
        assert_eq!(checked["result"]["evidence"], "certified_numeric");
        assert_eq!(checked["result"]["extremum"], extremum);
        assert_eq!(checked["result"]["point"], json!([point]));
        assert_eq!(
            checked["result"]["value"],
            json!({"lower":value,"upper":value})
        );
        assert_eq!(checked["result"]["gap"], "0");
        let exported = sender.send(export());
        assert_eq!(exported["status"], "success", "{exported}");
        let artifact = exported["result"]["artifact"].as_str().unwrap().to_owned();
        let content: Value = serde_json::from_str(&artifact).unwrap();
        for conclusion in ["accepted", "verified", "value", "gap", "point_value"] {
            assert!(content.get(conclusion).is_none());
        }
        sender.finish();

        let mut receiver = Cli::new(&[]);
        assert_eq!(receiver.send(registration(source))["status"], "success");
        let replayed = receiver.send(json!({"id":"replay","type":"ReplayBoxOptimization","payload":{
            "schema_version":1,"objective":"objective:global","optimization":"optimization:received",
            "domain":domain(),"extremum":extremum,"absolute_tolerance":"1/1000","artifact":artifact
        }}));
        assert_eq!(replayed["status"], "success", "{replayed}");
        assert_eq!(replayed["id"], "replay");
        for field in [
            "evidence",
            "claim",
            "extremum",
            "domain",
            "point",
            "value",
            "point_value",
            "absolute_tolerance",
            "gap",
            "leaf_count",
            "precision_digits",
            "context_digest",
        ] {
            assert_eq!(
                replayed["result"][field], checked["result"][field],
                "{field}"
            );
        }
        receiver.finish();
    }
}

#[test]
fn documented_nonconvex_global_minimum_is_certified_by_the_actual_cli() {
    let mut cli = Cli::new(&[]);
    assert_eq!(cli.send(registration("(x^2-2)^2"))["status"], "success");
    let mut request = optimize("minimum");
    request["payload"]["domain"] = json!([{"lower":"-2","upper":"2"}]);
    request["payload"]["precision_digits"] = json!(16);
    request["payload"]["max_leaves"] = json!(512);
    let checked = cli.send(request);
    assert_eq!(checked["status"], "success", "{checked}");
    let scalar = |value: &Value| {
        value
            .as_str()
            .unwrap()
            .parse::<frankensympy::BigRational>()
            .unwrap()
    };
    let zero = frankensympy::BigRational::from_integer(0.into());
    let tolerance = frankensympy::BigRational::new(1.into(), 1000.into());
    assert!(scalar(&checked["result"]["value"]["lower"]) <= zero);
    assert!(scalar(&checked["result"]["value"]["upper"]) >= zero);
    assert!(scalar(&checked["result"]["gap"]) <= tolerance);
    assert!(scalar(&checked["result"]["point_value"]["lower"]) > zero);
    assert!(checked["result"]["leaf_count"].as_u64().unwrap() > 1);
    assert_eq!(cli.send(export())["status"], "success");
    cli.finish();
}

#[test]
fn multichunk_optimization_proof_replays_with_the_original_large_source() {
    // The source remains below the immutable 4096-byte input cap while its
    // complete typed statement exceeds the inline proof transport envelope.
    let source = format!("x{}", "+0*x".repeat(600));
    // Source-preserving search and independent verification share one
    // cumulative allocation ledger even when verification has reserved steps.
    let arguments = [
        "--compute-steps",
        "100000000",
        "--allocations",
        "10000000",
    ];
    let mut sender = Cli::new(&arguments);
    let registered = sender.send(registration(&source));
    assert_eq!(registered["status"], "success", "{registered}");
    let checked = sender.send(optimize("minimum"));
    assert_eq!(checked["status"], "success", "{checked}");
    assert_eq!(
        checked["result"]["value"],
        json!({"lower":"-1","upper":"-1"})
    );
    let exported = sender.send(json!({"type":"BeginArtifactExport","payload":{
        "schema_version":1,"transfer":"transfer:send",
        "source":{"box_optimization":"optimization:sent"}
    }}));
    assert_eq!(exported["status"], "success", "{exported}");
    let manifest = exported["result"]["manifest"].clone();
    assert_eq!(manifest["kind"], "box_optimization");
    let byte_count = manifest["total_bytes"].as_u64().unwrap() as usize;
    assert!(
        byte_count > 24 * 1024,
        "actual artifact bytes: {byte_count}"
    );
    assert_eq!(sender.send(export())["code"], "resource_limit_exceeded");
    let chunks = byte_count.div_ceil(ARTIFACT_CHUNK_BYTES);
    assert!(chunks > 1);

    let mut receiver = Cli::new(&arguments);
    assert_eq!(receiver.send(registration(&source))["status"], "success");
    let imported = receiver.send(json!({"type":"BeginArtifactImport","payload":{
        "schema_version":1,"transfer":"transfer:receive","manifest":manifest
    }}));
    assert_eq!(imported["status"], "success", "{imported}");
    let replay = json!({"type":"ReplayBoxOptimizationFromArtifact","payload":{
        "schema_version":1,"objective":"objective:global","optimization":"optimization:received",
        "domain":domain(),"extremum":"minimum","absolute_tolerance":"1/1000",
        "transfer":"transfer:receive"
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
        assert_eq!(receiver.send(append)["status"], "success", "exact retry");
    }
    sender.finish();
    let unverified = receiver.send(json!({"type":"ExportBoxOptimization","payload":{
        "schema_version":1,"optimization":"optimization:received"
    }}));
    assert_eq!(unverified["code"], "unknown_object");
    let mut wrong_tolerance = replay.clone();
    wrong_tolerance["payload"]["absolute_tolerance"] = json!("1/100");
    assert_eq!(receiver.send(wrong_tolerance)["code"], "statement_mismatch");
    let accepted = receiver.send(replay);
    assert_eq!(accepted["status"], "success", "{accepted}");
    assert_eq!(accepted["result"]["value"], checked["result"]["value"]);
    assert_eq!(accepted["result"]["point"], checked["result"]["point"]);
    receiver.finish();
}

#[test]
fn no_verifier_reserve_never_publishes_a_global_optimization() {
    let mut cli = Cli::new(&["--verifier-steps", "0"]);
    assert_eq!(cli.send(registration("x^2"))["status"], "success");
    let response = cli.send(optimize("minimum"));
    assert_eq!(response["code"], "budget_exhausted", "{response}");
    assert_eq!(cli.send(export())["code"], "unknown_object");
    cli.finish();
}

#[test]
fn native_wire_refuses_unknown_fields_duplicate_tolerance_and_kind_typos() {
    let mut session = Session::with_budget_limits(
        frankensympy::BudgetLimits::uniform(200_000_000, 100_000_000),
        256,
    );
    let registration: Value =
        serde_json::from_str(&session.handle_envelope(&registration("x^2").to_string())).unwrap();
    assert_eq!(registration["status"], "success");
    for (field, value, expected) in [
        ("verified", json!(true), "malformed_request"),
        ("extremum", json!("min"), "malformed_request"),
        ("schema_version", json!(2), "unsupported_schema"),
    ] {
        let mut request = optimize("minimum");
        request["payload"][field] = value;
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
        assert_eq!(response["code"], expected, "{response}");
    }
    let duplicate = r#"{"type":"OptimizeBox","payload":{"schema_version":1,"objective":"objective:global","optimization":"optimization:sent","domain":[{"lower":"-1","upper":"1"}],"extremum":"minimum","absolute_tolerance":"1/1000","absolute_tolerance":"10","precision_digits":24,"max_leaves":256}}"#;
    let response: Value = serde_json::from_str(&session.handle_envelope(duplicate)).unwrap();
    assert_eq!(response["code"], "malformed_request");
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&export().to_string())).unwrap();
    assert_eq!(response["code"], "unknown_object");
    let checked: Value =
        serde_json::from_str(&session.handle_envelope(&optimize("minimum").to_string())).unwrap();
    assert_eq!(checked["status"], "success", "{checked}");
}

#[test]
fn optimization_artifact_integrity_identity_is_distinct_from_other_proof_kinds() {
    let bytes = b"same candidate bytes";
    let mut meter = fsym_budget::Unbounded;
    let optimization =
        artifact_content_id(ArtifactKind::BoxOptimization, bytes, &mut meter).unwrap();
    for kind in [
        ArtifactKind::RootCertificate,
        ArtifactKind::RootIsolation,
        ArtifactKind::IntegralCertificate,
        ArtifactKind::RootSearchCheckpoint,
        ArtifactKind::StationaryCertificate,
    ] {
        assert_ne!(
            optimization,
            artifact_content_id(kind, bytes, &mut meter).unwrap()
        );
    }
}
