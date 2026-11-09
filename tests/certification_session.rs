//! End-to-end native sessions and real fresh-process certificate transport.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval, ExactRootWitness,
    ResidualSystemId, RootCertificateId, Session,
};
use fsym_budget::Dimension;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(2_000_000, 10_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits
}

fn session() -> Session {
    Session::with_budget_limits(limits(), 256)
}
fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:sqrt2").unwrap()
}
fn certificate(name: &str) -> RootCertificateId {
    RootCertificateId::new(format!("certificate:{name}")).unwrap()
}
fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}

fn register(s: &mut Session, residuals: &[&str], variables: &[&str]) {
    s.certification_request(CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: residuals.iter().map(|s| (*s).into()).collect(),
        variables: variables.iter().map(|s| (*s).into()).collect(),
    })
    .unwrap();
}

fn witness() -> ExactRootWitness {
    ExactRootWitness {
        domain: vec![interval("1.4", "3/2")],
        center: vec!["29/20".into()],
        preconditioner: vec![vec!["1/3".into()]],
        precision_digits: 24,
    }
}

fn certify(name: &str) -> CertificationRequest {
    CertificationRequest::CertifyRootBox {
        schema_version: 1,
        system: system(),
        certificate: certificate(name),
        witness: witness(),
    }
}

fn export(s: &mut Session, name: &str) -> String {
    let response = s
        .certification_request(CertificationRequest::ExportRootCertificate {
            schema_version: 1,
            certificate: certificate(name),
        })
        .unwrap();
    let CertificationResponse::RootCertificateExported { artifact, .. } = response else {
        panic!("export response")
    };
    artifact
}

#[test]
fn exact_affine_enclosures_match_hand_derived_values_and_variable_order() {
    let mut s = session();
    register(&mut s, &["2*x+y-5", "x-3*y+2"], &["x", "y"]);
    let response = s
        .certification_request(CertificationRequest::EncloseResidualJacobian {
            schema_version: 1,
            system: system(),
            domain: vec![interval("1", "2"), interval("1", "1")],
            precision_digits: 24,
        })
        .unwrap();
    let CertificationResponse::ResidualJacobianEnclosed {
        residuals,
        jacobian,
        evidence,
        claim,
        ..
    } = response
    else {
        panic!("enclosure response")
    };
    assert_eq!(residuals, [interval("-2", "0"), interval("0", "1")]);
    assert_eq!(
        jacobian,
        [
            vec![interval("2", "2"), interval("1", "1")],
            vec![interval("1", "1"), interval("-3", "-3")]
        ]
    );
    assert_eq!(evidence, "certified_numeric");
    assert_eq!(claim, "whole_box_c1_residual_jacobian_enclosure");
}

#[test]
fn root_certificate_is_source_bound_and_workspace_mutation_cannot_change_it() {
    let mut s = session();
    register(&mut s, &["x^2-2"], &["x"]);
    s.construct("x", "0").unwrap();
    let response = s.certification_request(certify("original")).unwrap();
    let CertificationResponse::RootBoxCertified {
        contraction_bound,
        evidence,
        claim,
        ..
    } = response
    else {
        panic!("root response")
    };
    assert_eq!(contraction_bound, "1/15");
    assert_eq!(evidence, "certificate_verified");
    assert_eq!(claim, "unique_nonsingular_real_root_in_box");
    let original = s.root_certificate(&certificate("original")).unwrap();
    assert_eq!(
        original.residuals(),
        s.residual_problem(&system()).unwrap().residuals()
    );
    let artifact = export(&mut s, "original");
    let mut receiver = session();
    register(&mut receiver, &["x^2-2"], &["x"]);
    receiver
        .certification_request(CertificationRequest::ReplayRootCertificate {
            schema_version: 1,
            system: system(),
            certificate: certificate("replayed"),
            artifact,
        })
        .unwrap();
    assert_eq!(
        s.root_certificate(&certificate("original")),
        receiver.root_certificate(&certificate("replayed"))
    );
}

#[test]
fn undefined_original_source_cannot_be_hidden_by_zero_or_cancellation() {
    for source in ["0*(1/x)", "log(x)-log(x)", "0/(x-x)", "x+1/0"] {
        let mut s = session();
        register(&mut s, &[source], &["x"]);
        let result = s.certification_request(CertificationRequest::EncloseResidualJacobian {
            schema_version: 1,
            system: system(),
            domain: vec![interval("0", "0")],
            precision_digits: 24,
        });
        assert!(result.is_err(), "accepted undefined source {source}");
    }
}

#[test]
fn failed_mathematical_check_leaves_certificate_id_available_for_valid_retry() {
    let mut s = session();
    register(&mut s, &["x^2-2"], &["x"]);
    let before = s.budget_snapshot();
    let mut invalid = witness();
    invalid.preconditioner[0][0] = "0".into();
    let result = s.certification_request(CertificationRequest::CertifyRootBox {
        schema_version: 1,
        system: system(),
        certificate: certificate("retry"),
        witness: invalid,
    });
    assert!(result.is_err());
    assert!(s.root_certificate(&certificate("retry")).is_none());
    assert!(s.budget_snapshot().verifier_pool < before.verifier_pool);
    s.certification_request(certify("retry")).unwrap();
    assert!(s.root_certificate(&certificate("retry")).is_some());
    assert_eq!(
        s.certification_request(certify("retry"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn protected_verification_fuel_is_cumulative_and_cannot_borrow_ordinary_fuel() {
    let mut measured = session();
    register(&mut measured, &["x^2-2"], &["x"]);
    let before = measured.budget_snapshot();
    measured.certification_request(certify("once")).unwrap();
    let spent = before.verifier_pool - measured.budget_snapshot().verifier_pool;
    assert!(spent > 0);
    let mut exact_limits = limits();
    exact_limits.verifier_pool = spent;
    let mut s = Session::with_budget_limits(exact_limits, 256);
    register(&mut s, &["x^2-2"], &["x"]);
    s.certification_request(certify("once")).unwrap();
    assert_eq!(s.budget_snapshot().verifier_pool, 0);
    let failure = s.certification_request(certify("twice")).unwrap_err();
    assert_eq!(failure.code(), "budget_exhausted");
    assert!(s.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    assert!(s.root_certificate(&certificate("twice")).is_none());
    assert!(s.root_certificate(&certificate("once")).is_some());
}

#[test]
fn response_budget_failure_after_verification_cannot_publish_a_certificate() {
    let mut measured = session();
    register(&mut measured, &["x^2-2"], &["x"]);
    measured
        .certification_request(certify("publication"))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut s = Session::with_budget_limits(limited, 256);
    register(&mut s, &["x^2-2"], &["x"]);
    let error = s.certification_request(certify("publication")).unwrap_err();
    assert_eq!(error.code(), "budget_exhausted");
    assert_eq!(
        s.budget_snapshot().verifier_pool,
        after.verifier_pool,
        "the same complete independent verification ran before output refused"
    );
    assert!(s.root_certificate(&certificate("publication")).is_none());
}

#[test]
fn replay_rejects_different_equations_axes_and_forged_witness_without_publication() {
    let mut sender = session();
    register(&mut sender, &["x^2-2"], &["x"]);
    sender.certification_request(certify("original")).unwrap();
    let artifact = export(&mut sender, "original");
    for (source, axis) in [("x^2-3", "x"), ("y^2-2", "y")] {
        let mut receiver = session();
        register(&mut receiver, &[source], &[axis]);
        let error = receiver
            .certification_request(CertificationRequest::ReplayRootCertificate {
                schema_version: 1,
                system: system(),
                certificate: certificate("forged"),
                artifact: artifact.clone(),
            })
            .unwrap_err();
        assert_eq!(error.code(), "source_mismatch");
        assert!(receiver.root_certificate(&certificate("forged")).is_none());
    }
    let mut forged: Value = serde_json::from_str(&artifact).unwrap();
    forged["witness"]["preconditioner"][0][0] = serde_json::to_value(
        fsym_core::BigRational::from_integer(fsym_core::BigInt::from(0)),
    )
    .unwrap();
    let mut receiver = session();
    register(&mut receiver, &["x^2-2"], &["x"]);
    let error = receiver
        .certification_request(CertificationRequest::ReplayRootCertificate {
            schema_version: 1,
            system: system(),
            certificate: certificate("forged"),
            artifact: forged.to_string(),
        })
        .unwrap_err();
    assert!(matches!(error.code(), "invalid_artifact" | "refused"));
    assert!(receiver.root_certificate(&certificate("forged")).is_none());
}

#[test]
fn wire_rejects_unknown_fields_versions_nonexact_scalars_and_large_ids() {
    let mut s = session();
    let mut request = json!({"type":"RegisterResidualSystem","id":"registration","payload":{
        "schema_version":1,"system":"system:sqrt2","residuals":["x^2-2"],"variables":["x"]
    }});
    request["payload"]["schema_version"] = json!(2);
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&request.to_string())).unwrap()["code"],
        "unsupported_schema"
    );
    assert!(s.residual_problem(&system()).is_none());
    request["payload"]["schema_version"] = json!(1);
    request["payload"]["accepted"] = json!(true);
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&request.to_string())).unwrap()["code"],
        "malformed_request"
    );
    request["payload"]
        .as_object_mut()
        .unwrap()
        .remove("accepted");
    request["id"] = json!("a".repeat(257));
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&request.to_string())).unwrap()["code"],
        "invalid_request_id"
    );
    assert!(s.residual_problem(&system()).is_none());
    request["id"] = json!("registration");
    assert_eq!(
        serde_json::from_str::<Value>(&s.handle_envelope(&request.to_string())).unwrap()["status"],
        "success"
    );
    for scalar in ["1e2", "nan", "1/0", "1+1", " 1", "0.1.2"] {
        let mut invalid = witness();
        invalid.center[0] = scalar.into();
        assert_eq!(
            s.certification_request(CertificationRequest::CertifyRootBox {
                schema_version: 1,
                system: system(),
                certificate: certificate("bad-scalar"),
                witness: invalid,
            })
            .unwrap_err()
            .code(),
            "invalid_input"
        );
    }
}

#[test]
fn duplicate_transport_keys_are_rejected_before_source_publication() {
    for request in [
        r#"{"type":"RegisterResidualSystem","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:sqrt2","residuals":["x^2-2"],"variables":["x"]}}"#,
        r#"{"type":"RegisterResidualSystem","payload":{"schema_version":2,"schema_version":1,"system":"system:sqrt2","residuals":["x^2-2"],"variables":["x"]}}"#,
        r#"{"id":"a","id":"b","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:sqrt2","residuals":["x^2-2"],"variables":["x"]}}"#,
    ] {
        let mut s = session();
        let response: Value = serde_json::from_str(&s.handle_envelope(request)).unwrap();
        assert_eq!(response["code"], "malformed_request");
        assert!(s.residual_problem(&system()).is_none());
    }
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
        "schema_version":1,"system":"system:sqrt2","residuals":["x^2-2"],"variables":["x"]
    }})
}

#[test]
fn interactive_cli_exports_and_replays_an_actual_root_in_a_fresh_process() {
    let mut sender = Cli::new(&[]);
    assert_eq!(sender.send(registration())["status"], "success");
    let response = sender.send(serde_json::to_value(certify("original")).unwrap());
    assert_eq!(response["status"], "success", "{response}");
    assert_eq!(response["result"]["contraction_bound"], "1/15");
    let original_result = response["result"].clone();
    let exported = sender.send(
        json!({"id":"export","type":"ExportRootCertificate","payload":{
            "schema_version":1,"certificate":"certificate:original"
        }}),
    );
    assert_eq!(exported["status"], "success", "{exported}");
    let artifact = exported["result"]["artifact"].as_str().unwrap();
    assert!(artifact.len() < frankensympy::MAX_ENVELOPE_BYTES);
    sender.finish();

    let mut receiver = Cli::new(&[]);
    assert_eq!(receiver.send(registration())["status"], "success");
    let replayed = receiver.send(json!({"id":"replay","type":"ReplayRootCertificate","payload":{
        "schema_version":1,"system":"system:sqrt2","certificate":"certificate:replayed","artifact":artifact
    }}));
    assert_eq!(replayed["id"], "replay");
    assert_eq!(replayed["status"], "success", "{replayed}");
    for field in [
        "domain",
        "root_enclosure",
        "contraction_bound",
        "context_digest",
        "claim",
        "evidence",
    ] {
        assert_eq!(replayed["result"][field], original_result[field], "{field}");
    }
    receiver.finish();
}

#[test]
fn cli_zero_verifier_budget_refuses_and_keeps_session_usable() {
    let mut cli = Cli::new(&["--verifier-steps", "0"]);
    assert_eq!(cli.send(registration())["status"], "success");
    assert_eq!(
        cli.send(serde_json::to_value(certify("blocked")).unwrap())["code"],
        "budget_exhausted"
    );
    assert_eq!(
        cli.send(json!({"type":"ExportRootCertificate","payload":{
            "schema_version":1,"certificate":"certificate:blocked"
        }}))["code"],
        "unknown_object"
    );
    assert_eq!(
        cli.send(json!({"type":"Eval","payload":{"expr":"1+1"}}))["result"],
        "2"
    );
    cli.finish();
}

#[test]
fn cli_reports_invalid_budget_options_without_starting_a_session() {
    let output = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
        .args(["--memory-bytes", "-1"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("nonnegative integer")
    );
}
