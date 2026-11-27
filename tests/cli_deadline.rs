//! Fresh-process CLI deadlines, framing, and unchanged ordinary responses.

use frankensympy::MAX_ENVELOPE_BYTES;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(arguments: &[&str], requests: &[String]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut input = child.stdin.take().unwrap();
        for request in requests {
            writeln!(input, "{request}").unwrap();
        }
    }
    child.wait_with_output().unwrap()
}

fn responses(output: &Output) -> Vec<Value> {
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    std::str::from_utf8(&output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn registration() -> Value {
    json!({"id":"same-id","type":"RegisterResidualSystem","payload":{
        "schema_version":1,"system":"system:deadline-cli",
        "residuals":["x^2-2"],"variables":["x"]
    }})
}

fn certify() -> Value {
    json!({"id":"same-id","type":"CertifyRootBox","payload":{
        "schema_version":1,"system":"system:deadline-cli",
        "certificate":"certificate:deadline-cli",
        "witness":{
            "domain":[{"lower":"7/5","upper":"3/2"}],
            "center":["29/20"],"preconditioner":[["1/3"]],"precision_digits":16
        }
    }})
}

fn export() -> Value {
    json!({"id":"same-id","type":"ExportRootCertificate","payload":{
        "schema_version":1,"certificate":"certificate:deadline-cli"
    }})
}

fn assert_timing(response: &Value, allowance_ms: u64, triggered: bool) {
    let timing = &response["request_timing"];
    assert_eq!(timing["allowance_ms"], allowance_ms, "{response}");
    assert_eq!(timing["deadline_triggered"], triggered, "{response}");
    assert!(timing["elapsed_us"].is_u64(), "{response}");
}

#[test]
fn default_legacy_response_bytes_stay_unchanged() {
    let request = json!({"id":"ordinary","type":"Eval","payload":{"expr":"2+2"}});
    let output = run(&[], &[request.to_string()]);
    let parsed = responses(&output);
    assert_eq!(parsed.len(), 1);
    let expected = json!({"id":"ordinary","status":"success","result":"4"});
    assert_eq!(
        std::str::from_utf8(&output.stdout).unwrap(),
        format!("{expected}\n")
    );
    assert!(parsed[0].get("request_timing").is_none());
}

#[test]
fn zero_deadline_returns_timed_out_per_frame_without_unparsed_ids() {
    let request = registration().to_string();
    let output = run(
        &["--request-timeout-ms", "0"],
        &[request.clone(), "{broken".into(), request],
    );
    let parsed = responses(&output);
    assert_eq!(parsed.len(), 3);
    for response in parsed {
        assert_eq!(response["status"], "error");
        assert_eq!(response["code"], "timed_out");
        assert!(response.get("id").is_none());
        assert!(response.get("result").is_none());
        assert_timing(&response, 0, true);
    }
}

#[test]
fn timed_cli_compiles_certifies_and_exports_the_same_mathematical_result() {
    let requests = [
        registration(),
        json!({"id":"same-id","type":"CompileResidualSystem","payload":{
            "schema_version":1,"system":"system:deadline-cli","program":"program:deadline-cli"
        }}),
        certify(),
        export(),
    ]
    .map(|request| request.to_string());
    let ordinary = responses(&run(&[], &requests));
    let timed = responses(&run(&["--request-timeout-ms", "60000"], &requests));
    assert_eq!(ordinary.len(), 4);
    assert_eq!(timed.len(), 4);
    for (expected, mut actual) in ordinary.into_iter().zip(timed.iter().cloned()) {
        assert_eq!(actual["status"], "success", "{actual}");
        assert_eq!(actual["id"], "same-id");
        assert_timing(&actual, 60_000, false);
        actual.as_object_mut().unwrap().remove("request_timing");
        assert_eq!(actual, expected);
    }
    assert_eq!(timed[2]["result"]["contraction_bound"], "1/15");
    assert_eq!(timed[2]["result"]["evidence"], "certificate_verified");
    assert_eq!(
        timed[2]["result"]["claim"],
        "unique_nonsingular_real_root_in_box"
    );
    assert!(!timed[3]["result"]["artifact"].as_str().unwrap().is_empty());
}

#[test]
fn deadline_mode_refuses_legacy_requests_and_keeps_the_session_usable() {
    let requests = [
        json!({"id":1,"type":"Bind","payload":{"symbol":"a","expr":"7"}}),
        json!({"id":2,"type":"Eval","payload":{"expr":"2+2"}}),
        json!({"id":3,"type":"ConstructClaim","payload":{"source":"x=x"}}),
        registration(),
        certify(),
    ]
    .map(|request| request.to_string());
    let parsed = responses(&run(
        &["--request-timeout-ms", "18446744073709551615"],
        &requests,
    ));
    assert_eq!(parsed.len(), 5);
    for (index, response) in parsed[..3].iter().enumerate() {
        assert_eq!(response["code"], "unsupported_operation");
        assert_eq!(response["id"], index + 1);
        assert_timing(response, u64::MAX, false);
    }
    for response in &parsed[3..] {
        assert_eq!(response["status"], "success", "{response}");
        assert_timing(response, u64::MAX, false);
    }
}

#[test]
fn resource_exhaustion_remains_distinct_from_a_deadline() {
    let requests = [registration(), certify(), export()].map(|request| request.to_string());
    let parsed = responses(&run(
        &["--request-timeout-ms", "60000", "--verifier-steps", "0"],
        &requests,
    ));
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0]["status"], "success");
    assert_eq!(parsed[1]["code"], "budget_exhausted");
    assert_eq!(parsed[2]["code"], "unknown_object");
    for response in parsed {
        assert_timing(&response, 60_000, false);
    }

    let parsed = responses(&run(
        &["--request-timeout-ms", "60000", "--compute-steps", "0"],
        &[registration().to_string()],
    ));
    assert_eq!(parsed[0]["code"], "budget_exhausted");
    assert_eq!(parsed[0]["id"], "same-id");
    assert_timing(&parsed[0], 60_000, false);
}

#[test]
fn malformed_and_oversized_frames_keep_typed_errors_and_frame_boundaries() {
    let mut future = registration();
    future["payload"]["schema_version"] = json!(2);
    let mut invalid_id = registration();
    invalid_id["id"] = json!({"not":"an id"});
    let duplicate = r#"{"id":"duplicate","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:one","system":"system:two","residuals":["x"],"variables":["x"]}}"#;
    let requests = [
        "x".repeat(MAX_ENVELOPE_BYTES + 1),
        "{broken".into(),
        future.to_string(),
        invalid_id.to_string(),
        duplicate.into(),
        registration().to_string(),
    ];
    let parsed = responses(&run(&["--request-timeout-ms", "60000"], &requests));
    assert_eq!(parsed.len(), requests.len());
    for (response, code) in parsed.iter().zip([
        "request_too_large",
        "malformed_request",
        "unsupported_schema",
        "invalid_request_id",
        "malformed_request",
    ]) {
        assert_eq!(response["status"], "error");
        assert_eq!(response["code"], code, "{response}");
    }
    // Framing rejects oversize before starting the request execution timer.
    assert!(parsed[0].get("request_timing").is_none());
    for response in &parsed[1..] {
        assert_timing(response, 60_000, false);
    }
    assert_eq!(parsed.last().unwrap()["status"], "success");
}

#[test]
fn malformed_deadline_options_fail_before_starting_a_session() {
    for arguments in [
        vec!["--request-timeout-ms"],
        vec!["--request-timeout-ms", "-1"],
        vec!["--request-timeout-ms", "0.5"],
        vec!["--request-timeout-ms", "18446744073709551616"],
        vec!["--request-timeout-ms", "1", "--request-timeout-ms", "2"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args(&arguments)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{arguments:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{arguments:?}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("--request-timeout-ms"),
            "{arguments:?}: {output:?}"
        );
    }
}
