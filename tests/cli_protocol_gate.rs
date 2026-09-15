//! CLI and session gate tests for fra-rc-cli-u9o.
//!
//! Obligations: unknown schema, oversized frame, EOF mid-request, proof
//! pagination, stdout protocol purity, repeated request IDs, same-print
//! conflicting universe, candidate-versus-accepted distinction. The
//! spawned-CLI tests exercise a real fresh process; `Command::output`
//! waits for the child, so no uncontrolled orphans are possible.

#![forbid(unsafe_code)]

use frankensympy::{ClaimStatus, Session, SessionBudgets, SessionError};

fn spawn_cli(requests: &[String]) -> (String, String, i32) {
    let cli = env!("CARGO_BIN_EXE_frankensympy");
    use std::io::Write;
    let mut child = std::process::Command::new(cli)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawns the CLI");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        for request in requests {
            stdin.write_all(request.as_bytes()).expect("writes");
            stdin.write_all(b"\n").expect("writes newline");
        }
    }
    let output = child.wait_with_output().expect("child exits");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.code().unwrap_or(-1),
    )
}

#[test]
fn unknown_schema_gets_a_typed_malformed_response() {
    let (stdout, stderr, code) = spawn_cli(&[r#"{"type":"teleport","payload":{}}"#.to_string()]);
    assert_eq!(code, 0, "clean EOF exit");
    assert!(stderr.is_empty(), "stdout purity: stderr must be unused here");
    let response: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json line");
    assert_eq!(response["status"], "error");
    assert_eq!(response["code"], "malformed_request");
}

#[test]
fn oversized_frame_gets_a_typed_too_large_response() {
    let big = format!("{{\"junk\":\"{}\"}}", "x".repeat(70_000));
    let (stdout, _stderr, code) = spawn_cli(&[big]);
    assert_eq!(code, 0);
    let response: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json line");
    assert_eq!(response["status"], "error");
    assert!(
        stdout.contains("request_too_large")
            || response["error"]
                .as_str()
                .is_some_and(|e| e.contains("envelope")),
        "typed refusal required, got {response}"
    );
}

#[test]
fn eof_mid_request_exits_cleanly_with_one_response() {
    // A final line without a trailing newline is still processed, then EOF
    // ends the session with exit 0 and exactly one response line.
    let cli = env!("CARGO_BIN_EXE_frankensympy");
    use std::io::Write;
    let mut child = std::process::Command::new(cli)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawns");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(b"{\"type\":\"eval\"}") // no trailing newline: EOF mid-stream
        .expect("writes partial");
    let output = child.wait_with_output().expect("child exits");
    assert_eq!(output.status.code(), Some(0), "EOF must exit cleanly");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.lines().count(), 1, "exactly one response line");
}

#[test]
fn stdout_protocol_purity_responses_only() {
    let requests = vec![
        r#"{"type":"bind","symbol":"x","expr":"x^2 + 1"}"#.to_string(),
        r#"{"type":"diff","expr":"sin(x) + x^2","var":"x"}"#.to_string(),
        r#"{"type":"eval","expr":"x + 1"}"#.to_string(),
        "not json at all".to_string(),
    ];
    let (stdout, stderr, code) = spawn_cli(&requests);
    assert_eq!(code, 0);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), requests.len(), "one response per request");
    for line in &lines {
        let parsed: serde_json::Value =
            serde_json::from_str(line).expect("every stdout line is one JSON response");
        assert!(
            parsed.get("status").is_some(),
            "protocol purity: {line}"
        );
    }
    assert!(stderr.is_empty() || !stderr.contains("panic"));
}

#[test]
fn repeated_request_ids_are_echoed_and_processed_independently() {
    let requests = vec![
        r#"{"id":"req-7","type":"bind","symbol":"x","expr":"1"}"#.to_string(),
        r#"{"id":"req-7","type":"bind","symbol":"x","expr":"2"}"#.to_string(),
    ];
    let (stdout, _stderr, code) = spawn_cli(&requests);
    assert_eq!(code, 0);
    let lines: Vec<serde_json::Value> =
        stdout.lines().map(|l| serde_json::from_str(l).expect("json")).collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["id"], "req-7");
    assert_eq!(lines[1]["id"], "req-7");
    // Independent processing: the second bind wins; both succeeded.
    assert_eq!(lines[0]["status"], "success");
    assert_eq!(lines[1]["status"], "success");
}

#[test]
fn candidate_claims_are_never_accepted_and_exports_refuse_them() {
    let mut session = Session::new(SessionBudgets::default());
    let candidate = session.construct_claim("1 + 1").expect("claim");
    assert_eq!(candidate.status, ClaimStatus::Candidate);
    // Export refuses candidates outright: no unverified evidence leaves.
    assert!(matches!(
        session.export_claim(&candidate, 0, 16),
        Err(SessionError::CandidateNotAccepted)
    ));
    let accepted = session.verify_claim(&candidate).expect("verifies");
    assert_eq!(accepted.status, ClaimStatus::Accepted);
    assert!(session.export_claim(&accepted, 0, 16).is_ok());
}

#[test]
fn proof_pagination_splits_multi_step_derivations() {
    let mut session = Session::new(SessionBudgets::default());
    let candidate = session.construct_claim("42").expect("claim");
    let accepted = session.verify_claim(&candidate).expect("verifies");
    // The verify path builds a 2-step derivation (reflexivity + symmetry).
    let page0 = session.export_claim(&accepted, 0, 1).expect("page 0");
    let page1 = session.export_claim(&accepted, 1, 1).expect("page 1");
    assert_eq!(page0["page"], 0);
    assert_eq!(page1["page"], 1);
    assert_eq!(page0["total_pages"], 2);
    assert_eq!(page1["total_pages"], 2);
    assert!(matches!(
        session.export_claim(&accepted, 2, 1),
        Err(SessionError::PageOutOfRange { .. })
    ));
    // Oversized page size refuses.
    assert!(matches!(
        session.export_claim(&accepted, 0, 4096),
        Err(SessionError::PageSizeTooLarge { .. })
    ));
}

#[test]
fn same_print_symbols_from_conflicting_universes_refuse_import() {
    let context_a = {
        let mut context = fsym_assumptions::AssumptionsContext::new();
        let _ = context.assume(
            fsym_core::Symbol::new("x"),
            fsym_assumptions::Predicate::Real,
        );
        context.snapshot()
    };
    let context_b = {
        let mut context = fsym_assumptions::AssumptionsContext::new();
        let _ = context.assume(
            fsym_core::Symbol::new("x"),
            fsym_assumptions::Predicate::Integer,
        );
        context.snapshot()
    };
    assert_ne!(
        context_a.digest(),
        context_b.digest(),
        "the two universes must have distinct context digests"
    );

    let mut session_a = Session::with_context(
        std::sync::Arc::new(context_a),
        SessionBudgets::default(),
    );
    let candidate = session_a.construct_claim("x + 1").expect("claim");
    let accepted = session_a.verify_claim(&candidate).expect("verifies");
    let export = session_a.export_claim(&accepted, 0, 16).expect("exports");

    // Same-print symbol "x" exists in universe B too, but the universe
    // digests differ: importing A's export into B refuses.
    let mut session_b = Session::with_context(
        std::sync::Arc::new(context_b),
        SessionBudgets::default(),
    );
    let b_claim = session_b.construct_claim("x + 1").expect("claim in B");
    let _ = session_b.verify_claim(&b_claim).expect("verifies in B");
    assert!(matches!(
        session_b.replay_claim(&export, session_b.context_digest()),
        Err(SessionError::UniverseMismatch { .. })
    ));
}
