//! Root-existence certificates through the public session and fresh NDJSON processes.

use frankensympy::{
    ArtifactKind, ArtifactSource, ArtifactTransferId, BigRational, BudgetLimits,
    CertificationRequest, CertificationResponse, Dimension, ExactInterval, MeterError,
    ResidualSystemId, RootCertificateId, ScalarRootBracketId, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(20_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits
}
fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:scalar").unwrap()
}
fn bracket(name: &str) -> ScalarRootBracketId {
    ScalarRootBracketId::new(format!("bracket:{name}")).unwrap()
}
fn registration(source: &str) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec![source.into()],
        variables: vec!["x".into()],
    }
}
fn registered(source: &str) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    session.certification_request(registration(source)).unwrap();
    session
}
fn certify(name: &str, lower: &str, upper: &str) -> CertificationRequest {
    CertificationRequest::CertifyScalarRootBracket {
        schema_version: 1,
        system: system(),
        bracket: bracket(name),
        lower: lower.into(),
        upper: upper.into(),
        absolute_tolerance: "1/1000".into(),
        precision_digits: 24,
        max_bisections: 64,
    }
}
fn export(session: &mut Session, name: &str) -> String {
    let CertificationResponse::ScalarRootBracketExported { artifact, .. } = session
        .certification_request(CertificationRequest::ExportScalarRootBracket {
            schema_version: 1,
            bracket: bracket(name),
        })
        .unwrap()
    else {
        panic!("bracket export response");
    };
    artifact
}
fn replay(name: &str, artifact: &str) -> CertificationRequest {
    CertificationRequest::ReplayScalarRootBracket {
        schema_version: 1,
        system: system(),
        bracket: bracket(name),
        lower: "-1".into(),
        upper: "1".into(),
        absolute_tolerance: "1/1000".into(),
        artifact: artifact.into(),
    }
}
fn expect_failure(session: &mut Session, request: CertificationRequest, name: &str, code: &str) {
    let error = session.certification_request(request).unwrap_err();
    assert_eq!(error.code(), code, "{error}");
    assert!(session.scalar_root_bracket(&bracket(name)).is_none());
}

#[test]
fn repeated_nonsmooth_and_endpoint_roots_publish_existence_only() {
    for (source, lower, upper, expected) in [
        ("(x-1)^3", "0", "2", "1"),
        ("x+abs(x)/2", "-1", "1", "0"),
        ("(x-2)^2", "2", "3", "2"),
        ("(x-2)^2", "2", "2", "2"),
    ] {
        let mut session = registered(source);
        session.construct("x", "100").unwrap();
        let before = session.budget_snapshot().verifier_pool;
        let CertificationResponse::ScalarRootBracketCertified {
            evidence,
            claim,
            enclosure,
            endpoint_values,
            context_digest,
            ..
        } = session
            .certification_request(certify("root", lower, upper))
            .unwrap()
        else {
            panic!("bracket response");
        };
        assert_eq!(evidence, "certified_numeric");
        assert_eq!(claim, "at_least_one_real_root_in_closed_interval");
        assert_eq!(
            enclosure,
            ExactInterval {
                lower: expected.into(),
                upper: expected.into()
            }
        );
        assert_eq!(
            endpoint_values,
            vec![
                ExactInterval {
                    lower: "0".into(),
                    upper: "0".into()
                };
                2
            ]
        );
        assert_eq!(context_digest, session.context_digest());
        assert!(session.budget_snapshot().verifier_pool < before);
        assert!(session.scalar_root_bracket(&bracket("root")).is_some());
    }
    assert!(RootCertificateId::new("bracket:root").is_err());
    assert!(ScalarRootBracketId::new("certificate:root").is_err());
}

#[test]
fn irrational_and_transcendental_roots_have_exact_width_and_sign_bounds() {
    for source in ["(x^2-2)^3", "cos(x)-x"] {
        let (lower, upper) = if source.starts_with('(') {
            ("1", "2")
        } else {
            ("0", "1")
        };
        let mut session = registered(source);
        session
            .certification_request(certify("root", lower, upper))
            .unwrap();
        let receipt = session.scalar_root_bracket(&bracket("root")).unwrap();
        assert!(receipt.root_enclosure().width() <= BigRational::new(1.into(), 1000.into()));
        let values = receipt.endpoint_values();
        let zero = BigRational::from_integer(0.into());
        assert!(
            (values[0].upper() < zero && values[1].lower() > zero)
                || (values[1].upper() < zero && values[0].lower() > zero)
        );
        if source.starts_with('(') {
            let lo = receipt.root_enclosure().lower();
            let hi = receipt.root_enclosure().upper();
            let two = BigRational::from_integer(2.into());
            assert!(&lo * &lo < two && &hi * &hi > two);
        } else {
            assert!(receipt.root_enclosure().lower() > BigRational::new(73.into(), 100.into()));
            assert!(receipt.root_enclosure().upper() < BigRational::new(75.into(), 100.into()));
        }
    }
}

#[test]
fn broad_bracket_with_three_roots_never_claims_uniqueness_or_completeness() {
    let mut session = registered("x^3-x");
    let mut request = certify("several", "-2", "2");
    if let CertificationRequest::CertifyScalarRootBracket {
        absolute_tolerance,
        max_bisections,
        ..
    } = &mut request
    {
        *absolute_tolerance = "4".into();
        *max_bisections = 0;
    }
    let response = serde_json::to_value(session.certification_request(request).unwrap()).unwrap();
    assert_eq!(
        response["claim"],
        "at_least_one_real_root_in_closed_interval"
    );
    assert_eq!(response["enclosure"], json!({"lower":"-2","upper":"2"}));
    assert!(response.get("completeness").is_none());
    assert!(response.get("unique").is_none());
    assert!(
        session
            .root_certificate(&RootCertificateId::new("certificate:several").unwrap())
            .is_none()
    );
}

#[test]
fn original_interval_poles_refuse_even_when_a_root_exists_elsewhere() {
    for (source, lower, upper) in [
        ("1/x", "-1", "1"),
        ("(x-1)/x", "-2", "2"),
        ("x+0*(1/x)", "-1", "1"),
        ("log(x)", "-1", "2"),
    ] {
        let mut session = registered(source);
        assert!(
            session
                .certification_request(certify("invalid", lower, upper))
                .is_err(),
            "{source}"
        );
        assert!(session.scalar_root_bracket(&bracket("invalid")).is_none());
    }
}

#[test]
fn inconclusive_signs_and_exhausted_search_publish_nothing_and_allow_retry() {
    for source in ["x^2", "x^2+1"] {
        let mut session = registered(source);
        expect_failure(
            &mut session,
            certify("unknown", "-1", "1"),
            "unknown",
            "inconclusive",
        );
    }
    let mut session = registered("x^2-2");
    let mut request = certify("retry", "1", "2");
    if let CertificationRequest::CertifyScalarRootBracket { max_bisections, .. } = &mut request {
        *max_bisections = 0;
    }
    expect_failure(&mut session, request, "retry", "inconclusive");
    let after = session.budget_snapshot();
    session
        .certification_request(certify("retry", "1", "2"))
        .unwrap();
    assert!(
        session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
            < after.dimensions[Dimension::ComputeSteps.index()]
    );
    let old = session
        .scalar_root_bracket(&bracket("retry"))
        .unwrap()
        .clone();
    assert_eq!(
        session
            .certification_request(certify("retry", "1", "2"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    assert_eq!(session.scalar_root_bracket(&bracket("retry")), Some(&old));
}

#[test]
fn invalid_scalars_shapes_and_controls_cannot_publish() {
    for (field, value, code) in [
        ("absolute_tolerance", json!("0"), "invalid_input"),
        ("absolute_tolerance", json!("-1"), "invalid_input"),
        ("lower", json!("1/0"), "invalid_input"),
        ("upper", json!("NaN"), "invalid_input"),
        ("lower", json!("2"), "invalid_input"),
        ("precision_digits", json!(7), "invalid_input"),
        ("max_bisections", json!(1025), "resource_limit_exceeded"),
    ] {
        let mut session = registered("x");
        let mut request = serde_json::to_value(certify("invalid", "-1", "1")).unwrap();
        request["payload"][field] = value;
        expect_failure(
            &mut session,
            serde_json::from_value(request).unwrap(),
            "invalid",
            code,
        );
    }
    for (sources, axes) in [(vec!["x", "x^2"], vec!["x"]), (vec!["x+y"], vec!["x", "y"])] {
        let mut session = Session::with_budget_limits(limits(), 256);
        session
            .certification_request(CertificationRequest::RegisterResidualSystem {
                schema_version: 1,
                system: system(),
                residuals: sources.into_iter().map(str::to_owned).collect(),
                variables: axes.into_iter().map(str::to_owned).collect(),
            })
            .unwrap();
        expect_failure(
            &mut session,
            certify("invalid", "-1", "1"),
            "invalid",
            "invalid_input",
        );
    }
}

#[test]
fn replay_binds_the_exact_source_statement_and_receiver_context() {
    let mut sender = registered("x^3");
    sender
        .certification_request(certify("source", "-1", "1"))
        .unwrap();
    let artifact = export(&mut sender, "source");
    let mut receiver = registered("x^3");
    let before = receiver.budget_snapshot().verifier_pool;
    receiver
        .certification_request(replay("copy", &artifact))
        .unwrap();
    assert!(receiver.budget_snapshot().verifier_pool < before);
    assert_eq!(
        sender.scalar_root_bracket(&bracket("source")),
        receiver.scalar_root_bracket(&bracket("copy"))
    );
    for (field, value) in [
        ("lower", "-2"),
        ("upper", "2"),
        ("absolute_tolerance", "1/100"),
    ] {
        let mut request = serde_json::to_value(replay("wrong", &artifact)).unwrap();
        request["payload"][field] = json!(value);
        expect_failure(
            &mut receiver,
            serde_json::from_value(request).unwrap(),
            "wrong",
            "statement_mismatch",
        );
    }
    let mut different = registered("x^3+1");
    expect_failure(
        &mut different,
        replay("wrong", &artifact),
        "wrong",
        "source_mismatch",
    );
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(HashMap::new(), HashMap::new(), "other bracket context")
        .unwrap();
    let mut different = Session::with_context_and_budget_limits(context, limits(), 256);
    different
        .certification_request(registration("x^3"))
        .unwrap();
    expect_failure(
        &mut different,
        replay("wrong", &artifact),
        "wrong",
        "universe_mismatch",
    );
}

#[test]
fn forged_acceptance_and_noncanonical_bytes_refuse() {
    let mut sender = registered("x");
    sender
        .certification_request(certify("source", "-1", "1"))
        .unwrap();
    let artifact = export(&mut sender, "source");
    let mut forged: Value = serde_json::from_str(&artifact).unwrap();
    forged["accepted"] = json!(true);
    for bytes in [forged.to_string(), format!(" {artifact}"), "{}".into()] {
        let mut receiver = registered("x");
        expect_failure(
            &mut receiver,
            replay("forged", &bytes),
            "forged",
            "invalid_artifact",
        );
    }
}

#[test]
fn protected_verifier_allowance_is_required_and_cumulative() {
    let mut measured = registered("x");
    let before = measured.budget_snapshot().verifier_pool;
    measured
        .certification_request(certify("one", "-1", "1"))
        .unwrap();
    let spent = before - measured.budget_snapshot().verifier_pool;
    assert!(spent > 0);
    for allowance in [0, spent] {
        let mut limited = limits();
        limited.verifier_pool = allowance;
        let mut session = Session::with_budget_limits(limited, 256);
        session.certification_request(registration("x")).unwrap();
        if allowance > 0 {
            session
                .certification_request(certify("one", "-1", "1"))
                .unwrap();
        }
        expect_failure(
            &mut session,
            certify("unverified", "-1", "1"),
            "unverified",
            "budget_exhausted",
        );
        assert_eq!(session.budget_snapshot().verifier_pool, 0);
        assert!(session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    }
}

#[test]
fn failed_output_admission_keeps_an_independently_checked_bracket_unpublished() {
    let mut measured = registered("x");
    measured
        .certification_request(certify("output", "-1", "1"))
        .unwrap();
    let after = measured.budget_snapshot();
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] -=
        after.dimensions[Dimension::ComputeSteps.index()] + 1;
    let mut session = Session::with_budget_limits(limited, 256);
    session.certification_request(registration("x")).unwrap();
    expect_failure(
        &mut session,
        certify("output", "-1", "1"),
        "output",
        "budget_exhausted",
    );
    assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
}

#[test]
fn cancellation_reaches_generation_verification_and_final_publication() {
    let mut measured = registered("x");
    let mut checkpoints = 0;
    let before = measured.budget_snapshot().verifier_pool;
    measured
        .certification_request_with_checkpoint(certify("cancel", "-1", "1"), || {
            checkpoints += 1;
            Ok(())
        })
        .unwrap();
    let verification_work = before - measured.budget_snapshot().verifier_pool;
    let mut interrupted_verification = false;
    for target in [
        1,
        checkpoints / 4,
        checkpoints / 2,
        checkpoints * 3 / 4,
        checkpoints * 7 / 8,
        checkpoints - 1,
        checkpoints,
    ] {
        let mut session = registered("x");
        let mut observed = 0;
        let before = session.budget_snapshot();
        let error = session
            .certification_request_with_checkpoint(certify("cancel", "-1", "1"), || {
                observed += 1;
                if observed == target {
                    Err(MeterError::Cancelled)
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(error.code(), "cancelled");
        assert_eq!(observed, target);
        assert!(session.scalar_root_bracket(&bracket("cancel")).is_none());
        let after = session.budget_snapshot();
        let work = before.verifier_pool - after.verifier_pool;
        interrupted_verification |= work > 0 && work < verification_work;
        if target == checkpoints {
            assert_eq!(work, verification_work);
        }
        session
            .certification_request(certify("cancel", "-1", "1"))
            .unwrap();
        assert!(session.budget_snapshot().verifier_pool < after.verifier_pool);
    }
    assert!(interrupted_verification);
}

#[test]
fn chunked_brackets_have_a_distinct_kind_and_require_protected_replay() {
    let mut sender = registered("x^3");
    sender
        .certification_request(certify("source", "-1", "1"))
        .unwrap();
    let transfer = ArtifactTransferId::new("transfer:bracket").unwrap();
    let CertificationResponse::ArtifactExportStarted { manifest, .. } = sender
        .certification_request(CertificationRequest::BeginArtifactExport {
            schema_version: 1,
            transfer: transfer.clone(),
            source: ArtifactSource::ScalarRootBracket(bracket("source")),
        })
        .unwrap()
    else {
        panic!("artifact manifest");
    };
    assert_eq!(manifest.kind, ArtifactKind::ScalarRootBracket);
    let mut receiver = registered("x^3");
    receiver
        .certification_request(CertificationRequest::BeginArtifactImport {
            schema_version: 1,
            transfer: transfer.clone(),
            manifest: manifest.clone(),
        })
        .unwrap();
    let request = CertificationRequest::ReplayScalarRootBracketFromArtifact {
        schema_version: 1,
        system: system(),
        bracket: bracket("copy"),
        lower: "-1".into(),
        upper: "1".into(),
        absolute_tolerance: "1/1000".into(),
        transfer: transfer.clone(),
    };
    expect_failure(
        &mut receiver,
        request.clone(),
        "copy",
        "incomplete_transfer",
    );
    let protected = receiver.budget_snapshot().verifier_pool;
    for index in 0..manifest.chunk_count() {
        let CertificationResponse::ArtifactChunkRead { chunk, .. } = sender
            .certification_request(CertificationRequest::ReadArtifactChunk {
                schema_version: 1,
                transfer: transfer.clone(),
                index,
            })
            .unwrap()
        else {
            panic!("artifact chunk");
        };
        receiver
            .certification_request(CertificationRequest::AppendArtifactChunk {
                schema_version: 1,
                transfer: transfer.clone(),
                chunk,
            })
            .unwrap();
    }
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    assert!(receiver.scalar_root_bracket(&bracket("copy")).is_none());
    assert_eq!(
        receiver
            .certification_request(CertificationRequest::ReplayRootCertificateFromArtifact {
                schema_version: 1,
                system: system(),
                certificate: RootCertificateId::new("certificate:wrong-kind").unwrap(),
                transfer,
            })
            .unwrap_err()
            .code(),
        "artifact_kind_mismatch"
    );
    receiver.certification_request(request).unwrap();
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    assert_eq!(
        sender.scalar_root_bracket(&bracket("source")),
        receiver.scalar_root_bracket(&bracket("copy"))
    );
}

fn process(requests: &[Value]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut responses = Vec::new();
    for request in requests {
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        responses.push(serde_json::from_str(&line).unwrap());
    }
    drop(input);
    assert!(child.wait().unwrap().success());
    responses
}

#[test]
fn fresh_default_cli_replays_a_flat_root_in_another_process() {
    let registered = serde_json::to_value(registration("x^3")).unwrap();
    let mut export_request = serde_json::to_value(CertificationRequest::ExportScalarRootBracket {
        schema_version: 1,
        bracket: bracket("source"),
    })
    .unwrap();
    export_request["id"] = json!("portable-root");
    let sent = process(&[
        registered.clone(),
        serde_json::to_value(certify("source", "-1", "1")).unwrap(),
        export_request,
    ]);
    assert_eq!(sent[1]["status"], "success", "{}", sent[1]);
    assert_eq!(sent[2]["id"], "portable-root");
    let artifact = sent[2]["result"]["artifact"].as_str().unwrap();
    let received = process(&[
        registered,
        serde_json::to_value(replay("copy", artifact)).unwrap(),
    ]);
    assert_eq!(received[1]["status"], "success", "{}", received[1]);
    assert_eq!(
        received[1]["result"]["kind"],
        "scalar_root_bracket_certified"
    );
    assert_eq!(
        received[1]["result"]["claim"],
        "at_least_one_real_root_in_closed_interval"
    );
    assert_eq!(
        received[1]["result"]["enclosure"],
        json!({"lower":"0","upper":"0"})
    );
}

#[test]
fn cli_refusals_preserve_stream_and_never_export_a_failed_bracket() {
    let replies = process(&[
        serde_json::to_value(registration("1/x")).unwrap(),
        serde_json::to_value(certify("pole", "-1", "1")).unwrap(),
        json!({"type":"ExportScalarRootBracket","payload":{"schema_version":1,"bracket":"bracket:pole"}}),
        json!({"type":"CertifyScalarRootBracket","payload":{
            "schema_version":99,"system":"system:scalar","bracket":"bracket:bad-version",
            "lower":"-1","upper":"1","absolute_tolerance":"1/1000",
            "precision_digits":24,"max_bisections":64
        }}),
    ]);
    assert_eq!(replies[1]["code"], "domain_not_certified", "{}", replies[1]);
    assert_eq!(replies[2]["code"], "unknown_object");
    assert_eq!(replies[3]["code"], "unsupported_schema");
}
