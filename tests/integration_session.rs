//! Proper integral certificates through the actual native and NDJSON boundary.

use frankensympy::{
    ArtifactKind, ArtifactSource, ArtifactTransferId, BigRational, BudgetLimits,
    CertificationRequest, CertificationResponse, ExactInterval, IntegralCertificateId,
    ResidualSystemId, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Dimension;
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
    ResidualSystemId::new("system:integrand").unwrap()
}

fn integral(name: &str) -> IntegralCertificateId {
    IntegralCertificateId::new(format!("integral:{name}")).unwrap()
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
    CertificationRequest::CertifyIntegral {
        schema_version: 1,
        system: system(),
        integral: integral(name),
        lower: lower.into(),
        upper: upper.into(),
        absolute_tolerance: "1/1000".into(),
        precision_digits: 16,
        max_intervals: 128,
    }
}

fn export(session: &mut Session, name: &str) -> String {
    let CertificationResponse::IntegralCertificateExported { artifact, .. } = session
        .certification_request(CertificationRequest::ExportIntegralCertificate {
            schema_version: 1,
            integral: integral(name),
        })
        .unwrap()
    else {
        panic!("integral export response");
    };
    artifact
}

fn replay(name: &str, artifact: &str) -> CertificationRequest {
    CertificationRequest::ReplayIntegralCertificate {
        schema_version: 1,
        system: system(),
        integral: integral(name),
        lower: "0".into(),
        upper: "3".into(),
        absolute_tolerance: "1/1000".into(),
        artifact: artifact.into(),
    }
}

#[test]
fn exact_quadratic_integral_retains_orientation_and_frozen_source() {
    let mut session = registered("x^2");
    session.construct("x", "100").unwrap();
    for (name, lower, upper, answer) in [
        ("forward", "0", "3", "9"),
        ("reverse", "3", "0", "-9"),
        ("point", "2", "2", "0"),
    ] {
        let CertificationResponse::IntegralCertified {
            enclosure,
            evidence,
            claim,
            context_digest,
            ..
        } = session
            .certification_request(certify(name, lower, upper))
            .unwrap()
        else {
            panic!("integral response");
        };
        assert_eq!(
            enclosure,
            ExactInterval {
                lower: answer.into(),
                upper: answer.into()
            }
        );
        assert_eq!(evidence, "certified_numeric");
        assert_eq!(
            claim,
            "proper_real_definite_integral_with_absolute_enclosure_width_bound"
        );
        assert_eq!(context_digest, session.context_digest());
        assert!(session.integral_certificate(&integral(name)).is_some());
    }
}

#[test]
fn transcendental_enclosure_has_requested_width_and_independent_reference_bounds() {
    let mut session = registered("exp(x)");
    session
        .certification_request(certify("exponential", "0", "1"))
        .unwrap();
    let result = session
        .integral_certificate(&integral("exponential"))
        .unwrap();
    let rational = |n: i64, d: i64| BigRational::new(n.into(), d.into());
    assert!(result.enclosure().lower() > rational(1717i64, 1000i64));
    assert!(result.enclosure().upper() < rational(1720i64, 1000i64));
    assert!(result.enclosure().upper() - result.enclosure().lower() <= rational(1, 1000));
    assert!(result.interval_count() > 1);
}

#[test]
fn original_poles_cannot_be_erased_and_refusals_publish_no_certificate() {
    for (source, lower, upper) in [
        ("0*(1/x)", "-1", "1"),
        ("log(x)-log(x)", "0", "1"),
        ("1/x", "0", "0"),
        ("sqrt(x)", "-1", "1"),
    ] {
        let mut session = registered(source);
        assert!(
            session
                .certification_request(certify("invalid", lower, upper))
                .is_err(),
            "{source}"
        );
        assert!(session.integral_certificate(&integral("invalid")).is_none());
    }
}

#[test]
fn adaptive_exhaustion_is_inconclusive_and_label_can_be_retried() {
    let mut session = registered("x^4");
    let mut request = certify("retry", "0", "1");
    if let CertificationRequest::CertifyIntegral { max_intervals, .. } = &mut request {
        *max_intervals = 1;
    }
    assert_eq!(
        session.certification_request(request).unwrap_err().code(),
        "inconclusive"
    );
    assert!(session.integral_certificate(&integral("retry")).is_none());
    session
        .certification_request(certify("retry", "0", "1"))
        .unwrap();
    assert!(session.integral_certificate(&integral("retry")).is_some());
    assert_eq!(
        session
            .certification_request(certify("retry", "0", "1"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn wrong_dimensions_and_nonpositive_tolerance_refuse_before_publication() {
    let mut session = Session::with_budget_limits(limits(), 256);
    session
        .certification_request(CertificationRequest::RegisterResidualSystem {
            schema_version: 1,
            system: system(),
            residuals: vec!["x+y".into()],
            variables: vec!["x".into(), "y".into()],
        })
        .unwrap();
    assert_eq!(
        session
            .certification_request(certify("wrong", "0", "1"))
            .unwrap_err()
            .code(),
        "invalid_input"
    );
    for tolerance in ["0", "-1", "1/0", "NaN"] {
        let mut session = registered("x^2");
        let mut request = certify("wrong", "0", "1");
        if let CertificationRequest::CertifyIntegral {
            absolute_tolerance, ..
        } = &mut request
        {
            *absolute_tolerance = tolerance.into();
        }
        assert_eq!(
            session.certification_request(request).unwrap_err().code(),
            "invalid_input"
        );
        assert!(session.integral_certificate(&integral("wrong")).is_none());
    }
}

#[test]
fn replay_binds_exact_source_bounds_tolerance_and_context() {
    let mut sender = registered("x^2");
    sender
        .certification_request(certify("source", "0", "3"))
        .unwrap();
    let artifact = export(&mut sender, "source");
    let mut receiver = registered("x^2");
    receiver
        .certification_request(replay("copy", &artifact))
        .unwrap();
    assert_eq!(
        sender.integral_certificate(&integral("source")),
        receiver.integral_certificate(&integral("copy"))
    );
    for field in ["lower", "upper", "absolute_tolerance"] {
        let mut request = serde_json::to_value(replay("wrong", &artifact)).unwrap();
        request["payload"][field] = json!("2");
        let request = serde_json::from_value(request).unwrap();
        assert_eq!(
            receiver.certification_request(request).unwrap_err().code(),
            "statement_mismatch"
        );
        assert!(receiver.integral_certificate(&integral("wrong")).is_none());
    }
    let mut different = registered("x^2+1");
    assert_eq!(
        different
            .certification_request(replay("wrong", &artifact))
            .unwrap_err()
            .code(),
        "source_mismatch"
    );
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(HashMap::new(), HashMap::new(), "different integral context")
        .unwrap();
    let mut different = Session::with_context_and_budget_limits(context, limits(), 256);
    different
        .certification_request(registration("x^2"))
        .unwrap();
    assert_eq!(
        different
            .certification_request(replay("wrong", &artifact))
            .unwrap_err()
            .code(),
        "universe_mismatch"
    );
}

#[test]
fn omitted_partition_endpoints_and_forged_acceptance_refuse() {
    let mut sender = registered("x^2");
    sender
        .certification_request(certify("source", "0", "3"))
        .unwrap();
    let artifact = export(&mut sender, "source");
    for mutation in ["partition", "accepted", "schema_version"] {
        let mut forged: Value = serde_json::from_str(&artifact).unwrap();
        match mutation {
            "partition" => {
                forged["witness"]["partition"].as_array_mut().unwrap().pop();
            }
            "accepted" => {
                forged["accepted"] = json!(true);
            }
            _ => {
                forged["schema_version"] = json!(99);
            }
        }
        let mut receiver = registered("x^2");
        assert!(
            receiver
                .certification_request(replay("forged", &forged.to_string()))
                .is_err()
        );
        assert!(receiver.integral_certificate(&integral("forged")).is_none());
    }
}

#[test]
fn protected_verification_budget_is_cumulative_and_never_borrowed() {
    let mut measured = registered("x^2");
    let before = measured.budget_snapshot();
    measured
        .certification_request(certify("first", "0", "3"))
        .unwrap();
    let spent = before.verifier_pool - measured.budget_snapshot().verifier_pool;
    assert!(spent > 0);
    let mut limited = limits();
    limited.verifier_pool = spent;
    let mut session = Session::with_budget_limits(limited, 256);
    session.certification_request(registration("x^2")).unwrap();
    session
        .certification_request(certify("first", "0", "3"))
        .unwrap();
    assert_eq!(session.budget_snapshot().verifier_pool, 0);
    assert_eq!(
        session
            .certification_request(certify("second", "0", "3"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    assert!(session.integral_certificate(&integral("second")).is_none());
}

#[test]
fn failed_output_admission_cannot_publish_an_independently_checked_integral() {
    let mut measured = registered("x^2");
    measured
        .certification_request(certify("output", "0", "3"))
        .unwrap();
    let after = measured.budget_snapshot();
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] -=
        after.dimensions[Dimension::ComputeSteps.index()] + 1;
    let mut session = Session::with_budget_limits(limited, 256);
    session.certification_request(registration("x^2")).unwrap();
    assert_eq!(
        session
            .certification_request(certify("output", "0", "3"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
    assert!(session.integral_certificate(&integral("output")).is_none());
}

#[test]
fn chunked_integral_transport_preserves_kind_and_requires_reverification() {
    let mut sender = registered("x^2");
    sender
        .certification_request(certify("source", "0", "3"))
        .unwrap();
    let transfer = ArtifactTransferId::new("transfer:integral").unwrap();
    let CertificationResponse::ArtifactExportStarted { manifest, .. } = sender
        .certification_request(CertificationRequest::BeginArtifactExport {
            schema_version: 1,
            transfer: transfer.clone(),
            source: ArtifactSource::IntegralCertificate(integral("source")),
        })
        .unwrap()
    else {
        panic!("export manifest");
    };
    assert_eq!(manifest.kind, ArtifactKind::IntegralCertificate);
    let mut receiver = registered("x^2");
    receiver
        .certification_request(CertificationRequest::BeginArtifactImport {
            schema_version: 1,
            transfer: transfer.clone(),
            manifest: manifest.clone(),
        })
        .unwrap();
    let replay = CertificationRequest::ReplayIntegralCertificateFromArtifact {
        schema_version: 1,
        system: system(),
        integral: integral("copy"),
        lower: "0".into(),
        upper: "3".into(),
        absolute_tolerance: "1/1000".into(),
        transfer: transfer.clone(),
    };
    assert_eq!(
        receiver
            .certification_request(replay.clone())
            .unwrap_err()
            .code(),
        "incomplete_transfer"
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
            panic!("chunk");
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
    assert!(receiver.integral_certificate(&integral("copy")).is_none());
    receiver.certification_request(replay).unwrap();
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    assert_eq!(
        sender.integral_certificate(&integral("source")),
        receiver.integral_certificate(&integral("copy"))
    );
}

#[test]
fn fresh_default_cli_exports_and_replays_integral_in_a_new_process() {
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
    let registered = serde_json::to_value(registration("x^2")).unwrap();
    let sent = process(&[
        registered.clone(),
        serde_json::to_value(certify("source", "0", "3")).unwrap(),
        json!({"id":"proof", "type":"ExportIntegralCertificate", "payload":{
            "schema_version":1, "integral":"integral:source"
        }}),
    ]);
    assert_eq!(sent[1]["status"], "success", "{}", sent[1]);
    assert_eq!(sent[2]["id"], "proof");
    let artifact = sent[2]["result"]["artifact"].as_str().unwrap();
    let received = process(&[
        registered,
        serde_json::to_value(replay("copy", artifact)).unwrap(),
    ]);
    assert_eq!(received[1]["status"], "success", "{}", received[1]);
    assert_eq!(
        received[1]["result"]["enclosure"],
        json!({"lower":"9","upper":"9"})
    );
}
