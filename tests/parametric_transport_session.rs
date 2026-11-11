//! Parameter-family transport preserves receiver-owned statements and publication.

use frankensympy::certification_session::MAX_SESSION_ROOT_ARTIFACT_BYTES;
use frankensympy::{
    ArtifactExport, ArtifactKind, ArtifactSource, ArtifactTransferId, BigRational, BudgetLimits,
    CertificationRequest, CertificationResponse, ExactInterval, ExactParametricRootWitness,
    ParametricFamilyId, ParametricProblemId, ParametricSensitivityId, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Dimension, Unbounded};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 2_000_000;
    limits
}

fn problem() -> ParametricProblemId {
    ParametricProblemId::new("parametric:transport").unwrap()
}

fn family(name: &str) -> ParametricFamilyId {
    ParametricFamilyId::new(format!("family:{name}")).unwrap()
}

fn sensitivity(name: &str) -> ParametricSensitivityId {
    ParametricSensitivityId::new(format!("sensitivity:{name}")).unwrap()
}

fn transfer(name: &str) -> ArtifactTransferId {
    ArtifactTransferId::new(format!("transfer:{name}")).unwrap()
}

fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}

fn unknown_domain() -> Vec<ExactInterval> {
    vec![interval("-2", "2")]
}

fn parameter_domain() -> Vec<ExactInterval> {
    vec![interval("-1", "1")]
}

fn registration(source: &str, unknown: &str, parameter: &str) -> CertificationRequest {
    CertificationRequest::RegisterParametricProblem {
        schema_version: 1,
        problem: problem(),
        residuals: vec![source.into()],
        unknowns: vec![unknown.into()],
        parameters: vec![parameter.into()],
    }
}

fn registered(source: &str) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    session
        .certification_request(registration(source, "x", "p"))
        .unwrap();
    session
}

fn certify(name: &str) -> CertificationRequest {
    CertificationRequest::CertifyParametricRoot {
        schema_version: 1,
        problem: problem(),
        family: family(name),
        witness: ExactParametricRootWitness {
            unknown_domain: unknown_domain(),
            parameter_domain: parameter_domain(),
            center: vec!["0".into()],
            preconditioner: vec![vec!["1".into()]],
            precision_digits: 16,
        },
    }
}

fn replay(name: &str, artifact: &str) -> CertificationRequest {
    CertificationRequest::ReplayParametricFamily {
        schema_version: 1,
        problem: problem(),
        family: family(name),
        unknown_domain: unknown_domain(),
        parameter_domain: parameter_domain(),
        artifact: artifact.into(),
    }
}

fn replay_transfer(name: &str, source: &str) -> CertificationRequest {
    CertificationRequest::ReplayParametricFamilyFromArtifact {
        schema_version: 1,
        problem: problem(),
        family: family(name),
        unknown_domain: unknown_domain(),
        parameter_domain: parameter_domain(),
        transfer: transfer(source),
    }
}

fn differentiate(name: &str) -> CertificationRequest {
    CertificationRequest::EncloseParametricSensitivity {
        schema_version: 1,
        family: family(name),
        sensitivity: sensitivity(name),
    }
}

fn export(session: &mut Session, name: &str) -> String {
    let protected = session.budget_snapshot().verifier_pool;
    let response = session
        .certification_request(CertificationRequest::ExportParametricFamily {
            schema_version: 1,
            family: family(name),
        })
        .unwrap();
    assert_eq!(session.budget_snapshot().verifier_pool, protected);
    let CertificationResponse::ParametricFamilyExported {
        problem: exported_problem,
        family: exported_family,
        artifact,
        ..
    } = response
    else {
        panic!("exported parameter family");
    };
    assert_eq!(exported_problem, problem());
    assert_eq!(exported_family, family(name));
    artifact
}

fn affine_artifact() -> String {
    let mut sender = registered("x-p");
    sender.certification_request(certify("sender")).unwrap();
    export(&mut sender, "sender")
}

fn unpublished(session: &Session, name: &str) {
    assert!(session.parametric_family(&family(name)).is_none());
    assert!(session.parametric_sensitivity(&sensitivity(name)).is_none());
}

#[test]
fn inline_uniform_replay_recomputes_the_family_and_implicit_derivative() {
    let mut sender = registered("x-p");
    sender.certification_request(certify("sender")).unwrap();
    let artifact = export(&mut sender, "sender");
    assert_eq!(artifact, export(&mut sender, "sender"));
    for forbidden in [
        "root_enclosure",
        "contraction_bound",
        "solution_jacobian",
        "accepted",
        "verified",
    ] {
        assert!(!artifact.contains(forbidden), "{forbidden}");
    }
    drop(sender);

    let mut receiver = registered("x-p");
    receiver.construct("x", "100").unwrap();
    receiver.construct("p", "200").unwrap();
    let protected = receiver.budget_snapshot().verifier_pool;
    let response = receiver
        .certification_request(replay("received", &artifact))
        .unwrap();
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    let CertificationResponse::ParametricFamilyCertified {
        evidence,
        claim,
        unknown_domain: unknown,
        parameter_domain: parameters,
        root_enclosure,
        leaf_count,
        ..
    } = response
    else {
        panic!("replayed parameter family");
    };
    assert_eq!(evidence, "certified_numeric");
    assert_eq!(
        claim,
        "for_every_parameter_exactly_one_nonsingular_root_in_unknown_box"
    );
    assert_eq!(unknown, unknown_domain());
    assert_eq!(parameters, parameter_domain());
    assert_eq!(root_enclosure, [interval("-1", "1")]);
    assert_eq!(leaf_count, 1);
    assert!(
        receiver
            .parametric_sensitivity(&sensitivity("received"))
            .is_none()
    );
    let protected = receiver.budget_snapshot().verifier_pool;
    let CertificationResponse::ParametricSensitivityEnclosed {
        solution_jacobian, ..
    } = receiver
        .certification_request(differentiate("received"))
        .unwrap()
    else {
        panic!("fresh implicit derivative");
    };
    assert_eq!(solution_jacobian, [vec![interval("1", "1")]]);
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    assert_eq!(artifact, export(&mut receiver, "received"));
}

#[test]
fn adaptive_cover_replay_retains_all_parameter_cells_and_analytic_sensitivities() {
    let mut sender = registered("(1+p^2)*x-1");
    sender
        .certification_request(CertificationRequest::CertifyParametricCover {
            schema_version: 1,
            problem: problem(),
            family: family("sender"),
            unknown_domain: vec![interval("0", "2")],
            parameter_domain: vec![interval("0", "2")],
            precision_digits: 16,
            max_leaves: 128,
        })
        .unwrap();
    let original_leaves = sender
        .parametric_family(&family("sender"))
        .unwrap()
        .leaf_count();
    assert!(original_leaves > 1);
    let artifact = export(&mut sender, "sender");
    drop(sender);
    let mut receiver = registered("(1+p^2)*x-1");
    receiver
        .certification_request(CertificationRequest::ReplayParametricFamily {
            schema_version: 1,
            problem: problem(),
            family: family("received"),
            unknown_domain: vec![interval("0", "2")],
            parameter_domain: vec![interval("0", "2")],
            artifact,
        })
        .unwrap();
    let receipt = receiver.parametric_family(&family("received")).unwrap();
    assert_eq!(receipt.leaf_count(), original_leaves);
    for (numerator, denominator) in [(1, 1), (1, 2), (1, 5)] {
        assert!(
            receipt.root_enclosure()[0]
                .contains(&BigRational::new(numerator.into(), denominator.into()))
        );
    }
    receiver
        .certification_request(differentiate("received"))
        .unwrap();
    let derivative = receiver
        .parametric_sensitivity(&sensitivity("received"))
        .unwrap();
    // Independently known x(p)=1/(1+p^2), with derivative -2p/(1+p^2)^2.
    for (numerator, denominator) in [(0, 1), (-1, 2), (-4, 25)] {
        assert!(
            derivative.solution_jacobian()[0][0]
                .contains(&BigRational::new(numerator.into(), denominator.into()))
        );
    }
}

#[test]
fn multichunk_family_import_grants_no_acceptance_until_protected_replay() {
    let source = format!("x-p{}", "+0*x".repeat(900));
    let mut sender = registered(&source);
    sender.certification_request(certify("sender")).unwrap();
    let protected = sender.budget_snapshot().verifier_pool;
    assert_eq!(
        sender
            .certification_request(CertificationRequest::ExportParametricFamily {
                schema_version: 1,
                family: family("sender"),
            })
            .unwrap_err()
            .code(),
        "resource_limit_exceeded"
    );
    let CertificationResponse::ArtifactExportStarted { manifest, .. } = sender
        .certification_request(CertificationRequest::BeginArtifactExport {
            schema_version: 1,
            transfer: transfer("out"),
            source: ArtifactSource::ParametricFamily(family("sender")),
        })
        .unwrap()
    else {
        panic!("chunked family export");
    };
    assert_eq!(manifest.kind, ArtifactKind::ParametricFamily);
    assert!(manifest.total_bytes > MAX_SESSION_ROOT_ARTIFACT_BYTES);
    assert!(manifest.chunk_count() > 1);
    assert_eq!(sender.budget_snapshot().verifier_pool, protected);

    let mut receiver = registered(&source);
    let protected = receiver.budget_snapshot().verifier_pool;
    receiver
        .certification_request(CertificationRequest::BeginArtifactImport {
            schema_version: 1,
            transfer: transfer("in"),
            manifest: manifest.clone(),
        })
        .unwrap();
    assert_eq!(
        receiver
            .certification_request(replay_transfer("received", "in"))
            .unwrap_err()
            .code(),
        "incomplete_transfer"
    );
    unpublished(&receiver, "received");
    for index in 0..manifest.chunk_count() {
        let CertificationResponse::ArtifactChunkRead { chunk, .. } = sender
            .certification_request(CertificationRequest::ReadArtifactChunk {
                schema_version: 1,
                transfer: transfer("out"),
                index,
            })
            .unwrap()
        else {
            panic!("family chunk");
        };
        receiver
            .certification_request(CertificationRequest::AppendArtifactChunk {
                schema_version: 1,
                transfer: transfer("in"),
                chunk,
            })
            .unwrap();
    }
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    unpublished(&receiver, "received");
    receiver
        .certification_request(replay_transfer("received", "in"))
        .unwrap();
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    receiver
        .certification_request(differentiate("received"))
        .unwrap();
    assert_eq!(
        receiver
            .parametric_sensitivity(&sensitivity("received"))
            .unwrap()
            .solution_jacobian()[0][0],
        frankensympy::RealBall::from_i64(1)
    );
    let CertificationResponse::ArtifactExportStarted {
        manifest: reexported,
        ..
    } = receiver
        .certification_request(CertificationRequest::BeginArtifactExport {
            schema_version: 1,
            transfer: transfer("again"),
            source: ArtifactSource::ParametricFamily(family("received")),
        })
        .unwrap()
    else {
        panic!("reexported family");
    };
    assert_eq!(reexported, manifest);
}

#[test]
fn source_roles_both_domains_and_context_cannot_be_substituted() {
    let artifact = affine_artifact();
    for (source, unknown, parameter) in [
        ("x-2*p", "x", "p"),
        ("x-p", "p", "x"),
        ("x-p+0*(1/p)", "x", "p"),
    ] {
        let mut receiver = Session::with_budget_limits(limits(), 256);
        receiver
            .certification_request(registration(source, unknown, parameter))
            .unwrap();
        assert_eq!(
            receiver
                .certification_request(replay("wrong", &artifact))
                .unwrap_err()
                .code(),
            "source_mismatch"
        );
        unpublished(&receiver, "wrong");
    }
    for unknown in [true, false] {
        let mut receiver = registered("x-p");
        let mut request = replay("wrong", &artifact);
        let CertificationRequest::ReplayParametricFamily {
            unknown_domain,
            parameter_domain,
            ..
        } = &mut request
        else {
            unreachable!();
        };
        if unknown {
            *unknown_domain = vec![interval("-3", "3")];
        } else {
            *parameter_domain = vec![interval("-1/2", "1/2")];
        }
        assert_eq!(
            receiver.certification_request(request).unwrap_err().code(),
            "domain_mismatch"
        );
        unpublished(&receiver, "wrong");
    }
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(
            HashMap::new(),
            HashMap::new(),
            "independent family receiver",
        )
        .unwrap();
    let mut receiver = Session::with_context_and_budget_limits(context, limits(), 256);
    receiver
        .certification_request(registration("x-p", "x", "p"))
        .unwrap();
    assert_eq!(
        receiver
            .certification_request(replay("wrong", &artifact))
            .unwrap_err()
            .code(),
        "universe_mismatch"
    );
    unpublished(&receiver, "wrong");
}

#[test]
fn a_forged_witness_does_not_publish_or_reset_the_budget_and_can_be_retried() {
    let artifact = affine_artifact();
    let matrix = |value: i64| {
        serde_json::to_string(&vec![vec![BigRational::from_integer(value.into())]]).unwrap()
    };
    let original = format!("\"preconditioner\":{}", matrix(1));
    let replacement = format!("\"preconditioner\":{}", matrix(0));
    assert!(artifact.contains(&original));
    let forged = artifact.replacen(&original, &replacement, 1);
    let mut receiver = registered("x-p");
    let before = receiver.budget_snapshot();
    assert_eq!(
        receiver
            .certification_request(replay("retry", &forged))
            .unwrap_err()
            .code(),
        "refused"
    );
    unpublished(&receiver, "retry");
    let after = receiver.budget_snapshot();
    assert!(after.verifier_pool < before.verifier_pool);
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    receiver
        .certification_request(replay("retry", &artifact))
        .unwrap();
    assert!(receiver.parametric_family(&family("retry")).is_some());
    assert!(receiver.budget_snapshot().verifier_pool < after.verifier_pool);
}

#[test]
fn complete_valid_bytes_with_the_wrong_artifact_kind_never_reach_acceptance() {
    let artifact = affine_artifact();
    let packed = ArtifactExport::new(
        ArtifactKind::RootCertificate,
        artifact.into_bytes(),
        &mut Unbounded,
    )
    .unwrap();
    let mut receiver = registered("x-p");
    let protected = receiver.budget_snapshot().verifier_pool;
    receiver
        .certification_request(CertificationRequest::BeginArtifactImport {
            schema_version: 1,
            transfer: transfer("wrong-kind"),
            manifest: packed.manifest().clone(),
        })
        .unwrap();
    for index in 0..packed.manifest().chunk_count() {
        receiver
            .certification_request(CertificationRequest::AppendArtifactChunk {
                schema_version: 1,
                transfer: transfer("wrong-kind"),
                chunk: packed.chunk(index, &mut Unbounded).unwrap(),
            })
            .unwrap();
    }
    assert_eq!(
        receiver
            .certification_request(replay_transfer("wrong", "wrong-kind"))
            .unwrap_err()
            .code(),
        "artifact_kind_mismatch"
    );
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    unpublished(&receiver, "wrong");
}

#[test]
fn verifier_exhaustion_and_final_response_refusal_publish_no_family() {
    let artifact = affine_artifact();
    let mut bounded = limits();
    bounded.verifier_pool = 0;
    let mut receiver = Session::with_budget_limits(bounded, 256);
    receiver
        .certification_request(registration("x-p", "x", "p"))
        .unwrap();
    assert_eq!(
        receiver
            .certification_request(replay("blocked", &artifact))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    unpublished(&receiver, "blocked");
    assert!(receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);

    let mut measured = registered("x-p");
    measured
        .certification_request(replay("publication", &artifact))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut bounded = limits();
    bounded.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut receiver = Session::with_budget_limits(bounded, 256);
    receiver
        .certification_request(registration("x-p", "x", "p"))
        .unwrap();
    assert_eq!(
        receiver
            .certification_request(replay("publication", &artifact))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(
        receiver.budget_snapshot().verifier_pool,
        after.verifier_pool
    );
    unpublished(&receiver, "publication");
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
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{}", serde_json::to_string(&request).unwrap()).unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["status"], "success", "{response}");
        response
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
fn separate_default_cli_processes_replay_the_family_and_rederive_sensitivity() {
    let mut sender = Cli::new();
    sender.send(registration("x-p", "x", "p"));
    let accepted = sender.send(certify("sender"));
    let exported = sender.send(CertificationRequest::ExportParametricFamily {
        schema_version: 1,
        family: family("sender"),
    });
    let artifact = exported["result"]["artifact"].as_str().unwrap().to_owned();
    sender.finish();
    let mut receiver = Cli::new();
    receiver.send(registration("x-p", "x", "p"));
    let replayed = receiver.send(replay("received", &artifact));
    for field in [
        "claim",
        "evidence",
        "unknown_domain",
        "parameter_domain",
        "root_enclosure",
        "contraction_bound",
        "leaf_count",
        "precision_digits",
        "context_digest",
    ] {
        assert_eq!(
            replayed["result"][field], accepted["result"][field],
            "{field}"
        );
    }
    let differentiated = receiver.send(differentiate("received"));
    assert_eq!(
        differentiated["result"]["solution_jacobian"],
        json!([[{"lower":"1","upper":"1"}]])
    );
    let reexported = receiver.send(CertificationRequest::ExportParametricFamily {
        schema_version: 1,
        family: family("received"),
    });
    assert_eq!(reexported["result"]["artifact"], artifact);
    receiver.finish();
}
