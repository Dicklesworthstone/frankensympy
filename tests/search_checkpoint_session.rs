//! Portable candidate continuations never bypass complete-cover acceptance.

use frankensympy::{
    ArtifactKind, ArtifactSource, ArtifactTransferId, BudgetLimits, CertificationRequest,
    CertificationResponse, ExactInterval, ResidualSystemId, RootIsolationId, RootIsolationSearchId,
    Session,
};
use fsym_budget::Dimension;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:roots").unwrap()
}
fn search(name: &str) -> RootIsolationSearchId {
    RootIsolationSearchId::new(format!("search:{name}")).unwrap()
}
fn isolation(name: &str) -> RootIsolationId {
    RootIsolationId::new(format!("isolation:{name}")).unwrap()
}
fn domain() -> Vec<ExactInterval> {
    vec![ExactInterval {
        lower: "-2".into(),
        upper: "2".into(),
    }]
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

fn begin(name: &str) -> CertificationRequest {
    CertificationRequest::BeginRootIsolationSearch {
        schema_version: 1,
        system: system(),
        search: search(name),
        domain: domain(),
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn advance(name: &str, cells: usize) -> CertificationRequest {
    CertificationRequest::AdvanceRootIsolationSearch {
        schema_version: 1,
        search: search(name),
        max_cells: cells,
    }
}

fn finish(name: &str) -> CertificationRequest {
    CertificationRequest::FinishRootIsolationSearch {
        schema_version: 1,
        search: search(name),
        isolation: isolation(name),
    }
}

fn export(session: &mut Session, name: &str) -> String {
    let CertificationResponse::RootIsolationSearchExported {
        artifact, evidence, ..
    } = session
        .certification_request(CertificationRequest::ExportRootIsolationSearch {
            schema_version: 1,
            search: search(name),
        })
        .unwrap()
    else {
        panic!("checkpoint export");
    };
    assert_eq!(evidence, "heuristic_candidate");
    artifact
}

fn restore(name: &str, artifact: &str) -> CertificationRequest {
    CertificationRequest::RestoreRootIsolationSearch {
        schema_version: 1,
        system: system(),
        search: search(name),
        domain: domain(),
        precision_digits: 16,
        max_leaves: 512,
        artifact: artifact.into(),
    }
}

fn paused(source: &str) -> Session {
    let mut session = registered(source);
    session.certification_request(begin("paused")).unwrap();
    session.certification_request(advance("paused", 2)).unwrap();
    session
}

#[test]
fn resumed_work_matches_uninterrupted_cover_and_remains_candidate_until_finish() {
    let mut sender = paused("x^2-2");
    let saved = sender
        .root_isolation_search_progress(&search("paused"))
        .unwrap();
    assert!(saved.processed_cells > 0 && saved.pending_cells > 0);
    let artifact = export(&mut sender, "paused");
    let mut receiver = registered("x^2-2");
    let protected = receiver.budget_snapshot().verifier_pool;
    let CertificationResponse::RootIsolationSearchRestored {
        evidence, progress, ..
    } = receiver
        .certification_request(restore("copy", &artifact))
        .unwrap()
    else {
        panic!("restored candidate");
    };
    assert_eq!(evidence, "heuristic_candidate");
    assert_eq!(progress, saved);
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    assert!(receiver.root_isolation(&isolation("copy")).is_none());
    assert_eq!(
        receiver
            .certification_request(finish("copy"))
            .unwrap_err()
            .code(),
        "search_not_ready"
    );
    receiver
        .certification_request(advance("copy", 2047))
        .unwrap();
    receiver.certification_request(finish("copy")).unwrap();
    sender
        .certification_request(advance("paused", 2047))
        .unwrap();
    sender.certification_request(finish("paused")).unwrap();
    assert_eq!(
        sender.root_isolation(&isolation("paused")),
        receiver.root_isolation(&isolation("copy"))
    );
    assert_eq!(
        receiver
            .root_isolation(&isolation("copy"))
            .unwrap()
            .root_count(),
        2
    );
    assert!(receiver.budget_snapshot().verifier_pool < protected);
}

#[test]
fn frozen_options_domain_and_original_source_are_receiver_requirements() {
    let mut sender = paused("x^2-2");
    let artifact = export(&mut sender, "paused");
    for field in ["precision_digits", "max_leaves", "domain"] {
        let mut request = serde_json::to_value(restore("mismatch", &artifact)).unwrap();
        request["payload"][field] = match field {
            "precision_digits" => json!(24),
            "max_leaves" => json!(256),
            _ => json!([{"lower":"-3","upper":"3"}]),
        };
        let mut receiver = registered("x^2-2");
        assert!(
            receiver
                .certification_request(serde_json::from_value(request).unwrap())
                .is_err()
        );
        assert!(
            receiver
                .root_isolation_search_progress(&search("mismatch"))
                .is_none()
        );
    }
    let mut receiver = registered("x^2-3");
    assert_eq!(
        receiver
            .certification_request(restore("mismatch", &artifact))
            .unwrap_err()
            .code(),
        "source_mismatch"
    );
}

#[test]
fn altered_counters_or_incomplete_transfers_cannot_publish_candidate_handles() {
    let mut sender = paused("x^2-2");
    let artifact = export(&mut sender, "paused");
    for field in ["steps", "allocation_bytes", "source_bytes", "evaluations"] {
        let parsed: Value = serde_json::from_str(&artifact).unwrap();
        let original = format!("\"{field}\":{}", parsed["counters"][field]);
        assert_eq!(artifact.matches(&original).count(), 1);
        // Change only the counter: preserve source/options key order so the
        // request reaches intrinsic history validation, not canonical refusal.
        let forged = artifact.replacen(&original, &format!("\"{field}\":0"), 1);
        let mut receiver = registered("x^2-2");
        assert_eq!(
            receiver
                .certification_request(restore("forged", &forged))
                .unwrap_err()
                .code(),
            "invalid_artifact"
        );
        assert!(
            receiver
                .root_isolation_search_progress(&search("forged"))
                .is_none()
        );
    }
    let mut receiver = registered("x^2-2");
    let mut truncated = artifact.clone();
    truncated.pop();
    assert!(
        receiver
            .certification_request(restore("retry", &truncated))
            .is_err()
    );
    receiver
        .certification_request(restore("retry", &artifact))
        .unwrap();
    assert_eq!(
        receiver
            .certification_request(restore("retry", &artifact))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn restore_and_release_never_refund_the_receiving_budget() {
    let mut sender = paused("x^2-2");
    let artifact = export(&mut sender, "paused");
    let mut receiver = registered("x^2-2");
    let before = receiver.budget_snapshot();
    receiver
        .certification_request(restore("copy", &artifact))
        .unwrap();
    let restored = receiver.budget_snapshot();
    assert!(
        restored.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    assert!(
        restored.dimensions[Dimension::MemoryBytes.index()]
            < before.dimensions[Dimension::MemoryBytes.index()]
    );
    receiver
        .certification_request(CertificationRequest::ReleaseRootIsolationSearch {
            schema_version: 1,
            search: search("copy"),
        })
        .unwrap();
    receiver
        .certification_request(restore("copy", &artifact))
        .unwrap();
    assert!(
        receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
            < restored.dimensions[Dimension::ComputeSteps.index()]
    );
    assert_eq!(
        receiver.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
}

#[test]
fn output_refusal_after_prefix_validation_is_atomic() {
    let mut sender = paused("x^2-2");
    let artifact = export(&mut sender, "paused");
    let mut measured = registered("x^2-2");
    measured
        .certification_request(restore("copy", &artifact))
        .unwrap();
    let after = measured.budget_snapshot();
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] -=
        after.dimensions[Dimension::ComputeSteps.index()] + 1;
    let mut receiver = Session::with_budget_limits(limited, 256);
    receiver
        .certification_request(registration("x^2-2"))
        .unwrap();
    assert_eq!(
        receiver
            .certification_request(restore("copy", &artifact))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(
        receiver
            .root_isolation_search_progress(&search("copy"))
            .is_none()
    );
    assert!(receiver.root_isolation(&isolation("copy")).is_none());
}

#[test]
fn completed_root_free_checkpoint_still_requires_independent_finish() {
    let mut sender = registered("x^2+1");
    sender.certification_request(begin("empty")).unwrap();
    sender
        .certification_request(advance("empty", 2047))
        .unwrap();
    let artifact = export(&mut sender, "empty");
    let mut receiver = registered("x^2+1");
    receiver
        .certification_request(restore("empty", &artifact))
        .unwrap();
    assert!(
        receiver
            .root_isolation_search_progress(&search("empty"))
            .unwrap()
            .ready_for_verification
    );
    assert!(receiver.root_isolation(&isolation("empty")).is_none());
    receiver.certification_request(finish("empty")).unwrap();
    assert!(
        receiver
            .root_isolation(&isolation("empty"))
            .unwrap()
            .is_root_free()
    );
}

#[test]
fn restore_needs_no_verifier_allowance_but_finish_cannot_borrow_ordinary_work() {
    let mut sender = registered("x^2-2");
    sender.certification_request(begin("complete")).unwrap();
    sender
        .certification_request(advance("complete", 2047))
        .unwrap();
    let artifact = export(&mut sender, "complete");
    let mut allowance = limits();
    allowance.verifier_pool = 0;
    let mut receiver = Session::with_budget_limits(allowance, 256);
    receiver
        .certification_request(registration("x^2-2"))
        .unwrap();
    receiver
        .certification_request(restore("copy", &artifact))
        .unwrap();
    assert!(
        receiver
            .root_isolation_search_progress(&search("copy"))
            .unwrap()
            .ready_for_verification
    );
    assert_eq!(
        receiver
            .certification_request(finish("copy"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(receiver.root_isolation(&isolation("copy")).is_none());
    assert!(
        receiver
            .root_certificate(
                &frankensympy::RootCertificateId::new("certificate:copy.root-0").unwrap()
            )
            .is_none()
    );
    assert!(receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
}

#[test]
fn staged_search_checkpoint_has_a_separate_kind_and_no_acceptance_on_import() {
    let mut sender = paused("x^2-2");
    let transfer = ArtifactTransferId::new("transfer:search").unwrap();
    let CertificationResponse::ArtifactExportStarted { manifest, .. } = sender
        .certification_request(CertificationRequest::BeginArtifactExport {
            schema_version: 1,
            transfer: transfer.clone(),
            source: ArtifactSource::RootIsolationSearch(search("paused")),
        })
        .unwrap()
    else {
        panic!("checkpoint manifest");
    };
    assert_eq!(manifest.kind, ArtifactKind::RootSearchCheckpoint);
    let mut receiver = registered("x^2-2");
    receiver
        .certification_request(CertificationRequest::BeginArtifactImport {
            schema_version: 1,
            transfer: transfer.clone(),
            manifest: manifest.clone(),
        })
        .unwrap();
    let request = CertificationRequest::RestoreRootIsolationSearchFromArtifact {
        schema_version: 1,
        system: system(),
        search: search("copy"),
        domain: domain(),
        precision_digits: 16,
        max_leaves: 512,
        transfer: transfer.clone(),
    };
    assert_eq!(
        receiver
            .certification_request(request.clone())
            .unwrap_err()
            .code(),
        "incomplete_transfer"
    );
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
    assert!(
        receiver
            .root_isolation_search_progress(&search("copy"))
            .is_none()
    );
    receiver.certification_request(request).unwrap();
    assert!(receiver.root_isolation(&isolation("copy")).is_none());
    assert_eq!(
        receiver
            .certification_request(CertificationRequest::ReplayRootIsolationFromArtifact {
                schema_version: 1,
                system: system(),
                isolation: isolation("invalid"),
                domain: domain(),
                transfer,
            })
            .unwrap_err()
            .code(),
        "artifact_kind_mismatch"
    );
}

struct Cli {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl Cli {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        Self {
            input: child.stdin.take().unwrap(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        }
    }
    fn request(&mut self, request: CertificationRequest) -> Value {
        writeln!(self.input, "{}", serde_json::to_string(&request).unwrap()).unwrap();
        self.input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["status"], "success", "{response}");
        response
    }
    fn close(mut self) {
        drop(self.input);
        assert!(self.child.wait().unwrap().success());
    }
}

#[test]
fn saved_checkpoint_resumes_after_sender_process_exits_using_default_cli_limits() {
    let mut sender = Cli::start();
    sender.request(registration("x^2-2"));
    sender.request(begin("paused"));
    let before = sender.request(advance("paused", 2));
    let exported = sender.request(CertificationRequest::ExportRootIsolationSearch {
        schema_version: 1,
        search: search("paused"),
    });
    let path = std::env::temp_dir().join(format!(
        "frankensympy-root-search-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, exported["result"]["artifact"].as_str().unwrap()).unwrap();
    sender.close();
    let saved = std::fs::read_to_string(&path).unwrap();
    let mut receiver = Cli::start();
    receiver.request(registration("x^2-2"));
    let restored = receiver.request(restore("copy", &saved));
    assert_eq!(restored["result"]["evidence"], "heuristic_candidate");
    assert_eq!(restored["result"]["progress"], before["result"]["progress"]);
    receiver.request(advance("copy", 2047));
    let result = receiver.request(finish("copy"));
    assert_eq!(result["result"]["root_count"], 2);
    assert_eq!(result["result"]["evidence"], "certificate_verified");
    receiver.close();
}
