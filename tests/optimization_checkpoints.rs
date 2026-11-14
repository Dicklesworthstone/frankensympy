//! Portable optimization progress stays a candidate across native and CLI processes.

use frankensympy::{
    ArtifactKind, ArtifactSource, ArtifactTransferId, BigRational, BoxExtremumKind,
    BoxOptimizationId, BoxOptimizationOptions, BoxOptimizationSearch, BoxOptimizationSearchId,
    BoxOptimizationSearchStatus, BudgetLimits, CertificationError, CertificationRequest,
    CertificationResponse, ExactInterval, ObjectiveId, ObjectiveProblem, OptimizationError,
    RealBall, Session, export_box_optimization_checkpoint, restore_box_optimization_checkpoint,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Dimension, Unbounded};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn domain() -> Vec<ExactInterval> {
    vec![ExactInterval { lower: "-2".into(), upper: "2".into() }]
}

fn native_domain() -> Vec<RealBall> {
    vec![RealBall::new(q(0, 1), q(2, 1)).unwrap()]
}

fn options() -> BoxOptimizationOptions {
    BoxOptimizationOptions { precision_digits: 16, max_leaves: 256 }
}

fn problem(source: &str) -> ObjectiveProblem {
    ObjectiveProblem::parse_source(source, &["x".into()], &mut Unbounded).unwrap()
}

fn native_search(source: &str) -> BoxOptimizationSearch {
    problem(source)
        .begin_box_optimization_search(
            &native_domain(), BoxExtremumKind::Minimum, &q(1, 1000),
            options(), &mut Unbounded,
        )
        .unwrap()
}

fn restore_native(bytes: &[u8]) -> Result<BoxOptimizationSearch, OptimizationError> {
    restore_box_optimization_checkpoint(
        bytes, &problem("(x^2-2)^2"), &native_domain(),
        BoxExtremumKind::Minimum, &q(1, 1000), options(), [3; 32], &mut Unbounded,
    )
}

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits.dimensions[Dimension::DepthLimit.index()] = 256;
    limits
}

fn objective() -> ObjectiveId {
    ObjectiveId::new("objective:well").unwrap()
}

fn search(name: &str) -> BoxOptimizationSearchId {
    BoxOptimizationSearchId::new(format!("optimization-search:{name}")).unwrap()
}

fn optimization(name: &str) -> BoxOptimizationId {
    BoxOptimizationId::new(format!("optimization:{name}")).unwrap()
}

fn registration(source: &str) -> CertificationRequest {
    CertificationRequest::RegisterObjective {
        schema_version: 1, objective: objective(), source: source.into(),
        variables: vec!["x".into()],
    }
}

fn registered(source: &str) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    session.certification_request(registration(source)).unwrap();
    session
}

fn begin(name: &str) -> CertificationRequest {
    CertificationRequest::BeginBoxOptimization {
        schema_version: 1, objective: objective(), search: search(name), domain: domain(),
        extremum: BoxExtremumKind::Minimum, absolute_tolerance: "1/1000".into(),
        precision_digits: 16, max_leaves: 256,
    }
}

fn advance(name: &str, splits: usize) -> CertificationRequest {
    CertificationRequest::AdvanceBoxOptimization {
        schema_version: 1, search: search(name), max_splits: splits,
    }
}

fn finish(name: &str) -> CertificationRequest {
    CertificationRequest::FinishBoxOptimization {
        schema_version: 1, search: search(name), optimization: optimization(name),
    }
}

fn export(session: &mut Session, name: &str) -> String {
    let CertificationResponse::BoxOptimizationSearchExported { artifact, evidence, .. } =
        session.certification_request(CertificationRequest::ExportBoxOptimizationSearch {
            schema_version: 1, search: search(name),
        }).unwrap()
    else { panic!("candidate checkpoint"); };
    assert_eq!(evidence, "heuristic_candidate");
    artifact
}

fn restore(name: &str, artifact: &str) -> CertificationRequest {
    CertificationRequest::RestoreBoxOptimizationSearch {
        schema_version: 1, objective: objective(), search: search(name), domain: domain(),
        extremum: BoxExtremumKind::Minimum, absolute_tolerance: "1/1000".into(),
        precision_digits: 16, max_leaves: 256, artifact: artifact.into(),
    }
}

fn paused() -> Session {
    let mut session = registered("(x^2-2)^2");
    session.certification_request(begin("paused")).unwrap();
    session.certification_request(advance("paused", 1)).unwrap();
    session
}

fn complete(session: &mut Session, name: &str) {
    for _ in 0..256 {
        let progress = session.box_optimization_search_progress(&search(name)).unwrap();
        if progress.status == BoxOptimizationSearchStatus::Complete {
            return;
        }
        session.certification_request(advance(name, 3)).unwrap();
    }
    panic!("bounded search did not reach a complete candidate");
}

#[test]
fn canonical_native_checkpoint_recomputes_a_source_bound_frontier_and_final_witness() {
    let mut original = native_search("(x^2-2)^2");
    original.advance(1, &mut Unbounded).unwrap();
    let bytes = export_box_optimization_checkpoint(&mut original, [3; 32], &mut Unbounded).unwrap();
    let wire: Value = serde_json::from_slice(&bytes).unwrap();
    for absent in ["accepted", "verified", "value", "gap", "bounds", "priority_queue"] {
        assert!(wire.get(absent).is_none(), "{absent}");
    }
    let mut restored = restore_native(&bytes).unwrap();
    assert_eq!(restored.progress().leaf_count, original.progress().leaf_count);
    assert_eq!(restored.progress().split_count, original.progress().split_count);
    assert!(restored.counters().steps > original.counters().steps);
    original.advance(1024, &mut Unbounded).unwrap();
    restored.advance(1024, &mut Unbounded).unwrap();
    assert_eq!(restored.completed_witness(), original.completed_witness());
    let accepted = problem("(x^2-2)^2").verify_box_optimization(
        &native_domain(), BoxExtremumKind::Minimum, &q(1, 1000),
        restored.completed_witness().unwrap(), &mut Unbounded,
    ).unwrap();
    assert!(accepted.value().lower() <= q(0, 1));
    assert!(accepted.value().upper() >= q(0, 1));
    assert!(accepted.gap() <= &q(1, 1000));
}

#[test]
fn every_immutable_statement_component_is_bound_before_imported_arithmetic() {
    let mut original = native_search("(x^2-2)^2");
    let bytes = export_box_optimization_checkpoint(&mut original, [3; 32], &mut Unbounded).unwrap();
    let baseline = problem("(x^2-2)^2");
    assert!(matches!(
        restore_box_optimization_checkpoint(
            &bytes, &problem("(x^2-2)^2+1"), &native_domain(),
            BoxExtremumKind::Minimum, &q(1, 1000), options(), [3; 32], &mut Unbounded,
        ),
        Err(OptimizationError::Certification(CertificationError::SourceMismatch))
    ));
    let different = vec![RealBall::new(q(0, 1), q(3, 1)).unwrap()];
    assert!(matches!(
        restore_box_optimization_checkpoint(
            &bytes, &baseline, &different, BoxExtremumKind::Minimum,
            &q(1, 1000), options(), [3; 32], &mut Unbounded,
        ),
        Err(OptimizationError::Certification(CertificationError::DomainMismatch))
    ));
    for (kind, tolerance, changed_options) in [
        (BoxExtremumKind::Maximum, q(1, 1000), options()),
        (BoxExtremumKind::Minimum, q(1, 100), options()),
        (BoxExtremumKind::Minimum, q(1, 1000),
            BoxOptimizationOptions { precision_digits: 24, ..options() }),
        (BoxExtremumKind::Minimum, q(1, 1000),
            BoxOptimizationOptions { max_leaves: 128, ..options() }),
    ] {
        assert!(matches!(
            restore_box_optimization_checkpoint(
                &bytes, &baseline, &native_domain(), kind, &tolerance,
                changed_options, [3; 32], &mut Unbounded,
            ),
            Err(OptimizationError::StatementMismatch)
        ));
    }
    assert!(matches!(
        restore_box_optimization_checkpoint(
            &bytes, &baseline, &native_domain(), BoxExtremumKind::Minimum,
            &q(1, 1000), options(), [4; 32], &mut Unbounded,
        ),
        Err(OptimizationError::Certification(CertificationError::UniverseMismatch { .. }))
    ));

    // Corrupt arithmetic cannot mask a statement mismatch. The receiver never
    // deserializes the imported point when its original objective differs.
    let mut corrupted: Value = serde_json::from_slice(&bytes).unwrap();
    corrupted["incumbent_point"] = json!("not a rational coordinate list");
    assert!(matches!(
        restore_box_optimization_checkpoint(
            &serde_json::to_vec(&corrupted).unwrap(), &problem("x"), &native_domain(),
            BoxExtremumKind::Minimum, &q(1, 1000), options(), [3; 32], &mut Unbounded,
        ),
        Err(OptimizationError::Certification(CertificationError::SourceMismatch))
    ));
}

#[test]
fn malformed_noncanonical_and_oversized_artifacts_never_restore_a_search() {
    let mut original = native_search("(x^2-2)^2");
    let bytes = export_box_optimization_checkpoint(&mut original, [3; 32], &mut Unbounded).unwrap();
    let mut trailing = bytes.clone();
    trailing.push(b' ');
    assert!(matches!(
        restore_native(&trailing),
        Err(OptimizationError::Certification(CertificationError::NonCanonical))
    ));
    for (field, value) in [
        ("schema_version", json!(2)),
        ("accepted", json!(true)),
        ("partition", json!(null)),
        ("split_order", json!("wrong kind")),
        ("incumbent_point", json!(["malformed scalar"])),
        ("counters", json!({"steps":0})),
    ] {
        let mut malformed: Value = serde_json::from_slice(&bytes).unwrap();
        malformed[field] = value;
        assert!(restore_native(&serde_json::to_vec(&malformed).unwrap()).is_err(), "{field}");
    }
    assert!(restore_native(b"{}").is_err());
    let oversized = vec![b' '; frankensympy::MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES + 1];
    assert!(matches!(
        restore_native(&oversized),
        Err(OptimizationError::Certification(CertificationError::ByteLimit { .. }))
    ));
}

#[test]
fn session_resume_preserves_candidate_status_budgets_and_the_accepted_result() {
    let mut sender = paused();
    let saved = sender.box_optimization_search_progress(&search("paused")).unwrap();
    assert_eq!(saved.status, BoxOptimizationSearchStatus::Searching);
    let artifact = export(&mut sender, "paused");
    let spent = sender.box_optimization_search_counters(&search("paused")).unwrap();
    let mut receiver = registered("(x^2-2)^2");
    let protected = receiver.budget_snapshot().verifier_pool;
    let CertificationResponse::BoxOptimizationSearchRestored {
        evidence, progress, counters, ..
    } = receiver.certification_request(restore("resumed", &artifact)).unwrap()
    else { panic!("restored progress"); };
    assert_eq!(evidence, "heuristic_candidate");
    assert_eq!(progress.leaf_count, saved.leaf_count);
    assert_eq!(progress.split_count, saved.split_count);
    assert!(counters.steps > spent.steps);
    assert!(counters.allocation_bytes > spent.allocation_bytes);
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    assert!(receiver.box_optimization(&optimization("resumed")).is_none());
    assert_eq!(
        receiver.certification_request(finish("resumed")).unwrap_err().code(),
        "search_not_ready"
    );
    complete(&mut sender, "paused");
    complete(&mut receiver, "resumed");
    sender.certification_request(finish("paused")).unwrap();
    receiver.certification_request(finish("resumed")).unwrap();
    assert_eq!(
        sender.box_optimization(&optimization("paused")),
        receiver.box_optimization(&optimization("resumed"))
    );
    assert!(receiver.budget_snapshot().verifier_pool < protected);
}

#[test]
fn restoration_rejects_changed_context_and_failed_import_leaves_the_label_available() {
    let mut sender = paused();
    let artifact = export(&mut sender, "paused");
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(HashMap::new(), HashMap::new(), "checkpoint-other-universe").unwrap();
    let mut receiver = Session::with_context_and_budget_limits(context, limits(), 256);
    receiver.certification_request(registration("(x^2-2)^2")).unwrap();
    assert_eq!(
        receiver.certification_request(restore("copy", &artifact)).unwrap_err().code(),
        "universe_mismatch"
    );
    assert!(receiver.box_optimization_search_progress(&search("copy")).is_none());

    let mut receiver = registered("(x^2-2)^2");
    assert!(receiver.certification_request(restore("copy", "{}")).is_err());
    assert!(receiver.box_optimization_search_progress(&search("copy")).is_none());
    receiver.certification_request(restore("copy", &artifact)).unwrap();
    assert_eq!(
        receiver.certification_request(restore("copy", &artifact)).unwrap_err().code(),
        "duplicate_object"
    );
    let before_release = receiver.budget_snapshot();
    receiver.certification_request(CertificationRequest::ReleaseBoxOptimizationSearch {
        schema_version: 1, search: search("copy"),
    }).unwrap();
    assert!(receiver.box_optimization_search_progress(&search("copy")).is_none());
    assert!(receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
        <= before_release.dimensions[Dimension::ComputeSteps.index()]);
}

#[test]
fn a_complete_restored_candidate_still_requires_protected_independent_verification() {
    let mut sender = paused();
    complete(&mut sender, "paused");
    let artifact = export(&mut sender, "paused");
    let mut allowance = limits();
    allowance.verifier_pool = 0;
    let mut receiver = Session::with_budget_limits(allowance, 256);
    receiver.certification_request(registration("(x^2-2)^2")).unwrap();
    receiver.certification_request(restore("complete", &artifact)).unwrap();
    assert_eq!(
        receiver.box_optimization_search_progress(&search("complete")).unwrap().status,
        BoxOptimizationSearchStatus::Complete
    );
    assert_eq!(
        receiver.certification_request(finish("complete")).unwrap_err().code(),
        "budget_exhausted"
    );
    assert!(receiver.box_optimization(&optimization("complete")).is_none());
}

#[test]
fn chunked_transfer_preserves_kind_and_waits_for_complete_import_before_restore() {
    let mut sender = paused();
    let transfer = ArtifactTransferId::new("transfer:optimization-checkpoint").unwrap();
    let CertificationResponse::ArtifactExportStarted { manifest, .. } =
        sender.certification_request(CertificationRequest::BeginArtifactExport {
            schema_version: 1, transfer: transfer.clone(),
            source: ArtifactSource::BoxOptimizationSearch(search("paused")),
        }).unwrap()
    else { panic!("checkpoint manifest"); };
    assert_eq!(manifest.kind, ArtifactKind::BoxOptimizationCheckpoint);
    let mut receiver = registered("(x^2-2)^2");
    receiver.certification_request(CertificationRequest::BeginArtifactImport {
        schema_version: 1, transfer: transfer.clone(), manifest: manifest.clone(),
    }).unwrap();
    let request = CertificationRequest::RestoreBoxOptimizationSearchFromArtifact {
        schema_version: 1, objective: objective(), search: search("copy"), domain: domain(),
        extremum: BoxExtremumKind::Minimum, absolute_tolerance: "1/1000".into(),
        precision_digits: 16, max_leaves: 256, transfer: transfer.clone(),
    };
    assert_eq!(
        receiver.certification_request(request.clone()).unwrap_err().code(),
        "incomplete_transfer"
    );
    for index in 0..manifest.chunk_count() {
        let CertificationResponse::ArtifactChunkRead { chunk, .. } =
            sender.certification_request(CertificationRequest::ReadArtifactChunk {
                schema_version: 1, transfer: transfer.clone(), index,
            }).unwrap()
        else { panic!("checkpoint chunk"); };
        receiver.certification_request(CertificationRequest::AppendArtifactChunk {
            schema_version: 1, transfer: transfer.clone(), chunk,
        }).unwrap();
    }
    assert!(receiver.box_optimization_search_progress(&search("copy")).is_none());
    receiver.certification_request(request).unwrap();
    assert!(receiver.box_optimization(&optimization("copy")).is_none());
    assert_eq!(
        receiver.certification_request(CertificationRequest::ReplayBoxOptimizationFromArtifact {
            schema_version: 1, objective: objective(), optimization: optimization("wrong"),
            domain: domain(), extremum: BoxExtremumKind::Minimum,
            absolute_tolerance: "1/1000".into(), transfer,
        }).unwrap_err().code(),
        "artifact_kind_mismatch"
    );
}

#[test]
fn ndjson_checkpoint_dispatch_rejects_unknown_fields_versions_and_duplicate_keys() {
    let mut session = paused();
    let request = CertificationRequest::ExportBoxOptimizationSearch {
        schema_version: 1, search: search("paused"),
    };
    let reply: Value = serde_json::from_str(
        &session.handle_envelope(&serde_json::to_string(&request).unwrap())
    ).unwrap();
    assert_eq!(reply["status"], "success");
    assert_eq!(reply["result"]["kind"], "box_optimization_search_exported");
    let artifact = reply["result"]["artifact"].as_str().unwrap();
    let mut receiver = registered("(x^2-2)^2");
    let request = serde_json::to_string(&restore("wire", artifact)).unwrap();
    for bad in [
        request.replace("\"schema_version\":1", "\"schema_version\":2"),
        request.replacen("\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1", 1),
        request.replacen("\"schema_version\":1", "\"unknown\":true,\"schema_version\":1", 1),
    ] {
        let reply: Value = serde_json::from_str(&receiver.handle_envelope(&bad)).unwrap();
        assert_eq!(reply["status"], "error", "{reply}");
        assert!(receiver.box_optimization_search_progress(&search("wire")).is_none());
    }
    let reply: Value = serde_json::from_str(&receiver.handle_envelope(&request)).unwrap();
    assert_eq!(reply["status"], "success");
    assert_eq!(reply["result"]["evidence"], "heuristic_candidate");
}

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args([
                "--compute-steps", "100000000",
                "--memory-bytes", "1073741824",
                "--allocations", "4000000",
            ])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().unwrap();
        Self {
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        }
    }

    fn send(&mut self, request: CertificationRequest) -> Value {
        let request = serde_json::to_string(&request).unwrap();
        assert!(request.len() < frankensympy::MAX_ENVELOPE_BYTES);
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
        let reply: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(reply["status"], "success", "{reply}");
        reply
    }

    fn close(&mut self) {
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
fn candidate_can_resume_after_its_originating_cli_process_exits() {
    let mut sender = Cli::new();
    sender.send(registration("(x^2-2)^2"));
    sender.send(begin("original"));
    let progress = sender.send(advance("original", 1));
    let exported = sender.send(CertificationRequest::ExportBoxOptimizationSearch {
        schema_version: 1, search: search("original"),
    });
    let saved = exported["result"]["artifact"].as_str().unwrap().to_owned();
    sender.close();

    let mut receiver = Cli::new();
    receiver.send(registration("(x^2-2)^2"));
    let restored = receiver.send(restore("continued", &saved));
    assert_eq!(restored["result"]["evidence"], "heuristic_candidate");
    assert_eq!(
        restored["result"]["progress"]["split_count"],
        progress["result"]["progress"]["split_count"]
    );
    let mut ready = restored["result"]["progress"]["status"] == "complete";
    for _ in 0..256 {
        if ready { break; }
        let advanced = receiver.send(advance("continued", 3));
        ready = advanced["result"]["progress"]["status"] == "complete";
    }
    assert!(ready);
    let checked = receiver.send(finish("continued"));
    assert_eq!(checked["result"]["evidence"], "certified_numeric");
    assert_eq!(checked["result"]["claim"], "global_box_optimum_with_absolute_value_gap_bound");
    receiver.close();
}
