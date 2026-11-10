//! Chunked proof exchange retains the source-bound mathematical trust boundary.

use frankensympy::artifact_stream::{
    ARTIFACT_CHUNK_BYTES, ArtifactChunk, ArtifactExport, ArtifactKind, ArtifactManifest,
    ArtifactProgress, ArtifactTransferId, artifact_chunk_digest,
};
use frankensympy::certification_session::{
    ArtifactSource, MAX_SESSION_ARTIFACT_TRANSFERS, MAX_SESSION_ROOT_ARTIFACT_BYTES,
};
use frankensympy::session::MAX_ENVELOPE_BYTES;
use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, ExactInterval, ExactRootWitness,
    ResidualSystemId, RootCertificateId, RootIsolationId, Session,
};
use fsym_budget::{Budget, Dimension};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(200_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits.dimensions[Dimension::DepthLimit.index()] = 256;
    limits
}

fn transfer(name: &str) -> ArtifactTransferId {
    ArtifactTransferId::new(format!("transfer:{name}")).unwrap()
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:stream").unwrap()
}

fn isolation(name: &str) -> RootIsolationId {
    RootIsolationId::new(format!("isolation:{name}")).unwrap()
}

fn certificate(name: &str) -> RootCertificateId {
    RootCertificateId::new(format!("certificate:{name}")).unwrap()
}

fn child(name: &str, index: usize) -> RootCertificateId {
    certificate(&format!("{name}.root-{index}"))
}

fn interval(lower: &str, upper: &str) -> ExactInterval {
    ExactInterval {
        lower: lower.into(),
        upper: upper.into(),
    }
}

fn domain() -> Vec<ExactInterval> {
    vec![interval("-2", "2")]
}

fn registration(source: &str) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec![source.into()],
        variables: vec!["x".into()],
    }
}

fn registered(source: &str, budget: BudgetLimits) -> Session {
    let mut session = Session::with_budget_limits(budget, 256);
    session.certification_request(registration(source)).unwrap();
    session
}

fn isolate(name: &str) -> CertificationRequest {
    CertificationRequest::IsolateRealRootsInBox {
        schema_version: 1,
        system: system(),
        isolation: isolation(name),
        domain: domain(),
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn begin_import(name: &str, manifest: &ArtifactManifest) -> CertificationRequest {
    CertificationRequest::BeginArtifactImport {
        schema_version: 1,
        transfer: transfer(name),
        manifest: manifest.clone(),
    }
}

fn append(name: &str, chunk: &ArtifactChunk) -> CertificationRequest {
    CertificationRequest::AppendArtifactChunk {
        schema_version: 1,
        transfer: transfer(name),
        chunk: chunk.clone(),
    }
}

fn replay(name: &str, input: &str) -> CertificationRequest {
    CertificationRequest::ReplayRootIsolationFromArtifact {
        schema_version: 1,
        system: system(),
        isolation: isolation(name),
        domain: domain(),
        transfer: transfer(input),
    }
}

struct Packed {
    manifest: ArtifactManifest,
    chunks: Vec<ArtifactChunk>,
}

fn pack(kind: ArtifactKind, bytes: Vec<u8>) -> Packed {
    let mut meter = Budget::new(limits());
    let export = ArtifactExport::new(kind, bytes, &mut meter).unwrap();
    Packed {
        manifest: export.manifest().clone(),
        chunks: (0..export.manifest().chunk_count())
            .map(|index| export.chunk(index, &mut meter).unwrap())
            .collect(),
    }
}

fn stage(session: &mut Session, name: &str, packed: &Packed) {
    let protected = session.budget_snapshot().verifier_pool;
    session
        .certification_request(begin_import(name, &packed.manifest))
        .unwrap();
    for chunk in &packed.chunks {
        session.certification_request(append(name, chunk)).unwrap();
    }
    assert_eq!(session.budget_snapshot().verifier_pool, protected);
}

fn proof(source: &str) -> Vec<u8> {
    let mut sender = registered(source, limits());
    sender.certification_request(isolate("sender")).unwrap();
    let result = sender
        .certification_request(CertificationRequest::ExportRootIsolation {
            schema_version: 1,
            isolation: isolation("sender"),
        })
        .unwrap();
    let CertificationResponse::RootIsolationExported { artifact, .. } = result else {
        panic!("complete proof export");
    };
    artifact.into_bytes()
}

fn unpublished(session: &Session, name: &str) {
    assert!(session.root_isolation(&isolation(name)).is_none());
    assert!(session.root_certificate(&child(name, 0)).is_none());
    assert!(session.root_certificate(&child(name, 1)).is_none());
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    output
}

#[test]
fn staged_chunks_are_ordered_retryable_and_never_mathematical_evidence() {
    let bytes = vec![b'a'; ARTIFACT_CHUNK_BYTES + 3];
    let packed = pack(ArtifactKind::RootIsolation, bytes.clone());
    let mut receiver = registered("x", limits());
    receiver
        .certification_request(begin_import("input", &packed.manifest))
        .unwrap();
    let protected = receiver.budget_snapshot().verifier_pool;
    assert_eq!(
        receiver
            .certification_request(replay("missing", "input"))
            .unwrap_err()
            .code(),
        "incomplete_transfer",
    );
    assert_eq!(
        receiver
            .certification_request(append("input", &packed.chunks[1]))
            .unwrap_err()
            .code(),
        "out_of_order_chunk",
    );
    let first = receiver
        .certification_request(append("input", &packed.chunks[0]))
        .unwrap();
    let before_retry = receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let retried = receiver
        .certification_request(append("input", &packed.chunks[0]))
        .unwrap();
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::to_value(retried).unwrap()
    );
    assert!(receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] < before_retry);

    let changed = vec![b'b'; ARTIFACT_CHUNK_BYTES];
    let conflicting = ArtifactChunk {
        index: 0,
        digest: artifact_chunk_digest(
            packed.manifest.content_id,
            0,
            &changed,
            &mut Budget::new(limits()),
        )
        .unwrap(),
        data_hex: hex(&changed),
    };
    assert_eq!(
        receiver
            .certification_request(append("input", &conflicting))
            .unwrap_err()
            .code(),
        "conflicting_chunk",
    );
    receiver
        .certification_request(append("input", &packed.chunks[1]))
        .unwrap();
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    assert_eq!(
        receiver
            .certification_request(replay("invalid", "input"))
            .unwrap_err()
            .code(),
        "invalid_artifact",
    );
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    unpublished(&receiver, "invalid");

    let mut mismatched = registered("x", limits());
    mismatched
        .certification_request(begin_import("input", &packed.manifest))
        .unwrap();
    mismatched
        .certification_request(append("input", &conflicting))
        .unwrap();
    mismatched
        .certification_request(append("input", &packed.chunks[1]))
        .unwrap();
    let protected = mismatched.budget_snapshot().verifier_pool;
    assert_eq!(
        mismatched
            .certification_request(replay("digest", "input"))
            .unwrap_err()
            .code(),
        "invalid_artifact",
    );
    assert_eq!(mismatched.budget_snapshot().verifier_pool, protected);
    unpublished(&mismatched, "digest");
}

#[test]
fn digest_valid_transfers_still_require_expected_kind_sources_domain_and_mathematics() {
    let valid = proof("x^2-2");
    let packed = pack(ArtifactKind::RootIsolation, valid.clone());
    let mut receiver = registered("x^2-2", limits());
    stage(&mut receiver, "proof", &packed);
    let protected = receiver.budget_snapshot().verifier_pool;
    let wrong_kind = CertificationRequest::ReplayRootCertificateFromArtifact {
        schema_version: 1,
        system: system(),
        certificate: certificate("wrong-kind"),
        transfer: transfer("proof"),
    };
    assert_eq!(
        receiver
            .certification_request(wrong_kind)
            .unwrap_err()
            .code(),
        "artifact_kind_mismatch"
    );
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    let wrong_domain = CertificationRequest::ReplayRootIsolationFromArtifact {
        schema_version: 1,
        system: system(),
        isolation: isolation("received"),
        domain: vec![interval("-3", "3")],
        transfer: transfer("proof"),
    };
    assert_eq!(
        receiver
            .certification_request(wrong_domain)
            .unwrap_err()
            .code(),
        "domain_mismatch"
    );
    unpublished(&receiver, "received");
    receiver
        .certification_request(replay("received", "proof"))
        .unwrap();
    assert_eq!(
        receiver
            .root_isolation(&isolation("received"))
            .unwrap()
            .root_count(),
        2
    );

    let mut wrong_source = registered("x^2-3", limits());
    stage(&mut wrong_source, "proof", &packed);
    assert_eq!(
        wrong_source
            .certification_request(replay("changed", "proof"))
            .unwrap_err()
            .code(),
        "source_mismatch"
    );
    unpublished(&wrong_source, "changed");

    let original = String::from_utf8(proof("x^2+1")).unwrap();
    let source_field = |source| {
        format!(
            "\"residuals\":{},\"variables\":",
            serde_json::to_string(&[fsym_core::parse_raw(source).unwrap()]).unwrap(),
        )
    };
    let old = source_field("x^2+1");
    assert_eq!(original.matches(&old).count(), 1);
    let forged = original.replacen(&old, &source_field("x^2-2"), 1);
    // A fresh, correct content ID and every correct chunk digest accompany the
    // canonical but mathematically false zero-root cover.
    let forged = pack(ArtifactKind::RootIsolation, forged.into_bytes());
    let mut rejected = registered("x^2-2", limits());
    stage(&mut rejected, "forged", &forged);
    let protected = rejected.budget_snapshot().verifier_pool;
    assert_eq!(
        rejected
            .certification_request(replay("forged", "forged"))
            .unwrap_err()
            .code(),
        "inconclusive"
    );
    assert!(rejected.budget_snapshot().verifier_pool < protected);
    unpublished(&rejected, "forged");
}

#[test]
fn chunk_acknowledgement_budget_refusal_does_not_advance_the_prefix() {
    let packed = pack(
        ArtifactKind::RootIsolation,
        vec![b'a'; ARTIFACT_CHUNK_BYTES + 1],
    );
    let mut measured = registered("x", limits());
    measured
        .certification_request(begin_import("bounded", &packed.manifest))
        .unwrap();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - measured.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let acknowledgement = CertificationResponse::ArtifactChunkAppended {
        schema_version: 1,
        transfer: transfer("bounded"),
        progress: ArtifactProgress {
            received_bytes: ARTIFACT_CHUNK_BYTES,
            next_chunk: 1,
        },
    };
    let output_steps = serde_json::to_vec(&acknowledgement).unwrap().len() as u64;
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] = spent + output_steps - 1;
    let mut receiver = registered("x", limited);
    receiver
        .certification_request(begin_import("bounded", &packed.manifest))
        .unwrap();
    let protected = receiver.budget_snapshot().verifier_pool;
    assert_eq!(
        receiver
            .certification_request(append("bounded", &packed.chunks[0]))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    // This valid second chunk must still be out of order. The refused first
    // acknowledgement did not append any bytes despite the complete input.
    assert_eq!(
        receiver
            .certification_request(append("bounded", &packed.chunks[1]))
            .unwrap_err()
            .code(),
        "out_of_order_chunk"
    );
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
}

#[test]
fn replay_uses_one_cumulative_protected_pool_and_preserves_the_staged_bytes() {
    let packed = pack(ArtifactKind::RootIsolation, proof("x^2-2"));
    let mut measured = registered("x^2-2", limits());
    stage(&mut measured, "input", &packed);
    let protected = measured.budget_snapshot().verifier_pool;
    measured
        .certification_request(replay("first", "input"))
        .unwrap();
    let spent = protected - measured.budget_snapshot().verifier_pool;
    assert!(spent > 0);
    let mut limited = limits();
    limited.verifier_pool = spent;
    let mut receiver = registered("x^2-2", limited);
    stage(&mut receiver, "input", &packed);
    receiver
        .certification_request(replay("first", "input"))
        .unwrap();
    assert_eq!(receiver.budget_snapshot().verifier_pool, 0);
    assert_eq!(
        receiver
            .certification_request(replay("second", "input"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    unpublished(&receiver, "second");
    assert!(receiver.root_isolation(&isolation("first")).is_some());
    // Imports remain available after successful and refused mathematical
    // replay. Retrying an old chunk neither restores fuel nor adds bytes.
    receiver
        .certification_request(append("input", &packed.chunks[0]))
        .unwrap();
    assert_eq!(receiver.budget_snapshot().verifier_pool, 0);
    assert_eq!(
        receiver
            .certification_request(begin_import("input", &packed.manifest))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
}

#[test]
fn verified_replay_and_late_child_collision_publish_no_partial_root_set() {
    let packed = pack(ArtifactKind::RootIsolation, proof("x^2-2"));
    let mut measured = registered("x^2-2", limits());
    stage(&mut measured, "input", &packed);
    measured
        .certification_request(replay("publication", "input"))
        .unwrap();
    let after = measured.budget_snapshot();
    let spent = limits().dimensions[Dimension::ComputeSteps.index()]
        - after.dimensions[Dimension::ComputeSteps.index()];
    let mut limited = limits();
    limited.dimensions[Dimension::ComputeSteps.index()] = spent - 1;
    let mut receiver = registered("x^2-2", limited);
    stage(&mut receiver, "input", &packed);
    assert_eq!(
        receiver
            .certification_request(replay("publication", "input"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert_eq!(
        receiver.budget_snapshot().verifier_pool,
        after.verifier_pool
    );
    unpublished(&receiver, "publication");
    assert_eq!(
        receiver
            .certification_request(begin_import("input", &packed.manifest))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );

    let mut collision = registered("x^2-2", limits());
    stage(&mut collision, "input", &packed);
    collision
        .certification_request(CertificationRequest::CertifyRootBox {
            schema_version: 1,
            system: system(),
            certificate: child("blocked", 1),
            witness: ExactRootWitness {
                domain: vec![interval("1.4", "3/2")],
                center: vec!["29/20".into()],
                preconditioner: vec![vec!["1/3".into()]],
                precision_digits: 24,
            },
        })
        .unwrap();
    let old = collision
        .root_certificate(&child("blocked", 1))
        .unwrap()
        .clone();
    assert_eq!(
        collision
            .certification_request(replay("blocked", "input"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    assert!(collision.root_isolation(&isolation("blocked")).is_none());
    assert!(collision.root_certificate(&child("blocked", 0)).is_none());
    assert_eq!(collision.root_certificate(&child("blocked", 1)), Some(&old));
    collision
        .certification_request(replay("retry", "input"))
        .unwrap();
    assert_eq!(
        collision
            .root_isolation(&isolation("retry"))
            .unwrap()
            .root_count(),
        2
    );
}

#[test]
fn transfer_slots_release_explicitly_and_wire_inputs_remain_strict() {
    let packed = pack(ArtifactKind::RootIsolation, b"{}".to_vec());
    let mut receiver = registered("x", limits());
    for index in 0..MAX_SESSION_ARTIFACT_TRANSFERS {
        receiver
            .certification_request(begin_import(&format!("slot-{index}"), &packed.manifest))
            .unwrap();
    }
    assert_eq!(
        receiver
            .certification_request(begin_import("overflow", &packed.manifest))
            .unwrap_err()
            .code(),
        "resource_limit_exceeded"
    );
    let protected = receiver.budget_snapshot().verifier_pool;
    let before = receiver.budget_snapshot().dimensions[Dimension::MemoryBytes.index()];
    receiver
        .certification_request(CertificationRequest::ReleaseArtifactTransfer {
            schema_version: 1,
            transfer: transfer("slot-0"),
        })
        .unwrap();
    assert!(receiver.budget_snapshot().dimensions[Dimension::MemoryBytes.index()] < before);
    receiver
        .certification_request(begin_import("slot-0", &packed.manifest))
        .unwrap();
    assert_eq!(
        receiver
            .certification_request(CertificationRequest::ReadArtifactChunk {
                schema_version: 1,
                transfer: transfer("slot-0"),
                index: 0,
            })
            .unwrap_err()
            .code(),
        "wrong_transfer_mode"
    );
    assert_eq!(
        receiver
            .certification_request(CertificationRequest::ReleaseArtifactTransfer {
                schema_version: 1,
                transfer: transfer("missing"),
            })
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);

    let mut malformed = wire(begin_import("ambiguous", &packed.manifest));
    malformed["payload"]["manifest"]["accepted"] = json!(true);
    let response: Value =
        serde_json::from_str(&receiver.handle_envelope(&malformed.to_string())).unwrap();
    assert_eq!(response["code"], "malformed_request");
    let mut future = wire(begin_import("future", &packed.manifest));
    future["payload"]["schema_version"] = json!(2);
    let response: Value =
        serde_json::from_str(&receiver.handle_envelope(&future.to_string())).unwrap();
    assert_eq!(response["code"], "unsupported_schema");
    let duplicate = wire(append("slot-0", &packed.chunks[0]))
        .to_string()
        .replace("\"index\":0", "\"index\":0,\"index\":0");
    assert!(duplicate.contains("\"index\":0,\"index\":0"));
    let response: Value = serde_json::from_str(&receiver.handle_envelope(&duplicate)).unwrap();
    assert_eq!(response["code"], "malformed_request");
    let mut oversized = wire(append("slot-0", &packed.chunks[0]));
    oversized["payload"]["chunk"]["data_hex"] = json!("0".repeat(MAX_ENVELOPE_BYTES));
    let response: Value =
        serde_json::from_str(&receiver.handle_envelope(&oversized.to_string())).unwrap();
    assert_eq!(response["code"], "request_too_large");
    receiver
        .certification_request(append("slot-0", &packed.chunks[0]))
        .unwrap();
}

fn wire(request: CertificationRequest) -> Value {
    serde_json::to_value(request).unwrap()
}

fn success(response: &Value) {
    assert_eq!(response["status"], "success", "{response}");
}

struct Cli {
    label: String,
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new(label: String, arguments: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        Self {
            label,
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        }
    }

    fn send(&mut self, request: Value) -> Value {
        let operation = request["type"].as_str().unwrap().to_owned();
        let transfer = request["payload"]["transfer"].as_str().map(str::to_owned);
        let request = request.to_string();
        assert!(request.len() < MAX_ENVELOPE_BYTES);
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "response before EOF"
        );
        assert!(
            line.len() < MAX_ENVELOPE_BYTES,
            "bounded response: {} bytes",
            line.len()
        );
        let response: Value = serde_json::from_str(&line).unwrap();
        if response["status"] == "error" {
            eprintln!(
                "CLI {} operation={} transfer={:?} refused code={}",
                self.label, operation, transfer, response["code"],
            );
        }
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
fn fresh_cli_exchanges_both_proof_kinds_above_the_inline_limit_in_bounded_frames() {
    let large = format!("x{}", "+0*x".repeat(900));
    for (source, root_count, large_case) in [("x^2-2", 2, false), (large.as_str(), 1, true)] {
        // The long source needs explicit ordinary fuel for its quadratic
        // parser allowance and cumulative allocations for original-source
        // checking. The normal two-root exchange uses untouched defaults.
        let arguments: &[&str] = if large_case {
            &["--compute-steps", "200000000", "--allocations", "10000000"]
        } else {
            &[]
        };
        let case = if large_case {
            "large-linear"
        } else {
            "default-quadratic"
        };
        let mut sender = Cli::new(format!("{case}/sender"), arguments);
        let mut receiver = Cli::new(format!("{case}/receiver"), arguments);
        success(&sender.send(wire(registration(source))));
        success(&receiver.send(wire(registration(source))));
        let isolated = sender.send(wire(isolate("sender")));
        success(&isolated);
        assert_eq!(isolated["result"]["root_count"], root_count);
        for (stem, object) in [
            (
                "complete",
                ArtifactSource::RootIsolation(isolation("sender")),
            ),
            ("local", ArtifactSource::RootCertificate(child("sender", 0))),
        ] {
            if large_case {
                let inline = match &object {
                    ArtifactSource::RootIsolation(isolation) => {
                        CertificationRequest::ExportRootIsolation {
                            schema_version: 1,
                            isolation: isolation.clone(),
                        }
                    }
                    ArtifactSource::RootCertificate(certificate) => {
                        CertificationRequest::ExportRootCertificate {
                            schema_version: 1,
                            certificate: certificate.clone(),
                        }
                    }
                    ArtifactSource::IntegralCertificate(integral) => {
                        CertificationRequest::ExportIntegralCertificate {
                            schema_version: 1,
                            integral: integral.clone(),
                        }
                    }
                    ArtifactSource::RootIsolationSearch(search) => {
                        CertificationRequest::ExportRootIsolationSearch {
                            schema_version: 1,
                            search: search.clone(),
                        }
                    }
                };
                assert_eq!(sender.send(wire(inline))["code"], "resource_limit_exceeded");
            }
            let exported = sender.send(wire(CertificationRequest::BeginArtifactExport {
                schema_version: 1,
                transfer: transfer(stem),
                source: object,
            }));
            success(&exported);
            let manifest: ArtifactManifest =
                serde_json::from_value(exported["result"]["manifest"].clone()).unwrap();
            if large_case {
                assert!(manifest.total_bytes > MAX_SESSION_ROOT_ARTIFACT_BYTES);
                assert!(manifest.chunk_count() > 1);
            }
            success(&receiver.send(wire(begin_import(stem, &manifest))));
            for index in 0..manifest.chunk_count() {
                let chunk = sender.send(wire(CertificationRequest::ReadArtifactChunk {
                    schema_version: 1,
                    transfer: transfer(stem),
                    index,
                }));
                success(&chunk);
                let chunk: ArtifactChunk =
                    serde_json::from_value(chunk["result"]["chunk"].clone()).unwrap();
                let accepted = receiver.send(wire(append(stem, &chunk)));
                success(&accepted);
                assert_eq!(accepted["result"]["progress"]["next_chunk"], index + 1);
                if index == 0 {
                    let retry = receiver.send(wire(append(stem, &chunk)));
                    assert_eq!(retry, accepted);
                }
            }
            let request = if stem == "complete" {
                replay("received", stem)
            } else {
                CertificationRequest::ReplayRootCertificateFromArtifact {
                    schema_version: 1,
                    system: system(),
                    certificate: certificate("local"),
                    transfer: transfer(stem),
                }
            };
            let mut request = wire(request);
            request["id"] = json!(format!("replay-{stem}"));
            let replayed = receiver.send(request);
            success(&replayed);
            assert_eq!(replayed["id"], format!("replay-{stem}"));
            assert_eq!(replayed["result"]["evidence"], "certificate_verified");
            if stem == "complete" {
                assert_eq!(replayed["result"]["root_count"], root_count);
                assert_eq!(replayed["result"]["domain"], isolated["result"]["domain"]);
            }
            let replayed_source = if stem == "complete" {
                ArtifactSource::RootIsolation(isolation("received"))
            } else {
                ArtifactSource::RootCertificate(certificate("local"))
            };
            let reexported = receiver.send(wire(CertificationRequest::BeginArtifactExport {
                schema_version: 1,
                transfer: transfer(&format!("again-{stem}")),
                source: replayed_source,
            }));
            success(&reexported);
            assert_eq!(
                reexported["result"]["manifest"],
                exported["result"]["manifest"]
            );
            success(
                &receiver.send(wire(CertificationRequest::ReleaseArtifactTransfer {
                    schema_version: 1,
                    transfer: transfer(stem),
                })),
            );
        }
        sender.finish();
        receiver.finish();
    }
}
