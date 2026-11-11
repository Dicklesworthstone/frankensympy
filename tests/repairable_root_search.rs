//! Independent end-to-end tests of recovery without mathematical promotion.

use frankensympy::certification::CertificationError;
use frankensympy::repairable_search::{
    ROOT_SEARCH_REPAIR_CHUNK_SCHEMA, ROOT_SEARCH_REPAIR_MANIFEST_SCHEMA,
    ROOT_SEARCH_SOURCE_CHUNK_SCHEMA, RepairableRootSearchCheckpoint,
    RepairableRootSearchCheckpointStore, RepairableSearchError,
};
use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, Dimension, ExactInterval, RealBall,
    ResidualProblem, ResidualSystemId, RootIsolationId, RootIsolationOptions, RootIsolationSearch,
    RootIsolationSearchId, Session,
};
use fsym_budget::Unbounded;
use fsym_durable::{DurableRecord, DurableStore, FileStore, hex_lower};
use fsym_runtime::RepairSidecar;
use serde_json::{Value, json};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

fn directory(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "fsym-repairable-search-{label}-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

fn options() -> RootIsolationOptions {
    RootIsolationOptions {
        precision_digits: 16,
        max_leaves: 512,
    }
}
fn native_domain() -> Vec<RealBall> {
    vec![RealBall::new(0_i64.into(), 2_i64.into()).unwrap()]
}
fn problem(source: &str) -> ResidualProblem {
    ResidualProblem::parse_sources(&[source.into()], &["x".into()], &mut Unbounded).unwrap()
}
fn paused() -> RootIsolationSearch {
    let mut search = problem("x^2-2")
        .begin_isolation_search(&native_domain(), options(), &mut Unbounded)
        .unwrap();
    search.advance(2, &mut Unbounded).unwrap();
    assert!(search.progress().pending_cells > 0);
    search
}
fn save(root: &Path, search: &mut RootIsolationSearch) -> RepairableRootSearchCheckpoint {
    let store = RepairableRootSearchCheckpointStore::open(root).unwrap();
    let prepared = store.prepare(search, [7; 32], &mut Unbounded).unwrap();
    store.commit(prepared, &mut Unbounded).unwrap()
}
fn manifest(root: &Path, key: RepairableRootSearchCheckpoint) -> (DurableRecord, Value) {
    let record = FileStore::open(root)
        .unwrap()
        .load_committed_record(key.manifest_digest)
        .unwrap();
    let parsed = serde_json::from_slice(&record.payload).unwrap();
    (record, parsed)
}
fn digest(value: &Value) -> [u8; 32] {
    serde_json::from_value(value.clone()).unwrap()
}
fn chunk_path(
    root: &Path,
    context: [u8; 32],
    schema: &str,
    key: &Value,
    symbol_size: usize,
) -> PathBuf {
    root.join(hex_lower(&context))
        .join(schema)
        .join("committed")
        .join(format!(
            "{}-{}.record",
            hex_lower(&digest(&key["record_digest"])),
            symbol_size + 4
        ))
}
fn move_chunk(root: &Path, path: &Path, label: &str) {
    // Preserve the fixture bytes while simulating an unavailable symbol.
    fs::rename(path, root.join(format!("unavailable-{label}"))).unwrap();
}
fn restore(
    root: &Path,
    key: RepairableRootSearchCheckpoint,
) -> Result<
    (
        RootIsolationSearch,
        frankensympy::repairable_search::RootSearchRepairReport,
    ),
    RepairableSearchError,
> {
    RepairableRootSearchCheckpointStore::open(root)
        .unwrap()
        .restore(
            key,
            &problem("x^2-2"),
            &native_domain(),
            options(),
            [7; 32],
            &mut Unbounded,
        )
}

#[test]
fn actual_missing_and_corrupt_source_and_repair_records_recover_the_same_candidate() {
    let root = directory("recover");
    let mut original = paused();
    let progress = original.progress();
    let key = save(&root, &mut original);
    let (record, manifest) = manifest(&root, key);
    let size = manifest["symbol_size"].as_u64().unwrap() as usize;
    assert!(manifest["sources"].as_array().unwrap().len() >= 2);
    let source = |index| {
        chunk_path(
            &root,
            record.universe_id,
            ROOT_SEARCH_SOURCE_CHUNK_SCHEMA,
            &manifest["sources"][index],
            size,
        )
    };
    let repair = |index| {
        chunk_path(
            &root,
            record.universe_id,
            ROOT_SEARCH_REPAIR_CHUNK_SCHEMA,
            &manifest["repairs"][index],
            size,
        )
    };
    move_chunk(&root, &source(0), "source-0");
    fs::write(source(1), b"torn source record").unwrap();
    move_chunk(&root, &repair(0), "repair-0");
    fs::write(repair(1), b"torn repair record").unwrap();

    let (mut recovered, report) = restore(&root, key).unwrap();
    assert!(report.recovered);
    assert_eq!(
        (
            report.source_missing,
            report.source_corrupt,
            report.repair_missing,
            report.repair_corrupt
        ),
        (1, 1, 1, 1)
    );
    assert_eq!(recovered.progress(), progress);
    assert!(recovered.completed_witness().is_none());
    recovered.advance(2047, &mut Unbounded).unwrap();
    original.advance(2047, &mut Unbounded).unwrap();
    assert_eq!(recovered.completed_witness(), original.completed_witness());
    let receipt = problem("x^2-2")
        .verify_isolation(
            &native_domain(),
            recovered.completed_witness().unwrap(),
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(receipt.root_count(), 2);
}

#[test]
fn loss_beyond_available_equations_refuses_instead_of_reusing_an_old_checkpoint() {
    let root = directory("unrecoverable");
    let mut original = paused();
    let key = save(&root, &mut original);
    let (record, manifest) = manifest(&root, key);
    let size = manifest["symbol_size"].as_u64().unwrap() as usize;
    for (field, schema) in [
        ("sources", ROOT_SEARCH_SOURCE_CHUNK_SCHEMA),
        ("repairs", ROOT_SEARCH_REPAIR_CHUNK_SCHEMA),
    ] {
        for (index, chunk) in manifest[field].as_array().unwrap().iter().enumerate() {
            let path = chunk_path(&root, record.universe_id, schema, chunk, size);
            move_chunk(&root, &path, &format!("{field}-{index}"));
        }
    }
    assert!(matches!(
        restore(&root, key),
        Err(RepairableSearchError::Repair(_))
    ));
    assert!(
        FileStore::open(&root)
            .unwrap()
            .load_committed_record(key.manifest_digest)
            .is_ok()
    );
}

#[test]
fn repaired_bytes_still_bind_the_receivers_source_domain_options_and_context() {
    let root = directory("binding");
    let key = save(&root, &mut paused());
    let store = RepairableRootSearchCheckpointStore::open(&root).unwrap();
    assert!(matches!(
        store.restore(
            key,
            &problem("x^2-3"),
            &native_domain(),
            options(),
            [7; 32],
            &mut Unbounded
        ),
        Err(RepairableSearchError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    assert!(matches!(
        store.restore(
            key,
            &problem("x^2-2"),
            &[RealBall::new(0_i64.into(), 3_i64.into()).unwrap()],
            options(),
            [7; 32],
            &mut Unbounded
        ),
        Err(RepairableSearchError::Certification(
            CertificationError::DomainMismatch
        ))
    ));
    for different in [
        RootIsolationOptions {
            precision_digits: 24,
            ..options()
        },
        RootIsolationOptions {
            max_leaves: 256,
            ..options()
        },
    ] {
        assert!(matches!(
            store.restore(
                key,
                &problem("x^2-2"),
                &native_domain(),
                different,
                [7; 32],
                &mut Unbounded
            ),
            Err(RepairableSearchError::Certification(_))
        ));
    }
    assert!(
        store
            .restore(
                key,
                &problem("x^2-2"),
                &native_domain(),
                options(),
                [8; 32],
                &mut Unbounded
            )
            .is_err()
    );
}

fn publish(store: &FileStore, record: &DurableRecord) {
    store.prepare(record).unwrap();
    store.commit_record(record).unwrap();
}

// Preserve every canonical field's original order while changing selected
// values. Nested digest arrays and one-field chunk keys have no map-order
// dependency, including when serde_json enables preserve_order.
fn replace_field(text: &mut String, parsed: &mut Value, field: &str, replacement: Value) {
    let old = format!("\"{field}\":{}", parsed[field]);
    assert_eq!(text.matches(&old).count(), 1, "unique field {field}");
    *text = text.replacen(&old, &format!("\"{field}\":{replacement}"), 1);
    parsed[field] = replacement;
}

#[test]
fn canonical_self_consistent_repair_envelope_cannot_authorize_forged_search_history() {
    let root = directory("false-prefix");
    let mut candidate = paused();
    let key = save(&root, &mut candidate);
    let (stored, mut manifest) = manifest(&root, key);
    let old_size = manifest["symbol_size"].as_u64().unwrap() as usize;
    let store = FileStore::open(&root).unwrap();
    let mut payload = Vec::new();
    for chunk in manifest["sources"].as_array().unwrap() {
        let record = store
            .load_committed_record(digest(&chunk["record_digest"]))
            .unwrap();
        assert_eq!(record.payload.len(), old_size + 4);
        payload.extend_from_slice(&record.payload[4..]);
    }
    payload.truncate(key.payload_bytes);
    let mut checkpoint = String::from_utf8(payload).unwrap();
    let parsed: Value = serde_json::from_str(&checkpoint).unwrap();
    let steps = parsed["counters"]["steps"].as_u64().unwrap();
    assert!(steps > 0);
    let original = format!("\"steps\":{steps}");
    assert_eq!(checkpoint.matches(&original).count(), 1);
    checkpoint = checkpoint.replacen(&original, "\"steps\":0", 1);
    let payload = checkpoint.as_bytes();
    let size = payload.len().div_ceil(64).max(256);
    let sidecar = RepairSidecar::encode(payload, size, 16).unwrap();
    let mut sources = Vec::new();
    for index in 0..sidecar.num_source_symbols {
        let mut symbol = vec![0; size + 4];
        symbol[..4].copy_from_slice(&(index as u32).to_le_bytes());
        let start = index * size;
        let end = (start + size).min(payload.len());
        symbol[4..4 + end - start].copy_from_slice(&payload[start..end]);
        let record = DurableRecord::new(
            stored.universe_id,
            ROOT_SEARCH_SOURCE_CHUNK_SCHEMA,
            symbol,
            stored.dependencies.clone(),
        )
        .unwrap();
        sources.push(json!({"record_digest": record.record_digest}));
        publish(&store, &record);
    }
    let mut repairs = Vec::new();
    for (esi, bytes) in &sidecar.repair_symbols {
        let mut symbol = esi.to_le_bytes().to_vec();
        symbol.extend_from_slice(bytes);
        let record = DurableRecord::new(
            stored.universe_id,
            ROOT_SEARCH_REPAIR_CHUNK_SCHEMA,
            symbol,
            stored.dependencies.clone(),
        )
        .unwrap();
        repairs.push(json!({"record_digest": record.record_digest}));
        publish(&store, &record);
    }
    let mut manifest_text = String::from_utf8(stored.payload).unwrap();
    for (field, replacement) in [
        ("original_len", json!(payload.len())),
        ("symbol_size", json!(sidecar.symbol_size)),
        ("num_source_symbols", json!(sidecar.num_source_symbols)),
        ("codec_seed", json!(sidecar.codec_seed)),
        ("canonical_digest", json!(sidecar.canonical_digest)),
        (
            "source_symbol_digests",
            json!(sidecar.source_symbol_digests),
        ),
        (
            "repair_symbol_digests",
            json!(sidecar.repair_symbol_digests),
        ),
        ("sources", json!(sources)),
        ("repairs", json!(repairs)),
    ] {
        replace_field(&mut manifest_text, &mut manifest, field, replacement);
    }
    let poisoned = DurableRecord::new(
        stored.universe_id,
        ROOT_SEARCH_REPAIR_MANIFEST_SCHEMA,
        manifest_text.into_bytes(),
        stored.dependencies,
    )
    .unwrap();
    publish(&store, &poisoned);
    let poisoned_key = RepairableRootSearchCheckpoint {
        manifest_digest: poisoned.record_digest,
        manifest_bytes: poisoned.payload.len(),
        payload_bytes: payload.len(),
    };
    assert!(
        matches!(
            restore(&root, poisoned_key),
            Err(RepairableSearchError::Certification(_))
        ),
        "checksummed, decodable canonical bytes must still fail the independent prefix checker"
    );
    assert!(restore(&root, key).is_ok());
}

#[test]
fn missing_manifest_and_forged_bounds_never_trigger_symbol_discovery() {
    let root = directory("manifest-admission");
    let key = save(&root, &mut paused());
    for invalid in [
        RepairableRootSearchCheckpoint {
            manifest_bytes: 0,
            ..key
        },
        RepairableRootSearchCheckpoint {
            manifest_bytes: 64 * 1024 + 1,
            ..key
        },
        RepairableRootSearchCheckpoint {
            payload_bytes: 0,
            ..key
        },
        RepairableRootSearchCheckpoint {
            payload_bytes: 1024 * 1024 + 1,
            ..key
        },
        RepairableRootSearchCheckpoint {
            manifest_digest: [0; 32],
            ..key
        },
    ] {
        assert!(restore(&root, invalid).is_err());
    }
    let (record, _) = manifest(&root, key);
    let path = root
        .join(hex_lower(&record.universe_id))
        .join(ROOT_SEARCH_REPAIR_MANIFEST_SCHEMA)
        .join("committed")
        .join(format!(
            "{}-{}.record",
            hex_lower(&key.manifest_digest),
            key.manifest_bytes
        ));
    fs::write(path, b"truncated manifest").unwrap();
    assert!(matches!(
        restore(&root, key),
        Err(RepairableSearchError::Storage(_))
    ));
}

fn limits() -> BudgetLimits {
    let mut allowance = BudgetLimits::uniform(100_000_000, 100_000_000);
    allowance.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    allowance.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    allowance
}
fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:repair").unwrap()
}
fn search(name: &str) -> RootIsolationSearchId {
    RootIsolationSearchId::new(format!("search:{name}")).unwrap()
}
fn registration() -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec!["x^2-2".into()],
        variables: vec!["x".into()],
    }
}
fn begin() -> CertificationRequest {
    CertificationRequest::BeginRootIsolationSearch {
        schema_version: 1,
        system: system(),
        search: search("paused"),
        domain: vec![ExactInterval {
            lower: "-2".into(),
            upper: "2".into(),
        }],
        precision_digits: 16,
        max_leaves: 512,
    }
}
fn advance(name: &str, max_cells: usize) -> CertificationRequest {
    CertificationRequest::AdvanceRootIsolationSearch {
        schema_version: 1,
        search: search(name),
        max_cells,
    }
}
fn load(name: &str, checkpoint: RepairableRootSearchCheckpoint) -> CertificationRequest {
    CertificationRequest::LoadRepairableRootIsolationSearch {
        schema_version: 1,
        system: system(),
        search: search(name),
        domain: vec![ExactInterval {
            lower: "-2".into(),
            upper: "2".into(),
        }],
        precision_digits: 16,
        max_leaves: 512,
        checkpoint,
    }
}
fn finish(name: &str) -> CertificationRequest {
    CertificationRequest::FinishRootIsolationSearch {
        schema_version: 1,
        search: search(name),
        isolation: RootIsolationId::new(format!("isolation:{name}")).unwrap(),
    }
}

#[test]
fn session_recovery_grants_no_evidence_or_protected_budget_until_finish() {
    let root = directory("session");
    let mut sender = Session::with_budget_limits(limits(), 256)
        .with_checkpoint_directory(&root)
        .unwrap();
    sender.certification_request(registration()).unwrap();
    sender.certification_request(begin()).unwrap();
    sender.certification_request(advance("paused", 2)).unwrap();
    let progress = sender
        .root_isolation_search_progress(&search("paused"))
        .unwrap();
    let protected = sender.budget_snapshot().verifier_pool;
    let CertificationResponse::RootIsolationSearchSavedWithRepair {
        checkpoint,
        evidence,
        ..
    } = sender
        .certification_request(CertificationRequest::SaveRepairableRootIsolationSearch {
            schema_version: 1,
            search: search("paused"),
        })
        .unwrap()
    else {
        panic!("repairable save")
    };
    assert_eq!(evidence, "heuristic_candidate");
    assert_eq!(sender.budget_snapshot().verifier_pool, protected);
    let (record, manifest) = manifest(&root, checkpoint);
    let path = chunk_path(
        &root,
        record.universe_id,
        ROOT_SEARCH_SOURCE_CHUNK_SCHEMA,
        &manifest["sources"][0],
        manifest["symbol_size"].as_u64().unwrap() as usize,
    );
    move_chunk(&root, &path, "session-source");
    let mut receiver = Session::with_budget_limits(limits(), 256)
        .with_checkpoint_directory(&root)
        .unwrap();
    receiver.certification_request(registration()).unwrap();
    let protected = receiver.budget_snapshot().verifier_pool;
    let CertificationResponse::RootIsolationSearchRecovered {
        evidence,
        repair,
        progress: restored,
        ..
    } = receiver
        .certification_request(load("recovered", checkpoint))
        .unwrap()
    else {
        panic!("recovered candidate")
    };
    assert_eq!(evidence, "heuristic_candidate");
    assert_eq!(restored, progress);
    assert!(repair.recovered);
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    assert!(
        receiver
            .root_isolation(&RootIsolationId::new("isolation:recovered").unwrap())
            .is_none()
    );
    receiver
        .certification_request(advance("recovered", 2047))
        .unwrap();
    receiver.certification_request(finish("recovered")).unwrap();
    assert_eq!(
        receiver
            .root_isolation(&RootIsolationId::new("isolation:recovered").unwrap())
            .unwrap()
            .root_count(),
        2
    );
    assert!(receiver.budget_snapshot().verifier_pool < protected);
}

struct Cli {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl Cli {
    fn start(root: &Path, repair_budget: bool) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_frankensympy"));
        command.arg("--checkpoint-dir").arg(root);
        if repair_budget {
            command.args(["--compute-steps", "100000000"]);
        }
        let mut child = command
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
    fn exchange(&mut self, request: CertificationRequest) -> Value {
        writeln!(self.input, "{}", serde_json::to_string(&request).unwrap()).unwrap();
        self.input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
        serde_json::from_str(&line).unwrap()
    }
    fn request(&mut self, request: CertificationRequest) -> Value {
        let response = self.exchange(request);
        assert_eq!(response["status"], "success", "{response}");
        response
    }
    fn close(mut self) {
        drop(self.input);
        assert!(self.child.wait().unwrap().success());
    }
}

#[test]
fn default_cli_refuses_unfunded_repair_work_without_publishing_a_manifest() {
    let root = directory("default-budget");
    let mut cli = Cli::start(&root, false);
    cli.request(registration());
    cli.request(begin());
    let progress = cli.request(advance("paused", 2))["result"]["progress"].clone();
    let refused = cli.exchange(CertificationRequest::SaveRepairableRootIsolationSearch {
        schema_version: 1,
        search: search("paused"),
    });
    assert_eq!(refused["status"], "error");
    assert_eq!(refused["code"], "budget_exhausted");
    let current = cli.request(CertificationRequest::InspectRootIsolationSearch {
        schema_version: 1,
        search: search("paused"),
    });
    assert_eq!(current["result"]["progress"], progress);
    cli.close();
    assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
}

#[test]
fn fresh_cli_process_recovers_missing_bytes_and_independently_finishes_roots() {
    let root = directory("fresh-process");
    let mut sender = Cli::start(&root, true);
    sender.request(registration());
    sender.request(begin());
    let progress = sender.request(advance("paused", 2))["result"]["progress"].clone();
    let saved = sender.request(CertificationRequest::SaveRepairableRootIsolationSearch {
        schema_version: 1,
        search: search("paused"),
    });
    let key: RepairableRootSearchCheckpoint =
        serde_json::from_value(saved["result"]["checkpoint"].clone()).unwrap();
    sender.close();
    let (record, manifest) = manifest(&root, key);
    let path = chunk_path(
        &root,
        record.universe_id,
        ROOT_SEARCH_SOURCE_CHUNK_SCHEMA,
        &manifest["sources"][0],
        manifest["symbol_size"].as_u64().unwrap() as usize,
    );
    move_chunk(&root, &path, "fresh-source");
    let mut receiver = Cli::start(&root, true);
    receiver.request(registration());
    let restored = receiver.request(load("fresh", key));
    assert_eq!(restored["result"]["progress"], progress);
    assert_eq!(restored["result"]["evidence"], "heuristic_candidate");
    assert_eq!(restored["result"]["repair"]["source_missing"], 1);
    assert_eq!(restored["result"]["repair"]["recovered"], true);
    receiver.request(advance("fresh", 2047));
    let accepted = receiver.request(finish("fresh"));
    assert_eq!(accepted["result"]["root_count"], 2);
    assert_eq!(accepted["result"]["evidence"], "certificate_verified");
    receiver.close();
}
