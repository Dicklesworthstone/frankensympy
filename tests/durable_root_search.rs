//! File-backed continuation remains a source-bound candidate until verification.

use frankensympy::durable_search::{
    DURABLE_ROOT_SEARCH_SCHEMA, DurableSearchError, RootSearchCheckpointStore,
    StoredRootSearchCheckpoint,
};
use frankensympy::{
    Budget, BudgetLimits, BudgetMeter, CertificationRequest, CertificationResponse, Dimension,
    ExactInterval, MAX_ROOT_SEARCH_CHECKPOINT_BYTES, MeterError, RealBall, ResidualProblem,
    ResidualSystemId, RootIsolationId, RootIsolationOptions, RootIsolationSearchId, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Unbounded;
use fsym_durable::{DurableRecord, DurableStore, FileStore, hex_lower};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

fn directory(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "fsym-durable-search-{label}-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:durable").unwrap()
}
fn search(label: &str) -> RootIsolationSearchId {
    RootIsolationSearchId::new(format!("search:{label}")).unwrap()
}
fn isolation(label: &str) -> RootIsolationId {
    RootIsolationId::new(format!("isolation:{label}")).unwrap()
}
fn domain() -> Vec<ExactInterval> {
    vec![ExactInterval {
        lower: "-2".into(),
        upper: "2".into(),
    }]
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
fn native_problem() -> ResidualProblem {
    ResidualProblem::parse_sources(&["x^2-2".into()], &["x".into()], &mut Unbounded).unwrap()
}

fn register(source: &str) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec![source.into()],
        variables: vec!["x".into()],
    }
}
fn begin(label: &str) -> CertificationRequest {
    CertificationRequest::BeginRootIsolationSearch {
        schema_version: 1,
        system: system(),
        search: search(label),
        domain: domain(),
        precision_digits: 16,
        max_leaves: 512,
    }
}
fn advance(label: &str, max_cells: usize) -> CertificationRequest {
    CertificationRequest::AdvanceRootIsolationSearch {
        schema_version: 1,
        search: search(label),
        max_cells,
    }
}
fn finish(label: &str) -> CertificationRequest {
    CertificationRequest::FinishRootIsolationSearch {
        schema_version: 1,
        search: search(label),
        isolation: isolation(label),
    }
}
fn load(label: &str, checkpoint: StoredRootSearchCheckpoint) -> CertificationRequest {
    CertificationRequest::LoadRootIsolationSearch {
        schema_version: 1,
        system: system(),
        search: search(label),
        domain: domain(),
        precision_digits: 16,
        max_leaves: 512,
        checkpoint,
    }
}
fn registered(root: &Path, source: &str) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256)
        .with_checkpoint_directory(root)
        .unwrap();
    session.certification_request(register(source)).unwrap();
    session
}
fn paused(root: &Path) -> Session {
    let mut session = registered(root, "x^2-2");
    session.certification_request(begin("paused")).unwrap();
    session.certification_request(advance("paused", 2)).unwrap();
    session
}
fn save(session: &mut Session, label: &str) -> StoredRootSearchCheckpoint {
    let CertificationResponse::RootIsolationSearchSaved {
        checkpoint,
        evidence,
        ..
    } = session
        .certification_request(CertificationRequest::SaveRootIsolationSearch {
            schema_version: 1,
            search: search(label),
        })
        .unwrap()
    else {
        panic!("save response")
    };
    assert_eq!(evidence, "heuristic_candidate");
    checkpoint
}
fn no_publication(session: &Session, label: &str) {
    assert!(
        session
            .root_isolation_search_progress(&search(label))
            .is_none()
    );
    assert!(session.root_isolation(&isolation(label)).is_none());
}
fn record_path(root: &Path, context: [u8; 32], key: StoredRootSearchCheckpoint) -> PathBuf {
    root.join(hex_lower(&context))
        .join(DURABLE_ROOT_SEARCH_SCHEMA)
        .join("committed")
        .join(format!(
            "{}-{}.record",
            hex_lower(&key.record_digest),
            key.payload_bytes
        ))
}

#[test]
fn real_file_round_trip_preserves_partial_progress_and_requires_protected_finish() {
    let root = directory("round-trip");
    let mut sender = paused(&root);
    let before = sender
        .root_isolation_search_progress(&search("paused"))
        .unwrap();
    assert!(before.processed_cells > 0 && before.pending_cells > 0);
    let protected = sender.budget_snapshot().verifier_pool;
    let key = save(&mut sender, "paused");
    assert_eq!(sender.budget_snapshot().verifier_pool, protected);
    assert!(record_path(&root, sender.context_digest(), key).is_file());

    let mut receiver = registered(&root, "x^2-2");
    let before_budget = receiver.budget_snapshot();
    let CertificationResponse::RootIsolationSearchRestored {
        evidence, progress, ..
    } = receiver
        .certification_request(load("resumed", key))
        .unwrap()
    else {
        panic!("candidate restore")
    };
    assert_eq!(evidence, "heuristic_candidate");
    assert_eq!(progress, before);
    assert_eq!(
        receiver.budget_snapshot().verifier_pool,
        before_budget.verifier_pool
    );
    assert!(
        receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
            < before_budget.dimensions[Dimension::ComputeSteps.index()]
    );
    assert!(receiver.root_isolation(&isolation("resumed")).is_none());
    assert_eq!(
        receiver
            .certification_request(finish("resumed"))
            .unwrap_err()
            .code(),
        "search_not_ready"
    );
    receiver
        .certification_request(advance("resumed", 2047))
        .unwrap();
    receiver.certification_request(finish("resumed")).unwrap();
    sender
        .certification_request(advance("paused", 2047))
        .unwrap();
    sender.certification_request(finish("paused")).unwrap();
    assert_eq!(
        receiver.root_isolation(&isolation("resumed")),
        sender.root_isolation(&isolation("paused"))
    );
    assert_eq!(
        receiver
            .root_isolation(&isolation("resumed"))
            .unwrap()
            .root_count(),
        2
    );
    assert!(receiver.budget_snapshot().verifier_pool < before_budget.verifier_pool);
}

#[test]
fn stored_state_cannot_supply_protected_verifier_authority() {
    let root = directory("protected");
    let mut sender = paused(&root);
    sender
        .certification_request(advance("paused", 2047))
        .unwrap();
    let key = save(&mut sender, "paused");
    let mut allowance = limits();
    allowance.verifier_pool = 0;
    let mut receiver = Session::with_budget_limits(allowance, 256)
        .with_checkpoint_directory(&root)
        .unwrap();
    receiver.certification_request(register("x^2-2")).unwrap();
    receiver.certification_request(load("ready", key)).unwrap();
    assert!(
        receiver
            .root_isolation_search_progress(&search("ready"))
            .unwrap()
            .ready_for_verification
    );
    assert_eq!(
        receiver
            .certification_request(finish("ready"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(receiver.root_isolation(&isolation("ready")).is_none());
    assert!(receiver.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
}

#[test]
fn source_domain_options_and_receiver_context_remain_immutable() {
    let root = directory("universe");
    let mut sender = paused(&root);
    let key = save(&mut sender, "paused");
    let mut foreign_source = registered(&root, "x^2-3");
    assert_eq!(
        foreign_source
            .certification_request(load("foreign", key))
            .unwrap_err()
            .code(),
        "source_mismatch"
    );
    no_publication(&foreign_source, "foreign");
    for field in ["domain", "precision_digits", "max_leaves"] {
        let mut request = serde_json::to_value(load("foreign", key)).unwrap();
        request["payload"][field] = match field {
            "domain" => json!([{"lower":"-3","upper":"3"}]),
            "precision_digits" => json!(24),
            _ => json!(256),
        };
        let mut receiver = registered(&root, "x^2-2");
        assert!(
            receiver
                .certification_request(serde_json::from_value(request).unwrap())
                .is_err()
        );
        no_publication(&receiver, "foreign");
    }
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(HashMap::new(), HashMap::new(), "different-durable-context")
        .unwrap();
    let mut receiver = Session::with_context_and_budget_limits(context, limits(), 256)
        .with_checkpoint_directory(&root)
        .unwrap();
    receiver.certification_request(register("x^2-2")).unwrap();
    assert_ne!(receiver.context_digest(), sender.context_digest());
    assert!(
        receiver
            .certification_request(load("foreign", key))
            .is_err()
    );
    no_publication(&receiver, "foreign");
}

#[test]
fn requests_cannot_enable_persistence_or_choose_a_path() {
    let root = directory("capability");
    let mut sender = paused(&root);
    let key = save(&mut sender, "paused");
    let mut receiver = Session::with_budget_limits(limits(), 256);
    receiver.certification_request(register("x^2-2")).unwrap();
    receiver.certification_request(begin("ephemeral")).unwrap();
    for request in [
        CertificationRequest::SaveRootIsolationSearch {
            schema_version: 1,
            search: search("ephemeral"),
        },
        load("disabled", key),
    ] {
        assert_eq!(
            receiver.certification_request(request).unwrap_err().code(),
            "persistence_disabled"
        );
    }
    no_publication(&receiver, "disabled");
    let mut path_request = serde_json::to_value(load("injected", key)).unwrap();
    path_request["payload"]["checkpoint"]["path"] = json!(root.join("other").display().to_string());
    let response: Value =
        serde_json::from_str(&sender.handle_envelope(&path_request.to_string())).unwrap();
    assert_eq!(response["status"], "error");
    no_publication(&sender, "injected");
    assert!(!root.join("other").exists());
}

#[test]
fn forged_keys_refuse_without_occupying_the_requested_candidate_slot() {
    let root = directory("keys");
    let mut sender = paused(&root);
    let key = save(&mut sender, "paused");
    let mut receiver = registered(&root, "x^2-2");
    for invalid in [
        StoredRootSearchCheckpoint {
            payload_bytes: 0,
            ..key
        },
        StoredRootSearchCheckpoint {
            payload_bytes: MAX_ROOT_SEARCH_CHECKPOINT_BYTES + 1,
            ..key
        },
        StoredRootSearchCheckpoint {
            payload_bytes: key.payload_bytes - 1,
            ..key
        },
        StoredRootSearchCheckpoint {
            payload_bytes: key.payload_bytes + 1,
            ..key
        },
        StoredRootSearchCheckpoint {
            record_digest: [0; 32],
            ..key
        },
    ] {
        assert!(
            receiver
                .certification_request(load("retry", invalid))
                .is_err()
        );
        no_publication(&receiver, "retry");
    }
    receiver.certification_request(load("retry", key)).unwrap();
    assert!(
        receiver
            .root_isolation_search_progress(&search("retry"))
            .is_some()
    );
}

#[test]
fn truncation_and_byte_corruption_cannot_publish_a_search() {
    for corruption in ["truncate", "flip"] {
        let root = directory(corruption);
        let mut sender = paused(&root);
        let key = save(&mut sender, "paused");
        let path = record_path(&root, sender.context_digest(), key);
        let mut wire = fs::read(&path).unwrap();
        if corruption == "truncate" {
            wire.truncate(wire.len() / 2);
        } else {
            let last = wire.len() - 1;
            wire[last] ^= 1;
        }
        fs::write(path, wire).unwrap();
        let mut receiver = registered(&root, "x^2-2");
        let protected = receiver.budget_snapshot().verifier_pool;
        assert!(
            receiver
                .certification_request(load("corrupt", key))
                .is_err()
        );
        no_publication(&receiver, "corrupt");
        assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    }
}

#[test]
fn digest_consistent_false_progress_still_requires_independent_restoration() {
    let root = directory("false-progress");
    let mut sender = paused(&root);
    let key = save(&mut sender, "paused");
    let store = FileStore::open(&root).unwrap();
    let stored = store.load_committed_record(key.record_digest).unwrap();
    let text = String::from_utf8(stored.payload.clone()).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let original = format!("\"steps\":{}", value["counters"]["steps"]);
    assert_eq!(text.matches(&original).count(), 1);
    assert!(value["counters"]["steps"].as_u64().unwrap() > 0);
    let forged_payload = text.replacen(&original, "\"steps\":0", 1).into_bytes();
    let forged = DurableRecord::new(
        stored.universe_id,
        stored.payload_schema,
        forged_payload,
        stored.dependencies,
    )
    .unwrap();
    let handle = store.prepare(&forged).unwrap();
    store.commit(&handle).unwrap();
    let forged_key = StoredRootSearchCheckpoint {
        record_digest: forged.record_digest,
        payload_bytes: forged.payload.len(),
    };
    let mut receiver = registered(&root, "x^2-2");
    assert_eq!(
        receiver
            .certification_request(load("forged", forged_key))
            .unwrap_err()
            .code(),
        "invalid_artifact"
    );
    no_publication(&receiver, "forged");
    receiver
        .certification_request(load("original", key))
        .unwrap();
}

#[test]
fn exact_older_snapshot_remains_available_after_later_progress_is_saved() {
    let root = directory("versions");
    let mut sender = paused(&root);
    let prior = sender
        .root_isolation_search_progress(&search("paused"))
        .unwrap();
    let first = save(&mut sender, "paused");
    sender
        .certification_request(advance("paused", 2047))
        .unwrap();
    let ready = sender
        .root_isolation_search_progress(&search("paused"))
        .unwrap();
    let second = save(&mut sender, "paused");
    assert_ne!(first.record_digest, second.record_digest);
    let mut receiver = registered(&root, "x^2-2");
    receiver
        .certification_request(load("first", first))
        .unwrap();
    receiver
        .certification_request(load("second", second))
        .unwrap();
    assert_eq!(
        receiver
            .root_isolation_search_progress(&search("first"))
            .unwrap(),
        prior
    );
    assert_eq!(
        receiver
            .root_isolation_search_progress(&search("second"))
            .unwrap(),
        ready
    );
    assert!(!prior.ready_for_verification && ready.ready_for_verification);
    assert!(receiver.root_isolation(&isolation("second")).is_none());
}

#[test]
fn exhausted_receiver_cannot_borrow_verifier_resources_or_publish_a_search() {
    let root = directory("budget");
    let mut sender = paused(&root);
    let key = save(&mut sender, "paused");
    let mut measurement = Session::with_budget_limits(limits(), 256);
    measurement
        .certification_request(register("x^2-2"))
        .unwrap();
    let registration_cost = limits().dimensions[Dimension::ComputeSteps.index()]
        - measurement.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
    let mut allowance = limits();
    allowance.dimensions[Dimension::ComputeSteps.index()] = registration_cost + 1;
    let mut receiver = Session::with_budget_limits(allowance, 256)
        .with_checkpoint_directory(&root)
        .unwrap();
    receiver.certification_request(register("x^2-2")).unwrap();
    let protected = receiver.budget_snapshot().verifier_pool;
    assert_eq!(
        receiver
            .certification_request(load("starved", key))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    no_publication(&receiver, "starved");
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
}

struct CancelAfter {
    remaining: usize,
    budget: Budget,
}
impl BudgetMeter for CancelAfter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.checkpoint()?;
        self.budget.charge(dimension, amount)
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()?;
        self.budget.charge_batch(charges)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.remaining == 0 {
            return Err(MeterError::Cancelled);
        }
        self.remaining -= 1;
        Ok(())
    }
}

#[test]
fn cancellation_at_each_store_safe_point_cleans_staging_and_preserves_prior_checkpoint() {
    let problem = native_problem();
    let domain = native_domain();
    let mut reached_success = false;
    let mut cancelled_after_staging = 0;
    for safe_points in 0..16 {
        let root = directory("cancel");
        let store = RootSearchCheckpointStore::open(&root).unwrap();
        let mut live = problem
            .begin_isolation_search(&domain, options(), &mut Unbounded)
            .unwrap();
        live.advance(1, &mut Unbounded).unwrap();
        let old_progress = live.progress();
        let previous = store.prepare(&mut live, [7; 32], &mut Unbounded).unwrap();
        let old_key = store.commit(previous, &mut Unbounded).unwrap();
        live.advance(1, &mut Unbounded).unwrap();
        let pending = store.prepare(&mut live, [7; 32], &mut Unbounded).unwrap();
        let key = pending.key();
        let mut meter = CancelAfter {
            remaining: safe_points,
            budget: Budget::new(limits()),
        };
        match store.commit(pending, &mut meter) {
            Ok(returned) => {
                assert_eq!(returned, key);
                reached_success = true;
                break;
            }
            Err(DurableSearchError::Meter(MeterError::Cancelled)) => {
                let files = FileStore::open(&root).unwrap();
                assert_eq!(files.staging_len().unwrap(), 0);
                assert!(files.load_committed_record(key.record_digest).is_err());
                assert_eq!(
                    store
                        .restore(
                            old_key,
                            &problem,
                            &domain,
                            options(),
                            [7; 32],
                            &mut Unbounded
                        )
                        .unwrap()
                        .progress(),
                    old_progress
                );
                assert_eq!(meter.budget.verifier_remaining(), limits().verifier_pool);
                if safe_points >= 2 {
                    cancelled_after_staging += 1;
                }
            }
            Err(error) => panic!("unexpected refusal: {error}"),
        }
    }
    assert!(
        reached_success,
        "bounded cancellation exploration must reach the successful path"
    );
    assert!(
        cancelled_after_staging > 0,
        "must exercise cleanup after staging exists"
    );
}

struct Cli {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl Cli {
    fn start(root: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .arg("--checkpoint-dir")
            .arg(root)
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
fn default_cli_restores_from_disk_after_the_saving_process_exits() {
    let root = directory("cli");
    let mut sender = Cli::start(&root);
    sender.request(register("x^2-2"));
    sender.request(begin("paused"));
    let progress = sender.request(advance("paused", 2))["result"]["progress"].clone();
    let saved = sender.request(CertificationRequest::SaveRootIsolationSearch {
        schema_version: 1,
        search: search("paused"),
    });
    assert_eq!(saved["result"]["evidence"], "heuristic_candidate");
    let key: StoredRootSearchCheckpoint =
        serde_json::from_value(saved["result"]["checkpoint"].clone()).unwrap();
    sender.close();
    let mut receiver = Cli::start(&root);
    receiver.request(register("x^2-2"));
    let resumed = receiver.request(load("fresh", key));
    assert_eq!(resumed["result"]["evidence"], "heuristic_candidate");
    assert_eq!(resumed["result"]["progress"], progress);
    receiver.request(advance("fresh", 2047));
    let accepted = receiver.request(finish("fresh"));
    assert_eq!(accepted["result"]["root_count"], 2);
    assert_eq!(accepted["result"]["evidence"], "certificate_verified");
    receiver.close();
}
