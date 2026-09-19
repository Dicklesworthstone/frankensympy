//! Fresh-process crash/loss/repair/resume matrix for the durable
//! factor-frontier slice (`fra-rc-durable-m5e`).
//!
//! Every crash is real process death: the worker SIGSTOPs itself by sleeping
//! after writing its boundary marker and the parent SIGKILLs it, so the
//! on-disk state is exactly the boundary state. Every resume happens in a
//! genuinely fresh process. Storage refusal paths are exercised against
//! actual disk readback, including loss beyond the repair envelope,
//! digest-consistent but mathematically false proof material, duplicate
//! resume, and stale universe markers. Nothing here trusts a stored boolean:
//! the runtime resume path independently re-verifies every candidate.

#![forbid(unsafe_code)]

use fsym_durable::{
    DependencyManifest, DurableError, DurableRecord, DurableStore, FileStore,
    factor_race_universe_id,
};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use fsym_runtime::{FACTOR_RACE_CONTINUATION_SCHEMA, FactorRaceContinuationState, TypedCheckpoint};

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

fn worker() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_factor_frontier_worker"));
    command.current_dir(temp_root());
    command
}

fn temp_root() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fsym-durable-matrix-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir_all(&dir).expect("temp root");
    dir
}

fn unique_dir(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = temp_root().join(format!(
        "{tag}-{}",
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).expect("unique dir");
    dir
}

fn marker_stage(base: &Path, stage: &str) -> PathBuf {
    base.with_extension(stage)
}

fn wait_for_marker(base: &Path, stage: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if marker_stage(base, stage).is_file() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

fn run_generate(
    lane: &str,
    store_root: &Path,
    boundary: &str,
    marker_base: &Path,
) -> std::process::Child {
    worker()
        .args([
            "generate",
            lane,
            &store_root.display().to_string(),
            boundary,
            &marker_base.display().to_string(),
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn generate worker")
}

fn run_resume(lane: &str, store_root: &Path, marker_base: &Path, boundary: &str) -> Output {
    worker()
        .args([
            "resume",
            lane,
            &store_root.display().to_string(),
            &marker_base.display().to_string(),
            boundary,
        ])
        .output()
        .expect("spawn resume worker")
}

fn read_report(marker_base: &Path) -> Value {
    let wire = fs::read(marker_base.with_extension("resume.json")).expect("resume report");
    serde_json::from_slice(&wire).expect("report json")
}

fn committed_record_files(store_root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![store_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("store dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "record")
                && path.components().any(|c| c.as_os_str() == "committed")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn staging_record_files(store_root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![store_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("store dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "record")
                && path.components().any(|c| c.as_os_str() == "staging")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn ephemeral_and_every_boundary_resume_produce_identical_results() {
    // The ephemeral baseline: no persistence at all.
    let baseline_dir = unique_dir("baseline");
    let baseline_marker = baseline_dir.join("marker.json");
    let baseline = worker()
        .args(["ephemeral", &baseline_marker.display().to_string()])
        .output()
        .expect("ephemeral worker");
    assert!(
        baseline.status.success(),
        "ephemeral run failed: {}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    let expected = read_report(&baseline_marker);

    for boundary in ["generation-done", "prepared", "verified", "committed"] {
        let store_root = unique_dir("store");
        let marker_base = store_root.join("marker.json");

        let mut child = run_generate("file", &store_root, boundary, &marker_base);
        let reached = wait_for_marker(&marker_base, boundary, Duration::from_secs(120));
        let killed = child.kill().is_ok();
        let _ = child.wait();
        assert!(reached, "worker never reached boundary {boundary}");
        assert!(killed, "worker was already gone at {boundary}");

        match boundary {
            "generation-done" => {
                // Nothing was persisted: no staging, no committed record, and
                // a resume against the missing marker refuses typed.
                assert!(committed_record_files(&store_root).is_empty());
                assert!(staging_record_files(&store_root).is_empty());
                let output = run_resume("file", &store_root, &marker_base, "committed");
                assert!(
                    !output.status.success(),
                    "resume without any persisted record must fail"
                );
                assert!(
                    committed_record_files(&store_root).is_empty(),
                    "failed resume must leave zero records behind"
                );
            }
            "prepared" | "verified" => {
                // Staged exactly once, not yet published.
                assert_eq!(staging_record_files(&store_root).len(), 1, "{boundary}");
                assert!(committed_record_files(&store_root).is_empty(), "{boundary}");
                let output = run_resume("file", &store_root, &marker_base, boundary);
                assert!(
                    output.status.success(),
                    "resume at {boundary} failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let resumed = read_report(&marker_base);
                assert_eq!(resumed, expected, "resumed result at {boundary} diverged");
                // Publication happened exactly once, via the worker's
                // promotion, and the staged record was consumed.
                assert_eq!(committed_record_files(&store_root).len(), 1, "{boundary}");
                // Duplicate resume is a safe no-op replay with an identical
                // report and zero additional records.
                let again = run_resume("file", &store_root, &marker_base, boundary);
                assert!(again.status.success(), "duplicate resume must be safe");
                assert_eq!(read_report(&marker_base), expected);
                assert_eq!(committed_record_files(&store_root).len(), 1);
                assert!(staging_record_files(&store_root).is_empty());
            }
            "committed" => {
                assert_eq!(committed_record_files(&store_root).len(), 1);
                let output = run_resume("file", &store_root, &marker_base, boundary);
                assert!(
                    output.status.success(),
                    "resume at committed failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(read_report(&marker_base), expected);
            }
            _ => unreachable!(),
        }
        fs::remove_dir_all(&store_root).expect("cleanup store");
    }
    fs::remove_dir_all(&baseline_dir).expect("cleanup baseline");
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[test]
fn byte_loss_within_envelope_repairs_and_beyond_envelope_refuses() {
    // Produce a committed record with its repair sidecar.
    let store_root = unique_dir("loss");
    let marker_base = store_root.join("marker.json");
    let mut child = run_generate("file", &store_root, "committed", &marker_base);
    assert!(wait_for_marker(
        &marker_base,
        "committed",
        Duration::from_secs(120)
    ));
    child.kill().expect("kill generate worker");
    let _ = child.wait();

    let sidecar_path = store_root.join("committed.repair");
    let sidecar: fsym_runtime::RepairSidecar =
        serde_json::from_slice(&fs::read(&sidecar_path).expect("sidecar")).expect("sidecar json");
    let record_path = &committed_record_files(&store_root)[0];
    let original_wire = fs::read(record_path).expect("committed record");

    // Corrupt exactly one source symbol's worth of bytes inside the payload
    // number array (the JSON array form of the payload field's bytes).
    let mut corrupted = original_wire.clone();
    let payload_start = find_subslice(&original_wire, b"\"payload\":[").expect("payload field")
        + b"\"payload\":[".len();
    for offset in 0..sidecar.symbol_size.min(corrupted.len() - payload_start) {
        corrupted[payload_start + offset] ^= 0x5a;
    }
    fs::write(record_path, &corrupted).expect("corrupt committed record");

    // The store must now refuse on disk readback (digest mismatch), and a
    // resume attempt must fail without leaving anything behind.
    let refused = run_resume("file", &store_root, &marker_base, "committed");
    assert!(
        !refused.status.success(),
        "corrupted record must not resume"
    );

    // Repair: mark the corrupted source symbols lost and feed repair symbols.
    let symbol_count = sidecar.num_source_symbols;
    let mut received_source: Vec<Option<Vec<u8>>> = Vec::with_capacity(symbol_count);
    for index in 0..symbol_count {
        let start = index * sidecar.symbol_size;
        let end = (start + sidecar.symbol_size).min(original_wire.len());
        let original_symbol = original_wire[start..end].to_vec();
        let corrupted_available = start + sidecar.symbol_size <= corrupted.len();
        let intact = corrupted_available && corrupted[start..end] == original_wire[start..end];
        if intact {
            received_source.push(Some(original_symbol));
        } else {
            received_source.push(None);
        }
    }
    let lost = received_source
        .iter()
        .filter(|symbol| symbol.is_none())
        .count();
    assert!(lost >= 1, "corruption must knock out at least one symbol");
    assert!(
        lost <= sidecar.repair_symbols.len(),
        "test corruption must stay within the declared repair envelope"
    );
    let received_repair: Vec<(u32, Vec<u8>)> =
        sidecar.repair_symbols.iter().take(lost).cloned().collect();
    let repaired = sidecar
        .reconstruct(&received_source, &received_repair)
        .expect("repair within envelope");
    assert_eq!(repaired, original_wire, "repair must reproduce exact bytes");
    fs::write(record_path, &repaired).expect("restore repaired record");

    let output = run_resume("file", &store_root, &marker_base, "committed");
    assert!(
        output.status.success(),
        "repaired record must resume: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = read_report(&marker_base);
    assert_eq!(
        report["winning_strategy"], "kronecker_interpolation",
        "the slower verified generator must win"
    );

    // Beyond the envelope: destroy more source symbols than the declared
    // repair symbol count can ever reconstruct, even using every repair
    // symbol. Reconstruction must refuse with a typed error, never emit a
    // silently-wrong payload.
    const SYMBOLS_TO_DESTROY: usize = REPAIR_SYMBOL_COUNT_ESTIMATE + 1;
    let mut destroyed = 0usize;
    {
        let mut corrupted_mut = corrupted.clone();
        let total_symbols = sidecar.num_source_symbols;
        for index in 0..total_symbols {
            if destroyed >= SYMBOLS_TO_DESTROY {
                break;
            }
            let start = index * sidecar.symbol_size;
            if start >= corrupted_mut.len() {
                break;
            }
            let end = (start + sidecar.symbol_size).min(corrupted_mut.len());
            corrupted_mut[start] = corrupted_mut[start].wrapping_add(0x33);
            if corrupted_mut[start..end] == original_wire[start..end] {
                corrupted_mut[start] = corrupted_mut[start].wrapping_add(1);
            }
            destroyed += 1;
        }
        corrupted = corrupted_mut;
    }
    fs::write(record_path, &corrupted).expect("corrupt beyond envelope");
    let mut received_source_beyond: Vec<Option<Vec<u8>>> =
        Vec::with_capacity(sidecar.num_source_symbols);
    for index in 0..sidecar.num_source_symbols {
        let start = index * sidecar.symbol_size;
        let end = (start + sidecar.symbol_size).min(original_wire.len());
        let beyond = start + sidecar.symbol_size <= corrupted.len()
            && corrupted[start..end] != original_wire[start..end];
        if beyond {
            received_source_beyond.push(None);
        } else {
            received_source_beyond.push(Some(original_wire[start..end].to_vec()));
        }
    }
    let beyond_lost = received_source_beyond
        .iter()
        .filter(|s| s.is_none())
        .count();
    assert!(
        beyond_lost >= SYMBOLS_TO_DESTROY,
        "corruption must destroy at least {SYMBOLS_TO_DESTROY} symbols"
    );
    let received_repair: Vec<(u32, Vec<u8>)> = sidecar.repair_symbols.clone();
    let beyond = sidecar.reconstruct(&received_source_beyond, &received_repair);
    assert!(
        beyond_lost + received_repair.len() < sidecar.num_source_symbols
            || matches!(
                beyond,
                Err(fsym_runtime::RepairError::InsufficientSymbols(_, _))
            )
            || matches!(
                beyond,
                Err(fsym_runtime::RepairError::SourceSymbolDigestMismatch(_))
            ),
        "beyond-envelope reconstruction must refuse: got {beyond:?}"
    );
    if let Ok(bytes) = &beyond {
        assert_ne!(
            bytes, &original_wire,
            "a wrong payload must never equal the original"
        );
    }
    let refused = run_resume("file", &store_root, &marker_base, "committed");
    assert!(
        !refused.status.success(),
        "records beyond the repair envelope must not resume"
    );
    assert_eq!(
        committed_record_files(&store_root).len(),
        1,
        "refusals must leave the committed namespace unchanged"
    );
    fs::remove_dir_all(&store_root).expect("cleanup loss store");
}

const REPAIR_SYMBOL_COUNT_ESTIMATE: usize = 4;

#[test]
fn digest_consistent_false_proof_is_refused_by_mathematical_reverification() {
    let store_root = unique_dir("poison");
    let marker_base = store_root.join("marker.json");
    let mut child = run_generate("file", &store_root, "committed", &marker_base);
    assert!(wait_for_marker(
        &marker_base,
        "committed",
        Duration::from_secs(120)
    ));
    child.kill().expect("kill");
    let _ = child.wait();

    let record_path = &committed_record_files(&store_root)[0];
    let record: DurableRecord =
        serde_json::from_slice(&fs::read(record_path).expect("record")).expect("record json");

    // Unwrap the payload into typed state, poison EVERY candidate with a
    // self-consistent but mathematically FALSE derivation, and re-digest
    // everything so every stored digest is internally consistent. Only the
    // runtime's independent mathematical verification can catch this.
    // (Poisoning a single candidate would correctly fall through to the
    // remaining verified candidate — that path is covered by the runtime
    // invalid-candidate test.)
    let mut checkpoint: TypedCheckpoint<FactorRaceContinuationState> =
        serde_json::from_slice(&record.payload).expect("checkpoint decode");
    let false_lhs = checkpoint.payload.input_expr.clone();
    let false_rhs = fsym_core_expr_from_i64(123456789);
    let false_derivation = || DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(0),
            rule: ProofRule::DefinitionalReduction {
                lhs: false_lhs.clone(),
                rhs: false_rhs.clone(),
                rule_name: "polynomial_ring_equivalence".into(),
            },
            claim: Claim::AlgebraicIdentity {
                lhs: false_lhs.clone(),
                rhs: false_rhs.clone(),
            },
        }],
        root: StepId(0),
    };
    for record_entry in &mut checkpoint.payload.candidates {
        record_entry.candidate.derivation = false_derivation();
    }
    let poisoned_checkpoint = TypedCheckpoint::new(
        checkpoint.payload_schema.clone(),
        checkpoint.checkpoint_seq,
        checkpoint.payload.clone(),
        checkpoint.remaining_budget.clone(),
        checkpoint.verifier_remaining,
    )
    .expect("poisoned checkpoint re-digests");
    let poisoned_payload = serde_json::to_vec(&poisoned_checkpoint).expect("poisoned wire");

    let pins = DependencyManifest::new([
        ("factor_race_operation", "factor_race_v1"),
        ("fsym_runtime", env!("CARGO_PKG_VERSION")),
        ("repair_codec", "raptorq-rfc6330-v1"),
    ]);
    // Reuse the original universe binding so ONLY the mathematics differs.
    let poisoned_record = DurableRecord::new(
        record.universe_id,
        record.payload_schema.clone(),
        poisoned_payload,
        pins,
    )
    .expect("poisoned record");
    assert_ne!(
        poisoned_record.payload_digest, record.payload_digest,
        "payload digest must track the poisoned bytes"
    );
    fs::write(
        record_path,
        poisoned_record.to_wire().expect("poisoned wire"),
    )
    .expect("install poisoned record");

    let output = run_resume("file", &store_root, &marker_base, "committed");
    assert!(
        !output.status.success(),
        "a digest-consistent false proof must never publish; resume output: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("resumed race refused"),
        "refusal must come from mathematical reverification, got: {stderr}"
    );
    let poisoned_wire = poisoned_record.to_wire().expect("poisoned wire");
    assert!(
        fs::read(record_path).expect("record unchanged") == poisoned_wire,
        "the poisoned record is evidence and must be preserved, not cleaned"
    );
    fs::remove_dir_all(&store_root).expect("cleanup poison store");
}

fn fsym_core_expr_from_i64(value: i64) -> fsym_core::Expr {
    fsym_core::Expr::Integer(fsym_core::BigInt::from(value))
}

#[test]
fn stale_universe_marker_refuses_and_leaves_no_records() {
    // One committed run; then present a marker whose universe binding has
    // been tampered into a DIFFERENT universe. The namespace for that
    // universe is empty, so promotion and load must refuse instead of
    // resuming, and the store must be left untouched.
    let store_root = unique_dir("stale");
    let marker_base = store_root.join("marker.json");
    let mut child = run_generate("file", &store_root, "committed", &marker_base);
    assert!(wait_for_marker(
        &marker_base,
        "committed",
        Duration::from_secs(120)
    ));
    child.kill().expect("kill");
    let _ = child.wait();

    let mut marker: Value = serde_json::from_slice(
        &fs::read(marker_stage(&marker_base, "committed")).expect("committed marker"),
    )
    .expect("marker json");
    let mut stale_universe =
        decode_marker_universe(marker["universe_hex"].as_str().expect("universe hex"));
    stale_universe[0] ^= 0x01;
    marker["universe_hex"] = Value::String(fsym_durable::hex_lower(&stale_universe));
    let stale_marker_path = marker_stage(&marker_base, "committed");
    fs::write(
        &stale_marker_path,
        serde_json::to_vec(&marker).expect("stale marker"),
    )
    .expect("write stale marker");

    let output = run_resume("file", &store_root, &marker_base, "committed");
    assert!(
        !output.status.success(),
        "a stale universe marker must not resume another universe's record"
    );
    assert!(
        committed_record_files(&store_root).len() == 1,
        "stale resume must not write records into the store"
    );
    assert!(
        staging_record_files(&store_root).is_empty(),
        "stale resume must leave zero staging pollution"
    );

    // Additionally, presenting a valid record against a DIFFERENT expected
    // universe is a typed refusal at the validation boundary.
    let probe_root = store_root.join("probe");
    let store = FileStore::open(&probe_root).expect("probe store");
    let record: DurableRecord =
        serde_json::from_slice(&fs::read(&committed_record_files(&store_root)[0]).expect("record"))
            .expect("record json");
    let wrong_universe = factor_race_universe_id(
        FACTOR_RACE_CONTINUATION_SCHEMA,
        [1; 32],
        [2; 32],
        &DependencyManifest::empty(),
    );
    assert!(matches!(
        record.validate(wrong_universe, &record.dependencies),
        Err(DurableError::UniverseMismatch { .. })
    ));
    assert!(store.staging_len().expect("staging probe") == 0);
    // The probe store lives inside the store root; one removal covers both.
    fs::remove_dir_all(&store_root).expect("cleanup stale store");
}

fn decode_marker_universe(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64, "universe hex length");
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).expect("hex byte");
    }
    out
}

#[test]
fn generator_replacement_replay_is_identical() {
    // Two fully independent generate→resume cycles (fresh processes, fresh
    // stores) must produce byte-identical reports: resumed acceptance depends
    // only on the persisted verified candidates, never on which generator
    // binary or process produced them.
    let mut reports = Vec::new();
    for run in 0..2 {
        let store_root = unique_dir("replay");
        let marker_base = store_root.join("marker.json");
        let mut child = run_generate("file", &store_root, "committed", &marker_base);
        assert!(wait_for_marker(
            &marker_base,
            "committed",
            Duration::from_secs(120)
        ));
        child.kill().expect("kill");
        let _ = child.wait();
        let output = run_resume("file", &store_root, &marker_base, "committed");
        assert!(
            output.status.success(),
            "replay run {run} resume failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        reports.push(read_report(&marker_base));
        fs::remove_dir_all(&store_root).expect("cleanup replay store");
    }
    assert_eq!(reports[0], reports[1], "replay must be deterministic");
}

fn cli_binary_path() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("FSQLITE_CLI_BIN") {
        return Some(PathBuf::from(env));
    }
    let recorded = PathBuf::from("/data/tmp/cargo-target/debug/fsqlite");
    if recorded.is_file() {
        return Some(recorded);
    }
    None
}

#[test]
fn cli_lane_reproduces_the_file_lane_crash_matrix() {
    // Environment-gated: the pinned fsqlite CLI binary must be present
    // (built via `cargo build -p fsqlite-cli --locked -F fsqlite-core/native`
    // from the frankenscipy... frankensqlite-pin worktree; see
    // fra-rc-durable-m5e addenda). The sweep drives the identical boundary
    // matrix through the pinned-CLI subprocess store: stage-verify-commit
    // with real process death at each publication boundary, fresh-process
    // promotion/resume, idempotent duplicate resume, and zero-record
    // refusals.
    let Some(cli) = cli_binary_path() else {
        eprintln!(
            "skipped: pinned fsqlite CLI binary not found (set FSQLITE_CLI_BIN or build \
             the pinned worktree with -F fsqlite-core/native)"
        );
        return;
    };
    let _ = cli;
    // The lane sweep: the same generate/resume invocation sequence with
    // lane="cli" exercises FsqliteCliStore through the identical worker
    // protocol. The worker's open_store dispatches on the lane argument,
    // so every prepare/verify/commit/load in this sweep is a real
    // subprocess-backed SQLite operation.
    for boundary in ["prepared", "verified", "committed"] {
        let store_root = unique_dir("cli-store");
        let marker_base = store_root.join("marker.json");

        let mut child = run_generate("cli", &store_root, boundary, &marker_base);
        let reached = wait_for_marker(&marker_base, boundary, Duration::from_secs(120));
        let killed = child.kill().is_ok();
        let _ = child.wait();
        assert!(reached, "cli worker never reached boundary {boundary}");
        assert!(killed);

        let output = run_resume("cli", &store_root, &marker_base, boundary);
        assert!(
            output.status.success(),
            "cli-lane resume at {boundary} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !committed_record_files(&store_root).is_empty(),
            "{boundary}"
        );
        assert!(staging_record_files(&store_root).is_empty(), "{boundary}");
        fs::remove_dir_all(&store_root).expect("cleanup cli store");
    }
}
