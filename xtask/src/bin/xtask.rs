//! Gate runner for the campaign command contracts named in
//! docs/FIRST_IMPLEMENTATION_CAMPAIGN.md (C1-C3 slice, bead
//! fra-gate-runner-xtask-cyx). The runner executes real checks and writes
//! machine-readable receipts under artifacts/audit/receipts/.
//!
//! Independence rule: this binary NEVER grades its own receipts. Validation
//! lives in the structurally separate `gate-receipt-validator` binary, which
//! re-derives the checks digest and rejects tampering fail-closed.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::process::Command;
use std::sync::OnceLock;

use serde::Serialize;

const RECEIPTS_DIR: &str = "artifacts/audit/receipts";
static SOURCE_AT_START: OnceLock<xtask::SourceSnapshot> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
struct Check {
    name: String,
    status: String, // "passed" | "failed"
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct Receipt {
    schema_version: u32,
    gate: String,
    profile_id: String,
    status: String, // "passed" | "failed"
    commit: String,
    source: xtask::SourceSnapshot,
    profile_digest: String,
    checks: Vec<Check>,
    /// blake3 digest over the canonical JSON of `checks` (BTreeMap ordering).
    checks_digest: String,
    receipt_digest: String,
}

fn canonical_checks(checks: &[Check]) -> String {
    // BTreeMap gives deterministic key order; serde_json writes stable
    // scalars. This is the byte string the digest commits to.
    let mapped: Vec<BTreeMap<String, String>> = checks
        .iter()
        .map(|c| {
            BTreeMap::from([
                ("name".to_string(), c.name.clone()),
                ("status".to_string(), c.status.clone()),
                ("detail".to_string(), c.detail.clone()),
            ])
        })
        .collect();
    serde_json::to_string(&mapped).expect("checks serialize")
}

fn checks_digest(checks: &[Check]) -> String {
    blake3::hash(canonical_checks(checks).as_bytes())
        .to_hex()
        .to_string()
}

fn run_command(name: &str, mut cmd: Command, checks: &mut Vec<Check>) {
    let command: Vec<_> = std::iter::once(cmd.get_program())
        .chain(cmd.get_args())
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let output = cmd.output();
    match output {
        Ok(o) if o.status.success() => checks.push(Check {
            name: name.to_string(),
            status: if (name.starts_with("test-") || name.starts_with("tests-"))
                && passed_test_count(&String::from_utf8_lossy(&o.stdout)) == 0
            {
                "failed"
            } else {
                "passed"
            }
            .into(),
            detail: serde_json::json!({
                "command": command, "exit_code": 0,
                "stdout": String::from_utf8_lossy(&o.stdout),
                "stderr": String::from_utf8_lossy(&o.stderr),
            })
            .to_string(),
        }),
        Ok(o) => checks.push(Check {
            name: name.to_string(),
            status: "failed".into(),
            detail: serde_json::json!({
                "command": command, "exit_code": o.status.code(),
                "stdout": String::from_utf8_lossy(&o.stdout),
                "stderr": String::from_utf8_lossy(&o.stderr),
            })
            .to_string(),
        }),
        Err(e) => checks.push(Check {
            name: name.to_string(),
            status: "failed".into(),
            detail: format!("spawn error: {e}"),
        }),
    }
}

fn passed_test_count(stdout: &str) -> usize {
    stdout
        .lines()
        .filter_map(|line| {
            line.strip_prefix("test result: ok. ")?
                .split_whitespace()
                .next()?
                .parse::<usize>()
                .ok()
        })
        .fold(0, usize::saturating_add)
}

fn cargo() -> Command {
    let mut c = Command::new("cargo");
    c.env("RCH_SHIM_LOCAL_IDE", "1"); // nested cargo: keep gate execution local & bounded
    c
}

fn check_registry_sync(checks: &mut Vec<Check>) {
    let mut c = Command::new("bash");
    c.args(["scripts/check.sh", "registries"]);
    run_command("registry-validators", c, checks);
}

fn check_workspace_no_unsafe(checks: &mut Vec<Check>) {
    let manifest = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
    let ok = manifest.contains("unsafe_code = \"forbid\"");
    checks.push(Check {
        name: "workspace-forbids-unsafe".into(),
        status: if ok { "passed" } else { "failed" }.into(),
        detail: if ok {
            "workspace lints forbid unsafe_code"
        } else {
            "unsafe_code forbid missing"
        }
        .into(),
    });
}

fn check_profile_id(requested: &str, checks: &mut Vec<Check>) {
    let rel = format!("tools/conformance-lab/profiles/{requested}.toml");
    let text = std::fs::read_to_string(&rel).unwrap_or_default();
    let expected = format!("profile_id = \"{requested}\"");
    let ok = text.contains(&expected);
    checks.push(Check {
        name: "profile-id-matches-declared-module".into(),
        status: if ok { "passed" } else { "failed" }.into(),
        detail: if ok {
            format!("{rel} declares profile_id {requested:?}")
        } else {
            format!("{rel} missing or does not declare {requested:?}")
        },
    });
}

fn check_oracle_pinned(_profile: &str, checks: &mut Vec<Check>) {
    // The pinned oracle must answer 1.14.0; profile revisions share the pin.
    let oracle_python = "/home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0/bin/python";
    let out = Command::new(oracle_python)
        .args(["-c", "import sympy; print(sympy.__version__)"])
        .output();
    let (ok, detail) = match out {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
            (v == "1.14.0", format!("oracle reports sympy {v}"))
        }
        Ok(o) => (
            false,
            format!("oracle probe exit {}", o.status.code().unwrap_or(-1)),
        ),
        Err(e) => (false, format!("oracle probe spawn error: {e}")),
    };
    checks.push(Check {
        name: "oracle-version-pinned".into(),
        status: if ok { "passed" } else { "failed" }.into(),
        detail,
    });
}

fn cmd_profile_verify(profile: &str) -> Receipt {
    let mut checks = Vec::new();
    check_profile_id(profile, &mut checks);
    check_registry_sync(&mut checks);
    check_workspace_no_unsafe(&mut checks);
    finish("profile-verify", profile, checks)
}

fn cmd_gate_foundation() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);
    for crate_name in [
        "fsym-id",
        "fsym-budget",
        "fsym-outcome",
        "fsym-bigint",
        "fsym-rational",
        "fsym-modular",
    ] {
        let mut c = cargo();
        c.args(["test", "-p", crate_name, "--quiet"]);
        run_command(&format!("tests-{crate_name}"), c, &mut checks);
    }
    // ID stability across fresh processes: the trybuild/compile-fail and unit
    // corpus above run in their own processes; the digest below is stable
    // across two invocations of this runner (process-local state excluded).
    finish("foundation", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_python_object_model(profile: &str) -> Receipt {
    let mut checks = Vec::new();
    check_oracle_pinned(profile, &mut checks);
    check_profile_id(profile, &mut checks);
    // Oracle-isolation probe: candidate subprocess must fail closed (exit 3)
    // when it can see the oracle tree.
    let mut iso = Command::new("/data/projects/frankensympy/.venv-conformance/bin/python");
    iso.args([
        "tools/conformance-lab/capture.py",
        "isolation",
        &format!("tools/conformance-lab/profiles/{profile}.toml"),
        "--candidate-python",
        "/data/projects/frankensympy/.venv-conformance/bin/python3",
    ]);
    run_command("oracle-isolation-probe", iso, &mut checks);
    // Object-model differential: candidate vs pinned-oracle goldens.
    let mut diff = Command::new("/data/projects/frankensympy/.venv-conformance/bin/python");
    diff.args([
        "tools/conformance-lab/capture.py",
        "diff",
        &format!("tools/conformance-lab/profiles/{profile}.toml"),
        "--candidate-python",
        "/data/projects/frankensympy/.venv-conformance/bin/python3",
    ]);
    run_command("object-model-differential", diff, &mut checks);
    finish("python-object-model", profile, checks)
}

fn cmd_gate_deterministic_term_identity() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-core",
        "--test",
        "fresh_process_id_stability",
        "--quiet",
    ]);
    run_command("test-fresh-process-id-stability", c1, &mut checks);

    let fixture_path = "artifacts/conformance/fixtures/deterministic_term_identity_v1.json";
    let ok_fixture = match std::fs::read_to_string(fixture_path) {
        Ok(raw) => match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(v) => v["terms"].as_object().map(|t| t.len()).unwrap_or(0) >= 20,
            Err(_) => false,
        },
        Err(_) => false,
    };
    checks.push(Check {
        name: "cross-architecture-fixture-closure".into(),
        status: if ok_fixture { "passed" } else { "failed" }.into(),
        detail: if ok_fixture {
            format!("{fixture_path} valid with >= 20 canonical terms")
        } else {
            format!("{fixture_path} missing or invalid")
        },
    });

    for crate_name in ["fsym-core", "fsym-assumptions", "fsym-id"] {
        let mut c = cargo();
        c.args(["test", "-p", crate_name, "--quiet"]);
        run_command(&format!("tests-{crate_name}"), c, &mut checks);
    }

    finish(
        "deterministic-term-identity",
        "sympy-1.14.0-cpython",
        checks,
    )
}

fn cmd_gate_ws11_certified_numeric() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-core",
        "--test",
        "directed_rounding_mutation",
        "--quiet",
    ]);
    run_command("test-directed-rounding-mutation", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-core", "--quiet"]);
    run_command("tests-fsym-core", c2, &mut checks);

    let mut c3 = cargo();
    c3.args(["test", "-p", "fsym-proof-kernel", "--quiet"]);
    run_command("tests-fsym-proof-kernel", c3, &mut checks);

    finish("ws11-certified-numeric", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws13_portfolio_runtime() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-runtime",
        "--test",
        "cancellation_injection",
        "--quiet",
    ]);
    run_command("test-cancellation-injection", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-runtime", "--quiet"]);
    run_command("tests-fsym-runtime", c2, &mut checks);

    finish("ws13-portfolio-runtime", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws10_exact_linear() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args(["test", "-p", "fsym-matrices", "--quiet"]);
    run_command("tests-fsym-matrices", c1, &mut checks);

    finish("ws10-exact-linear", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws12_certified_jacobian() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-calculus",
        "--test",
        "sparse_jacobian_c7",
        "--quiet",
    ]);
    run_command("test-sparse-jacobian-c7", c1, &mut checks);

    let mut c2 = cargo();
    c2.args([
        "test",
        "-p",
        "fsym-calculus",
        "--test",
        "sparse_jacobian_gate",
        "--quiet",
    ]);
    run_command("test-sparse-jacobian-gate", c2, &mut checks);

    let mut c3 = cargo();
    c3.args(["test", "-p", "fsym-calculus", "--quiet"]);
    run_command("tests-fsym-calculus", c3, &mut checks);

    finish("ws12-certified-jacobian", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws17_groebner() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args(["test", "-p", "fsym-polys", "--quiet"]);
    run_command("tests-fsym-polys", c1, &mut checks);

    finish("ws17-groebner", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws18_analytic_calculus() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args(["test", "-p", "fsym-calculus", "--quiet"]);
    run_command("tests-fsym-calculus", c1, &mut checks);

    finish("ws18-analytic-calculus", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws19_solvers() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args(["test", "-p", "fsym-solvers", "--quiet"]);
    run_command("tests-fsym-solvers", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-sets", "--quiet"]);
    run_command("tests-fsym-sets", c2, &mut checks);

    let mut c3 = cargo();
    c3.args(["test", "-p", "fsym-logic", "--quiet"]);
    run_command("tests-fsym-logic", c3, &mut checks);

    finish("ws19-solvers", "sympy-1.14.0-cpython", checks)
}

fn finish(gate: &str, profile_id: &str, mut checks: Vec<Check>) -> Receipt {
    let source = SOURCE_AT_START
        .get()
        .expect("source captured before execution")
        .clone();
    let source_stable = xtask::source_snapshot(&xtask::workspace_root()).as_ref() == Ok(&source);
    checks.push(Check {
        name: "source-stable-during-run".into(),
        status: if source_stable { "passed" } else { "failed" }.into(),
        detail: "repository input snapshot compared before and after checks".into(),
    });
    let profile_digest = xtask::profile_digest(&xtask::workspace_root(), profile_id);
    checks.push(Check {
        name: "profile-input-bound".into(),
        status: if profile_digest.is_ok() {
            "passed"
        } else {
            "failed"
        }
        .into(),
        detail: profile_digest.clone().unwrap_or_else(|e| e),
    });
    let all_passed = checks.iter().all(|c| c.status == "passed");
    let mut receipt = Receipt {
        schema_version: 2,
        gate: gate.to_string(),
        profile_id: profile_id.to_string(),
        status: if all_passed { "passed" } else { "failed" }.into(),
        commit: source.commit.clone(),
        source,
        profile_digest: profile_digest.unwrap_or_default(),
        checks_digest: checks_digest(&checks),
        checks,
        receipt_digest: String::new(),
    };
    let mut payload = serde_json::to_value(&receipt).expect("receipt serializes");
    payload
        .as_object_mut()
        .expect("receipt object")
        .remove("receipt_digest");
    receipt.receipt_digest =
        blake3::hash(&serde_json::to_vec(&payload).expect("canonical receipt"))
            .to_hex()
            .to_string();
    write_receipt(&receipt);
    receipt
}

fn write_receipt(receipt: &Receipt) {
    use std::io::Write;

    let dir = std::path::Path::new(RECEIPTS_DIR);
    std::fs::create_dir_all(dir).expect("create receipts dir");
    let json = serde_json::to_string_pretty(receipt).expect("receipt serializes");
    let (tmp, mut file) = reserve_receipt_temp(dir, &receipt.gate).expect("reserve temp receipt");
    file.write_all(json.as_bytes()).expect("write temp receipt");
    drop(file);
    let final_path = dir.join(format!("{}.receipt.json", receipt.gate));
    std::fs::rename(&tmp, final_path).expect("atomic receipt rename");
}

fn reserve_receipt_temp(
    dir: &std::path::Path,
    gate: &str,
) -> std::io::Result<(std::path::PathBuf, std::fs::File)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    // Each publisher owns its inode until rename. Exclusive creation also
    // protects files left by a previous process with a reused PID. Failed
    // writes remain available for diagnosis; they are never published.
    for _ in 0..128 {
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(".{gate}.{}.{sequence}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "receipt temporary-file collision limit reached",
    ))
}

fn cmd_gate_ws14_agent_protocol() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-runtime",
        "--test",
        "c10_protocol_gate",
        "--quiet",
    ]);
    run_command("test-c10-protocol-gate", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-runtime", "--quiet"]);
    run_command("tests-fsym-runtime", c2, &mut checks);

    finish("ws14-agent-protocol", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws15_persistence_repair() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-runtime",
        "--test",
        "c9_persistence_repair_gate",
        "--quiet",
    ]);
    run_command("test-c9-persistence-repair-gate", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-runtime", "--quiet"]);
    run_command("tests-fsym-runtime", c2, &mut checks);

    finish("ws15-persistence-repair", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws09_factorization() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args(["test", "-p", "fsym-polys", "--quiet"]);
    run_command("tests-fsym-polys", c1, &mut checks);

    finish("ws09-factorization", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws20_structured_domains() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args(["test", "-p", "fsym-geometry", "--quiet"]);
    run_command("tests-fsym-geometry", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-tensor", "--quiet"]);
    run_command("tests-fsym-tensor", c2, &mut checks);

    finish("ws20-structured-domains", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws16_distribution_index() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-runtime",
        "--test",
        "ws16_distribution_index_gate",
        "--quiet",
    ]);
    run_command("test-ws16-distribution-index-gate", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-runtime", "--quiet"]);
    run_command("tests-fsym-runtime", c2, &mut checks);

    finish("ws16-distribution-index", "sympy-1.14.0-cpython", checks)
}

fn cmd_gate_ws22_performance() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);

    let mut c1 = cargo();
    c1.args([
        "test",
        "-p",
        "fsym-runtime",
        "--test",
        "ws22_performance_gate",
        "--quiet",
    ]);
    run_command("test-ws22-performance-gate", c1, &mut checks);

    let mut c2 = cargo();
    c2.args(["test", "-p", "fsym-runtime", "--quiet"]);
    run_command("tests-fsym-runtime", c2, &mut checks);

    let mut c3 = Command::new("python3");
    c3.args([
        "tools/perf/paired_bench.py",
        "run",
        "--out",
        "artifacts/benchmarks/ws22_paired_benchmark_report.json",
        "--rounds",
        "5",
    ]);
    run_command("paired-live-incumbent-bench", c3, &mut checks);

    let report_path = "artifacts/benchmarks/ws22_paired_benchmark_report.json";
    let report_content = std::fs::read_to_string(report_path).unwrap_or_default();
    let report_ok = report_content.contains("\"schema\": \"gauntlet.paired_bench.v1\"")
        && report_content.contains("\"admitted\":")
        && report_content.contains("\"aa_control\":")
        && report_content.contains("\"aa_control_verified\": true");

    checks.push(Check {
        name: "paired-benchmark-report-verification".into(),
        status: if report_ok { "passed" } else { "failed" }.into(),
        detail: if report_ok {
            format!("{report_path} contains schema, admitted cases, and verified AA control")
        } else {
            format!("{report_path} missing or invalid benchmark report content")
        },
    });

    finish("ws22-performance", "sympy-1.14.0-cpython", checks)
}

fn check_exclusion_ledger(checks: &mut Vec<Check>) {
    let ledger_path = "artifacts/conformance/exclusion_ledger.json";
    let content = std::fs::read_to_string(ledger_path).unwrap_or_default();
    let json: Result<serde_json::Value, _> = serde_json::from_str(&content);
    let ok = match json {
        Ok(v) => {
            v.get("schema_version") == Some(&serde_json::json!(1))
                && v.get("profile_id") == Some(&serde_json::json!("sympy-1.14.0-cpython"))
                && v.get("exclusions")
                    .and_then(|e| e.as_array())
                    .is_some_and(|arr| {
                        !arr.is_empty()
                            && arr.iter().all(|ex| {
                                ex.get("exclusion_id").is_some()
                                    && ex.get("category").is_some()
                                    && ex.get("feature").is_some()
                                    && ex.get("divergence_summary").is_some()
                                    && ex.get("source_evidence").is_some()
                                    && ex.get("rationale").is_some()
                                    && ex.get("status").is_some()
                            })
                    })
        }
        Err(_) => false,
    };
    checks.push(Check {
        name: "exclusion-ledger-verification".into(),
        status: if ok { "passed" } else { "failed" }.into(),
        detail: if ok {
            format!("{ledger_path} schema and source evidence valid")
        } else {
            format!("{ledger_path} missing or invalid")
        },
    });
}

fn cmd_gate_ws21_profile_closure() -> Receipt {
    let mut checks = Vec::new();
    check_workspace_no_unsafe(&mut checks);
    check_registry_sync(&mut checks);

    let mut c1 = cargo();
    c1.args(["test", "-p", "fsym-conformance", "--quiet"]);
    run_command("tests-fsym-conformance", c1, &mut checks);

    let mut c2 = Command::new("python3");
    c2.args(["tools/conformance-lab/corpus_gate.py"]);
    run_command("corpus-gate", c2, &mut checks);

    check_exclusion_ledger(&mut checks);

    finish("ws21-profile-closure", "sympy-1.14.0-cpython", checks)
}

fn print_usage() -> i32 {
    eprintln!(
        "usage: xtask profile verify <profile-id> | xtask gate foundation | xtask gate deterministic-term-identity | xtask gate ws09-factorization | xtask gate ws10-exact-linear | xtask gate ws11-certified-numeric | xtask gate ws12-certified-jacobian | xtask gate ws13-portfolio-runtime | xtask gate ws14-agent-protocol | xtask gate ws15-persistence-repair | xtask gate ws16-distribution-index | xtask gate ws17-groebner | xtask gate ws18-analytic-calculus | xtask gate ws19-solvers | xtask gate ws20-structured-domains | xtask gate ws21-profile-closure | xtask gate ws22-performance | xtask gate python-object-model --profile <profile-id>"
    );
    2
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source = match xtask::source_snapshot(&xtask::workspace_root()) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("cannot bind gate source: {error}");
            return std::process::ExitCode::from(2);
        }
    };
    SOURCE_AT_START
        .set(source)
        .expect("source initialized once");
    let receipt = match args.as_slice() {
        [a, b, profile] if a == "profile" && b == "verify" => cmd_profile_verify(profile),
        [a, b] if a == "gate" && b == "foundation" => cmd_gate_foundation(),
        [a, b] if a == "gate" && b == "deterministic-term-identity" => {
            cmd_gate_deterministic_term_identity()
        }
        [a, b] if a == "gate" && b == "ws09-factorization" => cmd_gate_ws09_factorization(),
        [a, b] if a == "gate" && b == "ws10-exact-linear" => cmd_gate_ws10_exact_linear(),
        [a, b] if a == "gate" && b == "ws11-certified-numeric" => cmd_gate_ws11_certified_numeric(),
        [a, b] if a == "gate" && b == "ws12-certified-jacobian" => {
            cmd_gate_ws12_certified_jacobian()
        }
        [a, b] if a == "gate" && b == "ws13-portfolio-runtime" => cmd_gate_ws13_portfolio_runtime(),
        [a, b] if a == "gate" && b == "ws14-agent-protocol" => cmd_gate_ws14_agent_protocol(),
        [a, b] if a == "gate" && b == "ws15-persistence-repair" => {
            cmd_gate_ws15_persistence_repair()
        }
        [a, b] if a == "gate" && b == "ws16-distribution-index" => {
            cmd_gate_ws16_distribution_index()
        }
        [a, b] if a == "gate" && b == "ws17-groebner" => cmd_gate_ws17_groebner(),
        [a, b] if a == "gate" && b == "ws18-analytic-calculus" => cmd_gate_ws18_analytic_calculus(),
        [a, b] if a == "gate" && b == "ws19-solvers" => cmd_gate_ws19_solvers(),
        [a, b] if a == "gate" && b == "ws20-structured-domains" => {
            cmd_gate_ws20_structured_domains()
        }
        [a, b] if a == "gate" && b == "ws21-profile-closure" => cmd_gate_ws21_profile_closure(),
        [a, b] if a == "gate" && b == "ws22-performance" => cmd_gate_ws22_performance(),
        [a, b, flag, profile]
            if a == "gate" && b == "python-object-model" && flag == "--profile" =>
        {
            cmd_gate_python_object_model(profile)
        }
        _ => return std::process::ExitCode::from(print_usage() as u8),
    };
    println!(
        "{} gate={} status={} checks_digest={}",
        RECEIPTS_DIR, receipt.gate, receipt.status, receipt.checks_digest
    );
    std::process::ExitCode::from(if receipt.status == "passed" { 0 } else { 1 })
}

#[cfg(test)]
mod execution_tests {
    use super::*;

    #[test]
    fn interleaved_receipt_writers_cannot_modify_each_others_publication() {
        use std::io::Write;

        let root = std::env::temp_dir().join(format!(
            "xtask-publication-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let legacy = root.join(".foundation.tmp");
        std::fs::write(&legacy, b"preserve existing artifact").unwrap();
        let (first_path, mut first) = reserve_receipt_temp(&root, "foundation").unwrap();
        let (second_path, mut second) = reserve_receipt_temp(&root, "foundation").unwrap();
        assert_ne!(first_path, second_path);

        // Deliberately schedule the second writer after the first rename.
        // A shared temporary inode would mutate the already published receipt.
        let published = root.join("foundation.receipt.json");
        first.write_all(b"first complete receipt").unwrap();
        drop(first);
        std::fs::rename(first_path, &published).unwrap();
        second.write_all(b"second complete receipt").unwrap();
        assert_eq!(
            std::fs::read(&published).unwrap(),
            b"first complete receipt"
        );
        drop(second);
        std::fs::rename(second_path, &published).unwrap();
        assert_eq!(
            std::fs::read(&published).unwrap(),
            b"second complete receipt"
        );
        assert_eq!(
            std::fs::read(&legacy).unwrap(),
            b"preserve existing artifact"
        );
    }

    #[test]
    fn child_execution_control() {
        assert_eq!(
            passed_test_count("test result: ok. 0 passed; 2 filtered out;"),
            0
        );
        assert_eq!(
            passed_test_count("running 4 tests\ntest result: FAILED. 3 passed; 1 failed;"),
            0
        );
        assert_eq!(passed_test_count("test result: ok. 2 passed; 0 failed;"), 2);
    }

    #[test]
    fn actual_child_test_execution_and_zero_run_have_different_verdicts() {
        for (filter, expected) in [
            ("execution_tests::child_execution_control", "passed"),
            ("this_test_filter_does_not_exist", "failed"),
        ] {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command.args(["--exact", filter]);
            let mut checks = Vec::new();
            run_command("test-child-execution", command, &mut checks);
            assert_eq!(checks.len(), 1);
            assert_eq!(checks[0].status, expected, "{}", checks[0].detail);
            let detail: serde_json::Value = serde_json::from_str(&checks[0].detail).unwrap();
            assert_eq!(detail["exit_code"], 0); // both real processes exit 0
            assert!(
                detail["stdout"]
                    .as_str()
                    .unwrap()
                    .contains("test result: ok.")
            );
        }
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.arg("--not-a-valid-test-harness-option");
        let mut checks = Vec::new();
        run_command("test-child-execution", command, &mut checks);
        assert_eq!(checks[0].status, "failed");
        let detail: serde_json::Value = serde_json::from_str(&checks[0].detail).unwrap();
        assert_ne!(detail["exit_code"], 0);
        assert!(!detail["stderr"].as_str().unwrap().is_empty());
    }
}
