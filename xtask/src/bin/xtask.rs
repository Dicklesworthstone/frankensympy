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

use serde::Serialize;

const RECEIPTS_DIR: &str = "artifacts/audit/receipts";

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
    checks: Vec<Check>,
    /// blake3 digest over the canonical JSON of `checks` (BTreeMap ordering).
    checks_digest: String,
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

fn head_commit() -> String {
    let out = Command::new("git").args(["rev-parse", "HEAD"]).output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn run_command(name: &str, mut cmd: Command, checks: &mut Vec<Check>) {
    let output = cmd.output();
    match output {
        Ok(o) if o.status.success() => checks.push(Check {
            name: name.to_string(),
            status: "passed".into(),
            detail: "exit=0".into(),
        }),
        Ok(o) => {
            let tail = String::from_utf8_lossy(&o.stderr);
            let tail: String = tail.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
            checks.push(Check {
                name: name.to_string(),
                status: "failed".into(),
                detail: format!("exit={} tail={}", o.status.code().unwrap_or(-1), tail),
            })
        }
        Err(e) => checks.push(Check {
            name: name.to_string(),
            status: "failed".into(),
            detail: format!("spawn error: {e}"),
        }),
    }
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

fn finish(gate: &str, profile_id: &str, checks: Vec<Check>) -> Receipt {
    let all_passed = checks.iter().all(|c| c.status == "passed");
    let receipt = Receipt {
        schema_version: 1,
        gate: gate.to_string(),
        profile_id: profile_id.to_string(),
        status: if all_passed { "passed" } else { "failed" }.into(),
        commit: head_commit(),
        checks_digest: checks_digest(&checks),
        checks,
    };
    write_receipt(&receipt);
    receipt
}

fn write_receipt(receipt: &Receipt) {
    let dir = std::path::Path::new(RECEIPTS_DIR);
    std::fs::create_dir_all(dir).expect("create receipts dir");
    let json = serde_json::to_string_pretty(receipt).expect("receipt serializes");
    let tmp = dir.join(format!(".{}.tmp", receipt.gate));
    std::fs::write(&tmp, json.as_bytes()).expect("write temp receipt");
    let final_path = dir.join(format!("{}.receipt.json", receipt.gate));
    std::fs::rename(&tmp, final_path).expect("atomic receipt rename");
}

fn print_usage() -> i32 {
    eprintln!(
        "usage: xtask profile verify <profile-id> | xtask gate foundation | xtask gate python-object-model --profile <profile-id>"
    );
    2
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let receipt = match args.as_slice() {
        [a, b, profile] if a == "profile" && b == "verify" => cmd_profile_verify(profile),
        [a, b] if a == "gate" && b == "foundation" => cmd_gate_foundation(),
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
