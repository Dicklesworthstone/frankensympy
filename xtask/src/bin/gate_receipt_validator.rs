//! Independent receipt validator (structurally separate from the xtask
//! runner; the runner must never grade itself). Re-derives the checks digest
//! from the receipt's own commitments and rejects tampering, unknown gate
//! names, inconsistent statuses, and schema drift — fail closed.
//!
//! Usage: gate-receipt-validator <receipt.json> [<receipt.json> ...]
//! Exit 0 iff every receipt validates against current repository inputs and
//! has passed checks. This is not execution attestation. Negative corpus lives in
//! xtask/tests/receipt_tamper.rs (runner-weakening mutants must flip these
//! verdicts).

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

const KNOWN_GATES: [&str; 18] = [
    "profile-verify",
    "foundation",
    "python-object-model",
    "deterministic-term-identity",
    "ws09-factorization",
    "ws10-exact-linear",
    "ws11-certified-numeric",
    "ws12-certified-jacobian",
    "ws13-portfolio-runtime",
    "ws14-agent-protocol",
    "ws15-persistence-repair",
    "ws16-distribution-index",
    "ws17-groebner",
    "ws18-analytic-calculus",
    "ws19-solvers",
    "ws20-structured-domains",
    "ws21-profile-closure",
    "ws22-performance",
];
const KNOWN_CHECK_STATUSES: [&str; 2] = ["passed", "failed"];
const KNOWN_STATUSES: [&str; 2] = ["passed", "failed"];

fn required_checks(gate: &str) -> Vec<&'static str> {
    let mut checks = match gate {
        "profile-verify" => vec![
            "profile-id-matches-declared-module",
            "registry-validators",
            "workspace-forbids-unsafe",
        ],
        "foundation" => vec![
            "workspace-forbids-unsafe",
            "tests-fsym-id",
            "tests-fsym-budget",
            "tests-fsym-outcome",
            "tests-fsym-bigint",
            "tests-fsym-rational",
            "tests-fsym-modular",
        ],
        "python-object-model" => vec![
            "oracle-version-pinned",
            "profile-id-matches-declared-module",
            "oracle-isolation-probe",
            "object-model-differential",
        ],
        "deterministic-term-identity" => vec![
            "workspace-forbids-unsafe",
            "test-fresh-process-id-stability",
            "cross-architecture-fixture-closure",
            "tests-fsym-core",
            "tests-fsym-assumptions",
            "tests-fsym-id",
        ],
        "ws09-factorization" | "ws17-groebner" => {
            vec!["workspace-forbids-unsafe", "tests-fsym-polys"]
        }
        "ws10-exact-linear" => vec!["workspace-forbids-unsafe", "tests-fsym-matrices"],
        "ws11-certified-numeric" => vec![
            "workspace-forbids-unsafe",
            "test-directed-rounding-mutation",
            "tests-fsym-core",
            "tests-fsym-proof-kernel",
        ],
        "ws12-certified-jacobian" => vec![
            "workspace-forbids-unsafe",
            "test-sparse-jacobian-c7",
            "test-sparse-jacobian-gate",
            "tests-fsym-calculus",
        ],
        "ws13-portfolio-runtime" => vec![
            "workspace-forbids-unsafe",
            "test-cancellation-injection",
            "tests-fsym-runtime",
        ],
        "ws14-agent-protocol" => vec![
            "workspace-forbids-unsafe",
            "test-c10-protocol-gate",
            "tests-fsym-runtime",
        ],
        "ws15-persistence-repair" => vec![
            "workspace-forbids-unsafe",
            "test-c9-persistence-repair-gate",
            "tests-fsym-runtime",
        ],
        "ws16-distribution-index" => vec![
            "workspace-forbids-unsafe",
            "test-ws16-distribution-index-gate",
            "tests-fsym-runtime",
        ],
        "ws18-analytic-calculus" => vec!["workspace-forbids-unsafe", "tests-fsym-calculus"],
        "ws19-solvers" => vec![
            "workspace-forbids-unsafe",
            "tests-fsym-solvers",
            "tests-fsym-sets",
            "tests-fsym-logic",
        ],
        "ws20-structured-domains" => vec![
            "workspace-forbids-unsafe",
            "tests-fsym-geometry",
            "tests-fsym-tensor",
        ],
        "ws21-profile-closure" => vec![
            "workspace-forbids-unsafe",
            "registry-validators",
            "tests-fsym-conformance",
            "corpus-gate",
            "exclusion-ledger-verification",
        ],
        "ws22-performance" => vec![
            "workspace-forbids-unsafe",
            "test-ws22-performance-gate",
            "tests-fsym-runtime",
            "paired-live-incumbent-bench",
            "paired-benchmark-report-verification",
        ],
        _ => vec![],
    };
    checks.extend(["source-stable-during-run", "profile-input-bound"]);
    checks
}

fn fail(what: &str, why: &str) -> i32 {
    eprintln!("REJECT {what}: {why}");
    1
}

fn validate(path: &str, source_root: &std::path::Path) -> i32 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() && metadata.len() <= 16 * 1024 * 1024 => {}
        _ => return fail(path, "receipt must be a regular file within 16 MiB"),
    }
    let raw = match std::fs::read_to_string(path) {
        Ok(r) => r,
        Err(e) => return fail(path, &format!("unreadable: {e}")),
    };
    let v: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => return fail(path, &format!("not JSON: {e}")),
    };
    let obj = match v.as_object() {
        Some(o) => o,
        None => return fail(path, "receipt is not an object"),
    };

    // Schema fields, fail-closed on unknown fields.
    let required = [
        "schema_version",
        "gate",
        "profile_id",
        "status",
        "commit",
        "checks",
        "checks_digest",
        "source",
        "profile_digest",
        "receipt_digest",
    ];
    for key in required {
        if !obj.contains_key(key) {
            return fail(path, &format!("missing required field {key}"));
        }
    }
    let allowed: std::collections::HashSet<&str> = required.iter().copied().collect();
    for key in obj.keys() {
        if !allowed.contains(key.as_str()) {
            return fail(path, &format!("unknown field {key} (fail closed)"));
        }
    }
    if obj["schema_version"] != serde_json::json!(2) {
        return fail(path, "unsupported schema_version");
    }
    let gate = obj["gate"].as_str().unwrap_or("");
    if !KNOWN_GATES.contains(&gate) {
        return fail(path, &format!("unknown gate {gate:?}"));
    }
    let status = obj["status"].as_str().unwrap_or("");
    if !KNOWN_STATUSES.contains(&status) {
        return fail(path, &format!("unknown status {status:?}"));
    }
    let source: xtask::SourceSnapshot = match serde_json::from_value(obj["source"].clone()) {
        Ok(source) => source,
        Err(error) => return fail(path, &format!("invalid source snapshot: {error}")),
    };
    let expected = match xtask::source_snapshot(source_root) {
        Ok(expected) => expected,
        Err(error) => return fail(path, &format!("cannot inspect expected source: {error}")),
    };
    if source != expected || obj["commit"].as_str() != Some(expected.commit.as_str()) {
        return fail(
            path,
            "receipt source does not match inspected repository inputs",
        );
    }
    let profile = obj["profile_id"].as_str().unwrap_or("");
    if !["profile-verify", "python-object-model"].contains(&gate)
        && profile != "sympy-1.14.0-cpython"
    {
        return fail(path, "gate profile does not match its declared contract");
    }
    let profile_digest = match xtask::profile_digest(source_root, profile) {
        Ok(digest) => digest,
        Err(error) => return fail(path, &format!("invalid profile: {error}")),
    };
    if obj["profile_digest"].as_str() != Some(profile_digest.as_str()) {
        return fail(path, "profile digest mismatch");
    }

    let checks = match obj["checks"].as_array() {
        Some(c) if !c.is_empty() => c,
        _ => return fail(path, "checks must be a non-empty array"),
    };
    let mut canonical_rows: Vec<BTreeMap<&str, String>> = Vec::new();
    let expected_checks: std::collections::BTreeSet<_> =
        required_checks(gate).into_iter().collect();
    let mut seen = std::collections::BTreeSet::new();
    for check in checks {
        let cobj = match check.as_object() {
            Some(o) => o,
            None => return fail(path, "check entry is not an object"),
        };
        let ckeys = ["name", "status", "detail"];
        for key in ckeys {
            if !cobj.contains_key(key) {
                return fail(path, &format!("check missing field {key}"));
            }
        }
        for key in cobj.keys() {
            if !ckeys.contains(&key.as_str()) {
                return fail(path, &format!("check has unknown field {key}"));
            }
        }
        let cstatus = cobj["status"].as_str().unwrap_or("");
        if !KNOWN_CHECK_STATUSES.contains(&cstatus) {
            return fail(path, &format!("unknown check status {cstatus:?}"));
        }
        let name = cobj["name"].as_str().unwrap_or("");
        if !expected_checks.contains(name) || !seen.insert(name.to_string()) {
            return fail(path, "unknown or duplicated check name");
        }
        if cobj["detail"].as_str().is_none_or(str::is_empty) {
            return fail(path, "check detail must be a nonempty string");
        }
        if cstatus == "passed" && (name.starts_with("test-") || name.starts_with("tests-")) {
            let detail: serde_json::Value =
                match serde_json::from_str(cobj["detail"].as_str().unwrap()) {
                    Ok(detail) => detail,
                    Err(_) => return fail(path, "test check lacks execution transcript"),
                };
            let ran_tests = detail["stdout"]
                .as_str()
                .unwrap_or("")
                .lines()
                .filter_map(|line| {
                    let rest = line.strip_prefix("test result: ok. ")?;
                    rest.split_whitespace().next()?.parse::<usize>().ok()
                })
                .any(|count| count > 0);
            if detail["exit_code"] != serde_json::json!(0)
                || !ran_tests
                || detail["command"].as_array().is_none_or(Vec::is_empty)
                || detail["stderr"].as_str().is_none()
            {
                return fail(path, "test check has no successful nonzero test execution");
            }
        }
        canonical_rows.push(BTreeMap::from([
            ("name", cobj["name"].as_str().unwrap_or("").to_string()),
            ("status", cstatus.to_string()),
            ("detail", cobj["detail"].as_str().unwrap_or("").to_string()),
        ]));
    }
    if seen.len() != expected_checks.len() {
        return fail(path, "required check set is incomplete");
    }

    // Status consistency: receipt status must follow from the checks.
    let all_passed = canonical_rows.iter().all(|r| r["status"] == "passed");
    let derived = if all_passed { "passed" } else { "failed" };
    if status != derived {
        return fail(
            path,
            &format!("status {status:?} inconsistent with checks (derived {derived:?})"),
        );
    }

    // Digest re-derivation: canonical JSON of checks, blake3, hex.
    let canonical = serde_json::to_string(&canonical_rows).expect("canonical rows serialize");
    // blake3 is available to the validator too; compute directly.
    let computed = blake3_hash(canonical.as_bytes());
    let claimed = obj["checks_digest"].as_str().unwrap_or("");
    if claimed != computed {
        return fail(
            path,
            &format!("checks_digest mismatch: claimed {claimed}, recomputed {computed}"),
        );
    }

    // Bind metadata as well as check rows. This detects alteration; it is
    // not a signature or independent attestation that commands executed.
    let mut payload = obj.clone();
    payload.remove("receipt_digest");
    let computed = blake3_hash(&serde_json::to_vec(&payload).expect("JSON payload"));
    let claimed = obj["receipt_digest"].as_str().unwrap_or("");
    if claimed != computed {
        return fail(
            path,
            &format!("receipt_digest mismatch: claimed {claimed}, recomputed {computed}"),
        );
    }

    println!(
        "VALID {path} gate={gate} status={status} checks={}",
        checks.len()
    );
    if all_passed { 0 } else { 1 }
}

fn blake3_hash(data: &[u8]) -> String {
    // Intentionally recomputes from the receipt bytes rather than trusting
    // any runner-supplied digest; no runner code is linked here.
    {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(data);
        hasher.finalize().to_hex().to_string()
    }
}

fn main() -> std::process::ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let source_root = if args.first().is_some_and(|arg| arg == "--source-root") && args.len() >= 2 {
        args.remove(0);
        std::path::PathBuf::from(args.remove(0))
    } else {
        xtask::workspace_root()
    };
    if args.is_empty() {
        eprintln!("usage: gate-receipt-validator [--source-root PATH] <receipt.json> [...]");
        return std::process::ExitCode::from(2);
    }
    let mut worst = 0;
    for path in &args {
        let code = validate(path, &source_root);
        worst = worst.max(code);
    }
    std::process::ExitCode::from(worst as u8)
}
