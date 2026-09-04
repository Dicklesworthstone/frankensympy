//! Independent receipt validator (structurally separate from the xtask
//! runner; the runner must never grade itself). Re-derives the checks digest
//! from the receipt's own commitments and rejects tampering, unknown gate
//! names, inconsistent statuses, and schema drift — fail closed.
//!
//! Usage: gate-receipt-validator <receipt.json> [<receipt.json> ...]
//! Exit 0 iff every receipt validates. Registered negative corpus lives in
//! xtask/tests/receipt_tamper.rs (runner-weakening mutants must flip these
//! verdicts).

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::process::Command;

const KNOWN_GATES: [&str; 3] = ["profile-verify", "foundation", "python-object-model"];
const KNOWN_CHECK_STATUSES: [&str; 2] = ["passed", "failed"];
const KNOWN_STATUSES: [&str; 2] = ["passed", "failed"];

fn fail(what: &str, why: &str) -> i32 {
    eprintln!("REJECT {what}: {why}");
    1
}

fn validate(path: &str) -> i32 {
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
    if obj["schema_version"] != serde_json::json!(1) {
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
    if obj["commit"].as_str().unwrap_or("").is_empty() {
        return fail(path, "commit must be a non-empty string");
    }

    let checks = match obj["checks"].as_array() {
        Some(c) if !c.is_empty() => c,
        _ => return fail(path, "checks must be a non-empty array"),
    };
    let mut canonical_rows: Vec<BTreeMap<&str, String>> = Vec::new();
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
        canonical_rows.push(BTreeMap::from([
            ("name", cobj["name"].as_str().unwrap_or("").to_string()),
            ("status", cstatus.to_string()),
            ("detail", cobj["detail"].as_str().unwrap_or("").to_string()),
        ]));
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
    let digest = Command::new("sha256sum").arg("/dev/null").output(); // placeholder to keep no external deps; real digest below
    let _ = digest;
    // blake3 is available to the validator too; compute directly.
    let computed = blake3_hash(canonical.as_bytes());
    let claimed = obj["checks_digest"].as_str().unwrap_or("");
    if claimed != computed {
        return fail(
            path,
            &format!("checks_digest mismatch: claimed {claimed}, recomputed {computed}"),
        );
    }

    // Digest re-derivation: canonical JSON of checks, blake3, hex. This
    // intentionally recomputes from the receipt bytes rather than trusting
    // any runner-supplied digest; no runner code is linked here.
    let computed = blake3_hash(canonical.as_bytes());
    let claimed = obj["checks_digest"].as_str().unwrap_or("");
    if claimed != computed {
        return fail(
            path,
            &format!("checks_digest mismatch: claimed {claimed}, recomputed {computed}"),
        );
    }

    println!(
        "ACCEPT {path} gate={gate} status={status} checks={}",
        checks.len()
    );
    0
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: gate-receipt-validator <receipt.json> [...]");
        return std::process::ExitCode::from(2);
    }
    let mut worst = 0;
    for path in &args {
        let code = validate(path);
        worst = worst.max(code);
    }
    std::process::ExitCode::from(worst as u8)
}
