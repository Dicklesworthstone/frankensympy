//! Registered negative corpus for the receipt validator: the runner-weakening
//! mutants MUST flip the validator's verdict (bead fra-gate-runner-xtask-cyx;
//! mutation discipline per Art. VIII.5). These tests build receipts by hand —
//! they do NOT link the runner's builder — so a validator that trusts
//! runner-shaped input still gets caught by the tamper cases.

use std::process::Command;

fn validator() -> Command {
    Command::new(env!("CARGO_BIN_EXE_gate-receipt-validator"))
}

fn base_receipt() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "gate": "foundation",
        "profile_id": "sympy-1.14.0-cpython",
        "status": "passed",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "checks": [
            {"name": "workspace-forbids-unsafe", "status": "passed", "detail": "ok"},
            {"name": "tests-fsym-id", "status": "passed", "detail": "exit=0"}
        ],
        "checks_digest": ""
    })
}

fn canonical(receipt: &serde_json::Value) -> String {
    let rows: Vec<std::collections::BTreeMap<&str, &str>> = receipt["checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            std::collections::BTreeMap::from([
                ("name", c["name"].as_str().unwrap()),
                ("status", c["status"].as_str().unwrap()),
                ("detail", c["detail"].as_str().unwrap()),
            ])
        })
        .collect();
    serde_json::to_string(&rows).unwrap()
}

fn digest(receipt: &serde_json::Value) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(canonical(receipt).as_bytes());
    h.finalize().to_hex().to_string()
}

fn write(
    tmp: &std::path::Path,
    mut receipt: serde_json::Value,
    tamper: impl FnOnce(&mut serde_json::Value),
) -> std::path::PathBuf {
    tamper(&mut receipt);
    receipt["checks_digest"] = serde_json::Value::String(digest(&receipt));
    let path = tmp.join(format!("{}.json", uuidish(&receipt)));
    std::fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    path
}

fn uuidish(receipt: &serde_json::Value) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    receipt.to_string().hash(&mut h);
    format!("case-{:016x}", h.finish())
}

#[test]
fn well_formed_receipt_is_accepted() {
    let tmp = std::env::temp_dir().join("xtask-validator-accept");
    std::fs::create_dir_all(&tmp).unwrap();
    let mut receipt = base_receipt();
    receipt["checks_digest"] = serde_json::Value::String(digest(&receipt));
    let path = tmp.join("good.json");
    std::fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    let status = validator().arg(&path).status().unwrap();
    assert!(status.success(), "validator rejected a well-formed receipt");
}

#[test]
fn mutant_tampered_check_status_is_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-tamper1");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = write(&tmp, base_receipt(), |r| {
        // Simulates a runner weakening: a failed check relabeled as passed
        // AFTER digest re-computation over the relabeled checks is bypassed by
        // flipping the top-level status instead.
        r["status"] = serde_json::json!("passed");
        r["checks"][0]["status"] = serde_json::json!("failed");
        // re-digest with the flipped check so ONLY the status inconsistency remains
        let rows: Vec<std::collections::BTreeMap<&str, &str>> = r["checks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                std::collections::BTreeMap::from([
                    ("name", c["name"].as_str().unwrap()),
                    ("status", c["status"].as_str().unwrap()),
                    ("detail", c["detail"].as_str().unwrap()),
                ])
            })
            .collect();
        let canon = serde_json::to_string(&rows).unwrap();
        use blake3::Hasher;
        let mut h = Hasher::new();
        h.update(canon.as_bytes());
        r["checks_digest"] = serde_json::Value::String(h.finalize().to_hex().to_string());
    });
    let status = validator().arg(&path).status().unwrap();
    assert!(
        !status.success(),
        "validator accepted status inconsistent with checks"
    );
}

#[test]
fn mutant_tampered_digest_is_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-tamper2");
    std::fs::create_dir_all(&tmp).unwrap();
    // Sign FIRST, tamper AFTER: the detail field is modified post-signing and
    // the digest is deliberately NOT recomputed, so the validator must catch
    // the mismatch (this is the mutation that kills a runner that weakens a
    // check after computing its digest).
    let mut receipt = base_receipt();
    receipt["checks_digest"] = serde_json::Value::String(digest(&receipt));
    receipt["checks"][0]["detail"] = serde_json::json!("tampered after signing");
    let path = tmp.join("tampered-after-signing.json");
    std::fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    let status = validator().arg(&path).status().unwrap();
    assert!(
        !status.success(),
        "validator accepted a digest-mismatched receipt"
    );
}

#[test]
fn mutant_unknown_gate_is_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-tamper3");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = write(&tmp, base_receipt(), |r| {
        r["gate"] = serde_json::json!("self-graded-gate");
    });
    let status = validator().arg(&path).status().unwrap();
    assert!(!status.success(), "validator accepted an unknown gate name");
}

#[test]
fn mutant_unknown_field_is_rejected_fail_closed() {
    let tmp = std::env::temp_dir().join("xtask-validator-tamper4");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = write(&tmp, base_receipt(), |r| {
        r["override_evidence"] = serde_json::json!(true);
    });
    let status = validator().arg(&path).status().unwrap();
    assert!(
        !status.success(),
        "validator accepted an unknown (override-shaped) field"
    );
}

#[test]
fn mutant_empty_checks_is_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-tamper5");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = write(&tmp, base_receipt(), |r| {
        r["checks"] = serde_json::json!([]);
    });
    let status = validator().arg(&path).status().unwrap();
    assert!(
        !status.success(),
        "validator accepted a receipt with no checks"
    );
}
