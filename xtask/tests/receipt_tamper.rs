//! Registered negative corpus for the receipt validator: the runner-weakening
//! mutants MUST flip the validator's verdict (bead fra-gate-runner-xtask-cyx;
//! mutation discipline per Art. VIII.5). These tests build receipts by hand —
//! they do NOT link the runner's builder — so a validator that trusts
//! runner-shaped input still gets caught by the tamper cases.

use std::process::Command;

fn validator() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gate-receipt-validator"));
    command.arg("--source-root").arg(test_source());
    command
}

fn test_source() -> &'static std::path::Path {
    // Real isolated Git source, not the remote build worker's checkout metadata
    // (RCH transfers source without .git). These are parser/identity controls,
    // not a claim that the synthetic receipt's gate commands actually ran.
    static SOURCE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    SOURCE
        .get_or_init(|| {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root =
                std::env::temp_dir().join(format!("xtask-source-{}-{nonce}", std::process::id()));
            std::fs::create_dir(&root).unwrap();
            let profiles = root.join("tools/conformance-lab/profiles");
            std::fs::create_dir_all(&profiles).unwrap();
            std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
            std::fs::write(root.join("Cargo.lock"), "version = 4\n").unwrap();
            std::fs::write(
                root.join("rust-toolchain.toml"),
                "[toolchain]\nchannel = \"nightly-2026-08-20\"\n",
            )
            .unwrap();
            std::fs::write(
                profiles.join("sympy-1.14.0-cpython.toml"),
                "profile_id = \"sympy-1.14.0-cpython\"\n",
            )
            .unwrap();
            for args in [
                vec!["init", "-q"],
                vec!["add", "."],
                vec![
                    "-c",
                    "user.name=GateTest",
                    "-c",
                    "user.email=gate-test@example.invalid",
                    "commit",
                    "-qm",
                    "isolated source fixture",
                ],
            ] {
                let output = Command::new("git")
                    .current_dir(&root)
                    .args(args)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            root
        })
        .as_path()
}

fn base_receipt() -> serde_json::Value {
    let source = xtask::source_snapshot(test_source()).unwrap();
    let mut receipt = serde_json::json!({
        "schema_version": 2,
        "gate": "foundation",
        "profile_id": "sympy-1.14.0-cpython",
        "status": "passed",
        "commit": source.commit,
        "source": source,
        "profile_digest": xtask::profile_digest(test_source(), "sympy-1.14.0-cpython").unwrap(),
        "checks": [
            {"name": "workspace-forbids-unsafe", "status": "passed", "detail": "ok"},
            {"name": "source-stable-during-run", "status": "passed", "detail": "ok"},
            {"name": "profile-input-bound", "status": "passed", "detail": "ok"}
        ],
        "checks_digest": "", "receipt_digest": ""
    });
    // Synthetic parser/control data, NOT evidence these crate tests executed.
    for name in [
        "fsym-id",
        "fsym-budget",
        "fsym-outcome",
        "fsym-bigint",
        "fsym-rational",
        "fsym-modular",
    ] {
        receipt["checks"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "name": format!("tests-{name}"), "status": "passed",
                "detail": serde_json::json!({"command": ["cargo", "test", "-p", name, "--quiet"],
                    "exit_code": 0, "stdout": "test result: ok. 1 passed; 0 failed; 0 ignored;\n",
                    "stderr": ""}).to_string()
            }));
    }
    receipt
}

fn seal(receipt: &mut serde_json::Value) {
    receipt["checks_digest"] = serde_json::Value::String(digest(receipt));
    let mut payload = receipt.as_object().unwrap().clone();
    payload.remove("receipt_digest");
    receipt["receipt_digest"] = serde_json::json!(
        blake3::hash(&serde_json::to_vec(&payload).unwrap())
            .to_hex()
            .to_string()
    );
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
    seal(&mut receipt);
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
fn matching_command_with_zero_tests_is_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-matching-command-zero-tests");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = write(&tmp, base_receipt(), |r| {
        let check = &mut r["checks"][3];
        let mut detail: serde_json::Value =
            serde_json::from_str(check["detail"].as_str().unwrap()).unwrap();
        detail["stdout"] = serde_json::json!("test result: ok. 0 passed; 0 failed;\n");
        check["detail"] = serde_json::json!(detail.to_string());
    });
    let output = validator().arg(path).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("nonzero test execution"));
}

#[test]
fn integration_test_target_is_bound_to_its_check() {
    let tmp = std::env::temp_dir().join("xtask-validator-integration-command");
    std::fs::create_dir_all(&tmp).unwrap();
    let mut receipt = base_receipt();
    receipt["gate"] = serde_json::json!("ws12-certified-jacobian");
    receipt["checks"].as_array_mut().unwrap().truncate(3);
    for (name, command) in [
        (
            "test-sparse-jacobian-c7",
            vec![
                "cargo",
                "test",
                "-p",
                "fsym-calculus",
                "--test",
                "sparse_jacobian_c7",
                "--quiet",
            ],
        ),
        (
            "test-sparse-jacobian-gate",
            vec![
                "cargo",
                "test",
                "-p",
                "fsym-calculus",
                "--test",
                "sparse_jacobian_gate",
                "--quiet",
            ],
        ),
        (
            "tests-fsym-calculus",
            vec!["cargo", "test", "-p", "fsym-calculus", "--quiet"],
        ),
    ] {
        receipt["checks"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "name": name, "status": "passed",
                "detail": serde_json::json!({"command": command, "exit_code": 0,
                    "stdout": "test result: ok. 1 passed; 0 failed;\n", "stderr": ""}).to_string()
            }));
    }
    let path = write(&tmp, receipt.clone(), |_| {});
    assert!(validator().arg(path).status().unwrap().success());
    let path = write(&tmp, receipt, |r| {
        let check = &mut r["checks"][3];
        let mut detail: serde_json::Value =
            serde_json::from_str(check["detail"].as_str().unwrap()).unwrap();
        detail["command"][5] = serde_json::json!("sparse_jacobian_gate");
        check["detail"] = serde_json::json!(detail.to_string());
    });
    let output = validator().arg(path).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("command"));
}

#[test]
fn resigned_command_substitution_is_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-command-substitution");
    std::fs::create_dir_all(&tmp).unwrap();
    for command in [
        serde_json::json!(["echo", "test result: ok. 1 passed;"]),
        serde_json::json!(["cargo", "test", "-p", "fsym-core", "--quiet"]),
        serde_json::json!(["cargo", "test", "-p", "fsym-id", "one_test", "--quiet"]),
        serde_json::json!([
            "cargo",
            "test",
            "-p",
            "fsym-id",
            "--quiet",
            "--",
            "--ignored"
        ]),
        serde_json::json!(["cargo", "test", "-p", "fsym-id", 0]),
    ] {
        let path = write(&tmp, base_receipt(), |r| {
            let check = &mut r["checks"][3];
            let mut detail: serde_json::Value =
                serde_json::from_str(check["detail"].as_str().unwrap()).unwrap();
            detail["command"] = command;
            check["detail"] = serde_json::json!(detail.to_string());
        });
        let output = validator().arg(path).output().unwrap();
        assert!(
            !output.status.success(),
            "validator accepted substituted command"
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("command"));
    }
}

#[test]
fn well_formed_receipt_is_accepted() {
    let tmp = std::env::temp_dir().join("xtask-validator-accept");
    std::fs::create_dir_all(&tmp).unwrap();
    let mut receipt = base_receipt();
    seal(&mut receipt);
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
    seal(&mut receipt);
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

#[test]
fn resigned_metadata_missing_check_zero_run_and_failed_gate_are_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-binding-mutants");
    std::fs::create_dir_all(&tmp).unwrap();
    for mutation in 0..9 {
        let path = write(&tmp, base_receipt(), |receipt| match mutation {
            0 => receipt["commit"] = serde_json::json!("unknown"),
            1 => receipt["profile_id"] = serde_json::json!("sympy-1.14.0-cpython-r2-corpus"),
            2 => receipt["gate"] = serde_json::json!("ws10-exact-linear"),
            3 => {
                receipt["checks"].as_array_mut().unwrap().pop();
            }
            4 => {
                let duplicate = receipt["checks"][0].clone();
                receipt["checks"].as_array_mut().unwrap().push(duplicate);
            }
            5 => receipt["source"]["inputs_digest"] = serde_json::json!("changed"),
            6 => receipt["profile_digest"] = serde_json::json!("changed"),
            7 => {
                receipt["checks"][3]["detail"] = serde_json::json!(
                    serde_json::json!({
                        "command": ["cargo", "test", "absent-filter"], "exit_code": 0,
                        "stdout": "test result: ok. 0 passed; 0 failed; 20 filtered out;",
                        "stderr": ""
                    })
                    .to_string()
                )
            }
            8 => {
                receipt["checks"][0]["status"] = serde_json::json!("failed");
                receipt["status"] = serde_json::json!("failed");
            }
            _ => unreachable!(),
        });
        assert!(
            !validator().arg(path).status().unwrap().success(),
            "mutation {mutation} survived"
        );
    }
}

#[test]
fn metadata_change_without_resealing_is_rejected() {
    let tmp = std::env::temp_dir().join("xtask-validator-metadata-digest");
    std::fs::create_dir_all(&tmp).unwrap();
    let mut receipt = base_receipt();
    seal(&mut receipt);
    receipt["profile_digest"] = serde_json::json!("tampered");
    let path = tmp.join("metadata.json");
    std::fs::write(&path, receipt.to_string()).unwrap();
    assert!(!validator().arg(path).status().unwrap().success());
}
