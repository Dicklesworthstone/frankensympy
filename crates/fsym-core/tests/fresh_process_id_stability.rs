//! Integration tests for fresh-process ID stability and deterministic term identity (WS04 / SEMANTIC-001).
//!
//! Verifies:
//! 1. Two fresh OS processes constructing the canonical term suite produce identical TermIds and BLAKE3 digests.
//! 2. Alpha-equivalence: Lambdas with different bound variable names (e.g. `\x. x + 1` vs `\y. y + 1`) intern to identical TermIds and digests.
//! 3. Cross-architecture stability: All computed term digests match the golden specification in `artifacts/conformance/fixtures/deterministic_term_identity_v1.json`.
//! 4. Insertion-order independence: Inserting terms in forward vs reverse order produces identical TermIds.
//! 5. Mutation rejection: Any perturbation in node payload or domain changes the computed digest.

#![forbid(unsafe_code)]

use fsym_bigint::BigInt;
use fsym_core::dag::{TermDag, TermNode, compute_term_digest, compute_term_digest_in_domain};
use fsym_core::{Symbol, TermDomain};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TermIdentityRecord {
    term_id_raw: u64,
    digest_hex: String,
    domain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenFixture {
    schema_version: u32,
    profile_id: String,
    hasher: String,
    terms: BTreeMap<String, TermIdentityRecord>,
}

fn run_probe_process() -> BTreeMap<String, TermIdentityRecord> {
    let probe_bin = env!("CARGO_BIN_EXE_term_identity_probe");
    let output = Command::new(probe_bin)
        .output()
        .expect("spawn term_identity_probe process");

    assert!(
        output.status.success(),
        "probe exited with failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).expect("valid JSON from probe")
}

#[test]
fn fresh_processes_produce_identical_term_ids_and_digests() {
    // Spawn two completely separate OS processes with distinct address spaces.
    let proc1_results = run_probe_process();
    let proc2_results = run_probe_process();

    assert!(!proc1_results.is_empty(), "probe returned no terms");
    assert_eq!(
        proc1_results.len(),
        proc2_results.len(),
        "process result counts differ"
    );

    for (name, rec1) in &proc1_results {
        let rec2 = proc2_results
            .get(name)
            .unwrap_or_else(|| panic!("term {name} missing in process 2"));
        assert_eq!(
            rec1, rec2,
            "term {name} differed between process 1 and process 2: {rec1:?} vs {rec2:?}"
        );
    }
}

#[test]
fn de_bruijn_lambdas_alpha_normalize_to_identical_id_and_digest() {
    let results = run_probe_process();

    let lambda_x = results
        .get("lambda_x_plus_1")
        .expect("lambda_x_plus_1 exists");
    let lambda_y = results
        .get("lambda_y_plus_1")
        .expect("lambda_y_plus_1 exists");

    assert_eq!(
        lambda_x.term_id_raw, lambda_y.term_id_raw,
        "Lambda with 'x' vs 'y' parameter must have identical TermId under alpha-equivalence"
    );
    assert_eq!(
        lambda_x.digest_hex, lambda_y.digest_hex,
        "Lambda with 'x' vs 'y' parameter must have identical BLAKE3 digest"
    );
}

#[test]
fn canonical_terms_match_golden_specification() {
    let results = run_probe_process();

    // Locate golden fixture relative to workspace root
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("artifacts/conformance/fixtures/deterministic_term_identity_v1.json");

    assert!(
        fixture_path.exists(),
        "golden fixture missing at {:?}",
        fixture_path
    );

    let raw = std::fs::read_to_string(&fixture_path).expect("read golden fixture");
    let golden: GoldenFixture = serde_json::from_str(&raw).expect("parse golden fixture");

    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.profile_id, "deterministic-term-identity-v1");
    assert_eq!(golden.hasher, "blake3-256");

    for (name, expected_rec) in &golden.terms {
        let actual_rec = results
            .get(name)
            .unwrap_or_else(|| panic!("term {name} missing in computed results"));
        assert_eq!(
            actual_rec.term_id_raw, expected_rec.term_id_raw,
            "term {name} TermId mismatch: computed {} vs golden {}",
            actual_rec.term_id_raw, expected_rec.term_id_raw
        );
        assert_eq!(
            actual_rec.digest_hex, expected_rec.digest_hex,
            "term {name} BLAKE3 digest mismatch: computed {} vs golden {}",
            actual_rec.digest_hex, expected_rec.digest_hex
        );
        assert_eq!(
            actual_rec.domain, expected_rec.domain,
            "term {name} domain mismatch"
        );
    }
}

#[test]
fn insertion_order_does_not_affect_term_identity() {
    let mut dag_forward = TermDag::new();
    let x_f = dag_forward
        .insert_node(TermNode::Sym(Symbol::new("x")))
        .unwrap();
    let y_f = dag_forward
        .insert_node(TermNode::Sym(Symbol::new("y")))
        .unwrap();
    let add_f = dag_forward
        .insert_node(TermNode::Add(vec![x_f, y_f]))
        .unwrap();

    let mut dag_reverse = TermDag::new();
    let y_r = dag_reverse
        .insert_node(TermNode::Sym(Symbol::new("y")))
        .unwrap();
    let x_r = dag_reverse
        .insert_node(TermNode::Sym(Symbol::new("x")))
        .unwrap();
    let add_r = dag_reverse
        .insert_node(TermNode::Add(vec![x_r, y_r]))
        .unwrap();

    assert_eq!(x_f, x_r);
    assert_eq!(y_f, y_r);
    assert_eq!(add_f, add_r);

    assert_eq!(
        dag_forward.term_digest(add_f),
        dag_reverse.term_digest(add_r)
    );
}

#[test]
fn mutation_rejection_and_domain_separation() {
    let node_sym_x = TermNode::Sym(Symbol::new("x"));
    let node_sym_x_prime = TermNode::Sym(Symbol::new("x_prime"));
    let node_int_0 = TermNode::Integer(BigInt::from(0));
    let node_int_1 = TermNode::Integer(BigInt::from(1));

    let dig_x = compute_term_digest(&node_sym_x).unwrap();
    let dig_x_prime = compute_term_digest(&node_sym_x_prime).unwrap();
    let dig_0 = compute_term_digest(&node_int_0).unwrap();
    let dig_1 = compute_term_digest(&node_int_1).unwrap();

    // Small mutations in string or numeric payload completely change digest
    assert_ne!(dig_x, dig_x_prime);
    assert_ne!(dig_0, dig_1);
    assert_ne!(dig_x, dig_0);

    // Domain separation: identical node in Expression vs Integer domain produces distinct digest
    let dig_expr = compute_term_digest_in_domain(&node_int_0, TermDomain::Expression).unwrap();
    let dig_integer = compute_term_digest_in_domain(&node_int_0, TermDomain::Integer).unwrap();
    assert_ne!(
        dig_expr, dig_integer,
        "domain separation must change the canonical preimage digest"
    );
}
