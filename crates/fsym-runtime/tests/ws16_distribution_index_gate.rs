//! WS16 Remote workers and distributed graph indexing gate test suite.
//!
//! Constitutional requirements (WORKSTREAM_GRAPH.md §21, §32):
//! - Untrusted remote workers produce candidates, never accepted state.
//! - Independent coordinator verifier verifies candidates before acceptance.
//! - Zero verified cache pollution: invalid/malicious candidates rejected fail-closed.
//! - Wire decoding bounds and schema strictness enforced (deny_unknown_fields).
//! - Semantic knowledge graph indexing with cycle detection and reproducible reachability.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Unbounded;
use fsym_core::Expr;
use fsym_proof_kernel::{Claim, ProofKernel, StepId};
use fsym_runtime::{
    CoordinatorVerifier, NodeKind, RemoteCandidate, RemoteWorkerError, SemanticGraphIndex,
};

// ============================================================================
// 1. Untrusted Remote Candidate Verification & Adversarial Rejections
// ============================================================================

#[test]
fn test_remote_worker_candidate_acceptance_and_adversarial_rejection() {
    let context = ImmutableAssumptionsSnapshot::empty();
    let x = Expr::symbol("x");
    let task_id = 777;
    let expected_claim = Claim::equality(x.clone(), x.clone());

    let coordinator = CoordinatorVerifier::new(task_id, expected_claim.clone(), context);

    let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
    let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
    let valid_derivation = kernel.export_derivation(step).unwrap();

    // 1. Valid candidate passes verification
    let valid_candidate = RemoteCandidate {
        worker_id: "worker_alpha".to_string(),
        task_id,
        result: x.clone(),
        claim: expected_claim.clone(),
        derivation: valid_derivation.clone(),
        worker_signature: vec![0xAA, 0xBB, 0xCC],
    };

    let accepted = coordinator
        .verify_remote_candidate(&valid_candidate)
        .expect("legitimate candidate must be accepted");
    assert_eq!(accepted.task_id(), task_id);
    assert_eq!(accepted.result(), &x);
    assert_eq!(accepted.claim(), &expected_claim);

    // 2. Mismatched task ID rejected immediately
    let wrong_task = RemoteCandidate {
        task_id: 888,
        ..valid_candidate.clone()
    };
    assert_eq!(
        coordinator
            .verify_remote_candidate(&wrong_task)
            .unwrap_err(),
        RemoteWorkerError::TaskMismatch
    );

    // 3. Forged claimed result rejected
    let forged_result = RemoteCandidate {
        result: Expr::from_i64(12345),
        ..valid_candidate.clone()
    };
    assert_eq!(
        coordinator
            .verify_remote_candidate(&forged_result)
            .unwrap_err(),
        RemoteWorkerError::ClaimForgery
    );

    // 4. Mismatched claim rejected
    let mismatched_claim = RemoteCandidate {
        claim: Claim::equality(x.clone(), Expr::from_i64(12345)),
        result: Expr::from_i64(12345),
        ..valid_candidate.clone()
    };
    assert_eq!(
        coordinator
            .verify_remote_candidate(&mismatched_claim)
            .unwrap_err(),
        RemoteWorkerError::TaskMismatch
    );

    // 5. Tampered derivation root step rejected
    let mut tampered_derivation = valid_derivation.clone();
    tampered_derivation.root = StepId(9999);
    let tampered_candidate = RemoteCandidate {
        derivation: tampered_derivation,
        ..valid_candidate
    };
    assert!(matches!(
        coordinator
            .verify_remote_candidate(&tampered_candidate)
            .unwrap_err(),
        RemoteWorkerError::VerificationFailed(_)
    ));
}

// ============================================================================
// 2. Zero Verified Cache Pollution Under Adversarial Attack
// ============================================================================

#[test]
fn test_zero_cache_pollution_under_adversarial_stream() {
    let context = ImmutableAssumptionsSnapshot::empty();
    let expr = Expr::symbol("y");
    let task_id = 1001;
    let expected_claim = Claim::equality(expr.clone(), expr.clone());

    let coordinator = CoordinatorVerifier::new(task_id, expected_claim, context);

    let mut accepted_count = 0;
    let mut rejected_count = 0;

    // Stream of 10 hostile candidates with various corruptions
    for i in 0..10 {
        let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
        let step = kernel
            .prove_reflexivity(expr.clone(), &mut Unbounded)
            .unwrap();
        let mut derivation = kernel.export_derivation(step).unwrap();

        // Inject tampering on every item
        derivation.root = StepId(100 + i);

        let hostile_candidate = RemoteCandidate {
            worker_id: format!("adversarial_worker_{i}"),
            task_id,
            result: expr.clone(),
            claim: Claim::equality(expr.clone(), expr.clone()),
            derivation,
            worker_signature: vec![i as u8],
        };

        match coordinator.verify_remote_candidate(&hostile_candidate) {
            Ok(_) => accepted_count += 1,
            Err(_) => rejected_count += 1,
        }
    }

    assert_eq!(
        accepted_count, 0,
        "Zero unverified entries admitted under adversarial attack"
    );
    assert_eq!(
        rejected_count, 10,
        "All 10 hostile candidates rejected fail-closed"
    );
}

// ============================================================================
// 3. Wire Decoding Bounds & Fail-Closed Deserialization
// ============================================================================

#[test]
fn test_remote_worker_wire_decoding_bounds_and_unknown_fields() {
    // 1. Unknown fields rejected fail-closed
    let unknown_field_json = r#"{
        "worker_id": "worker1",
        "task_id": 1,
        "result": {"Sym": {"name": "x"}},
        "claim": {"Equality": [{"Sym": {"name": "x"}}, {"Sym": {"name": "x"}}]},
        "derivation": {
            "root": 1,
            "steps": [[1, {"claim": {"Equality": [{"Sym": {"name": "x"}}, {"Sym": {"name": "x"}}]}, "rule": "Reflexivity", "premises": []}]]
        },
        "worker_signature": [1, 2, 3],
        "injected_extra_field": "exploit"
    }"#;

    let res = RemoteCandidate::decode_json(unknown_field_json.as_bytes());
    assert_eq!(res.unwrap_err(), RemoteWorkerError::CorruptedPayload);

    // 2. Oversized payload rejected before full parse
    let huge_payload = vec![b' '; fsym_runtime::MAX_REMOTE_CANDIDATE_BYTES + 1];
    let res_huge = RemoteCandidate::decode_json(&huge_payload);
    assert!(matches!(
        res_huge.unwrap_err(),
        RemoteWorkerError::PayloadTooLarge { .. }
    ));
}

// ============================================================================
// 4. Semantic Knowledge Graph Indexing & Rebuildability
// ============================================================================

#[test]
fn test_semantic_graph_indexing_and_rebuildability() {
    let mut index = SemanticGraphIndex::new();

    // Add nodes across multiple kinds
    index.add_node("ws_main", NodeKind::Workspace);
    index.add_node("ws_feature", NodeKind::Workspace);
    index.add_node("sym_x", NodeKind::Symbol);
    index.add_node("sym_y", NodeKind::Symbol);
    index.add_node("thm_pythagoras", NodeKind::Theorem);
    index.add_node("deriv_step1", NodeKind::Derivation);

    // Add dependency edges
    index.add_edge("ws_main", "ws_feature");
    index.add_edge("ws_feature", "thm_pythagoras");
    index.add_edge("thm_pythagoras", "deriv_step1");
    index.add_edge("deriv_step1", "sym_x");
    index.add_edge("deriv_step1", "sym_y");

    // Transitive reachability
    let reach = index.transitive_dependencies("ws_main");
    assert!(reach.contains("ws_feature"));
    assert!(reach.contains("thm_pythagoras"));
    assert!(reach.contains("deriv_step1"));
    assert!(reach.contains("sym_x"));
    assert!(reach.contains("sym_y"));
    assert_eq!(reach.len(), 5);

    // Acyclic check
    assert!(!index.has_cycle(), "Graph must be acyclic");

    // Serialization and rebuildability
    let serialized = serde_json::to_string(&index).expect("graph serializes");
    let rebuilt: SemanticGraphIndex =
        serde_json::from_str(&serialized).expect("graph deserializes");

    assert_eq!(
        index, rebuilt,
        "Rebuilt graph must match original bit-for-bit"
    );
    assert_eq!(rebuilt.transitive_dependencies("ws_main"), reach);
    assert!(!rebuilt.has_cycle());

    // Cycle detection
    let mut cyclic_index = index.clone();
    cyclic_index.add_edge("sym_x", "ws_main"); // Creates cycle
    assert!(cyclic_index.has_cycle(), "Cycle must be detected");
}
