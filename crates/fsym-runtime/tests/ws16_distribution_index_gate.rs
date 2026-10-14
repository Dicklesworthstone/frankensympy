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

// ============================================================================
// 5. FrankenGraphDB Projection Rebuild From Authoritative Workspace Artifacts
// ============================================================================

#[test]
fn test_frankengraphdb_projection_rebuild_from_authoritative_workspace() {
    use fsym_core::{Symbol, TermDag};
    use fsym_runtime::{
        AuthoritativeWorkspaceManifest, FrankenGraphDbProjectionAdapter, ReachabilityQueryResult,
        SemanticWorkspace, TestFixtureDestroyPolicy, verify_reachability_path,
        verify_topological_order,
    };
    use std::collections::BTreeSet;

    let mut ws = SemanticWorkspace::new("hero_branch");
    let x = Expr::symbol("x");
    let y = Expr::symbol("y");
    let f_expr = Expr::Add(vec![x.clone(), Expr::from_i64(1)]);
    let g_expr = Expr::Mul(vec![x.clone(), y.clone()]);
    let f_term_id_before = TermDag::new().insert_expr(&f_expr).unwrap();
    let g_term_id_before = TermDag::new().insert_expr(&g_expr).unwrap();

    ws.bind(Symbol::new("f"), f_expr.clone());
    ws.bind(Symbol::new("g"), g_expr.clone());

    let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
    let step = kernel
        .prove_reflexivity(f_expr.clone(), &mut Unbounded)
        .unwrap();
    let deriv = kernel.export_derivation(step).unwrap();
    ws.derivations.push(deriv);

    let manifest = AuthoritativeWorkspaceManifest::from_workspace(
        &ws,
        42,
        Some("workspace_proof_closure_v1".into()),
        BTreeSet::new(),
    );

    let test_dir = std::env::temp_dir().join(format!(
        "fsym_graph_proj_test_{}_{}",
        std::process::id(),
        42
    ));
    let mut adapter =
        FrankenGraphDbProjectionAdapter::create_isolated_test_adapter(&test_dir).unwrap();
    let proj_before = adapter
        .rebuild_from_authoritative(&manifest)
        .unwrap()
        .clone();

    // Verify TermId invariance
    assert_eq!(
        TermDag::new().insert_expr(&f_expr).unwrap(),
        f_term_id_before
    );
    assert_eq!(
        TermDag::new().insert_expr(&g_expr).unwrap(),
        g_term_id_before
    );

    // Query reachability and topological order before destroy
    let q_before = adapter
        .query_reachability(
            "ws:hero_branch",
            "sym:x",
            manifest.ingestion_watermark,
            manifest.source_manifest_root,
            manifest.universe_digest,
        )
        .unwrap();
    let ReachabilityQueryResult::Reachable {
        path: path_before,
        certificate: cert_before,
    } = &q_before
    else {
        panic!("expected Reachable path from ws:hero_branch to sym:x");
    };
    verify_reachability_path(&proj_before.index, cert_before).unwrap();
    let topo_before = proj_before.index.topological_order().unwrap();
    verify_topological_order(&proj_before.index, &topo_before).unwrap();

    // Destroy isolated test index under explicit policy
    let policy = TestFixtureDestroyPolicy::AllowIsolatedTestIndex {
        dir: test_dir.clone(),
    };
    adapter.destroy_isolated_test_index(&policy).unwrap();
    assert!(adapter.projection().is_none());

    // Rebuild from authoritative workspace manifest and reload from disk
    let proj_after = adapter
        .rebuild_from_authoritative(&manifest)
        .unwrap()
        .clone();
    let reloaded = FrankenGraphDbProjectionAdapter::load_from_disk(
        &test_dir,
        manifest.ingestion_watermark,
        manifest.source_manifest_root,
        manifest.universe_digest,
    )
    .unwrap();

    assert_eq!(proj_before, proj_after);
    assert_eq!(reloaded.projection().unwrap(), &proj_after);

    let q_after = reloaded
        .query_reachability(
            "ws:hero_branch",
            "sym:x",
            manifest.ingestion_watermark,
            manifest.source_manifest_root,
            manifest.universe_digest,
        )
        .unwrap();
    assert_eq!(q_before, q_after);
    assert_eq!(path_before.first().unwrap(), "ws:hero_branch");
    assert_eq!(path_before.last().unwrap(), "sym:x");
}

// ============================================================================
// 6. Graph Projection Adversaries & Coordinator Session Deduplication
// ============================================================================

#[test]
fn test_graph_projection_adversaries_staleness_context_malicious_private_unavailable() {
    use fsym_core::Symbol;
    use fsym_runtime::{
        AuthoritativeWorkspaceManifest, CoordinatorSession, FrankenGraphDbProjectionAdapter,
        GraphIndexError, ReachabilityQueryResult, SemanticWorkspace, TestFixtureDestroyPolicy,
    };
    use std::collections::BTreeSet;

    let mut ws = SemanticWorkspace::new("adv_branch");
    ws.bind(Symbol::new("pub_root"), Expr::symbol("secret_leaf"));
    ws.bind(Symbol::new("isolated_sym"), Expr::from_i64(7));

    let mut private_objs = BTreeSet::new();
    private_objs.insert("sym:secret_leaf".to_string());

    // Incomplete basis manifest (`None`) -> missing paths must return NotObserved, never DoesNotExist
    let manifest_incomplete =
        AuthoritativeWorkspaceManifest::from_workspace(&ws, 10, None, private_objs.clone());
    let mut adapter = FrankenGraphDbProjectionAdapter::new_in_memory();
    adapter
        .rebuild_from_authoritative(&manifest_incomplete)
        .unwrap();

    let missing_incomplete = adapter
        .query_reachability(
            "sym:isolated_sym",
            "sym:pub_root",
            manifest_incomplete.ingestion_watermark,
            manifest_incomplete.source_manifest_root,
            manifest_incomplete.universe_digest,
        )
        .unwrap();
    assert!(matches!(
        missing_incomplete,
        ReachabilityQueryResult::NotObserved { .. }
    ));

    // Complete basis manifest -> missing path between existing nodes returns DoesNotExist
    let manifest_complete = AuthoritativeWorkspaceManifest::from_workspace(
        &ws,
        11,
        Some("workspace_proof_closure_v1".into()),
        private_objs,
    );
    adapter
        .rebuild_from_authoritative(&manifest_complete)
        .unwrap();
    let missing_complete = adapter
        .query_reachability(
            "sym:isolated_sym",
            "sym:pub_root",
            manifest_complete.ingestion_watermark,
            manifest_complete.source_manifest_root,
            manifest_complete.universe_digest,
        )
        .unwrap();
    assert!(matches!(
        missing_complete,
        ReachabilityQueryResult::DoesNotExist { .. }
    ));

    // Stale watermark rejected
    assert!(matches!(
        adapter.query_reachability(
            "ws:adv_branch",
            "sym:pub_root",
            999,
            manifest_complete.source_manifest_root,
            manifest_complete.universe_digest,
        ),
        Err(GraphIndexError::StaleWatermark {
            found: 11,
            expected: 999
        })
    ));

    // Context mismatch rejected
    assert!(matches!(
        adapter.query_reachability(
            "ws:adv_branch",
            "sym:pub_root",
            manifest_complete.ingestion_watermark,
            manifest_complete.source_manifest_root,
            [0xFF; 32],
        ),
        Err(GraphIndexError::UniverseContextMismatch { .. })
    ));

    // Malicious index answer rejected
    assert!(matches!(
        adapter.verify_candidate_index_answer(
            "sym:isolated_sym",
            "sym:pub_root",
            vec!["sym:isolated_sym".into(), "sym:pub_root".into()],
        ),
        Err(GraphIndexError::MaliciousIndexAnswer(_))
    ));

    // Private object export denied when reachable from requested root
    assert_eq!(
        adapter.export_public_slice(&manifest_complete, &["sym:pub_root".into()]),
        Err(GraphIndexError::PrivateObjectExportDenied(
            "sym:secret_leaf".into()
        ))
    );
    // Public slice without private dependency succeeds
    let pub_slice = adapter
        .export_public_slice(&manifest_complete, &["sym:isolated_sym".into()])
        .unwrap();
    assert!(pub_slice.contains("sym:isolated_sym"));

    // Graph-unavailable operation degrades cleanly to NotObserved without affecting core workspace
    adapter.set_available(false);
    let unavail = adapter
        .query_reachability(
            "ws:adv_branch",
            "sym:pub_root",
            manifest_complete.ingestion_watermark,
            manifest_complete.source_manifest_root,
            manifest_complete.universe_digest,
        )
        .unwrap();
    assert!(matches!(
        unavail,
        ReachabilityQueryResult::NotObserved { .. }
    ));
    assert_eq!(
        ws.bindings.get(&Symbol::new("isolated_sym")),
        Some(&Expr::from_i64(7))
    );

    // Destroy policy refusal on non-isolated adapter
    let bad_policy = TestFixtureDestroyPolicy::AllowIsolatedTestIndex {
        dir: std::env::temp_dir().join("nonexistent_unmarked_dir"),
    };
    assert!(matches!(
        adapter.destroy_isolated_test_index(&bad_policy),
        Err(GraphIndexError::DestroyPolicyRefused(_))
    ));

    // CoordinatorSession refuses duplicate and late remote worker responses
    let x = Expr::symbol("x");
    let claim = Claim::equality(x.clone(), x.clone());
    let verifier =
        CoordinatorVerifier::new(505, claim.clone(), ImmutableAssumptionsSnapshot::empty());
    let mut session = CoordinatorSession::new(verifier);
    let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
    let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
    let cand = RemoteCandidate {
        worker_id: "w1".into(),
        task_id: 505,
        result: x.clone(),
        claim: claim.clone(),
        derivation: kernel.export_derivation(step).unwrap(),
        worker_signature: vec![1],
    };
    assert!(session.accept_once(&cand).is_ok());
    assert_eq!(
        session.accept_once(&cand),
        Err(RemoteWorkerError::DuplicateOrLateResponse { task_id: 505 })
    );

    let mut cancelled_session = CoordinatorSession::new(CoordinatorVerifier::new(
        506,
        claim,
        ImmutableAssumptionsSnapshot::empty(),
    ));
    cancelled_session.cancel_assignment();
    let mut late_cand = cand;
    late_cand.task_id = 506;
    assert_eq!(
        cancelled_session.accept_once(&late_cand),
        Err(RemoteWorkerError::DuplicateOrLateResponse { task_id: 506 })
    );
}
