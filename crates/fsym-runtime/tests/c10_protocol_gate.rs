//! Campaign Stage C10: Agent Protocol and Semantic Workspaces Gate.
//!
//! Constitutional and architectural requirements (WORKSTREAM_GRAPH.md §19, FIRST_IMPLEMENTATION_CAMPAIGN.md §15):
//! - NDJSON request/event/terminal protocol with strict resource bounds.
//! - Unknown and oversized schemas fail closed before state mutation.
//! - Semantic workspace fork, atomic patch, and proof-aware merge.
//! - Negative corpus: binding conflicts, assumption mismatches, and unverified derivations are rejected.
//! - Candidate vs accepted separation: untrusted remote candidates verified before acceptance.
//! - Transcript-free session replay reconstructs deterministic state bit-for-bit.

#![forbid(unsafe_code)]

use fsym_assumptions::{Domain, ImmutableAssumptionsSnapshot};
use fsym_budget::{Dimension, Unbounded};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, ProofKernel};
use fsym_runtime::{
    CoordinatorVerifier, MAX_AGENT_NDJSON_REQUEST_BYTES, RemoteCandidate, RemoteWorkerError,
    ReplayLog, SemanticWorkspace, WorkspaceError, WorkspacePatch, handle_agent_ndjson,
};
use std::collections::HashMap;

// ============================================================================
// 1. NDJSON Protocol & Schema Strictness (Fail Closed)
// ============================================================================

#[test]
fn test_c10_ndjson_protocol_wire_and_resource_bounds() {
    let mut ws = SemanticWorkspace::new("c10_agent");

    // 1. Valid Bind
    let bind_req = r#"{"type":"Bind","payload":{"symbol":"a","expr":"7"}}"#;
    let bind_resp = handle_agent_ndjson(bind_req, &mut ws);
    assert!(
        bind_resp.contains(r#""status":"Success""#),
        "Bind must succeed: {bind_resp}"
    );

    // 2. Valid Eval using bound symbol
    let eval_req = r#"{"type":"Eval","payload":{"expr":"a + 3"}}"#;
    let eval_resp = handle_agent_ndjson(eval_req, &mut ws);
    assert!(
        eval_resp.contains(r#""result":"10""#),
        "Eval must yield 10: {eval_resp}"
    );

    // 3. Valid Diff
    let diff_req = r#"{"type":"Diff","payload":{"expr":"x^3","var":"x"}}"#;
    let diff_resp = handle_agent_ndjson(diff_req, &mut ws);
    assert!(
        diff_resp.contains("3*x**2") || diff_resp.contains("3*x^2") || diff_resp.contains("3 * x"),
        "Diff must compute derivative: {diff_resp}"
    );

    // 4. Unknown schema fields fail closed before mutation
    let unknown_field_req =
        r#"{"type":"Bind","payload":{"symbol":"b","expr":"1","unsupported_extra_field":true}}"#;
    let unknown_resp = handle_agent_ndjson(unknown_field_req, &mut ws);
    assert!(
        unknown_resp.contains(r#""code":"malformed_request""#),
        "Unknown fields must fail closed: {unknown_resp}"
    );
    assert!(
        !ws.bindings.contains_key(&Symbol::new("b")),
        "Failed request must not mutate workspace"
    );

    // 5. Oversized envelope rejected before allocation
    let oversized_payload = " ".repeat(MAX_AGENT_NDJSON_REQUEST_BYTES + 10);
    let oversized_resp = handle_agent_ndjson(&oversized_payload, &mut ws);
    assert!(
        oversized_resp.contains(r#""code":"request_too_large""#),
        "Oversized payload must be rejected: {oversized_resp}"
    );

    // 6. Invalid symbol names rejected (constants / reserved words)
    for reserved in ["pi", "E", "I", "oo", "zoo", "nan"] {
        let req = format!(r#"{{"type":"Bind","payload":{{"symbol":"{reserved}","expr":"1"}}}}"#);
        let resp = handle_agent_ndjson(&req, &mut ws);
        assert!(
            resp.contains(r#""code":"invalid_name""#),
            "Reserved name {reserved} must be rejected: {resp}"
        );
    }
}

// ============================================================================
// 2. Workspace Fork & Atomic Semantic Patching
// ============================================================================

#[test]
fn test_c10_workspace_fork_and_patch_atomicity() {
    let mut main_ws = SemanticWorkspace::new("main");
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let z = Symbol::new("z");

    main_ws.bind(x.clone(), Expr::from_i64(10));
    main_ws.bind(y.clone(), Expr::from_i64(20));

    // Fork creates isolated child
    let mut child_ws = main_ws.fork("feature_patch");
    assert_eq!(child_ws.eval(&Expr::Sym(x.clone())), Expr::from_i64(10));

    // Valid patch: update y, insert z, remove x
    let mut updates = HashMap::new();
    updates.insert(y.clone(), Expr::from_i64(25));
    updates.insert(z.clone(), Expr::from_i64(30));
    let patch = WorkspacePatch {
        updated_bindings: updates,
        removed_bindings: vec![x.clone()],
    };

    child_ws
        .apply_patch(&patch)
        .expect("valid patch must apply");
    assert!(!child_ws.bindings.contains_key(&x));
    assert_eq!(child_ws.eval(&Expr::Sym(y.clone())), Expr::from_i64(25));
    assert_eq!(child_ws.eval(&Expr::Sym(z.clone())), Expr::from_i64(30));

    // Parent workspace remains unchanged (isolation)
    assert_eq!(main_ws.eval(&Expr::Sym(x.clone())), Expr::from_i64(10));
    assert_eq!(main_ws.eval(&Expr::Sym(y.clone())), Expr::from_i64(20));
    assert!(!main_ws.bindings.contains_key(&z));

    // Ambiguous patch: simultaneously update and remove the same symbol
    let mut bad_updates = HashMap::new();
    bad_updates.insert(y.clone(), Expr::from_i64(99));
    let bad_patch = WorkspacePatch {
        updated_bindings: bad_updates,
        removed_bindings: vec![y.clone()],
    };
    let err = child_ws
        .apply_patch(&bad_patch)
        .expect_err("ambiguous edit must fail");
    assert_eq!(
        err,
        WorkspaceError::AmbiguousBindingEdit { symbol: y.clone() }
    );
    // Verify atomicity: y is still 25, not 99 or removed
    assert_eq!(child_ws.eval(&Expr::Sym(y.clone())), Expr::from_i64(25));
}

// ============================================================================
// 3. Negative Corpus: Semantic Merge Conflicts & Verification Rejections
// ============================================================================

#[test]
fn test_c10_semantic_merge_negative_corpus() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");

    // Negative Case 1: Conflicting overlapping binding
    {
        let mut base = SemanticWorkspace::new("main");
        base.bind(x.clone(), Expr::from_i64(100));

        let mut branch = base.fork("branch_conflict");
        branch.bind(x.clone(), Expr::from_i64(200));

        let err = base
            .merge(&branch)
            .expect_err("conflicting binding must fail");
        assert_eq!(err, WorkspaceError::BindingConflict { symbol: x.clone() });
    }

    // Negative Case 2: Assumption context mismatch
    {
        let mut base = SemanticWorkspace::new("main");
        base.bind(x.clone(), Expr::from_i64(100));

        let mut branch = base.fork("branch_diff_domain");
        let mut domains = HashMap::new();
        domains.insert(x.clone(), Domain::QQ);
        let child_ctx = base
            .assumptions
            .derive_child(HashMap::new(), domains, "branch_diff_domain")
            .unwrap();
        branch.assumptions = child_ctx;

        let err = base
            .merge(&branch)
            .expect_err("mismatched assumptions must fail");
        assert!(matches!(
            err,
            WorkspaceError::AssumptionContextMismatch { .. }
        ));
    }

    // Negative Case 3: Derivation verification failure on unverified/tampered derivation
    {
        let mut base = SemanticWorkspace::new("main");
        base.bind(x.clone(), Expr::from_i64(100));

        let mut branch = base.fork("branch_with_derivation");
        branch.bind(y.clone(), Expr::from_i64(50));

        // Create a valid derivation
        let mut kernel = ProofKernel::new((*base.assumptions).clone());
        let step = kernel
            .prove_reflexivity(Expr::symbol("x"), &mut Unbounded)
            .unwrap();
        let mut tampered_deriv = kernel.export_derivation(step).unwrap();

        // Tamper with derivation root step ID so verification fails
        tampered_deriv.root = fsym_proof_kernel::StepId(9999);
        branch.derivations.push(tampered_deriv);

        let err = base
            .merge(&branch)
            .expect_err("unverified derivation must fail merge");
        assert_eq!(err, WorkspaceError::DerivationVerificationFailed);
    }

    // Positive Case: Clean merge with valid derivation and disjoint bindings
    {
        let mut base = SemanticWorkspace::new("main");
        base.bind(x.clone(), Expr::from_i64(100));

        let mut branch = base.fork("branch_clean");
        branch.bind(y.clone(), Expr::from_i64(50));

        let mut kernel = ProofKernel::new((*base.assumptions).clone());
        let step = kernel
            .prove_reflexivity(Expr::symbol("y"), &mut Unbounded)
            .unwrap();
        let valid_deriv = kernel.export_derivation(step).unwrap();
        branch.derivations.push(valid_deriv);

        let receipt = base.merge(&branch).expect("clean merge must succeed");
        assert_eq!(receipt.merged_bindings_count, 2);
        assert_eq!(base.derivations.len(), 1);
        assert_eq!(receipt.source_branch, "branch_clean");
        assert_eq!(receipt.target_branch, "main");
        assert_ne!(receipt.content_digest, [0u8; 32]);
    }
}

// ============================================================================
// 4. Candidate vs Accepted Separation (Remote Worker Verification)
// ============================================================================

#[test]
fn test_c10_candidate_vs_accepted_separation() {
    let context = ImmutableAssumptionsSnapshot::empty();
    let z = Expr::symbol("z");
    let task_id = 42;
    let coordinator =
        CoordinatorVerifier::new(task_id, Claim::equality(z.clone(), z.clone()), context);

    let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
    let step = kernel.prove_reflexivity(z.clone(), &mut Unbounded).unwrap();
    let derivation = kernel.export_derivation(step).unwrap();
    let claim = Claim::equality(z.clone(), z.clone());

    // 1. Legitimate remote candidate passes verification
    let valid_candidate = RemoteCandidate {
        worker_id: "trusted_worker".to_string(),
        task_id,
        result: z.clone(),
        claim: claim.clone(),
        derivation: derivation.clone(),
        worker_signature: vec![1, 2, 3, 4],
    };
    let verified = coordinator
        .verify_remote_candidate(&valid_candidate)
        .expect("valid candidate must verify");
    assert_eq!(verified.task_id(), task_id);

    // 2. Mismatched task ID rejected immediately
    let wrong_task_candidate = RemoteCandidate {
        task_id: 999, // Wrong task ID
        ..valid_candidate.clone()
    };
    let err = coordinator
        .verify_remote_candidate(&wrong_task_candidate)
        .expect_err("wrong task must fail");
    assert_eq!(err, RemoteWorkerError::TaskMismatch);

    // 3. Forged claim/result rejected (claim matches expected, but result does not match claimed result)
    let forged_result_candidate = RemoteCandidate {
        result: Expr::from_i64(999),
        ..valid_candidate.clone()
    };
    let err = coordinator
        .verify_remote_candidate(&forged_result_candidate)
        .expect_err("forged result must fail");
    assert_eq!(err, RemoteWorkerError::ClaimForgery);

    // 4. Tampered derivation rejected
    let mut tampered_deriv = derivation.clone();
    tampered_deriv.root = fsym_proof_kernel::StepId(8888);
    let tampered_deriv_candidate = RemoteCandidate {
        derivation: tampered_deriv,
        ..valid_candidate
    };
    let err = coordinator
        .verify_remote_candidate(&tampered_deriv_candidate)
        .expect_err("tampered proof must fail");
    assert!(matches!(err, RemoteWorkerError::VerificationFailed(_)));
}

// ============================================================================
// 5. Transcript-Free Session Replay Determinism
// ============================================================================

#[test]
fn test_c10_transcript_free_session_replay_determinism() {
    let seed = 987654321;
    let op_name = "algebraic_simplification_replay";

    let run_session = || -> (ReplayLog, [u8; 32]) {
        let mut log = ReplayLog::new(seed, op_name).expect("replay log init");
        log.record_event(
            "expand_terms",
            vec![(Dimension::ComputeSteps, 50)],
            b"step1_expand",
        )
        .expect("record event 1");
        log.record_event(
            "factor_subexpression",
            vec![(Dimension::ComputeSteps, 30)],
            b"step2_factor",
        )
        .expect("record event 2");
        log.record_event(
            "collect_terms",
            vec![(Dimension::ComputeSteps, 20)],
            b"step3_collect",
        )
        .expect("record event 3");
        let digest = log.finalize().expect("log finalize");
        (log, digest)
    };

    let (log1, digest1) = run_session();
    let (log2, digest2) = run_session();

    // Deterministic replay: digests match bit-for-bit without string transcript dependencies
    assert_eq!(
        digest1, digest2,
        "Deterministic execution must produce identical BLAKE3 digest"
    );
    assert!(
        log1.verify_replay_match(&log2),
        "Replay match verification must succeed"
    );

    // Mutation detection: tampered event payload is rejected
    let mut tampered_log = log1.clone();
    // Tamper with one event in the log
    if let Some(event) = tampered_log.events.get_mut(1) {
        event.payload[0] ^= 0xFF;
    }
    assert!(
        !tampered_log.verify_integrity(),
        "Tampered event payload must invalidate replay integrity"
    );
}
