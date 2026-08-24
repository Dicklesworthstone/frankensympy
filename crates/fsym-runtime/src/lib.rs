//! # fsym-runtime
//!
//! Structured execution runtime, portfolios, candidate racing, winner verification,
//! resource bounding, replay logs, and typed checkpoints (WS13).

#![forbid(unsafe_code)]

pub mod benchmarks;
pub mod checkpoint;
pub mod cx;
pub mod graph_index;
pub mod ledger;
pub mod portfolio;
pub mod protocol;
pub mod remote_worker;
pub mod repair;
pub mod replay;
pub mod rng;
pub mod workspace;

pub use benchmarks::*;
pub use checkpoint::*;
pub use cx::FsymCx;
pub use graph_index::*;
pub use ledger::*;
pub use portfolio::*;
pub use protocol::*;
pub use remote_worker::*;
pub use repair::*;
pub use replay::*;
pub use workspace::*;

pub use fsym_budget::{Budget, BudgetLimits, ChargeReceipt, Dimension};
pub use fsym_outcome::{ExecutionOutcome, MathOutcome};

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RuntimeError {
    #[error("Evaluation budget exceeded: max steps {0}")]
    BudgetExceeded(usize),
    #[error("Recursion depth limit exceeded: {0}")]
    RecursionLimitExceeded(usize),
    #[error("Execution timed out after {0:?}")]
    Timeout(Duration),
    #[error("Hardened mode policy violation: {0}")]
    PolicyViolation(String),
}

/// Runtime execution mode split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeMode {
    /// Strict mode: exact SymPy compatibility, maximum observable equivalence.
    Strict,
    /// Hardened mode: defensive resource limits, fail-closed on unbounded recursion.
    Hardened,
}

/// Resource and execution budget configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeBudget {
    pub max_eval_steps: usize,
    pub max_recursion_depth: usize,
    pub timeout: Duration,
    pub mode: RuntimeMode,
}

impl Default for RuntimeBudget {
    fn default() -> Self {
        Self {
            max_eval_steps: 1_000_000,
            max_recursion_depth: 256,
            timeout: Duration::from_secs(30),
            mode: RuntimeMode::Strict,
        }
    }
}

/// Audit event emitted for CAS decision trail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionAuditEvent {
    pub action: String,
    pub algorithm: String,
    pub rationale: String,
    pub budget_consumed_steps: usize,
    pub payload_hash: String,
}

impl DecisionAuditEvent {
    pub fn new(
        action: impl Into<String>,
        algorithm: impl Into<String>,
        rationale: impl Into<String>,
        budget_consumed_steps: usize,
        data: &[u8],
    ) -> Self {
        let hash = blake3::hash(data).to_hex().to_string();
        Self {
            action: action.into(),
            algorithm: algorithm.into(),
            rationale: rationale.into(),
            budget_consumed_steps,
            payload_hash: hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::Cx;
    use fsym_assumptions::ImmutableAssumptionsSnapshot;
    use fsym_budget::Unbounded;
    use fsym_core::{Expr, Symbol};
    use fsym_proof_kernel::ProofKernel;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn test_audit_event_hash() {
        let ev = DecisionAuditEvent::new(
            "solve_polynomial",
            "quadratic_formula",
            "degree_2_exact",
            42,
            b"x^2 - 4 = 0",
        );
        assert_eq!(ev.action, "solve_polynomial");
        assert_eq!(ev.algorithm, "quadratic_formula");
        assert_eq!(ev.budget_consumed_steps, 42);
        assert!(!ev.payload_hash.is_empty());
    }

    #[test]
    fn test_typed_checkpoint_integrity() {
        let mut budget = BTreeMap::new();
        budget.insert(Dimension::ComputeSteps, 500);
        let checkpoint = TypedCheckpoint::new(1, "state_snapshot_42".to_string(), budget, 10);
        assert!(checkpoint.verify_integrity());
    }

    #[test]
    fn test_replay_log_reproduces_digest_bit_for_bit() {
        let mut log1 = ReplayLog::new(12345, "karatsuba_mul");
        log1.record_event("mul_step_1", vec![(Dimension::ComputeSteps, 10)], b"a*b");
        log1.record_event("mul_step_2", vec![(Dimension::ComputeSteps, 5)], b"res");
        let d1 = log1.finalize();

        let mut log2 = ReplayLog::new(12345, "karatsuba_mul");
        log2.record_event("mul_step_1", vec![(Dimension::ComputeSteps, 10)], b"a*b");
        log2.record_event("mul_step_2", vec![(Dimension::ComputeSteps, 5)], b"res");
        let d2 = log2.finalize();

        assert_eq!(d1, d2);
        assert!(log1.verify_replay_match(&log2));
    }

    #[test]
    fn test_portfolio_race_with_winner_verification() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);

        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let requested_claim = fsym_proof_kernel::Claim::equality(x.clone(), x.clone());

        let s1_expr = x.clone();
        let s2_expr = x.clone();

        let strategy1 = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let step = kernel
                .prove_reflexivity(s1_expr.clone(), &mut Unbounded)
                .unwrap();
            let derivation = kernel.export_derivation(step).unwrap();
            let claim = fsym_proof_kernel::Claim::equality(s1_expr.clone(), s1_expr.clone());
            Ok(PortfolioCandidate {
                strategy_name: "strategy_reflexive".into(),
                result: s1_expr.clone(),
                claim,
                derivation,
            })
        });

        let strategy2 = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let step = kernel
                .prove_reflexivity(s2_expr.clone(), &mut Unbounded)
                .unwrap();
            let derivation = kernel.export_derivation(step).unwrap();
            let claim = fsym_proof_kernel::Claim::equality(s2_expr.clone(), s2_expr.clone());
            Ok(PortfolioCandidate {
                strategy_name: "strategy_alternative".into(),
                result: s2_expr.clone(),
                claim,
                derivation,
            })
        });

        let outcome = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &requested_claim,
            vec![("strategy_1", strategy1), ("strategy_2", strategy2)],
        )
        .unwrap();

        assert_eq!(outcome.winning_strategy, "strategy_reflexive");
        assert_eq!(outcome.result, x);
        assert!(outcome.evidence.verify_integrity());
    }

    #[test]
    fn test_portfolio_rejects_forged_candidate_winner() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);

        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let requested_claim = fsym_proof_kernel::Claim::equality(x.clone(), y.clone());

        let forged_strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
            let mut derivation = kernel.export_derivation(step).unwrap();
            // Forge the claim: x = y (which is invalid for reflexivity rule on x)
            derivation.steps[0].claim = fsym_proof_kernel::Claim::equality(x.clone(), y.clone());

            Ok(PortfolioCandidate {
                strategy_name: "forged_strategy".into(),
                result: y.clone(),
                claim: fsym_proof_kernel::Claim::equality(x.clone(), y.clone()),
                derivation,
            })
        });

        let res = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &requested_claim,
            vec![("forged", forged_strategy)],
        );

        assert!(matches!(
            res,
            Err(PortfolioError::WinnerVerificationFailed(_))
        ));
    }

    #[test]
    fn test_raptorq_repair_recovery_and_digest_validation() {
        let payload =
            b"FrankenSymPy certified mathematical payload with exact proofs and receipts.".to_vec();
        let symbol_size = 16;
        let sidecar = RepairSidecar::encode(&payload, symbol_size, 4).unwrap();

        // Simulate 5 source symbols
        let mut source_symbols = Vec::new();
        for i in 0..sidecar.num_source_symbols {
            let start = i * symbol_size;
            let end = (start + symbol_size).min(payload.len());
            let mut sym = vec![0u8; symbol_size];
            sym[..(end - start)].copy_from_slice(&payload[start..end]);
            source_symbols.push(Some(sym));
        }

        // Drop symbol index 2 (packet loss)
        source_symbols[2] = None;

        // Reconstruct using repair sidecar
        let recovered = sidecar
            .reconstruct(&source_symbols, &sidecar.repair_symbols)
            .unwrap();
        assert_eq!(recovered, payload);

        // Test corruption detection: tamper a bit in a source symbol
        source_symbols[0].as_mut().unwrap()[0] ^= 0xFF;
        let corrupted_res = sidecar.reconstruct(&source_symbols, &sidecar.repair_symbols);
        assert_eq!(
            corrupted_res,
            Err(RepairError::SourceSymbolDigestMismatch(0))
        );
    }

    #[test]
    fn test_durable_ledger_hash_chain_and_checkpoints() {
        let mut ledger = DurableLedger::new();

        let mut budget = BTreeMap::new();
        budget.insert(Dimension::ComputeSteps, 500);
        let cp = TypedCheckpoint::new(0, "mathematical_checkpoint_payload".to_string(), budget, 50);

        let seq0 = ledger.append_checkpoint(&cp).unwrap();
        assert_eq!(seq0, 0);

        let seq1 = ledger.append(b"audit_event_1".to_vec());
        assert_eq!(seq1, 1);

        assert!(ledger.verify_chain());

        // Tamper with a record
        ledger.records[0].payload[0] ^= 0xFF;
        assert!(
            !ledger.verify_chain(),
            "Tampered ledger record must fail verification"
        );
    }

    #[test]
    fn test_workspace_fork_and_clean_merge() {
        let mut base = SemanticWorkspace::new("main");
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        base.bind(x.clone(), Expr::from_i64(10));

        let mut branch = base.fork("feature_a");
        branch.bind(y.clone(), Expr::from_i64(20));

        let receipt = base.merge(&branch).unwrap();
        assert_eq!(receipt.merged_bindings_count, 2);
        assert_eq!(base.eval(&Expr::Sym(x)), Expr::from_i64(10));
        assert_eq!(base.eval(&Expr::Sym(y)), Expr::from_i64(20));
    }

    #[test]
    fn test_negative_corpus_semantic_conflict_merge_rejected() {
        let mut base = SemanticWorkspace::new("main");
        let x = Symbol::new("x");
        base.bind(x.clone(), Expr::from_i64(10));

        let mut conflicting_branch = base.fork("conflict_branch");
        // Conflicting binding on same symbol x
        conflicting_branch.bind(x.clone(), Expr::from_i64(999));

        let merge_res = base.merge(&conflicting_branch);
        assert!(matches!(
            merge_res,
            Err(WorkspaceError::BindingConflict(name, e1, e2)) if name == "x" && e1 == "10" && e2 == "999"
        ));
    }

    #[test]
    fn test_ndjson_protocol_dispatch() {
        let mut ws = SemanticWorkspace::new("agent_session");

        // 1. Bind x = 5
        let req1 = r#"{"type":"Bind","payload":{"symbol":"x","expr":"5"}}"#;
        let resp1 = handle_agent_ndjson(req1, &mut ws);
        assert!(resp1.contains(r#""status":"Success""#));

        // 2. Evaluate x + 3
        let req2 = r#"{"type":"Eval","payload":{"expr":"x + 3"}}"#;
        let resp2 = handle_agent_ndjson(req2, &mut ws);
        assert!(resp2.contains(r#""result":"8""#));

        // 3. Diff x^2 with respect to x
        let req3 = r#"{"type":"Diff","payload":{"expr":"x^2","var":"x"}}"#;
        let resp3 = handle_agent_ndjson(req3, &mut ws);
        assert!(resp3.contains(r#"2*x"#) || resp3.contains(r#"2 * x"#));
    }

    #[test]
    fn test_remote_worker_candidate_verification_and_adversarial_rejection() {
        let context = ImmutableAssumptionsSnapshot::empty();
        let x = Expr::symbol("x");
        let coordinator = CoordinatorVerifier::new(
            101,
            fsym_proof_kernel::Claim::equality(x.clone(), x.clone()),
            context,
        );
        let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
        let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
        let derivation = kernel.export_derivation(step).unwrap();
        let claim = fsym_proof_kernel::Claim::equality(x.clone(), x.clone());

        let valid_candidate = RemoteCandidate {
            worker_id: "worker_node_42".to_string(),
            task_id: 101,
            result: x.clone(),
            claim: claim.clone(),
            derivation: derivation.clone(),
            worker_signature: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };

        // Valid candidate verified and accepted
        let verified = coordinator
            .verify_remote_candidate(&valid_candidate)
            .unwrap();
        assert_eq!(verified.task_id, 101);

        // Adversarial test: Worker signs a forged claim (x = 999) with non-matching derivation
        let forged_candidate = RemoteCandidate {
            worker_id: "malicious_worker".to_string(),
            task_id: 101,
            result: Expr::from_i64(999),
            claim: fsym_proof_kernel::Claim::equality(x.clone(), Expr::from_i64(999)),
            derivation,
            worker_signature: vec![0xCA, 0xFE, 0xBA, 0xBE],
        };

        let reject_res = coordinator.verify_remote_candidate(&forged_candidate);
        assert!(matches!(reject_res, Err(RemoteWorkerError::ClaimForgery)));
    }

    #[test]
    fn test_semantic_graph_indexing_and_cycle_detection() {
        let mut graph = SemanticGraphIndex::new();
        graph.add_node("ws_main", NodeKind::Workspace);
        graph.add_node("ws_algebra", NodeKind::Workspace);
        graph.add_node("ws_calculus", NodeKind::Workspace);
        graph.add_node("thm_pythagoras", NodeKind::Theorem);

        graph.add_edge("ws_main", "ws_algebra");
        graph.add_edge("ws_algebra", "thm_pythagoras");

        assert!(!graph.has_cycle());

        let deps = graph.transitive_dependencies("ws_main");
        assert!(deps.contains("ws_algebra"));
        assert!(deps.contains("thm_pythagoras"));

        // Add a cycle: thm_pythagoras -> ws_main
        graph.add_edge("thm_pythagoras", "ws_main");
        assert!(graph.has_cycle(), "Cycle must be detected in graph index");
    }

    #[test]
    fn test_paired_benchmark_with_semantic_admission() {
        // Paired run where candidate matches reference
        let res = run_paired_benchmark(
            "karatsuba_vs_naive_poly_mul",
            || (42, 10),
            || (42, 25),
            |c, r| c == r,
        )
        .unwrap();

        assert_eq!(res.benchmark_name, "karatsuba_vs_naive_poly_mul");
        assert!(res.semantic_equivalence_verified);
        assert_eq!(res.candidate_steps, 10);
        assert_eq!(res.reference_steps, 25);

        // Adversarial test: Semantic admission failure (diverging output) must fail closed
        let fail_res = run_paired_benchmark(
            "diverging_algorithm",
            || (42, 5),
            || (999, 10),
            |c, r| c == r,
        );

        assert!(matches!(
            fail_res,
            Err(BenchmarkError::SemanticAdmissionFailed)
        ));
    }

    #[test]
    fn test_standard_ws22_suite_execution() {
        let results = run_standard_ws22_suite().unwrap();
        assert_eq!(results.len(), 2);
        for res in results {
            assert!(res.semantic_equivalence_verified);
        }
    }
}
