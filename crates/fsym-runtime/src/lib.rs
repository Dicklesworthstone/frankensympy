//! # fsym-runtime
//!
//! Structured execution runtime, portfolios, candidate racing, winner verification,
//! resource bounding, replay logs, and typed checkpoints (WS13).

#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod cx;
pub mod portfolio;
pub mod replay;
pub mod rng;

pub use checkpoint::*;
pub use cx::FsymCx;
pub use portfolio::*;
pub use replay::*;

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
    use fsym_core::Expr;
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
                steps_consumed: 1,
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
                steps_consumed: 2,
            })
        });

        let outcome = run_portfolio_race(
            &mut fsym_cx,
            &context,
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
                steps_consumed: 1,
            })
        });

        let res = run_portfolio_race(&mut fsym_cx, &context, vec![("forged", forged_strategy)]);

        assert!(matches!(
            res,
            Err(PortfolioError::WinnerVerificationFailed(_))
        ));
    }
}
