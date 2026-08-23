//! # fsym-runtime
//!
//! Structured async execution runtime, resource bounding, recursion depth limiters,
//! audit ledgers, and strict vs. hardened execution mode policy.

#![forbid(unsafe_code)]

// Canonical budget/outcome types live in the fsym-budget and fsym-outcome
// crates (registry-aligned evidence classes, single-issuance verifier lease).
pub mod rng;

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

    #[test]
    fn test_audit_event_hash() {
        let ev = DecisionAuditEvent::new(
            "solve_polynomial",
            "quadratic_formula",
            "degree_2_monic",
            42,
            b"test_payload",
        );
        assert_eq!(ev.budget_consumed_steps, 42);
        assert!(!ev.payload_hash.is_empty());
    }
}
