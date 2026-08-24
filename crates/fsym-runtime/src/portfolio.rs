//! Structured portfolio execution, candidate racing, and winner verification (WS13).
//!
//! # Architecture Invariants (§7.7, §7.8)
//! - Multidimensional budgets with protected verifier reservation.
//! - Generators cannot consume verifier-reserved budget.
//! - Candidate generation and acceptance are separate phases: winner must pass independent verification before publication.
//! - Controlled cancellation: request -> drain -> finalize (zero orphan tasks).

#![forbid(unsafe_code)]

use crate::cx::FsymCx;
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Budget, BudgetLimits, Dimension};
use fsym_evidence::EvidenceEnvelope;
use fsym_proof_kernel::{Claim, DerivationTree, verify_derivation_independent};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PortfolioError {
    #[error("All candidate strategies failed or refused: {0}")]
    AllStrategiesFailed(String),
    #[error("Winner candidate failed independent verification: {0}")]
    WinnerVerificationFailed(String),
    #[error("Execution budget exhausted: {0}")]
    BudgetExhausted(String),
    #[error("Portfolio execution was cancelled by owning region")]
    Cancelled,
    #[error("No verifier lease available for protected verification")]
    NoVerifierLease,
}

/// A candidate produced by an algorithm generator in the portfolio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioCandidate<T> {
    pub strategy_name: String,
    pub result: T,
    pub claim: Claim,
    pub derivation: DerivationTree,
    pub steps_consumed: u64,
}

/// A verified accepted outcome from a portfolio execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPortfolioOutcome<T> {
    pub winning_strategy: String,
    pub result: T,
    pub evidence: EvidenceEnvelope,
    pub total_steps_consumed: u64,
}

/// Function signature for a candidate strategy runner.
pub type StrategyRunner<T, Caps> =
    Box<dyn Fn(&mut FsymCx<'_, Caps>) -> Result<PortfolioCandidate<T>, PortfolioError>>;

/// A named strategy runner pair.
pub type NamedStrategy<T, Caps> = (&'static str, StrategyRunner<T, Caps>);

/// Executes a portfolio race between two or more candidate generation strategies,
/// followed by mandatory protected verification of the winning candidate before publication.
pub fn run_portfolio_race<T: Clone, Caps>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    strategies: Vec<NamedStrategy<T, Caps>>,
) -> Result<VerifiedPortfolioOutcome<T>, PortfolioError> {
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    let mut first_winner: Option<PortfolioCandidate<T>> = None;
    let mut failure_reasons: Vec<String> = Vec::new();

    // Candidate Generation Phase: sequentially or concurrently execute strategies
    for (name, strategy) in strategies {
        if cx.check_cancelled() {
            return Err(PortfolioError::Cancelled);
        }

        // Subdivide budget for strategy candidate generation
        let child_limits = BudgetLimits::uniform(cx.remaining(Dimension::ComputeSteps), 0);
        let child_budget = Budget::new(child_limits);
        let mut child_cx = FsymCx::new(cx.asupersync(), child_budget, child_limits);

        match strategy(&mut child_cx) {
            Ok(candidate) => {
                first_winner = Some(candidate);
                break;
            }
            Err(e) => {
                failure_reasons.push(format!("{name}: {e}"));
            }
        }
    }

    let winner = first_winner
        .ok_or_else(|| PortfolioError::AllStrategiesFailed(failure_reasons.join("; ")))?;

    // Protected Verification Phase: obtain verifier lease and check derivation
    let lease = cx.verifier_lease().ok_or(PortfolioError::NoVerifierLease)?;
    cx.charge_verifier(&lease, 1)
        .map_err(|e| PortfolioError::BudgetExhausted(e.to_string()))?;

    verify_derivation_independent(&winner.derivation, context).map_err(|e| {
        PortfolioError::WinnerVerificationFailed(format!(
            "Verifier rejected candidate from strategy `{}`: {e}",
            winner.strategy_name
        ))
    })?;

    let receipt_id = fsym_id::ReceiptId::new(1).expect("valid id");
    let receipt = fsym_evidence::VerificationReceipt::issue(
        receipt_id,
        &winner.claim,
        fsym_outcome::EvidenceClass::KernelProved,
        format!("portfolio-verifier:{}", winner.strategy_name),
        1,
        Some(winner.derivation.digest()),
    );

    let evidence = EvidenceEnvelope::new(
        winner.claim,
        fsym_outcome::EvidenceClass::KernelProved,
        receipt,
        Some(winner.derivation),
    );

    Ok(VerifiedPortfolioOutcome {
        winning_strategy: winner.strategy_name,
        result: winner.result,
        evidence,
        total_steps_consumed: winner.steps_consumed,
    })
}
