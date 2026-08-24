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
pub struct PortfolioCandidate {
    pub strategy_name: String,
    pub result: fsym_core::Expr,
    pub claim: Claim,
    pub derivation: DerivationTree,
    pub steps_consumed: u64,
}

/// A verified accepted outcome from a portfolio execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPortfolioOutcome {
    pub winning_strategy: String,
    pub result: fsym_core::Expr,
    pub evidence: EvidenceEnvelope,
    pub total_steps_consumed: u64,
}

/// Function signature for a candidate strategy runner.
pub type StrategyRunner<Caps> =
    Box<dyn Fn(&mut FsymCx<'_, Caps>) -> Result<PortfolioCandidate, PortfolioError>>;

/// A named strategy runner pair.
pub type NamedStrategy<Caps> = (&'static str, StrategyRunner<Caps>);

/// Executes a portfolio race between two or more candidate generation strategies,
/// followed by mandatory protected verification of the winning candidate before publication.
pub fn run_portfolio_race<Caps>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    strategies: Vec<NamedStrategy<Caps>>,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    let mut first_winner: Option<PortfolioCandidate> = None;
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

    let verified_claim =
        verify_derivation_independent(&winner.derivation, context).map_err(|e| {
            PortfolioError::WinnerVerificationFailed(format!(
                "Verifier rejected candidate from strategy `{}`: {e}",
                winner.strategy_name
            ))
        })?;
    if verified_claim != winner.claim {
        return Err(PortfolioError::WinnerVerificationFailed(format!(
            "Verifier established `{verified_claim}`, but strategy `{}` requested publication of `{}`",
            winner.strategy_name, winner.claim
        )));
    }
    if portfolio_claimed_result(&verified_claim) != &winner.result {
        return Err(PortfolioError::WinnerVerificationFailed(format!(
            "Verified claim `{verified_claim}` does not bind the result returned by strategy `{}`",
            winner.strategy_name
        )));
    }

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

fn portfolio_claimed_result(claim: &Claim) -> &fsym_core::Expr {
    match claim {
        Claim::Equality { rhs, .. } | Claim::AlgebraicIdentity { rhs, .. } => rhs,
        Claim::PredicateHold { expr, .. }
        | Claim::DomainMembership { expr, .. }
        | Claim::NonZero(expr) => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::Cx;
    use fsym_budget::Unbounded;
    use fsym_core::Expr;
    use fsym_proof_kernel::ProofKernel;

    #[test]
    fn rejects_claim_that_is_not_the_verified_derivation_root() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);

        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        let mismatched_strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
            let derivation = kernel.export_derivation(step).unwrap();

            Ok(PortfolioCandidate {
                strategy_name: "mismatched_claim".into(),
                result: y.clone(),
                claim: Claim::equality(x.clone(), y.clone()),
                derivation,
                steps_consumed: 1,
            })
        });

        let result = run_portfolio_race(
            &mut fsym_cx,
            &context,
            vec![("mismatched", mismatched_strategy)],
        );

        assert!(matches!(
            result,
            Err(PortfolioError::WinnerVerificationFailed(message))
                if message.contains("requested publication")
        ));
    }

    #[test]
    fn rejects_result_that_is_not_bound_to_the_verified_claim() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);

        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let incorrect_result_strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
            let derivation = kernel.export_derivation(step).unwrap();

            Ok(PortfolioCandidate {
                strategy_name: "incorrect_result".into(),
                result: Expr::from_i64(999),
                claim: Claim::equality(x.clone(), x.clone()),
                derivation,
                steps_consumed: 1,
            })
        });

        let result = run_portfolio_race(
            &mut fsym_cx,
            &context,
            vec![("incorrect-result", incorrect_result_strategy)],
        );

        assert!(matches!(
            result,
            Err(PortfolioError::WinnerVerificationFailed(message))
                if message.contains("does not bind the result")
        ));
    }
}
