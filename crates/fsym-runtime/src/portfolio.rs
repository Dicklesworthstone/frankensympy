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
use fsym_budget::{BudgetLimits, Dimension};
use fsym_evidence::EvidenceEnvelope;
use fsym_proof_kernel::{
    Claim, DerivationTree, derivation_verification_units, verify_derivation_independent,
};
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
    #[error("Child budget accounting failed: {0}")]
    BudgetAccountingFailed(String),
}

/// A candidate produced by an algorithm generator in the portfolio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioCandidate {
    pub strategy_name: String,
    pub result: fsym_core::Expr,
    pub claim: Claim,
    pub derivation: DerivationTree,
}

/// A verified accepted outcome from a portfolio execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPortfolioOutcome {
    pub winning_strategy: String,
    pub result: fsym_core::Expr,
    pub evidence: EvidenceEnvelope,
    pub context_digest: [u8; 32],
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
    requested_claim: &Claim,
    strategies: Vec<NamedStrategy<Caps>>,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    if !cx.has_verifier_authority() {
        return Err(PortfolioError::NoVerifierLease);
    }
    let initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);
    let mut failure_reasons: Vec<String> = Vec::new();
    let mut verification_attempted = false;

    // Candidate Generation Phase: each strategy gets a real reserved child ledger.
    // This baseline is deliberately sequential until the asupersync region race lands.
    for (name, strategy) in strategies {
        cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

        let child_limits = remaining_generator_limits(cx);
        let mut child_cx = cx
            .reserve_child(child_limits)
            .map_err(|error| PortfolioError::BudgetAccountingFailed(error.to_string()))?;
        let generated = strategy(&mut child_cx);
        cx.merge_child(child_cx)
            .map_err(|error| PortfolioError::BudgetAccountingFailed(error.to_string()))?;
        cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

        let winner = match generated {
            Ok(candidate) => candidate,
            Err(e) => {
                failure_reasons.push(format!("{name}: {e}"));
                continue;
            }
        };
        verification_attempted = true;

        // Charge the preflight itself before inspecting an untrusted candidate. A structurally
        // oversized derivation is therefore a paid rejection rather than a free verifier DoS.
        let mut verifier_charge = cx
            .charge_verifier(1)
            .map_err(|e| PortfolioError::BudgetExhausted(e.to_string()))?;
        let verifier_units = match derivation_verification_units(&winner.derivation) {
            Ok(units) => units,
            Err(error) => {
                failure_reasons.push(format!(
                    "{name}: verifier preflight rejected candidate from `{}`: {error}",
                    winner.strategy_name
                ));
                continue;
            }
        };
        if verifier_units > 1 {
            verifier_charge = cx
                .charge_verifier(verifier_units - 1)
                .map_err(|e| PortfolioError::BudgetExhausted(e.to_string()))?;
        }
        cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

        let verified_claim = match verify_derivation_independent(&winner.derivation, context) {
            Ok(claim) => claim,
            Err(error) => {
                failure_reasons.push(format!(
                    "{name}: verifier rejected candidate from `{}`: {error}",
                    winner.strategy_name
                ));
                continue;
            }
        };
        if verified_claim != winner.claim {
            failure_reasons.push(format!(
                "{name}: verifier established `{verified_claim}`, but `{}` requested publication of `{}`",
                winner.strategy_name, winner.claim
            ));
            continue;
        }
        if &verified_claim != requested_claim {
            failure_reasons.push(format!(
                "{name}: verified claim `{verified_claim}` does not answer requested claim `{requested_claim}`"
            ));
            continue;
        }
        if portfolio_claimed_result(&verified_claim) != &winner.result {
            failure_reasons.push(format!(
                "{name}: verified claim `{verified_claim}` does not bind the result returned by `{}`",
                winner.strategy_name
            ));
            continue;
        }
        cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

        let receipt_id = fsym_id::ReceiptId::new(verifier_charge.seq()).map_err(|error| {
            PortfolioError::WinnerVerificationFailed(format!(
                "invalid verifier receipt sequence: {error}"
            ))
        })?;
        let receipt = fsym_evidence::VerificationReceipt::issue(
            receipt_id,
            &winner.claim,
            fsym_outcome::EvidenceClass::KernelProved,
            format!("portfolio-verifier:{}", winner.strategy_name),
            verifier_charge.seq(),
            Some(winner.derivation.digest()),
        );

        let evidence = EvidenceEnvelope::new(
            winner.claim,
            fsym_outcome::EvidenceClass::KernelProved,
            receipt,
            Some(winner.derivation),
        );

        let compute_remaining = cx.remaining(Dimension::ComputeSteps);
        let total_steps_consumed = initial_compute_remaining
            .checked_sub(compute_remaining)
            .ok_or_else(|| {
                PortfolioError::BudgetAccountingFailed(format!(
                    "compute allowance increased from {initial_compute_remaining} to {compute_remaining}"
                ))
            })?;
        return Ok(VerifiedPortfolioOutcome {
            winning_strategy: winner.strategy_name,
            result: winner.result,
            evidence,
            context_digest: context.digest(),
            total_steps_consumed,
        });
    }

    let failures = failure_reasons.join("; ");
    if verification_attempted {
        Err(PortfolioError::WinnerVerificationFailed(failures))
    } else {
        Err(PortfolioError::AllStrategiesFailed(failures))
    }
}

fn remaining_generator_limits<Caps>(cx: &FsymCx<'_, Caps>) -> BudgetLimits {
    let mut dimensions = [0; fsym_budget::DIMENSION_COUNT];
    for dimension in Dimension::ALL {
        dimensions[dimension.index()] = cx.remaining(dimension);
    }
    BudgetLimits {
        dimensions,
        verifier_pool: 0,
    }
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
    use fsym_budget::{Budget, Unbounded};
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
        let requested = Claim::equality(x.clone(), y.clone());

        let mismatched_strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
            let derivation = kernel.export_derivation(step).unwrap();

            Ok(PortfolioCandidate {
                strategy_name: "mismatched_claim".into(),
                result: y.clone(),
                claim: Claim::equality(x.clone(), y.clone()),
                derivation,
            })
        });

        let result = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &requested,
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
        let requested = Claim::equality(x.clone(), x.clone());
        let incorrect_result_strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let step = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
            let derivation = kernel.export_derivation(step).unwrap();

            Ok(PortfolioCandidate {
                strategy_name: "incorrect_result".into(),
                result: Expr::from_i64(999),
                claim: Claim::equality(x.clone(), x.clone()),
                derivation,
            })
        });

        let result = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &requested,
            vec![("incorrect-result", incorrect_result_strategy)],
        );

        assert!(matches!(
            result,
            Err(PortfolioError::WinnerVerificationFailed(message))
                if message.contains("does not bind the result")
        ));
    }

    #[test]
    fn rejected_candidate_falls_back_without_refunding_consumed_work() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = fsym_budget::Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");

        let rejected_x = x.clone();
        let rejected = Box::new(move |cx: &mut FsymCx<'_, _>| {
            cx.charge(Dimension::ComputeSteps, 2).unwrap();
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(rejected_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "rejected-result".into(),
                result: Expr::from_i64(999),
                claim: Claim::equality(rejected_x.clone(), rejected_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let accepted_x = x.clone();
        let accepted = Box::new(move |cx: &mut FsymCx<'_, _>| {
            cx.charge(Dimension::ComputeSteps, 3).unwrap();
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(accepted_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "accepted".into(),
                result: accepted_x.clone(),
                claim: Claim::equality(accepted_x.clone(), accepted_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let outcome = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &Claim::equality(x.clone(), x.clone()),
            vec![("rejected", rejected), ("accepted", accepted)],
        )
        .unwrap();

        assert_eq!(outcome.winning_strategy, "accepted");
        assert_eq!(outcome.result, x);
        assert_eq!(outcome.total_steps_consumed, 5);
        assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 95);
        assert_eq!(fsym_cx.verifier_remaining(), 8);
    }

    #[test]
    fn fallback_never_resets_parent_budget() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = fsym_budget::Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");

        let first = Box::new(|cx: &mut FsymCx<'_, _>| {
            cx.charge(Dimension::ComputeSteps, 60).unwrap();
            Err(PortfolioError::AllStrategiesFailed(
                "planned first-strategy refusal".into(),
            ))
        });
        let second = Box::new(|cx: &mut FsymCx<'_, _>| {
            cx.charge(Dimension::ComputeSteps, 50)
                .map_err(|error| PortfolioError::BudgetExhausted(error.to_string()))?;
            Err(PortfolioError::AllStrategiesFailed(
                "unexpected charge success".into(),
            ))
        });

        let result = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &Claim::equality(x.clone(), x),
            vec![("first", first), ("second", second)],
        );

        assert!(matches!(
            result,
            Err(PortfolioError::AllStrategiesFailed(_))
        ));
        assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 40);
        assert_eq!(fsym_cx.verifier_remaining(), 10);
    }

    #[test]
    fn rejects_valid_but_irrelevant_claim() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let requested = Claim::equality(Expr::symbol("requested"), Expr::symbol("requested"));
        let irrelevant = Expr::symbol("irrelevant");

        let strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(irrelevant.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "irrelevant".into(),
                result: irrelevant.clone(),
                claim: Claim::equality(irrelevant.clone(), irrelevant.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let result = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &requested,
            vec![("irrelevant", strategy)],
        );

        assert!(matches!(
            result,
            Err(PortfolioError::WinnerVerificationFailed(message))
                if message.contains("does not answer requested claim")
        ));
    }

    #[test]
    fn repeated_portfolios_reuse_the_regions_single_verifier_capability() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let requested = Claim::equality(x.clone(), x.clone());

        for iteration in 0..2 {
            let candidate_x = x.clone();
            let strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
                let mut kernel =
                    ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
                let root = kernel
                    .prove_reflexivity(candidate_x.clone(), &mut Unbounded)
                    .unwrap();
                Ok(PortfolioCandidate {
                    strategy_name: format!("iteration-{iteration}"),
                    result: candidate_x.clone(),
                    claim: Claim::equality(candidate_x.clone(), candidate_x.clone()),
                    derivation: kernel.export_derivation(root).unwrap(),
                })
            });

            run_portfolio_race(
                &mut fsym_cx,
                &context,
                &requested,
                vec![("repeat", strategy)],
            )
            .expect("the verifier capability remains owned by the region");
        }
        assert_eq!(fsym_cx.verifier_remaining(), 8);
    }
}
