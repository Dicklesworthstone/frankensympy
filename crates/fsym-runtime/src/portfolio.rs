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
    Claim, DerivationTree, claim_verification_units, derivation_verification_units,
    expression_verification_units, verify_derivation_independent,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

const MAX_CANDIDATE_STRATEGY_NAME_BYTES: usize = 256;
const MAX_PORTFOLIO_STRATEGIES: usize = 64;

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
    #[error("Invalid portfolio configuration: {0}")]
    InvalidPortfolio(String),
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
    winning_strategy: String,
    result: fsym_core::Expr,
    evidence: EvidenceEnvelope,
    context_digest: [u8; 32],
    total_steps_consumed: u64,
}

impl VerifiedPortfolioOutcome {
    pub fn winning_strategy(&self) -> &str {
        &self.winning_strategy
    }

    pub fn result(&self) -> &fsym_core::Expr {
        &self.result
    }

    pub fn evidence(&self) -> &EvidenceEnvelope {
        &self.evidence
    }

    pub fn context_digest(&self) -> [u8; 32] {
        self.context_digest
    }

    /// Generator compute steps consumed across all attempted strategies.
    /// Protected verifier units are reported by their separate budget pool.
    pub fn generator_steps_consumed(&self) -> u64 {
        self.total_steps_consumed
    }
}

/// Function signature for a candidate strategy runner.
pub type StrategyRunner<Caps> =
    Box<dyn Fn(&mut FsymCx<'_, Caps>) -> Result<PortfolioCandidate, PortfolioError>>;

/// A named strategy runner pair.
pub type NamedStrategy<Caps> = (&'static str, StrategyRunner<Caps>);

/// Named concurrent strategy runner pair for parallel racing.
pub type ConcurrentStrategyRunner = Box<
    dyn Fn(&mut FsymCx<'_, asupersync::cx::cap::None>) -> Result<PortfolioCandidate, PortfolioError>
        + Send
        + Sync,
>;

/// A named concurrent strategy runner pair.
pub type NamedConcurrentStrategy = (&'static str, ConcurrentStrategyRunner);

/// Performs the shared protected verification, claim binding, receipt issuance, and envelope construction
/// on a generated candidate.
fn verify_and_publish_candidate<Caps>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    name: &str,
    winner: PortfolioCandidate,
    initial_compute_remaining: u64,
) -> Result<VerifiedPortfolioOutcome, String> {
    if winner.strategy_name.is_empty()
        || winner.strategy_name.len() > MAX_CANDIDATE_STRATEGY_NAME_BYTES
    {
        return Err(format!(
            "{name}: candidate strategy name must contain 1..={MAX_CANDIDATE_STRATEGY_NAME_BYTES} bytes"
        ));
    }
    let verifier_units = match (
        claim_verification_units(&winner.claim),
        expression_verification_units(&winner.result),
        derivation_verification_units(&winner.derivation),
    ) {
        (Ok(claim_units), Ok(result_units), Ok(derivation_units)) => claim_units
            .checked_add(result_units)
            .and_then(|units| units.checked_add(derivation_units))
            .unwrap_or(u64::MAX),
        (Err(error), _, _) => {
            return Err(format!(
                "{name}: verifier preflight rejected candidate from `{}`: {error}",
                winner.strategy_name
            ));
        }
        (_, Err(error), _) | (_, _, Err(error)) => {
            return Err(format!(
                "{name}: verifier preflight rejected candidate from `{}`: {error}",
                winner.strategy_name
            ));
        }
    };
    let mut verifier_charge = cx
        .charge_verifier(1)
        .map_err(|e| format!("{name}: verifier charge exhausted: {e}"))?;
    if verifier_units > 1 {
        verifier_charge = cx
            .charge_verifier(verifier_units - 1)
            .map_err(|e| format!("{name}: verifier charge exhausted: {e}"))?;
    }
    cx.checkpoint()
        .map_err(|_| format!("{name}: cancelled during verifier charging"))?;

    let verified_claim = match verify_derivation_independent(&winner.derivation, context) {
        Ok(claim) => claim,
        Err(error) => {
            return Err(format!(
                "{name}: verifier rejected candidate from `{}`: {error}",
                winner.strategy_name
            ));
        }
    };
    if verified_claim != winner.claim {
        return Err(format!(
            "{name}: verifier established `{verified_claim}`, but `{}` requested publication of `{}`",
            winner.strategy_name, winner.claim
        ));
    }
    if &verified_claim != requested_claim {
        return Err(format!(
            "{name}: verified claim `{verified_claim}` does not answer requested claim `{requested_claim}`"
        ));
    }
    if portfolio_claimed_result(&verified_claim) != &winner.result {
        return Err(format!(
            "{name}: verified claim `{verified_claim}` does not bind the result returned by `{}`",
            winner.strategy_name
        ));
    }
    cx.checkpoint()
        .map_err(|_| format!("{name}: cancelled after claim verification"))?;

    let receipt_id = fsym_id::ReceiptId::new(verifier_charge.seq())
        .map_err(|error| format!("{name}: invalid verifier receipt sequence: {error}"))?;
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
    if !evidence.verify_integrity() {
        return Err(format!(
            "{name}: verified candidate produced an invalid structural evidence envelope"
        ));
    }

    let compute_remaining = cx.remaining(Dimension::ComputeSteps);
    let total_steps_consumed = initial_compute_remaining
        .checked_sub(compute_remaining)
        .ok_or_else(|| {
            format!(
                "{name}: compute allowance increased from {initial_compute_remaining} to {compute_remaining}"
            )
        })?;
    Ok(VerifiedPortfolioOutcome {
        winning_strategy: winner.strategy_name,
        result: winner.result,
        evidence,
        context_digest: context.digest(),
        total_steps_consumed,
    })
}

/// Executes a sequential portfolio fallback race between candidate generation strategies,
/// followed by mandatory protected verification of the winning candidate before publication.
pub fn run_portfolio_race<Caps>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    strategies: Vec<NamedStrategy<Caps>>,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    if strategies.is_empty() || strategies.len() > MAX_PORTFOLIO_STRATEGIES {
        return Err(PortfolioError::InvalidPortfolio(format!(
            "strategy count must be in 1..={MAX_PORTFOLIO_STRATEGIES}"
        )));
    }
    if strategies
        .iter()
        .any(|(name, _)| name.is_empty() || name.len() > MAX_CANDIDATE_STRATEGY_NAME_BYTES)
    {
        return Err(PortfolioError::InvalidPortfolio(format!(
            "registered strategy names must contain 1..={MAX_CANDIDATE_STRATEGY_NAME_BYTES} bytes"
        )));
    }

    if !cx.has_verifier_authority() {
        return Err(PortfolioError::NoVerifierLease);
    }
    // Bound and pay for the caller's requested claim before any candidate is compared against
    // it. Otherwise an oversized request could bypass the candidate preflight and make the final
    // equality comparison itself the unmetered trust boundary.
    let _requested_preflight_charge = cx
        .charge_verifier(1)
        .map_err(|error| PortfolioError::BudgetExhausted(error.to_string()))?;
    let requested_claim_units = claim_verification_units(requested_claim).map_err(|error| {
        PortfolioError::WinnerVerificationFailed(format!(
            "requested claim failed verifier preflight: {error}"
        ))
    })?;
    if requested_claim_units > 1 {
        let _requested_remainder_charge = cx
            .charge_verifier(requested_claim_units - 1)
            .map_err(|error| PortfolioError::BudgetExhausted(error.to_string()))?;
    }
    let initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);
    let mut failure_reasons: Vec<String> = Vec::new();
    let mut verification_attempted = false;

    // Candidate Generation Phase: each strategy gets a real reserved child ledger.
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

        match verify_and_publish_candidate(
            cx,
            context,
            requested_claim,
            name,
            winner,
            initial_compute_remaining,
        ) {
            Ok(outcome) => return Ok(outcome),
            Err(err_msg) => {
                failure_reasons.push(err_msg);
            }
        }
    }

    let failures = failure_reasons.join("; ");
    if verification_attempted {
        Err(PortfolioError::WinnerVerificationFailed(failures))
    } else {
        Err(PortfolioError::AllStrategiesFailed(failures))
    }
}

/// Executes a concurrent portfolio race between two or more candidate generation strategies
/// using asupersync scoped CPU execution, with zero-orphan drain semantics and mandatory protected
/// verification of the winning candidate before publication.
pub fn run_portfolio_concurrent_race<Caps: Send + Sync + 'static>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    strategies: Vec<NamedConcurrentStrategy>,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    if strategies.is_empty() || strategies.len() > MAX_PORTFOLIO_STRATEGIES {
        return Err(PortfolioError::InvalidPortfolio(format!(
            "strategy count must be in 1..={MAX_PORTFOLIO_STRATEGIES}"
        )));
    }
    if strategies
        .iter()
        .any(|(name, _)| name.is_empty() || name.len() > MAX_CANDIDATE_STRATEGY_NAME_BYTES)
    {
        return Err(PortfolioError::InvalidPortfolio(format!(
            "registered strategy names must contain 1..={MAX_CANDIDATE_STRATEGY_NAME_BYTES} bytes"
        )));
    }

    if !cx.has_verifier_authority() {
        return Err(PortfolioError::NoVerifierLease);
    }
    let _requested_preflight_charge = cx
        .charge_verifier(1)
        .map_err(|error| PortfolioError::BudgetExhausted(error.to_string()))?;
    let requested_claim_units = claim_verification_units(requested_claim).map_err(|error| {
        PortfolioError::WinnerVerificationFailed(format!(
            "requested claim failed verifier preflight: {error}"
        ))
    })?;
    if requested_claim_units > 1 {
        let _requested_remainder_charge = cx
            .charge_verifier(requested_claim_units - 1)
            .map_err(|error| PortfolioError::BudgetExhausted(error.to_string()))?;
    }
    let initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);

    let num_strategies = strategies.len();
    let mut child_dim_limits = [0; fsym_budget::DIMENSION_COUNT];
    for dim in Dimension::ALL {
        let total_avail = cx.remaining(dim);
        child_dim_limits[dim.index()] = total_avail / (num_strategies as u64);
    }
    let child_limits = BudgetLimits {
        dimensions: child_dim_limits,
        verifier_pool: 0,
    };

    let mut child_budgets = Vec::with_capacity(num_strategies);
    for _ in 0..num_strategies {
        let child = cx
            .reserve_child(child_limits)
            .map_err(|error| PortfolioError::BudgetAccountingFailed(error.to_string()))?;
        child_budgets.push(child.into_budget());
    }

    struct WorkerOutcome {
        idx: usize,
        name: &'static str,
        result: Result<PortfolioCandidate, PortfolioError>,
        budget: fsym_budget::Budget,
    }

    let shared_outcomes: Arc<std::sync::Mutex<Vec<WorkerOutcome>>> =
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(num_strategies)));
    let winner_found = Arc::new(AtomicBool::new(false));

    let detached_cancel_cx = asupersync::Cx::detached_cancel_context();

    let scoped_result = cx.asupersync().scoped_cpu(num_strategies, |scope| {
        for (i, ((name, strategy), budget)) in strategies.into_iter().zip(child_budgets).enumerate()
        {
            let outcomes_ref = Arc::clone(&shared_outcomes);
            let winner_found_ref = Arc::clone(&winner_found);
            let detached_ref = &detached_cancel_cx;

            let _ = scope.spawn(move |cpu_child| {
                let mut worker_cx = FsymCx::new(detached_ref, budget, child_limits);
                if winner_found_ref.load(Ordering::Acquire) || cpu_child.checkpoint().is_err() {
                    let mut lock = outcomes_ref.lock().unwrap();
                    lock.push(WorkerOutcome {
                        idx: i,
                        name,
                        result: Err(PortfolioError::Cancelled),
                        budget: worker_cx.into_budget(),
                    });
                    return;
                }

                let gen_result = strategy(&mut worker_cx);
                if gen_result.is_ok() {
                    winner_found_ref.store(true, Ordering::Release);
                }
                let mut lock = outcomes_ref.lock().unwrap();
                lock.push(WorkerOutcome {
                    idx: i,
                    name,
                    result: gen_result,
                    budget: worker_cx.into_budget(),
                });
            });
        }
    });

    let mut outcomes = std::mem::take(&mut *shared_outcomes.lock().unwrap());

    // Sort outcomes by original strategy registration index to maintain deterministic winner priority
    outcomes.sort_by_key(|o| o.idx);

    // Merge child budgets back so work is charged
    for outcome in &mut outcomes {
        // Replace with dummy empty budget while transferring ownership to parent
        let dummy = fsym_budget::Budget::new(BudgetLimits::uniform(0, 0));
        let outcome_budget = std::mem::replace(&mut outcome.budget, dummy);
        cx.merge_child_budget(outcome_budget)
            .map_err(|error| PortfolioError::BudgetAccountingFailed(error.to_string()))?;
    }

    if let Err(_e) = scoped_result {
        return Err(PortfolioError::Cancelled);
    }

    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    let mut failure_reasons: Vec<String> = Vec::new();
    let mut verification_attempted = false;

    for outcome in outcomes {
        let winner = match outcome.result {
            Ok(c) => c,
            Err(e) => {
                failure_reasons.push(format!("{}: {e}", outcome.name));
                continue;
            }
        };
        verification_attempted = true;
        match verify_and_publish_candidate(
            cx,
            context,
            requested_claim,
            outcome.name,
            winner,
            initial_compute_remaining,
        ) {
            Ok(verified_outcome) => return Ok(verified_outcome),
            Err(err_msg) => {
                failure_reasons.push(err_msg);
            }
        }
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
    use std::sync::atomic::{AtomicBool, Ordering};

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

        assert_eq!(outcome.winning_strategy(), "accepted");
        assert_eq!(outcome.result(), &x);
        assert_eq!(outcome.generator_steps_consumed(), 5);
        assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 95);
        assert_eq!(fsym_cx.verifier_remaining(), 3);
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
        assert_eq!(fsym_cx.verifier_remaining(), 9);
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
                let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
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
        assert_eq!(fsym_cx.verifier_remaining(), 2);
    }

    #[test]
    fn oversized_requested_claim_is_a_paid_refusal_before_generation() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut deep = Expr::symbol("x");
        for _ in 0..300 {
            deep = Expr::Add(vec![deep]);
        }
        let requested = Claim::equality(deep.clone(), deep);

        let generation_ran = Arc::new(AtomicBool::new(false));
        let generation_ran_in_strategy = Arc::clone(&generation_ran);
        let strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            generation_ran_in_strategy.store(true, Ordering::SeqCst);
            Err(PortfolioError::AllStrategiesFailed(
                "unexpected generation".to_string(),
            ))
        });
        assert!(matches!(
            run_portfolio_race(
                &mut fsym_cx,
                &context,
                &requested,
                vec![("must-not-run", strategy)],
            ),
            Err(PortfolioError::WinnerVerificationFailed(message))
                if message.contains("requested claim failed verifier preflight")
        ));
        assert!(!generation_ran.load(Ordering::SeqCst));
        assert_eq!(fsym_cx.verifier_remaining(), 9);
    }

    #[test]
    fn empty_portfolio_is_a_typed_configuration_refusal() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");

        assert!(matches!(
            run_portfolio_race(
                &mut fsym_cx,
                &context,
                &Claim::equality(x.clone(), x),
                Vec::new(),
            ),
            Err(PortfolioError::InvalidPortfolio(_))
        ));
        assert_eq!(fsym_cx.verifier_remaining(), 10);
    }

    #[test]
    fn concurrent_race_winner_verified_and_published() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(200, 10);
        let budget = fsym_budget::Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let requested = Claim::equality(x.clone(), x.clone());

        let slow_x = x.clone();
        let slow_strategy = Box::new(move |cx: &mut FsymCx<'_, asupersync::cx::cap::None>| {
            cx.charge(Dimension::ComputeSteps, 5).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(slow_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "slow".into(),
                result: slow_x.clone(),
                claim: Claim::equality(slow_x.clone(), slow_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let fast_x = x.clone();
        let fast_strategy = Box::new(move |cx: &mut FsymCx<'_, asupersync::cx::cap::None>| {
            cx.charge(Dimension::ComputeSteps, 2).unwrap();
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(fast_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "fast".into(),
                result: fast_x.clone(),
                claim: Claim::equality(fast_x.clone(), fast_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let outcome = run_portfolio_concurrent_race(
            &mut fsym_cx,
            &context,
            &requested,
            vec![("slow", slow_strategy), ("fast", fast_strategy)],
        )
        .expect("concurrent race should produce a verified winner");

        assert!(outcome.winning_strategy() == "fast" || outcome.winning_strategy() == "slow");
        assert_eq!(outcome.result(), &x);
        assert_eq!(outcome.evidence().claim, requested);
        assert!(outcome.evidence().verify_integrity());
        assert!(fsym_cx.verifier_remaining() < 10);
    }

    #[test]
    fn concurrent_race_all_fail_returns_typed_error() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");

        let s1 = Box::new(|_cx: &mut FsymCx<'_, asupersync::cx::cap::None>| {
            Err(PortfolioError::AllStrategiesFailed("failed s1".into()))
        });
        let s2 = Box::new(|_cx: &mut FsymCx<'_, asupersync::cx::cap::None>| {
            Err(PortfolioError::AllStrategiesFailed("failed s2".into()))
        });

        let result = run_portfolio_concurrent_race(
            &mut fsym_cx,
            &context,
            &Claim::equality(x.clone(), x),
            vec![("s1", s1), ("s2", s2)],
        );

        assert!(matches!(
            result,
            Err(PortfolioError::AllStrategiesFailed(_))
        ));
    }
}
