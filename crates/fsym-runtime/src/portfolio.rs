//! Structured portfolio execution, candidate racing, and winner verification (WS13).
//!
//! # Architecture Invariants (§7.7, §7.8)
//! - Multidimensional budgets with protected verifier reservation.
//! - Generators cannot consume verifier-reserved budget.
//! - Candidate generation and acceptance are separate phases: winner must pass independent verification before publication.
//! - Controlled cancellation: request -> drain -> finalize (zero orphan tasks).

#![forbid(unsafe_code)]

use crate::checkpoint::TypedCheckpoint;
use crate::cx::{FsymCpuCx, FsymCx};
use asupersync::cx::ScopedCpuError;
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{BudgetLimits, Dimension};
use fsym_evidence::EvidenceEnvelope;
use fsym_proof_kernel::{
    Claim, DerivationTree, claim_verification_units, derivation_verification_units,
    expression_verification_units, verify_derivation_independent,
};
use std::sync::{Arc, Mutex};
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
    #[error("Structured portfolio execution failed: {0}")]
    StructuredExecutionFailed(String),
}

/// A candidate produced by an algorithm generator in the portfolio.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// Candidate generator accepted by the scoped parallel lane.
pub type ConcurrentStrategyRunner<Caps> = Box<
    dyn Fn(&mut FsymCpuCx<'_, Caps>) -> Result<PortfolioCandidate, PortfolioError> + Send + Sync,
>;

/// A registered strategy name and its scoped parallel candidate generator.
pub type NamedConcurrentStrategy<Caps> = (&'static str, ConcurrentStrategyRunner<Caps>);

#[cfg(test)]
thread_local! {
    // Verification/publication runs on the owner thread, not a generator worker.
    // Taking the hook before invocation prevents reentrancy and cross-test reuse.
    static BEFORE_PUBLICATION: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Performs the shared protected verification, claim binding, receipt issuance, and envelope construction
/// on a generated candidate.
fn verify_and_publish_candidate<Caps>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    name: &str,
    winner: PortfolioCandidate,
    initial_compute_remaining: u64,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
    if winner.strategy_name.is_empty()
        || winner.strategy_name.len() > MAX_CANDIDATE_STRATEGY_NAME_BYTES
    {
        return Err(PortfolioError::WinnerVerificationFailed(format!(
            "{name}: candidate strategy name must contain 1..={MAX_CANDIDATE_STRATEGY_NAME_BYTES} bytes"
        )));
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
            return Err(PortfolioError::WinnerVerificationFailed(format!(
                "{name}: verifier preflight rejected candidate from `{}`: {error}",
                winner.strategy_name
            )));
        }
        (_, Err(error), _) | (_, _, Err(error)) => {
            return Err(PortfolioError::WinnerVerificationFailed(format!(
                "{name}: verifier preflight rejected candidate from `{}`: {error}",
                winner.strategy_name
            )));
        }
    };
    let mut verifier_charge = cx.charge_verifier(1).map_err(|error| {
        PortfolioError::BudgetExhausted(format!("{name}: verifier charge exhausted: {error}"))
    })?;
    if verifier_units > 1 {
        verifier_charge = cx.charge_verifier(verifier_units - 1).map_err(|error| {
            PortfolioError::BudgetExhausted(format!("{name}: verifier charge exhausted: {error}"))
        })?;
    }
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    let verified_claim = match verify_derivation_independent(&winner.derivation, context) {
        Ok(claim) => claim,
        Err(_) => {
            return Err(PortfolioError::WinnerVerificationFailed(format!(
                "{name}: verifier rejected candidate from `{}`",
                winner.strategy_name
            )));
        }
    };
    if verified_claim != winner.claim {
        return Err(PortfolioError::WinnerVerificationFailed(format!(
            "{name}: candidate `{}` requested publication of a claim not established by its derivation",
            winner.strategy_name
        )));
    }
    if &verified_claim != requested_claim {
        return Err(PortfolioError::WinnerVerificationFailed(format!(
            "{name}: verified candidate `{}` does not answer requested claim",
            winner.strategy_name
        )));
    }
    if portfolio_claimed_result(&verified_claim) != &winner.result {
        return Err(PortfolioError::WinnerVerificationFailed(format!(
            "{name}: verified claim does not bind the result returned by `{}`",
            winner.strategy_name
        )));
    }
    // Publication boundary: verification has fully succeeded, but nothing is
    // published yet. This checkpoint is the sole post-verifier cancellation
    // seam: cancelling here must refuse publication while leaving the ledger
    // and any captured continuation otherwise intact.
    #[cfg(test)]
    if let Some(hook) = BEFORE_PUBLICATION.with(|slot| slot.borrow_mut().take()) {
        hook();
    }
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    let receipt_id = fsym_id::ReceiptId::new(verifier_charge.seq()).map_err(|error| {
        PortfolioError::StructuredExecutionFailed(format!(
            "{name}: invalid verifier receipt sequence: {error}"
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
    if !evidence.verify_integrity() {
        return Err(PortfolioError::StructuredExecutionFailed(format!(
            "{name}: verified candidate produced an invalid structural evidence envelope"
        )));
    }

    let compute_remaining = cx.remaining(Dimension::ComputeSteps);
    let total_steps_consumed = initial_compute_remaining
        .checked_sub(compute_remaining)
        .ok_or_else(|| {
            PortfolioError::BudgetAccountingFailed(format!(
                "{name}: compute allowance increased from {initial_compute_remaining} to {compute_remaining}"
            ))
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
        let generated =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| strategy(&mut child_cx)))
                .unwrap_or_else(|_| {
                    Err(PortfolioError::StructuredExecutionFailed(format!(
                        "{name}: candidate generator panicked"
                    )))
                });
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
            Err(PortfolioError::WinnerVerificationFailed(message)) => {
                failure_reasons.push(message);
            }
            Err(error) => return Err(error),
        }
    }

    let failures = failure_reasons.join("; ");
    if verification_attempted {
        Err(PortfolioError::WinnerVerificationFailed(failures))
    } else {
        Err(PortfolioError::AllStrategiesFailed(failures))
    }
}

/// Generates candidates from two or more strategies in an asupersync scoped CPU region, drains
/// every worker, then verifies candidates in deterministic registration order. Generator success
/// never cancels a sibling or selects a winner: only protected verification can do that.
pub fn run_portfolio_concurrent_race<Caps: Send + Sync + 'static>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    strategies: Vec<NamedConcurrentStrategy<Caps>>,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
    let initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);
    let outcomes = generate_concurrent_candidates(cx, requested_claim, strategies)?;
    accept_concurrent_candidates(
        cx,
        context,
        requested_claim,
        outcomes,
        initial_compute_remaining,
    )
}

pub(crate) struct WorkerOutcome {
    pub(crate) idx: usize,
    pub(crate) name: String,
    pub(crate) result: Result<PortfolioCandidate, PortfolioError>,
}

pub(crate) fn generate_concurrent_candidates<Caps: Send + Sync + 'static>(
    cx: &mut FsymCx<'_, Caps>,
    requested_claim: &Claim,
    strategies: Vec<NamedConcurrentStrategy<Caps>>,
) -> Result<Vec<WorkerOutcome>, PortfolioError> {
    cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;

    if strategies.len() < 2 || strategies.len() > MAX_PORTFOLIO_STRATEGIES {
        return Err(PortfolioError::InvalidPortfolio(format!(
            "concurrent strategy count must be in 2..={MAX_PORTFOLIO_STRATEGIES}"
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
        child_budgets.push(Arc::new(Mutex::new(child.into_budget())));
    }

    let shared_outcomes: Arc<Mutex<Vec<WorkerOutcome>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_strategies)));

    let scoped_result = cx.asupersync().scoped_cpu(num_strategies, |scope| {
        for (i, (name, strategy)) in strategies.into_iter().enumerate() {
            let outcomes_ref = Arc::clone(&shared_outcomes);
            let budget_ref = Arc::clone(&child_budgets[i]);

            if let Err(error) = scope.spawn(move |cpu_child| {
                let gen_result = {
                    let mut budget = budget_ref
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let mut worker_cx = FsymCpuCx::new(cpu_child, &mut budget, child_limits);
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        strategy(&mut worker_cx)
                    }))
                    .unwrap_or_else(|_| {
                        Err(PortfolioError::StructuredExecutionFailed(format!(
                            "{name}: candidate generator panicked"
                        )))
                    })
                };
                let mut outcomes = outcomes_ref
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                outcomes.push(WorkerOutcome {
                    idx: i,
                    name: name.to_owned(),
                    result: gen_result,
                });
            }) {
                let mut outcomes = shared_outcomes
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                outcomes.push(WorkerOutcome {
                    idx: i,
                    name: name.to_owned(),
                    result: Err(PortfolioError::StructuredExecutionFailed(format!(
                        "{name}: scoped CPU spawn refused: {error}"
                    ))),
                });
            }
        }
    });

    let mut outcomes = std::mem::take(
        &mut *shared_outcomes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );

    // Sort outcomes by original strategy registration index to maintain deterministic winner priority
    outcomes.sort_by_key(|o| o.idx);

    // The budget stays in this owned cell even if a generator panics, so every
    // reservation can be reconciled after the scoped region has drained.
    for budget in child_budgets {
        let budget = Arc::try_unwrap(budget)
            .map_err(|_| {
                PortfolioError::BudgetAccountingFailed(
                    "scoped CPU worker retained its child budget after drain".into(),
                )
            })?
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cx.merge_child_budget(budget)
            .map_err(|error| PortfolioError::BudgetAccountingFailed(error.to_string()))?;
    }

    match scoped_result {
        Ok(()) => {}
        Err(ScopedCpuError::Cancelled(_)) => return Err(PortfolioError::Cancelled),
        Err(error) => {
            return Err(PortfolioError::StructuredExecutionFailed(error.to_string()));
        }
    }
    if outcomes.len() != num_strategies {
        return Err(PortfolioError::StructuredExecutionFailed(format!(
            "scoped CPU region produced {} outcomes for {num_strategies} strategies",
            outcomes.len()
        )));
    }

    Ok(outcomes)
}

fn accept_concurrent_candidates<Caps>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    outcomes: Vec<WorkerOutcome>,
    initial_compute_remaining: u64,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
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
            &outcome.name,
            winner,
            initial_compute_remaining,
        ) {
            Ok(verified_outcome) => return Ok(verified_outcome),
            Err(PortfolioError::WinnerVerificationFailed(message)) => {
                failure_reasons.push(message);
            }
            Err(error) => return Err(error),
        }
    }

    let failures = failure_reasons.join("; ");
    if verification_attempted {
        Err(PortfolioError::WinnerVerificationFailed(failures))
    } else {
        Err(PortfolioError::AllStrategiesFailed(failures))
    }
}
/// Registered portfolio id this race integration is filed under.
pub const FACTOR_RACE_PORTFOLIO_ID: &str = "univariate_product_identity_v1";
pub const FACTOR_RACE_CONTINUATION_SCHEMA: &str = "fsym.portfolio.factor_race.continuation.v1";

/// Diagnostic decision card for one factor-race plan.
///
/// This is planning evidence only (docs/ALGORITHM_PORTFOLIOS.md §5): it records
/// the fixed launch set, the exact requested claim, and the protected verifier
/// reserve. It never authorizes a candidate or promotes evidence.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorRaceDecisionCard {
    pub portfolio_id: String,
    pub planner_version: String,
    /// BLAKE3 digest of the immutable input expression.
    pub input_digest: [u8; 32],
    /// Exact requested claim digest; never winner-derived.
    pub requested_claim_digest: [u8; 32],
    /// Fixed launch set in registration (acceptance priority) order.
    pub launch_order: Vec<String>,
    pub completion_policy: String,
    pub protected_verifier_reserve: u64,
}

impl FactorRaceDecisionCard {
    fn build(
        requested_claim: &Claim,
        strategies: &[NamedConcurrentStrategy<impl Send + Sync + 'static>],
        verifier_reserve: u64,
    ) -> Result<Self, PortfolioError> {
        if strategies.is_empty() {
            return Err(PortfolioError::InvalidPortfolio(
                "decision card requires at least one strategy".into(),
            ));
        }
        let Claim::AlgebraicIdentity { lhs, .. } = requested_claim else {
            return Err(PortfolioError::InvalidPortfolio(
                "factor race requests must be Claim::AlgebraicIdentity".into(),
            ));
        };
        let serialized = serde_json::to_vec(lhs).map_err(|error| {
            PortfolioError::InvalidPortfolio(format!("input serialization failed: {error}"))
        })?;
        Ok(Self {
            portfolio_id: FACTOR_RACE_PORTFOLIO_ID.into(),
            planner_version: "factor_race_v1".into(),
            input_digest: *blake3::hash(&serialized).as_bytes(),
            requested_claim_digest: requested_claim.digest(),
            launch_order: strategies.iter().map(|(name, _)| (*name).into()).collect(),
            completion_policy: "registration_order_strict".into(),
            protected_verifier_reserve: verifier_reserve,
        })
    }
}

/// Serialized state bound into a typed factor-race continuation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorRaceContinuationState {
    pub operation_version: String,
    /// The immutable input expression whose factor race produced these
    /// candidates. Bound to the decision-card input digest contract.
    pub input_expr: fsym_core::Expr,
    pub requested_claim: Claim,
    pub context_digest: [u8; 32],
    /// Drained candidates at the generator/verifier boundary in registration
    /// order. None has been accepted or published yet.
    pub candidates: Vec<FactorRaceCandidateRecord>,
}

impl FactorRaceContinuationState {
    pub const LEDGER_IDENTITY_SCHEMA: &str = "fsym.portfolio.factor_race.ledger.v1";
}

/// One drained candidate plus its original registration index, so resumed
/// acceptance preserves the exact fixed registration-order priority even when
/// a strategy name is not `&'static`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorRaceCandidateRecord {
    pub registration_index: usize,
    pub candidate: PortfolioCandidate,
}

/// Typed continuation over drained factor-race candidates.
///
/// The ledger is NOT part of this state: resume requires the same live owned
/// [`FsymCx`] region (Budget plus protected verifier lease). Checkpoint budget
/// counters are observational integrity copies only; they are never restored
/// into a ledger, and no ledger is ever reset.
pub struct FactorRaceContinuation {
    checkpoint: TypedCheckpoint<FactorRaceContinuationState>,
    /// Live capability captured from the owning ledger at generation drain
    /// time. Kept OUT of the wire payload: `from_wire` resumes require the
    /// caller to present the live ledger, and `resume` compares this
    /// capability against it. Wire bytes alone can never authorize a resume.
    /// The paired counter is the pre-generation baseline, never a restored allowance.
    ledger: Option<(fsym_budget::BudgetLedgerIdentity, u64)>,
}

impl FactorRaceContinuation {
    pub fn payload_schema(&self) -> &str {
        &self.checkpoint.payload_schema
    }

    pub fn verify_integrity(&self) -> bool {
        self.checkpoint.verify_integrity()
    }

    pub fn checkpoint_seq(&self) -> u64 {
        self.checkpoint.checkpoint_seq
    }

    /// Serialize the typed continuation for durable storage.
    pub fn to_wire(&self) -> Result<Vec<u8>, PortfolioError> {
        serde_json::to_vec(&self.checkpoint).map_err(|error| {
            PortfolioError::StructuredExecutionFailed(format!(
                "continuation serialization failed: {error}"
            ))
        })
    }

    /// Decode a continuation from durable bytes without trusting any counters.
    ///
    /// The decoded continuation carries NO ledger capability: wire bytes can
    /// never authorize a resume on their own. `resume` binds the decoded
    /// state to the live ledger the caller presents.
    pub fn from_wire(wire: &[u8]) -> Result<Self, PortfolioError> {
        let checkpoint: TypedCheckpoint<FactorRaceContinuationState> = serde_json::from_slice(wire)
            .map_err(|error| {
                PortfolioError::InvalidPortfolio(format!("continuation decode refused: {error}"))
            })?;
        Ok(Self {
            checkpoint,
            ledger: None,
        })
    }

    /// Resume acceptance of the drained candidates against the same live ledger.
    ///
    /// Refuses foreign ledgers (capability mismatch), tampered state, schema
    /// mismatches, and non-identical observational counters before any
    /// resumed work or charge.
    pub fn resume<Caps>(
        self,
        cx: &mut FsymCx<'_, Caps>,
        context: &Arc<ImmutableAssumptionsSnapshot>,
        requested_claim: &Claim,
    ) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
        if self.checkpoint.payload_schema != FACTOR_RACE_CONTINUATION_SCHEMA {
            return Err(PortfolioError::InvalidPortfolio(format!(
                "continuation schema mismatch: expected {FACTOR_RACE_CONTINUATION_SCHEMA}, got {}",
                self.checkpoint.payload_schema
            )));
        }
        if !self.checkpoint.verify_integrity() {
            return Err(PortfolioError::InvalidPortfolio(
                "continuation integrity verification failed; refusing foreign or tampered state"
                    .into(),
            ));
        }
        let live_ledger = cx.ledger_identity();
        let initial_compute_remaining = match &self.ledger {
            Some((captured, initial)) if *captured == live_ledger => *initial,
            _ => {
                return Err(PortfolioError::InvalidPortfolio(
                    "continuation was not captured by the presented live ledger; wire decode alone never authorizes a resume"
                        .into(),
                ));
            }
        };
        let state = &self.checkpoint.payload;
        if state.requested_claim != *requested_claim {
            return Err(PortfolioError::InvalidPortfolio(
                "continuation claim does not match the live requested claim".into(),
            ));
        }
        if state.context_digest != context.digest() {
            return Err(PortfolioError::InvalidPortfolio(
                "continuation assumptions context digest does not match the live context".into(),
            ));
        }
        // Observational counters are never restored, but a resumed state must
        // describe the live ledger exactly: any drift means the state was
        // captured for different work than is being resumed.
        for dimension in Dimension::ALL {
            let captured = state_remaining(&self.checkpoint, dimension);
            let live = cx.remaining(dimension);
            if captured != live {
                return Err(PortfolioError::InvalidPortfolio(format!(
                    "continuation budget counter mismatch for {}: captured remaining {captured}, live remaining {live}",
                    dimension.as_str(),
                )));
            }
        }
        if self.checkpoint.verifier_remaining != cx.verifier_remaining() {
            return Err(PortfolioError::InvalidPortfolio(format!(
                "continuation verifier reserve mismatch: captured {}, live {}",
                self.checkpoint.verifier_remaining,
                cx.verifier_remaining()
            )));
        }
        cx.checkpoint().map_err(|_| PortfolioError::Cancelled)?;
        let outcomes = state
            .candidates
            .iter()
            .map(|record| WorkerOutcome {
                idx: record.registration_index,
                name: record.candidate.strategy_name.clone(),
                result: Ok(record.candidate.clone()),
            })
            .collect();
        accept_concurrent_candidates(
            cx,
            context,
            requested_claim,
            outcomes,
            initial_compute_remaining,
        )
    }
}

fn state_remaining(
    checkpoint: &TypedCheckpoint<FactorRaceContinuationState>,
    dimension: Dimension,
) -> u64 {
    checkpoint
        .remaining_budget
        .get(&dimension)
        .copied()
        .unwrap_or(0)
}

/// Resumes drained factor-race candidates from durable typed-checkpoint state
/// in a fresh process or region.
///
/// This is the durable-storage resume path (`fra-rc-durable-m5e`), distinct
/// from [`FactorRaceContinuation::resume`]: the live ledger capability that
/// the in-region continuation requires cannot survive a process boundary, so
/// the fresh region presents its own newly constructed budget region. The
/// captured per-dimension counters inside `checkpoint` are observational
/// integrity copies only — they are never restored into, and never reset, any
/// ledger. The fresh region's remaining allowance is its own; durable
/// accounting is the caller's composition of both processes' consumptions.
///
/// Refuses before any charge or verification work: schema mismatch, failed
/// content-integrity replay, unknown operation version, claim or context
/// divergence from the presenting caller, and input-expression divergence
/// from the requested claim's left-hand side. Acceptance re-runs the full
/// independent mathematical verification for every candidate in registration
/// order; digests alone never authorize publication.
pub fn resume_factor_race_from_checkpoint<Caps>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    checkpoint: &TypedCheckpoint<FactorRaceContinuationState>,
) -> Result<VerifiedPortfolioOutcome, PortfolioError> {
    if checkpoint.payload_schema != FACTOR_RACE_CONTINUATION_SCHEMA {
        return Err(PortfolioError::InvalidPortfolio(format!(
            "durable continuation schema mismatch: expected {FACTOR_RACE_CONTINUATION_SCHEMA}, got {}",
            checkpoint.payload_schema
        )));
    }
    if !checkpoint.verify_integrity() {
        return Err(PortfolioError::InvalidPortfolio(
            "durable continuation integrity verification failed; refusing foreign or tampered state"
                .into(),
        ));
    }
    let state = &checkpoint.payload;
    if state.operation_version != "factor_race_v1" {
        return Err(PortfolioError::InvalidPortfolio(format!(
            "durable continuation operation version mismatch: {}",
            state.operation_version
        )));
    }
    if state.requested_claim != *requested_claim {
        return Err(PortfolioError::InvalidPortfolio(
            "durable continuation claim does not match the presenting requested claim".into(),
        ));
    }
    if state.context_digest != context.digest() {
        return Err(PortfolioError::InvalidPortfolio(
            "durable continuation assumptions context digest does not match the presenting context"
                .into(),
        ));
    }
    let Claim::AlgebraicIdentity { lhs, .. } = requested_claim else {
        return Err(PortfolioError::InvalidPortfolio(
            "factor race requests must be Claim::AlgebraicIdentity".into(),
        ));
    };
    if state.input_expr != *lhs {
        return Err(PortfolioError::InvalidPortfolio(
            "durable continuation input expression does not match the requested claim input".into(),
        ));
    }
    if state.candidates.is_empty() {
        return Err(PortfolioError::InvalidPortfolio(
            "durable continuation carries no drained candidates to resume".into(),
        ));
    }
    let initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);
    let outcomes = state
        .candidates
        .iter()
        .map(|record| WorkerOutcome {
            idx: record.registration_index,
            name: record.candidate.strategy_name.clone(),
            result: Ok(record.candidate.clone()),
        })
        .collect();
    accept_concurrent_candidates(
        cx,
        context,
        requested_claim,
        outcomes,
        initial_compute_remaining,
    )
}

/// Runs the drained concurrent candidate-generation phase for a factor race,
/// emitting the registered decision card and a typed continuation bound to the
/// caller's still-owned live ledger.
///
/// The continuation must be resumed with the same live [`FsymCx`]; this API
/// never serializes or restores ledger counters.
pub fn factor_race_generate<Caps: Send + Sync + 'static>(
    cx: &mut FsymCx<'_, Caps>,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    requested_claim: &Claim,
    strategies: Vec<NamedConcurrentStrategy<Caps>>,
) -> Result<(FactorRaceDecisionCard, FactorRaceContinuation), PortfolioError> {
    let decision_card =
        FactorRaceDecisionCard::build(requested_claim, &strategies, cx.verifier_remaining())?;
    let initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);
    let outcomes = generate_concurrent_candidates(cx, requested_claim, strategies)?;
    let ledger_identity = cx.ledger_identity();
    let candidates = outcomes
        .into_iter()
        .enumerate()
        .filter_map(|(registration_index, outcome)| {
            outcome
                .result
                .ok()
                .map(|candidate| FactorRaceCandidateRecord {
                    registration_index,
                    candidate,
                })
        })
        .collect::<Vec<_>>();
    let mut remaining_budget = std::collections::BTreeMap::new();
    for dimension in Dimension::ALL {
        remaining_budget.insert(dimension, cx.remaining(dimension));
    }
    let input_expr = match requested_claim {
        Claim::AlgebraicIdentity { lhs, .. } => lhs.clone(),
        _ => {
            return Err(PortfolioError::InvalidPortfolio(
                "factor race requests must be Claim::AlgebraicIdentity".into(),
            ));
        }
    };
    let checkpoint = TypedCheckpoint::new(
        FACTOR_RACE_CONTINUATION_SCHEMA,
        0,
        FactorRaceContinuationState {
            operation_version: "factor_race_v1".into(),
            input_expr,
            requested_claim: requested_claim.clone(),
            context_digest: context.digest(),
            candidates,
        },
        remaining_budget,
        cx.verifier_remaining(),
    )
    .map_err(|error| {
        PortfolioError::StructuredExecutionFailed(format!(
            "factor-race continuation capture failed: {error}"
        ))
    })?;
    Ok((
        decision_card,
        FactorRaceContinuation {
            checkpoint,
            ledger: Some((ledger_identity, initial_compute_remaining)),
        },
    ))
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

    /// Normalizes a factorization into the canonical product expression the
    /// tests fix as the requested claim: scale term (only when nonunit),
    /// then each factor, powered when its multiplicity exceeds one.
    fn normalize_factor_product<I>(
        scale: &fsym_core::BigRational,
        factors: I,
    ) -> Result<Expr, PortfolioError>
    where
        I: IntoIterator<Item = (fsym_polys::univariate::UnivariatePoly, usize)>,
    {
        use fsym_core::{BigInt, BigRational};
        let mut terms = Vec::new();
        let one = BigRational::from_integer(BigInt::from(1));
        if *scale != one {
            terms.push(Expr::Rational(scale.clone()));
        }
        let mut factors: Vec<_> = factors.into_iter().collect();
        factors.sort_by(|(left, _), (right, _)| {
            left.degree()
                .cmp(&right.degree())
                .then_with(|| left.coeffs.cmp(&right.coeffs))
        });
        for (poly, multiplicity) in factors {
            let term = poly.to_expr();
            terms.push(if multiplicity == 1 {
                term
            } else {
                let multiplicity = u64::try_from(multiplicity).map_err(|_| {
                    PortfolioError::InvalidPortfolio("factor multiplicity exceeds u64".into())
                })?;
                Expr::Pow(
                    Arc::new(term),
                    Arc::new(Expr::Integer(BigInt::from(multiplicity))),
                )
            });
        }
        Ok(match terms.len() {
            0 => Expr::from_i64(1),
            1 => terms.pop().expect("one factor"),
            _ => Expr::Mul(terms),
        })
    }

    #[test]
    fn durable_checkpoint_resume_refuses_every_mismatch_class() {
        use fsym_core::{BigInt, BigRational, Symbol};
        use fsym_polys::factorization::metered_complete_factorization;
        use fsym_polys::univariate::UnivariatePoly;

        let polynomial = |coeffs: &[i64]| {
            UnivariatePoly::new(
                Symbol::new("x"),
                coeffs
                    .iter()
                    .map(|c| BigRational::from_integer(BigInt::from(*c)))
                    .collect(),
            )
        };
        let input = Arc::new(polynomial(&[4, 0, 0, 0, 1]));
        let expected_product = Expr::Mul(vec![
            polynomial(&[2, -2, 1]).to_expr(),
            polynomial(&[2, 2, 1]).to_expr(),
        ]);
        let requested = Claim::AlgebraicIdentity {
            lhs: input.to_expr(),
            rhs: expected_product.clone(),
        };
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());

        let generator_context = Arc::clone(&context);
        let generator_input = Arc::clone(&input);
        let generator = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            let factorization = metered_complete_factorization(&generator_input, cx)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let product = normalize_factor_product(
                &factorization.scale,
                factorization
                    .factors
                    .into_iter()
                    .map(|factor| (factor.poly, factor.multiplicity))
                    .collect::<Vec<_>>(),
            )?;
            let lhs = generator_input.to_expr();
            let mut kernel = ProofKernel::new((**generator_context).clone());
            let root = kernel
                .prove_definitional_reduction(
                    lhs.clone(),
                    product.clone(),
                    "polynomial_ring_equivalence",
                    cx,
                )
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let derivation = kernel
                .export_derivation(root)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            Ok(PortfolioCandidate {
                strategy_name: "zassenhaus_modular".into(),
                result: product.clone(),
                claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                derivation,
            })
        });

        let raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(10_000_000, 100_000);
        let mut cx = FsymCx::new(&raw, Budget::new(limits), limits);
        let (_card, continuation) = factor_race_generate(
            &mut cx,
            &context,
            &requested,
            vec![
                ("zassenhaus_modular", generator),
                (
                    "refusing",
                    Box::new(|_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
                        Err(PortfolioError::AllStrategiesFailed("refusing".into()))
                    }),
                ),
            ],
        )
        .expect("generation drains");
        let wire = continuation.to_wire().expect("wire");
        let checkpoint: TypedCheckpoint<FactorRaceContinuationState> =
            serde_json::from_slice(&wire).expect("decode");

        // A fresh region resumes the committed checkpoint successfully.
        let resume_limits = BudgetLimits {
            dimensions: limits.dimensions,
            verifier_pool: checkpoint.verifier_remaining,
        };
        let resume_raw = Cx::detached_cancel_context();
        let mut fresh = FsymCx::new(&resume_raw, Budget::new(resume_limits), resume_limits);
        let outcome =
            resume_factor_race_from_checkpoint(&mut fresh, &context, &requested, &checkpoint)
                .expect("fresh-region resume publishes");
        assert_eq!(outcome.result(), &expected_product);

        // Schema mismatch refuses.
        let mut wrong_schema = checkpoint.clone();
        wrong_schema.payload_schema = "fsym.portfolio.something_else.v9".into();
        let resume_raw = Cx::detached_cancel_context();
        let mut probe = FsymCx::new(&resume_raw, Budget::new(resume_limits), resume_limits);
        assert!(matches!(
            resume_factor_race_from_checkpoint(&mut probe, &context, &requested, &wrong_schema),
            Err(PortfolioError::InvalidPortfolio(message))
                if message.contains("schema mismatch")
        ));

        // Tampered state (digest no longer replays) refuses before any charge.
        let mut tampered = checkpoint.clone();
        tampered.payload.candidates[0].candidate.strategy_name = "forged".into();
        let resume_raw = Cx::detached_cancel_context();
        let mut probe = FsymCx::new(&resume_raw, Budget::new(resume_limits), resume_limits);
        assert!(matches!(
            resume_factor_race_from_checkpoint(&mut probe, &context, &requested, &tampered),
            Err(PortfolioError::InvalidPortfolio(message))
                if message.contains("integrity")
        ));

        // A different requested claim refuses.
        let other_claim = Claim::AlgebraicIdentity {
            lhs: polynomial(&[1]).to_expr(),
            rhs: polynomial(&[1]).to_expr(),
        };
        let resume_raw = Cx::detached_cancel_context();
        let mut probe = FsymCx::new(&resume_raw, Budget::new(resume_limits), resume_limits);
        assert!(matches!(
            resume_factor_race_from_checkpoint(&mut probe, &context, &other_claim, &checkpoint),
            Err(PortfolioError::InvalidPortfolio(message))
                if message.contains("does not match the presenting requested claim")
        ));

        // A different assumptions context refuses.
        let mut facts = std::collections::HashMap::new();
        facts.insert(
            fsym_core::Symbol::new("x"),
            vec![fsym_assumptions::predicate::Predicate::Positive],
        );
        let other_context = ImmutableAssumptionsSnapshot::empty()
            .derive_child(
                facts,
                std::collections::HashMap::new(),
                "durable-refusal-probe",
            )
            .expect("child context");
        let resume_raw = Cx::detached_cancel_context();
        let mut probe = FsymCx::new(&resume_raw, Budget::new(resume_limits), resume_limits);
        assert!(matches!(
            resume_factor_race_from_checkpoint(&mut probe, &other_context, &requested, &checkpoint),
            Err(PortfolioError::InvalidPortfolio(message))
                if message.contains("context digest")
        ));

        // Refusals charged nothing: the fresh regions keep their full
        // verifier reserve after every refusal above.
        assert_eq!(probe.verifier_remaining(), resume_limits.verifier_pool);
    }

    #[test]
    fn real_factor_cancellation_after_verifier_refuses_publication() {
        use fsym_core::{BigInt, BigRational, Symbol};
        use fsym_polys::factorization::metered_complete_factorization;
        use fsym_polys::univariate::UnivariatePoly;
        use std::cell::Cell;
        use std::rc::Rc;

        // Clear even if an assertion panics before the publication boundary.
        struct ClearHook;
        impl Drop for ClearHook {
            fn drop(&mut self) {
                BEFORE_PUBLICATION.with(|slot| slot.borrow_mut().take());
            }
        }
        let _clear_hook = ClearHook;
        let polynomial = |coefficients: &[i64]| {
            UnivariatePoly::new(
                Symbol::new("x"),
                coefficients
                    .iter()
                    .map(|c| BigRational::from_integer(BigInt::from(*c)))
                    .collect(),
            )
        };
        let input = polynomial(&[4, 0, 0, 0, 1]);
        let requested = Claim::AlgebraicIdentity {
            lhs: input.to_expr(),
            rhs: Expr::Mul(vec![
                polynomial(&[2, -2, 1]).to_expr(),
                polynomial(&[2, 2, 1]).to_expr(),
            ]),
        };
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let generator_context = Arc::clone(&context);
        let verification_units = Rc::new(Cell::new(0));
        let candidate_units = Rc::clone(&verification_units);
        let generator = Box::new(move |cx: &mut FsymCx<'_, asupersync::cx::cap::None>| {
            let factorization = metered_complete_factorization(&input, cx)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let product = normalize_factor_product(
                &factorization.scale,
                factorization
                    .factors
                    .into_iter()
                    .map(|factor| (factor.poly, factor.multiplicity)),
            )?;
            let lhs = input.to_expr();
            let mut kernel = ProofKernel::new((**generator_context).clone());
            let root = kernel
                .prove_definitional_reduction(
                    lhs.clone(),
                    product.clone(),
                    "polynomial_ring_equivalence",
                    cx,
                )
                .unwrap();
            let candidate = PortfolioCandidate {
                strategy_name: "zassenhaus_modular".into(),
                result: product.clone(),
                claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                derivation: kernel.export_derivation(root).unwrap(),
            };
            candidate_units.set(
                claim_verification_units(&candidate.claim).unwrap()
                    + expression_verification_units(&candidate.result).unwrap()
                    + derivation_verification_units(&candidate.derivation).unwrap(),
            );
            Ok(candidate)
        });
        let raw = Cx::detached_cancel_context();
        let cancel = raw.clone();
        let reached_publication = Rc::new(Cell::new(false));
        let observed = Rc::clone(&reached_publication);
        BEFORE_PUBLICATION.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                observed.set(true);
                cancel.cancel_with(
                    asupersync::CancelKind::User,
                    Some("after real factor verification"),
                );
            }));
        });
        let limits = BudgetLimits::uniform(10_000_000, 100_000);
        let mut cx = FsymCx::new(&raw, Budget::new(limits), limits);
        let result = run_portfolio_race(
            &mut cx,
            &context,
            &requested,
            vec![("zassenhaus_modular", generator)],
        );
        assert!(
            reached_publication.get(),
            "independent verification must finish before cancellation"
        );
        assert_eq!(result, Err(PortfolioError::Cancelled));
        assert_eq!(
            cx.verifier_remaining(),
            limits.verifier_pool
                - claim_verification_units(&requested).unwrap()
                - verification_units.get()
        );
        assert!(
            cx.remaining(Dimension::ComputeSteps)
                < limits.dimensions[Dimension::ComputeSteps.index()]
        );
    }

    #[test]
    fn zassenhaus_quartic_product_passes_scoped_independent_verification() {
        use fsym_core::{BigInt, BigRational, Symbol};
        use fsym_polys::factorization::metered_complete_factorization;
        use fsym_polys::univariate::UnivariatePoly;

        let polynomial = |coefficients: &[i64]| {
            UnivariatePoly::new(
                Symbol::new("x"),
                coefficients
                    .iter()
                    .map(|coefficient| BigRational::from_integer(BigInt::from(*coefficient)))
                    .collect(),
            )
        };
        let input = Arc::new(polynomial(&[4, 0, 0, 0, 1]));
        // Fix the requested product independently before running the generator.
        // This is a product-identity claim, not an irreducibility claim.
        let expected_product = Expr::Mul(vec![
            polynomial(&[2, -2, 1]).to_expr(),
            polynomial(&[2, 2, 1]).to_expr(),
        ]);
        let requested = Claim::AlgebraicIdentity {
            lhs: input.to_expr(),
            rhs: expected_product.clone(),
        };
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let generator_context = Arc::clone(&context);
        let generator_input = Arc::clone(&input);
        let generator = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            let mut factorization = metered_complete_factorization(&generator_input, cx)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            factorization
                .factors
                .sort_by(|left, right| left.poly.coeffs.cmp(&right.poly.coeffs));
            let mut terms = Vec::new();
            let one = BigRational::from_integer(BigInt::from(1));
            if factorization.scale != one {
                terms.push(Expr::Rational(factorization.scale));
            }
            for factor in factorization.factors {
                let term = factor.poly.to_expr();
                terms.push(if factor.multiplicity == 1 {
                    term
                } else {
                    let multiplicity = u64::try_from(factor.multiplicity).map_err(|_| {
                        PortfolioError::InvalidPortfolio("factor multiplicity exceeds u64".into())
                    })?;
                    Expr::Pow(
                        Arc::new(term),
                        Arc::new(Expr::Integer(BigInt::from(multiplicity))),
                    )
                });
            }
            let product = match terms.len() {
                0 => Expr::from_i64(1),
                1 => terms.pop().expect("one factor"),
                _ => Expr::Mul(terms),
            };
            let lhs = generator_input.to_expr();
            let mut kernel = ProofKernel::new((**generator_context).clone());
            let root = kernel
                .prove_definitional_reduction(
                    lhs.clone(),
                    product.clone(),
                    "polynomial_ring_equivalence",
                    cx,
                )
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let derivation = kernel
                .export_derivation(root)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            Ok(PortfolioCandidate {
                strategy_name: "zassenhaus".into(),
                result: product.clone(),
                claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                derivation,
            })
        });
        // This test isolates the existing real generator; a refusing sibling is
        // not evidence that the two-real-strategy portfolio gate has passed.
        let refusing = Box::new(|_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            Err(PortfolioError::AllStrategiesFailed(
                "second real strategy is outside this single-generator regression".into(),
            ))
        });
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(10_000_000, 100_000);
        let mut cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let outcome = run_portfolio_concurrent_race(
            &mut cx,
            &context,
            &requested,
            vec![("zassenhaus", generator), ("refusing", refusing)],
        )
        .expect("real Zassenhaus product must pass protected independent verification");

        assert_eq!(outcome.winning_strategy(), "zassenhaus");
        assert_eq!(outcome.result(), &expected_product);
        assert_eq!(outcome.evidence().claim, requested);
        assert!(outcome.evidence().verify_integrity());
        assert!(outcome.generator_steps_consumed() > 0);
        assert_eq!(
            cx.remaining(Dimension::ComputeSteps),
            limits.dimensions[Dimension::ComputeSteps.index()] - outcome.generator_steps_consumed(),
        );
        assert!(cx.verifier_remaining() < limits.verifier_pool);
        assert!(cx.verifier_remaining() > 0);
    }
    #[test]
    fn rejects_claim_that_is_not_the_verified_derivation_root() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let budget = Budget::new(limits);
        let mut fsym_cx = FsymCx::new(&cx_raw, budget, limits);

        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let y = Expr::symbol("private_formula_symbol_must_not_leak");
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

        let Err(PortfolioError::WinnerVerificationFailed(message)) = result else {
            panic!("expected a typed winner-verification refusal");
        };
        assert!(message.contains("requested publication"));
        assert!(!message.contains("private_formula_symbol_must_not_leak"));
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
    fn sequential_generator_panic_is_contained_and_budget_is_reconciled() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let requested = Claim::equality(x.clone(), x.clone());

        let panicking = Box::new(
            |cx: &mut FsymCx<'_, asupersync::cx::cap::None>| -> Result<_, PortfolioError> {
                cx.charge(Dimension::ComputeSteps, 7).unwrap();
                panic!("planned sequential candidate-generator panic");
            },
        );

        let accepted_x = x.clone();
        let accepted = Box::new(move |cx: &mut FsymCx<'_, asupersync::cx::cap::None>| {
            cx.charge(Dimension::ComputeSteps, 2).unwrap();
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(accepted_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "sequential-panic-fallback".into(),
                result: accepted_x.clone(),
                claim: Claim::equality(accepted_x.clone(), accepted_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let outcome = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &requested,
            vec![("panicking", panicking), ("fallback", accepted)],
        )
        .expect("a contained generator panic must not suppress a verified fallback");

        assert_eq!(outcome.winning_strategy(), "sequential-panic-fallback");
        assert_eq!(outcome.generator_steps_consumed(), 9);
        assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 91);
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
    fn verifier_pool_exhaustion_is_not_misreported_as_candidate_rejection() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 3);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let requested = Claim::equality(x.clone(), x.clone());
        let candidate_x = x.clone();
        let strategy = Box::new(move |_cx: &mut FsymCx<'_, _>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(candidate_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "verifier-budget-probe".into(),
                result: candidate_x.clone(),
                claim: Claim::equality(candidate_x.clone(), candidate_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let result = run_portfolio_race(
            &mut fsym_cx,
            &context,
            &requested,
            vec![("verifier-budget-probe", strategy)],
        );

        assert!(matches!(result, Err(PortfolioError::BudgetExhausted(_))));
        assert_eq!(fsym_cx.verifier_remaining(), 1);
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
        let slow_strategy = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
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
        let fast_strategy = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
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

        assert_eq!(outcome.winning_strategy(), "slow");
        assert_eq!(outcome.result(), &x);
        assert_eq!(outcome.evidence().claim, requested);
        assert!(outcome.evidence().verify_integrity());
        assert!(fsym_cx.verifier_remaining() < 10);
    }

    #[test]
    fn concurrent_race_verifies_before_selecting_a_fallback() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(200, 20);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let requested = Claim::equality(x.clone(), x.clone());

        let rejected_x = x.clone();
        let rejected = Box::new(move |_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(rejected_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "unverified-first".into(),
                result: Expr::from_i64(999),
                claim: Claim::equality(rejected_x.clone(), rejected_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let accepted_x = x.clone();
        let accepted = Box::new(move |_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(accepted_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "verified-fallback".into(),
                result: accepted_x.clone(),
                claim: Claim::equality(accepted_x.clone(), accepted_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let outcome = run_portfolio_concurrent_race(
            &mut fsym_cx,
            &context,
            &requested,
            vec![
                ("unverified-first", rejected),
                ("verified-fallback", accepted),
            ],
        )
        .expect("an unverified generator success must not suppress a valid fallback");

        assert_eq!(outcome.winning_strategy(), "verified-fallback");
        assert_eq!(outcome.result(), &x);
    }

    #[test]
    fn concurrent_generator_panic_is_contained_and_budget_is_reconciled() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(200, 20);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let requested = Claim::equality(x.clone(), x.clone());

        let panicking = Box::new(
            |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| -> Result<_, PortfolioError> {
                cx.charge(Dimension::ComputeSteps, 7).unwrap();
                panic!("planned candidate-generator panic");
            },
        );

        let accepted_x = x.clone();
        let accepted = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            cx.charge(Dimension::ComputeSteps, 2).unwrap();
            let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
            let root = kernel
                .prove_reflexivity(accepted_x.clone(), &mut Unbounded)
                .unwrap();
            Ok(PortfolioCandidate {
                strategy_name: "panic-fallback".into(),
                result: accepted_x.clone(),
                claim: Claim::equality(accepted_x.clone(), accepted_x.clone()),
                derivation: kernel.export_derivation(root).unwrap(),
            })
        });

        let outcome = run_portfolio_concurrent_race(
            &mut fsym_cx,
            &context,
            &requested,
            vec![("panicking", panicking), ("panic-fallback", accepted)],
        )
        .expect("a contained generator panic must not suppress a verified fallback");

        assert_eq!(outcome.winning_strategy(), "panic-fallback");
        assert_eq!(outcome.generator_steps_consumed(), 9);
        assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 191);
    }

    #[test]
    fn concurrent_worker_checkpoint_observes_owner_cancellation() {
        let cx_raw = Cx::detached_cancel_context();
        let cancel_cx = cx_raw.clone();
        let limits = BudgetLimits::uniform(200, 20);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");

        let entered = Arc::new(AtomicBool::new(false));
        let observed = Arc::new(AtomicBool::new(false));
        let entered_in_worker = Arc::clone(&entered);
        let observed_in_worker = Arc::clone(&observed);
        let cancellation_sensitive =
            Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
                entered_in_worker.store(true, Ordering::Release);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while std::time::Instant::now() < deadline {
                    if cx.checkpoint().is_err() {
                        observed_in_worker.store(true, Ordering::Release);
                        return Err(PortfolioError::Cancelled);
                    }
                    std::hint::spin_loop();
                }
                Err(PortfolioError::StructuredExecutionFailed(
                    "worker did not observe owner cancellation within its bounded loop".into(),
                ))
            });
        let companion = Box::new(|_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            Err(PortfolioError::AllStrategiesFailed(
                "planned companion refusal".into(),
            ))
        });

        let entered_for_controller = Arc::clone(&entered);
        let controller = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !entered_for_controller.load(Ordering::Acquire) {
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                std::thread::yield_now();
            }
            cancel_cx.cancel_with(asupersync::CancelKind::User, Some("portfolio test"));
            true
        });

        let result = run_portfolio_concurrent_race(
            &mut fsym_cx,
            &context,
            &Claim::equality(x.clone(), x),
            vec![
                ("cancellation-sensitive", cancellation_sensitive),
                ("companion", companion),
            ],
        );
        assert!(controller.join().unwrap(), "worker never entered its body");

        assert_eq!(result, Err(PortfolioError::Cancelled));
        assert!(observed.load(Ordering::Acquire));
        assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 200);
    }

    #[test]
    fn concurrent_singleton_is_a_typed_configuration_refusal() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let strategy = Box::new(|_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            Err(PortfolioError::AllStrategiesFailed(
                "must not execute".into(),
            ))
        });

        assert!(matches!(
            run_portfolio_concurrent_race(
                &mut fsym_cx,
                &context,
                &Claim::equality(x.clone(), x),
                vec![("singleton", strategy)],
            ),
            Err(PortfolioError::InvalidPortfolio(_))
        ));
        assert_eq!(fsym_cx.verifier_remaining(), 10);
    }

    #[test]
    fn concurrent_race_all_fail_returns_typed_error() {
        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");

        let s1 = Box::new(|_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            Err(PortfolioError::AllStrategiesFailed("failed s1".into()))
        });
        let s2 = Box::new(|_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
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

    #[test]
    fn factor_race_two_real_strategies_verify_same_product_claim() {
        use fsym_core::{BigInt, BigRational, Symbol};
        use fsym_polys::factorization::metered_complete_factorization;
        use fsym_polys::univariate::UnivariatePoly;

        for zassenhaus_first in [true, false] {
            let (z_done_tx, z_done_rx) = std::sync::mpsc::channel();
            let (k_done_tx, k_done_rx) = std::sync::mpsc::channel();
            let z_done_rx = Mutex::new(z_done_rx);
            let k_done_rx = Mutex::new(k_done_rx);
            let (completion_tx, completion_rx) = std::sync::mpsc::channel();
            // x^4 + 4 = (x^2 - 2x + 2)(x^2 + 2x + 2): one product identity
            // raced by two mathematically distinct real generators.
            let ipoly = |coeffs: &[i64]| {
                UnivariatePoly::new(
                    Symbol::new("x"),
                    coeffs
                        .iter()
                        .map(|c| BigRational::from_integer(BigInt::from(*c)))
                        .collect(),
                )
            };
            let input = Arc::new(ipoly(&[4, 0, 0, 0, 1]));
            let expected_product = Expr::Mul(vec![
                ipoly(&[2, -2, 1]).to_expr(),
                ipoly(&[2, 2, 1]).to_expr(),
            ]);
            let requested = Claim::AlgebraicIdentity {
                lhs: input.to_expr(),
                rhs: expected_product.clone(),
            };
            let context = Arc::new(ImmutableAssumptionsSnapshot::empty());

            // Zassenhaus: metered modular complete factorization of the input.
            let z_input = Arc::clone(&input);
            let z_ctx = Arc::clone(&context);
            let z_completions = completion_tx.clone();
            let zassenhaus = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
                if !zassenhaus_first {
                    k_done_rx
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .expect("Kronecker must finish before releasing Zassenhaus");
                }
                let factorization = metered_complete_factorization(&z_input, cx)
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                let product = normalize_factor_product(
                    &factorization.scale,
                    factorization
                        .factors
                        .into_iter()
                        .map(|factor| (factor.poly, factor.multiplicity)),
                )?;
                let lhs = z_input.to_expr();
                let mut kernel = ProofKernel::new((**z_ctx).clone());
                let root = kernel
                    .prove_definitional_reduction(
                        lhs.clone(),
                        product.clone(),
                        "polynomial_ring_equivalence",
                        cx,
                    )
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                let derivation = kernel
                    .export_derivation(root)
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                z_completions.send("zassenhaus_modular").unwrap();
                if zassenhaus_first {
                    z_done_tx
                        .send(())
                        .expect("Zassenhaus completion receiver remains live");
                }
                Ok(PortfolioCandidate {
                    strategy_name: "zassenhaus_modular".into(),
                    result: product.clone(),
                    claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                    derivation,
                })
            });
            // Kronecker: the second real generator path; also meters its work.
            let k_input = Arc::clone(&input);
            let k_ctx = Arc::clone(&context);
            let k_completions = completion_tx;
            let kronecker = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
                if zassenhaus_first {
                    z_done_rx
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .expect("Zassenhaus must finish before releasing Kronecker");
                }
                let factorization =
                    fsym_polys::factorization::metered_kronecker_factorization(&k_input, cx)
                        .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                let product = normalize_factor_product(
                    &factorization.scale,
                    factorization
                        .factors
                        .into_iter()
                        .map(|factor| (factor.poly, factor.multiplicity)),
                )?;
                let lhs = k_input.to_expr();
                let mut kernel = ProofKernel::new((**k_ctx).clone());
                let root = kernel
                    .prove_definitional_reduction(
                        lhs.clone(),
                        product.clone(),
                        "polynomial_ring_equivalence",
                        cx,
                    )
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                let derivation = kernel
                    .export_derivation(root)
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                k_completions.send("kronecker_interpolation").unwrap();
                if !zassenhaus_first {
                    k_done_tx
                        .send(())
                        .expect("Kronecker completion receiver remains live");
                }
                Ok(PortfolioCandidate {
                    strategy_name: "kronecker_interpolation".into(),
                    result: product.clone(),
                    claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                    derivation,
                })
            });

            let cx_raw = Cx::detached_cancel_context();
            let limits = BudgetLimits::uniform(10_000_000, 100_000);
            let mut cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
            let initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);
            let outcomes = generate_concurrent_candidates(
                &mut cx,
                &requested,
                vec![
                    ("kronecker_interpolation", kronecker),
                    ("zassenhaus_modular", zassenhaus),
                ],
            )
            .expect("both real generators must drain");
            assert_eq!(
                completion_rx.try_iter().collect::<Vec<_>>(),
                if zassenhaus_first {
                    vec!["zassenhaus_modular", "kronecker_interpolation"]
                } else {
                    vec!["kronecker_interpolation", "zassenhaus_modular"]
                },
            );
            assert_eq!(outcomes.len(), 2);
            for outcome in &outcomes {
                let candidate = outcome.result.as_ref().expect("real generator completes");
                let verified = verify_and_publish_candidate(
                    &mut cx,
                    &context,
                    &requested,
                    &outcome.name,
                    candidate.clone(),
                    initial_compute_remaining,
                )
                .expect("each real generator independently verifies the fixed product claim");
                assert_eq!(verified.result(), &expected_product);
                assert_eq!(verified.evidence().claim, requested);
            }
            let outcome = accept_concurrent_candidates(
                &mut cx,
                &context,
                &requested,
                outcomes,
                initial_compute_remaining,
            )
            .expect("strict acceptance selects a verified real factor candidate");

            assert_eq!(
                outcome.result(),
                &expected_product,
                "both real strategies must verify the identical factor product"
            );
            assert_eq!(outcome.evidence().claim, requested);
            assert!(outcome.evidence().verify_integrity());
            assert!(outcome.generator_steps_consumed() > 0);
            assert_eq!(outcome.winning_strategy(), "kronecker_interpolation");
        }
    }

    #[test]
    fn invalid_fast_real_candidate_cannot_publish_before_slower_verified_one() {
        use fsym_core::{BigInt, BigRational, Symbol};
        use fsym_polys::factorization::{
            metered_complete_factorization, metered_kronecker_factorization,
        };
        use fsym_polys::univariate::UnivariatePoly;

        let ipoly = |coeffs: &[i64]| {
            UnivariatePoly::new(
                Symbol::new("x"),
                coeffs
                    .iter()
                    .map(|c| BigRational::from_integer(BigInt::from(*c)))
                    .collect(),
            )
        };
        let input = Arc::new(ipoly(&[4, 0, 0, 0, 1]));
        let expected_product = Expr::Mul(vec![
            ipoly(&[2, -2, 1]).to_expr(),
            ipoly(&[2, 2, 1]).to_expr(),
        ]);
        let requested = Claim::AlgebraicIdentity {
            lhs: input.to_expr(),
            rhs: expected_product.clone(),
        };
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());

        // Real Zassenhaus factors, but the candidate misbinds the requested
        // claim: its claimed identity does not match its verified product.
        let z_input = Arc::clone(&input);
        let z_ctx = Arc::clone(&context);
        let invalid_zassenhaus =
            Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
                let factorization = metered_complete_factorization(&z_input, cx)
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                let product = normalize_factor_product(
                    &factorization.scale,
                    factorization
                        .factors
                        .into_iter()
                        .map(|factor| (factor.poly, factor.multiplicity)),
                )?;
                let lhs = z_input.to_expr();
                let mut kernel = ProofKernel::new((**z_ctx).clone());
                let root = kernel
                    .prove_definitional_reduction(
                        lhs.clone(),
                        product.clone(),
                        "polynomial_ring_equivalence",
                        cx,
                    )
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                let derivation = kernel
                    .export_derivation(root)
                    .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
                Ok(PortfolioCandidate {
                    strategy_name: "zassenhaus_modular".into(),
                    result: product.clone(),
                    // Registered claim misbinds the result to the raw input.
                    claim: Claim::AlgebraicIdentity {
                        lhs: lhs.clone(),
                        rhs: lhs,
                    },
                    derivation,
                })
            });
        let k_input = Arc::clone(&input);
        let k_ctx = Arc::clone(&context);
        let kronecker = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            let factorization = metered_kronecker_factorization(&k_input, cx)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let product = normalize_factor_product(
                &factorization.scale,
                factorization
                    .factors
                    .into_iter()
                    .map(|factor| (factor.poly, factor.multiplicity)),
            )?;
            let lhs = k_input.to_expr();
            let mut kernel = ProofKernel::new((**k_ctx).clone());
            let root = kernel
                .prove_definitional_reduction(
                    lhs.clone(),
                    product.clone(),
                    "polynomial_ring_equivalence",
                    cx,
                )
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let derivation = kernel
                .export_derivation(root)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            Ok(PortfolioCandidate {
                strategy_name: "kronecker_interpolation".into(),
                result: product.clone(),
                claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                derivation,
            })
        });

        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(10_000_000, 100_000);
        let mut cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let _initial_compute_remaining = cx.remaining(Dimension::ComputeSteps);
        let outcome = run_portfolio_concurrent_race(
            &mut cx,
            &context,
            &requested,
            vec![
                ("zassenhaus_modular", invalid_zassenhaus),
                ("kronecker_interpolation", kronecker),
            ],
        )
        .expect("slower verified generator must win after invalid candidate refusal");
        assert_eq!(outcome.result(), &expected_product);
        assert_eq!(outcome.winning_strategy(), "kronecker_interpolation");
        assert_eq!(outcome.evidence().claim, requested);
        assert!(outcome.evidence().verify_integrity());
    }

    #[test]
    fn factor_race_continuation_resume_accepts_on_same_live_ledger_only() {
        use fsym_core::{BigInt, BigRational, Symbol};
        use fsym_polys::univariate::UnivariatePoly;

        let polynomial = |coeffs: &[i64]| {
            UnivariatePoly::new(
                Symbol::new("x"),
                coeffs
                    .iter()
                    .map(|c| BigRational::from_integer(BigInt::from(*c)))
                    .collect(),
            )
        };
        let input = Arc::new(polynomial(&[4, 0, 0, 0, 1]));
        let expected_product = Expr::Mul(vec![
            polynomial(&[2, -2, 1]).to_expr(),
            polynomial(&[2, 2, 1]).to_expr(),
        ]);
        let requested = Claim::AlgebraicIdentity {
            lhs: input.to_expr(),
            rhs: expected_product.clone(),
        };
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());

        let i_ctx = Arc::clone(&context);
        let i_input = Arc::clone(&input);
        let generator = Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
            use fsym_polys::factorization::metered_complete_factorization;
            let factorization = metered_complete_factorization(&i_input, cx)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let product = normalize_factor_product(
                &factorization.scale,
                factorization
                    .factors
                    .into_iter()
                    .map(|factor| (factor.poly, factor.multiplicity)),
            )?;
            let lhs = i_input.to_expr();
            let mut kernel = ProofKernel::new((**i_ctx).clone());
            let root = kernel
                .prove_definitional_reduction(
                    lhs.clone(),
                    product.clone(),
                    "polynomial_ring_equivalence",
                    cx,
                )
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            let derivation = kernel
                .export_derivation(root)
                .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
            Ok(PortfolioCandidate {
                strategy_name: "zassenhaus_modular".into(),
                result: product.clone(),
                claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                derivation,
            })
        });

        let cx_raw = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(10_000_000, 100_000);
        let mut cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
        let (_card, continuation) = factor_race_generate(
            &mut cx,
            &context,
            &requested,
            vec![
                ("zassenhaus_modular", generator),
                (
                    "refusing",
                    Box::new(|_cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
                        Err(PortfolioError::AllStrategiesFailed("refusing".into()))
                    }),
                ),
            ],
        )
        .expect("generation drains with the live ledger");

        // Wire round-trip: the decoded continuation has no ledger capability
        // and must refuse a fresh equal-budget ledger before any charge.
        let wire = continuation.to_wire().expect("wire encode");
        let decoded = FactorRaceContinuation::from_wire(&wire).expect("wire decode");
        let foreign_limits = BudgetLimits::uniform(10_000_000, 100_000);
        let foreign_raw = Cx::detached_cancel_context();
        let mut foreign_cx = FsymCx::new(&foreign_raw, Budget::new(foreign_limits), foreign_limits);
        assert!(matches!(
            decoded.resume(&mut foreign_cx, &context, &requested),
            Err(PortfolioError::InvalidPortfolio(_))
        ));
        assert_eq!(
            foreign_cx.remaining(Dimension::ComputeSteps),
            foreign_limits.dimensions[Dimension::ComputeSteps.index()],
            "refused resume must charge nothing"
        );

        // The same live ledger with identical counters resumes acceptance.
        let outcome = continuation
            .resume(&mut cx, &context, &requested)
            .expect("same live ledger resumes and publishes");
        assert_eq!(outcome.result(), &expected_product);
        assert!(outcome.evidence().verify_integrity());
        assert_eq!(
            outcome.generator_steps_consumed(),
            limits.dimensions[Dimension::ComputeSteps.index()]
                - cx.remaining(Dimension::ComputeSteps),
            "resumed accounting must include generation before the checkpoint",
        );
    }
}
