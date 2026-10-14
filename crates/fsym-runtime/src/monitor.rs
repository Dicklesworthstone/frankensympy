//! Registered operational monitors (constitution Article XV).
//!
//! This module implements the registered `compatibility_drift_v1` monitor: an
//! anytime-valid e-process over a stream of profile-differential observations.
//!
//! **Declared construction** (the monitor must state these, per Article XV.2):
//! * null hypothesis: the discrepancy rate is at most `p0`, per observation;
//! * alternative: the rate is at least `p1 > p0`;
//! * filtration: observations arrive one at a time; the wealth is measurable
//!   with respect to the observations seen so far, and no decision depends on
//!   future outcomes;
//! * statistic: the product of likelihood ratios, accumulated in log space, so
//!   the process is a nonnegative supermartingale under the null and Ville's
//!   inequality bounds the probability of ever crossing `1 / alpha` by `alpha`;
//! * reset policy: a reset opens a new generation, returns the log wealth to
//!   zero, and preserves every cumulative counter (history is not erased);
//! * censoring policy: timeouts, cancellations, and resource exhaustion are
//!   *observations*, never dropped — they are counted in their own buckets and
//!   charged against the wealth as observed divergence, because silently
//!   discarding them would bias the stream toward "no drift";
//! * fault policy: a non-finite update marks the monitor faulted and does not
//!   alarm; the caller falls back to its safe path (`increase_sampling`).
//!
//! **What it cannot do** (Article XV.4): an alarm is operational evidence about
//! a stream. It never proves or refutes any individual mathematical claim, and
//! it never promotes a compatibility profile. The recommended actions are the
//! registered ones; executing them is the caller's decision.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Registered monitor identifier implemented by this module.
pub const COMPATIBILITY_DRIFT_MONITOR_ID: &str = "compatibility_drift_v1";
/// Registered monitor identifier for portfolio outcome regression.
pub const PORTFOLIO_OUTCOME_REGRESSION_MONITOR_ID: &str = "portfolio_outcome_regression_v1";
/// Registered monitor identifier for dual-checked verifier disagreement.
pub const VERIFIER_DISAGREEMENT_MONITOR_ID: &str = "verifier_disagreement_v1";
/// Registered monitor identifier for runtime obligation leaks.
pub const OBLIGATION_LEAK_MONITOR_ID: &str = "obligation_leak_v1";
/// Registered monitor identifier for performance route regression.
pub const PERFORMANCE_REGRESSION_MONITOR_ID: &str = "performance_regression_v1";

/// All 5 registered operational monitor identifiers from `registries/monitor_profiles.toml`.
pub const REGISTERED_MONITOR_IDS: [&str; 5] = [
    PORTFOLIO_OUTCOME_REGRESSION_MONITOR_ID,
    COMPATIBILITY_DRIFT_MONITOR_ID,
    VERIFIER_DISAGREEMENT_MONITOR_ID,
    OBLIGATION_LEAK_MONITOR_ID,
    PERFORMANCE_REGRESSION_MONITOR_ID,
];

/// Minimum admitted null rate: open interval bound keeps the likelihood ratio finite.
const MIN_RATE: f64 = 1e-9;
/// Maximum admitted rate.
const MAX_RATE: f64 = 1.0 - 1e-9;
/// Bound on a single observation's log-likelihood contribution.
const MAX_LOG_CONTRIBUTION: f64 = 64.0;

/// Why the monitor refused to be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonitorError {
    /// `p0`/`p1`/`alpha` outside the admitted domain, or `p1 <= p0`.
    InvalidSpec(&'static str),
}

impl std::fmt::Display for MonitorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitorError::InvalidSpec(why) => write!(formatter, "invalid monitor spec: {why}"),
        }
    }
}

impl std::error::Error for MonitorError {}

/// A failure that is still an observation of the monitored population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureKind {
    /// The differential observation hit its wall-clock bound.
    Timeout,
    /// The differential observation was cancelled.
    Cancelled,
    /// The differential observation exhausted a declared budget.
    ResourceExhausted,
}

/// One element of the monitored stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Observation {
    /// A completed differential observation: did the candidate diverge?
    Completed {
        /// True when the observation recorded a divergence.
        discrepancy: bool,
    },
    /// A failed observation. Never dropped: counted and charged as divergence.
    Failed {
        /// Which failure occurred.
        kind: FailureKind,
    },
}

/// What the monitor recommends after an observation.
#[derive(Debug, Clone, PartialEq)]
pub enum MonitorDecision {
    /// Keep observing; the wealth has not crossed the threshold.
    Continue,
    /// The tail bound was crossed: open a discrepancy and block promotion.
    Alarm {
        /// The generation in which the alarm fired.
        generation: u64,
        /// `ln(1 / alpha)`.
        threshold: f64,
        /// Log wealth at the crossing.
        log_wealth: f64,
    },
    /// Arithmetic left the finite domain: investigate, do not alarm.
    Fault {
        /// The generation in which the fault occurred.
        generation: u64,
        /// Number of faults observed so far.
        faults: u64,
    },
}

/// Observable monitor state; plain numbers, so it serializes and resumes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonitorSnapshot {
    /// Registered monitor identifier.
    pub monitor_id: String,
    /// Null rate.
    pub null_rate: f64,
    /// Alternative rate.
    pub alternative_rate: f64,
    /// Significance level of the tail bound.
    pub alpha: f64,
    /// Current generation (0 before the first reset).
    pub generation: u64,
    /// Number of resets performed.
    pub resets: u64,
    /// Log wealth in the current generation.
    pub log_wealth: f64,
    /// Completed observations in the current generation.
    pub observations: u64,
    /// Completed observations with a divergence, current generation.
    pub discrepancies: u64,
    /// Failures charged in the current generation.
    pub failures: u64,
    /// Timeouts charged across all generations.
    pub timeouts: u64,
    /// Cancellations charged across all generations.
    pub cancellations: u64,
    /// Resource exhaustions charged across all generations.
    pub resource_exhaustions: u64,
    /// Total observations charged across all generations.
    pub total_observations: u64,
    /// Alarms raised across all generations.
    pub alarms: u64,
    /// Arithmetic faults across all generations.
    pub faults: u64,
    /// True once the current generation has alarmed (one-way until reset).
    pub alarmed: bool,
    /// True once an arithmetic fault occurred in the current generation.
    pub faulted: bool,
}

impl MonitorSnapshot {
    /// `ln(1 / alpha)`, the Ville threshold the wealth must reach.
    pub fn threshold(&self) -> f64 {
        (1.0 / self.alpha).ln()
    }

    /// Current wealth as a ratio (may exceed 1).
    pub fn wealth(&self) -> f64 {
        self.log_wealth.exp()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct EProcessCore {
    monitor_id: &'static str,
    null_rate: f64,
    alternative_rate: f64,
    alpha: f64,
    log_discrepancy: f64,
    log_agreement: f64,
    state: MonitorSnapshot,
}

impl EProcessCore {
    fn new(
        monitor_id: &'static str,
        null_rate: f64,
        alternative_rate: f64,
        alpha: f64,
    ) -> Result<Self, MonitorError> {
        if !(MIN_RATE..=MAX_RATE).contains(&null_rate) {
            return Err(MonitorError::InvalidSpec("null rate outside (0, 1)"));
        }
        if !(MIN_RATE..=MAX_RATE).contains(&alternative_rate) {
            return Err(MonitorError::InvalidSpec("alternative rate outside (0, 1)"));
        }
        if alternative_rate <= null_rate {
            return Err(MonitorError::InvalidSpec(
                "alternative rate must exceed the null rate",
            ));
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(MonitorError::InvalidSpec("alpha outside (0, 1)"));
        }
        let log_discrepancy = (alternative_rate / null_rate).ln();
        let log_agreement = ((1.0 - alternative_rate) / (1.0 - null_rate)).ln();
        if !log_discrepancy.is_finite()
            || !log_agreement.is_finite()
            || log_discrepancy > MAX_LOG_CONTRIBUTION
            || log_agreement.abs() > MAX_LOG_CONTRIBUTION
        {
            return Err(MonitorError::InvalidSpec(
                "declared rates produce an unrepresentable likelihood ratio",
            ));
        }
        Ok(Self {
            monitor_id,
            null_rate,
            alternative_rate,
            alpha,
            log_discrepancy,
            log_agreement,
            state: MonitorSnapshot {
                monitor_id: monitor_id.to_string(),
                null_rate,
                alternative_rate,
                alpha,
                generation: 0,
                resets: 0,
                log_wealth: 0.0,
                observations: 0,
                discrepancies: 0,
                failures: 0,
                timeouts: 0,
                cancellations: 0,
                resource_exhaustions: 0,
                total_observations: 0,
                alarms: 0,
                faults: 0,
                alarmed: false,
                faulted: false,
            },
        })
    }

    fn observe(&mut self, observation: Observation) -> MonitorDecision {
        // One-way until reset: once alarmed, every later observation keeps
        // returning Alarm (a non-finite candidate still returns Fault, per
        // the fault policy). Either way the observation is charged FIRST —
        // a failure that arrives after the alarm is still an observed
        // failure and must land in its own bucket, never vanish from the
        // stream (gate finding fra-rc-monitor-gate-p1m).
        let already_alarmed = self.state.alarmed;
        let (diverged, failure) = match observation {
            Observation::Completed { discrepancy } => (discrepancy, None),
            Observation::Failed { kind } => (true, Some(kind)),
        };
        let contribution = if diverged {
            self.log_discrepancy
        } else {
            self.log_agreement
        };
        let candidate = self.state.log_wealth + contribution;
        self.state.observations += 1;
        self.state.total_observations += 1;
        if diverged {
            self.state.discrepancies += 1;
        }
        if let Some(kind) = failure {
            self.state.failures += 1;
            match kind {
                FailureKind::Timeout => self.state.timeouts += 1,
                FailureKind::Cancelled => self.state.cancellations += 1,
                FailureKind::ResourceExhausted => self.state.resource_exhaustions += 1,
            }
        }
        if !candidate.is_finite() {
            // Fault policy: do not alarm on arithmetic that left the domain.
            self.state.faults += 1;
            self.state.faulted = true;
            return MonitorDecision::Fault {
                generation: self.state.generation,
                faults: self.state.faults,
            };
        }
        self.state.log_wealth = candidate;
        if already_alarmed || candidate >= self.state.threshold() {
            if !already_alarmed {
                // Count the transition once; post-alarm observations are
                // charged above but never raise new alarms.
                self.state.alarmed = true;
                self.state.alarms += 1;
            }
            return MonitorDecision::Alarm {
                generation: self.state.generation,
                threshold: self.state.threshold(),
                log_wealth: self.state.log_wealth,
            };
        }
        MonitorDecision::Continue
    }

    fn reset(&mut self) {
        self.state.generation += 1;
        self.state.resets += 1;
        self.state.log_wealth = 0.0;
        self.state.observations = 0;
        self.state.discrepancies = 0;
        self.state.failures = 0;
        self.state.alarmed = false;
        self.state.faulted = false;
    }

    fn snapshot(&self) -> MonitorSnapshot {
        self.state.clone()
    }

    fn resume(monitor_id: &'static str, snapshot: MonitorSnapshot) -> Result<Self, MonitorError> {
        let rebuilt = Self::new(
            monitor_id,
            snapshot.null_rate,
            snapshot.alternative_rate,
            snapshot.alpha,
        )?;
        if snapshot.monitor_id != monitor_id {
            return Err(MonitorError::InvalidSpec("snapshot names another monitor"));
        }
        Ok(Self {
            state: snapshot,
            ..rebuilt
        })
    }
}

/// The registered `compatibility_drift_v1` e-process monitor.
#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityDriftMonitor {
    core: EProcessCore,
}

impl CompatibilityDriftMonitor {
    /// Construct the monitor for a declared null and alternative rate.
    ///
    /// `0 < p0 < p1 < 1` and `0 < alpha < 1` are required; anything else is a
    /// typed refusal rather than a silent clamp.
    pub fn new(null_rate: f64, alternative_rate: f64, alpha: f64) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::new(
                COMPATIBILITY_DRIFT_MONITOR_ID,
                null_rate,
                alternative_rate,
                alpha,
            )?,
        })
    }

    /// The registered monitor identifier.
    pub fn monitor_id(&self) -> &'static str {
        COMPATIBILITY_DRIFT_MONITOR_ID
    }

    /// Registered safe fallback for `compatibility_drift_v1`.
    pub fn safe_fallback(&self) -> &'static str {
        "retain_previous_certified_profile"
    }

    /// Observe one stream element and return the monitor's recommendation.
    ///
    /// A failure is an observation: it advances the wealth as an observed
    /// divergence and lands in its own counter, so no failure can vanish from
    /// the monitored stream.
    pub fn observe(&mut self, observation: Observation) -> MonitorDecision {
        self.core.observe(observation)
    }

    /// Open a new generation: zero the wealth, keep every cumulative counter.
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Current observable state.
    pub fn snapshot(&self) -> MonitorSnapshot {
        self.core.snapshot()
    }

    /// Resume from a snapshot produced by this monitor construction.
    ///
    /// The declared rates must match, so a snapshot cannot silently change the
    /// hypothesis it was computed under.
    pub fn resume(snapshot: MonitorSnapshot) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::resume(COMPATIBILITY_DRIFT_MONITOR_ID, snapshot)?,
        })
    }

    /// The registered action recommended for the current state.
    pub fn recommended_action(&self) -> &'static str {
        if self.core.state.faulted {
            "increase_sampling"
        } else if self.core.state.alarmed {
            "block_profile_promotion"
        } else {
            "open_discrepancy"
        }
    }
}

/// The registered `portfolio_outcome_regression_v1` e-process monitor.
///
/// Population: `completed_and_inconclusive_portfolio_attempts`.
/// Actions: `["warn", "quarantine_route", "baseline_fallback", "recalibrate"]`.
/// Safe fallback: `"deterministic_baseline"`.
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioOutcomeRegressionMonitor {
    core: EProcessCore,
}

impl PortfolioOutcomeRegressionMonitor {
    /// Construct the monitor for a declared null and alternative inconclusive/regression rate.
    pub fn new(null_rate: f64, alternative_rate: f64, alpha: f64) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::new(
                PORTFOLIO_OUTCOME_REGRESSION_MONITOR_ID,
                null_rate,
                alternative_rate,
                alpha,
            )?,
        })
    }

    /// Construct with the default operational hypothesis (`p0 = 0.05, p1 = 0.25, alpha = 0.05`).
    pub fn default_spec() -> Self {
        Self::new(0.05, 0.25, 0.05).expect("default portfolio regression spec is valid")
    }

    /// The registered monitor identifier.
    pub fn monitor_id(&self) -> &'static str {
        PORTFOLIO_OUTCOME_REGRESSION_MONITOR_ID
    }

    /// Registered safe fallback for `portfolio_outcome_regression_v1`.
    pub fn safe_fallback(&self) -> &'static str {
        "deterministic_baseline"
    }

    /// Observe one portfolio attempt (completed verified vs failed/inconclusive/timeout/cancelled).
    pub fn observe(&mut self, observation: Observation) -> MonitorDecision {
        self.core.observe(observation)
    }

    /// Open a new generation: zero the wealth, keep every cumulative counter.
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Current observable state.
    pub fn snapshot(&self) -> MonitorSnapshot {
        self.core.snapshot()
    }

    /// Resume from a snapshot produced by this monitor construction.
    pub fn resume(snapshot: MonitorSnapshot) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::resume(PORTFOLIO_OUTCOME_REGRESSION_MONITOR_ID, snapshot)?,
        })
    }

    /// The registered action recommended for the current state.
    pub fn recommended_action(&self) -> &'static str {
        if self.core.state.faulted {
            "recalibrate"
        } else if self.core.state.alarmed {
            "quarantine_route"
        } else if self.core.state.log_wealth > 0.0 {
            "warn"
        } else {
            "baseline_fallback"
        }
    }
}

impl Default for PortfolioOutcomeRegressionMonitor {
    fn default() -> Self {
        Self::default_spec()
    }
}

/// The registered `verifier_disagreement_v1` exact-fault / e-process monitor.
///
/// Population: `dual_checked_certificates`.
/// Actions: `["immediate_quarantine", "block_publication", "preserve_repro_bundle"]`.
/// Safe fallback: `"reference_checker_only_after_investigation"`.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifierDisagreementMonitor {
    core: EProcessCore,
}

impl VerifierDisagreementMonitor {
    /// Construct the monitor for a declared null and alternative disagreement rate.
    pub fn new(null_rate: f64, alternative_rate: f64, alpha: f64) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::new(
                VERIFIER_DISAGREEMENT_MONITOR_ID,
                null_rate,
                alternative_rate,
                alpha,
            )?,
        })
    }

    /// Construct with the registered exact-fault-trigger calibration (`p0 = 1e-6, p1 = 0.99, alpha = 0.05`)
    /// so that a single disagreement or failure immediately crosses `ln(1 / alpha)` and alarms.
    pub fn new_exact_fault_trigger() -> Self {
        Self::new(1e-6, 0.99, 0.05).expect("exact fault trigger spec is valid")
    }

    /// The registered monitor identifier.
    pub fn monitor_id(&self) -> &'static str {
        VERIFIER_DISAGREEMENT_MONITOR_ID
    }

    /// Registered safe fallback for `verifier_disagreement_v1`.
    pub fn safe_fallback(&self) -> &'static str {
        "reference_checker_only_after_investigation"
    }

    /// Observe one dual-checked certificate comparison.
    pub fn observe(&mut self, observation: Observation) -> MonitorDecision {
        self.core.observe(observation)
    }

    /// Observe whether the optimized verifier and reference verifier agreed.
    pub fn observe_dual_check(&mut self, agreed: bool) -> MonitorDecision {
        self.observe(Observation::Completed {
            discrepancy: !agreed,
        })
    }

    /// Open a new generation: zero the wealth, keep every cumulative counter.
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Current observable state.
    pub fn snapshot(&self) -> MonitorSnapshot {
        self.core.snapshot()
    }

    /// Resume from a snapshot produced by this monitor construction.
    pub fn resume(snapshot: MonitorSnapshot) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::resume(VERIFIER_DISAGREEMENT_MONITOR_ID, snapshot)?,
        })
    }

    /// The registered action recommended for the current state.
    pub fn recommended_action(&self) -> &'static str {
        if self.core.state.faulted {
            "preserve_repro_bundle"
        } else if self.core.state.alarmed {
            "immediate_quarantine"
        } else {
            "block_publication"
        }
    }
}

impl Default for VerifierDisagreementMonitor {
    fn default() -> Self {
        Self::new_exact_fault_trigger()
    }
}

/// The registered `obligation_leak_v1` anytime-valid / exact-balance monitor.
///
/// Population: `all_acquired_runtime_obligations`.
/// Actions: `["warn", "cancel_region", "block_release"]`.
/// Safe fallback: `"exact_balance_invariant_and_conservative_shutdown"`.
#[derive(Debug, Clone, PartialEq)]
pub struct ObligationLeakMonitor {
    core: EProcessCore,
}

impl ObligationLeakMonitor {
    /// Construct the monitor for a declared null and alternative leak rate.
    pub fn new(null_rate: f64, alternative_rate: f64, alpha: f64) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::new(OBLIGATION_LEAK_MONITOR_ID, null_rate, alternative_rate, alpha)?,
        })
    }

    /// Construct with the registered exact-fault calibration (`p0 = 1e-6, p1 = 0.99, alpha = 0.05`)
    /// so that a single obligation leak immediately crosses `ln(1 / alpha)` and blocks release.
    pub fn new_exact_fault_trigger() -> Self {
        Self::new(1e-6, 0.99, 0.05).expect("obligation leak exact fault trigger spec is valid")
    }

    /// The registered monitor identifier.
    pub fn monitor_id(&self) -> &'static str {
        OBLIGATION_LEAK_MONITOR_ID
    }

    /// Registered safe fallback for `obligation_leak_v1`.
    pub fn safe_fallback(&self) -> &'static str {
        "exact_balance_invariant_and_conservative_shutdown"
    }

    /// Observe one obligation lifecycle outcome.
    pub fn observe(&mut self, observation: Observation) -> MonitorDecision {
        self.core.observe(observation)
    }

    /// Observe a region's obligation balance (`acquired` vs `reconciled`).
    pub fn observe_balance(&mut self, acquired: u64, reconciled: u64) -> MonitorDecision {
        self.observe(Observation::Completed {
            discrepancy: acquired != reconciled,
        })
    }

    /// Open a new generation: zero the wealth, keep every cumulative counter.
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Current observable state.
    pub fn snapshot(&self) -> MonitorSnapshot {
        self.core.snapshot()
    }

    /// Resume from a snapshot produced by this monitor construction.
    pub fn resume(snapshot: MonitorSnapshot) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::resume(OBLIGATION_LEAK_MONITOR_ID, snapshot)?,
        })
    }

    /// The registered action recommended for the current state.
    pub fn recommended_action(&self) -> &'static str {
        if self.core.state.faulted {
            "cancel_region"
        } else if self.core.state.alarmed {
            "block_release"
        } else {
            "warn"
        }
    }
}

impl Default for ObligationLeakMonitor {
    fn default() -> Self {
        Self::new_exact_fault_trigger()
    }
}

/// The registered `performance_regression_v1` e-process / conformal ratio monitor.
///
/// Population: `pinned_route_corpus_machine_observations`.
/// Actions: `["warn", "quarantine_optimized_route", "revert_default"]`.
/// Safe fallback: `"last_certified_route"`.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceRegressionMonitor {
    core: EProcessCore,
}

impl PerformanceRegressionMonitor {
    /// Construct the monitor for a declared null and alternative regression exceedance rate.
    pub fn new(null_rate: f64, alternative_rate: f64, alpha: f64) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::new(
                PERFORMANCE_REGRESSION_MONITOR_ID,
                null_rate,
                alternative_rate,
                alpha,
            )?,
        })
    }

    /// Construct with the default operational hypothesis (`p0 = 0.05, p1 = 0.25, alpha = 0.05`).
    pub fn default_spec() -> Self {
        Self::new(0.05, 0.25, 0.05).expect("default performance regression spec is valid")
    }

    /// The registered monitor identifier.
    pub fn monitor_id(&self) -> &'static str {
        PERFORMANCE_REGRESSION_MONITOR_ID
    }

    /// Registered safe fallback for `performance_regression_v1`.
    pub fn safe_fallback(&self) -> &'static str {
        "last_certified_route"
    }

    /// Observe one benchmark/route observation.
    pub fn observe(&mut self, observation: Observation) -> MonitorDecision {
        self.core.observe(observation)
    }

    /// Observe a paired latency measurement against a conformal ratio bound `max_ratio`.
    pub fn observe_latency_ratio(
        &mut self,
        candidate_ns: u64,
        baseline_ns: u64,
        max_ratio: f64,
    ) -> MonitorDecision {
        if baseline_ns == 0 || !max_ratio.is_finite() || max_ratio <= 0.0 {
            return self.observe(Observation::Failed {
                kind: FailureKind::ResourceExhausted,
            });
        }
        let regressed = (candidate_ns as f64) > (baseline_ns as f64) * max_ratio;
        self.observe(Observation::Completed {
            discrepancy: regressed,
        })
    }

    /// Open a new generation: zero the wealth, keep every cumulative counter.
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Current observable state.
    pub fn snapshot(&self) -> MonitorSnapshot {
        self.core.snapshot()
    }

    /// Resume from a snapshot produced by this monitor construction.
    pub fn resume(snapshot: MonitorSnapshot) -> Result<Self, MonitorError> {
        Ok(Self {
            core: EProcessCore::resume(PERFORMANCE_REGRESSION_MONITOR_ID, snapshot)?,
        })
    }

    /// The registered action recommended for the current state.
    pub fn recommended_action(&self) -> &'static str {
        if self.core.state.faulted {
            "revert_default"
        } else if self.core.state.alarmed {
            "quarantine_optimized_route"
        } else {
            "warn"
        }
    }
}

impl Default for PerformanceRegressionMonitor {
    fn default() -> Self {
        Self::default_spec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> CompatibilityDriftMonitor {
        CompatibilityDriftMonitor::new(0.05, 0.20, 0.05).expect("valid spec")
    }

    #[test]
    fn spec_validation_is_fail_closed() {
        assert!(CompatibilityDriftMonitor::new(0.0, 0.2, 0.05).is_err());
        assert!(CompatibilityDriftMonitor::new(0.05, 0.05, 0.05).is_err());
        assert!(CompatibilityDriftMonitor::new(0.05, 0.2, 0.0).is_err());
        assert!(CompatibilityDriftMonitor::new(0.05, 1.0, 0.05).is_err());
        assert!(CompatibilityDriftMonitor::new(0.05, 0.2, 0.05).is_ok());
    }

    #[test]
    fn wealth_is_exactly_the_likelihood_ratio_product() {
        let mut monitor = monitor();
        // Two discrepancies stay below ln(20); the third crosses (see the alarm test).
        for _ in 0..2 {
            assert_eq!(
                monitor.observe(Observation::Completed { discrepancy: true }),
                MonitorDecision::Continue
            );
        }
        let expected = 2.0 * (0.20_f64 / 0.05).ln();
        assert!((monitor.snapshot().log_wealth - expected).abs() < 1e-12);
        assert_eq!(
            monitor.observe(Observation::Completed { discrepancy: false }),
            MonitorDecision::Continue
        );
        let after = expected + ((1.0 - 0.20_f64) / (1.0 - 0.05)).ln();
        assert!((monitor.snapshot().log_wealth - after).abs() < 1e-12);
        assert_eq!(monitor.snapshot().discrepancies, 2);
        assert_eq!(monitor.snapshot().observations, 3);
    }

    #[test]
    fn crossing_the_tail_bound_alarms_once_and_stays_alarmed() {
        let mut monitor = monitor();
        let threshold = (1.0 / 0.05_f64).ln();
        let mut alarm = None;
        for step in 0..64 {
            if let MonitorDecision::Alarm { generation, .. } =
                monitor.observe(Observation::Completed { discrepancy: true })
            {
                alarm = Some((step, generation));
                break;
            }
        }
        let (step, generation) = alarm.expect("all-discrepancy stream must alarm");
        assert_eq!(generation, 0);
        assert!(monitor.snapshot().log_wealth >= threshold);
        assert_eq!(
            step, 2,
            "three discrepancies at a 4x likelihood ratio first cross ln(20)"
        );
        // Agreement after an alarm never clears it.
        let decision = monitor.observe(Observation::Completed { discrepancy: false });
        assert!(matches!(decision, MonitorDecision::Alarm { .. }));
        assert!(monitor.snapshot().alarmed);
    }

    #[test]
    fn failures_are_observations_and_never_silently_dropped() {
        let mut monitor = monitor();
        monitor.observe(Observation::Failed {
            kind: FailureKind::Timeout,
        });
        monitor.observe(Observation::Failed {
            kind: FailureKind::Cancelled,
        });
        monitor.observe(Observation::Failed {
            kind: FailureKind::ResourceExhausted,
        });
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.total_observations, 3);
        assert_eq!(snapshot.timeouts, 1);
        assert_eq!(snapshot.cancellations, 1);
        assert_eq!(snapshot.resource_exhaustions, 1);
        assert_eq!(
            snapshot.discrepancies, 3,
            "failures count as observed divergence"
        );
        assert!(
            snapshot.log_wealth > 0.0,
            "failures are not treated as agreement"
        );
    }

    #[test]
    fn reset_opens_a_generation_and_preserves_history() {
        let mut monitor = monitor();
        for _ in 0..3 {
            monitor.observe(Observation::Completed { discrepancy: true });
        }
        assert!(monitor.snapshot().alarmed);
        monitor.reset();
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.generation, 1);
        assert_eq!(snapshot.resets, 1);
        assert_eq!(snapshot.log_wealth, 0.0);
        assert_eq!(snapshot.observations, 0);
        assert!(!snapshot.alarmed);
        assert_eq!(snapshot.total_observations, 3, "history survives the reset");
        assert_eq!(snapshot.alarms, 1, "raised alarms are historical facts");
        assert_eq!(monitor.recommended_action(), "open_discrepancy");
    }

    #[test]
    fn snapshot_resume_continues_the_same_wealth() {
        let mut monitor = monitor();
        monitor.observe(Observation::Completed { discrepancy: true });
        let snapshot = monitor.snapshot();
        let mut resumed = CompatibilityDriftMonitor::resume(snapshot.clone()).expect("resume");
        assert_eq!(resumed.snapshot(), snapshot);
        monitor.observe(Observation::Completed { discrepancy: true });
        resumed.observe(Observation::Completed { discrepancy: true });
        assert_eq!(resumed.snapshot(), monitor.snapshot());
        // A snapshot whose declared hypothesis is not a valid monitor spec is
        // refused rather than resumed under a different construction.
        let mut mismatched = snapshot.clone();
        mismatched.alternative_rate = mismatched.null_rate;
        assert!(CompatibilityDriftMonitor::resume(mismatched).is_err());
        let mut foreign = snapshot.clone();
        foreign.monitor_id = "another_monitor_v1".to_string();
        assert!(CompatibilityDriftMonitor::resume(foreign).is_err());
    }

    #[test]
    fn corrupt_state_faults_instead_of_alarming() {
        // Persisted state is untrusted input: a corrupt snapshot must lead to a
        // fault and a safe action, never to an alarm.
        let mut snapshot = monitor().snapshot();
        snapshot.log_wealth = f64::INFINITY;
        let mut monitor =
            CompatibilityDriftMonitor::resume(snapshot).expect("resume accepts raw state");
        match monitor.observe(Observation::Completed { discrepancy: true }) {
            MonitorDecision::Fault { generation, faults } => {
                assert_eq!(generation, 0);
                assert_eq!(faults, 1);
            }
            other => panic!("expected Fault, got {other:?}"),
        }
        assert!(!monitor.snapshot().alarmed);
        assert_eq!(monitor.recommended_action(), "increase_sampling");
    }

    #[test]
    fn all_five_registered_monitors_instantiate_and_enforce_policies() {
        assert_eq!(REGISTERED_MONITOR_IDS.len(), 5);

        // 1. portfolio_outcome_regression_v1
        let mut portfolio_mon = PortfolioOutcomeRegressionMonitor::default_spec();
        assert_eq!(
            portfolio_mon.monitor_id(),
            PORTFOLIO_OUTCOME_REGRESSION_MONITOR_ID
        );
        assert_eq!(portfolio_mon.safe_fallback(), "deterministic_baseline");
        assert_eq!(
            portfolio_mon.observe(Observation::Completed { discrepancy: true }),
            MonitorDecision::Continue
        );
        assert_eq!(portfolio_mon.recommended_action(), "warn");
        portfolio_mon.observe(Observation::Failed {
            kind: FailureKind::Timeout,
        });
        assert!(matches!(
            portfolio_mon.observe(Observation::Completed { discrepancy: true }),
            MonitorDecision::Alarm { .. }
        ));
        assert_eq!(portfolio_mon.recommended_action(), "quarantine_route");

        // 2. verifier_disagreement_v1: exact fault trigger alarms immediately on a single disagreement
        let mut verifier_mon = VerifierDisagreementMonitor::new_exact_fault_trigger();
        assert_eq!(verifier_mon.monitor_id(), VERIFIER_DISAGREEMENT_MONITOR_ID);
        assert_eq!(
            verifier_mon.safe_fallback(),
            "reference_checker_only_after_investigation"
        );
        assert_eq!(verifier_mon.observe_dual_check(true), MonitorDecision::Continue);
        assert!(matches!(
            verifier_mon.observe_dual_check(false),
            MonitorDecision::Alarm { .. }
        ));
        assert_eq!(verifier_mon.recommended_action(), "immediate_quarantine");

        // 3. obligation_leak_v1: exact fault trigger alarms immediately on a single balance mismatch
        let mut leak_mon = ObligationLeakMonitor::new_exact_fault_trigger();
        assert_eq!(leak_mon.monitor_id(), OBLIGATION_LEAK_MONITOR_ID);
        assert_eq!(
            leak_mon.safe_fallback(),
            "exact_balance_invariant_and_conservative_shutdown"
        );
        assert_eq!(leak_mon.observe_balance(4, 4), MonitorDecision::Continue);
        assert!(matches!(
            leak_mon.observe_balance(4, 3),
            MonitorDecision::Alarm { .. }
        ));
        assert_eq!(leak_mon.recommended_action(), "block_release");

        // 4. performance_regression_v1: ratio exceedance e-process
        let mut perf_mon = PerformanceRegressionMonitor::default_spec();
        assert_eq!(perf_mon.monitor_id(), PERFORMANCE_REGRESSION_MONITOR_ID);
        assert_eq!(perf_mon.safe_fallback(), "last_certified_route");
        assert_eq!(
            perf_mon.observe_latency_ratio(110, 100, 1.25),
            MonitorDecision::Continue
        );
        assert_eq!(
            perf_mon.observe_latency_ratio(150, 100, 1.25),
            MonitorDecision::Continue
        );
        assert_eq!(
            perf_mon.observe_latency_ratio(160, 100, 1.25),
            MonitorDecision::Continue
        );
        assert!(matches!(
            perf_mon.observe_latency_ratio(200, 100, 1.25),
            MonitorDecision::Alarm { .. }
        ));
        assert_eq!(perf_mon.recommended_action(), "quarantine_optimized_route");
    }
}
