//! Reviewer-owned adversarial gate for the registered compatibility-drift
//! e-process monitor (`fra-rc-monitor-gate-p1m`, reviewing implementation
//! `fra-rc-monitor-5ja` at commit `1dc3b9a`).
//!
//! Independent authorship: written by the gate reviewer, not by the
//! implementation author, through the public API only. Attack obligations:
//! reset must zero the wealth while preserving every cumulative counter and
//! un-alarming; censoring must be impossible (a Failed observation is charged
//! as divergence in its own bucket and a failure-only stream must alarm);
//! the alarm is one-way within a generation; the wealth arithmetic crosses
//! the Ville threshold exactly; the fault policy can never become an alarm;
//! resume refuses a foreign monitor id and re-evaluates the bound honestly
//! against an inflated snapshot; the registered actions are exactly the
//! registered strings and no path admits a claim or promotes a profile.
//!
//! Verdicts assert exact state, not substrings, so a plausible-but-wrong
//! implementation fails rather than passing on phrasing.

#![forbid(unsafe_code)]

use fsym_runtime::monitor::{
    COMPATIBILITY_DRIFT_MONITOR_ID, CompatibilityDriftMonitor, FailureKind, MonitorDecision,
    MonitorSnapshot, Observation,
};

/// Independent deterministic 64-bit LCG (distinct multiplier from the
/// implementer's fixture on purpose).
struct ReviewLcg(u64);

impl ReviewLcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn draw(&mut self, probability: f64) -> bool {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64) < probability
    }
}

fn fresh() -> CompatibilityDriftMonitor {
    CompatibilityDriftMonitor::new(0.05, 0.20, 0.05).expect("valid spec")
}

fn completed(discrepancy: bool) -> Observation {
    Observation::Completed { discrepancy }
}

// ---------------------------------------------------------------------------
// Reset semantics
// ---------------------------------------------------------------------------

#[test]
fn reset_zeroes_wealth_but_preserves_every_cumulative_counter() {
    let mut monitor = fresh();
    for _ in 0..3 {
        let _ = monitor.observe(completed(true));
    }
    // Force a failure and an agreement into the same generation.
    let _ = monitor.observe(Observation::Failed { kind: FailureKind::Timeout });
    let _ = monitor.observe(completed(false));
    let before = monitor.snapshot();
    assert!(before.alarmed, "3 discrepancies at ln(4) each must cross ln(20)");
    assert_eq!(before.alarms, 1);

    monitor.reset();
    let after = monitor.snapshot();
    assert_eq!(after.generation, before.generation + 1, "generation advances");
    assert_eq!(after.resets, 1);
    assert_eq!(after.log_wealth, 0.0, "wealth zeroes");
    assert_eq!(after.observations, 0, "generation-local window zeroes");
    assert_eq!(after.discrepancies, 0);
    assert_eq!(after.failures, 0);
    assert!(!after.alarmed, "reset re-opens the generation");
    assert!(!after.faulted);
    // Cumulative counters survive: censoring history via reset is refused.
    assert_eq!(after.total_observations, before.total_observations);
    assert_eq!(after.timeouts, 1);
    assert_eq!(after.alarms, 1, "alarm history is not erased");
    // The stream resumes with a Continue, not a stale verdict.
    assert!(matches!(monitor.observe(completed(false)), MonitorDecision::Continue));
}

// ---------------------------------------------------------------------------
// Censoring adversaries
// ---------------------------------------------------------------------------

#[test]
fn a_failure_only_stream_alarms_and_lands_in_its_own_buckets() {
    let mut monitor = fresh();
    let mut decision = MonitorDecision::Continue;
    for _ in 0..3 {
        decision = monitor.observe(Observation::Failed { kind: FailureKind::Timeout });
    }
    assert!(
        matches!(decision, MonitorDecision::Alarm { .. }),
        "failures charged as agreement (censoring) would never alarm"
    );
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.failures, 3);
    assert_eq!(snapshot.timeouts, 3);
    assert_eq!(snapshot.cancellations, 0);
    assert_eq!(snapshot.resource_exhaustions, 0);
    assert_eq!(snapshot.discrepancies, 3, "failures are divergences");
}

#[test]
fn every_failure_kind_is_charged_in_its_own_counter() {
    let mut monitor = fresh();
    let _ = monitor.observe(Observation::Failed { kind: FailureKind::Cancelled });
    let _ = monitor.observe(Observation::Failed { kind: FailureKind::ResourceExhausted });
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.cancellations, 1);
    assert_eq!(snapshot.resource_exhaustions, 1);
    assert_eq!(snapshot.timeouts, 0);
    assert_eq!(snapshot.total_observations, 2, "no observation vanishes");
}

// ---------------------------------------------------------------------------
// One-way alarm + exact arithmetic
// ---------------------------------------------------------------------------

#[test]
fn alarm_is_one_way_within_a_generation() {
    let mut monitor = fresh();
    for _ in 0..3 {
        let _ = monitor.observe(completed(true));
    }
    for _ in 0..10 {
        let decision = monitor.observe(completed(false));
        match decision {
            MonitorDecision::Alarm { generation, threshold, log_wealth } => {
                assert_eq!(generation, 0);
                assert_eq!(threshold, (1.0 / 0.05f64).ln());
                let snapshot = monitor.snapshot();
                assert_eq!(log_wealth, snapshot.log_wealth, "wealth frozen at crossing");
            }
            other => panic!("agreement must not un-alarm, got {other:?}"),
        }
    }
    assert_eq!(monitor.snapshot().alarms, 1, "one alarm, not one per observation");
}

#[test]
fn wealth_crosses_the_ville_threshold_exactly_at_ln_4_contribution() {
    let mut monitor = fresh();
    // p1/p0 = 4, so wealth after k divergences is 4^k; ln(1/0.05) = ln(20)
    // is crossed exactly at k = 3 (64 >= 20), not at k = 2 (16 < 20).
    assert!(matches!(monitor.observe(completed(true)), MonitorDecision::Continue));
    let second = monitor.observe(completed(true));
    match second {
        MonitorDecision::Continue => {}
        other => panic!("16 < 20 must not alarm, got {other:?}"),
    }
    let third = monitor.observe(completed(true));
    match third {
        MonitorDecision::Alarm { threshold, log_wealth, .. } => {
            assert!((threshold - 20f64.ln()).abs() < 1e-12);
            assert!((log_wealth - 64f64.ln()).abs() < 1e-9);
            assert!((monitor.snapshot().wealth() - 64.0).abs() < 1e-6);
        }
        other => panic!("64 >= 20 must alarm, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Null alarm rate and power (deterministic seeded streams)
// ---------------------------------------------------------------------------

fn alarm_rate(rate: f64, streams: u64, steps: u64, seed: u64) -> f64 {
    let mut alarmed = 0u64;
    for stream in 0..streams {
        let mut monitor = fresh();
        let mut rng = ReviewLcg::new(seed ^ stream.wrapping_mul(0x9e37_79b9));
        let mut fired = false;
        for _ in 0..steps {
            if let MonitorDecision::Alarm { .. } =
                monitor.observe(completed(rng.draw(rate)))
            {
                fired = true;
                break;
            }
        }
        if fired {
            alarmed += 1;
        }
    }
    alarmed as f64 / streams as f64
}

#[test]
fn null_alarm_rate_is_bounded_by_alpha() {
    let rate = alarm_rate(0.05, 2_000, 200, 0x5eed_c020);
    assert!(
        rate <= 0.05,
        "Ville bound violated: null alarm rate {rate} exceeds alpha 0.05"
    );
}

#[test]
fn drift_at_the_alternative_rate_is_detected_quickly() {
    let rate = alarm_rate(0.40, 200, 100, 0x5eed_d1f7);
    assert!(
        rate >= 0.90,
        "no power: only {rate} of 0.40-drift streams alarmed within 100 steps"
    );
}

// ---------------------------------------------------------------------------
// Resume and snapshot adversaries
// ---------------------------------------------------------------------------

#[test]
fn resume_refuses_a_foreign_monitor_id_and_invalid_spec() {
    let mut monitor = fresh();
    let _ = monitor.observe(completed(true));
    let mut foreign = monitor.snapshot();
    foreign.monitor_id = "something_else_v9".to_string();
    assert!(CompatibilityDriftMonitor::resume(foreign).is_err());

    let mut bad = monitor.snapshot();
    bad.alpha = 1.5;
    assert!(CompatibilityDriftMonitor::resume(bad).is_err());

    let mut empty = MonitorSnapshot {
        monitor_id: COMPATIBILITY_DRIFT_MONITOR_ID.to_string(),
        null_rate: 0.05,
        alternative_rate: 0.20,
        alpha: 0.05,
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
    };
    assert!(CompatibilityDriftMonitor::resume(empty.clone()).is_ok());
    empty.log_wealth = f64::NAN;
    assert!(
        CompatibilityDriftMonitor::resume(empty).is_ok(),
        "snapshot fields are plain numbers; honesty is enforced on the next observation"
    );
}

#[test]
fn resumed_stream_is_identical_to_a_never_interrupted_one() {
    let mut straight = fresh();
    let mut split = fresh();
    for _ in 0..5 {
        let _ = straight.observe(completed(true));
        let _ = split.observe(completed(true));
    }
    let resumed = CompatibilityDriftMonitor::resume(split.snapshot()).expect("valid snapshot");
    let mut split = resumed;
    let _ = straight.observe(completed(false));
    let _ = split.observe(completed(false));
    assert_eq!(straight.snapshot(), split.snapshot());
}

#[test]
fn an_inflated_snapshot_wealth_still_re_evaluates_the_bound_honestly() {
    let mut inflated = fresh().snapshot();
    inflated.log_wealth = 1e300;
    let mut monitor = CompatibilityDriftMonitor::resume(inflated).expect("resumes");
    let decision = monitor.observe(completed(false));
    assert!(
        matches!(decision, MonitorDecision::Alarm { .. }),
        "a forged above-threshold snapshot must alarm on the next observation, never Continue"
    );
}

// ---------------------------------------------------------------------------
// Fault policy and registered action wiring
// ---------------------------------------------------------------------------

#[test]
fn fault_policy_recommends_increase_sampling_and_never_alarms() {
    let mut snapshot = fresh().snapshot();
    snapshot.log_wealth = f64::NAN;
    snapshot.faulted = false;
    let mut monitor = CompatibilityDriftMonitor::resume(snapshot).expect("resumes");
    match monitor.observe(completed(true)) {
        MonitorDecision::Fault { faults, .. } => assert_eq!(faults, 1),
        other => panic!("non-finite arithmetic must fault, got {other:?}"),
    }
    assert_eq!(monitor.recommended_action(), "increase_sampling");
    assert!(!monitor.snapshot().alarmed, "a fault is never an alarm");
}

#[test]
fn registered_actions_are_exactly_the_registered_strings() {
    assert_eq!(fresh().recommended_action(), "open_discrepancy");
    assert_eq!(COMPATIBILITY_DRIFT_MONITOR_ID, "compatibility_drift_v1");
    let mut monitor = fresh();
    for _ in 0..3 {
        let _ = monitor.observe(completed(true));
    }
    assert_eq!(monitor.recommended_action(), "block_profile_promotion");
}
