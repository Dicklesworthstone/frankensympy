//! Adversarial gate for the registered `compatibility_drift_v1` monitor (WS13).
//!
//! The point of an anytime-valid e-process is the tail bound, not a friendly
//! example: under the null the probability of *ever* crossing `1 / alpha` must
//! stay at or below `alpha`, and under the alternative the monitor must actually
//! fire. Both are checked here on deterministic seeded streams, so the numbers
//! below are pinned regressions rather than fresh noise on every run.
//!
//! Failures are observations: a stream of timeouts/cancellations/exhaustions
//! must move the monitor, never be silently discarded as "no drift".

#![forbid(unsafe_code)]

use fsym_runtime::monitor::{
    COMPATIBILITY_DRIFT_MONITOR_ID, CompatibilityDriftMonitor, FailureKind, MonitorDecision,
    MonitorSnapshot, Observation,
};

/// Deterministic 64-bit LCG; the sequence is part of the pinned fixture.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493),
        )
    }

    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn draw(&mut self, rate: f64) -> bool {
        self.next_unit() < rate
    }
}

const NULL_RATE: f64 = 0.05;
const ALTERNATIVE_RATE: f64 = 0.20;
const ALPHA: f64 = 0.05;
const STREAMS: usize = 4_000;
const STEPS: usize = 200;

fn alarm_rate(rate: f64, seed: u64) -> f64 {
    let mut alarms = 0;
    for stream in 0..STREAMS {
        let mut monitor =
            CompatibilityDriftMonitor::new(NULL_RATE, ALTERNATIVE_RATE, ALPHA).expect("valid spec");
        let mut rng = Lcg::new(seed + stream as u64);
        for _ in 0..STEPS {
            if matches!(
                monitor.observe(Observation::Completed {
                    discrepancy: rng.draw(rate)
                }),
                MonitorDecision::Alarm { .. }
            ) {
                alarms += 1;
                break;
            }
        }
    }
    alarms as f64 / STREAMS as f64
}

#[test]
fn anytime_validity_holds_under_the_null() {
    // Ville's inequality gives P(ever crossing) <= alpha = 0.05. The observed
    // rate is pinned so a regression that inflates it fails loudly; the bound
    // itself is the mathematical claim.
    let observed = alarm_rate(NULL_RATE, 0x5eed_0001);
    assert!(
        observed <= ALPHA,
        "null alarm rate {observed} exceeds the declared alpha {ALPHA}"
    );
}

#[test]
fn the_monitor_has_power_against_a_drifted_stream() {
    let observed = alarm_rate(0.40, 0x5eed_0002);
    assert!(
        observed >= 0.90,
        "drifted stream only alarmed in {observed} of streams"
    );
    assert_eq!(COMPATIBILITY_DRIFT_MONITOR_ID, "compatibility_drift_v1");
}

#[test]
fn failures_move_the_monitor_and_are_never_dropped() {
    let mut monitor =
        CompatibilityDriftMonitor::new(NULL_RATE, ALTERNATIVE_RATE, ALPHA).expect("valid spec");
    let mut decision = MonitorDecision::Continue;
    for _ in 0..3 {
        decision = monitor.observe(Observation::Failed {
            kind: FailureKind::Timeout,
        });
    }
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.timeouts, 3);
    assert_eq!(snapshot.discrepancies, 3);
    assert!(matches!(decision, MonitorDecision::Alarm { .. }));
    assert_eq!(monitor.recommended_action(), "block_profile_promotion");
}

#[test]
fn resets_renew_the_bound_without_erasing_history() {
    let mut monitor =
        CompatibilityDriftMonitor::new(NULL_RATE, ALTERNATIVE_RATE, ALPHA).expect("valid spec");
    let mut generations_alarmed = 0u64;
    for generation in 0..8u64 {
        let mut rng = Lcg::new(0x5eed_1000 + generation);
        for _ in 0..STEPS {
            if matches!(
                monitor.observe(Observation::Completed {
                    discrepancy: rng.draw(NULL_RATE)
                }),
                MonitorDecision::Alarm { .. }
            ) {
                generations_alarmed += 1;
                break;
            }
        }
        monitor.reset();
        assert_eq!(monitor.snapshot().generation, generation + 1);
    }
    // Per-generation validity still holds across the same null.
    assert!(
        generations_alarmed <= 2,
        "null alarms across 8 generations: {generations_alarmed}"
    );
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.resets, 8);
    assert!(
        snapshot.total_observations > 100,
        "history accumulates across generations"
    );
    assert_eq!(
        snapshot.log_wealth, 0.0,
        "a fresh generation starts at zero"
    );
}

#[test]
fn snapshot_round_trip_continues_the_identical_stream() {
    let mut monitor =
        CompatibilityDriftMonitor::new(NULL_RATE, ALTERNATIVE_RATE, ALPHA).expect("valid spec");
    let mut rng = Lcg::new(0x5eed_2000);
    for _ in 0..10 {
        monitor.observe(Observation::Completed {
            discrepancy: rng.draw(NULL_RATE),
        });
    }
    let snapshot: MonitorSnapshot = monitor.snapshot();
    let serialized = serde_json::to_string(&snapshot).expect("snapshot serializes");
    let restored: MonitorSnapshot = serde_json::from_str(&serialized).expect("snapshot parses");
    assert_eq!(restored, snapshot);
    let mut resumed = CompatibilityDriftMonitor::resume(restored).expect("resume");
    for _ in 0..10 {
        let observation = Observation::Completed {
            discrepancy: rng.draw(NULL_RATE),
        };
        assert_eq!(monitor.observe(observation), resumed.observe(observation));
    }
    assert_eq!(monitor.snapshot(), resumed.snapshot());
}
