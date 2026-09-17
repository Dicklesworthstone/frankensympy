//! Cancellation-injection lab run and replay verification suite (WS13 / C6 gate).
//!
//! Constitutional requirements (Art. VII.7, VIII.7, XVII; WORKSTREAM_GRAPH.md §18, §32):
//! - Cancellation is request -> drain -> finalize with zero controlled orphan tasks.
//! - Candidate publication and accepted publication are separate phases.
//! - Verifier pool is protected and cannot be consumed by cancelled generators.
//! - Fallback and cancellation never reset budget accounting.
//! - This integration matrix covers before reservation, during generator batches,
//!   and before verifier execution. The real-factor post-verifier publication
//!   boundary is tested in portfolio.rs with a test-only injection seam.
//! - Replay serialization round trips are checked here, not execution replay.

#![forbid(unsafe_code)]

use asupersync::Cx;
use asupersync::cx::cap::None as CapNone;
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
    Unbounded,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::factorization::{metered_complete_factorization, metered_kronecker_factorization};
use fsym_polys::univariate::UnivariatePoly;
use fsym_proof_kernel::{Claim, ProofKernel, claim_verification_units};
use fsym_runtime::{
    FsymCpuCx, FsymCx, PortfolioCandidate, PortfolioError, ReplayLog,
    run_portfolio_concurrent_race, run_portfolio_race,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

fn make_candidate(x: &Expr, name: &str) -> PortfolioCandidate {
    let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
    let root = kernel
        .prove_reflexivity(x.clone(), &mut Unbounded)
        .expect("reflexivity must prove");
    PortfolioCandidate {
        strategy_name: name.to_string(),
        result: x.clone(),
        claim: Claim::equality(x.clone(), x.clone()),
        derivation: kernel
            .export_derivation(root)
            .expect("export derivation must succeed"),
    }
}

const LIFECYCLE_TIMEOUT: Duration = Duration::from_secs(10);

fn real_factor_request() -> (UnivariatePoly, Claim) {
    let polynomial = |coefficients: &[i64]| {
        UnivariatePoly::new(
            Symbol::new("x"),
            coefficients
                .iter()
                .map(|coefficient| BigRational::from_integer(BigInt::from(*coefficient)))
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
    (input, requested)
}

struct ActiveGuard(Arc<AtomicUsize>);

impl ActiveGuard {
    fn enter(count: &Arc<AtomicUsize>) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(count))
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

// Observe real child-ledger charges, never simulate a refusal or cancellation.
// The optional callback pauses once, after an actual atomic batch has been paid.
struct ObservedMeter<'a, M, F> {
    inner: &'a mut M,
    spent: [u64; DIMENSION_COUNT],
    last_error: Option<MeterError>,
    after_batch: Option<F>,
}

impl<M: BudgetMeter, F: FnMut()> BudgetMeter for ObservedMeter<'_, M, F> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        let result = self.inner.charge(dimension, amount);
        match result {
            Ok(()) => self.spent[dimension.index()] += amount,
            Err(error) => self.last_error = Some(error),
        }
        result
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let result = self.inner.charge_batch(charges);
        match result {
            Ok(()) => {
                for &(dimension, amount) in charges {
                    self.spent[dimension.index()] += amount;
                }
                if let Some(mut callback) = self.after_batch.take() {
                    callback();
                }
            }
            Err(error) => self.last_error = Some(error),
        }
        result
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.last_error = Some(error);
        }
        result
    }
}

fn run_real_generator(
    zassenhaus: bool,
    input: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<(), fsym_polys::PolyError> {
    if zassenhaus {
        metered_complete_factorization(input, meter).map(|_| ())
    } else {
        metered_kronecker_factorization(input, meter).map(|_| ())
    }
}

// ============================================================================
// Phase 1: Cancellation Injected BEFORE Reservation
// ============================================================================

#[test]
fn test_cancellation_matrix_before_reservation() {
    let cx_raw = Cx::detached_cancel_context();
    cx_raw.cancel_with(asupersync::CancelKind::User, Some("pre-cancel"));

    let limits = BudgetLimits::uniform(100, 10);
    let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
    let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
    let x = Expr::symbol("x");
    let requested = Claim::equality(x.clone(), x.clone());

    // 1. Concurrent race with pre-cancelled context
    let s1 = Box::new(|_cx: &mut FsymCpuCx<'_, CapNone>| {
        panic!("worker 1 must not run when pre-cancelled");
    });
    let s2 = Box::new(|_cx: &mut FsymCpuCx<'_, CapNone>| {
        panic!("worker 2 must not run when pre-cancelled");
    });

    let result_concurrent = run_portfolio_concurrent_race(
        &mut fsym_cx,
        &context,
        &requested,
        vec![("s1", s1), ("s2", s2)],
    );
    assert_eq!(result_concurrent, Err(PortfolioError::Cancelled));
    assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 100);
    assert_eq!(fsym_cx.verifier_remaining(), 10);

    // 2. Sequential race with pre-cancelled context
    let seq_s = Box::new(|_cx: &mut FsymCx<'_, CapNone>| {
        panic!("sequential strategy must not run when pre-cancelled");
    });
    let result_seq = run_portfolio_race(&mut fsym_cx, &context, &requested, vec![("seq", seq_s)]);
    assert_eq!(result_seq, Err(PortfolioError::Cancelled));
    assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 100);
    assert_eq!(fsym_cx.verifier_remaining(), 10);
}

// ============================================================================
// Phase 2: Cancellation Injected DURING Generator Batch (Concurrent Workers)
// ============================================================================

#[test]
fn real_factor_generators_exhaust_budget_without_spending_verifier_reserve() {
    let cx_raw = Cx::detached_cancel_context();
    let mut limits = BudgetLimits::uniform(100_000_000, 100);
    // Kronecker can pay its first batches, but neither generator can finish.
    // The odd unit also checks reconciliation of the unreserved remainder.
    limits.dimensions[Dimension::ComputeSteps.index()] = 13;
    let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
    let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
    let (input, requested) = real_factor_request();
    let preflight = claim_verification_units(&requested).unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let (report_tx, report_rx) = mpsc::channel();
    let mut strategies: Vec<fsym_runtime::portfolio::NamedConcurrentStrategy<CapNone>> = Vec::new();
    for (name, zassenhaus) in [("zassenhaus", true), ("kronecker", false)] {
        let input = input.clone();
        let active = Arc::clone(&active);
        let report_tx = report_tx.clone();
        strategies.push((
            name,
            Box::new(move |cx| {
                let _guard = ActiveGuard::enter(&active);
                let mut meter = ObservedMeter {
                    inner: cx,
                    spent: [0; DIMENSION_COUNT],
                    last_error: None,
                    after_batch: None::<fn()>,
                };
                let error = run_real_generator(zassenhaus, &input, &mut meter)
                    .expect_err("real generator must refuse the insufficient child allowance");
                assert!(matches!(
                    meter.last_error,
                    Some(MeterError::Budget(BudgetError::Exhausted {
                        dimension: Dimension::ComputeSteps,
                        ..
                    }))
                ));
                report_tx.send((name, meter.spent)).unwrap();
                Err(PortfolioError::BudgetExhausted(error.to_string()))
            }),
        ));
    }
    drop(report_tx);

    let result = run_portfolio_concurrent_race(&mut fsym_cx, &context, &requested, strategies);
    assert!(matches!(
        result,
        Err(PortfolioError::AllStrategiesFailed(_))
    ));
    assert_eq!(active.load(Ordering::SeqCst), 0);
    let mut reports: Vec<_> = report_rx.try_iter().collect();
    reports.sort_by_key(|(name, _)| *name);
    assert_eq!(
        reports.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        ["kronecker", "zassenhaus"]
    );
    assert!(reports[0].1[Dimension::ComputeSteps.index()] > 0);
    for dimension in Dimension::ALL {
        let spent: u64 = reports
            .iter()
            .map(|(_, spent)| spent[dimension.index()])
            .sum();
        assert_eq!(
            fsym_cx.remaining(dimension),
            limits.dimensions[dimension.index()] - spent,
            "unused {dimension} reservations must return without refunding real work"
        );
    }
    assert_eq!(
        fsym_cx.verifier_remaining(),
        limits.verifier_pool - preflight
    );
}

#[test]
fn real_factor_cancellation_at_charged_batch_drains_delayed_sibling_and_callback() {
    let cx_raw = Cx::detached_cancel_context();
    let cancel_cx = cx_raw.clone();
    let limits = BudgetLimits::uniform(1_000_000_000, 100);
    let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
    let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
    let (input, requested) = real_factor_request();
    let preflight = claim_verification_units(&requested).unwrap();
    let active_workers = Arc::new(AtomicUsize::new(0));
    let active_callbacks = Arc::new(AtomicUsize::new(0));
    let returned = Arc::new(AtomicBool::new(false));
    let (ready_tx, ready_rx) = mpsc::channel();
    let (drained_tx, drained_rx) = mpsc::channel();
    let (report_tx, report_rx) = mpsc::channel();
    let mut releases = Vec::new();
    let mut strategies: Vec<fsym_runtime::portfolio::NamedConcurrentStrategy<CapNone>> = Vec::new();
    for (name, zassenhaus) in [("zassenhaus", true), ("kronecker", false)] {
        let input = input.clone();
        let workers = Arc::clone(&active_workers);
        let callbacks = Arc::clone(&active_callbacks);
        let ready_tx = ready_tx.clone();
        let drained_tx = drained_tx.clone();
        let report_tx = report_tx.clone();
        let (release_tx, release_rx) = mpsc::channel();
        releases.push(release_tx);
        let release_rx = Mutex::new(release_rx);
        strategies.push((
            name,
            Box::new(move |cx| {
                let guard = ActiveGuard::enter(&workers);
                let mut meter = ObservedMeter {
                    inner: cx,
                    spent: [0; DIMENSION_COUNT],
                    last_error: None,
                    after_batch: Some(|| {
                        let _callback_guard = ActiveGuard::enter(&callbacks);
                        ready_tx.send(name).unwrap();
                        release_rx
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .recv_timeout(LIFECYCLE_TIMEOUT)
                            .expect("controller must release the charged-batch callback");
                    }),
                };
                // Both real algorithms are inside a paid batch before the controller cancels.
                let error = run_real_generator(zassenhaus, &input, &mut meter).expect_err(
                    "the generator must observe owner cancellation, not publish factors",
                );
                assert_eq!(
                    meter.last_error,
                    Some(MeterError::Cancelled),
                    "{name}: {error}"
                );
                report_tx.send((name, meter.spent)).unwrap();
                drop(guard);
                drained_tx.send(name).unwrap();
                Err(PortfolioError::Cancelled)
            }),
        ));
    }
    drop(ready_tx);
    drop(drained_tx);
    drop(report_tx);

    let workers_for_controller = Arc::clone(&active_workers);
    let callbacks_for_controller = Arc::clone(&active_callbacks);
    let returned_for_controller = Arc::clone(&returned);
    let controller = std::thread::spawn(move || {
        let mut ready = [
            ready_rx
                .recv_timeout(LIFECYCLE_TIMEOUT)
                .expect("first paid batch"),
            ready_rx
                .recv_timeout(LIFECYCLE_TIMEOUT)
                .expect("second paid batch"),
        ];
        ready.sort();
        assert_eq!(ready, ["kronecker", "zassenhaus"]);
        cancel_cx.cancel_with(
            asupersync::CancelKind::User,
            Some("real factor charged batch"),
        );
        assert_eq!(workers_for_controller.load(Ordering::SeqCst), 2);
        assert_eq!(callbacks_for_controller.load(Ordering::SeqCst), 2);
        assert!(!returned_for_controller.load(Ordering::SeqCst));

        // Drain Kronecker first while Zassenhaus remains in its callback. This
        // orders the delay by events, not scheduler speed or a sleep duration.
        releases[1].send(()).unwrap();
        assert_eq!(
            drained_rx
                .recv_timeout(LIFECYCLE_TIMEOUT)
                .expect("Kronecker drain"),
            "kronecker"
        );
        assert_eq!(workers_for_controller.load(Ordering::SeqCst), 1);
        assert_eq!(callbacks_for_controller.load(Ordering::SeqCst), 1);
        assert!(!returned_for_controller.load(Ordering::SeqCst));
        releases[0].send(()).unwrap();
        assert_eq!(
            drained_rx
                .recv_timeout(LIFECYCLE_TIMEOUT)
                .expect("Zassenhaus drain"),
            "zassenhaus"
        );
    });

    let result = run_portfolio_concurrent_race(&mut fsym_cx, &context, &requested, strategies);
    returned.store(true, Ordering::SeqCst);
    // Capture quiescence at public return, before joining the external controller.
    let workers_at_return = active_workers.load(Ordering::SeqCst);
    let callbacks_at_return = active_callbacks.load(Ordering::SeqCst);
    controller
        .join()
        .expect("controller must complete without a timeout");
    assert_eq!(result, Err(PortfolioError::Cancelled));
    assert_eq!(workers_at_return, 0);
    assert_eq!(callbacks_at_return, 0);
    let mut reports: Vec<_> = report_rx.try_iter().collect();
    reports.sort_by_key(|(name, _)| *name);
    assert_eq!(
        reports.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        ["kronecker", "zassenhaus"]
    );
    for (_, spent) in &reports {
        assert!(spent[Dimension::ComputeSteps.index()] > 0);
        assert!(spent[Dimension::MemoryBytes.index()] > 0);
        assert!(spent[Dimension::AllocationCount.index()] > 0);
    }
    for dimension in Dimension::ALL {
        let spent: u64 = reports
            .iter()
            .map(|(_, spent)| spent[dimension.index()])
            .sum();
        assert_eq!(
            fsym_cx.remaining(dimension),
            limits.dimensions[dimension.index()] - spent,
            "cancellation must reconcile {dimension}, retaining all paid batches"
        );
    }
    assert_eq!(
        fsym_cx.verifier_remaining(),
        limits.verifier_pool - preflight
    );
}

// ============================================================================
// Phase 3: Cancellation Injected BEFORE Verifier Execution
// ============================================================================

#[test]
fn test_cancellation_matrix_before_verifier() {
    let cx_raw = Cx::detached_cancel_context();
    let cancel_cx = cx_raw.clone();
    let limits = BudgetLimits::uniform(200, 20);
    let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
    let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
    let x = Expr::symbol("x");
    let requested = Claim::equality(x.clone(), x.clone());

    let w1_x = x.clone();
    let cancel_cx_for_w1 = cancel_cx.clone();
    let s1 = Box::new(move |cx: &mut FsymCpuCx<'_, CapNone>| {
        cx.charge(Dimension::ComputeSteps, 5).unwrap();
        let cand = make_candidate(&w1_x, "s1");
        // Cancel right before returning the candidate from generator batch
        cancel_cx_for_w1.cancel_with(asupersync::CancelKind::User, Some("cancel before verifier"));
        Ok(cand)
    });

    let w2_x = x.clone();
    let s2 = Box::new(move |cx: &mut FsymCpuCx<'_, CapNone>| {
        cx.charge(Dimension::ComputeSteps, 3).unwrap();
        Ok(make_candidate(&w2_x, "s2"))
    });

    let result = run_portfolio_concurrent_race(
        &mut fsym_cx,
        &context,
        &requested,
        vec![("s1", s1), ("s2", s2)],
    );

    // Line 476 catches cancellation after drain and BEFORE any verifier runs
    assert_eq!(result, Err(PortfolioError::Cancelled));

    // Verifier pool was NOT touched by candidate verification (only 1 unit for requested-claim preflight)
    assert_eq!(
        fsym_cx.verifier_remaining(),
        19,
        "Verifier pool must remain intact without candidate verification charges when cancelled before verifier"
    );
}

// Actual post-verifier cancellation is exercised by the real-factor unit test
// in portfolio.rs, using its test-only owner-thread publication seam.

// ============================================================================
// Phase 5: Deterministic Replay Bit-for-Bit Reproduction
// ============================================================================

#[test]
fn test_cancellation_replay_log_deterministic_bit_for_bit() {
    let mut log = ReplayLog::new(42, "portfolio-cancellation-replay-v1").expect("new log");
    log.record_event(
        "portfolio.reservation",
        vec![(Dimension::ComputeSteps, 1)],
        b"limits=uniform(200,20);strategies=2",
    )
    .expect("record reservation");

    log.record_event(
        "portfolio.generator.batch",
        vec![(Dimension::ComputeSteps, 5)],
        b"worker=s1;steps=5",
    )
    .expect("record generator");

    log.record_event(
        "portfolio.cancellation",
        vec![(Dimension::ComputeSteps, 1)],
        b"reason=user_cancel;zero_orphans=true",
    )
    .expect("record cancel");

    let digest1 = log.finalize().expect("finalize log");

    // Verify hash chain and event integrity
    assert!(log.verify_integrity());

    // Serialize to JSON wire
    let wire = serde_json::to_string(&log).expect("serialize replay log");

    // Deserialize
    let log2: ReplayLog = serde_json::from_str(&wire).expect("deserialize replay log");
    assert!(log2.verify_integrity());
    assert!(log.verify_replay_match(&log2));
    let digest2 = log2.final_digest;

    assert_eq!(
        digest1, digest2,
        "Replay log must reproduce digest bit-for-bit across serialization round-trip"
    );
    assert_eq!(log.events.len(), log2.events.len());
}
