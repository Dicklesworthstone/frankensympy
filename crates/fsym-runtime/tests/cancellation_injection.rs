//! Cancellation-injection lab run and replay verification suite (WS13 / C6 gate).
//!
//! Constitutional requirements (Art. VII.7, VIII.7, XVII; WORKSTREAM_GRAPH.md §18, §32):
//! - Cancellation is request -> drain -> finalize with zero controlled orphan tasks.
//! - Candidate publication and accepted publication are separate phases.
//! - Verifier pool is protected and cannot be consumed by cancelled generators.
//! - Fallback and cancellation never reset budget accounting.
//! - Cancellation-injection matrix covering:
//!   1. Before reservation
//!   2. During generator batch
//!   3. Before verifier execution
//!   4. After verifier / before publication
//! - Replay log reproduces traces bit-for-bit.

#![forbid(unsafe_code)]

use asupersync::Cx;
use asupersync::cx::cap::None as CapNone;
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Budget, BudgetLimits, Dimension, Unbounded};
use fsym_core::Expr;
use fsym_proof_kernel::{Claim, ProofKernel};
use fsym_runtime::{
    FsymCpuCx, FsymCx, PortfolioCandidate, PortfolioError, ReplayLog,
    run_portfolio_concurrent_race, run_portfolio_race,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

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
fn test_cancellation_matrix_during_generator_batch() {
    let cx_raw = Cx::detached_cancel_context();
    let cancel_cx = cx_raw.clone();
    let limits = BudgetLimits::uniform(200, 20);
    let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
    let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
    let x = Expr::symbol("x");
    let requested = Claim::equality(x.clone(), x.clone());

    let worker_entered = Arc::new(AtomicBool::new(false));
    let worker_observed_cancel = Arc::new(AtomicBool::new(false));
    let active_worker_count = Arc::new(AtomicUsize::new(0));

    let entered_for_worker = Arc::clone(&worker_entered);
    let observed_for_worker = Arc::clone(&worker_observed_cancel);
    let count_for_worker1 = Arc::clone(&active_worker_count);

    let worker1 = Box::new(move |cx: &mut FsymCpuCx<'_, CapNone>| {
        count_for_worker1.fetch_add(1, Ordering::SeqCst);
        entered_for_worker.store(true, Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if cx.checkpoint().is_err() {
                observed_for_worker.store(true, Ordering::Release);
                count_for_worker1.fetch_sub(1, Ordering::SeqCst);
                return Err(PortfolioError::Cancelled);
            }
            std::hint::spin_loop();
        }
        count_for_worker1.fetch_sub(1, Ordering::SeqCst);
        Err(PortfolioError::StructuredExecutionFailed(
            "worker1 did not observe cancellation".into(),
        ))
    });

    let count_for_worker2 = Arc::clone(&active_worker_count);
    let worker2 = Box::new(move |_cx: &mut FsymCpuCx<'_, CapNone>| {
        count_for_worker2.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(10));
        count_for_worker2.fetch_sub(1, Ordering::SeqCst);
        Err(PortfolioError::AllStrategiesFailed("companion exit".into()))
    });

    let entered_for_controller = Arc::clone(&worker_entered);
    let controller = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !entered_for_controller.load(Ordering::Acquire) {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::yield_now();
        }
        cancel_cx.cancel_with(asupersync::CancelKind::User, Some("in-flight cancel"));
        true
    });

    let result = run_portfolio_concurrent_race(
        &mut fsym_cx,
        &context,
        &requested,
        vec![("w1", worker1), ("w2", worker2)],
    );

    assert!(controller.join().unwrap(), "controller thread must finish");
    assert_eq!(result, Err(PortfolioError::Cancelled));
    assert!(worker_observed_cancel.load(Ordering::Acquire));
    // Verify zero orphan workers remaining active
    assert_eq!(
        active_worker_count.load(Ordering::SeqCst),
        0,
        "Zero orphan workers must remain after scoped CPU drain"
    );
    // Budget was fully reconciled back to parent
    assert_eq!(fsym_cx.remaining(Dimension::ComputeSteps), 200);
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

// ============================================================================
// Phase 4: Cancellation Injected AFTER Verifier / BEFORE Publication
// ============================================================================

#[test]
fn test_cancellation_matrix_after_verifier_before_publication() {
    let cx_raw = Cx::detached_cancel_context();
    let cancel_cx = cx_raw.clone();
    let limits = BudgetLimits::uniform(200, 20);
    let mut fsym_cx = FsymCx::new(&cx_raw, Budget::new(limits), limits);
    let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
    let x = Expr::symbol("x");
    let requested = Claim::equality(x.clone(), x.clone());

    let cand_x = x.clone();
    let strategy = Box::new(move |cx: &mut FsymCx<'_, CapNone>| {
        cx.charge(Dimension::ComputeSteps, 5).unwrap();
        let cand = make_candidate(&cand_x, "strat");
        cancel_cx.cancel_with(
            asupersync::CancelKind::User,
            Some("cancel at publication barrier"),
        );
        Ok(cand)
    });

    let result = run_portfolio_race(
        &mut fsym_cx,
        &context,
        &requested,
        vec![("strat", strategy)],
    );

    // Post-verification checkpoint catches cancellation and refuses publication
    assert_eq!(
        result,
        Err(PortfolioError::Cancelled),
        "Cancellation after verifier must refuse publication"
    );
}

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
