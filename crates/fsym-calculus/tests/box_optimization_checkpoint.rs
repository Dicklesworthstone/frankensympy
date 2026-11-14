//! Portable optimization candidates retain complete coverage and consumed work.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::{
    BoxOptimizationOptions, BoxOptimizationSearch, BoxOptimizationSearchCheckpoint,
    BoxOptimizationSearchStatus, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, propose_box_optimization,
};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxPartitionStep, Expr,
    MAX_BOX_OPTIMIZATION_DEPTH, MAX_BOX_OPTIMIZATION_LEAVES, MAX_ROOT_BOX_ALLOCATION_BYTES,
    MAX_ROOT_BOX_STEPS, RealBall, RootBoxError, Symbol, SymbolIdentity, verify_box_optimization,
};
use std::sync::Arc;

const ORDINARY_ALLOWANCE: u64 = 1 << 35;
const VERIFIER_ALLOWANCE: u64 = 100_000_000;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn power(source: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(source), Arc::new(Expr::from_i64(exponent)))
}

fn double_well(x: &Symbol) -> Expr {
    power(
        Expr::Add(vec![power(Expr::Sym(x.clone()), 2), Expr::from_i64(-2)]),
        2,
    )
}

fn options() -> BoxOptimizationOptions {
    BoxOptimizationOptions {
        precision_digits: 16,
        max_leaves: 256,
    }
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(
        ORDINARY_ALLOWANCE,
        VERIFIER_ALLOWANCE,
    ))
}

fn fixture(meter: &mut impl BudgetMeter) -> BoxOptimizationSearch {
    let x = Symbol::new("x");
    BoxOptimizationSearch::new(
        &double_well(&x),
        &[x],
        &[interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        options(),
        meter,
    )
    .unwrap()
}

fn saved() -> BoxOptimizationSearchCheckpoint {
    let mut search = fixture(&mut Unbounded);
    search.advance(7, &mut Unbounded).unwrap();
    search.checkpoint(&mut Unbounded).unwrap()
}

fn restore(
    checkpoint: &BoxOptimizationSearchCheckpoint,
    meter: &mut impl BudgetMeter,
) -> Result<BoxOptimizationSearch, BoxOptimizationError> {
    let x = Symbol::new("x");
    BoxOptimizationSearch::from_checkpoint(
        &double_well(&x),
        &[x],
        &[interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        checkpoint,
        meter,
    )
}

#[test]
fn restored_prefixes_keep_the_exact_witness_and_independently_verified_optimum() {
    let x = Symbol::new("x");
    let objective = double_well(&x);
    let domain = [interval(-2, 2)];
    let expected = propose_box_optimization(
        &objective,
        std::slice::from_ref(&x),
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        options(),
        &mut Unbounded,
    )
    .unwrap();
    let mut saw_searching = false;
    let mut saw_complete = false;
    for prefix in [0, 1, 3, 7, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS] {
        let mut source = fixture(&mut Unbounded);
        if prefix != 0 {
            source.advance(prefix, &mut Unbounded).unwrap();
        }
        let before = source.progress();
        saw_searching |= before.status == BoxOptimizationSearchStatus::Searching;
        saw_complete |= before.status == BoxOptimizationSearchStatus::Complete;
        let checkpoint = source.checkpoint(&mut Unbounded).unwrap();
        let frozen = checkpoint.clone();
        assert_eq!(checkpoint.counters, source.counters());
        let mut receiver = budget();
        let mut resumed = restore(&checkpoint, &mut receiver).unwrap();
        assert_eq!(checkpoint, frozen);
        assert_eq!(resumed.progress().status, before.status);
        assert_eq!(resumed.progress().leaf_count, before.leaf_count);
        assert!(resumed.counters().steps > checkpoint.counters.steps);
        assert!(resumed.counters().allocation_bytes > checkpoint.counters.allocation_bytes);
        assert!(resumed.counters().evaluations > checkpoint.counters.evaluations);
        assert!(receiver.remaining(Dimension::ComputeSteps) < ORDINARY_ALLOWANCE);
        assert_eq!(receiver.verifier_remaining(), VERIFIER_ALLOWANCE);
        resumed
            .advance(MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, &mut receiver)
            .unwrap();
        assert_eq!(resumed.completed_witness(), Some(&expected));
        let lease = receiver.verifier_lease().unwrap();
        let receipt = verify_box_optimization(
            &objective,
            std::slice::from_ref(&x),
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            resumed.completed_witness().unwrap(),
            &mut receiver.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
        // The nonnegative square has exact zeros at both +/-sqrt(2).
        assert!(receipt.value().contains(&q(0, 1)));
        assert!(receipt.gap() <= &q(1, 1000));
    }
    assert!(saw_searching && saw_complete);
}

#[test]
fn split_order_is_preserved_and_malformed_coverage_or_history_is_rejected() {
    let original = saved();
    assert!(original.split_order.len() >= 3);
    let mut resumed = restore(&original, &mut Unbounded).unwrap();
    let roundtrip = resumed.checkpoint(&mut Unbounded).unwrap();
    assert_eq!(roundtrip.partition, original.partition);
    assert_eq!(roundtrip.split_order, original.split_order);
    assert_eq!(roundtrip.incumbent_point, original.incumbent_point);
    assert!(roundtrip.counters.steps > original.counters.steps);

    let mut bad = original.clone();
    bad.partition.clear();
    assert!(restore(&bad, &mut Unbounded).is_err());
    let mut bad = original.clone();
    bad.partition[0] = BoxPartitionStep::Leaf;
    assert!(restore(&bad, &mut Unbounded).is_err(), "trailing subtrees");
    let mut bad = original.clone();
    *bad.partition.last_mut().unwrap() = BoxPartitionStep::Split { axis: 0 };
    assert!(restore(&bad, &mut Unbounded).is_err(), "missing children");
    let mut bad = original.clone();
    bad.partition[0] = BoxPartitionStep::Split { axis: usize::MAX };
    assert!(restore(&bad, &mut Unbounded).is_err(), "unknown coordinate");
    let mut bad = original.clone();
    bad.split_order[1] = bad.split_order[0];
    assert!(restore(&bad, &mut Unbounded).is_err(), "repeated split");
    let mut bad = original.clone();
    bad.split_order.swap(0, 1);
    assert!(restore(&bad, &mut Unbounded).is_err(), "child before parent");
    let mut bad = original.clone();
    bad.split_order[0] = usize::MAX;
    assert!(restore(&bad, &mut Unbounded).is_err(), "unbounded index");
    let mut bad = original.clone();
    bad.split_order.pop();
    assert!(restore(&bad, &mut Unbounded).is_err(), "omitted split");

    let depth = MAX_BOX_OPTIMIZATION_DEPTH + 1;
    let mut bad = original;
    bad.partition = vec![BoxPartitionStep::Split { axis: 0 }; depth];
    bad.partition
        .extend(std::iter::repeat_n(BoxPartitionStep::Leaf, depth + 1));
    bad.split_order = (0..depth).collect();
    bad.counters.evaluations = bad.counters.evaluations.max(bad.partition.len());
    bad.counters.steps += bad.partition.len() as u64;
    assert!(restore(&bad, &mut Unbounded).is_err(), "excessive depth");
}

#[test]
fn same_name_typed_axes_and_fixed_coordinates_survive_frontier_reconstruction() {
    let x = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([2; 32]));
    let objective = Expr::Add(vec![double_well(&x), power(Expr::Sym(y.clone()), 2)]);
    let variables = [x, y];
    let domain = [interval(-2, 2), interval(1, 1)];
    let tolerance = q(1, 1000);
    let mut source = BoxOptimizationSearch::new(
        &objective,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        options(),
        &mut Unbounded,
    )
    .unwrap();
    source.advance(3, &mut Unbounded).unwrap();
    let checkpoint = source.checkpoint(&mut Unbounded).unwrap();
    let mut resumed = BoxOptimizationSearch::from_checkpoint(
        &objective,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        &checkpoint,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(resumed.variables(), variables);
    assert_eq!(resumed.domain(), domain);
    resumed
        .advance(MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, &mut Unbounded)
        .unwrap();
    let receipt = verify_box_optimization(
        &objective,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        resumed.completed_witness().unwrap(),
        &mut Unbounded,
    )
    .unwrap();
    assert!(receipt.value().contains(&q(1, 1)));
    let mut bad = checkpoint;
    bad.partition[0] = BoxPartitionStep::Split { axis: 1 };
    assert!(BoxOptimizationSearch::from_checkpoint(
        &objective,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        &bad,
        &mut Unbounded,
    )
    .is_err());
}

#[test]
fn saved_points_and_completion_are_recomputed_without_trusting_prior_bounds() {
    let original = saved();
    let mut bad = original.clone();
    bad.incumbent_point = Some(vec![q(3, 1)]);
    assert!(restore(&bad, &mut Unbounded).is_err(), "infeasible point");
    let mut bad = original.clone();
    bad.incumbent_point = Some(vec![]);
    assert!(restore(&bad, &mut Unbounded).is_err(), "wrong dimension");
    let mut coarse = original;
    coarse.partition = vec![BoxPartitionStep::Leaf];
    coarse.split_order.clear();
    coarse.incumbent_point = Some(vec![q(0, 1)]);
    let mut resumed = restore(&coarse, &mut Unbounded).unwrap();
    assert_eq!(resumed.progress().status, BoxOptimizationSearchStatus::Searching);
    assert!(resumed.completed_witness().is_none());
    assert_eq!(resumed.incumbent_bound(), Some(&q(4, 1)));
    resumed
        .advance(MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, &mut Unbounded)
        .unwrap();
    let x = Symbol::new("x");
    let receipt = verify_box_optimization(
        &double_well(&x),
        &[x],
        &[interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        resumed.completed_witness().unwrap(),
        &mut Unbounded,
    )
    .unwrap();
    assert!(receipt.value().contains(&q(0, 1)));
}

#[test]
fn unknown_source_domain_and_absent_incumbent_never_become_accepted_zero() {
    let x = Symbol::new("x");
    let objective = Expr::Mul(vec![Expr::from_i64(0), power(Expr::Sym(x.clone()), -1)]);
    let variables = [x];
    let domain = [interval(-1, 1)];
    let tolerance = q(1, 1000);
    let mut search = BoxOptimizationSearch::new(
        &objective,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        options(),
        &mut Unbounded,
    )
    .unwrap();
    let checkpoint = search.checkpoint(&mut Unbounded).unwrap();
    assert!(checkpoint.incumbent_point.is_none());
    let resumed = BoxOptimizationSearch::from_checkpoint(
        &objective,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        &checkpoint,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(resumed.progress().status, BoxOptimizationSearchStatus::Searching);
    assert!(resumed.completed_witness().is_none());
    assert!(resumed.optimistic_bound().is_none());
    let mut bad = checkpoint;
    bad.incumbent_point = Some(vec![q(0, 1)]);
    assert!(BoxOptimizationSearch::from_checkpoint(
        &objective,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        &bad,
        &mut Unbounded,
    )
    .is_err(), "the saved point must retain the hidden pole");
}

#[test]
fn invalid_options_and_forged_consumption_cannot_reset_lifetime_limits() {
    let original = saved();
    for count in [0, MAX_BOX_OPTIMIZATION_LEAVES + 1, usize::MAX] {
        let mut bad = original.clone();
        bad.options.max_leaves = count;
        assert!(restore(&bad, &mut Unbounded).is_err());
    }
    for precision in [0, 7, 129, u32::MAX] {
        let mut bad = original.clone();
        bad.options.precision_digits = precision;
        assert!(restore(&bad, &mut Unbounded).is_err());
    }
    for field in 0..4 {
        let mut bad = original.clone();
        match field {
            0 => bad.counters.steps = 0,
            1 => bad.counters.allocation_bytes = 0,
            2 => bad.counters.source_bytes += 1,
            _ => bad.counters.evaluations = 0,
        }
        assert!(restore(&bad, &mut Unbounded).is_err(), "history field {field}");
    }
    for count in [MAX_ROOT_BOX_STEPS, u64::MAX] {
        let mut bad = original.clone();
        bad.counters.steps = count;
        assert!(restore(&bad, &mut Unbounded).is_err());
    }
    for count in [MAX_ROOT_BOX_ALLOCATION_BYTES, u64::MAX] {
        let mut bad = original.clone();
        bad.counters.allocation_bytes = count;
        assert!(restore(&bad, &mut Unbounded).is_err());
    }
    let mut bad = original;
    bad.counters.evaluations = usize::MAX;
    assert!(restore(&bad, &mut Unbounded).is_err());
}

#[test]
fn repeated_restore_adds_work_until_the_unchanged_intrinsic_limit_refuses() {
    let mut checkpoint = fixture(&mut Unbounded).checkpoint(&mut Unbounded).unwrap();
    let mut meter = budget();
    let mut restored = 0;
    for _ in 0..64 {
        let before = meter.remaining(Dimension::MemoryBytes);
        match restore(&checkpoint, &mut meter) {
            Ok(mut search) => {
                assert!(search.counters().steps > checkpoint.counters.steps);
                assert!(search.counters().allocation_bytes > checkpoint.counters.allocation_bytes);
                checkpoint = search.checkpoint(&mut meter).unwrap();
                restored += 1;
            }
            Err(BoxOptimizationError::Verification(RootBoxError::ResourceLimit(_))) => {
                assert!(restored >= 1);
                assert!(meter.remaining(Dimension::MemoryBytes) < before);
                assert_eq!(meter.verifier_remaining(), VERIFIER_ALLOWANCE);
                return;
            }
            Err(error) => panic!("unexpected restore refusal: {error}"),
        }
    }
    panic!("restoration reset the lifetime allocation limit");
}

struct CancelAt<'a> {
    budget: &'a mut Budget,
    checkpoints: usize,
    at: Option<usize>,
    stopped: bool,
}

impl BudgetMeter for CancelAt<'_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "work continued after cancellation");
        self.budget.charge(dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "work continued after cancellation");
        self.budget.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "checkpoint continued after cancellation");
        self.checkpoints += 1;
        if self.at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            self.budget.checkpoint()
        }
    }
}

#[test]
fn cancellation_at_restore_safe_points_never_publishes_a_partial_candidate() {
    let checkpoint = saved();
    let frozen = checkpoint.clone();
    let mut probe_budget = budget();
    let mut probe = CancelAt {
        budget: &mut probe_budget,
        checkpoints: 0,
        at: None,
        stopped: false,
    };
    restore(&checkpoint, &mut probe).unwrap();
    let count = probe.checkpoints;
    assert!(count > 3);
    let mut points = vec![1, 2, count / 2, count - 1, count];
    points.sort_unstable();
    points.dedup();
    for at in points {
        let mut meter = budget();
        let mut cancelled = CancelAt {
            budget: &mut meter,
            checkpoints: 0,
            at: Some(at),
            stopped: false,
        };
        let error = restore(&checkpoint, &mut cancelled).err().expect("cancellation");
        assert_eq!(
            error,
            BoxOptimizationError::Verification(RootBoxError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.checkpoints, at);
        assert_eq!(checkpoint, frozen);
        assert_eq!(meter.verifier_remaining(), VERIFIER_ALLOWANCE);
    }
}

#[test]
fn terminal_failures_cannot_export_half_completed_search_state() {
    let mut meter = budget();
    let mut search = fixture(&mut meter);
    let mut cancelled = CancelAt {
        budget: &mut meter,
        checkpoints: 0,
        at: Some(2),
        stopped: false,
    };
    let error = search.advance(1, &mut cancelled).unwrap_err();
    assert_eq!(search.progress().status, BoxOptimizationSearchStatus::Failed);
    let before = search.counters();
    // A stopped meter panics if consulted again: a terminal search must refuse
    // before attempting any copying, resource admission or publication.
    assert_eq!(search.checkpoint(&mut cancelled).unwrap_err(), error);
    assert_eq!(search.counters(), before);
    assert!(search.completed_witness().is_none());
    assert_eq!(meter.verifier_remaining(), VERIFIER_ALLOWANCE);
}
