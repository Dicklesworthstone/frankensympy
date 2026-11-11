//! Bounded optimization continuations retain their original statement and work.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::{
    BoxOptimizationOptions, BoxOptimizationSearch, BoxOptimizationSearchStatus,
    MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, propose_box_optimization,
};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationPrepared, Expr,
    MAX_ROOT_BOX_ALLOCATION_BYTES, MAX_ROOT_BOX_STEPS, RealBall, RootBoxError, Symbol,
    SymbolIdentity, verify_box_optimization,
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

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(
        ORDINARY_ALLOWANCE,
        VERIFIER_ALLOWANCE,
    ))
}

fn balances(budget: &Budget) -> Vec<u64> {
    Dimension::ALL
        .iter()
        .map(|&dimension| budget.remaining(dimension))
        .collect()
}

fn options() -> BoxOptimizationOptions {
    BoxOptimizationOptions {
        precision_digits: 16,
        max_leaves: 256,
    }
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

fn finish(search: &mut BoxOptimizationSearch, quantum: usize, meter: &mut impl BudgetMeter) {
    for _ in 0..options().max_leaves {
        let before = search.progress();
        if before.status == BoxOptimizationSearchStatus::Complete {
            return;
        }
        assert_eq!(before.status, BoxOptimizationSearchStatus::Searching);
        assert!(search.completed_witness().is_none());
        let after = search.advance(quantum, meter).unwrap();
        assert_eq!(after, search.progress());
        assert!(after.split_count > before.split_count);
        assert!(after.split_count - before.split_count <= quantum);
        assert_eq!(after.leaf_count, after.split_count + 1);
        assert!(after.evaluation_count >= before.evaluation_count);
    }
    panic!("the bounded search did not finish");
}

#[test]
fn schedules_preserve_the_exact_candidate_and_independent_global_claim() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let difference = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(y.clone())]),
    ]);
    let cases = [
        (
            double_well(&x),
            vec![x.clone()],
            vec![interval(-2, 2)],
            BoxExtremumKind::Minimum,
            q(0, 1),
        ),
        (
            Expr::Mul(vec![Expr::from_i64(-1), power(difference, 2)]),
            vec![y, x],
            vec![interval(-1, 1); 2],
            BoxExtremumKind::Minimum,
            q(-4, 1),
        ),
    ];
    for (source, variables, domain, kind, true_value) in cases {
        let tolerance = q(1, 1000);
        let one_shot = propose_box_optimization(
            &source,
            &variables,
            &domain,
            kind,
            &tolerance,
            options(),
            &mut Unbounded,
        )
        .unwrap();
        let mut expected_counters = None;
        for quantum in [1, 3, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS] {
            let mut meter = budget();
            let lease = meter.verifier_lease().unwrap();
            let mut search = BoxOptimizationSearch::new(
                &source,
                &variables,
                &domain,
                kind,
                &tolerance,
                options(),
                &mut meter,
            )
            .unwrap();
            assert_eq!(search.objective(), &source);
            assert_eq!(search.variables(), variables);
            assert_eq!(search.domain(), domain);
            assert_eq!(search.kind(), kind);
            assert_eq!(search.absolute_tolerance(), &tolerance);
            assert_eq!(search.options(), options());
            assert_eq!(search.progress().split_count, 0);
            assert!(search.progress().evaluation_count > 0);
            finish(&mut search, quantum, &mut meter);
            assert_eq!(search.completed_witness(), Some(&one_shot));
            assert_eq!(meter.verifier_remaining(), VERIFIER_ALLOWANCE);
            let counters = search.counters();
            assert!(counters.steps > 0);
            assert!(counters.allocation_bytes > 0);
            assert_eq!(counters.evaluations, search.progress().evaluation_count);
            if let Some(expected) = expected_counters {
                assert_eq!(counters, expected);
            } else {
                expected_counters = Some(counters);
            }
            let receipt = verify_box_optimization(
                &source,
                &variables,
                &domain,
                kind,
                &tolerance,
                search.completed_witness().unwrap(),
                &mut meter.verifier_meter(&lease).unwrap(),
            )
            .unwrap();
            assert!(receipt.value().contains(&true_value));
            assert!(receipt.gap() <= &tolerance);
            assert_eq!(receipt.leaf_count(), search.progress().leaf_count);
            assert!(meter.verifier_remaining() < VERIFIER_ALLOWANCE);
            let finished = search.progress();
            let before = balances(&meter);
            assert_eq!(search.advance(1, &mut meter).unwrap(), finished);
            assert_eq!(balances(&meter), before);
            assert_eq!(search.counters(), counters);
        }
    }
}

#[test]
fn frozen_search_owns_raw_source_buffers_and_typed_coordinate_identity() {
    let identity = SymbolIdentity::from_assumptions_digest([17; 32]);
    let mut name = String::with_capacity(64 * 1024);
    name.push('x');
    let variable = Symbol::with_identity(name, identity);
    let shared = Arc::new(power(Expr::Sym(variable.clone()), -1));
    let raw_reciprocal = Expr::Pow(Arc::clone(&shared), Arc::new(Expr::from_i64(0)));
    let mut operands = Vec::with_capacity(4096);
    operands.extend([Expr::from_i64(0), raw_reciprocal]);
    let mut source = Expr::Mul(operands);
    let original = source.clone();
    let mut variables = vec![variable.clone()];
    let mut domain = vec![interval(1, 2)];
    let search = BoxOptimizationSearch::new(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Maximum,
        &q(1, 1000),
        options(),
        &mut Unbounded,
    )
    .unwrap();
    let Expr::Mul(copied) = search.objective() else {
        panic!("raw zero product was lost");
    };
    assert_eq!(copied.capacity(), copied.len());
    let Expr::Pow(copied, exponent) = &copied[1] else {
        panic!("raw zero power was lost");
    };
    assert!(!Arc::ptr_eq(copied, &shared));
    assert_eq!(exponent.as_ref(), &Expr::from_i64(0));
    assert_eq!(search.variables()[0].identity, Some(identity));
    assert_eq!(search.variables()[0].name.capacity(), 1);
    source = Expr::from_i64(7);
    variables[0] = Symbol::new("different");
    domain[0] = interval(-1, 1);
    assert_ne!(search.objective(), &source);
    assert_eq!(search.objective(), &original);
    assert_eq!(search.variables(), [variable]);
    assert_eq!(search.domain(), [interval(1, 2)]);
}

#[test]
fn invalid_quanta_leave_the_frontier_and_ledger_unchanged() {
    let mut meter = budget();
    let mut search = fixture(&mut meter);
    let before = search.progress();
    let counters = search.counters();
    let remaining = balances(&meter);
    for quantum in [0, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS + 1, usize::MAX] {
        assert!(search.advance(quantum, &mut meter).is_err());
        assert_eq!(search.progress(), before);
        assert_eq!(search.counters(), counters);
        assert_eq!(balances(&meter), remaining);
        assert!(search.failure().is_none());
    }
    finish(&mut search, 1, &mut meter);
    assert!(search.completed_witness().is_some());
}

fn assert_failed(search: &BoxOptimizationSearch, expected: &BoxOptimizationError) {
    assert_eq!(
        search.progress().status,
        BoxOptimizationSearchStatus::Failed
    );
    assert_eq!(search.failure(), Some(expected));
    assert!(search.completed_witness().is_none());
    assert!(search.incumbent_point().is_none());
    assert!(search.incumbent_bound().is_none());
    assert!(search.optimistic_bound().is_none());
    assert!(search.value_gap().is_none());
}

#[test]
fn cumulative_ordinary_exhaustion_is_terminal_and_cannot_borrow_verifier_compute() {
    for dimension in [Dimension::ComputeSteps, Dimension::MemoryBytes] {
        let mut meter = budget();
        let mut search = fixture(&mut meter);
        search.advance(1, &mut meter).unwrap();
        let worked = search.counters();
        let remaining = meter.remaining(dimension);
        meter.charge(dimension, remaining).unwrap();
        let error = search.advance(1, &mut meter).unwrap_err();
        assert!(matches!(
            error,
            BoxOptimizationError::Verification(RootBoxError::Meter(MeterError::Budget(_)))
        ));
        assert_failed(&search, &error);
        assert!(search.counters().steps >= worked.steps);
        assert!(search.counters().allocation_bytes >= worked.allocation_bytes);
        assert!(search.counters().evaluations >= worked.evaluations);
        let failed_counters = search.counters();
        let failed_balances = balances(&meter);
        assert_eq!(search.advance(1, &mut meter).unwrap_err(), error);
        assert_eq!(search.counters(), failed_counters);
        assert_eq!(balances(&meter), failed_balances);
        assert_eq!(meter.verifier_remaining(), VERIFIER_ALLOWANCE);
    }
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
fn cancellation_during_split_never_exposes_half_published_bounds_or_restarts_work() {
    let mut probe_budget = budget();
    let mut probe = fixture(&mut probe_budget);
    let mut probe_meter = CancelAt {
        budget: &mut probe_budget,
        checkpoints: 0,
        at: None,
        stopped: false,
    };
    probe.advance(1, &mut probe_meter).unwrap();
    let count = probe_meter.checkpoints;
    assert!(count > 3);
    let mut points = vec![1, 2, count / 4, count / 2, count - 1, count];
    points.sort_unstable();
    points.dedup();
    for at in points {
        let mut meter = budget();
        let mut search = fixture(&mut meter);
        let before = search.counters();
        let mut cancelled = CancelAt {
            budget: &mut meter,
            checkpoints: 0,
            at: Some(at),
            stopped: false,
        };
        let error = search.advance(1, &mut cancelled).unwrap_err();
        assert_eq!(
            error,
            BoxOptimizationError::Verification(RootBoxError::Meter(MeterError::Cancelled))
        );
        assert_failed(&search, &error);
        assert!(search.counters().steps >= before.steps);
        assert!(search.counters().evaluations >= before.evaluations);
        // Even the same already-cancelled meter must not be touched on retry.
        assert_eq!(search.advance(1, &mut cancelled).unwrap_err(), error);
        assert_eq!(cancelled.checkpoints, at);
        assert_eq!(meter.verifier_remaining(), VERIFIER_ALLOWANCE);
    }
}

#[test]
fn intrinsic_leaf_limits_are_not_refilled_by_small_advance_requests() {
    let x = Symbol::new("x");
    let source = double_well(&x);
    let options = BoxOptimizationOptions {
        max_leaves: 3,
        ..options()
    };
    let expected = propose_box_optimization(
        &source,
        std::slice::from_ref(&x),
        &[interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1_000_000_000),
        options,
        &mut Unbounded,
    )
    .unwrap_err();
    let mut search = BoxOptimizationSearch::new(
        &source,
        &[x],
        &[interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1_000_000_000),
        options,
        &mut Unbounded,
    )
    .unwrap();
    let mut error = None;
    for _ in 0..options.max_leaves {
        if let Err(refused) = search.advance(1, &mut Unbounded) {
            error = Some(refused);
            break;
        }
    }
    assert_eq!(error, Some(expected.clone()));
    assert_failed(&search, &expected);
    assert!(search.progress().leaf_count <= options.max_leaves);
    assert_eq!(search.advance(1, &mut Unbounded).unwrap_err(), expected);
}

#[test]
fn failed_workspace_borrows_preserve_intrinsic_arithmetic_and_allocation_caps() {
    let x = Symbol::new("x");
    for allocation in [false, true] {
        let mut prepared = BoxOptimizationPrepared::new(
            &Expr::Sym(x.clone()),
            std::slice::from_ref(&x),
            &[interval(-1, 1)],
            16,
            &mut Unbounded,
        )
        .unwrap();
        let initial = prepared.counters();
        let cap = if allocation {
            MAX_ROOT_BOX_ALLOCATION_BYTES
        } else {
            MAX_ROOT_BOX_STEPS
        };
        let spent = if allocation {
            initial.allocation_bytes
        } else {
            initial.steps
        };
        let error = prepared
            .with_workspace(&mut Unbounded, |workspace| {
                if allocation {
                    workspace.account(0, cap - spent - 1, 0)?;
                    workspace.account(0, 2, 0)
                } else {
                    workspace.account(cap - spent - 1, 0, 0)?;
                    workspace.account(2, 0, 0)
                }
            })
            .unwrap_err();
        assert!(matches!(
            error,
            BoxOptimizationError::Verification(RootBoxError::ResourceLimit(_))
        ));
        let failed = prepared.counters();
        assert_eq!(
            if allocation {
                failed.allocation_bytes
            } else {
                failed.steps
            },
            cap - 1,
            "successful work preceding a refusal must remain charged"
        );
        prepared
            .with_workspace(&mut Unbounded, |workspace| {
                if allocation {
                    workspace.account(0, 1, 0)
                } else {
                    workspace.account(1, 0, 0)
                }
            })
            .unwrap();
        let exhausted = prepared.counters();
        assert!(
            prepared
                .with_workspace(&mut Unbounded, |workspace| {
                    if allocation {
                        workspace.account(0, 1, 0)
                    } else {
                        workspace.account(1, 0, 0)
                    }
                })
                .is_err()
        );
        assert_eq!(prepared.counters(), exhausted);
    }
}

#[test]
fn held_poles_cannot_become_completed_zero_objectives() {
    let x = Symbol::new("x");
    let reciprocal = power(Expr::Sym(x.clone()), -1);
    for source in [
        Expr::Mul(vec![Expr::from_i64(0), reciprocal.clone()]),
        power(reciprocal, 0),
    ] {
        let started = BoxOptimizationSearch::new(
            &source,
            std::slice::from_ref(&x),
            &[interval(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            BoxOptimizationOptions {
                max_leaves: 4,
                ..options()
            },
            &mut Unbounded,
        );
        if let Ok(mut search) = started {
            let mut failure = None;
            for _ in 0..4 {
                assert!(search.completed_witness().is_none());
                match search.advance(1, &mut Unbounded) {
                    Ok(_) => {}
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            assert_failed(
                &search,
                &failure.expect("unresolved source pole must refuse"),
            );
        }
    }
}
