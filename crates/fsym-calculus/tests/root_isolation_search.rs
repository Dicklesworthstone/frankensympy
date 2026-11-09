use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::root_isolation::{
    MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationOptions, RootIsolationSearch,
    propose_root_isolation,
};
use fsym_core::{
    BigRational, Expr, MAX_ROOT_ISOLATION_LEAVES, RealBall, RootBoxError, RootIsolationError,
    RootIsolationPrepared, RootIsolationWitness, Symbol, SymbolIdentity, verify_root_isolation,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

const ALLOWANCE: u64 = 1 << 35;
const VERIFIER_ALLOWANCE: u64 = 100_000_000;

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(ALLOWANCE, VERIFIER_ALLOWANCE))
}

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn power(source: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(source), Arc::new(Expr::from_i64(exponent)))
}

fn square(variable: &Symbol, constant: i64) -> Expr {
    Expr::Add(vec![
        power(Expr::Sym(variable.clone()), 2),
        Expr::from_i64(constant),
    ])
}

fn options() -> RootIsolationOptions {
    RootIsolationOptions {
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn finish(search: &mut RootIsolationSearch, batch: usize, meter: &mut impl BudgetMeter) {
    for _ in 0..MAX_ROOT_ISOLATION_ADVANCE_CELLS {
        if search.progress().ready_for_verification {
            return;
        }
        let before = search.progress();
        let after = search.advance(batch, meter).unwrap();
        assert_eq!(after, search.progress());
        assert!(after.processed_cells > before.processed_cells);
        assert!(after.processed_cells - before.processed_cells <= batch);
        assert_eq!(after.ready_for_verification, after.pending_cells == 0);
        assert_eq!(
            after.processed_cells + after.pending_cells,
            2 * after.leaf_count - 1
        );
        if !after.ready_for_verification {
            assert!(search.completed_witness().is_none());
        }
    }
    panic!("bounded search did not finish");
}

#[test]
fn batch_schedules_preserve_exact_cover_and_independent_root_counts() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let px = square(&x, -1);
    let cases = [
        (
            vec![square(&x, -2)],
            vec![x.clone()],
            vec![interval(-2, 2)],
            2,
        ),
        (
            vec![square(&x, 1)],
            vec![x.clone()],
            vec![interval(-2, 2)],
            0,
        ),
        (
            vec![px.clone(), Expr::Add(vec![px, square(&y, -1)])],
            vec![x, y],
            vec![interval(-2, 2), interval(-2, 2)],
            4,
        ),
    ];
    for (sources, variables, domain, count) in cases {
        let one_shot =
            propose_root_isolation(&sources, &variables, &domain, options(), &mut Unbounded)
                .unwrap();
        for batch in [1, 2, 7, MAX_ROOT_ISOLATION_ADVANCE_CELLS] {
            let mut meter = budget();
            let lease = meter.verifier_lease().unwrap();
            let mut search =
                RootIsolationSearch::new(&sources, &variables, &domain, options(), &mut meter)
                    .unwrap();
            assert_eq!(search.residuals(), sources);
            assert_eq!(search.variables(), variables);
            assert_eq!(search.domain(), domain);
            assert_eq!(search.options(), options());
            assert_eq!(search.progress().processed_cells, 0);
            assert_eq!(search.progress().pending_cells, 1);
            assert!(!search.progress().ready_for_verification);
            assert!(search.completed_witness().is_none());
            finish(&mut search, batch, &mut meter);
            assert_eq!(search.completed_witness().unwrap(), &one_shot);
            assert_eq!(search.progress().candidate_root_count, count);
            assert_eq!(meter.verifier_remaining(), VERIFIER_ALLOWANCE);
            // The completed candidate still goes through the independent
            // original-source complete-cover verifier on protected compute.
            let receipt = verify_root_isolation(
                &sources,
                &variables,
                &domain,
                search.completed_witness().unwrap(),
                &mut meter.verifier_meter(&lease).unwrap(),
            )
            .unwrap();
            assert_eq!(receipt.root_count(), count);
            assert_eq!(receipt.leaf_count(), search.progress().leaf_count);
            assert_eq!(
                receipt.excluded_leaf_count(),
                search.progress().excluded_cells
            );
            assert!(meter.verifier_remaining() < VERIFIER_ALLOWANCE);
            let completed = search.progress();
            assert_eq!(search.advance(1, &mut meter).unwrap(), completed);
            assert_eq!(search.completed_witness().unwrap(), &one_shot);
        }
    }
}

#[test]
fn freezing_rebuilds_arc_buffers_and_retains_raw_source_and_typed_identity() {
    let identity = SymbolIdentity::from_assumptions_digest([7; 32]);
    let mut name = String::with_capacity(64 * 1024);
    name.push('x');
    let variable = Symbol::with_identity(name, identity);
    let mut operands = Vec::with_capacity(4096);
    operands.extend([Expr::from_i64(0), Expr::Sym(variable.clone())]);
    let shared = Arc::new(Expr::Mul(operands));
    let expression = Expr::Pow(Arc::clone(&shared), Arc::new(Expr::from_i64(0)));
    let mut sources = vec![expression.clone()];
    let mut variables = vec![variable.clone()];
    let mut domain = vec![interval(-1, 1)];
    let mut meter = budget();
    let search =
        RootIsolationSearch::new(&sources, &variables, &domain, options(), &mut meter).unwrap();
    let Expr::Pow(copied, exponent) = &search.residuals()[0] else {
        panic!("raw power retained");
    };
    assert_eq!(exponent.as_ref(), &Expr::from_i64(0));
    assert!(!Arc::ptr_eq(copied, &shared));
    let Expr::Mul(copied_operands) = copied.as_ref() else {
        panic!("raw zero product retained");
    };
    assert_eq!(copied_operands.len(), 2);
    assert_eq!(copied_operands.capacity(), 2);
    let Expr::Sym(copied_symbol) = &copied_operands[1] else {
        panic!("typed original symbol retained");
    };
    assert_eq!(copied_symbol.identity, Some(identity));
    assert_eq!(copied_symbol.name.capacity(), 1);
    assert_eq!(search.variables()[0].name.capacity(), 1);
    sources[0] = Expr::from_i64(99);
    variables[0] = Symbol::new("other");
    domain[0] = interval(10, 11);
    assert_eq!(search.residuals(), &[expression]);
    assert_eq!(search.variables(), &[variable]);
    assert_eq!(search.domain(), &[interval(-1, 1)]);
}

struct CancelMeter<'a> {
    ledger: &'a mut Budget,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CancelMeter<'_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.ledger.charge(dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.ledger.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            self.ledger.checkpoint()
        }
    }
}

fn search_fixture(meter: &mut impl BudgetMeter) -> RootIsolationSearch {
    let x = Symbol::new("x");
    RootIsolationSearch::new(
        &[square(&x, -2)],
        &[x],
        &[interval(-2, 2)],
        options(),
        meter,
    )
    .unwrap()
}

fn reference_witness() -> RootIsolationWitness {
    let x = Symbol::new("x");
    propose_root_isolation(
        &[square(&x, -2)],
        &[x],
        &[interval(-2, 2)],
        options(),
        &mut Unbounded,
    )
    .unwrap()
}

#[test]
fn every_advance_cancellation_safe_point_preserves_a_resumable_cover_on_the_same_ledger() {
    let mut probe_budget = budget();
    let mut probe = search_fixture(&mut probe_budget);
    let mut probe_meter = CancelMeter {
        ledger: &mut probe_budget,
        checkpoints: 0,
        cancel_at: None,
    };
    probe.advance(3, &mut probe_meter).unwrap();
    let safe_points = probe_meter.checkpoints;
    assert!(safe_points > 3);
    let expected = reference_witness();
    for cancel_at in 1..=safe_points {
        let mut ledger = budget();
        let lease = ledger.verifier_lease().unwrap();
        let mut search = search_fixture(&mut ledger);
        let remaining = ledger.remaining(Dimension::ComputeSteps);
        let mut meter = CancelMeter {
            ledger: &mut ledger,
            checkpoints: 0,
            cancel_at: Some(cancel_at),
        };
        assert_eq!(
            search.advance(3, &mut meter),
            Err(RootIsolationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            ))),
        );
        assert_eq!(meter.checkpoints, cancel_at);
        assert!(meter.ledger.remaining(Dimension::ComputeSteps) <= remaining);
        let paused = search.progress();
        assert!(paused.pending_cells > 0);
        assert_eq!(
            paused.processed_cells + paused.pending_cells,
            2 * paused.leaf_count - 1
        );
        assert!(search.completed_witness().is_none());
        meter.cancel_at = None;
        finish(&mut search, 4, &mut meter);
        assert_eq!(
            search.completed_witness().unwrap(),
            &expected,
            "cancel point {cancel_at}"
        );
        assert_eq!(meter.ledger.verifier_remaining(), VERIFIER_ALLOWANCE);
        let receipt = verify_root_isolation(
            search.residuals(),
            search.variables(),
            search.domain(),
            search.completed_witness().unwrap(),
            &mut meter.ledger.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt.root_count(), 2);
    }
}

#[test]
fn budget_refusal_keeps_prior_cell_transitions_and_cannot_replenish_by_retry() {
    let mut probe_budget = budget();
    let mut probe = search_fixture(&mut probe_budget);
    let construction = ALLOWANCE - probe_budget.remaining(Dimension::ComputeSteps);
    let before = probe_budget.remaining(Dimension::ComputeSteps);
    let first = probe.advance(1, &mut probe_budget).unwrap();
    let first_cost = before - probe_budget.remaining(Dimension::ComputeSteps);
    assert_eq!(first.processed_cells, 1);
    assert_eq!(first.pending_cells, 2);

    let mut limits = BudgetLimits::uniform(ALLOWANCE, VERIFIER_ALLOWANCE);
    limits.dimensions[Dimension::ComputeSteps.index()] = construction + first_cost;
    let mut ledger = Budget::new(limits);
    let mut search = search_fixture(&mut ledger);
    assert!(matches!(
        search.advance(2, &mut ledger),
        Err(RootIsolationError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(search.progress(), first);
    assert!(search.completed_witness().is_none());
    assert_eq!(ledger.remaining(Dimension::ComputeSteps), 0);
    let memory = ledger.remaining(Dimension::MemoryBytes);
    for _ in 0..3 {
        assert!(search.advance(1, &mut ledger).is_err());
        assert_eq!(search.progress(), first);
        assert_eq!(ledger.remaining(Dimension::ComputeSteps), 0);
        assert_eq!(ledger.remaining(Dimension::MemoryBytes), memory);
        assert_eq!(ledger.verifier_remaining(), VERIFIER_ALLOWANCE);
    }
}

#[test]
fn invalid_batches_and_hard_search_limits_do_not_discard_or_accept_frontier() {
    let mut meter = budget();
    let mut search = search_fixture(&mut meter);
    let initial = search.progress();
    for batch in [0, MAX_ROOT_ISOLATION_ADVANCE_CELLS + 1, usize::MAX] {
        assert!(matches!(
            search.advance(batch, &mut meter),
            Err(RootIsolationError::Verification(
                RootBoxError::ResourceLimit(_)
            ))
        ));
        assert_eq!(search.progress(), initial);
    }
    let x = Symbol::new("x");
    let mut limited = RootIsolationSearch::new(
        &[square(&x, -2)],
        &[x],
        &[interval(-2, 2)],
        RootIsolationOptions {
            max_leaves: 1,
            ..options()
        },
        &mut meter,
    )
    .unwrap();
    let initial = limited.progress();
    for _ in 0..2 {
        let before = meter.remaining(Dimension::ComputeSteps);
        assert!(matches!(
            limited.advance(1, &mut meter),
            Err(RootIsolationError::Incomplete(_))
        ));
        assert_eq!(limited.progress(), initial);
        assert!(limited.completed_witness().is_none());
        assert!(meter.remaining(Dimension::ComputeSteps) < before);
    }
}

#[test]
fn prepared_workspace_keeps_scratch_and_intrinsic_counters_across_failure_and_unwinding() {
    let x = Symbol::new("x");
    let domain = [interval(-1, 1)];
    let mut meter = budget();
    let mut prepared =
        RootIsolationPrepared::new(&[Expr::from_i64(1)], &[x], &domain, 16, &mut meter).unwrap();
    let before = meter.remaining(Dimension::MemoryBytes);
    for _ in 0..100 {
        prepared
            .with_workspace(&mut meter, |workspace| workspace.account(1, 0, 0))
            .unwrap();
    }
    assert_eq!(
        meter.remaining(Dimension::MemoryBytes),
        before,
        "scratch is admitted once"
    );
    let refused: Result<(), RootIsolationError> =
        prepared.with_workspace(&mut meter, |workspace| {
            workspace.enclose_cell(&domain)?;
            Err(RootIsolationError::Incomplete(
                "injected after interval work",
            ))
        });
    assert!(refused.is_err());
    assert_eq!(prepared.evaluation_count(), 1);
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _: Result<(), RootIsolationError> = prepared.with_workspace(&mut meter, |workspace| {
            workspace.enclose_cell(&domain)?;
            panic!("injected after interval work");
        });
    }));
    assert!(panicked.is_err());
    assert_eq!(prepared.evaluation_count(), 2);

    prepared
        .with_workspace(&mut meter, |workspace| workspace.account(50_000_000, 0, 0))
        .unwrap();
    assert!(matches!(
        prepared.with_workspace(&mut meter, |workspace| workspace.account(50_000_000, 0, 0)),
        Err(RootIsolationError::Verification(
            RootBoxError::ResourceLimit("cumulative arithmetic work")
        )),
    ));
    let x = Symbol::new("x");
    let mut prepared =
        RootIsolationPrepared::new(&[Expr::from_i64(1)], &[x], &domain, 16, &mut meter).unwrap();
    prepared
        .with_workspace(&mut meter, |workspace| {
            workspace.account(1, 100 * 1024 * 1024, 1)
        })
        .unwrap();
    assert!(matches!(
        prepared.with_workspace(&mut meter, |workspace| workspace.account(
            1,
            100 * 1024 * 1024,
            1
        )),
        Err(RootIsolationError::Verification(
            RootBoxError::ResourceLimit("cumulative allocation")
        )),
    ));
}

#[test]
fn evaluation_limit_is_shared_by_all_prepared_batches() {
    let x = Symbol::new("x");
    let domain = [interval(-1, 1)];
    let mut prepared =
        RootIsolationPrepared::new(&[Expr::from_i64(1)], &[x], &domain, 16, &mut Unbounded)
            .unwrap();
    let mut refused = None;
    for _ in 0..500 {
        let result = prepared.with_workspace(&mut Unbounded, |workspace| {
            for _ in 0..37 {
                workspace.enclose_cell(&domain)?;
            }
            Ok(())
        });
        if let Err(error) = result {
            refused = Some(error);
            break;
        }
    }
    assert_eq!(
        refused,
        Some(RootIsolationError::Verification(
            RootBoxError::ResourceLimit("root-isolation evaluation count")
        ))
    );
    assert_eq!(
        prepared.evaluation_count(),
        16 * MAX_ROOT_ISOLATION_LEAVES + 8
    );
    assert!(
        prepared
            .with_workspace(&mut Unbounded, |workspace| workspace.enclose_cell(&domain))
            .is_err()
    );
    assert_eq!(
        prepared.evaluation_count(),
        16 * MAX_ROOT_ISOLATION_LEAVES + 8
    );
}

#[test]
fn original_poles_and_distinct_same_print_symbols_remain_unresolved() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let hidden = Expr::Mul(vec![Expr::from_i64(0), power(Expr::Sym(x.clone()), -1)]);
    let mut search = RootIsolationSearch::new(
        std::slice::from_ref(&hidden),
        std::slice::from_ref(&x),
        &[interval(-1, 1)],
        RootIsolationOptions {
            max_leaves: 8,
            ..options()
        },
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(search.residuals(), &[hidden]);
    assert!(
        search
            .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
            .is_err()
    );
    assert!(!search.progress().ready_for_verification);
    assert!(search.completed_witness().is_none());
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let mut mismatched = RootIsolationSearch::new(
        &[Expr::Sym(x)],
        &[other],
        &[interval(-1, 1)],
        options(),
        &mut Unbounded,
    )
    .unwrap();
    let initial = mismatched.progress();
    assert!(matches!(
        mismatched.advance(1, &mut Unbounded),
        Err(RootIsolationError::Verification(RootBoxError::Unsupported(
            _
        )))
    ));
    assert_eq!(mismatched.progress(), initial);
    assert!(mismatched.completed_witness().is_none());
}
