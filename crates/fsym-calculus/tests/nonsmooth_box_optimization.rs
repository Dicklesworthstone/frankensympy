//! Candidate search and continuation at Abs corners; acceptance independently
//! reconstructs every original-source bound from the complete cover.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::{
    BoxOptimizationOptions, BoxOptimizationSearch, BoxOptimizationSearchStatus,
    MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, propose_box_optimization,
};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxPartitionStep, Expr, RealBall,
    RootBoxError, Symbol, SymbolIdentity, verify_box_optimization,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn bounds(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn abs(argument: Expr) -> Expr {
    Expr::Function("abs".into(), vec![argument])
}

fn shifted_corner(variable: &Symbol) -> Expr {
    abs(Expr::Add(vec![
        Expr::Sym(variable.clone()),
        Expr::Rational(q(-1, 3)),
    ]))
}

fn options() -> BoxOptimizationOptions {
    BoxOptimizationOptions {
        precision_digits: 16,
        max_leaves: 64,
    }
}

fn new_search(variable: &Symbol, meter: &mut impl BudgetMeter) -> BoxOptimizationSearch {
    BoxOptimizationSearch::new(
        &shifted_corner(variable),
        std::slice::from_ref(variable),
        &[bounds(0, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 65536),
        options(),
        meter,
    )
    .unwrap()
}

fn finish(search: &mut BoxOptimizationSearch, quantum: usize, meter: &mut impl BudgetMeter) {
    for _ in 0..options().max_leaves {
        if search.progress().status == BoxOptimizationSearchStatus::Complete {
            return;
        }
        let before = search.progress();
        let after = search.advance(quantum, meter).unwrap();
        assert!(after.split_count > before.split_count);
        assert!(after.split_count - before.split_count <= quantum);
    }
    panic!("bounded corner search did not finish");
}

#[test]
fn ordinary_search_finds_exact_cusp_optima_and_a_boundary_maximum() {
    let variable = Symbol::new("x");
    let absolute = abs(Expr::Sym(variable.clone()));
    for (source, domain, kind, point, value) in [
        (
            absolute.clone(),
            bounds(-1, 1),
            BoxExtremumKind::Minimum,
            q(0, 1),
            q(0, 1),
        ),
        (
            Expr::Mul(vec![Expr::from_i64(-1), absolute.clone()]),
            bounds(-1, 1),
            BoxExtremumKind::Maximum,
            q(0, 1),
            q(0, 1),
        ),
        (
            absolute,
            bounds(-1, 2),
            BoxExtremumKind::Maximum,
            q(2, 1),
            q(2, 1),
        ),
    ] {
        let proposed = propose_box_optimization(
            &source,
            std::slice::from_ref(&variable),
            std::slice::from_ref(&domain),
            kind,
            &q(1, 1000),
            options(),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(proposed.point, [point]);
        let receipt = verify_box_optimization(
            &source,
            std::slice::from_ref(&variable),
            std::slice::from_ref(&domain),
            kind,
            &q(1, 1000),
            &proposed,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(receipt.value(), &RealBall::exact(value));
        assert_eq!(receipt.gap(), &q(0, 1));
    }
}

#[test]
fn typed_multivariate_absolute_loss_uses_the_original_order_and_coordinates() {
    let plain = Symbol::new("same");
    let tagged = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([7; 32]));
    let source = Expr::Add(vec![
        abs(Expr::Add(vec![Expr::Sym(plain.clone()), Expr::from_i64(1)])),
        abs(Expr::Add(vec![
            Expr::Sym(tagged.clone()),
            Expr::from_i64(-2),
        ])),
    ]);
    let variables = [tagged, plain];
    let domain = [bounds(1, 3), bounds(-2, 0)];
    let candidate = propose_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        options(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(candidate.point, [q(2, 1), q(-1, 1)]);
    let receipt = verify_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        &candidate,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.variables(), variables);
    assert_eq!(receipt.value(), &RealBall::from_i64(0));
    assert!(
        verify_box_optimization(
            &source,
            &[variables[1].clone(), variables[0].clone()],
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn nondyadic_corner_searches_preserve_candidates_across_batch_sizes() {
    let variable = Symbol::new("x");
    let source = shifted_corner(&variable);
    let mut expected = None;
    for quantum in [1, 3, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS] {
        let mut search = new_search(&variable, &mut Unbounded);
        assert_eq!(
            search.progress().status,
            BoxOptimizationSearchStatus::Searching
        );
        finish(&mut search, quantum, &mut Unbounded);
        let witness = search.completed_witness().unwrap();
        assert_ne!(witness.point, [q(1, 3)]);
        assert!(search.progress().leaf_count > 1);
        let receipt = verify_box_optimization(
            &source,
            std::slice::from_ref(&variable),
            &[bounds(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 65536),
            witness,
            &mut Unbounded,
        )
        .unwrap();
        assert!(receipt.value().contains(&q(0, 1)));
        assert!(receipt.gap() > &q(0, 1) && receipt.gap() <= &q(1, 65536));
        assert_eq!(receipt.value().lower(), q(0, 1));
        if let Some(previous) = &expected {
            assert_eq!(witness, previous);
        } else {
            expected = Some(witness.clone());
        }
    }
}

#[test]
fn restored_corner_frontiers_recompute_bounds_and_retain_consumed_work() {
    let variable = Symbol::new("x");
    let source = shifted_corner(&variable);
    let mut original = new_search(&variable, &mut Unbounded);
    original.advance(3, &mut Unbounded).unwrap();
    let checkpoint = original.checkpoint(&mut Unbounded).unwrap();
    let frozen = checkpoint.clone();
    let allowance = 1 << 35;
    let mut receiver = Budget::new(BudgetLimits::uniform(allowance, 100_000_000));
    let mut restored = BoxOptimizationSearch::from_checkpoint(
        &source,
        std::slice::from_ref(&variable),
        &[bounds(0, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 65536),
        &checkpoint,
        &mut receiver,
    )
    .unwrap();
    assert_eq!(checkpoint, frozen);
    assert_eq!(
        restored.progress().leaf_count,
        original.progress().leaf_count
    );
    assert!(restored.counters().steps > checkpoint.counters.steps);
    assert!(restored.counters().allocation_bytes > checkpoint.counters.allocation_bytes);
    assert!(restored.counters().evaluations > checkpoint.counters.evaluations);
    assert!(receiver.remaining(Dimension::ComputeSteps) < allowance);
    finish(&mut original, 1, &mut Unbounded);
    finish(&mut restored, 3, &mut receiver);
    assert_eq!(restored.completed_witness(), original.completed_witness());
    let lease = receiver.verifier_lease().unwrap();
    let receipt = verify_box_optimization(
        &source,
        std::slice::from_ref(&variable),
        &[bounds(0, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 65536),
        restored.completed_witness().unwrap(),
        &mut receiver.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert!(receipt.value().contains(&q(0, 1)));
    let mut malformed = checkpoint.clone();
    malformed.partition = vec![BoxPartitionStep::Leaf];
    assert!(
        BoxOptimizationSearch::from_checkpoint(
            &source,
            std::slice::from_ref(&variable),
            &[bounds(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 65536),
            &malformed,
            &mut Unbounded
        )
        .is_err()
    );
    let mut forged = checkpoint;
    forged.incumbent_point = Some(vec![q(2, 1)]);
    assert!(
        BoxOptimizationSearch::from_checkpoint(
            &source,
            &[variable],
            &[bounds(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 65536),
            &forged,
            &mut Unbounded
        )
        .is_err()
    );
}

struct CancelAt {
    checkpoints: usize,
    at: Option<usize>,
    stopped: bool,
}

impl BudgetMeter for CancelAt {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        assert!(!self.stopped);
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped);
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.checkpoints += 1;
        if self.at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancelled_corner_subdivision_is_terminal_and_cannot_publish_or_reset_work() {
    let variable = Symbol::new("x");
    let mut probe = new_search(&variable, &mut Unbounded);
    let mut count = CancelAt {
        checkpoints: 0,
        at: None,
        stopped: false,
    };
    probe.advance(1, &mut count).unwrap();
    for at in [
        1,
        count.checkpoints / 3,
        count.checkpoints / 2,
        count.checkpoints,
    ] {
        let mut search = new_search(&variable, &mut Unbounded);
        let before = search.counters();
        let mut meter = CancelAt {
            checkpoints: 0,
            at: Some(at),
            stopped: false,
        };
        let error = search.advance(1, &mut meter).unwrap_err();
        assert_eq!(
            error,
            BoxOptimizationError::Verification(RootBoxError::Meter(MeterError::Cancelled))
        );
        assert_eq!(
            search.progress().status,
            BoxOptimizationSearchStatus::Failed
        );
        assert!(search.completed_witness().is_none());
        assert!(search.counters().steps >= before.steps);
        assert!(search.counters().evaluations >= before.evaluations);
        assert_eq!(search.advance(1, &mut meter).unwrap_err(), error);
        assert_eq!(meter.checkpoints, at);
        assert!(search.checkpoint(&mut Unbounded).is_err());
    }
}

#[test]
fn leaf_and_work_limits_never_turn_corner_candidates_into_accepted_results() {
    let variable = Symbol::new("x");
    let source = shifted_corner(&variable);
    assert!(matches!(
        propose_box_optimization(
            &source,
            std::slice::from_ref(&variable),
            &[bounds(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 65536),
            BoxOptimizationOptions {
                max_leaves: 1,
                ..options()
            },
            &mut Unbounded
        ),
        Err(BoxOptimizationError::AccuracyNotReached)
    ));
    let mut search = new_search(&variable, &mut Unbounded);
    let mut limits = BudgetLimits::uniform(100_000_000, 17);
    limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut meter = Budget::new(limits);
    let error = search.advance(1, &mut meter).unwrap_err();
    assert!(matches!(
        error,
        BoxOptimizationError::Verification(RootBoxError::Meter(_))
    ));
    assert!(search.completed_witness().is_none());
    assert_eq!(meter.verifier_remaining(), 17);
    assert_eq!(search.advance(1, &mut meter).unwrap_err(), error);
}

#[test]
fn original_zero_multiplied_poles_survive_search_subdivision() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    let reciprocal = Expr::Pow(Arc::new(x.clone()), Arc::new(Expr::from_i64(-1)));
    let source = Expr::Add(vec![abs(x), Expr::Mul(vec![Expr::from_i64(0), reciprocal])]);
    let result = propose_box_optimization(
        &source,
        &[variable],
        &[bounds(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        BoxOptimizationOptions {
            max_leaves: 8,
            ..options()
        },
        &mut Unbounded,
    );
    assert!(matches!(result, Err(BoxOptimizationError::Verification(_))));
}
