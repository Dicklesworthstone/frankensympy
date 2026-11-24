//! Complete-cover search at finite continuous boundaries. The proposal is
//! independently replayed by the core verifier; no derivative or optimizer
//! output is treated as mathematical evidence.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::{
    BoxOptimizationOptions, BoxOptimizationSearch, BoxOptimizationSearchStatus,
    propose_box_optimization,
};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxPartitionStep, Expr, RealBall,
    RootBoxError, Symbol, SymbolIdentity, verify_box_optimization,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: BigRational) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::Rational(exponent)))
}

fn options() -> BoxOptimizationOptions {
    BoxOptimizationOptions {
        precision_digits: 16,
        max_leaves: 64,
    }
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
    panic!("bounded continuous search did not finish");
}

#[test]
fn search_certifies_finite_endpoint_extrema_by_value_gap() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    for (source, domain, kind, expected) in [
        (
            function("sqrt", x.clone()),
            interval(0, 4),
            BoxExtremumKind::Minimum,
            0,
        ),
        (
            function("sqrt", x.clone()),
            interval(0, 4),
            BoxExtremumKind::Maximum,
            2,
        ),
        (
            power(x.clone(), q(1, 3)),
            interval(0, 8),
            BoxExtremumKind::Minimum,
            0,
        ),
        (
            function("acos", x.clone()),
            interval(0, 1),
            BoxExtremumKind::Minimum,
            0,
        ),
        (
            function("acosh", x),
            interval(1, 2),
            BoxExtremumKind::Minimum,
            0,
        ),
    ] {
        let tolerance = q(1, 16);
        let candidate = propose_box_optimization(
            &source,
            std::slice::from_ref(&variable),
            std::slice::from_ref(&domain),
            kind,
            &tolerance,
            options(),
            &mut Unbounded,
        )
        .unwrap();
        let receipt = verify_box_optimization(
            &source,
            std::slice::from_ref(&variable),
            std::slice::from_ref(&domain),
            kind,
            &tolerance,
            &candidate,
            &mut Unbounded,
        )
        .unwrap();
        assert!(receipt.value().contains(&q(expected, 1)), "{source}");
        assert!(receipt.gap() <= &tolerance);
        assert!(receipt.leaf_count() <= options().max_leaves);
        assert_eq!(receipt.objective(), &source);
        assert_eq!(receipt.witness(), &candidate);
    }
}

fn cusp(variable: &Symbol) -> Expr {
    function(
        "sqrt",
        function(
            "abs",
            Expr::Add(vec![Expr::Sym(variable.clone()), Expr::Rational(q(-1, 3))]),
        ),
    )
}

fn cusp_search(variable: &Symbol, meter: &mut impl BudgetMeter) -> BoxOptimizationSearch {
    BoxOptimizationSearch::new(
        &cusp(variable),
        std::slice::from_ref(variable),
        &[interval(0, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 32),
        options(),
        meter,
    )
    .unwrap()
}

#[test]
fn nonlipschitz_interior_cusp_is_deterministic_across_cooperative_batches() {
    let variable = Symbol::new("x");
    let mut expected = None;
    let mut counters = None;
    for quantum in [1, 3, 64] {
        let mut search = cusp_search(&variable, &mut Unbounded);
        assert_eq!(
            search.progress().status,
            BoxOptimizationSearchStatus::Searching
        );
        finish(&mut search, quantum, &mut Unbounded);
        let candidate = search.completed_witness().unwrap();
        let receipt = verify_box_optimization(
            &cusp(&variable),
            std::slice::from_ref(&variable),
            &[interval(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 32),
            candidate,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(receipt.value().lower(), q(0, 1));
        assert!(receipt.gap() > &q(0, 1));
        assert!(receipt.gap() <= &q(1, 32));
        assert_ne!(candidate.point, [q(1, 3)]);
        if let Some(previous) = &expected {
            assert_eq!(candidate, previous);
        } else {
            expected = Some(candidate.clone());
        }
        if let Some(previous) = counters {
            assert_eq!(search.counters(), previous);
        } else {
            counters = Some(search.counters());
        }
    }
}

#[test]
fn range_only_cells_choose_width_without_fabricating_zero_slopes() {
    let variables = [Symbol::new("x"), Symbol::new("unused")];
    let source = function("sqrt", Expr::Sym(variables[0].clone()));
    let mut search = BoxOptimizationSearch::new(
        &source,
        &variables,
        &[interval(0, 1), interval(0, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 16),
        options(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        search.incumbent_point(),
        Some([q(1, 2), q(1, 1)].as_slice())
    );
    search.advance(1, &mut Unbounded).unwrap();
    let checkpoint = search.checkpoint(&mut Unbounded).unwrap();
    // No slope certificate exists on the original box. Both widths remain
    // candidates even when one variable does not occur in the source.
    assert_eq!(
        checkpoint.partition[0],
        BoxPartitionStep::Split { axis: 1 }
    );
    assert_eq!(checkpoint.partition.len(), 3);
}

#[test]
fn continuous_checkpoint_replays_sources_and_retains_prior_consumption() {
    let variable = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32]));
    let source = cusp(&variable);
    let mut original = cusp_search(&variable, &mut Unbounded);
    original.advance(2, &mut Unbounded).unwrap();
    let checkpoint = original.checkpoint(&mut Unbounded).unwrap();
    let saved = checkpoint.clone();
    let allowance = 1 << 40;
    let mut receiver = Budget::new(BudgetLimits::uniform(allowance, 17));
    let mut restored = BoxOptimizationSearch::from_checkpoint(
        &source,
        std::slice::from_ref(&variable),
        &[interval(0, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 32),
        &checkpoint,
        &mut receiver,
    )
    .unwrap();
    assert_eq!(checkpoint, saved);
    assert!(restored.counters().steps > checkpoint.counters.steps);
    assert!(restored.counters().allocation_bytes > checkpoint.counters.allocation_bytes);
    assert!(restored.counters().evaluations > checkpoint.counters.evaluations);
    assert!(receiver.remaining(Dimension::ComputeSteps) < allowance);
    assert_eq!(receiver.verifier_remaining(), 17);
    finish(&mut original, 1, &mut Unbounded);
    finish(&mut restored, 3, &mut receiver);
    assert_eq!(restored.completed_witness(), original.completed_witness());
    let receipt = verify_box_optimization(
        &source,
        std::slice::from_ref(&variable),
        &[interval(0, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 32),
        restored.completed_witness().unwrap(),
        &mut Unbounded,
    )
    .unwrap();
    assert!(receipt.value().contains(&q(0, 1)));
    assert!(receipt.gap() <= &q(1, 32));

    // An identically printed but different axis cannot stand in for the
    // original typed coordinate during restored bound evaluation.
    assert!(
        BoxOptimizationSearch::from_checkpoint(
            &source,
            &[Symbol::new("x")],
            &[interval(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 32),
            &checkpoint,
            &mut Unbounded,
        )
        .is_err()
    );
    let mut malformed = checkpoint;
    malformed.partition.pop();
    assert!(
        BoxOptimizationSearch::from_checkpoint(
            &source,
            &[variable],
            &[interval(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 32),
            &malformed,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn genuine_domain_failures_never_become_complete_candidates() {
    let variable = Symbol::new("x");
    let source = Expr::Add(vec![
        function("sqrt", Expr::Sym(variable.clone())),
        Expr::Mul(vec![
            Expr::from_i64(0),
            power(Expr::Sym(variable.clone()), q(-1, 1)),
        ]),
    ]);
    let result = propose_box_optimization(
        &source,
        &[variable],
        &[interval(0, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 16),
        BoxOptimizationOptions {
            max_leaves: 8,
            ..options()
        },
        &mut Unbounded,
    );
    assert!(result.is_err());
}

struct Cancel;

impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn cancellation_and_exhaustion_are_terminal_without_a_candidate_or_budget_reset() {
    let variable = Symbol::new("x");
    for cancel in [false, true] {
        let mut search = cusp_search(&variable, &mut Unbounded);
        let before = search.counters();
        let mut tiny = Budget::new(BudgetLimits::uniform(0, 17));
        let failure = if cancel {
            search.advance(1, &mut Cancel)
        } else {
            search.advance(1, &mut tiny)
        }
        .unwrap_err();
        assert!(matches!(
            &failure,
            BoxOptimizationError::Verification(RootBoxError::Meter(_)),
        ));
        assert_eq!(
            search.progress().status,
            BoxOptimizationSearchStatus::Failed
        );
        assert!(search.completed_witness().is_none());
        assert!(search.incumbent_point().is_none());
        assert!(search.counters().steps >= before.steps);
        assert_eq!(search.advance(1, &mut Unbounded), Err(failure));
        assert_eq!(tiny.verifier_remaining(), 17);
    }
}
