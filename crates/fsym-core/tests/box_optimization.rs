//! Exact analytic and adversarial checks of complete finite-box optimization
//! witnesses. No candidate search algorithm is used to construct expectations.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationReceipt,
    BoxOptimizationWitness, BoxOptimizationWorkspace, BoxPartitionStep, Expr,
    MAX_BOX_OPTIMIZATION_DEPTH, MAX_BOX_OPTIMIZATION_NODES, RealBall, RootBoxError, Symbol,
    certify_box_optimization, verify_box_optimization,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn bounds(lo: i64, hi: i64) -> RealBall {
    RealBall::new(q(lo + hi, 2), q(hi - lo, 2)).unwrap()
}
fn pow(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn partition(depth: usize, axis: usize) -> Vec<BoxPartitionStep> {
    if depth == 0 {
        vec![BoxPartitionStep::Leaf]
    } else {
        let mut result = vec![BoxPartitionStep::Split { axis }];
        result.extend(partition(depth - 1, axis));
        result.extend(partition(depth - 1, axis));
        result
    }
}
fn witness(point: Vec<BigRational>, depth: usize) -> BoxOptimizationWitness {
    BoxOptimizationWitness {
        partition: partition(depth, 0),
        point,
        precision_digits: 24,
    }
}
fn check(
    source: &Expr,
    vars: &[Symbol],
    domain: &[RealBall],
    kind: BoxExtremumKind,
    tolerance: &BigRational,
    proposed: &BoxOptimizationWitness,
) -> Result<BoxOptimizationReceipt, BoxOptimizationError> {
    certify_box_optimization(source, vars, domain, kind, tolerance, proposed)
}

#[test]
fn nonconvex_quartic_has_a_certified_global_value_with_irrational_optimizers() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    // f = (x^2 - 3/2)^2 - 9/4, whose minimum is exactly -9/4.
    let source = pow(x.clone(), 4) - Expr::from_i64(3) * pow(x, 2);
    let proposed = witness(vec![q(39, 32)], 8);
    let tolerance = q(1, 1000);
    let receipt = check(
        &source,
        &vars,
        &[bounds(-2, 2)],
        BoxExtremumKind::Minimum,
        &tolerance,
        &proposed,
    )
    .unwrap();
    assert_eq!(receipt.objective(), &source);
    assert_eq!(receipt.variables(), vars);
    assert_eq!(receipt.domain(), [bounds(-2, 2)]);
    assert_eq!(receipt.witness(), &proposed);
    assert_eq!(receipt.point(), [q(39, 32)]);
    assert_eq!(receipt.kind(), BoxExtremumKind::Minimum);
    assert_eq!(receipt.absolute_tolerance(), &tolerance);
    assert_eq!(receipt.precision_digits(), 24);
    assert_eq!(receipt.leaf_count(), 256);
    assert!(receipt.value().contains(&q(-9, 4)));
    assert!(receipt.gap() <= &tolerance);
    assert_eq!(&receipt.value().width(), receipt.gap());
    let candidate_error = q(225, 1_048_576);
    assert_eq!(
        receipt.point_value(),
        &RealBall::exact(q(-9, 4) + &candidate_error)
    );
    assert!(candidate_error <= *receipt.gap());

    let maximum = check(
        &(-source),
        &vars,
        &[bounds(-2, 2)],
        BoxExtremumKind::Maximum,
        &tolerance,
        &proposed,
    )
    .unwrap();
    assert!(maximum.value().contains(&q(9, 4)));
    assert!(maximum.gap() <= &tolerance);
}

#[test]
fn centered_taylor_bound_proves_expanded_quadratic_optimality_without_subdivision() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let source = pow(x.clone(), 2) + Expr::from_i64(2) * x + Expr::from_i64(1);
    let receipt = check(
        &source,
        &vars,
        &[bounds(-2, 0)],
        BoxExtremumKind::Minimum,
        &q(1, 1_000_000),
        &witness(vec![q(-1, 1)], 0),
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(0));
    assert_eq!(receipt.gap(), &q(0, 1));
    assert_eq!(receipt.leaf_count(), 1);
}

#[test]
fn taylor_mixed_hessian_and_negative_diagonal_cannot_create_false_bounds() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = Expr::from_i64(2) * Expr::Sym(vars[0].clone()) * Expr::Sym(vars[1].clone());
    // The true minimum is -2. Dropping one mixed Hessian entry and applying
    // the 1/2 factor would fabricate a lower bound -1 and accept this point.
    assert_eq!(
        check(
            &source,
            &vars,
            &[bounds(-1, 1), bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &witness(vec![q(0, 1), q(0, 1)], 0)
        ),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
    let source = -pow(Expr::Sym(vars[0].clone()), 2);
    assert_eq!(
        check(
            &source,
            &vars[..1],
            &[bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 2),
            &witness(vec![q(0, 1)], 0)
        ),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
    let maximum = check(
        &source,
        &vars[..1],
        &[bounds(-1, 1)],
        BoxExtremumKind::Maximum,
        &q(1, 2),
        &witness(vec![q(0, 1)], 0),
    )
    .unwrap();
    assert_eq!(maximum.value(), &RealBall::from_i64(0));
}

#[test]
fn global_bound_uses_every_leaf_in_the_correct_direction() {
    let vars = [Symbol::new("x")];
    let source = -Expr::Sym(vars[0].clone());
    // The candidate value is 1, but the global minimum is -1 in the right
    // leaf. Aggregating the larger lower bound would falsely report gap 1.
    assert_eq!(
        check(
            &source,
            &vars,
            &[bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(3, 2),
            &witness(vec![q(-1, 1)], 1)
        ),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
    assert_eq!(
        check(
            &source,
            &vars,
            &[bounds(-1, 1)],
            BoxExtremumKind::Maximum,
            &q(3, 2),
            &witness(vec![q(1, 1)], 1)
        ),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
}

#[test]
fn a_zero_gap_in_one_leaf_never_hides_an_original_source_pole_elsewhere() {
    let vars = [Symbol::new("x")];
    let denominator = Expr::Sym(vars[0].clone()) - Expr::Rational(q(1, 2));
    let source = Expr::Mul(vec![Expr::from_i64(0), pow(denominator, -1)]);
    assert!(
        check(
            &source,
            &vars,
            &[bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 100),
            &witness(vec![q(-1, 1)], 1),
        )
        .is_err()
    );
}

#[test]
fn complete_partition_checks_do_not_disappear_for_constant_objectives() {
    let vars = [Symbol::new("x")];
    let source = Expr::from_i64(0);
    let domain = [bounds(-1, 1)];
    let cases = [
        vec![],
        vec![BoxPartitionStep::Split { axis: 0 }, BoxPartitionStep::Leaf],
        vec![BoxPartitionStep::Leaf, BoxPartitionStep::Leaf],
        vec![
            BoxPartitionStep::Split { axis: 1 },
            BoxPartitionStep::Leaf,
            BoxPartitionStep::Leaf,
        ],
        vec![
            BoxPartitionStep::Split { axis: usize::MAX },
            BoxPartitionStep::Leaf,
            BoxPartitionStep::Leaf,
        ],
        vec![BoxPartitionStep::Leaf; MAX_BOX_OPTIMIZATION_NODES + 1],
    ];
    for partition in cases {
        let mut proposed = witness(vec![q(0, 1)], 0);
        proposed.partition = partition;
        assert!(matches!(
            check(
                &source,
                &vars,
                &domain,
                BoxExtremumKind::Minimum,
                &q(1, 1),
                &proposed
            ),
            Err(BoxOptimizationError::InvalidPartition(_)),
        ));
    }
    let mut proposed = witness(vec![q(0, 1)], 0);
    proposed.partition = vec![BoxPartitionStep::Split { axis: 0 }; MAX_BOX_OPTIMIZATION_DEPTH + 1];
    proposed
        .partition
        .extend(vec![BoxPartitionStep::Leaf; MAX_BOX_OPTIMIZATION_DEPTH + 2]);
    assert!(matches!(
        check(
            &source,
            &vars,
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &proposed
        ),
        Err(BoxOptimizationError::InvalidPartition("partition depth")),
    ));
    assert!(
        check(
            &source,
            &vars,
            &[bounds(0, 0)],
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &witness(vec![q(0, 1)], 1),
        )
        .is_err()
    );
}

#[test]
fn tolerance_comparison_and_extremum_orientation_are_exact_below_binary64() {
    let vars = [Symbol::new("x")];
    let tiny = BigRational::new(1.into(), BigInt::from(10).pow(400));
    let source = Expr::Rational(tiny.clone()) * Expr::Sym(vars[0].clone());
    let tolerance = &tiny / q(2, 1);
    for (kind, bad, good, expected) in [
        (BoxExtremumKind::Minimum, q(1, 1), q(0, 1), q(0, 1)),
        (BoxExtremumKind::Maximum, q(0, 1), q(1, 1), tiny.clone()),
    ] {
        assert_eq!(
            check(
                &source,
                &vars,
                &[bounds(0, 1)],
                kind,
                &tolerance,
                &witness(vec![bad], 0)
            ),
            Err(BoxOptimizationError::AccuracyNotReached),
        );
        let receipt = check(
            &source,
            &vars,
            &[bounds(0, 1)],
            kind,
            &tolerance,
            &witness(vec![good], 0),
        )
        .unwrap();
        assert_eq!(receipt.value(), &RealBall::exact(expected));
    }
    let source = Expr::Sym(vars[0].clone());
    let proposed = witness(vec![q(1, 1)], 0);
    assert!(
        check(
            &source,
            &vars,
            &[bounds(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &proposed
        )
        .is_ok()
    );
    let just_below = q(1, 1) - BigRational::new(1.into(), BigInt::from(2).pow(200));
    assert_eq!(
        check(
            &source,
            &vars,
            &[bounds(0, 1)],
            BoxExtremumKind::Minimum,
            &just_below,
            &proposed
        ),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
}

#[test]
fn candidate_containment_typed_binding_and_precision_are_checked() {
    let vars = [Symbol::new("x")];
    let source = pow(Expr::Sym(vars[0].clone()) - Expr::from_i64(2), 2);
    assert!(
        check(
            &source,
            &vars,
            &[bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &witness(vec![q(2, 1)], 0)
        )
        .is_err()
    );
    assert!(
        check(
            &source,
            &[Symbol::new("y")],
            &[bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &witness(vec![q(0, 1)], 0)
        )
        .is_err()
    );
    assert!(
        check(
            &source,
            &vars,
            &[bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &witness(vec![], 0)
        )
        .is_err()
    );
    for tolerance in [q(0, 1), q(-1, 1)] {
        assert_eq!(
            check(
                &source,
                &vars,
                &[bounds(-1, 1)],
                BoxExtremumKind::Minimum,
                &tolerance,
                &witness(vec![q(0, 1)], 0)
            ),
            Err(BoxOptimizationError::InvalidTolerance)
        );
    }
    for precision in [7, 129] {
        let mut proposed = witness(vec![q(0, 1)], 0);
        proposed.precision_digits = precision;
        assert!(
            check(
                &source,
                &vars,
                &[bounds(-1, 1)],
                BoxExtremumKind::Minimum,
                &q(1, 1),
                &proposed
            )
            .is_err()
        );
    }
}

#[test]
fn point_boxes_and_maximum_dimension_retain_the_original_claim() {
    let vars: Vec<_> = (0..16).map(|i| Symbol::new(format!("x{i}"))).collect();
    let source = Expr::from_i64(7);
    let domain = vec![bounds(0, 0); vars.len()];
    let receipt = check(
        &source,
        &vars,
        &domain,
        BoxExtremumKind::Maximum,
        &q(1, 1000),
        &witness(vec![q(0, 1); vars.len()], 0),
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(7));
    assert_eq!(receipt.point_value(), &RealBall::from_i64(7));
    assert_eq!(receipt.gap(), &q(0, 1));
}

#[test]
fn reusable_workspace_checks_containment_and_reuses_only_the_scratch_reservation() {
    let vars = [Symbol::new("x")];
    let source = Expr::Sym(vars[0].clone());
    let domain = [bounds(-1, 1)];
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    let mut workspace =
        BoxOptimizationWorkspace::new(&source, &vars, &domain, 24, &mut budget).unwrap();
    for _ in 0..8 {
        assert_eq!(
            workspace.enclose_cell(&domain).unwrap().range(),
            &bounds(-1, 1)
        );
    }
    let (left, right) = workspace.bisect(&domain, 0).unwrap();
    assert_eq!(left[0].upper(), right[0].lower());
    assert_eq!(left[0].lower(), domain[0].lower());
    assert_eq!(right[0].upper(), domain[0].upper());
    assert!(workspace.enclose_cell(&[bounds(-2, 1)]).is_err());
    assert!(workspace.evaluate_point(&[q(2, 1)]).is_err());
    assert_eq!(workspace.multiply(&q(2, 3), &q(3, 5)).unwrap(), q(2, 5));
    assert_eq!(workspace.difference(&q(2, 3), &q(1, 3)).unwrap(), q(1, 3));
}

#[derive(Default)]
struct Cancel {
    visits: usize,
    stop: Option<usize>,
}
impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let here = self.visits;
        self.visits += 1;
        if self.stop == Some(here) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_checkpoint_prevents_global_receipt_publication() {
    let vars = [Symbol::new("x")];
    let source = Expr::Sym(vars[0].clone());
    let domain = [bounds(-1, 1)];
    let proposed = witness(vec![q(-1, 1)], 1);
    let mut count = Cancel::default();
    verify_box_optimization(
        &source,
        &vars,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1),
        &proposed,
        &mut count,
    )
    .unwrap();
    for stop in 0..count.visits {
        let mut meter = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            verify_box_optimization(
                &source,
                &vars,
                &domain,
                BoxExtremumKind::Minimum,
                &q(1, 1),
                &proposed,
                &mut meter
            ),
            Err(BoxOptimizationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            ))),
            "safe point {stop}",
        );
    }
}

#[test]
fn consumed_budget_survives_refusal_and_protected_resources_are_untouched() {
    let vars = [Symbol::new("x")];
    let source = Expr::Sym(vars[0].clone());
    let domain = [bounds(-1, 1)];
    let proposed = witness(vec![q(-1, 1)], 1);
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    verify_box_optimization(
        &source,
        &vars,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1),
        &proposed,
        &mut budget,
    )
    .unwrap();
    let consumed = budget.snapshot();
    assert_eq!(budget.verifier_remaining(), 10_000);
    assert!(matches!(
        verify_box_optimization(
            &source,
            &vars,
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1),
            &proposed,
            &mut budget
        ),
        Err(BoxOptimizationError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        ))),
    ));
    assert_eq!(budget.snapshot(), consumed);
    assert_eq!(budget.verifier_remaining(), 10_000);
}

#[test]
fn independently_checked_exponential_point_rounding_is_kept_in_the_gap() {
    let vars = [Symbol::new("x")];
    let source = Expr::Function("exp".into(), vec![Expr::Sym(vars[0].clone())]);
    let domain = [bounds(1, 1)];
    let mut proposed = witness(vec![q(1, 1)], 0);
    proposed.precision_digits = 8;
    let mut meter = Unbounded;
    let coarse = verify_box_optimization(
        &source,
        &vars,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        &proposed,
        &mut meter,
    )
    .unwrap();
    assert!(coarse.gap() > &q(0, 1));
    assert_eq!(
        check(
            &source,
            &vars,
            &domain,
            BoxExtremumKind::Minimum,
            &(coarse.gap() / q(2, 1)),
            &proposed
        ),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
}
