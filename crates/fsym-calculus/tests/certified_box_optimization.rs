use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::{BoxOptimizationOptions, propose_box_optimization};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationReceipt,
    BoxOptimizationWitness, BoxPartitionStep, Expr, RealBall, RootBoxError, Symbol,
    verify_box_optimization,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn power(value: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(value), Arc::new(Expr::from_i64(exponent)))
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

fn checked(
    source: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    kind: BoxExtremumKind,
    tolerance: &BigRational,
) -> BoxOptimizationReceipt {
    let mut budget = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0));
    let witness = propose_box_optimization(
        source,
        variables,
        domain,
        kind,
        tolerance,
        options(),
        &mut budget,
    )
    .unwrap();
    let receipt = verify_box_optimization(
        source,
        variables,
        domain,
        kind,
        tolerance,
        &witness,
        &mut budget,
    )
    .unwrap();
    assert_eq!(receipt.objective(), source);
    assert_eq!(receipt.variables(), variables);
    assert_eq!(receipt.domain(), domain);
    assert_eq!(receipt.kind(), kind);
    assert_eq!(receipt.absolute_tolerance(), tolerance);
    assert!(receipt.gap() >= &q(0, 1));
    assert!(receipt.gap() <= tolerance);
    assert_eq!(receipt.value().width(), *receipt.gap());
    assert!(
        domain
            .iter()
            .zip(receipt.point())
            .all(|(bound, value)| bound.contains(value))
    );
    let leaves = witness
        .partition
        .iter()
        .filter(|step| **step == BoxPartitionStep::Leaf)
        .count();
    assert_eq!(receipt.leaf_count(), leaves);
    assert_eq!(witness.partition.len(), 2 * leaves - 1);
    match kind {
        BoxExtremumKind::Minimum => {
            assert_eq!(receipt.value().upper(), receipt.point_value().upper());
        }
        BoxExtremumKind::Maximum => {
            assert_eq!(receipt.value().lower(), receipt.point_value().lower());
        }
    }
    receipt
}

#[test]
fn mixed_sign_affine_minimum_and_maximum_find_exact_boundary_points() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let source = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Mul(vec![Expr::from_i64(-2), Expr::Sym(y.clone())]),
    ]);
    let domain = [interval(-1, 2), interval(-3, 4)];
    for (kind, expected_point, value) in [
        (BoxExtremumKind::Minimum, vec![q(-1, 1), q(4, 1)], -9),
        (BoxExtremumKind::Maximum, vec![q(2, 1), q(-3, 1)], 8),
    ] {
        let receipt = checked(&source, &[x.clone(), y.clone()], &domain, kind, &q(1, 1000));
        assert_eq!(receipt.point(), expected_point);
        assert_eq!(receipt.value(), &RealBall::exact(q(value, 1)));
        assert_eq!(receipt.leaf_count(), 1);
    }
}

#[test]
fn nonconvex_irrational_minima_are_bounded_without_claiming_an_exact_optimizer() {
    let x = Symbol::new("x");
    let source = double_well(&x);
    let result = checked(
        &source,
        &[x],
        &[interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
    );
    // The two exact minimizers are +/-sqrt(2). Every rational candidate has
    // strictly positive value, but the independently checked global gap is small.
    assert!(result.value().contains(&q(0, 1)));
    assert!(result.gap() > &q(0, 1));
    assert!(result.point_value().lower() > q(0, 1));
    assert!(result.leaf_count() > 1);
    let displacement = &result.point()[0] * &result.point()[0] - q(2, 1);
    assert_eq!(
        result.point_value(),
        &RealBall::exact(&displacement * &displacement)
    );
}

#[test]
fn nonconvex_two_dimensional_search_keeps_all_wells_and_finds_mixed_corners() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let difference = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(y.clone())]),
    ]);
    let source = Expr::Mul(vec![Expr::from_i64(-1), power(difference, 2)]);
    let domain = [interval(-1, 1), interval(-1, 1)];
    let minimum = checked(
        &source,
        &[x.clone(), y.clone()],
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
    );
    assert_eq!(minimum.value(), &RealBall::exact(q(-4, 1)));
    assert_eq!(&minimum.point()[0] + &minimum.point()[1], q(0, 1));
    assert_eq!(&minimum.point()[0] * &minimum.point()[0], q(1, 1));
    assert!(minimum.leaf_count() > 1);
    let maximum = checked(
        &source,
        &[x, y],
        &domain,
        BoxExtremumKind::Maximum,
        &q(1, 1000),
    );
    assert_eq!(maximum.value(), &RealBall::exact(q(0, 1)));
    assert_eq!(maximum.leaf_count(), 1);
}

#[test]
fn two_dimensional_separable_quartic_finds_a_global_well() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let well = |variable: &Symbol| {
        power(
            Expr::Add(vec![
                power(Expr::Sym(variable.clone()), 2),
                Expr::from_i64(-1),
            ]),
            2,
        )
    };
    let source = Expr::Add(vec![well(&x), well(&y)]);
    let result = checked(
        &source,
        &[x, y],
        &[interval(-2, 2), interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
    );
    assert_eq!(result.value(), &RealBall::exact(q(0, 1)));
    assert!(result.leaf_count() > 1);
    assert!(result.point().iter().all(|value| value * value == q(1, 1)));
}

#[test]
fn an_irrational_transcendental_maximizer_has_a_certified_rational_candidate() {
    let x = Symbol::new("x");
    let source = Expr::Function("sin".into(), vec![Expr::Sym(x.clone())]);
    let result = checked(
        &source,
        &[x],
        &[interval(0, 4)],
        BoxExtremumKind::Maximum,
        &q(1, 1000),
    );
    // pi/2 lies in [0,4], so the independently known maximum is exactly one.
    assert!(result.value().contains(&q(1, 1)));
    assert!(result.gap() > &q(0, 1));
    assert!(result.point_value().upper() < q(1, 1));
    assert!(result.leaf_count() > 1);
}

#[test]
fn monotone_elementary_objectives_find_boundary_values_in_one_leaf() {
    let x = Symbol::new("x");
    for (name, domain, expected) in [
        ("exp", interval(0, 1), q(1, 1)),
        ("log", interval(1, 4), q(0, 1)),
    ] {
        let source = Expr::Function(name.into(), vec![Expr::Sym(x.clone())]);
        let result = checked(
            &source,
            std::slice::from_ref(&x),
            &[domain],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
        );
        assert!(result.value().contains(&expected));
        assert_eq!(result.leaf_count(), 1);
    }
}

#[test]
fn fixed_coordinates_and_flat_objectives_do_not_require_strict_curvature() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let source = Expr::Add(vec![power(Expr::Sym(x.clone()), 2), Expr::Sym(y.clone())]);
    let result = checked(
        &source,
        &[x, y],
        &[RealBall::exact(q(2, 1)), interval(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
    );
    assert_eq!(result.point(), [q(2, 1), q(-1, 1)]);
    assert_eq!(result.value(), &RealBall::exact(q(3, 1)));
    let variables: Vec<_> = (0..16).map(|i| Symbol::new(format!("x{i}"))).collect();
    let domain = vec![interval(-1, 1); 16];
    let flat = checked(
        &Expr::from_i64(7),
        &variables,
        &domain,
        BoxExtremumKind::Maximum,
        &q(1, 1000),
    );
    assert_eq!(flat.value(), &RealBall::exact(q(7, 1)));
    assert_eq!(flat.point(), vec![q(0, 1); 16]);
    assert_eq!(flat.leaf_count(), 1);
}

#[test]
fn bounded_subdivision_resolves_original_source_interval_dependency() {
    let x = Symbol::new("x");
    let sx = Expr::Sym(x.clone());
    // Held x*x on [-1,1] overestimates [-1,1], so the initial reciprocal
    // evaluation cannot prove its denominator positive. The closed half-boxes
    // can, and the checker must replay those exact unsimplified obligations.
    let denominator = Expr::Add(vec![Expr::from_i64(1), Expr::Mul(vec![sx.clone(), sx])]);
    let source = power(denominator, -1);
    let result = checked(
        &source,
        &[x],
        &[interval(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
    );
    assert_eq!(result.value(), &RealBall::exact(q(1, 2)));
    assert!(result.leaf_count() > 1);
}

#[test]
fn unresolved_domain_leaf_blocks_success_even_when_valid_regions_attain_zero() {
    let x = Symbol::new("x");
    let inverse = power(Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(1)]), -1);
    for source in [
        Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
        power(inverse, 0),
    ] {
        assert!(matches!(
            propose_box_optimization(
                &source,
                std::slice::from_ref(&x),
                &[interval(-1, 1)],
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                BoxOptimizationOptions {
                    max_leaves: 8,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(BoxOptimizationError::Verification(
                RootBoxError::Domain(_) | RootBoxError::Ball(_)
            ))
        ));
    }
}

#[test]
fn independent_replay_rejects_missing_regions_outside_points_and_wrong_objectives() {
    let x = Symbol::new("x");
    let source = double_well(&x);
    let variables = [x];
    let domain = [interval(-2, 2)];
    let tolerance = q(1, 1000);
    let witness = propose_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        options(),
        &mut Unbounded,
    )
    .unwrap();
    let verify = |source: &Expr, witness: &BoxOptimizationWitness| {
        verify_box_optimization(
            source,
            &variables,
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            witness,
            &mut Unbounded,
        )
    };
    assert!(verify(&source, &witness).is_ok());
    let mut truncated = witness.clone();
    truncated.partition.pop();
    assert!(matches!(
        verify(&source, &truncated),
        Err(BoxOptimizationError::InvalidPartition(_))
    ));
    let mut outside = witness.clone();
    outside.point[0] = q(3, 1);
    assert!(matches!(
        verify(&source, &outside),
        Err(BoxOptimizationError::Verification(
            RootBoxError::CenterOutsideBox { .. }
        ))
    ));
    let mut bad_point = witness.clone();
    bad_point.point[0] = q(0, 1);
    assert_eq!(
        verify(&source, &bad_point),
        Err(BoxOptimizationError::AccuracyNotReached)
    );
    let opposite = Expr::Mul(vec![Expr::from_i64(-1), source.clone()]);
    assert_eq!(
        verify(&opposite, &witness),
        Err(BoxOptimizationError::AccuracyNotReached)
    );
}

#[test]
fn fixed_ordering_produces_the_same_complete_partition_and_incumbent() {
    let x = Symbol::new("x");
    let source = double_well(&x);
    let propose = || {
        propose_box_optimization(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            options(),
            &mut Unbounded,
        )
        .unwrap()
    };
    assert_eq!(propose(), propose());
}

#[test]
fn leaf_precision_tolerance_and_source_limits_refuse_without_partial_witnesses() {
    let x = Symbol::new("x");
    let source = double_well(&x);
    let propose = |tolerance: &BigRational, options: BoxOptimizationOptions| {
        propose_box_optimization(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            BoxExtremumKind::Minimum,
            tolerance,
            options,
            &mut Unbounded,
        )
    };
    assert_eq!(
        propose(
            &q(1, 1000),
            BoxOptimizationOptions {
                max_leaves: 1,
                ..options()
            }
        ),
        Err(BoxOptimizationError::AccuracyNotReached)
    );
    for max_leaves in [0, 1025, usize::MAX] {
        assert!(matches!(
            propose(
                &q(1, 1000),
                BoxOptimizationOptions {
                    max_leaves,
                    ..options()
                }
            ),
            Err(BoxOptimizationError::Verification(
                RootBoxError::ResourceLimit(_)
            ))
        ));
    }
    for precision_digits in [7, 129, u32::MAX] {
        assert!(matches!(
            propose(
                &q(1, 1000),
                BoxOptimizationOptions {
                    precision_digits,
                    ..options()
                }
            ),
            Err(BoxOptimizationError::Verification(
                RootBoxError::InvalidPrecision
            ))
        ));
    }
    for tolerance in [q(0, 1), q(-1, 1)] {
        assert_eq!(
            propose(&tolerance, options()),
            Err(BoxOptimizationError::InvalidTolerance)
        );
    }
    for unsupported in [
        Expr::Function("gamma".into(), vec![Expr::Sym(x.clone())]),
        Expr::Sym(Symbol::new("unbound")),
    ] {
        assert!(matches!(
            propose_box_optimization(
                &unsupported,
                std::slice::from_ref(&x),
                &[interval(1, 2)],
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                options(),
                &mut Unbounded,
            ),
            Err(BoxOptimizationError::Verification(
                RootBoxError::Unsupported(_)
            ))
        ));
    }
}

#[test]
fn generator_and_verifier_spend_one_cumulative_caller_budget() {
    let x = Symbol::new("x");
    let variables = [x.clone()];
    let source = Expr::Sym(x);
    let domain = [interval(-1, 1)];
    // One workspace fits; the second independent phase cannot obtain its
    // fixed scratch reservation. The generator's previous costs remain used.
    let limit = 100 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    let witness = propose_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        options(),
        &mut budget,
    )
    .unwrap();
    let remaining = budget.remaining(Dimension::MemoryBytes);
    assert!(remaining < limit - 64 * 1024 * 1024);
    assert!(matches!(
        verify_box_optimization(
            &source,
            &variables,
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness,
            &mut budget,
        ),
        Err(BoxOptimizationError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.remaining(Dimension::MemoryBytes), remaining);
}

#[test]
fn failed_accuracy_does_not_refund_work_before_a_later_search() {
    let x = Symbol::new("x");
    let source = double_well(&x);
    let mut budget = Budget::new(BudgetLimits::uniform(100 * 1024 * 1024, 0));
    assert_eq!(
        propose_box_optimization(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            BoxOptimizationOptions {
                max_leaves: 1,
                ..options()
            },
            &mut budget,
        ),
        Err(BoxOptimizationError::AccuracyNotReached)
    );
    assert!(budget.remaining(Dimension::ComputeSteps) < 100 * 1024 * 1024);
    assert!(matches!(
        propose_box_optimization(
            &Expr::Sym(x.clone()),
            &[x],
            &[interval(-2, 2)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            options(),
            &mut budget,
        ),
        Err(BoxOptimizationError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
}

#[derive(Default)]
struct CheckpointMeter {
    checks: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CheckpointMeter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checks += 1;
        if self.cancel_at == Some(self.checks) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_entry_during_search_and_before_publication_has_no_witness() {
    let x = Symbol::new("x");
    let source = double_well(&x);
    let propose = |meter: &mut CheckpointMeter| {
        propose_box_optimization(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            options(),
            meter,
        )
    };
    let mut completed = CheckpointMeter::default();
    assert!(propose(&mut completed).is_ok());
    for cancel_at in [1, completed.checks / 2, completed.checks] {
        let mut meter = CheckpointMeter {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            propose(&mut meter),
            Err(BoxOptimizationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        );
        assert_eq!(meter.checks, cancel_at);
    }
}
