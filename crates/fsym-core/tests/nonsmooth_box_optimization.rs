//! Original-source value-gap certificates at continuous absolute-value corners.
//! Exact inequalities and reference values are independent of the search.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationPrepared,
    BoxOptimizationWitness, BoxOptimizationWorkspace, BoxPartitionStep, Expr, RealBall,
    RootBoxError, RootBoxWitness, Symbol, certify_box_optimization, certify_root_box,
    certify_stationary_point, enclose_objective, enclose_residual_jacobian,
    verify_box_optimization,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn bounds(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn abs(argument: Expr) -> Expr {
    function("abs", argument)
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn witness(point: Vec<BigRational>) -> BoxOptimizationWitness {
    BoxOptimizationWitness {
        partition: vec![BoxPartitionStep::Leaf],
        point,
        precision_digits: 24,
    }
}

#[test]
fn absolute_value_has_exact_minimum_and_negative_absolute_value_exact_maximum() {
    let variables = [Symbol::new("x")];
    let absolute = abs(Expr::Sym(variables[0].clone()));
    let domain = [bounds(-1, 1)];
    for (source, kind) in [
        (absolute.clone(), BoxExtremumKind::Minimum),
        (
            Expr::Mul(vec![Expr::from_i64(-1), absolute]),
            BoxExtremumKind::Maximum,
        ),
    ] {
        let proposed = witness(vec![q(0, 1)]);
        let receipt = certify_box_optimization(
            &source,
            &variables,
            &domain,
            kind,
            &q(1, 1_000_000),
            &proposed,
        )
        .unwrap();
        assert_eq!(receipt.objective(), &source);
        assert_eq!(receipt.variables(), variables);
        assert_eq!(receipt.domain(), domain);
        assert_eq!(receipt.witness(), &proposed);
        assert_eq!(receipt.value(), &RealBall::from_i64(0));
        assert_eq!(receipt.point_value(), &RealBall::from_i64(0));
        assert_eq!(receipt.gap(), &q(0, 1));
        assert_eq!(receipt.leaf_count(), 1);
    }
}

#[test]
fn coupled_corners_nested_values_and_held_constants_keep_exact_bounds() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(variables[0].clone());
    let y = Expr::Sym(variables[1].clone());
    let difference = Expr::Add(vec![
        x.clone(),
        Expr::Mul(vec![Expr::from_i64(-1), y.clone()]),
    ]);
    let coupled = Expr::Add(vec![
        abs(difference),
        Expr::Mul(vec![Expr::from_i64(2), abs(y)]),
        Expr::from_i64(3),
    ]);
    let zero = Expr::Mul(vec![Expr::from_i64(0), abs(x.clone())]);
    for (source, expected) in [
        (coupled, 3),
        (abs(abs(x.clone())), 0),
        (zero, 0),
        (power(Expr::Add(vec![Expr::from_i64(1), abs(x)]), 0), 1),
        (function("Abs", Expr::from_i64(0)), 0),
    ] {
        let receipt = certify_box_optimization(
            &source,
            &variables,
            &[bounds(-1, 1), bounds(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness(vec![q(0, 1); 2]),
        )
        .unwrap();
        assert_eq!(receipt.value(), &RealBall::from_i64(expected));
        assert_eq!(receipt.gap(), &q(0, 1));
    }
}

#[test]
fn smooth_outer_functions_compose_with_absolute_value_corners() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let source = function("exp", abs(x.clone()));
    let receipt = certify_box_optimization(
        &source,
        &variables,
        &[bounds(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1_000_000),
        &witness(vec![q(0, 1)]),
    )
    .unwrap();
    assert!(receipt.value().contains(&q(1, 1)));
    assert!(receipt.gap() <= &q(1, 1_000_000));

    let source = function("log", Expr::Add(vec![Expr::from_i64(1), abs(x)]));
    let receipt = certify_box_optimization(
        &source,
        &variables,
        &[bounds(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1_000_000),
        &witness(vec![q(0, 1)]),
    )
    .unwrap();
    assert!(receipt.value().contains(&q(0, 1)));
    assert!(receipt.gap() <= &q(1, 1_000_000));
}

#[test]
fn one_sided_corners_keep_monotone_faces_and_crossing_slopes_stay_uncertain() {
    let variables = [Symbol::new("x")];
    let source = abs(Expr::Sym(variables[0].clone()));
    for (domain, slopes, range) in [
        (bounds(-2, 0), RealBall::from_i64(-1), bounds(0, 2)),
        (bounds(0, 2), RealBall::from_i64(1), bounds(0, 2)),
        (bounds(-2, 2), bounds(-1, 1), bounds(0, 2)),
    ] {
        let cell = BoxOptimizationWorkspace::new(
            &source,
            &variables,
            std::slice::from_ref(&domain),
            24,
            &mut Unbounded,
        )
        .unwrap()
        .enclose_cell(std::slice::from_ref(&domain))
        .unwrap();
        assert_eq!(cell.slope_bounds(), Some([slopes].as_slice()));
        assert_eq!(cell.range(), &range);
    }
    // A positive whole-cell coordinate slope remains meaningful even though
    // an inner Abs is not differentiable. The minimum is at the left face.
    let source = Expr::Add(vec![
        Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(variables[0].clone())]),
        source,
    ]);
    let receipt = certify_box_optimization(
        &source,
        &variables,
        &[bounds(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        &witness(vec![q(-1, 1)]),
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(-1));
}

#[test]
fn product_dynamic_power_and_binary_composition_enclose_independent_values() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(variables[0].clone());
    let y = Expr::Sym(variables[1].clone());
    let sources = [
        Expr::Mul(vec![abs(x.clone()), abs(y.clone())]),
        Expr::Pow(
            Arc::new(Expr::Add(vec![Expr::from_i64(1), abs(x.clone())])),
            Arc::new(Expr::Add(vec![Expr::from_i64(1), abs(y.clone())])),
        ),
        Expr::Function(
            "atan2".into(),
            vec![abs(x), Expr::Add(vec![Expr::from_i64(1), abs(y)])],
        ),
    ];
    for (source, expected) in sources.iter().zip([q(1, 4), q(27, 8), q(0, 1)]) {
        let domain = [bounds(-2, 2), bounds(-2, 2)];
        let cell = BoxOptimizationWorkspace::new(source, &variables, &domain, 24, &mut Unbounded)
            .unwrap()
            .enclose_cell(&domain)
            .unwrap();
        assert!(cell.range().contains(&expected), "{source}");
        assert_eq!(cell.slope_bounds().unwrap().len(), 2);
    }
}

#[test]
fn strict_c1_c2_root_and_stationary_checks_still_refuse_absolute_value_corners() {
    let variables = [Symbol::new("x")];
    let source = abs(Expr::Sym(variables[0].clone()));
    for domain in [[bounds(-1, 1)], [bounds(0, 1)], [bounds(0, 0)]] {
        assert!(matches!(
            enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24),
            Err(RootBoxError::Domain(_)),
        ));
        assert!(matches!(
            enclose_objective(&source, &variables, &domain, 24),
            Err(RootBoxError::Domain(_))
        ));
    }
    let root = RootBoxWitness {
        domain: vec![bounds(-1, 1)],
        center: vec![q(0, 1)],
        preconditioner: vec![vec![q(1, 1)]],
        precision_digits: 24,
    };
    assert!(matches!(
        certify_root_box(std::slice::from_ref(&source), &variables, &root),
        Err(RootBoxError::Domain(_))
    ));
    assert!(matches!(
        certify_stationary_point(&source, &variables, &root),
        Err(RootBoxError::Domain(_))
    ));
}

#[test]
fn fallback_never_cancels_away_original_poles_undefined_children_or_unknown_functions() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let domain = [bounds(-1, 1)];
    let undefined = [
        power(x.clone(), -1),
        function("log", x.clone()),
        function("log", abs(x.clone())),
        // sqrt(abs(x)) now has a continuous value enclosure. The negative
        // radicand remains non-real and must not be erased by a held parent.
        function("sqrt", Expr::Mul(vec![Expr::from_i64(-1), abs(x.clone())])),
        power(abs(x.clone()), 0),
        Expr::Function("atan2".into(), vec![abs(x.clone()), Expr::from_i64(-1)]),
        function("unknown", x.clone()),
    ];
    for child in undefined {
        for hidden in [
            abs(child.clone()),
            Expr::Mul(vec![Expr::from_i64(0), child.clone()]),
            power(child, 0),
        ] {
            let source = Expr::Add(vec![abs(x.clone()), hidden]);
            let mut meter = Unbounded;
            let mut workspace =
                BoxOptimizationWorkspace::new(&source, &variables, &domain, 24, &mut meter)
                    .unwrap();
            assert!(workspace.enclose_cell(&domain).is_err(), "{source}");
        }
    }
    for source in [
        Expr::Function("Abs".into(), vec![]),
        Expr::Function("abs".into(), vec![x.clone(), x]),
    ] {
        assert!(
            certify_box_optimization(
                &source,
                &variables,
                &domain,
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                &witness(vec![q(0, 1)])
            )
            .is_err()
        );
    }
}

#[test]
fn piecewise_only_relaxes_the_selected_value_not_guard_stability() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let source = |condition| {
        Expr::Function(
            "Piecewise".into(),
            vec![
                Expr::Function("Tuple".into(), vec![abs(x.clone()), condition]),
                Expr::Function(
                    "Tuple".into(),
                    vec![function("log", Expr::from_i64(0)), Expr::symbol("True")],
                ),
            ],
        )
    };
    let stable = source(Expr::Function(
        "StrictLessThan".into(),
        vec![x.clone(), Expr::from_i64(2)],
    ));
    let receipt = certify_box_optimization(
        &stable,
        &variables,
        &[bounds(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        &witness(vec![q(0, 1)]),
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(0));
    for condition in [
        Expr::Function("StrictLessThan".into(), vec![x.clone(), Expr::from_i64(0)]),
        Expr::Function("Equality".into(), vec![abs(x.clone()), abs(x.clone())]),
    ] {
        assert!(
            certify_box_optimization(
                &source(condition),
                &variables,
                &[bounds(-1, 1)],
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                &witness(vec![q(0, 1)])
            )
            .is_err()
        );
    }
}

#[test]
fn candidate_point_coverage_tolerance_and_typed_source_still_control_acceptance() {
    let variables = [Symbol::new("x")];
    let source = abs(Expr::Sym(variables[0].clone()));
    let domain = [bounds(-1, 1)];
    let check = |candidate: &BoxOptimizationWitness| {
        certify_box_optimization(
            &source,
            &variables,
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            candidate,
        )
    };
    assert_eq!(
        check(&witness(vec![q(1, 1)])),
        Err(BoxOptimizationError::AccuracyNotReached)
    );
    assert!(check(&witness(vec![q(2, 1)])).is_err());
    let mut candidate = witness(vec![q(0, 1)]);
    candidate.partition = vec![BoxPartitionStep::Split { axis: 0 }, BoxPartitionStep::Leaf];
    assert!(check(&candidate).is_err());
    candidate.partition = vec![BoxPartitionStep::Leaf, BoxPartitionStep::Leaf];
    assert!(check(&candidate).is_err());
    assert!(
        certify_box_optimization(
            &source,
            &[Symbol::new("y")],
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness(vec![q(0, 1)])
        )
        .is_err()
    );
}

#[derive(Default)]
struct Cancel {
    count: usize,
    at: Option<usize>,
}

impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let here = self.count;
        self.count += 1;
        if self.at == Some(here) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn corner_retry_retains_work_and_every_checkpoint_prevents_publication_when_cancelled() {
    let variables = [Symbol::new("x")];
    let source = abs(Expr::Sym(variables[0].clone()));
    let domain = [bounds(-1, 1)];
    let mut prepared =
        BoxOptimizationPrepared::new(&source, &variables, &domain, 24, &mut Unbounded).unwrap();
    let before = prepared.counters();
    prepared
        .with_workspace(&mut Unbounded, |workspace| {
            workspace.evaluate_point(&[q(0, 1)])
        })
        .unwrap();
    let corner = prepared.counters();
    assert_eq!(corner.evaluations - before.evaluations, 2);
    assert!(corner.steps > before.steps && corner.allocation_bytes > before.allocation_bytes);
    prepared
        .with_workspace(&mut Unbounded, |workspace| {
            workspace.evaluate_point(&[q(1, 1)])
        })
        .unwrap();
    assert_eq!(prepared.counters().evaluations - corner.evaluations, 1);

    let candidate = witness(vec![q(0, 1)]);
    let mut probe = Cancel::default();
    verify_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        &candidate,
        &mut probe,
    )
    .unwrap();
    for at in 0..probe.count {
        let mut meter = Cancel {
            count: 0,
            at: Some(at),
        };
        assert_eq!(
            verify_box_optimization(
                &source,
                &variables,
                &domain,
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                &candidate,
                &mut meter
            ),
            Err(BoxOptimizationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            ))),
            "checkpoint {at}"
        );
    }
}

#[test]
fn complete_verifier_work_and_late_publication_share_the_same_budget() {
    let variables = [Symbol::new("x")];
    let source = abs(Expr::Sym(variables[0].clone()));
    let domain = [bounds(-1, 1)];
    let candidate = witness(vec![q(0, 1)]);
    let allowance = 100_000_000;
    let mut probe = Budget::new(BudgetLimits::uniform(allowance, 17));
    verify_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        &candidate,
        &mut probe,
    )
    .unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let consumed = allowance - probe.remaining(dimension);
        assert!(consumed > 0);
        let mut limits = BudgetLimits::uniform(allowance, 17);
        limits.dimensions[dimension.index()] = consumed - 1;
        let mut meter = Budget::new(limits);
        assert!(matches!(
            verify_box_optimization(
                &source,
                &variables,
                &domain,
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                &candidate,
                &mut meter
            ),
            Err(BoxOptimizationError::Verification(RootBoxError::Meter(_)))
        ));
        assert!(meter.remaining(Dimension::ComputeSteps) < allowance);
        assert_eq!(meter.verifier_remaining(), 17);
    }
}
