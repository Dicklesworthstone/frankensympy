//! Range-only global certificates for continuous objectives with unbounded
//! endpoint slopes. Exact extrema and inequalities supply independent checks;
//! missing slope information must never become a Taylor or monotonicity proof.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationPrepared,
    BoxOptimizationWitness, BoxOptimizationWorkspace, BoxPartitionStep, Expr, RealBall,
    RootBoxError, RootBoxWitness, Symbol, SymbolIdentity, certify_box_optimization,
    certify_root_box, certify_stationary_point, enclose_objective, enclose_residual_jacobian,
    verify_box_optimization,
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

fn witness(point: Vec<BigRational>) -> BoxOptimizationWitness {
    BoxOptimizationWitness {
        partition: vec![BoxPartitionStep::Leaf],
        point,
        precision_digits: 16,
    }
}

#[test]
fn finite_boundary_extrema_are_checked_without_derivative_claims() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    for (source, domain, kind, point, expected) in [
        (
            function("sqrt", x.clone()),
            interval(0, 4),
            BoxExtremumKind::Minimum,
            0,
            0,
        ),
        (
            function("sqrt", x.clone()),
            interval(0, 4),
            BoxExtremumKind::Maximum,
            4,
            2,
        ),
        (
            power(x.clone(), q(1, 3)),
            interval(0, 8),
            BoxExtremumKind::Minimum,
            0,
            0,
        ),
        (
            power(x.clone(), q(2, 3)),
            interval(0, 27),
            BoxExtremumKind::Maximum,
            27,
            9,
        ),
        (
            function("acos", x.clone()),
            interval(0, 1),
            BoxExtremumKind::Minimum,
            1,
            0,
        ),
        (
            function("acosh", x),
            interval(1, 2),
            BoxExtremumKind::Minimum,
            1,
            0,
        ),
    ] {
        let proposed = witness(vec![q(point, 1)]);
        let tolerance = q(1, 1_000_000);
        let receipt = certify_box_optimization(
            &source,
            &variables,
            std::slice::from_ref(&domain),
            kind,
            &tolerance,
            &proposed,
        )
        .unwrap();
        assert!(receipt.value().contains(&q(expected, 1)), "{source}");
        assert!(receipt.point_value().contains(&q(expected, 1)), "{source}");
        assert!(receipt.gap() <= &tolerance);
        assert_eq!(receipt.objective(), &source);
        assert_eq!(receipt.variables(), variables);
        assert_eq!(receipt.domain(), [domain]);
        assert_eq!(receipt.witness(), &proposed);
        assert_eq!(receipt.leaf_count(), 1);
    }
}

#[test]
fn value_only_cells_keep_the_whole_range_and_explicitly_lack_slopes() {
    let variables = [Symbol::new("x")];
    let source = function("sqrt", Expr::Sym(variables[0].clone()));
    let domain = [interval(0, 4)];
    let cell = BoxOptimizationWorkspace::new(&source, &variables, &domain, 16, &mut Unbounded)
        .unwrap()
        .enclose_cell(&domain)
        .unwrap();
    assert_eq!(cell.range(), &interval(0, 2));
    assert_eq!(cell.slope_bounds(), None);
    assert_eq!(cell.center(), [q(2, 1)]);
    assert!(cell.center_value().lower() > q(1, 1));
    assert!(cell.center_value().upper() < q(2, 1));

    let smooth_domain = [interval(1, 4)];
    let smooth =
        BoxOptimizationWorkspace::new(&source, &variables, &smooth_domain, 16, &mut Unbounded)
            .unwrap()
            .enclose_cell(&smooth_domain)
            .unwrap();
    assert_eq!(smooth.slope_bounds().unwrap().len(), 1);
    assert!(smooth.slope_bounds().unwrap()[0].is_positive());
}

#[test]
fn continuous_compositions_include_interior_cusps_and_fixed_boundary_boxes() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    for (source, domain) in [
        (
            function("sqrt", function("abs", x.clone())),
            interval(-1, 1),
        ),
        (function("abs", function("sqrt", x.clone())), interval(0, 1)),
        (function("sqrt", x), interval(0, 0)),
    ] {
        let receipt = certify_box_optimization(
            &source,
            &variables,
            &[domain],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness(vec![q(0, 1)]),
        )
        .unwrap();
        assert_eq!(receipt.value(), &RealBall::from_i64(0));
        assert_eq!(receipt.gap(), &q(0, 1));
    }
}

#[test]
fn multivariate_range_only_bounds_preserve_typed_axes_and_nonzero_variation() {
    let x = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = Expr::Add(vec![
        function("sqrt", Expr::Sym(x.clone())),
        power(Expr::Sym(y.clone()), q(2, 1)),
    ]);
    let variables = [y.clone(), x.clone()];
    let domain = [interval(-2, 2), interval(0, 4)];
    let cell = BoxOptimizationWorkspace::new(&source, &variables, &domain, 16, &mut Unbounded)
        .unwrap()
        .enclose_cell(&domain)
        .unwrap();
    assert_eq!(cell.slope_bounds(), None);
    assert_eq!(cell.range(), &interval(0, 6));
    let receipt = certify_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        &witness(vec![q(0, 1), q(0, 1)]),
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(0));
    assert_eq!(receipt.variables(), variables);
    assert!(
        certify_box_optimization(
            &source,
            &[x, y],
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness(vec![q(0, 1), q(0, 1)]),
        )
        .is_err()
    );
}

#[test]
fn continuous_retry_retains_original_hidden_domain_obligations() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let root = function("sqrt", x.clone());
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), power(x.clone(), q(-1, 1))]),
        Expr::Mul(vec![Expr::from_i64(0), function("log", x.clone())]),
        power(Expr::from_i64(1), q(1, 3)),
        function("unknown", x.clone()),
    ] {
        // The exact constant is a control: the same range-only fallback must
        // admit a defined held child while rejecting poles and unknown nodes.
        let defined = matches!(&hidden, Expr::Pow(base, _) if **base == Expr::from_i64(1));
        let source = Expr::Add(vec![root.clone(), hidden]);
        let result = certify_box_optimization(
            &source,
            &variables,
            &[interval(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness(vec![q(0, 1)]),
        );
        if defined {
            assert!(result.unwrap().value().contains(&q(1, 1)));
        } else {
            assert!(result.is_err(), "{source}");
        }
    }
    let exponent_pole = Expr::Pow(Arc::new(Expr::from_i64(1)), Arc::new(power(x, q(-1, 1))));
    let source = Expr::Add(vec![root, exponent_pole]);
    assert!(
        certify_box_optimization(
            &source,
            &variables,
            &[interval(0, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness(vec![q(0, 1)]),
        )
        .is_err()
    );
}

#[test]
fn nonreal_unbounded_and_unstable_piecewise_sources_still_refuse() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    for (source, domain, point) in [
        (function("sqrt", x.clone()), interval(-1, 1), 0),
        (function("asin", x.clone()), interval(-1, 2), 0),
        (function("acos", x.clone()), interval(-2, 1), 0),
        (function("acosh", x.clone()), interval(0, 2), 1),
        (power(x.clone(), q(-1, 2)), interval(0, 1), 1),
        (function("log", x.clone()), interval(0, 1), 1),
        (function("atanh", x.clone()), interval(0, 1), 0),
    ] {
        assert!(
            certify_box_optimization(
                &source,
                &variables,
                &[domain],
                BoxExtremumKind::Minimum,
                &q(100, 1),
                &witness(vec![q(point, 1)]),
            )
            .is_err(),
            "{source}"
        );
    }
    let source = Expr::Function(
        "Piecewise".into(),
        vec![
            Expr::Function(
                "Tuple".into(),
                vec![
                    function("sqrt", x.clone()),
                    Expr::Function("StrictLessThan".into(), vec![x, Expr::Rational(q(1, 2))]),
                ],
            ),
            Expr::Function(
                "Tuple".into(),
                vec![Expr::from_i64(10), Expr::symbol("True")],
            ),
        ],
    );
    assert!(
        certify_box_optimization(
            &source,
            &variables,
            &[interval(0, 1)],
            BoxExtremumKind::Minimum,
            &q(100, 1),
            &witness(vec![q(0, 1)]),
        )
        .is_err()
    );
}

#[test]
fn range_only_witnesses_cannot_forge_accuracy_coverage_or_slopes() {
    let variables = [Symbol::new("x")];
    let source = function("sqrt", Expr::Sym(variables[0].clone()));
    let domain = [interval(0, 4)];
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
        check(&witness(vec![q(2, 1)])),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
    assert!(check(&witness(vec![q(-1, 1)])).is_err());
    let mut missing = witness(vec![q(0, 1)]);
    missing.partition = vec![BoxPartitionStep::Split { axis: 0 }, BoxPartitionStep::Leaf];
    assert!(matches!(
        check(&missing),
        Err(BoxOptimizationError::InvalidPartition(_))
    ));
    let mut forged = serde_json::to_value(witness(vec![q(0, 1)])).unwrap();
    forged["slope_bounds"] = serde_json::json!([0]);
    assert!(serde_json::from_value::<BoxOptimizationWitness>(forged).is_err());
}

#[test]
fn endpoint_value_admission_never_promotes_smooth_root_or_stationary_receipts() {
    let variables = [Symbol::new("x")];
    let source = function("sqrt", Expr::Sym(variables[0].clone()));
    let domain = [interval(0, 1)];
    assert!(
        enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 16).is_err()
    );
    assert!(enclose_objective(&source, &variables, &domain, 16).is_err());
    let candidate = RootBoxWitness {
        domain: domain.to_vec(),
        center: vec![q(1, 2)],
        preconditioner: vec![vec![q(1, 1)]],
        precision_digits: 16,
    };
    assert!(certify_root_box(std::slice::from_ref(&source), &variables, &candidate).is_err());
    assert!(certify_stationary_point(&source, &variables, &candidate).is_err());
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
fn continuous_fallback_keeps_cumulative_work_and_cancelled_checks_never_publish() {
    let variables = [Symbol::new("x")];
    let source = function("sqrt", Expr::Sym(variables[0].clone()));
    let domain = [interval(0, 1)];
    let mut prepared =
        BoxOptimizationPrepared::new(&source, &variables, &domain, 16, &mut Unbounded).unwrap();
    let before = prepared.counters();
    prepared
        .with_workspace(&mut Unbounded, |workspace| {
            workspace.evaluate_point(&[q(0, 1)])
        })
        .unwrap();
    let endpoint = prepared.counters();
    assert_eq!(endpoint.evaluations - before.evaluations, 2);
    assert!(endpoint.steps > before.steps);
    assert!(endpoint.allocation_bytes > before.allocation_bytes);
    prepared
        .with_workspace(&mut Unbounded, |workspace| {
            workspace.evaluate_point(&[q(1, 1)])
        })
        .unwrap();
    assert_eq!(prepared.counters().evaluations - endpoint.evaluations, 1);

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
    for at in [0, probe.count / 3, 2 * probe.count / 3, probe.count - 1] {
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
                &mut meter,
            ),
            Err(BoxOptimizationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            ))),
        );
    }
}

#[test]
fn one_less_than_actual_consumption_refuses_without_replenishing_verifier_budget() {
    let variables = [Symbol::new("x")];
    let source = function("sqrt", Expr::Sym(variables[0].clone()));
    let domain = [interval(0, 1)];
    let candidate = witness(vec![q(0, 1)]);
    let allowance = 1 << 40;
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
                &mut meter,
            ),
            Err(BoxOptimizationError::Verification(RootBoxError::Meter(_))),
        ));
        assert_eq!(meter.verifier_remaining(), 17);
    }
}
