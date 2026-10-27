use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_QUADRATURE_INTERVALS, QuadratureError, QuadratureWitness,
    RootBoxError, Symbol, SymbolIdentity, certify_integral_enclosure, verify_integral_enclosure,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(value: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(value), Arc::new(Expr::from_i64(exponent)))
}
fn witness(points: Vec<BigRational>) -> QuadratureWitness {
    QuadratureWitness {
        partition: points,
        precision_digits: 24,
    }
}

#[test]
fn centered_derivative_bound_integrates_affine_sources_exactly() {
    let x = Symbol::new("integral_x");
    let source = Expr::from_i64(3) * Expr::Sym(x.clone()) + Expr::from_i64(2);
    let tolerance = BigRational::new(1.into(), BigInt::from(10).pow(100));
    let proposed = witness(vec![q(-2, 1), q(5, 1)]);
    let receipt =
        certify_integral_enclosure(&source, &x, &q(-2, 1), &q(5, 1), &tolerance, &proposed)
            .unwrap();
    assert_eq!(receipt.enclosure().lower(), q(91, 2));
    assert_eq!(receipt.enclosure().upper(), q(91, 2));
    assert_eq!(receipt.integrand(), &source);
    assert_eq!(receipt.variable(), &x);
    assert_eq!(receipt.witness(), &proposed);
    assert_eq!(receipt.absolute_tolerance(), &tolerance);
}

#[test]
fn complete_quadratic_partition_bounds_integral_and_rejects_false_accuracy() {
    let x = Symbol::new("x");
    let source = power(Expr::Sym(x.clone()), 2);
    let proposed = witness((0..=8).map(|i| q(i, 8)).collect());
    let receipt =
        certify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 100), &proposed).unwrap();
    assert!(receipt.enclosure().contains(&q(1, 3)));
    assert!(receipt.enclosure().width() <= q(1, 100));
    assert!(matches!(
        certify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 10_000), &proposed,),
        Err(QuadratureError::AccuracyNotReached)
    ));
    let coarse = witness(vec![q(0, 1), q(1, 1)]);
    assert!(matches!(
        certify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 100), &coarse,),
        Err(QuadratureError::AccuracyNotReached)
    ));
}

#[test]
fn negative_non_grid_bounds_round_outward_without_losing_integral_values() {
    let x = Symbol::new("x");
    let source = -power(Expr::Sym(x.clone()), 2);
    let receipt = certify_integral_enclosure(
        &source,
        &x,
        &q(0, 1),
        &q(1, 3),
        &q(1, 10),
        &witness(vec![q(0, 1), q(1, 3)]),
    )
    .unwrap();
    // Exact centered bounds are [-1/54,0]. The lower endpoint is not on
    // any finite decimal grid, so flooring a negative quotient matters.
    assert!(receipt.enclosure().lower() <= q(-1, 54));
    assert!(receipt.enclosure().upper() >= q(0, 1));
    assert!(receipt.enclosure().contains(&q(-1, 81)));
}

#[test]
fn reversed_and_point_intervals_preserve_the_exact_original_claim() {
    let x = Symbol::new("x");
    let source = Expr::Sym(x.clone());
    let receipt = certify_integral_enclosure(
        &source,
        &x,
        &q(3, 1),
        &q(-1, 1),
        &q(1, 10),
        &witness(vec![q(-1, 1), q(1, 1), q(3, 1)]),
    )
    .unwrap();
    assert_eq!(receipt.enclosure().lower(), q(-4, 1));
    assert_eq!(receipt.lower(), &q(3, 1));
    assert_eq!(receipt.upper(), &q(-1, 1));
    let point = certify_integral_enclosure(
        &source,
        &x,
        &q(2, 1),
        &q(2, 1),
        &q(1, 10),
        &witness(vec![q(2, 1)]),
    )
    .unwrap();
    assert_eq!(point.enclosure().lower(), q(0, 1));
    assert_eq!(point.enclosure().upper(), q(0, 1));
    assert_eq!(point.interval_count(), 0);
    assert!(
        certify_integral_enclosure(
            &power(source, -1),
            &x,
            &q(0, 1),
            &q(0, 1),
            &q(1, 10),
            &witness(vec![q(0, 1)]),
        )
        .is_err()
    );
}

#[test]
fn partition_source_and_typed_variable_mutations_cannot_relabel_acceptance() {
    let x = Symbol::new("x");
    let source = Expr::Sym(x.clone());
    for points in [
        vec![],
        vec![q(0, 1)],
        vec![q(0, 1), q(1, 2)],
        vec![q(1, 1), q(0, 1)],
        vec![q(0, 1), q(1, 2), q(1, 2), q(1, 1)],
        vec![q(0, 1), q(3, 4), q(1, 2), q(1, 1)],
        vec![q(0, 1); MAX_QUADRATURE_INTERVALS + 2],
    ] {
        assert!(
            certify_integral_enclosure(
                &source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                &witness(points),
            )
            .is_err()
        );
    }
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let proposed = witness(vec![q(0, 1), q(1, 1)]);
    assert!(
        certify_integral_enclosure(&source, &other, &q(0, 1), &q(1, 1), &q(1, 100), &proposed,)
            .is_err()
    );
    let original =
        certify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 100), &proposed).unwrap();
    let changed_source = source + Expr::Rational(q(1, 1_000_000));
    let changed = certify_integral_enclosure(
        &changed_source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 100),
        &proposed,
    )
    .unwrap();
    assert_ne!(original.integrand(), changed.integrand());
    assert_ne!(original.enclosure(), changed.enclosure());
}

#[test]
fn original_domains_are_checked_before_cancellation_or_zero_multiplication() {
    let x = Symbol::new("x");
    let sx = Expr::Sym(x.clone());
    let reciprocal = power(sx.clone(), -1);
    let sources = [
        reciprocal.clone(),
        Expr::Mul(vec![Expr::from_i64(0), reciprocal.clone()]),
        Expr::Mul(vec![sx, reciprocal.clone()]),
        power(reciprocal, 0),
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
        Expr::Function("sqrt".into(), vec![Expr::Sym(x.clone())]),
    ];
    for source in sources {
        assert!(
            certify_integral_enclosure(
                &source,
                &x,
                &q(-1, 1),
                &q(1, 1),
                &q(1, 1),
                &witness(vec![q(-1, 1), q(0, 1), q(1, 1)]),
            )
            .is_err()
        );
    }
}

#[test]
fn bounded_transport_and_request_validation_refuse_malformed_inputs() {
    let proposed = witness(vec![q(0, 1), q(1, 2), q(1, 1)]);
    let wire = serde_json::to_value(&proposed).unwrap();
    assert_eq!(
        serde_json::from_value::<QuadratureWitness>(wire.clone()).unwrap(),
        proposed
    );
    let mut bad = wire.clone();
    bad["partition"] = serde_json::json!(vec![q(0, 1); MAX_QUADRATURE_INTERVALS + 2]);
    assert!(serde_json::from_value::<QuadratureWitness>(bad).is_err());
    let mut bad = wire.clone();
    bad["accepted"] = serde_json::json!(true);
    assert!(serde_json::from_value::<QuadratureWitness>(bad).is_err());
    for scalar in [
        serde_json::json!([[1, [2]], [1, [2]]]),
        serde_json::json!([[1, [1]], [-1, [1]]]),
        serde_json::json!([[0, [1]], [1, [1]]]),
    ] {
        let mut bad = wire.clone();
        bad["partition"][0] = scalar;
        assert!(serde_json::from_value::<QuadratureWitness>(bad).is_err());
    }
    let huge = BigRational::from_integer(BigInt::from(1) << 4096u32);
    let x = Symbol::new("x");
    assert!(
        certify_integral_enclosure(
            &Expr::Sym(x.clone()),
            &x,
            &q(0, 1),
            &huge,
            &q(1, 10),
            &proposed,
        )
        .is_err()
    );
    for tolerance in [q(0, 1), q(-1, 1)] {
        assert!(matches!(
            certify_integral_enclosure(
                &Expr::Sym(x.clone()),
                &x,
                &q(0, 1),
                &q(1, 1),
                &tolerance,
                &proposed,
            ),
            Err(QuadratureError::InvalidTolerance)
        ));
    }
}

#[test]
fn actual_budget_is_charged_atomically_and_admits_successful_enclosures() {
    let x = Symbol::new("x");
    let source = power(Expr::Sym(x.clone()), 2);
    let proposed = witness((0..=8).map(|i| q(i, 8)).collect());
    let mut budget = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let receipt = verify_integral_enclosure(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 100),
        &proposed,
        &mut budget,
    )
    .unwrap();
    assert!(receipt.enclosure().contains(&q(1, 3)));
    assert!(budget.remaining(Dimension::ComputeSteps) < 512 * 1024 * 1024);
    let mut insufficient = Budget::new(BudgetLimits::uniform(1, 0));
    let before = insufficient.snapshot();
    assert!(
        verify_integral_enclosure(
            &source,
            &x,
            &q(0, 1),
            &q(1, 1),
            &q(1, 100),
            &proposed,
            &mut insufficient,
        )
        .is_err()
    );
    assert_eq!(insufficient.snapshot(), before);
}

struct StopAt {
    calls: usize,
    stop: usize,
}
impl BudgetMeter for StopAt {
    fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.calls += 1;
        if self.calls == self.stop {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_before_during_and_before_publication_returns_no_receipt() {
    let x = Symbol::new("x");
    let source = Expr::Sym(x.clone());
    let proposed = witness(vec![q(0, 1), q(1, 1)]);
    let mut observed = StopAt {
        calls: 0,
        stop: usize::MAX,
    };
    verify_integral_enclosure(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 100),
        &proposed,
        &mut observed,
    )
    .unwrap();
    for stop in [1, observed.calls / 2, observed.calls] {
        assert!(matches!(
            verify_integral_enclosure(
                &source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                &proposed,
                &mut StopAt { calls: 0, stop },
            ),
            Err(QuadratureError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        ));
    }
    // The ordinary explicit unbounded convenience path still enforces the
    // same exact tolerance; this is not a fallback from interrupted work.
    assert!(
        verify_integral_enclosure(
            &source,
            &x,
            &q(0, 1),
            &q(1, 1),
            &q(1, 100),
            &proposed,
            &mut Unbounded,
        )
        .is_ok()
    );
}
