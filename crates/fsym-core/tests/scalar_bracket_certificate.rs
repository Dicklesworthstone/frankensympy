use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness, ScalarRootBracketError,
    ScalarRootBracketWitness, ScalarRootBracketWorkspace, Symbol, SymbolIdentity, certify_root_box,
    certify_scalar_root_bracket, enclose_residual_jacobian, verify_scalar_root_bracket,
};
use std::cmp::Ordering;
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(value: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(value), Arc::new(Expr::from_i64(exponent)))
}
fn abs(value: Expr) -> Expr {
    Expr::Function("Abs".into(), vec![value])
}
fn witness(lower: BigRational, upper: BigRational) -> ScalarRootBracketWitness {
    ScalarRootBracketWitness {
        lower,
        upper,
        precision_digits: 24,
    }
}

#[test]
fn odd_multiple_root_has_existence_evidence_without_a_nonsingular_jacobian() {
    let x = Symbol::new("multiple_x");
    let source = power(Expr::Sym(x.clone()), 3);
    let proposed = witness(q(-1, 1000), q(1, 1000));
    let receipt = certify_scalar_root_bracket(
        &source,
        &x,
        &q(-1, 1),
        &q(1, 1),
        &q(1, 100),
        &proposed,
    )
    .unwrap();
    assert_eq!(receipt.expression(), &source);
    assert_eq!(receipt.variable(), &x);
    assert_eq!(receipt.lower(), &q(-1, 1));
    assert_eq!(receipt.upper(), &q(1, 1));
    assert_eq!(receipt.absolute_tolerance(), &q(1, 100));
    assert_eq!(receipt.witness(), &proposed);
    assert!(receipt.root_enclosure().contains(&q(0, 1)));
    assert!(receipt.endpoint_values()[0].is_negative());
    assert!(receipt.endpoint_values()[1].is_positive());

    // x^3 has a zero derivative at its root. The stronger existing claim
    // must remain unavailable even though the sign-bracket claim succeeds.
    let stronger = RootBoxWitness {
        domain: vec![RealBall::new(q(0, 1), q(1, 1)).unwrap()],
        center: vec![q(0, 1)],
        preconditioner: vec![vec![q(1, 1)]],
        precision_digits: 24,
    };
    assert!(matches!(
        certify_root_box(&[source], &[x], &stronger),
        Err(RootBoxError::NotContractive)
    ));
}

#[test]
fn absolute_value_corner_has_a_root_without_a_classical_derivative() {
    let x = Symbol::new("corner_x");
    let sx = Expr::Sym(x.clone());
    let source = Expr::Add(vec![
        Expr::Mul(vec![Expr::from_i64(2), sx.clone()]),
        abs(sx),
    ]);
    let receipt = certify_scalar_root_bracket(
        &source,
        &x,
        &q(-1, 1),
        &q(1, 1),
        &q(1, 100),
        &witness(q(0, 1), q(0, 1)),
    )
    .unwrap();
    assert_eq!(receipt.root_enclosure(), &RealBall::from_i64(0));
    assert_eq!(
        receipt.endpoint_values(),
        &[RealBall::from_i64(0), RealBall::from_i64(0)],
    );
    assert!(
        enclose_residual_jacobian(
            &[source],
            &[x],
            &[RealBall::new(q(0, 1), q(1, 1)).unwrap()],
            24,
        )
        .is_err()
    );
}

#[test]
fn multiple_roots_in_one_bracket_do_not_become_a_uniqueness_claim() {
    let x = Symbol::new("several_x");
    let sx = Expr::Sym(x.clone());
    let source = Expr::Mul(vec![
        sx.clone(),
        Expr::Add(vec![power(sx, 2), Expr::from_i64(-1)]),
    ]);
    let receipt = certify_scalar_root_bracket(
        &source,
        &x,
        &q(-2, 1),
        &q(2, 1),
        &q(4, 1),
        &witness(q(-2, 1), q(2, 1)),
    )
    .unwrap();
    // The polynomial x*(x^2-1) has these three distinct exact roots.
    for root in [-1, 0, 1] {
        assert!(receipt.root_enclosure().contains(&q(root, 1)));
    }
}

#[test]
fn closed_endpoint_and_degenerate_interval_need_an_exact_zero() {
    let x = Symbol::new("endpoint_x");
    let source = power(Expr::Sym(x.clone()), 2);
    for upper in [q(0, 1), q(2, 1)] {
        let receipt = certify_scalar_root_bracket(
            &source,
            &x,
            &q(0, 1),
            &upper,
            &q(1, 100),
            &witness(q(0, 1), q(0, 1)),
        )
        .unwrap();
        assert_eq!(receipt.root_enclosure(), &RealBall::from_i64(0));
    }
    assert!(matches!(
        certify_scalar_root_bracket(
            &source,
            &x,
            &q(1, 1),
            &q(1, 1),
            &q(1, 100),
            &witness(q(1, 1), q(1, 1)),
        ),
        Err(ScalarRootBracketError::NoSignChange)
    ));
}

#[test]
fn wrong_source_and_same_spelling_different_identity_cannot_replay_a_root() {
    let x = Symbol::new("x");
    let source = Expr::Sym(x.clone());
    let proposed = witness(q(-1, 100), q(1, 100));
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    assert!(matches!(
        certify_scalar_root_bracket(
            &source,
            &other,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 10),
            &proposed,
        ),
        Err(ScalarRootBracketError::Enclosure(RootBoxError::Unsupported(_)))
    ));
    let changed = Expr::Add(vec![source, Expr::from_i64(1)]);
    assert!(matches!(
        certify_scalar_root_bracket(
            &changed,
            &x,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 10),
            &proposed,
        ),
        Err(ScalarRootBracketError::NoSignChange)
    ));
}

#[test]
fn endpoint_containment_order_and_width_are_independently_checked() {
    let x = Symbol::new("mutant_x");
    let source = Expr::Sym(x.clone());
    for proposed in [
        witness(q(-2, 1), q(0, 1)),
        witness(q(0, 1), q(2, 1)),
        witness(q(1, 2), q(-1, 2)),
    ] {
        assert!(matches!(
            certify_scalar_root_bracket(
                &source,
                &x,
                &q(-1, 1),
                &q(1, 1),
                &q(4, 1),
                &proposed,
            ),
            Err(ScalarRootBracketError::InvalidBracket(_))
        ));
    }
    assert!(matches!(
        certify_scalar_root_bracket(
            &source,
            &x,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 100),
            &witness(q(-1, 1), q(1, 1)),
        ),
        Err(ScalarRootBracketError::AccuracyNotReached)
    ));
}

#[test]
fn original_poles_survive_zero_factors_cancellation_and_zeroth_powers() {
    let x = Symbol::new("guard_x");
    let sx = Expr::Sym(x.clone());
    let reciprocal = power(sx.clone(), -1);
    for source in [
        reciprocal.clone(),
        Expr::Add(vec![
            sx.clone(),
            Expr::Mul(vec![Expr::from_i64(0), reciprocal.clone()]),
        ]),
        Expr::Add(vec![
            Expr::Mul(vec![sx.clone(), reciprocal.clone()]),
            Expr::from_i64(-1),
        ]),
        Expr::Add(vec![power(reciprocal, 0), Expr::from_i64(-1)]),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function("log".into(), vec![sx]),
        ]),
    ] {
        assert!(
            certify_scalar_root_bracket(
                &source,
                &x,
                &q(-1, 1),
                &q(1, 1),
                &q(2, 1),
                &witness(q(-1, 1), q(1, 1)),
            )
            .is_err()
        );
    }
}

#[test]
fn a_narrow_valid_root_cannot_hide_a_pole_elsewhere_in_the_requested_interval() {
    let x = Symbol::new("full_domain_x");
    let sx = Expr::Sym(x.clone());
    let source = Expr::Mul(vec![
        sx.clone(),
        power(Expr::Add(vec![sx, Expr::from_i64(1)]), -1),
    ]);
    assert!(
        certify_scalar_root_bracket(
            &source,
            &x,
            &q(-2, 1),
            &q(2, 1),
            &q(1, 100),
            &witness(q(0, 1), q(0, 1)),
        )
        .is_err()
    );
}

fn piecewise(value: Expr, condition: Expr, fallback: Expr) -> Expr {
    Expr::Function(
        "Piecewise".into(),
        vec![
            Expr::Function("Tuple".into(), vec![value, condition]),
            Expr::Function("Tuple".into(), vec![fallback, Expr::symbol("True")]),
        ],
    )
}

#[test]
fn discontinuous_piecewise_sign_changes_do_not_certify_roots() {
    let x = Symbol::new("jump_x");
    let source = piecewise(
        Expr::from_i64(-1),
        Expr::Function(
            "StrictLessThan".into(),
            vec![Expr::Sym(x.clone()), Expr::from_i64(0)],
        ),
        Expr::from_i64(1),
    );
    assert!(
        certify_scalar_root_bracket(
            &source,
            &x,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 10),
            &witness(q(-1, 100), q(1, 100)),
        )
        .is_err()
    );
}

#[test]
fn a_stable_piecewise_branch_retains_lazy_domain_semantics() {
    let x = Symbol::new("stable_x");
    let sx = Expr::Sym(x.clone());
    let source = piecewise(
        abs(sx.clone()),
        Expr::Function("StrictLessThan".into(), vec![sx, Expr::from_i64(2)]),
        Expr::Function("log".into(), vec![Expr::from_i64(0)]),
    );
    let receipt = certify_scalar_root_bracket(
        &source,
        &x,
        &q(-1, 1),
        &q(1, 1),
        &q(1, 100),
        &witness(q(0, 1), q(0, 1)),
    )
    .unwrap();
    assert_eq!(receipt.root_enclosure(), &RealBall::from_i64(0));
}

#[test]
fn enclosure_containing_zero_does_not_mean_an_exact_zero() {
    let x = Symbol::new("sign_x");
    let source = Expr::Sym(x.clone());
    let mut meter = Unbounded;
    let mut workspace = ScalarRootBracketWorkspace::new(&source, &x, 24, &mut meter).unwrap();
    let tiny = q(1, 1_000_000);
    assert_eq!(
        workspace
            .sign(&RealBall::new(q(0, 1), tiny.clone()).unwrap())
            .unwrap(),
        None
    );
    assert_eq!(
        workspace.sign(&RealBall::exact(tiny)).unwrap(),
        Some(Ordering::Greater)
    );
    assert_eq!(
        workspace.sign(&RealBall::from_i64(0)).unwrap(),
        Some(Ordering::Equal)
    );
    assert_eq!(
        workspace.sign(&RealBall::from_i64(-1)).unwrap(),
        Some(Ordering::Less)
    );
}

#[test]
fn bounded_witness_transport_rejects_claim_inflation_and_malformed_rationals() {
    let proposed = witness(q(-1, 100), q(1, 100));
    let wire = serde_json::to_value(&proposed).unwrap();
    assert_eq!(
        serde_json::from_value::<ScalarRootBracketWitness>(wire.clone()).unwrap(),
        proposed
    );
    for key in ["accepted", "unique", "endpoint_values"] {
        let mut bad = wire.clone();
        bad[key] = serde_json::json!(true);
        assert!(serde_json::from_value::<ScalarRootBracketWitness>(bad).is_err());
    }
    for scalar in [
        serde_json::json!([[1, [2]], [1, [2]]]),
        serde_json::json!([[1, [1]], [-1, [1]]]),
        serde_json::json!([[0, [1]], [1, [1]]]),
    ] {
        let mut bad = wire.clone();
        bad["lower"] = scalar;
        assert!(serde_json::from_value::<ScalarRootBracketWitness>(bad).is_err());
    }
    for precision in [0, 7, 129, u32::MAX] {
        let mut bad = wire.clone();
        bad["precision_digits"] = serde_json::json!(precision);
        assert!(serde_json::from_value::<ScalarRootBracketWitness>(bad).is_err());
    }
    let mut bad = wire.clone();
    bad["lower"] = serde_json::json!(q(2, 1));
    assert!(serde_json::from_value::<ScalarRootBracketWitness>(bad).is_err());
    let huge = BigRational::from_integer(BigInt::from(1) << 4096u32);
    let mut bad = wire;
    bad["upper"] = serde_json::json!(huge);
    assert!(serde_json::from_value::<ScalarRootBracketWitness>(bad).is_err());
}

#[test]
fn invalid_requests_and_intrinsic_source_bounds_refuse_before_recursion() {
    let x = Symbol::new("bounded_x");
    let source = Expr::Sym(x.clone());
    let proposed = witness(q(0, 1), q(0, 1));
    for tolerance in [q(0, 1), q(-1, 1)] {
        assert!(matches!(
            certify_scalar_root_bracket(
                &source,
                &x,
                &q(-1, 1),
                &q(1, 1),
                &tolerance,
                &proposed,
            ),
            Err(ScalarRootBracketError::InvalidTolerance)
        ));
    }
    assert!(matches!(
        certify_scalar_root_bracket(
            &source,
            &x,
            &q(1, 1),
            &q(-1, 1),
            &q(1, 10),
            &proposed,
        ),
        Err(ScalarRootBracketError::InvalidBracket(_))
    ));
    let mut deep = source;
    for _ in 0..40 {
        deep = power(deep, 1);
    }
    assert!(matches!(
        certify_scalar_root_bracket(
            &deep,
            &x,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 10),
            &proposed,
        ),
        Err(ScalarRootBracketError::Enclosure(RootBoxError::ResourceLimit(_)))
    ));
}

#[test]
fn real_budgets_are_charged_and_failed_reservations_are_atomic() {
    let x = Symbol::new("budget_x");
    let source = power(Expr::Sym(x.clone()), 3);
    let proposed = witness(q(-1, 100), q(1, 100));
    let limit = 512 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    let receipt = verify_scalar_root_bracket(
        &source,
        &x,
        &q(-1, 1),
        &q(1, 1),
        &q(1, 10),
        &proposed,
        &mut budget,
    )
    .unwrap();
    assert!(receipt.root_enclosure().contains(&q(0, 1)));
    assert!(budget.remaining(Dimension::ComputeSteps) < limit);
    assert!(budget.remaining(Dimension::MemoryBytes) < limit);

    let mut insufficient = Budget::new(BudgetLimits::uniform(1, 0));
    let before = insufficient.snapshot();
    assert!(matches!(
        verify_scalar_root_bracket(
            &source,
            &x,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 10),
            &proposed,
            &mut insufficient,
        ),
        Err(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(insufficient.snapshot(), before);
}

#[test]
fn reusable_evaluation_keeps_one_reservation_and_cumulative_work() {
    let x = Symbol::new("reuse_x");
    let source = power(Expr::Sym(x.clone()), 3);
    let limit = 96 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    {
        let mut workspace = ScalarRootBracketWorkspace::new(&source, &x, 24, &mut budget).unwrap();
        for numerator in -4..=4 {
            let value = workspace.enclose(&q(numerator, 4), &q(numerator, 4)).unwrap();
            assert_eq!(value, RealBall::exact(q(numerator * numerator * numerator, 64)));
        }
    }
    let consumed = budget.remaining(Dimension::MemoryBytes);
    assert!(consumed < limit);
    assert!(ScalarRootBracketWorkspace::new(&source, &x, 24, &mut budget).is_err());
    assert_eq!(budget.remaining(Dimension::MemoryBytes), consumed);
}

#[derive(Default)]
struct Checkpoints {
    calls: usize,
    cancel_at: Option<usize>,
}
impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.calls += 1;
        if self.cancel_at.is_some_and(|at| self.calls >= at) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_before_during_and_after_receipt_construction_never_returns_acceptance() {
    let x = Symbol::new("cancel_x");
    let source = power(Expr::Sym(x.clone()), 3);
    let proposed = witness(q(-1, 100), q(1, 100));
    let run = |meter: &mut Checkpoints| {
        verify_scalar_root_bracket(
            &source,
            &x,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 10),
            &proposed,
            meter,
        )
    };
    let mut baseline = Checkpoints::default();
    run(&mut baseline).unwrap();
    assert!(baseline.calls > 4);
    for cancel_at in [1, baseline.calls / 2, baseline.calls] {
        let mut meter = Checkpoints {
            calls: 0,
            cancel_at: Some(cancel_at),
        };
        assert!(matches!(
            run(&mut meter),
            Err(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        ));
    }
}
