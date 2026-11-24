//! Public verifier regressions for continuity without differentiability.
//!
//! Exact root locations and elementary integral values supply independent
//! mathematical oracles. The checker receives only source and partitions;
//! no value, derivative, error estimate or trusted acceptance bit is supplied.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigRational, Expr, QuadratureError, QuadratureWitness, RealBall, RootBoxError,
    ScalarRootBracketError, ScalarRootBracketWitness, Symbol, certify_integral_enclosure,
    certify_scalar_root_bracket, enclose_objective, enclose_residual_jacobian,
    verify_integral_enclosure, verify_scalar_root_bracket,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: BigRational) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::Rational(exponent)))
}

fn subtract(source: Expr, value: BigRational) -> Expr {
    Expr::Add(vec![source, Expr::Rational(-value)])
}

fn bracket(lower: BigRational, upper: BigRational) -> ScalarRootBracketWitness {
    ScalarRootBracketWitness {
        lower,
        upper,
        precision_digits: 16,
    }
}

fn partition(points: Vec<BigRational>) -> QuadratureWitness {
    QuadratureWitness {
        partition: points,
        precision_digits: 16,
    }
}

#[test]
fn finite_closed_endpoints_admit_exact_roots_without_derivatives() {
    let x = Symbol::new("continuous_root_x");
    let sx = Expr::Sym(x.clone());
    for (source, lower, upper, root) in [
        (function("sqrt", sx.clone()), q(0, 1), q(1, 1), q(0, 1)),
        (
            subtract(function("sqrt", sx.clone()), q(1, 2)),
            q(0, 1),
            q(1, 1),
            q(1, 4),
        ),
        (function("asin", sx.clone()), q(-1, 1), q(1, 1), q(0, 1)),
        (function("acos", sx.clone()), q(-1, 1), q(1, 1), q(1, 1)),
        (function("acosh", sx), q(1, 1), q(2, 1), q(1, 1)),
    ] {
        let proposed = bracket(root.clone(), root.clone());
        let receipt =
            certify_scalar_root_bracket(&source, &x, &lower, &upper, &q(1, 100), &proposed)
                .unwrap();
        assert_eq!(receipt.expression(), &source);
        assert_eq!(receipt.lower(), &lower);
        assert_eq!(receipt.upper(), &upper);
        assert_eq!(receipt.witness(), &proposed);
        assert_eq!(receipt.root_enclosure(), &RealBall::exact(root));
        assert_eq!(
            receipt.endpoint_values(),
            &[RealBall::from_i64(0), RealBall::from_i64(0)],
        );
    }
}

#[test]
fn positive_fractional_power_signs_certify_an_interior_root_from_a_zero_endpoint_domain() {
    let x = Symbol::new("fractional_root_x");
    let source = subtract(power(Expr::Sym(x.clone()), q(1, 3)), q(1, 2));
    // The exact root is 1/8. Neither proposed endpoint is a root, so the
    // independent exponential/logarithmic value bounds must prove both signs.
    let receipt = certify_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 8),
        &bracket(q(1, 16), q(3, 16)),
    )
    .unwrap();
    assert!(receipt.root_enclosure().contains(&q(1, 8)));
    assert!(receipt.endpoint_values()[0].is_negative());
    assert!(receipt.endpoint_values()[1].is_positive());
}

#[test]
fn a_valid_root_subinterval_never_hides_an_original_domain_hole() {
    let x = Symbol::new("original_domain_x");
    let sx = Expr::Sym(x.clone());
    let hidden_pole = power(subtract(sx.clone(), q(1, 2)), q(-1, 1));
    for (source, lower, upper, root) in [
        (function("sqrt", sx.clone()), q(-1, 1), q(1, 1), q(0, 1)),
        (
            Expr::Mul(vec![Expr::from_i64(0), hidden_pole]),
            q(0, 1),
            q(1, 1),
            q(0, 1),
        ),
        (
            Expr::Mul(vec![Expr::from_i64(0), function("log", sx.clone())]),
            q(0, 1),
            q(1, 1),
            q(1, 1),
        ),
        (
            subtract(power(function("sqrt", sx.clone()), q(-1, 1)), q(2, 1)),
            q(0, 1),
            q(1, 1),
            q(1, 4),
        ),
        (
            subtract(power(sx.clone(), q(-1, 3)), q(1, 1)),
            q(0, 1),
            q(1, 1),
            q(1, 1),
        ),
        (
            subtract(
                Expr::Pow(Arc::new(sx.clone()), Arc::new(sx.clone())),
                q(1, 1),
            ),
            q(0, 1),
            q(1, 1),
            q(1, 1),
        ),
        (
            Expr::Mul(vec![
                Expr::from_i64(0),
                function("asin", Expr::Mul(vec![Expr::from_i64(2), sx])),
            ]),
            q(0, 1),
            q(1, 1),
            q(0, 1),
        ),
    ] {
        assert!(
            certify_scalar_root_bracket(
                &source,
                &x,
                &lower,
                &upper,
                &q(1, 10),
                &bracket(root.clone(), root),
            )
            .is_err(),
            "the complete original interval must be admitted: {source:?}",
        );
    }
}

#[test]
fn a_boundary_value_receipt_cannot_accept_a_forged_root_or_a_changed_source() {
    let x = Symbol::new("forged_root_x");
    let sx = Expr::Sym(x.clone());
    let source = subtract(function("sqrt", sx.clone()), q(1, 2));
    let valid = bracket(q(1, 4), q(1, 4));
    certify_scalar_root_bracket(&source, &x, &q(0, 1), &q(1, 1), &q(1, 100), &valid).unwrap();
    for (changed_source, proposed) in [
        (source, bracket(q(1, 1), q(1, 1))),
        (subtract(function("sqrt", sx), q(3, 4)), valid),
    ] {
        assert!(matches!(
            certify_scalar_root_bracket(
                &changed_source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                &proposed,
            ),
            Err(ScalarRootBracketError::NoSignChange)
        ));
    }
}

#[test]
fn continuous_value_support_does_not_relax_c1_or_c2_claims() {
    let x = Symbol::new("regularity_boundary_x");
    let sx = Expr::Sym(x.clone());
    for (source, domain) in [
        (
            function("sqrt", sx.clone()),
            RealBall::new(q(1, 2), q(1, 2)).unwrap(),
        ),
        (
            power(sx.clone(), q(1, 3)),
            RealBall::new(q(1, 2), q(1, 2)).unwrap(),
        ),
        (
            function("asin", sx.clone()),
            RealBall::new(q(0, 1), q(1, 1)).unwrap(),
        ),
        (
            function("acos", sx.clone()),
            RealBall::new(q(0, 1), q(1, 1)).unwrap(),
        ),
        (
            function("acosh", sx),
            RealBall::new(q(3, 2), q(1, 2)).unwrap(),
        ),
    ] {
        assert!(
            enclose_objective(
                &source,
                std::slice::from_ref(&x),
                std::slice::from_ref(&domain),
                16,
            )
            .is_err(),
        );
        assert!(
            enclose_residual_jacobian(
                std::slice::from_ref(&source),
                std::slice::from_ref(&x),
                std::slice::from_ref(&domain),
                16,
            )
            .is_err(),
        );
    }
}

#[test]
fn a_continuous_only_integral_cell_uses_the_entire_range_without_fake_curvature() {
    let x = Symbol::new("range_integral_x");
    let source = function("sqrt", Expr::Sym(x.clone()));
    let proposed = partition(vec![q(0, 1), q(1, 1)]);
    let receipt =
        certify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 1), &proposed).unwrap();
    assert_eq!(receipt.enclosure().lower(), q(0, 1));
    assert_eq!(receipt.enclosure().upper(), q(1, 1));
    assert!(receipt.enclosure().contains(&q(2, 3)));
    // A midpoint value with absent derivatives treated as zero would wrongly
    // claim arbitrary accuracy. This partition supplies no such information.
    assert!(matches!(
        certify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 100), &proposed,),
        Err(QuadratureError::AccuracyNotReached)
    ));
    let reversed =
        certify_integral_enclosure(&source, &x, &q(1, 1), &q(0, 1), &q(1, 1), &proposed).unwrap();
    assert_eq!(reversed.enclosure().lower(), q(-1, 1));
    assert_eq!(reversed.enclosure().upper(), q(0, 1));
    assert!(reversed.enclosure().contains(&q(-2, 3)));
    let zero = certify_integral_enclosure(
        &source,
        &x,
        &q(0, 1),
        &q(0, 1),
        &q(1, 100),
        &partition(vec![q(0, 1)]),
    )
    .unwrap();
    assert_eq!(zero.enclosure(), &RealBall::from_i64(0));
}

#[test]
fn complete_partitions_enclose_independently_known_endpoint_integrals() {
    let x = Symbol::new("partition_integral_x");
    let sx = Expr::Sym(x.clone());
    let proposed = partition((0..=32).map(|i| q(i, 32)).collect());
    for (source, exact_integral) in [
        (function("sqrt", sx.clone()), q(2, 3)),
        (power(sx.clone(), q(1, 3)), q(3, 4)),
        (power(sx.clone(), q(3, 2)), q(2, 5)),
        (function("acos", sx), q(1, 1)),
    ] {
        let receipt =
            certify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 16), &proposed)
                .unwrap();
        assert_eq!(receipt.integrand(), &source);
        assert_eq!(receipt.interval_count(), 32);
        assert!(receipt.enclosure().contains(&exact_integral));
        assert!(receipt.enclosure().upper() - receipt.enclosure().lower() <= q(1, 16));
    }
}

#[test]
fn improper_integrability_never_excuses_an_undefined_proper_endpoint() {
    let x = Symbol::new("proper_endpoint_x");
    let sx = Expr::Sym(x.clone());
    for source in [
        power(function("sqrt", sx.clone()), q(-1, 1)),
        power(sx.clone(), q(-1, 3)),
        function("log", sx),
    ] {
        // These improper integrals have finite values, but no value at zero
        // was supplied by the source and this API makes a proper claim.
        for points in [vec![q(0, 1)], vec![q(0, 1), q(1, 1)]] {
            let upper = points.last().unwrap().clone();
            assert!(
                certify_integral_enclosure(
                    &source,
                    &x,
                    &q(0, 1),
                    &upper,
                    &q(1, 1),
                    &partition(points),
                )
                .is_err(),
            );
        }
    }
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
fn continuity_retries_preserve_real_budgets_and_late_cancellation() {
    let x = Symbol::new("continuous_budget_x");
    let source = function("sqrt", Expr::Sym(x.clone()));
    let root = bracket(q(0, 1), q(0, 1));
    let cells = partition(vec![q(0, 1), q(1, 1)]);
    let limit = 512 * 1024 * 1024;
    let mut root_budget = Budget::new(BudgetLimits::uniform(limit, 0));
    let mut integral_budget = Budget::new(BudgetLimits::uniform(limit, 0));
    verify_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 100),
        &root,
        &mut root_budget,
    )
    .unwrap();
    verify_integral_enclosure(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 1),
        &cells,
        &mut integral_budget,
    )
    .unwrap();
    for budget in [&root_budget, &integral_budget] {
        assert!(budget.remaining(Dimension::ComputeSteps) < limit);
        assert!(budget.remaining(Dimension::MemoryBytes) < limit);
    }

    let run_root = |meter: &mut Checkpoints| {
        verify_scalar_root_bracket(&source, &x, &q(0, 1), &q(1, 1), &q(1, 100), &root, meter)
    };
    let run_integral = |meter: &mut Checkpoints| {
        verify_integral_enclosure(&source, &x, &q(0, 1), &q(1, 1), &q(1, 1), &cells, meter)
    };
    let mut root_baseline = Checkpoints::default();
    let mut integral_baseline = Checkpoints::default();
    run_root(&mut root_baseline).unwrap();
    run_integral(&mut integral_baseline).unwrap();
    assert!(root_baseline.calls > 4);
    assert!(integral_baseline.calls > 4);
    for cancel_at in [1, root_baseline.calls / 2, root_baseline.calls] {
        let mut meter = Checkpoints {
            calls: 0,
            cancel_at: Some(cancel_at),
        };
        assert!(matches!(
            run_root(&mut meter),
            Err(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        ));
    }
    for cancel_at in [1, integral_baseline.calls / 2, integral_baseline.calls] {
        let mut meter = Checkpoints {
            calls: 0,
            cancel_at: Some(cancel_at),
        };
        assert!(matches!(
            run_integral(&mut meter),
            Err(QuadratureError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        ));
    }
}
