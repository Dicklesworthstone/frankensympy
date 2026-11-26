//! Independent rational anchors, original domains and operational failures for
//! the reciprocal-inverse numerical/certification paths.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BallError, BigInt, BigRational, Expr, QuadratureWitness, RealBall, RootBoxError,
    RootBoxWitness, Symbol, certify_integral_enclosure, certify_root_box, enclose_objective,
    enclose_residual_jacobian, metered_enclose_objective, metered_enclose_residual_jacobian,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn value(name: &str, argument: &RealBall, digits: u32) -> Result<RealBall, BallError> {
    match name {
        "acot" => argument.acot(digits),
        "acoth" => argument.acoth(digits),
        "asec" => argument.asec(digits),
        "acsc" => argument.acsc(digits),
        "asech" => argument.asech(digits),
        "acsch" => argument.acsch(digits),
        _ => unreachable!(),
    }
}

/// Alternating Taylor bounds, independent of the production inverse kernels.
fn atan_bounds(argument: BigRational) -> (BigRational, BigRational) {
    let mut power = argument.clone();
    let square = &argument * &argument;
    let mut sum = q(0, 1);
    for index in 0..160 {
        let term = &power / q(2 * index + 1, 1);
        if index % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
        power *= &square;
    }
    (sum.clone(), sum + power / q(321, 1))
}

/// Positive atanh series with its geometric tail proves the log(2) bounds.
fn log_two_bounds() -> (BigRational, BigRational) {
    let mut power = q(1, 3);
    let mut sum = q(0, 1);
    for index in 0..96 {
        sum += q(2, 2 * index + 1) * &power;
        power /= q(9, 1);
    }
    let tail = q(2, 193) * power / (q(1, 1) - q(1, 9));
    (sum.clone(), sum + tail)
}

fn half_pi_bounds() -> (BigRational, BigRational) {
    // Machin's identity pi/2 = 8 atan(1/5) - 2 atan(1/239).
    let (a, b) = atan_bounds(q(1, 5));
    let (c, d) = atan_bounds(q(1, 239));
    (q(8, 1) * a - q(2, 1) * d, q(8, 1) * b - q(2, 1) * c)
}

fn contains_bounds(value: &RealBall, bounds: &(BigRational, BigRational)) {
    assert!(value.lower() <= bounds.0);
    assert!(value.upper() >= bounds.1);
}

fn anchors() -> [(&'static str, BigRational, BigRational, BigRational); 6] {
    [
        ("acot", q(4, 3), q(-9, 25), q(216, 625)),
        ("acoth", q(5, 3), q(-9, 16), q(135, 128)),
        ("asec", q(5, 4), q(16, 15), q(-2176, 675)),
        ("acsc", q(5, 3), q(-9, 20), q(1107, 1600)),
        ("asech", q(4, 5), q(-25, 12), q(-875, 432)),
        ("acsch", q(4, 3), q(-9, 20), q(1107, 2000)),
    ]
}

#[test]
fn principal_values_match_independent_series_in_both_ground_lanes() {
    let circular = atan_bounds(q(3, 4));
    let hyperbolic = log_two_bounds();
    for (name, point, _, _) in anchors() {
        let bounds = if matches!(name, "acot" | "asec" | "acsc") {
            &circular
        } else {
            &hyperbolic
        };
        let result = value(name, &RealBall::exact(point.clone()), 18).unwrap();
        contains_bounds(&result, bounds);
        assert!(result.radius() < &q(1, 1_000_000_000_000_000));
        let source = function(name, Expr::Rational(point.clone()));
        contains_bounds(&source.eval_real_ball(18).unwrap(), bounds);
        let complex = source.eval_complex_ball(18).unwrap();
        contains_bounds(complex.real(), bounds);
        assert_eq!(complex.imag(), &RealBall::from_i64(0));
        if matches!(name, "acot" | "acoth" | "acsc" | "acsch") {
            let negative = value(name, &RealBall::exact(-point), 18).unwrap();
            contains_bounds(&negative, &(-bounds.1.clone(), -bounds.0.clone()));
        }
    }
}

#[test]
fn closed_value_endpoints_and_complete_interval_ranges_are_retained() {
    let angle = atan_bounds(q(3, 4));
    let logarithm = log_two_bounds();
    let half_pi = half_pi_bounds();
    contains_bounds(&RealBall::from_i64(0).acot(18).unwrap(), &half_pi);
    assert_eq!(
        RealBall::from_i64(1).asec(18).unwrap(),
        RealBall::from_i64(0)
    );
    assert_eq!(
        RealBall::from_i64(1).asech(18).unwrap(),
        RealBall::from_i64(0)
    );
    contains_bounds(&RealBall::from_i64(1).acsc(18).unwrap(), &half_pi);
    let secant = interval(q(1, 1), q(5, 4)).asec(18).unwrap();
    assert!(secant.contains(&q(0, 1)));
    contains_bounds(&secant, &angle);
    let cosecant = interval(q(1, 1), q(5, 3)).acsc(18).unwrap();
    contains_bounds(&cosecant, &angle);
    contains_bounds(&cosecant, &half_pi);
    let sech = interval(q(4, 5), q(1, 1)).asech(18).unwrap();
    assert!(sech.contains(&q(0, 1)));
    contains_bounds(&sech, &logarithm);
    let negative_secant = interval(q(-5, 4), q(-1, 1)).asec(18).unwrap();
    contains_bounds(
        &negative_secant,
        &(q(2, 1) * &half_pi.0 - &angle.1, q(2, 1) * &half_pi.1),
    );
}

#[test]
fn exact_real_domain_and_precision_refusals_precede_large_work() {
    for name in ["acot", "acoth", "asec", "acsc", "asech", "acsch"] {
        assert!(value(name, &interval(q(-2, 1), q(2, 1)), 18).is_err());
        for precision in [
            0,
            RealBall::MAX_RECIPROCAL_INVERSE_PRECISION_DIGITS + 1,
            u32::MAX,
        ] {
            assert!(value(name, &RealBall::from_i64(2), precision).is_err());
        }
        let huge = BigRational::new(BigInt::one(), BigInt::one() << 140_000u32);
        assert!(matches!(
            value(name, &RealBall::exact(huge), 18),
            Err(BallError::ResourceLimitExceeded { .. })
        ));
    }
    for name in ["acoth", "asec", "acsc"] {
        assert!(value(name, &RealBall::exact(q(1, 2)), 18).is_err());
    }
    assert!(RealBall::from_i64(1).acoth(18).is_err());
    assert!(RealBall::from_i64(-1).acoth(18).is_err());
    assert!(RealBall::from_i64(-1).asech(18).is_err());
    assert!(RealBall::from_i64(2).asech(18).is_err());
    assert!(RealBall::from_i64(0).acsch(18).is_err());
}

#[test]
fn independently_known_derivatives_preserve_signs_and_multivariate_chain_terms() {
    let axes = [Symbol::new("x"), Symbol::new("y")];
    let inner = Expr::Add(vec![
        Expr::Pow(
            Arc::new(Expr::Sym(axes[0].clone())),
            Arc::new(Expr::from_i64(2)),
        ),
        Expr::Sym(axes[1].clone()),
    ]);
    for (name, point, first, second) in anchors() {
        for sign in [1, -1] {
            if sign == -1 && name == "asech" {
                continue;
            }
            let center = q(sign, 1) * &point;
            let first = first.clone();
            let second = q(sign, 1) * &second;
            let source = function(name, Expr::Sym(axes[0].clone()));
            let domain = [RealBall::exact(center.clone())];
            let c1 =
                enclose_residual_jacobian(std::slice::from_ref(&source), &axes[..1], &domain, 18)
                    .unwrap();
            assert!(
                c1.jacobian()[0][0].contains(&first),
                "{name} first derivative"
            );
            assert!(!c1.jacobian()[0][0].contains(&(-first.clone())));
            let c2 = enclose_objective(&source, &axes[..1], &domain, 18).unwrap();
            assert!(c2.gradient()[0].contains(&first));
            assert!(
                c2.hessian()[0][0].contains(&second),
                "{name} second derivative"
            );
            let composed = function(name, inner.clone());
            let composed_domain = [RealBall::exact(q(1, 2)), RealBall::exact(center - q(1, 4))];
            let jet = enclose_objective(&composed, &axes, &composed_domain, 18).unwrap();
            assert!(jet.gradient().iter().all(|entry| entry.contains(&first)));
            assert!(jet.hessian()[0][0].contains(&(&second + q(2, 1) * &first)));
            for (row, column) in [(0, 1), (1, 0), (1, 1)] {
                assert!(jet.hessian()[row][column].contains(&second));
            }
        }
    }
}

#[test]
fn original_poles_branch_cuts_and_nonsmooth_endpoints_cannot_be_erased() {
    let axis = Symbol::new("x");
    for (name, point) in [
        ("acot", 0),
        ("acoth", 1),
        ("acoth", -1),
        ("asec", 1),
        ("asec", -1),
        ("acsc", 1),
        ("acsc", -1),
        ("asech", 1),
        ("asech", 0),
        ("acsch", 0),
    ] {
        let inverse = function(name, Expr::Sym(axis.clone()));
        for source in [
            inverse.clone(),
            Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
            Expr::Pow(Arc::new(inverse), Arc::new(Expr::from_i64(0))),
        ] {
            let domain = [RealBall::from_i64(point)];
            assert!(matches!(
                enclose_residual_jacobian(
                    std::slice::from_ref(&source),
                    std::slice::from_ref(&axis),
                    &domain,
                    18,
                ),
                Err(RootBoxError::Domain(_))
            ));
            assert!(matches!(
                enclose_objective(&source, std::slice::from_ref(&axis), &domain, 18),
                Err(RootBoxError::Domain(_))
            ));
        }
    }
    let invalid_child = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
    ]);
    for name in ["acot", "acoth", "asec", "acsc", "asech", "acsch"] {
        assert!(
            function(name, invalid_child.clone())
                .eval_real_ball(18)
                .is_err()
        );
    }
}

#[test]
fn independent_root_acceptance_rejects_changed_sources_and_inverse_sign_mutants() {
    let axes = [Symbol::new("x")];
    for (name, center, slope, _) in anchors() {
        let target = if matches!(name, "acot" | "asec" | "acsc") {
            function("atan", Expr::Rational(q(3, 4)))
        } else {
            function("log", Expr::from_i64(2))
        };
        let source = function(name, Expr::Sym(axes[0].clone())) - target;
        let witness = RootBoxWitness {
            domain: vec![RealBall::new(center.clone(), q(1, 100)).unwrap()],
            center: vec![center.clone()],
            preconditioner: vec![vec![q(1, 1) / slope]],
            precision_digits: 18,
        };
        let receipt = certify_root_box(std::slice::from_ref(&source), &axes, &witness).unwrap();
        assert!(receipt.root_enclosure()[0].contains(&center));
        assert!(receipt.contraction_bound() < &q(1, 1));
        assert!(certify_root_box(&[source.clone() + Expr::from_i64(1)], &axes, &witness).is_err());
        let mut altered = witness;
        altered.preconditioner[0][0] = -altered.preconditioner[0][0].clone();
        assert!(certify_root_box(&[source], &axes, &altered).is_err());
    }
}

#[test]
fn finite_nonsmooth_endpoints_can_prove_proper_integrals_without_smooth_receipts() {
    let axis = Symbol::new("x");
    let source = function("asec", Expr::Sym(axis.clone()));
    let witness = QuadratureWitness {
        partition: vec![q(1, 1), q(5, 4)],
        precision_digits: 18,
    };
    let receipt =
        certify_integral_enclosure(&source, &axis, &q(1, 1), &q(5, 4), &q(1, 1), &witness).unwrap();
    // Integral_1^(5/4) asec(x) dx = (5/4)*atan(3/4) - log(2).
    let angle = atan_bounds(q(3, 4));
    let logarithm = log_two_bounds();
    contains_bounds(
        receipt.enclosure(),
        &(
            q(5, 4) * angle.0 - logarithm.1,
            q(5, 4) * angle.1 - logarithm.0,
        ),
    );
    assert!(
        certify_integral_enclosure(
            &source,
            &axis,
            &q(1, 1),
            &q(5, 4),
            &q(1, 1_000_000),
            &witness,
        )
        .is_err()
    );
}

struct Checkpoints {
    visits: usize,
    stop: Option<usize>,
    stopped: bool,
}

impl BudgetMeter for Checkpoints {
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
        self.visits += 1;
        if self.stop == Some(self.visits) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn metered_case(
    name: &str,
    second: bool,
    meter: &mut impl BudgetMeter,
) -> Result<(), RootBoxError> {
    let (_, point, _, _) = anchors().into_iter().find(|case| case.0 == name).unwrap();
    let axes = [Symbol::new("x")];
    let source = function(name, Expr::Sym(axes[0].clone()));
    let domain = [RealBall::exact(point)];
    if second {
        metered_enclose_objective(&source, &axes, &domain, 8, meter).map(|_| ())
    } else {
        metered_enclose_residual_jacobian(&[source], &axes, &domain, 8, meter).map(|_| ())
    }
}

#[test]
fn every_function_and_derivative_order_retains_terminal_cancellation_and_budget_errors() {
    for (name, _, _, _) in anchors() {
        for second in [false, true] {
            let mut baseline = Checkpoints {
                visits: 0,
                stop: None,
                stopped: false,
            };
            metered_case(name, second, &mut baseline).unwrap();
            for stop in [1, baseline.visits / 2, baseline.visits] {
                let mut meter = Checkpoints {
                    visits: 0,
                    stop: Some(stop),
                    stopped: false,
                };
                assert_eq!(
                    metered_case(name, second, &mut meter),
                    Err(RootBoxError::Meter(MeterError::Cancelled))
                );
            }
            let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
            assert!(matches!(
                metered_case(name, second, &mut budget),
                Err(RootBoxError::Meter(MeterError::Budget(_)))
            ));
        }
    }
}
