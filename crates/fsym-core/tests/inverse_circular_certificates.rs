//! Independent principal-branch and original-source inverse-circular regressions.
//! Rational series and fixed analytic derivatives are test oracles, never inputs
//! to the certificate checker. No floating approximation supplies a bound.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BallError, BigInt, BigRational, Constant, Expr, RealBall, RootBoxError, RootBoxWitness,
    StationaryPointKind, Symbol, certify_root_box, certify_stationary_point, enclose_objective,
    enclose_residual_jacobian, metered_enclose_objective, metered_enclose_residual_jacobian,
    verify_root_box, verify_stationary_point,
};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

/// For 0 <= x < 1 every arcsine series term is positive and the ratio of
/// successive terms is below x^2. A geometric tail therefore encloses the
/// omitted terms. This oracle uses neither atan nor the half-angle formula.
fn asin_series(point: &BigRational) -> (BigRational, BigRational) {
    assert!(point.abs() < BigRational::one());
    if point.is_negative() {
        let (lower, upper) = asin_series(&(-point.clone()));
        return (-upper, -lower);
    }
    let square = point * point;
    let mut term = point.clone();
    let mut sum = BigRational::zero();
    for n in 0..192 {
        sum += &term;
        term = term * &square * q((2 * n + 1) * (2 * n + 1), 2 * (n + 1) * (2 * n + 3));
    }
    let tail = term / (BigRational::one() - square);
    (sum.clone(), sum + tail)
}

fn encloses(ball: &RealBall, lower: &BigRational, upper: &BigRational) {
    assert!(ball.lower() <= *lower && ball.upper() >= *upper, "{ball}");
}

fn half_pi() -> Expr {
    Expr::Mul(vec![Expr::Rational(q(1, 2)), Expr::Const(Constant::Pi)])
}

#[test]
fn point_values_agree_with_independent_exact_series_on_both_principal_branches() {
    let pi = RealBall::pi(40)
        .unwrap()
        .div(&RealBall::from_i64(2))
        .unwrap();
    let target = q(1, 1_000_000_000_000_000_000);
    for point in [
        q(-4, 5),
        q(-3, 5),
        q(-1, 5),
        q(0, 1),
        q(1, 5),
        q(3, 5),
        q(4, 5),
    ] {
        let (lower, upper) = asin_series(&point);
        let argument = RealBall::exact(point);
        let sine_inverse = argument.asin(18).unwrap();
        encloses(&sine_inverse, &lower, &upper);
        assert!(sine_inverse.radius() <= &target);
        let cosine_inverse = argument.acos(18).unwrap();
        encloses(
            &cosine_inverse,
            &(pi.lower() - &upper),
            &(pi.upper() - &lower),
        );
        assert!(cosine_inverse.radius() <= &target);
    }
}

#[test]
fn wide_signed_intervals_keep_the_tight_monotone_endpoint_ranges() {
    let argument = interval(q(-4, 5), q(3, 5));
    let (left_lower, _) = asin_series(&q(-4, 5));
    let (_, right_upper) = asin_series(&q(3, 5));
    let tolerance = q(1, 1_000_000_000_000_000);
    let result = argument.asin(18).unwrap();
    assert!(result.lower() <= left_lower);
    assert!(result.upper() >= right_upper);
    assert!(result.lower() >= &left_lower - &tolerance);
    assert!(result.upper() <= &right_upper + &tolerance);
    assert!(result.contains(&q(0, 1)));

    let half = RealBall::pi(40)
        .unwrap()
        .div(&RealBall::from_i64(2))
        .unwrap();
    let result = argument.acos(18).unwrap();
    assert!(result.lower() <= half.lower() - &right_upper);
    assert!(result.upper() >= half.upper() - &left_lower);
    assert!(result.lower() >= half.lower() - right_upper - &tolerance);
    assert!(result.upper() <= half.upper() - left_lower + tolerance);
}

#[test]
fn closed_value_endpoints_keep_the_principal_values_without_division_by_zero() {
    let pi = RealBall::pi(40).unwrap();
    let half = pi.div(&RealBall::from_i64(2)).unwrap();
    assert_eq!(
        RealBall::from_i64(0).asin(18).unwrap(),
        RealBall::from_i64(0)
    );
    assert_eq!(
        RealBall::from_i64(1).acos(18).unwrap(),
        RealBall::from_i64(0)
    );
    encloses(
        &RealBall::from_i64(1).asin(18).unwrap(),
        &half.lower(),
        &half.upper(),
    );
    encloses(
        &RealBall::from_i64(-1).asin(18).unwrap(),
        &(-half.upper()),
        &(-half.lower()),
    );
    encloses(
        &RealBall::from_i64(0).acos(18).unwrap(),
        &half.lower(),
        &half.upper(),
    );
    encloses(
        &RealBall::from_i64(-1).acos(18).unwrap(),
        &pi.lower(),
        &pi.upper(),
    );
    let complete_domain = interval(q(-1, 1), q(1, 1));
    let result = complete_domain.asin(18).unwrap();
    assert!(result.lower() <= -half.upper());
    assert!(result.upper() >= half.upper());
    let result = complete_domain.acos(18).unwrap();
    assert!(result.contains(&q(0, 1)));
    assert!(result.upper() >= pi.upper());
}

#[test]
fn exact_tiny_outside_points_and_precision_overflow_are_never_clipped() {
    let epsilon = BigRational::new(BigInt::from(1), BigInt::from(1) << 120u32);
    for argument in [
        RealBall::exact(q(1, 1) + &epsilon),
        RealBall::exact(q(-1, 1) - &epsilon),
        RealBall::new(q(0, 1), q(1, 1) + &epsilon).unwrap(),
        RealBall::from_i64(2),
        RealBall::from_i64(-2),
    ] {
        assert!(matches!(
            argument.asin(18),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
        assert!(matches!(
            argument.acos(18),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
    }
    for precision in [
        0,
        RealBall::MAX_INVERSE_CIRCULAR_PRECISION_DIGITS + 1,
        u32::MAX,
    ] {
        assert!(matches!(
            RealBall::from_i64(0).asin(precision),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
        assert!(matches!(
            RealBall::from_i64(1).acos(precision),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
    }
    // The convenience kernel must reject excessive rational work before
    // squaring its huge denominator, even though this tiny point is in-domain.
    let tiny = RealBall::exact(BigRational::new(
        BigInt::from(1),
        BigInt::from(1) << 140_000u32,
    ));
    assert!(matches!(
        tiny.asin(18),
        Err(BallError::ResourceLimitExceeded { .. })
    ));
    assert!(matches!(
        tiny.acos(18),
        Err(BallError::ResourceLimitExceeded { .. })
    ));
}

#[test]
fn c1_and_c2_enclosures_keep_exact_derivative_signs_at_rational_anchors() {
    let variables = [Symbol::new("x")];
    for point_sign in [-1, 1] {
        let domain = [RealBall::exact(q(3 * point_sign, 5))];
        for (name, function_sign) in [("asin", 1), ("acos", -1)] {
            let source = function(name, Expr::Sym(variables[0].clone()));
            let first = q(5 * function_sign, 4);
            let second = q(75 * point_sign * function_sign, 64);
            let c1 =
                enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 18)
                    .unwrap();
            assert!(c1.jacobian()[0][0].contains(&first));
            assert!(!c1.jacobian()[0][0].contains(&(-first.clone())));
            assert_eq!(c1.residuals(), std::slice::from_ref(&source));
            assert_eq!(c1.variables(), variables);
            assert_eq!(c1.domain(), domain);
            let c2 = enclose_objective(&source, &variables, &domain, 18).unwrap();
            assert!(c2.gradient()[0].contains(&first));
            assert!(c2.hessian()[0][0].contains(&second));
            assert!(!c2.hessian()[0][0].contains(&(-second)));
            assert_eq!(c2.objective(), &source);
        }
    }
    let source = function("atan", Expr::Sym(variables[0].clone()));
    let c2 = enclose_objective(&source, &variables, &[RealBall::from_i64(1)], 18).unwrap();
    assert!(c2.gradient()[0].contains(&q(1, 2)));
    assert!(c2.hessian()[0][0].contains(&q(-1, 2)));
}

#[test]
fn nonlinear_chain_keeps_both_first_and_second_derivative_terms() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let inner = Expr::Add(vec![
        Expr::Rational(q(3, 5)),
        Expr::Mul(vec![Expr::from_i64(2), x.clone()]),
        Expr::Mul(vec![Expr::from_i64(3), power(x, 2)]),
    ]);
    for (name, sign) in [("asin", 1), ("acos", -1)] {
        let source = function(name, inner.clone());
        let result = enclose_objective(&source, &variables, &[RealBall::from_i64(0)], 18).unwrap();
        assert!(result.gradient()[0].contains(&q(5 * sign, 2)));
        // f''(3/5) * 2^2 + f'(3/5) * 6 = +/-195/16.
        assert!(result.hessian()[0][0].contains(&q(195 * sign, 16)));
        assert!(!result.hessian()[0][0].contains(&q(75 * sign, 16)));
        assert!(!result.hessian()[0][0].contains(&q(15 * sign, 2)));
    }
}

#[test]
fn mixed_partials_follow_every_coordinate_chain_factor() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let inner = Expr::Add(vec![
        Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(variables[0].clone())]),
        Expr::Sym(variables[1].clone()),
    ]);
    let domain = [RealBall::exact(q(1, 5)), RealBall::exact(q(1, 5))];
    for (name, sign) in [("asin", 1), ("acos", -1)] {
        let result =
            enclose_objective(&function(name, inner.clone()), &variables, &domain, 18).unwrap();
        assert!(result.gradient()[0].contains(&q(5 * sign, 2)));
        assert!(result.gradient()[1].contains(&q(5 * sign, 4)));
        for (row, column, numerator, denominator) in [
            (0, 0, 75, 16),
            (0, 1, 75, 32),
            (1, 0, 75, 32),
            (1, 1, 75, 64),
        ] {
            assert!(result.hessian()[row][column].contains(&q(numerator * sign, denominator)));
        }
    }
}

#[test]
fn derivative_domains_reject_endpoints_whole_boxes_and_held_zero_factors() {
    let variables = [Symbol::new("x")];
    let epsilon = BigRational::new(BigInt::from(1), BigInt::from(1) << 120u32);
    for name in ["asin", "acos"] {
        let inverse = function(name, Expr::Sym(variables[0].clone()));
        let held = Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]);
        for domain in [
            RealBall::from_i64(-1),
            RealBall::from_i64(1),
            interval(q(-1, 1), q(1, 1)),
            interval(q(-2, 1), q(-1, 1)),
            interval(q(1, 1), q(2, 1)),
            RealBall::new(q(1, 1), epsilon.clone()).unwrap(),
        ] {
            for source in [&inverse, &held] {
                assert!(matches!(
                    enclose_residual_jacobian(
                        std::slice::from_ref(source),
                        &variables,
                        std::slice::from_ref(&domain),
                        18,
                    ),
                    Err(RootBoxError::Domain(_))
                ));
                assert!(matches!(
                    enclose_objective(source, &variables, std::slice::from_ref(&domain), 18),
                    Err(RootBoxError::Domain(_))
                ));
            }
        }
        // This whole interval is strictly interior, although binary64 would
        // round its endpoints to one. Exact domain admission must retain it.
        let domain = [RealBall::new(q(1, 1) - &epsilon, &epsilon / q(2, 1)).unwrap()];
        let c1 = enclose_residual_jacobian(std::slice::from_ref(&inverse), &variables, &domain, 24)
            .unwrap();
        let c2 = enclose_objective(&inverse, &variables, &domain, 24).unwrap();
        if name == "asin" {
            assert!(c1.jacobian()[0][0].lower() > q(1, 1));
            assert!(c2.hessian()[0][0].lower() > q(0, 1));
        } else {
            assert!(c1.jacobian()[0][0].upper() < q(-1, 1));
            assert!(c2.hessian()[0][0].upper() < q(0, 1));
        }
    }
}

fn root_source(name: &str, variable: &Symbol) -> Expr {
    let inverse = function(name, Expr::Sym(variable.clone()));
    if name == "acos" {
        inverse - half_pi()
    } else {
        inverse
    }
}

fn witness(preconditioner: BigRational, precision_digits: u32) -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(q(0, 1), q(1, 20)).unwrap()],
        center: vec![q(0, 1)],
        preconditioner: vec![vec![preconditioner]],
        precision_digits,
    }
}

#[test]
fn root_receipts_bind_the_original_principal_source_and_reject_wrong_witnesses() {
    let variables = [Symbol::new("x")];
    for (name, sign) in [("asin", 1), ("acos", -1)] {
        let source = vec![root_source(name, &variables[0])];
        let proposal = witness(q(sign, 1), 18);
        let receipt = certify_root_box(&source, &variables, &proposal).unwrap();
        assert_eq!(receipt.residuals(), source);
        assert_eq!(receipt.variables(), variables);
        assert_eq!(receipt.witness(), &proposal);
        assert!(receipt.root_enclosure()[0].contains(&q(0, 1)));
        assert!(receipt.root_enclosure()[0].lower() > proposal.domain[0].lower());
        assert!(receipt.root_enclosure()[0].upper() < proposal.domain[0].upper());
        assert!(receipt.contraction_bound() < &q(1, 1));

        let shifted = [source[0].clone() + Expr::from_i64(1)];
        assert!(certify_root_box(&shifted, &variables, &proposal).is_err());
        let mut wrong_sign = proposal.clone();
        wrong_sign.preconditioner[0][0] = q(-sign, 1);
        assert!(matches!(
            certify_root_box(&source, &variables, &wrong_sign),
            Err(RootBoxError::NotContractive)
        ));
        assert_eq!(receipt.residuals(), source);
    }
}

#[test]
fn squared_arcsine_has_a_source_checked_strict_minimum() {
    let variables = [Symbol::new("x")];
    let source = power(function("asin", Expr::Sym(variables[0].clone())), 2);
    let proposal = witness(q(1, 2), 18);
    let receipt = certify_stationary_point(&source, &variables, &proposal).unwrap();
    assert_eq!(receipt.kind(), StationaryPointKind::StrictMinimum);
    assert_eq!(receipt.objective(), &source);
    assert_eq!(receipt.variables(), variables);
    assert_eq!(receipt.witness(), &proposal);
    assert!(receipt.stationary_enclosure()[0].contains(&q(0, 1)));
    assert!(receipt.objective_enclosure().contains(&q(0, 1)));
    assert!(receipt.hessian_enclosure()[0][0].lower() > q(0, 1));
    assert!(receipt.contraction_bound() < &q(1, 1));
    let shifted = source.clone() + Expr::Sym(variables[0].clone());
    assert!(certify_stationary_point(&shifted, &variables, &proposal).is_err());
}

#[derive(Clone, Copy, Debug)]
enum Lane {
    First,
    Second,
    Root,
    Stationary,
}

fn metered_case(lane: Lane, name: &str, meter: &mut impl BudgetMeter) -> Result<(), RootBoxError> {
    let variables = [Symbol::new("x")];
    let source = function(name, Expr::Sym(variables[0].clone()));
    let domain = [RealBall::exact(q(3, 5))];
    match lane {
        Lane::First => {
            metered_enclose_residual_jacobian(&[source], &variables, &domain, 8, meter).map(|_| ())
        }
        Lane::Second => {
            metered_enclose_objective(&source, &variables, &domain, 8, meter).map(|_| ())
        }
        Lane::Root => verify_root_box(
            &[root_source(name, &variables[0])],
            &variables,
            &witness(q(if name == "acos" { -1 } else { 1 }, 1), 8),
            meter,
        )
        .map(|_| ()),
        Lane::Stationary => {
            verify_stationary_point(&power(source, 2), &variables, &witness(q(1, 2), 8), meter)
                .map(|_| ())
        }
    }
}

const METER_CASES: [(Lane, &str); 6] = [
    (Lane::First, "asin"),
    (Lane::First, "acos"),
    (Lane::Second, "asin"),
    (Lane::Second, "acos"),
    (Lane::Root, "asin"),
    (Lane::Stationary, "asin"),
];
const LIMIT: u64 = 1_000_000_000_000;
const VERIFIER_RESERVE: u64 = 37;

struct Checkpoints {
    visits: usize,
    stop: Option<usize>,
    stopped: bool,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after a terminal cancellation");
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after a terminal cancellation");
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "continued after a terminal cancellation");
        self.visits += 1;
        if self.stop == Some(self.visits) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_inverse_circular_safe_point_refuses_without_publishing_a_receipt() {
    for (lane, name) in METER_CASES {
        let mut count = Checkpoints {
            visits: 0,
            stop: None,
            stopped: false,
        };
        metered_case(lane, name, &mut count).unwrap();
        assert!(count.visits > 0);
        for stop in 1..=count.visits {
            let mut meter = Checkpoints {
                visits: 0,
                stop: Some(stop),
                stopped: false,
            };
            assert_eq!(
                metered_case(lane, name, &mut meter),
                Err(RootBoxError::Meter(MeterError::Cancelled)),
                "{lane:?} {name}, checkpoint {stop}"
            );
            assert!(meter.stopped);
            assert_eq!(meter.visits, stop);
        }
    }
}

#[test]
fn cumulative_resource_limits_include_the_last_unit_and_protect_the_verifier_reserve() {
    let dimensions = [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ];
    for (lane, name) in METER_CASES {
        let mut measured = Budget::new(BudgetLimits::uniform(LIMIT, VERIFIER_RESERVE));
        metered_case(lane, name, &mut measured).unwrap();
        assert_eq!(measured.verifier_remaining(), VERIFIER_RESERVE);
        for dimension in dimensions {
            let used = LIMIT - measured.remaining(dimension);
            assert!(used > 0);
            let mut limits = BudgetLimits::uniform(LIMIT, VERIFIER_RESERVE);
            limits.dimensions[dimension.index()] = used;
            let mut exact = Budget::new(limits);
            metered_case(lane, name, &mut exact).unwrap();
            assert_eq!(exact.remaining(dimension), 0);
            let after_first = exact.snapshot();
            assert!(matches!(
                metered_case(lane, name, &mut exact),
                Err(RootBoxError::Meter(MeterError::Budget(BudgetError::Exhausted {
                    dimension: exhausted, ..
                }))) if exhausted == dimension
            ));
            assert_eq!(exact.snapshot(), after_first);
            assert_eq!(exact.verifier_remaining(), VERIFIER_RESERVE);

            limits.dimensions[dimension.index()] = used - 1;
            let mut short = Budget::new(limits);
            assert!(matches!(
                metered_case(lane, name, &mut short),
                Err(RootBoxError::Meter(MeterError::Budget(BudgetError::Exhausted {
                    dimension: exhausted, ..
                }))) if exhausted == dimension
            ));
            assert_eq!(short.verifier_remaining(), VERIFIER_RESERVE);
        }
    }
}
