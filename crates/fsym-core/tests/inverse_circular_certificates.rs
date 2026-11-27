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

/// Integer bisection supplies a rational square-root bracket independent of
/// the production ball square root. The fixtures below have 0 <= x <= 4.
fn reference_sqrt(value: &BigRational) -> (BigRational, BigRational) {
    assert!(value >= &q(0, 1) && value <= &q(4, 1));
    let scale = BigInt::one() << 192u32;
    let target = value.numer() * &scale * &scale;
    let mut lower = BigInt::zero();
    let mut upper = &scale * BigInt::from(2);
    while &upper - &lower > BigInt::one() {
        let middle = (&lower + &upper) / BigInt::from(2);
        if &middle * &middle * value.denom() <= target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    (
        BigRational::new(lower, scale.clone()),
        BigRational::new(upper, scale),
    )
}

/// For x >= 1, log(x) = 2 sum(t^(2k+1)/(2k+1)), t=(x-1)/(x+1).
/// Directed fixed-point integer arithmetic bounds every retained term, and
/// the geometric tail bounds all omitted terms. No production logarithm,
/// inverse function, or floating approximation supplies these bounds.
fn reference_log(value: &BigRational) -> (BigRational, BigRational) {
    assert!(value >= &q(1, 1));
    let scale = BigInt::one() << 192u32;
    let ratio = (value - q(1, 1)) / (value + q(1, 1));
    let ceil = |numerator: BigInt, denominator: &BigInt| {
        (numerator + denominator - BigInt::one()) / denominator
    };
    let scaled = ratio.numer() * &scale;
    let mut lower_power = &scaled / ratio.denom();
    let mut upper_power = ceil(scaled, ratio.denom());
    let lower_square = &lower_power * &lower_power / &scale;
    let upper_square = ceil(&upper_power * &upper_power, &scale);
    assert!(upper_square < scale);
    let mut lower_sum = BigInt::zero();
    let mut upper_sum = BigInt::zero();
    for index in 0..128 {
        let denominator = BigInt::from(2 * index + 1);
        lower_sum += &lower_power / &denominator;
        upper_sum += ceil(upper_power.clone(), &denominator);
        lower_power = lower_power * &lower_square / &scale;
        upper_power = ceil(upper_power * &upper_square, &scale);
    }
    upper_sum += ceil(
        upper_power * &scale,
        &(BigInt::from(257) * (&scale - upper_square)),
    );
    (
        BigRational::new(BigInt::from(2) * lower_sum, scale.clone()),
        BigRational::new(BigInt::from(2) * upper_sum, scale),
    )
}

fn reference_acosh(value: &BigRational) -> (BigRational, BigRational) {
    let (lower, upper) = reference_sqrt(&(value * value - q(1, 1)));
    (
        reference_log(&(value + lower)).0,
        reference_log(&(value + upper)).1,
    )
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
fn held_inverse_circular_endpoints_reach_every_ground_ball_entry_point() {
    let pi = RealBall::pi(60).unwrap();
    let tolerance = BigRational::new(BigInt::from(1), BigInt::from(10).pow(24));
    for (name, point, multiple) in [
        ("asin", -1, q(-1, 2)),
        ("asin", 0, q(0, 1)),
        ("asin", 1, q(1, 2)),
        ("acos", -1, q(1, 1)),
        ("acos", 0, q(1, 2)),
        ("acos", 1, q(0, 1)),
    ] {
        // Raw Function nodes deliberately bypass symbolic special-value folding.
        let source = function(name, Expr::from_i64(point));
        let expected = pi.mul(&RealBall::exact(multiple));
        let complex = source.eval_complex_ball(24).unwrap();
        assert_eq!(complex.imag(), &RealBall::from_i64(0));
        for actual in [
            source.evalf_ball(24).unwrap(),
            source.eval_real_ball(24).unwrap(),
            complex.real().clone(),
        ] {
            assert!(actual.contains_ball(&expected), "{name}({point}): {actual}");
            assert!(actual.radius() <= &tolerance);
        }
    }
    assert_eq!(
        function("asin", Expr::from_i64(0))
            .evalf_decimal(18)
            .unwrap(),
        "0"
    );
    assert_eq!(
        function("acos", Expr::from_i64(1))
            .evalf_decimal(18)
            .unwrap(),
        "0"
    );
    assert_eq!(
        function("asin", Expr::from_i64(1))
            .evalf_decimal(18)
            .unwrap(),
        "1.57079632679489662e0"
    );
    assert_eq!(
        function("acos", Expr::from_i64(-1))
            .evalf_decimal(18)
            .unwrap(),
        "3.14159265358979324e0"
    );
}

#[test]
fn scalar_inverse_circular_dispatch_matches_independent_interior_series() {
    let half = RealBall::pi(60)
        .unwrap()
        .div(&RealBall::from_i64(2))
        .unwrap();
    for point in [q(-4, 5), q(-1, 5), q(1, 5), q(4, 5)] {
        let (lower, upper) = asin_series(&point);
        let argument = Expr::Rational(point);
        let sine = function("asin", argument.clone()).evalf_ball(24).unwrap();
        encloses(&sine, &lower, &upper);
        let cosine = function("acos", argument).evalf_ball(24).unwrap();
        encloses(&cosine, &(half.lower() - &upper), &(half.upper() - &lower));
    }
}

#[test]
fn exact_points_near_both_endpoints_remain_distinct_from_the_endpoint() {
    // t = 2^-40, x = 1 - 2*t^2. Binary64 rounds x to 1, while
    // acos(x) = 2*asin(t) is positive. The positive arcsine series gives
    // t <= asin(t) <= t/(1-t^2), independently of the atan implementation.
    let t = BigRational::new(BigInt::from(1), BigInt::from(1) << 40u32);
    let square = &t * &t;
    let point = q(1, 1) - q(2, 1) * &square;
    let angle_lower = q(2, 1) * &t;
    let angle_upper = &angle_lower / (q(1, 1) - square);
    let pi = RealBall::pi(60).unwrap();
    let half = pi.div(&RealBall::from_i64(2)).unwrap();
    for (point, sine_lower, sine_upper, cosine_lower, cosine_upper) in [
        (
            point.clone(),
            half.lower() - &angle_upper,
            half.upper() - &angle_lower,
            angle_lower.clone(),
            angle_upper.clone(),
        ),
        (
            -point,
            -half.upper() + &angle_lower,
            -half.lower() + &angle_upper,
            pi.lower() - &angle_upper,
            pi.upper() - &angle_lower,
        ),
    ] {
        let argument = Expr::Rational(point);
        let sine = function("asin", argument.clone()).evalf_ball(24).unwrap();
        let cosine = function("acos", argument).evalf_ball(24).unwrap();
        encloses(&sine, &sine_lower, &sine_upper);
        encloses(&cosine, &cosine_lower, &cosine_upper);
        assert!(sine.lower() > -half.lower());
        assert!(sine.upper() < half.lower());
        assert!(cosine.lower() > q(0, 1));
        assert!(cosine.upper() < pi.lower());
    }
}

#[test]
fn held_endpoint_evaluation_composes_without_changing_principal_values() {
    for (source, expected) in [
        (
            function("sin", function("asin", Expr::from_i64(-1))),
            q(-1, 1),
        ),
        (
            function("sin", function("asin", Expr::from_i64(1))),
            q(1, 1),
        ),
        (
            function("cos", function("acos", Expr::from_i64(-1))),
            q(-1, 1),
        ),
        (
            function("asin", function("acos", Expr::from_i64(1))),
            q(0, 1),
        ),
    ] {
        assert!(
            source.evalf_ball(24).unwrap().contains(&expected),
            "{source}"
        );
        let complex = source.eval_complex_ball(24).unwrap();
        assert!(complex.contains(&expected, &q(0, 1)), "{source}");
    }
}

#[test]
fn scalar_inverse_circular_endpoints_keep_original_domain_and_arity_refusals() {
    let epsilon = BigRational::new(BigInt::from(1), BigInt::from(1) << 120u32);
    let held_pole = Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]);
    for name in ["asin", "acos"] {
        for argument in [
            Expr::Rational(q(1, 1) + &epsilon),
            Expr::Rational(q(-1, 1) - &epsilon),
            Expr::from_i64(2),
            Expr::from_i64(-2),
            Expr::Const(Constant::I),
            Expr::Const(Constant::Infinity),
            Expr::Sym(Symbol::new("inverse_circular_unbound")),
            Expr::Add(vec![Expr::from_i64(1), held_pole.clone()]),
        ] {
            let source = function(name, argument);
            assert!(source.evalf_ball(24).is_err(), "{source}");
        }
        // Principal complex inverses now admit finite nonreal results. They
        // must still reject every original undefined or unbound source.
        for argument in [
            Expr::Const(Constant::Infinity),
            Expr::Sym(Symbol::new("inverse_circular_unbound")),
            Expr::Add(vec![Expr::from_i64(1), held_pole.clone()]),
        ] {
            let source = function(name, argument);
            assert!(source.eval_complex_ball(24).is_err(), "{source}");
        }
        for arguments in [vec![], vec![Expr::from_i64(0), Expr::from_i64(1)]] {
            let source = Expr::Function(name.into(), arguments);
            assert!(source.evalf_ball(24).is_err());
            assert!(source.eval_complex_ball(24).is_err());
        }
        for precision in [
            0,
            RealBall::MAX_INVERSE_CIRCULAR_PRECISION_DIGITS + 1,
            u32::MAX,
        ] {
            assert!(
                function(name, Expr::from_i64(1))
                    .evalf_ball(precision)
                    .is_err()
            );
        }
    }
}

#[test]
fn scalar_complex_inverse_branches_enclose_independent_values_without_chopping() {
    let epsilon = BigRational::new(BigInt::one(), BigInt::one() << 120u32);
    let near = q(1, 1) + epsilon;
    let near_magnitude = reference_acosh(&near);
    let far_magnitude = reference_acosh(&q(2, 1));
    let pi = RealBall::pi(60).unwrap();
    let half = pi.div(&RealBall::from_i64(2)).unwrap();
    let zero = q(0, 1);

    for name in ["asin", "acos"] {
        for (point, (lower, upper)) in [(&near, &near_magnitude), (&q(2, 1), &far_magnitude)] {
            for sign in [-1, 1] {
                let argument = Expr::Rational(point * q(sign, 1));
                let source = function(name, argument);
                let value = source.eval_complex_ball(24).unwrap();
                let real = match (name, sign) {
                    ("asin", -1) => half.neg(),
                    ("asin", _) => half.clone(),
                    ("acos", -1) => pi.clone(),
                    _ => RealBall::from_i64(0),
                };
                encloses(value.real(), &real.lower(), &real.upper());
                let positive_imaginary = (name == "acos") == (sign == 1);
                if positive_imaginary {
                    encloses(value.imag(), lower, upper);
                    assert!(value.imag().lower() > zero, "{source}");
                } else {
                    encloses(value.imag(), &(-upper), &(-lower));
                    assert!(value.imag().upper() < zero, "{source}");
                }
                if name == "acos" && sign == 1 {
                    assert_eq!(value.real(), &RealBall::from_i64(0));
                }
            }
        }
    }

    // asin(i)=i log(1+sqrt(2)); acos(i)=pi/2-i log(1+sqrt(2)).
    let (root_lower, root_upper) = reference_sqrt(&q(2, 1));
    let lower = reference_log(&(q(1, 1) + root_lower)).0;
    let upper = reference_log(&(q(1, 1) + root_upper)).1;
    let sine = function("asin", Expr::Const(Constant::I))
        .eval_complex_ball(24)
        .unwrap();
    assert_eq!(sine.real(), &RealBall::from_i64(0));
    encloses(sine.imag(), &lower, &upper);
    assert!(sine.imag().lower() > zero);
    let cosine = function("acos", Expr::Const(Constant::I))
        .eval_complex_ball(24)
        .unwrap();
    encloses(cosine.real(), &half.lower(), &half.upper());
    encloses(cosine.imag(), &(-upper), &(-lower));
    assert!(cosine.imag().upper() < zero);
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
