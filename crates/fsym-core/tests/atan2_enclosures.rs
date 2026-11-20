//! Principal-angle enclosures checked against exact unrounded series and
//! analytic rectangle extrema, with origin/cut and work-limit adversaries.

use fsym_core::{BallError, BigInt, BigRational, Expr, RealBall};
use num_traits::Zero;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn decimal_unit(digits: u32) -> BigRational {
    BigRational::new(BigInt::from(1), BigInt::from(10).pow(digits))
}

/// The exact alternating partial sum and its successor enclose atan(t).
/// Unlike the production kernel, no term, power or sum is rounded.
fn atan_reference(t: &BigRational) -> RealBall {
    assert!(*t >= q(0, 1) && *t <= q(1, 2));
    let square = t * t;
    let mut power = t.clone();
    let mut sum = BigRational::zero();
    for index in 0..128 {
        let term = &power / q(2 * index + 1, 1);
        if index % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
        power *= &square;
    }
    let next = &sum + power / q(257, 1);
    interval(sum, next)
}

fn pi_reference() -> RealBall {
    atan_reference(&q(1, 5))
        .mul(&RealBall::from_i64(16))
        .sub(&atan_reference(&q(1, 239)).mul(&RealBall::from_i64(4)))
}

fn enclosed_tightly(actual: &RealBall, expected: &RealBall) {
    assert!(actual.contains_ball(expected), "{actual} misses {expected}");
    let tolerance = decimal_unit(20);
    assert!(actual.lower() >= expected.lower() - &tolerance);
    assert!(actual.upper() <= expected.upper() + tolerance);
}

#[test]
fn exact_axis_values_preserve_argument_order_and_principal_negative_axis() {
    let zero = RealBall::from_i64(0);
    let positive = interval(q(1, 7), q(11, 3));
    let negative = positive.neg();
    let pi = pi_reference();
    let half_pi = pi.div(&RealBall::from_i64(2)).unwrap();
    assert_eq!(zero.atan2(&positive, 24).unwrap(), zero);
    enclosed_tightly(&zero.atan2(&negative, 24).unwrap(), &pi);
    enclosed_tightly(&positive.atan2(&zero, 24).unwrap(), &half_pi);
    enclosed_tightly(&negative.atan2(&zero, 24).unwrap(), &half_pi.neg());
}

#[test]
fn all_four_quadrants_and_reciprocal_ratios_enclose_independent_series() {
    let pi = pi_reference();
    let half_pi = pi.div(&RealBall::from_i64(2)).unwrap();
    for (small, large) in [(1, 2), (1, 3), (2, 5)] {
        let acute = atan_reference(&q(small, large));
        for (y, x, angle) in [
            (small, large, acute.clone()),
            (large, small, half_pi.sub(&acute)),
        ] {
            for (ysign, xsign, expected) in [
                (1, 1, angle.clone()),
                (-1, 1, angle.neg()),
                (1, -1, pi.sub(&angle)),
                (-1, -1, angle.sub(&pi)),
            ] {
                let actual = RealBall::from_i64(y * ysign)
                    .atan2(&RealBall::from_i64(x * xsign), 24)
                    .unwrap();
                enclosed_tightly(&actual, &expected);
            }
        }
    }
    let quarter_pi = pi.div(&RealBall::from_i64(4)).unwrap();
    for (y, x, multiplier) in [(1, 1, 1), (1, -1, 3), (-1, 1, -1), (-1, -1, -3)] {
        enclosed_tightly(
            &RealBall::from_i64(y)
                .atan2(&RealBall::from_i64(x), 24)
                .unwrap(),
            &quarter_pi.mul(&RealBall::from_i64(multiplier)),
        );
    }
}

#[test]
fn continuous_rectangles_keep_full_corner_ranges_across_both_axes() {
    let pi = pi_reference();
    let low = atan_reference(&q(1, 2));
    let high = pi.div(&RealBall::from_i64(2)).unwrap().sub(&low);
    let positive = interval(q(1, 1), q(2, 1));
    let crossing = interval(q(-2, 1), q(2, 1));
    for (y, x, lower, upper) in [
        (
            positive.clone(),
            positive.clone(),
            low.lower(),
            high.upper(),
        ),
        (
            positive.clone(),
            crossing.clone(),
            low.lower(),
            pi.upper() - low.lower(),
        ),
        (
            positive.neg(),
            crossing.clone(),
            low.lower() - pi.upper(),
            -low.lower(),
        ),
        (crossing, positive.clone(), -high.upper(), high.upper()),
        (
            positive.clone(),
            positive.neg(),
            pi.lower() - high.upper(),
            pi.upper() - low.lower(),
        ),
    ] {
        let actual = y.atan2(&x, 24).unwrap();
        enclosed_tightly(&actual, &interval(lower, upper));
    }
}

#[test]
fn exact_axis_edges_and_degenerate_rectangles_keep_their_complete_ranges() {
    let pi = pi_reference();
    let half_pi = pi.div(&RealBall::from_i64(2)).unwrap();
    let quarter_pi = pi.div(&RealBall::from_i64(4)).unwrap();
    let positive = interval(q(1, 1), q(2, 1));
    let nonnegative = interval(q(0, 1), q(1, 1));
    enclosed_tightly(
        &nonnegative.atan2(&positive, 24).unwrap(),
        &interval(q(0, 1), quarter_pi.upper()),
    );
    enclosed_tightly(
        &positive.atan2(&nonnegative, 24).unwrap(),
        &interval(quarter_pi.lower(), half_pi.upper()),
    );
    enclosed_tightly(
        &positive.atan2(&RealBall::from_i64(1), 24).unwrap(),
        &interval(
            quarter_pi.lower(),
            half_pi.upper() - atan_reference(&q(1, 2)).lower(),
        ),
    );
    enclosed_tightly(
        &RealBall::from_i64(1).atan2(&positive, 24).unwrap(),
        &interval(atan_reference(&q(1, 2)).lower(), quarter_pi.upper()),
    );
}

#[test]
fn rectangles_containing_or_merely_touching_the_origin_refuse() {
    let zero = RealBall::from_i64(0);
    let crossing = interval(q(-1, 1), q(1, 1));
    let right = interval(q(0, 1), q(1, 1));
    let left = right.neg();
    for y in [&zero, &crossing, &right, &left] {
        for x in [&zero, &crossing, &right, &left] {
            assert!(matches!(
                y.atan2(x, 24),
                Err(BallError::ArgumentOutsideDeclaredEnvelope(message))
                    if message.contains("origin")
            ));
        }
    }
}

#[test]
fn negative_axis_crossings_and_one_sided_touching_boxes_refuse_without_clipping() {
    let x = interval(q(-2, 1), q(-1, 1));
    for y in [
        interval(q(-1, 1), q(1, 1)),
        interval(q(0, 1), q(1, 1)),
        interval(q(-1, 1), q(0, 1)),
    ] {
        assert!(matches!(
            y.atan2(&x, 24),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(message))
                if message.contains("branch cut")
        ));
    }
    // A tiny exact sign cannot be rounded to zero to choose another branch.
    let tiny = decimal_unit(80);
    let above = interval(tiny.clone(), &tiny * q(2, 1));
    let upper = above.atan2(&x, 24).unwrap();
    let lower = above.neg().atan2(&x, 24).unwrap();
    assert!(upper.lower() > q(3, 1));
    assert!(lower.upper() < q(-3, 1));
    assert_eq!(upper.neg(), lower);
}

#[test]
fn exact_positive_rescaling_is_invariant_beyond_binary_float_ranges() {
    let x = interval(q(-7, 3), q(-2, 5));
    let y = interval(q(1, 7), q(13, 2));
    let expected = y.atan2(&x, 24).unwrap();
    let huge = BigInt::from(10).pow(1000);
    for factor in [
        BigRational::from_integer(huge.clone()),
        BigRational::new(BigInt::from(1), huge),
    ] {
        let factor = RealBall::exact(factor);
        assert_eq!(y.mul(&factor).atan2(&x.mul(&factor), 24).unwrap(), expected,);
    }
}

#[test]
fn declared_precision_and_rational_work_limits_are_checked_before_corner_work() {
    let zero = RealBall::from_i64(0);
    let one = RealBall::from_i64(1);
    for precision in [0, RealBall::MAX_ATAN2_PRECISION_DIGITS + 1, u32::MAX] {
        assert!(matches!(
            zero.atan2(&one, precision),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
    }
    assert_eq!(
        zero.atan2(&one, RealBall::MAX_ATAN2_PRECISION_DIGITS)
            .unwrap(),
        zero,
    );
    let oversized = RealBall::exact(BigRational::new(
        BigInt::from(1),
        BigInt::from(1) << 140_000u32,
    ));
    assert!(matches!(
        oversized.atan2(&one, 24),
        Err(BallError::ResourceLimitExceeded { .. })
    ));
    assert!(matches!(
        one.atan2(&oversized, 24),
        Err(BallError::ResourceLimitExceeded { .. })
    ));
}

#[test]
fn ground_expression_dispatch_keeps_quadrants_arity_and_both_original_domains() {
    let pi = pi_reference();
    let quarter = pi.div(&RealBall::from_i64(4)).unwrap();
    for (y, x, multiplier) in [(1, 1, 1), (1, -1, 3), (-1, 1, -1), (-1, -1, -3)] {
        let source = Expr::Function("atan2".into(), vec![Expr::from_i64(y), Expr::from_i64(x)]);
        let actual = source.evalf_ball(24).unwrap();
        enclosed_tightly(&actual, &quarter.mul(&RealBall::from_i64(multiplier)));
        assert!(actual.radius() <= &decimal_unit(24));
    }
    let negative_axis = Expr::Function("atan2".into(), vec![Expr::from_i64(0), Expr::from_i64(-1)]);
    enclosed_tightly(&negative_axis.evalf_ball(24).unwrap(), &pi);
    let origin = Expr::Function("atan2".into(), vec![Expr::from_i64(0), Expr::from_i64(0)]);
    assert!(origin.evalf_ball(24).is_err());
    for arity in [0, 1, 3] {
        let source = Expr::Function("atan2".into(), vec![Expr::from_i64(1); arity]);
        assert!(source.evalf_ball(24).is_err());
    }
    let undefined = Expr::Function("log".into(), vec![Expr::from_i64(-1)]);
    let held_zero = Expr::Mul(vec![Expr::from_i64(0), undefined]);
    for arguments in [
        vec![held_zero.clone(), Expr::from_i64(1)],
        vec![Expr::from_i64(0), held_zero],
    ] {
        assert!(
            Expr::Function("atan2".into(), arguments)
                .evalf_ball(24)
                .is_err()
        );
    }
}
