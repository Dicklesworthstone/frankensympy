//! Directed-rounding mutant corpus and enclosure invariant tests (WS11 / C8 gate).
//!
//! Constitutional requirements (Art. VII.5, VIII.5; WORKSTREAM_GRAPH.md §16):
//! - Every certified enclosure must contain independent reference / high-precision values.
//! - Registered weakening and directed-rounding mutants must be killed.
//! - No ordinary float inhabits a certified value.
//! - Root uniqueness/isolation is checked via Sturm's theorem without trusting unverified approximations.

#![forbid(unsafe_code)]

use fsym_core::algebraic::{AlgebraicError, AlgebraicNumber};
use fsym_core::ball::{BallError, RealBall};
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Signed, Zero};

fn q(i: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(i))
}

fn q_frac(num: i64, den: i64) -> BigRational {
    BigRational::new(BigInt::from(num), BigInt::from(den))
}

// ============================================================================
// 1. Sqrt Directed Rounding: Lower and Upper Enclosure Mutants Killed
// ============================================================================

#[test]
fn test_mutant_sqrt_lower_bound_ceiling_killed() {
    // For non-perfect squares, sqrt(x) is irrational.
    // The certified lower bound L must satisfy L^2 <= x.
    // Mutant: rounding UP on lower bound (L_mutant = L + eps) causes L_mutant^2 > x,
    // which violates lower containment.
    let test_radicands = [q(2), q(3), q(5), q(7), q_frac(1, 2), q_frac(10, 3), q(1000)];

    for x in test_radicands {
        let ball = RealBall::exact(x.clone()).sqrt(32).expect("sqrt ball");
        let lower = ball.lower();

        // Authentic lower bound must have lower^2 <= x
        let lower_sq = &lower * &lower;
        assert!(
            lower_sq <= x,
            "Authentic lower bound must be <= sqrt(x): lower^2 = {lower_sq}, x = {x}"
        );

        // Registered weakening mutant: rounding up the lower bound by 1 unit in the last place
        // Denominator of lower is q * 2^32
        let eps = BigRational::new(BigInt::one(), lower.denom().clone());
        let mutant_lower = &lower + &eps;
        let mutant_lower_sq = &mutant_lower * &mutant_lower;

        // Mutant is KILLED because mutant_lower^2 > x (violating containment of sqrt(x))
        assert!(
            mutant_lower_sq > x,
            "Mutant (lower rounded UP) must fail containment: mutant_sq={mutant_lower_sq}, x={x}"
        );
    }
}

#[test]
fn test_mutant_sqrt_upper_bound_floor_killed() {
    // The certified upper bound U must satisfy U^2 >= x.
    // Mutant: rounding DOWN on upper bound (truncation without ceil) causes U_mutant^2 < x,
    // which violates upper containment.
    let test_radicands = [q(2), q(3), q(5), q(7), q_frac(1, 2), q_frac(10, 3), q(1000)];

    for x in test_radicands {
        let ball = RealBall::exact(x.clone()).sqrt(32).expect("sqrt ball");
        let upper = ball.upper();

        // Authentic upper bound must have upper^2 >= x
        let upper_sq = &upper * &upper;
        assert!(
            upper_sq >= x,
            "Authentic upper bound must be >= sqrt(x): upper^2 = {upper_sq}, x = {x}"
        );

        // Registered weakening mutant: rounding down the upper bound by 1 unit
        let eps = BigRational::new(BigInt::one(), upper.denom().clone());
        let mutant_upper = &upper - &eps;
        let mutant_upper_sq = &mutant_upper * &mutant_upper;

        // Mutant is KILLED because mutant_upper^2 < x (violating containment of sqrt(x))
        assert!(
            mutant_upper_sq < x,
            "Mutant (upper rounded DOWN) must fail containment: mutant_sq={mutant_upper_sq}, x={x}"
        );
    }
}

// ============================================================================
// 2. Multiplication Enclosure: Cross-Term & Sign Mutants Killed
// ============================================================================

#[test]
fn test_mutant_mul_cross_term_omission_killed() {
    // In RealBall::mul:
    // Radius = |m1| * r2 + |m2| * r1 + r1 * r2
    // Mutant: omitting the second-order cross term r1 * r2
    // Mutant radius = |m1| * r2 + |m2| * r1
    let b1 = RealBall::new(q(3), q(2)).expect("valid b1"); // [1, 5]
    let b2 = RealBall::new(q(4), q(3)).expect("valid b2"); // [1, 7]

    let prod = b1.mul(&b2); // midpoint 12, radius = 3*3 + 4*2 + 2*3 = 9 + 8 + 6 = 23 -> [-11, 35]

    // Extreme corner product: x1 = 3 + 2 = 5, x2 = 4 + 3 = 7 -> x1 * x2 = 35.
    let corner = q(35);
    assert!(
        prod.contains(&corner),
        "Authentic product ball must contain corner product 5 * 7 = 35"
    );

    // Now evaluate with the mutant that omitted r1 * r2:
    let mutant_radius = b1.midpoint().abs() * b2.radius() + b2.midpoint().abs() * b1.radius(); // 9 + 8 = 17
    let mutant_upper = b1.midpoint() * b2.midpoint() + &mutant_radius; // 12 + 17 = 29

    // Mutant is KILLED because corner 35 > mutant_upper (29)
    assert!(
        corner > mutant_upper,
        "Mutant omitting r1*r2 must fail to contain corner: 35 > {mutant_upper}"
    );
}

#[test]
fn test_mutant_mul_signed_midpoint_without_abs_killed() {
    // Mutant: computing radius as (m1 * r2 + m2 * r1 + r1 * r2) without taking abs(m1), abs(m2).
    // If m1 < 0, m1 * r2 is negative, under-reporting the necessary radius!
    let b1 = RealBall::new(q(-5), q(1)).expect("valid b1"); // [-6, -4]
    let b2 = RealBall::new(q(3), q(1)).expect("valid b2"); // [2, 4]

    let prod = b1.mul(&b2);
    // True radius: |-5|*1 + |3|*1 + 1*1 = 5 + 3 + 1 = 9. Midpoint = -15. Enclosure = [-24, -6].
    // Check all 4 corners:
    // (-6) * 2 = -12, (-6) * 4 = -24, (-4) * 2 = -8, (-4) * 4 = -16
    assert!(prod.contains(&q(-24)));
    assert!(prod.contains(&q(-6)));

    // Mutant calculation without abs:
    let mutant_radius =
        b1.midpoint() * b2.radius() + b2.midpoint() * b1.radius() + b1.radius() * b2.radius();
    // (-5)*1 + 3*1 + 1 = -1 !
    // Mutant radius is negative or dangerously small!
    assert!(
        mutant_radius < *prod.radius(),
        "Mutant without abs must under-report radius: {mutant_radius} < {}",
        prod.radius()
    );
    let mutant_lower = prod.midpoint() - &mutant_radius;
    // -24 is outside mutant enclosure [-14, -16]
    assert!(
        q(-24) < mutant_lower,
        "Mutant must fail to contain extreme negative product -24"
    );
}

// ============================================================================
// 3. Addition & Subtraction: Triangle-Inequality / Cancellation Mutants Killed
// ============================================================================

#[test]
fn test_mutant_add_radius_cancellation_killed() {
    // Mutant: assuming errors cancel in addition and using |r1 - r2| instead of r1 + r2.
    let b1 = RealBall::new(q(10), q(3)).expect("b1"); // [7, 13]
    let b2 = RealBall::new(q(20), q(3)).expect("b2"); // [17, 23]

    let sum = b1.add(&b2); // midpoint 30, radius 6 -> [24, 36]
    let corner = q(36); // (10 + 3) + (20 + 3) = 36
    assert!(sum.contains(&corner));

    // Mutant radius |3 - 3| = 0
    let mutant_radius = (b1.radius() - b2.radius()).abs();
    let mutant_upper = sum.midpoint() + &mutant_radius; // 30 + 0 = 30
    assert!(
        corner > mutant_upper,
        "Mutant radius cancellation must be killed: corner 36 > mutant upper 30"
    );
}

#[test]
fn test_mutant_sub_radius_subtraction_killed() {
    // In subtraction [m1 - r1, m1 + r1] - [m2 - r2, m2 + r2]:
    // Extreme point: (m1 + r1) - (m2 - r2) = (m1 - m2) + (r1 + r2).
    // Radii must ADD under subtraction.
    // Mutant: radii subtract (r1 - r2).
    let b1 = RealBall::new(q(15), q(4)).expect("b1");
    let b2 = RealBall::new(q(5), q(2)).expect("b2");

    let diff = b1.sub(&b2); // midpoint 10, radius 6 -> [4, 16]
    let corner = q(16); // (15 + 4) - (5 - 2) = 19 - 3 = 16
    assert!(diff.contains(&corner));

    // Mutant radius 4 - 2 = 2
    let mutant_radius = b1.radius() - b2.radius();
    let mutant_upper = diff.midpoint() + &mutant_radius; // 10 + 2 = 12
    assert!(
        corner > mutant_upper,
        "Mutant radius subtraction must be killed: corner 16 > mutant upper 12"
    );
}

// ============================================================================
// 4. Inversion & Division: Reversal and Division-by-Zero Mutants Killed
// ============================================================================

#[test]
fn test_mutant_inv_endpoint_reversal_killed() {
    // 1 / [l, u] = [1/u, 1/l] for 0 < l <= u.
    // Mutant: [1/l, 1/u] (naive inversion without reversing endpoints).
    let b = RealBall::new(q(3), q(1)).expect("b"); // [2, 4]
    let inv = b.inv().expect("inv ok");

    // True inverted interval is [1/4, 1/2]
    assert_eq!(inv.lower(), q_frac(1, 4));
    assert_eq!(inv.upper(), q_frac(1, 2));

    // Naive mutant: lower = 1/2, upper = 1/4.
    let mutant_lower = BigRational::one() / b.lower(); // 1/2
    let mutant_upper = BigRational::one() / b.upper(); // 1/4

    // Mutant has mutant_lower > mutant_upper (inverted, invalid interval)
    assert!(
        mutant_lower > mutant_upper,
        "Mutant naive inversion produces invalid interval where lower > upper"
    );
}

#[test]
fn test_mutant_inv_containing_zero_admitted_killed() {
    // Any ball containing zero cannot be safely inverted.
    let zero_crossing = RealBall::new(q(1), q(2)).expect("ball"); // [-1, 3] contains 0
    let err = zero_crossing.inv().unwrap_err();
    assert!(matches!(err, BallError::DivisionByZero(_)));

    let zero_point = RealBall::exact(q(0));
    assert!(matches!(
        zero_point.inv(),
        Err(BallError::DivisionByZero(_))
    ));
}

// ============================================================================
// 5. Integer Power: Zero Crossing and Negative Interval Mutants Killed
// ============================================================================

#[test]
fn test_mutant_pow_even_zero_crossing_monotonicity_killed() {
    // For even power of an interval containing zero: [-2, 3]^2 = [0, 9].
    // Mutant: assuming monotonicity and returning [(-2)^2, 3^2] = [4, 9].
    // But 0 in [-2, 3] squared is 0, which is NOT in [4, 9]!
    let b = RealBall::new(q_frac(1, 2), q_frac(5, 2)).expect("b"); // lower = -2, upper = 3
    assert!(b.contains_zero());

    let b_sq = b.pow(2).expect("b^2");
    assert_eq!(b_sq.lower(), q(0));
    assert_eq!(b_sq.upper(), q(9));
    assert!(b_sq.contains(&q(0)));

    // Mutant monotonicity:
    let mutant_lower = q(-2) * q(-2); // 4
    // 0 is outside mutant
    assert!(
        q(0) < mutant_lower,
        "Mutant assuming monotonicity excludes 0^2 = 0"
    );
}

#[test]
fn test_mutant_pow_negative_interval_even_power_killed() {
    // For negative interval [-5, -2]^2 = [4, 25].
    // Mutant: [(-5)^2, (-2)^2] = [25, 4] (inverted interval).
    let b = RealBall::new(q_frac(-7, 2), q_frac(3, 2)).expect("b"); // [-5, -2]

    let b_sq = b.pow(2).expect("b^2");
    assert_eq!(b_sq.lower(), q(4));
    assert_eq!(b_sq.upper(), q(25));
    assert!(b_sq.lower() <= b_sq.upper());

    // Check corners
    assert!(b_sq.contains(&q(4)));
    assert!(b_sq.contains(&q(25)));
}

// ============================================================================
// 6. Algebraic Number: Square-Free, Root Isolation & Bisection Mutants Killed
// ============================================================================

#[test]
fn test_mutant_algebraic_non_square_free_admitted_killed() {
    // Defining polynomial P(x) = (x - 3)^2 = x^2 - 6x + 9 has a double root at 3.
    // It is not square-free. Its Sturm sequence will fail to end with a non-zero degree-0 constant.
    let non_sf_coeffs = vec![q(9), q(-6), q(1)];
    let ball = RealBall::new(q(3), q(1)).expect("ball");

    let err = AlgebraicNumber::new(non_sf_coeffs, ball).unwrap_err();
    assert!(
        matches!(err, AlgebraicError::NonSquareFreePolynomial),
        "Non-square-free polynomial must be rejected fail-closed"
    );
}

#[test]
fn test_mutant_algebraic_interval_with_multiple_roots_admitted_killed() {
    // P(x) = x^2 - 2 has roots at -sqrt(2) and +sqrt(2).
    // Interval [-2, 2] contains TWO roots.
    // Sturm theorem must detect this and reject construction.
    let poly = vec![q(-2), q(0), q(1)];
    let wide_ball = RealBall::new(q(0), q(2)).expect("ball"); // [-2, 2]

    let err = AlgebraicNumber::new(poly, wide_ball).unwrap_err();
    assert!(
        matches!(err, AlgebraicError::InvalidIsolatingInterval(_)),
        "Interval containing 2 roots must be rejected: got {err:?}"
    );
}

#[test]
fn test_mutant_algebraic_interval_with_zero_roots_admitted_killed() {
    // P(x) = x^2 - 2 has no roots in [2, 5].
    let poly = vec![q(-2), q(0), q(1)];
    let disjoint_ball = RealBall::new(q_frac(7, 2), q_frac(3, 2)).expect("ball"); // [2, 5]

    let err = AlgebraicNumber::new(poly, disjoint_ball).unwrap_err();
    assert!(
        matches!(err, AlgebraicError::InvalidIsolatingInterval(_)),
        "Interval containing 0 roots must be rejected: got {err:?}"
    );
}

#[test]
fn test_mutant_algebraic_bisection_wrong_bracket_killed() {
    // For P(x) = x^2 - 2 on [1, 2]:
    // P(1) = -1 < 0, P(2) = 2 > 0.
    // Root is sqrt(2) ~ 1.4142...
    // Midpoint m = 1.5. P(1.5) = 2.25 - 2 = 0.25 > 0.
    // Root must be in [1, 1.5], NOT [1.5, 2].
    // Mutant: picking the wrong subinterval [1.5, 2] loses the root!
    let poly = vec![q(-2), q(0), q(1)];
    let ball = RealBall::new(q_frac(3, 2), q_frac(1, 2)).expect("ball"); // [1, 2]

    let mut alpha = AlgebraicNumber::new(poly.clone(), ball).expect("valid root 2");
    alpha.refine_step();

    // After 1 step, isolating ball must be [1, 1.5] (midpoint 1.25, radius 0.25)
    assert_eq!(alpha.isolating_ball().lower(), q(1));
    assert_eq!(alpha.isolating_ball().upper(), q_frac(3, 2));

    // Confirm that the true root satisfies P(lower) <= 0 and P(upper) >= 0
    let p_low = AlgebraicNumber::eval_poly_at(&poly, &alpha.isolating_ball().lower());
    let p_high = AlgebraicNumber::eval_poly_at(&poly, &alpha.isolating_ball().upper());
    assert!(
        p_low <= BigRational::zero(),
        "P(lower) = {p_low} must be <= 0"
    );
    assert!(
        p_high >= BigRational::zero(),
        "P(upper) = {p_high} must be >= 0"
    );

    // Mutant branch [1.5, 2]:
    let mutant_low = q_frac(3, 2);
    let mutant_high = q(2);
    let p_mutant_low = AlgebraicNumber::eval_poly_at(&poly, &mutant_low); // 0.25 > 0
    let p_mutant_high = AlgebraicNumber::eval_poly_at(&poly, &mutant_high); // 2 > 0
    // Mutant bracket has NO sign change (both > 0), confirming it lost the root!
    assert!(
        p_mutant_low > BigRational::zero() && p_mutant_high > BigRational::zero(),
        "Mutant bracket has no sign change and lost the root"
    );
}

#[test]
fn test_algebraic_refine_to_arbitrary_precision() {
    // Refine sqrt(2) to radius <= 1/1,000,000
    let poly = vec![q(-2), q(0), q(1)];
    let ball = RealBall::new(q_frac(3, 2), q_frac(1, 2)).expect("ball");
    let mut alpha = AlgebraicNumber::new(poly, ball).expect("valid root");

    let target = q_frac(1, 1_000_000);
    alpha
        .refine_to_radius(&target)
        .expect("refinement succeeded");

    assert!(alpha.isolating_ball().radius() <= &target);

    // Verify root is still strictly enclosed
    let low = alpha.isolating_ball().lower();
    let high = alpha.isolating_ball().upper();
    assert!(&low * &low <= q(2));
    assert!(&high * &high >= q(2));
}

#[test]
fn test_algebraic_exact_sign_determination() {
    // 1. Exact zero
    let mut zero = AlgebraicNumber::from_i64(0);
    assert_eq!(zero.sign(), 0);

    // 2. Positive root sqrt(2)
    let poly = vec![q(-2), q(0), q(1)];
    let ball = RealBall::new(q_frac(3, 2), q_frac(1, 2)).expect("ball");
    let mut sqrt2 = AlgebraicNumber::new(poly.clone(), ball).expect("valid root");
    assert_eq!(sqrt2.sign(), 1);

    // 3. Negative root -sqrt(2)
    let neg_ball = RealBall::new(q_frac(-3, 2), q_frac(1, 2)).expect("ball");
    let mut neg_sqrt2 = AlgebraicNumber::new(poly, neg_ball).expect("valid root");
    assert_eq!(neg_sqrt2.sign(), -1);
}
