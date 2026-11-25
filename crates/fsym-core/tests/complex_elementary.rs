//! Independent exact witnesses for principal complex elementary enclosures.
//!
//! Gaussian rational squares provide exact square-root answers without using
//! a square-root implementation as the oracle. Logarithm references below
//! use finite exact rational series and proved remainders, independently of
//! the production rounded component kernels.

use fsym_core::ball::MAX_REAL_BALL_WORK_BITS;
use fsym_core::{BallError, BigInt, BigRational, ComplexBall, RealBall};

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    let two = q(2, 1);
    RealBall::new((&lower + &upper) / &two, (upper - lower) / two).unwrap()
}

fn rectangle(x0: i64, x1: i64, y0: i64, y1: i64) -> ComplexBall {
    ComplexBall::new(interval(q(x0, 1), q(x1, 1)), interval(q(y0, 1), q(y1, 1)))
}

fn tolerance(digits: u32) -> BigRational {
    BigRational::new(BigInt::one(), BigInt::from(10).pow(digits))
}

fn square(real: &BigRational, imag: &BigRational) -> ComplexBall {
    ComplexBall::exact_complex(real * real - imag * imag, q(2, 1) * real * imag)
}

/// Alternating-series bounds for atan(x), 0 <= x <= 1/2.
fn reference_atan(value: BigRational) -> (BigRational, BigRational) {
    assert!(value >= q(0, 1) && value <= q(1, 2));
    let square = &value * &value;
    let mut power = value;
    let mut sum = q(0, 1);
    for index in 0..20 {
        let term = &power / q(2 * index + 1, 1);
        if index % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
        power *= &square;
    }
    // Twenty included terms end with a subtraction. The next term is
    // positive, so these are ordered bounds, not an error estimate.
    let upper = &sum + power / q(41, 1);
    (sum, upper)
}

fn reference_pi() -> (BigRational, BigRational) {
    let (a0, a1) = reference_atan(q(1, 5));
    let (b0, b1) = reference_atan(q(1, 239));
    (q(16, 1) * a0 - q(4, 1) * b1, q(16, 1) * a1 - q(4, 1) * b0)
}

/// log(q) = 2 sum z^(2k+1)/(2k+1), z=(q-1)/(q+1).
/// Remaining positive terms are bounded by a geometric series.
fn reference_positive_log(value: BigRational) -> (BigRational, BigRational) {
    assert!(value >= q(1, 1) && value <= q(5, 1));
    let z = (&value - q(1, 1)) / (&value + q(1, 1));
    let z_squared = &z * &z;
    let mut power = z;
    let mut sum = q(0, 1);
    for index in 0..64 {
        sum += &power / q(2 * index + 1, 1);
        power *= &z_squared;
    }
    let lower = q(2, 1) * sum;
    let remainder = q(2, 1) * power / (q(129, 1) * (q(1, 1) - z_squared));
    let upper = &lower + remainder;
    (lower, upper)
}

fn reference_acute_angle() -> (BigRational, BigRational) {
    // atan(4/3) = pi/4 + atan(1/7).
    let (pi0, pi1) = reference_pi();
    let (a0, a1) = reference_atan(q(1, 7));
    (pi0 / q(4, 1) + a0, pi1 / q(4, 1) + a1)
}

fn inside_reference(value: &RealBall, bounds: &(BigRational, BigRational)) {
    assert!(
        value.lower() >= bounds.0,
        "lower endpoint escaped the exact reference"
    );
    assert!(
        value.upper() <= bounds.1,
        "upper endpoint escaped the exact reference"
    );
}

#[test]
fn principal_square_roots_recover_exact_gaussian_rational_squares() {
    for real in 0..=4 {
        for imag in -4..=4 {
            let a = q(real, 3);
            let b = q(imag, 5);
            let expected_imag = if real == 0 {
                q(imag.abs(), 5)
            } else {
                b.clone()
            };
            let root = square(&a, &b).sqrt(30).unwrap();
            assert!(root.contains(&a, &expected_imag));
            assert!(root.real().width() < tolerance(30));
            assert!(root.imag().width() < tolerance(30));
            assert!(root.real().lower() >= q(0, 1));
        }
    }
    // A wrong real-only fallback, a sign error, or omission of the modulus
    // factor cannot satisfy this rational identity in all four quadrants.
    assert_eq!(
        ComplexBall::exact_complex(q(-3, 1), q(-4, 1))
            .sqrt(30)
            .unwrap(),
        ComplexBall::exact_complex(q(1, 1), q(-2, 1)),
    );
}

/// Algebraic identities identify the inexact components of sqrt(1+i).
/// On u in [1,2], 4u^4-4u^2-1 is strictly increasing; on v in [0,1],
/// 4v^4+4v^2-1 is strictly increasing. Opposite endpoint signs therefore
/// prove that the respective unique positive roots are enclosed.
#[test]
fn inexact_square_root_rounding_encloses_independently_characterized_roots() {
    let polynomial = |value: &BigRational, quadratic: i64| {
        let square = value * value;
        q(4, 1) * &square * &square + q(quadratic, 1) * square - q(1, 1)
    };
    for digits in [1, 12, 40] {
        for sign in [-1, 1] {
            let result = ComplexBall::exact_complex(q(1, 1), q(sign, 1))
                .sqrt(digits)
                .unwrap();
            let u = result.real();
            let v = result.imag().abs();
            assert!(u.lower() >= q(1, 1) && u.upper() <= q(2, 1));
            assert!(v.lower() >= q(0, 1) && v.upper() <= q(1, 1));
            assert!(polynomial(&u.lower(), -4) <= q(0, 1));
            assert!(polynomial(&u.upper(), -4) >= q(0, 1));
            assert!(polynomial(&v.lower(), 4) <= q(0, 1));
            assert!(polynomial(&v.upper(), 4) >= q(0, 1));
            assert!(u.width() < tolerance(digits));
            assert!(v.width() < tolerance(digits));
            if sign < 0 {
                assert!(result.imag().upper() < q(0, 1));
            } else {
                assert!(result.imag().lower() > q(0, 1));
            }
        }
    }
}

#[test]
fn square_root_encloses_rectangles_on_both_sides_of_the_positive_axis() {
    let cases = [
        rectangle(0, 16, -12, 12),
        rectangle(-16, -1, 1, 16),
        rectangle(-16, -1, -16, -1),
        rectangle(-4, 4, 1, 16),
        rectangle(-4, 4, -16, -1),
    ];
    for source in cases {
        let result = source.sqrt(24).unwrap();
        let mut witnesses = 0;
        for real in 0..=8 {
            for imag in -8..=8 {
                let a = q(real, 2);
                let b = q(imag, 2);
                let input = square(&a, &b);
                if source.contains(input.real().midpoint(), input.imag().midpoint()) {
                    let expected_imag = if real == 0 { q(imag.abs(), 2) } else { b };
                    assert!(result.contains(&a, &expected_imag));
                    witnesses += 1;
                }
            }
        }
        assert!(
            witnesses >= 5,
            "fixture did not cover enough exact square images"
        );
    }
}

#[test]
fn real_axis_square_root_keeps_zero_and_the_principal_negative_side() {
    assert_eq!(
        ComplexBall::from_i64(0).sqrt(20).unwrap(),
        ComplexBall::from_i64(0)
    );
    assert_eq!(
        ComplexBall::from_i64(-4).sqrt(20).unwrap(),
        ComplexBall::exact_complex(q(0, 1), q(2, 1)),
    );
    let mixed = rectangle(-9, 4, 0, 0).sqrt(20).unwrap();
    assert_eq!(mixed.real(), &interval(q(0, 1), q(2, 1)));
    assert_eq!(mixed.imag(), &interval(q(0, 1), q(3, 1)));
    let right_half_plane = rectangle(0, 1, -1, 1).sqrt(20).unwrap();
    assert!(right_half_plane.contains(&q(0, 1), &q(0, 1)));
    assert!(right_half_plane.contains(&q(1, 1), &q(0, 1)));
}

#[test]
fn negative_axis_contact_is_a_refusal_for_genuinely_complex_boxes() {
    for source in [
        rectangle(-2, -1, -1, 1),
        rectangle(-2, -1, 0, 1),
        rectangle(-2, -1, -1, 0),
        rectangle(-1, 1, 0, 1),
    ] {
        assert!(matches!(
            source.sqrt(20),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
        assert!(matches!(
            source.ln(20),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
    }
    // Exact cut points retain their defined principal values.
    assert!(rectangle(-4, -1, 0, 0).sqrt(20).is_ok());
    assert!(rectangle(-4, -1, 0, 0).ln(20).is_ok());
}

#[test]
fn logarithm_rejects_every_rectangle_that_contains_the_origin() {
    for source in [
        ComplexBall::from_i64(0),
        rectangle(0, 1, 0, 1),
        rectangle(0, 1, -1, 1),
        rectangle(-1, 1, -1, 1),
        rectangle(-1, 0, 0, 0),
    ] {
        assert!(matches!(
            source.ln(20),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
    }
}

#[test]
fn logarithm_principal_axes_match_exact_constants_and_rational_pi_bounds() {
    assert_eq!(
        ComplexBall::from_i64(1).ln(40).unwrap(),
        ComplexBall::from_i64(0)
    );
    let (pi0, pi1) = reference_pi();
    for (source, lower, upper) in [
        (ComplexBall::from_i64(-1), pi0.clone(), pi1.clone()),
        (ComplexBall::i(), &pi0 / q(2, 1), &pi1 / q(2, 1)),
        (ComplexBall::i().neg(), -&pi1 / q(2, 1), -&pi0 / q(2, 1)),
    ] {
        let result = source.ln(40).unwrap();
        assert_eq!(result.real(), &RealBall::from_i64(0));
        inside_reference(result.imag(), &(lower, upper));
        assert!(result.imag().width() < tolerance(40));
    }
}

#[test]
fn logarithm_all_quadrants_agree_with_independent_exact_series() {
    let magnitude = reference_positive_log(q(5, 1));
    let acute = reference_acute_angle();
    let pi = reference_pi();
    for (real, imag) in [(3, 4), (-3, 4), (3, -4), (-3, -4)] {
        let angle = if real > 0 {
            acute.clone()
        } else {
            (&pi.0 - &acute.1, &pi.1 - &acute.0)
        };
        let angle = if imag > 0 {
            angle
        } else {
            (-angle.1, -angle.0)
        };
        let result = ComplexBall::exact_complex(q(real, 1), q(imag, 1))
            .ln(40)
            .unwrap();
        inside_reference(result.real(), &magnitude);
        inside_reference(result.imag(), &angle);
        assert!(result.real().width() < tolerance(40));
        assert!(result.imag().width() < tolerance(40));
    }
}

#[test]
fn wide_log_rectangles_keep_original_input_variation() {
    let magnitude = reference_positive_log(q(5, 1));
    let angle = reference_acute_angle();
    let expected_real = interval(magnitude.0, magnitude.1);
    let expected_imag = interval(angle.0, angle.1);
    for (source, sign) in [(rectangle(2, 4, 3, 5), 1), (rectangle(2, 4, -5, -3), -1)] {
        let result = source.ln(40).unwrap();
        assert!(result.real().contains_ball(&expected_real));
        let expected = if sign > 0 {
            expected_imag.clone()
        } else {
            expected_imag.neg()
        };
        assert!(result.imag().contains_ball(&expected));
        assert!(result.real().width() > q(1, 10));
        assert!(result.imag().width() > q(1, 10));
    }
    let crossing = rectangle(1, 2, -1, 1).ln(30).unwrap();
    assert!(crossing.contains(&q(0, 1), &q(0, 1)));
    assert!(crossing.imag().lower() < q(0, 1));
    assert!(crossing.imag().upper() > q(0, 1));
}

#[test]
fn conjugation_commutes_away_from_the_principal_branch_cut() {
    for source in [
        ComplexBall::exact_complex(q(-3, 1), q(4, 1)),
        rectangle(-5, 3, 1, 4),
        rectangle(1, 4, -2, 3),
    ] {
        assert_eq!(
            source.sqrt(30).unwrap().conj(),
            source.conj().sqrt(30).unwrap()
        );
        assert_eq!(source.ln(30).unwrap().conj(), source.conj().ln(30).unwrap());
    }
}

#[test]
fn square_root_handles_extreme_scale_without_float_bounds_or_cancellation_loss() {
    let tiny = tolerance(200);
    let huge = BigRational::from_integer(BigInt::from(10).pow(100));
    for (real, imag) in [
        (tiny.clone(), q(1, 1)),
        (tiny, q(-1, 1)),
        (huge, tolerance(100)),
    ] {
        let result = square(&real, &imag).sqrt(50).unwrap();
        assert!(result.contains(&real, &imag));
        assert!(result.real().width() < tolerance(50));
        assert!(result.imag().width() < tolerance(50));
    }
}

#[test]
fn complex_elementary_precision_is_checked_even_for_trivial_constants() {
    for digits in [
        0,
        ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS + 1,
        u32::MAX,
    ] {
        assert!(matches!(
            ComplexBall::from_i64(0).sqrt(digits),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_)),
        ));
        assert!(matches!(
            ComplexBall::from_i64(1).ln(digits),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_)),
        ));
    }
    assert!(
        ComplexBall::from_i64(0)
            .sqrt(ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS)
            .is_ok()
    );
    assert!(
        ComplexBall::from_i64(1)
            .ln(ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS)
            .is_ok()
    );
}

#[test]
fn oversized_rationals_refuse_before_nested_complex_arithmetic() {
    let shift = (MAX_REAL_BALL_WORK_BITS / 128 + 16) as u32;
    let enormous = BigInt::one() << shift;
    let values = [
        BigRational::from_integer(enormous.clone()),
        BigRational::new(BigInt::one(), enormous),
    ];
    for value in values {
        for source in [
            ComplexBall::exact_complex(value.clone(), q(1, 1)),
            ComplexBall::exact_complex(q(1, 1), value.clone()),
            ComplexBall::new(
                RealBall::new(q(1, 1), value.clone()).unwrap(),
                RealBall::from_i64(1),
            ),
        ] {
            assert!(matches!(
                source.sqrt(20),
                Err(BallError::ResourceLimitExceeded { .. })
            ));
            assert!(matches!(
                source.ln(20),
                Err(BallError::ResourceLimitExceeded { .. })
            ));
        }
    }
}

#[test]
fn integer_powers_preflight_growth_while_retaining_exact_unit_cycles() {
    assert!(matches!(
        ComplexBall::exact_complex(q(2, 1), q(1, 1)).pow(i32::MAX),
        Err(BallError::ResourceLimitExceeded { .. }),
    ));
    assert_eq!(
        ComplexBall::i().pow(i32::MIN).unwrap(),
        ComplexBall::from_i64(1)
    );
    let huge = BigRational::from_integer(BigInt::one() << 70_000u32);
    assert!(matches!(
        ComplexBall::exact_complex(huge, q(1, 1)).pow(2),
        Err(BallError::ResourceLimitExceeded { .. }),
    ));
}
