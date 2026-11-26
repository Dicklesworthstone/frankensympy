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

#[test]
fn principal_powers_recover_exact_gaussian_roots_and_reciprocals() {
    // These witnesses are ordinary integer products, with the root's angle
    // inside the principal sector: (2+i)^3=2+11i, (1+i)^3=-2+2i,
    // (3+i)^3=18+26i and (2+i)^5=-38+41i.
    for (real, imag, degree, root_real, root_imag) in [
        (2, 11, 3, 2, 1),
        (-2, 2, 3, 1, 1),
        (18, 26, 3, 3, 1),
        (-38, 41, 5, 2, 1),
    ] {
        for sign in [-1, 1] {
            let source = ComplexBall::exact_complex(q(real, 1), q(sign * imag, 1));
            for numerator in [-1, 1, 2] {
                let exponent = ComplexBall::from_real(RealBall::exact(q(numerator, degree)));
                let result = source.pow_principal(&exponent, 30).unwrap();
                let a = q(root_real, 1);
                let b = q(sign * root_imag, 1);
                let (expected_real, expected_imag) = match numerator {
                    -1 => {
                        let norm = &a * &a + &b * &b;
                        (&a / &norm, -&b / norm)
                    }
                    1 => (a, b),
                    _ => (&a * &a - &b * &b, q(2, 1) * a * b),
                };
                assert!(result.contains(&expected_real, &expected_imag));
                assert!(result.real().width() < tolerance(24));
                assert!(result.imag().width() < tolerance(24));
            }
        }
    }
}

#[test]
fn negative_real_cube_root_keeps_the_principal_complex_branch() {
    let exponent = ComplexBall::from_real(RealBall::exact(q(1, 3)));
    let result = ComplexBall::from_i64(-8)
        .pow_principal(&exponent, 30)
        .unwrap();
    // Principal (-8)^(1/3) = 1 + sqrt(3)*i; the real odd root -2
    // also cubes to -8 but is not the requested branch.
    assert!(result.real().contains(&q(1, 1)));
    assert!(result.imag().lower() > q(1, 1));
    assert!(result.imag().upper() < q(2, 1));
    assert!(result.imag().pow(2).unwrap().contains(&q(3, 1)));
    assert!(!result.contains(&q(-2, 1), &q(0, 1)));
    assert!(result.pow(3).unwrap().contains(&q(-8, 1), &q(0, 1)));
}

/// Alternating Taylor bounds for exp(-x), 0 <= x <= 4. Terms decrease
/// after the fourth; the remainder after 80 terms has the next term's
/// positive sign and magnitude at most x^80/80!.
fn reference_negative_exp(value: BigRational) -> (BigRational, BigRational) {
    assert!(value >= q(0, 1) && value <= q(4, 1));
    let mut term = q(1, 1);
    let mut sum = q(1, 1);
    for index in 1..80 {
        term = term * &value / q(index, 1);
        if index % 2 == 0 {
            sum += &term;
        } else {
            sum -= &term;
        }
    }
    let upper = &sum + term * value / q(80, 1);
    (sum, upper)
}

/// Positive exponentials use exact reciprocals of the independent
/// alternating bounds above, not the production exponential kernel.
fn reference_exp(value: BigRational) -> RealBall {
    if value < q(0, 1) {
        let (lower, upper) = reference_negative_exp(-value);
        interval(lower, upper)
    } else {
        let (lower, upper) = reference_negative_exp(value);
        interval(q(1, 1) / upper, q(1, 1) / lower)
    }
}

fn reference_hyperbolic(value: BigRational) -> (RealBall, RealBall) {
    let positive = reference_exp(value.clone());
    let negative = reference_exp(-value);
    let sinh = interval(
        (positive.lower() - negative.upper()) / q(2, 1),
        (positive.upper() - negative.lower()) / q(2, 1),
    );
    let cosh = interval(
        (positive.lower() + negative.lower()) / q(2, 1),
        (positive.upper() + negative.upper()) / q(2, 1),
    );
    (sinh, cosh)
}

#[test]
fn complex_exponential_keeps_real_interval_variation_and_exact_endpoint_bounds() {
    for (lower, upper) in [(-2, -1), (-1, 1), (0, 2)] {
        let source = rectangle(lower, upper, 0, 0);
        for digits in [8, 30] {
            let result = source.exp(digits).unwrap();
            for value in [lower, upper] {
                let reference = reference_exp(q(value, 1));
                assert!(result.real().contains_ball(&reference));
            }
            assert!(result.imag().contains(&q(0, 1)));
            assert!(result.real().width() > q(1, 100));
        }
    }
    // The real accuracy API retains its established refusal: input
    // variation cannot satisfy an arbitrarily small absolute radius.
    let wide = interval(q(-1, 1), q(1, 1));
    assert!(matches!(
        wide.exp(30),
        Err(BallError::ResourceLimitExceeded { .. })
    ));
    assert!(ComplexBall::from_real(wide).exp(30).is_ok());
}

#[test]
fn complex_exponential_keeps_phase_variation_in_both_components() {
    // This rectangle contains log(2) + i*theta at theta=0, +/-pi/2,
    // and pi. Its exponential therefore contains all four exact axis
    // points below. Midpoint-only real or imaginary evaluation loses them.
    let result = rectangle(-1, 1, -4, 4).exp(24).unwrap();
    for (real, imag) in [(2, 0), (-2, 0), (0, 2), (0, -2)] {
        assert!(result.contains(&q(real, 1), &q(imag, 1)));
    }
    assert!(result.real().width() >= q(4, 1));
    assert!(result.imag().width() >= q(4, 1));
}

#[test]
fn complex_circular_functions_keep_imaginary_ranges_and_component_signs() {
    let (pi_lower, pi_upper) = reference_pi();
    let half_pi = interval(pi_lower / q(2, 1), pi_upper / q(2, 1));
    for (lower, upper) in [(-2, -1), (-1, 1), (0, 2)] {
        let y = interval(q(lower, 1), q(upper, 1));
        for at_half_pi in [false, true] {
            let x = if at_half_pi {
                half_pi.clone()
            } else {
                RealBall::from_i64(0)
            };
            let source = ComplexBall::new(x, y.clone());
            let sine = source.sin(24).unwrap();
            let cosine = source.cos(24).unwrap();
            for value in [lower, upper] {
                let (sinh, cosh) = reference_hyperbolic(q(value, 1));
                if at_half_pi {
                    // sin(pi/2+iy)=cosh(y), cos(pi/2+iy)=-i*sinh(y).
                    assert!(sine.real().contains_ball(&cosh));
                    assert!(sine.imag().contains(&q(0, 1)));
                    assert!(cosine.real().contains(&q(0, 1)));
                    assert!(cosine.imag().contains_ball(&sinh.neg()));
                } else {
                    // sin(iy)=i*sinh(y), cos(iy)=cosh(y).
                    assert!(sine.real().contains(&q(0, 1)));
                    assert!(sine.imag().contains_ball(&sinh));
                    assert!(cosine.real().contains_ball(&cosh));
                    assert!(cosine.imag().contains(&q(0, 1)));
                }
            }
            let varying = if at_half_pi {
                cosine.imag()
            } else {
                sine.imag()
            };
            assert!(varying.width() > q(1, 1));
        }
    }
}

#[test]
fn elementary_compositions_keep_point_accuracy_and_conjugate_symmetry() {
    for value in [-2, -1, 0, 1, 2] {
        let result = ComplexBall::from_i64(value).exp(30).unwrap();
        assert!(result.real().contains_ball(&reference_exp(q(value, 1))));
        assert!(result.real().width() < tolerance(30));
        assert!(result.imag().width() < tolerance(30));
    }
    for source in [rectangle(-1, 2, -2, 1), rectangle(0, 1, 1, 2)] {
        assert_eq!(
            source.exp(20).unwrap().conj(),
            source.conj().exp(20).unwrap()
        );
        assert_eq!(
            source.sin(20).unwrap().conj(),
            source.conj().sin(20).unwrap()
        );
        assert_eq!(
            source.cos(20).unwrap().conj(),
            source.conj().cos(20).unwrap()
        );
    }
}

#[test]
fn exponential_compositions_admit_precision_and_whole_input_before_evaluation() {
    for digits in [
        0,
        ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS + 1,
        u32::MAX,
    ] {
        for result in [
            ComplexBall::from_i64(0).exp(digits),
            ComplexBall::from_i64(0).sin(digits),
            ComplexBall::from_i64(0).cos(digits),
        ] {
            assert!(matches!(
                result,
                Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
            ));
        }
    }
    let huge =
        BigRational::from_integer(BigInt::one() << (MAX_REAL_BALL_WORK_BITS / 128 + 16) as u32);
    for component in [huge.clone(), q(1, 1) / huge] {
        for source in [
            ComplexBall::exact_complex(component.clone(), q(0, 1)),
            ComplexBall::exact_complex(q(0, 1), component),
        ] {
            for result in [source.exp(20), source.sin(20), source.cos(20)] {
                assert!(matches!(
                    result,
                    Err(BallError::ResourceLimitExceeded { .. })
                ));
            }
        }
    }
    // No midpoint shortcut may hide the out-of-envelope upper endpoint.
    for source in [rectangle(-1, 100_001, 0, 0), rectangle(-100_001, 1, 0, 0)] {
        assert!(matches!(
            source.exp(20),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
    }
    let imaginary = rectangle(0, 0, -1, 100_001);
    assert!(matches!(
        imaginary.sin(20),
        Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
    ));
    assert!(matches!(
        imaginary.cos(20),
        Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
    ));
}

#[test]
fn imaginary_exponents_match_independent_exact_taylor_bounds() {
    let (pi_lower, pi_upper) = reference_pi();
    for (base, divisor) in [(ComplexBall::i(), 2), (ComplexBall::from_i64(-1), 1)] {
        // i^i = exp(-pi/2), (-1)^i = exp(-pi). The reference uses
        // Machin's formula and exact rational Taylor bounds, not ln/pow/exp.
        let lower = reference_negative_exp(&pi_upper / q(divisor, 1)).0;
        let upper = reference_negative_exp(&pi_lower / q(divisor, 1)).1;
        let result = base.pow_principal(&ComplexBall::i(), 40).unwrap();
        inside_reference(result.real(), &(lower, upper));
        assert!(result.imag().contains(&q(0, 1)));
        assert!(result.imag().width() < tolerance(30));
    }
}

#[test]
fn principal_power_retains_whole_base_and_exponent_intervals() {
    let base = ComplexBall::from_real(interval(q(4, 1), q(9, 1)));
    let exponent = ComplexBall::from_real(interval(q(1, 2), q(1, 1)));
    let result = base.pow_principal(&exponent, 30).unwrap();
    // The full image on these positive rectangles is [2, 9]. Selecting
    // either midpoint, or silently treating the exponent as an integer,
    // would exclude at least one of these independently known corners.
    for value in [2, 3, 4, 9] {
        assert!(result.contains(&q(value, 1), &q(0, 1)));
    }
    assert!(result.real().width() >= q(7, 1));

    // Both coordinates of the exponent vary. The real edge gives 1, I,
    // and -1; the imaginary edge exp(-pi*t), 0<=t<=1, contains 1/2.
    let complex_exponents = rectangle(0, 1, 0, 1);
    let result = ComplexBall::from_i64(-1)
        .pow_principal(&complex_exponents, 30)
        .unwrap();
    for (real, imag) in [
        (q(1, 1), q(0, 1)),
        (q(-1, 1), q(0, 1)),
        (q(0, 1), q(1, 1)),
        (q(1, 2), q(0, 1)),
    ] {
        assert!(result.contains(&real, &imag));
    }
}

#[test]
fn principal_power_branch_cut_is_separate_from_integer_power_domain() {
    let cube_root = ComplexBall::from_real(RealBall::exact(q(1, 3)));
    for source in [
        rectangle(-2, -1, -1, 1),
        rectangle(-2, -1, 0, 1),
        rectangle(-2, -1, -1, 0),
    ] {
        assert!(source.pow_principal(&cube_root, 30).is_err());
        assert!(source.pow_principal(&ComplexBall::from_i64(3), 30).is_ok());
    }
    let upper = rectangle(-2, -1, 1, 2)
        .pow_principal(&cube_root, 30)
        .unwrap();
    let lower = rectangle(-2, -1, -2, -1)
        .pow_principal(&cube_root, 30)
        .unwrap();
    assert!(upper.imag().lower() > q(0, 1));
    assert_eq!(upper.conj(), lower);
}

#[test]
fn zero_power_requires_a_defined_positive_real_exponent() {
    let zero = ComplexBall::from_i64(0);
    for exponent in [
        ComplexBall::from_real(RealBall::exact(q(1, 3))),
        ComplexBall::from_real(interval(q(1, 3), q(2, 3))),
        ComplexBall::from_i64(3),
    ] {
        assert_eq!(zero.pow_principal(&exponent, 30).unwrap(), zero);
    }
    for exponent in [
        zero.clone(),
        ComplexBall::from_i64(-1),
        ComplexBall::from_real(interval(q(0, 1), q(1, 3))),
        ComplexBall::i(),
        ComplexBall::exact_complex(q(1, 1), q(1, 1)),
    ] {
        assert!(zero.pow_principal(&exponent, 30).is_err());
    }
    let uncertain = rectangle(0, 1, 0, 0);
    let exponent = ComplexBall::from_real(RealBall::exact(q(1, 3)));
    assert!(uncertain.pow_principal(&exponent, 30).is_err());
}

#[test]
fn general_powers_preflight_both_inputs_and_precision_before_trivial_values() {
    for digits in [
        0,
        ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS + 1,
        u32::MAX,
    ] {
        assert!(
            ComplexBall::from_i64(1)
                .pow_principal(&ComplexBall::from_i64(0), digits)
                .is_err()
        );
    }
    let enormous = ComplexBall::from_real(RealBall::exact(BigRational::from_integer(
        BigInt::one() << (MAX_REAL_BALL_WORK_BITS / 128 + 16) as u32,
    )));
    for (base, exponent) in [
        (enormous.clone(), ComplexBall::from_i64(0)),
        (ComplexBall::from_i64(1), enormous.clone()),
        (ComplexBall::from_i64(0), enormous),
    ] {
        assert!(matches!(
            base.pow_principal(&exponent, 30),
            Err(BallError::ResourceLimitExceeded { .. }),
        ));
    }
}


mod inverse_functions {
    use fsym_core::{BallError, BigInt, BigRational, ComplexBall, RealBall};

    type Inverse = fn(&ComplexBall, u32) -> Result<ComplexBall, BallError>;

    fn q(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    fn point(real: i64, imag: i64) -> ComplexBall {
        ComplexBall::exact_complex(q(real, 1), q(imag, 1))
    }

    fn interval(lower: BigRational, upper: BigRational) -> RealBall {
        let two = q(2, 1);
        RealBall::new((&lower + &upper) / &two, (upper - lower) / two).unwrap()
    }

    fn inverses() -> [Inverse; 6] {
        [
            ComplexBall::asin,
            ComplexBall::acos,
            ComplexBall::atan,
            ComplexBall::asinh,
            ComplexBall::acosh,
            ComplexBall::atanh,
        ]
    }

    fn assert_components(actual: ComplexBall, real: &RealBall, imag: &RealBall) {
        assert!(actual.real().contains_ball(real), "{actual} excludes real {real}");
        assert!(actual.imag().contains_ball(imag), "{actual} excludes imaginary {imag}");
        let tolerance = BigRational::new(BigInt::from(1), BigInt::from(10).pow(18));
        assert!(actual.real().width() < tolerance, "{actual}");
        assert!(actual.imag().width() < tolerance, "{actual}");
    }

    #[test]
    fn exact_cut_values_keep_the_pinned_principal_branches() {
        let pi = RealBall::pi(70).unwrap();
        let half_pi = pi.div(&RealBall::from_i64(2)).unwrap();
        let a = RealBall::from_i64(2).acosh(70).unwrap();
        let l = RealBall::from_i64(3).ln(70).unwrap().div(&RealBall::from_i64(2)).unwrap();
        let zero = RealBall::from_i64(0);
        assert_components(point(2, 0).asin(28).unwrap(), &half_pi, &a.neg());
        assert_components(point(-2, 0).asin(28).unwrap(), &half_pi.neg(), &a);
        assert_components(point(2, 0).acos(28).unwrap(), &zero, &a);
        assert_components(point(-2, 0).acos(28).unwrap(), &pi, &a.neg());
        assert_components(point(0, 2).atan(28).unwrap(), &half_pi, &l);
        assert_components(point(0, -2).atan(28).unwrap(), &half_pi.neg(), &l.neg());
        assert_components(point(0, 2).asinh(28).unwrap(), &a, &half_pi);
        assert_components(point(0, -2).asinh(28).unwrap(), &a.neg(), &half_pi.neg());
        assert_components(point(-2, 0).acosh(28).unwrap(), &a, &pi);
        assert_components(point(2, 0).atanh(28).unwrap(), &l, &half_pi.neg());
        assert_components(point(-2, 0).atanh(28).unwrap(), &l.neg(), &half_pi);
    }

    #[test]
    fn finite_branch_points_and_zero_components_remain_exact() {
        let zero = RealBall::from_i64(0);
        let pi = RealBall::pi(70).unwrap();
        let half_pi = pi.div(&RealBall::from_i64(2)).unwrap();
        assert_components(point(1, 0).asin(28).unwrap(), &half_pi, &zero);
        assert_components(point(-1, 0).asin(28).unwrap(), &half_pi.neg(), &zero);
        assert_eq!(point(1, 0).acos(28).unwrap(), point(0, 0));
        assert_components(point(-1, 0).acos(28).unwrap(), &pi, &zero);
        assert_components(point(0, 1).asinh(28).unwrap(), &zero, &half_pi);
        assert_components(point(0, -1).asinh(28).unwrap(), &zero, &half_pi.neg());
        assert_eq!(point(1, 0).acosh(28).unwrap(), point(0, 0));
        assert_components(point(-1, 0).acosh(28).unwrap(), &zero, &pi);
        for inverse in [
            ComplexBall::asin as Inverse,
            ComplexBall::atan,
            ComplexBall::asinh,
            ComplexBall::atanh,
        ] {
            assert_eq!(inverse(&point(0, 0), 28).unwrap(), point(0, 0));
        }
    }

    #[test]
    fn four_quadrants_match_independent_modulus_and_angle_formulas() {
        // For z=1+i, |z+1|=sqrt(5), |z-1|=1. The real-modulus
        // formulas give asin(z)=asin(B)+i*acosh(A), where
        // A=(sqrt(5)+1)/2 and B=(sqrt(5)-1)/2. These do not
        // repeat the production complex logarithm/square-root composition.
        let one = RealBall::from_i64(1);
        let two = RealBall::from_i64(2);
        let root = RealBall::from_i64(5).sqrt(400).unwrap();
        let a = root.add(&one).div(&two).unwrap();
        let b = root.sub(&one).div(&two).unwrap();
        let sine_real = b.asin(70).unwrap();
        let sine_imag = a.acosh(70).unwrap();
        let half_pi = RealBall::pi(70).unwrap().div(&two).unwrap();
        let tangent_real = two.atan2(&RealBall::from_i64(-1), 70).unwrap().div(&two).unwrap();
        let tangent_imag = RealBall::from_i64(5).ln(70).unwrap().div(&RealBall::from_i64(4)).unwrap();
        for x in [-1, 1] {
            for y in [-1, 1] {
                let z = point(x, y);
                let ar = if x < 0 { sine_real.neg() } else { sine_real.clone() };
                let ai = if y < 0 { sine_imag.neg() } else { sine_imag.clone() };
                assert_components(z.asin(28).unwrap(), &ar, &ai);
                assert_components(z.acos(28).unwrap(), &half_pi.sub(&ar), &ai.neg());
                let tr = if x < 0 { tangent_real.neg() } else { tangent_real.clone() };
                let ti = if y < 0 { tangent_imag.neg() } else { tangent_imag.clone() };
                assert_components(z.atan(28).unwrap(), &tr, &ti);
                let hr = if x < 0 { sine_imag.neg() } else { sine_imag.clone() };
                let hi = if y < 0 { sine_real.neg() } else { sine_real.clone() };
                assert_components(z.asinh(28).unwrap(), &hr, &hi);
                let angle = if x < 0 { b.neg() } else { b.clone() }.acos(70).unwrap();
                let angle = if y < 0 { angle.neg() } else { angle };
                assert_components(z.acosh(28).unwrap(), &sine_imag, &angle);
                let hr = if x < 0 { tangent_imag.neg() } else { tangent_imag.clone() };
                let hi = if y < 0 { tangent_real.neg() } else { tangent_real.clone() };
                assert_components(z.atanh(28).unwrap(), &hr, &hi);
            }
        }
    }

    #[test]
    fn cut_sides_are_selected_from_the_full_source_sign() {
        let epsilon = q(1, 1000);
        let above = ComplexBall::exact_complex(q(2, 1), epsilon.clone());
        let below = ComplexBall::exact_complex(q(2, 1), -&epsilon);
        assert!(above.asin(28).unwrap().imag().is_positive());
        assert!(below.asin(28).unwrap().imag().is_negative());
        assert!(above.acos(28).unwrap().imag().is_negative());
        assert!(below.acos(28).unwrap().imag().is_positive());
        assert!(above.atanh(28).unwrap().imag().is_positive());
        assert!(below.atanh(28).unwrap().imag().is_negative());
        let right = ComplexBall::exact_complex(epsilon.clone(), q(2, 1));
        let left = ComplexBall::exact_complex(-&epsilon, q(2, 1));
        assert!(right.atan(28).unwrap().real().is_positive());
        assert!(left.atan(28).unwrap().real().is_negative());
        assert!(right.asinh(28).unwrap().real().is_positive());
        assert!(left.asinh(28).unwrap().real().is_negative());
        let upper_left = ComplexBall::exact_complex(q(-2, 1), epsilon.clone());
        let lower_left = ComplexBall::exact_complex(q(-2, 1), -epsilon);
        assert!(upper_left.acosh(28).unwrap().imag().is_positive());
        assert!(lower_left.acosh(28).unwrap().imag().is_negative());
    }

    #[test]
    fn complete_real_and_imaginary_axis_hulls_cross_finite_branch_points() {
        let real = ComplexBall::from_real(interval(q(-2, 1), q(2, 1)));
        let asin = real.asin(28).unwrap();
        let acos = real.acos(28).unwrap();
        let imaginary = ComplexBall::new(RealBall::from_i64(0), interval(q(-2, 1), q(2, 1)));
        let asinh = imaginary.asinh(28).unwrap();
        let acosh_source = ComplexBall::from_real(interval(q(-3, 1), q(2, 1)));
        let acosh = acosh_source.acosh(28).unwrap();
        for x in -2..=2 {
            for (outer, inner) in [
                (&asin, point(x, 0).asin(60).unwrap()),
                (&acos, point(x, 0).acos(60).unwrap()),
                (&asinh, point(0, x).asinh(60).unwrap()),
                (&acosh, point(x, 0).acosh(60).unwrap()),
            ] {
                assert!(outer.real().contains_ball(inner.real()));
                assert!(outer.imag().contains_ball(inner.imag()));
            }
        }
        let endpoint = point(-3, 0).acosh(60).unwrap();
        assert!(acosh.real().contains_ball(endpoint.real()));
        assert!(acosh.imag().contains_ball(endpoint.imag()));
        assert!(asin.imag().lower() < q(-1, 1));
        assert!(asin.imag().upper() > q(1, 1));
        assert!(acosh.real().contains(&q(0, 1)));
    }

    #[test]
    fn nonzero_input_width_is_retained_for_every_inverse() {
        let axis = interval(q(99, 100), q(101, 100));
        let source = ComplexBall::new(axis.clone(), axis);
        for inverse in inverses() {
            let result = inverse(&source, 30).unwrap();
            let refined = inverse(&source, 60).unwrap();
            for x in [99, 100, 101] {
                for y in [99, 100, 101] {
                    let value = inverse(&ComplexBall::exact_complex(q(x, 100), q(y, 100)), 70).unwrap();
                    assert!(result.real().contains_ball(value.real()));
                    assert!(result.imag().contains_ball(value.imag()));
                    assert!(refined.real().contains_ball(value.real()));
                    assert!(refined.imag().contains_ball(value.imag()));
                }
            }
            assert!(result.real().width() > q(1, 1000));
            assert!(refined.real().width() > q(1, 1000));
        }
    }

    #[test]
    fn exact_and_interval_poles_refuse_including_endpoint_contact() {
        for x in [-1, 1] {
            assert!(point(x, 0).atanh(28).is_err());
            assert!(point(0, x).atan(28).is_err());
        }
        for (lower, upper) in [(-2, -1), (-1, 0), (0, 1), (1, 2), (-2, 2)] {
            let axis = interval(q(lower, 1), q(upper, 1));
            assert!(ComplexBall::from_real(axis.clone()).atanh(28).is_err());
            assert!(ComplexBall::new(RealBall::from_i64(0), axis).atan(28).is_err());
        }
    }

    #[test]
    fn off_axis_rectangles_touching_cuts_refuse_even_with_safe_midpoints() {
        let touch = interval(q(0, 1), q(1, 1));
        for sign in [-1, 1] {
            let source = ComplexBall::new(RealBall::from_i64(2 * sign), touch.clone());
            assert!(source.asin(28).is_err());
            assert!(source.acos(28).is_err());
            assert!(source.atanh(28).is_err());
            let source = ComplexBall::new(touch.clone(), RealBall::from_i64(2 * sign));
            assert!(source.atan(28).is_err());
            assert!(source.asinh(28).is_err());
        }
        for x in [-2, 0, 1] {
            assert!(ComplexBall::new(RealBall::from_i64(x), touch.clone()).acosh(28).is_err());
        }
    }

    #[test]
    fn inverse_precision_and_source_height_are_preflighted() {
        let huge = ComplexBall::from_real(RealBall::exact(BigRational::from_integer(
            BigInt::from(2).pow(20_000),
        )));
        for inverse in inverses() {
            assert!(inverse(&point(1, 1), 0).is_err());
            assert!(inverse(&point(1, 1), ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS + 1).is_err());
            assert!(matches!(inverse(&huge, 24), Err(BallError::ResourceLimitExceeded { .. })));
        }
    }
}
