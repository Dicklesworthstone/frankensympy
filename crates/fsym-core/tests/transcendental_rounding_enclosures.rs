//! Exact rational reference inequalities for rounded transcendental series.
//! These tests never treat a floating approximation or tolerance as proof.
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness2, Symbol,
    certify_root_box_2x2,
};
use num_traits::Zero;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn decimal_unit(digits: u32) -> BigRational {
    BigRational::new(BigInt::one(), BigInt::from(10).pow(digits))
}

// An exact, unrounded alternating partial sum and its next partial sum
// enclose atan(x) for 0 <= x <= 1/2.
fn atan_reference(x: &BigRational, terms: u32) -> RealBall {
    assert!(*x >= q(0, 1) && *x <= q(1, 2));
    let square = x * x;
    let mut power = x.clone();
    let mut sum = BigRational::zero();
    for index in 0..terms {
        let term = &power / q(i64::from(2 * index + 1), 1);
        if index % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
        power *= &square;
    }
    let term = power / q(i64::from(2 * terms + 1), 1);
    let next = if terms % 2 == 0 {
        &sum + term
    } else {
        &sum - term
    };
    let lower = sum.clone().min(next.clone());
    let upper = sum.max(next);
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

// ln((1+z)/(1-z)) = 2 sum z^(2k+1)/(2k+1), for 0<=z<1.
// The omitted positive tail is at most the first omitted term/(1-z^2).
fn log_reference_from_z(z: &BigRational, terms: u32) -> RealBall {
    assert!(*z >= q(0, 1) && *z <= q(1, 2));
    let square = z * z;
    let mut power = z.clone();
    let mut sum = BigRational::zero();
    for index in 0..terms {
        sum += &power / q(i64::from(2 * index + 1), 1);
        power *= &square;
    }
    let tail = power / (q(i64::from(2 * terms + 1), 1) * (q(1, 1) - square));
    RealBall::new(q(2, 1) * sum + &tail, tail).unwrap()
}

#[test]
fn rounded_zero_atan_powers_keep_the_missing_exact_remainder() {
    for (digits, exponent) in [(8, 9), (24, 15), (80, 33)] {
        let x = decimal_unit(exponent);
        let reference = atan_reference(&x, 2);
        // The exact root is strictly below x; the gap is independently
        // bounded below by x^3/3 - x^5/5.
        assert!(reference.upper() < x);
        for sign in [1, -1] {
            let argument = &x * q(sign, 1);
            let expected = if sign > 0 {
                reference.clone()
            } else {
                reference.neg()
            };
            let actual = RealBall::exact(argument).atan(digits).unwrap();
            assert!(actual.contains_ball(&expected));
            assert!(!actual.radius().is_zero());
            assert!(actual.radius() < &decimal_unit(digits));
        }
    }
    assert_eq!(
        RealBall::from_i64(0).atan(24).unwrap(),
        RealBall::from_i64(0)
    );
}

#[test]
fn logarithm_preserves_power_rounding_error_on_both_sides_of_one() {
    for (digits, exponent) in [(8, 9), (24, 15), (80, 33)] {
        let z = decimal_unit(exponent);
        let argument = (q(1, 1) + &z) / (q(1, 1) - &z);
        let reference = log_reference_from_z(&z, 2);
        assert!(reference.lower() > q(2, 1) * &z);
        for (argument, expected) in [
            (argument.clone(), reference.clone()),
            (q(1, 1) / argument, reference.neg()),
        ] {
            let actual = RealBall::exact(argument).ln(digits).unwrap();
            assert!(actual.contains_ball(&expected));
            assert!(!actual.radius().is_zero());
            assert!(actual.radius() < &decimal_unit(digits));
        }
    }
    assert_eq!(RealBall::from_i64(1).ln(24).unwrap(), RealBall::from_i64(0));
}

#[test]
fn ordinary_atan_log_and_range_reductions_enclose_independent_exact_bounds() {
    let pi = atan_reference(&q(1, 5), 64)
        .mul(&RealBall::from_i64(16))
        .sub(&atan_reference(&q(1, 239), 64).mul(&RealBall::from_i64(4)));
    for (argument, reference) in [
        (q(1, 3), atan_reference(&q(1, 3), 64)),
        (q(1, 2), atan_reference(&q(1, 2), 64)),
        (q(1, 1), pi.div(&RealBall::from_i64(4)).unwrap()),
        (
            q(2, 1),
            pi.div(&RealBall::from_i64(2))
                .unwrap()
                .sub(&atan_reference(&q(1, 2), 64)),
        ),
    ] {
        for sign in [1, -1] {
            let reference = if sign > 0 {
                reference.clone()
            } else {
                reference.neg()
            };
            let actual = RealBall::exact(&argument * q(sign, 1)).atan(24).unwrap();
            assert!(actual.contains_ball(&reference));
            assert!(actual.radius() < &decimal_unit(24));
        }
    }
    for argument in [q(4, 3), q(3, 2), q(2, 1)] {
        let z = (&argument - q(1, 1)) / (&argument + q(1, 1));
        let reference = log_reference_from_z(&z, 64);
        for (argument, reference) in [
            (argument.clone(), reference.clone()),
            (q(1, 1) / argument, reference.neg()),
        ] {
            let actual = RealBall::exact(argument).ln(24).unwrap();
            assert!(actual.contains_ball(&reference));
            assert!(actual.radius() < &decimal_unit(24));
        }
    }
}

#[test]
fn monotone_ball_extensions_keep_both_endpoint_rounding_enclosures() {
    let lo = decimal_unit(9);
    let hi = q(2, 1) * &lo;
    let argument = RealBall::new((&lo + &hi) / q(2, 1), (&hi - &lo) / q(2, 1)).unwrap();
    let actual = argument.atan(8).unwrap();
    assert!(actual.contains_ball(&atan_reference(&lo, 2)));
    assert!(actual.contains_ball(&atan_reference(&hi, 2)));

    let log_lo = (q(1, 1) + &lo) / (q(1, 1) - &lo);
    let log_hi = (q(1, 1) + &hi) / (q(1, 1) - &hi);
    let argument =
        RealBall::new((&log_lo + &log_hi) / q(2, 1), (log_hi - log_lo) / q(2, 1)).unwrap();
    let actual = argument.ln(8).unwrap();
    assert!(actual.contains_ball(&log_reference_from_z(&lo, 2)));
    assert!(actual.contains_ball(&log_reference_from_z(&hi, 2)));
}

#[test]
fn rounded_transcendental_constants_cannot_certify_a_box_missing_the_root() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let [x, y] = variables.clone().map(Expr::Sym);
    for (digits, exponent, radius_exponent) in [(8, 9, 30), (24, 15, 50)] {
        let z = decimal_unit(exponent);
        let log_argument = (q(1, 1) + &z) / (q(1, 1) - &z);
        for (name, argument, center, reference) in [
            ("atan", z.clone(), z.clone(), atan_reference(&z, 2)),
            (
                "log",
                log_argument,
                q(2, 1) * &z,
                log_reference_from_z(&z, 2),
            ),
        ] {
            let equations = [
                x.clone() - Expr::Function(name.into(), vec![Expr::Rational(argument)]),
                y.clone(),
            ];
            let centers = [center, q(0, 1)];
            let mut witness = RootBoxWitness2 {
                domain: centers
                    .clone()
                    .map(|c| RealBall::new(c, decimal_unit(radius_exponent)).unwrap()),
                center: centers,
                preconditioner: [[q(1, 1), q(0, 1)], [q(0, 1), q(1, 1)]],
                precision_digits: digits,
            };
            // The independent exact-series interval is already disjoint
            // from the proposed x box. A root receipt would be false.
            assert!(reference.is_disjoint(&witness.domain[0]));
            assert!(matches!(
                certify_root_box_2x2(&equations, &variables, &witness),
                Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
            ));

            // A sufficiently wide box remains certifiable and the returned
            // enclosure contains the same independent exact bounds.
            witness.domain = witness
                .center
                .clone()
                .map(|c| RealBall::new(c, decimal_unit(digits + 8)).unwrap());
            let accepted = certify_root_box_2x2(&equations, &variables, &witness).unwrap();
            assert!(accepted.root_enclosure()[0].contains_ball(&reference));
            assert_eq!(accepted.contraction_bound(), &q(0, 1));
        }
    }
}

// An independent unscaled positive Taylor sum and a geometric bound on its
// omitted tail enclose exp(x). This reference uses no production rounding or
// argument-halving path, including for the regression argument x=40.
fn exp_reference(x: &BigRational, terms: u32) -> RealBall {
    assert!(*x >= q(0, 1) && *x < q(i64::from(terms + 2), 1));
    let mut term = q(1, 1);
    let mut sum = term.clone();
    for index in 1..=terms {
        term = term * x / q(i64::from(index), 1);
        sum += &term;
    }
    let first_omitted = term * x / q(i64::from(terms + 1), 1);
    let ratio_bound = x / q(i64::from(terms + 2), 1);
    let tail = first_omitted / (q(1, 1) - ratio_bound);
    RealBall::new(sum + &tail / q(2, 1), tail / q(2, 1)).unwrap()
}

// Exact alternating partial sums bound sin/cos at these rational arguments.
// Only the omitted tail needs decreasing terms; the fixture bounds ensure
// that condition long before the chosen truncation degree.
fn trig_reference(x: &BigRational, sine: bool, terms: u32) -> RealBall {
    assert!(*x >= q(0, 1) && *x <= q(4, 1) && terms >= 4);
    let square = x * x;
    let mut term = if sine { x.clone() } else { q(1, 1) };
    let mut sum = term.clone();
    for index in 1..=terms {
        let j = i64::from(index);
        let denominator = if sine {
            (2 * j) * (2 * j + 1)
        } else {
            (2 * j - 1) * (2 * j)
        };
        term = term * &square / q(denominator, 1);
        if index % 2 == 0 {
            sum += &term;
        } else {
            sum -= &term;
        }
    }
    let j = i64::from(terms + 1);
    let denominator = if sine {
        (2 * j) * (2 * j + 1)
    } else {
        (2 * j - 1) * (2 * j)
    };
    let omitted = term * square / q(denominator, 1);
    let next = if terms % 2 == 0 {
        &sum - omitted
    } else {
        &sum + omitted
    };
    let lower = sum.clone().min(next.clone());
    let upper = sum.max(next);
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

#[test]
fn exp_scaling_keeps_outward_enclosures_and_bounded_rational_operands() {
    for (argument, terms) in [(q(1, 3), 80), (q(10, 1), 192), (q(40, 1), 320)] {
        let reference = exp_reference(&argument, terms);
        let positive = RealBall::exact(argument.clone()).exp(24).unwrap();
        let negative = RealBall::exact(-argument).exp(24).unwrap();
        assert!(positive.contains_ball(&reference));
        assert!(negative.contains_ball(&reference.inv().unwrap()));
        assert!(positive.mul(&negative).contains(&q(1, 1)));
        for actual in [positive, negative] {
            assert!(actual.radius() <= &decimal_unit(24));
            assert!(actual.midpoint().height().max_bits() < 2048);
            assert!(actual.radius().height().max_bits() < 2048);
        }
    }
    // A wide image cannot meet the original absolute-radius contract.
    let wide = RealBall::new(q(0, 1), q(1, 100)).unwrap();
    assert!(matches!(
        wide.exp(24),
        Err(fsym_core::BallError::ResourceLimitExceeded { .. })
    ));
}

#[test]
fn trig_reduction_rounds_outward_and_preserves_exact_reference_bounds() {
    for argument in [q(1, 3), q(2, 1), q(4, 1)] {
        for sine in [false, true] {
            let expected = trig_reference(&argument, sine, 64);
            for sign in [-1, 1] {
                let x = RealBall::exact(&argument * q(sign, 1));
                let actual = if sine { x.sin(24) } else { x.cos(24) }.unwrap();
                let reference = if sine && sign < 0 {
                    expected.neg()
                } else {
                    expected.clone()
                };
                assert!(actual.contains_ball(&reference));
                assert!(actual.radius() <= &decimal_unit(24));
                assert!(actual.midpoint().height().max_bits() < 1024);
                assert!(actual.radius().height().max_bits() < 1024);
            }
        }
    }
    let x = RealBall::from_i64(1);
    let sin = x.sin(24).unwrap();
    let cos = x.cos(24).unwrap();
    let double_angle = RealBall::from_i64(2)
        .cos(24)
        .unwrap()
        .sub(&cos.mul(&cos))
        .add(&sin.mul(&sin));
    assert!(double_angle.contains_zero());
}

#[test]
fn wide_trigonometric_intervals_use_the_valid_global_range() {
    let expected = RealBall::new(q(0, 1), q(1, 1)).unwrap();
    for argument in [
        RealBall::new(q(0, 1), q(1000, 1)).unwrap(),
        RealBall::new(q(2, 1), q(1, 1)).unwrap(),
    ] {
        assert_eq!(argument.sin(24).unwrap(), expected);
        assert_eq!(argument.cos(24).unwrap(), expected);
    }
}
