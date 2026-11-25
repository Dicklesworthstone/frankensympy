//! Existing ground-expression evaluation must retain principal complex branches
//! and original undefined subexpressions when composing the ball primitives.

use fsym_core::{BigInt, BigRational, ComplexBall, Constant, Expr, RealBall, Symbol};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn tolerance() -> BigRational {
    BigRational::new(BigInt::from(1), BigInt::from(10).pow(16))
}

fn complex(real: i64, imag: i64) -> Expr {
    Expr::Add(vec![
        Expr::from_i64(real),
        Expr::Mul(vec![Expr::from_i64(imag), Expr::Const(Constant::I)]),
    ])
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn atan2(y: Expr, x: Expr) -> Expr {
    Expr::Function("atan2".into(), vec![y, x])
}

fn half_power(base: Expr, numerator: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::Rational(q(numerator, 2))))
}

fn rational_power(base: Expr, numerator: i64, denominator: i64) -> Expr {
    Expr::Pow(
        Arc::new(base),
        Arc::new(Expr::Rational(q(numerator, denominator))),
    )
}

fn assert_point(result: &ComplexBall, real: BigRational, imag: BigRational) {
    assert!(
        result.contains(&real, &imag),
        "{result} excludes {real} + {imag}*I"
    );
    assert!(result.real().width() < tolerance());
    assert!(result.imag().width() < tolerance());
}

#[test]
fn raw_function_and_half_power_use_the_same_principal_square_root() {
    // Exact independent witnesses: (2 +/- I)^2 = 3 +/- 4I and
    // (1 +/- 2I)^2 = -3 +/- 4I. The negative real axis uses +I.
    for (real, imag, root_real, root_imag) in [
        (3, 4, 2, 1),
        (3, -4, 2, -1),
        (-3, 4, 1, 2),
        (-3, -4, 1, -2),
        (-4, 0, 0, 2),
        (9, 0, 3, 0),
        (0, 0, 0, 0),
    ] {
        for source in [
            function("sqrt", complex(real, imag)),
            half_power(complex(real, imag), 1),
        ] {
            assert_point(
                &source.eval_complex_ball(24).unwrap(),
                q(root_real, 1),
                q(root_imag, 1),
            );
        }
    }
}

#[test]
fn signed_half_integer_powers_compose_without_changing_branches() {
    for (numerator, real, imag) in [
        (3, q(2, 1), q(11, 1)),
        (-1, q(2, 5), q(-1, 5)),
        (-3, q(2, 125), q(-11, 125)),
    ] {
        assert_point(
            &half_power(complex(3, 4), numerator)
                .eval_complex_ball(24)
                .unwrap(),
            real,
            imag,
        );
    }
    assert_point(
        &half_power(Expr::from_i64(-1), 3)
            .eval_complex_ball(24)
            .unwrap(),
        q(0, 1),
        q(-1, 1),
    );
    assert_point(
        &half_power(Expr::from_i64(-1), -1)
            .eval_complex_ball(24)
            .unwrap(),
        q(0, 1),
        q(-1, 1),
    );
    assert!(
        half_power(Expr::from_i64(0), -1)
            .eval_complex_ball(24)
            .is_err()
    );
}

#[test]
fn logarithm_aliases_respect_all_four_unit_axis_values() {
    let pi = RealBall::pi(60).unwrap();
    for (real, imag, pi_multiple) in [
        (1, 0, q(0, 1)),
        (0, 1, q(1, 2)),
        (-1, 0, q(1, 1)),
        (0, -1, q(-1, 2)),
    ] {
        let expected_imag = pi.mul(&RealBall::exact(pi_multiple));
        for name in ["log", "ln"] {
            let result = function(name, complex(real, imag))
                .eval_complex_ball(24)
                .unwrap();
            assert!(result.real().contains(&q(0, 1)));
            assert!(result.imag().contains_ball(&expected_imag));
            assert!(result.real().width() < tolerance());
            assert!(result.imag().width() < tolerance());
        }
    }
}

#[test]
fn logarithms_and_roots_compose_inside_existing_complex_arithmetic() {
    // log(3+4I) has real part log(5), independently of its argument.
    let value = complex(3, 4);
    let logarithm = function("log", value.clone());
    let result = logarithm.eval_complex_ball(24).unwrap();
    assert!(
        result
            .real()
            .contains_ball(&RealBall::from_i64(5).ln(60).unwrap())
    );
    assert!(result.imag().lower() > q(0, 1));
    assert!(result.imag().upper() < q(1, 1));
    assert_point(
        &function("exp", logarithm).eval_complex_ball(24).unwrap(),
        q(3, 1),
        q(4, 1),
    );
    assert_point(
        &function("sqrt", function("sqrt", complex(-7, 24)))
            .eval_complex_ball(24)
            .unwrap(),
        q(2, 1),
        q(1, 1),
    );
}

#[test]
fn complex_evaluation_preserves_original_poles_and_unknown_sources() {
    let reciprocal_zero = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    let held_zero_pole = Expr::Mul(vec![Expr::from_i64(0), reciprocal_zero]);
    for source in [
        function("log", Expr::from_i64(0)),
        function("sqrt", held_zero_pole.clone()),
        half_power(held_zero_pole, 1),
        Expr::Mul(vec![Expr::from_i64(0), function("log", Expr::from_i64(0))]),
        function("log", Expr::Sym(Symbol::new("x"))),
        function("sqrt", Expr::Sym(Symbol::new("x"))),
    ] {
        assert!(source.eval_complex_ball(24).is_err(), "{source}");
    }
}

#[test]
fn adding_complex_branches_does_not_weaken_real_domain_checks() {
    for source in [
        function("sqrt", Expr::from_i64(-4)),
        function("log", Expr::from_i64(-1)),
    ] {
        assert!(source.eval_real_ball(24).is_err());
        assert!(source.eval_complex_ball(24).is_ok());
    }
}

#[test]
fn exact_zero_bases_admit_positive_literal_rational_powers() {
    for (numerator, denominator) in [(1, 3), (2, 3), (4, 5), (7, 3), (1, 2), (3, 2), (2, 1)] {
        for base in [
            Expr::from_i64(0),
            Expr::Add(vec![Expr::from_i64(1), Expr::from_i64(-1)]),
            rational_power(Expr::from_i64(0), 1, 3),
        ] {
            let source = rational_power(base, numerator, denominator);
            assert_eq!(source.evalf_ball(24).unwrap(), RealBall::from_i64(0));
            assert_eq!(source.eval_real_ball(24).unwrap(), RealBall::from_i64(0));
            assert_eq!(source.eval_complex_ball(24).unwrap(), ComplexBall::from_i64(0));
        }
    }
}

#[test]
fn positive_zero_powers_compose_with_inverse_circular_endpoint_values() {
    for source in [
        function("asin", rational_power(Expr::from_i64(0), 1, 3)),
        rational_power(function("acos", Expr::from_i64(1)), 2, 3),
    ] {
        assert_eq!(source.evalf_ball(24).unwrap(), RealBall::from_i64(0));
        assert_point(&source.eval_complex_ball(24).unwrap(), q(0, 1), q(0, 1));
    }
    let pi = RealBall::pi(60).unwrap();
    let source = function("acos", rational_power(Expr::from_i64(0), 4, 5));
    let expected = pi.div(&RealBall::from_i64(2)).unwrap();
    assert!(source.evalf_ball(24).unwrap().contains_ball(&expected));
}

#[test]
fn rational_zero_endpoint_does_not_hide_poles_or_change_the_real_power_domain() {
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    for (numerator, denominator) in [(1, 3), (2, 3), (3, 2)] {
        for base in [
            Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
            Expr::Mul(vec![Expr::from_i64(0), function("log", Expr::from_i64(0))]),
        ] {
            let source = rational_power(base, numerator, denominator);
            assert!(source.evalf_ball(24).is_err());
            assert!(source.eval_complex_ball(24).is_err());
        }
        for base in [
            Expr::from_i64(-1),
            Expr::Const(Constant::I),
            Expr::Sym(Symbol::new("zero_power_unbound")),
        ] {
            assert!(
                rational_power(base, numerator, denominator)
                    .evalf_ball(24)
                    .is_err()
            );
        }
        let reciprocal = rational_power(Expr::from_i64(0), -numerator, denominator);
        assert!(reciprocal.evalf_ball(24).is_err());
        assert!(reciprocal.eval_complex_ball(24).is_err());
    }
    // The endpoint rule requires a literal positive rational exponent. An
    // unsupported exponent tree cannot disappear merely because its base is zero.
    let invalid_exponent = Expr::Add(vec![
        Expr::Rational(q(1, 3)),
        Expr::Mul(vec![Expr::from_i64(0), pole]),
    ]);
    let source = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(invalid_exponent));
    assert!(source.evalf_ball(24).is_err());
    assert!(source.eval_complex_ball(24).is_err());
}

#[test]
fn complex_entry_point_keeps_atan2_axes_quadrants_and_argument_order() {
    let pi = RealBall::pi(60).unwrap();
    for (y, x, multiple) in [
        (0, 1, q(0, 1)),
        (0, -1, q(1, 1)),
        (1, 0, q(1, 2)),
        (-1, 0, q(-1, 2)),
        (1, 1, q(1, 4)),
        (1, -1, q(3, 4)),
        (-1, 1, q(-1, 4)),
        (-1, -1, q(-3, 4)),
    ] {
        let source = atan2(Expr::from_i64(y), Expr::from_i64(x));
        let actual = source.eval_complex_ball(24).unwrap();
        let expected = pi.mul(&RealBall::exact(multiple));
        assert!(actual.real().contains_ball(&expected), "{source}: {actual}");
        assert_eq!(actual.imag(), &RealBall::from_i64(0));
        assert!(actual.real().width() < tolerance());
    }
}

#[test]
fn atan2_composes_into_complex_phases_with_exact_pythagorean_values() {
    // exp(I*atan2(y,x)) = (x + I*y)/5 for these exact 3-4-5 triples.
    // Both unequal argument orders and all quadrants are exercised.
    for (y, x) in [(3, 4), (4, 3)] {
        for (ysign, xsign) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let y = y * ysign;
            let x = x * xsign;
            let angle = atan2(Expr::from_i64(y), Expr::from_i64(x));
            let phase = function("exp", Expr::Mul(vec![Expr::Const(Constant::I), angle]));
            assert_point(&phase.eval_complex_ball(24).unwrap(), q(x, 5), q(y, 5));
        }
    }
}

#[test]
fn complex_atan2_preserves_both_original_domains_and_exact_arity() {
    let held_pole = Expr::Mul(vec![Expr::from_i64(0), function("log", Expr::from_i64(0))]);
    for invalid in [
        held_pole,
        Expr::Const(Constant::I),
        Expr::Const(Constant::Infinity),
        Expr::Sym(Symbol::new("atan2_unbound")),
    ] {
        for source in [
            atan2(invalid.clone(), Expr::from_i64(1)),
            atan2(Expr::from_i64(0), invalid),
        ] {
            assert!(source.eval_complex_ball(24).is_err(), "{source}");
        }
    }
    let origin = atan2(Expr::from_i64(0), Expr::from_i64(0));
    assert!(origin.eval_complex_ball(24).is_err());
    for arity in [0, 1, 3] {
        let source = Expr::Function("atan2".into(), vec![Expr::from_i64(1); arity]);
        assert!(source.eval_complex_ball(24).is_err());
    }
    // The explicit atan2 route does not admit arbitrary multiargument functions.
    let unknown = Expr::Function(
        "unregistered_binary".into(),
        vec![Expr::from_i64(1), Expr::from_i64(1)],
    );
    assert!(unknown.eval_complex_ball(24).is_err());
}

#[test]
fn complex_atan2_retains_principal_cut_and_precision_refusals() {
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(10).pow(80));
    let above = atan2(Expr::Rational(tiny.clone()), Expr::from_i64(-1))
        .eval_complex_ball(24)
        .unwrap();
    let below = atan2(Expr::Rational(-tiny), Expr::from_i64(-1))
        .eval_complex_ball(24)
        .unwrap();
    assert!(above.real().lower() > q(3, 1));
    assert!(below.real().upper() < q(-3, 1));
    // A raw sin(pi) enclosure crosses zero. Its uncertain side of the cut
    // cannot be collapsed to the exceptional exact-zero positive-pi value.
    let uncertain = atan2(function("sin", Expr::Const(Constant::Pi)), Expr::from_i64(-1));
    assert!(uncertain.eval_complex_ball(24).is_err());
    let source = atan2(Expr::from_i64(0), Expr::from_i64(1));
    for precision in [0, RealBall::MAX_ATAN2_PRECISION_DIGITS + 1, u32::MAX] {
        assert!(source.eval_complex_ball(precision).is_err());
    }
}
