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

fn half_power(base: Expr, numerator: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::Rational(q(numerator, 2))))
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
