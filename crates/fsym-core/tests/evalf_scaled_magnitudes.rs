//! Exact finite sources can leave binary64 range before their results do.

use fsym_core::{BigInt, BigRational, Constant, CoreError, Expr};
use std::sync::Arc;

fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}

fn rational(numerator: BigInt, denominator: BigInt) -> Expr {
    Expr::Rational(BigRational::new(numerator, denominator))
}

fn decimal_power(exponent: i32) -> Expr {
    let coefficient = BigInt::from(10).pow(exponent.unsigned_abs());
    if exponent >= 0 {
        Expr::Integer(coefficient)
    } else {
        rational(BigInt::from(1), coefficient)
    }
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn close(expression: Expr, expected: f64) {
    let actual = expression.evalf().unwrap();
    assert!(
        actual.is_finite(),
        "{expression}: expected {expected}, got {actual}"
    );
    assert!(
        (actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs(),
        "{expression}: expected {expected}, got {actual}"
    );
}

#[test]
fn rational_roots_recover_decimal_values_outside_binary64_input_range() {
    for sign in [-1, 1] {
        let source = decimal_power(sign * 1000);
        close(
            power(source.clone(), Expr::rational(1, 1000).unwrap()),
            if sign < 0 { 0.1 } else { 10.0 },
        );
        close(
            power(source, Expr::rational(-1, 1000).unwrap()),
            if sign < 0 { 10.0 } else { 0.1 },
        );
    }
    for sign in [-1, 1] {
        let coefficient = BigInt::from(2).pow(2048);
        let base = if sign < 0 {
            rational(BigInt::from(1), coefficient)
        } else {
            Expr::Integer(coefficient)
        };
        assert_eq!(
            power(base, Expr::rational(1, 2048).unwrap()).evalf(),
            Ok(if sign < 0 { 0.5 } else { 2.0 })
        );
    }
}

#[test]
fn square_roots_preserve_representable_extreme_results() {
    for (exponent, expected) in [(400, 1.0e200), (-600, 1.0e-300)] {
        let source = decimal_power(exponent);
        close(function("sqrt", source.clone()), expected);
        close(power(source, Expr::rational(1, 2).unwrap()), expected);
    }
    assert_eq!(function("sqrt", decimal_power(-1000)).evalf(), Ok(0.0));
    assert_eq!(
        function("sqrt", Expr::Integer(BigInt::from(2).pow(2048))).evalf(),
        Ok(f64::INFINITY)
    );
    // The largest finite float is an exact integer; squaring its exact source
    // must not make the subsequent square root spuriously overflow.
    let maximum = (BigInt::from(2).pow(53) - BigInt::from(1)) * BigInt::from(2).pow(971);
    let squared = Expr::Integer(&maximum * &maximum);
    assert_eq!(function("sqrt", squared.clone()).evalf(), Ok(f64::MAX));
    assert_eq!(
        power(squared, Expr::rational(1, 2).unwrap()).evalf(),
        Ok(f64::MAX)
    );
}

#[test]
fn logarithms_recover_finite_values_for_exact_extreme_sources() {
    for name in ["log", "ln"] {
        for sign in [-1, 1] {
            close(
                function(name, decimal_power(sign * 1000)),
                f64::from(sign) * 1000.0 * std::f64::consts::LN_10,
            );
        }
        for sign in [-1, 1] {
            let coefficient = BigInt::from(2).pow(2048);
            let source = if sign < 0 {
                rational(BigInt::from(1), coefficient)
            } else {
                Expr::Integer(coefficient)
            };
            close(
                function(name, source),
                f64::from(sign) * 2048.0 * std::f64::consts::LN_2,
            );
        }
    }
}

#[test]
fn exact_composite_sources_recover_before_rounding() {
    let tiny = Expr::Add(vec![Expr::from_i64(0), decimal_power(-1000)]);
    close(power(tiny, Expr::rational(1, 1000).unwrap()), 0.1);
    let large = Expr::Mul(vec![decimal_power(200), decimal_power(200)]);
    close(function("sqrt", large), 1.0e200);
    let large = power(Expr::from_i64(10), Expr::from_i64(1000));
    close(function("ln", large), 1000.0 * std::f64::consts::LN_10);
}

#[test]
fn subnormal_square_roots_round_only_after_binary_scaling() {
    // sqrt(n / 2^2156) lies below, at, and above half the minimum subnormal.
    // The tie rounds to even zero; the upper neighbor rounds to 2^-1074.
    for (numerator, expected_bits) in [(49, 0), (64, 0), (81, 1)] {
        let source = rational(BigInt::from(numerator), BigInt::from(2).pow(2156));
        for expression in [
            function("sqrt", source.clone()),
            power(source, Expr::rational(1, 2).unwrap()),
        ] {
            assert_eq!(expression.evalf().unwrap().to_bits(), expected_bits);
        }
    }
    // Both 1.5 and 2.5 minimum-subnormal units tie to the even value 2.
    for numerator in [9, 25] {
        let source = rational(BigInt::from(numerator), BigInt::from(2).pow(2150));
        for expression in [
            function("sqrt", source.clone()),
            power(source, Expr::rational(1, 2).unwrap()),
        ] {
            assert_eq!(expression.evalf().unwrap().to_bits(), 2);
        }
    }
}

#[test]
fn extreme_integer_powers_keep_sign_zero_and_saturation() {
    for source_exponent in [-1000, 1000] {
        let negative = Expr::Mul(vec![Expr::from_i64(-1), decimal_power(source_exponent)]);
        for exponent in [-3, -2, 0, 2, 3] {
            let actual = power(negative.clone(), Expr::from_i64(exponent))
                .evalf()
                .unwrap();
            assert!(!actual.is_nan());
            if exponent == 0 {
                assert_eq!(actual, 1.0);
            } else {
                let underflows = (source_exponent < 0) == (exponent > 0);
                assert_eq!(actual.abs(), if underflows { 0.0 } else { f64::INFINITY });
                assert_eq!(actual.is_sign_negative(), exponent % 2 != 0);
            }
        }
        assert!(matches!(
            power(negative, Expr::rational(1, 1000).unwrap()).evalf(),
            Err(CoreError::InvalidOperation(message)) if message.contains("not real-valued")
        ));
    }
    for exponent in [-1, 0, 1] {
        assert_eq!(
            power(Expr::from_i64(0), Expr::from_i64(exponent)).evalf(),
            Ok(match exponent {
                -1 => f64::INFINITY,
                0 => 1.0,
                _ => 0.0,
            })
        );
    }
}

#[test]
fn ordinary_magnitudes_and_explicit_nan_keep_existing_conventions() {
    for (base, numerator, denominator, expected) in
        [(9, 1, 2, 3.0), (2, 10, 1, 1024.0), (4, -1, 2, 0.5)]
    {
        assert_eq!(
            power(
                Expr::from_i64(base),
                Expr::rational(numerator, denominator).unwrap()
            )
            .evalf(),
            Ok(expected)
        );
    }
    assert!(
        power(decimal_power(1000), Expr::Const(Constant::NaN))
            .evalf()
            .unwrap()
            .is_nan()
    );
    assert_eq!(function("ln", Expr::from_i64(1)).evalf(), Ok(0.0));
    assert_eq!(
        function("ln", Expr::from_i64(0)).evalf(),
        Ok(f64::NEG_INFINITY)
    );
    assert_eq!(function("sqrt", Expr::from_i64(0)).evalf(), Ok(0.0));
    for name in ["sqrt", "log", "ln"] {
        assert_eq!(
            function(name, Expr::Const(Constant::Infinity)).evalf(),
            Ok(f64::INFINITY)
        );
    }
    for (exponent, expected) in [(-1, 0.0), (0, 1.0), (1, f64::INFINITY)] {
        assert_eq!(
            power(Expr::Const(Constant::Infinity), Expr::from_i64(exponent)).evalf(),
            Ok(expected)
        );
    }
    for source_exponent in [-1000, 1000] {
        for negative_exponent in [false, true] {
            let exponent = Expr::Const(if negative_exponent {
                Constant::NegativeInfinity
            } else {
                Constant::Infinity
            });
            let grows = (source_exponent > 0) != negative_exponent;
            assert_eq!(
                power(decimal_power(source_exponent), exponent).evalf(),
                Ok(if grows { f64::INFINITY } else { 0.0 })
            );
        }
    }
}

#[test]
fn recovery_cannot_erase_source_obligations_or_exceed_its_bound() {
    for invalid in [
        Expr::symbol("unbound"),
        function("unknown", Expr::from_i64(0)),
    ] {
        let source = Expr::Mul(vec![decimal_power(1000), Expr::from_i64(0), invalid]);
        assert!(function("sqrt", source.clone()).evalf().is_err());
        assert!(
            power(source, Expr::rational(1, 1000).unwrap())
                .evalf()
                .is_err()
        );
    }
    let too_large = Expr::Integer(BigInt::from(2).pow(5000));
    for expression in [
        function("sqrt", too_large.clone()),
        function("log", too_large.clone()),
        power(too_large, Expr::rational(1, 5000).unwrap()),
    ] {
        assert!(
            matches!(expression.evalf(), Err(CoreError::InvalidOperation(message))
            if message.contains("limits") || message.contains("bounded"))
        );
    }
    let nonrational = Expr::Mul(vec![
        Expr::Const(Constant::Pi),
        Expr::rational(1, 10000).unwrap(),
    ]);
    assert!(power(decimal_power(1000), nonrational).evalf().is_err());
}
