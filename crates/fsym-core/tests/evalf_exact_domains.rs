//! Binary64 rounding must not choose the domain or parity of an exact source.

use fsym_core::{BigInt, BigRational, Constant, CoreError, Expr};
use std::sync::Arc;

fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}

fn rational(numerator: BigInt, denominator: BigInt) -> Expr {
    Expr::Rational(BigRational::new(numerator, denominator))
}

fn nonreal(expression: Expr) {
    assert!(
        matches!(expression.evalf(), Err(CoreError::InvalidOperation(message))
            if message.contains("not real-valued")),
        "exact nonreal source must not become a real result: {expression}"
    );
}

#[test]
fn noninteger_exponents_do_not_become_integers_after_rounding() {
    let scale = BigInt::from(2).pow(54);
    for integer in [1, 2, 3] {
        for sign in [-1, 1] {
            let exponent = rational(
                (BigInt::from(integer) * &scale + BigInt::from(1)) * BigInt::from(sign),
                scale.clone(),
            );
            nonreal(power(Expr::from_i64(-1), exponent));
        }
    }
    for sign in [-1, 1] {
        nonreal(power(
            Expr::from_i64(-2),
            rational(BigInt::from(sign), BigInt::from(10).pow(1000)),
        ));
    }
}

#[test]
fn large_integer_powers_keep_exact_odd_even_parity() {
    for bits in [53, 54, 100, 4097] {
        let even = BigInt::from(2).pow(bits);
        for offset in [0, 1, 2, 3] {
            for sign in [-1, 1] {
                let integer = (&even + BigInt::from(offset)) * BigInt::from(sign);
                let expected = if offset % 2 == 0 { 1.0 } else { -1.0 };
                for exponent in [
                    Expr::Integer(integer.clone()),
                    rational(integer, BigInt::from(1)),
                ] {
                    assert_eq!(power(Expr::from_i64(-1), exponent).evalf(), Ok(expected));
                }
            }
        }
    }
}

#[test]
fn composed_exponents_are_classified_before_float_cancellation() {
    let large = BigInt::from(2).pow(54);
    let one = Expr::Add(vec![
        Expr::Integer(large.clone()),
        Expr::from_i64(1),
        Expr::Integer(-large),
    ]);
    assert_eq!(power(Expr::from_i64(-2), one.clone()).evalf(), Ok(-2.0));
    let half = Expr::Mul(vec![one, Expr::rational(1, 2).unwrap()]);
    nonreal(power(Expr::from_i64(-2), half));
    let integer = Expr::Mul(vec![
        Expr::rational(3, 2).unwrap(),
        Expr::from_i64(2),
        power(Expr::from_i64(2), Expr::from_i64(-1)),
        Expr::from_i64(2),
    ]);
    assert_eq!(power(Expr::from_i64(-2), integer).evalf(), Ok(-8.0));
}

#[test]
fn underflow_does_not_hide_negative_source_domains() {
    let tiny = rational(BigInt::from(-1), BigInt::from(10).pow(1000));
    for source in [tiny.clone(), Expr::Add(vec![Expr::from_i64(0), tiny])] {
        nonreal(power(source.clone(), Expr::rational(1, 2).unwrap()));
        for function in ["sqrt", "log", "ln"] {
            nonreal(Expr::Function(function.into(), vec![source.clone()]));
        }
    }
}

#[test]
fn zero_base_uses_exact_exponent_sign_even_after_underflow() {
    let scale = BigInt::from(10).pow(1000);
    assert_eq!(
        power(Expr::from_i64(0), rational(BigInt::from(1), scale.clone())).evalf(),
        Ok(0.0)
    );
    assert_eq!(
        power(Expr::from_i64(0), rational(BigInt::from(-1), scale)).evalf(),
        Ok(f64::INFINITY)
    );
    // This approximate convenience API retains its existing evaluated convention.
    assert_eq!(power(Expr::from_i64(0), Expr::from_i64(0)).evalf(), Ok(1.0));
}

#[test]
fn exact_classification_keeps_original_errors_and_nan_semantics() {
    for invalid in [
        power(Expr::from_i64(-1), Expr::rational(1, 2).unwrap()),
        Expr::symbol("unknown"),
        Expr::Function("unimplemented".into(), vec![Expr::from_i64(0)]),
    ] {
        let exponent = Expr::Mul(vec![Expr::from_i64(0), invalid]);
        assert!(power(Expr::from_i64(-2), exponent).evalf().is_err());
    }
    assert!(
        power(Expr::from_i64(-2), Expr::Const(Constant::NaN))
            .evalf()
            .unwrap()
            .is_nan()
    );
    assert_eq!(
        power(
            Expr::from_i64(-2),
            Expr::Function("sqrt".into(), vec![Expr::from_i64(4)]),
        )
        .evalf(),
        Ok(4.0)
    );
}

#[test]
fn unproved_integer_rounding_and_oversized_folds_are_refused() {
    let rounded_to_zero = Expr::Function("exp".into(), vec![Expr::from_i64(-1000)]);
    let error = power(Expr::from_i64(-1), rounded_to_zero)
        .evalf()
        .unwrap_err();
    assert!(matches!(error, CoreError::InvalidOperation(message)
        if message.contains("exact exponent")));
    let oversized = power(Expr::from_i64(2), Expr::from_i64(1_000_000_000));
    assert!(power(Expr::from_i64(-1), oversized).evalf().is_err());
}

#[test]
fn exhausted_exact_classification_cannot_retry_rounded_domain_decisions() {
    let tiny = rational(BigInt::from(-1), BigInt::from(10).pow(1000));
    let mut wide = vec![Expr::from_i64(0); 1024];
    wide.push(tiny.clone());
    let mut deep = tiny;
    for _ in 0..65 {
        deep = Expr::Add(vec![Expr::from_i64(0), deep]);
    }
    let large = BigInt::from(2).pow(4097);
    let oversized = Expr::Add(vec![
        Expr::Integer(large.clone()),
        Expr::from_i64(-1),
        Expr::Integer(-large),
    ]);
    for source in [Expr::Add(wide), deep, oversized] {
        for expression in [
            power(source.clone(), Expr::rational(1, 2).unwrap()),
            Expr::Function("sqrt".into(), vec![source.clone()]),
            Expr::Function("log".into(), vec![source]),
        ] {
            assert!(
                matches!(expression.evalf(), Err(CoreError::InvalidOperation(message))
                if message.contains("classification") && message.contains("limits"))
            );
        }
    }
}

#[test]
fn unknown_rounded_zero_is_not_an_exact_branch_boundary() {
    let underflow = Expr::Function("exp".into(), vec![Expr::from_i64(-1000)]);
    for expression in [
        power(underflow.clone(), Expr::rational(1, 2).unwrap()),
        Expr::Function("sqrt".into(), vec![underflow.clone()]),
        Expr::Function("log".into(), vec![underflow]),
    ] {
        assert!(
            matches!(expression.evalf(), Err(CoreError::InvalidOperation(message))
            if message.contains("exact source sign"))
        );
    }
}

#[test]
fn exact_rational_composite_exponent_corpus_retains_integer_values() {
    for exponent in -8_i32..=8 {
        for denominator in 1_i64..=16 {
            let fraction = Expr::rational(1, denominator).unwrap();
            let source = Expr::Add(vec![
                Expr::from_i64(i64::from(exponent)),
                fraction.clone(),
                Expr::Mul(vec![Expr::from_i64(-1), fraction]),
            ]);
            assert_eq!(
                power(Expr::from_i64(-2), source).evalf(),
                Ok((-2.0_f64).powi(exponent))
            );
        }
    }
}
