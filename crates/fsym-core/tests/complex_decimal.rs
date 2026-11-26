//! Component rounding, principal branches and original-source admission.
//! Decimal transcendental fixtures were checked with isolated SymPy 1.14.0;
//! rational ties and Gaussian square roots have independent exact witnesses.

use fsym_core::{BigInt, BigRational, ComplexDecimalError, Constant, Expr};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> Expr {
    Expr::Rational(BigRational::new(numerator.into(), denominator.into()))
}

fn point(real: Expr, imag: Expr) -> Expr {
    Expr::Add(vec![real, Expr::Mul(vec![imag, Expr::Const(Constant::I)])])
}

fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

#[test]
fn exact_components_use_independent_half_even_rounding() {
    for (real, imag, expected_real, expected_imag) in [
        (q(5, 4), q(27, 20), "1.2e0", "1.4e0"),
        (q(-5, 4), q(-27, 20), "-1.2e0", "-1.4e0"),
        (q(199, 20), q(201, 20), "1.0e1", "1.0e1"),
        (q(0, 1), q(1, 8), "0", "1.2e-1"),
    ] {
        assert_eq!(
            point(real, imag).evalf_complex_decimal(2).unwrap(),
            (expected_real.into(), expected_imag.into()),
        );
    }
    assert_eq!(
        point(q(1, 3), q(-2, 7)).evalf_complex_decimal(30).unwrap(),
        (
            "3.33333333333333333333333333333e-1".into(),
            "-2.85714285714285714285714285714e-1".into(),
        ),
    );
}

#[test]
fn exact_gaussian_roots_and_reciprocals_retain_zero_components() {
    // (2+I)^2=3+4I, and (1+I)*(1-I)/2=1.
    for (source, expected) in [
        (
            function("sqrt", point(q(3, 1), q(4, 1))),
            ("2.0000000000000000000e0", "1.0000000000000000000e0"),
        ),
        (function("sqrt", q(-4, 1)), ("0", "2.0000000000000000000e0")),
        (
            power(point(q(1, 1), q(1, 1)), q(-1, 1)),
            ("5.0000000000000000000e-1", "-5.0000000000000000000e-1"),
        ),
        (function("log", q(1, 1)), ("0", "0")),
    ] {
        assert_eq!(
            source.evalf_complex_decimal(20).unwrap(),
            (expected.0.into(), expected.1.into()),
        );
    }
}

#[test]
fn principal_logarithms_keep_both_certified_components() {
    for (source, real, imag) in [
        (
            function("log", point(q(1, 1), q(1, 1))),
            "3.465735902799726547086160607290882840378e-1",
            "7.853981633974483096156608458198757210493e-1",
        ),
        (
            function("log", q(-2, 1)),
            "6.931471805599453094172321214581765680755e-1",
            "3.141592653589793238462643383279502884197e0",
        ),
        (
            function("log", point(q(0, 1), q(-1, 1))),
            "0",
            "-1.570796326794896619231321691639751442099e0",
        ),
    ] {
        assert_eq!(
            source.evalf_complex_decimal(40).unwrap(),
            (real.into(), imag.into()),
        );
    }
}

#[test]
fn general_powers_keep_the_principal_complex_branch() {
    assert_eq!(
        power(q(-8, 1), q(1, 3)).evalf_complex_decimal(40).unwrap(),
        (
            "1.000000000000000000000000000000000000000e0".into(),
            "1.732050807568877293527446341505872366943e0".into(),
        ),
    );
    assert_eq!(
        power(point(q(1, 1), q(1, 1)), point(q(2, 1), q(1, 1)))
            .evalf_complex_decimal(40)
            .unwrap(),
        (
            "-3.097435049284935563846197793596651626076e-1".into(),
            "8.576580125887356986453040141946709992235e-1".into(),
        ),
    );
}

#[test]
fn nonpoint_intermediate_enclosures_survive_complex_functions() {
    assert_eq!(
        function(
            "sin",
            Expr::Mul(vec![Expr::Const(Constant::I), Expr::Const(Constant::Pi)]),
        )
        .evalf_complex_decimal(40)
        .unwrap(),
        (
            "0".into(),
            "1.154873935725774837797733431538840968450e1".into(),
        ),
    );
    assert_eq!(
        function("exp", point(function("sqrt", q(2, 1)), q(1, 3)))
            .evalf_complex_decimal(40)
            .unwrap(),
        (
            "3.886844517362653200830073535400820772902e0".into(),
            "1.345833710532538338886530237309876372002e0".into(),
        ),
    );
}

#[test]
fn tiny_components_are_never_chopped_or_converted_through_binary64() {
    let tiny = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(10).pow(400)));
    assert_eq!(
        point(q(1, 1), tiny).evalf_complex_decimal(30).unwrap(),
        (
            "1.00000000000000000000000000000e0".into(),
            "1.00000000000000000000000000000e-400".into(),
        ),
    );
    let cancellation = Expr::Add(vec![
        Expr::Const(Constant::Pi),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Const(Constant::Pi)]),
    ]);
    assert!(matches!(
        cancellation.evalf_complex_decimal(10),
        Err(ComplexDecimalError::RoundingInconclusive(10)),
    ));
}

#[test]
fn precision_bounds_are_checked_before_source_evaluation() {
    for digits in [0, 1001, u32::MAX] {
        assert_eq!(
            function("log", Expr::from_i64(0)).evalf_complex_decimal(digits),
            Err(ComplexDecimalError::InvalidPrecision(digits)),
        );
    }
    let (real, imag) = point(q(1, 8), q(0, 1)).evalf_complex_decimal(1000).unwrap();
    assert_eq!(real.len(), 1004);
    assert!(real.starts_with("1.25000"));
    assert!(real.ends_with("e-1"));
    assert_eq!(imag, "0");
}

#[test]
fn original_undefined_children_cannot_be_erased_by_simple_values() {
    let pole = power(q(0, 1), q(-1, 1));
    let undefined = power(q(0, 1), q(0, 1));
    for source in [
        Expr::Mul(vec![q(0, 1), pole.clone()]),
        power(pole.clone(), q(0, 1)),
        power(q(1, 1), pole),
        power(Expr::Const(Constant::I), undefined),
        function("log", q(0, 1)),
    ] {
        assert!(matches!(
            source.evalf_complex_decimal(25),
            Err(ComplexDecimalError::Enclosure(_)),
        ));
    }
}

#[test]
fn unsupported_grammar_stays_distinct_from_domain_and_resource_failures() {
    for source in [
        Expr::symbol("x"),
        Expr::Const(Constant::Infinity),
        function("unknown", q(1, 1)),
        function("asin", q(1, 2)),
        Expr::Function("sin".into(), vec![q(1, 1), q(2, 1)]),
    ] {
        assert!(matches!(
            source.evalf_complex_decimal(25),
            Err(ComplexDecimalError::UnsupportedSource(_)),
        ));
    }
    let huge = Expr::Integer(BigInt::from(1) << 16384u32);
    let mut deep = q(1, 1);
    for _ in 0..64 {
        deep = function("conjugate", deep);
    }
    for source in [
        Expr::Add(vec![q(0, 1); 4096]),
        deep,
        huge.clone(),
        Expr::Add(vec![Expr::symbol("x"), huge]),
        Expr::Add(vec![Expr::Integer(BigInt::from(1) << 15000u32); 5]),
    ] {
        assert!(matches!(
            source.evalf_complex_decimal(25),
            Err(ComplexDecimalError::SourceLimitExceeded { .. }),
        ));
    }
}
