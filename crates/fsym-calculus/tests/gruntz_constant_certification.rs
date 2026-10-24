//! Exact/certified constant decisions at the limit and definite-integral boundary.

use fsym_calculus::gruntz::{Direction, constant_sign, is_zero_coefficient, limit};
use fsym_calculus::integrate::definite;
use fsym_calculus::transforms::integrate_definite;
use fsym_core::elementary::eval_pow;
use fsym_core::{Constant, Expr, Symbol};

fn function(name: &str, arg: Expr) -> Expr {
    Expr::Function(name.to_owned(), vec![arg])
}

fn almost_real_constant() -> Expr {
    // The outer Add, imaginary Mul, and residual Add consume twelve
    // guard digits. Choose the rational midpoint of the matching E ball.
    let e = Expr::Const(Constant::E);
    let approximation = e.eval_complex_ball(30).unwrap().real().midpoint().clone();
    Expr::from_i64(1) + Expr::Const(Constant::I) * (e - Expr::Rational(approximation))
}

#[test]
fn an_imaginary_enclosure_containing_zero_does_not_prove_a_real_sign() {
    let coefficient = almost_real_constant();
    let coarse = coefficient.eval_complex_ball(18).unwrap();
    assert!(coarse.real().is_positive());
    assert!(coarse.imag().contains_zero());
    assert!(
        !coefficient
            .eval_complex_ball(96)
            .unwrap()
            .imag()
            .contains_zero()
    );
    assert_eq!(constant_sign(&coefficient), None);
}

#[test]
fn a_nonreal_coefficient_cannot_publish_a_real_infinite_endpoint() {
    let x = Symbol::new("x");
    let coefficient = almost_real_constant();
    let infinity = Expr::Const(Constant::Infinity);
    let expression = coefficient.clone() * Expr::Sym(x.clone());
    assert!(limit(&expression, &x, &infinity, Direction::Plus).is_err());
    assert!(definite(&coefficient, &x, &Expr::from_i64(0), &infinity).is_none());
    assert!(integrate_definite(&coefficient, &x, &Expr::from_i64(0), &infinity).is_err());
}

#[test]
fn floating_rounding_of_an_exact_identity_cannot_decide_an_infinite_limit() {
    let x = Symbol::new("x");
    let one = Expr::from_i64(1);
    let two = Expr::from_i64(2);
    // cos(2) - cos(1)^2 + sin(1)^2 = 0 exactly. Its rounded complex-f64
    // evaluation can be separated from zero after this exact amplification.
    let identity = function("cos", two.clone())
        - eval_pow(function("cos", one.clone()), two.clone())
        + eval_pow(function("sin", one), two);
    let coefficient = Expr::from_i64(1_000_000_000_000) * identity;
    assert!(coefficient.eval_complex_ball(24).unwrap().contains_zero());
    let infinity = Expr::Const(Constant::Infinity);
    if let Ok(value) = limit(
        &(coefficient.clone() * Expr::Sym(x.clone())),
        &x,
        &infinity,
        Direction::Plus,
    ) {
        assert!(
            value.is_zero(),
            "a proved-zero or unresolved coefficient yielded {value}"
        );
    }
    if let Some(value) = definite(&coefficient, &x, &Expr::from_i64(0), &infinity) {
        assert!(
            value.is_zero(),
            "a proved-zero or unresolved integrand yielded {value}"
        );
    }
    assert!(matches!(constant_sign(&coefficient), None | Some(0)));
    assert_ne!(is_zero_coefficient(&coefficient), Some(false));
}

#[test]
fn an_unevaluable_ground_coefficient_is_unknown_instead_of_nonzero() {
    let coefficient = function("uninterpreted", Expr::from_i64(0));
    assert_eq!(is_zero_coefficient(&coefficient), None);
}

#[test]
fn unknown_symbolic_coefficients_need_declared_nonzero_or_sign_facts() {
    use fsym_core::assume::{Facts, with_symbol_facts};
    use std::collections::HashMap;

    let a = Symbol::new("a");
    let coefficient = Expr::Sym(a.clone());
    assert_eq!(is_zero_coefficient(&coefficient), None);
    assert_eq!(constant_sign(&coefficient), None);
    for (fact, expected_sign) in [
        ("positive", Some(1)),
        ("negative", Some(-1)),
        ("nonzero", None),
    ] {
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&[fact]))]),
            || {
                assert_eq!(is_zero_coefficient(&coefficient), Some(false));
                assert_eq!(constant_sign(&coefficient), expected_sign);
            },
        );
    }
    assert_eq!(is_zero_coefficient(&coefficient), None);
}

#[test]
fn exact_zero_and_certified_real_signs_remain_available() {
    let a = Expr::symbol("a");
    let one = Expr::from_i64(1);
    let two = Expr::from_i64(2);
    let identity =
        eval_pow(a.clone() + one.clone(), two.clone()) - (a.clone() * a.clone() + two * a + one);
    assert_eq!(is_zero_coefficient(&identity), Some(true));
    assert_eq!(constant_sign(&identity), Some(0));

    let imaginary_square = Expr::Pow(
        std::sync::Arc::new(Expr::Const(Constant::I)),
        std::sync::Arc::new(Expr::from_i64(2)),
    );
    assert_eq!(constant_sign(&imaginary_square), Some(-1));
    assert_eq!(is_zero_coefficient(&imaginary_square), Some(false));
    assert_eq!(constant_sign(&function("sin", Expr::from_i64(1))), Some(1));
    assert_eq!(constant_sign(&function("log", Expr::from_i64(2))), Some(1));
    assert_eq!(constant_sign(&Expr::Const(Constant::Infinity)), Some(1));
    assert_eq!(
        constant_sign(&Expr::Const(Constant::NegativeInfinity)),
        Some(-1)
    );
    for infinity in [Constant::Infinity, Constant::NegativeInfinity] {
        assert_eq!(is_zero_coefficient(&Expr::Const(infinity)), None);
    }
}

#[test]
fn refinement_can_prove_a_tiny_constant_nonzero() {
    let e = Expr::Const(Constant::E);
    let approximation = e.eval_complex_ball(28).unwrap().real().midpoint().clone();
    let residual = e - Expr::Rational(approximation);
    assert!(residual.eval_complex_ball(24).unwrap().contains_zero());
    let refined = residual.eval_complex_ball(96).unwrap();
    assert!(!refined.contains_zero());
    assert_eq!(is_zero_coefficient(&residual), Some(false));
    let sign = if refined.real().is_positive() { 1 } else { -1 };
    assert_eq!(constant_sign(&residual), Some(sign));
}

#[test]
fn normalization_does_not_erase_undefined_original_children() {
    use std::sync::Arc;
    let undefined = function("log", Expr::from_i64(0));
    let held = [
        Expr::Mul(vec![Expr::from_i64(0), undefined.clone()]),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Const(Constant::Infinity)]),
        Expr::Pow(Arc::new(undefined.clone()), Arc::new(Expr::from_i64(0))),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(0))),
            Expr::symbol("a"),
        ]),
        Expr::Pow(Arc::new(Expr::symbol("a")), Arc::new(Expr::from_i64(0))),
        Expr::Add(vec![
            Expr::symbol("a"),
            Expr::Mul(vec![Expr::from_i64(0), undefined]),
        ]),
    ];
    for coefficient in held {
        assert_eq!(constant_sign(&coefficient), None, "{coefficient}");
        assert_eq!(is_zero_coefficient(&coefficient), None, "{coefficient}");
    }
}

#[test]
fn coefficient_shape_is_bounded_before_recursive_work() {
    let mut coefficient = Expr::from_i64(0);
    for _ in 0..32 {
        coefficient = Expr::Add(vec![coefficient]);
    }
    assert_eq!(constant_sign(&coefficient), Some(0));
    assert_eq!(is_zero_coefficient(&coefficient), Some(true));
    let excessive_depth = Expr::Add(vec![coefficient]);
    assert_eq!(constant_sign(&excessive_depth), None);
    assert_eq!(is_zero_coefficient(&excessive_depth), None);

    let excessive_width = Expr::Add(vec![Expr::from_i64(0); 8_192]);
    assert_eq!(constant_sign(&excessive_width), None);
    assert_eq!(is_zero_coefficient(&excessive_width), None);
}

#[test]
fn positive_limit_dummy_does_not_capture_an_existing_parameter() {
    use fsym_core::assume::{Facts, with_symbol_facts};
    use std::collections::HashMap;

    let x = Symbol::new("x");
    let parameter = Symbol::new("_fsym_limit_x");
    let coefficient = Expr::Sym(parameter.clone());
    let expression = coefficient.clone() * Expr::Sym(x.clone());
    let infinity = Expr::Const(Constant::Infinity);
    assert!(limit(&expression, &x, &infinity, Direction::Plus).is_err());
    let negative = with_symbol_facts(
        HashMap::from([(parameter, Facts::from_names(&["negative"]))]),
        || limit(&expression, &x, &infinity, Direction::Plus),
    );
    assert_eq!(negative.unwrap(), Expr::Const(Constant::NegativeInfinity));
    assert_eq!(constant_sign(&coefficient), None);
    assert_eq!(is_zero_coefficient(&coefficient), None);
}

#[test]
fn complex_oscillator_growth_is_not_a_decaying_product() {
    let x = Symbol::new("x");
    let variable = Expr::Sym(x.clone());
    let imaginary_argument = Expr::Const(Constant::I) * variable.clone();
    let decay = function("exp", -variable.clone());
    let half = Expr::Rational(fsym_core::BigRational::new(1.into(), 2.into()));
    let infinity = Expr::Const(Constant::Infinity);
    for (name, expected) in [
        ("cos", half.clone()),
        ("sin", half * Expr::Const(Constant::I)),
    ] {
        let expression = decay.clone() * function(name, imaginary_argument.clone());
        let value = limit(&expression, &x, &infinity, Direction::Plus).unwrap();
        assert_eq!(fsym_simplify::simplify(&value), expected);
    }
    let quotient = function("sin", imaginary_argument) * eval_pow(variable, Expr::from_i64(-1));
    let result = limit(&quotient, &x, &infinity, Direction::Plus);
    assert!(matches!(
        result,
        Err(fsym_calculus::gruntz::LimitError::Undetermined(_))
    ));
}

#[test]
fn mixed_complex_oscillator_growth_is_not_bounded() {
    let x = Symbol::new("x");
    let variable = Expr::Sym(x.clone());
    // Dividing by x cancels the exponential growth caused by Im(z)=log(x),
    // leaving a nonvanishing oscillatory term exp(-I*x)/2 (or I times it).
    let argument = variable.clone() + Expr::Const(Constant::I) * function("log", variable.clone());
    let reciprocal = eval_pow(variable, Expr::from_i64(-1));
    for name in ["cos", "sin"] {
        let expression = function(name, argument.clone()) * reciprocal.clone();
        let result = limit(
            &expression,
            &x,
            &Expr::Const(Constant::Infinity),
            Direction::Plus,
        );
        assert!(
            result.is_err(),
            "a nonzero-amplitude oscillation has no limit"
        );
    }
}

#[test]
fn real_oscillator_bounds_use_real_parameter_facts() {
    use fsym_core::assume::{Facts, with_symbol_facts};
    use std::collections::HashMap;

    let x = Symbol::new("x");
    let variable = Expr::Sym(x.clone());
    let reciprocal = eval_pow(variable.clone(), Expr::from_i64(-1));
    let infinity = Expr::Const(Constant::Infinity);
    let real_sine = function("sin", variable.clone()) * reciprocal.clone();
    assert_eq!(
        limit(&real_sine, &x, &infinity, Direction::Plus).unwrap(),
        Expr::from_i64(0)
    );
    let nested_sine = function("sin", function("sin", variable.clone())) * reciprocal.clone();
    assert_eq!(
        limit(&nested_sine, &x, &infinity, Direction::Plus).unwrap(),
        Expr::from_i64(0)
    );
    let parameter = Symbol::new("a");
    let argument = Expr::Sym(parameter.clone()) * variable.clone();
    for (name, decay) in [
        ("sin", reciprocal.clone()),
        ("cos", function("exp", -variable.clone())),
    ] {
        let expression = function(name, argument.clone()) * decay;
        assert!(limit(&expression, &x, &infinity, Direction::Plus).is_err());
        let result = with_symbol_facts(
            HashMap::from([(parameter.clone(), Facts::from_names(&["real"]))]),
            || limit(&expression, &x, &infinity, Direction::Plus),
        );
        assert_eq!(result.unwrap(), Expr::from_i64(0));
    }

    // Use held nodes so source-domain obligations reach the limit API.
    let reciprocal_zero = Expr::Pow(
        std::sync::Arc::new(Expr::from_i64(0)),
        std::sync::Arc::new(Expr::from_i64(-1)),
    );
    let zero_power = Expr::Pow(
        std::sync::Arc::new(Expr::from_i64(0)),
        std::sync::Arc::new(Expr::from_i64(0)),
    );
    for argument in [
        reciprocal_zero.clone(),
        Expr::Mul(vec![variable.clone(), zero_power]),
        Expr::Add(vec![
            variable,
            Expr::Mul(vec![Expr::from_i64(0), reciprocal_zero]),
        ]),
    ] {
        let sine = function("sin", argument);
        let expression = Expr::Mul(vec![sine.clone(), reciprocal.clone()]);
        let held_factor = Expr::Mul(vec![
            Expr::Pow(
                std::sync::Arc::new(sine),
                std::sync::Arc::new(Expr::from_i64(1)),
            ),
            reciprocal.clone(),
        ]);
        let held_expression = Expr::Pow(
            std::sync::Arc::new(expression.clone()),
            std::sync::Arc::new(Expr::from_i64(1)),
        );
        for source in [expression, held_factor, held_expression] {
            for (point, direction) in [
                (infinity.clone(), Direction::Plus),
                (Expr::Const(Constant::NegativeInfinity), Direction::Plus),
                (Expr::from_i64(0), Direction::Plus),
                (Expr::from_i64(0), Direction::Minus),
            ] {
                assert!(
                    limit(&source, &x, &point, direction).is_err(),
                    "undefined source {source} at {point}"
                );
            }
        }
    }
}
