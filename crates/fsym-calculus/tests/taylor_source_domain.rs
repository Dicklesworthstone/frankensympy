//! Pointwise Taylor coefficients must retain the original source domain.
//!
//! Formal Laurent series may describe a punctured neighborhood. A Taylor
//! polynomial also needs a defined value at its center; cancellation, zero
//! factors, and zeroth powers cannot supply that missing value.

use fsym_calculus::{CalculusError, series_expansion, taylor};
use fsym_core::{BigRational, Constant, Expr, Symbol, SymbolIdentity};
use fsym_simplify::{try_expand, try_simplify};
use std::sync::Arc;

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn assert_refused(expression: &Expr, variable: &Symbol, center: &Expr, degree: usize) {
    assert!(
        matches!(
            taylor(expression, variable, center, degree),
            Err(CalculusError::NonDifferentiable(_))
        ),
        "Taylor coefficients must refuse {expression} at {center}, degree {degree}"
    );
}

fn assert_equal(actual: Expr, expected: Expr) {
    let residual = try_expand(&(actual.clone() - expected.clone())).unwrap();
    assert!(
        try_simplify(&residual).unwrap().is_zero(),
        "expected {expected}, got {actual}"
    );
}

#[test]
fn nonrational_composition_cannot_erase_original_removable_holes() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let exponential = function("exp", xe.clone());
    let expressions = [
        Expr::Mul(vec![xe.clone(), power(xe.clone(), -1), exponential.clone()]),
        Expr::Mul(vec![power(power(xe.clone(), -1), -1), exponential.clone()]),
        Expr::Add(vec![
            exponential.clone(),
            Expr::Mul(vec![Expr::from_i64(0), power(xe.clone(), -1)]),
        ]),
        Expr::Mul(vec![exponential, power(power(xe, -1), 0)]),
    ];
    for expression in expressions {
        for degree in [0, 1, 4] {
            assert_refused(&expression, &x, &Expr::from_i64(0), degree);
        }
        assert!(taylor(&expression, &x, &Expr::from_i64(1), 3).is_ok());
    }
}

#[test]
fn logarithmic_poles_are_checked_before_parent_normalization() {
    let x = Symbol::new("x");
    let log_x = function("log", Expr::Sym(x.clone()));
    for expression in [
        log_x.clone(),
        function("exp", log_x.clone()),
        Expr::Mul(vec![Expr::from_i64(0), log_x.clone()]),
        power(log_x.clone(), 0),
        Expr::Add(vec![
            log_x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), log_x]),
            function("exp", Expr::Sym(x.clone())),
        ]),
        Expr::Mul(vec![
            Expr::from_i64(0),
            function(
                "ln",
                Expr::Add(vec![Expr::from_i64(-1), Expr::Sym(x.clone())]),
            ),
        ]),
    ] {
        let center = if matches!(&expression, Expr::Mul(xs) if xs.iter().any(|child| matches!(child, Expr::Function(name, _) if name == "ln")))
        {
            Expr::from_i64(1)
        } else {
            Expr::from_i64(0)
        };
        assert_refused(&expression, &x, &center, 0);
        assert_refused(&expression, &x, &center, 3);
    }
}

#[test]
fn undefined_constant_children_and_nonfinite_centers_refuse() {
    let x = Symbol::new("x");
    let zero = Expr::from_i64(0);
    let bad_log = function("log", zero.clone());
    for child in [
        bad_log.clone(),
        function("exp", bad_log.clone()),
        power(bad_log.clone(), 0),
        Expr::Mul(vec![zero.clone(), bad_log]),
        power(power(zero.clone(), -1), -1),
    ] {
        let expression = Expr::Add(vec![function("exp", Expr::Sym(x.clone())), child.clone()]);
        assert_refused(&expression, &x, &zero, 0);
        assert_refused(&Expr::from_i64(7), &x, &child, 0);
    }
    for center in [
        Expr::Const(Constant::Infinity),
        Expr::Const(Constant::NegativeInfinity),
        Expr::Const(Constant::ComplexInfinity),
        Expr::Const(Constant::NaN),
    ] {
        assert_refused(&Expr::from_i64(7), &x, &center, 0);
        assert_refused(&function("exp", Expr::Sym(x.clone())), &x, &center, 2);
    }
}

#[test]
fn finite_analytic_compositions_keep_exact_coefficients() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let zero = Expr::from_i64(0);
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let sixth = Expr::Rational(BigRational::new(1.into(), 6.into()));
    let log_one_plus_x = function("log", Expr::Add(vec![Expr::from_i64(1), xe.clone()]));
    assert_equal(
        taylor(&log_one_plus_x, &x, &zero, 3).unwrap(),
        xe.clone() - half.clone() * power(xe.clone(), 2)
            + Expr::Rational(BigRational::new(1.into(), 3.into())) * power(xe.clone(), 3),
    );
    assert_equal(
        taylor(&function("exp", xe.clone()), &x, &zero, 3).unwrap(),
        Expr::from_i64(1) + xe.clone() + half * power(xe.clone(), 2) + sixth * power(xe.clone(), 3),
    );
    assert_equal(
        taylor(
            &Expr::Mul(vec![power(xe.clone(), 2), function("exp", xe.clone())]),
            &x,
            &zero,
            3,
        )
        .unwrap(),
        power(xe.clone(), 2) + power(xe.clone(), 3),
    );
    assert_equal(
        taylor(
            &Expr::Mul(vec![xe.clone(), function("sin", xe.clone())]),
            &x,
            &zero,
            4,
        )
        .unwrap(),
        power(xe.clone(), 2)
            - Expr::Rational(BigRational::new(1.into(), 6.into())) * power(xe.clone(), 4),
    );
    // A removable rational factor is valid away from its original hole.
    let with_hole = Expr::Mul(vec![xe.clone(), power(xe.clone(), -1), function("exp", xe)]);
    assert_equal(
        taylor(&with_hole, &x, &Expr::from_i64(1), 2).unwrap(),
        taylor(
            &function("exp", Expr::Sym(x.clone())),
            &x,
            &Expr::from_i64(1),
            2,
        )
        .unwrap(),
    );
}

#[test]
fn inactive_piecewise_branches_do_not_invent_point_domain_failures() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let exponential = function("exp", xe.clone());
    let expression = Expr::Function(
        "Piecewise".into(),
        vec![
            Expr::Function(
                "Tuple".into(),
                vec![
                    power(xe.clone() - Expr::from_i64(1), -1),
                    Expr::Function("StrictLessThan".into(), vec![xe, Expr::from_i64(0)]),
                ],
            ),
            Expr::Function(
                "Tuple".into(),
                vec![exponential.clone(), Expr::symbol("True")],
            ),
        ],
    );
    assert_equal(
        taylor(&expression, &x, &Expr::from_i64(1), 3).unwrap(),
        taylor(&exponential, &x, &Expr::from_i64(1), 3).unwrap(),
    );
}

#[test]
fn point_substitution_respects_typed_coordinate_identity() {
    let x = Symbol::new("x");
    let other_x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32]));
    let expression = Expr::Mul(vec![
        power(Expr::Sym(other_x.clone()), -1),
        function("exp", Expr::Sym(x.clone())),
    ]);
    assert_equal(
        taylor(&expression, &x, &Expr::from_i64(0), 1).unwrap(),
        (Expr::from_i64(1) + Expr::Sym(x.clone())) * power(Expr::Sym(other_x.clone()), -1),
    );
    assert_refused(&expression, &other_x, &Expr::from_i64(0), 0);
    assert!(
        taylor(
            &function("exp", Expr::Sym(x.clone())),
            &x,
            &Expr::Sym(other_x),
            1
        )
        .is_ok()
    );
    assert_refused(
        &function("exp", Expr::Sym(x.clone())),
        &x,
        &Expr::Sym(x.clone()),
        1,
    );
}

#[test]
fn formal_punctured_series_remains_distinct_from_pointwise_taylor() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let source = Expr::Mul(vec![function("sin", xe.clone()), power(xe, -1)]);
    let (terms, _) = series_expansion(&source, &x, &Expr::from_i64(0), 4).unwrap();
    assert_eq!(
        terms,
        vec![
            (Expr::from_i64(1), BigRational::from_integer(0.into())),
            (
                Expr::Rational(BigRational::new((-1).into(), 6.into())),
                BigRational::from_integer(2.into()),
            ),
        ]
    );
    assert_refused(&source, &x, &Expr::from_i64(0), 3);
}

#[test]
fn point_substitution_admits_combined_size_and_depth_before_cloning() {
    let x = Symbol::new("x");
    let center = Expr::Add(
        (0..128)
            .map(|index| Expr::symbol(format!("a{index}")))
            .collect(),
    );
    let repeated = Expr::Add(vec![function("exp", Expr::Sym(x.clone())); 129]);
    assert_refused(&repeated, &x, &center, 0);

    let mut nested_source = Expr::Sym(x.clone());
    let mut nested_center = Expr::symbol("a");
    for _ in 0..64 {
        nested_source = function("sin", nested_source);
        nested_center = function("cos", nested_center);
    }
    assert_refused(&nested_source, &x, &nested_center, 0);
}

fn finite_literal_infinity_values() -> [(Expr, Expr); 3] {
    [
        (
            function("exp", Expr::Const(Constant::NegativeInfinity)),
            Expr::from_i64(0),
        ),
        (
            function("atan", Expr::Const(Constant::Infinity)),
            Expr::rational(1, 2).unwrap() * Expr::Const(Constant::Pi),
        ),
        (
            power(Expr::Const(Constant::Infinity), -1),
            Expr::from_i64(0),
        ),
    ]
}

#[test]
fn finite_literal_infinity_sources_remain_defined() {
    let x = Symbol::new("finite_literal_source_x");
    let xe = Expr::Sym(x.clone());
    let zero = Expr::from_i64(0);
    for (source, expected) in finite_literal_infinity_values() {
        for order in [0, 3] {
            assert_equal(taylor(&source, &x, &zero, order).unwrap(), expected.clone());
            let held_zero = Expr::Mul(vec![zero.clone(), source.clone()]);
            assert_equal(taylor(&held_zero, &x, &zero, order).unwrap(), zero.clone());
        }
        let composed = Expr::Add(vec![function("exp", xe.clone()), source]);
        let expected_source = function("exp", xe.clone()) + expected;
        assert_equal(
            taylor(&composed, &x, &zero, 3).unwrap(),
            taylor(&expected_source, &x, &zero, 3).unwrap(),
        );
    }
    // Generated infinities at excluded source points are not literal inputs
    // to a defined constant. Even a finite outer limit cannot repair them.
    for source in [
        function("log", zero.clone()),
        power(zero.clone(), -1),
        function("exp", function("log", zero.clone())),
        power(power(zero.clone(), -1), -1),
        function("exp", Expr::Mul(vec![Expr::from_i64(-1), power(xe, -2)])),
    ] {
        for order in [0, 3] {
            assert_refused(&source, &x, &zero, order);
        }
    }
    for nonfinite in [
        Expr::Const(Constant::Infinity),
        Expr::Const(Constant::NegativeInfinity),
        Expr::Const(Constant::ComplexInfinity),
        Expr::Const(Constant::NaN),
        function("exp", Expr::Const(Constant::Infinity)),
        function("log", Expr::Const(Constant::Infinity)),
    ] {
        let held_zero = Expr::Mul(vec![zero.clone(), nonfinite]);
        assert_refused(&held_zero, &x, &zero, 0);
        assert_refused(&held_zero, &x, &zero, 3);
    }
}

#[test]
fn finite_literal_infinity_centers_match_their_exact_values() {
    let x = Symbol::new("finite_literal_center_x");
    let source = function("exp", Expr::Sym(x.clone()));
    for (center, value) in finite_literal_infinity_values() {
        for order in [0, 3] {
            assert_equal(
                taylor(&source, &x, &center, order).unwrap(),
                taylor(&source, &x, &value, order).unwrap(),
            );
            let shifted_center = Expr::Add(vec![Expr::from_i64(1), center.clone()]);
            let shifted_value = Expr::from_i64(1) + value.clone();
            assert_equal(
                taylor(&source, &x, &shifted_center, order).unwrap(),
                taylor(&source, &x, &shifted_value, order).unwrap(),
            );
        }
    }
    for center in [
        function("exp", function("log", Expr::from_i64(0))),
        power(power(Expr::from_i64(0), -1), -1),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Const(Constant::Infinity)]),
        function("exp", Expr::Const(Constant::Infinity)),
        function("log", Expr::Const(Constant::Infinity)),
    ] {
        assert_refused(&source, &x, &center, 0);
        assert_refused(&source, &x, &center, 3);
    }
}
