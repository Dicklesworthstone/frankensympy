//! One-sided coordinates must retain exact Laurent and branch information.

use fsym_calculus::gruntz::Direction;
use fsym_calculus::{SeriesTerms, series_expansion, series_expansion_dir};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol, SymbolIdentity, parse};
use std::sync::Arc;

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn expanded(source: &str, point: &str, order: i64, direction: Direction) -> SeriesTerms {
    series_expansion_dir(&e(source), &Symbol::new("x"), &e(point), order, direction).unwrap()
}

fn assert_equal(actual: &Expr, expected: &Expr) {
    let residual = fsym_simplify::try_simplify(
        &fsym_simplify::try_expand(&(actual.clone() - expected.clone())).unwrap(),
    )
    .unwrap();
    assert!(residual.is_zero(), "{actual} != {expected}: {residual}");
}

#[test]
fn nonsmooth_sign_changes_on_the_selected_finite_side() {
    for (source, point) in [
        ("Abs(x)/x", "0"),
        ("Abs(x-2)/(x-2)", "2"),
        ("Abs(x+2/3)/(x+2/3)", "-2/3"),
    ] {
        let plus = expanded(source, point, 5, Direction::Plus);
        let minus = expanded(source, point, 5, Direction::Minus);
        assert_eq!(plus, (vec![(e("1"), q(0, 1))], None));
        assert_eq!(minus, (vec![(e("-1"), q(0, 1))], None));
    }
}

#[test]
fn checked_rational_laurent_coefficients_reflect_both_negative_and_positive_powers() {
    let source = "(1+2*x)/(x**3*(1-x))";
    let (plus, plus_order) = expanded(source, "0", 4, Direction::Plus);
    let (minus, minus_order) = expanded(source, "0", 4, Direction::Minus);
    assert_eq!(plus_order, Some(q(4, 1)));
    assert_eq!(minus_order, plus_order);
    assert_eq!(plus.len(), 7);
    assert_eq!(minus.len(), 7);
    for (index, ((a, ka), (b, kb))) in plus.iter().zip(&minus).enumerate() {
        let power = index as i64 - 3;
        let coefficient = if index == 0 { 1 } else { 3 };
        assert_eq!(*ka, q(power, 1));
        assert_eq!(ka, kb);
        assert_equal(a, &Expr::from_i64(coefficient));
        assert_equal(
            b,
            &Expr::from_i64(if power % 2 == 0 {
                coefficient
            } else {
                -coefficient
            }),
        );
    }
}

#[test]
fn analytic_series_uses_the_reflected_coordinate_without_changing_the_function() {
    let (terms, order) = expanded("exp(x)", "0", 6, Direction::Minus);
    assert_eq!(order, Some(q(6, 1)));
    assert_eq!(terms.len(), 6);
    for (index, ((coefficient, power), factorial)) in
        terms.iter().zip([1, 1, 2, 6, 24, 120]).enumerate()
    {
        assert_eq!(*power, q(index as i64, 1));
        let sign = if index % 2 == 0 { 1 } else { -1 };
        assert_equal(coefficient, &rational_expr(q(sign, factorial)));
    }
    let x = Symbol::new("x");
    assert_eq!(
        series_expansion(&e("exp(x)"), &x, &e("0"), 6).unwrap(),
        expanded("exp(x)", "0", 6, Direction::Plus),
    );
}

#[test]
fn principal_puiseux_factors_follow_the_selected_ray() {
    let left = expanded("sqrt(x)", "0", 4, Direction::Minus);
    let right = expanded("sqrt(x)", "0", 4, Direction::Plus);
    assert_eq!(left.1, None);
    assert_eq!(right.1, None);
    assert_eq!(left.0.len(), 1);
    assert_eq!(right.0.len(), 1);
    assert_eq!(left.0[0].1, q(1, 2));
    assert_eq!(right.0[0].1, q(1, 2));
    assert_equal(&left.0[0].0, &e("I"));
    assert_equal(&right.0[0].0, &e("1"));
    let negative = expanded("sqrt(-x)", "0", 4, Direction::Minus);
    assert_equal(&negative.0[0].0, &e("1"));
}

#[test]
fn logarithms_use_the_same_oriented_coordinate_as_power_terms() {
    let left = expanded("log(x)", "0", 4, Direction::Minus);
    assert_eq!(left.1, None);
    assert_eq!(left.0.len(), 1);
    assert_eq!(left.0[0].1, q(0, 1));
    assert_equal(&left.0[0].0, &e("log(-x)+I*pi"));
    let shifted = expanded("log(x-2)", "2", 4, Direction::Minus);
    assert_equal(&shifted.0[0].0, &e("log(2-x)+I*pi"));
    let negative = expanded("log(-x)", "0", 4, Direction::Minus);
    assert_equal(&negative.0[0].0, &e("log(-x)"));
}

#[test]
fn infinite_centers_keep_the_established_inward_coordinates() {
    for point in ["oo", "-oo"] {
        for source in ["Abs(x)", "1/x+2/x**2"] {
            assert_eq!(
                expanded(source, point, 5, Direction::Plus),
                expanded(source, point, 5, Direction::Minus),
            );
        }
    }
    let negative = expanded("1/x+2/x**2", "-oo", 5, Direction::Minus);
    assert_eq!(negative.1, None);
    assert_eq!(negative.0.len(), 2);
    assert_equal(&negative.0[0].0, &e("-1"));
    assert_eq!(negative.0[0].1, q(1, 1));
    assert_equal(&negative.0[1].0, &e("2"));
    assert_eq!(negative.0[1].1, q(2, 1));
}

#[test]
fn typed_time_and_user_placeholder_names_are_not_captured() {
    let x = Symbol::with_identity(
        "_fsym_series_t",
        SymbolIdentity::from_assumptions_digest([21; 32]),
    );
    let parameter = Symbol::with_identity(
        "_fsym_log_t",
        SymbolIdentity::from_assumptions_digest([22; 32]),
    );
    let source = Expr::Add(vec![
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
        Expr::Sym(parameter.clone()),
    ]);
    let (terms, order) = series_expansion_dir(&source, &x, &e("0"), 4, Direction::Minus).unwrap();
    assert_eq!(order, None);
    assert_eq!(terms.len(), 1);
    let symbols = terms[0].0.free_symbols();
    assert_eq!(symbols.len(), 2);
    assert!(symbols.contains(&x));
    assert!(symbols.contains(&parameter));
    let expected =
        Expr::Function("log".into(), vec![-Expr::Sym(x)]) + Expr::Sym(parameter) + e("I*pi");
    assert_equal(&terms[0].0, &expected);
}

#[test]
fn original_undefined_children_and_two_sided_requests_refuse() {
    let x = Symbol::new("x");
    let pole = Expr::Pow(Arc::new(e("0")), Arc::new(e("-1")));
    let source = Expr::Mul(vec![
        e("0"),
        pole,
        Expr::Function("Abs".into(), vec![Expr::Sym(x.clone())]),
    ]);
    for direction in [Direction::Plus, Direction::Minus] {
        assert!(series_expansion_dir(&source, &x, &e("0"), 5, direction).is_err());
    }
    assert!(series_expansion_dir(&e("Abs(x)"), &x, &e("0"), 5, Direction::Both).is_err());
}
