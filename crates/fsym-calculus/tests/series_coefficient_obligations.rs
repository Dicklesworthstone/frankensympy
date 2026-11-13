//! Local series cannot invert a parameter-dependent leading coefficient
//! unless its nonzero premise is established in the active typed context.

use fsym_calculus::series::{Series, series_in};
use fsym_calculus::{gruntz::Direction, limit_dir};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::{BigRational, Expr, Symbol, parse};
use std::collections::HashMap;

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}

fn expand(source: &str, order: i64) -> Result<Series, fsym_calculus::series::SeriesError> {
    series_in(&e(source), &Symbol::new("x"), &q(order), &e("log(x)"))
}

fn assert_coefficient(actual: &Expr, expected: &str) {
    let residual = fsym_simplify::try_simplify(
        &fsym_simplify::try_expand(&(actual.clone() - e(expected))).unwrap(),
    )
    .unwrap();
    assert!(residual.is_zero(), "{actual} != {expected}: {residual}");
}

#[test]
fn a_vanishing_parameter_cannot_change_the_reported_local_order() {
    with_symbol_facts(
        HashMap::from([(Symbol::new("a"), Facts::from_names(&["real"]))]),
        || {
            for source in [
                "(sqrt(a**2+x**2)-sqrt(a**2))/x",
                "(sqrt(a**2+x)-sqrt(a**2))/sqrt(x)",
                "(sqrt(a**2+x**20)-sqrt(a**2))/x**10",
            ] {
                for order in [1, 2, 4] {
                    assert!(expand(source, order).is_err(), "{source}, order {order}");
                }
            }
        },
    );
}

#[test]
fn logarithms_and_reciprocals_keep_the_same_nonzero_obligation() {
    with_symbol_facts(
        HashMap::from([(Symbol::new("a"), Facts::from_names(&["real"]))]),
        || {
            for source in ["log(a+x)", "1/(a+x)", "(a+x)**(-2)", "sqrt(a+x)"] {
                assert!(expand(source, 3).is_err(), "{source}");
            }
        },
    );
}

#[test]
fn positive_parameters_discharge_the_actual_reciprocal_premise() {
    with_symbol_facts(
        HashMap::from([(Symbol::new("a"), Facts::from_names(&["positive"]))]),
        || {
            let result = expand("(sqrt(a**2+x**2)-sqrt(a**2))/x", 2).unwrap();
            assert_eq!(result.terms.len(), 1);
            assert_coefficient(result.terms.get(&q(1)).unwrap(), "a**(-1)/2");
            assert_eq!(result.order, Some(q(2)));

            let reciprocal = expand("1/(a+x)", 3).unwrap();
            for (power, expected) in [(0, "1/a"), (1, "-1/a**2"), (2, "1/a**3")] {
                assert_coefficient(reciprocal.terms.get(&q(power)).unwrap(), expected);
            }
        },
    );
}

#[test]
fn the_zero_parameter_case_keeps_its_distinct_exact_answer() {
    for source in ["sqrt(x**2)/x", "sqrt(x**20)/x**10"] {
        let result = expand(source, 2).unwrap();
        assert_eq!(result.terms.len(), 1);
        assert_coefficient(result.terms.get(&q(0)).unwrap(), "1");
        assert_eq!(result.order, None);
    }
    let source = e("sqrt(x**2)/x");
    let variable = Symbol::new("x");
    for (direction, value) in [(Direction::Plus, "1"), (Direction::Minus, "-1")] {
        let result = limit_dir(&source, &variable, &e("0"), direction).unwrap();
        assert_coefficient(&result, value);
    }
    assert!(limit_dir(&source, &variable, &e("0"), Direction::Both).is_err());
}

#[test]
fn positive_integer_powers_do_not_invent_a_reciprocal_side_condition() {
    let result = expand("(a+x)**2", 3).unwrap();
    assert_eq!(result.terms.len(), 3);
    assert_coefficient(result.terms.get(&q(0)).unwrap(), "a**2");
    assert_coefficient(result.terms.get(&q(1)).unwrap(), "2*a");
    assert_coefficient(result.terms.get(&q(2)).unwrap(), "1");
    assert_eq!(result.order, None);

    let vanishing = expand("(a*x)**3", 4).unwrap();
    assert_eq!(vanishing.terms.len(), 1);
    assert_coefficient(vanishing.terms.get(&q(3)).unwrap(), "a**3");
    assert_eq!(vanishing.order, None);
}

#[test]
fn polynomial_multiplication_uses_safe_valuation_lower_bounds() {
    // a may be zero. The lower bound -1 still safely determines how much
    // of the second factor is required for the x coefficient of the square.
    let result = expand("(a/x+x**2)**2", 2).unwrap();
    assert_eq!(result.terms.len(), 2);
    assert_coefficient(result.terms.get(&q(-2)).unwrap(), "a**2");
    assert_coefficient(result.terms.get(&q(1)).unwrap(), "2*a");
    assert_eq!(result.order, Some(q(2)));

    let hidden = expand("(x**3)**2", 1).unwrap();
    assert!(hidden.terms.is_empty());
    assert_eq!(hidden.order, Some(q(1)));
}
