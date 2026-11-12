//! Principal values on a cut depend on the side of the complete source path.

use fsym_calculus::gruntz::Direction;
use fsym_calculus::{limit_dir, series_expansion_dir};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::{BigRational, Expr, Symbol, SymbolIdentity, parse};
use std::collections::HashMap;

fn expression(source: &str) -> Expr {
    parse(source).unwrap()
}

fn equal(actual: &Expr, expected: &str) {
    let residual = fsym_simplify::try_simplify(
        &fsym_simplify::try_expand(&(actual.clone() - expression(expected))).unwrap(),
    )
    .unwrap();
    assert!(residual.is_zero(), "{actual} != {expected}: {residual}");
}

#[test]
fn lower_cut_series_reproduce_the_principal_log_and_square_root() {
    let x = Symbol::new("x");
    for (source, expected) in [
        ("log(-1-I*x)", ["-I*pi", "I", "1/2"]),
        ("sqrt(-1-I*x)", ["-I", "1/2", "-I/8"]),
    ] {
        let (terms, order) = series_expansion_dir(
            &expression(source),
            &x,
            &expression("0"),
            3,
            Direction::Plus,
        )
        .unwrap();
        assert_eq!(order, Some(BigRational::from_integer(3.into())));
        assert_eq!(terms.len(), expected.len());
        for (index, ((coefficient, exponent), expected)) in terms.iter().zip(expected).enumerate() {
            assert_eq!(*exponent, BigRational::from_integer((index as i64).into()));
            equal(coefficient, expected);
        }
    }
}

#[test]
fn lower_cut_limits_do_not_substitute_the_value_on_the_cut() {
    let x = Symbol::new("x");
    for (source, expected) in [("log(-1-I*x)", "-I*pi"), ("sqrt(-1-I*x)", "-I")] {
        let limit = limit_dir(&expression(source), &x, &expression("0"), Direction::Plus).unwrap();
        equal(&limit, expected);
    }
}

#[test]
fn upper_and_reflected_paths_have_distinct_principal_values() {
    let x = Symbol::new("x");
    for (source, plus, minus) in [
        ("log(-1-I*x)", "-I*pi", "I*pi"),
        ("sqrt(-1-I*x)", "-I", "I"),
        ("log(-1+I*x)", "I*pi", "-I*pi"),
        ("sqrt(-1+I*x)", "I", "-I"),
    ] {
        for (direction, expected) in [(Direction::Plus, plus), (Direction::Minus, minus)] {
            let source = expression(source);
            let (terms, _) =
                series_expansion_dir(&source, &x, &expression("0"), 3, direction).unwrap();
            equal(&terms[0].0, expected);
            equal(
                &limit_dir(&source, &x, &expression("0"), direction).unwrap(),
                expected,
            );
        }
        assert!(
            limit_dir(&expression(source), &x, &expression("0"), Direction::Both).is_err(),
            "the cut jump of {source} must not have a two-sided limit",
        );
    }
}

#[test]
fn cut_side_uses_the_original_path_beyond_the_requested_series_order() {
    let x = Symbol::new("x");
    for (source, expected) in [
        ("log(-1-I*x**100)", "-I*pi"),
        ("sqrt(-1-I*x**100)", "-I"),
        ("log(-1+I*x**100)", "I*pi"),
        ("sqrt(-1+I*x**100)", "I"),
        ("log(-1-I*x+I*x**2)", "-I*pi"),
        ("sqrt(-1-I*x+I*x**2)", "-I"),
    ] {
        let (terms, _) = series_expansion_dir(
            &expression(source),
            &x,
            &expression("0"),
            3,
            Direction::Plus,
        )
        .unwrap();
        equal(&terms[0].0, expected);
    }
}

#[test]
fn fractional_powers_use_the_exponent_specific_phase() {
    let x = Symbol::new("x");
    for (source, expected) in [
        ("(-1-I*x)**(1/3)", "exp(-I*pi/3)"),
        ("(-1-I*x)**(-1/2)", "I"),
        ("(-1-I*x)**(3/2)", "I"),
        ("(-1-I*x)**2", "1"),
        ("(-1-I*x)**(-3)", "-1"),
    ] {
        let source = expression(source);
        let (terms, _) =
            series_expansion_dir(&source, &x, &expression("0"), 3, Direction::Plus).unwrap();
        equal(&terms[0].0, expected);
        equal(
            &limit_dir(&source, &x, &expression("0"), Direction::Plus).unwrap(),
            expected,
        );
    }
}

#[test]
fn real_paths_and_values_away_from_the_cut_retain_their_continuous_values() {
    let x = Symbol::new("x");
    for (source, expected) in [
        ("log(-1+x)", "I*pi"),
        ("sqrt(-1+x)", "I"),
        ("log(1+x)", "0"),
        ("sqrt(1+x)", "1"),
        ("log(I+x)", "I*pi/2"),
        ("sqrt(I+x)", "sqrt(I)"),
    ] {
        for direction in [Direction::Plus, Direction::Minus] {
            let source = expression(source);
            let (terms, _) =
                series_expansion_dir(&source, &x, &expression("0"), 3, direction).unwrap();
            let constant = terms
                .iter()
                .find(|(_, power)| *power == BigRational::from_integer(0.into()))
                .map(|(coefficient, _)| coefficient.clone())
                .unwrap_or_else(|| expression("0"));
            equal(&constant, expected);
            equal(
                &limit_dir(&source, &x, &expression("0"), direction).unwrap(),
                expected,
            );
        }
    }
}

#[test]
fn parameter_signs_decide_the_cut_without_leaking_into_other_calls() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    let source = expression("log(-1-I*a*x)");
    for (fact, expected) in [("positive", "-I*pi"), ("negative", "I*pi")] {
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&[fact]))]),
            || {
                let (terms, _) =
                    series_expansion_dir(&source, &x, &expression("0"), 3, Direction::Plus)
                        .unwrap();
                equal(&terms[0].0, expected);
                equal(
                    &limit_dir(&source, &x, &expression("0"), Direction::Plus).unwrap(),
                    expected,
                );
            },
        );
    }
    for facts in [Facts::default(), Facts::from_names(&["real"])] {
        with_symbol_facts(HashMap::from([(a.clone(), facts)]), || {
            assert!(
                series_expansion_dir(&source, &x, &expression("0"), 3, Direction::Plus).is_err()
            );
            assert!(limit_dir(&source, &x, &expression("0"), Direction::Plus).is_err());
        });
    }
}

#[test]
fn typed_parameter_facts_do_not_bind_another_same_named_symbol() {
    let x = Symbol::new("x");
    let positive = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([61; 32]));
    let unknown = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([62; 32]));
    let source = Expr::Function(
        "log".into(),
        vec![expression("-1") - expression("I") * Expr::Sym(unknown) * Expr::Sym(x.clone())],
    );
    with_symbol_facts(
        HashMap::from([(positive, Facts::from_names(&["positive"]))]),
        || {
            assert!(
                series_expansion_dir(&source, &x, &expression("0"), 3, Direction::Plus).is_err()
            );
            assert!(limit_dir(&source, &x, &expression("0"), Direction::Plus).is_err());
        },
    );
}
