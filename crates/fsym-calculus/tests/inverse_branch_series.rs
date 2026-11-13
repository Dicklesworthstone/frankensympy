//! Principal inverse functions retain the side of the complete complex path.

use fsym_calculus::gruntz::Direction;
use fsym_calculus::{limit_dir, series_expansion_dir};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::{BigRational, Expr, Symbol, SymbolIdentity, parse};
use std::collections::HashMap;
use std::sync::Arc;

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn equal(actual: &Expr, expected: &str) {
    // Canonical elementary identities (including odd inverse functions) are
    // exact; the simplifier does not itself evaluate every held function.
    let expected = e(expected).subs(&HashMap::new());
    let residual = fsym_simplify::try_simplify(
        &fsym_simplify::try_expand(&(actual.clone() - expected.clone())).unwrap(),
    )
    .unwrap();
    assert!(residual.is_zero(), "{actual} != {expected}: {residual}");
}

fn coefficient(source: &Expr, order: i64, exponent: i64, direction: Direction) -> Expr {
    let (terms, _) =
        series_expansion_dir(source, &Symbol::new("x"), &e("0"), order, direction).unwrap();
    terms
        .into_iter()
        .find(|(_, power)| *power == BigRational::from_integer(exponent.into()))
        .map(|(value, _)| value)
        .unwrap_or_else(|| e("0"))
}

#[test]
fn inverse_circular_limits_follow_both_sides_of_their_cuts() {
    let x = Symbol::new("x");
    for (source, plus, minus) in [
        ("asin(5/4+I*x)", "pi-asin(5/4)", "asin(5/4)"),
        ("asin(-5/4+I*x)", "-asin(5/4)", "-pi+asin(5/4)"),
        ("acos(5/4+I*x)", "-acos(5/4)", "acos(5/4)"),
        ("acos(-5/4+I*x)", "acos(-5/4)", "2*pi-acos(-5/4)"),
        ("atan(2*I+x)", "atan(2*I)", "atan(2*I)-pi"),
        ("atan(-2*I+x)", "atan(-2*I)+pi", "atan(-2*I)"),
        ("acot(I/2+x)", "acot(I/2)+pi", "acot(I/2)"),
        ("acot(-I/2+x)", "acot(-I/2)", "acot(-I/2)-pi"),
    ] {
        let source = e(source);
        for (direction, expected) in [(Direction::Plus, plus), (Direction::Minus, minus)] {
            equal(
                &limit_dir(&source, &x, &e("0"), direction).unwrap(),
                expected,
            );
            equal(&coefficient(&source, 2, 0, direction), expected);
        }
        assert!(limit_dir(&source, &x, &e("0"), Direction::Both).is_err());
    }
}

#[test]
fn inverse_hyperbolic_limits_follow_both_sides_of_their_cuts() {
    let x = Symbol::new("x");
    for (source, plus, minus) in [
        ("asinh(5*I/4+x)", "asinh(5*I/4)", "I*pi-asinh(5*I/4)"),
        ("asinh(-5*I/4+x)", "-I*pi-asinh(-5*I/4)", "asinh(-5*I/4)"),
        ("acosh(-5/4+I*x)", "acosh(-5/4)", "acosh(-5/4)-2*I*pi"),
        ("acosh(1/2+I*x)", "acosh(1/2)", "-acosh(1/2)"),
        ("atanh(5/4+I*x)", "atanh(5/4)+I*pi", "atanh(5/4)"),
        ("atanh(-5/4+I*x)", "atanh(-5/4)", "atanh(-5/4)-I*pi"),
        ("acoth(1/2+I*x)", "acoth(1/2)", "acoth(1/2)+I*pi"),
        ("acoth(-1/2+I*x)", "acoth(-1/2)-I*pi", "acoth(-1/2)"),
    ] {
        let source = e(source);
        for (direction, expected) in [(Direction::Plus, plus), (Direction::Minus, minus)] {
            equal(
                &limit_dir(&source, &x, &e("0"), direction).unwrap(),
                expected,
            );
            equal(&coefficient(&source, 2, 0, direction), expected);
        }
        assert!(limit_dir(&source, &x, &e("0"), Direction::Both).is_err());
    }
}

#[test]
fn branch_reflections_change_derivatives_as_well_as_the_constant() {
    // These coefficients follow the independent derivatives of the principal
    // logarithmic forms, including their reflected square-root boundary.
    for (source, linear, quadratic) in [
        ("asin(5/4+I*x)", "-4/3", "40*I/27"),
        ("acos(5/4+I*x)", "4/3", "-40*I/27"),
        ("asinh(5*I/4-x)", "-4*I/3", "-40/27"),
        ("acosh(1/2-I*x)", "2*sqrt(3)/3", "-2*I*sqrt(3)/9"),
        ("atan(2*I-x)", "1/3", "-2*I/9"),
        ("atanh(5/4+I*x)", "-16*I/9", "-320/81"),
        ("acot(I/2+x)", "-4/3", "8*I/9"),
        ("acoth(1/2-I*x)", "-4*I/3", "-8/9"),
    ] {
        let source = e(source);
        equal(&coefficient(&source, 3, 1, Direction::Plus), linear);
        equal(&coefficient(&source, 3, 2, Direction::Plus), quadratic);
    }
}

#[test]
fn complete_paths_decide_the_branch_beyond_the_requested_order() {
    for (source, expected) in [
        ("asin(5/4+I*x**100)", "pi-asin(5/4)"),
        ("acos(5/4+I*x**100)", "-acos(5/4)"),
        ("atan(2*I-x**100)", "atan(2*I)-pi"),
        ("asinh(5*I/4-x**100)", "I*pi-asinh(5*I/4)"),
        ("acosh(-5/4-I*x**100)", "acosh(-5/4)-2*I*pi"),
        ("atanh(5/4+I*x**100)", "atanh(5/4)+I*pi"),
        ("acoth(1/2-I*x**100)", "acoth(1/2)+I*pi"),
        ("acot(I/2+x**100)", "acot(I/2)+pi"),
    ] {
        equal(&coefficient(&e(source), 2, 0, Direction::Plus), expected);
    }
}

#[test]
fn paths_on_the_cut_keep_the_defined_principal_value() {
    for (source, expected) in [
        ("asin(5/4+x)", "asin(5/4)"),
        ("acos(5/4+x)", "acos(5/4)"),
        ("atan(2*I+I*x)", "atan(2*I)"),
        ("asinh(5*I/4+I*x)", "asinh(5*I/4)"),
        ("acosh(-5/4+x)", "acosh(-5/4)"),
        ("atanh(5/4+x)", "atanh(5/4)"),
        ("acoth(1/2+x)", "acoth(1/2)"),
        ("acot(I/2+I*x)", "acot(I/2)"),
    ] {
        equal(&coefficient(&e(source), 2, 0, Direction::Plus), expected);
    }
}

#[test]
fn ordinary_taylor_series_and_continuous_endpoint_limits_stay_available() {
    for (source, expected) in [
        ("asin(x)", "x+x**3/6"),
        ("acos(x)", "pi/2-x-x**3/6"),
        ("atan(x)", "x-x**3/3"),
        ("asinh(x)", "x-x**3/6"),
        ("atanh(x)", "x+x**3/3"),
    ] {
        let (terms, _) =
            series_expansion_dir(&e(source), &Symbol::new("x"), &e("0"), 4, Direction::Plus)
                .unwrap();
        let polynomial = terms.into_iter().fold(e("0"), |sum, (c, k)| {
            sum + c * Expr::pow(Expr::symbol("x"), fsym_core::elementary::rational_expr(k))
        });
        equal(&polynomial, expected);
    }
    for (source, point, expected) in [
        ("asin(x)", "1", "pi/2"),
        ("acos(x)", "-1", "pi"),
        ("atan(x)", "2", "atan(2)"),
        ("acosh(x)", "2", "acosh(2)"),
    ] {
        equal(
            &limit_dir(&e(source), &Symbol::new("x"), &e(point), Direction::Both).unwrap(),
            expected,
        );
    }
}

#[test]
fn reciprocal_inverse_functions_handle_the_jump_at_zero() {
    for (source, plus, minus) in [
        ("acot(x)", "pi/2", "-pi/2"),
        ("acot(I*x)", "-pi/2", "pi/2"),
        ("acoth(x)", "-I*pi/2", "I*pi/2"),
        ("acoth(I*x)", "-I*pi/2", "I*pi/2"),
    ] {
        let source = e(source);
        for (direction, expected) in [(Direction::Plus, plus), (Direction::Minus, minus)] {
            equal(&coefficient(&source, 2, 0, direction), expected);
        }
        assert!(limit_dir(&source, &Symbol::new("x"), &e("0"), Direction::Both).is_err());
    }
}

#[test]
fn parameter_cut_sides_require_typed_facts_and_do_not_leak() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    let source = e("asin(5/4+I*a*x)");
    for (fact, expected) in [("positive", "pi-asin(5/4)"), ("negative", "asin(5/4)")] {
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&[fact]))]),
            || equal(&coefficient(&source, 2, 0, Direction::Plus), expected),
        );
    }
    assert!(series_expansion_dir(&source, &x, &e("0"), 2, Direction::Plus).is_err());
    with_symbol_facts(HashMap::from([(a, Facts::from_names(&["real"]))]), || {
        assert!(series_expansion_dir(&source, &x, &e("0"), 2, Direction::Plus).is_err());
    });
    let first = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([71; 32]));
    let other = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([72; 32]));
    let source = Expr::Function(
        "asin".into(),
        vec![e("5/4") + e("I") * Expr::Sym(other) * Expr::Sym(x.clone())],
    );
    with_symbol_facts(
        HashMap::from([(first, Facts::from_names(&["positive"]))]),
        || {
            assert!(series_expansion_dir(&source, &x, &e("0"), 2, Direction::Plus).is_err());
        },
    );
}

#[test]
fn finite_boundary_limits_cover_shifted_points_without_nonzero_value_proofs() {
    let x = Symbol::new("x");
    for (source, point, direction, expected) in [
        ("asin(5/4+I*(x-2))", "2", Direction::Plus, "pi-asin(5/4)"),
        ("asin(5/4+I*(x-2))", "2", Direction::Minus, "asin(5/4)"),
        ("asin(x)", "0", Direction::Both, "0"),
        ("acos(x)", "1", Direction::Both, "0"),
        ("atan(1/x)", "0", Direction::Plus, "pi/2"),
        ("atan(1/x)", "0", Direction::Minus, "-pi/2"),
        ("acot(x)", "0", Direction::Plus, "pi/2"),
        ("acot(x)", "0", Direction::Minus, "-pi/2"),
        ("acoth(I*x)", "0", Direction::Plus, "-I*pi/2"),
        ("acoth(I*x)", "0", Direction::Minus, "I*pi/2"),
    ] {
        equal(
            &limit_dir(&e(source), &x, &e(point), direction).unwrap(),
            expected,
        );
    }
}

#[test]
fn inverse_branch_selection_does_not_erase_undefined_source_children() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    let unknown_pole = Expr::Pow(Arc::new(Expr::Sym(a.clone())), Arc::new(e("-1")));
    let fixed_pole = Expr::Function("tan".into(), vec![e("pi/2")]);
    for hidden in [unknown_pole, fixed_pole] {
        let argument = Expr::Add(vec![
            e("5/4"),
            e("I") * Expr::Sym(x.clone()),
            Expr::Mul(vec![e("0"), hidden]),
        ]);
        let source = Expr::Function("asin".into(), vec![argument]);
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&["real"]))]),
            || {
                assert!(series_expansion_dir(&source, &x, &e("0"), 2, Direction::Plus).is_err());
                assert!(limit_dir(&source, &x, &e("0"), Direction::Plus).is_err());
            },
        );
    }
}

#[test]
fn inverse_inputs_preserve_undefined_points_and_outer_function_poles() {
    let x = Symbol::new("x");
    let hidden_pole = Expr::Mul(vec![e("0"), Expr::Function("tan".into(), vec![e("pi/2")])]);
    let point = Expr::Add(vec![e("0"), hidden_pole]);
    assert!(limit_dir(&e("atan(x)"), &x, &point, Direction::Plus).is_err());
    assert!(series_expansion_dir(&e("atan(x)"), &x, &point, 2, Direction::Plus).is_err());

    let source = Expr::Add(vec![
        e("1"),
        Expr::Sym(x.clone()),
        Expr::Mul(vec![e("0"), Expr::Function("atanh".into(), vec![e("1")])]),
    ]);
    assert!(limit_dir(&source, &x, &e("0"), Direction::Plus).is_err());
    assert!(series_expansion_dir(&source, &x, &e("0"), 2, Direction::Plus).is_err());
}

#[test]
fn finite_inverse_limits_do_not_assume_a_parameter_stays_nonzero() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    for source in [
        "atan(1+I+(sqrt(a**2+x**2)-sqrt(a**2))/x)",
        "atan(1+I+(sqrt(a**2+x**100)-sqrt(a**2))/x**50)",
    ] {
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&["real"]))]),
            || {
                assert!(limit_dir(&e(source), &x, &e("0"), Direction::Plus).is_err());
            },
        );
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&["positive"]))]),
            || {
                equal(
                    &limit_dir(&e(source), &x, &e("0"), Direction::Plus)
                        .unwrap_or_else(|error| panic!("{source}: {error:?}")),
                    "atan(1+I)",
                );
            },
        );
    }
    equal(
        &limit_dir(&e("atan(1+I+sqrt(x**2)/x)"), &x, &e("0"), Direction::Plus).unwrap(),
        "atan(2+I)",
    );
}
