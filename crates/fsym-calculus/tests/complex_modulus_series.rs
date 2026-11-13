//! Complex moduli must not inherit the sign of a real leading coefficient.

use fsym_calculus::gruntz::Direction;
use fsym_calculus::series::series_in;
use fsym_calculus::{SeriesTerms, limit_dir, series_expansion_dir};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::{BigRational, Expr, Symbol, SymbolIdentity, parse};
use std::collections::HashMap;
use std::sync::Arc;

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn expanded(source: &str, order: i64, direction: Direction) -> SeriesTerms {
    series_expansion_dir(&e(source), &Symbol::new("x"), &e("0"), order, direction).unwrap()
}

fn assert_coefficient(actual: &Expr, expected: &str) {
    let residual = fsym_simplify::try_simplify(
        &fsym_simplify::try_expand(&(actual.clone() - e(expected))).unwrap(),
    )
    .unwrap();
    assert!(residual.is_zero(), "{actual} != {expected}: {residual}");
}

fn assert_terms(actual: &SeriesTerms, expected: &[(i64, &str)]) {
    assert_eq!(actual.0.len(), expected.len(), "{actual:?}");
    for ((coefficient, exponent), (power, value)) in actual.0.iter().zip(expected) {
        assert_eq!(*exponent, q(*power, 1));
        assert_coefficient(coefficient, value);
    }
}

#[test]
fn nonreal_tails_do_not_become_imaginary_modulus_coefficients() {
    for source in ["Abs(1+I*x)", "Abs(1-I*x)", "Abs(-1+I*x)"] {
        for direction in [Direction::Plus, Direction::Minus] {
            let low = expanded(source, 2, direction);
            assert_terms(&low, &[(0, "1")]);
            assert_eq!(low.1, Some(q(2, 1)));

            // Squaring these exact rational coefficients gives 1+x^2
            // through degree six, with the positive constant square root.
            let high = expanded(source, 7, direction);
            assert_terms(&high, &[(0, "1"), (2, "1/2"), (4, "-1/8"), (6, "1/16")]);
            assert_eq!(high.1, Some(q(7, 1)));
        }
    }
}

#[test]
fn complex_leading_coefficients_use_their_exact_euclidean_norm() {
    for direction in [Direction::Plus, Direction::Minus] {
        let imaginary_axis = expanded("Abs(I*x)", 4, direction);
        assert_terms(&imaginary_axis, &[(1, "1")]);
        assert_eq!(imaginary_axis.1, None);

        let pole = expanded("Abs((3+4*I)/x)", 4, direction);
        assert_terms(&pole, &[(-1, "5")]);
        assert_eq!(pole.1, None);

        let algebraic = expanded("Abs((1+I)/x**2)", 4, direction);
        assert_terms(&algebraic, &[(-2, "sqrt(2)")]);
    }
}

#[test]
fn vanishing_complex_paths_keep_the_required_relative_precision() {
    let result = expanded("Abs(I*x+x**2)", 5, Direction::Plus);
    assert_terms(&result, &[(1, "1"), (3, "1/2")]);
    assert_eq!(result.1, Some(q(5, 1)));

    // A complex term outside the requested order still belongs to the
    // source; a truncation cannot authorize treating that source as real.
    let late = expanded("Abs(1+I*x**20)", 3, Direction::Plus);
    assert_terms(&late, &[(0, "1")]);
    assert_eq!(late.1, Some(q(3, 1)));
}

#[test]
fn mixed_complex_paths_match_independent_squared_norm_coefficients() {
    // |3+4i+(1-2i)x|^2 = 25-10x+5x^2. The positive square root has
    // coefficients 5, -1, 2/5, 2/25, obtained by exact convolution.
    let right = expanded("Abs(3+4*I+(1-2*I)*x)", 4, Direction::Plus);
    assert_terms(&right, &[(0, "5"), (1, "-1"), (2, "2/5"), (3, "2/25")]);
    let left = expanded("Abs(3+4*I+(1-2*I)*x)", 4, Direction::Minus);
    assert_terms(&left, &[(0, "5"), (1, "1"), (2, "2/5"), (3, "-2/25")]);
}

#[test]
fn supported_entire_functions_preserve_conjugation_in_the_modulus() {
    let unit_circle = expanded("Abs(exp(I*x))", 5, Direction::Plus);
    assert_terms(&unit_circle, &[(0, "1")]);
    let hyperbolic = expanded("Abs(cos(I*x))", 5, Direction::Plus);
    assert_terms(&hyperbolic, &[(0, "1"), (2, "1/2"), (4, "1/24")]);
}

#[test]
fn modulus_cancellation_limits_return_real_exact_values() {
    let x = Symbol::new("x");
    for (source, point, expected) in [
        ("(Abs(1+I*x)-1)/x", "0", "0"),
        ("(Abs(1+I*x)-1)/x**2", "0", "1/2"),
        ("(Abs(x+I)-x)*x", "oo", "1/2"),
    ] {
        for direction in [Direction::Plus, Direction::Minus] {
            let result = limit_dir(&e(source), &x, &e(point), direction).unwrap();
            assert_coefficient(&result, expected);
        }
    }
}

#[test]
fn unknown_complex_parameters_remain_obligations_even_beyond_the_requested_order() {
    let x = Symbol::new("x");
    for source in ["Abs(1+a*x)", "Abs(1+a*x**20)"] {
        assert!(series_expansion_dir(&e(source), &x, &e("0"), 3, Direction::Plus).is_err());
    }
    let a = Symbol::new("a");
    with_symbol_facts(HashMap::from([(a, Facts::from_names(&["real"]))]), || {
        let real = expanded("Abs(1+a*x)", 3, Direction::Plus);
        assert_terms(&real, &[(0, "1"), (1, "a")]);
        assert_eq!(real.1, None);
    });
    assert!(series_expansion_dir(&e("Abs(1+a*x)"), &x, &e("0"), 3, Direction::Plus).is_err());
}

#[test]
fn typed_and_contradictory_parameter_facts_cannot_discharge_complex_obligations() {
    let x = Symbol::new("x");
    let declared = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([91; 32]));
    let unknown = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([92; 32]));
    let source = Expr::Function(
        "Abs".into(),
        vec![e("1") + Expr::Sym(unknown.clone()) * Expr::Sym(x.clone())],
    );
    for (symbol, facts) in [
        (declared, Facts::from_names(&["real"])),
        (unknown, Facts::from_names(&["positive", "negative"])),
    ] {
        with_symbol_facts(HashMap::from([(symbol, facts)]), || {
            assert!(series_expansion_dir(&source, &x, &e("0"), 3, Direction::Plus).is_err());
        });
    }
}

#[test]
fn a_potentially_zero_leading_norm_is_not_inverted() {
    let x = Symbol::new("x");
    with_symbol_facts(
        HashMap::from([(Symbol::new("a"), Facts::from_names(&["real"]))]),
        || {
            assert!(
                series_expansion_dir(&e("Abs(a+I*x)"), &x, &e("0"), 3, Direction::Plus).is_err()
            );
        },
    );
}

#[test]
fn exact_zero_and_real_meromorphic_sources_preserve_their_existing_meaning() {
    let x = Symbol::new("x");
    let raw_zero = Expr::Function(
        "Abs".into(),
        vec![Expr::Add(vec![Expr::Sym(x.clone()), -Expr::Sym(x.clone())])],
    );
    let result = series_in(&raw_zero, &x, &q(3, 1), &e("log(x)")).unwrap();
    assert!(result.terms.is_empty());
    assert_eq!(result.order, None);
    let real = expanded("Abs(tan(x))", 5, Direction::Plus);
    assert_terms(&real, &[(1, "1"), (3, "1/3")]);
}

#[test]
fn original_undefined_subexpressions_cannot_disappear_in_a_squared_modulus() {
    let x = Symbol::new("x");
    let source = Expr::Function(
        "Abs".into(),
        vec![Expr::Add(vec![
            e("1"),
            Expr::Mul(vec![
                e("0"),
                Expr::Pow(Arc::new(e("0")), Arc::new(e("-1"))),
                Expr::Sym(x.clone()),
            ]),
        ])],
    );
    assert!(series_expansion_dir(&source, &x, &e("0"), 3, Direction::Plus).is_err());
    assert!(series_in(&source, &x, &q(3, 1), &e("log(x)")).is_err());
}

#[test]
fn original_parameter_and_function_poles_refuse_before_public_substitution() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    let reciprocal = Expr::Pow(Arc::new(Expr::Sym(a.clone())), Arc::new(e("-1")));
    let tangent = Expr::Function("tan".into(), vec![Expr::Sym(a.clone())]);
    let fixed_pole = Expr::Function("tan".into(), vec![e("pi/2")]);
    let logarithmic_pole = Expr::Function("log".into(), vec![e("0")]);
    with_symbol_facts(HashMap::from([(a, Facts::from_names(&["real"]))]), || {
        for pole in [reciprocal, tangent, fixed_pole, logarithmic_pole] {
            for hidden in [
                Expr::Mul(vec![e("0"), pole.clone()]),
                Expr::Pow(Arc::new(pole), Arc::new(e("0"))),
            ] {
                let source = Expr::Function(
                    "Abs".into(),
                    vec![Expr::Add(vec![
                        e("1"),
                        Expr::Mul(vec![hidden.clone(), Expr::Sym(x.clone())]),
                    ])],
                );
                for direction in [Direction::Plus, Direction::Minus, Direction::Both] {
                    assert!(limit_dir(&source, &x, &e("0"), direction).is_err());
                }
                for direction in [Direction::Plus, Direction::Minus] {
                    assert!(series_expansion_dir(&source, &x, &e("0"), 3, direction).is_err());
                }
                assert!(series_in(&source, &x, &q(3, 1), &e("log(x)")).is_err());
                let constant = Expr::Function("Abs".into(), vec![hidden]);
                assert!(series_in(&constant, &x, &q(3, 1), &e("log(x)")).is_err());
            }
        }
    });
}

#[test]
fn nonzero_parameters_and_regular_local_denominators_remain_usable() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    let reciprocal = Expr::Pow(Arc::new(Expr::Sym(a.clone())), Arc::new(e("-1")));
    let source = Expr::Function(
        "Abs".into(),
        vec![Expr::Add(vec![
            e("1"),
            Expr::Mul(vec![e("0"), reciprocal, Expr::Sym(x.clone())]),
        ])],
    );
    with_symbol_facts(
        HashMap::from([(a, Facts::from_names(&["nonzero"]))]),
        || {
            let result = series_expansion_dir(&source, &x, &e("0"), 3, Direction::Plus).unwrap();
            assert_terms(&result, &[(0, "1")]);
            assert_eq!(
                limit_dir(&source, &x, &e("0"), Direction::Both).unwrap(),
                e("1")
            );
        },
    );
    let rational = expanded("Abs(1/(1-x))", 4, Direction::Plus);
    assert_terms(&rational, &[(0, "1"), (1, "1"), (2, "1"), (3, "1")]);
    let regular = expanded("Abs(sec(x))", 5, Direction::Plus);
    assert_terms(&regular, &[(0, "1"), (2, "1/2"), (4, "5/24")]);
}

#[test]
fn local_coordinate_copies_are_bounded_before_expansion() {
    let x = Symbol::new("x");
    let source = Expr::Function(
        "Abs".into(),
        vec![Expr::Add(vec![Expr::Sym(x.clone()); 64])],
    );
    let point = Expr::Add(vec![e("1"); 64]);
    assert!(series_expansion_dir(&source, &x, &point, 3, Direction::Plus).is_err());
    assert!(limit_dir(&source, &x, &point, Direction::Plus).is_err());
}
