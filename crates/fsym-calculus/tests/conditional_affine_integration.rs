//! Conditional elementary primitives: exact branch identities, typed parameters,
//! and original-source admission. These checks do not claim definite convergence.

use fsym_calculus::integrate::{antiderivative, definite};
use fsym_calculus::{diff, integrate};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::{Constant, Expr, Symbol, SymbolIdentity, parse};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

fn e(source: &str) -> Expr {
    // Ordinary expression fixtures use the evaluated constructor form (e.g.
    // x/2 has an exact rational coefficient). Held-domain fixtures below are
    // built explicitly and are never passed through this normalization.
    parse(source).unwrap().subs(&HashMap::new())
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.to_string(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn same(actual: Expr, expected: Expr) {
    // A mathematical exact-zero comparator, separate from the public shape
    // checks below. This never accepts a numerical sample as an identity.
    let residual = fsym_simplify::try_simplify(
        &fsym_simplify::try_expand(&(actual.clone() - expected.clone())).unwrap(),
    )
    .unwrap();
    assert!(
        residual.is_zero(),
        "expected {expected}, got {actual}: {residual}"
    );
}

fn branches<'a>(value: &'a Expr, parameter: &Symbol) -> (&'a Expr, &'a Expr) {
    let Expr::Function(name, pairs) = value else {
        panic!("expected a conditional primitive, got {value}");
    };
    assert_eq!(name, "Piecewise");
    assert_eq!(pairs.len(), 2);
    let pair = |index: usize| {
        let Expr::Function(name, items) = &pairs[index] else {
            panic!("malformed Piecewise branch");
        };
        assert_eq!(name, "Tuple");
        assert_eq!(items.len(), 2);
        items
    };
    assert_eq!(
        pair(0)[1],
        Expr::Function(
            "Unequality".to_string(),
            vec![Expr::Sym(parameter.clone()), Expr::from_i64(0)],
        )
    );
    assert_eq!(pair(1)[1], Expr::symbol("True"));
    (&pair(0)[0], &pair(1)[0])
}

#[test]
fn five_entire_kernels_keep_the_zero_slope_branch() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    for (source, nonzero, zero) in [
        ("exp(a*x)", "exp(a*x)/a", "x"),
        ("sin(a*x+b)", "-cos(a*x+b)/a", "x*sin(b)"),
        ("cos(a*x+b)", "sin(a*x+b)/a", "x*cos(b)"),
        ("sinh(a*x+b)", "cosh(a*x+b)/a", "x*sinh(b)"),
        ("cosh(a*x+b)", "sinh(a*x+b)/a", "x*cosh(b)"),
        ("sin(a*x)", "-cos(a*x)/a", "0"),
        ("sinh(a*x)", "cosh(a*x)/a", "0"),
    ] {
        let source = e(source);
        let result = integrate(&source, &x).unwrap();
        let (generic, degenerate) = branches(&result, &a);
        same(generic.clone(), e(nonzero));
        same(degenerate.clone(), e(zero));
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&["nonzero"]))]),
            || {
                same(diff(generic, &x), source.clone());
            },
        );
        same(
            diff(degenerate, &x),
            source.subs(&HashMap::from([(a.clone(), Expr::from_i64(0))])),
        );
        for value in [e("-3"), e("1/2"), e("5")] {
            let map = HashMap::from([(a.clone(), value)]);
            same(diff(&generic.subs(&map), &x), source.subs(&map));
        }
    }
}

#[test]
fn rational_scales_and_correlated_intercepts_specialize_exactly() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    for (source, nonzero, zero) in [
        ("exp(-3*a*x)", "-exp(-3*a*x)/(3*a)", "x"),
        ("sin(2*a*x+a+b)", "-cos(2*a*x+a+b)/(2*a)", "x*sin(b)"),
        ("cos(a*x/2+a*b)", "2*sin(a*x/2+a*b)/a", "x"),
        ("sinh(a*x+a*a+b)", "cosh(a*x+a*a+b)/a", "x*sinh(b)"),
        ("cosh(a*x+b*b+2)", "sinh(a*x+b*b+2)/a", "x*cosh(b*b+2)"),
        ("exp(a*x+a)", "exp(a*x+a)/a", "x"),
    ] {
        let source = e(source);
        let result = integrate(&source, &x).unwrap();
        let (generic, degenerate) = branches(&result, &a);
        same(generic.clone(), e(nonzero));
        same(degenerate.clone(), e(zero));
        with_symbol_facts(
            HashMap::from([(a.clone(), Facts::from_names(&["nonzero"]))]),
            || {
                same(diff(generic, &x), source.clone());
            },
        );
        same(
            diff(degenerate, &x),
            source.subs(&HashMap::from([(a.clone(), Expr::from_i64(0))])),
        );
    }
}

#[test]
fn exponential_coefficients_keep_the_pinned_branch_placement() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    for (source, nonzero, zero) in [
        ("2*exp(a*x)", "2*exp(a*x)/a", "2*x"),
        ("b*exp(a*x)", "b*exp(a*x)/a", "b*x"),
        ("exp(b)*exp(a*x)", "exp(b)*exp(a*x)/a", "x*exp(b)"),
        ("(a+1)*exp(2*a*x)", "(a+1)*exp(2*a*x)/(2*a)", "x*(a+1)"),
        ("sin(a)*exp(2*a*x)", "sin(a)*exp(2*a*x)/(2*a)", "x*sin(a)"),
        ("(a+b)*exp(a*x)", "(a+b)*exp(a*x)/a", "x*(a+b)"),
    ] {
        let result = integrate(&e(source), &x).unwrap();
        let (generic, degenerate) = branches(&result, &a);
        same(generic.clone(), e(nonzero));
        same(degenerate.clone(), e(zero));
        let at_zero = HashMap::from([(a.clone(), Expr::from_i64(0))]);
        same(
            diff(&degenerate.subs(&at_zero), &x),
            e(source).subs(&at_zero),
        );
    }
}

#[test]
fn an_entire_cancelled_exponential_scale_needs_no_condition() {
    let x = Symbol::new("x");
    for (source, expected) in [
        ("a*exp(a*x)", "exp(a*x)"),
        ("a*b*exp(2*a*x)", "b*exp(2*a*x)/2"),
        ("a*a*exp(2*a*x)", "a*exp(2*a*x)/2"),
    ] {
        let result = integrate(&e(source), &x).unwrap();
        same(result.clone(), e(expected));
        assert!(!matches!(result, Expr::Function(ref name, _) if name == "Piecewise"));
        same(diff(&result, &x), e(source));
    }
}

#[test]
fn typed_axis_slope_and_shift_keep_distinct_equal_names() {
    let x = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([41; 32]));
    let slope = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([42; 32]));
    let shift = Symbol::new("a");
    let source = function(
        "sin",
        Expr::Sym(slope.clone()) * Expr::Sym(x.clone()) + Expr::Sym(shift.clone()),
    );
    let result = integrate(&source, &x).unwrap();
    let (generic, degenerate) = branches(&result, &slope);
    let expected = Expr::Sym(x.clone()) * function("sin", Expr::Sym(shift.clone()));
    same(degenerate.clone(), expected);
    assert_eq!(
        result.free_symbols().into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([x.clone(), slope.clone(), shift.clone(), Symbol::new("True")])
    );
    let substitution = HashMap::from([(slope.clone(), Expr::from_i64(2))]);
    same(
        diff(&generic.subs(&substitution), &x),
        source.subs(&substitution),
    );
    with_symbol_facts(
        HashMap::from([(shift, Facts::from_names(&["nonzero"]))]),
        || {
            let still_conditional = integrate(&source, &x).unwrap();
            branches(&still_conditional, &slope);
        },
    );
}

#[test]
fn declared_nonzero_slopes_retain_the_existing_unconditional_table() {
    let x = Symbol::new("x");
    let a = Symbol::new("a");
    with_symbol_facts(
        HashMap::from([(a, Facts::from_names(&["nonzero"]))]),
        || {
            for (source, expected) in [
                ("exp(a*x+b)", "exp(a*x+b)/a"),
                ("sin(a*x+b)", "-cos(a*x+b)/a"),
                ("sinh(a*x+b)", "cosh(a*x+b)/a"),
                ("cosh(a*x+b)", "sinh(a*x+b)/a"),
            ] {
                let actual = integrate(&e(source), &x).unwrap();
                same(actual.clone(), e(expected));
                same(diff(&actual, &x), e(source));
            }
        },
    );
}

#[test]
fn unsupported_shifted_exponentials_cannot_reenter_through_constant_splitting() {
    let x = Symbol::new("x");
    for source in ["exp(a*x+b)", "exp(a*x+2)", "2*exp(a*x+b)"] {
        assert!(antiderivative(&e(source), &x).is_none(), "{source}");
        assert!(integrate(&e(source), &x).is_err(), "{source}");
    }
}

#[test]
fn composite_slopes_and_nonlinear_arguments_stay_outside_this_lane() {
    let x = Symbol::new("x");
    for source in ["sin(a*b*x+b)", "exp(a*b*x)", "sin(a*x*x+b)"] {
        assert!(antiderivative(&e(source), &x).is_none(), "{source}");
    }
}

#[test]
fn held_original_argument_holes_are_not_erased_into_conditional_primitives() {
    let x = Symbol::new("x");
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), power(e("x-1"), -1)]),
        Expr::Mul(vec![e("x-1"), power(e("x-1"), -1)]),
        Expr::Mul(vec![Expr::from_i64(0), function("log", e("x"))]),
        power(Expr::from_i64(0), 0),
        power(Expr::from_i64(0), -1),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Const(Constant::Infinity)]),
    ] {
        let source = function("sin", Expr::Add(vec![e("a*x+b"), hidden]));
        assert!(antiderivative(&source, &x).is_none(), "{source}");
        assert!(integrate(&source, &x).is_err(), "{source}");
    }
}

#[test]
fn fallback_constant_extraction_preserves_the_original_admission_scope() {
    let x = Symbol::new("x");
    for invalid_coefficient in [
        Expr::Mul(vec![Expr::from_i64(0), power(e("b"), -1)]),
        Expr::Mul(vec![e("b"), power(e("b"), -1)]),
        Expr::Mul(vec![Expr::from_i64(0), function("log", e("b"))]),
    ] {
        let source = Expr::Mul(vec![invalid_coefficient, e("exp(a*x)")]);
        assert!(integrate(&source, &x).is_err(), "{source}");
    }
}

#[test]
fn numeric_and_proved_nonzero_definite_routes_keep_their_values() {
    let x = Symbol::new("x");
    same(
        definite(&e("exp(2*x)"), &x, &e("0"), &e("1")).unwrap(),
        e("(exp(2)-1)/2"),
    );
    with_symbol_facts(
        HashMap::from([(Symbol::new("a"), Facts::from_names(&["positive"]))]),
        || {
            same(
                definite(&e("exp(a*x)"), &x, &e("0"), &e("1")).unwrap(),
                e("(exp(a)-1)/a"),
            );
        },
    );
    // Conditional indefinite identities alone do not discharge parameterized
    // endpoint/convergence obligations in the existing definite lane.
    assert!(definite(&e("exp(a*x)"), &x, &e("0"), &e("1")).is_none());
}
