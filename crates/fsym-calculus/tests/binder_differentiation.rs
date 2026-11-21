//! Formal differentiation must respect each limit's lexical scope.
//!
//! These are hand-derived fundamental-theorem/Leibniz identities, not
//! convergence or interchange certificates for arbitrary integrands.

use fsym_calculus::diff;
use fsym_calculus::proof::{verified_diff, verify_diff_derivation};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_simplify::simplify;

fn check(source: &str, variable: &str, expected: &str) {
    assert_eq!(
        diff(&parse(source).unwrap(), &Symbol::new(variable)),
        simplify(&parse(expected).unwrap()),
        "d/d{variable} {source}",
    );
}

#[test]
fn an_outer_primitive_keeps_a_coordinate_shadowed_in_an_inner_definite_integral() {
    // The first source equals x/2, so its derivative cannot be zero.
    check(
        "Integral(x, Tuple(x,0,1), x)",
        "x",
        "Integral(x, Tuple(x,0,1))",
    );
    check(
        "Integral(x*y, Tuple(x,0,1), Tuple(x))",
        "x",
        "Integral(x*y, Tuple(x,0,1))",
    );
    check("Integral(x, x, x)", "x", "Integral(x, x)");
    // Reversing the scopes binds the formerly free coordinate instead.
    check("Integral(x*y, x, Tuple(x,0,1))", "x", "0");
}

#[test]
fn a_definite_coordinate_is_bound_only_in_the_body_not_its_endpoints() {
    check("Integral(x*y, Tuple(x,0,x))", "x", "x*y");
    check("Integral(x^2, Tuple(x,x,x^2))", "x", "2*x^5-x^2");
    check("Integral(1, Tuple(x,x,2*x))", "x", "1");
    check(
        "Integral(x*y, Tuple(x,0,x))",
        "y",
        "Integral(x, Tuple(x,0,x))",
    );
}

#[test]
fn nested_moving_limits_use_every_boundary_in_inner_to_outer_order() {
    check(
        "Integral(x*y, Tuple(x,0,y), Tuple(y,0,z))",
        "z",
        "Integral(x*z, Tuple(x,0,z))",
    );
    check(
        "Integral(x*y, Tuple(x,0,z), Tuple(y,0,z))",
        "z",
        "Integral(x*z, Tuple(x,0,z))+Integral(y*z, Tuple(y,0,z))",
    );
    check("Integral(x*y, Tuple(x,0,y), Tuple(y,0,z))", "y", "0");
    check(
        "Integral(x*y*z, Tuple(x,0,1), Tuple(y,0,2))",
        "z",
        "Integral(x*y, Tuple(x,0,1), Tuple(y,0,2))",
    );
}

#[test]
fn a_boundary_substitution_cannot_capture_a_free_parameter() {
    // The integral is x^2/4. In the boundary term y -> x, the inner
    // bound x must be renamed so that the inserted free x stays free.
    let source = parse("Integral(x*y, Tuple(x,0,1), Tuple(y,0,x))").unwrap();
    let result = diff(&source, &Symbol::new("x"));
    assert_eq!(
        result,
        simplify(&parse("Integral(x*x_1, Tuple(x_1,0,1))").unwrap()),
    );
    assert_eq!(result.free_symbols(), vec![Symbol::new("x")]);
}

#[test]
fn an_unsupported_boundary_into_an_indefinite_coordinate_stays_formal() {
    // The first source evaluates to x^4/4, so its derivative is x^3.
    // Naively substituting y -> x inside the primitive Integral(x*y,x)
    // would produce the false boundary x^3/3 instead of x^3/2.
    for source in [
        "Integral(x*y, x, Tuple(y,0,x))",
        "Integral(x*y, x, x, Tuple(y,0,x))",
        "Integral(Integral(x*y, x), Tuple(y,0,x))",
        "Integral(x*y*z, x, Tuple(y,0,x+z))",
    ] {
        let original = parse(source).unwrap();
        let variable = Symbol::new("x");
        let expected = Expr::Function(
            "diff".into(),
            vec![original.clone(), Expr::Sym(variable.clone())],
        );
        assert_eq!(diff(&original, &variable), simplify(&expected));
    }
    // Substitution through an indefinite primitive remains valid when
    // the inserted parameter cannot collide with its free coordinate.
    check("Integral(x*y, x, Tuple(y,0,z))", "z", "Integral(x*z, x)");
}

#[test]
fn evaluation_point_limits_compose_with_definite_and_indefinite_limits() {
    check(
        "Integral(x*y, Tuple(x,z), Tuple(y,0,1))",
        "z",
        "Integral(y*z, Tuple(y,0,1))",
    );
    check(
        "Integral(x*y, Tuple(x,0,1), Tuple(y,z))",
        "z",
        "Integral(x*z, Tuple(x,0,1))",
    );
    check(
        "Integral(x*y, Tuple(x,z), y)",
        "y",
        "Integral(x*y, Tuple(x,z))",
    );
    check(
        "Integral(x, Tuple(x,0,1), Tuple(x,z))",
        "z",
        "Integral(x, Tuple(x,0,1))",
    );
}

#[test]
fn equal_printed_coordinates_keep_their_distinct_native_identities() {
    let coordinate = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let parameter = Symbol::new("x");
    let body = Expr::Sym(coordinate.clone()) * Expr::Sym(parameter.clone());
    let limit = Expr::Function(
        "Tuple".into(),
        vec![
            Expr::Sym(coordinate.clone()),
            Expr::from_i64(0),
            Expr::Sym(parameter.clone()),
        ],
    );
    let source = Expr::Function("Integral".into(), vec![body, limit.clone()]);
    assert_eq!(diff(&source, &coordinate), Expr::from_i64(0));
    let expected = Expr::Sym(parameter.clone()).pow(Expr::from_i64(2))
        + Expr::Function("Integral".into(), vec![Expr::Sym(coordinate), limit]);
    assert_eq!(diff(&source, &parameter), simplify(&expected));
}

#[test]
fn fixed_sum_limits_differentiate_the_body_but_moving_discrete_bounds_stay_formal() {
    check("Sum(x*y, Tuple(x,1,3))", "y", "Sum(x, Tuple(x,1,3))");
    check("Sum(x*y, Tuple(x,1,y), Tuple(y,1,3))", "y", "0");
    for source in ["Sum(x, Tuple(x,1,y))", "Sum(x, Tuple(x,1,y), Tuple(z,1,3))"] {
        let source = parse(source).unwrap();
        let result = diff(&source, &Symbol::new("y"));
        assert!(result.to_string().contains("diff("), "{result}");
    }
}

#[test]
fn unchanged_endpoints_do_not_create_spurious_undefined_boundary_terms() {
    check("Integral(log(x), Tuple(x,0,1))", "z", "0");
    check(
        "Integral(y*log(x), Tuple(x,0,1))",
        "y",
        "Integral(log(x), Tuple(x,0,1))",
    );
}

#[test]
fn malformed_binders_are_preserved_without_panics_or_false_zero_derivatives() {
    let variable = Symbol::new("x");
    for source in [
        "Integral()",
        "Sum()",
        "Integral(x)",
        "Sum(x)",
        "Integral(x, Tuple())",
        "Integral(x, Tuple(1,0,1))",
        "Integral(x, Tuple(x,0,1,2))",
        "Integral(x, Tuple(x,0,1,2), Tuple(x,0,1))",
        "Sum(x, x)",
        "Sum(x, Tuple(x,1))",
    ] {
        let source = parse(source).unwrap();
        let expected = Expr::Function(
            "diff".into(),
            vec![source.clone(), Expr::Sym(variable.clone())],
        );
        assert_eq!(diff(&source, &variable), expected);
    }
}

#[test]
fn formal_leibniz_results_do_not_gain_unproved_interchange_certificates() {
    let source = parse("Integral(x*y, Tuple(x,0,1), y)").unwrap();
    let variable = Symbol::new("y");
    let (derivative, proposal) = verified_diff(&source, &variable);
    assert!(verify_diff_derivation(&proposal, &source, &variable, &derivative).is_err());
}
