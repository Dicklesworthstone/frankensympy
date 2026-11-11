use fsym_calculus::diff;
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_simplify::simplify;

#[test]
fn evaluated_primitives_bind_body_coordinates_but_not_the_point() {
    for (source, variable, expected) in [
        ("Integral(x*y, Tuple(x,2))", "x", "0"),
        ("Integral(x*y, Tuple(x,2))", "y", "Integral(x, Tuple(x,2))"),
        ("Integral(x*y, Tuple(x,y))", "x", "0"),
        (
            "Integral(x*y, Tuple(x,y))",
            "y",
            "y^2+Integral(x, Tuple(x,y))",
        ),
        ("Integral(x*y, Tuple(x,x))", "x", "x*y"),
        (
            "Integral(x*y, Tuple(x,2*y))",
            "y",
            "4*y^2+Integral(x, Tuple(x,2*y))",
        ),
    ] {
        assert_eq!(
            diff(&parse(source).unwrap(), &Symbol::new(variable)),
            // diff returns the existing simplifier's normal form. The raw
            // parser has a different Add ordering, so construct the expected
            // normal form without changing the hand-derived formula.
            simplify(&parse(expected).unwrap()),
            "d/d{variable} {source}",
        );
    }
}

#[test]
fn point_derivatives_keep_exact_symbol_identity() {
    let coordinate = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let parameter = Symbol::new("x");
    let body = Expr::Sym(coordinate.clone()) * Expr::Sym(parameter.clone());
    let limit = Expr::Function(
        "Tuple".into(),
        vec![Expr::Sym(coordinate.clone()), Expr::Sym(parameter.clone())],
    );
    let source = Expr::Function("Integral".into(), vec![body, limit.clone()]);
    assert_eq!(diff(&source, &coordinate), Expr::from_i64(0));
    assert_eq!(
        diff(&source, &parameter),
        simplify(
            &(Expr::Sym(parameter.clone()).pow(Expr::from_i64(2))
                + Expr::Function("Integral".into(), vec![Expr::Sym(coordinate), limit])),
        ),
    );
}

#[test]
fn independent_parameter_avoids_unnecessary_boundary_substitution() {
    let source = parse("Integral(sin(x), Tuple(x,y))").unwrap();
    assert_eq!(diff(&source, &Symbol::new("z")), Expr::from_i64(0));
    assert_eq!(diff(&source, &Symbol::new("y")), parse("sin(y)").unwrap());
}
