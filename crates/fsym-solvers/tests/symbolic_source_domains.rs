//! Generic polynomial formulas still respect proved original-source poles.
//!
//! Coefficient reconstruction may remove a held zero factor or reciprocal.
//! Candidate admission must inspect that source before publishing its roots,
//! without interpreting an unresolved parameter condition as false.

use fsym_core::elementary::eval_pow;
use fsym_core::{Constant, Expr, Symbol, SymbolIdentity};
use fsym_solvers::{solve, univariate::solve_univariate};
use std::collections::HashMap;
use std::sync::Arc;

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn held_zero_log(argument: Expr) -> Expr {
    Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("log".into(), vec![argument]),
    ])
}

fn assert_roots(source: &Expr, coordinate: &Symbol, expected: Vec<Expr>) {
    assert_eq!(
        solve_univariate(source, coordinate),
        Ok(expected.clone()),
        "univariate source: {source}"
    );
    assert_eq!(
        solve(source, coordinate),
        Ok(expected),
        "public source: {source}"
    );
}

fn assert_zero(expression: Expr) {
    let expanded = fsym_simplify::try_expand(&expression).unwrap();
    let simplified = fsym_simplify::try_simplify(&expanded).unwrap();
    // Reducing a squared radical can expose a discriminant sum inside a
    // product. Distribute that new sum before checking the exact residual.
    let expanded = fsym_simplify::try_expand(&simplified).unwrap();
    let simplified = fsym_simplify::try_simplify(&expanded).unwrap();
    assert!(simplified.is_zero(), "nonzero residual: {simplified}");
}

#[test]
fn held_zero_logarithms_do_not_publish_their_original_poles() {
    let x = Symbol::new("symbolic_source_x");
    let factor = Expr::Sym(x.clone()) - Expr::from_i64(1);
    for polynomial in [factor.clone(), power(factor.clone(), 2)] {
        let source = Expr::Add(vec![polynomial, held_zero_log(factor.clone())]);
        assert_roots(&source, &x, vec![]);
    }
    let undefined_constant = Expr::Add(vec![factor, held_zero_log(Expr::from_i64(0))]);
    assert_roots(&undefined_constant, &x, vec![]);
}

#[test]
fn nonrational_nested_reciprocals_retain_the_original_hole() {
    let x = Symbol::new("symbolic_source_x");
    let factor = Expr::Sym(x.clone()) - Expr::from_i64(1);
    let source = Expr::Mul(vec![
        Expr::Const(Constant::Pi),
        power(power(factor, -1), -1),
    ]);
    // The rational parser cannot admit pi. Reconstructing the resulting
    // symbolic linear polynomial must not restore its missing point x=1.
    assert_roots(&source, &x, vec![]);
}

#[test]
fn held_zero_products_cannot_erase_literal_nonfinite_factors() {
    let x = Symbol::new("symbolic_source_x");
    let polynomial = Expr::Sym(x.clone()) - Expr::from_i64(1);
    for constant in [
        Constant::Infinity,
        Constant::NegativeInfinity,
        Constant::ComplexInfinity,
        Constant::NaN,
    ] {
        let source = Expr::Add(vec![
            polynomial.clone(),
            Expr::Mul(vec![Expr::from_i64(0), Expr::Const(constant)]),
        ]);
        assert_roots(&source, &x, vec![]);
        let source = Expr::Sym(x.clone()) - Expr::Const(constant);
        assert_roots(&source, &x, vec![]);
    }
    for name in ["exp", "log"] {
        let nonfinite_composition =
            Expr::Function(name.into(), vec![Expr::Const(Constant::Infinity)]);
        let source = Expr::Add(vec![
            polynomial.clone(),
            Expr::Mul(vec![Expr::from_i64(0), nonfinite_composition.clone()]),
        ]);
        assert_roots(&source, &x, vec![]);
        let source = Expr::Sym(x.clone()) - nonfinite_composition;
        assert_roots(&source, &x, vec![]);
    }
    for (finite_composition, value) in [
        (
            Expr::Function("exp".into(), vec![Expr::Const(Constant::NegativeInfinity)]),
            Expr::from_i64(0),
        ),
        (
            Expr::Function("atan".into(), vec![Expr::Const(Constant::Infinity)]),
            Expr::rational(1, 2).unwrap() * Expr::Const(Constant::Pi),
        ),
        (
            power(Expr::Const(Constant::Infinity), -1),
            Expr::from_i64(0),
        ),
    ] {
        let source = Expr::Add(vec![
            polynomial.clone(),
            Expr::Mul(vec![Expr::from_i64(0), finite_composition.clone()]),
        ]);
        assert_roots(&source, &x, vec![Expr::from_i64(1)]);
        let source = Expr::Sym(x.clone()) - finite_composition;
        assert_roots(&source, &x, vec![value]);
    }
}

#[test]
fn piecewise_domain_checks_only_follow_selected_branches() {
    let x = Symbol::new("symbolic_source_x");
    let xe = Expr::Sym(x.clone());
    let polynomial = xe.clone() - Expr::from_i64(1);
    let pole = Expr::Function("log".into(), vec![polynomial.clone()]);
    let piecewise = |condition| {
        Expr::Function(
            "Piecewise".into(),
            vec![
                Expr::Function("Tuple".into(), vec![pole.clone(), condition]),
                Expr::Function(
                    "Tuple".into(),
                    vec![Expr::from_i64(1), Expr::symbol("True")],
                ),
            ],
        )
    };
    for (condition, expected) in [
        (
            Expr::Function("StrictLessThan".into(), vec![xe.clone(), Expr::from_i64(0)]),
            vec![Expr::from_i64(1)],
        ),
        (
            Expr::Function("StrictGreaterThan".into(), vec![xe, Expr::from_i64(0)]),
            vec![],
        ),
        (
            Expr::symbol("symbolic_source_condition"),
            vec![Expr::from_i64(1)],
        ),
    ] {
        let source = Expr::Add(vec![
            polynomial.clone(),
            Expr::Mul(vec![Expr::from_i64(0), piecewise(condition)]),
        ]);
        assert_roots(&source, &x, expected);
    }
}

#[test]
fn regular_roots_survive_original_domain_filtering() {
    let x = Symbol::new("symbolic_source_x");
    let xe = Expr::Sym(x.clone());
    let first = xe.clone() - Expr::from_i64(1);
    let second = xe - Expr::from_i64(2);
    for source in [
        Expr::Add(vec![
            first.clone() * second.clone(),
            held_zero_log(first.clone()),
        ]),
        Expr::Mul(vec![
            Expr::Const(Constant::Pi),
            Expr::Add(vec![power(power(first, -1), -1), Expr::from_i64(-1)]),
        ]),
    ] {
        assert_roots(&source, &x, vec![Expr::from_i64(2)]);
    }
}

#[test]
fn unknown_parameter_conditions_remain_generic_solutions() {
    let x = Symbol::new("symbolic_source_x");
    let xe = Expr::Sym(x.clone());
    let parameter = Expr::symbol("symbolic_source_parameter");
    for source in [
        Expr::Add(vec![
            xe.clone() - parameter.clone(),
            held_zero_log(xe.clone() - Expr::from_i64(1)),
        ]),
        xe * power(parameter.clone(), -1) - Expr::from_i64(1),
    ] {
        assert_roots(&source, &x, vec![parameter.clone()]);
    }
}

#[test]
fn an_unknown_constraint_cannot_hide_a_proved_pole() {
    let x = Symbol::new("symbolic_source_x");
    let factor = Expr::Sym(x.clone()) - Expr::from_i64(1);
    let known_pole = held_zero_log(factor.clone());
    let unknown = held_zero_log(Expr::symbol("symbolic_source_parameter"));
    for constraints in [
        vec![known_pole.clone(), unknown.clone()],
        vec![unknown, known_pole],
    ] {
        let source = Expr::Add(vec![
            factor.clone(),
            constraints[0].clone(),
            constraints[1].clone(),
        ]);
        assert_roots(&source, &x, vec![]);
    }
}

#[test]
fn ordinary_symbolic_and_irrational_quadratic_formulas_are_preserved() {
    let x = Symbol::new("symbolic_source_x");
    let xe = Expr::Sym(x.clone());
    let a = Expr::symbol("symbolic_source_a");
    let b = Expr::symbol("symbolic_source_b");
    let c = Expr::symbol("symbolic_source_c");
    let symbolic = a * power(xe.clone(), 2) + b * xe.clone() + c;
    let roots = solve_univariate(&symbolic, &x).unwrap();
    assert_eq!(roots.len(), 2);
    assert_eq!(solve(&symbolic, &x), Ok(roots.clone()));
    for root in roots {
        assert!(!root.free_symbols().contains(&x));
        assert_zero(symbolic.subs(&HashMap::from([(x.clone(), root)])));
    }

    let irrational = Expr::Add(vec![
        power(xe.clone(), 2) - Expr::from_i64(2),
        held_zero_log(xe - Expr::from_i64(1)),
    ]);
    let roots = solve_univariate(&irrational, &x).unwrap();
    let square_root = eval_pow(Expr::from_i64(2), Expr::rational(1, 2).unwrap());
    assert_eq!(roots.len(), 2);
    assert_zero(roots[0].clone() + square_root.clone());
    assert_zero(roots[1].clone() - square_root);
    assert_eq!(solve(&irrational, &x), Ok(roots));
}

#[test]
fn source_domain_substitution_uses_the_full_coordinate_identity() {
    let plain = Symbol::new("symbolic_source_shared");
    let typed = Symbol::with_identity(
        "symbolic_source_shared",
        SymbolIdentity::from_assumptions_digest([83; 32]),
    );
    let source = Expr::Add(vec![
        Expr::Sym(typed.clone()) - Expr::from_i64(1),
        held_zero_log(Expr::Sym(plain.clone()) - Expr::from_i64(1)),
    ]);
    // The plain atom is a parameter. Giving the typed coordinate the value
    // one must not also substitute that parameter into its logarithmic pole.
    assert_roots(&source, &typed, vec![Expr::from_i64(1)]);

    let source = Expr::Add(vec![
        Expr::Sym(plain.clone()) - Expr::Sym(typed.clone()),
        held_zero_log(Expr::Sym(plain.clone()) - Expr::from_i64(1)),
    ]);
    assert_roots(&source, &plain, vec![Expr::Sym(typed)]);
}
