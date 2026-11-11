//! Lexical scope and capture regressions for the native expression API.
//!
//! Native substitution is deliberately capture-avoiding; Python's profile
//! substitution rules are tested separately at the compatibility boundary.

use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use std::collections::HashMap;

fn expression(source: &str) -> Expr {
    parse(source).unwrap()
}

fn replace(source: &str, variable: &str, replacement: &str) -> Expr {
    expression(source).subs(&HashMap::from([(
        Symbol::new(variable),
        expression(replacement),
    )]))
}

// A small independent lexical evaluator makes the finite-sum counterexamples
// observable numerically without using Expr::subs to evaluate a bound body.
fn evaluate(expr: &Expr, environment: &HashMap<Symbol, i64>) -> i64 {
    match expr {
        Expr::Integer(value) => value.to_i64().unwrap(),
        Expr::Sym(symbol) => environment[symbol],
        Expr::Add(terms) => terms.iter().map(|term| evaluate(term, environment)).sum(),
        Expr::Mul(terms) => terms
            .iter()
            .map(|term| evaluate(term, environment))
            .product(),
        Expr::Function(name, args) if name == "Sum" => {
            evaluate_sum(&args[0], &args[1..], environment)
        }
        _ => panic!("outside the independent finite-sum test grammar: {expr:?}"),
    }
}

fn evaluate_sum(body: &Expr, limits: &[Expr], environment: &HashMap<Symbol, i64>) -> i64 {
    let Some((limit, inner_limits)) = limits.split_last() else {
        return evaluate(body, environment);
    };
    let Expr::Function(name, parts) = limit else {
        panic!("expected a finite limit")
    };
    assert_eq!(name, "Tuple");
    let [Expr::Sym(variable), lower, upper] = parts.as_slice() else {
        panic!("expected a finite limit")
    };
    let lower = evaluate(lower, environment);
    let upper = evaluate(upper, environment);
    let mut scope = environment.clone();
    (lower..=upper)
        .map(|value| {
            scope.insert(variable.clone(), value);
            evaluate_sum(body, inner_limits, &scope)
        })
        .sum()
}

#[test]
fn outer_variables_bind_inner_limits() {
    for operator in ["Integral", "Sum", "Product"] {
        let source = format!("{operator}(x+y, Tuple(x,1,y), Tuple(y,1,3))");
        assert_eq!(replace(&source, "y", "9"), expression(&source));
        assert_eq!(replace(&source, "x", "9"), expression(&source));
    }
    assert_eq!(
        evaluate(
            &replace("Sum(x+y, Tuple(x,1,y), Tuple(y,1,3))", "y", "9"),
            &HashMap::new(),
        ),
        24,
    );
}

#[test]
fn alpha_renaming_reaches_dependent_inner_limits() {
    let result = replace("Sum(z, Tuple(x,1,y), Tuple(y,1,3))", "z", "y");
    assert_eq!(result.free_symbols(), vec![Symbol::new("y")]);
    assert_eq!(
        evaluate(&result, &HashMap::from([(Symbol::new("y"), 5)])),
        30,
    );
}

#[test]
fn substitution_into_a_limit_also_avoids_capture() {
    let result = replace("Sum(x, Tuple(x,1,z), Tuple(y,1,3))", "z", "y");
    assert_eq!(result.free_symbols(), vec![Symbol::new("y")]);
    assert_eq!(
        evaluate(&result, &HashMap::from([(Symbol::new("y"), 5)])),
        45,
    );
}

#[test]
fn a_limit_variable_does_not_bind_its_own_endpoints() {
    assert_eq!(
        replace("Integral(x, Tuple(x,0,x))", "x", "3"),
        expression("Integral(x, Tuple(x,0,3))"),
    );
    let source = "Sum(x, Tuple(x,1,x), Tuple(x,1,3))";
    assert_eq!(replace(source, "x", "9"), expression(source));
    assert_eq!(evaluate(&expression(source), &HashMap::new()), 10);
}

#[test]
fn fresh_binders_cannot_be_consumed_by_other_map_entries() {
    let result = expression("Lambda(x, x+y)").subs(&HashMap::from([
        (Symbol::new("y"), expression("x")),
        (Symbol::new("x_1"), expression("7")),
    ]));
    assert_eq!(result, expression("Lambda(x_2, x+x_2)"));
    assert_eq!(result.free_symbols(), vec![Symbol::new("x")]);
}

#[test]
fn compound_replacements_avoid_capture_for_all_native_binders() {
    for (source, expected) in [
        ("Lambda(x, x+cos(y))", "Lambda(x_1, x+x_1)"),
        ("Lambda(Tuple(x), x+cos(y))", "Lambda(Tuple(x_1), x+x_1)"),
        (
            "Integral(x+cos(y), Tuple(x,0,1))",
            "Integral(x+x_1, Tuple(x_1,0,1))",
        ),
        ("Sum(x+cos(y), Tuple(x,1,3))", "Sum(x+x_1, Tuple(x_1,1,3))"),
        (
            "Subs(x+cos(y), Tuple(x), Tuple(1))",
            "Subs(x+x_1, Tuple(x_1), Tuple(1))",
        ),
    ] {
        let result = expression(source).subs_expr(&expression("cos(y)"), &expression("x"));
        assert_eq!(result, expression(expected), "{source}");
        assert_eq!(result.free_symbols(), vec![Symbol::new("x")]);
    }
}

#[test]
fn compound_substitution_respects_nested_scope_and_own_endpoints() {
    let source = expression("Integral(x, Tuple(x,0,y+1), Tuple(y,0,y+1))");
    assert_eq!(
        source.subs_expr(&expression("y+1"), &expression("z")),
        expression("Integral(x, Tuple(x,0,y+1), Tuple(y,0,z))"),
    );
    let source = expression("Sum(cos(z), Tuple(x,1,y), Tuple(y,1,3))");
    let result = source.subs_expr(&expression("cos(z)"), &expression("y"));
    assert_eq!(result.free_symbols(), vec![Symbol::new("y")]);
    assert_eq!(
        evaluate(&result, &HashMap::from([(Symbol::new("y"), 5)])),
        30,
    );
}

#[test]
fn absent_compound_patterns_do_not_rename_binders() {
    for source in [
        "Lambda(x, x+y)",
        "Integral(x+y, Tuple(x,0,1))",
        "Sum(x, Tuple(x,1,y), Tuple(y,1,3))",
    ] {
        let source = expression(source);
        assert_eq!(
            source.subs_expr(&expression("cos(z)"), &expression("x")),
            source
        );
    }
}

#[test]
fn indefinite_variables_are_free_and_support_point_substitution() {
    assert_eq!(
        replace("Integral(x*y, Tuple(x))", "x", "z"),
        expression("Integral(y*z, Tuple(z))"),
    );
    assert_eq!(
        replace("Integral(x*y, x)", "x", "2"),
        expression("Integral(x*y, Tuple(x,2))"),
    );
    assert_eq!(
        replace("Integral(x+y, Tuple(x))", "y", "x"),
        expression("Integral(2*x, Tuple(x))"),
    );
    let source = expression("Integral(x*y, Tuple(x))");
    assert_eq!(
        source.subs(&HashMap::from([
            (Symbol::new("x"), expression("y")),
            (Symbol::new("y"), expression("2")),
        ])),
        expression("Integral(2*y, Tuple(y))"),
    );
}

#[test]
fn point_substitution_does_not_capture_other_replacements() {
    let source = expression("Integral(x+y, Tuple(x))");
    let result = source.subs(&HashMap::from([
        (Symbol::new("x"), expression("2")),
        (Symbol::new("y"), expression("x")),
    ]));
    assert_eq!(result, expression("Integral(x+x_1, Tuple(x_1,2))"));
    assert_eq!(result.free_symbols(), vec![Symbol::new("x")]);
}

#[test]
fn binding_uses_symbol_identity_and_preserves_it_when_renaming() {
    let identity = SymbolIdentity::from_assumptions_digest([1; 32]);
    let bound = Symbol::with_identity("x", identity);
    let plain = Symbol::new("x");
    let parameter = Symbol::new("y");
    let source = Expr::Function(
        "Lambda".into(),
        vec![
            Expr::Sym(bound.clone()),
            Expr::Sym(bound.clone()) + Expr::Sym(plain.clone()) + Expr::Sym(parameter.clone()),
        ],
    );
    let result = source.subs(&HashMap::from([
        (plain, Expr::from_i64(2)),
        (parameter, Expr::Sym(bound.clone())),
    ]));
    assert_eq!(result.free_symbols(), vec![bound]);
    let Expr::Function(_, args) = result else {
        panic!("expected lambda")
    };
    let Expr::Sym(renamed) = &args[0] else {
        panic!("expected binder")
    };
    assert_eq!(renamed.name, "x_1");
    assert_eq!(renamed.identity, Some(identity));
}

#[test]
fn flat_limit_lists_do_not_add_recursive_expression_depth() {
    let variable = Symbol::new("x");
    let limit = expression("Tuple(x,1,2)");
    let mut args = vec![Expr::symbol("z")];
    args.extend(std::iter::repeat_n(limit, 16_384));
    let source = Expr::Function("Sum".into(), args);
    assert_eq!(source.nesting_depth(), 3);
    let result = source.subs(&HashMap::from([(Symbol::new("z"), Expr::from_i64(7))]));
    assert_eq!(result.nesting_depth(), 3);
    let Expr::Function(_, args) = result else {
        panic!("expected the original flat sum shape")
    };
    assert_eq!(args.len(), 16_385);
    assert_eq!(args[0], Expr::from_i64(7));
    for limit in &args[1..] {
        let Expr::Function(_, parts) = limit else {
            panic!("expected a limit")
        };
        assert_eq!(parts[0], Expr::Sym(variable.clone()));
    }
}

#[test]
fn all_repeated_enclosing_binders_avoid_a_captured_replacement() {
    let mut args = vec![Expr::symbol("z")];
    args.extend(std::iter::repeat_n(expression("Tuple(x,1,2)"), 4096));
    let source = Expr::Function("Sum".into(), args);
    let result = source.subs(&HashMap::from([(Symbol::new("z"), Expr::symbol("x"))]));
    assert_eq!(result.nesting_depth(), 3);
    assert_eq!(result.free_symbols(), vec![Symbol::new("x")]);
    let Expr::Function(_, args) = result else {
        panic!("expected a flat sum")
    };
    assert_eq!(args[0], Expr::symbol("x"));
    let mut variables = std::collections::HashSet::new();
    for limit in &args[1..] {
        let Expr::Function(_, parts) = limit else {
            panic!("expected a limit")
        };
        assert!(variables.insert(parts[0].clone()));
    }
    assert_eq!(variables.len(), 4096);
}

#[test]
fn compound_replacement_matches_real_subtrees_without_synthetic_prefixes() {
    for operator in ["Integral", "Sum", "Product"] {
        let prefix = expression(&format!("{operator}(x, Tuple(x,1,3))"));
        let flat = expression(&format!("{operator}(x, Tuple(x,1,3), Tuple(y,1,2))"));
        assert_eq!(flat.subs_expr(&prefix, &expression("z")), flat);

        let nested = Expr::Function(
            operator.into(),
            vec![prefix.clone(), expression("Tuple(y,1,2)")],
        );
        assert_eq!(
            nested.subs_expr(&prefix, &expression("z")),
            expression(&format!("{operator}(z, Tuple(y,1,2))")),
        );
    }
    let mut args = vec![Expr::symbol("x")];
    args.extend(std::iter::repeat_n(expression("Tuple(x,1,2)"), 16_384));
    let flat = Expr::Function("Integral".into(), args);
    assert_eq!(
        flat.subs_expr(&expression("Integral(x, Tuple(x,1,2))"), &expression("y")),
        flat,
    );
}

#[test]
fn indefinite_coordinates_are_real_substitution_sites_under_outer_scopes() {
    assert_eq!(
        replace("Integral(1, Tuple(x), Tuple(y,0,1))", "x", "y"),
        expression("Integral(1, Tuple(y), Tuple(y_1,0,1))"),
    );
    assert_eq!(
        replace("Integral(1, Tuple(x), Tuple(y,0,1))", "x", "y+1"),
        expression("Integral(1, Tuple(x,y+1), Tuple(y_1,0,1))"),
    );
    assert_eq!(
        replace("Sum(x, Tuple(x,1,z))", "z", "x"),
        expression("Sum(x, Tuple(x,1,x))"),
    );
}

#[test]
fn flat_binder_renames_avoid_simultaneous_map_keys_and_preserve_typed_identity() {
    assert_eq!(
        expression("Sum(x+z, Tuple(x,1,2))").subs(&HashMap::from([
            (Symbol::new("z"), expression("x")),
            (Symbol::new("x_1"), expression("7")),
        ])),
        expression("Sum(x+x_2, Tuple(x_2,1,2))"),
    );
    let identity = SymbolIdentity::from_assumptions_digest([2; 32]);
    let variable = Symbol::with_identity("x", identity);
    let source = Expr::Function(
        "Sum".into(),
        vec![
            Expr::Sym(variable.clone()) + Expr::symbol("z"),
            Expr::Function(
                "Tuple".into(),
                vec![
                    Expr::Sym(variable.clone()),
                    Expr::from_i64(1),
                    Expr::from_i64(2),
                ],
            ),
        ],
    );
    let result = source.subs(&HashMap::from([(
        Symbol::new("z"),
        Expr::Sym(variable.clone()),
    )]));
    assert_eq!(result.free_symbols(), vec![variable]);
    let Expr::Function(_, args) = result else {
        panic!("expected sum")
    };
    let Expr::Function(_, parts) = &args[1] else {
        panic!("expected limit")
    };
    let Expr::Sym(renamed) = &parts[0] else {
        panic!("expected coordinate")
    };
    assert_eq!(renamed.name, "x_1");
    assert_eq!(renamed.identity, Some(identity));
}

#[test]
fn compound_patterns_match_before_alpha_renaming_normalizes_a_raw_subtree() {
    let pattern = Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(0)]);
    let source = Expr::Function(
        "Lambda".into(),
        vec![
            Expr::symbol("x"),
            Expr::Add(vec![Expr::symbol("x"), pattern.clone()]),
        ],
    );
    assert_eq!(
        source.subs_expr(&pattern, &Expr::symbol("x")),
        expression("Lambda(x_1,x+x_1)"),
    );
    assert_eq!(
        pattern.subs_expr(&expression("cos(z)"), &Expr::symbol("x")),
        pattern,
    );
}

#[test]
fn compound_substitution_does_not_leak_holes_or_confuse_user_function_names() {
    let source = expression(
        "Lambda(x, x+cos(y)+__fsym_substitution_hole(__fsym_substitution_2,__fsym_substitution_1))",
    );
    assert_eq!(
        source.subs_expr(&expression("cos(y)"), &expression("x")),
        expression(
            "Lambda(x_1, x+x_1+__fsym_substitution_hole(__fsym_substitution_2,__fsym_substitution_1))",
        ),
    );
    // The inner Lambda is initially an opaque, malformed binding shape.
    // A temporary symbol must not turn its argument into a bound marker.
    let source = expression("Integral(Lambda(cos(y),cos(y)+x),Tuple(x,0,1))");
    assert_eq!(
        source.subs_expr(&expression("cos(y)"), &expression("x")),
        expression("Integral(Lambda(x,x+x_1),Tuple(x_1,0,1))"),
    );
}

#[test]
fn bounded_symbol_substitution_agrees_with_independent_lexical_evaluation() {
    let environment = HashMap::from([
        (Symbol::new("x"), 2),
        (Symbol::new("y"), 3),
        (Symbol::new("z"), 4),
        (Symbol::new("w"), 5),
    ]);
    let sources = [
        "Sum(x+z,Tuple(x,1,y))",
        "Sum(x+y+z,Tuple(x,1,y),Tuple(y,1,3))",
        "Sum(z,Tuple(x,1,y),Tuple(y,1,w))",
        "Sum(x+z,Tuple(x,1,x),Tuple(x,1,3))",
        "Sum(x+w,Tuple(x,1,z),Tuple(y,1,2))",
    ];
    for source in sources {
        let source = expression(source);
        for target in ["x", "y", "z", "w"] {
            for replacement in ["x", "y", "z", "3", "x+y"] {
                let replacement = expression(replacement);
                let mut reference_environment = environment.clone();
                reference_environment
                    .insert(Symbol::new(target), evaluate(&replacement, &environment));
                let result =
                    source.subs(&HashMap::from([(Symbol::new(target), replacement.clone())]));
                assert_eq!(
                    evaluate(&result, &environment),
                    evaluate(&source, &reference_environment),
                    "{source}: {target} -> {replacement}",
                );
            }
        }
        let simultaneous = HashMap::from([
            (Symbol::new("z"), expression("x")),
            (Symbol::new("w"), expression("y")),
        ]);
        let mut reference_environment = environment.clone();
        for (target, replacement) in &simultaneous {
            reference_environment.insert(target.clone(), evaluate(replacement, &environment));
        }
        assert_eq!(
            evaluate(&source.subs(&simultaneous), &environment),
            evaluate(&source, &reference_environment),
            "simultaneous substitutions in {source}",
        );
    }
}
