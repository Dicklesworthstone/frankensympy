//! Scoped variable binding, alpha-equivalence, and capture-avoiding substitution for WS04.
//!
//! Handles scoped symbols in calculus integrals, derivatives, and lambda binders,
//! guaranteeing that substitution never unintentionally captures free variables.

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Strongly typed representation of a binder construct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinderNode {
    /// Lambda abstraction over bound parameter.
    Lambda { param: Symbol, body: Box<Expr> },
    /// Integral over integration variable with optional limits.
    Integral {
        var: Symbol,
        body: Box<Expr>,
        limits: Option<(Box<Expr>, Box<Expr>)>,
    },
    /// Derivative with respect to a differentiation variable.
    Derivative { var: Symbol, body: Box<Expr> },
}

/// De Bruijn canonical indexed term for syntax-invariant alpha comparison.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeBruijnExpr {
    /// Free symbol (not bound in enclosing scopes).
    Free(Symbol),
    /// Bound variable represented by its de Bruijn index (0 = innermost enclosing binder).
    Bound(usize),
    /// Exact integer literal.
    Integer(fsym_core::BigInt),
    /// Exact rational literal.
    Rational(fsym_core::BigRational),
    /// Mathematical constant.
    Const(fsym_core::Constant),
    /// Addition of subterms.
    Add(Vec<DeBruijnExpr>),
    /// Multiplication of subterms.
    Mul(Vec<DeBruijnExpr>),
    /// Power of base and exponent.
    Pow(Box<DeBruijnExpr>, Box<DeBruijnExpr>),
    /// Arbitrary named function application.
    Function(String, Vec<DeBruijnExpr>),
    /// Scoped binder (e.g. Lambda) with body in which index 0 refers to this binder.
    Binder(String, Box<DeBruijnExpr>),
}

/// Converts an expression into canonical De Bruijn indexed form.
pub fn to_de_bruijn(expr: &Expr) -> DeBruijnExpr {
    let mut scope = Vec::new();
    expr_to_de_bruijn(expr, &mut scope)
}

fn expr_to_de_bruijn(expr: &Expr, scope: &mut Vec<Symbol>) -> DeBruijnExpr {
    match expr {
        Expr::Sym(s) => {
            if let Some(pos) = scope.iter().rev().position(|sym| sym == s) {
                DeBruijnExpr::Bound(pos)
            } else {
                DeBruijnExpr::Free(s.clone())
            }
        }
        Expr::Integer(n) => DeBruijnExpr::Integer(n.clone()),
        Expr::Rational(q) => DeBruijnExpr::Rational(q.clone()),
        Expr::Const(c) => DeBruijnExpr::Const(*c),
        Expr::Add(terms) => {
            DeBruijnExpr::Add(terms.iter().map(|t| expr_to_de_bruijn(t, scope)).collect())
        }
        Expr::Mul(terms) => {
            DeBruijnExpr::Mul(terms.iter().map(|t| expr_to_de_bruijn(t, scope)).collect())
        }
        Expr::Pow(b, e) => DeBruijnExpr::Pow(
            Box::new(expr_to_de_bruijn(b, scope)),
            Box::new(expr_to_de_bruijn(e, scope)),
        ),
        Expr::Function(name, args) => {
            if name == "Lambda"
                && args.len() == 2
                && let Expr::Sym(param) = &args[0]
            {
                scope.push(param.clone());
                let body_db = expr_to_de_bruijn(&args[1], scope);
                scope.pop();
                return DeBruijnExpr::Binder("Lambda".into(), Box::new(body_db));
            }
            DeBruijnExpr::Function(
                name.clone(),
                args.iter().map(|a| expr_to_de_bruijn(a, scope)).collect(),
            )
        }
    }
}

/// Identifies binder function semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinderKind {
    /// Lambda(var, body): binds `var` inside `body`.
    Lambda,
    /// Integral(body, var): binds `var` inside `body`.
    Integral,
    /// Derivative(body, var): binds `var` inside `body`.
    Derivative,
}

fn classify_binder(name: &str, args_len: usize) -> Option<BinderKind> {
    if name == "Lambda" && args_len == 2 {
        Some(BinderKind::Lambda)
    } else if name == "Integral" && args_len >= 2 {
        Some(BinderKind::Integral)
    } else if name == "Derivative" && args_len == 2 {
        Some(BinderKind::Derivative)
    } else {
        None
    }
}

/// Extracts the set of free (unbound) symbols appearing in an expression.
pub fn free_symbols(expr: &Expr) -> BTreeSet<Symbol> {
    let mut free = BTreeSet::new();
    let mut bound_scope = BTreeSet::new();
    collect_free_symbols(expr, &mut bound_scope, &mut free);
    free
}

fn collect_free_symbols(
    expr: &Expr,
    bound_scope: &mut BTreeSet<Symbol>,
    free: &mut BTreeSet<Symbol>,
) {
    match expr {
        Expr::Sym(s) => {
            if !bound_scope.contains(s) {
                free.insert(s.clone());
            }
        }
        Expr::Add(args) | Expr::Mul(args) => {
            for arg in args {
                collect_free_symbols(arg, bound_scope, free);
            }
        }
        Expr::Pow(b, e) => {
            collect_free_symbols(b, bound_scope, free);
            collect_free_symbols(e, bound_scope, free);
        }
        Expr::Function(name, args) => {
            if let Some(kind) = classify_binder(name, args.len()) {
                match kind {
                    BinderKind::Lambda => {
                        // Lambda(var, body)
                        if let Expr::Sym(var) = &args[0] {
                            let newly_bound = bound_scope.insert(var.clone());
                            collect_free_symbols(&args[1], bound_scope, free);
                            if newly_bound {
                                bound_scope.remove(var);
                            }
                            return;
                        }
                    }
                    BinderKind::Integral => {
                        // Integral(body, var, ...)
                        if let Expr::Sym(var) = &args[1] {
                            let newly_bound = bound_scope.insert(var.clone());
                            collect_free_symbols(&args[0], bound_scope, free);
                            if newly_bound {
                                bound_scope.remove(var);
                            }
                            for limit in args.iter().skip(2) {
                                collect_free_symbols(limit, bound_scope, free);
                            }
                            return;
                        }
                    }
                    BinderKind::Derivative => {
                        // Derivative(body, var)
                        if let Expr::Sym(var) = &args[1] {
                            let newly_bound = bound_scope.insert(var.clone());
                            collect_free_symbols(&args[0], bound_scope, free);
                            if newly_bound {
                                bound_scope.remove(var);
                            }
                            return;
                        }
                    }
                }
            }

            for arg in args {
                collect_free_symbols(arg, bound_scope, free);
            }
        }
        _ => {}
    }
}

/// Generates a deterministic fresh symbol not contained in the `avoid` set.
pub fn fresh_symbol(base: &str, avoid: &BTreeSet<Symbol>) -> Symbol {
    let base_sym = Symbol::new(base);
    if !avoid.contains(&base_sym) {
        return base_sym;
    }
    let mut index = 1usize;
    loop {
        let candidate = Symbol::new(format!("{base}_{index}"));
        if !avoid.contains(&candidate) {
            return candidate;
        }
        index += 1;
    }
}

/// Tests structural equivalence under alpha-renaming of bound variables.
pub fn alpha_equivalent(a: &Expr, b: &Expr) -> bool {
    let mut a_to_b = BTreeMap::new();
    let mut b_to_a = BTreeMap::new();
    alpha_equiv_helper(a, b, &mut a_to_b, &mut b_to_a)
}

fn alpha_equiv_helper(
    a: &Expr,
    b: &Expr,
    a_to_b: &mut BTreeMap<Symbol, Symbol>,
    b_to_a: &mut BTreeMap<Symbol, Symbol>,
) -> bool {
    match (a, b) {
        (Expr::Sym(s1), Expr::Sym(s2)) => {
            if let Some(mapped) = a_to_b.get(s1) {
                mapped == s2
            } else if b_to_a.contains_key(s2) {
                false
            } else {
                s1 == s2
            }
        }
        (Expr::Integer(n1), Expr::Integer(n2)) => n1 == n2,
        (Expr::Rational(q1), Expr::Rational(q2)) => q1 == q2,
        (Expr::Const(c1), Expr::Const(c2)) => c1 == c2,
        (Expr::Add(args1), Expr::Add(args2)) | (Expr::Mul(args1), Expr::Mul(args2)) => {
            args1.len() == args2.len()
                && args1
                    .iter()
                    .zip(args2.iter())
                    .all(|(e1, e2)| alpha_equiv_helper(e1, e2, a_to_b, b_to_a))
        }
        (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) => {
            alpha_equiv_helper(b1, b2, a_to_b, b_to_a) && alpha_equiv_helper(e1, e2, a_to_b, b_to_a)
        }
        (Expr::Function(n1, a1), Expr::Function(n2, a2)) => {
            if n1 != n2 || a1.len() != a2.len() {
                return false;
            }

            if let (Some(k1), Some(k2)) =
                (classify_binder(n1, a1.len()), classify_binder(n2, a2.len()))
                && k1 == k2
            {
                match k1 {
                    BinderKind::Lambda => {
                        if let (Expr::Sym(v1), Expr::Sym(v2)) = (&a1[0], &a2[0]) {
                            let old_ab = a_to_b.insert(v1.clone(), v2.clone());
                            let old_ba = b_to_a.insert(v2.clone(), v1.clone());
                            let body_eq = alpha_equiv_helper(&a1[1], &a2[1], a_to_b, b_to_a);
                            match old_ab {
                                Some(prev) => {
                                    a_to_b.insert(v1.clone(), prev);
                                }
                                None => {
                                    a_to_b.remove(v1);
                                }
                            }
                            match old_ba {
                                Some(prev) => {
                                    b_to_a.insert(v2.clone(), prev);
                                }
                                None => {
                                    b_to_a.remove(v2);
                                }
                            }
                            return body_eq;
                        }
                    }
                    BinderKind::Integral | BinderKind::Derivative => {
                        if let (Expr::Sym(v1), Expr::Sym(v2)) = (&a1[1], &a2[1]) {
                            let old_ab = a_to_b.insert(v1.clone(), v2.clone());
                            let old_ba = b_to_a.insert(v2.clone(), v1.clone());
                            let body_eq = alpha_equiv_helper(&a1[0], &a2[0], a_to_b, b_to_a);
                            match old_ab {
                                Some(prev) => {
                                    a_to_b.insert(v1.clone(), prev);
                                }
                                None => {
                                    a_to_b.remove(v1);
                                }
                            }
                            match old_ba {
                                Some(prev) => {
                                    b_to_a.insert(v2.clone(), prev);
                                }
                                None => {
                                    b_to_a.remove(v2);
                                }
                            }
                            if !body_eq {
                                return false;
                            }
                            for (l1, l2) in a1.iter().skip(2).zip(a2.iter().skip(2)) {
                                if !alpha_equiv_helper(l1, l2, a_to_b, b_to_a) {
                                    return false;
                                }
                            }
                            return true;
                        }
                    }
                }
            }

            a1.iter()
                .zip(a2.iter())
                .all(|(e1, e2)| alpha_equiv_helper(e1, e2, a_to_b, b_to_a))
        }
        _ => false,
    }
}

/// Performs capture-avoiding substitution: replaces occurrences of `target` with `replacement` in `expr`.
pub fn capture_avoiding_subs(expr: &Expr, target: &Symbol, replacement: &Expr) -> Expr {
    let repl_free = free_symbols(replacement);
    subs_internal(expr, target, replacement, &repl_free)
}

fn subs_internal(
    expr: &Expr,
    target: &Symbol,
    replacement: &Expr,
    repl_free: &BTreeSet<Symbol>,
) -> Expr {
    match expr {
        Expr::Sym(s) => {
            if s == target {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        Expr::Add(args) => Expr::Add(
            args.iter()
                .map(|a| subs_internal(a, target, replacement, repl_free))
                .collect(),
        ),
        Expr::Mul(args) => Expr::Mul(
            args.iter()
                .map(|a| subs_internal(a, target, replacement, repl_free))
                .collect(),
        ),
        Expr::Pow(b, e) => Expr::Pow(
            Arc::new(subs_internal(b, target, replacement, repl_free)),
            Arc::new(subs_internal(e, target, replacement, repl_free)),
        ),
        Expr::Function(name, args) => {
            if let Some(kind) = classify_binder(name, args.len()) {
                match kind {
                    BinderKind::Lambda => {
                        if let Expr::Sym(bound_var) = &args[0] {
                            if bound_var == target {
                                return expr.clone(); // Shadowed
                            }
                            if repl_free.contains(bound_var) {
                                // Variable capture would occur: rename bound variable
                                let mut avoid = free_symbols(&args[1]);
                                avoid.extend(repl_free.iter().cloned());
                                avoid.insert(target.clone());
                                let fresh = fresh_symbol(&bound_var.name, &avoid);
                                let renamed_body = subs_internal(
                                    &args[1],
                                    bound_var,
                                    &Expr::Sym(fresh.clone()),
                                    &BTreeSet::new(),
                                );
                                let new_body =
                                    subs_internal(&renamed_body, target, replacement, repl_free);
                                return Expr::Function(
                                    name.clone(),
                                    vec![Expr::Sym(fresh), new_body],
                                );
                            } else {
                                let new_body =
                                    subs_internal(&args[1], target, replacement, repl_free);
                                return Expr::Function(
                                    name.clone(),
                                    vec![args[0].clone(), new_body],
                                );
                            }
                        }
                    }
                    BinderKind::Integral | BinderKind::Derivative => {
                        if let Expr::Sym(bound_var) = &args[1] {
                            if bound_var == target {
                                return expr.clone(); // Shadowed
                            }
                            if repl_free.contains(bound_var) {
                                let mut avoid = free_symbols(&args[0]);
                                avoid.extend(repl_free.iter().cloned());
                                avoid.insert(target.clone());
                                let fresh = fresh_symbol(&bound_var.name, &avoid);
                                let renamed_body = subs_internal(
                                    &args[0],
                                    bound_var,
                                    &Expr::Sym(fresh.clone()),
                                    &BTreeSet::new(),
                                );
                                let new_body =
                                    subs_internal(&renamed_body, target, replacement, repl_free);
                                let mut new_args = vec![new_body, Expr::Sym(fresh)];
                                for limit in args.iter().skip(2) {
                                    new_args.push(subs_internal(
                                        limit,
                                        target,
                                        replacement,
                                        repl_free,
                                    ));
                                }
                                return Expr::Function(name.clone(), new_args);
                            } else {
                                let new_body =
                                    subs_internal(&args[0], target, replacement, repl_free);
                                let mut new_args = vec![new_body, args[1].clone()];
                                for limit in args.iter().skip(2) {
                                    new_args.push(subs_internal(
                                        limit,
                                        target,
                                        replacement,
                                        repl_free,
                                    ));
                                }
                                return Expr::Function(name.clone(), new_args);
                            }
                        }
                    }
                }
            }

            Expr::Function(
                name.clone(),
                args.iter()
                    .map(|a| subs_internal(a, target, replacement, repl_free))
                    .collect(),
            )
        }
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_symbols_extraction_with_binders() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        // Lambda(x, x + y): x is bound, y is free
        let lambda_expr = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("x"),
                Expr::Add(vec![Expr::symbol("x"), Expr::symbol("y")]),
            ],
        );

        let free = free_symbols(&lambda_expr);
        assert_eq!(free.len(), 1);
        assert!(!free.contains(&x));
        assert!(free.contains(&y));
    }

    #[test]
    fn alpha_equivalence_under_binder_renaming() {
        // Lambda(x, x + 1) vs Lambda(y, y + 1)
        let l1 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("x"),
                Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]),
            ],
        );
        let l2 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("y"),
                Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(1)]),
            ],
        );
        let l3 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("y"),
                Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(2)]),
            ],
        );

        assert!(alpha_equivalent(&l1, &l2));
        assert!(!alpha_equivalent(&l1, &l3));
    }

    #[test]
    fn capture_avoiding_substitution_renames_bound_var() {
        // In Lambda(y, x + y), substituting x -> y + 1 must avoid capturing y!
        // Should produce Lambda(y_1, (y + 1) + y_1)
        let lambda_expr = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("y"),
                Expr::Add(vec![Expr::symbol("x"), Expr::symbol("y")]),
            ],
        );

        let replacement = Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(1)]);
        let substituted = capture_avoiding_subs(&lambda_expr, &Symbol::new("x"), &replacement);

        // Verify free symbols of result: y must be free, no x
        let free = free_symbols(&substituted);
        assert!(free.contains(&Symbol::new("y")));
        assert!(!free.contains(&Symbol::new("x")));

        // Verify alpha-equivalence with Lambda(z, (y + 1) + z)
        let expected = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("z"),
                Expr::Add(vec![
                    Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(1)]),
                    Expr::symbol("z"),
                ]),
            ],
        );
        assert!(alpha_equivalent(&substituted, &expected));
    }

    #[test]
    fn de_bruijn_conversion_guarantees_alpha_identity() {
        let l1 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("x"),
                Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]),
            ],
        );
        let l2 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("y"),
                Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(1)]),
            ],
        );

        // De Bruijn trees are strictly identical for alpha-equivalent terms
        let db1 = to_de_bruijn(&l1);
        let db2 = to_de_bruijn(&l2);
        assert_eq!(db1, db2);
    }

    #[test]
    fn binder_node_representation() {
        let b = BinderNode::Lambda {
            param: Symbol::new("t"),
            body: Box::new(Expr::Mul(vec![Expr::symbol("t"), Expr::from_i64(2)])),
        };
        match b {
            BinderNode::Lambda { param, body } => {
                assert_eq!(param.name, "t");
                assert_eq!(*body, Expr::Mul(vec![Expr::symbol("t"), Expr::from_i64(2)]));
            }
            _ => panic!("Expected Lambda"),
        }
    }

    #[test]
    fn substitution_free_symbols_soundness_property() {
        let expr = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("y"),
                Expr::Add(vec![
                    Expr::symbol("x"),
                    Expr::Mul(vec![Expr::symbol("y"), Expr::symbol("z")]),
                ]),
            ],
        );
        let repl = Expr::Add(vec![Expr::symbol("a"), Expr::symbol("b")]);
        let substituted = capture_avoiding_subs(&expr, &Symbol::new("x"), &repl);

        let free_sub = free_symbols(&substituted);
        let free_orig = free_symbols(&expr);
        let free_repl = free_symbols(&repl);

        // free(e[x/r]) subseteq (free(e) \ {x}) U free(r)
        let mut expected_superset = free_orig;
        expected_superset.remove(&Symbol::new("x"));
        expected_superset.extend(free_repl);

        for sym in &free_sub {
            assert!(
                expected_superset.contains(sym),
                "Symbol {:?} violated free symbol containment",
                sym
            );
        }
    }
}
