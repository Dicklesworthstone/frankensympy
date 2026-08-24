//! Scoped variable binding, alpha-equivalence, and capture-avoiding substitution for WS04.
//!
//! Handles scoped symbols in calculus integrals, sums, products, and function binders,
//! guaranteeing that substitution never unintentionally captures free variables.

use fsym_core::{Expr, Symbol};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Extracts the set of free (unbound) symbols appearing in an expression.
pub fn free_symbols(expr: &Expr) -> BTreeSet<Symbol> {
    let mut free = BTreeSet::new();
    collect_free_symbols(expr, &mut BTreeSet::new(), &mut free);
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
        Expr::Function(_name, args) => {
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
            n1 == n2
                && a1.len() == a2.len()
                && a1
                    .iter()
                    .zip(a2.iter())
                    .all(|(e1, e2)| alpha_equiv_helper(e1, e2, a_to_b, b_to_a))
        }
        _ => false,
    }
}

/// Performs capture-avoiding substitution: replaces occurrences of `target` with `replacement` in `expr`.
pub fn capture_avoiding_subs(expr: &Expr, target: &Symbol, replacement: &Expr) -> Expr {
    subs_internal(expr, target, replacement)
}

fn subs_internal(expr: &Expr, target: &Symbol, replacement: &Expr) -> Expr {
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
                .map(|a| subs_internal(a, target, replacement))
                .collect(),
        ),
        Expr::Mul(args) => Expr::Mul(
            args.iter()
                .map(|a| subs_internal(a, target, replacement))
                .collect(),
        ),
        Expr::Pow(b, e) => Expr::Pow(
            Arc::new(subs_internal(b, target, replacement)),
            Arc::new(subs_internal(e, target, replacement)),
        ),
        Expr::Function(name, args) => Expr::Function(
            name.clone(),
            args.iter()
                .map(|a| subs_internal(a, target, replacement))
                .collect(),
        ),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_symbols_extraction() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let expr = Expr::Add(vec![
            Expr::symbol("x"),
            Expr::Mul(vec![Expr::from_i64(2), Expr::symbol("y")]),
        ]);

        let free = free_symbols(&expr);
        assert_eq!(free.len(), 2);
        assert!(free.contains(&x));
        assert!(free.contains(&y));
    }

    #[test]
    fn fresh_symbol_avoids_collisions() {
        let mut avoid = BTreeSet::new();
        avoid.insert(Symbol::new("x"));
        avoid.insert(Symbol::new("x_1"));

        let fresh = fresh_symbol("x", &avoid);
        assert_eq!(fresh.name, "x_2");
    }

    #[test]
    fn substitution_replaces_target() {
        let x = Symbol::new("x");
        let expr = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]);
        let target = x;
        let replacement = Expr::symbol("y");

        let result = capture_avoiding_subs(&expr, &target, &replacement);
        assert_eq!(
            result,
            Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(1)])
        );
    }

    #[test]
    fn alpha_equivalence_properties() {
        let e1 = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(10)]);
        let e2 = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(10)]);
        let e3 = Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(10)]);

        assert!(alpha_equivalent(&e1, &e2));
        assert!(!alpha_equivalent(&e1, &e3));
    }

    #[test]
    fn nested_power_and_function_substitution() {
        let x = Symbol::new("x");
        let expr = Expr::Pow(
            std::sync::Arc::new(Expr::Function("sin".into(), vec![Expr::symbol("x")])),
            std::sync::Arc::new(Expr::from_i64(2)),
        );

        let replaced = capture_avoiding_subs(&expr, &x, &Expr::symbol("z"));
        assert_eq!(
            replaced,
            Expr::Pow(
                std::sync::Arc::new(Expr::Function("sin".into(), vec![Expr::symbol("z")])),
                std::sync::Arc::new(Expr::from_i64(2)),
            )
        );
    }
}
