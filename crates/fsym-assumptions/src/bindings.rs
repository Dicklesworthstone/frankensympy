//! Scoped variable binding, alpha-equivalence, and capture-avoiding substitution for WS04.
//!
//! Handles scoped symbols in calculus integrals, derivatives, and lambda binders,
//! guaranteeing that substitution never unintentionally captures free variables.

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use std::collections::BTreeSet;
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

impl BinderNode {
    /// Creates a lambda binder.
    pub fn lambda(param: impl Into<Symbol>, body: Expr) -> Self {
        Self::Lambda {
            param: param.into(),
            body: Box::new(body),
        }
    }

    /// Creates an indefinite integral binder.
    pub fn integral(var: impl Into<Symbol>, body: Expr) -> Self {
        Self::Integral {
            var: var.into(),
            body: Box::new(body),
            limits: None,
        }
    }

    /// Creates a definite integral binder with lower and upper limits.
    pub fn definite_integral(var: impl Into<Symbol>, body: Expr, lower: Expr, upper: Expr) -> Self {
        Self::Integral {
            var: var.into(),
            body: Box::new(body),
            limits: Some((Box::new(lower), Box::new(upper))),
        }
    }

    /// Creates a derivative binder.
    pub fn derivative(var: impl Into<Symbol>, body: Expr) -> Self {
        Self::Derivative {
            var: var.into(),
            body: Box::new(body),
        }
    }

    /// Converts this binder node into an expression tree.
    pub fn to_expr(&self) -> Expr {
        match self {
            Self::Lambda { param, body } => Expr::Function(
                "Lambda".to_string(),
                vec![Expr::Sym(param.clone()), (**body).clone()],
            ),
            Self::Integral { var, body, limits } => {
                if let Some((lower, upper)) = limits {
                    Expr::Function(
                        "Integral".to_string(),
                        vec![
                            (**body).clone(),
                            Expr::Sym(var.clone()),
                            (**lower).clone(),
                            (**upper).clone(),
                        ],
                    )
                } else {
                    Expr::Function(
                        "Integral".to_string(),
                        vec![(**body).clone(), Expr::Sym(var.clone())],
                    )
                }
            }
            Self::Derivative { var, body } => Expr::Function(
                "Derivative".to_string(),
                vec![(**body).clone(), Expr::Sym(var.clone())],
            ),
        }
    }

    /// Attempts to parse a binder node from an expression tree.
    pub fn try_from_expr(expr: &Expr) -> Option<Self> {
        if let Expr::Function(name, args) = expr {
            match name.as_str() {
                "Lambda" if args.len() == 2 => {
                    if let Expr::Sym(param) = &args[0] {
                        return Some(Self::Lambda {
                            param: param.clone(),
                            body: Box::new(args[1].clone()),
                        });
                    }
                }
                "Integral" => {
                    if args.len() == 2
                        && let Expr::Sym(var) = &args[1]
                    {
                        return Some(Self::Integral {
                            var: var.clone(),
                            body: Box::new(args[0].clone()),
                            limits: None,
                        });
                    } else if args.len() == 4
                        && let Expr::Sym(var) = &args[1]
                    {
                        return Some(Self::Integral {
                            var: var.clone(),
                            body: Box::new(args[0].clone()),
                            limits: Some((Box::new(args[2].clone()), Box::new(args[3].clone()))),
                        });
                    }
                }
                "Derivative" if args.len() == 2 => {
                    if let Expr::Sym(var) = &args[1] {
                        return Some(Self::Derivative {
                            var: var.clone(),
                            body: Box::new(args[0].clone()),
                        });
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Returns the bound variable symbol.
    pub fn bound_variable(&self) -> &Symbol {
        match self {
            Self::Lambda { param, .. } => param,
            Self::Integral { var, .. } | Self::Derivative { var, .. } => var,
        }
    }

    /// Returns the body expression.
    pub fn body(&self) -> &Expr {
        match self {
            Self::Lambda { body, .. }
            | Self::Integral { body, .. }
            | Self::Derivative { body, .. } => body,
        }
    }

    /// Returns the free symbols of this binder.
    pub fn free_symbols(&self) -> BTreeSet<Symbol> {
        let mut free = free_symbols(self.body());
        free.remove(self.bound_variable());
        if let Self::Integral {
            limits: Some((lower, upper)),
            ..
        } = self
        {
            free.extend(free_symbols(lower));
            free.extend(free_symbols(upper));
        }
        free
    }

    /// Checks alpha equivalence against another binder node.
    pub fn is_alpha_equivalent(&self, other: &Self) -> bool {
        alpha_equivalent(&self.to_expr(), &other.to_expr())
    }
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
            if let Some(binder) = BinderNode::try_from_expr(expr) {
                match binder {
                    BinderNode::Lambda { param, body } => {
                        scope.push(param);
                        let body_db = expr_to_de_bruijn(&body, scope);
                        scope.pop();
                        return DeBruijnExpr::Binder("Lambda".into(), Box::new(body_db));
                    }
                    BinderNode::Integral { var, body, limits } => {
                        scope.push(var);
                        let body_db = expr_to_de_bruijn(&body, scope);
                        scope.pop();
                        if let Some((lower, upper)) = limits {
                            let lower_db = expr_to_de_bruijn(&lower, scope);
                            let upper_db = expr_to_de_bruijn(&upper, scope);
                            return DeBruijnExpr::Function(
                                "Integral".into(),
                                vec![body_db, lower_db, upper_db],
                            );
                        } else {
                            return DeBruijnExpr::Binder("Integral".into(), Box::new(body_db));
                        }
                    }
                    BinderNode::Derivative { var, body } => {
                        scope.push(var);
                        let body_db = expr_to_de_bruijn(&body, scope);
                        scope.pop();
                        return DeBruijnExpr::Binder("Derivative".into(), Box::new(body_db));
                    }
                }
            }
            DeBruijnExpr::Function(
                name.clone(),
                args.iter().map(|a| expr_to_de_bruijn(a, scope)).collect(),
            )
        }
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
        Expr::Function(_, args) => {
            if let Some(binder) = BinderNode::try_from_expr(expr) {
                let var = binder.bound_variable();
                let newly_bound = bound_scope.insert(var.clone());
                collect_free_symbols(binder.body(), bound_scope, free);
                if newly_bound {
                    bound_scope.remove(var);
                }
                if let BinderNode::Integral {
                    limits: Some((lower, upper)),
                    ..
                } = binder
                {
                    collect_free_symbols(&lower, bound_scope, free);
                    collect_free_symbols(&upper, bound_scope, free);
                }
                return;
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
    to_de_bruijn(a) == to_de_bruijn(b)
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
            if let Some(binder) = BinderNode::try_from_expr(expr) {
                let bound_var = binder.bound_variable();
                if bound_var == target {
                    return expr.clone(); // Shadowed
                }
                let body = binder.body();
                let (new_bound_var, new_body) = if repl_free.contains(bound_var) {
                    let mut avoid = free_symbols(body);
                    avoid.extend(repl_free.iter().cloned());
                    avoid.insert(target.clone());
                    let fresh = fresh_symbol(&bound_var.name, &avoid);
                    let renamed_body =
                        subs_internal(body, bound_var, &Expr::Sym(fresh.clone()), &BTreeSet::new());
                    let new_body = subs_internal(&renamed_body, target, replacement, repl_free);
                    (fresh, new_body)
                } else {
                    let new_body = subs_internal(body, target, replacement, repl_free);
                    (bound_var.clone(), new_body)
                };

                match binder {
                    BinderNode::Lambda { .. } => {
                        return BinderNode::Lambda {
                            param: new_bound_var,
                            body: Box::new(new_body),
                        }
                        .to_expr();
                    }
                    BinderNode::Derivative { .. } => {
                        return BinderNode::Derivative {
                            var: new_bound_var,
                            body: Box::new(new_body),
                        }
                        .to_expr();
                    }
                    BinderNode::Integral { limits, .. } => {
                        let new_limits = limits.map(|(lower, upper)| {
                            (
                                Box::new(subs_internal(&lower, target, replacement, repl_free)),
                                Box::new(subs_internal(&upper, target, replacement, repl_free)),
                            )
                        });
                        return BinderNode::Integral {
                            var: new_bound_var,
                            body: Box::new(new_body),
                            limits: new_limits,
                        }
                        .to_expr();
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
            _ => unreachable!("Expected Lambda variant"),
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

    #[test]
    fn nested_binder_multi_level_capture_avoidance() {
        // Lambda(y, Lambda(z, x + y + z))
        // Substitute x -> y + z
        let inner = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("z"),
                Expr::Add(vec![
                    Expr::symbol("x"),
                    Expr::symbol("y"),
                    Expr::symbol("z"),
                ]),
            ],
        );
        let outer = Expr::Function("Lambda".into(), vec![Expr::symbol("y"), inner]);
        let repl = Expr::Add(vec![Expr::symbol("y"), Expr::symbol("z")]);

        let substituted = capture_avoiding_subs(&outer, &Symbol::new("x"), &repl);
        let free = free_symbols(&substituted);

        // y and z in replacement must remain free outside the binder
        assert!(free.contains(&Symbol::new("y")));
        assert!(free.contains(&Symbol::new("z")));
        assert!(!free.contains(&Symbol::new("x")));

        // Must be alpha-equivalent to Lambda(y_fresh, Lambda(z_fresh, (y + z) + y_fresh + z_fresh))
        let expected = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("y_1"),
                Expr::Function(
                    "Lambda".into(),
                    vec![
                        Expr::symbol("z_1"),
                        Expr::Add(vec![
                            Expr::Add(vec![Expr::symbol("y"), Expr::symbol("z")]),
                            Expr::symbol("y_1"),
                            Expr::symbol("z_1"),
                        ]),
                    ],
                ),
            ],
        );
        assert!(alpha_equivalent(&substituted, &expected));
    }

    #[test]
    fn integral_and_derivative_binder_capture_avoidance() {
        // Integral(x * y, x, a, b): x is bound, y, a, b are free
        let integral_expr = Expr::Function(
            "Integral".into(),
            vec![
                Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
                Expr::symbol("x"),
                Expr::symbol("a"),
                Expr::symbol("b"),
            ],
        );
        let free_int = free_symbols(&integral_expr);
        assert!(!free_int.contains(&Symbol::new("x")));
        assert!(free_int.contains(&Symbol::new("y")));
        assert!(free_int.contains(&Symbol::new("a")));
        assert!(free_int.contains(&Symbol::new("b")));

        // Substitute y -> x + 1 into Integral(x * y, x, a, b)
        // Must rename bound x to avoid capturing the x in (x + 1)
        let repl = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]);
        let substituted_int = capture_avoiding_subs(&integral_expr, &Symbol::new("y"), &repl);

        let free_sub_int = free_symbols(&substituted_int);
        assert!(free_sub_int.contains(&Symbol::new("x")));
        assert!(!free_sub_int.contains(&Symbol::new("y")));

        // Derivative(x * y, x): x is bound, y is free
        let deriv_expr = Expr::Function(
            "Derivative".into(),
            vec![
                Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
                Expr::symbol("x"),
            ],
        );
        let free_deriv = free_symbols(&deriv_expr);
        assert!(!free_deriv.contains(&Symbol::new("x")));
        assert!(free_deriv.contains(&Symbol::new("y")));

        // Substitute y -> x into Derivative(x * y, x)
        let substituted_deriv =
            capture_avoiding_subs(&deriv_expr, &Symbol::new("y"), &Expr::symbol("x"));
        let free_sub_deriv = free_symbols(&substituted_deriv);
        assert!(free_sub_deriv.contains(&Symbol::new("x")));
        assert!(!free_sub_deriv.contains(&Symbol::new("y")));
    }

    #[test]
    fn substitution_preserves_domain_typing_invariants() {
        use crate::domain::Domain;

        // In an expression e in ZZ[x, y], substituting x -> 3 (in ZZ) yields an expression in ZZ[y]
        let e = Expr::Add(vec![
            Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("x")]),
            Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
            Expr::from_i64(1),
        ]);
        let d_orig = Domain::of_expr(&e);
        assert!(matches!(d_orig, Domain::PolyRing { .. }));

        let substituted = capture_avoiding_subs(&e, &Symbol::new("x"), &Expr::from_i64(3));
        let d_sub = Domain::of_expr(&substituted);

        // Substituted domain ZZ[y] must coerce into original ZZ[x, y]
        assert!(d_sub.can_coerce_to(&d_orig));

        // Substituting x -> 1/2 (in QQ) yields an expression in QQ[y], which coerces into fraction field
        let q_sub = capture_avoiding_subs(
            &e,
            &Symbol::new("x"),
            &Expr::Rational(fsym_core::BigRational::new(
                fsym_core::BigInt::from(1),
                fsym_core::BigInt::from(2),
            )),
        );
        let d_q = Domain::of_expr(&q_sub);
        assert!(d_q.can_coerce_to(&Domain::FractionField {
            base: Box::new(Domain::ZZ),
            generators: vec![Symbol::new("x"), Symbol::new("y")],
        }));
    }

    #[test]
    fn alpha_equivalence_properties() {
        let e1 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("u"),
                Expr::Mul(vec![Expr::symbol("u"), Expr::symbol("u")]),
            ],
        );
        let e2 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("v"),
                Expr::Mul(vec![Expr::symbol("v"), Expr::symbol("v")]),
            ],
        );
        let e3 = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("w"),
                Expr::Mul(vec![Expr::symbol("w"), Expr::symbol("w")]),
            ],
        );

        // Reflexivity
        assert!(alpha_equivalent(&e1, &e1));
        // Symmetry
        assert!(alpha_equivalent(&e1, &e2));
        assert!(alpha_equivalent(&e2, &e1));
        // Transitivity
        assert!(alpha_equivalent(&e1, &e2) && alpha_equivalent(&e2, &e3));
        assert!(alpha_equivalent(&e1, &e3));
    }

    #[test]
    fn binder_node_conversions_and_alpha_equivalence() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let z = Symbol::new("z");

        // Lambda binder
        let lambda1 = BinderNode::lambda(
            x.clone(),
            Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
        );
        let expr1 = lambda1.to_expr();
        let parsed1 = BinderNode::try_from_expr(&expr1).expect("valid lambda binder");
        assert_eq!(parsed1, lambda1);
        assert_eq!(parsed1.bound_variable(), &x);
        let free1 = parsed1.free_symbols();
        assert!(free1.contains(&y));
        assert!(!free1.contains(&x));

        let lambda2 = BinderNode::lambda(
            z.clone(),
            Expr::Mul(vec![Expr::Sym(z.clone()), Expr::Sym(y.clone())]),
        );
        assert!(lambda1.is_alpha_equivalent(&lambda2));

        // Definite integral binder
        let int1 = BinderNode::definite_integral(
            x.clone(),
            Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
            Expr::from_i64(0),
            Expr::Sym(z.clone()),
        );
        let int_expr = int1.to_expr();
        let parsed_int = BinderNode::try_from_expr(&int_expr).expect("valid definite integral");
        assert_eq!(parsed_int, int1);
        let free_int = parsed_int.free_symbols();
        assert!(free_int.contains(&y));
        assert!(free_int.contains(&z));
        assert!(!free_int.contains(&x));

        // Derivative binder
        let deriv1 = BinderNode::derivative(
            x.clone(),
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
        );
        let deriv_expr = deriv1.to_expr();
        let parsed_deriv = BinderNode::try_from_expr(&deriv_expr).expect("valid derivative");
        assert_eq!(parsed_deriv, deriv1);
        assert_eq!(parsed_deriv.free_symbols().len(), 0);
    }

    #[test]
    fn alpha_equivalent_rejects_inner_binder_shadowing_outer_free() {
        // Audit counterexample: `Lambda(x, Lambda(y, x))` returns the outer x
        // (depends on the captured free variable). `Lambda(a, Lambda(a, a))`
        // shadows its parameter and returns the constant a. They are NOT
        // alpha-equivalent because shadowing the captured symbol changes
        // meaning, but `alpha_equivalent` currently returns `true` because
        // `alpha_equiv_helper` blindly overwrites the b_to_a entry when
        // entering the inner Lambda scope (line 393) instead of detecting
        // that `a` was already claimed by the outer `x -> a` mapping.
        let outer_dependent = BinderNode::lambda(
            Symbol::new("x"),
            Expr::Function("Lambda".into(), vec![Expr::symbol("y"), Expr::symbol("x")]),
        );
        let inner_shadow = BinderNode::lambda(
            Symbol::new("a"),
            Expr::Function("Lambda".into(), vec![Expr::symbol("a"), Expr::symbol("a")]),
        );
        assert!(
            !outer_dependent.is_alpha_equivalent(&inner_shadow),
            "shadowing inner binder that captures an outer free symbol must \
             not be alpha-equivalent to the non-shadowing version"
        );

        // Regression: a true alpha renaming must still be reported equivalent.
        let renamed = BinderNode::lambda(
            Symbol::new("b"),
            Expr::Function("Lambda".into(), vec![Expr::symbol("a"), Expr::symbol("b")]),
        );
        assert!(outer_dependent.is_alpha_equivalent(&renamed));
    }
}
