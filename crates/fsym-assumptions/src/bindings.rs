//! Scoped variable binding, alpha-equivalence, and capture-avoiding substitution for WS04.
//!
//! Handles scoped symbols in definite integrals and lambda binders, plus the
//! related (but non-alpha-bound) variables of indefinite integrals and
//! derivatives. Substitution never silently captures a free variable.

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use std::collections::BTreeSet;
use std::sync::Arc;
use thiserror::Error;

/// Refusal emitted when a binding operation cannot inspect the complete expression.
///
/// Binding operations must never return a partial semantic answer: an incomplete
/// free-symbol set can make substitution capture a variable, and a truncated
/// De Bruijn term can make distinct expressions compare as alpha-equivalent.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BindingError {
    /// The expression exceeds the maximum supported binding traversal depth.
    #[error("binding traversal exceeds the maximum depth of {limit}")]
    DepthExceeded { limit: usize },
    /// Capture avoidance would require renaming an operator variable whose name
    /// is semantically observable rather than alpha-bound.
    #[error(
        "capture-safe substitution under {operator} requires a non-alpha-equivalent variable rename"
    )]
    NonAlphaRenamingRequired { operator: &'static str },
    /// An operator variable can only be replaced directly by another symbol.
    /// More general evaluation-point substitutions require an explicit node
    /// that the current expression IR does not provide.
    #[error("{operator} variable substitution requires a symbol replacement")]
    UnsupportedOperatorVariableReplacement { operator: &'static str },
}

/// Strongly typed representation of a scoped or operator-variable construct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinderNode {
    /// Lambda abstraction over bound parameter.
    Lambda { param: Symbol, body: Box<Expr> },
    /// Integral over an integration variable with optional limits.
    ///
    /// The variable is alpha-bound in the body only for a definite integral.
    /// In an indefinite integral it remains part of the observable result.
    Integral {
        var: Symbol,
        body: Box<Expr>,
        limits: Option<(Box<Expr>, Box<Expr>)>,
    },
    /// Derivative with respect to a differentiation variable.
    ///
    /// The variable controls differentiation but is not alpha-bound: renaming
    /// it changes the expression's observable free-variable identity.
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

    /// Creates an indefinite integral node.
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

    /// Creates a derivative node.
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
                "Lambda" => {
                    return try_parse_lambda(args);
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

    /// Returns the construct's declared variable symbol.
    ///
    /// This variable is alpha-bound only for a Lambda or definite integral.
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
    pub fn free_symbols(&self) -> Result<BTreeSet<Symbol>, BindingError> {
        match self {
            Self::Lambda { param, body } => {
                let mut free = free_symbols(body)?;
                free.remove(param);
                Ok(free)
            }
            Self::Integral {
                var,
                body,
                limits: Some((lower, upper)),
            } => {
                let mut free = free_symbols(body)?;
                free.remove(var);
                free.extend(free_symbols(lower)?);
                free.extend(free_symbols(upper)?);
                Ok(free)
            }
            Self::Integral {
                var,
                body,
                limits: None,
            } => {
                let mut free = free_symbols(body)?;
                free.insert(var.clone());
                Ok(free)
            }
            Self::Derivative { body, .. } => free_symbols(body),
        }
    }

    /// Checks alpha equivalence against another binder node.
    pub fn is_alpha_equivalent(&self, other: &Self) -> Result<bool, BindingError> {
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
    /// Scoped binder whose additional arguments are evaluated outside its scope.
    ///
    /// Definite-integral limits use this form: index 0 is bound only in `body`,
    /// while `arguments` retain the surrounding scope. Keeping a distinct node
    /// prevents a genuine binder from colliding with an opaque function that
    /// happens to use the same printed name and arity.
    BinderWithArguments {
        name: String,
        body: Box<DeBruijnExpr>,
        arguments: Vec<DeBruijnExpr>,
    },
}

/// Checks that one flat parameter list contains distinct symbols only.
///
/// Repeating a name inside one signature is malformed and must remain an
/// opaque function. Shadowing in a separately nested Lambda remains valid.
fn has_distinct_symbol_parameters(parameters: &[Expr]) -> bool {
    let mut seen = BTreeSet::new();
    parameters.iter().all(|parameter| match parameter {
        Expr::Sym(symbol) => seen.insert(symbol),
        _ => false,
    })
}

/// Parses single-parameter, multi-parameter Tuple, and multi-argument Lambda forms.
fn try_parse_lambda(args: &[Expr]) -> Option<BinderNode> {
    if args.len() == 2 {
        match &args[0] {
            Expr::Sym(param) => Some(BinderNode::Lambda {
                param: param.clone(),
                body: Box::new(args[1].clone()),
            }),
            Expr::Function(tname, targs) if tname == "Tuple" => {
                if targs.is_empty() {
                    None
                } else if targs.len() == 1 {
                    if let Expr::Sym(param) = &targs[0] {
                        Some(BinderNode::Lambda {
                            param: param.clone(),
                            body: Box::new(args[1].clone()),
                        })
                    } else {
                        None
                    }
                } else if has_distinct_symbol_parameters(targs) {
                    if let Expr::Sym(first) = &targs[0] {
                        let rest_tuple = if targs.len() == 2 {
                            targs[1].clone()
                        } else {
                            Expr::Function("Tuple".to_string(), targs[1..].to_vec())
                        };
                        let inner_lambda =
                            Expr::Function("Lambda".to_string(), vec![rest_tuple, args[1].clone()]);
                        Some(BinderNode::Lambda {
                            param: first.clone(),
                            body: Box::new(inner_lambda),
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    } else if args.len() > 2 {
        if has_distinct_symbol_parameters(&args[..args.len() - 1]) {
            if let Expr::Sym(first) = &args[0] {
                let inner_lambda = if args.len() == 3 {
                    Expr::Function("Lambda".to_string(), vec![args[1].clone(), args[2].clone()])
                } else {
                    let inner_args = args[1..].to_vec();
                    Expr::Function("Lambda".to_string(), inner_args)
                };
                Some(BinderNode::Lambda {
                    param: first.clone(),
                    body: Box::new(inner_lambda),
                })
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

/// Maximum recursion depth allowed during binding traversal and substitution.
pub const MAX_BINDING_DEPTH: usize = 256;

/// Checks raw expression depth before binder parsing clones any body expression.
///
/// The recursive helper is itself safe because it refuses before descending past
/// [`MAX_BINDING_DEPTH`]. This preflight is necessary for a deeply nested body
/// directly beneath a binder: parsing that binder into an owned [`BinderNode`]
/// would otherwise clone the entire body before the traversal noticed its depth.
fn preflight_binding_depth(expr: &Expr, depth: usize) -> Result<(), BindingError> {
    if depth > MAX_BINDING_DEPTH {
        return Err(BindingError::DepthExceeded {
            limit: MAX_BINDING_DEPTH,
        });
    }
    match expr {
        Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
            for arg in args {
                preflight_binding_depth(arg, depth + 1)?;
            }
        }
        Expr::Pow(base, exponent) => {
            preflight_binding_depth(base, depth + 1)?;
            preflight_binding_depth(exponent, depth + 1)?;
        }
        Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
    }
    Ok(())
}

/// Converts an expression into canonical De Bruijn indexed form.
pub fn to_de_bruijn(expr: &Expr) -> Result<DeBruijnExpr, BindingError> {
    preflight_binding_depth(expr, 0)?;
    let mut scope = Vec::new();
    expr_to_de_bruijn(expr, &mut scope, 0)
}

fn expr_to_de_bruijn(
    expr: &Expr,
    scope: &mut Vec<Symbol>,
    depth: usize,
) -> Result<DeBruijnExpr, BindingError> {
    if depth > MAX_BINDING_DEPTH {
        return Err(BindingError::DepthExceeded {
            limit: MAX_BINDING_DEPTH,
        });
    }
    Ok(match expr {
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
        Expr::Add(terms) => DeBruijnExpr::Add(
            terms
                .iter()
                .map(|t| expr_to_de_bruijn(t, scope, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Mul(terms) => DeBruijnExpr::Mul(
            terms
                .iter()
                .map(|t| expr_to_de_bruijn(t, scope, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Pow(b, e) => DeBruijnExpr::Pow(
            Box::new(expr_to_de_bruijn(b, scope, depth + 1)?),
            Box::new(expr_to_de_bruijn(e, scope, depth + 1)?),
        ),
        Expr::Function(name, args) => {
            if let Some(binder) = BinderNode::try_from_expr(expr) {
                match binder {
                    BinderNode::Lambda { param, body } => {
                        scope.push(param);
                        let body_db = expr_to_de_bruijn(&body, scope, depth + 1);
                        scope.pop();
                        let body_db = body_db?;
                        return Ok(DeBruijnExpr::Binder("Lambda".into(), Box::new(body_db)));
                    }
                    BinderNode::Integral {
                        var,
                        body,
                        limits: Some((lower, upper)),
                    } => {
                        scope.push(var);
                        let body_db = expr_to_de_bruijn(&body, scope, depth + 1);
                        scope.pop();
                        let body_db = body_db?;
                        let lower_db = expr_to_de_bruijn(&lower, scope, depth + 1)?;
                        let upper_db = expr_to_de_bruijn(&upper, scope, depth + 1)?;
                        return Ok(DeBruijnExpr::BinderWithArguments {
                            name: "Integral".into(),
                            body: Box::new(body_db),
                            arguments: vec![lower_db, upper_db],
                        });
                    }
                    BinderNode::Integral {
                        var,
                        body,
                        limits: None,
                    } => {
                        return Ok(DeBruijnExpr::Function(
                            "Integral".into(),
                            vec![
                                expr_to_de_bruijn(&body, scope, depth + 1)?,
                                expr_to_de_bruijn(&Expr::Sym(var), scope, depth + 1)?,
                            ],
                        ));
                    }
                    BinderNode::Derivative { var, body } => {
                        return Ok(DeBruijnExpr::Function(
                            "Derivative".into(),
                            vec![
                                expr_to_de_bruijn(&body, scope, depth + 1)?,
                                expr_to_de_bruijn(&Expr::Sym(var), scope, depth + 1)?,
                            ],
                        ));
                    }
                }
            }
            DeBruijnExpr::Function(
                name.clone(),
                args.iter()
                    .map(|a| expr_to_de_bruijn(a, scope, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
    })
}

/// Extracts the set of free (unbound) symbols appearing in an expression.
pub fn free_symbols(expr: &Expr) -> Result<BTreeSet<Symbol>, BindingError> {
    preflight_binding_depth(expr, 0)?;
    let mut free = BTreeSet::new();
    let mut bound_scope = BTreeSet::new();
    collect_free_symbols(expr, &mut bound_scope, &mut free, 0)?;
    Ok(free)
}

/// Extracts every symbol appearing in an expression, including binder declarations.
///
/// Capture-avoiding freshening uses this stricter set because choosing an existing
/// inner binder name can change which declaration a renamed occurrence refers to.
fn all_symbols(expr: &Expr) -> Result<BTreeSet<Symbol>, BindingError> {
    let mut symbols = BTreeSet::new();
    collect_all_symbols(expr, &mut symbols, 0)?;
    Ok(symbols)
}

fn collect_all_symbols(
    expr: &Expr,
    symbols: &mut BTreeSet<Symbol>,
    depth: usize,
) -> Result<(), BindingError> {
    if depth > MAX_BINDING_DEPTH {
        return Err(BindingError::DepthExceeded {
            limit: MAX_BINDING_DEPTH,
        });
    }
    match expr {
        Expr::Sym(symbol) => {
            symbols.insert(symbol.clone());
        }
        Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
            for arg in args {
                collect_all_symbols(arg, symbols, depth + 1)?;
            }
        }
        Expr::Pow(base, exponent) => {
            collect_all_symbols(base, symbols, depth + 1)?;
            collect_all_symbols(exponent, symbols, depth + 1)?;
        }
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
    }
    Ok(())
}

fn collect_free_symbols(
    expr: &Expr,
    bound_scope: &mut BTreeSet<Symbol>,
    free: &mut BTreeSet<Symbol>,
    depth: usize,
) -> Result<(), BindingError> {
    if depth > MAX_BINDING_DEPTH {
        return Err(BindingError::DepthExceeded {
            limit: MAX_BINDING_DEPTH,
        });
    }
    match expr {
        Expr::Sym(s) => {
            if !bound_scope.contains(s) {
                free.insert(s.clone());
            }
        }
        Expr::Add(args) | Expr::Mul(args) => {
            for arg in args {
                collect_free_symbols(arg, bound_scope, free, depth + 1)?;
            }
        }
        Expr::Pow(b, e) => {
            collect_free_symbols(b, bound_scope, free, depth + 1)?;
            collect_free_symbols(e, bound_scope, free, depth + 1)?;
        }
        Expr::Function(_, args) => {
            if let Some(binder) = BinderNode::try_from_expr(expr) {
                match binder {
                    BinderNode::Lambda { param, body } => {
                        let newly_bound = bound_scope.insert(param.clone());
                        let body_result = collect_free_symbols(&body, bound_scope, free, depth + 1);
                        if newly_bound {
                            bound_scope.remove(&param);
                        }
                        body_result?;
                    }
                    BinderNode::Integral {
                        var,
                        body,
                        limits: Some((lower, upper)),
                    } => {
                        let newly_bound = bound_scope.insert(var.clone());
                        let body_result = collect_free_symbols(&body, bound_scope, free, depth + 1);
                        if newly_bound {
                            bound_scope.remove(&var);
                        }
                        body_result?;
                        collect_free_symbols(&lower, bound_scope, free, depth + 1)?;
                        collect_free_symbols(&upper, bound_scope, free, depth + 1)?;
                    }
                    BinderNode::Integral {
                        var,
                        body,
                        limits: None,
                    } => {
                        collect_free_symbols(&body, bound_scope, free, depth + 1)?;
                        if !bound_scope.contains(&var) {
                            free.insert(var);
                        }
                    }
                    BinderNode::Derivative { body, .. } => {
                        collect_free_symbols(&body, bound_scope, free, depth + 1)?;
                    }
                }
                return Ok(());
            }

            for arg in args {
                collect_free_symbols(arg, bound_scope, free, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
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
pub fn alpha_equivalent(a: &Expr, b: &Expr) -> Result<bool, BindingError> {
    Ok(to_de_bruijn(a)? == to_de_bruijn(b)?)
}

/// Performs capture-avoiding substitution: replaces occurrences of `target` with `replacement` in `expr`.
pub fn capture_avoiding_subs(
    expr: &Expr,
    target: &Symbol,
    replacement: &Expr,
) -> Result<Expr, BindingError> {
    preflight_binding_depth(expr, 0)?;
    let repl_free = free_symbols(replacement)?;
    subs_internal(expr, target, replacement, &repl_free, 0)
}

fn subs_internal(
    expr: &Expr,
    target: &Symbol,
    replacement: &Expr,
    repl_free: &BTreeSet<Symbol>,
    depth: usize,
) -> Result<Expr, BindingError> {
    if depth > MAX_BINDING_DEPTH {
        return Err(BindingError::DepthExceeded {
            limit: MAX_BINDING_DEPTH,
        });
    }
    Ok(match expr {
        Expr::Sym(s) => {
            if s == target {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        Expr::Add(args) => Expr::Add(
            args.iter()
                .map(|a| subs_internal(a, target, replacement, repl_free, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Mul(args) => Expr::Mul(
            args.iter()
                .map(|a| subs_internal(a, target, replacement, repl_free, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Pow(b, e) => Expr::Pow(
            Arc::new(subs_internal(b, target, replacement, repl_free, depth + 1)?),
            Arc::new(subs_internal(e, target, replacement, repl_free, depth + 1)?),
        ),
        Expr::Function(name, args) => {
            if let Some(binder) = BinderNode::try_from_expr(expr) {
                match &binder {
                    BinderNode::Derivative { var, body } => {
                        let target_occurs_in_body = free_symbols(body)?.contains(target);
                        if var == target {
                            let Expr::Sym(new_var) = replacement else {
                                return Err(BindingError::UnsupportedOperatorVariableReplacement {
                                    operator: "Derivative",
                                });
                            };
                            if new_var != var && free_symbols(body)?.contains(new_var) {
                                // `Derivative(y, x).subs(x, y)` cannot be represented
                                // by merely rebuilding the derivative: doing so would
                                // reinterpret the pre-existing free `y` as the new
                                // differentiation variable.  A profile-compatible
                                // shell can preserve this as an explicit `Subs` node;
                                // the current native IR cannot, so fail closed.
                                return Err(BindingError::NonAlphaRenamingRequired {
                                    operator: "Derivative",
                                });
                            }
                            return Ok(BinderNode::Derivative {
                                var: new_var.clone(),
                                body: Box::new(subs_internal(
                                    body,
                                    target,
                                    replacement,
                                    repl_free,
                                    depth + 1,
                                )?),
                            }
                            .to_expr());
                        }
                        if !target_occurs_in_body {
                            return Ok(expr.clone());
                        }
                        if repl_free.contains(var) {
                            return Err(BindingError::NonAlphaRenamingRequired {
                                operator: "Derivative",
                            });
                        }
                        return Ok(BinderNode::Derivative {
                            var: var.clone(),
                            body: Box::new(subs_internal(
                                body,
                                target,
                                replacement,
                                repl_free,
                                depth + 1,
                            )?),
                        }
                        .to_expr());
                    }
                    BinderNode::Integral {
                        var,
                        body,
                        limits: None,
                    } => {
                        let target_occurs_in_body = free_symbols(body)?.contains(target);
                        if var == target {
                            let Expr::Sym(new_var) = replacement else {
                                return Err(BindingError::UnsupportedOperatorVariableReplacement {
                                    operator: "Integral",
                                });
                            };
                            return Ok(BinderNode::Integral {
                                var: new_var.clone(),
                                body: Box::new(subs_internal(
                                    body,
                                    target,
                                    replacement,
                                    repl_free,
                                    depth + 1,
                                )?),
                                limits: None,
                            }
                            .to_expr());
                        }
                        if !target_occurs_in_body {
                            return Ok(expr.clone());
                        }
                        return Ok(BinderNode::Integral {
                            var: var.clone(),
                            body: Box::new(subs_internal(
                                body,
                                target,
                                replacement,
                                repl_free,
                                depth + 1,
                            )?),
                            limits: None,
                        }
                        .to_expr());
                    }
                    BinderNode::Lambda { .. }
                    | BinderNode::Integral {
                        limits: Some(_), ..
                    } => {}
                }
                let bound_var = binder.bound_variable().clone();
                if &bound_var == target {
                    // The declaration shadows `target` only in its body. Definite
                    // integral limits remain in the surrounding scope and must
                    // still receive the substitution.
                    return Ok(match binder {
                        BinderNode::Integral {
                            var,
                            body,
                            limits: Some((lower, upper)),
                        } => BinderNode::Integral {
                            var,
                            body,
                            limits: Some((
                                Box::new(subs_internal(
                                    &lower,
                                    target,
                                    replacement,
                                    repl_free,
                                    depth + 1,
                                )?),
                                Box::new(subs_internal(
                                    &upper,
                                    target,
                                    replacement,
                                    repl_free,
                                    depth + 1,
                                )?),
                            )),
                        }
                        .to_expr(),
                        BinderNode::Lambda { .. }
                        | BinderNode::Integral { limits: None, .. }
                        | BinderNode::Derivative { .. } => expr.clone(),
                    });
                }
                let body = binder.body();
                let target_occurs_in_body = free_symbols(body)?.contains(target);
                let (new_bound_var, new_body) = if !target_occurs_in_body {
                    (bound_var.clone(), body.clone())
                } else if repl_free.contains(&bound_var) {
                    let mut avoid = all_symbols(body)?;
                    avoid.extend(repl_free.iter().cloned());
                    avoid.insert(target.clone());
                    let fresh = fresh_symbol(&bound_var.name, &avoid);
                    let fresh_free = BTreeSet::from([fresh.clone()]);
                    let renamed_body = subs_internal(
                        body,
                        &bound_var,
                        &Expr::Sym(fresh.clone()),
                        &fresh_free,
                        depth + 1,
                    )?;
                    let new_body =
                        subs_internal(&renamed_body, target, replacement, repl_free, depth + 1)?;
                    (fresh, new_body)
                } else {
                    let new_body = subs_internal(body, target, replacement, repl_free, depth + 1)?;
                    (bound_var, new_body)
                };

                match binder {
                    BinderNode::Lambda { .. } => {
                        return Ok(BinderNode::Lambda {
                            param: new_bound_var,
                            body: Box::new(new_body),
                        }
                        .to_expr());
                    }
                    BinderNode::Derivative { .. } => {
                        return Ok(BinderNode::Derivative {
                            var: new_bound_var,
                            body: Box::new(new_body),
                        }
                        .to_expr());
                    }
                    BinderNode::Integral { limits, .. } => {
                        let new_limits = limits
                            .map(|(lower, upper)| {
                                Ok::<_, BindingError>((
                                    Box::new(subs_internal(
                                        &lower,
                                        target,
                                        replacement,
                                        repl_free,
                                        depth + 1,
                                    )?),
                                    Box::new(subs_internal(
                                        &upper,
                                        target,
                                        replacement,
                                        repl_free,
                                        depth + 1,
                                    )?),
                                ))
                            })
                            .transpose()?;
                        return Ok(BinderNode::Integral {
                            var: new_bound_var,
                            body: Box::new(new_body),
                            limits: new_limits,
                        }
                        .to_expr());
                    }
                }
            }

            Expr::Function(
                name.clone(),
                args.iter()
                    .map(|a| subs_internal(a, target, replacement, repl_free, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        _ => expr.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_symbols(expr: &Expr) -> BTreeSet<Symbol> {
        super::free_symbols(expr).expect("test expression is within the binding depth limit")
    }

    fn to_de_bruijn(expr: &Expr) -> DeBruijnExpr {
        super::to_de_bruijn(expr).expect("test expression is within the binding depth limit")
    }

    fn alpha_equivalent(a: &Expr, b: &Expr) -> bool {
        super::alpha_equivalent(a, b).expect("test expressions are within the binding depth limit")
    }

    fn capture_avoiding_subs(expr: &Expr, target: &Symbol, replacement: &Expr) -> Expr {
        super::capture_avoiding_subs(expr, target, replacement)
            .expect("test expressions are within the binding depth limit")
    }

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
    fn definite_integral_de_bruijn_form_retains_its_scope_boundary() {
        let genuine = BinderNode::lambda(
            Symbol::new("x"),
            BinderNode::definite_integral(
                Symbol::new("y"),
                Expr::symbol("y"),
                Expr::from_i64(0),
                Expr::from_i64(1),
            )
            .to_expr(),
        );
        let opaque = BinderNode::lambda(
            Symbol::new("x"),
            Expr::Function(
                "Integral".into(),
                vec![Expr::symbol("x"), Expr::from_i64(0), Expr::from_i64(1)],
            ),
        );

        assert!(
            !genuine
                .is_alpha_equivalent(&opaque)
                .expect("bounded binders")
        );

        let renamed = BinderNode::lambda(
            Symbol::new("a"),
            BinderNode::definite_integral(
                Symbol::new("b"),
                Expr::symbol("b"),
                Expr::from_i64(0),
                Expr::from_i64(1),
            )
            .to_expr(),
        );
        assert!(
            genuine
                .is_alpha_equivalent(&renamed)
                .expect("bounded binders")
        );
    }

    #[test]
    fn indefinite_integral_and_derivative_variables_are_not_alpha_bound() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let integral_x = BinderNode::integral(x.clone(), Expr::from_i64(1));
        let integral_y = BinderNode::integral(y.clone(), Expr::from_i64(1));
        assert_eq!(integral_x.free_symbols(), Ok(BTreeSet::from([x.clone()])));
        assert_eq!(
            super::free_symbols(&integral_x.to_expr()),
            Ok(BTreeSet::from([x.clone()]))
        );
        assert!(
            !integral_x
                .is_alpha_equivalent(&integral_y)
                .expect("bounded integrals")
        );

        let derivative_x = BinderNode::derivative(
            x.clone(),
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
        );
        let derivative_y = BinderNode::derivative(
            y.clone(),
            Expr::pow(Expr::Sym(y.clone()), Expr::from_i64(2)),
        );
        assert_eq!(derivative_x.free_symbols(), Ok(BTreeSet::from([x.clone()])));
        assert_eq!(
            super::free_symbols(&derivative_x.to_expr()),
            Ok(BTreeSet::from([x.clone()]))
        );
        assert!(
            !derivative_x
                .is_alpha_equivalent(&derivative_y)
                .expect("bounded derivatives")
        );

        let constant_derivative = BinderNode::derivative(x.clone(), Expr::from_i64(1));
        assert_eq!(constant_derivative.free_symbols(), Ok(BTreeSet::new()));
        assert_eq!(
            super::free_symbols(&constant_derivative.to_expr()),
            Ok(BTreeSet::new())
        );

        // An operator variable may still refer to a genuine enclosing binder.
        let nested_x = BinderNode::lambda(x, constant_derivative.to_expr());
        let nested_y = BinderNode::lambda(
            y.clone(),
            BinderNode::derivative(y, Expr::from_i64(1)).to_expr(),
        );
        assert!(
            nested_x
                .is_alpha_equivalent(&nested_y)
                .expect("bounded nested derivative")
        );
    }

    #[test]
    fn absent_substitution_target_does_not_rename_nested_binders() {
        let expr = BinderNode::lambda(
            Symbol::new("x"),
            BinderNode::lambda(Symbol::new("x_1"), Expr::symbol("x")).to_expr(),
        )
        .to_expr();

        let substituted = capture_avoiding_subs(&expr, &Symbol::new("y"), &Expr::symbol("x"));

        assert_eq!(substituted, expr);
    }

    #[test]
    fn freshening_avoids_names_declared_by_nested_binders() {
        let expr = BinderNode::lambda(
            Symbol::new("x"),
            BinderNode::lambda(
                Symbol::new("x_1"),
                Expr::Add(vec![Expr::symbol("y"), Expr::symbol("x")]),
            )
            .to_expr(),
        )
        .to_expr();

        let substituted = capture_avoiding_subs(&expr, &Symbol::new("y"), &Expr::symbol("x"));
        let expected = BinderNode::lambda(
            Symbol::new("x_2"),
            BinderNode::lambda(
                Symbol::new("x_1"),
                Expr::Add(vec![Expr::symbol("x"), Expr::symbol("x_2")]),
            )
            .to_expr(),
        )
        .to_expr();

        assert_eq!(substituted, expected);
    }

    #[test]
    fn shadowing_integral_binder_does_not_shadow_its_limits() {
        let expr = BinderNode::definite_integral(
            Symbol::new("x"),
            Expr::symbol("x"),
            Expr::from_i64(0),
            Expr::symbol("x"),
        )
        .to_expr();

        let substituted = capture_avoiding_subs(&expr, &Symbol::new("x"), &Expr::from_i64(1));
        let expected = BinderNode::definite_integral(
            Symbol::new("x"),
            Expr::symbol("x"),
            Expr::from_i64(0),
            Expr::from_i64(1),
        )
        .to_expr();

        assert_eq!(substituted, expected);
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
    fn definite_integral_binder_capture_avoidance() {
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
    }

    #[test]
    fn operator_substitution_handles_symbol_renaming_and_refuses_derivative_capture() {
        let derivative = BinderNode::derivative(
            Symbol::new("x"),
            Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
        )
        .to_expr();
        let indefinite_integral = BinderNode::integral(
            Symbol::new("x"),
            Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
        )
        .to_expr();

        assert_eq!(
            super::capture_avoiding_subs(&derivative, &Symbol::new("y"), &Expr::symbol("x")),
            Err(BindingError::NonAlphaRenamingRequired {
                operator: "Derivative",
            })
        );

        let renamed_derivative =
            super::capture_avoiding_subs(&derivative, &Symbol::new("x"), &Expr::symbol("z"))
                .expect("a symbolic differentiation-variable rename is representable");
        assert_eq!(
            renamed_derivative,
            BinderNode::derivative(
                Symbol::new("z"),
                Expr::Mul(vec![Expr::symbol("z"), Expr::symbol("y")]),
            )
            .to_expr()
        );

        let renamed_constant_derivative = super::capture_avoiding_subs(
            &BinderNode::derivative(Symbol::new("x"), Expr::from_i64(1)).to_expr(),
            &Symbol::new("x"),
            &Expr::symbol("z"),
        )
        .expect("the declared variable is structurally substitutable even for a constant body");
        assert_eq!(
            renamed_constant_derivative,
            BinderNode::derivative(Symbol::new("z"), Expr::from_i64(1)).to_expr()
        );

        let substituted_integral = super::capture_avoiding_subs(
            &indefinite_integral,
            &Symbol::new("y"),
            &Expr::symbol("x"),
        )
        .expect("an indefinite integral does not alpha-bind its variable");
        assert_eq!(
            substituted_integral,
            BinderNode::integral(
                Symbol::new("x"),
                Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("x")]),
            )
            .to_expr()
        );

        let renamed_integral = super::capture_avoiding_subs(
            &indefinite_integral,
            &Symbol::new("x"),
            &Expr::symbol("z"),
        )
        .expect("a symbolic integration-variable rename is representable");
        assert_eq!(
            renamed_integral,
            BinderNode::integral(
                Symbol::new("z"),
                Expr::Mul(vec![Expr::symbol("z"), Expr::symbol("y")]),
            )
            .to_expr()
        );

        let safe_derivative =
            super::capture_avoiding_subs(&derivative, &Symbol::new("y"), &Expr::symbol("z"))
                .expect("replacement does not collide with the differentiation variable");
        assert_eq!(
            safe_derivative,
            BinderNode::derivative(
                Symbol::new("x"),
                Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("z")]),
            )
            .to_expr()
        );

        for (operator, expression) in [
            ("Derivative", derivative),
            ("Integral", indefinite_integral),
        ] {
            assert_eq!(
                super::capture_avoiding_subs(&expression, &Symbol::new("x"), &Expr::from_i64(2),),
                Err(BindingError::UnsupportedOperatorVariableReplacement { operator })
            );
        }
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
        let free1 = parsed1.free_symbols().expect("bounded binder");
        assert!(free1.contains(&y));
        assert!(!free1.contains(&x));

        let lambda2 = BinderNode::lambda(
            z.clone(),
            Expr::Mul(vec![Expr::Sym(z.clone()), Expr::Sym(y.clone())]),
        );
        assert!(
            lambda1
                .is_alpha_equivalent(&lambda2)
                .expect("bounded binders")
        );

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
        let free_int = parsed_int.free_symbols().expect("bounded binder");
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
        assert_eq!(
            parsed_deriv.free_symbols().expect("bounded derivative"),
            BTreeSet::from([x])
        );
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
            !outer_dependent
                .is_alpha_equivalent(&inner_shadow)
                .expect("bounded binders"),
            "shadowing inner binder that captures an outer free symbol must \
             not be alpha-equivalent to the non-shadowing version"
        );

        // Regression: a true alpha renaming must still be reported equivalent.
        let renamed = BinderNode::lambda(
            Symbol::new("b"),
            Expr::Function("Lambda".into(), vec![Expr::symbol("a"), Expr::symbol("b")]),
        );
        assert!(
            outer_dependent
                .is_alpha_equivalent(&renamed)
                .expect("bounded binders")
        );
    }

    #[test]
    fn multi_parameter_lambda_tuple_and_multi_arg_alpha_equivalence() {
        // Lambda((x, y), x + y)
        let tuple_xy = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function("Tuple".into(), vec![Expr::symbol("x"), Expr::symbol("y")]),
                Expr::Add(vec![Expr::symbol("x"), Expr::symbol("y")]),
            ],
        );
        // Lambda((a, b), a + b)
        let tuple_ab = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function("Tuple".into(), vec![Expr::symbol("a"), Expr::symbol("b")]),
                Expr::Add(vec![Expr::symbol("a"), Expr::symbol("b")]),
            ],
        );
        // Lambda(x, y, x + y) (multi-arg representation)
        let multi_xy = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("x"),
                Expr::symbol("y"),
                Expr::Add(vec![Expr::symbol("x"), Expr::symbol("y")]),
            ],
        );
        // Curried Lambda(a, Lambda(b, a + b))
        let curried_ab = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("a"),
                Expr::Function(
                    "Lambda".into(),
                    vec![
                        Expr::symbol("b"),
                        Expr::Add(vec![Expr::symbol("a"), Expr::symbol("b")]),
                    ],
                ),
            ],
        );

        assert!(alpha_equivalent(&tuple_xy, &tuple_ab));
        assert!(alpha_equivalent(&tuple_xy, &multi_xy));
        assert!(alpha_equivalent(&tuple_xy, &curried_ab));
        assert!(alpha_equivalent(&multi_xy, &curried_ab));

        // 3-parameter tuple Lambda((x, y, z), x * y + z)
        let tuple_3_xyz = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function(
                    "Tuple".into(),
                    vec![Expr::symbol("x"), Expr::symbol("y"), Expr::symbol("z")],
                ),
                Expr::Add(vec![
                    Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
                    Expr::symbol("z"),
                ]),
            ],
        );
        let tuple_3_abc = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function(
                    "Tuple".into(),
                    vec![Expr::symbol("a"), Expr::symbol("b"), Expr::symbol("c")],
                ),
                Expr::Add(vec![
                    Expr::Mul(vec![Expr::symbol("a"), Expr::symbol("b")]),
                    Expr::symbol("c"),
                ]),
            ],
        );
        let multi_3_xyz = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::symbol("x"),
                Expr::symbol("y"),
                Expr::symbol("z"),
                Expr::Add(vec![
                    Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
                    Expr::symbol("z"),
                ]),
            ],
        );
        assert!(alpha_equivalent(&tuple_3_xyz, &tuple_3_abc));
        assert!(alpha_equivalent(&tuple_3_xyz, &multi_3_xyz));
    }

    #[test]
    fn duplicate_parameters_remain_opaque_instead_of_becoming_binders() {
        let duplicate_tuple = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function("Tuple".into(), vec![Expr::symbol("x"), Expr::symbol("x")]),
                Expr::symbol("x"),
            ],
        );
        let duplicate_multi_arg = Expr::Function(
            "Lambda".into(),
            vec![Expr::symbol("x"), Expr::symbol("x"), Expr::symbol("x")],
        );
        let renamed_duplicate_tuple = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function("Tuple".into(), vec![Expr::symbol("a"), Expr::symbol("a")]),
                Expr::symbol("a"),
            ],
        );

        for malformed in [&duplicate_tuple, &duplicate_multi_arg] {
            assert!(BinderNode::try_from_expr(malformed).is_none());
            assert_eq!(free_symbols(malformed), BTreeSet::from([Symbol::new("x")]));
        }
        assert!(!alpha_equivalent(
            &duplicate_tuple,
            &renamed_duplicate_tuple
        ));

        let legitimate_shadowing = BinderNode::lambda(
            Symbol::new("x"),
            BinderNode::lambda(Symbol::new("x"), Expr::symbol("x")).to_expr(),
        )
        .to_expr();
        assert!(BinderNode::try_from_expr(&legitimate_shadowing).is_some());
    }

    #[test]
    fn multi_parameter_lambda_free_symbols_and_capture_avoiding_subs() {
        // Lambda((x, y), x + y + z): free symbol is {z}
        let expr = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function("Tuple".into(), vec![Expr::symbol("x"), Expr::symbol("y")]),
                Expr::Add(vec![
                    Expr::symbol("x"),
                    Expr::symbol("y"),
                    Expr::symbol("z"),
                ]),
            ],
        );
        let free = free_symbols(&expr);
        assert_eq!(free.len(), 1);
        assert!(free.contains(&Symbol::new("z")));

        // Substitute z -> x + 1. Parameter x must be freshened to avoid capture!
        let target = Symbol::new("z");
        let replacement = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]);
        let substituted = capture_avoiding_subs(&expr, &target, &replacement);

        // Substituted expression must still have free symbols {x}
        let sub_free = free_symbols(&substituted);
        assert_eq!(sub_free.len(), 1);
        assert!(sub_free.contains(&Symbol::new("x")));

        // Must be alpha-equivalent to Lambda((x_fresh, y), (x_fresh + y + (x + 1)))
        let expected_alpha = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function("Tuple".into(), vec![Expr::symbol("x_1"), Expr::symbol("y")]),
                Expr::Add(vec![
                    Expr::symbol("x_1"),
                    Expr::symbol("y"),
                    Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]),
                ]),
            ],
        );
        assert!(alpha_equivalent(&substituted, &expected_alpha));
    }

    #[test]
    fn bindings_recursion_depth_limit_fails_closed() {
        let mut deep_x = Expr::symbol("x");
        let mut deep_y = Expr::symbol("y");
        for _ in 0..MAX_BINDING_DEPTH + 10 {
            deep_x = Expr::Add(vec![deep_x, Expr::from_i64(1)]);
            deep_y = Expr::Add(vec![deep_y, Expr::from_i64(1)]);
        }
        let expected = BindingError::DepthExceeded {
            limit: MAX_BINDING_DEPTH,
        };

        assert_eq!(super::free_symbols(&deep_x), Err(expected.clone()));
        assert_eq!(super::to_de_bruijn(&deep_x), Err(expected.clone()));
        assert_eq!(
            super::alpha_equivalent(&deep_x, &deep_y),
            Err(expected.clone())
        );
        assert_eq!(
            super::capture_avoiding_subs(&deep_x, &Symbol::new("x"), &Expr::from_i64(1)),
            Err(expected)
        );
    }

    #[test]
    fn operator_variable_substitution_with_non_symbol_replacement_refuses() {
        // Check landed in 08ef1c5: when substituting the operator variable of a
        // Derivative or indefinite Integral, the replacement must itself be a
        // single symbol. Anything else is refused with
        // BindingError::UnsupportedOperatorVariableReplacement instead of being
        // silently substituted or producing NonAlphaRenamingRequired.
        let deriv = BinderNode::derivative(
            Symbol::new("x"),
            Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]),
        )
        .to_expr();
        let err = super::capture_avoiding_subs(&deriv, &Symbol::new("x"), &Expr::from_i64(5))
            .unwrap_err();
        assert!(matches!(
            err,
            BindingError::UnsupportedOperatorVariableReplacement { operator } if operator == "Derivative"
        ));

        let integral = BinderNode::integral(Symbol::new("x"), Expr::symbol("x")).to_expr();
        let err = super::capture_avoiding_subs(
            &integral,
            &Symbol::new("x"),
            &Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(1)]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            BindingError::UnsupportedOperatorVariableReplacement { operator } if operator == "Integral"
        ));

        // The success path: substituting a fresh symbol is representable.
        let renamed = super::capture_avoiding_subs(&deriv, &Symbol::new("x"), &Expr::symbol("y"))
            .expect("symbolic operator-variable substitution must succeed");
        let expected = BinderNode::derivative(
            Symbol::new("y"),
            Expr::Add(vec![Expr::symbol("y"), Expr::from_i64(1)]),
        )
        .to_expr();
        assert_eq!(renamed, expected);
    }

    #[test]
    fn derivative_variable_substitution_refuses_existing_free_symbol_capture() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        for body in [
            Expr::Sym(y.clone()),
            Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
        ] {
            let derivative = BinderNode::derivative(x.clone(), body).to_expr();
            assert_eq!(
                super::capture_avoiding_subs(&derivative, &x, &Expr::Sym(y.clone())),
                Err(BindingError::NonAlphaRenamingRequired {
                    operator: "Derivative",
                })
            );
        }

        let derivative = BinderNode::derivative(y.clone(), Expr::Sym(y.clone())).to_expr();
        let renamed = super::capture_avoiding_subs(&derivative, &y, &Expr::symbol("z"))
            .expect("a fresh differentiation variable does not capture the body");
        assert_eq!(
            renamed,
            BinderNode::derivative(Symbol::new("z"), Expr::symbol("z")).to_expr()
        );

        let constant = BinderNode::derivative(x.clone(), Expr::from_i64(1)).to_expr();
        let renamed_constant = super::capture_avoiding_subs(&constant, &x, &Expr::Sym(y.clone()))
            .expect("a variable absent from the body cannot be captured");
        assert_eq!(
            renamed_constant,
            BinderNode::derivative(y, Expr::from_i64(1)).to_expr()
        );
    }
}

#[cfg(test)]
mod nested_tuple_documentation_tests {
    //! Nested-tuple Lambda is a known limitation of the current
    //! `try_parse_lambda` parser. `Lambda(Tuple(Tuple(x,y), z), body)` returns
    //! `None` from `BinderNode::try_from_expr` because the parameter-list
    //! parser only accepts flat `Tuple`s of `Expr::Sym` entries.
    //!
    //! The test below documents the current behavior. It is marked
    //! `#[ignore]` because lifting the limitation requires a recursive Tuple
    //! parser; the curried form `Lambda(a, Lambda(Tuple(b, c), ...))` covers
    //! the same surface and is the recommended workaround.
    use super::*;

    #[test]
    #[ignore = "nested Tuple Lambda is rejected by try_parse_lambda; \
                use curried Lambda(a, Lambda(Tuple(b,c), ...)) instead. \
                Removing this attribute requires a recursive Tuple parser."]
    fn nested_tuple_lambda_alpha_equivalence_and_substitution() {
        // Lambda((Tuple(x, y), z), x + y + z)
        let nested = Expr::Function(
            "Lambda".into(),
            vec![
                Expr::Function(
                    "Tuple".into(),
                    vec![
                        Expr::Function("Tuple".into(), vec![Expr::symbol("x"), Expr::symbol("y")]),
                        Expr::symbol("z"),
                    ],
                ),
                Expr::Add(vec![
                    Expr::symbol("x"),
                    Expr::symbol("y"),
                    Expr::symbol("z"),
                ]),
            ],
        );
        // Currently rejected by try_from_expr.
        assert!(
            BinderNode::try_from_expr(&nested).is_none(),
            "nested Tuple Lambda must be rejected by the current parser"
        );
    }
}
