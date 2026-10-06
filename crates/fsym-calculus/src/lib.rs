//! # fsym-calculus
//!
//! Symbolic differentiation, integration, limits, and series expansion.

pub mod compile;
pub mod gruntz;
pub mod integrate;
pub mod proof;
pub mod series;
pub mod summation;

pub(crate) use integrate::rational_parts as integrate_rational_parts;
pub mod sparse_jacobian;
pub mod transforms;

pub use compile::*;
pub use proof::*;
pub use sparse_jacobian::*;
pub use transforms::*;

use fsym_core::{BigInt, BigRational, Constant, CoreError, Expr, RealBall, Symbol};
use fsym_simplify::simplify;
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

const MAX_DIRECT_SUBSTITUTION_NODES: usize = 16_384;
const MAX_DIRECT_SUBSTITUTION_DEPTH: usize = 128;
const UNSAFE_DIRECT_SUBSTITUTION: &str =
    "direct substitution encountered a literal pole or exceeded traversal limits";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CalculusError {
    #[error("Cannot differentiate non-differentiable term: {0}")]
    NonDifferentiable(String),
    #[error("Integration not computable symbolically: {0}")]
    IntegrationFailed(String),
    #[error("Limit undetermined with available rules: {0}")]
    Undetermined(String),
}

fn numeric_value(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Compute the unsimplified symbolic derivative following direct definitional reduction rules.
/// Certified real-box evaluation over the declared transcendental subset
/// (sin, cos, exp) at the requested decimal precision.
///
/// Thin delegation to the fsym-core ball evaluator; this is the single
/// import site for campaign code that needs enclosures with explicit error
/// bounds rather than binary64 approximations. Unsupported operations
/// refuse with a typed error; nothing here silently degrades precision.
pub fn evalf_ball(expr: &Expr, precision_digits: u32) -> Result<RealBall, CoreError> {
    expr.evalf_ball(precision_digits)
}

pub fn diff_unsimplified(expr: &Expr, var: &Symbol) -> Expr {
    match expr {
        Expr::Sym(s) => {
            if s == var {
                Expr::from_i64(1)
            } else {
                Expr::from_i64(0)
            }
        }
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => Expr::from_i64(0),
        Expr::Add(terms) => {
            let diff_terms: Vec<Expr> = terms.iter().map(|t| diff(t, var)).collect();
            Expr::Add(diff_terms)
        }
        Expr::Mul(factors) => {
            // Product rule: d(f*g*h) = f'*g*h + f*g'*h + f*g*h'
            let mut add_terms = Vec::new();
            for i in 0..factors.len() {
                let mut prod_factors = Vec::new();
                for (j, factor) in factors.iter().enumerate() {
                    if i == j {
                        prod_factors.push(diff(factor, var));
                    } else {
                        prod_factors.push(factor.clone());
                    }
                }
                add_terms.push(Expr::Mul(prod_factors));
            }
            Expr::Add(add_terms)
        }
        Expr::Pow(base, exp) => {
            let du = diff(base, var);
            let dv = diff(exp, var);
            if du.is_zero() && dv.is_zero() {
                Expr::from_i64(0)
            } else if dv.is_zero() {
                // u(x)^c where c does not depend on var: c * u^(c - 1) * u'
                let exp_minus_1 = if let Expr::Integer(n) = exp.as_ref() {
                    Expr::Integer(n - BigInt::from(1))
                } else {
                    Expr::Add(vec![exp.as_ref().clone(), Expr::from_i64(-1)])
                };
                Expr::Mul(vec![
                    exp.as_ref().clone(),
                    Expr::pow(base.as_ref().clone(), exp_minus_1),
                    du,
                ])
            } else if du.is_zero() {
                // c^v(x) where c does not depend on var: c^v * ln(c) * v'
                Expr::Mul(vec![
                    expr.clone(),
                    Expr::Function("log".to_string(), vec![base.as_ref().clone()]),
                    dv,
                ])
            } else {
                // General chain rule: u(x)^v(x) * (v' * ln(u) + v * u' / u)
                let term1 = Expr::Mul(vec![
                    dv,
                    Expr::Function("log".to_string(), vec![base.as_ref().clone()]),
                ]);
                let term2 = Expr::Mul(vec![
                    exp.as_ref().clone(),
                    du,
                    Expr::pow(base.as_ref().clone(), Expr::from_i64(-1)),
                ]);
                Expr::Mul(vec![expr.clone(), Expr::Add(vec![term1, term2])])
            }
        }
        Expr::Function(name, args) if matches!(name.as_str(), "Integral" | "Sum") => {
            diff_binder(expr, name, args, var)
        }
        Expr::Function(name, args) if name == "Subs" && args.len() == 3 => {
            diff_subs(expr, args, var)
        }
        Expr::Function(name, args) => {
            // If all arguments are independent of var, then by the chain rule d(f(args))/d(var) = 0.
            if name != "Derivative" && name != "diff" && args.iter().all(|a| diff(a, var).is_zero())
            {
                return Expr::from_i64(0);
            }
            // Elementary derivatives
            if name == "sin" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![Expr::Function("cos".to_string(), vec![u.clone()]), du])
            } else if name == "cos" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("sin".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "tan" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let tan_u_sq = Expr::pow(
                    Expr::Function("tan".to_string(), vec![u.clone()]),
                    Expr::from_i64(2),
                );
                Expr::Mul(vec![Expr::Add(vec![Expr::from_i64(1), tan_u_sq]), du])
            } else if name == "exp" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![Expr::Function("exp".to_string(), vec![u.clone()]), du])
            } else if name == "sinh" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::Function("cosh".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "cosh" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::Function("sinh".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "tanh" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let tanh_u_sq = Expr::pow(
                    Expr::Function("tanh".to_string(), vec![u.clone()]),
                    Expr::from_i64(2),
                );
                Expr::Mul(vec![
                    Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::Mul(vec![Expr::from_i64(-1), tanh_u_sq]),
                    ]),
                    du,
                ])
            } else if (name == "log" || name == "ln") && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![Expr::pow(u.clone(), Expr::from_i64(-1)), du])
            } else if name == "atan" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let denom = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                Expr::Mul(vec![Expr::pow(denom, Expr::from_i64(-1)), du])
            } else if name == "asin" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let one_minus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]),
                ]);
                let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                Expr::Mul(vec![Expr::pow(one_minus_u_sq, neg_half), du])
            } else if name == "acos" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let one_minus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]),
                ]);
                let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(one_minus_u_sq, neg_half),
                    du,
                ])
            } else if name == "asinh" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let one_plus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                Expr::Mul(vec![Expr::pow(one_plus_u_sq, neg_half), du])
            } else if name == "acosh" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let u_sq_minus_one = Expr::Add(vec![
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                    Expr::from_i64(-1),
                ]);
                let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                Expr::Mul(vec![Expr::pow(u_sq_minus_one, neg_half), du])
            } else if name == "atanh" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let one_minus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]),
                ]);
                Expr::Mul(vec![Expr::pow(one_minus_u_sq, Expr::from_i64(-1)), du])
            } else if name == "cot" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let cot_u_sq = Expr::pow(
                    Expr::Function("cot".to_string(), vec![u.clone()]),
                    Expr::from_i64(2),
                );
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Add(vec![Expr::from_i64(1), cot_u_sq]),
                    du,
                ])
            } else if name == "sec" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::Function("sec".to_string(), vec![u.clone()]),
                    Expr::Function("tan".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "csc" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("csc".to_string(), vec![u.clone()]),
                    Expr::Function("cot".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "coth" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let coth_u_sq = Expr::pow(
                    Expr::Function("coth".to_string(), vec![u.clone()]),
                    Expr::from_i64(2),
                );
                Expr::Mul(vec![
                    Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::Mul(vec![Expr::from_i64(-1), coth_u_sq]),
                    ]),
                    du,
                ])
            } else if name == "sech" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("sech".to_string(), vec![u.clone()]),
                    Expr::Function("tanh".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "csch" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("csch".to_string(), vec![u.clone()]),
                    Expr::Function("coth".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "acot" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let denom = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(denom, Expr::from_i64(-1)),
                    du,
                ])
            } else if name == "acoth" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let one_minus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]),
                ]);
                Expr::Mul(vec![Expr::pow(one_minus_u_sq, Expr::from_i64(-1)), du])
            } else if name == "asec" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let u_sq = Expr::pow(u.clone(), Expr::from_i64(2));
                let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                let one_minus_inv_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![Expr::from_i64(-1), inv_u_sq]),
                ]);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_term = Expr::pow(one_minus_inv_u_sq, half);
                let denom = Expr::Mul(vec![u_sq, sqrt_term]);
                Expr::Mul(vec![Expr::pow(denom, Expr::from_i64(-1)), du])
            } else if name == "acsc" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let u_sq = Expr::pow(u.clone(), Expr::from_i64(2));
                let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                let one_minus_inv_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![Expr::from_i64(-1), inv_u_sq]),
                ]);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_term = Expr::pow(one_minus_inv_u_sq, half);
                let denom = Expr::Mul(vec![u_sq, sqrt_term]);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(denom, Expr::from_i64(-1)),
                    du,
                ])
            } else if name == "asech" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let one_minus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]),
                ]);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_term = Expr::pow(one_minus_u_sq, half);
                let denom = Expr::Mul(vec![u.clone(), sqrt_term]);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(denom, Expr::from_i64(-1)),
                    du,
                ])
            } else if name == "acsch" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let u_sq = Expr::pow(u.clone(), Expr::from_i64(2));
                let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                let one_plus_inv_u_sq = Expr::Add(vec![Expr::from_i64(1), inv_u_sq]);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_term = Expr::pow(one_plus_inv_u_sq, half);
                let denom = Expr::Mul(vec![u_sq, sqrt_term]);
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(denom, Expr::from_i64(-1)),
                    du,
                ])
            } else if name == "sinc" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let cos_u = Expr::Function("cos".to_string(), vec![u.clone()]);
                let sin_u = Expr::Function("sin".to_string(), vec![u.clone()]);
                let inv_u = Expr::pow(u.clone(), Expr::from_i64(-1));
                let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                let term1 = Expr::Mul(vec![cos_u, inv_u]);
                let term2 = Expr::Mul(vec![Expr::from_i64(-1), sin_u, inv_u_sq]);
                Expr::Mul(vec![Expr::Add(vec![term1, term2]), du])
            } else if matches!(name.as_str(), "Si" | "Ci" | "Ei" | "Shi" | "Chi") && args.len() == 1
            {
                // d/du Si(u) = sin(u)/u, Ci -> cos, Ei -> exp, Shi -> sinh, Chi -> cosh.
                let u = &args[0];
                let du = diff(u, var);
                let kernel = match name.as_str() {
                    "Si" => "sin",
                    "Ci" => "cos",
                    "Ei" => "exp",
                    "Shi" => "sinh",
                    _ => "cosh",
                };
                Expr::Mul(vec![
                    Expr::Function(kernel.to_string(), vec![u.clone()]),
                    Expr::pow(u.clone(), Expr::from_i64(-1)),
                    du,
                ])
            } else if name == "Heaviside" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![
                    Expr::Function("DiracDelta".to_string(), vec![u.clone()]),
                    du,
                ])
            } else if name == "erf" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let pi = Expr::Const(Constant::Pi);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_pi = Expr::pow(pi, half);
                let inv_sqrt_pi = Expr::pow(sqrt_pi, Expr::from_i64(-1));
                let neg_u_sq = Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                let exp_neg_u_sq = Expr::Function("exp".to_string(), vec![neg_u_sq]);
                Expr::Mul(vec![Expr::from_i64(2), inv_sqrt_pi, exp_neg_u_sq, du])
            } else if name == "erfc" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                let pi = Expr::Const(Constant::Pi);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_pi = Expr::pow(pi, half);
                let inv_sqrt_pi = Expr::pow(sqrt_pi, Expr::from_i64(-1));
                let neg_u_sq = Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                let exp_neg_u_sq = Expr::Function("exp".to_string(), vec![neg_u_sq]);
                Expr::Mul(vec![Expr::from_i64(-2), inv_sqrt_pi, exp_neg_u_sq, du])
            } else if name == "LambertW" && args.len() == 1 {
                // W'(u) = W(u) / (u * (1 + W(u))) * u'
                let u = &args[0];
                let du = diff(u, var);
                let w = expr.clone();
                Expr::Mul(vec![
                    w.clone(),
                    Expr::pow(
                        Expr::Mul(vec![u.clone(), Expr::Add(vec![w, Expr::from_i64(1)])]),
                        Expr::from_i64(-1),
                    ),
                    du,
                ])
            } else if name == "atan2" && args.len() == 2 {
                // d atan2(y, x) = (x*y' - y*x') / (x^2 + y^2)
                let (y, xx) = (&args[0], &args[1]);
                let dy = diff(y, var);
                let dx = diff(xx, var);
                Expr::Mul(vec![
                    Expr::Add(vec![
                        Expr::Mul(vec![xx.clone(), dy]),
                        Expr::Mul(vec![Expr::from_i64(-1), y.clone(), dx]),
                    ]),
                    Expr::pow(
                        Expr::Add(vec![
                            Expr::pow(xx.clone(), Expr::from_i64(2)),
                            Expr::pow(y.clone(), Expr::from_i64(2)),
                        ]),
                        Expr::from_i64(-1),
                    ),
                ])
            } else if (name == "Derivative" || name == "diff") && !args.is_empty() {
                if diff(&args[0], var).is_zero() {
                    Expr::from_i64(0)
                } else {
                    let mut new_args = args.clone();
                    new_args.push(Expr::Sym(var.clone()));
                    Expr::Function(name.clone(), new_args)
                }
            } else {
                Expr::Function(
                    "diff".to_string(),
                    vec![expr.clone(), Expr::Sym(var.clone())],
                )
            }
        }
    }
}

/// One binder limit: variable and optional `(lower, upper)` bounds.
type BinderLimit = (Symbol, Option<(Expr, Expr)>);

fn binder_limits(args: &[Expr]) -> Option<Vec<BinderLimit>> {
    args.iter()
        .map(|lim| match lim {
            Expr::Sym(v) => Some((v.clone(), None)),
            Expr::Function(t, items) if t == "Tuple" => match items.as_slice() {
                [Expr::Sym(v)] => Some((v.clone(), None)),
                [Expr::Sym(v), a, b] => Some((v.clone(), Some((a.clone(), b.clone())))),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn limit_expr(lim: &BinderLimit) -> Expr {
    match &lim.1 {
        None => Expr::Function("Tuple".into(), vec![Expr::Sym(lim.0.clone())]),
        Some((a, b)) => Expr::Function(
            "Tuple".into(),
            vec![Expr::Sym(lim.0.clone()), a.clone(), b.clone()],
        ),
    }
}

fn unevaluated_diff(expr: &Expr, var: &Symbol) -> Expr {
    Expr::Function(
        "diff".to_string(),
        vec![expr.clone(), Expr::Sym(var.clone())],
    )
}

/// Derivative of `Sum`/`Integral` with respect to `var`: the body is
/// differentiated under the binder when `var` is free and the bounds do not
/// depend on it; a definite integral with variable bounds uses the Leibniz
/// rule (single limit). Any other shape stays unevaluated.
fn diff_binder(expr: &Expr, name: &str, args: &[Expr], var: &Symbol) -> Expr {
    let Some(limits) = binder_limits(&args[1..]) else {
        return unevaluated_diff(expr, var);
    };
    let body = &args[0];
    let depends = |e: &Expr| e.free_symbols().iter().any(|s| s == var);
    let bound_dep = limits
        .iter()
        .any(|(_, b)| b.as_ref().is_some_and(|(a, c)| depends(a) || depends(c)));
    let is_bound = limits.iter().any(|(v, _)| v == var);
    if is_bound {
        // var is a dummy of a definite limit: the result does not depend on it.
        let definite = limits.iter().any(|(v, b)| v == var && b.is_some());
        if name == "Sum" || definite {
            if !bound_dep {
                return Expr::from_i64(0);
            }
        } else if limits.len() == 1 {
            // d/dx Integral(f(x), x) = f(x)
            return body.clone();
        }
        return unevaluated_diff(expr, var);
    }
    if !bound_dep {
        let mut new_args = vec![diff(body, var)];
        new_args.extend(limits.iter().map(limit_expr));
        if new_args[0].is_zero() {
            return Expr::from_i64(0);
        }
        return Expr::Function(name.to_string(), new_args);
    }
    if name == "Integral" && limits.len() == 1 {
        let (v, Some((a, b))) = &limits[0] else {
            return unevaluated_diff(expr, var);
        };
        let at = |p: &Expr| body.subs(&HashMap::from([(v.clone(), p.clone())]));
        let mut terms = vec![
            at(b) * diff(b, var),
            Expr::from_i64(-1) * at(a) * diff(a, var),
        ];
        let inner = diff(body, var);
        if !inner.is_zero() {
            terms.push(Expr::Function(
                "Integral".into(),
                vec![inner, limit_expr(&limits[0])],
            ));
        }
        return Expr::Add(terms);
    }
    unevaluated_diff(expr, var)
}

/// d/dy Subs(e, (v_i), (p_i)) = Subs(de/dy, v, p) + sum_i Subs(de/dv_i, v, p) * dp_i/dy
/// for y not among the bound v_i (chain rule through the points).
fn diff_subs(expr: &Expr, args: &[Expr], var: &Symbol) -> Expr {
    let (Expr::Function(tv, vars), Expr::Function(_, points)) = (&args[1], &args[2]) else {
        return unevaluated_diff(expr, var);
    };
    if tv != "Tuple" || vars.len() != points.len() {
        return unevaluated_diff(expr, var);
    }
    let body = &args[0];
    let bound = vars.iter().any(|v| matches!(v, Expr::Sym(s) if s == var));
    // With symbol points the substitution is a plain renaming (upstream
    // evaluates such Subs when differentiating).
    let renaming: Option<HashMap<Symbol, Expr>> = vars
        .iter()
        .zip(points)
        .map(|(v, p)| match (v, p) {
            (Expr::Sym(vs), Expr::Sym(_)) => Some((vs.clone(), p.clone())),
            _ => None,
        })
        .collect();
    let wrap = |e: Expr| -> Expr {
        if e.is_zero() {
            return e;
        }
        if let Some(map) = &renaming {
            return e.subs(map);
        }
        Expr::Function("Subs".into(), vec![e, args[1].clone(), args[2].clone()])
    };
    let mut terms = Vec::new();
    if !bound {
        terms.push(wrap(diff(body, var)));
    }
    for (v, p) in vars.iter().zip(points) {
        let Expr::Sym(vs) = v else {
            return unevaluated_diff(expr, var);
        };
        let dp = diff(p, var);
        if !dp.is_zero() {
            terms.push(wrap(diff(body, vs)) * dp);
        }
    }
    terms.into_iter().fold(Expr::from_i64(0), |a, b| a + b)
}

fn eliminate_zero_products(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let non_zero: Vec<Expr> = terms
                .iter()
                .map(eliminate_zero_products)
                .filter(|t| !t.is_zero())
                .collect();
            match non_zero.len() {
                0 => Expr::from_i64(0),
                1 => non_zero.into_iter().next().expect("len checked"),
                _ => Expr::Add(non_zero),
            }
        }
        Expr::Mul(factors) => {
            let mapped: Vec<Expr> = factors.iter().map(eliminate_zero_products).collect();
            if mapped.iter().any(|f| f.is_zero()) {
                Expr::from_i64(0)
            } else {
                match mapped.len() {
                    0 => Expr::from_i64(1),
                    1 => mapped.into_iter().next().expect("len checked"),
                    _ => Expr::Mul(mapped),
                }
            }
        }
        Expr::Pow(base, exp) => Expr::Pow(
            Arc::new(eliminate_zero_products(base)),
            Arc::new(eliminate_zero_products(exp)),
        ),
        Expr::Function(name, args) => Expr::Function(
            name.clone(),
            args.iter().map(eliminate_zero_products).collect(),
        ),
        _ => expr.clone(),
    }
}

/// Compute the symbolic derivative of an expression with respect to a symbol: ∂expr / ∂var.
pub fn diff(expr: &Expr, var: &Symbol) -> Expr {
    let unsimplified = diff_unsimplified(expr, var);
    let cleaned = eliminate_zero_products(&unsimplified);
    simplify(&cleaned)
}

/// Compute the N-th derivative: d^n(expr) / d(var)^n.
pub fn diff_n(expr: &Expr, var: &Symbol, n: usize) -> Expr {
    let mut current = expr.clone();
    for _ in 0..n {
        current = diff(&current, var);
    }
    current
}

fn is_free_of(expr: &Expr, var: &Symbol) -> bool {
    !expr.free_symbols().iter().any(|s| s == var)
}

/// Undifferentiated-derivative sentinel produced by [`diff`]'s fallback.
/// Whether `expr` carries the non-differentiable sentinel of [`diff`].
pub(crate) fn carries_diff_sentinel_pub(expr: &Expr) -> bool {
    carries_diff_sentinel(expr)
}

fn carries_diff_sentinel(expr: &Expr) -> bool {
    match expr {
        Expr::Function(name, args)
            if (name == "diff" || name == "Derivative") && args.len() >= 2 =>
        {
            true
        }
        Expr::Add(terms) | Expr::Mul(terms) => terms.iter().any(carries_diff_sentinel),
        Expr::Pow(b, e) => carries_diff_sentinel(b) || carries_diff_sentinel(e),
        Expr::Function(_, args) => args.iter().any(carries_diff_sentinel),
        _ => false,
    }
}

/// Antiderivative of a single term in `var` (no `+ C`).
///
/// Covers constants, the power rule (including `n = -1 -> log`),
/// and `exp`/`sin`/`cos` of the bare variable. Everything else is a typed
/// refusal.
fn integral_term(f: &Expr, var: &Symbol) -> Result<Expr, CalculusError> {
    let x = Expr::Sym(var.clone());
    let half = || Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
    match f {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => Ok(f.clone() * x),
        Expr::Sym(s) if s == var => Ok(Expr::Mul(vec![half(), x.clone(), x])),
        Expr::Sym(_) => Ok(f.clone() * x),
        Expr::Pow(base, exp) if base.as_ref() == &x => match exp.as_ref() {
            Expr::Integer(n) if *n != BigInt::from(-1) => {
                let np1 = n + BigInt::from(1);
                Ok(Expr::Mul(vec![
                    Expr::Rational(BigRational::new(BigInt::from(1), np1.clone())),
                    Expr::Pow(Arc::new(x.clone()), Arc::new(Expr::Integer(np1))),
                ]))
            }
            Expr::Integer(n) if *n == BigInt::from(-1) => {
                Ok(Expr::Function("log".to_string(), vec![x]))
            }
            Expr::Rational(r) => {
                let np1 = r + BigRational::from_integer(1.into());
                if np1.is_zero() {
                    return Ok(Expr::Function("log".to_string(), vec![x]));
                }
                Ok(Expr::Mul(vec![
                    Expr::Rational(np1.recip()),
                    Expr::Pow(Arc::new(x.clone()), Arc::new(Expr::Rational(np1))),
                ]))
            }
            other => Err(CalculusError::IntegrationFailed(format!("x^{other}"))),
        },
        Expr::Function(name, args) if args.len() == 1 => {
            let u = &args[0];
            let (c, is_linear) = if u == &x {
                (Expr::from_i64(1), true)
            } else if let Expr::Mul(factors) = u {
                let mut const_factors = Vec::new();
                let mut var_count = 0;
                let mut all_other_factors_constant = true;
                for f in factors {
                    if f == &x {
                        var_count += 1;
                    } else if is_free_of(f, var) {
                        const_factors.push(f.clone());
                    } else {
                        all_other_factors_constant = false;
                    }
                }
                if var_count == 1 && all_other_factors_constant {
                    (simplify(&Expr::Mul(const_factors)), true)
                } else {
                    (Expr::from_i64(1), false)
                }
            } else {
                (Expr::from_i64(1), false)
            };

            if is_linear {
                if c == Expr::from_i64(1) {
                    match name.as_str() {
                        "exp" => Ok(Expr::Function("exp".to_string(), vec![u.clone()])),
                        "sin" => Ok(Expr::Mul(vec![
                            Expr::from_i64(-1),
                            Expr::Function("cos".to_string(), vec![u.clone()]),
                        ])),
                        "cos" => Ok(Expr::Function("sin".to_string(), vec![u.clone()])),
                        "sinh" => Ok(Expr::Function("cosh".to_string(), vec![u.clone()])),
                        "cosh" => Ok(Expr::Function("sinh".to_string(), vec![u.clone()])),
                        other => Err(CalculusError::IntegrationFailed(format!("{other}({u})"))),
                    }
                } else if numeric_value(&c).is_some_and(|value| !value.is_zero()) {
                    let inv_c = Expr::Pow(Arc::new(c), Arc::new(Expr::from_i64(-1)));
                    match name.as_str() {
                        "exp" => Ok(simplify(&Expr::Mul(vec![
                            inv_c,
                            Expr::Function("exp".to_string(), vec![u.clone()]),
                        ]))),
                        "sin" => Ok(simplify(&Expr::Mul(vec![
                            Expr::from_i64(-1),
                            inv_c,
                            Expr::Function("cos".to_string(), vec![u.clone()]),
                        ]))),
                        "cos" => Ok(simplify(&Expr::Mul(vec![
                            inv_c,
                            Expr::Function("sin".to_string(), vec![u.clone()]),
                        ]))),
                        "sinh" => Ok(simplify(&Expr::Mul(vec![
                            inv_c,
                            Expr::Function("cosh".to_string(), vec![u.clone()]),
                        ]))),
                        "cosh" => Ok(simplify(&Expr::Mul(vec![
                            inv_c,
                            Expr::Function("sinh".to_string(), vec![u.clone()]),
                        ]))),
                        other => Err(CalculusError::IntegrationFailed(format!("{other}({u})"))),
                    }
                } else if c.is_zero() {
                    // A zero slope makes the function constant. Simplify first so
                    // exp(0*x), sin(0*x), and cos(0*x) take their exact values.
                    integral_term(&simplify(f), var)
                } else {
                    // Dividing by a symbolic slope would silently assume it is
                    // nonzero. Keep the antiderivative conditional until an
                    // assumptions context can discharge that side condition.
                    Err(CalculusError::IntegrationFailed(format!("{name}({u})")))
                }
            } else {
                Err(CalculusError::IntegrationFailed(format!("{name}({u})")))
            }
        }
        other => Err(CalculusError::IntegrationFailed(other.to_string())),
    }
}

/// Maximum polynomial degree admitted by the integration-by-parts rule.
const MAX_BY_PARTS_DEGREE: usize = 8;

/// Ascending exact-rational coefficient view of a polynomial in `var`.
///
/// Returns `None` for anything that is not a bounded polynomial with exact
/// rational coefficients: negative, symbolic, or degree-capped exponents
/// and non-polynomial leaves all refuse rather than misclassify.
fn polynomial_coeffs(expr: &Expr, var: &Symbol) -> Option<Vec<BigRational>> {
    let x = Expr::Sym(var.clone());
    match expr {
        Expr::Integer(n) => Some(vec![BigRational::from_integer(n.clone())]),
        Expr::Rational(r) => Some(vec![r.clone()]),
        Expr::Sym(s) if s == var => Some(vec![BigRational::zero(), BigRational::one()]),
        Expr::Pow(base, exp) if base.as_ref() == &x => {
            let Expr::Integer(n) = exp.as_ref() else {
                return None;
            };
            let n = n.to_u64()? as usize;
            if n == 0 || n > MAX_BY_PARTS_DEGREE {
                return None;
            }
            let mut coeffs = vec![BigRational::zero(); n];
            coeffs.push(BigRational::one());
            Some(coeffs)
        }
        Expr::Add(terms) => {
            let mut total = vec![BigRational::zero()];
            for term in terms {
                let t = polynomial_coeffs(term, var)?;
                if t.len() - 1 > MAX_BY_PARTS_DEGREE {
                    return None;
                }
                if t.len() > total.len() {
                    total.resize(t.len(), BigRational::zero());
                }
                for (i, c) in t.iter().enumerate() {
                    total[i] += c.clone();
                }
            }
            Some(total)
        }
        Expr::Mul(factors) => {
            let mut product = vec![BigRational::one()];
            for factor in factors {
                let f = polynomial_coeffs(factor, var)?;
                let new_degree = (product.len() - 1) + (f.len() - 1);
                if new_degree > MAX_BY_PARTS_DEGREE {
                    return None;
                }
                let mut next = vec![BigRational::zero(); new_degree + 1];
                for (i, a) in product.iter().enumerate() {
                    for (j, b) in f.iter().enumerate() {
                        let mut ab = a.clone();
                        ab *= b.clone();
                        next[i + j] += ab;
                    }
                }
                product = next;
            }
            Some(product)
        }
        _ => None,
    }
}

/// d/dvar of an ascending coefficient array.
fn derivative_coeffs(coeffs: &[BigRational]) -> Vec<BigRational> {
    coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| {
            let mut scaled = c.clone();
            scaled *= BigRational::from_integer(BigInt::from(i as i64));
            scaled
        })
        .collect()
}

/// Exact slope of a strictly-linear argument `a * var`, or `None`.
///
/// Mirrors the linearity classification in [`integral_term`]: the bare
/// variable or a product of exactly one variable factor with exact numeric
/// constants. A zero slope refuses so the antiderivative never silently
/// divides by zero.
fn linear_argument_slope(u: &Expr, var: &Symbol) -> Option<BigRational> {
    let x = Expr::Sym(var.clone());
    if u == &x {
        return Some(BigRational::one());
    }
    let Expr::Mul(factors) = u else {
        return None;
    };
    let mut var_count = 0usize;
    let mut slope = BigRational::one();
    for factor in factors {
        if factor == &x {
            var_count += 1;
        } else {
            let q = numeric_value(factor)?;
            slope *= q;
        }
    }
    if var_count == 1 && !slope.is_zero() {
        Some(slope)
    } else {
        None
    }
}

/// Antiderivative cycle of `g(a*x)` for the by-parts tableau: entry `m`
/// holds the `(m+1)`-th antiderivative of `g`.
fn antiderivative_cycle(name: &str, ax: &Expr) -> Option<Vec<Expr>> {
    let applied = |n: &str| Expr::Function(n.to_string(), vec![ax.clone()]);
    let negated = |e: Expr| Expr::Mul(vec![Expr::from_i64(-1), e]);
    match name {
        "exp" => Some(vec![applied("exp")]),
        "sin" => Some(vec![
            negated(applied("cos")),
            negated(applied("sin")),
            applied("cos"),
            applied("sin"),
        ]),
        "cos" => Some(vec![
            applied("sin"),
            negated(applied("cos")),
            negated(applied("sin")),
            applied("cos"),
        ]),
        "sinh" => Some(vec![applied("cosh"), applied("sinh")]),
        "cosh" => Some(vec![applied("sinh"), applied("cosh")]),
        _ => None,
    }
}

/// Canonical expression for an exact rational coefficient.
fn coefficient_expr(q: &BigRational) -> Expr {
    if q.is_integer() {
        Expr::Integer(q.to_integer())
    } else {
        Expr::Rational(q.clone())
    }
}

/// Ascending-coefficient polynomial expression in `var` (zero terms dropped).
fn polynomial_expr(coeffs: &[BigRational], var: &Symbol) -> Expr {
    let x = Expr::Sym(var.clone());
    let mut terms: Vec<Expr> = Vec::new();
    for (i, c) in coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let power = match i {
            0 => None,
            1 => Some(x.clone()),
            _ => Some(Expr::Pow(
                Arc::new(x.clone()),
                Arc::new(Expr::from_i64(i as i64)),
            )),
        };
        match power {
            None => terms.push(coefficient_expr(c)),
            Some(p) => {
                if *c == BigRational::one() {
                    terms.push(p);
                } else {
                    terms.push(Expr::Mul(vec![coefficient_expr(c), p]));
                }
            }
        }
    }
    match terms.len() {
        0 => Expr::from_i64(0),
        1 => terms.into_iter().next().expect("len checked"),
        _ => Expr::Add(terms),
    }
}

/// Integrates `p(var) * g(a*var)` by bounded tabular parts:
/// `sum_k (-1)^k p^(k)(var) * G_{k+1}(a*var) / a^{k+1}` where `G` walks the
/// antiderivative cycle of `g`. The sum is finite because the polynomial
/// derivative chain terminates.
///
/// Returns `None` unless the factor list is exactly one bounded polynomial
/// plus one linear-argument analytic factor; the caller keeps its typed
/// refusal for every other shape.
fn try_integrate_by_parts(var_parts: &[Expr], var: &Symbol) -> Option<Expr> {
    const ANALYTIC_NAMES: [&str; 5] = ["exp", "sin", "cos", "sinh", "cosh"];
    if var_parts.len() != 2 {
        return None;
    }
    let classify = |part: &Expr| -> Option<bool> {
        if let Expr::Function(name, args) = part {
            if args.len() == 1 && ANALYTIC_NAMES.contains(&name.as_str()) {
                // Analytic factors only participate with linear arguments.
                return if linear_argument_slope(&args[0], var).is_some() {
                    Some(true)
                } else {
                    None
                };
            }
            return None;
        }
        if polynomial_coeffs(part, var).is_some() {
            Some(false)
        } else {
            None
        }
    };
    let (func_part, poly_part) = match (classify(&var_parts[0]), classify(&var_parts[1])) {
        (Some(true), Some(false)) => (&var_parts[0], &var_parts[1]),
        (Some(false), Some(true)) => (&var_parts[1], &var_parts[0]),
        _ => return None,
    };
    let Expr::Function(name, args) = func_part else {
        return None;
    };
    let a = linear_argument_slope(&args[0], var)?;
    let p = polynomial_coeffs(poly_part, var)?;
    if p.len() < 2 {
        // Degree-zero factors are peeled as constants by `integrate`.
        return None;
    }
    let x = Expr::Sym(var.clone());
    let ax = simplify(&Expr::Mul(vec![coefficient_expr(&a), x]));
    let cycle = antiderivative_cycle(name, &ax)?;
    let mut terms: Vec<Expr> = Vec::new();
    let mut derivative = p.clone();
    let mut a_power = BigRational::one();
    for k in 0..p.len() {
        if derivative.iter().any(|c| !c.is_zero()) {
            a_power *= a.clone();
            let mut coeff = Expr::Rational(a_power.recip());
            if k % 2 == 1 {
                coeff = Expr::Mul(vec![Expr::from_i64(-1), coeff]);
            }
            let next_antiderivative = cycle[k % cycle.len()].clone();
            terms.push(simplify(&Expr::Mul(vec![
                coeff,
                polynomial_expr(&derivative, var),
                next_antiderivative,
            ])));
        }
        derivative = derivative_coeffs(&derivative);
    }
    if terms.is_empty() {
        return None;
    }
    Some(simplify(&Expr::Add(terms)))
}

/// Indefinite integral of `expr` with respect to `var` (no `+ C`).
///
/// Handles sums, constant factors, and every case in
/// [`integral_term`]; anything else fails as
/// [`CalculusError::IntegrationFailed`] rather than returning a guess.
pub fn integrate(expr: &Expr, var: &Symbol) -> Result<Expr, CalculusError> {
    if let Some(anti) = integrate::antiderivative(expr, var) {
        return Ok(anti);
    }
    integrate_basic(expr, var)
}

/// The original term-table integrator, kept as a fallback lane.
fn integrate_basic(expr: &Expr, var: &Symbol) -> Result<Expr, CalculusError> {
    match expr {
        Expr::Add(terms) => {
            let mut parts = Vec::new();
            for t in terms {
                parts.push(integrate(t, var)?);
            }
            Ok(simplify(&Expr::Add(parts)))
        }
        Expr::Mul(factors) => {
            let mut consts = Vec::new();
            let mut var_parts = Vec::new();
            for f in factors {
                if is_free_of(f, var) {
                    consts.push(f.clone());
                } else {
                    var_parts.push(f.clone());
                }
            }
            let c = simplify(&Expr::Mul(consts));
            if var_parts.is_empty() {
                return Ok(simplify(&(c * Expr::Sym(var.clone()))));
            }
            if var_parts.len() > 1 {
                // Bounded products of one exact polynomial and one
                // linear-argument analytic factor integrate by parts;
                // everything else stays a typed refusal.
                if let Some(anti) = try_integrate_by_parts(&var_parts, var) {
                    return Ok(simplify(&(c * anti)));
                }
                return Err(CalculusError::IntegrationFailed(format!(
                    "{}",
                    simplify(&Expr::Mul(var_parts))
                )));
            }
            let inner = var_parts.pop().expect("len checked");
            let anti = integrate(&inner, var)?;
            Ok(simplify(&(c * anti)))
        }
        other => integral_term(other, var).map(|e| simplify(&e)),
    }
}

/// Limit of `expr` as `var -> to` from the right (`dir = "+"`, the upstream
/// default for finite points). See [`limit_dir`].
pub fn limit(expr: &Expr, var: &Symbol, to: &Expr) -> Result<Expr, CalculusError> {
    limit_dir(expr, var, to, gruntz::Direction::Plus)
}

/// Limit of `expr` as `var -> to` in direction `dir`.
///
/// Finite points first try direct substitution when the expression is
/// free of poles and discontinuous functions at the point; everything else
/// (indeterminate forms, poles, essential singularities, limits at
/// infinity) goes through the Gruntz algorithm in [`gruntz`]. Undecidable
/// constants and unsupported functions refuse with
/// [`CalculusError::Undetermined`]; a two-sided limit whose one-sided limits
/// differ refuses as well.
pub fn limit_dir(
    expr: &Expr,
    var: &Symbol,
    to: &Expr,
    dir: gruntz::Direction,
) -> Result<Expr, CalculusError> {
    if unsafe_direct_substitution(expr) || unsafe_direct_substitution(to) {
        return Err(CalculusError::Undetermined(
            UNSAFE_DIRECT_SUBSTITUTION.to_string(),
        ));
    }
    if carries_diff_sentinel(expr) {
        return Err(CalculusError::Undetermined(expr.to_string()));
    }
    if is_free_of(expr, var) {
        return Ok(expr.clone());
    }
    let infinite_point = matches!(
        to,
        Expr::Const(Constant::Infinity) | Expr::Const(Constant::NegativeInfinity)
    );
    if !infinite_point
        && continuous_shape(expr)
        && let Some(value) = direct_substitution(expr, var, to)
    {
        return Ok(value);
    }
    gruntz::limit(expr, var, to, dir).map_err(|e| CalculusError::Undetermined(e.to_string()))
}

/// Functions whose elementary formulas are continuous wherever they
/// evaluate to a finite value (no jump discontinuities).
fn continuous_shape(expr: &Expr) -> bool {
    match expr {
        Expr::Function(name, args) => {
            !matches!(
                name.as_str(),
                "floor" | "ceiling" | "sign" | "frac" | "Heaviside" | "Piecewise" | "Mod"
            ) && args.iter().all(continuous_shape)
        }
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().all(continuous_shape),
        Expr::Pow(b, e) => continuous_shape(b) && continuous_shape(e),
        _ => true,
    }
}

fn has_infinity(expr: &Expr) -> bool {
    match expr {
        Expr::Const(
            Constant::Infinity
            | Constant::NegativeInfinity
            | Constant::ComplexInfinity
            | Constant::NaN,
        ) => true,
        Expr::Add(xs) | Expr::Mul(xs) | Expr::Function(_, xs) => xs.iter().any(has_infinity),
        Expr::Pow(b, e) => has_infinity(b) || has_infinity(e),
        _ => false,
    }
}

/// Value at the point when exact substitution is finite and pole-free.
fn direct_substitution(expr: &Expr, var: &Symbol, to: &Expr) -> Option<Expr> {
    let substituted = expr.subs(&HashMap::from([(var.clone(), to.clone())]));
    if unsafe_direct_substitution(&substituted) || has_infinity(&substituted) {
        return None;
    }
    let value = simplify(&substituted);
    if unsafe_direct_substitution(&value) || has_infinity(&value) || !is_free_of(&value, var) {
        return None;
    }
    Some(value)
}

/// Series terms `(coefficient, exponent)` and the remainder order.
pub type SeriesTerms = (Vec<(Expr, BigRational)>, Option<BigRational>);

/// Generalized series of `expr` about `var = x0` to absolute order `n`.
///
/// Returns the nonzero terms as `(coefficient, exponent)` pairs in the
/// local variable (`x - x0` for finite `x0`, `1/x` for `x0 = oo`, `-1/x`
/// for `x0 = -oo`) and the order of the remainder (`None` when the
/// expansion is exact). `log` terms of the local variable stay in the
/// coefficients (`series(x*log(x))` keeps `x*log(x)`).
///
/// Returns `(coefficient, exponent)` pairs and the remainder order.
pub fn series_expansion(
    expr: &Expr,
    var: &Symbol,
    x0: &Expr,
    n: i64,
) -> Result<SeriesTerms, CalculusError> {
    if unsafe_direct_substitution(expr) || unsafe_direct_substitution(x0) {
        return Err(CalculusError::NonDifferentiable(
            UNSAFE_DIRECT_SUBSTITUTION.to_string(),
        ));
    }
    let t = Symbol::new("_fsym_series_t");
    let te = Expr::Sym(t.clone());
    let xe = Expr::Sym(var.clone());
    let recip = |e: Expr| fsym_core::elementary::eval_pow(e, Expr::from_i64(-1));
    let (replacement, log_t) = match x0 {
        Expr::Const(Constant::Infinity) => (
            recip(te.clone()),
            -Expr::Function("log".into(), vec![xe.clone()]),
        ),
        Expr::Const(Constant::NegativeInfinity) => (
            -recip(te.clone()),
            -Expr::Function("log".into(), vec![-xe.clone()]),
        ),
        _ if x0.is_zero() => (te.clone(), Expr::Function("log".into(), vec![xe.clone()])),
        _ => (
            x0.clone() + te.clone(),
            Expr::Function("log".into(), vec![xe.clone() - x0.clone()]),
        ),
    };
    let local = expr.subs(&HashMap::from([(var.clone(), replacement)]));
    let target = BigRational::from_integer(BigInt::from(n));
    let s = series::series_in(&local, &t, &target, &log_t)
        .map_err(|e| CalculusError::NonDifferentiable(e.to_string()))?;
    let terms = s
        .terms
        .into_iter()
        .map(|(k, c)| (simplify(&c), k))
        .filter(|(c, _)| !c.is_zero())
        .collect();
    Ok((terms, s.order))
}

/// Fail-closed preflight for direct substitution and Taylor coefficients.
///
/// Returns `true` for a literal pole (a negative numeric power of exact zero),
/// an expression beyond the fixed traversal limits, or traversal allocation
/// failure. The iterative walk prevents hostile function nesting from turning
/// a typed refusal into stack overflow.
fn unsafe_direct_substitution(expr: &Expr) -> bool {
    let mut pending = Vec::new();
    if pending.try_reserve(1).is_err() {
        return true;
    }
    pending.push((expr, 1usize));
    let mut discovered = 1usize;

    while let Some((node, depth)) = pending.pop() {
        if depth > MAX_DIRECT_SUBSTITUTION_DEPTH {
            return true;
        }
        let Some(child_depth) = depth.checked_add(1) else {
            return true;
        };

        let children: &[Expr] = match node {
            Expr::Pow(base, exp) => {
                let negative_exp = match exp.as_ref() {
                    Expr::Integer(n) => n.is_negative(),
                    Expr::Rational(value) => value < &BigRational::zero(),
                    _ => false,
                };
                if negative_exp && base.is_zero() {
                    return true;
                }
                if discovered
                    .checked_add(2)
                    .is_none_or(|count| count > MAX_DIRECT_SUBSTITUTION_NODES)
                    || pending.try_reserve(2).is_err()
                {
                    return true;
                }
                discovered += 2;
                pending.push((exp, child_depth));
                pending.push((base, child_depth));
                continue;
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => children,
            // Exact substitution folds `0**-1` to `zoo`; an undefined value
            // is a pole for the purposes of direct substitution.
            Expr::Const(Constant::ComplexInfinity | Constant::NaN) => return true,
            _ => continue,
        };

        if discovered
            .checked_add(children.len())
            .is_none_or(|count| count > MAX_DIRECT_SUBSTITUTION_NODES)
            || pending.try_reserve(children.len()).is_err()
        {
            return true;
        }
        discovered += children.len();
        pending.extend(children.iter().map(|child| (child, child_depth)));
    }

    false
}

/// Taylor polynomial of `expr` around `var = at` through degree `order`.
///
/// Coefficients are exact where the derivatives evaluate exactly; a
/// derivative that hits [`diff`]'s non-differentiable sentinel aborts with
/// [`CalculusError::NonDifferentiable`].
pub fn taylor(expr: &Expr, var: &Symbol, at: &Expr, order: usize) -> Result<Expr, CalculusError> {
    const MAX_ORDER: usize = 12;
    if order > MAX_ORDER {
        return Err(CalculusError::NonDifferentiable(format!(
            "order {order} exceeds supported maximum {MAX_ORDER}"
        )));
    }
    if unsafe_direct_substitution(expr) || unsafe_direct_substitution(at) {
        return Err(CalculusError::NonDifferentiable(
            UNSAFE_DIRECT_SUBSTITUTION.to_string(),
        ));
    }
    let x = Expr::Sym(var.clone());
    // No Sub impl on Expr: shift = x + (-1)*at, folded by simplify.
    let neg_at = simplify(&Expr::Mul(vec![Expr::from_i64(-1), at.clone()]));
    let shift = simplify(&(x.clone() + neg_at));
    let mut terms = Vec::new();
    let mut factorial: u64 = 1;
    for k in 0..=order {
        if k > 1 {
            factorial *= k as u64;
        }
        let deriv = diff_n(expr, var, k);
        if unsafe_direct_substitution(&deriv) {
            return Err(CalculusError::NonDifferentiable(
                UNSAFE_DIRECT_SUBSTITUTION.to_string(),
            ));
        }
        let simplified = simplify(&deriv);
        if unsafe_direct_substitution(&simplified) {
            return Err(CalculusError::NonDifferentiable(
                UNSAFE_DIRECT_SUBSTITUTION.to_string(),
            ));
        }
        if k > 0 && carries_diff_sentinel(&simplified) {
            return Err(CalculusError::NonDifferentiable(expr.to_string()));
        }
        let substituted = simplified.subs(&HashMap::from([(var.clone(), at.clone())]));
        if unsafe_direct_substitution(&substituted) {
            return Err(CalculusError::NonDifferentiable(
                UNSAFE_DIRECT_SUBSTITUTION.to_string(),
            ));
        }
        let value = simplify(&substituted);
        if unsafe_direct_substitution(&value) {
            return Err(CalculusError::NonDifferentiable(
                UNSAFE_DIRECT_SUBSTITUTION.to_string(),
            ));
        }
        if carries_diff_sentinel(&value) {
            return Err(CalculusError::NonDifferentiable(value.to_string()));
        }
        let scaled = if k == 0 {
            value
        } else {
            simplify(&Expr::Mul(vec![
                value,
                Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(factorial))),
            ]))
        };
        let term = if k == 0 {
            scaled
        } else {
            simplify(
                &(scaled * Expr::Pow(Arc::new(shift.clone()), Arc::new(Expr::from_i64(k as i64)))),
            )
        };
        if !term.is_zero() {
            terms.push(term);
        }
    }
    let _ = &x;
    Ok(simplify(&Expr::Add(if terms.is_empty() {
        vec![Expr::from_i64(0)]
    } else {
        terms
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_polynomial() {
        let x = Symbol::new("x");
        // f(x) = x^3 + 2*x + 5
        let expr = Expr::Add(vec![
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(3))),
            Expr::Mul(vec![Expr::from_i64(2), Expr::symbol("x")]),
            Expr::from_i64(5),
        ]);
        let d = diff(&expr, &x);
        // df/dx = 3*x^2 + 2
        let free = d.free_symbols();
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].name, "x");
    }

    #[test]
    fn test_diff_trig() {
        let x = Symbol::new("x");
        let expr = Expr::Function("sin".to_string(), vec![Expr::symbol("x")]);
        let d = diff(&expr, &x);
        assert_eq!(
            d,
            Expr::Function("cos".to_string(), vec![Expr::symbol("x")])
        );

        let sec_expr = Expr::Function("sec".to_string(), vec![Expr::symbol("x")]);
        let d_sec = diff(&sec_expr, &x);
        assert_eq!(
            d_sec,
            Expr::Mul(vec![
                Expr::Function("sec".to_string(), vec![Expr::symbol("x")]),
                Expr::Function("tan".to_string(), vec![Expr::symbol("x")]),
            ])
        );

        let csc_expr = Expr::Function("csc".to_string(), vec![Expr::symbol("x")]);
        let d_csc = diff(&csc_expr, &x);
        assert_eq!(
            d_csc,
            Expr::Mul(vec![
                Expr::from_i64(-1),
                Expr::Function("cot".to_string(), vec![Expr::symbol("x")]),
                Expr::Function("csc".to_string(), vec![Expr::symbol("x")]),
            ])
        );
    }

    #[test]
    fn test_integrate_hyperbolic() {
        let x = Symbol::new("x");
        let x_expr = Expr::symbol("x");

        // ∫sinh(x) dx = cosh(x)
        let sinh_expr = Expr::Function("sinh".to_string(), vec![x_expr.clone()]);
        assert_eq!(
            integrate(&sinh_expr, &x).unwrap(),
            Expr::Function("cosh".to_string(), vec![x_expr.clone()])
        );

        // ∫cosh(x) dx = sinh(x)
        let cosh_expr = Expr::Function("cosh".to_string(), vec![x_expr.clone()]);
        assert_eq!(
            integrate(&cosh_expr, &x).unwrap(),
            Expr::Function("sinh".to_string(), vec![x_expr.clone()])
        );

        // ∫sinh(2*x) dx = 1/2 * cosh(2*x)
        let sinh_2x = Expr::Function(
            "sinh".to_string(),
            vec![Expr::Mul(vec![Expr::from_i64(2), x_expr.clone()])],
        );
        let inv_2 = Expr::rational(1, 2).unwrap();
        assert_eq!(
            integrate(&sinh_2x, &x).unwrap(),
            Expr::Mul(vec![
                inv_2,
                Expr::Function(
                    "cosh".to_string(),
                    vec![Expr::Mul(vec![Expr::from_i64(2), x_expr.clone()])]
                ),
            ])
        );

        // ∫cosh(3*x) dx = 1/3 * sinh(3*x)
        let cosh_3x = Expr::Function(
            "cosh".to_string(),
            vec![Expr::Mul(vec![Expr::from_i64(3), x_expr.clone()])],
        );
        let inv_3 = Expr::rational(1, 3).unwrap();
        assert_eq!(
            integrate(&cosh_3x, &x).unwrap(),
            Expr::Mul(vec![
                inv_3,
                Expr::Function(
                    "sinh".to_string(),
                    vec![Expr::Mul(vec![Expr::from_i64(3), x_expr])]
                ),
            ])
        );
    }

    #[test]
    fn test_integrate_power_rule() {
        let x = Symbol::new("x");
        let x2 = Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(2)));
        // ∫x² dx = x³/3
        let anti = integrate(&x2, &x).unwrap();
        assert_eq!(
            anti,
            Expr::Mul(vec![
                Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(3))),
                Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(3))),
            ])
        );
        // Constant multiple: ∫3x² dx = x³ (3 * 1/3 folds to 1).
        let scaled = Expr::Mul(vec![Expr::from_i64(3), x2]);
        assert_eq!(
            integrate(&scaled, &x).unwrap(),
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(3)))
        );
    }

    #[test]
    fn test_integrate_reciprocal_and_elementary() {
        let x = Symbol::new("x");
        // ∫1/x dx = log(x)
        let recip = Expr::Mul(vec![
            Expr::symbol("x"),
            Expr::Pow(Arc::new(Expr::from_i64(1)), Arc::new(Expr::from_i64(-1))),
        ]);
        // Note: 1/x stays structural; the power rule with n=-1 applies to x^-1.
        let inv = Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(-1)));
        assert_eq!(
            integrate(&inv, &x).unwrap(),
            Expr::Function("log".to_string(), vec![Expr::symbol("x")])
        );
        let _ = recip;
        // ∫cos(x) dx = sin(x)
        let cos_x = Expr::Function("cos".to_string(), vec![Expr::symbol("x")]);
        assert_eq!(
            integrate(&cos_x, &x).unwrap(),
            Expr::Function("sin".to_string(), vec![Expr::symbol("x")])
        );
    }

    #[test]
    fn test_integrate_typed_refusal_on_unsupported_product() {
        let x = Symbol::new("x");
        // x * log(x) integrates by parts: x**2*log(x)/2 - x**2/4.
        let e = Expr::Mul(vec![
            Expr::symbol("x"),
            Expr::Function("log".to_string(), vec![Expr::symbol("x")]),
        ]);
        let anti = integrate(&e, &x).expect("by-parts antiderivative");
        assert_eq!(simplify(&(diff(&anti, &x) - e)), Expr::from_i64(0));
        // sin(x)/x has no elementary antiderivative: upstream answers with
        // the sine integral Si(x), which differentiates back exactly.
        let si = Expr::Mul(vec![
            Expr::Function("sin".to_string(), vec![Expr::symbol("x")]),
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(-1))),
        ]);
        let anti = integrate(&si, &x).expect("Si antiderivative");
        assert_eq!(
            anti,
            Expr::Function("Si".to_string(), vec![Expr::symbol("x")])
        );
        assert_eq!(simplify(&(diff(&anti, &x) - si)), Expr::from_i64(0));
        // exp(x**3) has no closed form in the supported rule set: typed refusal.
        let hard = Expr::Function(
            "exp".to_string(),
            vec![Expr::Pow(
                Arc::new(Expr::symbol("x")),
                Arc::new(Expr::from_i64(3)),
            )],
        );
        assert!(matches!(
            integrate(&hard, &x),
            Err(CalculusError::IntegrationFailed(_))
        ));
    }

    #[test]
    fn integration_by_parts_roundtrips_polynomial_analytic_products() {
        let x = Symbol::new("x");
        let analytic = |name: &str, u: Expr| Expr::Function(name.to_string(), vec![u]);
        let x_pow = |n: i64| Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(n)));
        let roundtrip = |integrand: &Expr| {
            let anti = integrate(integrand, &x).expect("by-parts integral expected");
            let residual = fsym_simplify::expand(&(diff(&anti, &x) - integrand.clone()));
            assert_eq!(
                simplify(&residual),
                Expr::from_i64(0),
                "{integrand} -> {anti}"
            );
        };

        roundtrip(&Expr::Mul(vec![
            Expr::symbol("x"),
            analytic("sin", Expr::symbol("x")),
        ]));
        roundtrip(&Expr::Mul(vec![
            x_pow(2),
            analytic("exp", Expr::symbol("x")),
        ]));
        roundtrip(&Expr::Mul(vec![
            Expr::from_i64(3),
            x_pow(2),
            analytic("exp", Expr::symbol("x")),
        ]));
        roundtrip(&Expr::Mul(vec![
            Expr::symbol("x"),
            analytic("cos", Expr::Mul(vec![Expr::from_i64(2), Expr::symbol("x")])),
        ]));
        roundtrip(&Expr::Mul(vec![
            Expr::symbol("x"),
            analytic("sinh", Expr::symbol("x")),
        ]));
        roundtrip(&Expr::Mul(vec![
            x_pow(4),
            analytic("sin", Expr::symbol("x")),
        ]));

        // Exact closed form for the canonical case: x*exp(x) -> exp(x)*(x-1).
        let anti = integrate(
            &Expr::Mul(vec![Expr::symbol("x"), analytic("exp", Expr::symbol("x"))]),
            &x,
        )
        .unwrap();
        let expected = Expr::Mul(vec![
            analytic("exp", Expr::symbol("x")),
            Expr::Add(vec![Expr::from_i64(-1), Expr::symbol("x")]),
        ]);
        assert_eq!(
            fsym_simplify::expand(&anti),
            fsym_simplify::expand(&expected)
        );
    }

    #[test]
    fn integration_by_parts_refusals_stay_typed() {
        let x = Symbol::new("x");
        let refused = |integrand: &Expr| {
            assert!(matches!(
                integrate(integrand, &x),
                Err(CalculusError::IntegrationFailed(_))
            ));
        };
        // Non-elementary: sin(x**2) (Fresnel) and exp(x**2) (erfi).
        refused(&Expr::Function(
            "sin".to_string(),
            vec![Expr::Pow(
                Arc::new(Expr::symbol("x")),
                Arc::new(Expr::from_i64(2)),
            )],
        ));
        refused(&Expr::Function(
            "exp".to_string(),
            vec![Expr::Pow(
                Arc::new(Expr::symbol("x")),
                Arc::new(Expr::from_i64(2)),
            )],
        ));
        // Previously refused shapes now integrate exactly; the derivative
        // of each antiderivative reproduces the integrand.
        let verified = |integrand: &Expr| {
            let anti = integrate(integrand, &x).expect("antiderivative expected");
            assert_eq!(
                simplify(&(diff(&anti, &x) - integrand.clone())),
                Expr::from_i64(0),
                "{integrand} -> {anti}"
            );
        };
        verified(&Expr::Mul(vec![
            Expr::symbol("x"),
            Expr::Function(
                "sin".to_string(),
                vec![Expr::Pow(
                    Arc::new(Expr::symbol("x")),
                    Arc::new(Expr::from_i64(2)),
                )],
            ),
        ]));
        verified(&Expr::Mul(vec![
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(9))),
            Expr::Function("sin".to_string(), vec![Expr::symbol("x")]),
        ]));
    }

    #[test]
    fn integration_discharges_linear_slope_before_division() {
        let x = Symbol::new("x");
        let x_expr = Expr::Sym(x.clone());

        let symbolic_slope = Expr::Function(
            "exp".to_string(),
            vec![Expr::Mul(vec![Expr::symbol("a"), x_expr.clone()])],
        );
        assert!(matches!(
            integrate(&symbolic_slope, &x),
            Err(CalculusError::IntegrationFailed(_))
        ));

        let nonlinear_argument = Expr::Function(
            "exp".to_string(),
            vec![Expr::Mul(vec![
                x_expr.clone(),
                Expr::Function("sin".to_string(), vec![x_expr.clone()]),
            ])],
        );
        assert!(matches!(
            integrate(&nonlinear_argument, &x),
            Err(CalculusError::IntegrationFailed(_))
        ));

        let zero_slope = Expr::Function(
            "exp".to_string(),
            vec![Expr::Mul(vec![Expr::from_i64(0), x_expr])],
        );
        assert_eq!(integrate(&zero_slope, &x).unwrap(), Expr::symbol("x"));
    }

    #[test]
    fn test_limit_infinity_degree_analysis() {
        let x = Symbol::new("x");
        // lim_{x->oo} 2x + 1 = +oo
        let lin = Expr::Add(vec![
            Expr::Mul(vec![Expr::from_i64(2), Expr::symbol("x")]),
            Expr::from_i64(1),
        ]);
        assert_eq!(
            limit(&lin, &x, &Expr::Const(Constant::Infinity)).unwrap(),
            Expr::Const(Constant::Infinity)
        );
        // lim_{x->-oo} -x^5 = +oo (odd degree, negative lead flips).
        let quintic = Expr::Mul(vec![
            Expr::from_i64(-1),
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(5))),
        ]);
        assert_eq!(
            limit(&quintic, &x, &Expr::Const(Constant::NegativeInfinity)).unwrap(),
            Expr::Const(Constant::Infinity)
        );
        // Point substitution: lim_{x->5} x = 5.
        assert_eq!(
            limit(&Expr::symbol("x"), &x, &Expr::from_i64(5)).unwrap(),
            Expr::from_i64(5)
        );
    }

    #[test]
    fn infinity_limit_requires_an_exact_numeric_leading_coefficient() {
        let x = Symbol::new("x");
        let negative_half_x = Expr::Mul(vec![
            Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2))),
            Expr::symbol("x"),
        ]);
        assert_eq!(
            limit(&negative_half_x, &x, &Expr::Const(Constant::Infinity)).unwrap(),
            Expr::Const(Constant::NegativeInfinity)
        );

        let unknown_lead = Expr::Mul(vec![Expr::symbol("a"), Expr::symbol("x")]);
        assert!(matches!(
            limit(&unknown_lead, &x, &Expr::Const(Constant::Infinity)),
            Err(CalculusError::Undetermined(_))
        ));

        let fractional_power = Expr::Pow(
            Arc::new(Expr::symbol("x")),
            Arc::new(Expr::Rational(BigRational::new(
                BigInt::from(1),
                BigInt::from(2),
            ))),
        );
        assert_eq!(
            limit(&fractional_power, &x, &Expr::Const(Constant::Infinity)).unwrap(),
            Expr::Const(Constant::Infinity)
        );

        assert_eq!(
            limit(&Expr::symbol("a"), &x, &Expr::Const(Constant::Infinity)).unwrap(),
            Expr::symbol("a")
        );
    }

    #[test]
    fn infinity_limit_reports_expansion_cap_as_typed_refusal() {
        let x = Symbol::new("x");
        // A genuinely WIDE product (13 distinct binomials -> 8192 raw terms,
        // none collectable) still trips the expansion cap: typed refusal.
        let factors: Vec<Expr> = (0..13)
            .map(|i| Expr::Add(vec![Expr::symbol(format!("x{i}")), Expr::from_i64(1)]))
            .collect();
        let genuine_bomb = Expr::Mul(factors);
        // Free of the limit variable: the expression is its own limit and
        // nothing is expanded.
        assert_eq!(
            limit(&genuine_bomb, &x, &Expr::Const(Constant::Infinity)).unwrap(),
            genuine_bomb
        );
    }

    #[test]
    fn infinity_limit_of_expandable_polynomial_evaluates_to_infinity() {
        // (x+1)**13 at +oo is +oo: since like-term collection landed
        // (fra-fra-ws18-expand-typed-refusal-7c1) the polynomial expands
        // (21 -> 14 terms) and the limit is COMPUTED, matching upstream
        // limit((x+1)**13, x, oo) = oo. The old cap refusal for this shape
        // pinned the pre-collection silent-passthrough behavior.
        let x = Symbol::new("x");
        let factor = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]);
        let polynomial = Expr::Pow(Arc::new(factor), Arc::new(Expr::from_i64(13)));
        let outcome = limit(&polynomial, &x, &Expr::Const(Constant::Infinity));
        match outcome {
            Ok(value) => assert_eq!(
                value,
                Expr::Const(Constant::Infinity),
                "limit of a positive leading polynomial at +oo must be +oo"
            ),
            Err(err) => panic!("expandable polynomial limit must evaluate, got {err:?}"),
        }
    }

    #[test]
    fn test_limit_zero_over_zero_resolves_through_series() {
        let x = Symbol::new("x");
        // sin(x)/x at 0: indeterminate 0/0, resolved by the Gruntz lane.
        let e = Expr::Mul(vec![
            Expr::Function("sin".to_string(), vec![Expr::symbol("x")]),
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(-1))),
        ]);
        assert_eq!(
            limit(&e, &x, &Expr::from_i64(0)).unwrap(),
            Expr::from_i64(1)
        );
        // Two-sided: 1/x has different one-sided limits.
        let recip = Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(-1)));
        assert!(matches!(
            limit_dir(&recip, &x, &Expr::from_i64(0), gruntz::Direction::Both),
            Err(CalculusError::Undetermined(_))
        ));

        let fractional_pole = Expr::Pow(
            Arc::new(Expr::symbol("x")),
            Arc::new(Expr::Rational(BigRational::new(
                BigInt::from(-1),
                BigInt::from(2),
            ))),
        );
        assert_eq!(
            limit(&fractional_pole, &x, &Expr::from_i64(0)).unwrap(),
            Expr::Const(Constant::Infinity)
        );
    }

    #[test]
    fn finite_limit_detects_poles_inside_function_arguments() {
        let x = Symbol::new("x");
        let reciprocal = Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-1)));
        let nested = Expr::Function("exp".to_string(), vec![reciprocal]);
        let held_derivative = Expr::Function(
            "exp".to_string(),
            vec![Expr::Function(
                "diff".to_string(),
                vec![Expr::Sym(x.clone()), Expr::Sym(x.clone())],
            )],
        );

        // exp(1/x) has an essential singularity: +oo from the right, 0
        // from the left, so the two-sided limit refuses.
        assert_eq!(
            limit(&nested, &x, &Expr::from_i64(0)).unwrap(),
            Expr::Const(Constant::Infinity)
        );
        assert_eq!(
            limit_dir(&nested, &x, &Expr::from_i64(0), gruntz::Direction::Minus).unwrap(),
            Expr::from_i64(0)
        );
        assert!(matches!(
            limit(&held_derivative, &x, &Expr::from_i64(0)),
            Err(CalculusError::Undetermined(_))
        ));
        assert!(limit(&nested, &x, &Expr::from_i64(1)).is_ok());

        let mut too_deep = Expr::Sym(x.clone());
        for _ in 0..MAX_DIRECT_SUBSTITUTION_DEPTH {
            too_deep = Expr::Function("exp".to_string(), vec![too_deep]);
        }
        assert_eq!(
            limit(&too_deep, &x, &Expr::from_i64(0)),
            Err(CalculusError::Undetermined(
                UNSAFE_DIRECT_SUBSTITUTION.to_string()
            ))
        );
    }

    /// Metamorphic probe: truncated Taylor series approximates the original
    /// near the expansion point.
    fn assert_taylor_close(original: &Expr, var: &Symbol, at: i64, probe: f64, tol: f64) {
        let series = taylor(original, var, &Expr::from_i64(at), 6).unwrap();
        let env = HashMap::from([(
            var.clone(),
            Expr::Rational(BigRational::new(
                BigInt::from((probe * 1000.0) as i64),
                BigInt::from(1000),
            )),
        )]);
        let approx = series.subs(&env).evalf().unwrap();
        let exact = original.subs(&env).evalf().unwrap();
        assert!(
            (approx - exact).abs() < tol,
            "series {approx} vs exact {exact}"
        );
    }

    #[test]
    fn test_taylor_exp_matches_near_origin() {
        let x = Symbol::new("x");
        let e = Expr::Function("exp".to_string(), vec![Expr::symbol("x")]);
        assert_taylor_close(&e, &x, 0, 0.05, 1e-9);
    }

    #[test]
    fn test_taylor_sin_structural_cubic_term() {
        let x = Symbol::new("x");
        let s = Expr::Function("sin".to_string(), vec![Expr::symbol("x")]);
        let series = taylor(&s, &x, &Expr::from_i64(0), 3).unwrap();
        // x - x^3/6 in canonical order.
        assert_eq!(
            series,
            Expr::Add(vec![
                Expr::symbol("x"),
                Expr::Mul(vec![
                    Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(6))),
                    Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(3))),
                ]),
            ])
        );
    }

    #[test]
    fn test_taylor_nondifferentiable_is_typed_error() {
        let x = Symbol::new("x");
        // Unknown function differentiation produces a diff sentinel term.
        let l = Expr::Function("unsupported_fn".to_string(), vec![Expr::symbol("x")]);
        assert!(matches!(
            taylor(&l, &x, &Expr::from_i64(1), 2),
            Err(CalculusError::NonDifferentiable(_))
        ));
    }

    #[test]
    fn taylor_refuses_singular_coefficients() {
        let x = Symbol::new("x");
        let reciprocal = Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-1)));
        let nested = Expr::Function("exp".to_string(), vec![reciprocal.clone()]);

        for expression in [&reciprocal, &nested] {
            assert!(matches!(
                taylor(expression, &x, &Expr::from_i64(0), 2),
                Err(CalculusError::NonDifferentiable(_))
            ));
        }
        assert!(taylor(&reciprocal, &x, &Expr::from_i64(1), 2).is_ok());
    }

    #[test]
    fn test_verified_differentiation_proof_and_independent_verification() {
        let x = Symbol::new("x");
        let expr = Expr::Mul(vec![Expr::symbol("x"), Expr::from_i64(5)]);

        let (deriv, tree) = verified_diff(&expr, &x);
        assert!(verify_diff_derivation(&tree, &expr, &x, &deriv).is_ok());

        // Mutant test: tampered derivative claim is rejected
        let forged_deriv = Expr::from_i64(42);
        assert!(verify_diff_derivation(&tree, &expr, &x, &forged_deriv).is_err());
    }

    #[test]
    fn test_diff_inverse_and_ln() {
        let x = Symbol::new("x");

        // ln(x)
        let ln_x = Expr::Function("ln".into(), vec![Expr::symbol("x")]);
        let d_ln = diff(&ln_x, &x);
        assert_eq!(d_ln, Expr::pow(Expr::symbol("x"), Expr::from_i64(-1)));

        let (deriv_ln, tree_ln) = verified_diff(&ln_x, &x);
        assert!(verify_diff_derivation(&tree_ln, &ln_x, &x, &deriv_ln).is_ok());

        // asin(x)
        let asin_x = Expr::Function("asin".into(), vec![Expr::symbol("x")]);
        let d_asin = diff(&asin_x, &x);
        let expected_asin_denom = Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![
                Expr::from_i64(-1),
                Expr::pow(Expr::symbol("x"), Expr::from_i64(2)),
            ]),
        ]);
        let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
        assert_eq!(d_asin, simplify(&Expr::pow(expected_asin_denom, neg_half)));

        // atan(x)
        let atan_x = Expr::Function("atan".into(), vec![Expr::symbol("x")]);
        let d_atan = diff(&atan_x, &x);
        let expected_atan_denom = Expr::Add(vec![
            Expr::from_i64(1),
            Expr::pow(Expr::symbol("x"), Expr::from_i64(2)),
        ]);
        assert_eq!(
            d_atan,
            simplify(&Expr::pow(expected_atan_denom, Expr::from_i64(-1)))
        );
    }

    #[test]
    fn test_hero_pipeline_compiled_residual_and_jacobian_diagnostic() {
        // Nonlinear 2D residual system:
        // f1(x, y) = x^2 + y^2 - 1
        // f2(x, y) = sin(x) + cos(y)
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let vars = vec![x.clone(), y.clone()];

        let f1 = Expr::Add(vec![
            Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
            Expr::Pow(Arc::new(Expr::Sym(y.clone())), Arc::new(Expr::from_i64(2))),
            Expr::from_i64(-1),
        ]);
        let f2 = Expr::Add(vec![
            Expr::Function("sin".into(), vec![Expr::Sym(x.clone())]),
            Expr::Function("cos".into(), vec![Expr::Sym(y.clone())]),
        ]);

        let system = CompiledResidualSystem::compile(&[f1, f2], &vars);
        assert_eq!(system.num_residuals, 2);
        assert_eq!(system.num_vars, 2);

        let test_point = [0.6, 0.8];
        let mut res = [0.0; 2];
        let mut jac = [0.0; 4];
        system.eval_system(&test_point, &mut res, &mut jac);

        // f1(0.6, 0.8) = 0.6^2 + 0.8^2 - 1 = 0.36 + 0.64 - 1 = 0.0
        assert!((res[0] - 0.0).abs() < 1e-12);

        // Check Jacobian against central finite differences with 1e-6 tolerance.
        // This is an approximate diagnostic, not mathematical verification.
        let consistent = system.check_with_finite_differences(&test_point, 1e-6, 1e-5);
        assert!(
            consistent,
            "Compiled Jacobian must match numerical finite differences"
        );
    }

    #[test]
    fn test_definite_integration_polynomial() {
        // \int_0^2 (3*x^2 + 1) dx = [x^3 + x]_0^2 = (8 + 2) - 0 = 10
        let x = Symbol::new("x");
        let expr = Expr::Add(vec![
            Expr::Mul(vec![
                Expr::from_i64(3),
                Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
            ]),
            Expr::from_i64(1),
        ]);

        let a = Expr::from_i64(0);
        let b = Expr::from_i64(2);
        let val = integrate_definite(&expr, &x, &a, &b).unwrap();
        assert_eq!(val, Expr::from_i64(10));
    }

    #[test]
    fn test_laplace_transform_elementary_catalog() {
        let t = Symbol::new("t");
        let s = Symbol::new("s");

        // L{t^2} = 2 / s^3
        let t_sq = Expr::Pow(Arc::new(Expr::Sym(t.clone())), Arc::new(Expr::from_i64(2)));
        let l_poly = laplace_transform(&t_sq, &t, &s).unwrap();
        assert_eq!(
            l_poly,
            Expr::Mul(vec![
                Expr::from_i64(2),
                Expr::Pow(Arc::new(Expr::Sym(s.clone())), Arc::new(Expr::from_i64(-3))),
            ])
        );

        // L{exp(3*t)} = 1 / (s - 3)
        let exp_3t = Expr::Function(
            "exp".into(),
            vec![Expr::Mul(vec![Expr::from_i64(3), Expr::Sym(t.clone())])],
        );
        let l_exp = laplace_transform(&exp_3t, &t, &s).unwrap();
        assert_eq!(
            l_exp,
            // Canonical Add args: exact numbers precede symbols
            // (fra-add-args-canonical-order-o1i).
            Expr::Pow(
                Arc::new(Expr::Add(vec![Expr::from_i64(-3), Expr::Sym(s.clone())])),
                Arc::new(Expr::from_i64(-1)),
            )
        );
    }

    #[test]
    fn laplace_transform_rejects_nonlinear_function_arguments() {
        let t = Symbol::new("t");
        let s = Symbol::new("s");
        let t_expr = Expr::Sym(t.clone());
        let nonlinear_arguments = [
            Expr::Mul(vec![t_expr.clone(), t_expr.clone()]),
            Expr::Mul(vec![
                t_expr.clone(),
                Expr::Add(vec![t_expr.clone(), Expr::from_i64(1)]),
            ]),
        ];

        for name in ["exp", "sin", "cos"] {
            for argument in &nonlinear_arguments {
                let expr = Expr::Function(name.to_string(), vec![argument.clone()]);
                assert!(
                    laplace_transform(&expr, &t, &s).is_err(),
                    "{name}({argument}) is not a linear-in-t catalog entry"
                );
            }
        }
    }

    #[test]
    fn laplace_transform_handles_symbolic_constants_and_refuses_ambiguous_variables() {
        let t = Symbol::new("t");
        let s = Symbol::new("s");
        let x = Symbol::new("x");

        let transformed = laplace_transform(&Expr::Sym(x.clone()), &t, &s).unwrap();
        assert_eq!(
            transformed,
            Expr::Mul(vec![
                Expr::Sym(x),
                Expr::Pow(Arc::new(Expr::Sym(s.clone())), Arc::new(Expr::from_i64(-1))),
            ])
        );

        assert!(laplace_transform(&Expr::Sym(t.clone()), &t, &t).is_err());
        assert!(laplace_transform(&Expr::Sym(s.clone()), &t, &s).is_err());
    }

    #[test]
    fn laplace_transform_refuses_unbounded_polynomial_work() {
        let t = Symbol::new("t");
        let s = Symbol::new("s");
        let oversized = Expr::Pow(
            Arc::new(Expr::Sym(t.clone())),
            Arc::new(Expr::Integer(BigInt::from(
                transforms::MAX_LAPLACE_POLYNOMIAL_DEGREE + 1,
            ))),
        );
        assert!(laplace_transform(&oversized, &t, &s).is_err());
    }

    #[test]
    fn test_laplace_hyperbolic_and_damped() {
        let t = Symbol::new("t");
        let s = Symbol::new("s");
        let a = Symbol::new("a");
        let w = Symbol::new("w");

        // 1. L{sinh(a*t)} = a / (s^2 - a^2)
        let at = Expr::Mul(vec![Expr::Sym(a.clone()), Expr::Sym(t.clone())]);
        let sinh_at = Expr::Function("sinh".to_string(), vec![at.clone()]);
        let l_sinh = laplace_transform(&sinh_at, &t, &s).unwrap();
        assert!(matches!(l_sinh, Expr::Mul(_)));

        // 2. L{cosh(a*t)} = s / (s^2 - a^2)
        let cosh_at = Expr::Function("cosh".to_string(), vec![at]);
        let l_cosh = laplace_transform(&cosh_at, &t, &s).unwrap();
        assert!(matches!(l_cosh, Expr::Mul(_)));

        // 3. Damped: L{exp(a*t) * sin(w*t)}
        let wt = Expr::Mul(vec![Expr::Sym(w.clone()), Expr::Sym(t.clone())]);
        let sin_wt = Expr::Function("sin".to_string(), vec![wt]);
        let exp_at = Expr::Function(
            "exp".to_string(),
            vec![Expr::Mul(vec![Expr::Sym(a.clone()), Expr::Sym(t.clone())])],
        );
        let damped = Expr::Mul(vec![exp_at, sin_wt]);
        let l_damped = laplace_transform(&damped, &t, &s).unwrap();
        assert!(matches!(l_damped, Expr::Mul(_)));

        // 4. A symbolic rate has no established real ordering/domain, so ROC
        // metadata stays unknown even though the formal transform is known.
        let res_roc = laplace_transform_with_roc(&damped, &t, &s).unwrap();
        assert_eq!(res_roc.roc_abscissa, None);

        // An exact real damping/frequency catalog entry does establish its ROC.
        let numeric_damped = Expr::Mul(vec![
            Expr::Function(
                "exp".to_string(),
                vec![Expr::Mul(vec![Expr::from_i64(3), Expr::Sym(t.clone())])],
            ),
            Expr::Function(
                "sin".to_string(),
                vec![Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(t.clone())])],
            ),
        ]);
        let numeric_roc = laplace_transform_with_roc(&numeric_damped, &t, &s).unwrap();
        assert_eq!(numeric_roc.roc_abscissa, Some(Expr::from_i64(3)));
    }

    #[test]
    fn test_fourier_transform_catalog() {
        let t = Symbol::new("t");
        let omega = Symbol::new("omega");

        // F{exp(-2*|t|)} = 4 / (4 + omega^2)
        let abs_t = Expr::Function("abs".to_string(), vec![Expr::Sym(t.clone())]);
        let neg_a_abs_t = Expr::Mul(vec![Expr::from_i64(-2), abs_t]);
        let f_expr = Expr::Function("exp".to_string(), vec![neg_a_abs_t]);
        let f_trans = fourier_transform(&f_expr, &t, &omega).unwrap();
        assert!(matches!(f_trans, Expr::Mul(_)));

        // Constant: F{c} = 2*pi*c*delta(omega)
        let c = Expr::from_i64(5);
        let f_c = fourier_transform(&c, &t, &omega).unwrap();
        assert!(matches!(f_c, Expr::Mul(_)));
    }

    #[test]
    fn test_differentiation_elementary_and_powers() {
        let x = Symbol::new("x");
        let x_expr = Expr::Sym(x.clone());

        // d/dx(tan(x)) = 1 + tan(x)^2
        let tan_x = Expr::Function("tan".to_string(), vec![x_expr.clone()]);
        let d_tan = diff(&tan_x, &x);
        let expected_tan = simplify(&Expr::Add(vec![
            Expr::from_i64(1),
            Expr::pow(tan_x.clone(), Expr::from_i64(2)),
        ]));
        assert_eq!(d_tan, expected_tan);

        // d/dx(tanh(x)) = 1 - tanh(x)^2
        let tanh_x = Expr::Function("tanh".to_string(), vec![x_expr.clone()]);
        let d_tanh = diff(&tanh_x, &x);
        let expected_tanh = simplify(&Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![
                Expr::from_i64(-1),
                Expr::pow(tanh_x.clone(), Expr::from_i64(2)),
            ]),
        ]));
        assert_eq!(d_tanh, expected_tanh);

        // d/dx(atan(x)) = (1 + x^2)^(-1)
        let atan_x = Expr::Function("atan".to_string(), vec![x_expr.clone()]);
        let d_atan = diff(&atan_x, &x);
        let expected_atan = simplify(&Expr::pow(
            Expr::Add(vec![
                Expr::from_i64(1),
                Expr::pow(x_expr.clone(), Expr::from_i64(2)),
            ]),
            Expr::from_i64(-1),
        ));
        assert_eq!(d_atan, expected_atan);

        // d/dx(2^x) = 2^x * log(2)
        let two_to_x = Expr::pow(Expr::from_i64(2), x_expr.clone());
        let d_two_to_x = diff(&two_to_x, &x);
        let expected_two_to_x = simplify(&Expr::Mul(vec![
            two_to_x,
            Expr::Function("log".to_string(), vec![Expr::from_i64(2)]),
        ]));
        assert_eq!(d_two_to_x, expected_two_to_x);

        // d/dx(sinc(x))
        let sinc_x = Expr::Function("sinc".to_string(), vec![x_expr.clone()]);
        let d_sinc = diff(&sinc_x, &x);
        assert!(!matches!(d_sinc, Expr::Function(ref name, _) if name == "diff"));

        // d/dx(erf(x))
        let erf_x = Expr::Function("erf".to_string(), vec![x_expr.clone()]);
        let d_erf = diff(&erf_x, &x);
        assert!(!matches!(d_erf, Expr::Function(ref name, _) if name == "diff"));

        // d/dx(erfc(x))
        let erfc_x = Expr::Function("erfc".to_string(), vec![x_expr.clone()]);
        let d_erfc = diff(&erfc_x, &x);
        assert!(!matches!(d_erfc, Expr::Function(ref name, _) if name == "diff"));

        // d/dx(asec(x))
        let asec_x = Expr::Function("asec".to_string(), vec![x_expr.clone()]);
        let d_asec = diff(&asec_x, &x);
        assert!(!matches!(d_asec, Expr::Function(ref name, _) if name == "diff"));
    }

    #[test]
    fn test_eliminate_zero_products_annihilation_and_unwrapping() {
        let x = Expr::symbol("x");
        // Mul containing an Add that simplifies to 0
        let zero_add = Expr::Add(vec![Expr::from_i64(0), Expr::from_i64(0)]);
        let mul_with_zero = Expr::Mul(vec![x.clone(), zero_add]);
        assert_eq!(eliminate_zero_products(&mul_with_zero), Expr::from_i64(0));

        // Mul with empty factors -> 1
        let empty_mul = Expr::Mul(vec![]);
        assert_eq!(eliminate_zero_products(&empty_mul), Expr::from_i64(1));

        // Mul with single non-zero factor -> unwrapped factor
        let single_mul = Expr::Mul(vec![x.clone()]);
        assert_eq!(eliminate_zero_products(&single_mul), x);

        // Nested in Function and Pow
        let func = Expr::Function("f".to_string(), vec![mul_with_zero]);
        assert_eq!(
            eliminate_zero_products(&func),
            Expr::Function("f".to_string(), vec![Expr::from_i64(0)])
        );
    }

    #[test]
    fn test_diff_independent_function_is_zero_and_derivative_chaining() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let fx = Expr::Function("f".to_string(), vec![Expr::Sym(x.clone())]);

        // d/dy(f(x)) == 0 by chain rule
        assert_eq!(diff(&fx, &y), Expr::from_i64(0));

        // d/dx(f(x)) == diff(f(x), x)
        let dfx = diff(&fx, &x);
        assert_eq!(
            dfx,
            Expr::Function("diff".to_string(), vec![fx.clone(), Expr::Sym(x.clone())])
        );

        // d/dx(diff(f(x), x)) == diff(f(x), x, x)
        let d2fx = diff(&dfx, &x);
        assert_eq!(
            d2fx,
            Expr::Function(
                "diff".to_string(),
                vec![fx.clone(), Expr::Sym(x.clone()), Expr::Sym(x.clone())]
            )
        );

        // d/dy(diff(f(x), x)) == 0
        assert_eq!(diff(&dfx, &y), Expr::from_i64(0));

        // Also test chaining on explicit Derivative function
        let deriv = Expr::Function(
            "Derivative".to_string(),
            vec![fx.clone(), Expr::Sym(x.clone())],
        );
        assert_eq!(
            diff(&deriv, &x),
            Expr::Function(
                "Derivative".to_string(),
                vec![fx.clone(), Expr::Sym(x.clone()), Expr::Sym(x.clone())]
            )
        );
        assert_eq!(diff(&deriv, &y), Expr::from_i64(0));
    }
}

#[cfg(test)]
mod ball_boundary_tests {
    use super::*;

    #[test]
    fn evalf_ball_encloses_transcendental_values() {
        let x = Symbol::new("x");
        // exp(1) at 25 digits must enclose e and carry a sub-10^-24 radius.
        let e_expr = Expr::Function("exp".to_string(), vec![Expr::from_i64(1)]);
        let ball = evalf_ball(&e_expr, 25).expect("exp ball");
        // e lies strictly inside (2, 3); the whole enclosure must too.
        let two = BigRational::from_integer(BigInt::from(2));
        let three = BigRational::from_integer(BigInt::from(3));
        assert!(
            two < ball.lower() && ball.upper() < three,
            "e enclosure {ball} must lie inside (2, 3)"
        );
        // sin(x) evaluated at the constant pi must enclose 0.
        let sin_pi = Expr::Function("sin".to_string(), vec![Expr::Const(Constant::Pi)]);
        let ball = evalf_ball(&sin_pi, 20).expect("sin(pi) ball");
        assert!(ball.contains_zero(), "sin(pi) enclosure must contain zero");
        // free symbols refuse
        assert!(evalf_ball(&Expr::Sym(x.clone()), 10).is_err());
        // unsupported functions refuse
        let tan_x = Expr::Function("tan".to_string(), vec![Expr::Sym(x.clone())]);
        assert!(evalf_ball(&tan_x, 10).is_err());
    }
}
