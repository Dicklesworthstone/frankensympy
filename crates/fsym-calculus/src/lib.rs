//! # fsym-calculus
//!
//! Symbolic differentiation, integration, limits, and series expansion.

pub mod box_optimization;
pub mod compile;
pub mod delayed_laplace;
pub mod elementary_rational_definite;
pub mod exponential_sum;
pub mod forward_mellin;
pub mod fourier;
pub mod geometric_sum;
pub mod gruntz;
pub mod hermite;
pub mod integrate;
pub mod parametric_cover;
pub mod program_artifact;
pub mod proof;
pub mod quadrature;
pub mod rational_definite;
pub mod rational_laplace;
pub mod rational_limits;
pub mod rational_mellin;
pub mod rational_product;
pub mod rational_sum;
pub mod root_isolation;
pub mod series;
pub mod summation;

pub mod sparse_hessian;
pub mod sparse_jacobian;
pub mod transforms;
pub mod verified;

pub use compile::*;
pub use proof::*;
pub use sparse_hessian::*;
pub use sparse_jacobian::*;
pub use transforms::*;
pub use verified::*;

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
    diff_unsimplified_with_coth(expr, var, false)
}

/// The numerical derivative lane keeps tanh' and coth' away from subtraction
/// of two rounded ones, keeps the entire sinc jets finite at their removable
/// singularity, retains guarded absolute-value derivatives on nonzero real
/// branches, and avoids artificial poles in literal zero/one powers.
/// Public symbolic derivatives retain their existing forms.
pub(crate) fn numeric_diff_unsimplified(expr: &Expr, var: &Symbol) -> Expr {
    diff_unsimplified_with_coth(expr, var, true)
}

fn diff_unsimplified_with_coth(expr: &Expr, var: &Symbol, stable_coth: bool) -> Expr {
    // Every recursive chain/product derivative uses the same representation;
    // this also covers coth inside an outer function or a mixed derivative.
    let diff = |child: &Expr, variable: &Symbol| diff_with_coth(child, variable, stable_coth);
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
            // Numeric programs retain their original source separately. A
            // literal zero or one exponent therefore needs no artificial
            // 0*u^-1 or 1*u^0 factor in its derivative. Original base-domain
            // obligations still execute before any derivative is published.
            if stable_coth {
                match exp.as_ref() {
                    Expr::Integer(n) if n.is_zero() => return Expr::from_i64(0),
                    Expr::Rational(r) if r.numer().is_zero() => return Expr::from_i64(0),
                    Expr::Integer(n) if n.is_one() => return diff(base, var),
                    Expr::Rational(r) if r.numer() == r.denom() => return diff(base, var),
                    _ => {}
                }
            }
            let du = diff(base, var);
            let dv = diff(exp, var);
            if du.is_zero() && dv.is_zero() {
                Expr::from_i64(0)
            } else if dv.is_zero() {
                // u(x)^c where c does not depend on var: c * u^(c - 1) * u'
                let exp_minus_1 = match exp.as_ref() {
                    Expr::Integer(n) => Expr::Integer(n - BigInt::from(1)),
                    Expr::Rational(r) => {
                        fsym_core::elementary::rational_expr(r - BigRational::one())
                    }
                    _ => Expr::Add(vec![exp.as_ref().clone(), Expr::from_i64(-1)]),
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
                if stable_coth {
                    // 1 - tanh(u)^2 rounds to zero already near |u| = 20.
                    // The stable sech opcode retains representable tails and
                    // its derivative also preserves second-order curvature.
                    return Expr::Mul(vec![
                        Expr::pow(
                            Expr::Function("sech".to_string(), vec![u.clone()]),
                            Expr::from_i64(2),
                        ),
                        du,
                    ]);
                }
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
            } else if name == "sqrt" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                // The explicit principal-square-root node follows the same
                // formal rule as u^(1/2). Its inverse half power retains the
                // singularity at zero; real evaluators must still admit the
                // original source and every derivative at the requested point.
                Expr::Mul(vec![
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                    Expr::pow(
                        u.clone(),
                        Expr::Rational(BigRational::new((-1).into(), 2.into())),
                    ),
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
                let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                // Separate principal square roots retain the acosh branch.
                // Replacing their product by sqrt(u^2-1) reverses the sign
                // on parts of the complex domain, including Re(u) < 0.
                Expr::Mul(vec![
                    Expr::pow(
                        Expr::Add(vec![u.clone(), Expr::from_i64(-1)]),
                        neg_half.clone(),
                    ),
                    Expr::pow(Expr::Add(vec![u.clone(), Expr::from_i64(1)]), neg_half),
                    du,
                ])
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
                if stable_coth {
                    return Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(
                            Expr::Function("csch".to_string(), vec![u.clone()]),
                            Expr::from_i64(2),
                        ),
                        du,
                    ]);
                }
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
            } else if stable_coth
                && matches!(name.as_str(), "abs" | "Abs" | "_fsym_abs_d1")
                && args.len() == 1
            {
                // These private nodes denote the real derivatives only on
                // u != 0. Their evaluators reject zero rather than assigning
                // a subgradient; original smooth-source evaluation separately
                // retains that obligation when zero arithmetic removes a jet.
                let next = if name == "_fsym_abs_d1" {
                    "_fsym_abs_d2"
                } else {
                    "_fsym_abs_d1"
                };
                let u = &args[0];
                Expr::Mul(vec![
                    Expr::Function(next.to_string(), vec![u.clone()]),
                    diff(u, var),
                ])
            } else if name == "sinc" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                if stable_coth {
                    return Expr::Mul(vec![
                        Expr::Function("_fsym_sinc_d1".to_string(), vec![u.clone()]),
                        du,
                    ]);
                }
                let cos_u = Expr::Function("cos".to_string(), vec![u.clone()]);
                let sin_u = Expr::Function("sin".to_string(), vec![u.clone()]);
                let inv_u = Expr::pow(u.clone(), Expr::from_i64(-1));
                let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                let term1 = Expr::Mul(vec![cos_u, inv_u]);
                let term2 = Expr::Mul(vec![Expr::from_i64(-1), sin_u, inv_u_sq]);
                Expr::Mul(vec![Expr::Add(vec![term1, term2]), du])
            } else if name == "_fsym_sinc_d1" && args.len() == 1 {
                // This private node denotes the exact entire derivative of
                // sinc, not the quotient formula with an artificial hole.
                // Its separate numerical evaluator handles the second jet.
                let u = &args[0];
                Expr::Mul(vec![
                    Expr::Function("_fsym_sinc_d2".to_string(), vec![u.clone()]),
                    diff(u, var),
                ])
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
            } else if name == "Piecewise"
                && !args.is_empty()
                && args.iter().all(
                    |a| matches!(a, Expr::Function(t, pair) if t == "Tuple" && pair.len() == 2),
                )
            {
                // d/dx Piecewise((e_i, c_i), ...) = Piecewise((de_i/dx, c_i), ...)
                // (upstream Piecewise._eval_derivative; conditions are kept).
                let branches = args
                    .iter()
                    .map(|a| match a {
                        Expr::Function(_, pair) => Expr::Function(
                            "Tuple".to_string(),
                            vec![diff(&pair[0], var), pair[1].clone()],
                        ),
                        other => other.clone(),
                    })
                    .collect();
                Expr::Function("Piecewise".to_string(), branches)
            } else if matches!(
                name.as_str(),
                "legendre" | "chebyshevt" | "chebyshevu" | "hermite" | "laguerre"
            ) && args.len() == 2
                && is_free_of(&args[0], var)
            {
                // Upstream fdiff rules of the classical orthogonal families.
                let n = args[0].clone();
                let u = args[1].clone();
                let du = diff(&u, var);
                let one = Expr::from_i64(1);
                let f = |nm: &str, k: Expr| Expr::Function(nm.to_string(), vec![k, u.clone()]);
                let inner = match name.as_str() {
                    "legendre" => {
                        n.clone()
                            * (u.clone() * f("legendre", n.clone())
                                - f("legendre", n.clone() - one.clone()))
                            * Expr::pow(u.clone() * u.clone() - one.clone(), Expr::from_i64(-1))
                    }
                    "chebyshevt" => n.clone() * f("chebyshevu", n.clone() - one.clone()),
                    "chebyshevu" => {
                        ((n.clone() + one.clone()) * f("chebyshevt", n.clone() + one.clone())
                            - u.clone() * f("chebyshevu", n.clone()))
                            * Expr::pow(u.clone() * u.clone() - one.clone(), Expr::from_i64(-1))
                    }
                    "hermite" => {
                        Expr::from_i64(2) * n.clone() * f("hermite", n.clone() - one.clone())
                    }
                    _ => -Expr::Function(
                        "assoc_laguerre".to_string(),
                        vec![n.clone() - one.clone(), one.clone(), u.clone()],
                    ),
                };
                inner * du
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

/// The empty, one-point and two-endpoint integral limits have different
/// binding semantics. Sums require both endpoints.
struct BinderLimit<'a> {
    coordinate: &'a Symbol,
    endpoints: &'a [Expr],
}

fn binder_limit<'a>(name: &str, limit: &'a Expr) -> Option<BinderLimit<'a>> {
    match limit {
        Expr::Sym(coordinate) if name == "Integral" => Some(BinderLimit {
            coordinate,
            endpoints: &[],
        }),
        Expr::Function(tag, items) if tag == "Tuple" => {
            let [Expr::Sym(coordinate), endpoints @ ..] = items.as_slice() else {
                return None;
            };
            if endpoints.len() == 2 || (name == "Integral" && endpoints.len() <= 1) {
                Some(BinderLimit {
                    coordinate,
                    endpoints,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The inner expression preceding one flat limit. Limits are stored from
/// innermost to outermost; a prefix therefore retains its complete scope.
fn binder_prefix(name: &str, args: &[Expr]) -> Expr {
    if let [body] = args {
        body.clone()
    } else {
        Expr::Function(name.into(), args.to_vec())
    }
}

fn bind_derivative(name: &str, derivative: Expr, limit: &Expr) -> Expr {
    if derivative.is_zero() {
        return derivative;
    }
    if let Expr::Function(inner_name, inner_args) = &derivative
        && inner_name == name
        && inner_args.len() >= 2
        && inner_args[1..]
            .iter()
            .all(|limit| binder_limit(name, limit).is_some())
    {
        // Keep fixed-limit chains flat instead of introducing one recursive
        // AST layer per limit. This is the same order as the public Integral
        // constructor's flattening rule.
        let mut args = inner_args.clone();
        args.push(limit.clone());
        return Expr::Function(name.into(), args);
    }
    Expr::Function(name.into(), vec![derivative, limit.clone()])
}

/// An indefinite coordinate is free, so ordinary capture-avoiding substitution
/// does not alpha-rename it. Inserting a parameter containing that coordinate
/// under its integral would change a parameter of the primitive into its
/// integration variable. Such a boundary needs a separate evaluated-primitive
/// representation; preserve the formal derivative until that lane exists.
fn boundary_captures_primitive(inner: &Expr, coordinate: &Symbol, point: &Expr) -> bool {
    if !inner.free_symbols().contains(coordinate) {
        return false;
    }
    let point_symbols = point.free_symbols();
    if point_symbols.is_empty() {
        return false;
    }
    let mut pending = vec![inner];
    while let Some(node) = pending.pop() {
        match node {
            Expr::Function(name, args) => {
                if name == "Integral"
                    && args.len() >= 2
                    && node.free_symbols().contains(coordinate)
                    && args[1..].iter().any(|limit| {
                        binder_limit(name, limit).is_some_and(|limit| {
                            limit.endpoints.is_empty()
                                && limit.coordinate != coordinate
                                && point_symbols.contains(limit.coordinate)
                        })
                    })
                {
                    return true;
                }
                pending.extend(args);
            }
            Expr::Add(args) | Expr::Mul(args) => pending.extend(args),
            Expr::Pow(base, exponent) => {
                pending.push(base);
                pending.push(exponent);
            }
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
        }
    }
    false
}

fn unevaluated_diff(expr: &Expr, var: &Symbol) -> Expr {
    Expr::Function(
        "diff".to_string(),
        vec![expr.clone(), Expr::Sym(var.clone())],
    )
}

/// Formal fundamental-theorem and Leibniz rules, composed in lexical limit
/// order. Each coordinate binds only the inner expression, never its own
/// endpoints. A later indefinite limit can make an earlier bound symbol free
/// again. Boundary insertion uses capture-avoiding native substitution.
///
/// Moving discrete sum bounds, malformed limits and boundary substitutions
/// capturing a free indefinite coordinate stay unevaluated. These formal rules
/// do not establish convergence or interchange conditions, and do not add a
/// differentiation proof-kernel rule for binders.
fn diff_binder(expr: &Expr, name: &str, args: &[Expr], var: &Symbol) -> Expr {
    let Some((body, raw_limits)) = args.split_first() else {
        return unevaluated_diff(expr, var);
    };
    if raw_limits.is_empty() {
        return unevaluated_diff(expr, var);
    }
    let Some(limits) = raw_limits
        .iter()
        .map(|limit| binder_limit(name, limit))
        .collect::<Option<Vec<_>>>()
    else {
        return unevaluated_diff(expr, var);
    };

    // The last limit in this coordinate replaces the derivative of every
    // inner scope by its own fundamental-theorem/boundary contribution.
    // Avoid differentiating work which that binding would discard. Parsing
    // every original limit above still precedes this shortcut.
    let shadow = limits.iter().rposition(|limit| limit.coordinate == var);
    let mut derivative = if shadow.is_some() {
        Expr::from_i64(0)
    } else {
        diff(body, var)
    };
    for (index, limit) in limits.iter().enumerate().skip(shadow.unwrap_or(0)) {
        if limit.endpoints.is_empty() {
            derivative = if limit.coordinate == var {
                binder_prefix(name, &args[..index + 1])
            } else {
                bind_derivative(name, derivative, &raw_limits[index])
            };
            continue;
        }

        let mut boundary_terms = Vec::new();
        let mut moving_sum = false;
        for (endpoint_index, endpoint) in limit.endpoints.iter().enumerate() {
            let endpoint_derivative = diff(endpoint, var);
            if endpoint_derivative.is_zero() {
                // A fixed singular endpoint must not introduce 0*f(endpoint).
                continue;
            }
            if name == "Sum" {
                moving_sum = true;
                break;
            }
            let inner = binder_prefix(name, &args[..index + 1]);
            if boundary_captures_primitive(&inner, limit.coordinate, endpoint) {
                return unevaluated_diff(expr, var);
            }
            let boundary = inner.subs(&HashMap::from([(
                limit.coordinate.clone(),
                endpoint.clone(),
            )])) * endpoint_derivative;
            boundary_terms.push(if limit.endpoints.len() == 2 && endpoint_index == 0 {
                Expr::from_i64(-1) * boundary
            } else {
                boundary
            });
        }
        if moving_sum {
            derivative = unevaluated_diff(&binder_prefix(name, &args[..index + 2]), var);
            continue;
        }
        let partial = if limit.coordinate == var {
            Expr::from_i64(0)
        } else {
            bind_derivative(name, derivative, &raw_limits[index])
        };
        if !partial.is_zero() {
            boundary_terms.push(partial);
        }
        derivative = match boundary_terms.len() {
            0 => Expr::from_i64(0),
            1 => boundary_terms.pop().expect("one boundary or partial term"),
            _ => Expr::Add(boundary_terms),
        };
    }
    derivative
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
    diff_with_coth(expr, var, false)
}

pub(crate) fn numeric_diff(expr: &Expr, var: &Symbol) -> Expr {
    diff_with_coth(expr, var, true)
}

fn diff_with_coth(expr: &Expr, var: &Symbol, stable_coth: bool) -> Expr {
    let unsimplified = diff_unsimplified_with_coth(expr, var, stable_coth);
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
    integrate_in_source_scope(expr, var, integrate::conditional_source_admitted(expr))
}

fn integrate_in_source_scope(
    expr: &Expr,
    var: &Symbol,
    conditional_source: bool,
) -> Result<Expr, CalculusError> {
    if let Some(anti) = integrate::antiderivative_in_source_scope(expr, var, conditional_source) {
        return Ok(anti);
    }
    integrate_basic(expr, var, conditional_source)
}

/// The original term-table integrator, kept as a fallback lane.
fn integrate_basic(
    expr: &Expr,
    var: &Symbol,
    conditional_source: bool,
) -> Result<Expr, CalculusError> {
    match expr {
        Expr::Add(terms) => {
            let mut parts = Vec::new();
            for t in terms {
                parts.push(integrate_in_source_scope(t, var, conditional_source)?);
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
            let anti = integrate_in_source_scope(&inner, var, conditional_source)?;
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
/// Rational functions at rational points or infinity first use independently
/// checked local-order factorizations. Other finite points try direct
/// substitution when the expression is free of poles and discontinuous
/// functions; remaining cases use the Gruntz algorithm in [`gruntz`]. Undecidable
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
    series::validate_modulus_inputs(expr, var, to, dir)
        .map_err(|error| CalculusError::Undetermined(error.to_string()))?;
    series::validate_inverse_inputs(expr, var, to, dir)
        .map_err(|error| CalculusError::Undetermined(error.to_string()))?;
    if carries_diff_sentinel(expr) {
        return Err(CalculusError::Undetermined(expr.to_string()));
    }
    if let Some(result) = rational_limits::try_limit(expr, var, to, dir) {
        return result.map_err(|error| CalculusError::Undetermined(error.to_string()));
    }
    if let Some(result) = series::finite_inverse_limit(expr, var, to, dir) {
        return result.map_err(|error| CalculusError::Undetermined(error.to_string()));
    }
    if is_free_of(expr, var) {
        return Ok(expr.clone());
    }
    let infinite_point = matches!(
        to,
        Expr::Const(Constant::Infinity) | Expr::Const(Constant::NegativeInfinity)
    );
    if !infinite_point
        && continuous_shape(expr, var, to)
        && let Some(value) = direct_substitution(expr, var, to)
    {
        return Ok(value);
    }
    gruntz::limit(expr, var, to, dir).map_err(|e| CalculusError::Undetermined(e.to_string()))
}

/// Direct substitution requires continuity along the original source. A finite
/// principal value on a branch cut alone does not establish that continuity.
fn continuous_shape(expr: &Expr, var: &Symbol, to: &Expr) -> bool {
    if is_free_of(expr, var) {
        return true;
    }
    let branch_is_continuous = |base: &Expr| {
        is_free_of(base, var)
            || series::avoids_principal_cut(&base.subs(&HashMap::from([(var.clone(), to.clone())])))
    };
    match expr {
        Expr::Function(name, args) => {
            !matches!(
                name.as_str(),
                "floor" | "ceiling" | "sign" | "frac" | "Heaviside" | "Piecewise" | "Mod"
            ) && (!matches!(name.as_str(), "log" | "ln" | "sqrt")
                || (args.len() == 1 && branch_is_continuous(&args[0])))
                && (!matches!(
                    name.as_str(),
                    "asin" | "acos" | "atan" | "acot" | "asinh" | "acosh" | "atanh" | "acoth"
                ) || (args.len() == 1
                    && series::inverse_is_continuous(
                        name,
                        &args[0].subs(&HashMap::from([(var.clone(), to.clone())])),
                    )))
                && args.iter().all(|arg| continuous_shape(arg, var, to))
        }
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().all(|arg| continuous_shape(arg, var, to)),
        Expr::Pow(b, e) => {
            ((is_free_of(e, var) && fsym_core::assume::is_integer(e)) || branch_is_continuous(b))
                && continuous_shape(b, var, to)
                && continuous_shape(e, var, to)
        }
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
/// Bounded rational functions use independently checked coefficient
/// convolutions and can supply up to 256 terms beyond their valuation.
/// Formal truncation/exactness is separate from a general convergence claim.
pub fn series_expansion(
    expr: &Expr,
    var: &Symbol,
    x0: &Expr,
    n: i64,
) -> Result<SeriesTerms, CalculusError> {
    series_expansion_dir(expr, var, x0, n, gruntz::Direction::Plus)
}

/// One-sided generalized series, with the finite local coordinate selected by
/// `direction`: x-x0 from the right, x0-x from the left. At either infinity
/// the inward coordinates 1/x or -1/x apply independently of the supplied side.
/// Logarithmic coefficients use that same positive local coordinate; fractional
/// powers retain their principal branch factors. Two-sided series are refused.
pub fn series_expansion_dir(
    expr: &Expr,
    var: &Symbol,
    x0: &Expr,
    n: i64,
    direction: gruntz::Direction,
) -> Result<SeriesTerms, CalculusError> {
    if matches!(direction, gruntz::Direction::Both) {
        return Err(CalculusError::NonDifferentiable(
            "series direction must be one-sided".to_string(),
        ));
    }
    let from_left = matches!(direction, gruntz::Direction::Minus)
        && !matches!(
            x0,
            Expr::Const(Constant::Infinity | Constant::NegativeInfinity)
        );
    if unsafe_direct_substitution(expr) || unsafe_direct_substitution(x0) {
        return Err(CalculusError::NonDifferentiable(
            UNSAFE_DIRECT_SUBSTITUTION.to_string(),
        ));
    }
    series::validate_modulus_inputs(expr, var, x0, direction)
        .map_err(|error| CalculusError::NonDifferentiable(error.to_string()))?;
    series::validate_inverse_inputs(expr, var, x0, direction)
        .map_err(|error| CalculusError::NonDifferentiable(error.to_string()))?;
    if let Some(result) = rational_limits::try_series(expr, var, x0, n, from_left) {
        return result.map_err(|error| CalculusError::NonDifferentiable(error.to_string()));
    }
    let t = series::fresh_symbol("_fsym_series_t", &[expr, x0], &[var])
        .map_err(|error| CalculusError::NonDifferentiable(error.to_string()))?;
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
        _ if from_left => (
            x0.clone() - te.clone(),
            Expr::Function("log".into(), vec![x0.clone() - xe.clone()]),
        ),
        _ if x0.is_zero() => (te.clone(), Expr::Function("log".into(), vec![xe.clone()])),
        _ => (
            x0.clone() + te.clone(),
            Expr::Function("log".into(), vec![xe.clone() - x0.clone()]),
        ),
    };
    let local = gruntz::substitute_limit_source(expr, var, &replacement);
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

/// Evaluate a point-substituted Taylor source from its leaves upwards. Each
/// visited child must remain defined before its parent may cancel it, multiply
/// it by zero, or raise it to the zeroth power. This is a point-domain check,
/// not a convergence certificate or a restriction on formal Laurent series.
fn finite_taylor_point_value(expr: &Expr) -> Result<Expr, CalculusError> {
    fn refused() -> CalculusError {
        CalculusError::NonDifferentiable(
            "Taylor coefficients require defined finite source values at the center".into(),
        )
    }

    fn visit(expr: &Expr, depth: usize, remaining: &mut usize) -> Result<Expr, CalculusError> {
        if depth > MAX_DIRECT_SUBSTITUTION_DEPTH || *remaining == 0 {
            return Err(refused());
        }
        *remaining -= 1;
        let child_depth = depth + 1;
        // Literal real infinities can be valid inputs to exact finite
        // elementary values. Fold only that immediate, fully known parent;
        // an infinity produced by log(0) or 0**-1 must still be refused while
        // visiting the original child, before an outer function can hide it.
        let literal_infinity = |value: &Expr| {
            matches!(
                value,
                Expr::Const(Constant::Infinity | Constant::NegativeInfinity)
            )
        };
        let literal_children = match expr {
            Expr::Function(_, args) if args.len() == 1 && literal_infinity(&args[0]) => {
                Some(1usize)
            }
            Expr::Pow(base, exponent)
                if literal_infinity(base)
                    && matches!(exponent.as_ref(), Expr::Integer(_) | Expr::Rational(_)) =>
            {
                Some(2usize)
            }
            _ => None,
        };
        if let Some(count) = literal_children {
            if child_depth > MAX_DIRECT_SUBSTITUTION_DEPTH || *remaining < count {
                return Err(refused());
            }
            *remaining -= count;
            let value = match expr {
                Expr::Function(name, args) => {
                    fsym_core::elementary::eval_function(name, args).ok_or_else(refused)?
                }
                Expr::Pow(base, exponent) => fsym_core::elementary::eval_pow(
                    base.as_ref().clone(),
                    exponent.as_ref().clone(),
                ),
                _ => unreachable!("validated literal infinity composition"),
            };
            if has_infinity(&value) {
                return Err(refused());
            }
            // Charge and validate the finite result too, under the same
            // depth/node allowance. No whole-source simplification occurs.
            return visit(&value, depth, remaining);
        }
        let children = |args: &[Expr], remaining: &mut usize| {
            args.iter()
                .map(|arg| visit(arg, child_depth, remaining))
                .collect::<Result<Vec<_>, _>>()
        };
        let rebuilt = match expr {
            Expr::Add(args) => Expr::Add(children(args, remaining)?),
            Expr::Mul(args) => Expr::Mul(children(args, remaining)?),
            Expr::Pow(base, exponent) => {
                let base = visit(base, child_depth, remaining)?;
                let exponent = visit(exponent, child_depth, remaining)?;
                // The rational Taylor lane already rejects held 0**0. Keep
                // that pointwise policy when a transcendental composition
                // selects this general coefficient lane instead.
                if base.is_zero()
                    && numeric_value(&exponent).is_some_and(|power| power <= BigRational::zero())
                {
                    return Err(refused());
                }
                Expr::Pow(Arc::new(base), Arc::new(exponent))
            }
            Expr::Function(name, args) if name == "Piecewise" => {
                let mut kept = Vec::new();
                for branch in args {
                    if child_depth > MAX_DIRECT_SUBSTITUTION_DEPTH || *remaining == 0 {
                        return Err(refused());
                    }
                    *remaining -= 1;
                    let Expr::Function(tuple, pair) = branch else {
                        return Err(refused());
                    };
                    if tuple != "Tuple" || pair.len() != 2 {
                        return Err(refused());
                    }
                    let condition = visit(&pair[1], child_depth + 1, remaining)?;
                    let truth = match &condition {
                        Expr::Sym(symbol) if symbol.is_plain() && symbol.name == "True" => {
                            Some(true)
                        }
                        Expr::Sym(symbol) if symbol.is_plain() && symbol.name == "False" => {
                            Some(false)
                        }
                        _ => None,
                    };
                    if truth == Some(false) {
                        continue;
                    }
                    let value = visit(&pair[0], child_depth + 1, remaining)?;
                    if truth == Some(true) && kept.is_empty() {
                        return Ok(value);
                    }
                    kept.push(Expr::Function("Tuple".into(), vec![value, condition]));
                    if truth == Some(true) {
                        break;
                    }
                }
                if kept.is_empty() {
                    return Err(refused());
                }
                Expr::Function(name.clone(), kept)
            }
            Expr::Function(name, args) => Expr::Function(name.clone(), children(args, remaining)?),
            _ => expr.clone(),
        };
        if has_infinity(&rebuilt) {
            return Err(refused());
        }
        // Match exact substitution's elementary folds, but only after the
        // source children have passed their own point-domain checks.
        let evaluated = match rebuilt {
            Expr::Function(name, args) => fsym_core::elementary::eval_function(&name, &args)
                .unwrap_or(Expr::Function(name, args)),
            Expr::Pow(base, exponent) => fsym_core::elementary::eval_pow(
                Arc::unwrap_or_clone(base),
                Arc::unwrap_or_clone(exponent),
            ),
            value => value,
        };
        let value = fsym_simplify::try_simplify(&evaluated)
            .map_err(|error| CalculusError::NonDifferentiable(error.to_string()))?;
        if unsafe_direct_substitution(&value) || has_infinity(&value) {
            return Err(refused());
        }
        Ok(value)
    }

    let mut remaining = MAX_DIRECT_SUBSTITUTION_NODES;
    visit(expr, 1, &mut remaining)
}

/// Bound the expanded point tree before cloning the center into each source
/// occurrence. Counting bound occurrences too is conservative; the actual
/// substitution below retains the existing capture-aware binder semantics.
fn taylor_point_substitution(expr: &Expr, var: &Symbol, at: &Expr) -> Result<Expr, CalculusError> {
    fn bounds(
        expr: &Expr,
        replacement: Option<(&Symbol, usize, usize)>,
    ) -> Result<(usize, usize), CalculusError> {
        let refused = || {
            CalculusError::NonDifferentiable(
                "Taylor point substitution exceeds its expression bounds".into(),
            )
        };
        let mut pending = vec![(expr, 1usize)];
        let mut nodes = 0usize;
        let mut maximum_depth = 0usize;
        while let Some((node, depth)) = pending.pop() {
            let (width, height) = match (node, replacement) {
                (Expr::Sym(symbol), Some((var, width, height))) if symbol == var => (width, height),
                _ => (1, 1),
            };
            nodes = nodes.checked_add(width).ok_or_else(refused)?;
            let expanded_depth = depth.checked_add(height - 1).ok_or_else(refused)?;
            maximum_depth = maximum_depth.max(expanded_depth);
            if nodes > MAX_DIRECT_SUBSTITUTION_NODES
                || maximum_depth > MAX_DIRECT_SUBSTITUTION_DEPTH
            {
                return Err(refused());
            }
            let children: &[Expr] = match node {
                Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => args,
                Expr::Pow(base, exponent) => {
                    pending.try_reserve(2).map_err(|_| refused())?;
                    pending.push((base, depth + 1));
                    pending.push((exponent, depth + 1));
                    continue;
                }
                _ => continue,
            };
            if children.len() > MAX_DIRECT_SUBSTITUTION_NODES {
                return Err(refused());
            }
            pending.try_reserve(children.len()).map_err(|_| refused())?;
            pending.extend(children.iter().map(|child| (child, depth + 1)));
        }
        Ok((nodes, maximum_depth))
    }

    let (width, height) = bounds(at, None)?;
    bounds(expr, Some((var, width, height)))?;
    Ok(gruntz::substitute_limit_source(expr, var, at))
}

/// Taylor polynomial of `expr` around `var = at` through degree `order`.
///
/// Coefficients are exact where the derivatives evaluate exactly; a
/// derivative that hits [`diff`]'s non-differentiable sentinel aborts with
/// [`CalculusError::NonDifferentiable`].
/// Original source children must be defined at a finite center independent
/// of `var`; cancellation does not extend their pointwise domain. Symbolic
/// parameter coefficients retain their unevaluated domain conditions.
/// Rational functions at regular rational centers use a checked coefficient
/// recurrence for up to 256 coefficients; other expressions retain the
/// bounded derivative route.
pub fn taylor(expr: &Expr, var: &Symbol, at: &Expr, order: usize) -> Result<Expr, CalculusError> {
    if let Some(result) = rational_limits::try_taylor(expr, var, at, order) {
        return result.map_err(|error| CalculusError::NonDifferentiable(error.to_string()));
    }
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
    if !is_free_of(at, var) {
        return Err(CalculusError::NonDifferentiable(
            "the Taylor center must be independent of the expansion coordinate".into(),
        ));
    }
    let at = finite_taylor_point_value(at)?;
    // Ordinary substitution and simplification fold x*x**-1, exp(log(x)),
    // and zero-weighted terms. Retain the original scalar tree until its
    // point-domain obligations have been checked, including at order zero.
    let source_at_center = taylor_point_substitution(expr, var, &at)?;
    finite_taylor_point_value(&source_at_center)?;
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
        if k > 0 && carries_diff_sentinel(&deriv) {
            return Err(CalculusError::NonDifferentiable(expr.to_string()));
        }
        let substituted = taylor_point_substitution(&deriv, var, &at)?;
        let value = finite_taylor_point_value(&substituted)?;
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
            Ok(Expr::Function(name, branches)) if name == "Piecewise" && branches.len() == 2
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
