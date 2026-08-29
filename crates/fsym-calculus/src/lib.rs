//! # fsym-calculus
//!
//! Symbolic differentiation, integration, limits, and series expansion.

pub mod compile;
pub mod proof;
pub mod transforms;

pub use compile::*;
pub use proof::*;
pub use transforms::*;

use fsym_budget::Unbounded;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_simplify::{expand_with, simplify};
use num_traits::{Signed, Zero};
use std::collections::{BTreeMap, HashMap};
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
        Expr::Function(name, args) => {
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
            } else if name == "log" && args.len() == 1 {
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
            } else {
                Expr::Function(
                    "diff".to_string(),
                    vec![expr.clone(), Expr::Sym(var.clone())],
                )
            }
        }
    }
}

/// Compute the symbolic derivative of an expression with respect to a symbol: ∂expr / ∂var.
pub fn diff(expr: &Expr, var: &Symbol) -> Expr {
    simplify(&diff_unsimplified(expr, var))
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
fn carries_diff_sentinel(expr: &Expr) -> bool {
    match expr {
        Expr::Function(name, args) if name == "diff" && args.len() == 2 => true,
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

/// Indefinite integral of `expr` with respect to `var` (no `+ C`).
///
/// Handles sums, constant factors, and every case in
/// [`integral_term`]; anything else fails as
/// [`CalculusError::IntegrationFailed`] rather than returning a guess.
pub fn integrate(expr: &Expr, var: &Symbol) -> Result<Expr, CalculusError> {
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
                // Products of variable terms (x*sin(x), x*log(x), ...) have
                // no rule yet: typed refusal instead of unbounded recursion.
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

/// Limit of a univariate polynomial as `var` approaches a numeric point or
/// ±Infinity via direct substitution / degree analysis. Rational functions
/// at indeterminate points report [`CalculusError::Undetermined`].
pub fn limit(expr: &Expr, var: &Symbol, to: &Expr) -> Result<Expr, CalculusError> {
    match to {
        Expr::Const(Constant::Infinity) | Expr::Const(Constant::NegativeInfinity) => {
            let expanded = expand_with(expr, &mut Unbounded)
                .map_err(|error| CalculusError::Undetermined(error.to_string()))?;
            if is_free_of(&expanded, var) {
                return Ok(simplify(&expanded));
            }
            // Polynomial degree/leading-coefficient scan over additive terms.
            let terms: Vec<Expr> = match &expanded {
                Expr::Add(ts) => ts.clone(),
                single => vec![single.clone()],
            };
            let mut coefficients = BTreeMap::<u64, BigRational>::new();
            for t in &terms {
                let Some((degree, coefficient)) = polynomial_term(t, var) else {
                    return Err(CalculusError::Undetermined(expr.to_string()));
                };
                *coefficients.entry(degree).or_insert_with(BigRational::zero) += coefficient;
            }
            coefficients.retain(|_, coefficient| !coefficient.is_zero());
            let Some((&degree, leading_coefficient)) = coefficients.iter().next_back() else {
                return Ok(Expr::from_i64(0));
            };
            if degree == 0 {
                return Ok(simplify(&expanded));
            }
            let mut positive = leading_coefficient.is_positive();
            if *to == Expr::Const(Constant::NegativeInfinity) && degree % 2 == 1 {
                positive = !positive;
            }
            Ok(Expr::Const(if positive {
                Constant::Infinity
            } else {
                Constant::NegativeInfinity
            }))
        }
        point => {
            if unsafe_direct_substitution(expr) || unsafe_direct_substitution(point) {
                return Err(CalculusError::Undetermined(
                    UNSAFE_DIRECT_SUBSTITUTION.to_string(),
                ));
            }
            let substituted = expr.subs(&HashMap::from([(var.clone(), point.clone())]));
            if unsafe_direct_substitution(&substituted) {
                return Err(CalculusError::Undetermined(
                    UNSAFE_DIRECT_SUBSTITUTION.to_string(),
                ));
            }
            let value = simplify(&substituted);
            if unsafe_direct_substitution(&value) {
                return Err(CalculusError::Undetermined(
                    UNSAFE_DIRECT_SUBSTITUTION.to_string(),
                ));
            }
            if !is_free_of(&value, var) || carries_diff_sentinel(&value) {
                return Err(CalculusError::Undetermined(value.to_string()));
            }
            Ok(value)
        }
    }
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

fn monomial_degree(expr: &Expr, var: &Symbol) -> Option<u64> {
    match expr {
        Expr::Sym(s) if s == var => Some(1),
        Expr::Pow(b, e) => {
            let Expr::Integer(exponent) = e.as_ref() else {
                return None;
            };
            let exponent = u64::try_from(exponent).ok()?;
            let base_deg = monomial_degree(b, var)?;
            base_deg.checked_mul(exponent)
        }
        _ => None,
    }
}

/// Parse one expanded term as an exact univariate monomial.
///
/// Only numeric coefficients are admitted. A symbolic coefficient may be
/// positive, negative, or zero, so it cannot determine a limit at infinity
/// without assumptions.
fn polynomial_term(term: &Expr, var: &Symbol) -> Option<(u64, BigRational)> {
    let mut degree = 0u64;
    let mut coefficient = BigRational::from_integer(1.into());
    let mut factors = vec![term];
    while let Some(f) = factors.pop() {
        match f {
            Expr::Integer(value) => coefficient *= BigRational::from_integer(value.clone()),
            Expr::Rational(value) => coefficient *= value,
            Expr::Mul(nested) => factors.extend(nested),
            other => {
                let deg = monomial_degree(other, var)?;
                degree = degree.checked_add(deg)?;
            }
        }
    }
    Some((degree, coefficient))
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
    fn test_integrate_typed_refusal_on_product() {
        let x = Symbol::new("x");
        // x * sin(x) has no rule yet: typed refusal, never a guess.
        let e = Expr::Mul(vec![
            Expr::symbol("x"),
            Expr::Function("sin".to_string(), vec![Expr::symbol("x")]),
        ]);
        assert!(matches!(
            integrate(&e, &x),
            Err(CalculusError::IntegrationFailed(_))
        ));
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
        let expanded_quintic = expand_with(&quintic, &mut Unbounded).unwrap();
        assert_eq!(
            polynomial_term(&expanded_quintic, &x),
            Some((5, BigRational::from_integer(BigInt::from(-1)))),
            "expanded quintic: {expanded_quintic:?}"
        );
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
        assert!(matches!(
            limit(&fractional_power, &x, &Expr::Const(Constant::Infinity)),
            Err(CalculusError::Undetermined(_))
        ));

        assert_eq!(
            limit(&Expr::symbol("a"), &x, &Expr::Const(Constant::Infinity)).unwrap(),
            Expr::symbol("a")
        );
    }

    #[test]
    fn infinity_limit_reports_expansion_cap_as_typed_refusal() {
        let x = Symbol::new("x");
        let factor = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]);
        let expansion_bomb = Expr::Mul(vec![factor; 13]);
        assert!(matches!(
            limit(&expansion_bomb, &x, &Expr::Const(Constant::Infinity)),
            Err(CalculusError::Undetermined(_))
        ));
    }

    #[test]
    fn test_limit_zero_over_zero_is_undetermined() {
        let x = Symbol::new("x");
        // sin(x)/x at 0: structurally 0 * 0^-1 -> typed Undetermined.
        let e = Expr::Mul(vec![
            Expr::Function("sin".to_string(), vec![Expr::symbol("x")]),
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(-1))),
        ]);
        assert!(matches!(
            limit(&e, &x, &Expr::from_i64(0)),
            Err(CalculusError::Undetermined(_))
        ));

        let fractional_pole = Expr::Pow(
            Arc::new(Expr::symbol("x")),
            Arc::new(Expr::Rational(BigRational::new(
                BigInt::from(-1),
                BigInt::from(2),
            ))),
        );
        assert!(matches!(
            limit(&fractional_pole, &x, &Expr::from_i64(0)),
            Err(CalculusError::Undetermined(_))
        ));
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

        assert!(matches!(
            limit(&nested, &x, &Expr::from_i64(0)),
            Err(CalculusError::Undetermined(_))
        ));
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
            Expr::Pow(
                Arc::new(Expr::Add(vec![Expr::Sym(s.clone()), Expr::from_i64(-3),])),
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
    }
}
