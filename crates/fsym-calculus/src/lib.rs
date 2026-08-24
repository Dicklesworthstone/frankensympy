//! # fsym-calculus
//!
//! Symbolic differentiation, integration, limits, and series expansion.

pub mod compile;
pub mod proof;
pub mod transforms;

pub use compile::*;
pub use proof::*;
pub use transforms::*;

use fsym_core::{BigInt, Constant, Expr, Symbol};
use fsym_simplify::{expand, simplify};
use num_traits::Zero;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CalculusError {
    #[error("Cannot differentiate non-differentiable term: {0}")]
    NonDifferentiable(String),
    #[error("Integration not computable symbolically: {0}")]
    IntegrationFailed(String),
    #[error("Limit undetermined with available rules: {0}")]
    Undetermined(String),
}

use fsym_core::BigRational;
use std::collections::HashMap;

/// Compute the symbolic derivative of an expression with respect to a symbol: ∂expr / ∂var.
pub fn diff(expr: &Expr, var: &Symbol) -> Expr {
    let unsimplified = match expr {
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
            // d(u^v) = u^v * (v' * ln(u) + v * u' / u)
            // Special case when exponent is constant integer n: d(u^n) = n * u^(n-1) * u'
            if let Expr::Integer(n) = exp.as_ref() {
                let n_minus_1 = n - BigInt::from(1);
                let du = diff(base, var);
                Expr::Mul(vec![
                    Expr::Integer(n.clone()),
                    Expr::Pow(base.clone(), Arc::new(Expr::Integer(n_minus_1))),
                    du,
                ])
            } else {
                // General chain rule fallback
                Expr::Function(
                    "diff".to_string(),
                    vec![expr.clone(), Expr::Sym(var.clone())],
                )
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
            } else if name == "exp" && args.len() == 1 {
                let u = &args[0];
                let du = diff(u, var);
                Expr::Mul(vec![Expr::Function("exp".to_string(), vec![u.clone()]), du])
            } else {
                Expr::Function(
                    "diff".to_string(),
                    vec![expr.clone(), Expr::Sym(var.clone())],
                )
            }
        }
    };
    simplify(&unsimplified)
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
                for f in factors {
                    if f == &x {
                        var_count += 1;
                    } else if is_free_of(f, var) {
                        const_factors.push(f.clone());
                    }
                }
                if var_count == 1 {
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
                        other => Err(CalculusError::IntegrationFailed(format!("{other}({u})"))),
                    }
                } else {
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
                        other => Err(CalculusError::IntegrationFailed(format!("{other}({u})"))),
                    }
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
            let expanded = expand(expr);
            // Polynomial degree/leading-coefficient scan over additive terms.
            let terms: Vec<Expr> = match &expanded {
                Expr::Add(ts) => ts.clone(),
                single => vec![single.clone()],
            };
            let mut best_degree: Option<i64> = None;
            let mut lead_coeff_sign: i64 = 0;
            for t in &terms {
                let (deg, sign) = term_degree_and_coeff_sign(t, var);
                if deg > best_degree.unwrap_or(i64::MIN) {
                    best_degree = Some(deg);
                    lead_coeff_sign = sign;
                }
            }
            let Some(d) = best_degree else {
                return Err(CalculusError::Undetermined(expr.to_string()));
            };
            if d < 0 {
                return Err(CalculusError::Undetermined(expr.to_string()));
            }
            if d == 0 {
                return Ok(simplify(&expanded));
            }
            let mut sign = lead_coeff_sign;
            if *to == Expr::Const(Constant::NegativeInfinity) && d % 2 == 1 {
                sign = -sign;
            }
            if sign == 0 {
                return Err(CalculusError::Undetermined(expr.to_string()));
            }
            Ok(Expr::Const(if sign > 0 {
                Constant::Infinity
            } else {
                Constant::NegativeInfinity
            }))
        }
        point => {
            let value = simplify(&expr.subs(&HashMap::from([(var.clone(), point.clone())])));
            if !is_free_of(&value, var) || carries_diff_sentinel(&value) || divides_by_zero(&value)
            {
                return Err(CalculusError::Undetermined(value.to_string()));
            }
            Ok(value)
        }
    }
}

/// Structural detection of literal division by zero after substitution:
/// any negative-integer power of an exactly-zero base.
fn divides_by_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Pow(base, exp) => {
            let negative_exp = matches!(exp.as_ref(), Expr::Integer(n) if n.is_negative());
            (negative_exp && base.is_zero()) || divides_by_zero(base) || divides_by_zero(exp)
        }
        Expr::Add(terms) | Expr::Mul(terms) => terms.iter().any(divides_by_zero),
        _ => false,
    }
}

/// Degree in `var` of one expanded additive term and the sign of its
/// leading coefficient contribution.
fn term_degree_and_coeff_sign(term: &Expr, var: &Symbol) -> (i64, i64) {
    let x = Expr::Sym(var.clone());
    let mut degree = 0i64;
    let mut sign = 1i64;
    let factors: Vec<Expr> = match term {
        Expr::Mul(fs) => fs.clone(),
        single => vec![single.clone()],
    };
    for f in &factors {
        match f {
            Expr::Sym(s) if s == var => degree += 1,
            Expr::Pow(b, e) if b.as_ref() == &x => {
                if let Expr::Integer(n) = e.as_ref() {
                    degree += i64::try_from(n.clone()).unwrap_or(0);
                }
            }
            Expr::Integer(n) => {
                if *n < BigInt::from(0) {
                    sign = -sign;
                }
            }
            Expr::Const(Constant::NegativeInfinity) => sign = -sign,
            _ => {}
        }
    }
    (degree, sign)
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
        let simplified = simplify(&deriv);
        if k > 0 && carries_diff_sentinel(&simplified) {
            return Err(CalculusError::NonDifferentiable(expr.to_string()));
        }
        let value = simplified.subs(&HashMap::from([(var.clone(), at.clone())]));
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
        // log(x) differentiation hits diff's unsupported-function sentinel.
        let l = Expr::Function("log".to_string(), vec![Expr::symbol("x")]);
        assert!(matches!(
            taylor(&l, &x, &Expr::from_i64(1), 2),
            Err(CalculusError::NonDifferentiable(_))
        ));
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
    fn test_hero_pipeline_compiled_residual_and_jacobian_certification() {
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

        // Verify Jacobian against central finite differences with 1e-6 tolerance
        let verified = system.verify_with_finite_differences(&test_point, 1e-6, 1e-5);
        assert!(
            verified,
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
}
