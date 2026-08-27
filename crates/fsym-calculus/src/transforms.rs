//! Integral transforms (Laplace, Fourier) and definite integration (WS18).

#![forbid(unsafe_code)]

use crate::{CalculusError, integrate};
use fsym_core::{BigInt, Expr, Symbol};
use fsym_simplify::simplify;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) const MAX_LAPLACE_POLYNOMIAL_DEGREE: u64 = 4_096;
const MAX_DEFINITE_POLYNOMIAL_POWER: u64 = 4_096;
const MAX_DEFINITE_INTEGRAND_NODES: usize = 16_384;
const MAX_DEFINITE_INTEGRAND_DEPTH: usize = 256;

/// Admit only the bounded expression fragment that is structurally total on
/// finite rational endpoints. This is deliberately narrower than
/// [`integrate`]: an antiderivative rule alone does not establish that endpoint
/// subtraction is valid across singularities or branch cuts.
fn is_total_polynomial_fragment(expr: &Expr) -> bool {
    let mut pending = Vec::new();
    if pending.try_reserve(1).is_err() {
        return false;
    }
    pending.push((expr, 1usize));
    let mut visited = 0usize;

    while let Some((node, depth)) = pending.pop() {
        if depth > MAX_DEFINITE_INTEGRAND_DEPTH {
            return false;
        }
        visited = match visited.checked_add(1) {
            Some(count) if count <= MAX_DEFINITE_INTEGRAND_NODES => count,
            _ => return false,
        };

        match node {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Sym(_) => {}
            Expr::Add(children) | Expr::Mul(children) if !children.is_empty() => {
                let Some(child_depth) = depth.checked_add(1) else {
                    return false;
                };
                if pending.try_reserve(children.len()).is_err() {
                    return false;
                }
                pending.extend(children.iter().map(|child| (child, child_depth)));
            }
            Expr::Pow(base, exponent) => {
                let Expr::Integer(power) = exponent.as_ref() else {
                    return false;
                };
                if power
                    .to_u64()
                    .is_none_or(|power| power > MAX_DEFINITE_POLYNOMIAL_POWER)
                {
                    return false;
                }
                let Some(child_depth) = depth.checked_add(1) else {
                    return false;
                };
                if child_depth > MAX_DEFINITE_INTEGRAND_DEPTH {
                    return false;
                }
                // The exponent is an expression child too, even though its
                // integer shape was checked directly above.
                visited = match visited.checked_add(1) {
                    Some(count) if count <= MAX_DEFINITE_INTEGRAND_NODES => count,
                    _ => return false,
                };
                if pending.try_reserve(1).is_err() {
                    return false;
                }
                pending.push((base, child_depth));
            }
            _ => return false,
        }
    }

    true
}

fn linear_argument_coefficient(arg: &Expr, t: &Symbol) -> Option<Expr> {
    let t_expr = Expr::Sym(t.clone());
    if arg == &t_expr {
        return Some(Expr::from_i64(1));
    }

    let Expr::Mul(factors) = arg else {
        return None;
    };
    let mut coefficient_factors = Vec::with_capacity(factors.len().saturating_sub(1));
    let mut t_count = 0usize;
    for factor in factors {
        if factor == &t_expr {
            t_count += 1;
        } else if factor.free_symbols().contains(t) {
            return None;
        } else {
            coefficient_factors.push(factor.clone());
        }
    }
    (t_count == 1).then(|| simplify(&Expr::Mul(coefficient_factors)))
}

/// Computes a definite integral by endpoint subtraction for the currently
/// admitted structurally polynomial fragment.
///
/// Both endpoints must be exact finite integers or rationals. Powers must be
/// non-negative integers no larger than 4096, and the input shape is bounded.
/// Analytic functions, negative or rational powers, symbolic endpoints, and
/// unbounded shapes are refused until singularity and branch evidence exists.
pub fn integrate_definite(
    expr: &Expr,
    var: &Symbol,
    a: &Expr,
    b: &Expr,
) -> Result<Expr, CalculusError> {
    let exact_finite_endpoint =
        |endpoint: &Expr| matches!(endpoint, Expr::Integer(_) | Expr::Rational(_));
    if !exact_finite_endpoint(a) || !exact_finite_endpoint(b) || !is_total_polynomial_fragment(expr)
    {
        return Err(CalculusError::IntegrationFailed(
            "definite integration requires bounded polynomial input and exact finite rational endpoints"
                .to_string(),
        ));
    }

    let anti = integrate(expr, var)?;
    let mut sub_b = HashMap::new();
    sub_b.insert(var.clone(), b.clone());
    let fb = anti.subs(&sub_b);

    let mut sub_a = HashMap::new();
    sub_a.insert(var.clone(), a.clone());
    let fa = anti.subs(&sub_a);

    let diff_expr = Expr::Add(vec![fb, Expr::Mul(vec![Expr::from_i64(-1), fa])]);
    Ok(simplify(&diff_expr))
}

#[cfg(test)]
mod definite_integral_tests {
    use super::*;

    #[test]
    fn rejects_singular_integrand_across_interval() {
        let x = Symbol::new("x");
        let reciprocal_square =
            Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-2)));

        assert!(
            integrate_definite(
                &reciprocal_square,
                &x,
                &Expr::from_i64(-1),
                &Expr::from_i64(1),
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_unadmitted_analytic_and_endpoint_cases() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let sin_x = Expr::Function("sin".to_string(), vec![Expr::Sym(x.clone())]);
        let reciprocal = Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-1)));

        assert!(integrate_definite(&sin_x, &x, &Expr::from_i64(0), &Expr::from_i64(1)).is_err());
        assert!(
            integrate_definite(&Expr::Sym(x.clone()), &x, &Expr::from_i64(0), &Expr::Sym(y),)
                .is_err()
        );
        assert!(
            integrate_definite(&reciprocal, &x, &Expr::from_i64(1), &Expr::from_i64(2),).is_err()
        );
    }

    #[test]
    fn rejects_oversized_power_and_depth() {
        let x = Symbol::new("x");
        let oversized_power = Expr::Pow(
            Arc::new(Expr::Sym(x.clone())),
            Arc::new(Expr::from_i64(4_097)),
        );
        assert!(
            integrate_definite(&oversized_power, &x, &Expr::from_i64(0), &Expr::from_i64(1),)
                .is_err()
        );

        let mut too_deep = Expr::Sym(x.clone());
        for _ in 0..MAX_DEFINITE_INTEGRAND_DEPTH {
            too_deep = Expr::Add(vec![too_deep]);
        }
        assert!(
            integrate_definite(&too_deep, &x, &Expr::from_i64(0), &Expr::from_i64(1),).is_err()
        );
    }
}

/// Exact unilateral Laplace transform: $\mathcal{L}\{f(t)\}(s) = \int_0^\infty e^{-st} f(t) dt$.
pub fn laplace_transform(expr: &Expr, t: &Symbol, s: &Symbol) -> Result<Expr, CalculusError> {
    if t == s {
        return Err(CalculusError::IntegrationFailed(
            "Laplace input and transform variables must be distinct".to_string(),
        ));
    }
    if expr.free_symbols().contains(s) {
        return Err(CalculusError::IntegrationFailed(format!(
            "Laplace input must be independent of transform variable {s}"
        )));
    }

    let s_sym = Expr::Sym(s.clone());
    let t_sym = Expr::Sym(t.clone());

    if !expr.free_symbols().contains(t) {
        let inv_s = Expr::Pow(Arc::new(s_sym), Arc::new(Expr::from_i64(-1)));
        return Ok(simplify(&(expr.clone() * inv_s)));
    }

    match expr {
        Expr::Add(terms) => {
            let mut transformed = Vec::with_capacity(terms.len());
            for term in terms {
                transformed.push(laplace_transform(term, t, s)?);
            }
            Ok(simplify(&Expr::Add(transformed)))
        }
        Expr::Mul(factors) => {
            let mut consts = Vec::new();
            let mut t_factors = Vec::new();
            for f in factors {
                if !f.free_symbols().contains(t) {
                    consts.push(f.clone());
                } else {
                    t_factors.push(f.clone());
                }
            }
            if t_factors.is_empty() {
                // Constant c -> c / s
                let c = simplify(&Expr::Mul(consts));
                let inv_s = Expr::Pow(Arc::new(s_sym), Arc::new(Expr::from_i64(-1)));
                return Ok(simplify(&(c * inv_s)));
            }
            if t_factors.len() == 1 {
                let c = simplify(&Expr::Mul(consts));
                let l_t = laplace_transform(&t_factors[0], t, s)?;
                return Ok(simplify(&(c * l_t)));
            }
            if t_factors.len() == 2 {
                let c = simplify(&Expr::Mul(consts));
                for (exp_idx, other_idx) in [(0, 1), (1, 0)] {
                    if let Expr::Function(name, args) = &t_factors[exp_idx]
                        && name == "exp"
                        && args.len() == 1
                        && let Some(a) = linear_argument_coefficient(&args[0], t)
                    {
                        let base_transform = laplace_transform(&t_factors[other_idx], t, s)?;
                        let s_minus_a =
                            Expr::Add(vec![s_sym.clone(), Expr::Mul(vec![Expr::from_i64(-1), a])]);
                        let mut sub_map = HashMap::new();
                        sub_map.insert(s.clone(), s_minus_a);
                        let shifted = base_transform.subs(&sub_map);
                        return Ok(simplify(&(c * shifted)));
                    }
                }
                for (t_idx, other_idx) in [(0, 1), (1, 0)] {
                    if t_factors[t_idx] == t_sym {
                        let base_transform = laplace_transform(&t_factors[other_idx], t, s)?;
                        let d_ds = crate::diff(&base_transform, s);
                        let res = Expr::Mul(vec![Expr::from_i64(-1), d_ds]);
                        return Ok(simplify(&(c * res)));
                    }
                }
            }
            Err(CalculusError::IntegrationFailed(format!(
                "Laplace transform not computable for product: {}",
                expr
            )))
        }
        Expr::Sym(sym) if sym == t => {
            // L{t} = 1 / s^2
            let s_sq = Expr::Pow(Arc::new(s_sym), Arc::new(Expr::from_i64(2)));
            let inv_s_sq = Expr::Pow(Arc::new(s_sq), Arc::new(Expr::from_i64(-1)));
            Ok(simplify(&inv_s_sq))
        }
        Expr::Pow(base, exp) if base.as_ref() == &t_sym => {
            // L{t^n} = n! / s^(n+1) for positive integer n
            if let Expr::Integer(n) = exp.as_ref()
                && let Some(n_val) = n.to_u64()
                && n_val <= MAX_LAPLACE_POLYNOMIAL_DEGREE
            {
                let mut fact = BigInt::from(1);
                for i in 1..=n_val {
                    fact *= BigInt::from(i);
                }
                let exponent = i64::try_from(n_val + 1).map_err(|_| {
                    CalculusError::IntegrationFailed(
                        "Laplace polynomial exponent exceeds the supported range".to_string(),
                    )
                })?;
                let neg_np1 = Expr::from_i64(-exponent);
                let inv = Expr::Pow(Arc::new(s_sym), Arc::new(neg_np1));
                return Ok(simplify(&(Expr::Integer(fact) * inv)));
            }
            Err(CalculusError::IntegrationFailed(format!(
                "Laplace transform of t^{exp}; supported powers are non-negative integers up to {MAX_LAPLACE_POLYNOMIAL_DEGREE}"
            )))
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let arg = &args[0];
            match name.as_str() {
                "exp" => {
                    // L{exp(a*t)} = 1 / (s - a)
                    let Some(a) = linear_argument_coefficient(arg, t) else {
                        return Err(CalculusError::IntegrationFailed(format!("exp({arg})")));
                    };
                    let denom = Expr::Add(vec![s_sym, Expr::Mul(vec![Expr::from_i64(-1), a])]);
                    let inv = Expr::Pow(Arc::new(denom), Arc::new(Expr::from_i64(-1)));
                    Ok(simplify(&inv))
                }
                "sin" => {
                    // L{sin(w*t)} = w / (s^2 + w^2)
                    let Some(w) = linear_argument_coefficient(arg, t) else {
                        return Err(CalculusError::IntegrationFailed(format!("sin({arg})")));
                    };
                    let s_sq = Expr::Pow(Arc::new(s_sym), Arc::new(Expr::from_i64(2)));
                    let w_sq = Expr::Pow(Arc::new(w.clone()), Arc::new(Expr::from_i64(2)));
                    let denom = Expr::Add(vec![s_sq, w_sq]);
                    let inv = Expr::Pow(Arc::new(denom), Arc::new(Expr::from_i64(-1)));
                    Ok(simplify(&(w * inv)))
                }
                "cos" => {
                    // L{cos(w*t)} = s / (s^2 + w^2)
                    let Some(w) = linear_argument_coefficient(arg, t) else {
                        return Err(CalculusError::IntegrationFailed(format!("cos({arg})")));
                    };
                    let s_sq = Expr::Pow(Arc::new(s_sym.clone()), Arc::new(Expr::from_i64(2)));
                    let w_sq = Expr::Pow(Arc::new(w), Arc::new(Expr::from_i64(2)));
                    let denom = Expr::Add(vec![s_sq, w_sq]);
                    let inv = Expr::Pow(Arc::new(denom), Arc::new(Expr::from_i64(-1)));
                    Ok(simplify(&(s_sym * inv)))
                }
                "sinh" => {
                    // L{sinh(a*t)} = a / (s^2 - a^2)
                    let Some(a) = linear_argument_coefficient(arg, t) else {
                        return Err(CalculusError::IntegrationFailed(format!("sinh({arg})")));
                    };
                    let s_sq = Expr::Pow(Arc::new(s_sym), Arc::new(Expr::from_i64(2)));
                    let a_sq = Expr::Pow(Arc::new(a.clone()), Arc::new(Expr::from_i64(2)));
                    let denom = Expr::Add(vec![s_sq, Expr::Mul(vec![Expr::from_i64(-1), a_sq])]);
                    let inv = Expr::Pow(Arc::new(denom), Arc::new(Expr::from_i64(-1)));
                    Ok(simplify(&(a * inv)))
                }
                "cosh" => {
                    // L{cosh(a*t)} = s / (s^2 - a^2)
                    let Some(a) = linear_argument_coefficient(arg, t) else {
                        return Err(CalculusError::IntegrationFailed(format!("cosh({arg})")));
                    };
                    let s_sq = Expr::Pow(Arc::new(s_sym.clone()), Arc::new(Expr::from_i64(2)));
                    let a_sq = Expr::Pow(Arc::new(a), Arc::new(Expr::from_i64(2)));
                    let denom = Expr::Add(vec![s_sq, Expr::Mul(vec![Expr::from_i64(-1), a_sq])]);
                    let inv = Expr::Pow(Arc::new(denom), Arc::new(Expr::from_i64(-1)));
                    Ok(simplify(&(s_sym * inv)))
                }
                other => Err(CalculusError::IntegrationFailed(format!(
                    "Laplace transform of {other}({arg})"
                ))),
            }
        }
        other => Err(CalculusError::IntegrationFailed(format!(
            "Laplace transform not implemented for {other}"
        ))),
    }
}

/// Typed Laplace transform output with Region of Convergence (ROC) metadata.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LaplaceResult {
    pub transform: Expr,
    pub roc_abscissa: Option<Expr>,
}

/// Computes the unilateral Laplace transform along with its Region of Convergence: $\text{Re}(s) > \sigma_0$.
pub fn laplace_transform_with_roc(
    expr: &Expr,
    t: &Symbol,
    s: &Symbol,
) -> Result<LaplaceResult, CalculusError> {
    let transform = laplace_transform(expr, t, s)?;
    let roc_abscissa = compute_roc_abscissa(expr, t);
    Ok(LaplaceResult {
        transform,
        roc_abscissa,
    })
}

fn compute_roc_abscissa(expr: &Expr, t: &Symbol) -> Option<Expr> {
    match expr {
        Expr::Function(name, args) if args.len() == 1 => {
            let arg = &args[0];
            match name.as_str() {
                "exp" => linear_argument_coefficient(arg, t),
                "sinh" | "cosh" => {
                    let a = linear_argument_coefficient(arg, t)?;
                    Some(Expr::Function("abs".to_string(), vec![a]))
                }
                "sin" | "cos" => Some(Expr::from_i64(0)),
                _ => None,
            }
        }
        Expr::Pow(base, _) if base.as_ref() == &Expr::Sym(t.clone()) => Some(Expr::from_i64(0)),
        Expr::Sym(sym) if sym == t => Some(Expr::from_i64(0)),
        _ if !expr.free_symbols().contains(t) => Some(Expr::from_i64(0)),
        Expr::Mul(factors) => {
            for f in factors {
                if let Expr::Function(name, args) = f
                    && name == "exp"
                    && args.len() == 1
                {
                    return linear_argument_coefficient(&args[0], t);
                }
            }
            Some(Expr::from_i64(0))
        }
        Expr::Add(terms) => {
            let mut max_abscissa = None;
            for term in terms {
                let roc = compute_roc_abscissa(term, t)?;
                max_abscissa = Some(roc);
            }
            max_abscissa
        }
        _ => None,
    }
}

/// Exact Fourier transform: $\mathcal{F}\{f(t)\}(\omega) = \int_{-\infty}^\infty f(t) e^{-i \omega t} dt$ for catalog functions.
pub fn fourier_transform(expr: &Expr, t: &Symbol, omega: &Symbol) -> Result<Expr, CalculusError> {
    if t == omega {
        return Err(CalculusError::IntegrationFailed(
            "Fourier input and transform variables must be distinct".to_string(),
        ));
    }
    if expr.free_symbols().contains(omega) {
        return Err(CalculusError::IntegrationFailed(format!(
            "Fourier input must be independent of transform variable {omega}"
        )));
    }
    let omega_sym = Expr::Sym(omega.clone());
    let t_sym = Expr::Sym(t.clone());

    if !expr.free_symbols().contains(t) {
        let two_pi = Expr::Mul(vec![
            Expr::from_i64(2),
            Expr::Const(fsym_core::Constant::Pi),
            expr.clone(),
        ]);
        let dirac = Expr::Function("dirac".to_string(), vec![omega_sym]);
        return Ok(simplify(&(two_pi * dirac)));
    }

    match expr {
        Expr::Add(terms) => {
            let mut transformed = Vec::with_capacity(terms.len());
            for term in terms {
                transformed.push(fourier_transform(term, t, omega)?);
            }
            Ok(simplify(&Expr::Add(transformed)))
        }
        Expr::Mul(factors) => {
            let mut consts = Vec::new();
            let mut t_factors = Vec::new();
            for f in factors {
                if !f.free_symbols().contains(t) {
                    consts.push(f.clone());
                } else {
                    t_factors.push(f.clone());
                }
            }
            if t_factors.len() == 1 {
                let c = simplify(&Expr::Mul(consts));
                let f_t = fourier_transform(&t_factors[0], t, omega)?;
                return Ok(simplify(&(c * f_t)));
            }
            Err(CalculusError::IntegrationFailed(format!(
                "Fourier transform not computable for product: {}",
                expr
            )))
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let arg = &args[0];
            match name.as_str() {
                "exp" => {
                    // Check for exp(-a * abs(t)) -> 2a / (a^2 + omega^2)
                    let abs_t_expr = Expr::Function("abs".to_string(), vec![t_sym.clone()]);
                    if let Expr::Mul(factors) = arg {
                        let mut non_abs = Vec::new();
                        let mut has_abs = false;
                        for f in factors {
                            if f == &abs_t_expr {
                                has_abs = true;
                            } else if f.free_symbols().contains(t) {
                                has_abs = false;
                                break;
                            } else {
                                non_abs.push(f.clone());
                            }
                        }
                        if has_abs {
                            let neg_a = simplify(&Expr::Mul(non_abs));
                            let a = simplify(&(Expr::from_i64(-1) * neg_a));
                            let two_a = Expr::Mul(vec![Expr::from_i64(2), a.clone()]);
                            let a_sq = Expr::Pow(Arc::new(a), Arc::new(Expr::from_i64(2)));
                            let w_sq = Expr::Pow(Arc::new(omega_sym), Arc::new(Expr::from_i64(2)));
                            let denom = Expr::Add(vec![a_sq, w_sq]);
                            let inv = Expr::Pow(Arc::new(denom), Arc::new(Expr::from_i64(-1)));
                            return Ok(simplify(&(two_a * inv)));
                        }
                    }
                    Err(CalculusError::IntegrationFailed(format!(
                        "Fourier transform of exp({arg})"
                    )))
                }
                other => Err(CalculusError::IntegrationFailed(format!(
                    "Fourier transform of {other}({arg})"
                ))),
            }
        }
        other => Err(CalculusError::IntegrationFailed(format!(
            "Fourier transform not implemented for {other}"
        ))),
    }
}
