//! Integral transforms (Laplace, Fourier) and definite integration (WS18).

#![forbid(unsafe_code)]

use crate::{CalculusError, integrate};
use fsym_core::{BigInt, Expr, Symbol};
use fsym_simplify::simplify;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) const MAX_LAPLACE_POLYNOMIAL_DEGREE: u64 = 4_096;

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

/// Computes the definite integral $\int_a^b f(x) dx = F(b) - F(a)$ via the Fundamental Theorem of Calculus.
pub fn integrate_definite(
    expr: &Expr,
    var: &Symbol,
    a: &Expr,
    b: &Expr,
) -> Result<Expr, CalculusError> {
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
                    if let Expr::Mul(factors) = arg
                        && factors.len() == 2
                        && factors.iter().any(|f| {
                            matches!(f, Expr::Function(n, a) if n == "abs" && a.len() == 1 && a[0] == t_sym)
                        })
                    {
                        let a_opt = factors
                            .iter()
                            .find(|f| !matches!(f, Expr::Function(n, _) if n == "abs"));
                        if let Some(neg_a) = a_opt {
                            let a = simplify(&(Expr::from_i64(-1) * neg_a.clone()));
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
