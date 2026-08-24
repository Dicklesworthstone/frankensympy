//! Integral transforms (Laplace, Fourier) and definite integration (WS18).

#![forbid(unsafe_code)]

use crate::{CalculusError, integrate};
use fsym_core::{BigInt, Expr, Symbol};
use fsym_simplify::simplify;
use std::collections::HashMap;
use std::sync::Arc;

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
    let s_sym = Expr::Sym(s.clone());
    let t_sym = Expr::Sym(t.clone());

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
            Err(CalculusError::IntegrationFailed(format!(
                "Laplace transform not computable for product: {}",
                expr
            )))
        }
        Expr::Integer(n) => {
            let inv_s = Expr::Pow(Arc::new(s_sym), Arc::new(Expr::from_i64(-1)));
            Ok(simplify(&(Expr::Integer(n.clone()) * inv_s)))
        }
        Expr::Rational(r) => {
            let inv_s = Expr::Pow(Arc::new(s_sym), Arc::new(Expr::from_i64(-1)));
            Ok(simplify(&(Expr::Rational(r.clone()) * inv_s)))
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
                && let Ok(n_val) = n.to_string().parse::<u64>()
            {
                let mut fact = BigInt::from(1);
                for i in 1..=n_val {
                    fact *= BigInt::from(i);
                }
                let neg_np1 = Expr::from_i64(-((n_val + 1) as i64));
                let inv = Expr::Pow(Arc::new(s_sym), Arc::new(neg_np1));
                return Ok(simplify(&(Expr::Integer(fact) * inv)));
            }
            Err(CalculusError::IntegrationFailed(format!(
                "Laplace transform of t^{exp}"
            )))
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let arg = &args[0];
            match name.as_str() {
                "exp" => {
                    // L{exp(a*t)} = 1 / (s - a)
                    let a = if arg == &t_sym {
                        Expr::from_i64(1)
                    } else if let Expr::Mul(factors) = arg {
                        let mut consts = Vec::new();
                        for f in factors {
                            if f != &t_sym {
                                consts.push(f.clone());
                            }
                        }
                        simplify(&Expr::Mul(consts))
                    } else {
                        return Err(CalculusError::IntegrationFailed(format!("exp({arg})")));
                    };
                    let denom = Expr::Add(vec![s_sym, Expr::Mul(vec![Expr::from_i64(-1), a])]);
                    let inv = Expr::Pow(Arc::new(denom), Arc::new(Expr::from_i64(-1)));
                    Ok(simplify(&inv))
                }
                "sin" => {
                    // L{sin(w*t)} = w / (s^2 + w^2)
                    let w = if arg == &t_sym {
                        Expr::from_i64(1)
                    } else if let Expr::Mul(factors) = arg {
                        let mut consts = Vec::new();
                        for f in factors {
                            if f != &t_sym {
                                consts.push(f.clone());
                            }
                        }
                        simplify(&Expr::Mul(consts))
                    } else {
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
                    let w = if arg == &t_sym {
                        Expr::from_i64(1)
                    } else if let Expr::Mul(factors) = arg {
                        let mut consts = Vec::new();
                        for f in factors {
                            if f != &t_sym {
                                consts.push(f.clone());
                            }
                        }
                        simplify(&Expr::Mul(consts))
                    } else {
                        return Err(CalculusError::IntegrationFailed(format!("cos({arg})")));
                    };
                    let s_sq = Expr::Pow(Arc::new(s_sym.clone()), Arc::new(Expr::from_i64(2)));
                    let w_sq = Expr::Pow(Arc::new(w), Arc::new(Expr::from_i64(2)));
                    let denom = Expr::Add(vec![s_sq, w_sq]);
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
