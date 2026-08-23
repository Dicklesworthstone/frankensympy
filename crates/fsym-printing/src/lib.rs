//! # fsym-printing
//!
//! Formatted display, LaTeX emission, Unicode math rendering, and multi-language code generation.

#![forbid(unsafe_code)]

use fsym_core::{Constant, Expr};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PrintingError {
    #[error("Failed to render expression: {0}")]
    RenderError(String),
}

/// Render symbolic expression as LaTeX code.
pub fn latex(expr: &Expr) -> String {
    match expr {
        Expr::Sym(s) => {
            // Check for greek letters
            if ["alpha", "beta", "gamma", "theta", "pi", "sigma", "omega"]
                .contains(&s.name.as_str())
            {
                format!("\\{}", s.name)
            } else {
                s.name.clone()
            }
        }
        Expr::Integer(n) => format!("{}", n),
        Expr::Rational(r) => format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom()),
        Expr::Const(c) => match c {
            Constant::Pi => "\\pi".to_string(),
            Constant::E => "e".to_string(),
            Constant::I => "i".to_string(),
            Constant::Infinity => "\\infty".to_string(),
            Constant::NegativeInfinity => "-\\infty".to_string(),
            Constant::ComplexInfinity => "\\tilde{\\infty}".to_string(),
            Constant::NaN => "\\text{NaN}".to_string(),
        },
        Expr::Add(terms) => terms.iter().map(latex).collect::<Vec<_>>().join(" + "),
        Expr::Mul(factors) => factors
            .iter()
            .map(|f| match f {
                Expr::Add(_) => format!("\\left({}\\right)", latex(f)),
                _ => latex(f),
            })
            .collect::<Vec<_>>()
            .join(" "),
        Expr::Pow(base, exp) => match exp.as_ref() {
            Expr::Rational(r) if r.numer() == &1.into() && r.denom() == &2.into() => {
                format!("\\sqrt{{{}}}", latex(base))
            }
            _ => format!("{}^{{{}}}", latex(base), latex(exp)),
        },
        Expr::Function(name, args) => {
            let arg_str = args.iter().map(latex).collect::<Vec<_>>().join(", ");
            format!("\\{}\\left({}\\right)", name, arg_str)
        }
    }
}

/// Render symbolic expression as Rust code snippet.
pub fn to_rust_code(expr: &Expr) -> String {
    match expr {
        Expr::Sym(s) => s.name.clone(),
        Expr::Integer(n) => format!("{}.0", n),
        Expr::Rational(r) => format!("({}.0 / {}.0)", r.numer(), r.denom()),
        Expr::Const(Constant::Pi) => "std::f64::consts::PI".to_string(),
        Expr::Const(Constant::E) => "std::f64::consts::E".to_string(),
        Expr::Const(Constant::Infinity) => "f64::INFINITY".to_string(),
        Expr::Const(Constant::NegativeInfinity) => "f64::NEG_INFINITY".to_string(),
        Expr::Const(Constant::NaN) => "f64::NAN".to_string(),
        Expr::Const(Constant::I) | Expr::Const(Constant::ComplexInfinity) => {
            "/* complex constant */".to_string()
        }
        Expr::Add(terms) => {
            let s = terms
                .iter()
                .map(to_rust_code)
                .collect::<Vec<_>>()
                .join(" + ");
            format!("({})", s)
        }
        Expr::Mul(factors) => {
            let s = factors
                .iter()
                .map(to_rust_code)
                .collect::<Vec<_>>()
                .join(" * ");
            format!("({})", s)
        }
        Expr::Pow(b, e) => format!("({}).powf({})", to_rust_code(b), to_rust_code(e)),
        Expr::Function(name, args) => {
            let arg_str = args.iter().map(to_rust_code).collect::<Vec<_>>().join(", ");
            format!("{}({})", name, arg_str)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latex_emission() {
        let x = Expr::symbol("x");
        let expr = Expr::Pow(
            std::sync::Arc::new(x),
            std::sync::Arc::new(Expr::from_i64(2)),
        );
        assert_eq!(latex(&expr), "x^{2}");
    }
}
