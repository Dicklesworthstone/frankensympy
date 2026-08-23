//! # fsym-printing
//!
//! Formatted display, LaTeX emission, Unicode math rendering, and multi-language code generation.

#![forbid(unsafe_code)]

use fsym_core::{Constant, Expr};
use num_bigint::BigInt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PrintingError {
    #[error("Failed to render expression: {0}")]
    RenderError(String),
}

/// Operator binding strength, mirroring mathematical notation.
const PREC_ADD: u8 = 1;
const PREC_MUL: u8 = 2;
const PREC_POW: u8 = 3;
const PREC_ATOM: u8 = 4;

fn prec_of(e: &Expr) -> u8 {
    match e {
        Expr::Add(_) => PREC_ADD,
        Expr::Mul(_) => PREC_MUL,
        Expr::Pow(..) => PREC_POW,
        _ => PREC_ATOM,
    }
}

fn zero() -> BigInt {
    BigInt::from(0)
}

fn one() -> BigInt {
    BigInt::from(1)
}

fn two() -> BigInt {
    BigInt::from(2)
}

/// Whether an additive term carries a leading negative sign in printed form.
fn negative_head(e: &Expr) -> bool {
    match e {
        Expr::Integer(n) => *n < zero(),
        Expr::Rational(r) => *r.numer() < zero(),
        Expr::Mul(fs) => fs.first().is_some_and(negative_head),
        _ => false,
    }
}

/// Return the positive-magnitude view of an additive term (negating the head factor).
fn flip_sign(e: &Expr) -> Expr {
    match e {
        Expr::Integer(n) => Expr::Integer(-n.clone()),
        Expr::Rational(r) => Expr::Rational(-r.clone()),
        Expr::Mul(fs) => Expr::Mul(
            fs.iter()
                .enumerate()
                .map(|(i, f)| if i == 0 { flip_sign(f) } else { f.clone() })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn constant_latex(c: Constant) -> String {
    match c {
        Constant::Pi => "\\pi".to_string(),
        Constant::E => "e".to_string(),
        Constant::I => "i".to_string(),
        Constant::Infinity => "\\infty".to_string(),
        Constant::NegativeInfinity => "-\\infty".to_string(),
        Constant::ComplexInfinity => "\\tilde{\\infty}".to_string(),
        Constant::NaN => "\\text{NaN}".to_string(),
    }
}

fn symbol_latex(name: &str) -> String {
    const GREEK: [&str; 7] = ["alpha", "beta", "gamma", "theta", "pi", "sigma", "omega"];
    if GREEK.contains(&name) {
        format!("\\{}", name)
    } else {
        name.to_string()
    }
}

/// Backslash-function names recognized by the LaTeX emitter; anything else
/// renders through `\operatorname`.
fn known_function(name: &str) -> bool {
    [
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "log", "ln", "exp",
        "gamma", "zeta",
    ]
    .contains(&name)
}

/// Render a power whose exponent is a rational `p/q`, using root and
/// reciprocal forms where they improve readability.
fn rational_power_latex(base_wrapped: &str, p: &BigInt, q: &BigInt) -> String {
    let neg = *p < zero();
    let p_abs = if neg { -p.clone() } else { p.clone() };
    let core = match (&p_abs, q) {
        (p, q) if *q == two() && *p == one() => format!("\\sqrt{{{}}}", base_wrapped),
        (p, q) if *p == one() => format!("\\sqrt[{}]{{{}}}", q, base_wrapped),
        (p, q) => format!("{}^{{\\frac{{{}}}{{{}}}}}", base_wrapped, p, q),
    };
    if neg {
        format!("\\frac{{1}}{{{}}}", core)
    } else {
        core
    }
}

fn latex_prec(expr: &Expr, parent_prec: u8) -> String {
    let body = match expr {
        Expr::Sym(s) => symbol_latex(&s.name),
        Expr::Integer(n) => format!("{}", n),
        Expr::Rational(r) => {
            format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom())
        }
        Expr::Const(c) => constant_latex(*c),
        Expr::Add(terms) => {
            let mut out = String::new();
            for (i, t) in terms.iter().enumerate() {
                if i == 0 {
                    out.push_str(&wrap_latex(t, PREC_ADD));
                } else if negative_head(t) {
                    out.push_str(" - ");
                    out.push_str(&wrap_latex(&flip_sign(t), PREC_ADD));
                } else {
                    out.push_str(" + ");
                    out.push_str(&wrap_latex(t, PREC_ADD));
                }
            }
            out
        }
        Expr::Mul(factors) => {
            // Numeric coefficient leads; remaining factors keep their order.
            let mut ordered: Vec<&Expr> = factors.iter().filter(|f| !is_numeric(f)).collect();
            let nums: Vec<&Expr> = factors.iter().filter(|f| is_numeric(f)).collect();
            for (pos, n) in nums.iter().enumerate() {
                ordered.insert(pos, n);
            }
            ordered
                .iter()
                .map(|f| wrap_latex(f, PREC_MUL))
                .collect::<Vec<_>>()
                .join(" ")
        }
        Expr::Pow(base, exp) => {
            let b = wrap_latex(base, PREC_POW);
            match exp.as_ref() {
                Expr::Integer(n) if *n < zero() => {
                    format!("\\frac{{1}}{{{}^{{{}}}}}", b, -n)
                }
                Expr::Rational(r) => rational_power_latex(&b, r.numer(), r.denom()),
                e => format!("{}^{{{}}}", b, latex_prec(e, PREC_ADD)),
            }
        }
        Expr::Function(name, args) => {
            let arg_str = args
                .iter()
                .map(|a| latex_prec(a, PREC_ADD))
                .collect::<Vec<_>>()
                .join(", ");
            if known_function(name) {
                format!("\\{}\\left({}\\right)", name, arg_str)
            } else {
                format!("\\operatorname{{{}}}\\left({}\\right)", name, arg_str)
            }
        }
    };
    if prec_of(expr) < parent_prec {
        format!("\\left({}\\right)", body)
    } else {
        body
    }
}

fn wrap_latex(e: &Expr, min_prec: u8) -> String {
    latex_prec(e, min_prec)
}

fn is_numeric(e: &Expr) -> bool {
    matches!(e, Expr::Integer(_) | Expr::Rational(_))
}

/// Render symbolic expression as LaTeX code.
pub fn latex(expr: &Expr) -> String {
    latex_prec(expr, PREC_ADD)
}

/// Superscript rendering of an integer exponent using Unicode digits.
fn unicode_superscript(n: &BigInt) -> String {
    const DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let mut s = if *n < zero() {
        "⁻".to_string()
    } else {
        String::new()
    };
    for ch in n.magnitude().to_string().chars() {
        s.push(DIGITS[ch.to_digit(10).expect("decimal digit") as usize]);
    }
    s
}

fn constant_pretty(c: Constant) -> String {
    match c {
        Constant::Pi => "π".to_string(),
        Constant::E => "ℯ".to_string(),
        Constant::I => "𝑖".to_string(),
        Constant::Infinity => "∞".to_string(),
        Constant::NegativeInfinity => "-∞".to_string(),
        Constant::ComplexInfinity => "⧜".to_string(),
        Constant::NaN => "NaN".to_string(),
    }
}

fn pretty_prec(expr: &Expr, parent_prec: u8) -> String {
    let body = match expr {
        Expr::Sym(s) => symbol_latex(&s.name),
        Expr::Integer(n) => format!("{}", n),
        Expr::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expr::Const(c) => constant_pretty(*c),
        Expr::Add(terms) => {
            let mut out = String::new();
            for (i, t) in terms.iter().enumerate() {
                if i == 0 {
                    out.push_str(&wrap_pretty(t, PREC_ADD));
                } else if negative_head(t) {
                    out.push_str(" − ");
                    out.push_str(&wrap_pretty(&flip_sign(t), PREC_ADD));
                } else {
                    out.push_str(" + ");
                    out.push_str(&wrap_pretty(t, PREC_ADD));
                }
            }
            out
        }
        Expr::Mul(factors) => {
            let mut ordered: Vec<&Expr> = factors.iter().filter(|f| !is_numeric(f)).collect();
            let nums: Vec<&Expr> = factors.iter().filter(|f| is_numeric(f)).collect();
            for (pos, n) in nums.iter().enumerate() {
                ordered.insert(pos, n);
            }
            ordered
                .iter()
                .map(|f| wrap_pretty(f, PREC_MUL))
                .collect::<Vec<_>>()
                .join("·")
        }
        Expr::Pow(base, exp) => match exp.as_ref() {
            Expr::Integer(n) => {
                format!("{}{}", wrap_pretty(base, PREC_POW), unicode_superscript(n))
            }
            Expr::Rational(r) if r.denom() == &two() => {
                format!("√({})", pretty_prec(base, PREC_ADD))
            }
            e => format!(
                "{}^({})",
                wrap_pretty(base, PREC_POW),
                pretty_prec(e, PREC_ADD)
            ),
        },
        Expr::Function(name, args) => {
            let arg_str = args
                .iter()
                .map(|a| pretty_prec(a, PREC_ADD))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", name, arg_str)
        }
    };
    if prec_of(expr) < parent_prec {
        format!("({})", body)
    } else {
        body
    }
}

fn wrap_pretty(e: &Expr, min_prec: u8) -> String {
    pretty_prec(e, min_prec)
}

/// Render symbolic expression as human-friendly Unicode math.
pub fn pretty(expr: &Expr) -> String {
    pretty_prec(expr, PREC_ADD)
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
    use num_rational::BigRational;
    use std::sync::Arc;

    fn pow(b: Expr, e: Expr) -> Expr {
        Expr::Pow(Arc::new(b), Arc::new(e))
    }

    fn add(ts: Vec<Expr>) -> Expr {
        Expr::Add(ts)
    }

    fn mul(fs: Vec<Expr>) -> Expr {
        Expr::Mul(fs)
    }

    fn sym(n: &str) -> Expr {
        Expr::symbol(n)
    }

    #[test]
    fn test_latex_emission() {
        assert_eq!(latex(&pow(sym("x"), Expr::from_i64(2))), "x^{2}");
    }

    #[test]
    fn test_latex_pow_over_add_is_parenthesized() {
        // The precedence bug: (x + y)^2 must not print as x + y^{2}.
        let e = pow(add(vec![sym("x"), sym("y")]), Expr::from_i64(2));
        assert_eq!(latex(&e), "\\left(x + y\\right)^{2}");
    }

    #[test]
    fn test_latex_add_subtraction_sign() {
        // x + (-3) renders as subtraction, not "x + -3".
        let e = add(vec![sym("x"), Expr::from_i64(-3)]);
        assert_eq!(latex(&e), "x - 3");
        // x + (-2 y): negative multiplicative term flips to subtraction.
        let e2 = add(vec![sym("x"), mul(vec![Expr::from_i64(-2), sym("y")])]);
        assert_eq!(latex(&e2), "x - 2 y");
    }

    #[test]
    fn test_latex_mul_numeric_leads_and_add_children_parenthesized() {
        // y * 2 orders as 2 y; (x + 1) * y parenthesizes the sum.
        let e = mul(vec![sym("y"), Expr::from_i64(2)]);
        assert_eq!(latex(&e), "2 y");
        let e2 = mul(vec![add(vec![sym("x"), Expr::from_i64(1)]), sym("y")]);
        assert_eq!(latex(&e2), "\\left(x + 1\\right) y");
    }

    #[test]
    fn test_latex_negative_exponent_is_reciprocal() {
        let e = pow(sym("x"), Expr::from_i64(-1));
        assert_eq!(latex(&e), "\\frac{1}{x^{1}}");
    }

    #[test]
    fn test_latex_sqrt_and_root_forms() {
        let half = BigRational::new(one(), two());
        let e = pow(sym("x"), Expr::Rational(half));
        assert_eq!(latex(&e), "\\sqrt{x}");
        let third = BigRational::new(one(), BigInt::from(3));
        let cube_root = pow(sym("x"), Expr::Rational(third));
        assert_eq!(latex(&cube_root), "\\sqrt[3]{x}");
    }

    #[test]
    fn test_latex_rational_and_functions() {
        let r = BigRational::new(BigInt::from(3), BigInt::from(4));
        assert_eq!(latex(&Expr::Rational(r)), "\\frac{3}{4}");
        let s = Expr::Function("sin".to_string(), vec![sym("x")]);
        assert_eq!(latex(&s), "\\sin\\left(x\\right)");
        let f = Expr::Function("foo".to_string(), vec![sym("x")]);
        assert_eq!(latex(&f), "\\operatorname{foo}\\left(x\\right)");
    }

    #[test]
    fn test_latex_greek_symbols() {
        assert_eq!(latex(&sym("alpha")), "\\alpha");
        assert_eq!(latex(&Expr::Const(Constant::Pi)), "\\pi");
    }

    #[test]
    fn test_pretty_superscripts_and_operators() {
        assert_eq!(pretty(&pow(sym("x"), Expr::from_i64(2))), "x²");
        assert_eq!(pretty(&pow(sym("x"), Expr::from_i64(-3))), "x⁻³");
        let e = mul(vec![Expr::from_i64(2), sym("x")]);
        assert_eq!(pretty(&e), "2·x");
        let d = add(vec![sym("x"), Expr::from_i64(-1)]);
        assert_eq!(pretty(&d), "x − 1");
    }
}
