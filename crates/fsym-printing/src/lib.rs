//! # fsym-printing
//!
//! Bounded LaTeX, Unicode-math, and Rust/Python/C expression rendering for the
//! native expression tree. These views are not semantic identity and do not
//! claim SymPy-profile printer compatibility.

#![forbid(unsafe_code)]

use fsym_core::{BigInt, Constant, Expr};

mod bounded;

pub use bounded::{PrintingError, PrintingLimits};
use bounded::{RenderTarget, validate_render};

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

/// Whether a term has an explicit, printable leading negative sign.
fn negative_head(e: &Expr) -> bool {
    match e {
        Expr::Integer(n) => *n < zero(),
        Expr::Rational(r) => *r.numer() < zero(),
        Expr::Mul(fs) => fs.iter().filter(|factor| negative_head(factor)).count() % 2 == 1,
        _ => false,
    }
}

/// Return the positive-magnitude view of a term with an explicit sign.
fn positive_magnitude(e: &Expr) -> Expr {
    match e {
        Expr::Integer(n) => Expr::Integer(-n.clone()),
        Expr::Rational(r) => Expr::Rational(-r.clone()),
        Expr::Mul(fs) => Expr::Mul(
            fs.iter()
                .map(|factor| {
                    if negative_head(factor) {
                        positive_magnitude(factor)
                    } else {
                        factor.clone()
                    }
                })
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
        let mut escaped = String::with_capacity(name.len());
        for ch in name.chars() {
            match ch {
                '\\' => escaped.push_str("\\backslash{}"),
                '{' => escaped.push_str("\\{"),
                '}' => escaped.push_str("\\}"),
                '$' => escaped.push_str("\\$"),
                '&' => escaped.push_str("\\&"),
                '#' => escaped.push_str("\\#"),
                '%' => escaped.push_str("\\%"),
                '_' => escaped.push_str("\\_"),
                '^' => escaped.push_str("\\^{}"),
                '~' => escaped.push_str("\\~{}"),
                _ => escaped.push(ch),
            }
        }
        escaped
    }
}

fn symbol_pretty(name: &str) -> String {
    match name {
        "alpha" => "α".to_string(),
        "beta" => "β".to_string(),
        "gamma" => "γ".to_string(),
        "theta" => "θ".to_string(),
        "pi" => "π".to_string(),
        "sigma" => "σ".to_string(),
        "omega" => "ω".to_string(),
        _ => name.to_string(),
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
            if terms.is_empty() {
                return "0".to_string();
            }
            let mut out = String::new();
            for (i, t) in terms.iter().enumerate() {
                if i == 0 {
                    out.push_str(&wrap_latex(t, PREC_ADD));
                } else if negative_head(t) {
                    out.push_str(" - ");
                    out.push_str(&wrap_latex(&positive_magnitude(t), PREC_ADD));
                } else {
                    out.push_str(" + ");
                    out.push_str(&wrap_latex(t, PREC_ADD));
                }
            }
            out
        }
        Expr::Mul(factors) => {
            if factors.is_empty() {
                return "1".to_string();
            }
            // Numeric coefficient leads; remaining factors keep their order.
            let mut ordered: Vec<&Expr> = factors.iter().filter(|f| !is_numeric(f)).collect();
            let nums: Vec<&Expr> = factors.iter().filter(|f| is_numeric(f)).collect();
            for (pos, n) in nums.iter().enumerate() {
                ordered.insert(pos, n);
            }
            let body = ordered
                .iter()
                .map(|factor| {
                    if negative_head(factor) {
                        wrap_latex(&positive_magnitude(factor), PREC_MUL)
                    } else {
                        wrap_latex(factor, PREC_MUL)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            if negative_head(expr) {
                format!("-{body}")
            } else {
                body
            }
        }
        Expr::Pow(base, exp) => {
            let b = if matches!(base.as_ref(), Expr::Pow(..)) || negative_head(base) {
                format!("\\left({}\\right)", latex_prec(base, PREC_ADD))
            } else {
                wrap_latex(base, PREC_POW)
            };
            match exp.as_ref() {
                Expr::Integer(n) if *n < zero() => {
                    if *n == -one() {
                        format!("\\frac{{1}}{{{}}}", b)
                    } else {
                        format!("\\frac{{1}}{{{}^{{{}}}}}", b, -n)
                    }
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
                format!(
                    "\\operatorname{{{}}}\\left({}\\right)",
                    symbol_latex(name),
                    arg_str
                )
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

/// Render symbolic expression as bounded LaTeX code.
pub fn latex(expr: &Expr) -> Result<String, PrintingError> {
    latex_with_limits(expr, PrintingLimits::default())
}

/// Render symbolic expression as LaTeX code with caller-provided limits.
pub fn latex_with_limits(expr: &Expr, limits: PrintingLimits) -> Result<String, PrintingError> {
    validate_render(expr, RenderTarget::Latex, limits)?;
    Ok(latex_prec(expr, PREC_ADD))
}

/// Superscript rendering of an integer exponent using Unicode digits.
fn unicode_superscript(n: &BigInt) -> String {
    const DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let mut s = if n.is_negative() {
        "⁻".to_string()
    } else {
        String::new()
    };
    for ch in n.abs().to_string().chars() {
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
        Expr::Sym(s) => symbol_pretty(&s.name),
        Expr::Integer(n) => format!("{}", n),
        Expr::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expr::Const(c) => constant_pretty(*c),
        Expr::Add(terms) => {
            if terms.is_empty() {
                return "0".to_string();
            }
            let mut out = String::new();
            for (i, t) in terms.iter().enumerate() {
                if i == 0 {
                    out.push_str(&wrap_pretty(t, PREC_ADD));
                } else if negative_head(t) {
                    out.push_str(" − ");
                    out.push_str(&wrap_pretty(&positive_magnitude(t), PREC_ADD));
                } else {
                    out.push_str(" + ");
                    out.push_str(&wrap_pretty(t, PREC_ADD));
                }
            }
            out
        }
        Expr::Mul(factors) => {
            if factors.is_empty() {
                return "1".to_string();
            }
            let mut ordered: Vec<&Expr> = factors.iter().filter(|f| !is_numeric(f)).collect();
            let nums: Vec<&Expr> = factors.iter().filter(|f| is_numeric(f)).collect();
            for (pos, n) in nums.iter().enumerate() {
                ordered.insert(pos, n);
            }
            let body = ordered
                .iter()
                .map(|factor| {
                    if negative_head(factor) {
                        wrap_pretty(&positive_magnitude(factor), PREC_MUL)
                    } else {
                        wrap_pretty(factor, PREC_MUL)
                    }
                })
                .collect::<Vec<_>>()
                .join("·");
            if negative_head(expr) {
                format!("−{body}")
            } else {
                body
            }
        }
        Expr::Pow(base, exp) => match exp.as_ref() {
            Expr::Integer(n) => {
                let base = if matches!(base.as_ref(), Expr::Pow(..)) || negative_head(base) {
                    format!("({})", pretty_prec(base, PREC_ADD))
                } else {
                    wrap_pretty(base, PREC_POW)
                };
                format!("{}{}", base, unicode_superscript(n))
            }
            Expr::Rational(r) if r.denom() == &two() && r.numer().abs() == one() => {
                let root = format!("√({})", pretty_prec(base, PREC_ADD));
                if r.numer().is_negative() {
                    format!("1/{root}")
                } else {
                    root
                }
            }
            e => {
                let base = if matches!(base.as_ref(), Expr::Pow(..)) || negative_head(base) {
                    format!("({})", pretty_prec(base, PREC_ADD))
                } else {
                    wrap_pretty(base, PREC_POW)
                };
                format!("{}^({})", base, pretty_prec(e, PREC_ADD))
            }
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

/// Render symbolic expression as bounded human-friendly Unicode math.
pub fn pretty(expr: &Expr) -> Result<String, PrintingError> {
    pretty_with_limits(expr, PrintingLimits::default())
}

/// Render symbolic expression as Unicode math with caller-provided limits.
pub fn pretty_with_limits(expr: &Expr, limits: PrintingLimits) -> Result<String, PrintingError> {
    validate_render(expr, RenderTarget::Pretty, limits)?;
    Ok(pretty_prec(expr, PREC_ADD))
}

fn rust_code_unchecked(expr: &Expr) -> String {
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
            if terms.is_empty() {
                return "0.0".to_string();
            }
            let s = terms
                .iter()
                .map(rust_code_unchecked)
                .collect::<Vec<_>>()
                .join(" + ");
            format!("({})", s)
        }
        Expr::Mul(factors) => {
            if factors.is_empty() {
                return "1.0".to_string();
            }
            let s = factors
                .iter()
                .map(rust_code_unchecked)
                .collect::<Vec<_>>()
                .join(" * ");
            format!("({})", s)
        }
        Expr::Pow(b, e) => format!(
            "({}).powf({})",
            rust_code_unchecked(b),
            rust_code_unchecked(e)
        ),
        Expr::Function(name, args) => {
            let arg_str = args
                .iter()
                .map(rust_code_unchecked)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", name, arg_str)
        }
    }
}

/// Render a bounded real-valued Rust expression.
///
/// Invalid identifiers, exact values outside the emitter's current `i64`
/// numeric lane, and complex constants are refused rather than emitted as
/// uncompilable or misleading source.
pub fn to_rust_code(expr: &Expr) -> Result<String, PrintingError> {
    to_rust_code_with_limits(expr, PrintingLimits::default())
}

/// Render a Rust expression with caller-provided limits.
pub fn to_rust_code_with_limits(
    expr: &Expr,
    limits: PrintingLimits,
) -> Result<String, PrintingError> {
    validate_render(expr, RenderTarget::Rust, limits)?;
    Ok(rust_code_unchecked(expr))
}

fn python_code_unchecked(expr: &Expr) -> String {
    match expr {
        Expr::Sym(s) => s.name.clone(),
        Expr::Integer(n) => format!("{}", n),
        Expr::Rational(r) => format!("fractions.Fraction({}, {})", r.numer(), r.denom()),
        Expr::Const(Constant::Pi) => "math.pi".to_string(),
        Expr::Const(Constant::E) => "math.e".to_string(),
        Expr::Const(Constant::Infinity) => "float('inf')".to_string(),
        Expr::Const(Constant::NegativeInfinity) => "float('-inf')".to_string(),
        Expr::Const(Constant::NaN) => "float('nan')".to_string(),
        Expr::Const(Constant::I) => "1j".to_string(),
        Expr::Const(Constant::ComplexInfinity) => "complex(float('nan'), float('nan'))".to_string(),
        Expr::Add(terms) => {
            if terms.is_empty() {
                return "0".to_string();
            }
            let s = terms
                .iter()
                .map(python_code_unchecked)
                .collect::<Vec<_>>()
                .join(" + ");
            format!("({})", s)
        }
        Expr::Mul(factors) => {
            if factors.is_empty() {
                return "1".to_string();
            }
            let s = factors
                .iter()
                .map(python_code_unchecked)
                .collect::<Vec<_>>()
                .join(" * ");
            format!("({})", s)
        }
        Expr::Pow(b, e) => format!(
            "(({}) ** ({}))",
            python_code_unchecked(b),
            python_code_unchecked(e)
        ),
        Expr::Function(name, args) => {
            let arg_str = args
                .iter()
                .map(python_code_unchecked)
                .collect::<Vec<_>>()
                .join(", ");
            let callable = match name.as_str() {
                "Abs" | "abs" => "abs".to_string(),
                "acos" | "acosh" | "asin" | "asinh" | "atan" | "atanh" | "cos" | "cosh" | "erf"
                | "erfc" | "exp" | "factorial" | "gamma" | "log" | "sin" | "sinh" | "sqrt"
                | "tan" | "tanh" => format!("math.{name}"),
                _ => name.clone(),
            };
            format!("{callable}({arg_str})")
        }
    }
}

/// Render a bounded Python expression.
///
/// The caller provides `math`, `fractions`, and any user-defined function names
/// referenced by the returned expression. This function emits source only; it
/// does not execute generated code.
pub fn to_python_code(expr: &Expr) -> Result<String, PrintingError> {
    to_python_code_with_limits(expr, PrintingLimits::default())
}

/// Render a Python expression with caller-provided limits.
pub fn to_python_code_with_limits(
    expr: &Expr,
    limits: PrintingLimits,
) -> Result<String, PrintingError> {
    validate_render(expr, RenderTarget::Python, limits)?;
    Ok(python_code_unchecked(expr))
}

fn c_code_unchecked(expr: &Expr) -> String {
    match expr {
        Expr::Sym(s) => s.name.clone(),
        Expr::Integer(n) => format!("{}.0", n),
        Expr::Rational(r) => format!("({}.0 / {}.0)", r.numer(), r.denom()),
        Expr::Const(Constant::Pi) => "acos(-1.0)".to_string(),
        Expr::Const(Constant::E) => "exp(1.0)".to_string(),
        Expr::Const(Constant::Infinity) => "INFINITY".to_string(),
        Expr::Const(Constant::NegativeInfinity) => "(-INFINITY)".to_string(),
        Expr::Const(Constant::NaN) => "NAN".to_string(),
        Expr::Const(Constant::I) | Expr::Const(Constant::ComplexInfinity) => {
            "/* complex constant */".to_string()
        }
        Expr::Add(terms) => {
            if terms.is_empty() {
                return "0.0".to_string();
            }
            let s = terms
                .iter()
                .map(c_code_unchecked)
                .collect::<Vec<_>>()
                .join(" + ");
            format!("({})", s)
        }
        Expr::Mul(factors) => {
            if factors.is_empty() {
                return "1.0".to_string();
            }
            let s = factors
                .iter()
                .map(c_code_unchecked)
                .collect::<Vec<_>>()
                .join(" * ");
            format!("({})", s)
        }
        Expr::Pow(b, e) => format!("pow({}, {})", c_code_unchecked(b), c_code_unchecked(e)),
        Expr::Function(name, args) => {
            let arg_str = args
                .iter()
                .map(c_code_unchecked)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", name, arg_str)
        }
    }
}

/// Render a bounded real-valued C expression.
pub fn to_c_code(expr: &Expr) -> Result<String, PrintingError> {
    to_c_code_with_limits(expr, PrintingLimits::default())
}

/// Render a C expression with caller-provided limits.
pub fn to_c_code_with_limits(expr: &Expr, limits: PrintingLimits) -> Result<String, PrintingError> {
    validate_render(expr, RenderTarget::C, limits)?;
    Ok(c_code_unchecked(expr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::BigRational;
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
        assert_eq!(latex(&pow(sym("x"), Expr::from_i64(2))).unwrap(), "x^{2}");
    }

    #[test]
    fn test_latex_pow_over_add_is_parenthesized() {
        // The precedence bug: (x + y)^2 must not print as x + y^{2}.
        let e = pow(add(vec![sym("x"), sym("y")]), Expr::from_i64(2));
        assert_eq!(latex(&e).unwrap(), "\\left(x + y\\right)^{2}");
    }

    #[test]
    fn test_latex_add_subtraction_sign() {
        // x + (-3) renders as subtraction, not "x + -3".
        let e = add(vec![sym("x"), Expr::from_i64(-3)]);
        assert_eq!(latex(&e).unwrap(), "x - 3");
        // x + (-2 y): negative multiplicative term flips to subtraction.
        let e2 = add(vec![sym("x"), mul(vec![Expr::from_i64(-2), sym("y")])]);
        assert_eq!(latex(&e2).unwrap(), "x - 2 y");
    }

    #[test]
    fn test_latex_mul_numeric_leads_and_add_children_parenthesized() {
        // y * 2 orders as 2 y; (x + 1) * y parenthesizes the sum.
        let e = mul(vec![sym("y"), Expr::from_i64(2)]);
        assert_eq!(latex(&e).unwrap(), "2 y");
        let e2 = mul(vec![add(vec![sym("x"), Expr::from_i64(1)]), sym("y")]);
        assert_eq!(latex(&e2).unwrap(), "\\left(x + 1\\right) y");
    }

    #[test]
    fn test_latex_negative_exponent_is_reciprocal() {
        let e = pow(sym("x"), Expr::from_i64(-1));
        assert_eq!(latex(&e).unwrap(), "\\frac{1}{x}");
        let e2 = pow(sym("x"), Expr::from_i64(-2));
        assert_eq!(latex(&e2).unwrap(), "\\frac{1}{x^{2}}");
    }

    #[test]
    fn test_latex_sqrt_and_root_forms() {
        let half = BigRational::new(one(), two());
        let e = pow(sym("x"), Expr::Rational(half));
        assert_eq!(latex(&e).unwrap(), "\\sqrt{x}");
        let third = BigRational::new(one(), BigInt::from(3));
        let cube_root = pow(sym("x"), Expr::Rational(third));
        assert_eq!(latex(&cube_root).unwrap(), "\\sqrt[3]{x}");
    }

    #[test]
    fn test_latex_rational_and_functions() {
        let r = BigRational::new(BigInt::from(3), BigInt::from(4));
        assert_eq!(latex(&Expr::Rational(r)).unwrap(), "\\frac{3}{4}");
        let s = Expr::Function("sin".to_string(), vec![sym("x")]);
        assert_eq!(latex(&s).unwrap(), "\\sin\\left(x\\right)");
        let f = Expr::Function("foo".to_string(), vec![sym("x")]);
        assert_eq!(latex(&f).unwrap(), "\\operatorname{foo}\\left(x\\right)");
    }

    #[test]
    fn test_latex_greek_symbols() {
        assert_eq!(latex(&sym("alpha")).unwrap(), "\\alpha");
        assert_eq!(latex(&Expr::Const(Constant::Pi)).unwrap(), "\\pi");
    }

    #[test]
    fn test_pretty_superscripts_and_operators() {
        assert_eq!(pretty(&pow(sym("x"), Expr::from_i64(2))).unwrap(), "x²");
        assert_eq!(pretty(&pow(sym("x"), Expr::from_i64(-3))).unwrap(), "x⁻³");
        let e = mul(vec![Expr::from_i64(2), sym("x")]);
        assert_eq!(pretty(&e).unwrap(), "2·x");
        let d = add(vec![sym("x"), Expr::from_i64(-1)]);
        assert_eq!(pretty(&d).unwrap(), "x − 1");
    }

    #[test]
    fn test_power_precedence_and_exact_rational_roots() {
        let nested = pow(pow(sym("x"), Expr::from_i64(2)), Expr::from_i64(3));
        assert_eq!(latex(&nested).unwrap(), "\\left(x^{2}\\right)^{3}");
        let negative_base = pow(Expr::from_i64(-2), Expr::from_i64(2));
        assert_eq!(latex(&negative_base).unwrap(), "\\left(-2\\right)^{2}");
        assert_eq!(pretty(&negative_base).unwrap(), "(-2)²");

        let three_halves = BigRational::new(BigInt::from(3), two());
        assert_eq!(
            pretty(&pow(sym("x"), Expr::Rational(three_halves))).unwrap(),
            "x^(3/2)"
        );
        let minus_half = BigRational::new(-one(), two());
        assert_eq!(
            pretty(&pow(sym("x"), Expr::Rational(minus_half))).unwrap(),
            "1/√(x)"
        );
    }

    #[test]
    fn test_sign_normalization_handles_noncanonical_products() {
        let negative = mul(vec![sym("y"), Expr::from_i64(-2)]);
        let sum = add(vec![sym("x"), negative]);
        assert_eq!(latex(&sum).unwrap(), "x - 2 y");
        assert_eq!(pretty(&sum).unwrap(), "x − 2·y");

        let positive = mul(vec![Expr::from_i64(-2), Expr::from_i64(-3), sym("x")]);
        assert_eq!(latex(&positive).unwrap(), "2 3 x");
        assert_eq!(pretty(&positive).unwrap(), "2·3·x");
    }

    #[test]
    fn test_symbol_views_are_target_specific_and_latex_is_escaped() {
        assert_eq!(pretty(&sym("alpha")).unwrap(), "α");
        assert_eq!(latex(&sym("x_1{bad}")).unwrap(), "x\\_1\\{bad\\}");
        let function = Expr::Function("f_1".to_string(), vec![sym("x")]);
        assert_eq!(
            latex(&function).unwrap(),
            "\\operatorname{f\\_1}\\left(x\\right)"
        );
    }

    #[test]
    fn test_empty_nary_nodes_render_identity_values() {
        assert_eq!(latex(&add(Vec::new())).unwrap(), "0");
        assert_eq!(latex(&mul(Vec::new())).unwrap(), "1");
        assert_eq!(pretty(&add(Vec::new())).unwrap(), "0");
        assert_eq!(pretty(&mul(Vec::new())).unwrap(), "1");
        assert_eq!(to_rust_code(&add(Vec::new())).unwrap(), "0.0");
        assert_eq!(to_rust_code(&mul(Vec::new())).unwrap(), "1.0");
    }

    #[test]
    fn test_rust_emitter_refuses_misleading_source() {
        assert_eq!(
            to_rust_code(&sym("not valid")),
            Err(PrintingError::InvalidRustIdentifier)
        );
        assert_eq!(
            to_rust_code(&sym("match")),
            Err(PrintingError::InvalidRustIdentifier)
        );
        assert_eq!(
            to_rust_code(&Expr::Const(Constant::I)),
            Err(PrintingError::UnsupportedRustConstant(Constant::I))
        );
        let huge = BigInt::from(i64::MAX) + BigInt::from(1);
        assert_eq!(
            to_rust_code(&Expr::Integer(huge)),
            Err(PrintingError::RustNumericValueOutOfRange)
        );
    }

    #[test]
    fn test_renderers_fail_closed_at_limits() {
        let expression = add(vec![sym("alpha"), sym("beta")]);
        let node_limited = PrintingLimits {
            max_nodes: 1,
            ..PrintingLimits::default()
        };
        assert_eq!(
            latex_with_limits(&expression, node_limited),
            Err(PrintingError::NodeLimitExceeded { max_nodes: 1 })
        );

        let byte_limited = PrintingLimits {
            max_output_bytes: 3,
            ..PrintingLimits::default()
        };
        assert_eq!(
            pretty_with_limits(&sym("alpha"), byte_limited),
            Err(PrintingError::OutputLimitExceeded {
                max_output_bytes: 3
            })
        );

        let depth_limited = PrintingLimits {
            max_depth: 0,
            ..PrintingLimits::default()
        };
        assert_eq!(
            to_rust_code_with_limits(&pow(sym("x"), Expr::from_i64(2)), depth_limited),
            Err(PrintingError::DepthLimitExceeded { max_depth: 0 })
        );

        assert_eq!(
            pretty(&sym("x\ny")),
            Err(PrintingError::InvalidNameControlCharacter)
        );
    }

    #[test]
    fn test_python_and_c_code_emission() {
        let e = add(vec![
            pow(sym("x"), Expr::from_i64(2)),
            mul(vec![Expr::from_i64(3), sym("x")]),
            Expr::from_i64(5),
        ]);
        assert_eq!(to_python_code(&e).unwrap(), "(((x) ** (2)) + (3 * x) + 5)");
        assert_eq!(to_c_code(&e).unwrap(), "(pow(x, 2.0) + (3.0 * x) + 5.0)");

        let trig = Expr::Function("sin".to_string(), vec![sym("x")]);
        assert_eq!(to_python_code(&trig).unwrap(), "math.sin(x)");
        assert_eq!(to_c_code(&trig).unwrap(), "sin(x)");

        let user_function = Expr::Function("f".to_string(), vec![sym("x")]);
        assert_eq!(to_python_code(&user_function).unwrap(), "f(x)");

        let rational = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(3)));
        assert_eq!(
            to_python_code(&rational).unwrap(),
            "fractions.Fraction(1, 3)"
        );

        let pi_expr = Expr::Const(Constant::Pi);
        assert_eq!(to_python_code(&pi_expr).unwrap(), "math.pi");
        assert_eq!(to_c_code(&pi_expr).unwrap(), "acos(-1.0)");

        assert_eq!(
            to_python_code(&sym("def")),
            Err(PrintingError::InvalidPythonIdentifier)
        );
        assert_eq!(
            to_c_code(&sym("while")),
            Err(PrintingError::InvalidCIdentifier)
        );
        assert_eq!(
            to_c_code(&sym("_Atomic")),
            Err(PrintingError::InvalidCIdentifier)
        );
        assert_eq!(
            to_python_code(&Expr::Const(Constant::ComplexInfinity)),
            Err(PrintingError::UnsupportedPythonConstant(
                Constant::ComplexInfinity
            ))
        );
        assert_eq!(
            to_c_code(&Expr::Const(Constant::I)),
            Err(PrintingError::UnsupportedCConstant(Constant::I))
        );

        let huge = Expr::Integer(BigInt::from(i64::MAX) + BigInt::from(1));
        assert_eq!(
            to_c_code(&huge),
            Err(PrintingError::CNumericValueOutOfRange)
        );
    }
}
