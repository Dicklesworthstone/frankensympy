//! Automatic evaluation of elementary functions at construction time.
//!
//! This is the native counterpart of the `eval` classmethods of the pinned
//! upstream elementary functions (`sin`, `cos`, `tan`, `cot`, `sec`, `csc`,
//! their inverses, the hyperbolic family, `exp`, `log`, `Abs`, `sign`). Every
//! rule here is an exact identity on the principal branch:
//!
//! * special values at rational multiples of `pi` with the upstream radical
//!   forms (denominators 1, 2, 3, 4, 5, 6, 8, 10, 12);
//! * reduction of any rational multiple of `pi` into `[0, pi/2]` with the
//!   resulting sign (`sin(8*pi/7) -> -sin(pi/7)`);
//! * peeling integer multiples of `pi/2` off an `Add` argument
//!   (`sin(x + pi/2) -> cos(x)`);
//! * parity (`sin(-x) -> -sin(x)`, `cos(-x) -> cos(x)`) when the argument
//!   carries an unambiguous minus sign;
//! * imaginary arguments (`sin(I*x) -> I*sinh(x)`);
//! * direct inverse compositions (`sin(asin(x)) -> x`, `exp(log(x)) -> x`);
//! * `exp` at `I*pi*q` for half-integer `q`, `exp(c*log(x)) -> x**c`;
//! * `log` of exact numbers on the principal branch.
//!
//! Anything not covered returns `None` and the caller keeps the unevaluated
//! application. No rule consults floating point.

#![forbid(unsafe_code)]

use crate::{BigInt, BigRational, Constant, Expr};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;

fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

/// Exact numeric literal value.
fn number(e: &Expr) -> Option<BigRational> {
    match e {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Canonical literal for an exact rational.
pub fn rational_expr(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Integer(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

fn floor_rat(r: &BigRational) -> BigInt {
    let n = r.numer();
    let d = r.denom();
    let q = n.clone() / d.clone();
    let back = q.clone() * d.clone();
    if &back != n && n.is_negative() {
        q - BigInt::from(1)
    } else {
        q
    }
}

/// `r mod m` in `[0, m)` for positive `m`.
fn mod_rat(r: &BigRational, m: &BigRational) -> BigRational {
    let q = floor_rat(&(r.clone() / m.clone()));
    r.clone() - m.clone() * BigRational::from_integer(q)
}

fn pi() -> Expr {
    Expr::Const(Constant::Pi)
}

fn imag() -> Expr {
    Expr::Const(Constant::I)
}

fn zoo() -> Expr {
    Expr::Const(Constant::ComplexInfinity)
}

fn sqrt_int(n: i64) -> Expr {
    Expr::Pow(
        Arc::new(Expr::from_i64(n)),
        Arc::new(Expr::Rational(rat(1, 2))),
    )
}

fn sqrt_of(e: Expr) -> Expr {
    Expr::Pow(Arc::new(e), Arc::new(Expr::Rational(rat(1, 2))))
}

fn q(n: i64, d: i64) -> Expr {
    rational_expr(rat(n, d))
}

/// `r * pi` in canonical product form.
fn pi_times(r: BigRational) -> Expr {
    if r.is_zero() {
        return Expr::from_i64(0);
    }
    rational_expr(r) * pi()
}

/// Coefficient `c` when `e == c*pi` for an exact rational `c`.
fn pi_coeff(e: &Expr) -> Option<BigRational> {
    match e {
        Expr::Const(Constant::Pi) => Some(BigRational::one()),
        Expr::Mul(factors) if factors.len() == 2 => match (&factors[0], &factors[1]) {
            (c, Expr::Const(Constant::Pi)) | (Expr::Const(Constant::Pi), c) => number(c),
            _ => None,
        },
        _ if e.is_zero() => Some(BigRational::zero()),
        _ => None,
    }
}

/// `Some(rest)` when `e == I*rest` with `rest` free of a second `I` factor.
fn imaginary_coeff(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Const(Constant::I) => Some(Expr::from_i64(1)),
        Expr::Mul(factors) => {
            let count = factors
                .iter()
                .filter(|f| matches!(f, Expr::Const(Constant::I)))
                .count();
            if count != 1 {
                return None;
            }
            let rest: Vec<Expr> = factors
                .iter()
                .filter(|f| !matches!(f, Expr::Const(Constant::I)))
                .cloned()
                .collect();
            Some(rest.into_iter().fold(Expr::from_i64(1), |acc, f| acc * f))
        }
        _ => None,
    }
}

fn is_negative_term(e: &Expr) -> bool {
    match e {
        Expr::Integer(n) => n.is_negative(),
        Expr::Rational(r) => r.is_negative(),
        Expr::Mul(factors) => factors.first().is_some_and(|f| match f {
            Expr::Integer(n) => n.is_negative(),
            Expr::Rational(r) => r.is_negative(),
            _ => false,
        }),
        Expr::Const(Constant::NegativeInfinity) => true,
        _ => false,
    }
}

/// `Some(-e)` when `e` carries an unambiguous leading minus sign.
///
/// Matches upstream `could_extract_minus_sign` for numbers, products with a
/// negative coefficient, and sums where strictly more terms are negative than
/// positive. Balanced sums are left alone (upstream breaks that tie by its
/// printing order, which is not decided natively).
pub fn extract_minus_sign(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Mul(_) | Expr::Const(_) => {
            if is_negative_term(e) {
                Some(negate(e))
            } else {
                None
            }
        }
        Expr::Add(terms) => {
            let negative = terms.iter().filter(|t| is_negative_term(t)).count();
            if 2 * negative > terms.len() {
                Some(negate(e))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Exact negation in canonical form.
pub fn negate(e: &Expr) -> Expr {
    match e {
        Expr::Integer(n) => Expr::Integer(-n.clone()),
        Expr::Rational(r) => Expr::Rational(-r.clone()),
        Expr::Const(Constant::Infinity) => Expr::Const(Constant::NegativeInfinity),
        Expr::Const(Constant::NegativeInfinity) => Expr::Const(Constant::Infinity),
        Expr::Add(terms) => terms
            .iter()
            .map(negate)
            .fold(Expr::from_i64(0), |acc, t| acc + t),
        other => other.clone() * Expr::from_i64(-1),
    }
}

/// Splits an `Add` into its rational-multiple-of-`pi` part and the rest.
fn split_pi_terms(terms: &[Expr]) -> (BigRational, Vec<Expr>) {
    let mut coeff = BigRational::zero();
    let mut rest = Vec::new();
    for t in terms {
        match pi_coeff(t) {
            Some(c) if !t.is_zero() => coeff += c,
            _ => rest.push(t.clone()),
        }
    }
    (coeff, rest)
}

fn sum(terms: Vec<Expr>) -> Expr {
    terms.into_iter().fold(Expr::from_i64(0), |acc, t| acc + t)
}

fn func(name: &str, arg: Expr) -> Expr {
    eval_function(name, std::slice::from_ref(&arg))
        .unwrap_or_else(|| Expr::Function(name.to_string(), vec![arg]))
}

// ---------------------------------------------------------------------------
// Special-value tables (upstream radical forms).
// ---------------------------------------------------------------------------

/// `cos(r*pi)` for `r` in `[0, 1/2]` when upstream has a closed form.
fn cos_table(r: &BigRational) -> Option<Expr> {
    let key = (r.numer().to_i64()?, r.denom().to_i64()?);
    Some(match key {
        (0, 1) => Expr::from_i64(1),
        (1, 2) => Expr::from_i64(0),
        (1, 3) => q(1, 2),
        (1, 4) => q(1, 2) * sqrt_int(2),
        (1, 6) => q(1, 2) * sqrt_int(3),
        (1, 5) => q(1, 4) + q(1, 4) * sqrt_int(5),
        (2, 5) => q(-1, 4) + q(1, 4) * sqrt_int(5),
        (1, 8) => sqrt_of(q(1, 4) * sqrt_int(2) + q(1, 2)),
        (3, 8) => sqrt_of(q(1, 2) + q(-1, 4) * sqrt_int(2)),
        (1, 10) => sqrt_of(q(1, 8) * sqrt_int(5) + q(5, 8)),
        (3, 10) => sqrt_of(q(5, 8) + q(-1, 8) * sqrt_int(5)),
        (1, 12) => q(1, 4) * sqrt_int(2) + q(1, 4) * sqrt_int(6),
        (5, 12) => q(-1, 4) * sqrt_int(2) + q(1, 4) * sqrt_int(6),
        _ => return None,
    })
}

/// `tan(r*pi)` for `r` in `[0, 1/2]`.
fn tan_table(r: &BigRational) -> Option<Expr> {
    let key = (r.numer().to_i64()?, r.denom().to_i64()?);
    Some(match key {
        (0, 1) => Expr::from_i64(0),
        (1, 2) => zoo(),
        (1, 3) => sqrt_int(3),
        (1, 4) => Expr::from_i64(1),
        (1, 6) => q(1, 3) * sqrt_int(3),
        (1, 5) => sqrt_of(Expr::from_i64(5) + Expr::from_i64(-2) * sqrt_int(5)),
        (2, 5) => sqrt_of(Expr::from_i64(2) * sqrt_int(5) + Expr::from_i64(5)),
        (1, 8) => Expr::from_i64(-1) + sqrt_int(2),
        (3, 8) => Expr::from_i64(1) + sqrt_int(2),
        (1, 10) => sqrt_of(Expr::from_i64(1) + q(-2, 5) * sqrt_int(5)),
        (3, 10) => sqrt_of(q(2, 5) * sqrt_int(5) + Expr::from_i64(1)),
        (1, 12) => Expr::from_i64(2) + Expr::from_i64(-1) * sqrt_int(3),
        (5, 12) => sqrt_int(3) + Expr::from_i64(2),
        _ => return None,
    })
}

/// `sec(r*pi)` for `r` in `[0, 1/2]`.
fn sec_table(r: &BigRational) -> Option<Expr> {
    let key = (r.numer().to_i64()?, r.denom().to_i64()?);
    match key {
        (0, 1) => return Some(Expr::from_i64(1)),
        (1, 2) => return Some(zoo()),
        (1, 3) => return Some(Expr::from_i64(2)),
        (1, 4) => return Some(sqrt_int(2)),
        (1, 6) => return Some(q(2, 3) * sqrt_int(3)),
        _ => {}
    }
    let c = cos_table(r)?;
    Some(match c {
        Expr::Pow(base, exp) if *exp == Expr::Rational(rat(1, 2)) => {
            Expr::Pow(base, Arc::new(Expr::Rational(rat(-1, 2))))
        }
        other => Expr::Pow(Arc::new(other), Arc::new(Expr::from_i64(-1))),
    })
}

fn signed(negative: bool, e: Expr) -> Expr {
    if negative { negate(&e) } else { e }
}

/// Evaluates `name(c*pi)` for an exact rational `c`.
///
/// Returns the reduced form `±name(r*pi)` when no closed form exists and the
/// reduction changed the argument; `None` when the argument is already
/// reduced and has no closed form.
fn eval_trig_pi(name: &str, c: &BigRational) -> Option<Expr> {
    let half = rat(1, 2);
    let one = BigRational::one();
    let two = BigRational::from_integer(BigInt::from(2));
    // Reduce into [0, 1/2] with a sign; `cofunction` selects the table.
    let (negative, r) = match name {
        "sin" | "csc" => {
            let mut r = mod_rat(c, &two);
            let mut neg = false;
            if r >= one {
                neg = true;
                r -= one.clone();
            }
            if r > half {
                r = one.clone() - r;
            }
            (neg, r)
        }
        "cos" | "sec" => {
            let mut r = mod_rat(c, &two);
            let mut neg = false;
            if r > one {
                r = two.clone() - r;
            }
            if r > half {
                neg = true;
                r = one.clone() - r;
            }
            (neg, r)
        }
        "tan" | "cot" => {
            let mut r = mod_rat(c, &one);
            let mut neg = false;
            if r > half {
                neg = true;
                r = one.clone() - r;
            }
            (neg, r)
        }
        _ => return None,
    };
    let value = match name {
        "sin" => cos_table(&(half.clone() - r.clone())),
        "cos" => cos_table(&r),
        "tan" => tan_table(&r),
        "cot" => tan_table(&(half.clone() - r.clone())),
        "sec" => sec_table(&r),
        "csc" => sec_table(&(half.clone() - r.clone())),
        _ => None,
    };
    if let Some(v) = value {
        if negative && v == zoo() {
            return Some(v);
        }
        return Some(signed(negative, v));
    }
    if !negative && &r == c {
        return None;
    }
    Some(signed(
        negative,
        Expr::Function(name.to_string(), vec![pi_times(r)]),
    ))
}

/// Peels integer multiples of `pi/2` off an `Add` argument.
fn peel_pi_half(name: &str, terms: &[Expr]) -> Option<Expr> {
    let (k, rest) = split_pi_terms(terms);
    if k.is_zero() {
        return None;
    }
    let half = rat(1, 2);
    let m = k.clone() - mod_rat(&k, &half);
    if m.is_zero() {
        return None;
    }
    let x = sum(rest) + pi_times(k - m.clone());
    // Index of m in units of pi/2 (mod 4).
    let idx = mod_rat(
        &(m * BigRational::from_integer(BigInt::from(2))),
        &BigRational::from_integer(BigInt::from(4)),
    )
    .to_integer()
    .to_i64()?;
    Some(match (name, idx) {
        ("sin", 0) => func("sin", x),
        ("sin", 1) => func("cos", x),
        ("sin", 2) => negate(&func("sin", x)),
        ("sin", 3) => negate(&func("cos", x)),
        ("cos", 0) => func("cos", x),
        ("cos", 1) => negate(&func("sin", x)),
        ("cos", 2) => negate(&func("cos", x)),
        ("cos", 3) => func("sin", x),
        ("tan", 0 | 2) => func("tan", x),
        ("tan", 1 | 3) => negate(&func("cot", x)),
        ("cot", 0 | 2) => func("cot", x),
        ("cot", 1 | 3) => negate(&func("tan", x)),
        ("sec", 0) => func("sec", x),
        ("sec", 2) => negate(&func("sec", x)),
        ("csc", 0) => func("csc", x),
        ("csc", 2) => negate(&func("csc", x)),
        _ => return None,
    })
}

fn inverse_of(arg: &Expr, inverse: &str) -> Option<Expr> {
    match arg {
        Expr::Function(name, args) if name == inverse && args.len() == 1 => Some(args[0].clone()),
        _ => None,
    }
}

fn eval_circular(name: &str, arg: &Expr) -> Option<Expr> {
    let odd = matches!(name, "sin" | "tan" | "cot" | "csc");
    if matches!(arg, Expr::Const(Constant::NaN)) || *arg == zoo() {
        return Some(Expr::Const(Constant::NaN));
    }
    if let Some(neg) = extract_minus_sign(arg) {
        let inner = func(name, neg);
        return Some(if odd { negate(&inner) } else { inner });
    }
    if let Some(i) = imaginary_coeff(arg) {
        return Some(match name {
            "sin" => imag() * func("sinh", i),
            "cos" => func("cosh", i),
            "tan" => imag() * func("tanh", i),
            "cot" => negate(&(imag() * func("coth", i))),
            "sec" => func("sech", i),
            "csc" => negate(&(imag() * func("csch", i))),
            _ => return None,
        });
    }
    if let Some(c) = pi_coeff(arg) {
        return eval_trig_pi(name, &c);
    }
    if let Expr::Add(terms) = arg
        && let Some(v) = peel_pi_half(name, terms)
    {
        return Some(v);
    }
    let inv = match name {
        "sin" => "asin",
        "cos" => "acos",
        "tan" => "atan",
        "cot" => "acot",
        "sec" => "asec",
        "csc" => "acsc",
        _ => return None,
    };
    inverse_of(arg, inv)
}

fn eval_hyperbolic(name: &str, arg: &Expr) -> Option<Expr> {
    let odd = matches!(name, "sinh" | "tanh" | "coth" | "csch");
    if let Some(neg) = extract_minus_sign(arg) {
        let inner = func(name, neg);
        return Some(if odd { negate(&inner) } else { inner });
    }
    if let Some(i) = imaginary_coeff(arg) {
        return Some(match name {
            "sinh" => imag() * func("sin", i),
            "cosh" => func("cos", i),
            "tanh" => imag() * func("tan", i),
            "coth" => negate(&(imag() * func("cot", i))),
            "sech" => func("sec", i),
            "csch" => negate(&(imag() * func("csc", i))),
            _ => return None,
        });
    }
    let inv = match name {
        "sinh" => "asinh",
        "cosh" => "acosh",
        "tanh" => "atanh",
        "coth" => "acoth",
        _ => return None,
    };
    inverse_of(arg, inv)
}

/// Recognizes the canonical radicals used by the inverse tables.
fn radical_key(arg: &Expr) -> Option<(bool, &'static str)> {
    let (negative, magnitude) = match extract_minus_sign(arg) {
        Some(m) => (true, m),
        None => (false, arg.clone()),
    };
    let key = if magnitude == Expr::from_i64(1) {
        "1"
    } else if magnitude == q(1, 2) {
        "1/2"
    } else if magnitude == q(1, 2) * sqrt_int(2) {
        "sqrt2/2"
    } else if magnitude == q(1, 2) * sqrt_int(3) {
        "sqrt3/2"
    } else if magnitude == sqrt_int(3) {
        "sqrt3"
    } else if magnitude == q(1, 3) * sqrt_int(3) {
        "sqrt3/3"
    } else if magnitude == Expr::Const(Constant::Infinity) {
        "oo"
    } else {
        return None;
    };
    Some((negative, key))
}

fn eval_inverse_circular(name: &str, arg: &Expr) -> Option<Expr> {
    if let Some((negative, key)) = radical_key(arg) {
        let r = match (name, key) {
            ("asin", "1") => Some(rat(1, 2)),
            ("asin", "1/2") => Some(rat(1, 6)),
            ("asin", "sqrt2/2") => Some(rat(1, 4)),
            ("asin", "sqrt3/2") => Some(rat(1, 3)),
            ("atan", "1") => Some(rat(1, 4)),
            ("atan", "sqrt3") => Some(rat(1, 3)),
            ("atan", "sqrt3/3") => Some(rat(1, 6)),
            ("atan", "oo") => Some(rat(1, 2)),
            ("acot", "1") => Some(rat(1, 4)),
            ("acot", "sqrt3") => Some(rat(1, 6)),
            ("acot", "sqrt3/3") => Some(rat(1, 3)),
            ("acot", "oo") => Some(BigRational::zero()),
            _ => None,
        };
        if let Some(r) = r {
            return Some(pi_times(if negative { -r } else { r }));
        }
        let r = match (name, key) {
            ("acos", "1") => Some(BigRational::zero()),
            ("acos", "1/2") => Some(rat(1, 3)),
            ("acos", "sqrt2/2") => Some(rat(1, 4)),
            ("acos", "sqrt3/2") => Some(rat(1, 6)),
            _ => None,
        };
        if let Some(r) = r {
            return Some(pi_times(if negative { BigRational::one() - r } else { r }));
        }
    }
    if arg.is_zero() {
        return match name {
            "asin" | "atan" => Some(Expr::from_i64(0)),
            "acos" | "acot" => Some(pi_times(rat(1, 2))),
            _ => None,
        };
    }
    if matches!(name, "asin" | "atan" | "acot")
        && let Some(neg) = extract_minus_sign(arg)
    {
        return Some(negate(&func(name, neg)));
    }
    None
}

fn eval_inverse_hyperbolic(name: &str, arg: &Expr) -> Option<Expr> {
    if matches!(name, "asinh" | "atanh" | "acoth" | "acsch")
        && let Some(neg) = extract_minus_sign(arg)
    {
        return Some(negate(&func(name, neg)));
    }
    None
}

/// `Some(x)` when `e == log(x)`.
fn log_arg(e: &Expr) -> Option<&Expr> {
    match e {
        Expr::Function(name, args) if name == "log" && args.len() == 1 => Some(&args[0]),
        _ => None,
    }
}

/// `x**c` for `c*log(x)` with exact rational `c`.
fn log_power(e: &Expr) -> Option<Expr> {
    if let Some(x) = log_arg(e) {
        return Some(x.clone());
    }
    if let Expr::Mul(factors) = e
        && factors.len() == 2
        && let Some(c) = number(&factors[0])
        && let Some(x) = log_arg(&factors[1])
    {
        return Some(power(x.clone(), rational_expr(c)));
    }
    None
}

/// Canonical power with exact folding of literal bases.
fn power(base: Expr, exp: Expr) -> Expr {
    if exp.is_one() {
        return base;
    }
    if exp.is_zero() {
        return Expr::from_i64(1);
    }
    if let (Some(b), Expr::Integer(e)) = (number(&base), &exp)
        && let Some(e) = e.to_i64()
        && let Ok(e) = i32::try_from(e)
        && let Ok(v) = b.pow(e)
    {
        return rational_expr(v);
    }
    match base {
        Expr::Pow(b, inner) => match (number(&inner), number(&exp)) {
            (Some(a), Some(c)) if exp_is_integer(&exp) => {
                Expr::Pow(b, Arc::new(rational_expr(a * c)))
            }
            _ => Expr::Pow(Arc::new(Expr::Pow(b, inner)), Arc::new(exp)),
        },
        other => Expr::Pow(Arc::new(other), Arc::new(exp)),
    }
}

fn exp_is_integer(e: &Expr) -> bool {
    matches!(e, Expr::Integer(_))
}

fn eval_exp(arg: &Expr) -> Option<Expr> {
    match arg {
        Expr::Const(Constant::Infinity) => return Some(Expr::Const(Constant::Infinity)),
        Expr::Const(Constant::NegativeInfinity) => return Some(Expr::from_i64(0)),
        Expr::Const(Constant::ComplexInfinity) | Expr::Const(Constant::NaN) => {
            return Some(Expr::Const(Constant::NaN));
        }
        _ => {}
    }
    if arg.is_one() {
        return Some(Expr::Const(Constant::E));
    }
    if let Some(p) = log_power(arg) {
        return Some(p);
    }
    // exp(I*pi*c) for half-integer c.
    if let Some(i) = imaginary_coeff(arg)
        && let Some(c) = pi_coeff(&i)
    {
        let doubled = c * BigRational::from_integer(BigInt::from(2));
        if doubled.is_integer() {
            let k = mod_rat(&doubled, &BigRational::from_integer(BigInt::from(4)))
                .to_integer()
                .to_i64()?;
            return Some(match k {
                0 => Expr::from_i64(1),
                1 => imag(),
                2 => Expr::from_i64(-1),
                _ => negate(&imag()),
            });
        }
        return None;
    }
    if let Expr::Add(terms) = arg {
        let mut out: Vec<Expr> = Vec::new();
        let mut kept: Vec<Expr> = Vec::new();
        for t in terms {
            if t.is_one() {
                kept.push(t.clone());
                continue;
            }
            match eval_exp(t) {
                Some(v) if !matches!(&v, Expr::Function(n, _) if n == "exp") => out.push(v),
                Some(Expr::Function(_, args)) => kept.push(args[0].clone()),
                _ => kept.push(t.clone()),
            }
        }
        if !out.is_empty() {
            let rest = sum(kept);
            let factor = func("exp", rest);
            return Some(out.into_iter().fold(factor, |acc, f| acc * f));
        }
    }
    None
}

fn eval_log(arg: &Expr) -> Option<Expr> {
    match arg {
        Expr::Const(Constant::E) => return Some(Expr::from_i64(1)),
        Expr::Const(Constant::Infinity) | Expr::Const(Constant::NegativeInfinity) => {
            return Some(Expr::Const(Constant::Infinity));
        }
        Expr::Const(Constant::ComplexInfinity) => return Some(Expr::Const(Constant::Infinity)),
        Expr::Const(Constant::NaN) => return Some(Expr::Const(Constant::NaN)),
        Expr::Const(Constant::I) => return Some(q(1, 2) * imag() * pi()),
        _ => {}
    }
    if arg.is_zero() {
        return Some(zoo());
    }
    if arg.is_one() {
        return Some(Expr::from_i64(0));
    }
    // log(exp(c)) = c for an exact real literal c.
    if let Expr::Function(name, args) = arg
        && name == "exp"
        && args.len() == 1
        && number(&args[0]).is_some()
    {
        return Some(args[0].clone());
    }
    if let Some(r) = number(arg) {
        if r.is_negative() {
            let magnitude = rational_expr(-r);
            return Some(func("log", magnitude) + imag() * pi());
        }
        if r.numer().is_one() {
            return Some(negate(&func("log", Expr::Integer(r.denom().clone()))));
        }
        return None;
    }
    // log(b*I) for an exact nonzero rational b.
    if let Some(b) = imaginary_coeff(arg)
        && let Some(r) = number(&b)
    {
        let half_pi = q(1, 2) * imag() * pi();
        let magnitude = rational_expr(r.abs());
        let lead = func("log", magnitude);
        return Some(if r.is_negative() {
            lead + negate(&half_pi)
        } else {
            lead + half_pi
        });
    }
    None
}

/// Exact square root of a non-negative rational in upstream canonical form.
pub fn sqrt_rational(r: &BigRational) -> Expr {
    if let Some(root) = r.exact_sqrt() {
        return rational_expr(root);
    }
    // sqrt(p/q) = sqrt(p*q)/q
    let pq = r.numer().clone() * r.denom().clone();
    let (outside, inside) = split_square(&pq);
    let coeff = BigRational::new(outside, r.denom().clone());
    let radical = Expr::Pow(
        Arc::new(Expr::Integer(inside)),
        Arc::new(Expr::Rational(rat(1, 2))),
    );
    if coeff.is_one() {
        radical
    } else {
        rational_expr(coeff) * radical
    }
}

/// `n = a^2 * b` with trial division over small primes.
fn split_square(n: &BigInt) -> (BigInt, BigInt) {
    let mut a = BigInt::from(1);
    let mut b = n.clone();
    let mut p: i64 = 2;
    while p <= 10_000 {
        let pp = BigInt::from(p * p);
        if pp > b {
            break;
        }
        loop {
            let qd = b.clone() / pp.clone();
            if qd.clone() * pp.clone() == b {
                b = qd;
                a *= BigInt::from(p);
            } else {
                break;
            }
        }
        p += if p == 2 { 1 } else { 2 };
    }
    (a, b)
}

fn is_positive_constant(e: &Expr) -> bool {
    matches!(e, Expr::Const(Constant::Pi | Constant::E))
        || matches!(e, Expr::Pow(b, x) if matches!(**b, Expr::Integer(ref n) if n.is_positive()) && number(x).is_some())
}

fn eval_abs(arg: &Expr) -> Option<Expr> {
    if let Some(r) = number(arg) {
        return Some(rational_expr(r.abs()));
    }
    if is_positive_constant(arg) {
        return Some(arg.clone());
    }
    match arg {
        Expr::Const(Constant::I) => return Some(Expr::from_i64(1)),
        Expr::Const(
            Constant::Infinity | Constant::NegativeInfinity | Constant::ComplexInfinity,
        ) => {
            return Some(Expr::Const(Constant::Infinity));
        }
        Expr::Const(Constant::NaN) => return Some(Expr::Const(Constant::NaN)),
        _ => {}
    }
    // Gaussian rationals a + b*I.
    if let Expr::Add(terms) = arg
        && terms.len() == 2
    {
        let mut re = None;
        let mut im = None;
        for t in terms {
            if let Some(r) = number(t) {
                re = Some(r);
            } else if let Some(b) = imaginary_coeff(t).and_then(|b| number(&b)) {
                im = Some(b);
            }
        }
        if let (Some(a), Some(b)) = (re, im) {
            return Some(sqrt_rational(&(a.clone() * a + b.clone() * b)));
        }
    }
    if let Expr::Mul(factors) = arg {
        let mut outside = Expr::from_i64(1);
        let mut inside: Vec<Expr> = Vec::new();
        let mut changed = false;
        for f in factors {
            if let Some(r) = number(f) {
                changed = true;
                outside = outside * rational_expr(r.abs());
            } else if matches!(f, Expr::Const(Constant::I)) {
                changed = true;
            } else if is_positive_constant(f) {
                changed = true;
                outside = outside * f.clone();
            } else {
                inside.push(f.clone());
            }
        }
        if changed {
            if inside.is_empty() {
                return Some(outside);
            }
            let rest = inside.into_iter().fold(Expr::from_i64(1), |acc, f| acc * f);
            return Some(outside * func("Abs", rest));
        }
    }
    None
}

fn eval_sign(arg: &Expr) -> Option<Expr> {
    if let Some(r) = number(arg) {
        return Some(Expr::from_i64(if r.is_zero() {
            0
        } else if r.is_negative() {
            -1
        } else {
            1
        }));
    }
    if is_positive_constant(arg) {
        return Some(Expr::from_i64(1));
    }
    match arg {
        Expr::Const(Constant::I) => return Some(imag()),
        Expr::Const(Constant::Infinity) => return Some(Expr::from_i64(1)),
        Expr::Const(Constant::NegativeInfinity) => return Some(Expr::from_i64(-1)),
        Expr::Const(Constant::NaN) => return Some(Expr::Const(Constant::NaN)),
        _ => {}
    }
    if let Expr::Mul(factors) = arg {
        let mut sign = Expr::from_i64(1);
        let mut inside: Vec<Expr> = Vec::new();
        let mut changed = false;
        for f in factors {
            if let Some(r) = number(f) {
                changed = true;
                if r.is_negative() {
                    sign = negate(&sign);
                }
            } else if matches!(f, Expr::Const(Constant::I)) {
                changed = true;
                sign = sign * imag();
            } else if is_positive_constant(f) {
                changed = true;
            } else {
                inside.push(f.clone());
            }
        }
        if changed {
            if inside.is_empty() {
                return Some(sign);
            }
            let rest = inside.into_iter().fold(Expr::from_i64(1), |acc, f| acc * f);
            return Some(sign * func("sign", rest));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Power construction (upstream Pow.__new__ / _eval_power subset).
// ---------------------------------------------------------------------------

fn raw_pow(base: Expr, exp: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exp))
}

/// Exact integer `n`-th root when `m` is a perfect `n`-th power.
fn integer_nthroot(m: &BigInt, n: u32) -> Option<BigInt> {
    if m.is_negative() {
        return None;
    }
    if m.is_zero() || m.is_one() || n == 1 {
        return Some(m.clone());
    }
    let f = m.to_f64()?;
    let guess = f.powf(1.0 / f64::from(n)).round();
    if !guess.is_finite() {
        return None;
    }
    let g = BigInt::from(guess as i64);
    [g.clone() - BigInt::from(1), g.clone(), g + BigInt::from(1)]
        .into_iter()
        .find(|cand| !cand.is_negative() && cand.pow(n) == *m)
}

/// Trial-division factorization of `m > 0`; the unfactored cofactor (if any)
/// is returned as one entry with multiplicity 1.
fn small_factors(m: &BigInt) -> Vec<(BigInt, u32)> {
    let mut out = Vec::new();
    let mut rest = m.clone();
    let mut p: i64 = 2;
    while p <= 32_768 {
        let bp = BigInt::from(p);
        if bp.clone() * bp.clone() > rest {
            break;
        }
        let mut k = 0u32;
        loop {
            let q = rest.clone() / bp.clone();
            if q.clone() * bp.clone() == rest {
                rest = q;
                k += 1;
            } else {
                break;
            }
        }
        if k > 0 {
            out.push((bp, k));
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if !rest.is_one() {
        out.push((rest, 1));
    }
    out
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd_u64(b, a % b) }
}

/// `m**(p/q)` for an Integer `m` and a non-integer rational exponent.
fn integer_rational_power(m: &BigInt, e: &BigRational) -> Expr {
    let p = e.numer().clone();
    let qd = e.denom().clone();
    let half = rat(1, 2);
    if m.is_negative() && *e == half {
        let pos = eval_pow(Expr::Integer(-m.clone()), Expr::Rational(half));
        return pos * imag();
    }
    if e.is_negative() {
        let ne = -e.clone();
        if m.is_negative() {
            let sign = eval_pow(Expr::from_i64(-1), Expr::Rational(e.clone()));
            let mag = rational_power(&BigRational::new(BigInt::from(1), -m.clone()), &ne);
            return sign * mag;
        }
        return rational_power(&BigRational::new(BigInt::from(1), m.clone()), &ne);
    }
    let (Some(pi64), Some(qi64)) = (p.to_i64(), qd.to_i64()) else {
        return raw_pow(Expr::Integer(m.clone()), Expr::Rational(e.clone()));
    };
    let (Ok(pu), Ok(qu)) = (u32::try_from(pi64), u32::try_from(qi64)) else {
        return raw_pow(Expr::Integer(m.clone()), Expr::Rational(e.clone()));
    };
    let mag = m.abs();
    let sign_part = || {
        if m.is_negative() {
            eval_pow(Expr::from_i64(-1), Expr::Rational(e.clone()))
        } else {
            Expr::from_i64(1)
        }
    };
    if let Some(root) = integer_nthroot(&mag, qu) {
        return Expr::Integer(root.pow(pu)) * sign_part();
    }
    if mag.bits() > 4096 {
        return raw_pow(Expr::Integer(m.clone()), Expr::Rational(e.clone()));
    }
    let factors = small_factors(&mag);
    let mut out_int = BigInt::from(1);
    let mut out_rad = Expr::from_i64(1);
    let mut sqr: Vec<(BigInt, u64)> = Vec::new();
    for (prime, mult) in factors {
        let exponent = u64::from(mult) * u64::from(pu);
        let div_e = exponent / u64::from(qu);
        let div_m = exponent % u64::from(qu);
        if div_e > 0 {
            out_int *= prime.pow(div_e as u32);
        }
        if div_m > 0 {
            let g = gcd_u64(div_m, u64::from(qu));
            if g != 1 {
                out_rad = out_rad
                    * raw_pow(
                        Expr::Integer(prime),
                        rational_expr(BigRational::new(
                            BigInt::from(div_m / g),
                            BigInt::from(u64::from(qu) / g),
                        )),
                    );
            } else {
                sqr.push((prime, div_m));
            }
        }
    }
    let mut sqr_gcd = 0u64;
    for (_, ex) in &sqr {
        sqr_gcd = if sqr_gcd == 0 {
            *ex
        } else {
            gcd_u64(sqr_gcd, *ex)
        };
    }
    let mut sqr_int = BigInt::from(1);
    for (k, v) in &sqr {
        sqr_int *= k.pow((*v / sqr_gcd.max(1)) as u32);
    }
    if sqr_int == mag && out_int.is_one() && out_rad.is_one() {
        return raw_pow(Expr::Integer(m.clone()), Expr::Rational(e.clone()));
    }
    let mut result = Expr::Integer(out_int) * out_rad;
    if sqr_gcd > 0 {
        result = result
            * raw_pow(
                Expr::Integer(sqr_int),
                rational_expr(BigRational::new(BigInt::from(sqr_gcd), qd.clone())),
            );
    }
    result * sign_part()
}

/// `r**e` for an exact rational base and non-integer rational exponent.
fn rational_power(r: &BigRational, e: &BigRational) -> Expr {
    if r.is_integer() {
        return integer_rational_power(&r.to_integer(), e);
    }
    let n = r.numer().clone();
    let d = r.denom().clone();
    if !n.is_one() {
        return integer_rational_power(&n, e) * integer_rational_power(&d, &(-e.clone()));
    }
    // (1/d)**(p/q) = d**(p*(q-1)/q) / d**p
    let p = e.numer().clone();
    let q = e.denom().clone();
    let num = BigRational::new(p.clone() * (q.clone() - BigInt::from(1)), q);
    let Some(pi) = p.to_i64().and_then(|v| i32::try_from(v).ok()) else {
        return raw_pow(Expr::Rational(r.clone()), Expr::Rational(e.clone()));
    };
    let Ok(den) = BigRational::from_integer(d.clone()).pow(pi) else {
        return raw_pow(Expr::Rational(r.clone()), Expr::Rational(e.clone()));
    };
    let lead = if num.is_integer() {
        rational_expr(
            BigRational::from_integer(d.clone())
                .pow(
                    num.to_integer()
                        .to_i64()
                        .and_then(|v| i32::try_from(v).ok())
                        .unwrap_or(1),
                )
                .unwrap_or_else(|_| BigRational::one()),
        )
    } else {
        integer_rational_power(&d, &num)
    };
    lead * rational_expr(den.recip())
}

fn is_nonnegative_factor(e: &Expr) -> bool {
    match e {
        Expr::Integer(n) => !n.is_negative(),
        Expr::Rational(r) => !r.is_negative(),
        Expr::Const(Constant::Pi | Constant::E) => true,
        // Declared facts in the active assumption scope.
        other => matches!(crate::assume::sign(other), Some(0 | 1)),
    }
}

/// `(a, b)` when `e == a + b*I` with exact rationals a, b.
fn gaussian_parts(e: &Expr) -> Option<(BigRational, BigRational)> {
    match e {
        Expr::Const(Constant::I) => Some((BigRational::zero(), BigRational::one())),
        Expr::Add(terms) => {
            let mut re = BigRational::zero();
            let mut im = BigRational::zero();
            for t in terms {
                if let Some(v) = number(t) {
                    re += v;
                } else {
                    im += imaginary_coeff(t).and_then(|c| number(&c))?;
                }
            }
            Some((re, im))
        }
        other => imaginary_coeff(other)
            .and_then(|c| number(&c))
            .map(|c| (BigRational::zero(), c)),
    }
}

/// Canonical `base**exp` (upstream automatic power evaluation).
pub fn eval_pow(base: Expr, exp: Expr) -> Expr {
    if exp.is_zero() {
        return Expr::from_i64(1);
    }
    if exp.is_one() {
        return base;
    }
    if matches!(base, Expr::Const(Constant::NaN)) || matches!(exp, Expr::Const(Constant::NaN)) {
        return Expr::Const(Constant::NaN);
    }
    if base.is_one() {
        if matches!(
            exp,
            Expr::Const(
                Constant::Infinity | Constant::NegativeInfinity | Constant::ComplexInfinity
            )
        ) {
            return Expr::Const(Constant::NaN);
        }
        return Expr::from_i64(1);
    }
    if matches!(base, Expr::Const(Constant::E)) {
        return func("exp", exp);
    }
    if let Some(bv) = number(&base) {
        if let Some(ev) = number(&exp) {
            if bv.is_zero() {
                return if ev.is_negative() {
                    zoo()
                } else {
                    Expr::from_i64(0)
                };
            }
            if ev.is_integer() {
                if let Some(k) = ev.to_integer().to_i64().and_then(|v| i32::try_from(v).ok())
                    && k.unsigned_abs() <= 100_000
                    && let Ok(v) = bv.pow(k)
                {
                    return rational_expr(v);
                }
                return raw_pow(base, exp);
            }
            if bv == -BigRational::one() {
                if ev.denom() == &BigInt::from(2) {
                    let k = mod_rat(
                        &BigRational::from_integer(ev.numer().clone()),
                        &BigRational::from_integer(BigInt::from(4)),
                    )
                    .to_integer()
                    .to_i64()
                    .unwrap_or(0);
                    return match k {
                        1 => imag(),
                        3 => negate(&imag()),
                        _ => raw_pow(base, exp),
                    };
                }
                return raw_pow(base, exp);
            }
            return rational_power(&bv, &ev);
        }
        if bv.is_zero() {
            if matches!(exp, Expr::Const(Constant::Infinity)) {
                return Expr::from_i64(0);
            }
            if matches!(exp, Expr::Const(Constant::NegativeInfinity)) {
                return zoo();
            }
        }
        // Negative bases to +-oo: |b| > 1 oscillates unboundedly (zoo),
        // |b| < 1 decays to 0 (upstream).
        if bv.is_negative() && bv != -BigRational::one() {
            let small = -bv.clone() < BigRational::one();
            match exp {
                Expr::Const(Constant::Infinity) => {
                    return if small { Expr::from_i64(0) } else { zoo() };
                }
                Expr::Const(Constant::NegativeInfinity) => {
                    return if small {
                        Expr::Const(Constant::NaN)
                    } else {
                        Expr::from_i64(0)
                    };
                }
                _ => {}
            }
        }
        // Positive rational bases to +-oo (upstream Pow._eval_power).
        if bv.is_positive() && !bv.is_one() {
            let small = bv < BigRational::one();
            match exp {
                Expr::Const(Constant::Infinity) => {
                    return if small {
                        Expr::from_i64(0)
                    } else {
                        Expr::Const(Constant::Infinity)
                    };
                }
                Expr::Const(Constant::NegativeInfinity) => {
                    return if small {
                        Expr::Const(Constant::Infinity)
                    } else {
                        Expr::from_i64(0)
                    };
                }
                _ => {}
            }
        }
        return raw_pow(base, exp);
    }
    // Negative integer powers of Gaussian rationals are evaluated:
    // (a + b*I)**-n = ((a - b*I)/(a^2 + b^2))**n, multiplied out exactly.
    if let Expr::Integer(k) = &exp
        && k.is_negative()
        && let Some((a, b)) = gaussian_parts(&base)
        && !b.is_zero()
        && let Some(n) = (-k.clone()).to_i64()
        && n <= 64
    {
        let norm = a.clone() * a.clone() + b.clone() * b.clone();
        let (wr, wi) = (a / norm.clone(), -b / norm);
        let (mut re, mut im) = (BigRational::one(), BigRational::zero());
        for _ in 0..n {
            let nr = re.clone() * wr.clone() - im.clone() * wi.clone();
            let ni = re * wi.clone() + im * wr.clone();
            re = nr;
            im = ni;
        }
        return rational_expr(re) + rational_expr(im) * imag();
    }
    match (&base, &exp) {
        (Expr::Const(Constant::I), Expr::Integer(k)) => {
            let m = mod_rat(
                &BigRational::from_integer(k.clone()),
                &BigRational::from_integer(BigInt::from(4)),
            )
            .to_integer()
            .to_i64()
            .unwrap_or(0);
            return match m {
                0 => Expr::from_i64(1),
                1 => imag(),
                2 => Expr::from_i64(-1),
                _ => negate(&imag()),
            };
        }
        (Expr::Const(Constant::Infinity), _) | (Expr::Const(Constant::ComplexInfinity), _) => {
            if let Some(ev) = number(&exp) {
                if ev.is_negative() {
                    return Expr::from_i64(0);
                }
                return base.clone();
            }
        }
        (Expr::Const(Constant::NegativeInfinity), Expr::Integer(k)) => {
            if k.is_negative() {
                return Expr::from_i64(0);
            }
            let odd = (k.clone() % BigInt::from(2)) != BigInt::from(0);
            return if odd {
                Expr::Const(Constant::NegativeInfinity)
            } else {
                Expr::Const(Constant::Infinity)
            };
        }
        (Expr::Pow(b, e1), _) => {
            // (b**e1)**e = b**(e1*e) for integer e, for |e1| < 1 with a
            // numeric e, and for any real exponents of a base that the
            // active facts prove positive.
            let positive_base = crate::assume::sign(b) == Some(1)
                && crate::assume::is_real(e1)
                && crate::assume::is_real(&exp);
            let combine = positive_base
                || matches!(exp, Expr::Integer(_))
                || number(e1).is_some_and(|v| v.abs() < BigRational::one())
                    && number(&exp).is_some();
            if combine {
                let product = (**e1).clone() * exp.clone();
                return eval_pow((**b).clone(), product);
            }
        }
        (Expr::Function(name, args), Expr::Integer(_)) if name == "exp" && args.len() == 1 => {
            return func("exp", args[0].clone() * exp.clone());
        }
        (Expr::Mul(factors), Expr::Integer(_)) => {
            return factors
                .iter()
                .map(|f| eval_pow(f.clone(), exp.clone()))
                .fold(Expr::from_i64(1), |acc, f| acc * f);
        }
        (Expr::Mul(factors), _) if number(&exp).is_some() => {
            // Split nonnegative numeric/constant factors out of a rational
            // power: sqrt(2*x) -> sqrt(2)*sqrt(x); sqrt(-2*x) -> sqrt(2)*sqrt(-x).
            let mut outside: Vec<Expr> = Vec::new();
            let mut inside: Vec<Expr> = Vec::new();
            for f in factors {
                match number(f) {
                    Some(v) if v.is_negative() && v != -BigRational::one() => {
                        outside.push(rational_expr(-v));
                        inside.push(Expr::from_i64(-1));
                    }
                    _ if is_nonnegative_factor(f) => outside.push(f.clone()),
                    _ => inside.push(f.clone()),
                }
            }
            if !outside.is_empty() {
                let rest = inside.into_iter().fold(Expr::from_i64(1), |acc, f| acc * f);
                let lead = outside
                    .into_iter()
                    .map(|f| eval_pow(f, exp.clone()))
                    .fold(Expr::from_i64(1), |acc, f| acc * f);
                return lead * eval_pow(rest, exp.clone());
            }
        }
        _ => {}
    }
    raw_pow(base, exp)
}

/// Bernoulli numbers B_0..B_n (B_1 = -1/2).
fn bernoulli_numbers(n: usize) -> Vec<BigRational> {
    let binom = |n: i64, r: i64| {
        let mut acc = BigRational::one();
        for i in 0..r {
            acc = acc * BigRational::from_integer(BigInt::from(n - i))
                / BigRational::from_integer(BigInt::from(i + 1));
        }
        acc
    };
    let mut b = vec![BigRational::one()];
    for m in 1..=n {
        let mut s = BigRational::zero();
        for (j, bj) in b.iter().enumerate() {
            s += binom(m as i64 + 1, j as i64) * bj.clone();
        }
        b.push(-s / BigRational::from_integer(BigInt::from(m as i64 + 1)));
    }
    b
}

/// zeta at 0, 1, positive even integers and negative integers (exact).
fn eval_zeta(arg: &Expr) -> Option<Expr> {
    let Expr::Integer(n) = arg else {
        return None;
    };
    let n = n.to_i64()?;
    if n == 0 {
        return Some(q(-1, 2));
    }
    if n == 1 {
        return Some(zoo());
    }
    if n > 0 && n % 2 == 0 && n <= 200 {
        // zeta(2m) = (-1)^(m+1) B_{2m} (2 pi)^{2m} / (2 (2m)!)
        let b = bernoulli_numbers(n as usize);
        let mut fact = BigInt::from(1);
        for i in 2..=n {
            fact *= BigInt::from(i);
        }
        let two_pow = BigInt::from(2).pow(n as u32);
        let sign = if (n / 2) % 2 == 1 { 1 } else { -1 };
        let coeff = b[n as usize].clone() * BigRational::from_integer(two_pow * BigInt::from(sign))
            / BigRational::from_integer(fact * BigInt::from(2));
        return Some(rational_expr(coeff) * eval_pow(pi(), Expr::from_i64(n)));
    }
    if (-200..0).contains(&n) {
        // zeta(-m) = -B_{m+1}/(m+1)  (B_1 convention irrelevant for m >= 1)
        let m = (-n) as usize;
        let b = bernoulli_numbers(m + 1);
        let mut bm = b[m + 1].clone();
        if m + 1 == 1 {
            bm = -bm;
        }
        return Some(rational_expr(
            -bm / BigRational::from_integer(BigInt::from(m as i64 + 1)),
        ));
    }
    None
}

/// `atan2(y, x)` for exact rationals: the principal argument of x + I*y.
fn eval_atan2(y: &Expr, x: &Expr) -> Option<Expr> {
    let (yv, xv) = (number(y)?, number(x)?);
    if xv.is_zero() && yv.is_zero() {
        return Some(Expr::Const(Constant::NaN));
    }
    if xv.is_zero() {
        return Some(pi_times(if yv.is_positive() {
            rat(1, 2)
        } else {
            rat(-1, 2)
        }));
    }
    let base = func("atan", rational_expr(yv.clone() / xv.clone()));
    if xv.is_positive() {
        return Some(base);
    }
    Some(if yv.is_negative() {
        base - pi()
    } else {
        base + pi()
    })
}

/// Integer-valued Mod for exact rationals: p - q*floor(p/q).
fn eval_mod(p: &Expr, qq: &Expr) -> Option<Expr> {
    let (pv, qv) = (number(p)?, number(qq)?);
    if qv.is_zero() {
        return Some(Expr::Const(Constant::NaN));
    }
    let f = floor_rat(&(pv.clone() / qv.clone()));
    Some(rational_expr(pv - qv * BigRational::from_integer(f)))
}

/// Max/Min of exact rationals (all arguments numeric).
fn eval_extremum(name: &str, args: &[Expr]) -> Option<Expr> {
    let vals: Option<Vec<BigRational>> = args.iter().map(number).collect();
    let vals = vals?;
    let pick = if name == "Max" {
        vals.into_iter().max()?
    } else {
        vals.into_iter().min()?
    };
    Some(rational_expr(pick))
}

/// Automatic evaluation of `name(args)`; `None` keeps the application.
pub fn eval_function(name: &str, args: &[Expr]) -> Option<Expr> {
    match (name, args) {
        ("atan2", [y, x]) => return eval_atan2(y, x),
        ("Mod", [p, qq]) => return eval_mod(p, qq),
        ("Max" | "Min", _) if !args.is_empty() => return eval_extremum(name, args),
        ("binomial", [n, k]) => return eval_binomial(n, k),
        _ => {}
    }
    let [arg] = args else {
        return None;
    };
    match name {
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => eval_circular(name, arg),
        "sinh" | "cosh" | "tanh" | "coth" | "sech" | "csch" => eval_hyperbolic(name, arg),
        "asin" | "acos" | "atan" | "acot" => eval_inverse_circular(name, arg),
        "asinh" | "acosh" | "atanh" | "acoth" | "acsch" => eval_inverse_hyperbolic(name, arg),
        "exp" => eval_exp(arg),
        "log" | "ln" => eval_log(arg),
        "Abs" => eval_abs(arg),
        "zeta" => eval_zeta(arg),
        "LambertW" => {
            if arg.is_zero() {
                Some(Expr::from_i64(0))
            } else if matches!(arg, Expr::Const(Constant::E)) {
                Some(Expr::from_i64(1))
            } else {
                None
            }
        }
        "factorial" => match arg {
            Expr::Integer(n) if !n.is_negative() && *n <= BigInt::from(1000) => {
                let mut acc = BigInt::from(1);
                let mut i = BigInt::from(2);
                while i <= *n {
                    acc *= i.clone();
                    i += BigInt::from(1);
                }
                Some(Expr::Integer(acc))
            }
            _ => None,
        },
        "sign" => eval_sign(arg),
        "Heaviside" => eval_sign(arg).and_then(|s| match number(&s) {
            Some(v) if v.is_negative() => Some(Expr::from_i64(0)),
            Some(v) if v.is_zero() => Some(rational_expr(rat(1, 2))),
            Some(_) => Some(Expr::from_i64(1)),
            None => None,
        }),
        "DiracDelta" => eval_sign(arg).and_then(|s| match number(&s) {
            Some(v) if !v.is_zero() => Some(Expr::from_i64(0)),
            _ => None,
        }),
        "subfactorial" => match arg {
            Expr::Integer(n) if !n.is_negative() && *n <= BigInt::from(1000) => {
                // !0 = 1, !1 = 0, !n = (n - 1) * (!(n-1) + !(n-2)).
                let n = n.to_i64()?;
                let (mut prev, mut cur) = (BigInt::from(1), BigInt::from(0));
                if n == 0 {
                    return Some(Expr::Integer(prev));
                }
                for i in 2..=n {
                    let next = BigInt::from(i - 1) * (cur.clone() + prev);
                    prev = cur;
                    cur = next;
                }
                Some(Expr::Integer(cur))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Upstream `binomial.eval` for an integer lower index: negative `k` gives
/// 0, `k = 0, 1` give `1, n`, and a numeric `n` evaluates the falling
/// factorial `n (n-1) ... (n-k+1) / k!` exactly (bounded `k`).
fn eval_binomial(n: &Expr, k: &Expr) -> Option<Expr> {
    let Expr::Integer(kk) = k else {
        return None;
    };
    if kk.is_negative() {
        return Some(Expr::from_i64(0));
    }
    if kk.is_zero() {
        return Some(Expr::from_i64(1));
    }
    if *kk == BigInt::from(1) {
        return Some(n.clone());
    }
    let nq = number(n)?;
    let kv = kk.to_i64()?;
    if kv > 1000 {
        return None;
    }
    if nq.is_integer() && !nq.is_negative() && nq < BigRational::from_integer(kk.clone()) {
        return Some(Expr::from_i64(0));
    }
    let mut acc = BigRational::one();
    for i in 0..kv {
        acc = acc * (nq.clone() - BigRational::from_integer(BigInt::from(i)))
            / BigRational::from_integer(BigInt::from(i + 1));
    }
    Some(rational_expr(acc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> Expr {
        Expr::symbol("x")
    }

    fn ev(name: &str, arg: Expr) -> String {
        func(name, arg).to_string()
    }

    #[test]
    fn trig_special_values() {
        assert_eq!(ev("sin", pi_times(rat(1, 6))), "1/2");
        assert_eq!(ev("cos", pi()), "-1");
        assert_eq!(ev("sin", pi()), "0");
        assert_eq!(ev("tan", pi_times(rat(1, 4))), "1");
        assert_eq!(ev("tan", pi_times(rat(1, 2))), "zoo");
        assert_eq!(ev("sin", pi_times(rat(-1, 2))), "-1");
        assert_eq!(
            func("sin", pi_times(rat(8, 7))),
            negate(&Expr::Function("sin".into(), vec![pi_times(rat(1, 7))]))
        );
        assert_eq!(
            func("sin", pi_times(rat(1, 7))),
            Expr::Function("sin".into(), vec![pi_times(rat(1, 7))])
        );
    }

    #[test]
    fn parity_and_inverse() {
        assert_eq!(
            func("cos", negate(&x())),
            Expr::Function("cos".into(), vec![x()])
        );
        assert_eq!(func("sin", Expr::Function("asin".into(), vec![x()])), x());
        assert_eq!(func("exp", Expr::Function("log".into(), vec![x()])), x());
        assert_eq!(ev("exp", imag() * pi()), "-1");
        assert_eq!(ev("log", Expr::Const(Constant::E)), "1");
        assert_eq!(func("asin", q(1, 2)), pi_times(rat(1, 6)));
        assert_eq!(func("acos", q(-1, 2)), pi_times(rat(2, 3)));
    }

    #[test]
    fn abs_gaussian_and_sqrt() {
        let z = Expr::from_i64(3) + Expr::from_i64(4) * imag();
        assert_eq!(func("Abs", z), Expr::from_i64(5));
        assert_eq!(sqrt_rational(&rat(1, 2)), q(1, 2) * sqrt_int(2));
        assert_eq!(sqrt_rational(&rat(8, 1)), Expr::from_i64(2) * sqrt_int(2));
    }
}
