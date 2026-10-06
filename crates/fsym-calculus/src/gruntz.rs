//! Limits by the Gruntz algorithm (most-rapidly-varying subexpressions).
//!
//! `limit(e, x, x0, dir)` reduces every case to `x -> +oo` (substituting
//! `x0 + 1/x`, `x0 - 1/x` or `-x`), then follows Gruntz's method: find the
//! set of most rapidly varying subexpressions, rewrite them in terms of a
//! single `w -> 0+`, take the leading term `c0*w^e0` of the series in `w`
//! and decide by the sign of `e0` (recursing on `c0` when `e0 = 0`).
//!
//! The limit variable is treated as positive and large, as upstream does
//! (`log(exp(a)) = a`, `log(x**p) = p*log(x)` for positive `x`). Every
//! decision that depends on the sign or zero-ness of a constant is made
//! exactly when possible; constants that cannot be decided make the limit
//! refuse with [`LimitError::Undetermined`] instead of guessing.

#![forbid(unsafe_code)]

use crate::series::{SeriesError, leading_term};
use fsym_core::elementary::{eval_function, eval_pow, rational_expr};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use num_complex::Complex64;
use num_traits::{Signed, Zero};
use std::collections::HashMap;
use thiserror::Error;

const MAX_DEPTH: usize = 40;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LimitError {
    #[error("limit undetermined: {0}")]
    Undetermined(String),
    #[error("limit does not exist: {0}")]
    DoesNotExist(String),
}

impl From<SeriesError> for LimitError {
    fn from(e: SeriesError) -> Self {
        LimitError::Undetermined(e.to_string())
    }
}

type LResult<T> = Result<T, LimitError>;

/// Direction of approach for a finite limit point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Plus,
    Minus,
    Both,
}

fn func(name: &str, arg: Expr) -> Expr {
    eval_function(name, std::slice::from_ref(&arg))
        .unwrap_or_else(|| Expr::Function(name.to_string(), vec![arg]))
}

fn oo() -> Expr {
    Expr::Const(Constant::Infinity)
}

fn neg_oo() -> Expr {
    Expr::Const(Constant::NegativeInfinity)
}

fn is_free_of(e: &Expr, x: &Symbol) -> bool {
    !e.free_symbols().iter().any(|s| s == x)
}

fn is_infinite(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Const(Constant::Infinity | Constant::NegativeInfinity)
    )
}

// ---------------------------------------------------------------------------
// Numeric helpers (decisions on constants)
// ---------------------------------------------------------------------------

/// Floating-point value of a symbol-free expression over the complex plane.
pub fn complex_value(e: &Expr) -> Option<Complex64> {
    Some(match e {
        Expr::Integer(n) => Complex64::new(n.to_f64()?, 0.0),
        Expr::Rational(r) => Complex64::new(r.to_f64()?, 0.0),
        Expr::Const(c) => match c {
            Constant::Pi => Complex64::new(std::f64::consts::PI, 0.0),
            Constant::E => Complex64::new(std::f64::consts::E, 0.0),
            Constant::I => Complex64::new(0.0, 1.0),
            _ => return None,
        },
        Expr::Sym(_) => return None,
        Expr::Add(xs) => {
            let mut acc = Complex64::new(0.0, 0.0);
            for x in xs {
                acc += complex_value(x)?;
            }
            acc
        }
        Expr::Mul(xs) => {
            let mut acc = Complex64::new(1.0, 0.0);
            for x in xs {
                acc *= complex_value(x)?;
            }
            acc
        }
        Expr::Pow(b, x) => {
            let bv = complex_value(b)?;
            if let Expr::Integer(n) = x.as_ref() {
                bv.powi(i32::try_from(n.to_i64()?).ok()?)
            } else {
                bv.powc(complex_value(x)?)
            }
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let a = complex_value(&args[0])?;
            match name.as_str() {
                "exp" => a.exp(),
                "log" | "ln" => a.ln(),
                "sin" => a.sin(),
                "cos" => a.cos(),
                "tan" => a.tan(),
                "cot" => a.tan().inv(),
                "sec" => a.cos().inv(),
                "csc" => a.sin().inv(),
                "sinh" => a.sinh(),
                "cosh" => a.cosh(),
                "tanh" => a.tanh(),
                "asin" => a.asin(),
                "acos" => a.acos(),
                "atan" => a.atan(),
                "asinh" => a.asinh(),
                "acosh" => a.acosh(),
                "atanh" => a.atanh(),
                "Abs" => Complex64::new(a.norm(), 0.0),
                "sqrt" => a.sqrt(),
                _ => return None,
            }
        }
        _ => return None,
    })
}

/// Sign of a real constant: `Some(1|-1|0)`, `None` when undecidable or not
/// a real constant.
pub fn constant_sign(c: &Expr) -> Option<i32> {
    match c {
        Expr::Integer(n) => {
            return Some(if n.is_zero() {
                0
            } else if n.is_negative() {
                -1
            } else {
                1
            });
        }
        Expr::Rational(r) => {
            return Some(if r.is_zero() {
                0
            } else if r.is_negative() {
                -1
            } else {
                1
            });
        }
        Expr::Const(Constant::Infinity) => return Some(1),
        Expr::Const(Constant::NegativeInfinity) => return Some(-1),
        _ => {}
    }
    if !c.free_symbols().is_empty() {
        return None;
    }
    let v = complex_value(c)?;
    if !v.re.is_finite() || v.im.abs() > 1e-12 * (1.0 + v.re.abs()) {
        return None;
    }
    if v.re.abs() > 1e-9 {
        return Some(if v.re > 0.0 { 1 } else { -1 });
    }
    if fsym_simplify::simplify(c).is_zero() {
        return Some(0);
    }
    None
}

/// Zero test for a series coefficient: `Some(true)` exact zero,
/// `Some(false)` provably (numerically well-separated) nonzero or symbolic,
/// `None` when a constant cannot be decided.
pub fn is_zero_coefficient(c: &Expr) -> Option<bool> {
    if c.is_zero() {
        return Some(true);
    }
    if !c.free_symbols().is_empty() {
        let s = fsym_simplify::simplify(c);
        return Some(s.is_zero());
    }
    match complex_value(c) {
        Some(v) if v.norm() > 1e-9 => Some(false),
        Some(_) | None => {
            if fsym_simplify::simplify(c).is_zero() {
                Some(true)
            } else if complex_value(c).is_none() {
                Some(false)
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Normalization under "x is positive and large"
// ---------------------------------------------------------------------------

struct Gruntz {
    x: Symbol,
    depth: usize,
    fresh: usize,
}

impl Gruntz {
    fn is_positive(&self, e: &Expr) -> bool {
        match e {
            Expr::Sym(s) => *s == self.x,
            Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => constant_sign(e) == Some(1),
            Expr::Function(name, _) if name == "exp" => true,
            Expr::Mul(xs) => xs.iter().all(|f| self.is_positive(f)),
            Expr::Add(xs) => xs.iter().all(|f| self.is_positive(f)),
            Expr::Pow(b, _) => self.is_positive(b),
            _ => e.free_symbols().is_empty() && constant_sign(e) == Some(1),
        }
    }

    /// Rebuilds `e` canonically, applying `log(exp(a)) = a`,
    /// `log(b**p) = p*log(b)` and `log(a*b) = log(a)+log(b)` for positive
    /// factors, and `x**p` combination for positive bases.
    fn normalize(&self, e: &Expr) -> Expr {
        match e {
            Expr::Add(xs) => xs
                .iter()
                .map(|t| self.normalize(t))
                .fold(Expr::from_i64(0), |a, b| a + b),
            Expr::Mul(xs) => xs
                .iter()
                .map(|t| self.normalize(t))
                .fold(Expr::from_i64(1), |a, b| a * b),
            Expr::Pow(b, p) => {
                let b = self.normalize(b);
                let p = self.normalize(p);
                if !is_free_of(&p, &self.x) {
                    // b**p = exp(p*log(b)) for an x-dependent exponent.
                    let l = self.log_positive(b);
                    return func("exp", self.normalize(&(p * l)));
                }
                if let Expr::Pow(bb, pp) = &b
                    && self.is_positive(bb)
                {
                    return eval_pow((**bb).clone(), (**pp).clone() * p);
                }
                eval_pow(b, p)
            }
            Expr::Function(name, args) if args.len() == 1 => {
                let a = self.normalize(&args[0]);
                if name == "log" {
                    return self.log_positive(a);
                }
                if !is_free_of(&a, &self.x)
                    && let Some(rewritten) = hyperbolic_as_exp(name, &a)
                {
                    return self.normalize(&rewritten);
                }
                func(name, a)
            }
            Expr::Function(name, args) => Expr::Function(
                name.clone(),
                args.iter().map(|a| self.normalize(a)).collect(),
            ),
            other => other.clone(),
        }
    }

    fn log_positive(&self, a: Expr) -> Expr {
        match &a {
            Expr::Function(n, inner) if n == "exp" && inner.len() == 1 => inner[0].clone(),
            Expr::Pow(b, p)
                if self.is_positive(b) && p.free_symbols().iter().all(|s| *s == self.x) =>
            {
                (**p).clone() * self.log_positive((**b).clone())
            }
            Expr::Mul(xs) if xs.iter().all(|f| self.is_positive(f)) && xs.len() > 1 => xs
                .iter()
                .map(|f| self.log_positive(f.clone()))
                .fold(Expr::from_i64(0), |a, b| a + b),
            _ => func("log", a),
        }
    }

    fn fresh_symbol(&mut self, stem: &str) -> Symbol {
        self.fresh += 1;
        Symbol::new(format!("_fsym_{stem}{}", self.fresh))
    }

    // -----------------------------------------------------------------------
    // Core recursion
    // -----------------------------------------------------------------------

    fn limitinf(&mut self, e: &Expr) -> LResult<Expr> {
        if self.depth > MAX_DEPTH {
            return Err(LimitError::Undetermined("recursion depth exceeded".into()));
        }
        self.depth += 1;
        let r = self.limitinf_inner(e);
        self.depth -= 1;
        r
    }

    fn limitinf_inner(&mut self, e: &Expr) -> LResult<Expr> {
        let e = self.normalize(e);
        if std::env::var("FSYM_TRACE").is_ok() {
            eprintln!("{:indent$}limitinf {e}", "", indent = self.depth);
        }
        if is_free_of(&e, &self.x) {
            return Ok(e);
        }
        if e == Expr::Sym(self.x.clone()) {
            return Ok(oo());
        }
        // bounded * (-> 0) -> 0 for oscillating sin/cos factors whose
        // argument diverges (no series exists at such a point).
        if let Expr::Mul(factors) = &e {
            let (osc, rest): (Vec<Expr>, Vec<Expr>) = factors
                .iter()
                .cloned()
                .partition(|f| is_bounded_oscillator(f) && !is_free_of(f, &self.x));
            if !osc.is_empty() {
                let mut diverging = false;
                for f in &osc {
                    if let Expr::Function(_, args) = f
                        && is_infinite(&self.limitinf(&args[0])?)
                    {
                        diverging = true;
                    }
                }
                if diverging {
                    let rest = rest.into_iter().fold(Expr::from_i64(1), |a, b| a * b);
                    let l = self.limitinf(&rest)?;
                    if l.is_zero() {
                        return Ok(Expr::from_i64(0));
                    }
                    return Err(LimitError::Undetermined(format!(
                        "oscillating factor times {l}"
                    )));
                }
            }
        }
        let (c0, e0) = self.mrv_leadterm(&e)?;
        if e0.is_positive() {
            return Ok(Expr::from_i64(0));
        }
        if e0.is_negative() {
            let s = self.sign(&c0)?;
            return match s {
                1 => Ok(oo()),
                -1 => Ok(neg_oo()),
                _ => Err(LimitError::Undetermined(format!("sign of {c0}"))),
            };
        }
        self.limitinf(&c0)
    }

    /// Sign of `e` as `x -> oo` (eventually constant).
    fn sign(&mut self, e: &Expr) -> LResult<i32> {
        if is_free_of(e, &self.x) {
            return constant_sign(e)
                .ok_or_else(|| LimitError::Undetermined(format!("sign of {e}")));
        }
        match e {
            Expr::Sym(_) => return Ok(1),
            Expr::Function(n, _) if n == "exp" => return Ok(1),
            Expr::Mul(xs) => {
                let mut s = 1;
                for f in xs {
                    s *= self.sign(f)?;
                }
                return Ok(s);
            }
            Expr::Pow(b, p) => {
                let sb = self.sign(b)?;
                if sb == 1 {
                    return Ok(1);
                }
                if let Expr::Integer(n) = p.as_ref() {
                    let odd = (n.clone() % BigInt::from(2)) != BigInt::from(0);
                    return Ok(if odd { sb } else { sb * sb });
                }
            }
            _ => {}
        }
        let (c0, _) = self.mrv_leadterm(e)?;
        self.sign(&c0)
    }

    /// Most rapidly varying subexpressions of `e` (each `x` or `exp(..)`).
    fn mrv(&mut self, e: &Expr) -> LResult<Vec<Expr>> {
        if is_free_of(e, &self.x) {
            return Ok(Vec::new());
        }
        match e {
            Expr::Sym(_) => Ok(vec![e.clone()]),
            Expr::Add(xs) | Expr::Mul(xs) => {
                let mut acc = Vec::new();
                for t in xs {
                    let m = self.mrv(t)?;
                    acc = self.mrv_max(acc, m)?;
                }
                Ok(acc)
            }
            Expr::Pow(b, p) => {
                if is_free_of(p, &self.x) {
                    self.mrv(b)
                } else {
                    let rewritten = self.normalize(&func(
                        "exp",
                        (**p).clone() * Expr::Function("log".into(), vec![(**b).clone()]),
                    ));
                    self.mrv(&rewritten)
                }
            }
            Expr::Function(name, args) if name == "exp" && args.len() == 1 => {
                let l = self.limitinf(&args[0])?;
                let inner = self.mrv(&args[0])?;
                if is_infinite(&l) {
                    self.mrv_max(vec![e.clone()], inner)
                } else {
                    Ok(inner)
                }
            }
            Expr::Function(_, args) => {
                let mut acc = Vec::new();
                for a in args {
                    let m = self.mrv(a)?;
                    acc = self.mrv_max(acc, m)?;
                }
                Ok(acc)
            }
            _ => Ok(Vec::new()),
        }
    }

    fn mrv_max(&mut self, f: Vec<Expr>, g: Vec<Expr>) -> LResult<Vec<Expr>> {
        if f.is_empty() {
            return Ok(g);
        }
        if g.is_empty() {
            return Ok(f);
        }
        let x = Expr::Sym(self.x.clone());
        if f.contains(&x) && g.contains(&x) {
            let mut out = f;
            for item in g {
                if !out.contains(&item) {
                    out.push(item);
                }
            }
            return Ok(out);
        }
        // x is the slowest scale: any exp(..) element dominates it.
        if f.contains(&x) {
            return Ok(g);
        }
        if g.contains(&x) {
            return Ok(f);
        }
        match self.compare(&f[0], &g[0])? {
            std::cmp::Ordering::Greater => Ok(f),
            std::cmp::Ordering::Less => Ok(g),
            std::cmp::Ordering::Equal => {
                let mut out = f;
                for item in g {
                    if !out.contains(&item) {
                        out.push(item);
                    }
                }
                Ok(out)
            }
        }
    }

    fn log_of(&self, a: &Expr) -> Expr {
        match a {
            Expr::Function(n, args) if n == "exp" && args.len() == 1 => args[0].clone(),
            other => self.log_positive(other.clone()),
        }
    }

    /// Compares growth classes: `Greater` when `a` varies faster than `b`.
    fn compare(&mut self, a: &Expr, b: &Expr) -> LResult<std::cmp::Ordering> {
        let la = self.log_of(a);
        let lb = self.log_of(b);
        let ratio = la * eval_pow(lb, Expr::from_i64(-1));
        let c = self.limitinf(&ratio)?;
        if c.is_zero() {
            Ok(std::cmp::Ordering::Less)
        } else if is_infinite(&c) {
            Ok(std::cmp::Ordering::Greater)
        } else {
            Ok(std::cmp::Ordering::Equal)
        }
    }

    fn subs_x(&self, e: &Expr, v: &Expr) -> Expr {
        self.normalize(&e.subs(&HashMap::from([(self.x.clone(), v.clone())])))
    }

    /// Leading term `(c0, e0)` of `e` in the most rapidly varying scale.
    fn mrv_leadterm(&mut self, e: &Expr) -> LResult<(Expr, BigRational)> {
        let omega = self.mrv(e)?;
        if omega.is_empty() {
            return Ok((e.clone(), BigRational::zero()));
        }
        let x = Expr::Sym(self.x.clone());
        if omega.contains(&x) {
            // Move up one level: x -> exp(x).
            let up = self.subs_x(e, &func("exp", x.clone()));
            let (c0, e0) = self.mrv_leadterm(&up)?;
            let down = self.subs_x(&c0, &func("log", x));
            return Ok((down, e0));
        }
        let w = self.fresh_symbol("w");
        let (f, logw) = self.rewrite(e, &omega, &w)?;
        if std::env::var("FSYM_TRACE").is_ok() {
            eprintln!(
                "{:indent$}rewrite {e} omega={} -> {f} logw={logw}",
                "",
                omega
                    .iter()
                    .map(|o| o.to_string())
                    .collect::<Vec<_>>()
                    .join(";"),
                indent = self.depth
            );
        }
        let (c0, e0) = leading_term(&f, &w, &logw)?;
        Ok((self.normalize(&c0), e0))
    }

    /// Rewrites `e` in terms of `w -> 0+` for the mrv set `omega`.
    ///
    /// With `g = exp(s)` an innermost element and `w = g**(-sign(s))`,
    /// every `f = exp(t)` in `omega` becomes `exp(t - c*s) * w**(-+c)` where
    /// `c = lim t/s` is a rational constant. Returns `(f(w), log(w))`.
    fn rewrite(&mut self, e: &Expr, omega: &[Expr], w: &Symbol) -> LResult<(Expr, Expr)> {
        let g = omega
            .iter()
            .find(|cand| {
                !omega
                    .iter()
                    .any(|o| *cand != o && contains_subexpr(cand, o))
            })
            .cloned()
            .ok_or_else(|| LimitError::Undetermined("no innermost mrv element".into()))?;
        let s = exp_argument(&g)?;
        let sig = self.sign(&s)?;
        let wexpr = Expr::Sym(w.clone());
        let mut args = Vec::with_capacity(omega.len());
        let mut ratios = Vec::with_capacity(omega.len());
        for f in omega {
            let t = exp_argument(f)?;
            let c = self.limitinf(&(t.clone() * eval_pow(s.clone(), Expr::from_i64(-1))))?;
            let cr = match c {
                Expr::Integer(n) => BigRational::from_integer(n),
                Expr::Rational(r) => r,
                other => {
                    return Err(LimitError::Undetermined(format!(
                        "non-rational mrv exponent ratio {other}"
                    )));
                }
            };
            args.push(t);
            ratios.push(cr);
        }
        // Indices ordered outermost (largest) first.
        let mut order: Vec<usize> = (0..omega.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(expr_size(&omega[i])));
        let mut built: Vec<Option<Expr>> = vec![None; omega.len()];
        // Build innermost first so nested replacements are available.
        for &i in order.iter().rev() {
            let mut t = args[i].clone();
            for &j in &order {
                if j != i
                    && contains_subexpr(&t, &omega[j])
                    && let Some(b) = &built[j]
                {
                    t = t.subs_expr(&omega[j], b);
                }
            }
            let reduced = self.normalize(&(t - rational_expr(ratios[i].clone()) * s.clone()));
            let wexp = if sig == 1 {
                -ratios[i].clone()
            } else {
                ratios[i].clone()
            };
            built[i] = Some(func("exp", reduced) * eval_pow(wexpr.clone(), rational_expr(wexp)));
        }
        let mut out = e.clone();
        for &i in &order {
            if let Some(b) = &built[i] {
                out = out.subs_expr(&omega[i], b);
            }
        }
        let out = self.normalize(&out);
        let logw = if sig == 1 { -s } else { s };
        Ok((out, logw))
    }

    fn limit_point(&mut self, e: &Expr, x0: &Expr, dir: Direction) -> LResult<Expr> {
        let x = Expr::Sym(self.x.clone());
        match x0 {
            Expr::Const(Constant::Infinity) => self.limitinf(e),
            Expr::Const(Constant::NegativeInfinity) => {
                let flipped = e.subs(&HashMap::from([(self.x.clone(), -x)]));
                self.limitinf(&flipped)
            }
            _ => {
                let recip = eval_pow(x.clone(), Expr::from_i64(-1));
                let shifted = match dir {
                    Direction::Minus => x0.clone() - recip,
                    _ => x0.clone() + recip,
                };
                let sub = e.subs(&HashMap::from([(self.x.clone(), shifted)]));
                self.limitinf(&sub)
            }
        }
    }
}

/// Hyperbolic functions in terms of `exp` (the tractable form).
fn hyperbolic_as_exp(name: &str, a: &Expr) -> Option<Expr> {
    let ep = func("exp", a.clone());
    let em = func("exp", -a.clone());
    let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
    let inv = |e: Expr| eval_pow(e, Expr::from_i64(-1));
    Some(match name {
        "sinh" => half * (ep - em),
        "cosh" => half * (ep + em),
        "tanh" => (ep.clone() - em.clone()) * inv(ep + em),
        "coth" => (ep.clone() + em.clone()) * inv(ep - em),
        "sech" => Expr::from_i64(2) * inv(ep + em),
        "csch" => Expr::from_i64(2) * inv(ep - em),
        _ => return None,
    })
}

/// Bounded oscillating factor: `sin`/`cos` of a real argument.
fn is_bounded_oscillator(e: &Expr) -> bool {
    matches!(e, Expr::Function(n, args) if (n == "sin" || n == "cos") && args.len() == 1)
}

fn expr_size(e: &Expr) -> usize {
    match e {
        Expr::Add(xs) | Expr::Mul(xs) | Expr::Function(_, xs) => {
            1 + xs.iter().map(expr_size).sum::<usize>()
        }
        Expr::Pow(b, p) => 1 + expr_size(b) + expr_size(p),
        _ => 1,
    }
}

fn contains_subexpr(outer: &Expr, inner: &Expr) -> bool {
    if outer == inner {
        return true;
    }
    match outer {
        Expr::Add(xs) | Expr::Mul(xs) | Expr::Function(_, xs) => {
            xs.iter().any(|x| contains_subexpr(x, inner))
        }
        Expr::Pow(b, p) => contains_subexpr(b, inner) || contains_subexpr(p, inner),
        _ => false,
    }
}

fn exp_argument(f: &Expr) -> LResult<Expr> {
    match f {
        Expr::Function(n, args) if n == "exp" && args.len() == 1 => Ok(args[0].clone()),
        other => Err(LimitError::Undetermined(format!(
            "mrv element {other} is not exp"
        ))),
    }
}

fn substitute_positive_dummy(e: &Expr, x: &Symbol) -> (Expr, Symbol) {
    let d = Symbol::new(format!("_fsym_limit_{}", x.name));
    (
        e.subs(&HashMap::from([(x.clone(), Expr::Sym(d.clone()))])),
        d,
    )
}

/// Limit of `e` as `x -> x0` in direction `dir` (ignored at infinity).
pub fn limit(e: &Expr, x: &Symbol, x0: &Expr, dir: Direction) -> LResult<Expr> {
    if is_free_of(e, x) {
        return Ok(e.clone());
    }
    if dir == Direction::Both && !is_infinite(x0) {
        let right = limit(e, x, x0, Direction::Plus)?;
        let left = limit(e, x, x0, Direction::Minus)?;
        if right == left {
            return Ok(right);
        }
        return Err(LimitError::DoesNotExist(format!(
            "left hand limit = {left} and right hand limit = {right}"
        )));
    }
    let (e, d) = substitute_positive_dummy(e, x);
    let mut g = Gruntz {
        x: d.clone(),
        depth: 0,
        fresh: 0,
    };
    let value = g.limit_point(&e, x0, dir)?;
    // A finite limit may still mention the dummy only through a failure.
    if !is_free_of(&value, &d) {
        return Err(LimitError::Undetermined(format!("{value}")));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> Symbol {
        Symbol::new("x")
    }

    fn xe() -> Expr {
        Expr::Sym(x())
    }

    fn lim(e: Expr, x0: Expr) -> String {
        match limit(&e, &x(), &x0, Direction::Plus) {
            Ok(v) => v.to_string(),
            Err(err) => format!("ERR {err}"),
        }
    }

    #[test]
    fn classic_limits() {
        let sinx_over_x = func("sin", xe()) * eval_pow(xe(), Expr::from_i64(-1));
        assert_eq!(lim(sinx_over_x, Expr::from_i64(0)), "1");
        let cos_term = (func("cos", xe()) - Expr::from_i64(1)) * eval_pow(xe(), Expr::from_i64(-2));
        assert_eq!(lim(cos_term, Expr::from_i64(0)), "-1/2");
        let compound = eval_pow(Expr::from_i64(1) + eval_pow(xe(), Expr::from_i64(-1)), xe());
        assert_eq!(lim(compound, oo()), "E");
        let xexp = xe() * func("exp", -xe());
        assert_eq!(lim(xexp, oo()), "0");
        let xlog = xe() * func("log", xe());
        assert_eq!(lim(xlog, Expr::from_i64(0)), "0");
        let recip = eval_pow(xe(), Expr::from_i64(-1));
        assert_eq!(lim(recip.clone(), Expr::from_i64(0)), "oo");
        assert!(limit(&recip, &x(), &Expr::from_i64(0), Direction::Both).is_err());
    }
}
