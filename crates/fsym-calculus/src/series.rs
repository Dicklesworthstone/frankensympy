//! Generalized power series in one variable `t -> 0+`.
//!
//! A [`Series`] is a finite sum `sum c_k t^k` over exact rational exponents
//! (Laurent and Puiseux terms) plus an optional order term `O(t^order)`; the
//! coefficients are expressions free of `t`. `log(t)` never becomes a power
//! of `t`: it is carried inside coefficients as the placeholder symbol
//! [`LOG_T`], which callers replace with the meaning of `log(t)` in their
//! context (for Gruntz limits, an expression in the limit variable).
//!
//! Expansion is computed to a requested absolute order. Operations that
//! need more precision from their operands (products with negative
//! valuations, reciprocals, logarithms) re-expand the operand at a raised
//! order, so every term reported below the result order is exact.
//!
//! Supported: `+`, `*`, powers with `t`-free exponents (integer, rational
//! or symbolic when the base has valuation 0), `exp`, `log`, `Abs`, the
//! circular/hyperbolic families and their inverses at finite points via
//! exact Taylor coefficients, and `atan`/`acot` at infinite arguments.
//! Anything else is refused with a typed error rather than approximated.

#![forbid(unsafe_code)]

use fsym_core::elementary::{eval_function, eval_pow, rational_expr};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use num_traits::{Signed, Zero};
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

/// Name of the placeholder symbol standing for `log(t)` in coefficients.
pub const LOG_T: &str = "_fsym_log_t";

/// Hard limits on expansion work.
const MAX_TERMS: usize = 64;
const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SeriesError {
    #[error("series expansion not supported: {0}")]
    Unsupported(String),
    #[error("essential singularity in series expansion: {0}")]
    Essential(String),
    #[error("series precision exhausted")]
    Precision,
}

type SResult<T> = Result<T, SeriesError>;

/// Truncated generalized power series.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    /// Exponent -> nonzero coefficient.
    pub terms: BTreeMap<BigRational, Expr>,
    /// Absolute order of the error term; `None` when the series is exact.
    pub order: Option<BigRational>,
}

fn q(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

fn min_order(a: &Option<BigRational>, b: &Option<BigRational>) -> Option<BigRational> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x.clone()),
        (Some(x), Some(y)) => Some(if x < y { x.clone() } else { y.clone() }),
    }
}

fn add_order(a: &Option<BigRational>, d: &BigRational) -> Option<BigRational> {
    a.as_ref().map(|x| x.clone() + d.clone())
}

/// Canonical form used for coefficient zero detection.
pub fn normalize(e: &Expr) -> Expr {
    fsym_simplify::expand(e)
}

impl Series {
    pub fn zero_exact() -> Self {
        Series {
            terms: BTreeMap::new(),
            order: None,
        }
    }

    pub fn constant(c: Expr) -> Self {
        let mut terms = BTreeMap::new();
        if !c.is_zero() {
            terms.insert(BigRational::zero(), c);
        }
        Series { terms, order: None }
    }

    pub fn monomial(c: Expr, k: BigRational) -> Self {
        let mut terms = BTreeMap::new();
        if !c.is_zero() {
            terms.insert(k, c);
        }
        Series { terms, order: None }
    }

    /// Lowest exponent with a nonzero coefficient.
    pub fn valuation(&self) -> Option<BigRational> {
        self.terms.keys().next().cloned()
    }

    /// Valuation, or the order when no term is known.
    fn val_or_order(&self) -> Option<BigRational> {
        self.valuation().or_else(|| self.order.clone())
    }

    pub fn lead(&self) -> Option<(BigRational, Expr)> {
        self.terms
            .iter()
            .next()
            .map(|(k, c)| (k.clone(), c.clone()))
    }

    fn truncate(mut self, order: &BigRational) -> Self {
        if self.order.is_none() && self.terms.keys().all(|k| k < order) {
            // Nothing dropped: an exact series stays exact.
            return self;
        }
        self.terms.retain(|k, _| k < order);
        self.order = min_order(&self.order, &Some(order.clone()));
        self
    }

    fn insert_sum(terms: &mut BTreeMap<BigRational, Expr>, k: BigRational, c: Expr) {
        let entry = terms.remove(&k);
        let total = match entry {
            Some(prev) => normalize(&(prev + c)),
            None => c,
        };
        if !total.is_zero() {
            terms.insert(k, total);
        }
    }

    pub fn add(&self, other: &Series) -> Series {
        let order = min_order(&self.order, &other.order);
        let mut terms = BTreeMap::new();
        for (k, c) in self.terms.iter().chain(other.terms.iter()) {
            if order.as_ref().is_none_or(|o| k < o) {
                Series::insert_sum(&mut terms, k.clone(), c.clone());
            }
        }
        Series { terms, order }
    }

    pub fn scale(&self, c: &Expr) -> Series {
        if c.is_zero() {
            return Series::zero_exact();
        }
        let mut terms = BTreeMap::new();
        for (k, v) in &self.terms {
            let p = normalize(&(v.clone() * c.clone()));
            if !p.is_zero() {
                terms.insert(k.clone(), p);
            }
        }
        Series {
            terms,
            order: self.order.clone(),
        }
    }

    pub fn shift(&self, d: &BigRational) -> Series {
        Series {
            terms: self
                .terms
                .iter()
                .map(|(k, c)| (k.clone() + d.clone(), c.clone()))
                .collect(),
            order: add_order(&self.order, d),
        }
    }

    pub fn mul(&self, other: &Series) -> Series {
        let va = self.val_or_order();
        let vb = other.val_or_order();
        let o1 = match (&self.order, &vb) {
            (Some(oa), Some(vb)) => Some(oa.clone() + vb.clone()),
            _ => None,
        };
        let o2 = match (&other.order, &va) {
            (Some(ob), Some(va)) => Some(ob.clone() + va.clone()),
            _ => None,
        };
        let order = min_order(&o1, &o2);
        let mut acc: BTreeMap<BigRational, Vec<Expr>> = BTreeMap::new();
        for (ka, ca) in &self.terms {
            for (kb, cb) in &other.terms {
                let k = ka.clone() + kb.clone();
                if order.as_ref().is_none_or(|o| &k < o) {
                    acc.entry(k).or_default().push(ca.clone() * cb.clone());
                }
            }
        }
        let mut terms = BTreeMap::new();
        for (k, parts) in acc {
            let total = normalize(&parts.into_iter().fold(Expr::from_i64(0), |a, b| a + b));
            if !total.is_zero() {
                terms.insert(k, total);
            }
        }
        Series { terms, order }
    }

    /// `sum_{k} coeff(k) * u^k` for a series `u` with positive valuation,
    /// computed to absolute order `target`.
    fn compose(u: &Series, coeff: impl Fn(usize) -> Expr, target: &BigRational) -> SResult<Series> {
        let mut result = Series::constant(coeff(0));
        if u.terms.is_empty() && u.order.is_none() {
            return Ok(result);
        }
        let Some(v) = u.valuation() else {
            // u = O(t^order): only the constant term is known.
            let order = min_order(&u.order, &Some(target.clone())).expect("bounded by target");
            result.terms.retain(|k, _| *k < order);
            result.order = Some(order);
            return Ok(result);
        };
        if !v.is_positive() {
            return Err(SeriesError::Unsupported(
                "composition needs positive valuation".into(),
            ));
        }
        let mut power = Series::constant(Expr::from_i64(1));
        let mut k = 0usize;
        loop {
            k += 1;
            if k > MAX_TERMS {
                break;
            }
            let lowest = v.clone() * q(k as i64);
            if &lowest >= target {
                break;
            }
            power = power.mul(u).truncate(target);
            let c = coeff(k);
            if !c.is_zero() {
                result = result.add(&power.scale(&c));
            }
        }
        // Remaining tail starts at k*v >= target (an infinite series is
        // never exact, whatever the operands were).
        let tail = v * q(k as i64);
        let order = min_order(
            &min_order(&result.order, &Some(tail)),
            &Some(target.clone()),
        );
        let order = min_order(&order, &u.order).expect("bounded by target");
        result.terms.retain(|k, _| *k < order);
        result.order = Some(order);
        Ok(result)
    }
}

/// Expansion context.
struct Ctx<'a> {
    t: &'a Symbol,
    log_t: Expr,
    depth: usize,
}

fn is_free_of(e: &Expr, t: &Symbol) -> bool {
    !e.free_symbols().iter().any(|s| s == t)
}

fn number(e: &Expr) -> Option<BigRational> {
    match e {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn factorial(k: usize) -> BigInt {
    let mut f = BigInt::from(1);
    for i in 2..=k {
        f *= BigInt::from(i as i64);
    }
    f
}

fn rat_expr(n: BigInt, d: BigInt) -> Expr {
    rational_expr(BigRational::new(n, d))
}

/// Generalized binomial coefficient `binom(a, k)` for an expression `a`.
fn binomial_coeff(a: &Expr, k: usize) -> Expr {
    let mut num = Expr::from_i64(1);
    for i in 0..k {
        num = num * (a.clone() - Expr::from_i64(i as i64));
    }
    normalize(&(num * rat_expr(BigInt::from(1), factorial(k))))
}

fn func(name: &str, arg: Expr) -> Expr {
    eval_function(name, std::slice::from_ref(&arg))
        .unwrap_or_else(|| Expr::Function(name.to_string(), vec![arg]))
}

fn has_undefined(e: &Expr) -> bool {
    match e {
        Expr::Const(Constant::ComplexInfinity | Constant::NaN) => true,
        Expr::Add(xs) | Expr::Mul(xs) | Expr::Function(_, xs) => xs.iter().any(has_undefined),
        Expr::Pow(b, x) => has_undefined(b) || has_undefined(x),
        _ => false,
    }
}

/// Taylor coefficient `f^(k)(a0) / k!` for a named analytic function.
fn taylor_coefficients(name: &str, a0: &Expr, n: usize) -> SResult<Vec<Expr>> {
    let z = Symbol::new("_fsym_series_z");
    let mut f = Expr::Function(name.to_string(), vec![Expr::Sym(z.clone())]);
    let mut out = Vec::with_capacity(n + 1);
    let map = |e: &Expr| e.subs(&HashMap::from([(z.clone(), a0.clone())]));
    for k in 0..=n {
        let value = map(&f);
        if has_undefined(&value) {
            return Err(SeriesError::Unsupported(format!(
                "{name} is not analytic at {a0}"
            )));
        }
        out.push(normalize(
            &(value * rat_expr(BigInt::from(1), factorial(k))),
        ));
        if k < n {
            f = fsym_simplify::simplify(&crate::diff(&f, &z));
            if crate::carries_diff_sentinel_pub(&f) {
                return Err(SeriesError::Unsupported(format!("derivative of {name}")));
            }
        }
    }
    Ok(out)
}

fn expand_at(e: &Expr, target: &BigRational, ctx: &mut Ctx) -> SResult<Series> {
    if ctx.depth > MAX_DEPTH {
        return Err(SeriesError::Unsupported("expression too deep".into()));
    }
    ctx.depth += 1;
    let r = expand_inner(e, target, ctx);
    ctx.depth -= 1;
    r
}

fn expand_inner(e: &Expr, target: &BigRational, ctx: &mut Ctx) -> SResult<Series> {
    if is_free_of(e, ctx.t) {
        if has_undefined(e) {
            return Err(SeriesError::Unsupported(format!("undefined constant {e}")));
        }
        return Ok(Series::constant(e.clone()));
    }
    match e {
        Expr::Sym(_) => Ok(Series::monomial(Expr::from_i64(1), q(1)).truncate(target)),
        Expr::Add(terms) => {
            let mut acc = Series::zero_exact();
            for term in terms {
                acc = acc.add(&expand_at(term, target, ctx)?);
            }
            Ok(acc.truncate(target))
        }
        Expr::Mul(factors) => expand_mul(factors, target, ctx),
        Expr::Pow(base, exp) => {
            if is_free_of(exp, ctx.t) {
                expand_pow(base, exp, target, ctx)
            } else {
                // b**e = exp(e*log(b))
                let rewritten = func(
                    "exp",
                    (**exp).clone() * Expr::Function("log".into(), vec![(**base).clone()]),
                );
                expand_at(&rewritten, target, ctx)
            }
        }
        Expr::Function(name, args) if args.len() == 1 => {
            expand_function(name, &args[0], target, ctx)
        }
        other => Err(SeriesError::Unsupported(format!("{other}"))),
    }
}

fn expand_mul(factors: &[Expr], target: &BigRational, ctx: &mut Ctx) -> SResult<Series> {
    // First pass at the requested order to learn valuations.
    let mut parts: Vec<Series> = Vec::with_capacity(factors.len());
    for f in factors {
        parts.push(expand_at(f, target, ctx)?);
    }
    let vals: Vec<Option<BigRational>> = parts.iter().map(|s| s.val_or_order()).collect();
    if vals.iter().any(|v| v.is_none()) {
        // An exact zero factor.
        if parts
            .iter()
            .any(|s| s.terms.is_empty() && s.order.is_none())
        {
            return Ok(Series::zero_exact());
        }
    }
    // Re-expand factors whose companions have negative total valuation.
    for i in 0..parts.len() {
        let others: BigRational = vals
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| v.clone().unwrap_or_else(BigRational::zero))
            .fold(BigRational::zero(), |a, b| a + b);
        if others.is_negative() && parts[i].order.is_some() {
            let raised = target.clone() - others;
            parts[i] = expand_at(&factors[i], &raised, ctx)?;
        }
    }
    let mut acc = Series::constant(Expr::from_i64(1));
    for p in &parts {
        acc = acc.mul(p);
    }
    Ok(acc.truncate(target))
}

/// Split `s = c0 * t^v * (1 + u)` with `val(u) > 0`.
fn split_lead(s: &Series) -> SResult<(BigRational, Expr, Series)> {
    let Some((v, c0)) = s.lead() else {
        return Err(SeriesError::Precision);
    };
    let inv = func_pow(&c0, &Expr::from_i64(-1));
    let mut u = Series {
        terms: BTreeMap::new(),
        order: s.order.clone().map(|o| o - v.clone()),
    };
    for (k, c) in s.terms.iter().skip(1) {
        let coeff = normalize(&(c.clone() * inv.clone()));
        if !coeff.is_zero() {
            u.terms.insert(k.clone() - v.clone(), coeff);
        }
    }
    Ok((v, c0, u))
}

fn func_pow(b: &Expr, e: &Expr) -> Expr {
    eval_pow(b.clone(), e.clone())
}

fn expand_pow(base: &Expr, exp: &Expr, target: &BigRational, ctx: &mut Ctx) -> SResult<Series> {
    let mut b = expand_at(base, target, ctx)?;
    if b.terms.is_empty() {
        b = retry_until_term(base, target, ctx)?;
    }
    let (v, _, _) = split_lead(&b)?;
    let a_num = number(exp);
    let scaled_v = match &a_num {
        Some(a) => v.clone() * a.clone(),
        None if v.is_zero() => BigRational::zero(),
        None => {
            return Err(SeriesError::Unsupported(format!(
                "power t**({exp}) with symbolic exponent"
            )));
        }
    };
    // Need relative precision (target - scaled_v) from b: raise its order.
    let need = target.clone() - scaled_v.clone() + v.clone();
    if b.order.as_ref().is_some_and(|o| o < &need) {
        b = expand_at(base, &need, ctx)?;
    }
    let (v, c0, u) = split_lead(&b)?;
    let rel_target = target.clone() - scaled_v.clone();
    // Small positive integer powers: repeated products keep exactness.
    if let Some(a) = &a_num
        && a.is_integer()
        && a.is_positive()
        && a.to_integer() <= BigInt::from(32)
    {
        let k = a.to_integer().to_i64().unwrap_or(1);
        let mut acc = Series::constant(Expr::from_i64(1));
        for _ in 0..k {
            acc = acc.mul(&b).truncate(target);
        }
        let _ = (v, c0, u, rel_target);
        return Ok(acc);
    }
    let lead = func_pow(&c0, exp);
    let body = Series::compose(&u, |k| binomial_coeff(exp, k), &rel_target)?;
    let _ = v;
    Ok(body.scale(&lead).shift(&scaled_v).truncate(target))
}

fn retry_until_term(e: &Expr, target: &BigRational, ctx: &mut Ctx) -> SResult<Series> {
    let mut t = target.clone();
    for _ in 0..6 {
        t = t + q(4);
        let s = expand_at(e, &t, ctx)?;
        if !s.terms.is_empty() {
            return Ok(s);
        }
        if s.order.is_none() {
            return Ok(s);
        }
    }
    Err(SeriesError::Precision)
}

/// Splits `a0 = alpha*LOG_T + beta` when linear in the placeholder.
fn split_log_linear(a0: &Expr) -> Option<(BigRational, Expr)> {
    let lt = Symbol::new(LOG_T);
    if is_free_of(a0, &lt) {
        return Some((BigRational::zero(), a0.clone()));
    }
    let at = |v: i64| normalize(&a0.subs(&HashMap::from([(lt.clone(), Expr::from_i64(v))])));
    let b0 = at(0);
    let b1 = at(1);
    let b2 = at(2);
    let alpha = normalize(&(b1.clone() - b0.clone()));
    let check = normalize(&(b2 - b0.clone() - alpha.clone() * Expr::from_i64(2)));
    if !check.is_zero() {
        return None;
    }
    number(&alpha).map(|a| (a, b0))
}

fn expand_function(name: &str, arg: &Expr, target: &BigRational, ctx: &mut Ctx) -> SResult<Series> {
    match name {
        "exp" => {
            let a = expand_at(arg, target, ctx)?;
            if a.valuation().is_some_and(|v| v.is_negative()) {
                return Err(SeriesError::Essential(format!("exp({arg})")));
            }
            let a0 = a
                .terms
                .get(&BigRational::zero())
                .cloned()
                .unwrap_or(Expr::from_i64(0));
            let mut u = a.clone();
            u.terms.remove(&BigRational::zero());
            let (alpha, beta) = split_log_linear(&a0)
                .ok_or_else(|| SeriesError::Unsupported(format!("exp of log term in {arg}")))?;
            let rel = target.clone() - alpha.clone();
            let u = if rel > *target && a.order.is_some() {
                let mut again = expand_at(arg, &rel, ctx)?;
                again.terms.remove(&BigRational::zero());
                again
            } else {
                u
            };
            let body = Series::compose(&u, |k| rat_expr(BigInt::from(1), factorial(k)), &rel)?;
            Ok(body
                .scale(&func("exp", beta))
                .shift(&alpha)
                .truncate(target))
        }
        "log" | "ln" => {
            let mut b = expand_at(arg, target, ctx)?;
            if b.terms.is_empty() {
                b = retry_until_term(arg, target, ctx)?;
            }
            let (v, _, _) = split_lead(&b)?;
            let need = target.clone() + v.clone();
            if b.order.as_ref().is_some_and(|o| o < &need) {
                b = expand_at(arg, &need, ctx)?;
            }
            let (v, c0, u) = split_lead(&b)?;
            let body = Series::compose(
                &u,
                |k| {
                    if k == 0 {
                        Expr::from_i64(0)
                    } else {
                        let sign = if k % 2 == 1 { 1 } else { -1 };
                        rat_expr(BigInt::from(sign), BigInt::from(k as i64))
                    }
                },
                target,
            )?;
            let constant = func("log", c0) + rational_expr(v) * ctx.log_t.clone();
            Ok(body
                .add(&Series::constant(normalize(&constant)))
                .truncate(target))
        }
        "Abs" => {
            let b = expand_at(arg, target, ctx)?;
            let Some((_, c0)) = b.lead() else {
                return Err(SeriesError::Precision);
            };
            match crate::gruntz::constant_sign(&c0) {
                Some(1) => Ok(b),
                Some(-1) => Ok(b.scale(&Expr::from_i64(-1))),
                _ => Err(SeriesError::Unsupported(format!("sign of {c0}"))),
            }
        }
        "atan" | "acot" => {
            let a = expand_at(arg, target, ctx)?;
            if a.valuation().is_some_and(|v| v.is_negative()) {
                // atan(a) = sign(a)*pi/2 - atan(1/a)
                let (_, c0, _) = split_lead(&a)?;
                let s = crate::gruntz::constant_sign(&c0)
                    .ok_or_else(|| SeriesError::Unsupported(format!("sign of {c0}")))?;
                let half_pi = Expr::Rational(BigRational::new(BigInt::from(s), BigInt::from(2)))
                    * Expr::Const(Constant::Pi);
                let recip = func_pow(arg, &Expr::from_i64(-1));
                let inner = expand_function("atan", &recip, target, ctx)?;
                return Ok(if name == "atan" {
                    Series::constant(half_pi).add(&inner.scale(&Expr::from_i64(-1)))
                } else {
                    inner
                }
                .truncate(target));
            }
            analytic(name, arg, a, target)
        }
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" | "sinh" | "cosh" | "tanh" | "coth"
        | "sech" | "csch" | "asin" | "acos" | "asinh" | "acosh" | "atanh" | "acoth" | "erf"
        | "erfc" | "gamma" | "sinc" => {
            let a = expand_at(arg, target, ctx)?;
            if a.valuation().is_some_and(|v| v.is_negative()) {
                return Err(SeriesError::Essential(format!("{name}({arg})")));
            }
            analytic(name, arg, a, target)
        }
        _ => Err(SeriesError::Unsupported(format!("{name}({arg})"))),
    }
}

fn analytic(name: &str, arg: &Expr, a: Series, target: &BigRational) -> SResult<Series> {
    let a0 = a
        .terms
        .get(&BigRational::zero())
        .cloned()
        .unwrap_or(Expr::from_i64(0));
    if !is_free_of(&a0, &Symbol::new(LOG_T)) {
        return Err(SeriesError::Unsupported(format!(
            "{name} of log term in {arg}"
        )));
    }
    let mut u = a;
    u.terms.remove(&BigRational::zero());
    let n = match u.valuation() {
        Some(v) if v.is_positive() => {
            let mut n = 0usize;
            while v.clone() * q(n as i64) < *target && n < MAX_TERMS {
                n += 1;
            }
            n
        }
        _ => 0,
    };
    let coeffs = taylor_coefficients(name, &a0, n)?;
    Series::compose(
        &u,
        |k| coeffs.get(k).cloned().unwrap_or(Expr::from_i64(0)),
        target,
    )
    .map(|s| s.truncate(target))
}

/// Expand `e` in `t -> 0+` to absolute order `target`; `log(t)` in the
/// coefficients is replaced by `log_t`.
pub fn series_in(
    e: &Expr,
    t: &Symbol,
    target: &BigRational,
    log_t: &Expr,
) -> Result<Series, SeriesError> {
    let mut ctx = Ctx {
        t,
        log_t: Expr::Sym(Symbol::new(LOG_T)),
        depth: 0,
    };
    let s = expand_at(e, target, &mut ctx)?;
    Ok(substitute_log(s, log_t))
}

fn substitute_log(s: Series, log_t: &Expr) -> Series {
    let lt = Symbol::new(LOG_T);
    let map = HashMap::from([(lt, log_t.clone())]);
    let mut terms = BTreeMap::new();
    for (k, c) in s.terms {
        let v = normalize(&c.subs(&map));
        if !v.is_zero() {
            terms.insert(k, v);
        }
    }
    Series {
        terms,
        order: s.order,
    }
}

/// Leading term `c * t^k` of `e` as `t -> 0+`, with `log(t)` replaced by
/// `log_t` in `c`. Coefficients whose zero-ness cannot be decided refuse.
pub fn leading_term(
    e: &Expr,
    t: &Symbol,
    log_t: &Expr,
) -> Result<(Expr, BigRational), SeriesError> {
    let lt = Symbol::new(LOG_T);
    let mut target = q(1);
    for _ in 0..8 {
        let mut ctx = Ctx {
            t,
            log_t: Expr::Sym(lt.clone()),
            depth: 0,
        };
        match expand_at(e, &target, &mut ctx) {
            Ok(s) => {
                for (k, c) in &s.terms {
                    let value = normalize(&c.subs(&HashMap::from([(lt.clone(), log_t.clone())])));
                    match crate::gruntz::is_zero_coefficient(&value) {
                        Some(true) => continue,
                        Some(false) => return Ok((value, k.clone())),
                        None => {
                            return Err(SeriesError::Unsupported(format!(
                                "cannot decide whether {value} is zero"
                            )));
                        }
                    }
                }
                if s.order.is_none() {
                    // Exactly zero.
                    return Ok((Expr::from_i64(0), BigRational::zero()));
                }
            }
            Err(SeriesError::Precision) => {}
            Err(other) => return Err(other),
        }
        target = target * q(2) + q(1);
    }
    Err(SeriesError::Precision)
}

/// Series of `e` in `t -> 0+` with at least `n` exponents of precision
/// beyond the valuation, as `(series, order)`.
pub fn series_terms(e: &Expr, t: &Symbol, n: i64, log_t: &Expr) -> Result<Series, SeriesError> {
    let target = q(n);
    let s = series_in(e, t, &target, log_t)?;
    Ok(s)
}
