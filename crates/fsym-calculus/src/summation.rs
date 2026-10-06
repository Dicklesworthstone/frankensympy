//! Closed forms for sums and products over an integer index.
//!
//! `summation(f, k, a, b)` is `sum_{k=a}^{b} f`; `b` may be symbolic or
//! `oo`. Exact rules only:
//!
//! * finite numeric ranges are added term by term (bounded);
//! * linearity and index-free factors;
//! * polynomials in `k` through Bernoulli-polynomial antidifferences
//!   (Faulhaber), so `sum k**2` is `n**3/3 + n**2/2 + n/6`;
//! * geometric terms `c*q**(u*k+v)` with a numeric ratio (`|q| < 1` for
//!   infinite sums);
//! * rational functions whose partial fractions telescope (linear factors at
//!   integer offsets, coefficients summing to zero), e.g.
//!   `sum 1/(k*(k+1)) = 1 - 1/(n+1)`;
//! * `sum_{k=1}^{oo} 1/k**s = zeta(s)` and `sum_{k=0}^{oo} x**k/k! = exp(x)`.
//!
//! `product(f, k, a, b)` covers numeric ranges, index-free factors,
//! `k -> factorial`, powers and geometric factors.

#![forbid(unsafe_code)]

use fsym_core::elementary::{eval_function, eval_pow, rational_expr};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use num_traits::{One, Signed, Zero};
use std::collections::HashMap;

const MAX_DIRECT_TERMS: i64 = 2000;

fn func(name: &str, arg: Expr) -> Expr {
    eval_function(name, std::slice::from_ref(&arg))
        .unwrap_or_else(|| Expr::Function(name.to_string(), vec![arg]))
}

fn number(e: &Expr) -> Option<BigRational> {
    match e {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn int_value(e: &Expr) -> Option<i64> {
    match e {
        Expr::Integer(n) => n.to_i64(),
        _ => None,
    }
}

fn is_free_of(e: &Expr, k: &Symbol) -> bool {
    !e.free_symbols().iter().any(|s| s == k)
}

fn at(e: &Expr, k: &Symbol, v: Expr) -> Expr {
    e.subs(&HashMap::from([(k.clone(), v)]))
}

fn is_oo(e: &Expr) -> bool {
    matches!(e, Expr::Const(Constant::Infinity))
}

fn expand(e: &Expr) -> Expr {
    fsym_simplify::expand(e)
}

fn binomial(n: i64, r: i64) -> BigRational {
    let mut acc = BigRational::one();
    for i in 0..r {
        acc = acc * BigRational::from_integer(BigInt::from(n - i))
            / BigRational::from_integer(BigInt::from(i + 1));
    }
    acc
}

/// Bernoulli numbers B_0..B_n with B_1 = -1/2.
fn bernoulli_numbers(n: usize) -> Vec<BigRational> {
    let mut b = vec![BigRational::one()];
    for m in 1..=n {
        let mut s = BigRational::zero();
        for (j, bj) in b.iter().enumerate() {
            s += binomial(m as i64 + 1, j as i64) * bj.clone();
        }
        b.push(-s / BigRational::from_integer(BigInt::from(m as i64 + 1)));
    }
    b
}

/// `F(x)` with `F(x+1) - F(x) = x**p` and `F(0) = 0`:
/// `(B_{p+1}(x) - B_{p+1})/(p+1)`.
fn power_antidifference(p: usize, x: &Expr) -> Expr {
    let n = p + 1;
    let b = bernoulli_numbers(n);
    let mut terms = Vec::new();
    for (j, bj) in b.iter().enumerate().take(n) {
        // B_n(x) = sum_j C(n,j) B_j x^(n-j); drop the constant j = n term.
        let c = binomial(n as i64, j as i64) * bj.clone()
            / BigRational::from_integer(BigInt::from(n as i64));
        if c.is_zero() {
            continue;
        }
        terms.push(rational_expr(c) * eval_pow(x.clone(), Expr::from_i64((n - j) as i64)));
    }
    terms.into_iter().fold(Expr::from_i64(0), |a, t| a + t)
}

/// Splits `e` into (index-free factor, index-dependent factors).
fn split_constant(e: &Expr, k: &Symbol) -> (Expr, Expr) {
    match e {
        Expr::Mul(fs) => {
            let mut c = Expr::from_i64(1);
            let mut rest = Expr::from_i64(1);
            for f in fs {
                if is_free_of(f, k) {
                    c = c * f.clone();
                } else {
                    rest = rest * f.clone();
                }
            }
            (c, rest)
        }
        other => (Expr::from_i64(1), other.clone()),
    }
}

/// `Some((q, c))` when `e == c * q**k` with `q`, `c` free of `k`.
fn geometric(e: &Expr, k: &Symbol) -> Option<(Expr, Expr)> {
    let ke = Expr::Sym(k.clone());
    let (base, exponent) = match e {
        Expr::Pow(b, x) => ((**b).clone(), (**x).clone()),
        Expr::Function(n, args) if n == "exp" && args.len() == 1 => {
            (Expr::Const(Constant::E), args[0].clone())
        }
        _ => return None,
    };
    if !is_free_of(&base, k) {
        return None;
    }
    let u = expand(&crate::diff(&exponent, k));
    if !is_free_of(&u, k) || u.is_zero() {
        return None;
    }
    let v = expand(&(exponent.clone() - u.clone() * ke));
    if !is_free_of(&v, k) {
        return None;
    }
    Some((eval_pow(base.clone(), u), eval_pow(base, v)))
}

fn polynomial_in(e: &Expr, k: &Symbol) -> Option<UnivariatePoly> {
    UnivariatePoly::from_expr(&expand(e), k).ok()
}

/// Sum over k in [a, b] of a monomial-sum polynomial with Expr coefficients.
fn sum_polynomial(e: &Expr, k: &Symbol, a: &Expr, b: &Expr) -> Option<Expr> {
    let p = polynomial_in(e, k)?;
    if is_oo(b) {
        // A nonzero polynomial summand diverges to the sign of its lead.
        let lead = p.coeffs.iter().rev().find(|c| !c.is_zero())?;
        return Some(Expr::Const(if lead.is_negative() {
            Constant::NegativeInfinity
        } else {
            Constant::Infinity
        }));
    }
    let mut total = Expr::from_i64(0);
    let upper = b.clone() + Expr::from_i64(1);
    for (deg, c) in p.coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let f_up = power_antidifference(deg, &upper);
        let f_lo = power_antidifference(deg, a);
        total = total + rational_expr(c.clone()) * (f_up - f_lo);
    }
    Some(expand(&total))
}

fn sum_geometric(e: &Expr, k: &Symbol, a: &Expr, b: &Expr) -> Option<Expr> {
    let (q, c) = geometric(e, k)?;
    let qv = crate::gruntz::complex_value(&q)?;
    if (qv.re - 1.0).abs() < 1e-12 && qv.im.abs() < 1e-12 {
        return None;
    }
    let one = Expr::from_i64(1);
    if is_oo(b) {
        if qv.norm() >= 1.0 {
            return None;
        }
        return Some(c * eval_pow(q.clone(), a.clone()) * eval_pow(one - q, Expr::from_i64(-1)));
    }
    let num = eval_pow(q.clone(), b.clone() + one.clone()) - eval_pow(q.clone(), a.clone());
    Some(fsym_simplify::simplify(
        &(c * num * eval_pow(q - one, Expr::from_i64(-1))),
    ))
}

/// Telescoping rational sums: partial fractions with linear factors
/// `k + r_i` at integer offsets whose coefficients cancel.
fn sum_rational(e: &Expr, k: &Symbol, a: &Expr, b: &Expr) -> Option<Expr> {
    let (num, den) = crate::integrate_rational_parts(e, k)?;
    if den.degree()? == 0 {
        return None;
    }
    let (q, rem) = num.div_rem(&den).ok()?;
    if !q.is_zero() {
        return None;
    }
    let fact = fsym_polys::factorization::complete_factorization(&den).ok()?;
    let scale = fact.scale.clone();
    // Only simple linear factors k + r with integer r.
    let mut roots: Vec<BigRational> = Vec::new();
    for t in &fact.factors {
        if t.multiplicity != 1 || t.poly.degree()? != 1 {
            return None;
        }
        let r = t.poly.coeffs[0].clone() / t.poly.coeffs[1].clone();
        if !r.is_integer() {
            return None;
        }
        roots.push(r);
    }
    // Residues: c_i = rem(-r_i) / (scale * prod_{j != i} (r_j - r_i)).
    let mut coeffs = Vec::new();
    for (i, ri) in roots.iter().enumerate() {
        let mut d = scale.clone();
        for (j, rj) in roots.iter().enumerate() {
            if i != j {
                d *= rj.clone() - ri.clone();
            }
        }
        coeffs.push(rem.eval(&(-ri.clone())) / d);
    }
    let total: BigRational = coeffs
        .iter()
        .fold(BigRational::zero(), |s, c| s + c.clone());
    if !total.is_zero() {
        return None;
    }
    // sum_{k=a}^{b} c_i/(k + r_i) = c_i*(H(b + r_i) - H(a - 1 + r_i)); with
    // m = min r_i, H(b + r) = H(b + m) + sum_{j=m+1}^{r} 1/(b + j) and the
    // H(b + m) parts cancel because the c_i sum to zero.
    let m = roots.iter().cloned().min()?;
    let mut out = Expr::from_i64(0);
    for (c, r) in coeffs.iter().zip(&roots) {
        let shift = (r.clone() - m.clone()).to_integer().to_i64()?;
        let ce = rational_expr(c.clone());
        // upper tail
        if !is_oo(b) {
            for j in 1..=shift {
                let jj = m.clone() + BigRational::from_integer(BigInt::from(j));
                out =
                    out + ce.clone() * eval_pow(b.clone() + rational_expr(jj), Expr::from_i64(-1));
            }
        }
        // lower: -c * H(a - 1 + r) relative to H(a - 1 + m)
        for j in 1..=shift {
            let jj = m.clone() + BigRational::from_integer(BigInt::from(j));
            out = out
                - ce.clone()
                    * eval_pow(
                        a.clone() - Expr::from_i64(1) + rational_expr(jj),
                        Expr::from_i64(-1),
                    );
        }
    }
    Some(fsym_simplify::simplify(&out))
}

fn sum_special_infinite(e: &Expr, k: &Symbol, a: &Expr) -> Option<Expr> {
    let ke = Expr::Sym(k.clone());
    // 1/k**s from a = 1 (or a > 1, subtracting the head).
    if let Expr::Pow(base, s) = e
        && **base == ke
        && let Some(sv) = number(s)
        && sv.is_negative()
    {
        let p = -sv;
        if p <= BigRational::one() {
            return Some(Expr::Const(Constant::Infinity));
        }
        let a0 = int_value(a)?;
        if a0 < 1 {
            return None;
        }
        let mut head = Expr::from_i64(0);
        for j in 1..a0 {
            head = head + eval_pow(Expr::from_i64(j), rational_expr(-p.clone()));
        }
        return Some(func("zeta", rational_expr(p)) - head);
    }
    None
}

/// `sum_{k=0}^{oo} x**k / k!` = `exp(x)`.
fn sum_exp_series(e: &Expr, k: &Symbol, a: &Expr) -> Option<Expr> {
    if !a.is_zero() {
        return None;
    }
    let ke = Expr::Sym(k.clone());
    let Expr::Mul(fs) = e else {
        return None;
    };
    let mut base = None;
    let mut has_fact = false;
    for f in fs {
        match f {
            Expr::Pow(b, p) if **p == ke && is_free_of(b, k) => base = Some((**b).clone()),
            Expr::Pow(b, p)
                if **p == Expr::from_i64(-1)
                    && matches!(b.as_ref(), Expr::Function(n, args) if n == "factorial" && args[0] == ke) =>
            {
                has_fact = true
            }
            _ => return None,
        }
    }
    if has_fact {
        base.map(|b| func("exp", b))
    } else {
        None
    }
}

/// Closed form of `sum_{k=a}^{b} f`, or `None`.
pub fn summation(f: &Expr, k: &Symbol, a: &Expr, b: &Expr) -> Option<Expr> {
    if let (Some(lo), Some(hi)) = (int_value(a), int_value(b)) {
        if hi < lo {
            if hi == lo - 1 {
                return Some(Expr::from_i64(0));
            }
            // upstream convention: sum_{a}^{b} = -sum_{b+1}^{a-1}
            return summation(f, k, &Expr::from_i64(hi + 1), &Expr::from_i64(lo - 1)).map(|s| -s);
        }
        if hi - lo <= MAX_DIRECT_TERMS {
            let mut total = Expr::from_i64(0);
            for i in lo..=hi {
                total = total + at(f, k, Expr::from_i64(i));
            }
            return Some(fsym_simplify::simplify(&total));
        }
    }
    if is_free_of(f, k) {
        if is_oo(b) {
            return if f.is_zero() {
                Some(Expr::from_i64(0))
            } else {
                None
            };
        }
        return Some(expand(
            &(f.clone() * (b.clone() - a.clone() + Expr::from_i64(1))),
        ));
    }
    if let Expr::Add(terms) = f {
        let mut parts = Vec::new();
        let mut poly_part = Expr::from_i64(0);
        for t in terms {
            if polynomial_in(t, k).is_some()
                && is_free_of(&split_constant(t, k).0, k)
                && !is_oo(b)
                && split_constant(t, k).0.free_symbols().is_empty()
            {
                poly_part = poly_part + t.clone();
            } else {
                parts.push(summation(t, k, a, b)?);
            }
        }
        let mut total = parts.into_iter().fold(Expr::from_i64(0), |x, y| x + y);
        if !poly_part.is_zero() {
            total = total + sum_polynomial(&poly_part, k, a, b)?;
        }
        return Some(total);
    }
    let (c, rest) = split_constant(f, k);
    if !c.is_one() {
        return summation(&rest, k, a, b).map(|s| c * s);
    }
    if let Some(s) = sum_polynomial(f, k, a, b) {
        return Some(s);
    }
    if let Some(s) = sum_geometric(f, k, a, b) {
        return Some(s);
    }
    if is_oo(b) {
        if let Some(s) = sum_special_infinite(f, k, a) {
            return Some(s);
        }
        if let Some(s) = sum_exp_series(f, k, a) {
            return Some(s);
        }
    }
    sum_rational(f, k, a, b)
}

/// Closed form of `prod_{k=a}^{b} f`, or `None`.
pub fn product(f: &Expr, k: &Symbol, a: &Expr, b: &Expr) -> Option<Expr> {
    if let (Some(lo), Some(hi)) = (int_value(a), int_value(b)) {
        if hi < lo {
            return if hi == lo - 1 {
                Some(Expr::from_i64(1))
            } else {
                None
            };
        }
        if hi - lo <= MAX_DIRECT_TERMS {
            let mut total = Expr::from_i64(1);
            for i in lo..=hi {
                total = total * at(f, k, Expr::from_i64(i));
            }
            return Some(fsym_simplify::simplify(&total));
        }
    }
    if is_oo(b) {
        return None;
    }
    let count = b.clone() - a.clone() + Expr::from_i64(1);
    if is_free_of(f, k) {
        return Some(eval_pow(f.clone(), count));
    }
    let ke = Expr::Sym(k.clone());
    if let Expr::Mul(fs) = f {
        let mut acc = Expr::from_i64(1);
        for g in fs {
            acc = acc * product(g, k, a, b)?;
        }
        return Some(acc);
    }
    if *f == ke && a.is_one() {
        return Some(func("factorial", b.clone()));
    }
    if let Expr::Pow(base, p) = f {
        if **base == ke && is_free_of(p, k) && a.is_one() {
            return Some(eval_pow(func("factorial", b.clone()), (**p).clone()));
        }
        if is_free_of(base, k) {
            let s = summation(p, k, a, b)?;
            return Some(eval_pow((**base).clone(), s));
        }
    }
    if let Expr::Function(n, args) = f
        && n == "exp"
    {
        let s = summation(&args[0], k, a, b)?;
        return Some(func("exp", s));
    }
    None
}
