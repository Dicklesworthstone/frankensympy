//! Closed forms for sums and products over an integer index.
//!
//! `summation(f, k, a, b)` is `sum_{k=a}^{b} f`; `b` may be symbolic or
//! `oo`. Exact rules only:
//!
//! * finite numeric ranges are added term by term (bounded);
//! * linearity and index-free factors;
//! * polynomials in `k` through Bernoulli-polynomial antidifferences
//!   (Faulhaber), so `sum k**2` is `n**3/3 + n**2/2 + n/6`;
//! * geometric terms `c*q**(u*k+v)` with an exact rational ratio (`|q| < 1` for
//!   infinite sums);
//! * independently checked rational telescoping, including repeated poles
//!   and fractional offsets, at exact integer bounds, e.g.
//!   `sum_{k=1}^{10000} 1/(k*(k+1)) = 10000/10001`;
//! * `sum_{k=1}^{oo} 1/k**s = zeta(s)` and `sum_{k=0}^{oo} x**k/k! = exp(x)`.
//!
//! `product(f, k, a, b)` covers numeric ranges, index-free factors,
//! `k -> factorial`, powers and geometric factors.

#![forbid(unsafe_code)]

use fsym_core::elementary::{eval_function, eval_pow, rational_expr};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_sum::RationalSumUpper;
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
    // Exact rational convergence decisions: a ratio arbitrarily close to
    // one must neither be rounded to one nor cross the convergence boundary.
    let qv = number(&q)?;
    let one = Expr::from_i64(1);
    if qv.is_one() {
        return (!is_oo(b)).then(|| c * (b.clone() - a.clone() + one));
    }
    if qv.is_zero() {
        // Splitting 0^(u*k+v) into 0^v * (0^u)^k can create undefined
        // factors even when the original term is defined. Admit only the
        // literal 0^k family here; other zero-base forms stay unevaluated.
        if !matches!(e, Expr::Pow(base, exponent)
            if number(base).is_some_and(|value| value.is_zero())
                && exponent.as_ref() == &Expr::Sym(k.clone()))
        {
            return None;
        }
        let lo = int_value(a)?;
        if lo < 0 {
            return None;
        }
        if is_oo(b) {
            return Some(if lo == 0 { c } else { Expr::from_i64(0) });
        }
        let hi = int_value(b)?;
        return (hi >= lo).then(|| if lo == 0 { c } else { Expr::from_i64(0) });
    }
    if is_oo(b) {
        if qv.abs() >= BigRational::one() || !matches!(a, Expr::Integer(_)) {
            return None;
        }
        return Some(c * eval_pow(q.clone(), a.clone()) * eval_pow(one - q, Expr::from_i64(-1)));
    }
    let num = eval_pow(q.clone(), b.clone() + one.clone()) - eval_pow(q.clone(), a.clone());
    Some(fsym_simplify::simplify(
        &(c * num * eval_pow(q - one, Expr::from_i64(-1))),
    ))
}

/// Integer-range rational sums preserve the original denominator before
/// cancellation. Unsupported symbolic bounds retain an unevaluated sum.
fn sum_rational(e: &Expr, k: &Symbol, a: &Expr, b: &Expr) -> Option<Expr> {
    let (num, den) = crate::rational_sum::original_rational_parts(e, k)?;
    sum_rational_parts(&num, &den, a, b)
}

fn sum_rational_parts(
    num: &UnivariatePoly,
    den: &UnivariatePoly,
    a: &Expr,
    b: &Expr,
) -> Option<Expr> {
    let Expr::Integer(lower) = a else {
        return None;
    };
    let upper = match b {
        Expr::Integer(upper) => RationalSumUpper::Finite(upper.clone()),
        Expr::Const(Constant::Infinity) => RationalSumUpper::PositiveInfinity,
        _ => return None,
    };
    // A rational function need not have a rational antidifference. Short
    // finite sums keep their direct exact lane, without first paying for
    // factorization. Original poles and coefficient growth are checked.
    if let RationalSumUpper::Finite(upper) = &upper
        && let (Some(mut lo), Some(mut hi)) = (lower.to_i64(), upper.to_i64())
    {
        let mut negate = false;
        if hi < lo {
            let next = hi.checked_add(1)?;
            if next == lo {
                return Some(Expr::from_i64(0));
            }
            hi = lo.checked_sub(1)?;
            lo = next;
            negate = true;
        }
        if hi
            .checked_sub(lo)
            .is_some_and(|span| span <= MAX_DIRECT_TERMS)
        {
            let mut total = BigRational::zero();
            for index in lo..=hi {
                let point = BigRational::from_integer(BigInt::from(index));
                let denominator = den.eval(&point);
                if denominator.is_zero() {
                    return None;
                }
                total += num.eval(&point) / denominator;
                if total.numer().bits().max(total.denom().bits()) > 16_384 {
                    return None;
                }
            }
            return Some(rational_expr(if negate { -total } else { total }));
        }
    }
    crate::rational_sum::rational_sum(num, den, lower, &upper)
        .ok()
        .map(|result| rational_expr(result.value))
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
        if a0 < 1 || a0.checked_sub(1)? > MAX_DIRECT_TERMS {
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
    // Try the whole rational expression before linearity: separate terms
    // can diverge while their rational combination telescopes. Retaining
    // the unsimplified denominator also prevents canceled input poles from
    // disappearing through direct evaluation or constant extraction.
    if crate::rational_sum::is_rational_quotient(f, k)? {
        let (numerator, denominator) = crate::rational_sum::original_rational_parts(f, k)?;
        return sum_rational_parts(&numerator, &denominator, a, b).or_else(|| {
            // The existing zeta family is not rationally telescoping. Its
            // separate convergence rule is safe only above the pole at zero.
            if is_oo(b) && int_value(a).is_some_and(|lower| lower >= 1) {
                sum_special_infinite(f, k, a)
            } else {
                None
            }
        });
    }
    if let (Some(lo), Some(hi)) = (int_value(a), int_value(b)) {
        if hi < lo {
            if hi.checked_add(1) == Some(lo) {
                return Some(Expr::from_i64(0));
            }
            // upstream convention: sum_{a}^{b} = -sum_{b+1}^{a-1}
            return summation(
                f,
                k,
                &Expr::from_i64(hi.checked_add(1)?),
                &Expr::from_i64(lo.checked_sub(1)?),
            )
            .map(|s| -s);
        }
        if hi
            .checked_sub(lo)
            .is_some_and(|length| length <= MAX_DIRECT_TERMS)
        {
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
