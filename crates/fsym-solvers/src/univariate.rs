//! Univariate equation solving: exact polynomial roots and inversion.
//!
//! * Polynomials over Q are factored completely; each irreducible factor is
//!   solved exactly when it is linear, quadratic, a binomial `x**n - c`,
//!   biquadratic, or a cubic (Cardano). Other irreducible factors refuse
//!   (upstream answers with `CRootOf`), so a returned list is always complete.
//! * Rational equations solve the numerator and drop roots of the
//!   denominator.
//! * Equations in one transcendental generator are solved by inversion
//!   (`exp`, `log`, `sin`, `cos`, `tan`, real powers) or by solving a
//!   polynomial in the generator first; candidate roots are checked by
//!   substitution and extraneous ones dropped.
//!
//! Results follow upstream ordering: real roots ascending, then non-real
//! roots by real part and imaginary part.

#![forbid(unsafe_code)]

use crate::SolverError;
use fsym_core::elementary::{eval_function, eval_pow, rational_expr};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use num_traits::{One, Signed, Zero};
use std::collections::HashMap;

fn func(name: &str, arg: Expr) -> Expr {
    eval_function(name, std::slice::from_ref(&arg))
        .unwrap_or_else(|| Expr::Function(name.to_string(), vec![arg]))
}

fn rq(r: BigRational) -> Expr {
    rational_expr(r)
}

fn q(n: i64, d: i64) -> Expr {
    rq(BigRational::new(BigInt::from(n), BigInt::from(d)))
}

fn pi() -> Expr {
    Expr::Const(Constant::Pi)
}

fn imag() -> Expr {
    Expr::Const(Constant::I)
}

fn is_free_of(e: &Expr, x: &Symbol) -> bool {
    !e.free_symbols().iter().any(|s| s == x)
}

fn number(e: &Expr) -> Option<BigRational> {
    match e {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn expand(e: &Expr) -> Expr {
    fsym_simplify::expand(e)
}

fn simplify(e: &Expr) -> Expr {
    fsym_simplify::simplify(e)
}

/// `(re, im)` floating value of a constant expression.
fn approx(e: &Expr) -> Option<(f64, f64)> {
    fsym_calculus::gruntz::complex_value(e).map(|v| (v.re, v.im))
}

fn certified_ball(e: &Expr) -> Option<fsym_core::ball::ComplexBall> {
    if !e.free_symbols().is_empty() {
        return None;
    }
    e.eval_complex_ball(24).ok()
}

/// Upstream root order: reals ascending, then complex by (re, im).
pub fn sort_roots(mut roots: Vec<Expr>) -> Vec<Expr> {
    if roots
        .iter()
        .any(|r| certified_ball(r).is_none() && approx(r).is_none())
    {
        // Symbolic roots keep their construction order (upstream lists the
        // minus branch of a symbolic radical first).
        return roots;
    }
    roots.sort_by(|a, b| {
        if let (Some(ba), Some(bb)) = (certified_ball(a), certified_ball(b)) {
            let a_real = ba.imag().contains_zero();
            let b_real = bb.imag().contains_zero();
            let real_tag_a = if a_real { 0u8 } else { 1u8 };
            let real_tag_b = if b_real { 0u8 } else { 1u8 };
            return real_tag_a
                .cmp(&real_tag_b)
                .then_with(|| ba.real().midpoint().cmp(bb.real().midpoint()))
                .then_with(|| ba.imag().midpoint().cmp(bb.imag().midpoint()));
        }
        let key = |e: &Expr| {
            let (re, im) = approx(e).unwrap_or((f64::NAN, 0.0));
            let is_real = im.abs() < 1e-12;
            (if is_real { 0 } else { 1 }, re, im)
        };
        let (ka, kb) = (key(a), key(b));
        ka.0.cmp(&kb.0)
            .then(ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
            .then(ka.2.partial_cmp(&kb.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    roots
}

fn dedup(roots: Vec<Expr>) -> Vec<Expr> {
    let mut out: Vec<Expr> = Vec::new();
    for r in roots {
        let dup = out.iter().any(|o| {
            if *o == r {
                return true;
            }
            let diff_expr = o.clone() - r.clone();
            if let (Some(ba), Some(bb)) = (certified_ball(o), certified_ball(&r)) {
                return ba.sub(&bb).contains_zero()
                    && (simplify(&expand(&diff_expr)).is_zero()
                        || fsym_simplify::cancel(&expand(&diff_expr)).is_zero());
            }
            match (approx(o), approx(&r)) {
                (Some(a), Some(b)) => {
                    (a.0 - b.0).abs() < 1e-12
                        && (a.1 - b.1).abs() < 1e-12
                        && simplify(&expand(&diff_expr)).is_zero()
                }
                _ => false,
            }
        });
        if !dup {
            out.push(r);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Polynomials over Q
// ---------------------------------------------------------------------------

/// `cos(2*pi*k/n) + I*sin(2*pi*k/n)` through the exact special-value table.
fn unit_root(k: i64, n: i64, offset: BigRational) -> Expr {
    let angle = (BigRational::from_integer(BigInt::from(2 * k)) + offset)
        / BigRational::from_integer(BigInt::from(n));
    let arg = rq(angle) * pi();
    expand(&(func("cos", arg.clone()) + imag() * func("sin", arg)))
}

fn quadratic_roots(a: &BigRational, b: &BigRational, c: &BigRational) -> Vec<Expr> {
    let disc =
        b.clone() * b.clone() - BigRational::from_integer(BigInt::from(4)) * a.clone() * c.clone();
    let sq = eval_pow(rq(disc), q(1, 2));
    let inv = rq(BigRational::one() / (BigRational::from_integer(BigInt::from(2)) * a.clone()));
    let mb = rq(-b.clone());
    vec![
        expand(&((mb.clone() - sq.clone()) * inv.clone())),
        expand(&((mb + sq) * inv)),
    ]
}

/// Roots of an irreducible (or square-free) factor over Q.
fn factor_roots(p: &UnivariatePoly) -> Result<Vec<Expr>, SolverError> {
    let d = p.degree().ok_or(SolverError::InfiniteSolutions)?;
    let c = &p.coeffs;
    match d {
        0 => Ok(vec![]),
        1 => Ok(vec![rq(-c[0].clone() / c[1].clone())]),
        2 => Ok(quadratic_roots(&c[2], &c[1], &c[0])),
        _ => {
            // Binomial a*x**n + c0.
            if c[1..d].iter().all(|v| v.is_zero()) {
                let r = -c[0].clone() / c[d].clone();
                let n = d as i64;
                let (mag, offset) = if r.is_negative() {
                    (-r, BigRational::one())
                } else {
                    (r, BigRational::zero())
                };
                let base = eval_pow(rq(mag), q(1, n));
                return Ok((0..n)
                    .map(|k| expand(&(base.clone() * unit_root(k, n, offset.clone()))))
                    .collect());
            }
            // Biquadratic a x^4 + b x^2 + c.
            if d == 4 && c[1].is_zero() && c[3].is_zero() {
                let mut out = Vec::new();
                for y in quadratic_roots(&c[4], &c[2], &c[0]) {
                    let s = eval_pow(y, q(1, 2));
                    out.push(expand(&(-s.clone())));
                    out.push(s);
                }
                return Ok(out);
            }
            if d == 3 {
                return Ok(cardano(&c[3], &c[2], &c[1], &c[0]));
            }
            Err(SolverError::UnsupportedDegree(d))
        }
    }
}

/// Cardano's formula for a cubic with rational coefficients.
fn cardano(a: &BigRational, b: &BigRational, c: &BigRational, d: &BigRational) -> Vec<Expr> {
    let three = BigRational::from_integer(BigInt::from(3));
    let (b, c, d) = (
        b.clone() / a.clone(),
        c.clone() / a.clone(),
        d.clone() / a.clone(),
    );
    // x = t - b/3 ; t^3 + p t + q = 0
    let p = c.clone() - b.clone() * b.clone() / three.clone();
    let qq = BigRational::from_integer(BigInt::from(2)) * b.clone() * b.clone() * b.clone()
        / BigRational::from_integer(BigInt::from(27))
        - b.clone() * c / three.clone()
        + d;
    let shift = rq(-b / three.clone());
    let half_q = rq(-qq.clone() / BigRational::from_integer(BigInt::from(2)));
    let inner = qq.clone() * qq / BigRational::from_integer(BigInt::from(4))
        + p.clone() * p.clone() * p.clone() / BigRational::from_integer(BigInt::from(27));
    let u = eval_pow(half_q + eval_pow(rq(inner), q(1, 2)), q(1, 3));
    let omega = [
        unit_root(0, 3, BigRational::zero()),
        unit_root(1, 3, BigRational::zero()),
        unit_root(2, 3, BigRational::zero()),
    ];
    omega
        .iter()
        .map(|w| {
            let uk = w.clone() * u.clone();
            let t = if p.is_zero() {
                uk
            } else {
                uk.clone() - rq(p.clone() / three.clone()) * eval_pow(uk, Expr::from_i64(-1))
            };
            expand(&(t + shift.clone()))
        })
        .collect()
}

/// Exact roots (with multiplicity) of a polynomial over Q.
pub fn poly_roots(p: &UnivariatePoly) -> Result<Vec<(Expr, usize)>, SolverError> {
    if p.is_zero() {
        return Err(SolverError::InfiniteSolutions);
    }
    if p.degree() == Some(0) {
        return Ok(vec![]);
    }
    let fact = fsym_polys::factorization::complete_factorization(p)
        .map_err(|e| SolverError::InvalidSystem(e.to_string()))?;
    let mut out = Vec::new();
    for t in &fact.factors {
        for r in factor_roots(&t.poly)? {
            out.push((r, t.multiplicity));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Rational equations
// ---------------------------------------------------------------------------

fn rational_parts(e: &Expr, x: &Symbol) -> Option<(UnivariatePoly, UnivariatePoly)> {
    let one = || UnivariatePoly::one(x.clone());
    match e {
        Expr::Integer(_) | Expr::Rational(_) => {
            Some((UnivariatePoly::new(x.clone(), vec![number(e)?]), one()))
        }
        Expr::Sym(s) if s == x => Some((
            UnivariatePoly::new(x.clone(), vec![BigRational::zero(), BigRational::one()]),
            one(),
        )),
        Expr::Add(ts) => {
            let mut acc: Option<(UnivariatePoly, UnivariatePoly)> = None;
            for t in ts {
                let (n, d) = rational_parts(t, x)?;
                acc = Some(match acc {
                    None => (n, d),
                    Some((an, ad)) => (
                        an.mul(&d).ok()?.add(&n.mul(&ad).ok()?).ok()?,
                        ad.mul(&d).ok()?,
                    ),
                });
            }
            acc
        }
        Expr::Mul(fs) => {
            let mut n = one();
            let mut d = one();
            for f in fs {
                let (fnn, fd) = rational_parts(f, x)?;
                n = n.mul(&fnn).ok()?;
                d = d.mul(&fd).ok()?;
            }
            Some((n, d))
        }
        Expr::Pow(b, p) => {
            let k = match p.as_ref() {
                Expr::Integer(k) => k.to_i64()?,
                _ => return None,
            };
            if k.unsigned_abs() > 64 {
                return None;
            }
            let (bn, bd) = rational_parts(b, x)?;
            let kk = u32::try_from(k.unsigned_abs()).ok()?;
            if k >= 0 {
                Some((bn.pow(kk).ok()?, bd.pow(kk).ok()?))
            } else {
                Some((bd.pow(kk).ok()?, bn.pow(kk).ok()?))
            }
        }
        _ => None,
    }
}

fn solve_rational(e: &Expr, x: &Symbol) -> Option<Result<Vec<Expr>, SolverError>> {
    let (mut num, den) = rational_parts(e, x)?;
    if den.is_zero() {
        return Some(Err(SolverError::InvalidSystem(
            "the original rational expression has an identically zero denominator".into(),
        )));
    }
    // A reciprocal of a reciprocal still excludes the inner pole. Keep
    // every original inverse's numerator as a domain constraint; the final
    // numerator/denominator pair alone cannot describe those holes.
    let mut excluded = vec![den];
    let mut stack = vec![e];
    while let Some(term) = stack.pop() {
        match term {
            Expr::Pow(base, exponent) => {
                if number(exponent).is_some_and(|n| n.is_negative()) {
                    let (base_num, _) = rational_parts(base, x)?;
                    if base_num.is_zero() {
                        return Some(Err(SolverError::InvalidSystem(
                            "an original reciprocal has an identically zero base".into(),
                        )));
                    }
                    excluded.push(base_num);
                }
                stack.push(base);
                stack.push(exponent);
            }
            Expr::Add(args) | Expr::Mul(args) => stack.extend(args),
            _ => {}
        }
    }
    if num.is_zero() {
        return Some(Err(SolverError::InfiniteSolutions));
    }
    // Remove every numerator root forbidden by the ORIGINAL denominator,
    // including algebraic poles and repeated numerator factors. Cancelling
    // a fraction once loses the hole at x = 1 in (x - 1)^2/(x - 1).
    // Repeated gcd division is exact saturation over Q; each iteration
    // strictly lowers the numerator degree.
    for denominator in excluded {
        loop {
            let g = num.gcd(&denominator).ok()?;
            if g.degree().unwrap_or(0) == 0 {
                break;
            }
            num = num.div_rem(&g).ok()?.0;
        }
    }
    let roots = match poly_roots(&num) {
        Ok(r) => r,
        Err(err) => return Some(Err(err)),
    };
    let mut out = Vec::new();
    for (r, _) in roots {
        out.push(r);
    }
    Some(Ok(sort_roots(dedup(out))))
}

// ---------------------------------------------------------------------------
// Inversion
// ---------------------------------------------------------------------------

/// Solves `lhs = rhs` for `x` by peeling invertible outer operations.
fn invert(lhs: &Expr, rhs: Expr, x: &Symbol, depth: usize) -> Result<Vec<Expr>, SolverError> {
    if depth > 24 {
        return Err(SolverError::NonLinear);
    }
    let xe = Expr::Sym(x.clone());
    if *lhs == xe {
        return Ok(vec![rhs]);
    }
    match lhs {
        Expr::Add(terms) => {
            let (dep, free): (Vec<Expr>, Vec<Expr>) =
                terms.iter().cloned().partition(|t| !is_free_of(t, x));
            if free.is_empty() {
                return solve_generic(&(lhs.clone() - rhs), x);
            }
            let moved = free.into_iter().fold(rhs, |acc, t| acc - t);
            let rest = dep.into_iter().fold(Expr::from_i64(0), |a, t| a + t);
            invert(&rest, moved, x, depth + 1)
        }
        Expr::Mul(factors) => {
            let (dep, free): (Vec<Expr>, Vec<Expr>) =
                factors.iter().cloned().partition(|t| !is_free_of(t, x));
            if free.is_empty() {
                return solve_generic(&(lhs.clone() - rhs), x);
            }
            let c = free.into_iter().fold(Expr::from_i64(1), |a, t| a * t);
            let rest = dep.into_iter().fold(Expr::from_i64(1), |a, t| a * t);
            invert(&rest, rhs * eval_pow(c, Expr::from_i64(-1)), x, depth + 1)
        }
        Expr::Pow(b, p) if is_free_of(p, x) => {
            let pv = number(p);
            match pv {
                Some(v) if v.is_integer() && v.is_positive() && !is_free_of(b, x) => {
                    // b**n = rhs -> b = rhs**(1/n) * unit roots (complex)
                    let n = v.to_integer().to_i64().ok_or(SolverError::NonLinear)?;
                    if n > 12 {
                        return Err(SolverError::NonLinear);
                    }
                    let principal = eval_pow(rhs, q(1, n));
                    let mut out = Vec::new();
                    // Square roots list the negative branch first, as
                    // upstream solve orders -sqrt(r), sqrt(r).
                    let order: Vec<i64> = if n == 2 { vec![1, 0] } else { (0..n).collect() };
                    for k in order {
                        let cand =
                            expand(&(principal.clone() * unit_root(k, n, BigRational::zero())));
                        out.extend(invert(b, cand, x, depth + 1)?);
                    }
                    Ok(out)
                }
                Some(v) if !v.is_integer() => {
                    // b**(p/q) = rhs -> b = rhs**(q/p), checked later.
                    let inv = rq(BigRational::one() / v);
                    invert(b, eval_pow(rhs, inv), x, depth + 1)
                }
                Some(v) if v.is_negative() && !is_free_of(b, x) => {
                    // b**(-n) = rhs -> b**n = 1/rhs: every n-th root
                    // branch, not only the principal one (1/y**2 = x has
                    // y = -1/sqrt(x) and y = 1/sqrt(x)).
                    let pos = eval_pow((**b).clone(), rq(-v));
                    invert(&pos, eval_pow(rhs, Expr::from_i64(-1)), x, depth + 1)
                }
                _ => Err(SolverError::NonLinear),
            }
        }
        Expr::Pow(b, p) if is_free_of(b, x) => {
            // b**p = rhs -> p = log(rhs)/log(b), exact when rhs = b**m.
            if let (Some(bv), Some(rv)) = (number(b), number(&rhs))
                && bv.is_positive()
                && !bv.is_one()
            {
                for m in -64i32..=64 {
                    if let Ok(v) = bv.pow(m)
                        && v == rv
                    {
                        return invert(p, Expr::from_i64(i64::from(m)), x, depth + 1);
                    }
                }
            }
            let target =
                func("log", rhs) * eval_pow(func("log", (**b).clone()), Expr::from_i64(-1));
            invert(p, target, x, depth + 1)
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let inner = &args[0];
            match name.as_str() {
                "exp" => invert(inner, func("log", rhs), x, depth + 1),
                "log" => invert(inner, func("exp", rhs), x, depth + 1),
                "sin" => {
                    let a = func("asin", rhs);
                    let mut out = invert(inner, a.clone(), x, depth + 1)?;
                    out.extend(invert(inner, pi() - a, x, depth + 1)?);
                    Ok(out)
                }
                "cos" => {
                    let a = func("acos", rhs);
                    let mut out = invert(inner, a.clone(), x, depth + 1)?;
                    out.extend(invert(inner, Expr::from_i64(2) * pi() - a, x, depth + 1)?);
                    Ok(out)
                }
                "tan" => invert(inner, func("atan", rhs), x, depth + 1),
                "asin" => invert(inner, func("sin", rhs), x, depth + 1),
                "acos" => invert(inner, func("cos", rhs), x, depth + 1),
                "atan" => invert(inner, func("tan", rhs), x, depth + 1),
                "sinh" => invert(inner, func("asinh", rhs), x, depth + 1),
                "tanh" => invert(inner, func("atanh", rhs), x, depth + 1),
                _ => Err(SolverError::NonLinear),
            }
        }
        _ => Err(SolverError::NonLinear),
    }
}

/// Polynomial in one generator g(x): solve in u, then invert g = u.
fn solve_in_generator(e: &Expr, x: &Symbol) -> Result<Vec<Expr>, SolverError> {
    let mut gens: Vec<Expr> = Vec::new();
    collect_generators(e, x, &mut gens);
    if gens.len() != 1 {
        return Err(SolverError::NonLinear);
    }
    let g = gens.pop().expect("one generator");
    let u = Symbol::new("_fsym_solve_u");
    let ue = Expr::Sym(u.clone());
    let in_u = rewrite_in_generator(e, &g, &ue, x).ok_or(SolverError::NonLinear)?;
    if !is_free_of(&in_u, x) {
        return Err(SolverError::NonLinear);
    }
    let us = solve_rational(&expand(&in_u), &u).ok_or(SolverError::NonLinear)??;
    let mut out = Vec::new();
    for val in us {
        // A generator value we cannot invert means the solution set is not
        // established: refuse instead of dropping it.
        out.extend(invert(&g, val, x, 0)?);
    }
    Ok(out)
}

/// exp(k*a) -> exp(a)**k for the generator exp(a).
fn rewrite_in_generator(e: &Expr, g: &Expr, u: &Expr, x: &Symbol) -> Option<Expr> {
    if e == g {
        return Some(u.clone());
    }
    if is_free_of(e, x) {
        return Some(e.clone());
    }
    match e {
        Expr::Add(xs) => Some(
            xs.iter()
                .map(|t| rewrite_in_generator(t, g, u, x))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .fold(Expr::from_i64(0), |a, b| a + b),
        ),
        Expr::Mul(xs) => Some(
            xs.iter()
                .map(|t| rewrite_in_generator(t, g, u, x))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .fold(Expr::from_i64(1), |a, b| a * b),
        ),
        Expr::Pow(b, p) => Some(eval_pow(rewrite_in_generator(b, g, u, x)?, (**p).clone())),
        Expr::Function(n, args) if n == "exp" => {
            let Expr::Function(_, gargs) = g else {
                return None;
            };
            let ratio =
                simplify(&(args[0].clone() * eval_pow(gargs[0].clone(), Expr::from_i64(-1))));
            let k = number(&ratio)?;
            if !k.is_integer() {
                return None;
            }
            Some(eval_pow(u.clone(), rq(k)))
        }
        _ => None,
    }
}

fn collect_generators(e: &Expr, x: &Symbol, out: &mut Vec<Expr>) {
    if is_free_of(e, x) {
        return;
    }
    match e {
        Expr::Sym(_) => {
            if !out.contains(e) {
                out.push(e.clone());
            }
        }
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().for_each(|t| collect_generators(t, x, out)),
        Expr::Pow(b, p) if is_free_of(p, x) && number(p).is_some_and(|v| v.is_integer()) => {
            collect_generators(b, x, out)
        }
        Expr::Function(n, args) if n == "exp" => {
            // exp(k*a) shares the generator exp(a) with the smallest |k|.
            let (c, rest) = split_coeff(&args[0]);
            let base = Expr::Function("exp".into(), vec![rest.clone()]);
            let _ = c;
            if !out.contains(&base) {
                out.push(base);
            }
        }
        other => {
            if !out.contains(other) {
                out.push(other.clone());
            }
        }
    }
}

fn split_coeff(e: &Expr) -> (BigRational, Expr) {
    if let Expr::Mul(fs) = e
        && let Some(c) = number(&fs[0])
    {
        let rest = fs[1..]
            .iter()
            .cloned()
            .fold(Expr::from_i64(1), |a, b| a * b);
        return (c, rest);
    }
    (BigRational::one(), e.clone())
}

fn solve_generic(e: &Expr, x: &Symbol) -> Result<Vec<Expr>, SolverError> {
    if let Some(r) = solve_rational(e, x) {
        return r;
    }
    solve_in_generator(e, x)
}

/// A checked algebraic identity, with resource/unsupported failures left
/// undecided. Callers check the original expression's domain separately.
fn exact_zero(e: &Expr) -> bool {
    let Ok(s) = fsym_simplify::try_simplify(e) else {
        return false;
    };
    if s.is_zero() {
        return true;
    }
    let Ok(expanded) = fsym_simplify::try_expand(&s) else {
        return false;
    };
    fsym_simplify::try_simplify(&expanded).is_ok_and(|v| v.is_zero())
        || fsym_simplify::try_cancel(&expanded).is_ok_and(|v| v.is_zero())
}

/// `Some(true)` proves zero; `Some(false)` proves nonzero. A ball merely
/// containing zero proves neither. Refinement is bounded and never falls
/// back to a floating-point tolerance or to acceptance of unknown values.
fn zero_status(e: &Expr) -> Option<bool> {
    if exact_zero(e) {
        return Some(true);
    }
    if !e.free_symbols().is_empty() {
        return None;
    }
    for digits in [24, 48, 96] {
        if let Ok(ball) = e.eval_complex_ball(digits) {
            if !ball.contains_zero() {
                return Some(false);
            }
            if ball.real().midpoint().is_zero()
                && ball.real().radius().is_zero()
                && ball.imag().midpoint().is_zero()
                && ball.imag().radius().is_zero()
            {
                return Some(true);
            }
        }
    }
    None
}

fn real_sign(e: &Expr) -> Option<std::cmp::Ordering> {
    if let Some(q) = number(e) {
        return Some(q.cmp(&BigRational::zero()));
    }
    if !e.free_symbols().is_empty() {
        return None;
    }
    for digits in [24, 48, 96] {
        if let Ok(ball) = e.evalf_ball(digits) {
            if ball.is_positive() {
                return Some(std::cmp::Ordering::Greater);
            }
            if ball.is_negative() {
                return Some(std::cmp::Ordering::Less);
            }
            if ball.midpoint().is_zero() && ball.radius().is_zero() {
                return Some(std::cmp::Ordering::Equal);
            }
        }
    }
    None
}

/// Verify a ground residual P + Q*sqrt(g) without mistaking a vanishing
/// numerical enclosure for equality. The polynomial identity P² = Q²*g
/// and the certified sign of -P/Q fix the principal square-root branch.
/// This retains irrational solutions such as sqrt(x) = 1 - x while
/// rejecting the other root introduced by squaring that equation.
fn square_root_residual(e: &Expr) -> Option<bool> {
    if !e.free_symbols().is_empty() {
        return None;
    }
    let mut stack = vec![e];
    let mut radicals = Vec::new();
    while let Some(term) = stack.pop() {
        match term {
            Expr::Pow(base, exponent) => {
                if number(exponent) == Some(BigRational::new(1.into(), 2.into())) {
                    radicals.push((term, base.as_ref()));
                }
                stack.push(base);
                stack.push(exponent);
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => stack.extend(args),
            _ => {}
        }
    }
    // The residual is ground, so this temporary symbol cannot capture any
    // free input symbol. No printer text participates in the identity check.
    let u = Symbol::new("_fsym_root_check");
    let ue = Expr::Sym(u.clone());
    for (radical, radicand) in radicals {
        let with_u = e.subs_expr(radical, &ue);
        let p = simplify(&with_u.subs(&HashMap::from([(u.clone(), Expr::from_i64(0))])));
        let q =
            simplify(&(with_u.subs(&HashMap::from([(u.clone(), Expr::from_i64(1))])) - p.clone()));
        if !exact_zero(&(with_u - p.clone() - q.clone() * ue.clone()))
            || zero_status(&q) != Some(false)
            || !exact_zero(&(p.clone() * p.clone() - q.clone() * q.clone() * radicand.clone()))
        {
            continue;
        }
        let target = simplify(&(-p * eval_pow(q, Expr::from_i64(-1))));
        if let Some(sign) = real_sign(&target) {
            return Some(sign != std::cmp::Ordering::Less);
        }
    }
    None
}

fn contains_nonfinite(e: &Expr) -> bool {
    let mut stack = vec![e];
    while let Some(term) = stack.pop() {
        match term {
            Expr::Const(
                Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity
                | Constant::NaN,
            ) => return true,
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => stack.extend(args),
            Expr::Pow(base, exponent) => {
                stack.push(base);
                stack.push(exponent);
            }
            _ => {}
        }
    }
    false
}

/// Check denominator/log constraints before substitution can cancel them.
fn candidate_in_domain(e: &Expr, x: &Symbol, root: &Expr) -> Result<bool, SolverError> {
    let replacements = HashMap::from([(x.clone(), root.clone())]);
    let mut stack = vec![e];
    while let Some(term) = stack.pop() {
        let nonzero_requirement = match term {
            Expr::Pow(base, exponent) if number(exponent).is_some_and(|n| n.is_negative()) => {
                Some(base.as_ref())
            }
            Expr::Function(name, args)
                if matches!(name.as_str(), "log" | "ln") && args.len() == 1 =>
            {
                Some(&args[0])
            }
            _ => None,
        };
        if let Some(base) = nonzero_requirement {
            let at_root = base.subs(&replacements);
            match zero_status(&at_root) {
                Some(true) => return Ok(false),
                Some(false) => {}
                None => {
                    return Err(SolverError::IncompleteSolutionSet(
                        "a candidate's original denominator or logarithm argument is undecidable"
                            .into(),
                    ));
                }
            }
        }
        match term {
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => stack.extend(args),
            Expr::Pow(base, exponent) => {
                stack.push(base);
                stack.push(exponent);
            }
            _ => {}
        }
    }
    Ok(true)
}

/// Return only independently checked original-equation roots. A residual
/// that cannot be proved zero or nonzero makes the complete result
/// inconclusive; it is never accepted or silently discarded.
fn check_roots(e: &Expr, x: &Symbol, roots: Vec<Expr>) -> Result<Vec<Expr>, SolverError> {
    if !crate::verifier_inputs_within_bounds(std::iter::once(e).chain(roots.iter())) {
        return Err(SolverError::IncompleteSolutionSet(
            "original-equation verification exceeds its bounded input envelope".into(),
        ));
    }
    let mut checked = Vec::new();
    for root in roots {
        if contains_nonfinite(&root) || !candidate_in_domain(e, x, &root)? {
            continue;
        }
        let residual = e.subs(&HashMap::from([(x.clone(), root.clone())]));
        if contains_nonfinite(&residual) {
            continue;
        }
        match zero_status(&residual).or_else(|| square_root_residual(&residual)) {
            Some(true) => checked.push(root),
            Some(false) => {}
            None => {
                return Err(SolverError::IncompleteSolutionSet(
                    "a candidate's residual in the original equation cannot be verified exactly"
                        .into(),
                ));
            }
        }
    }
    Ok(checked)
}

/// `Some([c0, c1, c2])` when `e` is a polynomial of degree <= 2 in `x`
/// with `x`-free (possibly symbolic) coefficients.
fn symbolic_quadratic(e: &Expr, x: &Symbol) -> Option<[Expr; 3]> {
    let zero = HashMap::from([(x.clone(), Expr::from_i64(0))]);
    let d1 = fsym_calculus::diff(e, x);
    let d2 = fsym_calculus::diff(&d1, x);
    let c0 = expand(&e.subs(&zero));
    let c1 = expand(&d1.subs(&zero));
    let c2 = expand(&(d2.subs(&zero) * q(1, 2)));
    if [&c0, &c1, &c2].iter().any(|c| !is_free_of(c, x)) {
        return None;
    }
    let xe = Expr::Sym(x.clone());
    let rebuilt =
        c2.clone() * eval_pow(xe.clone(), Expr::from_i64(2)) + c1.clone() * xe + c0.clone();
    if !expand(&(e.clone() - rebuilt)).is_zero() {
        return None;
    }
    Some([c0, c1, c2])
}

fn solve_symbolic_quadratic(e: &Expr, x: &Symbol) -> Option<Vec<Expr>> {
    let [c0, c1, c2] = symbolic_quadratic(e, x)?;
    if c2.is_zero() {
        if c1.is_zero() {
            return None;
        }
        return Some(vec![expand(&(-c0 * eval_pow(c1, Expr::from_i64(-1))))]);
    }
    let disc = expand(&(c1.clone() * c1.clone() - Expr::from_i64(4) * c0 * c2.clone()));
    let sq = sqrt_with_content(disc);
    let inv = eval_pow(Expr::from_i64(2) * c2, Expr::from_i64(-1));
    let mb = -c1;
    Some(vec![
        expand(&((mb.clone() - sq.clone()) * inv.clone())),
        expand(&((mb + sq) * inv)),
    ])
}

/// `sqrt(d)` with the positive rational content of a sum pulled out:
/// `sqrt(4 - 4*y**2) = 2*sqrt(1 - y**2)`.
fn sqrt_with_content(d: Expr) -> Expr {
    let Expr::Add(terms) = &d else {
        return eval_pow(d, q(1, 2));
    };
    let mut num = BigInt::zero();
    let mut den = BigInt::one();
    for t in terms {
        let c = number(t).unwrap_or_else(|| split_coeff(t).0);
        num = gcd_int(&num, c.numer());
        den = lcm_int(&den, c.denom());
    }
    if num.is_zero() {
        return eval_pow(d, q(1, 2));
    }
    let g = BigRational::new(num.abs(), den);
    if g.is_one() {
        return eval_pow(d, q(1, 2));
    }
    let rest = expand(&(d * rq(BigRational::one() / g.clone())));
    eval_pow(rq(g), q(1, 2)) * eval_pow(rest, q(1, 2))
}

fn gcd_int(a: &BigInt, b: &BigInt) -> BigInt {
    let (mut a, mut b) = (a.abs(), b.abs());
    while !b.is_zero() {
        let r = a.clone() % b.clone();
        a = b;
        b = r;
    }
    a
}

fn lcm_int(a: &BigInt, b: &BigInt) -> BigInt {
    let g = gcd_int(a, b);
    if g.is_zero() {
        return BigInt::zero();
    }
    (a.clone() * b.clone()).abs() / g
}

/// `e = P + Q*sqrt(g)`: solve `P**2 - Q**2*g = 0` (candidates are checked
/// against the original equation by the caller).
fn eliminate_radical(e: &Expr, x: &Symbol, depth: usize) -> Option<Vec<Expr>> {
    if depth > 3 {
        return None;
    }
    let mut radicals = Vec::new();
    find_square_roots(e, x, &mut radicals);
    let r = radicals.into_iter().next()?;
    let Expr::Pow(g, _) = &r else { return None };
    let u = Symbol::new(format!("_fsym_rad{depth}"));
    let ue = Expr::Sym(u.clone());
    let with_u = e.subs_expr(&r, &ue);
    let p = expand(&with_u.subs(&HashMap::from([(u.clone(), Expr::from_i64(0))])));
    let qq = expand(&fsym_calculus::diff(&with_u, &u));
    if !is_free_of(&qq, &u) || !expand(&(with_u - p.clone() - qq.clone() * ue)).is_zero() {
        return None;
    }
    let squared = expand(&(p.clone() * p - qq.clone() * qq * (**g).clone()));
    if squared.is_zero() {
        return None;
    }
    if let Some(Ok(rs)) = solve_rational(&squared, x) {
        return Some(rs);
    }
    eliminate_radical(&squared, x, depth + 1)
}

fn find_square_roots(e: &Expr, x: &Symbol, out: &mut Vec<Expr>) {
    match e {
        Expr::Pow(b, p)
            if !is_free_of(b, x) && number(p).is_some_and(|v| v.denom() == &BigInt::from(2)) =>
        {
            let root = eval_pow((**b).clone(), q(1, 2));
            if !out.contains(&root) {
                out.push(root);
            }
        }
        Expr::Add(xs) | Expr::Mul(xs) | Expr::Function(_, xs) => {
            xs.iter().for_each(|t| find_square_roots(t, x, out))
        }
        Expr::Pow(b, p) => {
            find_square_roots(b, x, out);
            find_square_roots(p, x, out);
        }
        _ => {}
    }
}

/// Exact solutions of `e = 0` in `x`, or a typed refusal.
pub fn solve_univariate(e: &Expr, x: &Symbol) -> Result<Vec<Expr>, SolverError> {
    if let Some(r) = solve_rational(e, x) {
        return r;
    }
    if let Some(rs) = solve_symbolic_quadratic(e, x) {
        return Ok(dedup(rs));
    }
    let candidates = match invert(e, Expr::from_i64(0), x, 0) {
        Ok(c) => c,
        Err(_) => match solve_in_generator(e, x) {
            Ok(c) => c,
            Err(err) => eliminate_radical(e, x, 0).ok_or(err)?,
        },
    };
    let checked = check_roots(e, x, candidates.into_iter().map(|c| simplify(&c)).collect())?;
    Ok(sort_roots(dedup(checked)))
}

#[cfg(test)]
mod original_equation_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn enclosure_containing_zero_does_not_accept_a_nonzero_residual() {
        let x = Symbol::new("x");
        // Addition evaluates its children with four guard digits. Use that
        // enclosure's rational midpoint, which cannot equal irrational e,
        // to force a genuinely undecided coarse residual enclosure.
        let approximation = Expr::Const(Constant::E)
            .eval_complex_ball(28)
            .unwrap()
            .real()
            .midpoint()
            .clone();
        let equation = Expr::Sym(x.clone()) + Expr::Const(Constant::E) - rq(approximation);
        let residual = equation.subs(&HashMap::from([(x.clone(), Expr::from_i64(0))]));
        assert!(residual.eval_complex_ball(24).unwrap().contains_zero());
        assert!(!residual.eval_complex_ball(96).unwrap().contains_zero());
        assert_eq!(
            check_roots(&equation, &x, vec![Expr::from_i64(0)]),
            Ok(Vec::new())
        );
    }

    #[test]
    fn unevaluable_residual_is_incomplete_not_a_root_or_an_empty_set() {
        let x = Symbol::new("x");
        let equation = Expr::Function("uninterpreted".into(), vec![Expr::Sym(x.clone())]);
        assert!(matches!(
            check_roots(&equation, &x, vec![Expr::from_i64(0)]),
            Err(SolverError::IncompleteSolutionSet(_))
        ));
        // Finding one valid candidate does not license silently omitting
        // another candidate whose original-equation check is unresolved.
        let product = Expr::Sym(x.clone()) * equation;
        assert!(matches!(
            check_roots(&product, &x, vec![Expr::from_i64(0), Expr::from_i64(1)]),
            Err(SolverError::IncompleteSolutionSet(_))
        ));
    }

    #[test]
    fn symbolic_principal_branch_obligation_survives_public_dispatch() {
        let x = Symbol::new("x");
        let y = Expr::symbol("y");
        let equation = eval_pow(Expr::Sym(x.clone()), q(1, 2)) + y;
        // Squaring proposes x = y^2, but sqrt(y^2) = -y is conditional.
        let direct = solve_univariate(&equation, &x);
        assert!(matches!(direct, Err(SolverError::IncompleteSolutionSet(_))));
        assert_eq!(crate::solve(&equation, &x), direct);
    }

    #[test]
    fn radical_elimination_preserves_exact_rational_and_irrational_roots() {
        let x = Symbol::new("x");
        let xe = Expr::Sym(x.clone());
        let rational =
            eval_pow(xe.clone() + Expr::from_i64(3), q(1, 2)) - xe.clone() + Expr::from_i64(3);
        assert_eq!(crate::solve(&rational, &x), Ok(vec![Expr::from_i64(6)]));

        let irrational = eval_pow(xe.clone(), q(1, 2)) + xe - Expr::from_i64(1);
        let expected = expand(&(q(3, 2) - q(1, 2) * eval_pow(Expr::from_i64(5), q(1, 2))));
        assert_eq!(crate::solve(&irrational, &x), Ok(vec![expected]));
        // The squared identity also holds at this other algebraic root;
        // the independent principal-branch sign obligation must reject it.
        let extraneous = expand(&(q(3, 2) + q(1, 2) * eval_pow(Expr::from_i64(5), q(1, 2))));
        let wrong_branch = irrational.subs(&HashMap::from([(x, extraneous)]));
        assert_eq!(square_root_residual(&wrong_branch), Some(false));
    }

    #[test]
    fn original_rational_poles_survive_cancellation_and_multiplicity() {
        let x = Symbol::new("x");
        let xe = Expr::Sym(x.clone());
        for pole_factor in [
            xe.clone() - Expr::from_i64(1),
            eval_pow(xe.clone(), Expr::from_i64(2)) - Expr::from_i64(2),
        ] {
            // Preserve the original fraction's domain instead of letting
            // eager multiplication simplify away its excluded points.
            let equation = Expr::Mul(vec![
                Expr::Pow(Arc::new(pole_factor.clone()), Arc::new(Expr::from_i64(3))),
                xe.clone() - Expr::from_i64(3),
                Expr::Pow(Arc::new(pole_factor), Arc::new(Expr::from_i64(-1))),
            ]);
            assert_eq!(crate::solve(&equation, &x), Ok(vec![Expr::from_i64(3)]));
        }
    }

    #[test]
    fn original_domain_and_finite_solution_checks_precede_simplification() {
        let x = Symbol::new("x");
        let factor = Expr::Sym(x.clone()) - Expr::from_i64(1);
        let equation = Expr::Mul(vec![
            factor.clone(),
            Expr::Pow(Arc::new(factor), Arc::new(Expr::from_i64(-1))),
        ]);
        assert_eq!(
            check_roots(&equation, &x, vec![Expr::from_i64(1)]),
            Ok(vec![])
        );
        // Infinity is a limiting point, not a finite solution of exp(x)=0.
        let exponential = Expr::Function("exp".into(), vec![Expr::Sym(x.clone())]);
        assert_eq!(crate::solve(&exponential, &x), Ok(vec![]));
    }

    #[test]
    fn nested_reciprocal_holes_are_not_forgotten_by_rational_rewriting() {
        let x = Symbol::new("x");
        let pole = Expr::Sym(x.clone()) - Expr::from_i64(1);
        let inner = Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(-1)));
        let reciprocal = Expr::Pow(Arc::new(inner), Arc::new(Expr::from_i64(-1)));
        assert_eq!(crate::solve(&reciprocal, &x), Ok(vec![]));
        let undefined_inner = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
        let undefined = Expr::Pow(Arc::new(undefined_inner), Arc::new(Expr::from_i64(-1)));
        assert!(matches!(
            crate::solve(&undefined, &x),
            Err(SolverError::InvalidSystem(_))
        ));
    }

    #[test]
    fn an_identically_zero_denominator_is_not_retried_as_a_polynomial() {
        let x = Symbol::new("x");
        let equation = Expr::Mul(vec![
            Expr::Sym(x.clone()),
            Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
        ]);
        let direct = solve_univariate(&equation, &x);
        assert!(matches!(direct, Err(SolverError::InvalidSystem(_))));
        assert_eq!(crate::solve(&equation, &x), direct);
    }
}
