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

/// Upstream root order: reals ascending, then complex by (re, im).
pub fn sort_roots(mut roots: Vec<Expr>) -> Vec<Expr> {
    if roots.iter().any(|r| approx(r).is_none()) {
        // Symbolic roots keep their construction order (upstream lists the
        // minus branch of a symbolic radical first).
        return roots;
    }
    let key = |e: &Expr| {
        let (re, im) = approx(e).unwrap_or((f64::NAN, 0.0));
        let is_real = im.abs() < 1e-12;
        (if is_real { 0 } else { 1 }, re, im)
    };
    roots.sort_by(|a, b| {
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
            *o == r
                || match (approx(o), approx(&r)) {
                    (Some(a), Some(b)) => {
                        (a.0 - b.0).abs() < 1e-12
                            && (a.1 - b.1).abs() < 1e-12
                            && simplify(&expand(&(o.clone() - r.clone()))).is_zero()
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
    let (num, den) = rational_parts(e, x)?;
    let g = num.gcd(&den).ok()?;
    let (num, _) = num.div_rem(&g).ok()?;
    let (den, _) = den.div_rem(&g).ok()?;
    if num.is_zero() {
        return Some(Err(SolverError::InfiniteSolutions));
    }
    let roots = match poly_roots(&num) {
        Ok(r) => r,
        Err(err) => return Some(Err(err)),
    };
    let mut out = Vec::new();
    for (r, _) in roots {
        if let Some(v) = number(&r)
            && den.eval(&v).is_zero()
        {
            continue;
        }
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
                    for k in 0..n {
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
                Some(v) if v.is_negative() => {
                    invert(b, eval_pow(rhs, rq(BigRational::one() / v)), x, depth + 1)
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
            if !out.iter().any(|o| *o == base) {
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

/// Drops candidates that do not satisfy `e = 0` (exactly, or numerically
/// with a wide margin when exact simplification is inconclusive).
fn check_roots(e: &Expr, x: &Symbol, roots: Vec<Expr>) -> Vec<Expr> {
    roots
        .into_iter()
        .filter(|r| {
            let v = e.subs(&HashMap::from([(x.clone(), r.clone())]));
            let s = simplify(&v);
            if s.is_zero() {
                return true;
            }
            match approx(&s) {
                Some((re, im)) => re.abs() < 1e-9 && im.abs() < 1e-9,
                None => true,
            }
        })
        .collect()
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
    let sq = eval_pow(disc, q(1, 2));
    let inv = eval_pow(Expr::from_i64(2) * c2, Expr::from_i64(-1));
    let mb = -c1;
    Some(vec![
        expand(&((mb.clone() - sq.clone()) * inv.clone())),
        expand(&((mb + sq) * inv)),
    ])
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
    let checked = check_roots(e, x, candidates.into_iter().map(|c| simplify(&c)).collect());
    Ok(sort_roots(dedup(checked)))
}
