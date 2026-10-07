//! Indefinite and definite integration.
//!
//! A constructive integrator over a fixed rule set; every rule is an exact
//! antiderivative identity (no numeric fitting):
//!
//! * linearity and constant extraction;
//! * a table of elementary antiderivatives with linear inner arguments
//!   (`exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `a**x`, `log`, `atan`,
//!   `asin`, `erf` for Gaussians, arcsine/arcsinh radicals, power rule);
//! * rational functions: polynomial part, exact partial fractions over the
//!   complete factorization of the denominator over Q, `log` terms for
//!   linear factors, `log` + `atan` terms (with the Hermite-style reduction
//!   for powers) for irreducible quadratics;
//! * `sin**m * cos**n` by odd-power substitution and the reduction formula;
//! * `exp(a*x+b) * sin/cos(c*x+d)` closed forms;
//! * integration by parts for polynomial times exp/trig/hyperbolic and for
//!   log/atan/asin factors;
//! * derivative-matching substitution `f(g(x)) * g'(x)`;
//! * expansion of products and integer powers as a last resort.
//!
//! Definite integrals evaluate the antiderivative at the endpoints through
//! [`crate::limit_dir`] (so infinite endpoints and integrable endpoint
//! singularities work) after refusing intervals that contain an interior
//! pole of the integrand.

#![forbid(unsafe_code)]

use crate::gruntz::Direction;
use fsym_core::elementary::{eval_function, eval_pow, rational_expr};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use num_traits::{One, Signed, Zero};
use std::collections::HashMap;

const MAX_DEPTH: usize = 12;

fn func(name: &str, arg: Expr) -> Expr {
    eval_function(name, std::slice::from_ref(&arg))
        .unwrap_or_else(|| Expr::Function(name.to_string(), vec![arg]))
}

fn pow(b: Expr, e: Expr) -> Expr {
    eval_pow(b, e)
}

fn recip(e: Expr) -> Expr {
    pow(e, Expr::from_i64(-1))
}

fn q(n: i64, d: i64) -> Expr {
    rational_expr(BigRational::new(BigInt::from(n), BigInt::from(d)))
}

fn rq(r: BigRational) -> Expr {
    rational_expr(r)
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

fn product(factors: impl IntoIterator<Item = Expr>) -> Expr {
    factors.into_iter().fold(Expr::from_i64(1), |a, b| a * b)
}

fn sum(terms: impl IntoIterator<Item = Expr>) -> Expr {
    terms.into_iter().fold(Expr::from_i64(0), |a, b| a + b)
}

/// `Some((a, b))` when `e == a*x + b` with `a, b` free of `x`, `a != 0`.
fn linear_coeffs(e: &Expr, x: &Symbol) -> Option<(Expr, Expr)> {
    let xe = Expr::Sym(x.clone());
    let zero = HashMap::from([(x.clone(), Expr::from_i64(0))]);
    let b = e.subs(&zero);
    let a = fsym_simplify::expand(&crate::diff(e, x));
    if a.is_zero() || !is_free_of(&a, x) || !is_free_of(&b, x) {
        return None;
    }
    // Dividing by a symbolic slope would be wrong where it vanishes
    // (upstream answers with a Piecewise); only provably nonzero slopes.
    if !provably_nonzero(&a) {
        return None;
    }
    let check = fsym_simplify::expand(&(e.clone() - (a.clone() * xe + b.clone())));
    if !check.is_zero() {
        return None;
    }
    Some((a, b))
}

/// A symbol-free constant whose value is well separated from zero.
fn provably_nonzero(e: &Expr) -> bool {
    if let Some(v) = number(e) {
        return !v.is_zero();
    }
    if fsym_core::assume::is_nonzero(e) {
        return true;
    }
    e.free_symbols().is_empty() && crate::gruntz::complex_value(e).is_some_and(|v| v.norm() > 1e-9)
}

/// Splits a product into (x-free factor, x-dependent factors).
fn split_constant(e: &Expr, x: &Symbol) -> (Expr, Vec<Expr>) {
    match e {
        Expr::Mul(factors) => {
            let mut c = Expr::from_i64(1);
            let mut rest = Vec::new();
            for f in factors {
                if is_free_of(f, x) {
                    c = c * f.clone();
                } else {
                    rest.push(f.clone());
                }
            }
            (c, rest)
        }
        other if is_free_of(other, x) => (other.clone(), Vec::new()),
        other => (Expr::from_i64(1), vec![other.clone()]),
    }
}

struct Integrator {
    x: Symbol,
    depth: usize,
}

impl Integrator {
    fn xe(&self) -> Expr {
        Expr::Sym(self.x.clone())
    }

    fn int(&mut self, f: &Expr) -> Option<Expr> {
        if self.depth > MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.int_inner(f);
        self.depth -= 1;
        r
    }

    fn int_inner(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        if is_free_of(f, &x) {
            return Some(f.clone() * self.xe());
        }
        if let Expr::Add(terms) = f {
            let mut parts = Vec::with_capacity(terms.len());
            for t in terms {
                parts.push(self.int(t)?);
            }
            return Some(sum(parts));
        }
        let (c, rest) = split_constant(f, &x);
        if !c.is_one() {
            let inner = product(rest);
            return self.int(&inner).map(|r| c * r);
        }
        if let Some(r) = self.table(f) {
            return Some(r);
        }
        if let Some(r) = self.special_quotient(f) {
            return Some(r);
        }
        // exp(a + g(x)) = exp(a)*exp(g(x)) with a free of x, for integrands
        // the table does not take whole (Gaussians in several variables).
        if let Some((free, dependent)) = split_exp_constant(f, &x) {
            return self.int(&dependent).map(|r| free * r);
        }
        if let Some(r) = self.quadratic_denominator(f) {
            return Some(r);
        }
        if let Some(r) = self.rational(f) {
            return Some(r);
        }
        if let Some(r) = self.biquadratic(f) {
            return Some(r);
        }
        if let Some(r) = self.trig_powers(f) {
            return Some(r);
        }
        if let Some(r) = self.exp_trig(f) {
            return Some(r);
        }
        if let Some(r) = self.substitution(f) {
            return Some(r);
        }
        if let Some(r) = self.parts(f) {
            return Some(r);
        }
        let expanded = fsym_simplify::expand(f);
        if &expanded != f && matches!(expanded, Expr::Add(_)) {
            return self.int(&expanded);
        }
        None
    }

    // -----------------------------------------------------------------------
    // Table
    // -----------------------------------------------------------------------

    fn table(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        let xe = self.xe();
        match f {
            Expr::Sym(_) => Some(q(1, 2) * pow(xe.clone(), Expr::from_i64(2))),
            Expr::Pow(b, e) => {
                let (b, e) = ((**b).clone(), (**e).clone());
                if e == Expr::from_i64(-2)
                    && let Expr::Function(fname, fargs) = &b
                    && fargs.len() == 1
                    && let Some((a, _)) = linear_coeffs(&fargs[0], &x)
                {
                    let u = fargs[0].clone();
                    let (s, c) = (func("sin", u.clone()), func("cos", u));
                    return match fname.as_str() {
                        "cos" => Some(recip(a) * s * recip(c)),
                        "sin" => Some(-recip(a) * c * recip(s)),
                        _ => None,
                    };
                }
                // 1/(1 + cos(u)) = tan(u/2)', 1/(1 - cos(u)) = -cot(u/2)'.
                if e == Expr::from_i64(-1)
                    && let Expr::Add(ts) = &b
                    && ts.len() == 2
                {
                    for (one, other) in [(&ts[0], &ts[1]), (&ts[1], &ts[0])] {
                        if !one.is_one() {
                            continue;
                        }
                        let (sign, cosine) = match other {
                            Expr::Function(n, a) if n == "cos" && a.len() == 1 => (1, &a[0]),
                            Expr::Mul(fs)
                                if fs.len() == 2
                                    && fs[0] == Expr::from_i64(-1)
                                    && matches!(&fs[1], Expr::Function(n, a) if n == "cos" && a.len() == 1) =>
                            {
                                let Expr::Function(_, a) = &fs[1] else {
                                    unreachable!()
                                };
                                (-1, &a[0])
                            }
                            _ => continue,
                        };
                        let Some((a, _)) = linear_coeffs(cosine, &x) else {
                            continue;
                        };
                        let half = q(1, 2) * cosine.clone();
                        return Some(if sign == 1 {
                            recip(a) * func("tan", half)
                        } else {
                            -recip(a) * func("cot", half)
                        });
                    }
                }
                if e == Expr::from_i64(2)
                    && let Expr::Function(fname, fargs) = &b
                    && fargs.len() == 1
                    && let Some((a, _)) = linear_coeffs(&fargs[0], &x)
                    && matches!(fname.as_str(), "cosh" | "sinh")
                {
                    let u = fargs[0].clone();
                    let sc = func("sinh", u.clone()) * func("cosh", u.clone());
                    return Some(if fname == "cosh" {
                        recip(a) * (q(1, 2) * u + q(1, 2) * sc)
                    } else {
                        recip(a) * (q(1, 2) * sc - q(1, 2) * u)
                    });
                }
                if e == Expr::from_i64(2)
                    && let Expr::Function(fname, fargs) = &b
                    && fargs.len() == 1
                    && let Some((a, _)) = linear_coeffs(&fargs[0], &x)
                {
                    let u = fargs[0].clone();
                    let (s, c) = (func("sin", u.clone()), func("cos", u.clone()));
                    match fname.as_str() {
                        "sec" => return Some(recip(a) * func("tan", u)),
                        "csc" => return Some(-recip(a) * func("cot", u)),
                        // tan**2 = sec**2 - 1, cot**2 = csc**2 - 1 (upstream forms).
                        "tan" => return Some(recip(a) * s * recip(c) - xe.clone()),
                        "cot" => return Some(-recip(a) * c * recip(s) - xe.clone()),
                        _ => {}
                    }
                }
                if is_free_of(&e, &x) {
                    let positive_integer =
                        number(&e).is_some_and(|v| v.is_integer() && v.is_positive());
                    if positive_integer && is_polynomial_in(&b, &x) {
                        // Upstream integrates expanded polynomials.
                        return None;
                    }
                    if let Some((a, _)) = linear_coeffs(&b, &x) {
                        // (a*x+b)**n
                        if e == Expr::from_i64(-1) {
                            return Some(recip(a) * func("log", b));
                        }
                        let np1 = e.clone() + Expr::from_i64(1);
                        return Some(pow(b, np1.clone()) * recip(a * np1));
                    }
                    if let Some(r) = self.radical_quadratic(&b, &e) {
                        return Some(r);
                    }
                    return None;
                }
                if is_free_of(&b, &x)
                    && let Some((a, _)) = linear_coeffs(&e, &x)
                {
                    // c**(a*x+b) = c**(...) / (a*log(c))
                    return Some(pow(b.clone(), e.clone()) * recip(a * func("log", b)));
                }
                None
            }
            Expr::Function(name, args) if args.len() == 1 => {
                let u = &args[0];
                if name == "exp"
                    && let Some(r) = self.gaussian(u)
                {
                    return Some(r);
                }
                let (a, _) = linear_coeffs(u, &x)?;
                let ia = recip(a);
                let u = u.clone();
                let r = match name.as_str() {
                    "exp" => func("exp", u),
                    "sin" => -func("cos", u),
                    "cos" => func("sin", u),
                    "tan" => -func("log", func("cos", u)),
                    "cot" => func("log", func("sin", u)),
                    "sinh" => func("cosh", u),
                    "cosh" => func("sinh", u),
                    "tanh" => func("log", func("cosh", u)),
                    "sec" => {
                        let su = func("sin", u);
                        q(1, 2)
                            * (func("log", su.clone() + Expr::from_i64(1))
                                - func("log", su - Expr::from_i64(1)))
                    }
                    "csc" => {
                        let cu = func("cos", u);
                        q(1, 2)
                            * (func("log", cu.clone() - Expr::from_i64(1))
                                - func("log", cu + Expr::from_i64(1)))
                    }
                    "log" => u.clone() * func("log", u.clone()) - u,
                    "atan" => {
                        u.clone() * func("atan", u.clone())
                            - q(1, 2) * func("log", pow(u, Expr::from_i64(2)) + Expr::from_i64(1))
                    }
                    "asin" => {
                        u.clone() * func("asin", u.clone())
                            + pow(Expr::from_i64(1) - pow(u, Expr::from_i64(2)), q(1, 2))
                    }
                    "acos" => {
                        u.clone() * func("acos", u.clone())
                            - pow(Expr::from_i64(1) - pow(u, Expr::from_i64(2)), q(1, 2))
                    }
                    "asinh" => {
                        u.clone() * func("asinh", u.clone())
                            - pow(pow(u, Expr::from_i64(2)) + Expr::from_i64(1), q(1, 2))
                    }
                    "erf" => {
                        u.clone() * func("erf", u.clone())
                            + func("exp", -pow(u, Expr::from_i64(2)))
                                * recip(pow(Expr::Const(Constant::Pi), q(1, 2)))
                    }
                    _ => return None,
                };
                Some(ia * r)
            }
            _ => None,
        }
    }

    /// `N(x)/(x**4 + p*x**2 + q)` with rational `p, q`, `p**2 < 4q` and
    /// `deg N <= 3`: the denominator splits over the reals as
    /// `(x**2 - s*x + r)(x**2 + s*x + r)`, `r = sqrt(q)`,
    /// `s = sqrt(2r - p)`; partial fractions solve in closed form and each
    /// quadratic term integrates to log + atan (upstream's form for
    /// `1/(x**4 + 1)`).
    fn biquadratic(&self, f: &Expr) -> Option<Expr> {
        let x = &self.x;
        let (num, den) = rational_parts(f, x)?;
        if den.degree()? != 4 || num.degree().unwrap_or(0) > 3 {
            return None;
        }
        let lc = den.coeffs.last()?.clone();
        let c = |i: usize, p: &UnivariatePoly| {
            p.coeffs.get(i).cloned().unwrap_or_else(BigRational::zero) / lc.clone()
        };
        if !c(1, &den).is_zero() || !c(3, &den).is_zero() {
            return None;
        }
        let (pv, qv) = (c(2, &den), c(0, &den));
        if !qv.is_positive()
            || pv.clone() * pv.clone() >= BigRational::from_integer(4.into()) * qv.clone()
        {
            return None;
        }
        let ncoef = |i: usize| rq(c(i, &num));
        let (n0, n1, n2, n3) = (ncoef(0), ncoef(1), ncoef(2), ncoef(3));
        let r = pow(rq(qv), q(1, 2));
        let s = pow(Expr::from_i64(2) * r.clone() - rq(pv), q(1, 2));
        let e = |v: Expr| fsym_simplify::expand(&v);
        let h = q(1, 2);
        let t1 = e((n2 - n0.clone() * recip(r.clone())) * recip(s.clone()));
        let t2 = e((n1 - r.clone() * n3.clone()) * recip(s.clone()));
        let base = e(n0 * recip(r.clone()));
        let alpha = e(h.clone() * (n3.clone() + t1.clone()));
        let gamma = e(h.clone() * (n3 - t1));
        let beta = e(h.clone() * (base.clone() + t2.clone()));
        let delta = e(h.clone() * (base - t2));
        let xe = self.xe();
        // int (P*x + Q)/(x**2 + b*x + c) for 4c - b**2 > 0.
        let term = |pp: Expr, qq: Expr, b: Expr, cc: Expr| -> Expr {
            let quad = e(pow(xe.clone(), Expr::from_i64(2)) + b.clone() * xe.clone() + cc.clone());
            let disc = e(Expr::from_i64(4) * cc - b.clone() * b.clone());
            let sd = pow(disc, q(1, 2));
            let log_part = h.clone() * pp.clone() * func("log", quad);
            let coeff = e(qq - h.clone() * pp * b.clone());
            let atan_arg = e((Expr::from_i64(2) * xe.clone() + b) * recip(sd.clone()));
            log_part + e(Expr::from_i64(2) * coeff * recip(sd)) * func("atan", atan_arg)
        };
        let ms = e(Expr::from_i64(-1) * s.clone());
        Some(term(alpha, beta, ms, r.clone()) + term(gamma, delta, s, r))
    }

    /// `T(a*x)/x` for the non-elementary integrals: sin -> Si, cos -> Ci,
    /// exp -> Ei, sinh -> Shi, cosh -> Chi.
    fn special_quotient(&self, f: &Expr) -> Option<Expr> {
        let Expr::Mul(fs) = f else {
            return None;
        };
        if fs.len() != 2 {
            return None;
        }
        let xe = self.xe();
        let inv_x = pow(xe.clone(), Expr::from_i64(-1));
        let other = if fs[0] == inv_x {
            &fs[1]
        } else if fs[1] == inv_x {
            &fs[0]
        } else {
            return None;
        };
        let Expr::Function(name, args) = other else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        let (_, b) = linear_coeffs(&args[0], &self.x)?;
        if !b.is_zero() {
            return None;
        }
        let target = match name.as_str() {
            "sin" => "Si",
            "cos" => "Ci",
            "exp" => "Ei",
            "sinh" => "Shi",
            "cosh" => "Chi",
            _ => return None,
        };
        Some(func(target, args[0].clone()))
    }

    /// `(p*x + q)/(A*x**2 + B*x + C)` with x-free (possibly symbolic)
    /// coefficients whose discriminant sign is decided (exactly or by the
    /// active facts): log + atan for `4AC - B^2 > 0`, log + log otherwise.
    fn quadratic_denominator(&self, f: &Expr) -> Option<Expr> {
        let x = &self.x;
        let factors: Vec<Expr> = match f {
            Expr::Mul(xs) => xs.clone(),
            other => vec![other.clone()],
        };
        let mut den = None;
        let mut num = Expr::from_i64(1);
        for g in factors {
            match &g {
                Expr::Pow(b, e) if **e == Expr::from_i64(-1) && den.is_none() => {
                    den = Some((**b).clone());
                }
                _ => num = num * g,
            }
        }
        let den = den?;
        if is_free_of(&den, x) || den.free_symbols().iter().all(|s| s == x) {
            // Pure-rational cases belong to the exact partial-fraction lane.
            return None;
        }
        let coeff = |e: &Expr, k: i64| -> Option<Expr> {
            let d = (0..k).fold(e.clone(), |acc, _| crate::diff(&acc, x));
            let v = d.subs(&HashMap::from([(x.clone(), Expr::from_i64(0))]));
            let fact: i64 = (1..=k).product();
            let c = fsym_simplify::expand(&(v * q(1, fact)));
            is_free_of(&c, x).then_some(c)
        };
        let (a2, b1, c0) = (coeff(&den, 2)?, coeff(&den, 1)?, coeff(&den, 0)?);
        if !fsym_simplify::expand(&crate::diff(&crate::diff(&crate::diff(&den, x), x), x)).is_zero()
            || a2.is_zero()
        {
            return None;
        }
        let (p1, p0) = (coeff(&num, 1)?, coeff(&num, 0)?);
        if !fsym_simplify::expand(&crate::diff(&crate::diff(&num, x), x)).is_zero() {
            return None;
        }
        let disc = fsym_simplify::expand(
            &(Expr::from_i64(4) * a2.clone() * c0.clone() - b1.clone() * b1.clone()),
        );
        let sign = match number(&disc) {
            Some(v) => i8::from(v.is_positive()) - i8::from(v.is_negative()),
            None => fsym_core::assume::sign(&disc)?,
        };
        if sign <= 0 || !provably_nonzero(&a2) {
            return None;
        }
        let xe = self.xe();
        let two_a = Expr::from_i64(2) * a2.clone();
        let log_part = p1.clone() * recip(two_a.clone()) * func("log", den.clone());
        let rest = p0 - p1 * b1.clone() * recip(two_a.clone());
        let sd = fsym_simplify::simplify(&pow(disc, q(1, 2)));
        let atan_part = Expr::from_i64(2)
            * rest
            * recip(sd.clone())
            * func(
                "atan",
                fsym_simplify::simplify(&((two_a * xe + b1) * recip(sd))),
            );
        Some(log_part + atan_part)
    }

    /// exp(-c*x**2) for a positive rational c: sqrt(pi)*erf(sqrt(c)*x)/(2*sqrt(c)).
    fn gaussian(&self, u: &Expr) -> Option<Expr> {
        let x = &self.x;
        let x2 = pow(self.xe(), Expr::from_i64(2));
        let c = fsym_simplify::expand(&(-u.clone() * recip(x2)));
        if !is_free_of(&c, x) {
            return None;
        }
        let positive = match number(&c) {
            Some(cv) => cv.is_positive(),
            None => fsym_core::assume::sign(&c) == Some(1),
        };
        if !positive {
            return None;
        }
        let sc = pow(c, q(1, 2));
        Some(
            pow(Expr::Const(Constant::Pi), q(1, 2))
                * func("erf", sc.clone() * self.xe())
                * recip(Expr::from_i64(2) * sc),
        )
    }

    /// (k - c*x**2)**(-1/2) -> asin, (c*x**2 + k)**(-1/2) -> asinh,
    /// (k - c*x**2)**(1/2) -> x*sqrt(...)/2 + k*asin(...)/(2*sqrt(c)).
    fn radical_quadratic(&self, b: &Expr, e: &Expr) -> Option<Expr> {
        let x = &self.x;
        let ev = number(e)?;
        let zero = HashMap::from([(x.clone(), Expr::from_i64(0))]);
        let k = b.subs(&zero);
        let c2 = fsym_simplify::expand(
            &((b.clone() - k.clone()) * recip(pow(self.xe(), Expr::from_i64(2)))),
        );
        if !is_free_of(&c2, x) {
            return None;
        }
        let (kv, cv) = (number(&k)?, number(&c2)?);
        if cv.is_zero() || kv.is_zero() {
            return None;
        }
        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        if cv.is_positive() {
            let sc = pow(rq(cv.clone()), q(1, 2));
            let root = pow(b.clone(), q(1, 2));
            // log(sqrt(c)*x + sqrt(c*x**2 + k))/sqrt(c) for k < 0.
            let log_form =
                || func("log", sc.clone() * self.xe() + root.clone()) * recip(sc.clone());
            if kv.is_negative() {
                if ev == -half.clone() {
                    return Some(log_form());
                }
                if ev == half {
                    return Some(
                        q(1, 2) * self.xe() * root.clone() + rq(kv.clone()) * q(1, 2) * log_form(),
                    );
                }
                return None;
            }
            if ev == half {
                let s = pow(rq(cv.clone() / kv.clone()), q(1, 2));
                return Some(
                    q(1, 2) * self.xe() * root
                        + q(1, 2) * rq(kv) * func("asinh", s * self.xe()) * recip(sc),
                );
            }
        }
        if !kv.is_positive() {
            return None;
        }
        let s = pow(rq(cv.abs() / kv.clone()), q(1, 2));
        let sc = pow(rq(cv.abs()), q(1, 2));
        let arg = s * self.xe();
        if ev == BigRational::new(BigInt::from(-1), BigInt::from(2)) {
            return Some(if cv.is_negative() {
                func("asin", arg) * recip(sc)
            } else {
                func("asinh", arg) * recip(sc)
            });
        }
        if ev == BigRational::new(BigInt::from(1), BigInt::from(2)) && cv.is_negative() {
            return Some(
                q(1, 2) * self.xe() * pow(b.clone(), q(1, 2))
                    + q(1, 2) * rq(kv) * func("asin", arg) * recip(sc),
            );
        }
        None
    }

    // -----------------------------------------------------------------------
    // Rational functions
    // -----------------------------------------------------------------------

    /// `x**(k-1) * R(x**k)` with `k >= 2` integrates as
    /// `(1/k) * int R(u) du` at `u = x**k` (upstream's results for
    /// x/(x**4 + 1) -> atan(x**2)/2, x**3/(x**8 + 1) -> atan(x**4)/4).
    fn rational_power_substitution(
        &mut self,
        num: &UnivariatePoly,
        den: &UnivariatePoly,
    ) -> Option<Expr> {
        let exps = |p: &UnivariatePoly, shift: usize| -> Vec<usize> {
            p.coeffs
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.is_zero())
                .map(|(i, _)| i + shift)
                .collect()
        };
        let mut g = 0usize;
        for e in exps(den, 0).into_iter().chain(exps(num, 1)) {
            g = gcd_usize(g, e);
        }
        if g < 2 || den.degree()? == 0 {
            return None;
        }
        let u = Symbol::new("_fsym_kronecker_u");
        let shrink = |p: &UnivariatePoly, shift: usize| -> UnivariatePoly {
            let deg = (p.coeffs.len() - 1 + shift) / g;
            let mut coeffs = vec![BigRational::zero(); deg + 1];
            for (i, c) in p.coeffs.iter().enumerate() {
                if !c.is_zero() {
                    coeffs[(i + shift) / g - usize::from(shift > 0)] += c.clone();
                }
            }
            UnivariatePoly::new(u.clone(), coeffs)
        };
        let reduced = shrink(num, 1).to_expr() * recip(shrink(den, 0).to_expr());
        let mut sub = Integrator {
            x: u.clone(),
            depth: self.depth + 1,
        };
        let anti = sub.int(&reduced)?;
        let xk = pow(self.xe(), Expr::from_i64(i64::try_from(g).ok()?));
        let back = anti.subs(&HashMap::from([(u, xk)]));
        Some(rq(BigRational::new(BigInt::one(), BigInt::from(g as u64))) * back)
    }

    fn rational(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        let (num, den) = rational_parts(f, &x)?;
        if let Some(r) = self.rational_power_substitution(&num, &den) {
            return Some(r);
        }
        if den.degree()? == 0 {
            let p = num
                .mul(&UnivariatePoly::new(x.clone(), vec![den.coeffs[0].recip()]))
                .ok()?;
            let anti = p.integrate(BigRational::zero()).ok()?;
            return Some(anti.to_expr());
        }
        let g = num.gcd(&den).ok()?;
        let (num, _) = num.div_rem(&g).ok()?;
        let (den, _) = den.div_rem(&g).ok()?;
        let (quot, rem) = num.div_rem(&den).ok()?;
        let mut out = quot.integrate(BigRational::zero()).ok()?.to_expr();
        if rem.is_zero() {
            return Some(out);
        }
        let fact = fsym_polys::factorization::complete_factorization(&den).ok()?;
        // den = scale * prod(p_i ** k_i), p_i monic irreducible.
        let scale = fact.scale.clone();
        let pieces: Vec<(UnivariatePoly, usize)> = fact
            .factors
            .iter()
            .map(|t| (t.poly.clone(), t.multiplicity))
            .collect();
        if pieces.iter().any(|(p, _)| p.degree().unwrap_or(0) > 2) {
            return None;
        }
        let mut logs: Vec<(BigRational, UnivariatePoly)> = Vec::new();
        let rem = rem
            .mul(&UnivariatePoly::new(x.clone(), vec![scale.recip()]))
            .ok()?;
        let monic_den = den
            .mul(&UnivariatePoly::new(x.clone(), vec![scale.recip()]))
            .ok()?;
        for (p, k) in &pieces {
            let pk = p.pow(u32::try_from(*k).ok()?).ok()?;
            let (other, _) = monic_den.div_rem(&pk).ok()?;
            // A = rem * other^{-1} mod p^k
            let bez = other.extended_gcd(&pk).ok()?;
            let inv = scale_poly(&bez.u, &bez.gcd.leading_coeff().recip())?;
            let a = rem.mul(&inv).ok()?.div_rem(&pk).ok()?.1;
            // Expand A in base p: A = sum c_j p^j.
            let mut digits = Vec::new();
            let mut cur = a;
            for _ in 0..*k {
                let (qq, rr) = cur.div_rem(p).ok()?;
                digits.push(rr);
                cur = qq;
            }
            for (j, c) in digits.into_iter().enumerate() {
                let m = k - j; // c / p**m
                if c.is_zero() {
                    continue;
                }
                out = out + self.partial_term(&c, p, m, &mut logs)?;
            }
        }
        // Upstream ratint returns the Hermite rational part as ONE fraction
        // in cancel form (-1/(2*x**2 + 2), (-2*x - 1)/(x**2 + x)): merge the
        // function-free, non-polynomial terms of `out`.
        out = merge_rational_part(out, &x);
        // Upstream's log part groups factors sharing a log coefficient into
        // one logarithm: log(x + 1) + log(x + 2) -> log(x**2 + 3*x + 2).
        let mut grouped: Vec<(BigRational, UnivariatePoly)> = Vec::new();
        for (c, p) in logs {
            match grouped.iter_mut().find(|(gc, _)| *gc == c) {
                Some((_, gp)) => *gp = gp.mul(&p).ok()?,
                None => grouped.push((c, p)),
            }
        }
        for (c, p) in grouped {
            out = out + rq(c) * func("log", primitive_expr(&p));
        }
        Some(out)
    }

    /// Antiderivative of c(x)/p(x)**m for monic irreducible p of degree 1/2.
    fn partial_term(
        &self,
        c: &UnivariatePoly,
        p: &UnivariatePoly,
        m: usize,
        logs: &mut Vec<(BigRational, UnivariatePoly)>,
    ) -> Option<Expr> {
        let xe = self.xe();
        match p.degree()? {
            1 => {
                let b = c.coeffs.first().cloned().unwrap_or_else(BigRational::zero);
                let lin = p.to_expr();
                if m == 1 {
                    logs.push((b, p.clone()));
                    Some(Expr::from_i64(0))
                } else {
                    let mm = BigInt::from((m - 1) as i64);
                    Some(
                        rq(-b / BigRational::from_integer(mm.clone()))
                            * pow(lin, Expr::Integer(-mm)),
                    )
                }
            }
            2 => {
                // p = x^2 + s x + t, c = A x + C
                let s = p.coeffs[1].clone();
                let t = p.coeffs[0].clone();
                let a = c.coeffs.get(1).cloned().unwrap_or_else(BigRational::zero);
                let cc = c.coeffs.first().cloned().unwrap_or_else(BigRational::zero);
                let delta = BigRational::from_integer(BigInt::from(4)) * t - s.clone() * s.clone();
                let pe = p.to_expr();
                let two_x_s = Expr::from_i64(2) * xe.clone() + rq(s.clone());
                let rest = cc - a.clone() * s / BigRational::from_integer(BigInt::from(2));
                let mut out = Expr::from_i64(0);
                if !a.is_zero() {
                    out = out
                        + if m == 1 {
                            logs.push((
                                a.clone() / BigRational::from_integer(BigInt::from(2)),
                                p.clone(),
                            ));
                            Expr::from_i64(0)
                        } else {
                            let mm = BigInt::from((m - 1) as i64);
                            rq(-a
                                / (BigRational::from_integer(BigInt::from(2))
                                    * BigRational::from_integer(mm.clone())))
                                * pow(pe.clone(), Expr::Integer(-mm))
                        };
                }
                if !rest.is_zero() {
                    out = out + rq(rest) * quadratic_reciprocal_power(&pe, &two_x_s, &delta, m);
                }
                Some(out)
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // sin**m * cos**n
    // -----------------------------------------------------------------------

    fn trig_powers(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        let factors: Vec<Expr> = match f {
            Expr::Mul(fs) => fs.clone(),
            other => vec![other.clone()],
        };
        let mut arg: Option<Expr> = None;
        let mut m: i64 = 0;
        let mut n: i64 = 0;
        for fac in &factors {
            let (base, e) = match fac {
                Expr::Pow(b, e) => ((**b).clone(), number(e)?),
                other => (other.clone(), BigRational::one()),
            };
            if !e.is_integer() || e.is_negative() {
                return None;
            }
            let k = e.to_integer().to_i64()?;
            let Expr::Function(name, args) = &base else {
                return None;
            };
            if args.len() != 1 {
                return None;
            }
            match &arg {
                Some(a) if *a != args[0] => return None,
                _ => arg = Some(args[0].clone()),
            }
            match name.as_str() {
                "sin" => m += k,
                "cos" => n += k,
                _ => return None,
            }
        }
        let u = arg?;
        let (a, _) = linear_coeffs(&u, &x)?;
        if m + n < 2 {
            return None;
        }
        let ia = recip(a);
        let s = func("sin", u.clone());
        let c = func("cos", u.clone());
        let t = Symbol::new("_fsym_int_u");
        let te = Expr::Sym(t.clone());
        if n % 2 == 1 {
            let poly = pow(
                Expr::from_i64(1) - pow(te.clone(), Expr::from_i64(2)),
                Expr::from_i64((n - 1) / 2),
            ) * pow(te.clone(), Expr::from_i64(m));
            let anti = poly_antiderivative(&poly, &t)?;
            let back = anti.subs(&HashMap::from([(t, s)]));
            return Some(ia * back);
        }
        if m % 2 == 1 {
            // u = cos: -int (1-u^2)^((m-1)/2) u^n du
            let poly = pow(
                Expr::from_i64(1) - pow(te.clone(), Expr::from_i64(2)),
                Expr::from_i64((m - 1) / 2),
            ) * pow(te.clone(), Expr::from_i64(n));
            let anti = poly_antiderivative(&poly, &t)?;
            let back = anti.subs(&HashMap::from([(t, c)]));
            return Some(-ia * back);
        }
        if n % 2 == 1 {
            let poly = pow(
                Expr::from_i64(1) - pow(te.clone(), Expr::from_i64(2)),
                Expr::from_i64((n - 1) / 2),
            ) * pow(te.clone(), Expr::from_i64(m));
            let anti = poly_antiderivative(&poly, &t)?;
            let back = anti.subs(&HashMap::from([(t, s)]));
            return Some(ia * back);
        }
        // Both even: reduction on the cosine power, then on the sine power.
        Some(ia * even_sin_cos(m, n, &s, &c, &u))
    }

    // -----------------------------------------------------------------------
    // exp * sin/cos
    // -----------------------------------------------------------------------

    fn exp_trig(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        let Expr::Mul(fs) = f else {
            return None;
        };
        if fs.len() != 2 {
            return None;
        }
        let (mut ex, mut tr) = (None, None);
        for fac in fs {
            match fac {
                Expr::Function(n, args) if n == "exp" && args.len() == 1 => {
                    ex = Some(args[0].clone())
                }
                Expr::Function(n, args) if (n == "sin" || n == "cos") && args.len() == 1 => {
                    tr = Some((n.clone(), args[0].clone()))
                }
                _ => return None,
            }
        }
        let (eu, (tn, tu)) = (ex?, tr?);
        let (a, _) = linear_coeffs(&eu, &x)?;
        let (c, _) = linear_coeffs(&tu, &x)?;
        let den = recip(a.clone() * a.clone() + c.clone() * c.clone());
        let e = func("exp", eu);
        let s = func("sin", tu.clone());
        let co = func("cos", tu);
        Some(match tn.as_str() {
            "sin" => e.clone() * s * (a * den.clone()) - e * co * (c * den),
            _ => e.clone() * co * (a * den.clone()) + e * s * (c * den),
        })
    }

    // -----------------------------------------------------------------------
    // Substitution: f(g(x)) * g'(x)
    // -----------------------------------------------------------------------

    fn substitution(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        let mut candidates = Vec::new();
        collect_candidates(f, &x, &mut candidates);
        let u = Symbol::new(format!("_fsym_sub{}", self.depth));
        let ue = Expr::Sym(u.clone());
        for g in candidates {
            let dg = crate::diff(&g, &x);
            if dg.is_zero() {
                continue;
            }
            // Never divide by a symbolic constant factor of g' (it may vanish).
            let (dc, _) = split_constant(&fsym_simplify::simplify(&dg), &x);
            if !provably_nonzero(&dc) {
                continue;
            }
            let ratio = f.clone() * recip(dg);
            let ratio = fsym_simplify::simplify(&ratio);
            let mut in_u = ratio.subs_expr(&g, &ue);
            if !is_free_of(&in_u, &x) {
                // u = x**d: replace x**(k*d) by u**k.
                if let Expr::Pow(base, d) = &g
                    && **base == self.xe()
                    && let Expr::Integer(d) = d.as_ref()
                {
                    in_u = replace_powers(&ratio, &x, d, &ue);
                } else if let Some((a, b)) = linear_coeffs(&g, &x) {
                    // u = a*x + b: rewrite x = (u - b)/a everywhere.
                    let xu = (ue.clone() - b) * recip(a);
                    in_u = fsym_simplify::simplify(&ratio.subs(&HashMap::from([(x.clone(), xu)])));
                }
            }
            if !is_free_of(&in_u, &x) {
                continue;
            }
            let saved = std::mem::replace(&mut self.x, u.clone());
            let anti = self.int(&in_u);
            self.x = saved;
            if let Some(anti) = anti {
                let back = anti.subs(&HashMap::from([(u.clone(), g)]));
                return Some(split_improper_powers(&back));
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Integration by parts
    // -----------------------------------------------------------------------

    fn parts(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        let factors: Vec<Expr> = match f {
            Expr::Mul(fs) => fs.clone(),
            other => vec![other.clone()],
        };
        // u = log/atan/asin/acos factor (possibly a power of log), dv = rest.
        for (i, fac) in factors.iter().enumerate() {
            let is_u = match fac {
                Expr::Function(n, args) => {
                    matches!(
                        n.as_str(),
                        "log" | "atan" | "asin" | "acos" | "asinh" | "acosh" | "atanh"
                    ) && args.len() == 1
                }
                Expr::Pow(b, e) => {
                    matches!(b.as_ref(), Expr::Function(n, _) if n == "log")
                        && number(e).is_some_and(|v| v.is_integer() && v.is_positive())
                }
                _ => false,
            };
            if !is_u {
                continue;
            }
            let dv = product(
                factors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, f)| f.clone()),
            );
            if !is_polynomial_in(&dv, &x) && !is_rational_power_of(&dv, &x) {
                continue;
            }
            let v = self.int(&dv)?;
            let du = crate::diff(fac, &x);
            let rest = fsym_simplify::simplify(&(v.clone() * du));
            let r = self.int(&rest)?;
            return Some(fac.clone() * v - r);
        }
        // u = polynomial, dv = exp/sin/cos/sinh/cosh of a linear argument.
        let (poly, other): (Vec<Expr>, Vec<Expr>) = factors
            .iter()
            .cloned()
            .partition(|f| is_polynomial_in(f, &x));
        if poly.is_empty() || other.is_empty() {
            return None;
        }
        // x**m * exp(q(x)) with q quadratic: u = x**(m-1), dv = x*exp(q)
        // (dv integrates by substitution; reduces the Gaussian moments).
        if other.len() == 1
            && let Expr::Function(n, args) = &other[0]
            && n == "exp"
            && args.len() == 1
            && !is_free_of(&args[0], &x)
            && linear_coeffs(&args[0], &x).is_none()
        {
            let p = product(poly.clone());
            let xe = self.xe();
            let u = fsym_simplify::simplify(&(p * recip(xe.clone())));
            if is_polynomial_in(&u, &x) && !is_free_of(&u, &x) {
                let dv = xe * other[0].clone();
                if let Some(v) = self.int(&dv) {
                    let du = crate::diff(&u, &x);
                    let rest = fsym_simplify::simplify(&(v.clone() * du));
                    if let Some(r) = self.int(&rest) {
                        return Some(u * v - r);
                    }
                }
            }
        }
        let linear_fn = |g: &Expr| {
            matches!(g, Expr::Function(n, args)
                if matches!(n.as_str(), "exp" | "sin" | "cos" | "sinh" | "cosh") && args.len() == 1
                    && linear_coeffs(&args[0], &x).is_some())
        };
        // dv: one linear exp/trig/hyperbolic factor, or exp times sin/cos.
        let ok = (other.len() == 1 && linear_fn(&other[0]))
            || (other.len() == 2
                && other.iter().all(linear_fn)
                && other
                    .iter()
                    .any(|g| matches!(g, Expr::Function(n, _) if n == "exp")));
        if !ok {
            return None;
        }
        let g_owned = product(other.clone());
        let g = &g_owned;
        let p = product(poly);
        if let Expr::Function(n, args) = g
            && n == "exp"
        {
            // int p*exp(a*x+b) = exp(a*x+b) * sum_k (-1)^k p^(k) / a^(k+1)
            let (a, _) = linear_coeffs(&args[0], &x)?;
            let mut acc = Expr::from_i64(0);
            let mut deriv = p;
            let mut scale = recip(a.clone());
            for _ in 0..64 {
                if deriv.is_zero() {
                    return Some(fsym_simplify::expand(&acc) * g.clone());
                }
                acc = acc + scale.clone() * deriv.clone();
                deriv = fsym_simplify::expand(&crate::diff(&deriv, &x));
                scale = -scale * recip(a.clone());
            }
            return None;
        }
        // int p*g = p*G1 - p'*G2 + p''*G3 - ...
        let mut out = Expr::from_i64(0);
        let mut deriv = p;
        let mut anti = g.clone();
        let mut sign = Expr::from_i64(1);
        for _ in 0..32 {
            if deriv.is_zero() {
                return Some(out);
            }
            anti = self.int(&anti)?;
            out = out + sign.clone() * deriv.clone() * anti.clone();
            deriv = fsym_simplify::expand(&crate::diff(&deriv, &x));
            sign = -sign;
        }
        None
    }
}

/// Rewrites `b**(n + f)` (integer `n >= 1`, `0 < f < 1`) as the expanded
/// `b**n` times `b**f`, the shape upstream returns for substitution results
/// (`(x**2 + 1)**(3/2)/3 -> x**2*sqrt(x**2 + 1)/3 + sqrt(x**2 + 1)/3`).
/// `exp(a + g)` with `a` free of `x` and `g` not: `(exp(a), exp(g))`.
fn split_exp_constant(f: &Expr, x: &Symbol) -> Option<(Expr, Expr)> {
    let Expr::Function(name, args) = f else {
        return None;
    };
    if name != "exp" || args.len() != 1 {
        return None;
    }
    let Expr::Add(terms) = &args[0] else {
        return None;
    };
    let (free, dep): (Vec<Expr>, Vec<Expr>) = terms.iter().cloned().partition(|t| is_free_of(t, x));
    if free.is_empty() || dep.is_empty() {
        return None;
    }
    Some((func("exp", sum(free)), func("exp", sum(dep))))
}

fn split_improper_powers(e: &Expr) -> Expr {
    fn walk(e: &Expr, changed: &mut bool) -> Expr {
        match e {
            Expr::Pow(b, p) => {
                let b = walk(b, changed);
                if let Some(r) = number(p)
                    && !r.is_integer()
                    && r > BigRational::one()
                    && matches!(b, Expr::Add(_))
                {
                    let n = r.numer().clone() / r.denom().clone();
                    let frac = r.clone() - BigRational::from_integer(n.clone());
                    *changed = true;
                    let whole = fsym_simplify::expand(&pow(b.clone(), Expr::Integer(n)));
                    let root = Expr::Pow(std::sync::Arc::new(b), std::sync::Arc::new(rq(frac)));
                    return distribute(vec![whole, root]);
                }
                pow(b, (**p).clone())
            }
            Expr::Add(xs) => sum(xs.iter().map(|t| walk(t, changed))),
            Expr::Mul(xs) => {
                let parts: Vec<Expr> = xs.iter().map(|t| walk(t, changed)).collect();
                if *changed {
                    distribute(parts)
                } else {
                    product(parts)
                }
            }
            other => other.clone(),
        }
    }
    let mut changed = false;
    let out = walk(e, &mut changed);
    if changed { out } else { e.clone() }
}

/// Product with every Add factor distributed over the remaining factors.
fn distribute(parts: Vec<Expr>) -> Expr {
    let mut terms = vec![Expr::from_i64(1)];
    for p in parts {
        let pieces: Vec<Expr> = match &p {
            Expr::Add(xs) => xs.clone(),
            Expr::Mul(xs) if xs.iter().any(|f| matches!(f, Expr::Add(_))) => {
                return product(terms.into_iter().map(|t| t * p.clone()));
            }
            other => vec![other.clone()],
        };
        terms = terms
            .iter()
            .flat_map(|t| pieces.iter().map(move |q| t.clone() * q.clone()))
            .collect();
    }
    sum(terms)
}

fn scale_poly(p: &UnivariatePoly, c: &BigRational) -> Option<UnivariatePoly> {
    p.mul(&UnivariatePoly::new(p.gen_sym.clone(), vec![c.clone()]))
        .ok()
}

/// `p` with integer coefficients, content removed and positive leading term.
fn primitive_expr(p: &UnivariatePoly) -> Expr {
    let mut den = BigInt::from(1);
    for c in &p.coeffs {
        let d = c.denom().clone();
        den = den.clone() * d.clone() / fsym_core::arith::gcd(&den, &d);
    }
    let scaled: Vec<BigInt> = p
        .coeffs
        .iter()
        .map(|c| (c.clone() * BigRational::from_integer(den.clone())).to_integer())
        .collect();
    let mut g = BigInt::from(0);
    for c in &scaled {
        g = fsym_core::arith::gcd(&g, c);
    }
    if g.is_zero() {
        return p.to_expr();
    }
    let mut coeffs: Vec<BigRational> = scaled
        .into_iter()
        .map(|c| BigRational::new(c, g.clone()))
        .collect();
    if coeffs.last().is_some_and(|c| c.is_negative()) {
        coeffs = coeffs.into_iter().map(|c| -c).collect();
    }
    UnivariatePoly::new(p.gen_sym.clone(), coeffs).to_expr()
}

/// `int dx / p**m` for an irreducible monic quadratic `p` with
/// `4t - s^2 = delta > 0`, `two_x_s = 2x + s`.
fn quadratic_reciprocal_power(p: &Expr, two_x_s: &Expr, delta: &BigRational, m: usize) -> Expr {
    let sd = pow(rq(delta.clone()), q(1, 2));
    let base = Expr::from_i64(2)
        * recip(sd.clone())
        * func(
            "atan",
            fsym_simplify::expand(&(two_x_s.clone() * recip(sd))),
        );
    let mut acc = base;
    for j in 2..=m {
        let jm1 = BigRational::from_integer(BigInt::from((j - 1) as i64));
        let first = two_x_s.clone()
            * recip(
                rq(jm1.clone() * delta.clone()) * pow(p.clone(), Expr::from_i64((j - 1) as i64)),
            );
        let coeff =
            BigRational::from_integer(BigInt::from(2 * (2 * j as i64 - 3))) / (jm1 * delta.clone());
        acc = first + rq(coeff) * acc;
    }
    acc
}

/// int sin^m cos^n du for even m, n >= 0 (reduction formulas).
fn even_sin_cos(m: i64, n: i64, s: &Expr, c: &Expr, u: &Expr) -> Expr {
    if m == 0 && n == 0 {
        return u.clone();
    }
    if n > 0 {
        // int s^m c^n = s^(m+1) c^(n-1)/(m+n) + (n-1)/(m+n) int s^m c^(n-2)
        let head = pow(s.clone(), Expr::from_i64(m + 1))
            * pow(c.clone(), Expr::from_i64(n - 1))
            * q(1, m + n);
        return head + q(n - 1, m + n) * even_sin_cos(m, n - 2, s, c, u);
    }
    // int s^m = -s^(m-1) c / m + (m-1)/m int s^(m-2)
    let head = -pow(s.clone(), Expr::from_i64(m - 1)) * c.clone() * q(1, m);
    head + q(m - 1, m) * even_sin_cos(m - 2, 0, s, c, u)
}

fn poly_antiderivative(p: &Expr, t: &Symbol) -> Option<Expr> {
    let poly = UnivariatePoly::from_expr(&fsym_simplify::expand(p), t).ok()?;
    Some(poly.integrate(BigRational::zero()).ok()?.to_expr())
}

/// `c * x**r` with `c` free of `x` and rational `r != -1`.
fn is_rational_power_of(e: &Expr, x: &Symbol) -> bool {
    let (_, rest) = split_constant(e, x);
    match rest.as_slice() {
        [Expr::Pow(b, r)] => {
            matches!(b.as_ref(), Expr::Sym(s) if s == x)
                && number(r).is_some_and(|v| v != -BigRational::one())
        }
        [Expr::Sym(s)] => s == x,
        _ => false,
    }
}

fn is_polynomial_in(e: &Expr, x: &Symbol) -> bool {
    match e {
        Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => true,
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().all(|t| is_polynomial_in(t, x)),
        Expr::Pow(b, p) => {
            is_free_of(e, x)
                || (is_polynomial_in(b, x)
                    && number(p).is_some_and(|v| v.is_integer() && !v.is_negative()))
        }
        other => is_free_of(other, x),
    }
}

/// Numerator/denominator polynomials over Q when `e` is a rational function
/// of `x` with rational coefficients.
pub(crate) fn rational_parts(e: &Expr, x: &Symbol) -> Option<(UnivariatePoly, UnivariatePoly)> {
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
                let (fn_, fd) = rational_parts(f, x)?;
                n = n.mul(&fn_).ok()?;
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
                if bn.is_zero() {
                    return None;
                }
                Some((bd.pow(kk).ok()?, bn.pow(kk).ok()?))
            }
        }
        _ => None,
    }
}

/// Replace `x**(k*d)` by `u**k` throughout `e` (for the substitution
/// `u = x**d`); any other occurrence of `x` is left (and later rejects).
fn replace_powers(e: &Expr, x: &Symbol, d: &BigInt, u: &Expr) -> Expr {
    match e {
        Expr::Pow(b, k) if matches!(b.as_ref(), Expr::Sym(s) if s == x) => match k.as_ref() {
            Expr::Integer(k) if (k % d).is_zero() => pow(u.clone(), Expr::Integer(k / d)),
            _ => e.clone(),
        },
        Expr::Add(xs) => sum(xs.iter().map(|t| replace_powers(t, x, d, u))),
        Expr::Mul(xs) => product(xs.iter().map(|t| replace_powers(t, x, d, u))),
        Expr::Pow(b, k) => pow(replace_powers(b, x, d, u), replace_powers(k, x, d, u)),
        Expr::Function(n, args) => Expr::Function(
            n.clone(),
            args.iter().map(|a| replace_powers(a, x, d, u)).collect(),
        ),
        other => other.clone(),
    }
}

fn collect_candidates(e: &Expr, x: &Symbol, out: &mut Vec<Expr>) {
    let push = |c: &Expr, out: &mut Vec<Expr>| {
        if !is_free_of(c, x) && !matches!(c, Expr::Sym(_)) && !out.contains(c) {
            out.push(c.clone());
        }
    };
    match e {
        Expr::Function(_, args) => {
            push(e, out);
            for a in args {
                push(a, out);
                collect_candidates(a, x, out);
            }
        }
        Expr::Pow(b, p) => {
            push(b, out);
            // x**k offers x**d for the proper divisors d of k (u = x**2 for x**4).
            if let (Expr::Sym(s), Expr::Integer(k)) = (b.as_ref(), p.as_ref())
                && s == x
                && let Some(k) = k.to_i64()
                && (2..=12).contains(&k)
            {
                for d in 2..k {
                    if k % d == 0 {
                        push(&pow(b.as_ref().clone(), Expr::from_i64(d)), out);
                    }
                }
            }
            collect_candidates(b, x, out);
            if !is_free_of(p, x) {
                push(p, out);
                collect_candidates(p, x, out);
            }
        }
        Expr::Add(xs) | Expr::Mul(xs) => {
            for t in xs {
                collect_candidates(t, x, out);
            }
        }
        _ => {}
    }
}

/// Antiderivative of `f` in `x` (no constant), or `None` when no rule
/// applies.
pub fn antiderivative(f: &Expr, x: &Symbol) -> Option<Expr> {
    let mut it = Integrator {
        x: x.clone(),
        depth: 0,
    };
    let r = it.int(f)?;
    // Rational integrands: upstream (ratint) reports the rational part as
    // one fraction in cancel form whichever lane found it.
    if rational_parts(f, x).is_some() {
        return Some(merge_rational_part(r, x));
    }
    Some(r)
}

/// Whether the integrand has a pole strictly inside `(a, b)`: exact for
/// rational integrands (real roots of the denominator), conservative
/// sampling of the integrand for the rest.
fn has_interior_singularity(f: &Expr, x: &Symbol, a: &Expr, b: &Expr) -> bool {
    let (Some(av), Some(bv)) = (
        crate::gruntz::complex_value(a),
        crate::gruntz::complex_value(b),
    ) else {
        // Infinite endpoints: sample a symmetric finite window instead.
        return sample_poles(f, x, -50.0, 50.0, a, b);
    };
    sample_poles(f, x, av.re, bv.re, a, b)
}

fn sample_poles(f: &Expr, x: &Symbol, lo: f64, hi: f64, a: &Expr, b: &Expr) -> bool {
    let finite = |e: &Expr| crate::gruntz::complex_value(e).map(|v| v.re);
    let lo = finite(a).unwrap_or(if lo < hi { lo } else { hi - 100.0 });
    let hi = finite(b).unwrap_or(if hi > lo {
        hi.max(lo + 100.0)
    } else {
        lo + 100.0
    });
    if let Some((_, den)) = rational_parts(f, x) {
        // Sign changes or exact zeros of the denominator on a fine grid.
        let steps = 2000;
        let mut prev: Option<f64> = None;
        for i in 1..steps {
            let t = lo + (hi - lo) * (i as f64) / (steps as f64);
            let v = den
                .coeffs
                .iter()
                .rev()
                .fold(0.0, |acc, c| acc * t + c.to_f64().unwrap_or(f64::NAN));
            if v == 0.0 {
                return true;
            }
            if let Some(p) = prev
                && p.signum() != v.signum()
            {
                return true;
            }
            prev = Some(v);
        }
        return false;
    }
    let steps = 400;
    for i in 1..steps {
        let t = lo + (hi - lo) * (i as f64) / (steps as f64);
        let at = f.subs(&HashMap::from([(
            x.clone(),
            Expr::Rational(BigRational::new(
                BigInt::from((t * 1_000_000.0).round() as i64),
                BigInt::from(1_000_000),
            )),
        )]));
        match crate::gruntz::complex_value(&at) {
            Some(v) if v.re.is_finite() && v.im.is_finite() => {}
            Some(_) => return true,
            None => {}
        }
    }
    false
}

/// Definite integral of `f` over `(a, b)` by the fundamental theorem.
pub fn definite(f: &Expr, x: &Symbol, a: &Expr, b: &Expr) -> Option<Expr> {
    if a == b {
        return Some(Expr::from_i64(0));
    }
    let anti = antiderivative(f, x)?;
    if has_interior_singularity(f, x, a, b) {
        return None;
    }
    let upper = crate::limit_dir(&anti, x, b, Direction::Minus).ok()?;
    let lower = crate::limit_dir(&anti, x, a, Direction::Plus).ok()?;
    if matches!(
        upper,
        Expr::Const(Constant::ComplexInfinity | Constant::NaN)
    ) || matches!(
        lower,
        Expr::Const(Constant::ComplexInfinity | Constant::NaN)
    ) {
        return None;
    }
    let value = upper - lower;
    Some(fsym_simplify::simplify(&value))
}

fn gcd_usize(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd_usize(b, a % b) }
}

fn has_function(e: &Expr) -> bool {
    match e {
        Expr::Function(_, _) => true,
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().any(has_function),
        Expr::Pow(b, k) => has_function(b) || has_function(k),
        _ => false,
    }
}

/// Integer cancel form of `num/den` (upstream ``cancel``): both sides
/// primitive over ZZ with the rational content split as p/q into the
/// numerator and denominator, denominator leading coefficient positive.
fn cancel_form(num: &UnivariatePoly, den: &UnivariatePoly) -> Option<Expr> {
    fn integer_primitive(p: &UnivariatePoly) -> (BigRational, Vec<BigInt>) {
        let mut l = BigInt::one();
        for c in &p.coeffs {
            let d = c.denom().clone();
            let g = fsym_core::arith::gcd(&l, &d);
            l = l * d / g;
        }
        let ints: Vec<BigInt> = p
            .coeffs
            .iter()
            .map(|c| (c.clone() * BigRational::from_integer(l.clone())).to_integer())
            .collect();
        let mut g = BigInt::zero();
        for c in &ints {
            g = fsym_core::arith::gcd(&g, c);
        }
        if g.is_zero() {
            return (BigRational::zero(), ints);
        }
        let lead_neg = ints
            .iter()
            .rev()
            .find(|c| !c.is_zero())
            .is_some_and(|c| c.is_negative());
        let g = if lead_neg { -g } else { g };
        let prim = ints.iter().map(|c| c.clone() / g.clone()).collect();
        (BigRational::new(g, l), prim)
    }
    let (cn, pn) = integer_primitive(num);
    let (cd, pd) = integer_primitive(den);
    if cd.is_zero() {
        return None;
    }
    let c = cn / cd;
    let (p, q) = (c.numer().clone(), c.denom().clone());
    let x = num.gen_sym.clone();
    let to_poly = |v: Vec<BigInt>, k: &BigInt| {
        UnivariatePoly::new(
            x.clone(),
            v.into_iter()
                .map(|a| BigRational::from_integer(a * k.clone()))
                .collect(),
        )
    };
    let top = to_poly(pn, &p).to_expr();
    let bottom = to_poly(pd, &q).to_expr();
    Some(top * Expr::pow(bottom, Expr::from_i64(-1)))
}

fn merge_rational_part(out: Expr, x: &Symbol) -> Expr {
    let terms = match &out {
        Expr::Add(ts) => ts.clone(),
        other => vec![other.clone()],
    };
    let (rational, rest): (Vec<Expr>, Vec<Expr>) = terms
        .into_iter()
        .partition(|t| !has_function(t) && !is_free_of(t, x) && !is_polynomial_in(t, x));
    if rational.is_empty() {
        return out;
    }
    let Some((num, den)) = rational_parts(&sum(rational.clone()), x) else {
        return out;
    };
    let Ok(g) = num.gcd(&den) else { return out };
    let (Ok((num, _)), Ok((den, _))) = (num.div_rem(&g), den.div_rem(&g)) else {
        return out;
    };
    match cancel_form(&num, &den) {
        Some(frac) => sum(rest.into_iter().chain(std::iter::once(frac))),
        None => out,
    }
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

    /// d/dx F - f simplifies to zero.
    fn check(f: Expr) {
        let anti = antiderivative(&f, &x()).unwrap_or_else(|| panic!("no antiderivative for {f}"));
        let back = fsym_simplify::simplify(&(crate::diff(&anti, &x()) - f.clone()));
        let num = crate::gruntz::complex_value(&back.subs(&HashMap::from([(x(), q(3, 7))])));
        assert!(
            back.is_zero() || num.is_some_and(|v| v.norm() < 1e-9),
            "F' - f = {back} for f = {f}, F = {anti}"
        );
    }

    #[test]
    fn rational_functions() {
        check(recip(pow(xe(), Expr::from_i64(2)) + Expr::from_i64(1)));
        check(recip(pow(xe(), Expr::from_i64(2)) - Expr::from_i64(1)));
        check(recip(pow(xe(), Expr::from_i64(3)) - Expr::from_i64(1)));
        check(
            xe() * recip(
                pow(xe(), Expr::from_i64(2)) + Expr::from_i64(2) * xe() + Expr::from_i64(5),
            ),
        );
        check(recip(pow(
            pow(xe(), Expr::from_i64(2)) + Expr::from_i64(1),
            Expr::from_i64(2),
        )));
        check(recip(pow(xe() + Expr::from_i64(1), Expr::from_i64(3))));
    }

    #[test]
    fn transcendental_rules() {
        check(pow(func("sin", xe()), Expr::from_i64(2)));
        check(pow(func("sin", xe()), Expr::from_i64(3)));
        check(func("log", xe()));
        check(xe() * func("exp", pow(xe(), Expr::from_i64(2))));
        check(func("exp", xe()) * func("sin", xe()));
        check(pow(xe(), Expr::from_i64(2)) * func("cos", xe()));
        check(func("atan", xe()));
        check(func("log", xe()) * recip(xe()));
    }
}
