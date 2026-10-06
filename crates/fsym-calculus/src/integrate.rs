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
        if let Some(r) = self.rational(f) {
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
                if e == Expr::from_i64(2)
                    && let Expr::Function(fname, fargs) = &b
                    && fargs.len() == 1
                    && let Some((a, _)) = linear_coeffs(&fargs[0], &x)
                {
                    let u = fargs[0].clone();
                    match fname.as_str() {
                        "sec" => return Some(recip(a) * func("tan", u)),
                        "csc" => return Some(-recip(a) * func("cot", u)),
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

    /// exp(-c*x**2) for a positive rational c: sqrt(pi)*erf(sqrt(c)*x)/(2*sqrt(c)).
    fn gaussian(&self, u: &Expr) -> Option<Expr> {
        let x = &self.x;
        let x2 = pow(self.xe(), Expr::from_i64(2));
        let c = fsym_simplify::expand(&(-u.clone() * recip(x2)));
        if !is_free_of(&c, x) {
            return None;
        }
        let cv = number(&c)?;
        if !cv.is_positive() {
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
        if !kv.is_positive() || cv.is_zero() {
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

    fn rational(&mut self, f: &Expr) -> Option<Expr> {
        let x = self.x.clone();
        let (num, den) = rational_parts(f, &x)?;
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
            let in_u = ratio.subs_expr(&g, &ue);
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
            if !is_polynomial_in(&dv, &x) {
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
    it.int(f)
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
