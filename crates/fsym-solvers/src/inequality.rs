//! Univariate real inequalities `f(x) op 0` over the reals.
//!
//! The solution set is assembled from *critical points*: real zeros of `f`,
//! real zeros of every `Abs` argument, denominator, even-root radicand and
//! `log` argument (domain boundaries). Between consecutive critical points
//! `f` has constant sign on its real domain, so one rational sample point per
//! open gap decides the gap; each critical point is decided by exact
//! substitution. Signs come from certified ball enclosures, never from
//! floating heuristics, and any undecided sign, symbolic parameter,
//! periodic generator, or incomplete root set refuses with a typed error.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::HashMap;

use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_sets::SymSet;
use fsym_sets::realline::{self, Piece, real_order};
use num_traits::{One, Signed, Zero};

use crate::SolverError;
use crate::univariate::solve_univariate;

/// Relational operator of `f(x) op 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IneqOp {
    Lt,
    Le,
    Gt,
    Ge,
}

impl IneqOp {
    pub fn from_str_op(op: &str) -> Option<Self> {
        Some(match op {
            "<" => IneqOp::Lt,
            "<=" => IneqOp::Le,
            ">" => IneqOp::Gt,
            ">=" => IneqOp::Ge,
            _ => return None,
        })
    }

    fn holds(self, sign: Ordering) -> bool {
        match self {
            IneqOp::Lt => sign == Ordering::Less,
            IneqOp::Le => sign != Ordering::Greater,
            IneqOp::Gt => sign == Ordering::Greater,
            IneqOp::Ge => sign != Ordering::Less,
        }
    }
}

fn refuse(msg: &str) -> SolverError {
    SolverError::IncompleteSolutionSet(msg.to_string())
}

fn zero() -> Expr {
    Expr::Integer(BigInt::zero())
}

fn contains_x(e: &Expr, x: &Symbol) -> bool {
    e.free_symbols().iter().any(|s| s == x)
}

/// Domain/branch structure of `f`: `Abs` arguments, denominators,
/// even-root radicands and log arguments that depend on `x`.
#[derive(Default)]
struct Structure {
    abs_args: Vec<Expr>,
    denominators: Vec<Expr>,
    nonneg: Vec<Expr>,
    positive: Vec<Expr>,
}

fn scan(e: &Expr, x: &Symbol, s: &mut Structure) -> Result<(), SolverError> {
    if !contains_x(e, x) {
        return Ok(());
    }
    match e {
        Expr::Sym(_) => Ok(()),
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().try_for_each(|t| scan(t, x, s)),
        Expr::Pow(b, p) => {
            if contains_x(p, x) {
                // b**g(x): real for b > 0 constant bases only.
                if contains_x(b, x) {
                    s.positive.push((**b).clone());
                } else if real_order(b, &zero()) != Some(Ordering::Greater) {
                    return Err(refuse("power with non-positive constant base"));
                }
                scan(b, x, s)?;
                return scan(p, x, s);
            }
            match p.as_ref() {
                Expr::Integer(k) => {
                    if k.is_negative() {
                        s.denominators.push((**b).clone());
                    }
                }
                Expr::Rational(r) => {
                    if r.denom() % BigInt::from(2) == BigInt::zero() {
                        s.nonneg.push((**b).clone());
                    }
                    if r.is_negative() {
                        s.denominators.push((**b).clone());
                    }
                }
                _ => return Err(refuse("symbolic exponent")),
            }
            scan(b, x, s)
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let a = &args[0];
            match name.as_str() {
                "Abs" => s.abs_args.push(a.clone()),
                "log" => s.positive.push(a.clone()),
                "exp" | "atan" | "sinh" | "asinh" | "tanh" | "cbrt" => {}
                _ => return Err(refuse("unsupported or periodic function of the variable")),
            }
            scan(a, x, s)
        }
        _ => Err(refuse("unsupported expression shape")),
    }
}

/// Real zeros of `g` (complete set) or a refusal.
fn real_zeros(g: &Expr, x: &Symbol) -> Result<Vec<Expr>, SolverError> {
    if !contains_x(g, x) {
        return Ok(Vec::new());
    }
    let roots = match solve_univariate(g, x) {
        Ok(r) => r,
        Err(SolverError::NoSolution) => Vec::new(),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for r in roots {
        if contains_x(&r, x) || !r.free_symbols().is_empty() {
            return Err(refuse("root depends on a parameter"));
        }
        // Keep certified real roots only; undecidable roots refuse.
        match real_order(&r, &zero()) {
            Some(_) => out.push(r),
            None => {
                if is_nonreal(&r) {
                    continue;
                }
                return Err(refuse("cannot certify whether a root is real"));
            }
        }
    }
    Ok(out)
}

fn is_nonreal(e: &Expr) -> bool {
    fsym_calculus::gruntz::complex_value(e).is_some_and(|v| v.im.abs() > 1e-12 * (1.0 + v.re.abs()))
}

fn subs(e: &Expr, x: &Symbol, v: &Expr) -> Expr {
    e.subs(&HashMap::from([(x.clone(), v.clone())]))
}

/// Certified sign of a real constant, `None` when undecided.
fn const_sign(c: &Expr) -> Option<Ordering> {
    if c.is_zero() {
        return Some(Ordering::Equal);
    }
    if let Some(o) = real_order(c, &zero()) {
        return Some(o);
    }
    let s = fsym_simplify::simplify(c);
    if s.is_zero() {
        return Some(Ordering::Equal);
    }
    real_order(&s, &zero())
}

/// `Some(sign)` of `f` at the point `v`, `None` when `f` is not real
/// (outside the domain) there, or a refusal when a sign is undecidable.
fn sign_at(
    f: &Expr,
    st: &Structure,
    x: &Symbol,
    v: &Expr,
) -> Result<Option<Ordering>, SolverError> {
    for d in &st.denominators {
        match const_sign(&subs(d, x, v)) {
            Some(Ordering::Equal) => return Ok(None),
            Some(_) => {}
            None => return Err(refuse("undecided denominator sign")),
        }
    }
    for r in &st.nonneg {
        match const_sign(&subs(r, x, v)) {
            Some(Ordering::Less) => return Ok(None),
            Some(_) => {}
            None => return Err(refuse("undecided radicand sign")),
        }
    }
    for p in &st.positive {
        match const_sign(&subs(p, x, v)) {
            Some(Ordering::Greater) => {}
            Some(_) => return Ok(None),
            None => return Err(refuse("undecided log-argument sign")),
        }
    }
    let val = subs(f, x, v);
    match const_sign(&val) {
        Some(o) => Ok(Some(o)),
        None => Err(refuse("undecided sign of the expression")),
    }
}

fn floor_q(q: &BigRational) -> BigRational {
    let t = BigRational::from_integer(q.to_integer());
    if &t > q { t - BigRational::one() } else { t }
}

/// A rational strictly between two certified-ordered endpoints.
fn sample_between(a: &Expr, b: &Expr) -> Result<Expr, SolverError> {
    let ninf = matches!(a, Expr::Const(Constant::NegativeInfinity));
    let pinf = matches!(b, Expr::Const(Constant::Infinity));
    let ball = |e: &Expr| {
        e.evalf_ball(30)
            .map_err(|_| refuse("cannot enclose a critical point"))
    };
    let q = match (ninf, pinf) {
        (true, true) => BigRational::zero(),
        (true, false) => floor_q(&ball(b)?.lower()) - BigRational::one(),
        (false, true) => floor_q(&ball(a)?.upper()) + BigRational::one() + BigRational::one(),
        (false, false) => {
            let lo = ball(a)?.upper();
            let hi = ball(b)?.lower();
            if lo >= hi {
                return Err(refuse("critical points too close to separate"));
            }
            // Prefer an integer in the gap, else the midpoint.
            let k = floor_q(&lo) + BigRational::one();
            if k < hi {
                k
            } else {
                (lo + hi) / BigRational::from_integer(BigInt::from(2))
            }
        }
    };
    Ok(fsym_core::elementary::rational_expr(q))
}

/// Solution set of `f(x) op 0` over the reals.
pub fn solve_univariate_inequality(
    f: &Expr,
    x: &Symbol,
    op: IneqOp,
) -> Result<SymSet, SolverError> {
    if f.free_symbols().iter().any(|s| s != x) {
        return Err(refuse("inequality with symbolic parameters"));
    }
    let mut st = Structure::default();
    scan(f, x, &mut st)?;

    let mut points: Vec<Expr> = Vec::new();
    for g in st
        .abs_args
        .iter()
        .chain(&st.denominators)
        .chain(&st.nonneg)
        .chain(&st.positive)
    {
        points.extend(real_zeros(g, x)?);
    }
    // Zeros of f on each sign region of the Abs arguments.
    let breaks = sorted_unique(points.clone())?;
    let mut regions: Vec<(Expr, Expr)> = Vec::new();
    let mut lo = Expr::Const(Constant::NegativeInfinity);
    for b in &breaks {
        regions.push((lo.clone(), b.clone()));
        lo = b.clone();
    }
    regions.push((lo, Expr::Const(Constant::Infinity)));
    for (a, b) in &regions {
        let probe = sample_between(a, b)?;
        let mut fr = f.clone();
        for arg in &st.abs_args {
            let sgn = const_sign(&subs(arg, x, &probe))
                .ok_or_else(|| refuse("undecided Abs-argument sign"))?;
            let replacement = if sgn == Ordering::Less {
                Expr::Mul(vec![Expr::from_i64(-1), arg.clone()])
            } else {
                arg.clone()
            };
            fr = fr.subs_expr(
                &Expr::Function("Abs".into(), vec![arg.clone()]),
                &replacement,
            );
        }
        let fr = fsym_simplify::simplify(&fr);
        for r in real_zeros(&fr, x)? {
            let inside = real_order(&r, a) == Some(Ordering::Greater)
                && real_order(&r, b) == Some(Ordering::Less);
            if inside {
                points.push(r);
            }
        }
    }
    let points = sorted_unique(points)?;

    let mut pieces: Vec<Piece> = Vec::new();
    let mut lo = Expr::Const(Constant::NegativeInfinity);
    let mut endpoints = points.clone();
    endpoints.push(Expr::Const(Constant::Infinity));
    for (i, hi) in endpoints.iter().enumerate() {
        let probe = sample_between(&lo, hi)?;
        if let Some(sign) = sign_at(f, &st, x, &probe)?
            && op.holds(sign)
        {
            pieces.push(Piece {
                lo: lo.clone(),
                hi: hi.clone(),
                lo_open: true,
                hi_open: true,
            });
        }
        if i < points.len()
            && let Some(sign) = sign_at(f, &st, x, hi)?
            && op.holds(sign)
        {
            pieces.push(Piece {
                lo: hi.clone(),
                hi: hi.clone(),
                lo_open: false,
                hi_open: false,
            });
        }
        lo = hi.clone();
    }
    let merged = realline::normalize(pieces).ok_or_else(|| refuse("unordered pieces"))?;
    Ok(realline::to_set(merged))
}

fn sorted_unique(mut pts: Vec<Expr>) -> Result<Vec<Expr>, SolverError> {
    let mut out: Vec<Expr> = Vec::new();
    pts.dedup();
    for p in pts {
        let mut pos = out.len();
        let mut dup = false;
        for (i, q) in out.iter().enumerate() {
            match real_order(&p, q).ok_or_else(|| refuse("unordered critical points"))? {
                Ordering::Equal => {
                    dup = true;
                    break;
                }
                Ordering::Less => {
                    pos = i;
                    break;
                }
                Ordering::Greater => {}
            }
        }
        if !dup {
            out.insert(pos, p);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::parse;

    fn solve(src: &str, op: IneqOp) -> String {
        let e = parse(src).unwrap();
        let x = Symbol::new("x");
        format!("{}", solve_univariate_inequality(&e, &x, op).unwrap())
    }

    #[test]
    fn quadratic_inequalities() {
        assert_eq!(solve("x**2 - 4", IneqOp::Lt), "Interval(-2, 2)");
        assert_eq!(
            solve("x**2 - 4", IneqOp::Gt),
            "Union(Interval(-oo, -2) | Interval(2, oo))"
        );
        assert_eq!(solve("x**2 - 4", IneqOp::Le), "Interval[-2, 2]");
    }

    #[test]
    fn rational_inequality_excludes_poles() {
        assert_eq!(solve("1/x", IneqOp::Gt), "Interval(0, oo)");
        assert_eq!(
            solve("(x - 1)/(x + 1)", IneqOp::Ge),
            "Union(Interval(-oo, -1) | Interval[1, oo))"
        );
    }

    #[test]
    fn abs_and_domain_boundaries() {
        assert_eq!(solve("Abs(x) - 3", IneqOp::Lt), "Interval(-3, 3)");
        assert_eq!(solve("x**(1/2) - 2", IneqOp::Lt), "Interval[0, 4)");
        assert_eq!(solve("log(x)", IneqOp::Gt), "Interval(1, oo)");
    }

    #[test]
    fn refuses_periodic_and_parametric() {
        let x = Symbol::new("x");
        let e = parse("sin(x)").unwrap();
        assert!(solve_univariate_inequality(&e, &x, IneqOp::Gt).is_err());
        let e = parse("x - a").unwrap();
        assert!(solve_univariate_inequality(&e, &x, IneqOp::Gt).is_err());
    }
}
