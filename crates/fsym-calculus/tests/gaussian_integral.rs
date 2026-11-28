//! Exact polynomial Gaussian moments through the ordinary calculus entry points.

use fsym_calculus::{integrate, integrate_definite};
use fsym_core::{Constant, Expr, Symbol, parse};
use std::collections::HashMap;

fn e(src: &str) -> Expr {
    parse(src).unwrap().subs(&HashMap::new())
}

fn same(value: Expr, expected: Expr) {
    let delta =
        fsym_simplify::simplify(&fsym_simplify::expand(&(value.clone() - expected.clone())));
    assert!(delta.is_zero(), "{value} != {expected}; residual {delta}");
}

#[test]
fn shifted_mass_and_moments_at_both_infinities() {
    let x = Symbol::new("x");
    let lo = Expr::Const(Constant::NegativeInfinity);
    let hi = Expr::Const(Constant::Infinity);
    for (p, factor) in [
        ("1", "1"),
        ("x", "2"),
        ("x^2", "9/2"),
        ("x^3", "11"),
        ("x^4", "115/4"),
        ("(x-2)^7", "0"),
        ("(x-2)^8", "105/16"),
    ] {
        let source = e(&format!("({p})*exp(-(x-2)^2)"));
        let expected = e(&format!("({factor})*sqrt(pi)"));
        same(
            integrate_definite(&source, &x, &lo, &hi).unwrap(),
            expected.clone(),
        );
        same(
            integrate_definite(&source, &x, &hi, &lo).unwrap(),
            -expected,
        );
    }
}

#[test]
fn residual_free_weight_gives_pure_exponential_boundary_terms() {
    let x = Symbol::new("x");
    // H = x^3 - 2, q = -2x^2 + 3x - 1; P = H' + q'H.
    let source = e("(-4*x^4+3*x^3+3*x^2+8*x-6)*exp(-2*x^2+3*x-1)");
    same(
        integrate(&source, &x).unwrap(),
        e("(x^3-2)*exp(-2*x^2+3*x-1)"),
    );
    same(
        integrate_definite(&source, &x, &e("1"), &Expr::Const(Constant::Infinity)).unwrap(),
        e("1"),
    );
    same(
        integrate_definite(&source, &x, &e("2"), &e("1")).unwrap(),
        e("-1-6*exp(-3)"),
    );
}

#[test]
fn scaled_translated_half_lines_and_finite_interval() {
    let x = Symbol::new("x");
    let f = e("exp(-4*x^2+4*x+2)");
    let ninf = Expr::Const(Constant::NegativeInfinity);
    let inf = Expr::Const(Constant::Infinity);
    let half = e("sqrt(pi)*exp(3)/4");
    let center = e("1/2");
    same(
        integrate_definite(&f, &x, &ninf, &center).unwrap(),
        half.clone(),
    );
    same(
        integrate_definite(&f, &x, &center, &inf).unwrap(),
        half.clone(),
    );
    same(
        integrate_definite(&f, &x, &center, &e("1")).unwrap(),
        half * e("erf(1)"),
    );
}

#[test]
fn held_interior_pole_and_unproved_decay_cannot_gain_gaussian_mass() {
    let x = Symbol::new("x");
    let pole = e("x-1");
    let source = Expr::Mul(vec![
        e("exp(-(x-1)^2)"),
        pole.clone(),
        Expr::Pow(pole.into(), Expr::from_i64(-1).into()),
    ]);
    let lo = Expr::Const(Constant::NegativeInfinity);
    let hi = Expr::Const(Constant::Infinity);
    assert!(integrate_definite(&source, &x, &lo, &hi).is_err());
    assert!(integrate_definite(&e("exp(a*x^2+2*x)"), &x, &lo, &hi).is_err());
}
