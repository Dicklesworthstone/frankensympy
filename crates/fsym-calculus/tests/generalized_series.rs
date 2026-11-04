//! Exact coefficients and refusal boundaries for the generalized series lane.

use fsym_calculus::series::{SeriesError, series_in};
use fsym_core::{BigRational, Expr, Symbol};
use std::collections::BTreeMap;
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn nonlinear_logarithms_cannot_be_proved_affine_by_three_samples() {
    let t = Symbol::new("t");
    let logarithm = function("log", Expr::Sym(t.clone()));
    // This nonzero cubic vanishes at all three points used by the old
    // recognizer. Its exponential is not the constant one near t = 0+.
    let impostor = logarithm.clone()
        * (logarithm.clone() - Expr::from_i64(1))
        * (logarithm.clone() - Expr::from_i64(2));
    for exponent in [
        impostor.clone(),
        impostor + Expr::from_i64(2) * logarithm.clone() + Expr::from_i64(7),
    ] {
        assert!(matches!(
            series_in(&function("exp", exponent), &t, &q(5, 1), &logarithm),
            Err(SeriesError::Unsupported(_))
        ));
    }
}

#[test]
fn exact_affine_logarithms_still_shift_the_exponential_series() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    // exp(3 log(t) + t) = t^3 exp(t); coefficients follow 1/k!.
    let source = function("exp", Expr::from_i64(3) * logarithm.clone() + te);
    let expanded = series_in(&source, &t, &q(7, 1), &logarithm).unwrap();
    assert_eq!(
        expanded.terms,
        BTreeMap::from([
            (q(3, 1), Expr::from_i64(1)),
            (q(4, 1), Expr::from_i64(1)),
            (q(5, 1), Expr::Rational(q(1, 2))),
            (q(6, 1), Expr::Rational(q(1, 6))),
        ])
    );
    assert_eq!(expanded.order, Some(q(7, 1)));
}

#[test]
fn powers_with_poles_keep_terms_needed_by_later_products() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    for exponent in 2..=8 {
        // (t^-1 + t^2)^m = sum_j binom(m,j) t^(3j-m).
        // The first factor's t^2 term lies beyond the requested order,
        // but products with remaining poles can bring it back below it.
        let source = power(power(te.clone(), -1) + power(te.clone(), 2), exponent);
        let expanded = series_in(&source, &t, &q(2, 1), &logarithm).unwrap();
        let mut expected = BTreeMap::new();
        let mut binomial = 1_i64;
        for j in 0..=exponent {
            if 3 * j - exponent < 2 {
                expected.insert(q(3 * j - exponent, 1), Expr::from_i64(binomial));
            }
            if j < exponent {
                binomial = binomial * (exponent - j) / (j + 1);
            }
        }
        assert_eq!(expanded.terms, expected, "power {exponent}");
        assert_eq!(expanded.order, Some(q(2, 1)));
    }
}

#[test]
fn analytic_laurent_powers_have_the_correct_constant_term() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    // (1/t + sin(t))^2 = t^-2 + 2 + 2/3 t^2 + O(t^4).
    let source = power(power(te.clone(), -1) + function("sin", te), 2);
    let expanded = series_in(&source, &t, &q(1, 1), &logarithm).unwrap();
    assert_eq!(
        expanded.terms,
        BTreeMap::from([(q(-2, 1), Expr::from_i64(1)), (q(0, 1), Expr::from_i64(2))])
    );
    assert_eq!(expanded.order, Some(q(1, 1)));
}
