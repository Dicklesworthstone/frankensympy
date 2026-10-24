//! Public definite-integration domain and endpoint regressions.

use fsym_calculus::integrate::definite;
use fsym_calculus::transforms::integrate_definite;
use fsym_core::elementary::eval_pow;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use std::sync::Arc;

fn q(n: i64, d: i64) -> Expr {
    Expr::Rational(BigRational::new(BigInt::from(n), BigInt::from(d)))
}

fn function(name: &str, arg: Expr) -> Expr {
    Expr::Function(name.to_owned(), vec![arg])
}

fn assert_exact(actual: Expr, expected: Expr) {
    let difference = fsym_simplify::try_simplify(&(actual.clone() - expected.clone())).unwrap();
    assert!(difference.is_zero(), "expected {expected}, got {actual}");
}

#[test]
fn transcendental_constants_do_not_hide_rational_poles() {
    let x = Symbol::new("x");
    let integrand = Expr::Const(Constant::Pi)
        * eval_pow(
            Expr::from_i64(3) * Expr::Sym(x.clone()) - Expr::from_i64(1),
            Expr::from_i64(-2),
        );
    let zero = Expr::from_i64(0);
    let one = Expr::from_i64(1);
    let two = Expr::from_i64(2);

    // The old finite sampling grid misses x = 1/3 and publishes -pi/2.
    assert!(definite(&integrand, &x, &zero, &one).is_none());
    assert!(integrate_definite(&integrand, &x, &zero, &one).is_err());
    assert_exact(
        integrate_definite(&integrand, &x, &one, &two).unwrap(),
        q(1, 10) * Expr::Const(Constant::Pi),
    );
    assert_exact(
        integrate_definite(&integrand, &x, &two, &one).unwrap(),
        q(-1, 10) * Expr::Const(Constant::Pi),
    );
}

#[test]
fn positive_branches_keep_integrable_endpoint_singularities() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let zero = Expr::from_i64(0);
    let one = Expr::from_i64(1);
    let inverse_root = eval_pow(xe.clone(), q(-1, 2));
    assert_exact(
        integrate_definite(&inverse_root, &x, &zero, &one).unwrap(),
        Expr::from_i64(2),
    );
    assert_exact(
        integrate_definite(&inverse_root, &x, &one, &zero).unwrap(),
        Expr::from_i64(-2),
    );
    assert_exact(
        integrate_definite(&eval_pow(xe.clone(), q(1, 2)), &x, &zero, &one).unwrap(),
        q(2, 3),
    );
    assert_exact(
        integrate_definite(&function("log", xe), &x, &zero, &one).unwrap(),
        Expr::from_i64(-1),
    );
}

#[test]
fn entire_functions_and_symbolic_polynomial_endpoints_still_work() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let xe = Expr::Sym(x.clone());
    let zero = Expr::from_i64(0);
    let one = Expr::from_i64(1);
    assert_exact(
        integrate_definite(&function("sin", xe.clone()), &x, &zero, &one).unwrap(),
        one.clone() - function("cos", one.clone()),
    );
    assert_exact(
        integrate_definite(&function("exp", xe.clone()), &x, &zero, &one).unwrap(),
        Expr::Const(Constant::E) - one,
    );
    assert_exact(
        integrate_definite(&xe, &x, &zero, &Expr::Sym(y.clone())).unwrap(),
        q(1, 2) * eval_pow(Expr::Sym(y), Expr::from_i64(2)),
    );
}

#[test]
fn certified_trigonometric_denominator_and_irrational_endpoint_work() {
    let x = Symbol::new("x");
    let integrand = eval_pow(function("cos", Expr::Sym(x.clone())), Expr::from_i64(-2));
    assert_exact(
        integrate_definite(
            &integrand,
            &x,
            &Expr::from_i64(0),
            &(q(1, 4) * Expr::Const(Constant::Pi)),
        )
        .unwrap(),
        Expr::from_i64(1),
    );
}

#[test]
fn a_pole_hidden_inside_an_endpoint_enclosure_is_not_missed() {
    let x = Symbol::new("x");
    let e = Expr::Const(Constant::E);
    let coarse = e.evalf_ball(24).unwrap();
    let refined = e.evalf_ball(48).unwrap();
    let midpoint = coarse.midpoint().clone();
    let two = BigRational::from_integer(BigInt::from(2));

    // Construct an exact rational pole strictly between the old endpoint
    // midpoint and the true endpoint. The refined ball proves which side
    // contains e, so this fixture does not assume a rounding direction.
    let (pole, a, b, rounded_a, rounded_b) = if refined.lower() > midpoint {
        (
            (&midpoint + refined.lower()) / two,
            Expr::from_i64(0),
            e.clone(),
            Expr::from_i64(0),
            Expr::Rational(midpoint),
        )
    } else {
        assert!(refined.upper() < midpoint);
        (
            (&midpoint + refined.upper()) / two,
            e.clone(),
            Expr::from_i64(4),
            Expr::Rational(midpoint),
            Expr::from_i64(4),
        )
    };
    let integrand = eval_pow(
        Expr::Sym(x.clone()) - Expr::Rational(pole),
        Expr::from_i64(-2),
    );
    assert!(definite(&integrand, &x, &rounded_a, &rounded_b).is_some());
    assert!(definite(&integrand, &x, &a, &b).is_none());
    assert!(definite(&integrand, &x, &b, &a).is_none());
    assert!(integrate_definite(&integrand, &x, &a, &b).is_err());
}

#[test]
fn undefined_held_constant_subexpressions_are_checked_before_cancellation() {
    let x = Symbol::new("x");
    let bad_log = function("log", Expr::from_i64(0));
    let held = [
        Expr::Pow(Arc::new(bad_log.clone()), Arc::new(Expr::from_i64(0))),
        Expr::Mul(vec![Expr::from_i64(0), bad_log]),
        Expr::Pow(
            Arc::new(Expr::Pow(
                Arc::new(Expr::from_i64(0)),
                Arc::new(Expr::from_i64(-1)),
            )),
            Arc::new(Expr::from_i64(-1)),
        ),
    ];
    for integrand in held {
        assert!(definite(&integrand, &x, &Expr::from_i64(0), &Expr::from_i64(1)).is_none());
        assert!(
            integrate_definite(&integrand, &x, &Expr::from_i64(0), &Expr::from_i64(1)).is_err()
        );
        // Bounds need the same original-source check: held log(0)**0
        // must not become the finite endpoint 1 during limit normalization.
        assert!(definite(&Expr::from_i64(1), &x, &integrand, &Expr::from_i64(2)).is_none());
        assert!(
            integrate_definite(&Expr::from_i64(1), &x, &integrand, &Expr::from_i64(2)).is_err()
        );
        assert!(definite(&Expr::from_i64(1), &x, &Expr::from_i64(2), &integrand).is_none());
    }
}

#[test]
fn unsupported_branches_and_uncertified_periodic_poles_are_refused() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    assert!(
        integrate_definite(
            &eval_pow(xe.clone(), q(1, 2)),
            &x,
            &Expr::from_i64(-1),
            &Expr::from_i64(1),
        )
        .is_err()
    );
    let shifted_sine = function("sin", Expr::from_i64(3) * xe.clone() - Expr::from_i64(1));
    assert!(
        integrate_definite(
            &eval_pow(shifted_sine, Expr::from_i64(-2)),
            &x,
            &Expr::from_i64(0),
            &Expr::from_i64(1),
        )
        .is_err()
    );
    let distant_pole =
        function("exp", xe.clone()) * eval_pow(xe - Expr::from_i64(100), Expr::from_i64(-2));
    assert!(
        integrate_definite(
            &distant_pole,
            &x,
            &Expr::from_i64(0),
            &Expr::Const(Constant::Infinity),
        )
        .is_err()
    );
}

#[test]
fn indeterminate_improper_endpoint_differences_are_refused() {
    let x = Symbol::new("x");
    assert!(
        integrate_definite(
            &Expr::Sym(x.clone()),
            &x,
            &Expr::Const(Constant::NegativeInfinity),
            &Expr::Const(Constant::Infinity),
        )
        .is_err()
    );
}
