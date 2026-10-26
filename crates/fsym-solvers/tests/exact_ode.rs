//! Exact potentials and independent gradient checks for polynomial one-forms.

use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::multivariate::MultivariatePoly;
use fsym_solvers::{SolverError, exact_ode_potential, verify_exact_ode_potential};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> Expr {
    rational_expr(BigRational::new(numerator.into(), denominator.into()))
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn assert_same_polynomial(actual: &Expr, expected: &Expr, x: &Symbol, y: &Symbol) {
    let generators = [x.clone(), y.clone()];
    assert_eq!(
        MultivariatePoly::from_expr(actual, &generators).unwrap(),
        MultivariatePoly::from_expr(expected, &generators).unwrap()
    );
}

fn mixed_fixture(x: &Symbol, y: &Symbol) -> (Expr, Expr, Expr) {
    let (xe, ye) = (Expr::Sym(x.clone()), Expr::Sym(y.clone()));
    // Hand-derived coefficients of F = x^2*y + x*y^3 + y^4.
    let m = Expr::from_i64(2) * xe.clone() * ye.clone() + power(ye.clone(), 3);
    let n = power(xe.clone(), 2)
        + Expr::from_i64(3) * xe.clone() * power(ye.clone(), 2)
        + Expr::from_i64(4) * power(ye.clone(), 3);
    let potential = power(xe.clone(), 2) * ye.clone() + xe * power(ye.clone(), 3) + power(ye, 4);
    (m, n, potential)
}

#[test]
fn nonlinear_mixed_form_has_the_hand_derived_implicit_potential() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (m, n, expected) = mixed_fixture(&x, &y);
    assert!(verify_exact_ode_potential(&expected, &m, &n, &x, &y));
    let actual = exact_ode_potential(&m, &n, &x, &y).unwrap();
    assert_same_polynomial(&actual, &expected, &x, &y);
}

#[test]
fn mixed_rational_coefficients_integrate_without_approximations() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (xe, ye) = (Expr::Sym(x.clone()), Expr::Sym(y.clone()));
    // F = x^3*y^2/2 + x*y/3 + y^5; neither gradient is separable.
    let m = q(3, 2) * power(xe.clone(), 2) * power(ye.clone(), 2) + q(1, 3) * ye.clone();
    let n = power(xe.clone(), 3) * ye.clone()
        + q(1, 3) * xe.clone()
        + Expr::from_i64(5) * power(ye.clone(), 4);
    let expected = q(1, 2) * power(xe.clone(), 3) * power(ye.clone(), 2)
        + q(1, 3) * xe * ye.clone()
        + power(ye, 5);
    let actual = exact_ode_potential(&m, &n, &x, &y).unwrap();
    assert_same_polynomial(&actual, &expected, &x, &y);
    assert!(verify_exact_ode_potential(&expected, &m, &n, &x, &y));
}

#[test]
fn constant_and_single_axis_forms_are_level_sets_and_zero_form_refuses() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    for (m, n) in [(2, -3), (0, 5), (7, 0)] {
        let m = Expr::from_i64(m);
        let n = Expr::from_i64(n);
        let expected = m.clone() * Expr::Sym(x.clone()) + n.clone() * Expr::Sym(y.clone());
        let actual = exact_ode_potential(&m, &n, &x, &y).unwrap();
        assert_same_polynomial(&actual, &expected, &x, &y);
    }
    let zero = Expr::from_i64(0);
    assert_eq!(
        exact_ode_potential(&zero, &zero, &x, &y),
        Err(SolverError::InfiniteSolutions)
    );
    assert!(!verify_exact_ode_potential(&zero, &zero, &zero, &x, &y));
}

#[test]
fn gradient_checker_accepts_only_an_additive_constant_not_a_tiny_wrong_gradient() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (m, n, potential) = mixed_fixture(&x, &y);
    assert!(verify_exact_ode_potential(
        &(potential.clone() + Expr::from_i64(7)),
        &m,
        &n,
        &x,
        &y,
    ));
    assert!(!verify_exact_ode_potential(
        &(Expr::from_i64(2) * potential.clone()),
        &m,
        &n,
        &x,
        &y,
    ));
    let epsilon = rational_expr(BigRational::new(BigInt::one(), BigInt::from(10).pow(80)));
    let altered_x = potential.clone() + epsilon.clone() * Expr::Sym(x.clone());
    assert!(!verify_exact_ode_potential(&altered_x, &m, &n, &x, &y));
    let altered_y = potential + epsilon.clone() * Expr::Sym(y.clone());
    assert!(!verify_exact_ode_potential(&altered_y, &m, &n, &x, &y));
    let wrong_n = n + epsilon * Expr::Sym(x.clone());
    assert!(matches!(
        exact_ode_potential(&m, &wrong_n, &x, &y),
        Err(SolverError::InvalidSystem(_))
    ));
}

#[test]
fn nonclosed_forms_and_aliased_variables_refuse() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let m = Expr::Sym(y.clone());
    let zero = Expr::from_i64(0);
    assert!(matches!(
        exact_ode_potential(&m, &zero, &x, &y),
        Err(SolverError::InvalidSystem(_))
    ));
    assert!(matches!(
        exact_ode_potential(&Expr::from_i64(1), &zero, &x, &x),
        Err(SolverError::InvalidSystem(_))
    ));
    assert!(!verify_exact_ode_potential(&m, &zero, &zero, &x, &x));
}

#[test]
fn original_nonpolynomial_and_malformed_subterms_cannot_hide_behind_zero() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let zero = Expr::from_i64(0);
    let one = Expr::from_i64(1);
    let unknown = Expr::Sym(Symbol::new("parameter"));
    for invalid in [
        unknown.clone(),
        Expr::Function("opaque".into(), vec![Expr::Sym(x.clone())]),
        Expr::Const(Constant::NaN),
        Expr::Const(Constant::Infinity),
        Expr::Add(vec![]),
        Expr::Mul(vec![]),
        power(zero.clone(), -1),
        power(Expr::Sym(x.clone()), -1),
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(q(1, 2))),
        Expr::Mul(vec![zero.clone(), unknown.clone()]),
        Expr::Mul(vec![zero.clone(), power(zero.clone(), -1)]),
        power(unknown, 0),
    ] {
        assert!(
            exact_ode_potential(&invalid, &one, &x, &y).is_err(),
            "unexpectedly admitted {invalid:?}"
        );
        assert!(!verify_exact_ode_potential(&invalid, &zero, &one, &x, &y));
    }
}

#[test]
fn preflight_rejects_nested_power_growth_depth_and_expanded_term_exhaustion() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let xe = Expr::Sym(x.clone());
    let zero = Expr::from_i64(0);
    let mut deep = xe.clone();
    for _ in 0..66 {
        deep = power(deep, 1);
    }
    for oversized in [
        // No huge integer or polynomial is constructed by these held powers.
        power(power(Expr::from_i64(2), 64), 64),
        power(power(xe.clone(), 4), 5),
        power(xe.clone(), i64::MAX),
        Expr::Integer(BigInt::one() << 4_097u32),
        deep,
        power(xe.clone() + Expr::Sym(y.clone()) + Expr::from_i64(1), 16),
    ] {
        assert!(exact_ode_potential(&oversized, &zero, &x, &y).is_err());
        assert!(!verify_exact_ode_potential(
            &oversized,
            &Expr::from_i64(1),
            &zero,
            &x,
            &y,
        ));
    }
}

#[test]
fn potential_degree_has_room_for_the_last_supported_input_power() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let m = power(Expr::Sym(x.clone()), 16);
    let n = Expr::from_i64(0);
    let expected = q(1, 17) * power(Expr::Sym(x.clone()), 17);
    let actual = exact_ode_potential(&m, &n, &x, &y).unwrap();
    assert_same_polynomial(&actual, &expected, &x, &y);
    assert!(verify_exact_ode_potential(&expected, &m, &n, &x, &y));
    assert!(exact_ode_potential(&power(Expr::Sym(x.clone()), 17), &n, &x, &y).is_err());
}
