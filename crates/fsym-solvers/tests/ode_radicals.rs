//! Exact radical ODE families, basis independence, and domain rejection.

use fsym_calculus::diff;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_simplify::{try_expand, try_simplify};
use fsym_solvers::{
    dsolve_cauchy_euler, dsolve_const_coeff_second_order,
    dsolve_const_coeff_second_order_nonhomogeneous, verify_cauchy_euler_solution,
    verify_const_coeff_second_order_nonhomogeneous_solution,
    verify_const_coeff_second_order_solution,
};
use std::collections::HashMap;

fn rational(numerator: BigInt, denominator: BigInt) -> Expr {
    fsym_core::elementary::rational_expr(BigRational::new(numerator, denominator))
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.to_string(), vec![argument])
}

fn sqrt_two() -> Expr {
    Expr::pow(Expr::from_i64(2), rational(1.into(), 2.into()))
}

// Extract the actual generated basis, differentiate it, and check its exact
// Wronskian at an ordinary point. A duplicated or missing basis must fail even
// when the residual checker accepts each individual solution.
fn wronskian_square(family: &Expr, x: &Symbol, c1: &Symbol, c2: &Symbol, at: i64) -> Expr {
    let first = diff(family, c1);
    let second = diff(family, c2);
    let determinant = Expr::Add(vec![
        Expr::Mul(vec![first.clone(), diff(&second, x)]),
        Expr::Mul(vec![Expr::from_i64(-1), diff(&first, x), second]),
    ]);
    let determinant = determinant.subs(&HashMap::from([(x.clone(), Expr::from_i64(at))]));
    let determinant = try_simplify(&try_expand(&determinant).unwrap()).unwrap();
    // Multiply explicitly: a held Pow(Mul(...), 2) need not be distributed by
    // the simplifier. This still checks W*W against the independently derived
    // exact rational value, without sampling or approximate comparisons.
    let squared = Expr::Mul(vec![determinant.clone(), determinant]);
    try_simplify(&try_expand(&squared).unwrap()).unwrap()
}

fn expected_wronskian_square(a: i64, b: i64, c: i64, euler: bool) -> Expr {
    let a = BigInt::from(a);
    let b = if euler {
        BigInt::from(b) - &a
    } else {
        BigInt::from(b)
    };
    let discriminant = &b * &b - BigInt::from(4) * &a * BigInt::from(c);
    if discriminant.is_positive() {
        rational(discriminant, &a * &a)
    } else {
        rational(-discriminant, BigInt::from(4) * &a * &a)
    }
}

#[test]
fn hand_constructed_irrational_solutions_have_exact_residuals() {
    let x = Symbol::new("x");
    let argument = Expr::Mul(vec![sqrt_two(), Expr::Sym(x.clone())]);
    let exponential = function("exp", argument.clone());
    assert!(verify_const_coeff_second_order_solution(
        &exponential,
        1,
        0,
        -2,
        &x
    ));
    assert!(!verify_const_coeff_second_order_solution(
        &exponential,
        1,
        0,
        -3,
        &x
    ));
    for name in ["sin", "cos"] {
        let solution = function(name, argument.clone());
        assert!(verify_const_coeff_second_order_solution(
            &solution, 1, 0, 2, &x
        ));
        assert!(!verify_const_coeff_second_order_solution(
            &solution, 1, 0, 3, &x
        ));
    }
    let perturbation = rational(1.into(), BigInt::from(10).pow(30));
    let wrong = Expr::Add(vec![exponential, perturbation]);
    assert!(!verify_const_coeff_second_order_solution(
        &wrong, 1, 0, -2, &x
    ));
}

#[test]
fn generated_irrational_families_are_independent_and_support_forcing() {
    let x = Symbol::new("x");
    let c1 = Symbol::new("C1");
    let c2 = Symbol::new("C2");
    for (a, b, c) in [(1, 0, -2), (1, 0, 2), (2, -3, -1), (-2, 3, 1), (1, 2, 3)] {
        let family = dsolve_const_coeff_second_order(a, b, c, &x, &c1, &c2).unwrap();
        assert!(family.free_symbols().contains(&c1));
        assert!(family.free_symbols().contains(&c2));
        assert!(verify_const_coeff_second_order_solution(
            &family, a, b, c, &x
        ));
        assert!(!verify_const_coeff_second_order_solution(
            &family,
            a,
            b,
            c + 1,
            &x
        ));
        assert_eq!(
            wronskian_square(&family, &x, &c1, &c2, 0),
            expected_wronskian_square(a, b, c, false),
            "the generated family must contain two independent solutions for ({a}, {b}, {c})"
        );
    }
    for forcing in [
        Expr::from_i64(1),
        function("exp", Expr::Sym(x.clone())),
        function("cos", Expr::Sym(x.clone())),
    ] {
        let family =
            dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 2, &forcing, &x, &c1, &c2)
                .unwrap();
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &family, 1, 0, 2, &forcing, &x
        ));
        assert!(!verify_const_coeff_second_order_nonhomogeneous_solution(
            &family,
            1,
            0,
            2,
            &Expr::Add(vec![forcing, Expr::from_i64(1)]),
            &x,
        ));
        assert_eq!(
            wronskian_square(&family, &x, &c1, &c2, 0),
            Expr::from_i64(2)
        );
    }
}

#[test]
fn cauchy_euler_irrational_families_are_independent_on_positive_axis() {
    let x = Symbol::new("x");
    let c1 = Symbol::new("C1");
    let c2 = Symbol::new("C2");
    for (a, b, c) in [(1, 1, -2), (1, 1, 2), (2, -1, -1), (-2, 1, 1), (1, 3, 3)] {
        let family = dsolve_cauchy_euler(a, b, c, &x, &c1, &c2).unwrap();
        assert!(verify_cauchy_euler_solution(&family, a, b, c, &x));
        assert!(!verify_cauchy_euler_solution(&family, a, b, c + 1, &x));
        assert_eq!(
            wronskian_square(&family, &x, &c1, &c2, 1),
            expected_wronskian_square(a, b, c, true),
            "the Euler family must contain two independent solutions for ({a}, {b}, {c})"
        );
    }
    let power = Expr::pow(Expr::Sym(x.clone()), sqrt_two());
    assert!(verify_cauchy_euler_solution(&power, 1, 1, -2, &x));
    let oscillation = function(
        "sin",
        Expr::Mul(vec![sqrt_two(), function("log", Expr::Sym(x.clone()))]),
    );
    assert!(verify_cauchy_euler_solution(&oscillation, 1, 1, 2, &x));
}

#[test]
fn domain_checks_precede_cancellation_even_for_zero_equations() {
    let x = Symbol::new("x");
    let symbol = Expr::Sym(x.clone());
    for invalid in [
        Expr::Const(Constant::Infinity),
        Expr::Const(Constant::NaN),
        Expr::pow(Expr::from_i64(0), Expr::from_i64(-1)),
        Expr::pow(
            Expr::Add(vec![symbol.clone(), Expr::from_i64(-1)]),
            Expr::from_i64(-1),
        ),
        function("log", Expr::Mul(vec![Expr::from_i64(-1), symbol.clone()])),
        function("log", Expr::Add(vec![symbol.clone(), Expr::from_i64(-1)])),
        function("opaque", symbol.clone()),
    ] {
        assert!(!verify_cauchy_euler_solution(&invalid, 0, 0, 0, &x));
        assert!(!verify_const_coeff_second_order_solution(
            &invalid, 0, 0, 0, &x
        ));
        let hidden = Expr::Mul(vec![Expr::from_i64(0), invalid]);
        assert!(!verify_cauchy_euler_solution(&hidden, 1, 1, 0, &x));
        assert!(!verify_const_coeff_second_order_solution(
            &hidden, 1, 0, 0, &x
        ));
    }
    let reciprocal = Expr::pow(symbol, Expr::from_i64(-1));
    assert!(verify_cauchy_euler_solution(&reciprocal, 1, 2, 0, &x));
    assert!(!verify_const_coeff_second_order_solution(
        &reciprocal,
        0,
        0,
        0,
        &x
    ));
}
