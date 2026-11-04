//! Exact radical normalization before quadratic roots enter nested expressions.

use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_solvers::{solve_poly, solve_quadratic};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn radical(value: i64) -> Expr {
    Expr::Pow(
        Arc::new(Expr::from_i64(value)),
        Arc::new(Expr::rational(1, 2).unwrap()),
    )
}

fn check_pair(coefficients: [BigRational; 3], expected: Vec<Expr>) {
    let x = Symbol::new("quadratic_radical_x");
    let polynomial = UnivariatePoly::new(x.clone(), coefficients.to_vec());
    let roots = solve_poly(&polynomial).unwrap();
    assert_eq!(roots, expected);
    let variable = Expr::Sym(x.clone());
    let source = (rational_expr(coefficients[2].clone()) * variable.clone()
        + rational_expr(coefficients[1].clone()))
        * variable
        + rational_expr(coefficients[0].clone());
    assert_eq!(solve_quadratic(&source, &x).unwrap(), roots);
    for root in roots {
        // Independent polynomial substitution checks both signs exactly.
        let residual = (rational_expr(coefficients[2].clone()) * root.clone()
            + rational_expr(coefficients[1].clone()))
            * root
            + rational_expr(coefficients[0].clone());
        let expanded = fsym_simplify::try_expand(&residual).unwrap();
        assert!(fsym_simplify::try_simplify(&expanded).unwrap().is_zero());
    }
}

#[test]
fn common_square_factors_cancel_before_gram_roots_are_published() {
    let center = Expr::from_i64(15);
    let offset = radical(221);
    let plus = center.clone() + offset.clone();
    let minus = center - offset;
    // The 1/12-scaled polynomial has discriminant 221/36; negative scaling
    // reverses the quadratic formula's sign order without changing its root set.
    for scale in [q(1, 1), q(1, 12), q(7, 5)] {
        check_pair(
            [q(4, 1) * &scale, q(-30, 1) * &scale, scale],
            vec![plus.clone(), minus.clone()],
        );
    }
    check_pair([q(-4, 1), q(30, 1), q(-1, 1)], vec![minus, plus]);
}

#[test]
fn nonsquare_rational_discriminants_keep_exact_denominators() {
    let center = Expr::rational(1, 3).unwrap();
    let offset = Expr::rational(1, 3).unwrap() * radical(2);
    check_pair(
        [q(-1, 9), q(-2, 3), q(1, 1)],
        vec![center.clone() + offset.clone(), center - offset],
    );
    check_pair([q(-2, 1), q(0, 1), q(1, 1)], vec![radical(2), -radical(2)]);
}

#[test]
fn bounded_factor_extraction_leaves_an_exact_unfactored_remainder() {
    // 10009 exceeds the core square-factor trial bound. No unbounded
    // factorization is needed to return and check these exact roots.
    let inside = 2 * 10_009_i64 * 10_009;
    check_pair(
        [q(-inside, 1), q(0, 1), q(1, 1)],
        vec![radical(inside), -radical(inside)],
    );
}

#[test]
fn exact_root_multiplicity_and_complex_branch_are_preserved() {
    let x = Symbol::new("quadratic_radical_x");
    let repeated = UnivariatePoly::new(x.clone(), vec![q(4, 1), q(-4, 1), q(1, 1)]);
    assert_eq!(
        solve_poly(&repeated).unwrap(),
        vec![Expr::from_i64(2), Expr::from_i64(2)]
    );
    let complex = UnivariatePoly::new(x, vec![q(1, 1), q(0, 1), q(1, 1)]);
    let roots = solve_poly(&complex).unwrap();
    // Preserve the full typed shape of the existing complex formula,
    // including its held quadratic numerator and reciprocal denominator.
    let imaginary_discriminant = Expr::Pow(
        Arc::new(Expr::Rational(q(-4, 1))),
        Arc::new(Expr::Rational(q(1, 2))),
    );
    let reciprocal_denominator = Expr::Pow(
        Arc::new(Expr::Rational(q(2, 1))),
        Arc::new(Expr::from_i64(-1)),
    );
    assert_eq!(
        roots,
        vec![
            Expr::Mul(vec![
                Expr::Add(vec![
                    Expr::Rational(q(0, 1)),
                    imaginary_discriminant.clone(),
                ]),
                reciprocal_denominator.clone(),
            ]),
            Expr::Mul(vec![
                Expr::Add(vec![
                    Expr::Rational(q(0, 1)),
                    Expr::Mul(vec![Expr::from_i64(-1), imaginary_discriminant]),
                ]),
                reciprocal_denominator,
            ]),
        ]
    );
    let close = BigRational::new(BigInt::from(1), BigInt::from(1) << 80u32);
    let a = q(1, 1);
    let b = -(q(2, 1) + &close);
    let c = q(1, 1) + &close;
    let polynomial = UnivariatePoly::new(Symbol::new("x"), vec![c, b, a]);
    assert_eq!(
        solve_poly(&polynomial).unwrap(),
        vec![Expr::from_i64(1), rational_expr(q(1, 1) + close)]
    );
}
