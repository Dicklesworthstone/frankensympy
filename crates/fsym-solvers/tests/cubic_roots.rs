//! Regressions for complete exact cubic solutions through the public APIs.

use fsym_core::elementary::{eval_pow, rational_expr};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_solvers::univariate::{poly_roots, solve_univariate};

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn shifted_cube(shift: BigRational, constant: BigRational, scale: BigRational) -> UnivariatePoly {
    // scale * ((x - shift)^3 + constant).
    UnivariatePoly::new(
        Symbol::new("x"),
        vec![
            scale.clone() * (constant - shift.clone() * shift.clone() * shift.clone()),
            scale.clone() * rational(3, 1) * shift.clone() * shift.clone(),
            scale.clone() * rational(-3, 1) * shift,
            scale,
        ],
    )
}

#[test]
fn shifted_pure_cubics_keep_all_three_roots_and_the_exact_real_root() {
    for (shift, constant, scale) in [
        (rational(-1, 1), rational(2, 1), rational(1, 1)),
        (rational(2, 1), rational(2, 1), rational(-3, 1)),
        (rational(1, 2), rational(2, 1), rational(5, 7)),
        (rational(-1, 1), rational(-2, 1), rational(1, 1)),
    ] {
        let polynomial = shifted_cube(shift.clone(), constant.clone(), scale);
        let x = Symbol::new("x");
        let roots = solve_univariate(&polynomial.to_expr(), &x).unwrap();
        assert_eq!(roots.len(), 3, "{polynomial:?}");
        assert!(!roots.contains(&rational_expr(shift.clone())));

        let magnitude = eval_pow(Expr::from_i64(2), Expr::Rational(rational(1, 3)));
        let expected_real = rational_expr(shift)
            + if constant > rational(0, 1) {
                -magnitude
            } else {
                magnitude
            };
        assert!(roots.contains(&expected_real), "{roots:?}");

        // Evaluate each reported root directly in the ORIGINAL cubic,
        // independently of the generator's change-of-variable parameters.
        for root in &roots {
            let residual = polynomial
                .coeffs
                .iter()
                .rev()
                .fold(Expr::from_i64(0), |acc, coefficient| {
                    acc * root.clone() + Expr::Rational(coefficient.clone())
                });
            let expanded = fsym_simplify::try_expand(&residual).unwrap();
            assert!(
                fsym_simplify::try_simplify(&expanded).unwrap().is_zero(),
                "residual={expanded}"
            );
        }
    }
}

#[test]
fn repeated_cubic_factors_preserve_root_multiplicities() {
    let cubic = shifted_cube(rational(-1, 1), rational(2, 1), rational(1, 1));
    let repeated = cubic.mul(&cubic).unwrap();
    let roots = poly_roots(&repeated).unwrap();
    assert_eq!(roots.len(), 3);
    assert!(roots.iter().all(|(_, multiplicity)| *multiplicity == 2));
    assert_eq!(
        roots
            .iter()
            .map(|(_, multiplicity)| *multiplicity)
            .sum::<usize>(),
        6
    );
}

#[test]
fn rational_equation_still_excludes_an_original_cubic_pole() {
    let cubic = shifted_cube(rational(-1, 1), rational(2, 1), rational(1, 1));
    let x = Symbol::new("x");
    let numerator = cubic.to_expr() * (Expr::Sym(x.clone()) - Expr::from_i64(4));
    let reciprocal = Expr::Pow(
        std::sync::Arc::new(cubic.to_expr()),
        std::sync::Arc::new(Expr::Integer(BigInt::from(-1))),
    );
    assert_eq!(
        solve_univariate(&(numerator * reciprocal), &x).unwrap(),
        vec![Expr::from_i64(4)]
    );
}
