//! Exact quartic solving through the native public root API.

use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_solvers::SolverError;
use fsym_solvers::univariate::{poly_roots, solve_univariate};

fn polynomial(coefficients: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(
        Symbol::new("x"),
        coefficients
            .iter()
            .map(|&c| BigRational::from_integer(c.into()))
            .collect(),
    )
}

#[test]
fn general_quartics_return_all_four_roots() {
    // Respectively zero, two, four, zero, zero, two, and two real roots. These are
    // separate from the exact publication check and exercise the returned
    // expressions through an independent approximate evaluator as well.
    for (coefficients, real_roots) in [
        (&[1, 1, 0, 0, 1][..], 0),
        (&[-1, 1, 0, 0, 1][..], 2),
        (&[1, 1, -5, 0, 1][..], 4),
        (&[1, 2, 3, 4, 5][..], 0),
        (&[3, 4, 6, 4, 1][..], 0),
        // Resolvents (u-4)^3+63 and (u+4)^3-65 exercise both
        // Cardano pure-cubic signs, including its zero plus-radicand.
        (&[-3, 1, -6, 0, 1][..], 2),
        (&[-3, 1, 6, 0, 1][..], 2),
    ] {
        let polynomial = polynomial(coefficients);
        let roots =
            poly_roots(&polynomial).unwrap_or_else(|error| panic!("{coefficients:?}: {error}"));
        assert_eq!(
            roots
                .iter()
                .map(|(_, multiplicity)| multiplicity)
                .sum::<usize>(),
            4
        );
        let approximate = roots
            .iter()
            .map(|(root, _)| {
                fsym_calculus::gruntz::complex_value(root).expect("finite algebraic root")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            approximate
                .iter()
                .filter(|root| root.im.abs() < 1e-7)
                .count(),
            real_roots
        );
        for root in approximate {
            let mut residual = root * 0.0;
            let mut scale = 0.0;
            for &coefficient in coefficients.iter().rev() {
                residual = residual * root + coefficient as f64;
                scale = scale * root.norm() + (coefficient as f64).abs();
            }
            assert!(
                residual.norm() < 1e-8 * scale.max(1.0),
                "{coefficients:?}: {root:?}, residual {residual:?}"
            );
        }
    }
}

#[test]
fn nonmonic_rational_scaling_preserves_the_complete_roots() {
    let source = polynomial(&[1, 2, 3, 4, 5]);
    let scaled = UnivariatePoly::new(
        source.gen_sym.clone(),
        source
            .coeffs
            .iter()
            .map(|c| c * BigRational::new((-7).into(), 11.into()))
            .collect(),
    );
    assert_eq!(poly_roots(&source).unwrap(), poly_roots(&scaled).unwrap());
}

#[test]
fn repeated_irreducible_quartics_preserve_all_multiplicities() {
    let quartic = polynomial(&[1, 1, 0, 0, 1]);
    let linear = polynomial(&[-2, 1]);
    let repeated = quartic
        .mul(&quartic)
        .unwrap()
        .mul(&linear.pow(3).unwrap())
        .unwrap();
    let roots = poly_roots(&repeated).unwrap();
    assert_eq!(roots.len(), 5);
    assert!(
        roots
            .iter()
            .any(|(root, multiplicity)| *root == Expr::from_i64(2) && *multiplicity == 3)
    );
    assert_eq!(
        roots
            .iter()
            .filter(|(_, multiplicity)| *multiplicity == 2)
            .count(),
        4
    );
}

#[test]
fn public_equation_solver_exposes_general_quartics() {
    let quartic = polynomial(&[-1, 1, 0, 0, 1]);
    let roots = solve_univariate(&quartic.to_expr(), &quartic.gen_sym).unwrap();
    assert_eq!(roots.len(), 4);
}

#[test]
fn original_quartic_denominator_holes_are_still_excluded() {
    let quartic = polynomial(&[1, 1, 0, 0, 1]);
    let x = Expr::Sym(quartic.gen_sym.clone());
    let numerator = quartic.to_expr() * (x - Expr::from_i64(3));
    let reciprocal = Expr::Pow(
        std::sync::Arc::new(quartic.to_expr()),
        std::sync::Arc::new(Expr::from_i64(-1)),
    );
    assert_eq!(
        solve_univariate(&(numerator * reciprocal), &quartic.gen_sym).unwrap(),
        vec![Expr::from_i64(3)]
    );
    // A simplified P²/P would leave P's roots. The original held quotient
    // excludes all of them, including the numerator's repeated factors.
    let held = Expr::Mul(vec![
        Expr::Pow(
            std::sync::Arc::new(quartic.to_expr()),
            std::sync::Arc::new(Expr::from_i64(2)),
        ),
        Expr::Pow(
            std::sync::Arc::new(quartic.to_expr()),
            std::sync::Arc::new(Expr::from_i64(-1)),
        ),
    ]);
    assert!(
        solve_univariate(&held, &quartic.gen_sym)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn irreducible_quintic_is_not_mistaken_for_a_quartic() {
    let quintic = polynomial(&[-1, -1, 0, 0, 0, 1]);
    assert!(matches!(
        poly_roots(&quintic),
        Err(SolverError::UnsupportedDegree(5))
    ));
}
