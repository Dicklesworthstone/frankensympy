//! Complete spectra of rational matrices beyond the quadratic-only lane.

use fsym_core::elementary::eval_pow;
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{Matrix, MatrixError};
use std::sync::Arc;

fn integer(n: i64) -> Expr {
    Expr::from_i64(n)
}

// The companion of x^n + c[n-1]x^(n-1) + ... + c[0]. This
// construction gives an independent characteristic-polynomial fixture.
fn companion(coefficients: &[i64]) -> Matrix {
    let n = coefficients.len();
    let mut entries = vec![integer(0); n * n];
    for row in 0..n {
        entries[row * n + n - 1] = integer(-coefficients[row]);
        if row > 0 {
            entries[row * n + row - 1] = integer(1);
        }
    }
    Matrix::new(n, n, entries).unwrap()
}

fn count(roots: &[Expr], value: &Expr) -> usize {
    roots.iter().filter(|root| *root == value).count()
}

#[test]
fn repeated_and_defective_spectra_keep_algebraic_multiplicity() {
    let jordan = Matrix::new(
        4,
        4,
        [2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, -1]
            .into_iter()
            .map(integer)
            .collect(),
    )
    .unwrap();
    assert_eq!(
        jordan.eigenvalues().unwrap(),
        vec![integer(-1), integer(2), integer(2), integer(2)]
    );
    assert_eq!(
        Matrix::eye(12).unwrap().eigenvalues().unwrap(),
        vec![integer(1); 12]
    );
    assert_eq!(
        Matrix::zeros(4, 4).unwrap().eigenvalues().unwrap(),
        vec![integer(0); 4]
    );
}

#[test]
fn mixed_real_complex_factors_return_the_entire_spectrum() {
    // (x-1)^2 (x+2) (x^2+1) = x^5 - 2x^3 + 2x^2 - 3x + 2.
    let roots = companion(&[2, -3, 2, -2, 0]).eigenvalues().unwrap();
    let imaginary = Expr::Const(Constant::I);
    assert_eq!(roots.len(), 5);
    assert_eq!(count(&roots, &integer(1)), 2);
    assert_eq!(count(&roots, &integer(-2)), 1);
    assert_eq!(count(&roots, &imaginary), 1);
    assert_eq!(count(&roots, &(-imaginary)), 1);
}

#[test]
fn repeated_irrational_factors_retain_each_multiplicity() {
    // (x^2-2)^2, with no rational roots and two repeated irrational roots.
    let roots = companion(&[4, 0, -4, 0]).eigenvalues().unwrap();
    let positive = eval_pow(integer(2), Expr::rational(1, 2).unwrap());
    assert_eq!(roots.len(), 4);
    assert_eq!(count(&roots, &positive), 2);
    assert_eq!(count(&roots, &(-positive)), 2);
}

#[test]
fn cubic_characteristic_roots_satisfy_the_original_polynomial() {
    // (x+1)^3 + 2, the pure-depressed-cubic branch that previously
    // produced an incorrect repeated shift in Cardano's formula.
    let roots = companion(&[3, 3, 3]).eigenvalues().unwrap();
    assert_eq!(roots.len(), 3);
    assert_ne!(roots[0], roots[1]);
    assert_ne!(roots[0], roots[2]);
    assert_ne!(roots[1], roots[2]);
    let cube_root = eval_pow(integer(2), Expr::rational(1, 3).unwrap());
    assert!(roots.contains(&(integer(-1) - cube_root)));
    for root in roots {
        let residual =
            ((root.clone() + integer(3)) * root.clone() + integer(3)) * root + integer(3);
        let expanded = fsym_simplify::try_expand(&residual).unwrap();
        assert!(fsym_simplify::try_simplify(&expanded).unwrap().is_zero());
    }
}

#[test]
fn nearby_exact_rational_eigenvalues_are_not_merged() {
    let tiny = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(10).pow(60)));
    let values = vec![integer(0), tiny.clone(), tiny.clone() * integer(2)];
    let matrix = Matrix::diag(values.clone()).unwrap();
    let roots = matrix.eigenvalues().unwrap();
    assert_eq!(roots.len(), 3);
    for value in values {
        assert_eq!(count(&roots, &value), 1);
    }
}

#[test]
fn an_unsolved_factor_refuses_instead_of_publishing_known_roots() {
    // (x-2)(x^5-x-1). A rational eigenvalue must not escape as if it
    // constituted the complete spectrum when the irreducible quintic refuses.
    let matrix = companion(&[2, 1, -1, 0, 0, -2]);
    assert_eq!(
        matrix.eigenvalues(),
        Err(MatrixError::EigenvaluesUnsupportedDegree(5))
    );
}

#[test]
fn shape_original_domain_and_resource_limits_remain_explicit() {
    assert!(
        Matrix::zeros(0, 0)
            .unwrap()
            .eigenvalues()
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        Matrix::zeros(2, 3).unwrap().eigenvalues(),
        Err(MatrixError::NotSquare(2, 3))
    ));
    assert!(matches!(
        Matrix::eye(33).unwrap().eigenvalues(),
        Err(MatrixError::ResourceLimit(_))
    ));

    for unsupported in [
        Expr::symbol("a"),
        Expr::Const(Constant::Infinity),
        Expr::Pow(Arc::new(integer(0)), Arc::new(integer(-1))),
    ] {
        // A zero diagonal would hide these entries in the characteristic
        // polynomial of a triangular matrix without original-domain checks.
        let matrix = Matrix::new(
            3,
            3,
            vec![
                integer(0),
                unsupported,
                integer(0),
                integer(0),
                integer(0),
                integer(0),
                integer(0),
                integer(0),
                integer(0),
            ],
        )
        .unwrap();
        assert_eq!(
            matrix.eigenvalues(),
            Err(MatrixError::SymbolicCharacteristicPolynomial)
        );
    }
    let huge = Expr::Integer(BigInt::from(1) << 65_537u32);
    let matrix = Matrix::diag(vec![huge, integer(1), integer(1)]).unwrap();
    assert!(matches!(
        matrix.eigenvalues(),
        Err(MatrixError::ResourceLimit(_))
    ));
}
