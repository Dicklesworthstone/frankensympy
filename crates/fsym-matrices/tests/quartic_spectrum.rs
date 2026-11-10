//! General quartic roots and multiplicities through the native matrix API.

use fsym_budget::Unbounded;
use fsym_core::{BigRational, Expr, Symbol, real_roots::isolate_real_roots};
use fsym_matrices::Matrix;
use fsym_polys::UnivariatePoly;
use fsym_solvers::univariate::poly_roots;

fn polynomial(coefficients: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(
        Symbol::new("lambda"),
        coefficients
            .iter()
            .map(|&value| BigRational::from_integer(value.into()))
            .collect(),
    )
}

fn quartic_companion() -> Matrix {
    Matrix::new(
        4,
        4,
        [0, 0, 0, -1, 1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0]
            .into_iter()
            .map(Expr::from_i64)
            .collect(),
    )
    .unwrap()
}

#[test]
fn companion_returns_the_complete_nonreal_quartic_spectrum() {
    let matrix = quartic_companion();
    // Supply the source coefficients independently of the matrix calculation.
    let source = polynomial(&[1, 1, 0, 0, 1]);
    assert_eq!(
        matrix.char_poly_with_certificate("lambda").unwrap().poly,
        source
    );

    // An independent exact Sturm check establishes that all four roots are
    // nonreal; approximate residuals are not used to admit this spectrum.
    let real_roots = isolate_real_roots(&source.coeffs, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(real_roots.count_with_multiplicity(), 0);

    let expected = poly_roots(&source).unwrap();
    assert_eq!(expected.len(), 4);
    assert!(expected.iter().all(|(_, multiplicity)| *multiplicity == 1));
    let spectrum = matrix.eigenvalues().unwrap();
    assert_eq!(spectrum.len(), 4);
    for (root, _) in expected {
        assert_eq!(spectrum.iter().filter(|&value| value == &root).count(), 1);
    }
}

#[test]
fn repeated_companion_blocks_retain_every_algebraic_multiplicity() {
    let companion = quartic_companion();
    let mut entries = vec![Expr::from_i64(0); 8 * 8];
    for offset in [0, 4] {
        for row in 0..4 {
            for column in 0..4 {
                entries[(row + offset) * 8 + column + offset] =
                    companion.data()[row * 4 + column].clone();
            }
        }
    }
    let matrix = Matrix::new(8, 8, entries).unwrap();
    // (lambda^4 + lambda + 1)^2, independently expanded.
    let source = polynomial(&[1, 2, 1, 0, 2, 2, 0, 0, 1]);
    assert_eq!(
        matrix.char_poly_with_certificate("lambda").unwrap().poly,
        source
    );

    let distinct = poly_roots(&polynomial(&[1, 1, 0, 0, 1])).unwrap();
    assert_eq!(distinct.len(), 4);
    assert!(distinct.iter().all(|(_, multiplicity)| *multiplicity == 1));
    let spectrum = matrix.eigenvalues().unwrap();
    assert_eq!(spectrum.len(), 8);
    for (root, _) in distinct {
        assert_eq!(spectrum.iter().filter(|&value| value == &root).count(), 2);
    }
}
