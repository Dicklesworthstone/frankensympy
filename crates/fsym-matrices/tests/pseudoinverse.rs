//! Exact values and adversarial Moore-Penrose identities, independent of rank-factorization details.

use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    Matrix, MatrixError, PseudoinverseCertificate, verify_pseudoinverse_certificate,
};

fn integer_matrix(rows: usize, cols: usize, entries: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        cols,
        entries.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn rational_matrix(rows: usize, cols: usize, entries: &[(i64, i64)]) -> Matrix {
    Matrix::new(
        rows,
        cols,
        entries
            .iter()
            .map(|&(numerator, denominator)| Expr::rational(numerator, denominator).unwrap())
            .collect(),
    )
    .unwrap()
}

#[test]
fn singular_square_has_an_exact_pseudoinverse() {
    let a = integer_matrix(2, 2, &[1, 2, 2, 4]);
    let expected = rational_matrix(2, 2, &[(1, 25), (2, 25), (2, 25), (4, 25)]);
    assert_eq!(a.pseudoinverse().unwrap(), expected);
}

#[test]
fn rectangular_rank_one_and_transpose_match_closed_form() {
    // A = u v^T, with u = [1, 2, 3]^T and v = [2, -1]^T.
    // A+ = v u^T / ((u^T u) (v^T v)).
    let a = integer_matrix(3, 2, &[2, -1, 4, -2, 6, -3]);
    let expected = rational_matrix(
        2,
        3,
        &[(1, 35), (2, 35), (3, 35), (-1, 70), (-1, 35), (-3, 70)],
    );
    assert_eq!(a.pseudoinverse().unwrap(), expected);
    assert_eq!(a.transpose().pseudoinverse().unwrap(), expected.transpose());
}

#[test]
fn rank_two_singular_matrix_matches_exact_fixture() {
    // Independently captured with the pinned SymPy 1.14.0 oracle.
    let a = integer_matrix(3, 3, &[1, 2, 3, 2, 4, 6, 0, 1, 1]);
    let expected = rational_matrix(
        3,
        3,
        &[
            (2, 15),
            (4, 15),
            (-5, 3),
            (-1, 15),
            (-2, 15),
            (4, 3),
            (1, 15),
            (2, 15),
            (-1, 3),
        ],
    );
    assert_eq!(a.pseudoinverse().unwrap(), expected);
}

#[test]
fn full_rank_rectangular_behavior_is_preserved() {
    let a = integer_matrix(3, 2, &[1, 2, 3, 4, 5, 6]);
    let expected = rational_matrix(
        2,
        3,
        &[(-4, 3), (-1, 3), (2, 3), (13, 12), (1, 3), (-5, 12)],
    );
    assert_eq!(a.pseudoinverse().unwrap(), expected);
}

#[test]
fn exhaustive_small_matrices_match_an_independent_closed_form() {
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    let matrix = integer_matrix(2, 2, &[a, b, c, d]);
                    let determinant = a * d - b * c;
                    let squared_norm = a * a + b * b + c * c + d * d;
                    let expected = if determinant != 0 {
                        rational_matrix(
                            2,
                            2,
                            &[
                                (d, determinant),
                                (-b, determinant),
                                (-c, determinant),
                                (a, determinant),
                            ],
                        )
                    } else if squared_norm != 0 {
                        // Every nonzero singular 2x2 matrix has rank one.
                        rational_matrix(
                            2,
                            2,
                            &[
                                (a, squared_norm),
                                (c, squared_norm),
                                (b, squared_norm),
                                (d, squared_norm),
                            ],
                        )
                    } else {
                        Matrix::zeros(2, 2).unwrap()
                    };
                    assert_eq!(matrix.pseudoinverse().unwrap(), expected, "{a},{b},{c},{d}");
                }
            }
        }
    }
}

#[test]
fn exact_scaling_and_involution_hold_for_rational_entries() {
    let a = rational_matrix(3, 2, &[(1, 3), (2, 7), (2, 3), (4, 7), (1, 1), (6, 7)]);
    let inverse = a.pseudoinverse().unwrap();
    assert_eq!(inverse.pseudoinverse().unwrap(), a);
    let scale = Expr::rational(-3, 7).unwrap();
    let inverse_scale = Expr::rational(-7, 3).unwrap();
    assert_eq!(
        a.scalar_mul(&scale).unwrap().pseudoinverse().unwrap(),
        inverse.scalar_mul(&inverse_scale).unwrap(),
    );
}

#[test]
fn zero_rank_and_zero_area_shapes_are_preserved() {
    for (rows, cols) in [(0, 0), (0, 3), (4, 0), (2, 3), (0, 262_144)] {
        let a = Matrix::zeros(rows, cols).unwrap();
        let inverse = a.pseudoinverse().unwrap();
        assert_eq!(inverse.rows(), cols);
        assert_eq!(inverse.cols(), rows);
        assert_eq!(inverse, Matrix::zeros(cols, rows).unwrap());
        verify_pseudoinverse_certificate(&a, &PseudoinverseCertificate { inverse }).unwrap();
    }
}

#[test]
fn singular_diagonal_beyond_old_determinant_limit() {
    let diagonal: Vec<_> = (0..12).map(Expr::from_i64).collect();
    let expected_diagonal: Vec<_> = (0..12)
        .map(|value| {
            if value == 0 {
                Expr::from_i64(0)
            } else {
                Expr::rational(1, value).unwrap()
            }
        })
        .collect();
    let a = Matrix::diag(diagonal).unwrap();
    assert_eq!(
        a.pseudoinverse().unwrap(),
        Matrix::diag(expected_diagonal).unwrap()
    );
}

#[test]
fn pseudoinverse_gives_the_minimum_norm_least_squares_solution() {
    let a = integer_matrix(2, 2, &[1, 2, 2, 4]);
    let b = integer_matrix(2, 1, &[3, 0]);
    let solution = a.pseudoinverse().unwrap().matmul(&b).unwrap();
    assert_eq!(solution, rational_matrix(2, 1, &[(3, 25), (6, 25)]));
    let residual = a.matmul(&solution).unwrap().sub(&b).unwrap();
    assert_eq!(
        a.transpose().matmul(&residual).unwrap(),
        Matrix::zeros(2, 1).unwrap()
    );
    let kernel_vector = integer_matrix(1, 2, &[-2, 1]);
    assert_eq!(
        kernel_vector.matmul(&solution).unwrap(),
        Matrix::zeros(1, 1).unwrap()
    );
}

#[test]
fn every_penrose_identity_is_required() {
    let a = integer_matrix(2, 2, &[1, 0, 0, 0]);
    for (entries, rejected_identity) in [
        ([0, 0, 0, 0], "A X A != A"),
        ([1, 0, 0, 1], "X A X != X"),
        ([1, 1, 0, 0], "A X is not symmetric"),
        ([1, 0, 1, 0], "X A is not symmetric"),
    ] {
        let cert = PseudoinverseCertificate {
            inverse: integer_matrix(2, 2, &entries),
        };
        assert_eq!(
            verify_pseudoinverse_certificate(&a, &cert).unwrap_err(),
            MatrixError::InvalidCertificate(rejected_identity.to_string()),
        );
    }
}

#[test]
fn tiny_exact_perturbations_are_rejected_without_tolerances() {
    let a = Matrix::eye(1).unwrap();
    let denominator = BigInt::from(10).pow(80);
    let perturbed = Expr::Rational(BigRational::new(
        &denominator + BigInt::from(1),
        denominator,
    ));
    let cert = PseudoinverseCertificate {
        inverse: Matrix::new(1, 1, vec![perturbed]).unwrap(),
    };
    assert!(matches!(
        verify_pseudoinverse_certificate(&a, &cert),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn malformed_shape_symbolic_domain_and_zero_mutants_are_rejected() {
    let a = Matrix::eye(2).unwrap();
    let wrong_shape = PseudoinverseCertificate {
        inverse: Matrix::zeros(1, 2).unwrap(),
    };
    assert!(matches!(
        verify_pseudoinverse_certificate(&a, &wrong_shape),
        Err(MatrixError::ShapeMismatch(..))
    ));
    let symbolic = Matrix::new(1, 1, vec![Expr::symbol("x")]).unwrap();
    assert_eq!(
        symbolic.pseudoinverse().unwrap_err(),
        MatrixError::UnsupportedCertificateDomain
    );
    assert_eq!(
        verify_pseudoinverse_certificate(
            &Matrix::eye(1).unwrap(),
            &PseudoinverseCertificate { inverse: symbolic }
        )
        .unwrap_err(),
        MatrixError::UnsupportedCertificateDomain,
    );
    let zero = Matrix::zeros(2, 2).unwrap();
    assert!(matches!(
        verify_pseudoinverse_certificate(&zero, &PseudoinverseCertificate { inverse: a }),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn reference_verifier_refuses_work_beyond_its_bound() {
    let a = Matrix::new(2000, 2, vec![Expr::from_i64(1); 4000]).unwrap();
    let inverse = Matrix::new(2, 2000, vec![Expr::rational(1, 4000).unwrap(); 4000]).unwrap();
    assert!(matches!(
        verify_pseudoinverse_certificate(&a, &PseudoinverseCertificate { inverse }),
        Err(MatrixError::ResourceLimit(_)),
    ));
}
