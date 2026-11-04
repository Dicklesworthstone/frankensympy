//! Complete unique rectangular solves reuse the independent affine checker.

use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{Matrix, MatrixError};

fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        values.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

#[test]
fn tall_systems_with_row_pivoting_and_multiple_rhs_have_exact_solutions() {
    let a = matrix(4, 2, &[0, 2, 3, 1, 3, 3, -3, 1]);
    let expected = matrix(2, 3, &[2, -1, 4, 3, 5, -2]);
    let rhs = matrix(4, 3, &[6, 10, -4, 9, 2, 10, 15, 12, 6, -3, 8, -14]);
    let solution = a.solve(&rhs).unwrap();
    assert_eq!(solution, expected);
    assert_eq!(a.matmul(&solution).unwrap(), rhs);
}

#[test]
fn small_tall_systems_agree_with_independent_two_by_two_determinants() {
    for a in -1i64..=1 {
        for b in -1i64..=1 {
            for c in -1i64..=1 {
                for d in -1i64..=1 {
                    let coefficients = matrix(3, 2, &[a, b, c, d, a + c, b + d]);
                    // A * [2, -3], computed directly from the original rows.
                    let rhs = matrix(
                        3,
                        1,
                        &[2 * a - 3 * b, 2 * c - 3 * d, 2 * (a + c) - 3 * (b + d)],
                    );
                    if a * d - b * c != 0 {
                        assert_eq!(coefficients.solve(&rhs).unwrap(), matrix(2, 1, &[2, -3]));
                    } else {
                        assert_eq!(coefficients.solve(&rhs), Err(MatrixError::SingularMatrix));
                    }
                }
            }
        }
    }
}

#[test]
fn tiny_exact_pivots_and_inconsistencies_are_not_rounded_away() {
    let tiny = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(10).pow(80)));
    let a = Matrix::new(
        3,
        2,
        vec![
            tiny.clone(),
            Expr::from_i64(0),
            Expr::from_i64(0),
            Expr::from_i64(1),
            tiny.clone(),
            Expr::from_i64(1),
        ],
    )
    .unwrap();
    let first = Expr::from_i64(2) * tiny.clone();
    let b = Matrix::new(
        3,
        1,
        vec![
            first.clone(),
            Expr::from_i64(-3),
            first.clone() - Expr::from_i64(3),
        ],
    )
    .unwrap();
    assert_eq!(a.solve(&b).unwrap(), matrix(2, 1, &[2, -3]));
    let inconsistent = Matrix::new(
        3,
        1,
        vec![
            first.clone(),
            Expr::from_i64(-3),
            first + tiny - Expr::from_i64(3),
        ],
    )
    .unwrap();
    assert_eq!(
        a.solve(&inconsistent),
        Err(MatrixError::InconsistentLinearSystem)
    );
}

#[test]
fn inconsistent_and_parametric_outcomes_are_distinct_refusals() {
    let tall = matrix(3, 2, &[1, 1, 2, 2, 3, 3]);
    assert_eq!(
        tall.solve(&matrix(3, 1, &[1, 2, 3])),
        Err(MatrixError::SingularMatrix)
    );
    assert_eq!(
        tall.solve(&matrix(3, 1, &[1, 2, 4])),
        Err(MatrixError::InconsistentLinearSystem)
    );
    let wide = matrix(1, 3, &[1, 2, 3]);
    assert_eq!(
        wide.solve(&matrix(1, 1, &[1])),
        Err(MatrixError::SingularMatrix)
    );
    assert_eq!(
        matrix(1, 3, &[0, 0, 0]).solve(&matrix(1, 1, &[1])),
        Err(MatrixError::InconsistentLinearSystem)
    );
}

#[test]
fn a_later_inconsistent_rhs_cannot_publish_a_partial_result() {
    let a = matrix(3, 2, &[1, 0, 0, 1, 1, 1]);
    let b = matrix(3, 2, &[2, 3, 4, 5, 6, 9]);
    assert_eq!(a.solve(&b), Err(MatrixError::InconsistentLinearSystem));
}

#[test]
fn zero_area_systems_preserve_unique_empty_vectors_and_rank_checks() {
    let no_variables = Matrix::zeros(3, 0).unwrap();
    assert_eq!(
        no_variables.solve(&Matrix::zeros(3, 2).unwrap()).unwrap(),
        Matrix::zeros(0, 2).unwrap()
    );
    assert_eq!(
        no_variables.solve(&Matrix::zeros(3, 0).unwrap()).unwrap(),
        Matrix::zeros(0, 0).unwrap()
    );
    assert_eq!(
        no_variables.solve(&matrix(3, 1, &[0, 0, 1])),
        Err(MatrixError::InconsistentLinearSystem)
    );
    assert_eq!(
        Matrix::zeros(0, 2)
            .unwrap()
            .solve(&Matrix::zeros(0, 1).unwrap()),
        Err(MatrixError::SingularMatrix)
    );
    let full_rank = matrix(3, 2, &[1, 0, 0, 1, 1, 1]);
    assert_eq!(
        full_rank.solve(&Matrix::zeros(3, 0).unwrap()).unwrap(),
        Matrix::zeros(2, 0).unwrap()
    );
    assert_eq!(
        Matrix::zeros(3, 2)
            .unwrap()
            .solve(&Matrix::zeros(3, 0).unwrap()),
        Err(MatrixError::SingularMatrix)
    );
}

#[test]
fn original_domains_shapes_and_aggregate_bounds_are_checked() {
    let a = matrix(3, 2, &[1, 0, 0, 1, 1, 1]);
    assert!(matches!(
        a.solve(&Matrix::zeros(2, 1).unwrap()),
        Err(MatrixError::ShapeMismatch(..))
    ));
    for entry in [
        Expr::symbol("parameter"),
        Expr::Const(Constant::I),
        Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")]),
    ] {
        let rhs = Matrix::new(
            3,
            2,
            vec![
                Expr::from_i64(0),
                Expr::from_i64(0),
                Expr::from_i64(0),
                Expr::from_i64(0),
                Expr::from_i64(0),
                entry,
            ],
        )
        .unwrap();
        assert_eq!(
            Matrix::zeros(3, 2).unwrap().solve(&rhs),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }
    let huge = Expr::Integer(BigInt::from(1) << 8192u32);
    let rhs = Matrix::new(3, 1, vec![Expr::from_i64(0), Expr::from_i64(0), huge]).unwrap();
    assert!(matches!(a.solve(&rhs), Err(MatrixError::ResourceLimit(_))));
    assert!(matches!(
        Matrix::zeros(129, 0)
            .unwrap()
            .solve(&Matrix::zeros(129, 1).unwrap()),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        a.solve(&Matrix::zeros(3, 129).unwrap()),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::zeros(65, 64)
            .unwrap()
            .solve(&Matrix::zeros(65, 16).unwrap()),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn established_square_solver_does_not_inherit_rectangular_dimension_caps() {
    let a = Matrix::eye(129).unwrap();
    let rhs = Matrix::new(129, 1, (0..129).map(Expr::from_i64).collect()).unwrap();
    assert_eq!(a.solve(&rhs).unwrap(), rhs);
    assert_eq!(
        matrix(2, 2, &[1, 2, 2, 4]).solve(&matrix(2, 1, &[1, 3])),
        Err(MatrixError::SingularMatrix)
    );
}
