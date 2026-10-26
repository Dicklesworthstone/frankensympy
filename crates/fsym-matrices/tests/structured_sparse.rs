//! Closed-form sparse systems, decomposition mutants and resource boundaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    MatrixError, SparseDeterminantCertificate, SparseLinearSystemCertificate, SparseLuCertificate,
    SparseMatrix, metered_verify_sparse_linear_system_certificate,
    verify_sparse_determinant_certificate, verify_sparse_linear_system_certificate,
    verify_sparse_lu_certificate,
};
use num_traits::One;
use std::collections::BTreeMap;
use std::sync::Arc;

fn integers(rows: usize, columns: usize, values: &[i64]) -> SparseMatrix {
    assert_eq!(values.len(), rows * columns);
    let entries = values
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0)
        .map(|(index, &value)| ((index / columns, index % columns), Expr::from_i64(value)))
        .collect();
    SparseMatrix::new(rows, columns, entries).unwrap()
}

fn changed(matrix: &SparseMatrix, row: usize, column: usize, value: Expr) -> SparseMatrix {
    let mut entries = matrix.entries().clone();
    entries.insert((row, column), value);
    SparseMatrix::new(matrix.rows(), matrix.cols(), entries).unwrap()
}

#[test]
fn row_pivoting_handles_zero_diagonals_and_multiple_right_hand_sides() {
    let a = integers(3, 3, &[0, 2, 1, 3, 4, 5, 6, 7, 8]);
    let b = integers(3, 2, &[3, 3, 6, 12, 12, 18]);
    let x = integers(3, 2, &[1, -1, 2, 0, -1, 3]);
    assert_eq!(a.det().unwrap(), Expr::from_i64(9));
    assert_eq!(a.solve(&b).unwrap(), x);
    let solution = a.solve_with_certificate(&b).unwrap();
    verify_sparse_linear_system_certificate(&a, &b, &solution).unwrap();
    assert_eq!(solution.factorization.row_permutation, vec![1, 0, 2]);
    // These factors were calculated by hand, independently of the generator.
    let lower = changed(
        &integers(3, 3, &[1, 0, 0, 0, 1, 0, 2, 0, 1]),
        2,
        1,
        Expr::rational(-1, 2).unwrap(),
    );
    let upper = changed(
        &integers(3, 3, &[3, 4, 5, 0, 2, 1, 0, 0, 0]),
        2,
        2,
        Expr::rational(-3, 2).unwrap(),
    );
    let manual = SparseLuCertificate {
        row_permutation: vec![1, 0, 2],
        lower,
        upper,
    };
    verify_sparse_lu_certificate(&a, &manual).unwrap();
    assert_eq!(solution.factorization, manual);
}

#[test]
fn all_small_two_by_two_matrices_match_closed_form_determinants_and_solves() {
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    let matrix = integers(2, 2, &[a, b, c, d]);
                    let det = a * d - b * c;
                    assert_eq!(matrix.det().unwrap(), Expr::from_i64(det));
                    let rhs = integers(2, 1, &[2, -3]);
                    if det == 0 {
                        assert_eq!(matrix.solve(&rhs), Err(MatrixError::SingularMatrix));
                    } else {
                        let expected = SparseMatrix::new(
                            2,
                            1,
                            BTreeMap::from([
                                ((0, 0), Expr::rational(2 * d + 3 * b, det).unwrap()),
                                ((1, 0), Expr::rational(-3 * a - 2 * c, det).unwrap()),
                            ]),
                        )
                        .unwrap();
                        assert_eq!(matrix.solve(&rhs).unwrap(), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn larger_small_determinants_match_the_six_term_formula() {
    // A fixed reproducible family includes row swaps, singular matrices and fill.
    for seed in 0..240_i64 {
        let mut state = seed + 11;
        let values: Vec<i64> = (0..9)
            .map(|_| {
                state = (state * 97 + 23) % 1021;
                state % 7 - 3
            })
            .collect();
        let [a, b, c, d, e, f, g, h, i] = values.as_slice() else {
            unreachable!()
        };
        let expected = a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h;
        let matrix = integers(3, 3, &values);
        assert_eq!(
            matrix.det().unwrap(),
            Expr::from_i64(expected),
            "seed {seed}"
        );
    }
}

#[test]
fn thousand_dimensional_banded_system_never_requires_dense_storage() {
    let n = 1_000;
    let mut entries = BTreeMap::new();
    let mut rhs = BTreeMap::new();
    for row in 0..n {
        entries.insert((row, row), Expr::from_i64(2));
        if row > 0 {
            entries.insert((row, row - 1), Expr::from_i64(-1));
        }
        if row + 1 < n {
            entries.insert((row, row + 1), Expr::from_i64(-1));
        }
        // A*[1,...,1]^T has only the two endpoint entries.
        if row == 0 || row + 1 == n {
            rhs.insert((row, 0), Expr::from_i64(1));
        }
    }
    let matrix = SparseMatrix::new(n, n, entries).unwrap();
    assert!(matches!(
        matrix.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    // det(tridiag(-1,2,-1)) = n+1, from D_n=2D_(n-1)-D_(n-2).
    assert_eq!(matrix.det().unwrap(), Expr::from_i64(1_001));
    let result = matrix
        .solve(&SparseMatrix::new(n, 1, rhs).unwrap())
        .unwrap();
    assert_eq!(result.nnz(), n);
    assert!(
        result
            .entries()
            .values()
            .all(|entry| entry == &Expr::from_i64(1))
    );
    assert_eq!(matrix.nnz(), 2_998);
}

#[test]
fn large_rational_diagonal_and_sparse_multiple_rhs_remain_exact() {
    let n = 600;
    let diagonal = Expr::rational(2, 3).unwrap();
    let matrix = SparseMatrix::new(
        n,
        n,
        (0..n).map(|row| ((row, row), diagonal.clone())).collect(),
    )
    .unwrap();
    let rhs = SparseMatrix::new(
        n,
        3,
        BTreeMap::from([
            ((0, 0), Expr::from_i64(2)),
            ((200, 1), Expr::rational(1, 3).unwrap()),
            ((599, 2), Expr::from_i64(-4)),
        ]),
    )
    .unwrap();
    let solution = matrix.solve(&rhs).unwrap();
    assert_eq!(solution.nnz(), 3);
    assert_eq!(solution.get(0, 0).unwrap(), Expr::from_i64(3));
    assert_eq!(solution.get(200, 1).unwrap(), Expr::rational(1, 2).unwrap());
    assert_eq!(solution.get(599, 2).unwrap(), Expr::from_i64(-6));
    assert_eq!(
        matrix.det().unwrap(),
        Expr::Rational(BigRational::new(
            BigInt::from(2).pow(600),
            BigInt::from(3).pow(600)
        ))
    );
}

#[test]
fn empty_and_singular_shapes_have_explicit_semantics() {
    let empty = SparseMatrix::zeros(0, 0);
    assert_eq!(empty.det().unwrap(), Expr::from_i64(1));
    assert_eq!(
        empty.solve(&SparseMatrix::zeros(0, 3)).unwrap(),
        SparseMatrix::zeros(0, 3)
    );
    assert_eq!(
        SparseMatrix::eye(3)
            .unwrap()
            .solve(&SparseMatrix::zeros(3, 0))
            .unwrap(),
        SparseMatrix::zeros(3, 0)
    );
    let singular = integers(3, 3, &[0, 1, 2, 0, 2, 4, 0, 3, 6]);
    let factors = singular.lu().unwrap();
    verify_sparse_lu_certificate(&singular, &factors).unwrap();
    assert_eq!(singular.det().unwrap(), Expr::from_i64(0));
    for rhs in [SparseMatrix::zeros(3, 1), SparseMatrix::zeros(3, 0)] {
        assert_eq!(singular.solve(&rhs), Err(MatrixError::SingularMatrix));
    }
    assert!(matches!(
        SparseMatrix::zeros(2, 3).det(),
        Err(MatrixError::NotSquare(2, 3))
    ));
    assert!(matches!(
        SparseMatrix::eye(2)
            .unwrap()
            .solve(&SparseMatrix::zeros(3, 1)),
        Err(MatrixError::ShapeMismatch(..))
    ));
}

#[test]
fn each_lu_obligation_is_checked_independently() {
    let identity = SparseMatrix::eye(2).unwrap();
    let valid = identity.lu().unwrap();
    let mut nonunit = valid.clone();
    nonunit.lower = integers(2, 2, &[2, 0, 0, 1]);
    nonunit.upper = changed(&identity, 0, 0, Expr::rational(1, 2).unwrap());
    // Product still equals I; checking only P A = L U would accept this.
    assert!(verify_sparse_lu_certificate(&identity, &nonunit).is_err());
    let mut upper_in_lower = valid.clone();
    upper_in_lower.lower = integers(2, 2, &[1, 1, 0, 1]);
    upper_in_lower.upper = integers(2, 2, &[1, -1, 0, 1]);
    assert!(verify_sparse_lu_certificate(&identity, &upper_in_lower).is_err());
    let mut lower_in_upper = valid.clone();
    lower_in_upper.lower = integers(2, 2, &[1, 0, 1, 1]);
    lower_in_upper.upper = integers(2, 2, &[1, 0, -1, 1]);
    assert!(verify_sparse_lu_certificate(&identity, &lower_in_upper).is_err());
    let zero = SparseMatrix::zeros(2, 2);
    let mut invalid_permutation = zero.lu().unwrap();
    invalid_permutation.row_permutation = vec![0, 0];
    assert!(verify_sparse_lu_certificate(&zero, &invalid_permutation).is_err());
    for permutation in [vec![0], vec![0, 2], vec![1, 0]] {
        let mut mutant = valid.clone();
        mutant.row_permutation = permutation;
        assert!(verify_sparse_lu_certificate(&identity, &mutant).is_err());
    }
    let mut wrong_shape = valid.clone();
    wrong_shape.upper = SparseMatrix::zeros(2, 1);
    assert!(verify_sparse_lu_certificate(&identity, &wrong_shape).is_err());
    let mut wrong_product = valid;
    wrong_product.upper = changed(&identity, 0, 1, Expr::from_i64(1));
    assert!(verify_sparse_lu_certificate(&identity, &wrong_product).is_err());
}

#[test]
fn determinant_mutants_and_tiny_solution_perturbations_are_rejected() {
    let a = integers(2, 2, &[0, 2, 3, 4]);
    let b = integers(2, 1, &[2, 7]);
    let det = a.det_with_certificate().unwrap();
    verify_sparse_determinant_certificate(&a, &det).unwrap();
    let wrong_det = SparseDeterminantCertificate {
        determinant: Expr::from_i64(6),
        ..det
    };
    assert!(verify_sparse_determinant_certificate(&a, &wrong_det).is_err());
    let mut solution = a.solve_with_certificate(&b).unwrap();
    assert_eq!(solution.solution, integers(2, 1, &[1, 1]));
    let denominator = BigInt::from(10).pow(80);
    let epsilon_away = BigRational::one() + BigRational::new(BigInt::one(), denominator);
    solution.solution = changed(&solution.solution, 1, 0, Expr::Rational(epsilon_away));
    assert!(verify_sparse_linear_system_certificate(&a, &b, &solution).is_err());
    let singular = SparseMatrix::zeros(2, 2);
    let rhs = SparseMatrix::zeros(2, 1);
    let nonunique = SparseLinearSystemCertificate {
        solution: rhs.clone(),
        factorization: singular.lu().unwrap(),
    };
    // The exact residual is zero, but the uniqueness obligation fails.
    assert!(verify_sparse_linear_system_certificate(&singular, &rhs, &nonunique).is_err());
}

#[test]
fn unsupported_original_entries_cannot_hide_behind_singular_or_zero_outputs() {
    let undefined = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    let zero_times_undefined = Expr::Mul(vec![Expr::from_i64(0), undefined]);
    for entry in [Expr::symbol("x"), zero_times_undefined] {
        let matrix = changed(&SparseMatrix::zeros(2, 2), 0, 1, entry.clone());
        assert_eq!(matrix.det(), Err(MatrixError::UnsupportedCertificateDomain));
        assert_eq!(
            matrix.solve(&SparseMatrix::zeros(2, 1)),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
        let rhs = changed(&SparseMatrix::zeros(2, 1), 1, 0, entry);
        assert_eq!(
            SparseMatrix::zeros(2, 2).solve(&rhs),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }
}

#[test]
fn dimensions_and_coefficient_growth_are_refused_before_large_allocation() {
    assert!(matches!(
        SparseMatrix::zeros(usize::MAX, usize::MAX).det(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let oversized = Expr::Integer(BigInt::one() << 16_384_u32);
    let matrix = changed(&SparseMatrix::eye(1).unwrap(), 0, 0, oversized);
    assert!(matches!(matrix.det(), Err(MatrixError::ResourceLimit(_))));
    // The inputs fit, but their uncancelled product exceeds the bound.
    let large = Expr::Integer(BigInt::one() << 9_000_u32);
    let matrix = SparseMatrix::new(
        2,
        2,
        BTreeMap::from([((0, 0), large.clone()), ((1, 1), large)]),
    )
    .unwrap();
    assert!(matches!(matrix.det(), Err(MatrixError::ResourceLimit(_))));
}

#[test]
fn elimination_fill_is_bounded_for_an_initially_sparse_arrowhead() {
    // Eliminating the first column fills the remaining square block. There
    // are only 1558 original entries, but more than 262144 resulting entries.
    let n = 520;
    let mut entries = BTreeMap::new();
    for index in 0..n {
        entries.insert((0, index), Expr::from_i64(1));
        entries.insert((index, 0), Expr::from_i64(1));
        if index != 0 {
            entries.insert((index, index), Expr::from_i64(2));
        }
    }
    let matrix = SparseMatrix::new(n, n, entries).unwrap();
    assert_eq!(matrix.nnz(), 1558);
    assert!(matches!(matrix.lu(), Err(MatrixError::ResourceLimit(_))));
    assert_eq!(matrix.nnz(), 1558);
}

#[derive(Default)]
struct RecordingMeter {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for RecordingMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert_ne!(amount, 0);
        self.charges[dimension.index()] += amount;
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn solve_accounts_generation_and_verification_and_cancels_at_every_safe_point() {
    let a = integers(2, 2, &[0, 2, 3, 4]);
    let b = integers(2, 1, &[2, 7]);
    let original = a.clone();
    let rhs_original = b.clone();
    let mut measured = RecordingMeter::default();
    let expected = a.metered_solve(&b, &mut measured).unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 17,
    };
    assert_eq!(
        a.metered_solve(&b, &mut Budget::new(limits)).unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] == 0 {
            continue;
        }
        let mut short = limits;
        short.dimensions[dimension.index()] -= 1;
        assert!(matches!(
            a.metered_solve(&b, &mut Budget::new(short)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    for cancel_at in 1..=measured.checkpoints {
        let mut meter = RecordingMeter {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            a.metered_solve(&b, &mut meter),
            Err(MatrixError::Cancelled),
            "safe point {cancel_at}"
        );
    }
    assert_eq!(a, original);
    assert_eq!(b, rhs_original);
    // A separately presented certificate also respects caller cancellation.
    let cert = a.solve_with_certificate(&b).unwrap();
    let mut cancelled = RecordingMeter {
        cancel_at: Some(1),
        ..Default::default()
    };
    assert_eq!(
        metered_verify_sparse_linear_system_certificate(&a, &b, &cert, &mut cancelled),
        Err(MatrixError::Cancelled)
    );
}
