//! Exact sparse least-squares anchors, independent acceptance and bounded work.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    MatrixError, SparseLeastSquaresCandidate, SparseMatrix,
    generate_sparse_least_squares_candidate, generate_sparse_rref_candidate,
    verify_sparse_least_squares_candidate,
};
use num_traits::One;
use std::collections::BTreeMap;
use std::sync::Arc;

fn integers(rows: usize, columns: usize, values: &[i64]) -> SparseMatrix {
    assert_eq!(values.len(), rows * columns);
    SparseMatrix::new(
        rows,
        columns,
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(index, &value)| ((index / columns, index % columns), Expr::from_i64(value)))
            .collect(),
    )
    .unwrap()
}

fn changed(matrix: &SparseMatrix, row: usize, column: usize, value: Expr) -> SparseMatrix {
    let mut entries = matrix.entries().clone();
    entries.insert((row, column), value);
    SparseMatrix::new(matrix.rows(), matrix.cols(), entries).unwrap()
}

#[test]
fn coupled_inconsistent_multiple_rhs_have_exact_unique_minimizers() {
    let a = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    let b = integers(3, 2, &[1, 2, 2, -1, 4, 0]);
    let expected = SparseMatrix::new(
        2,
        2,
        BTreeMap::from([
            ((0, 0), Expr::rational(4, 3).unwrap()),
            ((0, 1), Expr::rational(5, 3).unwrap()),
            ((1, 0), Expr::rational(7, 3).unwrap()),
            ((1, 1), Expr::rational(-4, 3).unwrap()),
        ]),
    )
    .unwrap();
    assert_eq!(a.solve(&b), Err(MatrixError::InconsistentLinearSystem));
    let verified = a.least_squares_with_certificate(&b).unwrap();
    assert_eq!(verified.solution(), &expected);
    assert_eq!(verified.rhs(), &b);
    assert_eq!(verified.rank(), 2);
    assert_eq!(verified.rank_evidence().source().rows, 3);
    assert_eq!(verified.rank_evidence().source().columns, 2);
    assert_eq!(verified.into_solution(), expected);
    // A^T B = [[5,2],[6,-1]], A^T A = [[2,1],[1,2]]. These
    // independently calculated values yield the fractions asserted above.
}

#[test]
fn all_small_single_column_systems_match_the_scalar_projection_formula() {
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                let matrix = integers(3, 1, &[a, b, c]);
                let rhs = integers(3, 2, &[2, -1, -1, 4, 3, 2]);
                let denominator = a * a + b * b + c * c;
                if denominator == 0 {
                    assert_eq!(
                        matrix.solve_least_squares(&rhs),
                        Err(MatrixError::SingularMatrix)
                    );
                    continue;
                }
                let result = matrix.solve_least_squares(&rhs).unwrap();
                assert_eq!(
                    result.get(0, 0).unwrap(),
                    Expr::rational(2 * a - b + 3 * c, denominator).unwrap()
                );
                assert_eq!(
                    result.get(0, 1).unwrap(),
                    Expr::rational(-a + 4 * b + 2 * c, denominator).unwrap()
                );
            }
        }
    }
}

#[test]
fn six_hundred_unknown_tall_system_stays_sparse_with_multiple_rhs() {
    // A=[I;2I] has 720000 dense cells, exceeding the dense storage bound.
    let n = 600;
    let mut entries = BTreeMap::new();
    let mut rhs = BTreeMap::new();
    for row in 0..n {
        entries.insert((row, row), Expr::from_i64(1));
        entries.insert((n + row, row), Expr::from_i64(2));
        rhs.insert((row, 0), Expr::from_i64(3));
        rhs.insert((n + row, 0), Expr::from_i64(4));
        rhs.insert((row, 1), Expr::from_i64(-1));
        rhs.insert((n + row, 1), Expr::from_i64(3));
    }
    let matrix = SparseMatrix::new(2 * n, n, entries).unwrap();
    let rhs = SparseMatrix::new(2 * n, 2, rhs).unwrap();
    assert!(matches!(
        matrix.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    let result = matrix.solve_least_squares(&rhs).unwrap();
    assert_eq!((result.rows(), result.cols(), result.nnz()), (n, 2, 2 * n));
    for row in 0..n {
        assert_eq!(result.get(row, 0).unwrap(), Expr::rational(11, 5).unwrap());
        assert_eq!(result.get(row, 1).unwrap(), Expr::from_i64(1));
    }
    assert_eq!(matrix.nnz(), 2 * n);
    assert_eq!(rhs.nnz(), 4 * n);
}

#[test]
fn tiny_rational_coefficients_are_not_rounded_to_zero() {
    let denominator = BigInt::from(10).pow(400);
    let tiny = Expr::Rational(BigRational::new(BigInt::one(), denominator.clone()));
    let twice = Expr::Rational(BigRational::new(BigInt::from(2), denominator.clone()));
    let matrix =
        SparseMatrix::new(2, 1, BTreeMap::from([((0, 0), tiny), ((1, 0), twice)])).unwrap();
    let result = matrix
        .solve_least_squares(&integers(2, 1, &[1, 1]))
        .unwrap();
    assert_eq!(
        result.get(0, 0).unwrap(),
        Expr::Integer(denominator * BigInt::from(3) / BigInt::from(5))
    );
}

#[test]
fn zero_area_shapes_and_empty_rhs_retain_the_uniqueness_obligation() {
    assert_eq!(
        SparseMatrix::zeros(0, 0)
            .solve_least_squares(&SparseMatrix::zeros(0, 3))
            .unwrap(),
        SparseMatrix::zeros(0, 3)
    );
    // With no unknowns the empty solution is unique, even for nonzero B.
    assert_eq!(
        SparseMatrix::zeros(3, 0)
            .solve_least_squares(&integers(3, 1, &[1, -2, 3]))
            .unwrap(),
        SparseMatrix::zeros(0, 1)
    );
    let full_rank = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    assert_eq!(
        full_rank
            .solve_least_squares(&SparseMatrix::zeros(3, 0))
            .unwrap(),
        SparseMatrix::zeros(2, 0)
    );
    for matrix in [
        integers(3, 2, &[1, 2, 2, 4, 3, 6]),
        integers(2, 3, &[1, 0, 0, 0, 1, 0]),
        SparseMatrix::zeros(0, 2),
    ] {
        for columns in [0, 1] {
            assert_eq!(
                matrix.solve_least_squares(&SparseMatrix::zeros(matrix.rows(), columns)),
                Err(MatrixError::SingularMatrix)
            );
        }
    }
    assert!(matches!(
        full_rank.solve_least_squares(&SparseMatrix::zeros(2, 1)),
        Err(MatrixError::ShapeMismatch(..))
    ));
}

#[test]
fn acceptance_rejects_wrong_solutions_sources_and_rank_witnesses() {
    let a = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    let b = integers(3, 1, &[1, 2, 4]);
    let valid = generate_sparse_least_squares_candidate(&a, &b, &mut Unbounded).unwrap();
    verify_sparse_least_squares_candidate(&a, &b, valid.clone(), &mut Unbounded).unwrap();
    let mut wrong = valid.clone();
    wrong.solution = changed(&wrong.solution, 0, 0, Expr::from_i64(1));
    assert!(verify_sparse_least_squares_candidate(&a, &b, wrong, &mut Unbounded).is_err());
    let mut wrong = valid.clone();
    wrong.solution = SparseMatrix::zeros(1, 1);
    assert!(verify_sparse_least_squares_candidate(&a, &b, wrong, &mut Unbounded).is_err());
    let mut wrong = valid.clone();
    wrong.rank_witness.witness.operations.clear();
    assert!(verify_sparse_least_squares_candidate(&a, &b, wrong, &mut Unbounded).is_err());
    let mut wrong = valid.clone();
    wrong.rank_witness.witness.pivots.pop();
    assert!(verify_sparse_least_squares_candidate(&a, &b, wrong, &mut Unbounded).is_err());
    let changed_a = changed(&a, 0, 0, Expr::from_i64(2));
    assert!(
        verify_sparse_least_squares_candidate(&changed_a, &b, valid.clone(), &mut Unbounded)
            .is_err()
    );
    let changed_b = changed(&b, 0, 0, Expr::from_i64(2));
    assert!(verify_sparse_least_squares_candidate(&a, &changed_b, valid, &mut Unbounded).is_err());

    // A zero residual is insufficient: the independently accepted rank must
    // also establish uniqueness, even when there are no RHS columns.
    let singular = integers(3, 2, &[1, 2, 2, 4, 3, 6]);
    for columns in [0, 1] {
        let rhs = SparseMatrix::zeros(3, columns);
        let candidate = SparseLeastSquaresCandidate {
            solution: SparseMatrix::zeros(2, columns),
            rank_witness: generate_sparse_rref_candidate(&singular, &mut Unbounded).unwrap(),
        };
        assert!(matches!(
            verify_sparse_least_squares_candidate(&singular, &rhs, candidate, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
}

#[test]
fn acceptance_detects_perturbations_smaller_than_machine_precision() {
    let a = integers(2, 1, &[1, 1]);
    let b = integers(2, 1, &[0, 2]);
    let mut candidate = generate_sparse_least_squares_candidate(&a, &b, &mut Unbounded).unwrap();
    let perturbed = BigRational::one() + BigRational::new(BigInt::one(), BigInt::from(10).pow(80));
    candidate.solution = changed(&candidate.solution, 0, 0, Expr::Rational(perturbed));
    assert!(matches!(
        verify_sparse_least_squares_candidate(&a, &b, candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn unsupported_original_cells_do_not_hide_behind_zero_or_rank_shortcuts() {
    let undefined = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    let hidden = Expr::Mul(vec![Expr::from_i64(0), undefined]);
    for entry in [Expr::symbol("x"), hidden] {
        let a = changed(&SparseMatrix::zeros(2, 1), 1, 0, entry.clone());
        assert_eq!(
            a.solve_least_squares(&SparseMatrix::zeros(2, 0)),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
        let rhs = changed(&SparseMatrix::zeros(2, 1), 1, 0, entry);
        assert_eq!(
            SparseMatrix::zeros(2, 0).solve_least_squares(&rhs),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }
}

#[test]
fn dimension_coefficient_and_gram_fill_limits_are_terminal() {
    let too_tall = SparseMatrix::zeros(8_193, 1);
    assert!(matches!(
        too_tall.solve_least_squares(&SparseMatrix::zeros(8_193, 1)),
        Err(MatrixError::ResourceLimit(_))
    ));
    let too_high = changed(
        &SparseMatrix::eye(1).unwrap(),
        0,
        0,
        Expr::Integer(BigInt::one() << 16_384_u32),
    );
    assert!(matches!(
        too_high.solve_least_squares(&integers(1, 1, &[1])),
        Err(MatrixError::ResourceLimit(_))
    ));
    // The original coefficients fit the RREF bound, but a normal-equation
    // product with this RHS exceeds the uncancelled rational height bound.
    let high_a = changed(
        &SparseMatrix::eye(1).unwrap(),
        0,
        0,
        Expr::Integer(BigInt::one() << 8_000_u32),
    );
    let high_b = changed(
        &SparseMatrix::eye(1).unwrap(),
        0,
        0,
        Expr::Integer(BigInt::one() << 9_000_u32),
    );
    assert!(matches!(
        high_a.solve_least_squares(&high_b),
        Err(MatrixError::ResourceLimit(_))
    ));

    // [I; ones] has only 1026 original entries, but its Gram matrix needs
    // 263169 stored entries. Fill or the cumulative ledger must refuse it.
    let n = 513;
    let mut entries = BTreeMap::new();
    for column in 0..n {
        entries.insert((column, column), Expr::from_i64(1));
        entries.insert((n, column), Expr::from_i64(1));
    }
    let wide_row = SparseMatrix::new(n + 1, n, entries).unwrap();
    assert!(matches!(
        wide_row.solve_least_squares(&SparseMatrix::zeros(n + 1, 1)),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert_eq!(wide_row.nnz(), 2 * n);
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
fn generation_acceptance_and_publication_share_the_budget_and_cancel_atomically() {
    let a = integers(2, 1, &[1, 2]);
    let b = integers(2, 1, &[1, 1]);
    let originals = (a.clone(), b.clone());
    let mut measured = RecordingMeter::default();
    let expected = a.metered_solve_least_squares(&b, &mut measured).unwrap();
    let mut generation = RecordingMeter::default();
    let candidate = generate_sparse_least_squares_candidate(&a, &b, &mut generation).unwrap();
    let mut acceptance = RecordingMeter::default();
    verify_sparse_least_squares_candidate(&a, &b, candidate.clone(), &mut acceptance).unwrap();
    for dimension in Dimension::ALL {
        assert_eq!(
            measured.charges[dimension.index()],
            generation.charges[dimension.index()] + acceptance.charges[dimension.index()]
        );
    }
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 17,
    };
    assert_eq!(
        a.metered_solve_least_squares(&b, &mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                a.metered_solve_least_squares(&b, &mut Budget::new(short)),
                Err(MatrixError::ResourceLimit(_))
            ));
        }
    }
    for cancel_at in 1..=measured.checkpoints {
        let mut meter = RecordingMeter {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            a.metered_solve_least_squares(&b, &mut meter),
            Err(MatrixError::Cancelled),
            "safe point {cancel_at}"
        );
    }
    assert_eq!((a.clone(), b.clone()), originals);
    for cancel_at in [1, acceptance.checkpoints] {
        let mut meter = RecordingMeter {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert!(matches!(
            verify_sparse_least_squares_candidate(&a, &b, candidate.clone(), &mut meter),
            Err(MatrixError::Cancelled)
        ));
    }
}
