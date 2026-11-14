//! Sparse pseudoinverse, minimum-norm exact anchors, and untrusted proposals.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    MatrixError, SparseMatrix, generate_sparse_minimum_norm_candidate,
    generate_sparse_pseudoinverse_candidate, verify_sparse_minimum_norm_candidate,
    verify_sparse_pseudoinverse_candidate,
};
use num_traits::One;
use std::collections::BTreeMap;
use std::sync::Arc;

fn matrix(rows: usize, columns: usize, values: &[i64], denominator: i64) -> SparseMatrix {
    assert_eq!(values.len(), rows * columns);
    SparseMatrix::new(
        rows,
        columns,
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(index, &value)| {
                (
                    (index / columns, index % columns),
                    Expr::rational(value, denominator).unwrap(),
                )
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn singular_exact_pseudoinverse_and_inconsistent_minimum_norm_multiple_rhs() {
    let a = matrix(2, 2, &[1, 2, 2, 4], 1);
    let b = matrix(2, 2, &[3, 0, 5, 1], 1);
    let evidence = a.pseudoinverse_with_certificate().unwrap();
    assert_eq!(evidence.matrix(), &matrix(2, 2, &[1, 2, 2, 4], 25));
    assert_eq!((evidence.source().rows, evidence.source().columns), (2, 2));
    assert_eq!(evidence.receipt().rows(), 2);
    let solve = a.minimum_norm_with_certificate(&b).unwrap();
    assert_eq!(solve.solution(), &matrix(2, 2, &[13, 2, 26, 4], 25));
    assert_eq!(solve.receipt().rhs_columns(), 2);
    assert_eq!(solve.rhs().entries.len(), 3);
    assert_eq!(a.solve_least_squares(&b), Err(MatrixError::SingularMatrix));
}

#[test]
fn coupled_wide_and_tall_full_rank_anchors() {
    let wide = matrix(2, 3, &[1, 0, 1, 0, 1, 1], 1);
    let expected = matrix(3, 2, &[2, -1, -1, 2, 1, 1], 3);
    assert_eq!(wide.pseudoinverse().unwrap(), expected);
    let b = matrix(2, 2, &[1, 3, 2, -1], 1);
    assert_eq!(
        wide.minimum_norm_least_squares(&b).unwrap(),
        matrix(3, 2, &[0, 7, 3, -5, 3, 2], 3)
    );
    let tall = matrix(3, 2, &[1, 0, 0, 1, 1, 1], 1);
    assert_eq!(
        tall.pseudoinverse().unwrap(),
        matrix(2, 3, &[2, -1, 1, -1, 2, 1], 3)
    );
    let rhs = matrix(3, 2, &[1, 2, 2, -1, 4, 0], 1);
    assert_eq!(
        tall.minimum_norm_least_squares(&rhs).unwrap(),
        matrix(2, 2, &[4, 5, 7, -4], 3)
    );
}

#[test]
fn scalar_rank_one_outer_products_match_the_closed_form() {
    // (u*v^T)^+ = v*u^T / ((u^T*u)*(v^T*v)), including negative entries.
    for left in -2..=2 {
        for right in -2..=2 {
            if left == 0 && right == 0 {
                continue;
            }
            let a = matrix(2, 3, &[left, -left, 2 * left, right, -right, 2 * right], 1);
            let denominator = 6 * (left * left + right * right);
            assert_eq!(
                a.pseudoinverse().unwrap(),
                matrix(
                    3,
                    2,
                    &[left, right, -left, -right, 2 * left, 2 * right],
                    denominator
                )
            );
        }
    }
}

#[test]
fn large_rank_deficient_block_system_never_requires_dense_materialization() {
    // A=[I I; 2I 2I] has rank n, but 4*n*n dense cells.
    let n = 600;
    let mut entries = BTreeMap::new();
    let mut rhs = BTreeMap::new();
    for i in 0..n {
        for column in [i, n + i] {
            entries.insert((i, column), Expr::from_i64(1));
            entries.insert((n + i, column), Expr::from_i64(2));
        }
        rhs.insert((i, 0), Expr::from_i64(3));
        rhs.insert((n + i, 0), Expr::from_i64(4));
    }
    let a = SparseMatrix::new(2 * n, 2 * n, entries).unwrap();
    assert!(matches!(
        a.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    let b = SparseMatrix::new(2 * n, 1, rhs).unwrap();
    let result = a.minimum_norm_with_certificate(&b).unwrap();
    assert_eq!(
        (
            result.solution().rows(),
            result.solution().cols(),
            result.solution().nnz()
        ),
        (2 * n, 1, 2 * n)
    );
    assert_eq!(
        result.witness().pseudoinverse.pseudoinverse.entries.len(),
        4 * n
    );
    for row in 0..2 * n {
        assert_eq!(
            result.solution().get(row, 0).unwrap(),
            Expr::rational(11, 10).unwrap()
        );
    }
}

#[test]
fn all_zero_area_and_rank_zero_shapes_have_minimum_norm_zero_results() {
    for (rows, columns) in [(0, 0), (0, 3), (3, 0), (2, 3)] {
        let a = SparseMatrix::zeros(rows, columns);
        assert_eq!(
            a.pseudoinverse().unwrap(),
            SparseMatrix::zeros(columns, rows)
        );
        for rhs_columns in [0, 2] {
            let b = SparseMatrix::zeros(rows, rhs_columns);
            assert_eq!(
                a.minimum_norm_least_squares(&b).unwrap(),
                SparseMatrix::zeros(columns, rhs_columns)
            );
        }
    }
    let a = SparseMatrix::zeros(2, 3);
    assert_eq!(
        a.minimum_norm_least_squares(&matrix(2, 1, &[3, -7], 1))
            .unwrap(),
        SparseMatrix::zeros(3, 1)
    );
    let rank_deficient = matrix(2, 2, &[1, 2, 2, 4], 1);
    assert_eq!(
        rank_deficient
            .minimum_norm_least_squares(&SparseMatrix::zeros(2, 0))
            .unwrap(),
        SparseMatrix::zeros(2, 0)
    );
}

#[test]
fn source_candidate_and_nullspace_mutations_cannot_gain_acceptance() {
    let a = matrix(2, 2, &[1, 2, 2, 4], 1);
    let candidate = generate_sparse_pseudoinverse_candidate(&a, &mut Unbounded).unwrap();
    let mut changed = candidate.clone();
    changed.witness.pseudoinverse.entries[0].value += BigRational::one();
    assert!(matches!(
        verify_sparse_pseudoinverse_candidate(&a, changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let mut changed = candidate.clone();
    changed.source.entries[0].value += BigRational::one();
    assert!(matches!(
        verify_sparse_pseudoinverse_candidate(&a, changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let other = matrix(2, 2, &[1, 2, 2, 5], 1);
    assert!(verify_sparse_pseudoinverse_candidate(&other, candidate, &mut Unbounded).is_err());
    let b = matrix(2, 1, &[3, 5], 1);
    let solve = generate_sparse_minimum_norm_candidate(&a, &b, &mut Unbounded).unwrap();
    let mut changed = solve.clone();
    changed.witness.solution.entries[0].value += BigRational::from_integer(2.into());
    changed.witness.solution.entries[1].value -= BigRational::one();
    assert!(matches!(
        verify_sparse_minimum_norm_candidate(&a, &b, changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let changed_b = matrix(2, 1, &[4, 5], 1);
    assert!(verify_sparse_minimum_norm_candidate(&a, &changed_b, solve, &mut Unbounded).is_err());
}

#[test]
fn original_symbolic_and_undefined_sources_are_rejected_even_in_zero_lanes() {
    let zero = Expr::from_i64(0);
    let undefined = Expr::Pow(Arc::new(zero.clone()), Arc::new(Expr::from_i64(-1)));
    for value in [Expr::symbol("x"), Expr::Mul(vec![zero, undefined])] {
        let a = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), value.clone())])).unwrap();
        assert_eq!(
            a.pseudoinverse(),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
        assert_eq!(
            a.minimum_norm_least_squares(&SparseMatrix::zeros(1, 0)),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
        let b = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), value)])).unwrap();
        assert_eq!(
            SparseMatrix::zeros(1, 1).minimum_norm_least_squares(&b),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }
}

#[test]
fn dimensions_height_shape_and_projector_fill_are_explicit_refusals() {
    assert!(matches!(
        SparseMatrix::zeros(8193, 0).pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let huge = Expr::Integer(BigInt::one() << 8192u32);
    let a = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), huge)])).unwrap();
    assert!(matches!(
        a.pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        SparseMatrix::zeros(2, 3).minimum_norm_least_squares(&SparseMatrix::zeros(3, 1)),
        Err(MatrixError::ShapeMismatch(..))
    ));
    let n = 513;
    let a = SparseMatrix::new(
        n,
        1,
        (0..n).map(|row| ((row, 0), Expr::from_i64(1))).collect(),
    )
    .unwrap();
    assert!(matches!(
        a.pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[derive(Default)]
struct Recording {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}
impl BudgetMeter for Recording {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
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
fn one_cumulative_account_covers_generation_independent_replay_and_publication() {
    let a = matrix(1, 2, &[1, 2], 1);
    let b = matrix(1, 1, &[3], 1);
    let mut measured = Recording::default();
    let expected = a
        .metered_minimum_norm_least_squares(&b, &mut measured)
        .unwrap();
    let mut generation = Recording::default();
    let candidate = generate_sparse_minimum_norm_candidate(&a, &b, &mut generation).unwrap();
    let mut acceptance = Recording::default();
    verify_sparse_minimum_norm_candidate(&a, &b, candidate, &mut acceptance).unwrap();
    for dimension in Dimension::ALL {
        assert_eq!(
            measured.charges[dimension.index()],
            generation.charges[dimension.index()] + acceptance.charges[dimension.index()]
        );
    }
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        a.metered_minimum_norm_least_squares(&b, &mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                a.metered_minimum_norm_least_squares(&b, &mut Budget::new(short)),
                Err(MatrixError::ResourceLimit(_))
            ));
        }
    }
    for cancel_at in 1..=measured.checkpoints {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            a.metered_minimum_norm_least_squares(&b, &mut meter),
            Err(MatrixError::Cancelled),
            "safe point {cancel_at}"
        );
    }
}
