//! Sparse complete least-squares families, original-source admission and budgets.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    MatrixError, SparseMatrix, generate_sparse_least_squares_family_candidate,
    verify_sparse_least_squares_family_candidate,
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
fn singular_inconsistent_multi_rhs_retains_the_full_family() {
    let a = matrix(2, 2, &[1, 2, 2, 4], 1);
    let b = matrix(2, 2, &[3, 0, 5, 1], 1);
    let family = a.least_squares_family(&b).unwrap();
    assert_eq!(family.minimum_norm(), &matrix(2, 2, &[13, 2, 26, 4], 25));
    let n = matrix(2, 2, &[4, -2, -2, 1], 5);
    assert_eq!(family.projector(), &n);
    assert_eq!(family.receipt().minimum_norm().rhs_columns(), 2);
    assert_eq!(family.source().entries.len(), 4);
    assert_eq!(family.rhs().entries.len(), 3);
    assert_eq!(family.witness().projector.entries.len(), 4);
    assert_eq!(n.transpose().unwrap(), n);
    assert_eq!(n.matmul(&n).unwrap(), n);
    assert_eq!(a.matmul(&n).unwrap(), SparseMatrix::zeros(2, 2));
    for w in [
        matrix(2, 2, &[1, 0, 0, 1], 1),
        matrix(2, 2, &[2, -5, 3, 7], 1),
        matrix(2, 2, &[1, 2, 5, -3], 7),
    ] {
        let x = family.minimum_norm().add(&n.matmul(&w).unwrap()).unwrap();
        let residual = a.matmul(&x).unwrap().sub(&b).unwrap();
        assert_eq!(
            a.transpose().unwrap().matmul(&residual).unwrap(),
            SparseMatrix::zeros(2, 2)
        );
    }
    // Every independently chosen nullspace displacement is retained unchanged.
    let displacement = matrix(2, 2, &[6, -10, -3, 5], 1);
    assert_eq!(n.matmul(&displacement).unwrap(), displacement);
    let (minimum, projector) = family.into_parts();
    assert_eq!(minimum, matrix(2, 2, &[13, 2, 26, 4], 25));
    assert_eq!(projector, n);
}

#[test]
fn wide_families_and_unique_tall_least_squares_have_exact_projectors() {
    let wide = matrix(2, 3, &[1, 0, 1, 0, 1, 1], 1);
    let b = matrix(2, 2, &[1, 3, 2, -1], 1);
    let family = wide.least_squares_family(&b).unwrap();
    assert_eq!(
        family.minimum_norm(),
        &matrix(3, 2, &[0, 7, 3, -5, 3, 2], 3)
    );
    assert_eq!(
        family.projector(),
        &matrix(3, 3, &[1, 1, -1, 1, 1, -1, -1, -1, 1], 3)
    );
    let tall = matrix(3, 2, &[1, 0, 0, 1, 1, 1], 1);
    let b = matrix(3, 2, &[1, 2, 2, -1, 4, 0], 1);
    let family = tall.least_squares_family(&b).unwrap();
    assert_eq!(family.minimum_norm(), &matrix(2, 2, &[4, 5, 7, -4], 3));
    assert_eq!(family.projector(), &SparseMatrix::zeros(2, 2));
}

#[test]
fn rank_zero_and_empty_shapes_preserve_identity_on_all_free_coordinates() {
    for (rows, columns) in [(0, 0), (0, 3), (3, 0), (2, 3)] {
        let a = SparseMatrix::zeros(rows, columns);
        for rhs_columns in [0, 2] {
            let b = matrix(rows, rhs_columns, &vec![7; rows * rhs_columns], 1);
            let family = a.least_squares_family(&b).unwrap();
            assert_eq!(
                family.minimum_norm(),
                &SparseMatrix::zeros(columns, rhs_columns)
            );
            assert_eq!(family.projector(), &SparseMatrix::eye(columns).unwrap());
        }
    }
    let a = matrix(2, 2, &[1, 2, 2, 4], 1);
    let family = a.least_squares_family(&SparseMatrix::zeros(2, 0)).unwrap();
    assert_eq!(family.minimum_norm(), &SparseMatrix::zeros(2, 0));
    assert_eq!(family.projector(), &matrix(2, 2, &[4, -2, -2, 1], 5));
}

#[test]
fn large_rank_deficient_family_stays_sparse_beyond_the_dense_entry_limit() {
    // A=[I I; 2I 2I], B=[3;4] in each independent 2x2 block.
    // P=[I 2I; I 2I]/10; X0=11/10; N=[I -I; -I I]/2.
    let n = 600;
    let mut entries = BTreeMap::new();
    let mut rhs = BTreeMap::new();
    let mut expected_projector = BTreeMap::new();
    for i in 0..n {
        for column in [i, n + i] {
            entries.insert((i, column), Expr::from_i64(1));
            entries.insert((n + i, column), Expr::from_i64(2));
        }
        rhs.insert((i, 0), Expr::from_i64(3));
        rhs.insert((n + i, 0), Expr::from_i64(4));
        for row in [i, n + i] {
            for column in [i, n + i] {
                expected_projector.insert(
                    (row, column),
                    Expr::rational(if row == column { 1 } else { -1 }, 2).unwrap(),
                );
            }
        }
    }
    let a = SparseMatrix::new(2 * n, 2 * n, entries).unwrap();
    assert!(matches!(
        a.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    let b = SparseMatrix::new(2 * n, 1, rhs).unwrap();
    let family = a.least_squares_family(&b).unwrap();
    assert_eq!(
        family.projector(),
        &SparseMatrix::new(2 * n, 2 * n, expected_projector).unwrap()
    );
    assert_eq!(family.minimum_norm().nnz(), 2 * n);
    assert_eq!(family.projector().nnz(), 4 * n);
    for row in 0..2 * n {
        assert_eq!(
            family.minimum_norm().get(row, 0).unwrap(),
            Expr::rational(11, 10).unwrap()
        );
    }
}

#[test]
fn source_rhs_and_incomplete_family_mutations_cannot_gain_acceptance() {
    let a = matrix(2, 2, &[1, 2, 2, 4], 1);
    let b = matrix(2, 1, &[3, 5], 1);
    let candidate = generate_sparse_least_squares_family_candidate(&a, &b, &mut Unbounded).unwrap();
    let mut changed = candidate.clone();
    changed.witness.projector.entries.clear();
    assert!(matches!(
        verify_sparse_least_squares_family_candidate(&a, &b, changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let mut changed = candidate.clone();
    changed.witness.minimum_norm.solution.entries[0].value += BigRational::from_integer(2.into());
    changed.witness.minimum_norm.solution.entries[1].value -= BigRational::one();
    assert!(matches!(
        verify_sparse_least_squares_family_candidate(&a, &b, changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let mut changed = candidate.clone();
    changed.source.entries[0].value += BigRational::one();
    assert!(matches!(
        verify_sparse_least_squares_family_candidate(&a, &b, changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let mut changed = candidate.clone();
    changed.rhs.entries[0].value += BigRational::one();
    assert!(matches!(
        verify_sparse_least_squares_family_candidate(&a, &b, changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    assert!(
        verify_sparse_least_squares_family_candidate(
            &matrix(2, 2, &[1, 2, 2, 5], 1),
            &b,
            candidate.clone(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_sparse_least_squares_family_candidate(
            &a,
            &matrix(2, 1, &[4, 5], 1),
            candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn unsupported_originals_and_resource_bounds_are_not_hidden_by_empty_outputs() {
    let zero = Expr::from_i64(0);
    let undefined = Expr::Pow(Arc::new(zero.clone()), Arc::new(Expr::from_i64(-1)));
    for value in [Expr::symbol("x"), Expr::Mul(vec![zero, undefined])] {
        let a = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), value.clone())])).unwrap();
        assert!(matches!(
            a.least_squares_family(&SparseMatrix::zeros(1, 0)),
            Err(MatrixError::UnsupportedCertificateDomain)
        ));
        let b = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), value)])).unwrap();
        assert!(matches!(
            SparseMatrix::zeros(1, 0).least_squares_family(&b),
            Err(MatrixError::UnsupportedCertificateDomain)
        ));
    }
    assert!(matches!(
        SparseMatrix::zeros(0, 8193).least_squares_family(&SparseMatrix::zeros(0, 0)),
        Err(MatrixError::ResourceLimit(_))
    ));
    let huge = Expr::Integer(BigInt::one() << 8192u32);
    let a = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), huge)])).unwrap();
    assert!(matches!(
        a.least_squares_family(&SparseMatrix::zeros(1, 0)),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        SparseMatrix::zeros(2, 3).least_squares_family(&SparseMatrix::zeros(3, 1)),
        Err(MatrixError::ShapeMismatch(..))
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
fn one_cumulative_budget_covers_generation_family_proof_and_both_output_lifts() {
    let a = matrix(1, 2, &[1, 0], 1);
    let b = matrix(1, 1, &[2], 1);
    let mut measured = Recording::default();
    let expected = a
        .metered_least_squares_family(&b, &mut measured)
        .unwrap()
        .into_parts();
    let mut generation = Recording::default();
    let candidate =
        generate_sparse_least_squares_family_candidate(&a, &b, &mut generation).unwrap();
    let mut acceptance = Recording::default();
    verify_sparse_least_squares_family_candidate(&a, &b, candidate, &mut acceptance).unwrap();
    for dimension in Dimension::ALL {
        assert_eq!(
            measured.charges[dimension.index()],
            generation.charges[dimension.index()] + acceptance.charges[dimension.index()]
        );
    }
    assert_eq!(
        measured.checkpoints,
        generation.checkpoints + acceptance.checkpoints
    );
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        a.metered_least_squares_family(&b, &mut Budget::new(limits))
            .unwrap()
            .into_parts(),
        expected
    );
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                a.metered_least_squares_family(&b, &mut Budget::new(short)),
                Err(MatrixError::ResourceLimit(_))
            ));
        }
    }
    for cancel_at in 1..=measured.checkpoints {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert!(
            matches!(
                a.metered_least_squares_family(&b, &mut meter),
                Err(MatrixError::Cancelled)
            ),
            "safe point {cancel_at}"
        );
    }
}
