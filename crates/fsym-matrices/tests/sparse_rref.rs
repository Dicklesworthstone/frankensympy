//! Exact sparse rank/nullspace functionality and independent acceptance boundaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    MatrixError, SparseMatrix, generate_sparse_rref_candidate, verify_sparse_rref_candidate,
};
use fsym_proof_kernel::linear_system::LinearRowOperation;
use fsym_proof_kernel::sparse_rref::{MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION};
use std::collections::BTreeMap;

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

#[test]
fn rectangular_rref_rank_and_complete_nullspace_match_hand_elimination() {
    let matrix = integers(3, 5, &[0, 2, 4, 0, 6, 0, 0, 0, 3, 9, 0, 4, 8, 3, 21]);
    let expected = integers(3, 5, &[0, 1, 2, 0, 3, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0]);
    assert_eq!(matrix.rref().unwrap(), (expected, vec![1, 3]));
    assert_eq!(matrix.rank().unwrap(), 2);
    assert_eq!(
        matrix.nullspace().unwrap(),
        vec![
            integers(5, 1, &[1, 0, 0, 0, 0]),
            integers(5, 1, &[0, -2, 1, 0, 0]),
            integers(5, 1, &[0, -3, 0, -3, 1]),
        ]
    );
    assert_eq!(
        matrix.columnspace().unwrap(),
        vec![integers(3, 1, &[2, 0, 4]), integers(3, 1, &[0, 3, 3])]
    );
}

#[test]
fn row_swaps_zero_columns_and_later_elimination_preserve_column_incidence() {
    let matrix = integers(4, 4, &[0, 0, 2, 4, 0, 3, 6, 9, 1, 2, 3, 4, 2, 4, 6, 8]);
    let (reference, pivots) = matrix.to_dense().unwrap().rref().unwrap();
    assert_eq!(
        matrix.rref().unwrap(),
        (reference.to_sparse().unwrap(), pivots)
    );
    assert_eq!(matrix.rank().unwrap(), 3);
    assert_eq!(
        matrix.nullspace().unwrap(),
        vec![integers(4, 1, &[0, 1, -2, 1])]
    );
}

#[test]
fn all_729_small_rectangular_matrices_match_the_separate_dense_lane() {
    for code in 0..729 {
        let mut remaining = code;
        let mut values = [0; 6];
        for value in &mut values {
            *value = (remaining % 3) - 1;
            remaining /= 3;
        }
        let sparse = integers(2, 3, &values);
        let dense = sparse.to_dense().unwrap();
        let (dense_reduced, dense_pivots) = dense.rref().unwrap();
        let (sparse_reduced, sparse_pivots) = sparse.rref().unwrap();
        assert_eq!(
            sparse_reduced.to_dense().unwrap(),
            dense_reduced,
            "input {values:?}"
        );
        assert_eq!(sparse_pivots, dense_pivots, "input {values:?}");
        let expected: Vec<_> = dense
            .nullspace()
            .unwrap()
            .iter()
            .map(|vector| vector.to_sparse().unwrap())
            .collect();
        assert_eq!(sparse.nullspace().unwrap(), expected, "input {values:?}");
    }
}

#[test]
fn rational_and_arbitrarily_small_exact_pivots_are_not_rounded_to_zero() {
    let rational = SparseMatrix::new(
        2,
        3,
        BTreeMap::from([
            ((0, 0), Expr::rational(1, 2).unwrap()),
            ((0, 1), Expr::rational(1, 3).unwrap()),
            ((0, 2), Expr::rational(-5, 6).unwrap()),
            ((1, 0), Expr::rational(1, 4).unwrap()),
            ((1, 1), Expr::rational(1, 6).unwrap()),
            ((1, 2), Expr::rational(-5, 12).unwrap()),
        ]),
    )
    .unwrap();
    assert_eq!(rational.rank().unwrap(), 1);
    assert_eq!(
        rational.nullspace().unwrap(),
        vec![
            SparseMatrix::new(
                3,
                1,
                BTreeMap::from([
                    ((0, 0), Expr::rational(-2, 3).unwrap()),
                    ((1, 0), Expr::from_i64(1)),
                ])
            )
            .unwrap(),
            SparseMatrix::new(
                3,
                1,
                BTreeMap::from([
                    ((0, 0), Expr::rational(5, 3).unwrap()),
                    ((2, 0), Expr::from_i64(1)),
                ])
            )
            .unwrap(),
        ]
    );
    let denominator = BigInt::from(10).pow(80);
    let almost_one = BigRational::new(&denominator + BigInt::one(), denominator);
    let perturbed = SparseMatrix::new(
        2,
        2,
        BTreeMap::from([
            ((0, 0), Expr::from_i64(1)),
            ((0, 1), Expr::from_i64(1)),
            ((1, 0), Expr::from_i64(1)),
            ((1, 1), Expr::Rational(almost_one)),
        ]),
    )
    .unwrap();
    assert_eq!(perturbed.rank().unwrap(), 2);
    assert_eq!(perturbed.rref().unwrap().0, SparseMatrix::eye(2).unwrap());
    assert!(perturbed.nullspace().unwrap().is_empty());
}

#[test]
fn thousand_dimensional_rectangular_kernel_and_pivots_never_require_dense_storage() {
    let n = 1_024;
    let mut entries: BTreeMap<_, _> = (0..n).map(|row| ((row, row), Expr::from_i64(2))).collect();
    entries.insert((0, n), Expr::from_i64(4));
    entries.insert((n - 1, n), Expr::from_i64(6));
    let matrix = SparseMatrix::new(n, n + 1, entries).unwrap();
    assert!(matches!(
        matrix.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    let (reduced, pivots) = matrix.rref().unwrap();
    assert_eq!(pivots, (0..n).collect::<Vec<_>>());
    assert_eq!(reduced.nnz(), n + 2);
    assert_eq!(reduced.get(0, n).unwrap(), Expr::from_i64(2));
    assert_eq!(reduced.get(n - 1, n).unwrap(), Expr::from_i64(3));
    assert_eq!(matrix.rank().unwrap(), n);
    let nullspace = matrix.nullspace().unwrap();
    assert_eq!(nullspace.len(), 1);
    assert_eq!(
        nullspace[0],
        SparseMatrix::new(
            n + 1,
            1,
            BTreeMap::from([
                ((0, 0), Expr::from_i64(-2)),
                ((n - 1, 0), Expr::from_i64(-3)),
                ((n, 0), Expr::from_i64(1)),
            ])
        )
        .unwrap()
    );
    let columns = matrix.columnspace().unwrap();
    assert_eq!(columns.len(), n);
    for (column, vector) in columns.iter().enumerate() {
        assert_eq!((vector.rows(), vector.cols(), vector.nnz()), (n, 1, 1));
        assert_eq!(vector.get(column, 0).unwrap(), Expr::from_i64(2));
    }
}

#[test]
fn large_singular_square_matrix_has_a_complete_kernel() {
    let n = 1_000;
    let matrix = SparseMatrix::new(
        n,
        n,
        (0..n - 1)
            .map(|row| ((row, row), Expr::from_i64(1)))
            .collect(),
    )
    .unwrap();
    assert_eq!(matrix.rank().unwrap(), n - 1);
    assert_eq!(
        matrix.nullspace().unwrap(),
        vec![SparseMatrix::new(n, 1, BTreeMap::from([((n - 1, 0), Expr::from_i64(1))]),).unwrap()]
    );
}

#[test]
fn zero_area_and_zero_rank_shapes_have_their_complete_sparse_bases() {
    for (rows, columns) in [(0, 0), (0, 5), (5, 0), (3, 5)] {
        let matrix = SparseMatrix::zeros(rows, columns);
        assert_eq!(matrix.rank().unwrap(), 0);
        assert_eq!(matrix.rref().unwrap(), (matrix.clone(), Vec::new()));
        assert!(matrix.columnspace().unwrap().is_empty());
        let vectors = matrix.nullspace().unwrap();
        assert_eq!(vectors.len(), columns);
        for (index, vector) in vectors.iter().enumerate() {
            assert_eq!(
                vector,
                &SparseMatrix::new(
                    columns,
                    1,
                    BTreeMap::from([((index, 0), Expr::from_i64(1))])
                )
                .unwrap()
            );
        }
    }
    // Sparse output does not inherit the dense nullspace's quadratic-size cap.
    let matrix = SparseMatrix::zeros(0, 1_024);
    let basis = matrix.nullspace().unwrap();
    assert_eq!(basis.len(), 1_024);
    assert_eq!(basis.iter().map(SparseMatrix::nnz).sum::<usize>(), 1_024);
}

#[test]
fn original_source_and_rank_mutants_cannot_acquire_accepted_wrappers() {
    let matrix = integers(2, 3, &[0, 2, 4, 3, 6, 12]);
    let candidate = generate_sparse_rref_candidate(&matrix, &mut Unbounded).unwrap();
    let checked = verify_sparse_rref_candidate(&matrix, candidate.clone(), &mut Unbounded).unwrap();
    assert_eq!(checked.rank(), 2);
    assert_eq!(checked.receipt().pivots(), &[0, 1]);
    assert_eq!(checked.source(), &candidate.source);
    assert_eq!(checked.reduced(), &candidate.witness.reduced);
    let mut mutations = Vec::new();
    let mut omitted = candidate.clone();
    omitted.witness.operations.pop();
    mutations.push(omitted);
    let mut zero_scale = candidate.clone();
    zero_scale
        .witness
        .operations
        .push(LinearRowOperation::Scale {
            row: 0,
            factor: BigRational::from_integer(BigInt::from(0)),
        });
    mutations.push(zero_scale);
    let mut rank = candidate.clone();
    rank.witness.pivots.pop();
    mutations.push(rank);
    let mut output = candidate.clone();
    output.witness.reduced.entries[0].value = BigRational::from_integer(BigInt::from(2));
    mutations.push(output);
    let mut source = candidate.clone();
    source.source.entries[0].value = BigRational::from_integer(BigInt::from(99));
    mutations.push(source);
    let mut version = candidate.clone();
    version.witness.schema_version += 1;
    mutations.push(version);
    let mut permutation = candidate.clone();
    permutation.witness.reduced.entries.swap(0, 1);
    mutations.push(permutation);
    for mutated in mutations {
        assert!(matches!(
            verify_sparse_rref_candidate(&matrix, mutated, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
    // Coordinated replacement of source and witness still cannot replace the
    // expected caller input at the acceptance boundary.
    let other = integers(2, 3, &[1, 0, 0, 0, 1, 0]);
    let replaced = generate_sparse_rref_candidate(&other, &mut Unbounded).unwrap();
    assert!(matches!(
        verify_sparse_rref_candidate(&matrix, replaced, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn source_domain_dimension_and_coefficient_admission_fail_before_shortcuts() {
    let symbolic = SparseMatrix::new(
        2,
        2,
        BTreeMap::from([((0, 0), Expr::from_i64(1)), ((1, 1), Expr::symbol("x"))]),
    )
    .unwrap();
    for outcome in [
        symbolic.rank().map(|_| ()),
        symbolic.nullspace().map(|_| ()),
    ] {
        assert_eq!(outcome, Err(MatrixError::UnsupportedCertificateDomain));
    }
    for (rows, columns) in [
        (MAX_SPARSE_RREF_DIMENSION + 1, 0),
        (0, MAX_SPARSE_RREF_DIMENSION + 1),
        (usize::MAX, usize::MAX),
    ] {
        assert!(matches!(
            SparseMatrix::zeros(rows, columns).rank(),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let oversized = Expr::Integer(BigInt::one() << MAX_SPARSE_RREF_COEFFICIENT_BITS as u32);
    let matrix = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), oversized)])).unwrap();
    assert!(matches!(matrix.rank(), Err(MatrixError::ResourceLimit(_))));
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    used: [u64; 3],
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        let index = match dimension {
            Dimension::ComputeSteps => Some(0),
            Dimension::MemoryBytes => Some(1),
            Dimension::AllocationCount => Some(2),
            _ => None,
        };
        if let Some(index) = index {
            self.used[index] += amount;
        }
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
fn cumulative_budgets_span_replay_basis_checks_and_publication() {
    let source = integers(2, 3, &[0, 2, 4, 3, 6, 12]);
    let mut measured = Meter::default();
    let expected = source.metered_nullspace(&mut measured).unwrap();
    for (dimension, index) in [
        (Dimension::ComputeSteps, 0),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 2),
    ] {
        assert!(measured.used[index] > 0);
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = measured.used[index];
        assert_eq!(
            source.metered_nullspace(&mut Budget::new(limits)).unwrap(),
            expected
        );
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = measured.used[index] - 1;
        assert!(matches!(
            source.metered_nullspace(&mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = measured.used[0] * 2 - 1;
    let mut shared = Budget::new(limits);
    assert_eq!(source.metered_nullspace(&mut shared).unwrap(), expected);
    assert!(matches!(
        source.metered_nullspace(&mut shared),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn cancellation_at_every_safe_point_publishes_no_partial_basis() {
    let source = integers(2, 3, &[0, 2, 4, 3, 6, 12]);
    let mut measured = Meter::default();
    source.metered_nullspace(&mut measured).unwrap();
    for stop in 1..=measured.checkpoints {
        let mut meter = Meter {
            cancel_at: Some(stop),
            ..Meter::default()
        };
        assert_eq!(
            source.metered_nullspace(&mut meter),
            Err(MatrixError::Cancelled),
            "safe point {stop}"
        );
    }
}

#[test]
fn separately_owned_generator_and_verifier_meters_do_not_reset_failed_work() {
    let source = integers(2, 3, &[0, 2, 4, 3, 6, 12]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    let accepted = source
        .rref_checked_with_meters(&mut generator, &mut verifier)
        .unwrap();
    assert_eq!(accepted.rank(), 2);
    for (dimension, index) in [
        (Dimension::ComputeSteps, 0),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 2),
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = generator.used[index] - 1;
        let mut untouched = Meter::default();
        assert!(matches!(
            source.rref_checked_with_meters(&mut Budget::new(limits), &mut untouched),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert_eq!(untouched.checkpoints, 0);
        assert_eq!(untouched.used, [0; 3]);
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = verifier.used[index] - 1;
        assert!(matches!(
            source.rref_checked_with_meters(&mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let mut cancelled = Meter {
        cancel_at: Some(1),
        ..Meter::default()
    };
    assert!(matches!(
        source.rref_checked_with_meters(&mut Unbounded, &mut cancelled),
        Err(MatrixError::Cancelled)
    ));
}
