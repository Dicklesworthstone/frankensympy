//! Complete exact sparse rectangular solves, including refusal and resource semantics.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{MatrixError, SparseMatrix};
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
fn tall_pivoted_system_solves_all_rhs_with_original_sources_unchanged() {
    let matrix = integers(4, 2, &[0, 2, 3, 4, 6, 8, -3, 0]);
    let rhs = integers(4, 3, &[6, 2, 0, 15, -2, 0, 30, -4, 0, -3, 6, 0]);
    let expected = integers(2, 3, &[1, -2, 0, 3, 1, 0]);
    let original_a = matrix.clone();
    let original_b = rhs.clone();
    assert_eq!(matrix.solve(&rhs).unwrap(), expected);
    assert_eq!(matrix, original_a);
    assert_eq!(rhs, original_b);
}

#[test]
fn all_729_three_by_two_inputs_distinguish_unique_from_nonunique_solutions() {
    let expected = integers(2, 2, &[2, 0, -1, 3]);
    for code in 0..729 {
        let mut remaining = code;
        let mut values = [0; 6];
        for value in &mut values {
            *value = (remaining % 3) - 1;
            remaining /= 3;
        }
        let matrix = integers(3, 2, &values);
        let rhs = matrix.matmul(&expected).unwrap();
        let rank = matrix.to_dense().unwrap().rank().unwrap();
        if rank == 2 {
            assert_eq!(matrix.solve(&rhs).unwrap(), expected, "input {values:?}");
        } else {
            assert_eq!(
                matrix.solve(&rhs),
                Err(MatrixError::SingularMatrix),
                "input {values:?}"
            );
        }
    }
}

#[test]
fn any_inconsistent_rhs_refuses_the_entire_result() {
    let matrix = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    // The first RHS is consistent; the second fails the last original row.
    let rhs = integers(3, 2, &[2, 7, 3, 8, 5, 16]);
    assert_eq!(
        matrix.solve(&rhs),
        Err(MatrixError::InconsistentLinearSystem)
    );
    let deficient = integers(3, 2, &[1, 2, 2, 4, 0, 0]);
    assert_eq!(
        deficient.solve(&integers(3, 1, &[1, 2, 1])),
        Err(MatrixError::InconsistentLinearSystem)
    );
    assert_eq!(
        SparseMatrix::zeros(3, 2).solve(&integers(3, 1, &[0, 1, 0])),
        Err(MatrixError::InconsistentLinearSystem)
    );
}

#[test]
fn rank_deficiency_never_becomes_a_unique_zero_solution() {
    let matrix = integers(3, 2, &[1, 2, 2, 4, 3, 6]);
    assert_eq!(
        matrix.solve(&integers(3, 1, &[5, 10, 15])),
        Err(MatrixError::SingularMatrix)
    );
    assert_eq!(
        matrix.solve(&SparseMatrix::zeros(3, 2)),
        Err(MatrixError::SingularMatrix)
    );
    assert_eq!(
        matrix.solve(&SparseMatrix::zeros(3, 0)),
        Err(MatrixError::SingularMatrix)
    );
    let wide = integers(2, 3, &[1, 0, 1, 0, 1, 1]);
    assert_eq!(
        wide.solve(&integers(2, 1, &[1, 2])),
        Err(MatrixError::SingularMatrix)
    );
}

#[test]
fn empty_rhs_and_zero_unknowns_keep_shape_and_consistency_obligations() {
    let full_rank = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    assert_eq!(
        full_rank.solve(&SparseMatrix::zeros(3, 0)).unwrap(),
        SparseMatrix::zeros(2, 0)
    );
    let no_unknowns = SparseMatrix::zeros(4, 0);
    for columns in [0, 1, 3] {
        assert_eq!(
            no_unknowns.solve(&SparseMatrix::zeros(4, columns)).unwrap(),
            SparseMatrix::zeros(0, columns)
        );
    }
    let impossible =
        SparseMatrix::new(4, 3, BTreeMap::from([((3, 2), Expr::from_i64(1))])).unwrap();
    assert_eq!(
        no_unknowns.solve(&impossible),
        Err(MatrixError::InconsistentLinearSystem)
    );
    assert_eq!(
        SparseMatrix::zeros(0, 2).solve(&SparseMatrix::zeros(0, 1)),
        Err(MatrixError::SingularMatrix)
    );
    // Square empty inputs retain the original LU route and its shape contract.
    assert_eq!(
        SparseMatrix::zeros(0, 0)
            .solve(&SparseMatrix::zeros(0, 3))
            .unwrap(),
        SparseMatrix::zeros(0, 3)
    );
}

#[test]
fn tiny_rational_pivots_remain_exact_in_tall_systems() {
    let epsilon = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(10).pow(80)));
    let matrix = SparseMatrix::new(
        3,
        2,
        BTreeMap::from([
            ((0, 0), epsilon.clone()),
            ((1, 1), Expr::rational(1, 3).unwrap()),
            ((2, 0), epsilon),
            ((2, 1), Expr::rational(1, 3).unwrap()),
        ]),
    )
    .unwrap();
    let expected = SparseMatrix::new(
        2,
        2,
        BTreeMap::from([
            ((0, 0), Expr::rational(2, 5).unwrap()),
            ((0, 1), Expr::from_i64(-7)),
            ((1, 0), Expr::rational(-5, 7).unwrap()),
            ((1, 1), Expr::from_i64(3)),
        ]),
    )
    .unwrap();
    let rhs = matrix.matmul(&expected).unwrap();
    assert_eq!(matrix.solve(&rhs).unwrap(), expected);
}

#[test]
fn large_overdetermined_system_and_result_stay_sparse() {
    let n = 600;
    let mut entries = BTreeMap::new();
    let mut rhs = BTreeMap::new();
    let mut solution = BTreeMap::new();
    for row in 0..n {
        entries.insert((row, row), Expr::from_i64(2));
        entries.insert((n + row, row), Expr::from_i64(3));
        if row % 7 == 0 {
            solution.insert((row, 0), Expr::from_i64(5));
            rhs.insert((row, 0), Expr::from_i64(10));
            rhs.insert((n + row, 0), Expr::from_i64(15));
        }
        if row % 11 == 0 {
            solution.insert((row, 2), Expr::from_i64(-2));
            rhs.insert((row, 2), Expr::from_i64(-4));
            rhs.insert((n + row, 2), Expr::from_i64(-6));
        }
    }
    let matrix = SparseMatrix::new(2 * n, n, entries).unwrap();
    assert!(matches!(
        matrix.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    let rhs = SparseMatrix::new(2 * n, 3, rhs).unwrap();
    let expected = SparseMatrix::new(n, 3, solution).unwrap();
    assert_eq!(matrix.solve(&rhs).unwrap(), expected);
}

#[test]
fn original_a_and_b_domains_are_admitted_even_with_empty_or_impossible_shapes() {
    let literal = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    let symbolic_rhs =
        SparseMatrix::new(3, 1, BTreeMap::from([((2, 0), Expr::symbol("x"))])).unwrap();
    assert_eq!(
        literal.solve(&symbolic_rhs),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
    let symbolic_a =
        SparseMatrix::new(3, 2, BTreeMap::from([((2, 1), Expr::symbol("x"))])).unwrap();
    assert_eq!(
        symbolic_a.solve(&SparseMatrix::zeros(3, 0)),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
    let wide_symbolic =
        SparseMatrix::new(1, 2, BTreeMap::from([((0, 1), Expr::symbol("x"))])).unwrap();
    assert_eq!(
        wide_symbolic.solve(&SparseMatrix::zeros(1, 0)),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
    let oversized = Expr::Integer(BigInt::one() << MAX_SPARSE_RREF_COEFFICIENT_BITS as u32);
    let rhs = SparseMatrix::new(3, 1, BTreeMap::from([((2, 0), oversized)])).unwrap();
    assert!(matches!(
        literal.solve(&rhs),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        literal.solve(&SparseMatrix::zeros(2, 1)),
        Err(MatrixError::ShapeMismatch(..))
    ));
    let rhs = SparseMatrix::zeros(3, MAX_SPARSE_RREF_DIMENSION);
    assert!(matches!(
        literal.solve(&rhs),
        Err(MatrixError::ResourceLimit(_))
    ));
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
fn one_budget_covers_augmented_replay_all_rhs_residuals_and_publication() {
    let matrix = integers(3, 1, &[2, 3, -1]);
    let rhs = integers(3, 2, &[4, -6, 6, -9, -2, 3]);
    let mut measured = Meter::default();
    let expected = matrix.metered_solve(&rhs, &mut measured).unwrap();
    assert_eq!(expected, integers(1, 2, &[2, -3]));
    for (dimension, index) in [
        (Dimension::ComputeSteps, 0),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 2),
    ] {
        assert!(measured.used[index] > 0);
        let mut exact = BudgetLimits::uniform(u64::MAX, 0);
        exact.dimensions[dimension.index()] = measured.used[index];
        assert_eq!(
            matrix.metered_solve(&rhs, &mut Budget::new(exact)).unwrap(),
            expected
        );
        let mut short = BudgetLimits::uniform(u64::MAX, 0);
        short.dimensions[dimension.index()] = measured.used[index] - 1;
        assert!(matches!(
            matrix.metered_solve(&rhs, &mut Budget::new(short)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let mut shared_limits = BudgetLimits::uniform(u64::MAX, 0);
    shared_limits.dimensions[Dimension::ComputeSteps.index()] = measured.used[0] * 2 - 1;
    let mut shared = Budget::new(shared_limits);
    assert_eq!(matrix.metered_solve(&rhs, &mut shared).unwrap(), expected);
    assert!(matches!(
        matrix.metered_solve(&rhs, &mut shared),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn cancellation_at_every_safe_point_never_publishes_a_partial_rhs_result() {
    let matrix = integers(2, 1, &[2, 3]);
    let rhs = integers(2, 2, &[4, -6, 6, -9]);
    let mut measured = Meter::default();
    matrix.metered_solve(&rhs, &mut measured).unwrap();
    for stop in 1..=measured.checkpoints {
        let mut meter = Meter {
            cancel_at: Some(stop),
            ..Meter::default()
        };
        assert_eq!(
            matrix.metered_solve(&rhs, &mut meter),
            Err(MatrixError::Cancelled),
            "safe point {stop}"
        );
    }
}

#[test]
fn square_solve_and_invertibility_certificate_keep_the_existing_lu_contract() {
    let matrix = integers(2, 2, &[0, 2, 3, 4]);
    let rhs = integers(2, 1, &[6, 15]);
    let certificate = matrix.solve_with_certificate(&rhs).unwrap();
    assert_eq!(certificate.solution, integers(2, 1, &[1, 3]));
    assert_eq!(
        matrix.metered_solve(&rhs, &mut Unbounded).unwrap(),
        certificate.solution
    );
    let tall = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    assert!(matches!(
        tall.solve_with_certificate(&integers(3, 1, &[2, 3, 5])),
        Err(MatrixError::NotSquare(..))
    ));
}
