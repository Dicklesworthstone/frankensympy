//! Exact Hermitian least squares, original domains and cumulative budgets.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, Constant, Expr};
use fsym_matrices::{Matrix, MatrixError, SparseMatrix};
use std::collections::BTreeMap;

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> Expr {
    Expr::rational(real, denominator).unwrap()
        + Expr::rational(imaginary, denominator).unwrap() * Expr::Const(Constant::I)
}

fn matrix(rows: usize, cols: usize, values: &[(i64, i64)], denominator: i64) -> Matrix {
    Matrix::new(
        rows,
        cols,
        values
            .iter()
            .map(|&(real, imaginary)| gaussian(real, imaginary, denominator))
            .collect(),
    )
    .unwrap()
}

#[test]
fn hermitian_metric_corrects_the_transpose_only_wrong_answer() {
    let source = matrix(2, 1, &[(1, 0), (0, 2)], 1);
    let rhs = matrix(2, 1, &[(1, 0), (0, 0)], 1);
    let expected = matrix(1, 1, &[(1, 0)], 5);
    assert_eq!(source.solve_least_squares(&rhs).unwrap(), expected);
    assert_eq!(
        source
            .metered_gaussian_solve_least_squares(&rhs, &mut Unbounded)
            .unwrap(),
        expected
    );
    assert_ne!(expected, matrix(1, 1, &[(-1, 0)], 3));
}

#[test]
fn isotropic_transpose_does_not_make_a_complex_full_rank_source_singular() {
    // A^T A=0, whereas A* A=2 and the unique minimizer is 1/2.
    let source = matrix(2, 1, &[(1, 0), (0, 1)], 1);
    let rhs = matrix(2, 1, &[(1, 0), (0, 0)], 1);
    assert_eq!(
        source.solve_least_squares(&rhs).unwrap(),
        matrix(1, 1, &[(1, 0)], 2)
    );
}

#[test]
fn every_complex_rhs_is_projected_with_the_adjoint() {
    let source = matrix(2, 1, &[(1, 0), (0, 2)], 1);
    let rhs = matrix(2, 2, &[(1, 0), (0, 0), (0, 0), (1, 0)], 1);
    assert_eq!(
        source.solve_least_squares(&rhs).unwrap(),
        matrix(1, 2, &[(1, 0), (0, -2)], 5)
    );
    // A rational coefficient matrix still needs Q(i) admission when B is complex.
    let rational = matrix(2, 1, &[(1, 0), (2, 0)], 1);
    let rhs = matrix(2, 1, &[(0, 1), (0, 0)], 1);
    assert_eq!(
        rational.solve_least_squares(&rhs).unwrap(),
        matrix(1, 1, &[(0, 1)], 5)
    );
}

#[test]
fn tall_multiple_variable_and_rhs_systems_have_hand_derived_solutions() {
    let source = matrix(3, 2, &[(1, 0), (0, 0), (0, 1), (0, 0), (0, 0), (1, 1)], 1);
    let rhs = matrix(3, 2, &[(1, 1), (0, 0), (0, 0), (0, 1), (2, 0), (1, -1)], 1);
    assert_eq!(
        source.solve_least_squares(&rhs).unwrap(),
        matrix(2, 2, &[(1, 1), (1, 0), (2, -2), (0, -2)], 2)
    );
    let square = matrix(1, 1, &[(1, 1)], 1);
    assert_eq!(
        square
            .solve_least_squares(&matrix(1, 1, &[(3, 1)], 1))
            .unwrap(),
        matrix(1, 1, &[(2, -1)], 1)
    );
}

#[test]
fn literal_rational_full_rank_behavior_is_preserved() {
    let source = matrix(3, 2, &[(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (3, 0)], 1);
    let rhs = matrix(3, 1, &[(1, 0), (2, 0), (2, 0)], 1);
    let expected = matrix(2, 1, &[(4, 0), (3, 0)], 6);
    assert_eq!(source.solve_least_squares(&rhs).unwrap(), expected);
    assert_eq!(
        source
            .metered_gaussian_solve_least_squares(&rhs, &mut Unbounded)
            .unwrap(),
        expected
    );
}

#[test]
fn rank_deficient_wide_and_zero_sources_remain_nonunique() {
    for (source, rhs) in [
        (
            matrix(2, 2, &[(1, 0), (2, 0), (0, 1), (0, 2)], 1),
            matrix(2, 1, &[(1, 0), (0, 0)], 1),
        ),
        (
            matrix(1, 2, &[(1, 0), (0, 1)], 1),
            matrix(1, 1, &[(1, 0)], 1),
        ),
        (
            Matrix::zeros(2, 1).unwrap(),
            matrix(2, 1, &[(0, 1), (0, 0)], 1),
        ),
        (
            matrix(2, 2, &[(1, 0), (2, 0), (0, 1), (0, 2)], 1),
            Matrix::zeros(2, 0).unwrap(),
        ),
    ] {
        assert_eq!(
            source.solve_least_squares(&rhs),
            Err(MatrixError::SingularMatrix)
        );
    }
}

#[test]
fn empty_rhs_keeps_full_rank_obligation_and_empty_unknowns_preserve_shape() {
    let source = matrix(2, 1, &[(1, 0), (0, 1)], 1);
    assert_eq!(
        source
            .solve_least_squares(&Matrix::zeros(2, 0).unwrap())
            .unwrap(),
        Matrix::zeros(1, 0).unwrap()
    );
    for (rows, columns) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
        let source = Matrix::zeros(rows, 0).unwrap();
        let rhs = Matrix::zeros(rows, columns).unwrap();
        let mut measured = Recording::default();
        let result = source
            .metered_gaussian_solve_least_squares(&rhs, &mut measured)
            .unwrap();
        assert_eq!(result, Matrix::zeros(0, columns).unwrap());
        assert_eq!(
            source
                .metered_gaussian_solve_least_squares(
                    &rhs,
                    &mut Budget::new(BudgetLimits {
                        dimensions: measured.charges,
                        verifier_pool: 0,
                    }),
                )
                .unwrap(),
            result
        );
    }
}

#[test]
fn both_original_domains_are_checked_before_normal_products_or_rank_shortcuts() {
    let pole = Expr::Pow(Box::new(Expr::from_i64(0)), Box::new(Expr::from_i64(-1)));
    let hidden = Expr::Mul(vec![Expr::from_i64(0), pole]);
    let invalid = Matrix::new(1, 1, vec![hidden]).unwrap();
    let one = matrix(1, 1, &[(1, 0)], 1);
    assert_eq!(
        invalid.solve_least_squares(&one),
        Err(MatrixError::DivisionByZero)
    );
    assert_eq!(
        one.solve_least_squares(&invalid),
        Err(MatrixError::DivisionByZero)
    );
    // B's original pole must not disappear behind a zero A or rank refusal.
    assert_eq!(
        Matrix::zeros(1, 1).unwrap().solve_least_squares(&invalid),
        Err(MatrixError::DivisionByZero)
    );
    let symbolic = Matrix::new(
        1,
        1,
        vec![Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")])],
    )
    .unwrap();
    assert_eq!(
        one.solve_least_squares(&symbolic),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
}

#[test]
fn held_gaussian_source_arithmetic_is_exactly_admitted() {
    let reciprocal = Expr::Pow(Box::new(gaussian(1, 1, 1)), Box::new(Expr::from_i64(-1)));
    let source = Matrix::new(1, 1, vec![reciprocal]).unwrap();
    assert_eq!(
        source
            .solve_least_squares(&matrix(1, 1, &[(1, 0)], 1))
            .unwrap(),
        matrix(1, 1, &[(1, 1)], 1)
    );
}

#[test]
fn bounded_admission_never_falls_back_to_plain_transpose() {
    let one = matrix(1, 1, &[(1, 0)], 1);
    let oversized_rhs = Matrix::new(1, 33, vec![Expr::Const(Constant::I); 33]).unwrap();
    assert!(matches!(
        one.solve_least_squares(&oversized_rhs),
        Err(MatrixError::ResourceLimit(_))
    ));
    let oversized_source = Matrix::new(33, 1, vec![Expr::Const(Constant::I); 33]).unwrap();
    assert!(matches!(
        oversized_source.solve_least_squares(&Matrix::zeros(33, 1).unwrap()),
        Err(MatrixError::ResourceLimit(_))
    ));
    let tall = Matrix::new(
        1,
        1,
        vec![Expr::Add(vec![
            Expr::Integer(BigInt::from(1) << 4096),
            Expr::Const(Constant::I),
        ])],
    )
    .unwrap();
    assert!(matches!(
        one.solve_least_squares(&tall),
        Err(MatrixError::ResourceLimit(_))
    ));
    let unsupported = Matrix::new(1, 1, vec![Expr::symbol("x")]).unwrap();
    assert_eq!(
        unsupported.solve_least_squares(&one),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
}

#[test]
fn incompatible_rhs_shape_is_rejected() {
    assert!(matches!(
        matrix(2, 1, &[(1, 0), (0, 1)], 1).solve_least_squares(&matrix(1, 1, &[(1, 0)], 1)),
        Err(MatrixError::ShapeMismatch(..))
    ));
}

#[test]
fn borrowed_sparse_rhs_admission_precedes_any_dense_conversion() {
    let rhs = SparseMatrix::new(3, 2, BTreeMap::from([((2, 1), gaussian(1, 1, 1))])).unwrap();
    let mut measured = Recording::default();
    rhs.validate_gaussian_pseudoinverse_source_with_meter(&mut measured)
        .unwrap();
    rhs.validate_gaussian_pseudoinverse_source_with_meter(&mut Budget::new(BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    }))
    .unwrap();
    for source in [
        SparseMatrix::zeros(33, 1),
        SparseMatrix::zeros(0, usize::MAX),
    ] {
        assert!(matches!(
            source.validate_gaussian_pseudoinverse_source_with_meter(&mut Unbounded),
            Err(MatrixError::ResourceLimit(_)),
        ));
    }
    let pole = Expr::Pow(Box::new(Expr::from_i64(0)), Box::new(Expr::from_i64(-1)));
    let invalid = SparseMatrix::new(
        1,
        2,
        BTreeMap::from([((0, 1), Expr::Mul(vec![Expr::from_i64(0), pole]))]),
    )
    .unwrap();
    assert_eq!(
        invalid.validate_gaussian_pseudoinverse_source_with_meter(&mut Unbounded),
        Err(MatrixError::DivisionByZero),
    );
    let mut cancelled = Recording {
        cancel_at: Some(1),
        ..Default::default()
    };
    assert_eq!(
        rhs.validate_gaussian_pseudoinverse_source_with_meter(&mut cancelled),
        Err(MatrixError::Cancelled),
    );
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
fn source_generation_rank_replay_residual_check_and_publication_share_one_budget() {
    let source = matrix(2, 1, &[(1, 0), (0, 2)], 1);
    let rhs = matrix(2, 1, &[(1, 0), (0, 0)], 1);
    let mut measured = Recording::default();
    let expected = source
        .metered_gaussian_solve_least_squares(&rhs, &mut measured)
        .unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .metered_gaussian_solve_least_squares(&rhs, &mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] == 0 {
            continue;
        }
        let mut short = limits;
        short.dimensions[dimension.index()] -= 1;
        assert!(matches!(
            source.metered_gaussian_solve_least_squares(&rhs, &mut Budget::new(short)),
            Err(MatrixError::ResourceLimit(_)),
        ));
    }
    for cancel_at in [
        1,
        measured.checkpoints / 4,
        measured.checkpoints / 2,
        measured.checkpoints - 1,
        measured.checkpoints,
    ] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.metered_gaussian_solve_least_squares(&rhs, &mut meter),
            Err(MatrixError::Cancelled),
            "checkpoint {cancel_at}",
        );
    }
}
