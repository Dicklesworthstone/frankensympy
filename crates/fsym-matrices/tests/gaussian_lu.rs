//! Rectangular Q(i) LU: independent anchors, complete rank and false witnesses.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{GaussianLuCandidate, Matrix, MatrixError};
use std::sync::Arc;

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> Expr {
    let real = Expr::rational(real, denominator).unwrap();
    if imaginary == 0 {
        return real;
    }
    real + Expr::rational(imaginary, denominator).unwrap() * Expr::Const(Constant::I)
}

fn matrix(rows: usize, columns: usize, entries: &[(i64, i64)], denominator: i64) -> Matrix {
    Matrix::new(
        rows,
        columns,
        entries
            .iter()
            .map(|&(r, i)| gaussian(r, i, denominator))
            .collect(),
    )
    .unwrap()
}

fn integers(rows: usize, columns: usize, entries: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        entries.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn result(source: &Matrix) -> GaussianLuCandidate {
    source.metered_gaussian_lu(&mut Unbounded).unwrap()
}

fn replace(source: &Matrix, index: usize, value: Expr) -> Matrix {
    let mut entries = source.data().to_vec();
    entries[index] = value;
    Matrix::new(source.rows(), source.cols(), entries).unwrap()
}

#[test]
fn imaginary_pivots_have_exact_unit_lower_factors() {
    let source = matrix(1, 1, &[(0, 1)], 1);
    let actual = result(&source);
    assert_eq!(actual.p, Matrix::eye(1).unwrap());
    assert_eq!(actual.l, Matrix::eye(1).unwrap());
    assert_eq!(actual.u, source);
    assert!(actual.row_swaps.is_empty());
    assert_eq!(actual.pivots, vec![0]);
}

#[test]
fn exact_pivot_policy_uses_the_first_nonzero_row_not_largest_magnitude() {
    let source = integers(2, 2, &[1, 2, 4, 5]);
    let actual = result(&source);
    assert_eq!(actual.p, Matrix::eye(2).unwrap());
    assert_eq!(actual.l, integers(2, 2, &[1, 0, 4, 1]));
    assert_eq!(actual.u, integers(2, 2, &[1, 2, 0, -3]));
    assert!(actual.row_swaps.is_empty());
    assert_eq!(actual.pivots, vec![0, 1]);
}

#[test]
fn gaussian_division_retains_exact_fractional_complex_multipliers() {
    let source = matrix(2, 2, &[(1, 1), (2, 0), (1, 0), (1, -1)], 1);
    let actual = result(&source);
    assert_eq!(
        actual.l,
        matrix(2, 2, &[(2, 0), (0, 0), (1, -1), (2, 0)], 2)
    );
    assert_eq!(actual.u, matrix(2, 2, &[(1, 1), (2, 0), (0, 0), (0, 0)], 1));
    assert_eq!(actual.pivots, vec![0]);
}

#[test]
fn tall_matrices_keep_full_square_lower_factor_and_original_upper_shape() {
    let source = matrix(3, 2, &[(0, 0), (1, 1), (2, 0), (3, 0), (4, 0), (8, 2)], 1);
    let actual = result(&source);
    assert_eq!(actual.p, integers(3, 3, &[0, 1, 0, 1, 0, 0, 0, 0, 1]));
    assert_eq!(actual.l, integers(3, 3, &[1, 0, 0, 0, 1, 0, 2, 2, 1]));
    assert_eq!(
        actual.u,
        matrix(3, 2, &[(2, 0), (3, 0), (0, 0), (1, 1), (0, 0), (0, 0)], 1)
    );
    assert_eq!(actual.row_swaps, vec![(0, 1)]);
    assert_eq!(actual.pivots, vec![0, 1]);
}

#[test]
fn wide_matrices_skip_zero_columns_and_place_multipliers_by_pivot_row() {
    let source = integers(3, 4, &[0, 1, 2, 3, 0, 2, 4, 8, 0, 3, 6, 10]);
    let actual = result(&source);
    assert_eq!(
        actual.l,
        matrix(
            3,
            3,
            &[
                (2, 0),
                (0, 0),
                (0, 0),
                (4, 0),
                (2, 0),
                (0, 0),
                (6, 0),
                (1, 0),
                (2, 0)
            ],
            2
        )
    );
    assert_eq!(
        actual.u,
        integers(3, 4, &[0, 1, 2, 3, 0, 0, 0, 2, 0, 0, 0, 0])
    );
    assert_eq!(actual.pivots, vec![1, 3]);
    assert!(actual.row_swaps.is_empty());
}

#[test]
fn late_row_swaps_move_previously_computed_lower_multipliers() {
    let source = integers(3, 3, &[1, 2, 3, 2, 4, 8, 3, 7, 10]);
    let actual = result(&source);
    assert_eq!(actual.l, integers(3, 3, &[1, 0, 0, 3, 1, 0, 2, 0, 1]));
    assert_eq!(actual.u, integers(3, 3, &[1, 2, 3, 0, 1, 1, 0, 0, 2]));
    assert_eq!(actual.p, integers(3, 3, &[1, 0, 0, 0, 0, 1, 0, 1, 0]));
    assert_eq!(actual.row_swaps, vec![(1, 2)]);
    assert_eq!(actual.pivots, vec![0, 1, 2]);
}

#[test]
fn multiple_forward_swaps_have_the_correct_permutation_orientation() {
    let source = matrix(
        3,
        3,
        &[
            (0, 0),
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (2, 0),
            (0, 0),
        ],
        1,
    );
    let actual = result(&source);
    assert_eq!(actual.p, integers(3, 3, &[0, 1, 0, 0, 0, 1, 1, 0, 0]));
    assert_eq!(actual.l, Matrix::eye(3).unwrap());
    assert_eq!(
        actual.u,
        matrix(
            3,
            3,
            &[
                (1, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (2, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 1)
            ],
            1
        )
    );
    assert_eq!(actual.row_swaps, vec![(0, 1), (1, 2)]);
}

#[test]
fn rank_deficient_zero_leading_columns_do_not_require_diagonal_pivots() {
    let source = integers(3, 3, &[0, 1, 2, 0, 2, 4, 0, 3, 6]);
    let actual = result(&source);
    assert_eq!(actual.l, integers(3, 3, &[1, 0, 0, 2, 1, 0, 3, 0, 1]));
    assert_eq!(actual.u, integers(3, 3, &[0, 1, 2, 0, 0, 0, 0, 0, 0]));
    assert_eq!(actual.pivots, vec![1]);
}

#[test]
fn zero_and_empty_shapes_retain_full_lower_identity_without_invented_rank() {
    for (rows, columns) in [(0, 0), (0, 4), (3, 0), (3, 4), (32, 0), (0, 32)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        let actual = result(&source);
        assert_eq!(actual.p, Matrix::eye(rows).unwrap());
        assert_eq!(actual.l, Matrix::eye(rows).unwrap());
        assert_eq!(actual.u, source);
        assert!(actual.row_swaps.is_empty());
        assert!(actual.pivots.is_empty());
    }
}

#[test]
fn exhaustive_two_by_two_ranks_agree_with_integer_pair_determinants() {
    let values = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let product = |a: (i64, i64), b: (i64, i64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    for a in values {
        for b in values {
            for c in values {
                for d in values {
                    let rank = if product(a, d) != product(b, c) {
                        2
                    } else {
                        usize::from([a, b, c, d].iter().any(|&entry| entry != (0, 0)))
                    };
                    let actual = result(&matrix(2, 2, &[a, b, c, d], 1));
                    assert_eq!(actual.pivots.len(), rank, "{a:?} {b:?} {c:?} {d:?}");
                    if rank == 1 {
                        assert_eq!(actual.pivots, vec![usize::from(a == (0, 0) && c == (0, 0))]);
                    }
                }
            }
        }
    }
}

#[test]
fn checker_accepts_a_hand_built_complex_rectangular_witness() {
    let source = matrix(2, 3, &[(0, 0), (0, 1), (1, 0), (1, 0), (0, 0), (2, 0)], 1);
    let proposed = GaussianLuCandidate {
        p: integers(2, 2, &[0, 1, 1, 0]),
        l: Matrix::eye(2).unwrap(),
        u: matrix(2, 3, &[(1, 0), (0, 0), (2, 0), (0, 0), (0, 1), (1, 0)], 1),
        row_swaps: vec![(0, 1)],
        pivots: vec![0, 1],
    };
    source
        .verify_gaussian_lu_candidate(&proposed, &mut Unbounded)
        .unwrap();
}

#[test]
fn zero_permutations_cannot_erase_source_rows_under_a_valid_product_identity() {
    let source = Matrix::eye(2).unwrap();
    let proposed = GaussianLuCandidate {
        p: Matrix::zeros(2, 2).unwrap(),
        l: Matrix::eye(2).unwrap(),
        u: Matrix::zeros(2, 2).unwrap(),
        row_swaps: vec![],
        pivots: vec![],
    };
    assert!(
        matches!(source.verify_gaussian_lu_candidate(&proposed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(reason)) if reason.contains("permutation"))
    );
}

#[test]
fn reconstruction_alone_does_not_establish_unit_lower_or_echelon_structure() {
    let source = integers(2, 2, &[1, 2, 0, 1]);
    for lower in [integers(2, 2, &[2, 0, 0, 2]), source.clone()] {
        let upper = if lower == source {
            Matrix::eye(2).unwrap()
        } else {
            matrix(2, 2, &[(1, 0), (2, 0), (0, 0), (1, 0)], 2)
        };
        let proposed = GaussianLuCandidate {
            p: Matrix::eye(2).unwrap(),
            l: lower,
            u: upper,
            row_swaps: vec![],
            pivots: vec![0, 1],
        };
        assert!(
            matches!(source.verify_gaussian_lu_candidate(&proposed, &mut Unbounded), Err(MatrixError::InvalidCertificate(reason)) if reason.contains("lower triangular"))
        );
    }
    let source = integers(2, 2, &[0, 1, 1, 0]);
    let proposed = GaussianLuCandidate {
        p: Matrix::eye(2).unwrap(),
        l: Matrix::eye(2).unwrap(),
        u: source.clone(),
        row_swaps: vec![],
        pivots: vec![0, 1],
    };
    assert!(
        matches!(source.verify_gaussian_lu_candidate(&proposed, &mut Unbounded), Err(MatrixError::InvalidCertificate(reason)) if reason.contains("echelon"))
    );
}

#[test]
fn malformed_shapes_swaps_and_rank_reports_are_rejected_before_indexing() {
    let source = integers(3, 3, &[0, 0, 1, 1, 0, 0, 0, 1, 0]);
    let good = result(&source);
    let mut mutations = Vec::new();
    for swaps in [
        vec![],
        vec![(1, 2), (0, 1)],
        vec![(0, usize::MAX)],
        vec![(0, 0)],
        vec![(2, 1)],
        vec![(0, 1); 33],
    ] {
        let mut changed = good.clone();
        changed.row_swaps = swaps;
        mutations.push(changed);
    }
    for pivots in [
        vec![],
        vec![0, 1],
        vec![0, 0, 2],
        vec![2, 1, 0],
        vec![0, 1, usize::MAX],
        vec![0; 33],
    ] {
        let mut changed = good.clone();
        changed.pivots = pivots;
        mutations.push(changed);
    }
    let mut changed = good.clone();
    changed.p = Matrix::eye(2).unwrap();
    mutations.push(changed);
    let mut changed = good.clone();
    changed.l = Matrix::eye(2).unwrap();
    mutations.push(changed);
    let mut changed = good;
    changed.u = Matrix::zeros(3, 2).unwrap();
    mutations.push(changed);
    for changed in mutations {
        assert!(matches!(
            source.verify_gaussian_lu_candidate(&changed, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
}

#[test]
fn exact_original_source_and_tiny_complex_factor_mutants_are_rejected() {
    let source = matrix(2, 2, &[(1, 1), (2, 0), (1, 0), (1, -1)], 1);
    let good = result(&source);
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(90)));
    let mut changed = good.clone();
    changed.u = replace(
        &changed.u,
        1,
        changed.u.get(0, 1).unwrap().clone() + epsilon.clone(),
    );
    assert!(matches!(
        source.verify_gaussian_lu_candidate(&changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let mut changed = good.clone();
    changed.l = replace(
        &changed.l,
        2,
        changed.l.get(1, 0).unwrap().clone() + epsilon * Expr::Const(Constant::I),
    );
    assert!(matches!(
        source.verify_gaussian_lu_candidate(&changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let changed_source = replace(&source, 0, gaussian(1, -1, 1));
    assert!(matches!(
        changed_source.verify_gaussian_lu_candidate(&good, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn held_poles_and_unsupported_children_are_not_erased_by_zero_rows() {
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    for entry in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
    ] {
        let source = Matrix::new(1, 1, vec![entry]).unwrap();
        assert_eq!(
            source.metered_gaussian_lu(&mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
        let proposed = result(&Matrix::zeros(1, 1).unwrap());
        assert_eq!(
            source.verify_gaussian_lu_candidate(&proposed, &mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
    }
    for entry in [
        Expr::symbol("x"),
        Expr::Const(Constant::Pi),
        Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")]),
    ] {
        assert_eq!(
            Matrix::new(1, 1, vec![entry])
                .unwrap()
                .metered_gaussian_lu(&mut Unbounded),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }
}

#[test]
fn forged_undefined_factors_and_oversized_inputs_fail_closed() {
    let source = Matrix::eye(1).unwrap();
    let mut proposed = result(&source);
    proposed.u = Matrix::new(
        1,
        1,
        vec![Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
        ])],
    )
    .unwrap();
    assert_eq!(
        source.verify_gaussian_lu_candidate(&proposed, &mut Unbounded),
        Err(MatrixError::DivisionByZero)
    );
    for source in [
        Matrix::zeros(0, 33).unwrap(),
        Matrix::zeros(33, 0).unwrap(),
        Matrix::new(1, 1, vec![Expr::Integer(BigInt::from(1) << 4096u32)]).unwrap(),
    ] {
        assert!(matches!(
            source.metered_gaussian_lu(&mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}

#[derive(Default)]
struct Recording {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Recording {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(amount > 0);
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
fn one_cumulative_budget_covers_generation_reconstruction_and_publication() {
    let source = matrix(3, 2, &[(0, 0), (1, 1), (2, 0), (3, 0), (4, 0), (8, 2)], 1);
    let mut recorded = Recording::default();
    let expected = source.metered_gaussian_lu(&mut recorded).unwrap();
    let limits = BudgetLimits {
        dimensions: recorded.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .metered_gaussian_lu(&mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if recorded.charges[dimension.index()] > 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                source.metered_gaussian_lu(&mut Budget::new(short)),
                Err(MatrixError::ResourceLimit(_))
            ));
        }
    }
    for cancel_at in [
        1,
        recorded.checkpoints / 4,
        recorded.checkpoints / 2,
        recorded.checkpoints * 3 / 4,
        recorded.checkpoints - 1,
        recorded.checkpoints,
    ] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.metered_gaussian_lu(&mut meter),
            Err(MatrixError::Cancelled)
        );
    }
}

#[test]
fn independent_replay_has_terminal_cancel_and_resource_limits() {
    let source = matrix(2, 2, &[(0, 0), (0, 1), (1, 0), (1, 0)], 1);
    let proposed = result(&source);
    let mut recorded = Recording::default();
    source
        .verify_gaussian_lu_candidate(&proposed, &mut recorded)
        .unwrap();
    let mut short = BudgetLimits {
        dimensions: recorded.charges,
        verifier_pool: 0,
    };
    short.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        source.verify_gaussian_lu_candidate(&proposed, &mut Budget::new(short)),
        Err(MatrixError::ResourceLimit(_))
    ));
    for cancel_at in [1, recorded.checkpoints / 2, recorded.checkpoints] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.verify_gaussian_lu_candidate(&proposed, &mut meter),
            Err(MatrixError::Cancelled)
        );
    }
}

#[test]
fn empty_factor_publication_accepts_exact_recorded_budgets() {
    for (rows, columns) in [(0, 0), (0, 3), (4, 0)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        let mut recorded = Recording::default();
        let expected = source.metered_gaussian_lu(&mut recorded).unwrap();
        assert_eq!(
            source
                .metered_gaussian_lu(&mut Budget::new(BudgetLimits {
                    dimensions: recorded.charges,
                    verifier_pool: 0
                }))
                .unwrap(),
            expected
        );
    }
}
