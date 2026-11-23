//! Exact Q(i) thin QR anchors, rank evidence, mutations and resource boundaries.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{GaussianQrCandidate, Matrix, MatrixError};
use std::sync::Arc;

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> Expr {
    let real = Expr::rational(real, denominator).unwrap();
    if imaginary == 0 {
        return real;
    }
    let imaginary = Expr::rational(imaginary, denominator).unwrap() * Expr::Const(Constant::I);
    real + imaginary
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

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn candidate(source: &Matrix) -> GaussianQrCandidate {
    source.metered_gaussian_qr(&mut Unbounded).unwrap()
}

#[test]
fn isotropic_columns_require_the_hermitian_inner_product() {
    // This nonzero column has v^T v=0 and v* v=2.
    let source = matrix(2, 1, &[(1, 0), (0, 1)], 1);
    let result = candidate(&source);
    assert_eq!(result.q, source);
    assert_eq!(result.r, Matrix::eye(1).unwrap());
    assert_eq!(result.norm_squared, vec![rational(2, 1)]);
    assert_eq!(result.pivots, vec![0]);
}

#[test]
fn nonorthogonal_complex_columns_have_hand_computed_exact_projections() {
    let source = matrix(2, 2, &[(1, 0), (1, 1), (0, 1), (2, 0)], 1);
    let result = candidate(&source);
    assert_eq!(
        result.q,
        matrix(2, 2, &[(2, 0), (1, 3), (0, 2), (3, -1)], 2)
    );
    assert_eq!(
        result.r,
        matrix(2, 2, &[(2, 0), (1, -1), (0, 0), (2, 0)], 2)
    );
    assert_eq!(result.norm_squared, vec![rational(2, 1), rational(5, 1)]);
    assert_eq!(result.pivots, vec![0, 1]);
}

#[test]
fn zero_and_dependent_columns_are_retained_in_rectangular_r() {
    let source = matrix(
        2,
        4,
        &[
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (1, 0),
        ],
        1,
    );
    let result = candidate(&source);
    assert_eq!(result.pivots, vec![1, 3]);
    assert_eq!(result.norm_squared, vec![rational(2, 1), rational(1, 2)]);
    assert_eq!(
        result.q,
        matrix(2, 2, &[(0, 2), (0, -1), (2, 0), (1, 0)], 2)
    );
    assert_eq!(
        result.r,
        matrix(
            2,
            4,
            &[
                (0, 0),
                (2, 0),
                (4, 0),
                (1, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (2, 0)
            ],
            2,
        )
    );
}

#[test]
fn normalized_pythagorean_columns_and_scalar_phases_are_exact() {
    let source = matrix(2, 1, &[(0, 3), (4, 0)], 1);
    let (q, r) = source
        .metered_gaussian_qr_normalized(&mut Unbounded)
        .unwrap();
    assert_eq!(q, matrix(2, 1, &[(0, 3), (4, 0)], 5));
    assert_eq!(r, matrix(1, 1, &[(5, 0)], 1));
    for (source, phase, magnitude) in [
        ((0, 1), (0, 1), 1),
        ((0, -3), (0, -1), 3),
        ((-5, 0), (-1, 0), 5),
    ] {
        let (q, r) = matrix(1, 1, &[source], 1)
            .metered_gaussian_qr_normalized(&mut Unbounded)
            .unwrap();
        assert_eq!(q, matrix(1, 1, &[phase], 1));
        assert_eq!(r, matrix(1, 1, &[(magnitude, 0)], 1));
    }
}

fn product(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[test]
fn exhaustive_small_complex_ranks_match_the_independent_determinant() {
    let values = [(0, 0), (1, 0), (0, 1), (1, 1)];
    for a in values {
        for b in values {
            for c in values {
                for d in values {
                    let ad = product(a, d);
                    let bc = product(b, c);
                    let rank = if ad != bc {
                        2
                    } else {
                        usize::from([a, b, c, d].iter().any(|entry| *entry != (0, 0)))
                    };
                    let result = candidate(&matrix(2, 2, &[a, b, c, d], 1));
                    assert_eq!(result.q.cols(), rank, "{a:?} {b:?} {c:?} {d:?}");
                    assert_eq!(result.r.rows(), rank);
                    assert_eq!(result.pivots.len(), rank);
                }
            }
        }
    }
}

#[test]
fn original_tall_source_and_its_conjugation_keep_the_expected_rank() {
    let source = matrix(3, 2, &[(1, 1), (2, 2), (0, 1), (0, 2), (3, -1), (6, -2)], 3);
    let result = candidate(&source);
    assert_eq!(result.pivots, vec![0]);
    assert_eq!(result.norm_squared, vec![rational(13, 9)]);
    assert_eq!(result.r, matrix(1, 2, &[(1, 0), (2, 0)], 1));
    let conjugated = matrix(
        3,
        2,
        &[(1, -1), (2, -2), (0, -1), (0, -2), (3, 1), (6, 2)],
        3,
    );
    assert_eq!(candidate(&conjugated).norm_squared, result.norm_squared);
}

#[test]
fn empty_and_zero_rank_shapes_are_thin_without_false_basis_vectors() {
    for (rows, cols) in [(0, 0), (0, 4), (3, 0), (3, 4)] {
        let source = Matrix::zeros(rows, cols).unwrap();
        let result = candidate(&source);
        assert_eq!(result.q, Matrix::zeros(rows, 0).unwrap());
        assert_eq!(result.r, Matrix::zeros(0, cols).unwrap());
        assert!(result.norm_squared.is_empty());
        assert!(result.pivots.is_empty());
        let pair = source
            .metered_gaussian_qr_normalized(&mut Unbounded)
            .unwrap();
        assert_eq!(pair, (result.q, result.r));
    }
    let held = Expr::Add(vec![Expr::Const(Constant::I), gaussian(0, -1, 1)]);
    let result = candidate(&Matrix::new(1, 2, vec![held.clone(), held]).unwrap());
    assert_eq!(result.q.cols(), 0);
}

#[test]
fn complete_hermitian_gram_check_rejects_imaginary_cross_terms() {
    // Q's columns are dependent: q2=i*q1. Their real cross product is zero,
    // but their Hermitian inner product is 2i. Reconstruction alone passes.
    let source = matrix(2, 2, &[(1, 0), (0, 1), (0, 1), (-1, 0)], 1);
    let forged = GaussianQrCandidate {
        q: source.clone(),
        r: Matrix::eye(2).unwrap(),
        pivots: vec![0, 1],
        norm_squared: vec![rational(2, 1), rational(2, 1)],
    };
    assert!(matches!(
        source.verify_gaussian_qr_candidate(&forged, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(reason)) if reason.contains("Hermitian Gram")
    ));
}

#[test]
fn unit_pivot_minor_rejects_rank_inflation_even_when_reconstruction_and_gram_pass() {
    let source = matrix(2, 2, &[(1, 0), (0, 0), (0, 0), (0, 0)], 1);
    let forged = GaussianQrCandidate {
        q: Matrix::eye(2).unwrap(),
        r: source.clone(),
        pivots: vec![0, 1],
        norm_squared: vec![rational(1, 1), rational(1, 1)],
    };
    assert!(matches!(
        source.verify_gaussian_qr_candidate(&forged, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(reason)) if reason.contains("pivot")
    ));
}

#[test]
fn source_norm_pivot_and_factor_mutants_are_rejected_exactly() {
    let source = matrix(2, 2, &[(1, 0), (1, 1), (0, 1), (2, 0)], 1);
    let good = candidate(&source);
    let mut mutations = Vec::new();
    let mut changed = good.clone();
    changed.norm_squared[0] = rational(3, 1);
    mutations.push(changed);
    let mut changed = good.clone();
    changed.norm_squared[0] = rational(-2, 1);
    mutations.push(changed);
    let mut changed = good.clone();
    changed.pivots = vec![1, 0];
    mutations.push(changed);
    let mut changed = good.clone();
    changed.pivots = vec![0, 0];
    mutations.push(changed);
    let mut changed = good.clone();
    changed.q = Matrix::eye(2).unwrap();
    mutations.push(changed);
    let mut changed = good.clone();
    let mut values = changed.r.data().to_vec();
    values[1] =
        values[1].clone() + Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(80)));
    changed.r = Matrix::new(2, 2, values).unwrap();
    mutations.push(changed);
    for changed in mutations {
        assert!(matches!(
            source.verify_gaussian_qr_candidate(&changed, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
    let changed_source = matrix(2, 2, &[(1, 0), (1, 1), (0, -1), (2, 0)], 1);
    assert!(matches!(
        changed_source.verify_gaussian_qr_candidate(&good, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn zero_shortcuts_never_discard_original_poles_or_unsupported_children() {
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    for source in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
    ] {
        let source = Matrix::new(1, 1, vec![source]).unwrap();
        assert_eq!(
            source.metered_gaussian_qr(&mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
        let empty = GaussianQrCandidate {
            q: Matrix::zeros(1, 0).unwrap(),
            r: Matrix::zeros(0, 1).unwrap(),
            pivots: vec![],
            norm_squared: vec![],
        };
        assert_eq!(
            source.verify_gaussian_qr_candidate(&empty, &mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
    }
    let source = Matrix::new(
        1,
        1,
        vec![Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("x")])],
    )
    .unwrap();
    assert_eq!(
        source.metered_gaussian_qr(&mut Unbounded),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
}

#[test]
fn dimensions_and_exact_bit_growth_remain_bounded() {
    for source in [
        Matrix::new(33, 1, vec![Expr::Const(Constant::I); 33]).unwrap(),
        Matrix::zeros(0, 33).unwrap(),
        Matrix::new(
            1,
            1,
            vec![Expr::Add(vec![
                Expr::Integer(BigInt::from(1) << 4096),
                Expr::Const(Constant::I),
            ])],
        )
        .unwrap(),
    ] {
        assert!(matches!(
            source.metered_gaussian_qr(&mut Unbounded),
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
fn generation_verification_normalization_and_publication_share_one_meter() {
    let source = matrix(2, 2, &[(1, 0), (1, 1), (0, 1), (2, 0)], 1);
    let mut measured = Recording::default();
    let expected = source
        .metered_gaussian_qr_normalized(&mut measured)
        .unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .metered_gaussian_qr_normalized(&mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                source.metered_gaussian_qr_normalized(&mut Budget::new(short)),
                Err(MatrixError::ResourceLimit(_))
            ));
        }
    }
    for cancel_at in [
        1,
        measured.checkpoints / 4,
        measured.checkpoints / 2,
        measured.checkpoints * 3 / 4,
        measured.checkpoints - 1,
        measured.checkpoints,
    ] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.metered_gaussian_qr_normalized(&mut meter),
            Err(MatrixError::Cancelled)
        );
    }
}

#[test]
fn independent_replay_has_its_own_cancellation_and_budget_controls() {
    let source = matrix(2, 1, &[(1, 0), (0, 1)], 1);
    let candidate = candidate(&source);
    let mut measured = Recording::default();
    source
        .verify_gaussian_qr_candidate(&candidate, &mut measured)
        .unwrap();
    let mut limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    limits.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        source.verify_gaussian_qr_candidate(&candidate, &mut Budget::new(limits)),
        Err(MatrixError::ResourceLimit(_))
    ));
    for cancel_at in [1, measured.checkpoints / 2, measured.checkpoints] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.verify_gaussian_qr_candidate(&candidate, &mut meter),
            Err(MatrixError::Cancelled)
        );
    }
}

#[test]
fn zero_rank_outputs_work_with_exact_recorded_no_zero_charge_budgets() {
    for (rows, columns) in [(0, 0), (0, 3), (4, 0)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        let mut recorded = Recording::default();
        let expected = source
            .metered_gaussian_qr_normalized(&mut recorded)
            .unwrap();
        assert_eq!(
            source
                .metered_gaussian_qr_normalized(&mut Budget::new(BudgetLimits {
                    dimensions: recorded.charges,
                    verifier_pool: 0,
                }))
                .unwrap(),
            expected
        );
    }
}

#[test]
fn nonsquare_normalization_reserves_repeated_arithmetic_allocations() {
    // The norm 100000001 exceeds 10^8 and is not a square, so canonical
    // radical extraction visits the complete small-prime trial range.
    let source = matrix(2, 1, &[(10_000, 0), (0, 1)], 1);
    let mut raw = Recording::default();
    source.metered_gaussian_qr(&mut raw).unwrap();
    let mut normalized = Recording::default();
    source
        .metered_gaussian_qr_normalized(&mut normalized)
        .unwrap();
    let allocation = Dimension::AllocationCount.index();
    assert!(
        normalized.charges[allocation] >= raw.charges[allocation] + 20_000,
        "thousands of trial divisions cannot be charged as one allocation"
    );
    let mut limits = BudgetLimits {
        dimensions: normalized.charges,
        verifier_pool: 0,
    };
    limits.dimensions[allocation] -= 1;
    assert!(matches!(
        source.metered_gaussian_qr_normalized(&mut Budget::new(limits)),
        Err(MatrixError::ResourceLimit(_))
    ));
}
