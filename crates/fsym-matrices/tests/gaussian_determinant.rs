//! Checked Gaussian determinants beyond the old factorial expansion lane.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{Matrix, MatrixError};
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

#[test]
fn complex_triangular_determinants_reach_the_thirty_two_axis_boundary() {
    // (1+i)^9=16+16i; (1+i)^16=256; (1+i)^32=65536.
    // Nonzero superdiagonal cells make these full triangular examples, not
    // diagonal-only dispatch. The old complex Laplace lane stopped at8.
    for (size, expected) in [(9, (16, 16)), (16, (256, 0)), (32, (65_536, 0))] {
        let mut entries = Vec::new();
        for row in 0..size {
            for column in 0..size {
                entries.push(if row == column {
                    gaussian(1, 1, 1)
                } else {
                    Expr::from_i64(i64::from(row < column))
                });
            }
        }
        let source = Matrix::new(size, size, entries).unwrap();
        let expected = gaussian(expected.0, expected.1, 1);
        assert_eq!(
            source
                .metered_gaussian_det(&mut Unbounded)
                .unwrap_or_else(|error| panic!("triangular dimension {size}: {error}")),
            expected,
            "triangular dimension {size}"
        );
        if size == 9 {
            assert_eq!(source.det().unwrap(), expected);
        }
    }
}

#[test]
fn exact_odd_and_even_row_permutations_contribute_the_correct_sign() {
    let odd = matrix(2, 2, &[(0, 0), (0, 1), (1, 0), (0, 0)], 1);
    assert_eq!(
        odd.metered_gaussian_det(&mut Unbounded).unwrap(),
        gaussian(0, -1, 1)
    );
    let even = matrix(
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
    assert_eq!(
        even.metered_gaussian_det(&mut Unbounded).unwrap(),
        gaussian(0, 2, 1)
    );
}

#[test]
fn rational_complex_components_remain_exact_without_binary64_conversion() {
    let source = matrix(2, 2, &[(0, 1), (1, 0), (1, 0), (0, 1)], 7);
    assert_eq!(source.det().unwrap(), Expr::rational(-2, 49).unwrap());
    let source = matrix(1, 1, &[(3, -4)], 5);
    assert_eq!(
        source.metered_gaussian_det(&mut Unbounded).unwrap(),
        gaussian(3, -4, 5)
    );
    let tiny = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(400)));
    let source = Matrix::new(
        2,
        2,
        vec![
            tiny.clone(),
            Expr::Const(Constant::I),
            Expr::from_i64(0),
            Expr::from_i64(1),
        ],
    )
    .unwrap();
    assert_eq!(source.det().unwrap(), tiny);
}

#[test]
fn complete_rank_check_establishes_zero_for_singular_complex_sources() {
    let rank_one = matrix(2, 2, &[(1, 0), (0, 1), (0, 1), (-1, 0)], 1);
    assert_eq!(
        rank_one.metered_gaussian_det(&mut Unbounded).unwrap(),
        Expr::from_i64(0)
    );
    let mut entries = vec![Expr::from_i64(0); 81];
    for index in 0..8 {
        entries[index * 9 + index] = Expr::Const(Constant::I);
    }
    assert_eq!(
        Matrix::new(9, 9, entries).unwrap().det().unwrap(),
        Expr::from_i64(0)
    );
}

#[test]
fn exhaustive_two_by_two_results_match_integer_pair_ad_minus_bc() {
    let values = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let product = |a: (i64, i64), b: (i64, i64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    for a in values {
        for b in values {
            for c in values {
                for d in values {
                    let ad = product(a, d);
                    let bc = product(b, c);
                    let expected = gaussian(ad.0 - bc.0, ad.1 - bc.1, 1);
                    assert_eq!(
                        matrix(2, 2, &[a, b, c, d], 1)
                            .metered_gaussian_det(&mut Unbounded)
                            .unwrap(),
                        expected,
                        "{a:?} {b:?} {c:?} {d:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn empty_determinant_is_one_and_nonempty_zero_determinants_are_zero() {
    for size in 0..=3 {
        assert_eq!(
            Matrix::zeros(size, size)
                .unwrap()
                .metered_gaussian_det(&mut Unbounded)
                .unwrap(),
            Expr::from_i64(i64::from(size == 0))
        );
    }
}

#[test]
fn nonsquare_matrices_refuse_including_zero_area_rectangles() {
    for (rows, columns) in [(0, 3), (3, 0), (2, 3), (3, 2)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        assert_eq!(
            source.metered_gaussian_det(&mut Unbounded),
            Err(MatrixError::NotSquare(rows, columns))
        );
        assert_eq!(source.det(), Err(MatrixError::NotSquare(rows, columns)));
    }
}

#[test]
fn default_dispatch_preserves_the_rational_and_symbolic_lanes() {
    // The established rational determinant has a larger dimension envelope.
    assert_eq!(Matrix::eye(33).unwrap().det().unwrap(), Expr::from_i64(1));
    let source = Matrix::new(
        2,
        2,
        vec![
            Expr::symbol("x"),
            Expr::from_i64(0),
            Expr::from_i64(0),
            Expr::from_i64(1),
        ],
    )
    .unwrap();
    assert_eq!(source.det().unwrap(), Expr::symbol("x"));
    assert_eq!(
        source.metered_gaussian_det(&mut Unbounded),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
    let source = Matrix::new(1, 1, vec![Expr::Const(Constant::Pi)]).unwrap();
    assert_eq!(source.det().unwrap(), Expr::Const(Constant::Pi));
}

#[test]
fn closed_exact_held_arithmetic_is_selected_before_simplification() {
    let reciprocal = Expr::Pow(
        Arc::new(Expr::Const(Constant::I)),
        Arc::new(Expr::from_i64(-1)),
    );
    let sum = Expr::Add(vec![Expr::from_i64(1), Expr::from_i64(1)]);
    let source = Matrix::new(
        2,
        2,
        vec![reciprocal, Expr::from_i64(0), Expr::from_i64(0), sum],
    )
    .unwrap();
    assert_eq!(source.det().unwrap(), gaussian(0, -2, 1));
}

#[test]
fn original_held_poles_cannot_disappear_in_a_zero_determinant() {
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    for entry in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
    ] {
        let source = Matrix::new(
            2,
            2,
            vec![
                Expr::from_i64(0),
                entry,
                Expr::from_i64(0),
                Expr::from_i64(0),
            ],
        )
        .unwrap();
        assert_eq!(
            source.metered_gaussian_det(&mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
        assert_eq!(source.det(), Err(MatrixError::DivisionByZero));
    }
}

#[test]
fn dimensions_input_height_and_published_height_are_bounded() {
    let mut oversized = vec![Expr::from_i64(0); 33 * 33];
    oversized[0] = Expr::Const(Constant::I);
    let source = Matrix::new(33, 33, oversized).unwrap();
    assert!(matches!(source.det(), Err(MatrixError::ResourceLimit(_))));
    let source = Matrix::new(
        1,
        1,
        vec![Expr::Add(vec![
            Expr::Integer(BigInt::from(1) << 4096u32),
            Expr::Const(Constant::I),
        ])],
    )
    .unwrap();
    assert!(matches!(
        source.metered_gaussian_det(&mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
    // Each source and factor is inside4096 bits, but their determinant is not.
    let large = Expr::Integer(BigInt::from(1) << 3000u32);
    let source = Matrix::new(
        2,
        2,
        vec![
            large.clone(),
            Expr::Const(Constant::I),
            Expr::from_i64(0),
            large,
        ],
    )
    .unwrap();
    assert!(matches!(
        source.metered_gaussian_det(&mut Unbounded),
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
fn determinant_uses_one_ledger_through_independent_check_and_scalar_publication() {
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
    let mut recorded = Recording::default();
    let expected = source.metered_gaussian_det(&mut recorded).unwrap();
    let limits = BudgetLimits {
        dimensions: recorded.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .metered_gaussian_det(&mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if recorded.charges[dimension.index()] > 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                source.metered_gaussian_det(&mut Budget::new(short)),
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
            source.metered_gaussian_det(&mut meter),
            Err(MatrixError::Cancelled)
        );
    }
}

#[test]
fn empty_determinant_still_has_a_checked_publication_checkpoint() {
    let source = Matrix::zeros(0, 0).unwrap();
    let mut recorded = Recording::default();
    assert_eq!(
        source.metered_gaussian_det(&mut recorded).unwrap(),
        Expr::from_i64(1)
    );
    let mut meter = Recording {
        cancel_at: Some(recorded.checkpoints),
        ..Default::default()
    };
    assert_eq!(
        source.metered_gaussian_det(&mut meter),
        Err(MatrixError::Cancelled)
    );
}
