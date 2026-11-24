//! Independent exact anchors and adversarial Hermitian LDL/Cholesky witnesses.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{GaussianCholeskyCandidate, GaussianLdlCandidate, Matrix, MatrixError};
use std::sync::Arc;

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> Expr {
    let real = Expr::rational(real, denominator).unwrap();
    if imaginary == 0 {
        return real;
    }
    real + Expr::rational(imaginary, denominator).unwrap() * Expr::Const(Constant::I)
}

fn integers(rows: usize, columns: usize, entries: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        entries.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn matrix(rows: usize, columns: usize, entries: &[(i64, i64)], denominator: i64) -> Matrix {
    Matrix::new(
        rows,
        columns,
        entries
            .iter()
            .map(|&(real, imaginary)| gaussian(real, imaginary, denominator))
            .collect(),
    )
    .unwrap()
}

fn root(integer: i64) -> Expr {
    Expr::Pow(
        Arc::new(Expr::from_i64(integer)),
        Arc::new(Expr::rational(1, 2).unwrap()),
    )
}

fn replace(source: &Matrix, index: usize, value: Expr) -> Matrix {
    let mut entries = source.data().to_vec();
    entries[index] = value;
    Matrix::new(source.rows(), source.cols(), entries).unwrap()
}

fn ldl(source: &Matrix) -> GaussianLdlCandidate {
    source.metered_gaussian_ldl(&mut Unbounded).unwrap()
}

fn cholesky(source: &Matrix) -> GaussianCholeskyCandidate {
    let candidate = source
        .gaussian_cholesky_candidate_with_meter(&mut Unbounded)
        .unwrap();
    source
        .verify_gaussian_cholesky_candidate(&candidate, &mut Unbounded)
        .unwrap();
    assert_eq!(
        source.metered_gaussian_cholesky(&mut Unbounded).unwrap(),
        candidate.factor,
    );
    candidate
}

fn complex_source() -> Matrix {
    matrix(2, 2, &[(2, 0), (0, 1), (0, -1), (2, 0)], 1)
}

#[test]
fn textbook_rational_three_by_three_has_hand_computed_exact_factors() {
    let source = integers(3, 3, &[4, 12, -16, 12, 37, -43, -16, -43, 98]);
    let expected = GaussianLdlCandidate {
        l: integers(3, 3, &[1, 0, 0, 3, 1, 0, -4, 5, 1]),
        d: integers(3, 3, &[4, 0, 0, 0, 1, 0, 0, 0, 9]),
    };
    assert_eq!(ldl(&source), expected);
    source
        .verify_gaussian_ldl_candidate(&expected, &mut Unbounded)
        .unwrap();
    assert_eq!(
        cholesky(&source).factor,
        integers(3, 3, &[2, 0, 0, 6, 1, 0, -8, 5, 3]),
    );
}

#[test]
fn imaginary_off_diagonals_use_conjugate_transpose_and_positive_real_pivots() {
    let source = complex_source();
    let expected = GaussianLdlCandidate {
        l: matrix(2, 2, &[(2, 0), (0, 0), (0, -1), (2, 0)], 2),
        d: matrix(2, 2, &[(4, 0), (0, 0), (0, 0), (3, 0)], 2),
    };
    assert_eq!(ldl(&source), expected);
    let expected_factor = Matrix::new(
        2,
        2,
        vec![
            root(2),
            Expr::from_i64(0),
            gaussian(0, -1, 2) * root(2),
            Expr::rational(1, 2).unwrap() * root(6),
        ],
    )
    .unwrap();
    assert_eq!(cholesky(&source).factor, expected_factor);
    let hand_built = GaussianCholeskyCandidate {
        ldl: expected,
        scales: vec![root(2), Expr::rational(1, 2).unwrap() * root(6)],
        factor: expected_factor,
    };
    source
        .verify_gaussian_cholesky_candidate(&hand_built, &mut Unbounded)
        .unwrap();
}

#[test]
fn exhaustive_small_positive_hermitian_pairs_match_scalar_schur_complements() {
    // A=[[a,b],[conj(b),(|b|^2+k)/a]] has determinant k and leading minor a.
    // Expected L and D come from these scalar identities, not another matrix
    // decomposition implementation.
    for a in 1..=3 {
        for real in -1..=1 {
            for imaginary in -1..=1 {
                for k in 1..=3 {
                    let source = Matrix::new(
                        2,
                        2,
                        vec![
                            Expr::from_i64(a),
                            gaussian(real, imaginary, 1),
                            gaussian(real, -imaginary, 1),
                            Expr::rational(real * real + imaginary * imaginary + k, a).unwrap(),
                        ],
                    )
                    .unwrap();
                    let expected = GaussianLdlCandidate {
                        l: Matrix::new(
                            2,
                            2,
                            vec![
                                Expr::from_i64(1),
                                Expr::from_i64(0),
                                gaussian(real, -imaginary, a),
                                Expr::from_i64(1),
                            ],
                        )
                        .unwrap(),
                        d: Matrix::new(
                            2,
                            2,
                            vec![
                                Expr::from_i64(a),
                                Expr::from_i64(0),
                                Expr::from_i64(0),
                                Expr::rational(k, a).unwrap(),
                            ],
                        )
                        .unwrap(),
                    };
                    assert_eq!(ldl(&source), expected, "{a} {real} {imaginary} {k}");
                    cholesky(&source);
                }
            }
        }
    }
}

#[test]
fn every_original_entry_is_checked_for_hermitian_symmetry_without_tolerance() {
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(90)));
    for source in [
        matrix(2, 2, &[(2, 0), (0, 1), (0, 1), (2, 0)], 1),
        matrix(1, 1, &[(2, 1)], 1),
        Matrix::new(
            1,
            1,
            vec![Expr::from_i64(2) + epsilon * Expr::Const(Constant::I)],
        )
        .unwrap(),
        integers(2, 2, &[2, 1, 0, 2]),
    ] {
        assert_eq!(
            source.metered_gaussian_ldl(&mut Unbounded),
            Err(MatrixError::NotHermitian),
        );
        assert_eq!(
            source.metered_gaussian_cholesky(&mut Unbounded),
            Err(MatrixError::NotHermitian),
        );
    }
}

#[test]
fn indefinite_semidefinite_and_zero_leading_pivots_are_not_positive_definite() {
    for source in [
        integers(1, 1, &[-1]),
        integers(1, 1, &[0]),
        integers(2, 2, &[1, 0, 0, -1]),
        integers(2, 2, &[1, 0, 0, 0]),
        integers(2, 2, &[1, 1, 1, 1]),
        integers(2, 2, &[0, 1, 1, 0]),
        Matrix::zeros(2, 2).unwrap(),
    ] {
        assert_eq!(
            source.metered_gaussian_ldl(&mut Unbounded),
            Err(MatrixError::NotPositiveDefinite),
        );
        assert_eq!(
            source.metered_gaussian_cholesky(&mut Unbounded),
            Err(MatrixError::NotPositiveDefinite),
        );
    }
}

#[test]
fn empty_square_factors_are_valid_and_rectangular_shapes_are_refused() {
    let empty = Matrix::zeros(0, 0).unwrap();
    assert_eq!(
        ldl(&empty),
        GaussianLdlCandidate {
            l: empty.clone(),
            d: empty.clone(),
        },
    );
    let candidate = cholesky(&empty);
    assert!(candidate.scales.is_empty());
    assert_eq!(candidate.factor, empty);
    for (rows, columns) in [(0, 3), (3, 0), (2, 3), (3, 2)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        assert_eq!(
            source.metered_gaussian_ldl(&mut Unbounded),
            Err(MatrixError::NotSquare(rows, columns)),
        );
        assert_eq!(
            source.metered_gaussian_cholesky(&mut Unbounded),
            Err(MatrixError::NotSquare(rows, columns)),
        );
    }
}

#[test]
fn thirty_two_axes_are_supported_and_dimension_and_height_bounds_remain_closed() {
    for size in [9, 16, 32] {
        let mut entries = vec![Expr::from_i64(0); size * size];
        for index in 0..size {
            entries[index * size + index] = Expr::from_i64(2);
        }
        let source = Matrix::new(size, size, entries).unwrap();
        let result = ldl(&source);
        assert_eq!(result.l, Matrix::eye(size).unwrap());
        assert_eq!(result.d, source);
        let factor = cholesky(&source).factor;
        for row in 0..size {
            for column in 0..size {
                let expected = if row == column {
                    root(2)
                } else {
                    Expr::from_i64(0)
                };
                assert_eq!(factor.get(row, column).unwrap(), &expected);
            }
        }
    }
    for source in [
        Matrix::eye(33).unwrap(),
        Matrix::new(1, 1, vec![Expr::Integer(BigInt::from(1) << 4096u32)]).unwrap(),
    ] {
        assert!(matches!(
            source.metered_gaussian_ldl(&mut Unbounded),
            Err(MatrixError::ResourceLimit(_)),
        ));
        assert!(matches!(
            source.metered_gaussian_cholesky(&mut Unbounded),
            Err(MatrixError::ResourceLimit(_)),
        ));
    }
}

#[test]
fn tiny_positive_rational_pivots_are_retained_exactly() {
    let tiny = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(120)));
    let source = Matrix::new(
        2,
        2,
        vec![
            Expr::from_i64(1),
            Expr::from_i64(0),
            Expr::from_i64(0),
            tiny,
        ],
    )
    .unwrap();
    let result = ldl(&source);
    assert_eq!(result.l, Matrix::eye(2).unwrap());
    assert_eq!(result.d, source);
}

#[test]
fn defined_held_arithmetic_is_admitted_but_every_original_pole_is_preserved() {
    let held_two = Expr::Add(vec![Expr::from_i64(1), Expr::from_i64(1)]);
    assert_eq!(
        ldl(&Matrix::new(1, 1, vec![held_two]).unwrap()).d,
        integers(1, 1, &[2]),
    );
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    for held in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
    ] {
        // The invalid operand is above the diagonal; a lower-only generator
        // must not silently erase it before the independent source reader.
        let source = replace(&Matrix::eye(2).unwrap(), 1, held);
        assert_eq!(
            source.metered_gaussian_ldl(&mut Unbounded),
            Err(MatrixError::DivisionByZero),
        );
        assert_eq!(
            source.metered_gaussian_cholesky(&mut Unbounded),
            Err(MatrixError::DivisionByZero),
        );
        let proposed = ldl(&Matrix::eye(2).unwrap());
        assert_eq!(
            source.verify_gaussian_ldl_candidate(&proposed, &mut Unbounded),
            Err(MatrixError::DivisionByZero),
        );
    }
}

#[test]
fn unsupported_source_children_do_not_disappear_under_zero_shortcuts() {
    for entry in [
        Expr::symbol("x"),
        Expr::Const(Constant::Pi),
        Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")]),
    ] {
        let source = replace(&Matrix::eye(2).unwrap(), 1, entry);
        assert_eq!(
            source.metered_gaussian_ldl(&mut Unbounded),
            Err(MatrixError::UnsupportedCertificateDomain),
        );
        assert_eq!(
            source.metered_gaussian_cholesky(&mut Unbounded),
            Err(MatrixError::UnsupportedCertificateDomain),
        );
    }
}

#[test]
fn checker_accepts_a_hand_built_complex_positive_definite_ldl_witness() {
    // L=[[1,0],[1+i,1]], D=diag(2,3), so A=[[2,2-2i],[2+2i,7]].
    let source = matrix(2, 2, &[(2, 0), (2, -2), (2, 2), (7, 0)], 1);
    let candidate = GaussianLdlCandidate {
        l: matrix(2, 2, &[(1, 0), (0, 0), (1, 1), (1, 0)], 1),
        d: integers(2, 2, &[2, 0, 0, 3]),
    };
    source
        .verify_gaussian_ldl_candidate(&candidate, &mut Unbounded)
        .unwrap();
    assert_eq!(ldl(&source), candidate);
}

#[test]
fn factor_identity_alone_does_not_replace_unit_lower_and_positive_diagonal_checks() {
    let source = Matrix::eye(2).unwrap();
    let wrong_unit = GaussianLdlCandidate {
        l: integers(2, 2, &[2, 0, 0, 2]),
        d: matrix(2, 2, &[(1, 0), (0, 0), (0, 0), (1, 0)], 4),
    };
    assert!(matches!(
        source.verify_gaussian_ldl_candidate(&wrong_unit, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
    let wrong_triangle = GaussianLdlCandidate {
        l: integers(2, 2, &[1, 1, 0, 1]),
        d: Matrix::eye(2).unwrap(),
    };
    let same_product = integers(2, 2, &[2, 1, 1, 1]);
    assert!(matches!(
        same_product.verify_gaussian_ldl_candidate(&wrong_triangle, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
    // This is a valid indefinite L D L^H identity, outside the promised
    // positive-definite LDL contract.
    let indefinite = integers(2, 2, &[1, 0, 0, -1]);
    let candidate = GaussianLdlCandidate {
        l: Matrix::eye(2).unwrap(),
        d: indefinite.clone(),
    };
    assert!(matches!(
        indefinite.verify_gaussian_ldl_candidate(&candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
}

#[test]
fn shape_diagonal_upper_triangle_and_imaginary_diagonal_mutants_are_rejected() {
    let source = complex_source();
    let good = ldl(&source);
    let mut mutations = Vec::new();
    for factor in [Matrix::eye(1).unwrap(), Matrix::zeros(2, 3).unwrap()] {
        let mut candidate = good.clone();
        candidate.l = factor.clone();
        mutations.push(candidate);
        let mut candidate = good.clone();
        candidate.d = factor;
        mutations.push(candidate);
    }
    for (index, value) in [
        (0, Expr::from_i64(2)),
        (1, Expr::from_i64(1)),
        (3, Expr::Const(Constant::I)),
    ] {
        let mut candidate = good.clone();
        candidate.l = replace(&candidate.l, index, value);
        mutations.push(candidate);
    }
    for (index, value) in [
        (0, Expr::from_i64(0)),
        (0, Expr::from_i64(-2)),
        (1, Expr::from_i64(1)),
        (3, gaussian(3, 1, 2)),
    ] {
        let mut candidate = good.clone();
        candidate.d = replace(&candidate.d, index, value);
        mutations.push(candidate);
    }
    for candidate in mutations {
        assert!(matches!(
            source.verify_gaussian_ldl_candidate(&candidate, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_)),
        ));
    }
}

#[test]
fn exact_source_binding_rejects_changed_sources_and_tiny_factor_perturbations() {
    let source = complex_source();
    let good = ldl(&source);
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(90)));
    let mut changed = good.clone();
    changed.l = replace(
        &changed.l,
        2,
        changed.l.get(1, 0).unwrap().clone() + epsilon.clone(),
    );
    assert!(matches!(
        source.verify_gaussian_ldl_candidate(&changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
    let mut changed = good.clone();
    changed.d = replace(
        &changed.d,
        3,
        changed.d.get(1, 1).unwrap().clone() + epsilon,
    );
    assert!(matches!(
        source.verify_gaussian_ldl_candidate(&changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
    let changed_source = replace(&source, 0, Expr::from_i64(3));
    assert!(matches!(
        changed_source.verify_gaussian_ldl_candidate(&good, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
    let conjugated = matrix(2, 2, &[(2, 0), (0, -1), (0, 1), (2, 0)], 1);
    assert!(matches!(
        conjugated.verify_gaussian_ldl_candidate(&good, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
}

#[test]
fn undefined_candidate_children_are_parsed_before_zero_factor_shortcuts() {
    let source = Matrix::eye(2).unwrap();
    let good = ldl(&source);
    let hidden = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
    ]);
    for lower in [false, true] {
        let mut candidate = good.clone();
        if lower {
            candidate.l = replace(&candidate.l, 1, hidden.clone());
        } else {
            candidate.d = replace(&candidate.d, 1, hidden.clone());
        }
        assert_eq!(
            source.verify_gaussian_ldl_candidate(&candidate, &mut Unbounded),
            Err(MatrixError::DivisionByZero),
        );
    }
}

#[test]
fn independent_cholesky_checker_accepts_an_equivalent_noncanonical_positive_root() {
    let source = integers(1, 1, &[2]);
    let scaled_root = Expr::Mul(vec![Expr::rational(1, 2).unwrap(), root(8)]);
    let candidate = GaussianCholeskyCandidate {
        ldl: GaussianLdlCandidate {
            l: Matrix::eye(1).unwrap(),
            d: source.clone(),
        },
        scales: vec![scaled_root.clone()],
        factor: Matrix::new(1, 1, vec![scaled_root]).unwrap(),
    };
    source
        .verify_gaussian_cholesky_candidate(&candidate, &mut Unbounded)
        .unwrap();
}

#[test]
fn cholesky_rejects_negative_and_imaginary_column_phases_even_with_same_gram() {
    let source = integers(1, 1, &[2]);
    let good = cholesky(&source);
    for phase in [Expr::from_i64(-1), Expr::Const(Constant::I)] {
        let mut candidate = good.clone();
        candidate.factor = Matrix::new(1, 1, vec![phase * root(2)]).unwrap();
        assert!(matches!(
            source.verify_gaussian_cholesky_candidate(&candidate, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_)),
        ));
    }
    let source = complex_source();
    let good = cholesky(&source);
    let mut candidate = good.clone();
    candidate.factor = replace(&candidate.factor, 1, root(2));
    assert!(matches!(
        source.verify_gaussian_cholesky_candidate(&candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
}

#[test]
fn cholesky_requires_correct_positive_scales_complete_shapes_and_checked_ldl() {
    let source = complex_source();
    let good = cholesky(&source);
    let mut mutations = Vec::new();
    for scale in [
        Expr::from_i64(0),
        Expr::from_i64(-1) * root(2),
        root(3),
        Expr::Const(Constant::I),
    ] {
        let mut candidate = good.clone();
        candidate.scales[0] = scale;
        mutations.push(candidate);
    }
    let mut candidate = good.clone();
    candidate.scales.pop();
    mutations.push(candidate);
    let mut candidate = good.clone();
    candidate.scales.push(root(2));
    mutations.push(candidate);
    let mut candidate = good.clone();
    candidate.factor = Matrix::eye(1).unwrap();
    mutations.push(candidate);
    let mut candidate = good.clone();
    candidate.ldl.l = Matrix::eye(2).unwrap();
    mutations.push(candidate);
    for candidate in mutations {
        assert!(matches!(
            source.verify_gaussian_cholesky_candidate(&candidate, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_)),
        ));
    }
}

#[test]
fn normalized_entry_equality_is_exact_even_when_component_signs_agree() {
    let source = integers(1, 1, &[2]);
    let mut candidate = cholesky(&source);
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(50)));
    candidate.factor = Matrix::new(1, 1, vec![(Expr::from_i64(1) + epsilon) * root(2)]).unwrap();
    assert!(matches!(
        source.verify_gaussian_cholesky_candidate(&candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
}

#[test]
fn complete_normalized_forest_is_bounded_before_scalar_interpretation() {
    let source = integers(1, 1, &[2]);
    let good = cholesky(&source);
    let large_root = Expr::Pow(
        Arc::new(Expr::Integer(BigInt::from(1) << 8192u32)),
        Arc::new(Expr::rational(1, 2).unwrap()),
    );
    for invalid in [
        Expr::symbol("hidden"),
        root(-2),
        Expr::Pow(
            Arc::new(Expr::from_i64(2)),
            Arc::new(Expr::rational(1, 3).unwrap()),
        ),
        Expr::Add(vec![]),
        Expr::Mul(vec![Expr::from_i64(1); 17]),
        Expr::Mul(vec![root(2), root(3)]),
        Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")]),
        large_root,
    ] {
        let mut candidate = good.clone();
        candidate.factor = Matrix::new(1, 1, vec![invalid]).unwrap();
        assert!(
            source
                .verify_gaussian_cholesky_candidate(&candidate, &mut Unbounded)
                .is_err()
        );
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

fn limits(recorded: &Recording) -> BudgetLimits {
    BudgetLimits {
        dimensions: recorded.charges,
        verifier_pool: 0,
    }
}

#[test]
fn generation_reconstruction_normalization_and_publication_share_one_budget() {
    let source = complex_source();
    for normalized in [false, true] {
        let mut recorded = Recording::default();
        if normalized {
            source.metered_gaussian_cholesky(&mut recorded).unwrap();
        } else {
            source.metered_gaussian_ldl(&mut recorded).unwrap();
        }
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            assert!(recorded.charges[dimension.index()] > 0);
        }
        let mut exact = Budget::new(limits(&recorded));
        if normalized {
            source.metered_gaussian_cholesky(&mut exact).unwrap();
        } else {
            source.metered_gaussian_ldl(&mut exact).unwrap();
        }
        for dimension in Dimension::ALL {
            if recorded.charges[dimension.index()] == 0 {
                continue;
            }
            let mut short = limits(&recorded);
            short.dimensions[dimension.index()] -= 1;
            let mut budget = Budget::new(short);
            let error = if normalized {
                source.metered_gaussian_cholesky(&mut budget).unwrap_err()
            } else {
                source.metered_gaussian_ldl(&mut budget).unwrap_err()
            };
            assert!(matches!(error, MatrixError::ResourceLimit(_)));
        }
        let mut cumulative = Budget::new(limits(&recorded));
        if normalized {
            source.metered_gaussian_cholesky(&mut cumulative).unwrap();
            assert!(source.metered_gaussian_cholesky(&mut cumulative).is_err());
        } else {
            source.metered_gaussian_ldl(&mut cumulative).unwrap();
            assert!(source.metered_gaussian_ldl(&mut cumulative).is_err());
        }
    }
}

#[test]
fn cancellation_during_every_phase_and_final_publication_is_terminal() {
    for source in [complex_source(), Matrix::zeros(0, 0).unwrap()] {
        for normalized in [false, true] {
            let mut recorded = Recording::default();
            if normalized {
                source.metered_gaussian_cholesky(&mut recorded).unwrap();
            } else {
                source.metered_gaussian_ldl(&mut recorded).unwrap();
            }
            for cancel_at in [
                1,
                recorded.checkpoints.div_ceil(3),
                recorded.checkpoints / 2,
                recorded.checkpoints - 1,
                recorded.checkpoints,
            ] {
                let mut cancelled = Recording {
                    cancel_at: Some(cancel_at),
                    ..Default::default()
                };
                let error = if normalized {
                    source
                        .metered_gaussian_cholesky(&mut cancelled)
                        .unwrap_err()
                } else {
                    source.metered_gaussian_ldl(&mut cancelled).unwrap_err()
                };
                assert_eq!(error, MatrixError::Cancelled);
                assert_eq!(cancelled.checkpoints, cancel_at);
            }
        }
    }
}

#[test]
fn independent_replay_uses_exact_budgets_and_checks_final_cancellation() {
    let source = complex_source();
    let proposed = cholesky(&source);
    for normalized in [false, true] {
        let mut recorded = Recording::default();
        if normalized {
            source
                .verify_gaussian_cholesky_candidate(&proposed, &mut recorded)
                .unwrap();
        } else {
            source
                .verify_gaussian_ldl_candidate(&proposed.ldl, &mut recorded)
                .unwrap();
        }
        let mut exact = Budget::new(limits(&recorded));
        if normalized {
            source
                .verify_gaussian_cholesky_candidate(&proposed, &mut exact)
                .unwrap();
        } else {
            source
                .verify_gaussian_ldl_candidate(&proposed.ldl, &mut exact)
                .unwrap();
        }
        for dimension in Dimension::ALL {
            if recorded.charges[dimension.index()] == 0 {
                continue;
            }
            let mut short = limits(&recorded);
            short.dimensions[dimension.index()] -= 1;
            let mut budget = Budget::new(short);
            let error = if normalized {
                source
                    .verify_gaussian_cholesky_candidate(&proposed, &mut budget)
                    .unwrap_err()
            } else {
                source
                    .verify_gaussian_ldl_candidate(&proposed.ldl, &mut budget)
                    .unwrap_err()
            };
            assert!(matches!(error, MatrixError::ResourceLimit(_)));
        }
        for cancel_at in [1, recorded.checkpoints / 2, recorded.checkpoints] {
            let mut cancelled = Recording {
                cancel_at: Some(cancel_at),
                ..Default::default()
            };
            let error = if normalized {
                source
                    .verify_gaussian_cholesky_candidate(&proposed, &mut cancelled)
                    .unwrap_err()
            } else {
                source
                    .verify_gaussian_ldl_candidate(&proposed.ldl, &mut cancelled)
                    .unwrap_err()
            };
            assert_eq!(error, MatrixError::Cancelled);
            assert_eq!(cancelled.checkpoints, cancel_at);
        }
    }
}

#[test]
fn empty_publication_obeys_exact_measured_limits_for_both_operations() {
    let source = Matrix::zeros(0, 0).unwrap();
    for normalized in [false, true] {
        let mut recorded = Recording::default();
        if normalized {
            source.metered_gaussian_cholesky(&mut recorded).unwrap();
        } else {
            source.metered_gaussian_ldl(&mut recorded).unwrap();
        }
        let mut exact = Budget::new(limits(&recorded));
        if normalized {
            source.metered_gaussian_cholesky(&mut exact).unwrap();
        } else {
            source.metered_gaussian_ldl(&mut exact).unwrap();
        }
    }
}

#[test]
fn legacy_rational_ldl_still_accepts_its_transpose_symmetric_indefinite_contract() {
    let source = integers(2, 2, &[1, 0, 0, -1]);
    let legacy = source.ldl().unwrap();
    assert_eq!(legacy.l, Matrix::eye(2).unwrap());
    assert_eq!(legacy.d, source);
}
