//! Exact Gaussian RREF, complete affine families and independent witnesses.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{GaussianAffineCandidate, GaussianRrefCandidate, Matrix, MatrixError};
use std::sync::Arc;

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> Expr {
    let real = Expr::rational(real, denominator).unwrap();
    if imaginary == 0 {
        return real;
    }
    real + Expr::rational(imaginary, denominator).unwrap() * Expr::Const(Constant::I)
}

fn matrix(rows: usize, cols: usize, values: &[(i64, i64)]) -> Matrix {
    Matrix::new(
        rows,
        cols,
        values.iter().map(|&(r, i)| gaussian(r, i, 1)).collect(),
    )
    .unwrap()
}

fn replace(matrix: &Matrix, index: usize, value: Expr) -> Matrix {
    let mut data = matrix.data().to_vec();
    data[index] = value;
    Matrix::new(matrix.rows(), matrix.cols(), data).unwrap()
}

fn candidate(source: &Matrix) -> GaussianRrefCandidate {
    source
        .gaussian_rref_candidate_with_meter(&mut Unbounded)
        .unwrap()
}

#[test]
fn imaginary_pivots_support_rank_nullspace_and_existing_exact_solves() {
    let source = matrix(1, 1, &[(0, 1)]);
    assert_eq!(source.rref().unwrap(), (Matrix::eye(1).unwrap(), vec![0]));
    assert_eq!(source.rank().unwrap(), 1);
    assert_eq!(source.nullspace().unwrap(), Vec::<Matrix>::new());
    assert_eq!(
        source.solve(&Matrix::eye(1).unwrap()).unwrap(),
        matrix(1, 1, &[(0, -1)])
    );
    // I is supported exact arithmetic, not an unsupported-domain sentinel.
    assert_eq!(
        Matrix::zeros(1, 1).unwrap().solve(&source),
        Err(MatrixError::SingularMatrix)
    );
}

#[test]
fn isotropic_rank_deficiency_has_a_complete_complex_kernel() {
    let source = matrix(2, 2, &[(1, 0), (0, 1), (0, 1), (-1, 0)]);
    assert_eq!(
        source.metered_gaussian_rref(&mut Unbounded).unwrap(),
        (matrix(2, 2, &[(1, 0), (0, 1), (0, 0), (0, 0)]), vec![0])
    );
    let expected = vec![matrix(2, 1, &[(0, -1), (1, 0)])];
    assert_eq!(
        source.metered_gaussian_nullspace(&mut Unbounded).unwrap(),
        expected
    );
    assert_eq!(source.nullspace().unwrap(), expected);
    assert_eq!(source.rank().unwrap(), 1);
}

#[test]
fn row_swaps_and_skipped_columns_preserve_canonical_pivots() {
    let source = matrix(
        3,
        4,
        &[
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 0),
            (2, 0),
            (1, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 2),
            (0, 4),
        ],
    );
    let expected = matrix(
        3,
        4,
        &[
            (0, 0),
            (1, 0),
            (0, 0),
            (-1, 0),
            (0, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
    );
    assert_eq!(
        source.metered_gaussian_rref(&mut Unbounded).unwrap(),
        (expected, vec![1, 2])
    );
    assert_eq!(
        source.metered_gaussian_nullspace(&mut Unbounded).unwrap(),
        vec![
            matrix(4, 1, &[(1, 0), (0, 0), (0, 0), (0, 0)]),
            matrix(4, 1, &[(0, 0), (1, 0), (-2, 0), (1, 0)]),
        ]
    );
}

#[test]
fn determinant_anchors_cover_all_two_by_two_gaussian_patterns() {
    let values = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let product = |a: (i64, i64), b: (i64, i64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    for &a in &values {
        for &b in &values {
            for &c in &values {
                for &d in &values {
                    let ad = product(a, d);
                    let bc = product(b, c);
                    let determinant = (ad.0 - bc.0, ad.1 - bc.1);
                    let rank = if determinant != (0, 0) {
                        2
                    } else if [a, b, c, d].iter().any(|&value| value != (0, 0)) {
                        1
                    } else {
                        0
                    };
                    let source = matrix(2, 2, &[a, b, c, d]);
                    let (reduced, pivots) = source.metered_gaussian_rref(&mut Unbounded).unwrap();
                    assert_eq!(pivots.len(), rank, "source={source:?}");
                    if rank == 2 {
                        assert_eq!(reduced, Matrix::eye(2).unwrap());
                    }
                    if rank == 1 {
                        assert_eq!(pivots, vec![usize::from(a == (0, 0) && c == (0, 0))]);
                    }
                    assert_eq!(
                        source
                            .metered_gaussian_nullspace(&mut Unbounded)
                            .unwrap()
                            .len(),
                        2 - rank
                    );
                }
            }
        }
    }
}

#[test]
fn a_hand_built_inverse_transform_is_checked_without_generator_authority() {
    let source = matrix(2, 2, &[(0, 0), (0, 1), (1, 0), (0, 0)]);
    let proposed = GaussianRrefCandidate {
        reduced: Matrix::eye(2).unwrap(),
        transform: matrix(2, 2, &[(0, 0), (1, 0), (0, -1), (0, 0)]),
        inverse_transform: source.clone(),
        pivots: vec![0, 1],
    };
    source
        .verify_gaussian_rref_candidate(&proposed, &mut Unbounded)
        .unwrap();
}

#[test]
fn zero_transforms_cannot_erase_original_rank() {
    let source = matrix(2, 2, &[(1, 0), (0, 1), (0, 0), (1, 0)]);
    let forged = GaussianRrefCandidate {
        reduced: Matrix::zeros(2, 2).unwrap(),
        transform: Matrix::zeros(2, 2).unwrap(),
        inverse_transform: Matrix::zeros(2, 2).unwrap(),
        pivots: vec![],
    };
    assert!(matches!(
        source.verify_gaussian_rref_candidate(&forged, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn malformed_pivots_transforms_and_tiny_reduction_mutations_are_rejected() {
    let source = matrix(2, 2, &[(0, 1), (1, 0), (1, 0), (0, 0)]);
    let good = candidate(&source);
    let mut mutations = Vec::new();
    for pivots in [vec![], vec![1, 0], vec![0, 0], vec![usize::MAX]] {
        let mut changed = good.clone();
        changed.pivots = pivots;
        mutations.push(changed);
    }
    let mut changed = good.clone();
    changed.inverse_transform = Matrix::eye(2).unwrap();
    mutations.push(changed);
    let mut changed = good.clone();
    changed.transform = Matrix::eye(2).unwrap();
    mutations.push(changed);
    let mut changed = good.clone();
    changed.reduced = replace(
        &changed.reduced,
        1,
        Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(90))),
    );
    mutations.push(changed);
    let mut changed = good.clone();
    changed.reduced = Matrix::eye(1).unwrap();
    mutations.push(changed);
    for changed in mutations {
        assert!(matches!(
            source.verify_gaussian_rref_candidate(&changed, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
    let changed_source = replace(&source, 0, gaussian(0, -1, 1));
    assert!(matches!(
        changed_source.verify_gaussian_rref_candidate(&good, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn invertible_equivalent_forms_must_still_be_in_reduced_row_echelon_form() {
    let source = Matrix::eye(2).unwrap();
    let scaled = matrix(2, 2, &[(2, 0), (0, 0), (0, 0), (2, 0)]);
    let proposed = GaussianRrefCandidate {
        reduced: scaled.clone(),
        transform: scaled,
        inverse_transform: Matrix::new(
            2,
            2,
            vec![
                gaussian(1, 0, 2),
                gaussian(0, 0, 1),
                gaussian(0, 0, 1),
                gaussian(1, 0, 2),
            ],
        )
        .unwrap(),
        pivots: vec![0, 1],
    };
    assert!(matches!(
        source.verify_gaussian_rref_candidate(&proposed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn exact_affine_particular_and_kernel_keep_all_complex_solution_freedom() {
    let source = matrix(1, 2, &[(0, 1), (1, 0)]);
    let rhs = Matrix::eye(1).unwrap();
    let family = source
        .metered_gaussian_affine_solution(&rhs, &mut Unbounded)
        .unwrap()
        .unwrap();
    assert_eq!(family.coefficients(), &source);
    assert_eq!(family.rhs(), &rhs);
    assert_eq!(family.particular(), &matrix(2, 1, &[(0, -1), (0, 0)]));
    assert_eq!(family.kernel(), &matrix(2, 1, &[(0, 1), (1, 0)]));
    assert_eq!(family.free_columns(), &[1]);
}

#[test]
fn multiple_right_hand_sides_have_independent_row_major_parameters() {
    let source = matrix(2, 3, &[(1, 0), (0, 1), (0, 0), (0, 0), (0, 0), (0, 1)]);
    let rhs = matrix(2, 2, &[(1, 0), (2, 0), (0, 1), (0, 2)]);
    let family = source
        .metered_gaussian_affine_solution(&rhs, &mut Unbounded)
        .unwrap()
        .unwrap();
    assert_eq!(
        family.particular(),
        &matrix(3, 2, &[(1, 0), (2, 0), (0, 0), (0, 0), (1, 0), (2, 0)])
    );
    assert_eq!(family.kernel(), &matrix(3, 1, &[(0, -1), (1, 0), (0, 0)]));
    let (solution, parameters, free) = source
        .metered_gaussian_gauss_jordan_solve(&rhs, &mut Unbounded)
        .unwrap()
        .unwrap();
    assert_eq!(free, vec![1]);
    assert_eq!(
        parameters,
        Matrix::new(1, 2, vec![Expr::symbol("tau0"), Expr::symbol("tau1")]).unwrap()
    );
    assert_eq!(solution.get(1, 0).unwrap(), &Expr::symbol("tau0"));
    assert_eq!(solution.get(1, 1).unwrap(), &Expr::symbol("tau1"));
    assert_eq!(solution.get(2, 0).unwrap(), &Expr::from_i64(1));
    assert_eq!(solution.get(2, 1).unwrap(), &Expr::from_i64(2));
}

#[test]
fn inconsistency_in_any_rhs_column_refuses_the_whole_family() {
    let source = matrix(2, 1, &[(0, 1), (0, 2)]);
    let rhs = matrix(2, 2, &[(1, 0), (1, 0), (2, 0), (3, 0)]);
    let proposed = source
        .gaussian_affine_candidate_with_meter(&rhs, &mut Unbounded)
        .unwrap();
    assert!(proposed.particular.is_none());
    assert!(proposed.kernel.is_none());
    assert_eq!(
        source
            .verify_gaussian_affine_candidate(&rhs, &proposed, &mut Unbounded)
            .unwrap(),
        None
    );
    assert_eq!(
        source
            .metered_gaussian_gauss_jordan_solve(&rhs, &mut Unbounded)
            .unwrap(),
        None
    );
}

#[test]
fn residual_only_incomplete_or_noncanonical_families_are_rejected() {
    let source = matrix(1, 2, &[(0, 1), (1, 0)]);
    let rhs = Matrix::eye(1).unwrap();
    let good = source
        .gaussian_affine_candidate_with_meter(&rhs, &mut Unbounded)
        .unwrap();
    let mut mutations = Vec::new();
    let mut changed = good.clone();
    changed.kernel = Some(Matrix::zeros(2, 0).unwrap());
    mutations.push(changed);
    let mut changed = good.clone();
    changed.kernel = Some(Matrix::zeros(2, 1).unwrap());
    mutations.push(changed);
    let mut changed = good.clone();
    changed.kernel = Some(matrix(2, 1, &[(0, 2), (2, 0)]));
    mutations.push(changed);
    let mut changed = good.clone();
    changed.particular = Some(matrix(2, 1, &[(0, 0), (1, 0)]));
    mutations.push(changed);
    let mut changed = good.clone();
    changed.particular = Some(matrix(2, 1, &[(0, 1), (0, 0)]));
    mutations.push(changed);
    mutations.push(GaussianAffineCandidate {
        reduction: good.reduction.clone(),
        particular: None,
        kernel: None,
    });
    for changed in mutations {
        assert!(matches!(
            source.verify_gaussian_affine_candidate(&rhs, &changed, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
    let different_rhs = matrix(1, 1, &[(2, 0)]);
    assert!(matches!(
        source.verify_gaussian_affine_candidate(&different_rhs, &good, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn tiny_exact_complex_pivots_are_not_numerical_zero() {
    let tiny = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(100)));
    let source = Matrix::new(
        2,
        2,
        vec![
            Expr::Const(Constant::I) * tiny.clone(),
            Expr::from_i64(0),
            Expr::from_i64(0),
            tiny,
        ],
    )
    .unwrap();
    assert_eq!(
        source.metered_gaussian_rref(&mut Unbounded).unwrap(),
        (Matrix::eye(2).unwrap(), vec![0, 1])
    );
}

#[test]
fn zero_area_and_zero_rank_sources_retain_all_coordinate_freedom() {
    for (rows, columns) in [(0, 0), (0, 3), (4, 0), (2, 4)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        let (reduced, pivots) = source.metered_gaussian_rref(&mut Unbounded).unwrap();
        assert_eq!(reduced, source);
        assert!(pivots.is_empty());
        let nullspace = source.metered_gaussian_nullspace(&mut Unbounded).unwrap();
        assert_eq!(nullspace.len(), columns);
        for (index, vector) in nullspace.iter().enumerate() {
            assert_eq!((vector.rows(), vector.cols()), (columns, 1));
            for row in 0..columns {
                assert_eq!(
                    vector.get(row, 0).unwrap(),
                    &Expr::from_i64(i64::from(row == index))
                );
            }
        }
        let rhs = Matrix::zeros(rows, 2).unwrap();
        let family = source
            .metered_gaussian_affine_solution(&rhs, &mut Unbounded)
            .unwrap()
            .unwrap();
        assert_eq!(family.particular(), &Matrix::zeros(columns, 2).unwrap());
        assert_eq!(family.kernel(), &Matrix::eye(columns).unwrap());
    }
    assert_eq!(
        Matrix::zeros(2, 0)
            .unwrap()
            .metered_gaussian_affine_solution(&matrix(2, 1, &[(0, 0), (0, 1)]), &mut Unbounded)
            .unwrap(),
        None
    );
    let source = matrix(1, 2, &[(0, 1), (1, 0)]);
    let (solution, parameters, free) = source
        .metered_gaussian_gauss_jordan_solve(&Matrix::zeros(1, 0).unwrap(), &mut Unbounded)
        .unwrap()
        .unwrap();
    assert_eq!(solution, Matrix::zeros(2, 0).unwrap());
    assert_eq!(parameters, Matrix::zeros(1, 0).unwrap());
    assert_eq!(free, vec![1]);
}

#[test]
fn unknown_and_rhs_axes_have_separate_thirty_two_coordinate_bounds() {
    let mut coefficients = vec![Expr::from_i64(0); 32];
    coefficients[0] = Expr::Const(Constant::I);
    let source = Matrix::new(1, 32, coefficients).unwrap();
    let rhs = Matrix::new(1, 32, vec![Expr::from_i64(1); 32]).unwrap();
    let (solution, parameters, free) = source
        .metered_gaussian_gauss_jordan_solve(&rhs, &mut Unbounded)
        .unwrap()
        .unwrap();
    assert_eq!((solution.rows(), solution.cols()), (32, 32));
    assert_eq!((parameters.rows(), parameters.cols()), (31, 32));
    assert_eq!(free, (1..32).collect::<Vec<_>>());
    assert_eq!(solution.get(0, 31).unwrap(), &gaussian(0, -1, 1));
    assert_eq!(solution.get(31, 31).unwrap(), &Expr::symbol("tau991"));
}

#[test]
fn original_poles_and_unsupported_children_survive_zero_shortcuts() {
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    for entry in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
    ] {
        let source = Matrix::new(1, 1, vec![entry]).unwrap();
        assert_eq!(
            source.metered_gaussian_rref(&mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
        assert_eq!(
            Matrix::zeros(1, 1)
                .unwrap()
                .metered_gaussian_affine_solution(&source, &mut Unbounded),
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
                .metered_gaussian_rref(&mut Unbounded),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }
}

#[test]
fn malformed_witness_domains_and_oversized_sources_fail_closed() {
    let source = Matrix::eye(1).unwrap();
    let mut proposed = candidate(&source);
    proposed.inverse_transform = Matrix::new(
        1,
        1,
        vec![Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
        ])],
    )
    .unwrap();
    assert_eq!(
        source.verify_gaussian_rref_candidate(&proposed, &mut Unbounded),
        Err(MatrixError::DivisionByZero)
    );
    for source in [
        Matrix::zeros(0, 33).unwrap(),
        Matrix::zeros(33, 0).unwrap(),
        Matrix::new(1, 1, vec![Expr::Integer(BigInt::from(1) << 4096u32)]).unwrap(),
    ] {
        assert!(matches!(
            source.metered_gaussian_rref(&mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    assert!(matches!(
        Matrix::eye(1)
            .unwrap()
            .metered_gaussian_affine_solution(&Matrix::zeros(2, 1).unwrap(), &mut Unbounded),
        Err(MatrixError::ShapeMismatch(..))
    ));
}

#[test]
fn sparse_syntax_selection_does_not_materialize_a_large_symbolic_matrix() {
    use fsym_matrices::SparseMatrix;
    use std::collections::BTreeMap;

    let source = SparseMatrix::new(
        100_000,
        100_000,
        BTreeMap::from([((0, 0), Expr::symbol("x"))]),
    )
    .unwrap();
    assert!(
        !source
            .has_gaussian_rational_syntax_with_meter(&mut Unbounded)
            .unwrap()
    );
    let source = SparseMatrix::new(
        100_000,
        100_000,
        BTreeMap::from([((0, 0), Expr::Const(Constant::I))]),
    )
    .unwrap();
    assert!(
        source
            .has_gaussian_rational_syntax_with_meter(&mut Unbounded)
            .unwrap()
    );
    assert!(matches!(
        source.validate_gaussian_pseudoinverse_source_with_meter(&mut Unbounded),
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

fn execute(
    operation: u8,
    source: &Matrix,
    rhs: &Matrix,
    meter: &mut impl BudgetMeter,
) -> Result<(), MatrixError> {
    match operation {
        0 => {
            source.metered_gaussian_rref(meter)?;
        }
        1 => {
            source.metered_gaussian_nullspace(meter)?;
        }
        2 => {
            source.metered_gaussian_gauss_jordan_solve(rhs, meter)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[test]
fn generation_verification_and_parameter_publication_share_each_exact_budget() {
    let source = matrix(1, 2, &[(0, 1), (1, 1)]);
    let rhs = matrix(1, 2, &[(1, 0), (2, 0)]);
    for operation in 0..3 {
        let mut recorded = Recording::default();
        execute(operation, &source, &rhs, &mut recorded).unwrap();
        let limits = BudgetLimits {
            dimensions: recorded.charges,
            verifier_pool: 0,
        };
        execute(operation, &source, &rhs, &mut Budget::new(limits)).unwrap();
        for dimension in Dimension::ALL {
            if recorded.charges[dimension.index()] == 0 {
                continue;
            }
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                execute(operation, &source, &rhs, &mut Budget::new(short)),
                Err(MatrixError::ResourceLimit(_))
            ));
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
                execute(operation, &source, &rhs, &mut meter),
                Err(MatrixError::Cancelled)
            );
        }
    }
}

#[test]
fn independent_affine_replay_has_its_own_budget_and_final_cancellation_point() {
    let source = matrix(1, 2, &[(0, 1), (1, 1)]);
    let rhs = matrix(1, 1, &[(1, 0)]);
    let proposed = source
        .gaussian_affine_candidate_with_meter(&rhs, &mut Unbounded)
        .unwrap();
    let mut recorded = Recording::default();
    let expected = source
        .verify_gaussian_affine_candidate(&rhs, &proposed, &mut recorded)
        .unwrap();
    let mut limits = BudgetLimits {
        dimensions: recorded.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .verify_gaussian_affine_candidate(&rhs, &proposed, &mut Budget::new(limits))
            .unwrap(),
        expected
    );
    limits.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        source.verify_gaussian_affine_candidate(&rhs, &proposed, &mut Budget::new(limits)),
        Err(MatrixError::ResourceLimit(_))
    ));
    for cancel_at in [1, recorded.checkpoints / 2, recorded.checkpoints] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.verify_gaussian_affine_candidate(&rhs, &proposed, &mut meter),
            Err(MatrixError::Cancelled)
        );
    }
}
