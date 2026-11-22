//! Hand-derived Q(i) pseudoinverses and independent Hermitian adversaries.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, Constant, Expr};
use fsym_matrices::{
    Matrix, MatrixError, PseudoinverseCertificate, verify_pseudoinverse_certificate,
};
use fsym_proof_kernel::hermitian_inertia::{HermitianInertiaError, read_gaussian_rational_matrix};

fn gaussian(real: i64, imaginary: i64, denominator: i64) -> Expr {
    let real = Expr::rational(real, denominator).unwrap();
    if imaginary == 0 {
        return real;
    }
    let imaginary = if imaginary == denominator {
        Expr::Const(Constant::I)
    } else {
        Expr::Mul(vec![
            Expr::rational(imaginary, denominator).unwrap(),
            Expr::Const(Constant::I),
        ])
    };
    if real.is_zero() {
        imaginary
    } else {
        Expr::Add(vec![real, imaginary])
    }
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

fn assert_inverse(source: &Matrix, expected: Matrix) {
    let actual = source.pseudoinverse().unwrap();
    assert_eq!(actual, expected);
    source
        .verify_gaussian_pseudoinverse(&actual, &mut Unbounded)
        .unwrap();
    verify_pseudoinverse_certificate(source, &PseudoinverseCertificate { inverse: actual })
        .unwrap();
}

#[test]
fn scalar_complex_inverses_are_exact_and_conjugated() {
    assert_inverse(&matrix(1, 1, &[(0, 1)], 1), matrix(1, 1, &[(0, -1)], 1));
    assert_inverse(&matrix(1, 1, &[(1, 1)], 1), matrix(1, 1, &[(1, -1)], 2));
    assert_inverse(&matrix(1, 1, &[(3, -4)], 1), matrix(1, 1, &[(3, 4)], 25));
    let held = Matrix::new(
        1,
        1,
        vec![Expr::Pow(
            Box::new(gaussian(1, 1, 1)),
            Box::new(Expr::from_i64(-1)),
        )],
    )
    .unwrap();
    assert_inverse(&held, matrix(1, 1, &[(1, 1)], 1));
}

#[test]
fn published_gaussian_values_match_ordinary_canonical_expression_arithmetic() {
    for (real, imaginary) in [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (3, -4)] {
        let denominator = real * real + imaginary * imaginary;
        let expected = Expr::rational(real, denominator).unwrap()
            + Expr::rational(-imaginary, denominator).unwrap() * Expr::Const(Constant::I);
        let actual = matrix(1, 1, &[(real, imaginary)], 1)
            .pseudoinverse()
            .unwrap();
        assert_eq!(actual.get(0, 0).unwrap(), &expected);
    }
}

#[test]
fn isotropic_complex_row_and_column_require_adjoint_not_transpose() {
    // A A^T = 0 for this nonzero row; A A* = 2.
    let row = matrix(1, 2, &[(1, 0), (0, 1)], 1);
    assert_inverse(&row, matrix(2, 1, &[(1, 0), (0, -1)], 2));
    assert_inverse(&row.transpose(), matrix(1, 2, &[(1, 0), (0, -1)], 2));
}

#[test]
fn rank_one_rectangular_formula_retains_every_complex_component() {
    // A = u v^T, u=(1+i,2-i,-i), v=(i,1-i).
    // ||u||^2 ||v||^2=8*3, so A+ = A*/24.
    let source = matrix(
        3,
        2,
        &[(-1, 1), (2, 0), (1, 2), (1, -3), (1, 0), (-1, -1)],
        1,
    );
    let expected = matrix(
        2,
        3,
        &[(-1, -1), (1, -2), (1, 0), (2, 0), (1, 3), (-1, 1)],
        24,
    );
    assert_inverse(&source, expected.clone());
    assert_inverse(&source.transpose(), expected.transpose());
    assert_eq!(expected.pseudoinverse().unwrap(), source);
}

#[test]
fn full_column_and_row_rank_rectangles_have_the_expected_inverse() {
    let source = matrix(3, 2, &[(1, 0), (0, 0), (0, 1), (0, 0), (0, 0), (1, 1)], 1);
    let expected = matrix(2, 3, &[(1, 0), (0, -1), (0, 0), (0, 0), (0, 0), (1, -1)], 2);
    assert_inverse(&source, expected.clone());
    assert_inverse(&source.transpose(), expected.transpose());
    assert_inverse(
        &matrix(2, 2, &[(1, 1), (2, 0), (0, 0), (1, -1)], 1),
        matrix(2, 2, &[(1, -1), (-2, 0), (0, 0), (1, 1)], 2),
    );
}

fn product(left: (i64, i64), right: (i64, i64)) -> (i64, i64) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

#[test]
fn exhaustive_small_matrices_match_cofactor_or_rank_one_closed_forms() {
    let values = [(0, 0), (1, 0), (0, 1), (1, 1)];
    for a in values {
        for b in values {
            for c in values {
                for d in values {
                    let ad = product(a, d);
                    let bc = product(b, c);
                    let determinant = (ad.0 - bc.0, ad.1 - bc.1);
                    let determinant_norm = determinant.0.pow(2) + determinant.1.pow(2);
                    let squared_norm = [a, b, c, d].iter().map(|v| v.0.pow(2) + v.1.pow(2)).sum();
                    let expected = if determinant_norm != 0 {
                        let conjugate = (determinant.0, -determinant.1);
                        matrix(
                            2,
                            2,
                            &[
                                product(d, conjugate),
                                product((-b.0, -b.1), conjugate),
                                product((-c.0, -c.1), conjugate),
                                product(a, conjugate),
                            ],
                            determinant_norm,
                        )
                    } else if squared_norm != 0 {
                        // Every nonzero singular 2x2 complex matrix has rank one.
                        matrix(
                            2,
                            2,
                            &[(a.0, -a.1), (c.0, -c.1), (b.0, -b.1), (d.0, -d.1)],
                            squared_norm,
                        )
                    } else {
                        Matrix::zeros(2, 2).unwrap()
                    };
                    let source = matrix(2, 2, &[a, b, c, d], 1);
                    assert_eq!(
                        source.pseudoinverse().unwrap(),
                        expected,
                        "{a:?},{b:?},{c:?},{d:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn all_four_penrose_identities_reject_separate_complex_mutants() {
    let source = matrix(2, 2, &[(0, 1), (0, 0), (0, 0), (0, 0)], 1);
    for (values, identity) in [
        ([(0, 0), (0, 0), (0, 0), (0, 0)], "A X A != A"),
        ([(0, -1), (0, 0), (0, 0), (1, 0)], "X A X != X"),
        ([(0, -1), (1, 0), (0, 0), (0, 0)], "A X is not Hermitian"),
        ([(0, -1), (0, 0), (1, 0), (0, 0)], "X A is not Hermitian"),
    ] {
        let candidate = matrix(2, 2, &values, 1);
        assert_eq!(
            source.verify_gaussian_pseudoinverse(&candidate, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(identity.into())),
        );
    }
}

#[test]
fn symmetric_projectors_do_not_satisfy_complex_hermitian_obligations() {
    // X = A^T/(A^T A) satisfies the two reflexive identities, and AX is
    // symmetric, but fails its required conjugate-transpose identity.
    let source = matrix(2, 1, &[(1, 0), (0, 2)], 1);
    let wrong_transpose = matrix(1, 2, &[(-1, 0), (0, -2)], 3);
    assert_eq!(
        source.verify_gaussian_pseudoinverse(&wrong_transpose, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "A X is not Hermitian".into()
        )),
    );
    // P^2=P, trace(P)=1 and off-diagonal conjugacy holds. Only checking
    // off-diagonal entries would miss the two nonreal diagonal entries.
    let projector = matrix(2, 2, &[(4, 3), (5, 0), (5, 0), (4, -3)], 8);
    assert_eq!(
        projector.verify_gaussian_pseudoinverse(&projector, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "A X is not Hermitian".into()
        )),
    );
}

#[test]
fn tiny_exact_imaginary_mutation_cannot_pass_any_tolerance() {
    let source = matrix(1, 1, &[(0, 1)], 1);
    let tiny = Expr::Rational(fsym_core::BigRational::new(
        BigInt::from(1),
        BigInt::from(10).pow(80),
    ));
    let candidate = Matrix::new(1, 1, vec![Expr::Add(vec![gaussian(0, -1, 1), tiny])]).unwrap();
    assert!(matches!(
        source.verify_gaussian_pseudoinverse(&candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_)),
    ));
}

#[test]
fn literal_rational_fast_path_and_explicit_metered_reference_agree() {
    let source = matrix(3, 2, &[(1, 0), (2, 0), (2, 0), (4, 0), (3, 0), (6, 0)], 1);
    assert_eq!(
        source.pseudoinverse().unwrap(),
        source
            .metered_gaussian_pseudoinverse(&mut Unbounded)
            .unwrap()
    );
    // Existing literal-rational zero-area behavior is not capped by the
    // deliberately bounded complex source reader.
    assert_eq!(
        Matrix::zeros(0, 262_144).unwrap().pseudoinverse().unwrap(),
        Matrix::zeros(262_144, 0).unwrap()
    );
}

#[test]
fn zero_area_and_held_zero_sources_preserve_exact_transposed_shape() {
    for (rows, cols) in [(0, 0), (0, 3), (4, 0), (2, 3)] {
        let source = Matrix::zeros(rows, cols).unwrap();
        let expected = Matrix::zeros(cols, rows).unwrap();
        assert_eq!(
            source
                .metered_gaussian_pseudoinverse(&mut Unbounded)
                .unwrap(),
            expected
        );
    }
    let held_zero = Expr::Add(vec![Expr::Const(Constant::I), gaussian(0, -1, 1)]);
    assert_inverse(
        &Matrix::new(1, 2, vec![held_zero.clone(), held_zero]).unwrap(),
        Matrix::zeros(2, 1).unwrap(),
    );
    let zero = Matrix::zeros(1, 1).unwrap();
    let false_inverse = matrix(1, 1, &[(0, 1)], 1);
    assert_eq!(
        zero.verify_gaussian_pseudoinverse(&false_inverse, &mut Unbounded),
        Err(MatrixError::InvalidCertificate("X A X != X".into()))
    );
}

#[test]
fn empty_shapes_accept_strict_recorded_budgets_without_zero_charges() {
    for (rows, cols) in [(0, 0), (0, 3), (4, 0)] {
        let source = Matrix::zeros(rows, cols).unwrap();
        let mut measured = Recording::default();
        let expected = source
            .metered_gaussian_pseudoinverse(&mut measured)
            .unwrap();
        let limits = BudgetLimits {
            dimensions: measured.charges,
            verifier_pool: 0,
        };
        assert_eq!(
            source
                .metered_gaussian_pseudoinverse(&mut Budget::new(limits))
                .unwrap(),
            expected
        );
        let mut replay = Recording::default();
        source
            .verify_gaussian_pseudoinverse(&expected, &mut replay)
            .unwrap();
        source
            .verify_gaussian_pseudoinverse(
                &expected,
                &mut Budget::new(BudgetLimits {
                    dimensions: replay.charges,
                    verifier_pool: 0,
                }),
            )
            .unwrap();
    }
}

#[test]
fn original_poles_and_unsupported_children_survive_zero_simplifications() {
    let pole = Expr::Pow(Box::new(Expr::from_i64(0)), Box::new(Expr::from_i64(-1)));
    let sources = [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Box::new(pole.clone()), Box::new(Expr::from_i64(0))),
        Expr::Pow(
            Box::new(Expr::Add(vec![
                Expr::Const(Constant::I),
                gaussian(0, -1, 1),
            ])),
            Box::new(Expr::from_i64(0)),
        ),
    ];
    for expression in sources {
        let source = Matrix::new(1, 1, vec![expression]).unwrap();
        assert_eq!(source.pseudoinverse(), Err(MatrixError::DivisionByZero));
        assert_eq!(
            source.validate_gaussian_pseudoinverse_source_with_meter(&mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
    }
    let symbolic = Matrix::new(
        1,
        1,
        vec![Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("x")])],
    )
    .unwrap();
    assert_eq!(
        symbolic.pseudoinverse(),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
    let output_pole = Matrix::new(1, 1, vec![Expr::Mul(vec![Expr::from_i64(0), pole])]).unwrap();
    assert_eq!(
        Matrix::zeros(1, 1)
            .unwrap()
            .verify_gaussian_pseudoinverse(&output_pole, &mut Unbounded),
        Err(MatrixError::DivisionByZero)
    );
}

#[test]
fn original_source_is_rechecked_separately_from_the_candidate() {
    let original = matrix(1, 2, &[(1, 0), (0, 1)], 1);
    let candidate = original.pseudoinverse().unwrap();
    let changed = matrix(1, 2, &[(1, 0), (0, -1)], 1);
    assert!(matches!(
        changed.verify_gaussian_pseudoinverse(&candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    assert!(matches!(
        original.verify_gaussian_pseudoinverse(&Matrix::zeros(2, 2).unwrap(), &mut Unbounded,),
        Err(MatrixError::ShapeMismatch(..))
    ));
}

#[test]
fn source_and_intermediate_bounds_refuse_before_expensive_work() {
    let oversized_axis = Matrix::new(33, 1, vec![Expr::Const(Constant::I); 33]).unwrap();
    assert!(matches!(
        oversized_axis.pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let oversized = Matrix::new(
        1,
        1,
        vec![Expr::Add(vec![
            Expr::Integer(BigInt::from(1) << 4096),
            Expr::Const(Constant::I),
        ])],
    )
    .unwrap();
    assert!(matches!(
        oversized.pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let intermediate = Matrix::new(
        1,
        1,
        vec![Expr::Pow(
            Box::new(Expr::Integer(BigInt::from(1) << 2000)),
            Box::new(Expr::from_i64(64)),
        )],
    )
    .unwrap();
    assert!(matches!(
        intermediate.pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let too_many = Matrix::new(
        1,
        1,
        vec![Expr::Add(vec![Expr::Const(Constant::I); 16_385])],
    )
    .unwrap();
    assert!(matches!(
        too_many.pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let power = Matrix::new(
        1,
        1,
        vec![Expr::Pow(
            Box::new(Expr::Const(Constant::I)),
            Box::new(Expr::from_i64(65)),
        )],
    )
    .unwrap();
    assert!(matches!(
        power.pseudoinverse(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn rectangular_admission_preserves_the_square_inertia_contract() {
    let value = Expr::Const(Constant::I);
    let rectangle = Matrix::new(1, 2, vec![Expr::from_i64(0), value.clone()]).unwrap();
    rectangle
        .validate_gaussian_pseudoinverse_source_with_meter(&mut Unbounded)
        .unwrap();
    assert_inverse(&rectangle, matrix(2, 1, &[(0, 0), (0, -1)], 1));
    assert!(matches!(
        read_gaussian_rational_matrix(1, 2, std::iter::once(((0, 1), &value)), &mut Unbounded,),
        Err(HermitianInertiaError::NotSquare(1, 2))
    ));
    assert!(
        read_gaussian_rational_matrix(
            1,
            1,
            [((0, 0), &value), ((0, 0), &value)].into_iter(),
            &mut Unbounded,
        )
        .is_err()
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
fn one_cumulative_budget_covers_proposal_replay_and_final_publication() {
    let source = matrix(1, 1, &[(1, 1)], 1);
    let mut measured = Recording::default();
    let expected = source
        .metered_gaussian_pseudoinverse(&mut measured)
        .unwrap();
    let mut replay = Recording::default();
    source
        .verify_gaussian_pseudoinverse(&expected, &mut replay)
        .unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .metered_gaussian_pseudoinverse(&mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(
                matches!(
                    source.metered_gaussian_pseudoinverse(&mut Budget::new(short)),
                    Err(MatrixError::ResourceLimit(_))
                ),
                "{dimension:?}"
            );
        }
    }
    let verification_start = measured.checkpoints - replay.checkpoints - 1;
    for cancel_at in [
        1,
        verification_start / 2,
        verification_start + 1,
        verification_start + replay.checkpoints / 2,
        measured.checkpoints - 1,
        measured.checkpoints,
    ] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.metered_gaussian_pseudoinverse(&mut meter),
            Err(MatrixError::Cancelled),
            "safe point {cancel_at}"
        );
    }
    let mut cancelled = Recording {
        cancel_at: Some(1),
        ..Default::default()
    };
    assert_eq!(
        source.validate_gaussian_pseudoinverse_source_with_meter(&mut cancelled),
        Err(MatrixError::Cancelled)
    );
}

#[test]
fn verifier_budget_and_cancellation_remain_independent_public_controls() {
    let source = matrix(1, 2, &[(1, 0), (0, 1)], 1);
    let candidate = matrix(2, 1, &[(1, 0), (0, -1)], 2);
    let mut measured = Recording::default();
    source
        .verify_gaussian_pseudoinverse(&candidate, &mut measured)
        .unwrap();
    let mut limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    limits.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        source.verify_gaussian_pseudoinverse(&candidate, &mut Budget::new(limits)),
        Err(MatrixError::ResourceLimit(_))
    ));
    for cancel_at in [1, measured.checkpoints / 2, measured.checkpoints] {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            source.verify_gaussian_pseudoinverse(&candidate, &mut meter),
            Err(MatrixError::Cancelled)
        );
    }
}
