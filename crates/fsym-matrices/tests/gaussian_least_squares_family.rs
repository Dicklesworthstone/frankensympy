//! Complete complex least-squares families with independently authored
//! witnesses, exact counterexamples, original domains and cumulative limits.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, Constant, Expr};
use fsym_matrices::{GaussianLeastSquaresCandidate, Matrix, MatrixError};
use std::sync::Arc;

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

fn row_case() -> (Matrix, Matrix, GaussianLeastSquaresCandidate) {
    // A=(1,i), ||A||^2=2. Its complete kernel is spanned by (-i,1).
    (
        matrix(1, 2, &[(1, 0), (0, 1)], 1),
        matrix(1, 1, &[(1, 0)], 1),
        GaussianLeastSquaresCandidate {
            pseudoinverse: matrix(2, 1, &[(1, 0), (0, -1)], 2),
            minimum_norm: matrix(2, 1, &[(1, 0), (0, -1)], 2),
            projector: matrix(2, 2, &[(1, 0), (0, -1), (0, 1), (1, 0)], 2),
        },
    )
}

#[test]
fn an_independently_authored_complex_family_is_accepted_without_generation() {
    let (source, rhs, candidate) = row_case();
    let checked = source
        .verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded)
        .unwrap();
    assert_eq!(checked.coefficients(), &source);
    assert_eq!(checked.rhs(), &rhs);
    assert_eq!(checked.minimum_norm(), &candidate.minimum_norm);
    assert_eq!(checked.projector(), &candidate.projector);
    assert_eq!(
        source
            .metered_gaussian_least_squares_family(&rhs, &mut Unbounded)
            .unwrap(),
        checked
    );
}

#[test]
fn rank_deficient_inconsistent_sources_keep_the_complete_complex_kernel() {
    // A=u*v^T with u=(1,2), v=(1,i). A+=v* u^T/(2*5).
    // B=(1,0) is inconsistent. The LS residual is (-4/5,2/5).
    let source = matrix(2, 2, &[(1, 0), (0, 1), (2, 0), (0, 2)], 1);
    let rhs = matrix(2, 1, &[(1, 0), (0, 0)], 1);
    let checked = source
        .metered_gaussian_least_squares_family(&rhs, &mut Unbounded)
        .unwrap();
    assert_eq!(
        checked.minimum_norm(),
        &matrix(2, 1, &[(1, 0), (0, -1)], 10)
    );
    assert_eq!(checked.projector(), &row_case().2.projector);
    // The unique-solution API must still refuse this genuinely nonunique case.
    assert_eq!(
        source.metered_gaussian_solve_least_squares(&rhs, &mut Unbounded),
        Err(MatrixError::SingularMatrix)
    );
}

#[test]
fn multiple_complex_right_hand_sides_share_one_complete_family() {
    let source = matrix(2, 2, &[(1, 0), (0, 1), (2, 0), (0, 2)], 1);
    let rhs = matrix(2, 2, &[(1, 0), (0, 1), (0, 0), (2, -1)], 1);
    let checked = source
        .metered_gaussian_least_squares_family(&rhs, &mut Unbounded)
        .unwrap();
    assert_eq!(
        checked.minimum_norm(),
        &matrix(2, 2, &[(1, 0), (4, -1), (0, -1), (-1, -4)], 10)
    );
    assert_eq!(checked.projector(), &row_case().2.projector);
}

#[test]
fn full_column_rank_complex_families_have_no_free_directions() {
    let source = matrix(2, 1, &[(1, 0), (0, 2)], 1);
    let rhs = matrix(2, 1, &[(1, 0), (0, 0)], 1);
    let checked = source
        .metered_gaussian_least_squares_family(&rhs, &mut Unbounded)
        .unwrap();
    assert_eq!(checked.minimum_norm(), &matrix(1, 1, &[(1, 0)], 5));
    assert_eq!(checked.projector(), &Matrix::zeros(1, 1).unwrap());
}

#[test]
fn complex_family_agrees_with_the_existing_exact_rational_reference() {
    for source in [
        matrix(3, 2, &[(1, 0), (2, 0), (2, 0), (4, 0), (3, 0), (6, 0)], 1),
        matrix(3, 2, &[(1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0)], 1),
    ] {
        let rhs = matrix(3, 2, &[(1, 0), (2, 0), (3, 0), (1, 0), (-1, 0), (2, 0)], 3);
        let rational = source.least_squares_family(&rhs).unwrap();
        let gaussian = source
            .metered_gaussian_least_squares_family(&rhs, &mut Unbounded)
            .unwrap();
        assert_eq!(gaussian.minimum_norm(), rational.minimum_norm());
        assert_eq!(gaussian.projector(), rational.projector());
    }
}

#[test]
fn zero_and_empty_sources_retain_every_arbitrary_coordinate() {
    for (rows, unknowns, rhs_columns) in [(2, 3, 1), (0, 3, 2), (3, 0, 2), (2, 3, 0), (0, 0, 0)] {
        let source = Matrix::zeros(rows, unknowns).unwrap();
        let rhs = Matrix::new(
            rows,
            rhs_columns,
            vec![gaussian(1, 2, 3); rows * rhs_columns],
        )
        .unwrap();
        let checked = source
            .metered_gaussian_least_squares_family(&rhs, &mut Unbounded)
            .unwrap();
        assert_eq!(
            checked.minimum_norm(),
            &Matrix::zeros(unknowns, rhs_columns).unwrap()
        );
        assert_eq!(checked.projector(), &Matrix::eye(unknowns).unwrap());
        assert_eq!(checked.rhs(), &rhs);
    }
}

#[test]
fn nonzero_source_with_empty_rhs_still_requires_complete_kernel_evidence() {
    let (source, _, mut candidate) = row_case();
    let rhs = Matrix::zeros(1, 0).unwrap();
    candidate.minimum_norm = Matrix::zeros(2, 0).unwrap();
    let checked = source
        .verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded)
        .unwrap();
    assert_eq!(checked.minimum_norm(), &Matrix::zeros(2, 0).unwrap());
    candidate.projector = Matrix::zeros(2, 2).unwrap();
    assert_eq!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "least-squares kernel projector is incomplete".into()
        ))
    );
}

#[test]
fn residual_only_zero_projector_cannot_claim_a_complete_family() {
    let (source, rhs, mut candidate) = row_case();
    candidate.projector = Matrix::zeros(2, 2).unwrap();
    assert_eq!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "least-squares kernel projector is incomplete".into()
        ))
    );
}

#[test]
fn a_kernel_shifted_solution_is_not_the_minimum_norm_particular() {
    let (source, rhs, mut candidate) = row_case();
    // This still solves A*x=1, because the added vector (-i,1) is in ker(A).
    candidate.minimum_norm = matrix(2, 1, &[(1, -2), (2, -1)], 2);
    assert_eq!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "least-squares particular is not minimum norm".into()
        ))
    );
}

#[test]
fn each_projector_obligation_rejects_an_explicit_false_witness() {
    let source = Matrix::zeros(2, 2).unwrap();
    let rhs = Matrix::zeros(2, 1).unwrap();
    let mut candidate = GaussianLeastSquaresCandidate {
        pseudoinverse: Matrix::zeros(2, 2).unwrap(),
        minimum_norm: Matrix::zeros(2, 1).unwrap(),
        projector: matrix(2, 2, &[(1, 0), (0, 1), (0, 0), (0, 0)], 1),
    };
    assert_eq!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "least-squares kernel projector is not Hermitian".into()
        ))
    );
    candidate.projector = matrix(2, 2, &[(2, 0), (0, 0), (0, 0), (2, 0)], 1);
    assert_eq!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "least-squares kernel projector is not idempotent".into()
        ))
    );
    let (source, rhs, mut candidate) = row_case();
    candidate.projector = Matrix::eye(2).unwrap();
    assert_eq!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(
            "least-squares projector leaves the original kernel".into()
        ))
    );
}

#[test]
fn all_four_hermitian_penrose_obligations_are_required_by_family_acceptance() {
    let source = matrix(2, 2, &[(0, 1), (0, 0), (0, 0), (0, 0)], 1);
    let rhs = Matrix::zeros(2, 1).unwrap();
    for (values, identity) in [
        ([(0, 0), (0, 0), (0, 0), (0, 0)], "A X A != A"),
        ([(0, -1), (0, 0), (0, 0), (1, 0)], "X A X != X"),
        ([(0, -1), (1, 0), (0, 0), (0, 0)], "A X is not Hermitian"),
        ([(0, -1), (0, 0), (1, 0), (0, 0)], "X A is not Hermitian"),
    ] {
        let candidate = GaussianLeastSquaresCandidate {
            pseudoinverse: matrix(2, 2, &values, 1),
            minimum_norm: Matrix::zeros(2, 1).unwrap(),
            projector: matrix(2, 2, &[(0, 0), (0, 0), (0, 0), (1, 0)], 1),
        };
        assert_eq!(
            source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
            Err(MatrixError::InvalidCertificate(identity.into()))
        );
    }
}

#[test]
fn wrong_transpose_particular_and_tiny_exact_mutations_fail_original_normals() {
    let source = matrix(2, 1, &[(1, 0), (0, 2)], 1);
    let rhs = matrix(2, 1, &[(1, 0), (0, 0)], 1);
    let mut candidate = GaussianLeastSquaresCandidate {
        pseudoinverse: matrix(1, 2, &[(1, 0), (0, -2)], 5),
        minimum_norm: matrix(1, 1, &[(-1, 0)], 3),
        projector: Matrix::zeros(1, 1).unwrap(),
    };
    assert!(matches!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    candidate.minimum_norm = Matrix::new(
        1,
        1,
        vec![Expr::Add(vec![
            Expr::rational(1, 5).unwrap(),
            Expr::Rational(fsym_core::BigRational::new(
                BigInt::from(1),
                BigInt::from(10).pow(80),
            )),
        ])],
    )
    .unwrap();
    assert!(matches!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn certificates_are_bound_to_both_original_sources_and_every_shape() {
    let (source, rhs, candidate) = row_case();
    for (a, b) in [
        (matrix(1, 2, &[(1, 0), (0, 2)], 1), rhs.clone()),
        (source.clone(), matrix(1, 1, &[(2, 0)], 1)),
        (source.clone(), Matrix::zeros(1, 2).unwrap()),
        (Matrix::zeros(2, 2).unwrap(), rhs.clone()),
    ] {
        assert!(
            a.verify_gaussian_least_squares_candidate(&b, &candidate, &mut Unbounded)
                .is_err()
        );
    }
    let mut malformed = candidate;
    malformed.projector = Matrix::zeros(2, 1).unwrap();
    assert!(matches!(
        source.verify_gaussian_least_squares_candidate(&rhs, &malformed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    assert!(matches!(
        source.metered_gaussian_least_squares_family(&Matrix::zeros(2, 1).unwrap(), &mut Unbounded),
        Err(MatrixError::ShapeMismatch(..))
    ));
}

#[test]
fn original_domain_holes_cannot_disappear_in_zero_products_or_witnesses() {
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    let invalid = Matrix::new(1, 1, vec![Expr::Mul(vec![Expr::from_i64(0), pole])]).unwrap();
    let zero = Matrix::zeros(1, 1).unwrap();
    for (source, rhs) in [(&invalid, &zero), (&zero, &invalid)] {
        assert_eq!(
            source.metered_gaussian_least_squares_family(rhs, &mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
    }
    let candidate = GaussianLeastSquaresCandidate {
        pseudoinverse: zero.clone(),
        minimum_norm: invalid,
        projector: Matrix::eye(1).unwrap(),
    };
    assert_eq!(
        zero.verify_gaussian_least_squares_candidate(&zero, &candidate, &mut Unbounded),
        Err(MatrixError::DivisionByZero)
    );
    let hidden_symbol = Matrix::new(
        1,
        1,
        vec![Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("x")])],
    )
    .unwrap();
    assert_eq!(
        zero.metered_gaussian_least_squares_family(&hidden_symbol, &mut Unbounded),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
}

#[test]
fn held_exact_gaussian_sources_have_normalized_immutable_receipt_snapshots() {
    let held = Matrix::new(
        1,
        1,
        vec![Expr::Pow(
            Arc::new(gaussian(1, 1, 1)),
            Arc::new(Expr::from_i64(-1)),
        )],
    )
    .unwrap();
    let rhs = matrix(1, 1, &[(1, 0)], 1);
    let checked = held
        .metered_gaussian_least_squares_family(&rhs, &mut Unbounded)
        .unwrap();
    assert_eq!(checked.coefficients(), &matrix(1, 1, &[(1, -1)], 2));
    assert_eq!(checked.minimum_norm(), &matrix(1, 1, &[(1, 1)], 1));
    assert_eq!(checked.projector(), &Matrix::zeros(1, 1).unwrap());
}

#[test]
fn source_and_witness_height_and_dimension_bounds_precede_publication() {
    let source = matrix(1, 1, &[(0, 1)], 1);
    for rhs in [
        Matrix::new(1, 33, vec![Expr::Const(Constant::I); 33]).unwrap(),
        Matrix::new(1, 1, vec![Expr::Integer(BigInt::from(1) << 4096)]).unwrap(),
    ] {
        assert!(matches!(
            source.metered_gaussian_least_squares_family(&rhs, &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let (source, rhs, mut candidate) = row_case();
    candidate.projector = Matrix::zeros(33, 33).unwrap();
    assert!(matches!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Unbounded),
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
fn proposal_original_verification_and_publication_share_the_same_budget() {
    let (source, rhs, _) = row_case();
    let mut measured = Recording::default();
    let expected = source
        .metered_gaussian_least_squares_family(&rhs, &mut measured)
        .unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .metered_gaussian_least_squares_family(&rhs, &mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if limits.dimensions[dimension.index()] == 0 {
            continue;
        }
        let mut short = limits;
        short.dimensions[dimension.index()] -= 1;
        assert!(matches!(
            source.metered_gaussian_least_squares_family(&rhs, &mut Budget::new(short)),
            Err(MatrixError::ResourceLimit(_))
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
            source.metered_gaussian_least_squares_family(&rhs, &mut meter),
            Err(MatrixError::Cancelled),
            "checkpoint {cancel_at}"
        );
    }
}

#[test]
fn the_standalone_verifier_charges_and_cancels_every_phase_too() {
    let (source, rhs, candidate) = row_case();
    let mut measured = Recording::default();
    let expected = source
        .verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut measured)
        .unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        source
            .verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Budget::new(limits))
            .unwrap(),
        expected
    );
    let mut short = limits;
    short.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut Budget::new(short)),
        Err(MatrixError::ResourceLimit(_))
    ));
    let mut cancelled = Recording {
        cancel_at: Some(measured.checkpoints),
        ..Default::default()
    };
    assert_eq!(
        source.verify_gaussian_least_squares_candidate(&rhs, &candidate, &mut cancelled),
        Err(MatrixError::Cancelled)
    );
    let mut proposal = Recording::default();
    source
        .gaussian_least_squares_candidate_with_meter(&rhs, &mut proposal)
        .unwrap();
    assert!(proposal.charges[Dimension::ComputeSteps.index()] > 0);
    assert!(measured.charges[Dimension::ComputeSteps.index()] > 0);
}
