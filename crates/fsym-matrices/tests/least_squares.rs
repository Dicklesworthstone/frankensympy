//! Complete exact least-squares families and independent rejection obligations.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{Matrix, MatrixError, verify_least_squares_candidate};

fn matrix(rows: usize, cols: usize, entries: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        cols,
        entries.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}
fn rational(rows: usize, cols: usize, entries: &[(i64, i64)]) -> Matrix {
    Matrix::new(
        rows,
        cols,
        entries
            .iter()
            .map(|&(n, d)| Expr::rational(n, d).unwrap())
            .collect(),
    )
    .unwrap()
}

#[test]
fn rank_deficient_tall_system_has_a_complete_minimum_norm_family() {
    // A = [1,2,3]^T [2,-1].  A+ = [2,-1]^T [1,2,3]/70.
    let a = matrix(3, 2, &[2, -1, 4, -2, 6, -3]);
    let b = matrix(3, 2, &[3, 1, 0, 2, 1, 0]);
    let checked = a.least_squares_family(&b).unwrap();
    assert_eq!(
        checked.minimum_norm(),
        &rational(2, 2, &[(6, 35), (1, 7), (-3, 35), (-1, 14)])
    );
    assert_eq!(
        checked.projector(),
        &rational(2, 2, &[(1, 5), (2, 5), (2, 5), (4, 5)])
    );
    assert_eq!(checked.coefficients(), &a);
    assert_eq!(checked.rhs(), &b);
    assert_eq!(checked.rank(), 1);
    for u in -2..=2 {
        for v in -2..=2 {
            let w = matrix(2, 2, &[u, v, 2 * u, -v]);
            let family = checked
                .minimum_norm()
                .add(&checked.projector().matmul(&w).unwrap())
                .unwrap();
            let residual = a.matmul(&family).unwrap().sub(&b).unwrap();
            assert_eq!(
                a.transpose().matmul(&residual).unwrap(),
                Matrix::zeros(2, 2).unwrap()
            );
        }
    }
}

#[test]
fn wide_exact_solutions_retain_every_free_direction() {
    let a = matrix(1, 3, &[1, 2, 3]);
    let b = matrix(1, 2, &[2, 4]);
    let checked = a.least_squares_family(&b).unwrap();
    assert_eq!(
        checked.minimum_norm(),
        &rational(3, 2, &[(1, 7), (2, 7), (2, 7), (4, 7), (3, 7), (6, 7)])
    );
    assert_eq!(
        checked.projector(),
        &rational(
            3,
            3,
            &[
                (13, 14),
                (-1, 7),
                (-3, 14),
                (-1, 7),
                (5, 7),
                (-3, 7),
                (-3, 14),
                (-3, 7),
                (5, 14)
            ]
        )
    );
    assert_eq!(a.matmul(checked.minimum_norm()).unwrap(), b);
    assert_eq!(checked.rank(), 1);
}

#[test]
fn exhaustive_two_by_two_inputs_match_closed_form_rank_cases() {
    let b = matrix(2, 2, &[2, -1, 3, 1]);
    for a in -1..=1 {
        for c in -1..=1 {
            for d in -1..=1 {
                for e in -1..=1 {
                    let input = matrix(2, 2, &[a, c, d, e]);
                    let det = a * e - c * d;
                    let norm = a * a + c * c + d * d + e * e;
                    let (minimum, projector, rank) = if det != 0 {
                        (
                            rational(
                                2,
                                2,
                                &[
                                    (2 * e - 3 * c, det),
                                    (-e - c, det),
                                    (3 * a - 2 * d, det),
                                    (a + d, det),
                                ],
                            ),
                            Matrix::zeros(2, 2).unwrap(),
                            2,
                        )
                    } else if norm != 0 {
                        (
                            rational(
                                2,
                                2,
                                &[
                                    (2 * a + 3 * d, norm),
                                    (-a + d, norm),
                                    (2 * c + 3 * e, norm),
                                    (-c + e, norm),
                                ],
                            ),
                            rational(
                                2,
                                2,
                                &[
                                    (norm - a * a - d * d, norm),
                                    (-a * c - d * e, norm),
                                    (-a * c - d * e, norm),
                                    (norm - c * c - e * e, norm),
                                ],
                            ),
                            1,
                        )
                    } else {
                        (Matrix::zeros(2, 2).unwrap(), Matrix::eye(2).unwrap(), 0)
                    };
                    let checked = input.least_squares_family(&b).unwrap();
                    assert_eq!(checked.minimum_norm(), &minimum, "{a},{c};{d},{e}");
                    assert_eq!(checked.projector(), &projector, "{a},{c};{d},{e}");
                    assert_eq!(checked.rank(), rank);
                }
            }
        }
    }
}

#[test]
fn full_rank_least_squares_and_existing_default_behavior_are_preserved() {
    let a = matrix(3, 2, &[1, 2, 2, 3, 3, 4]);
    let b = matrix(3, 1, &[8, 14, 18]);
    let expected = rational(2, 1, &[(5, 3), (10, 3)]);
    let checked = a.least_squares_family(&b).unwrap();
    assert_eq!(checked.minimum_norm(), &expected);
    assert_eq!(checked.projector(), &Matrix::zeros(2, 2).unwrap());
    assert_eq!(a.solve_least_squares(&b).unwrap(), expected);
    let singular = matrix(2, 2, &[1, 1, 2, 2]);
    assert!(matches!(
        singular.solve_least_squares(&matrix(2, 1, &[1, 0])),
        Err(MatrixError::SingularMatrix)
    ));
}

#[test]
fn zero_rank_zero_area_and_empty_rhs_have_complete_shapes() {
    for (rows, cols, rhs_columns) in [
        (0, 0, 0),
        (0, 0, 2),
        (0, 3, 2),
        (4, 0, 3),
        (3, 2, 2),
        (2, 3, 0),
    ] {
        let a = Matrix::zeros(rows, cols).unwrap();
        let b = Matrix::new(
            rows,
            rhs_columns,
            vec![Expr::from_i64(7); rows * rhs_columns],
        )
        .unwrap();
        let checked = a.least_squares_family(&b).unwrap();
        assert_eq!(
            checked.minimum_norm(),
            &Matrix::zeros(cols, rhs_columns).unwrap()
        );
        assert_eq!(checked.projector(), &Matrix::eye(cols).unwrap());
        assert_eq!(checked.rank(), 0);
    }
    let a = matrix(3, 2, &[1, 0, 0, 1, 1, 1]);
    let checked = a
        .least_squares_family(&Matrix::zeros(3, 0).unwrap())
        .unwrap();
    assert_eq!(checked.minimum_norm(), &Matrix::zeros(2, 0).unwrap());
    assert_eq!(checked.projector(), &Matrix::zeros(2, 2).unwrap());
}

#[test]
fn every_verifier_identity_is_necessary() {
    let a = matrix(2, 2, &[1, 0, 0, 0]);
    let b = matrix(2, 1, &[1, 3]);
    let good = a
        .least_squares_candidate_with_meter(&b, &mut Unbounded)
        .unwrap();
    verify_least_squares_candidate(&a, &b, good.clone(), &mut Unbounded).unwrap();
    for (projector, reason) in [
        (matrix(2, 2, &[0, 0, 1, 1]), "not symmetric"),
        (Matrix::eye(2).unwrap(), "range is not"),
        (Matrix::zeros(2, 2).unwrap(), "omits a complete"),
    ] {
        let mut mutant = good.clone();
        mutant.projector = projector;
        let error = verify_least_squares_candidate(&a, &b, mutant, &mut Unbounded).unwrap_err();
        assert!(error.to_string().contains(reason), "{error}");
    }
    for (minimum_norm, reason) in [
        (matrix(2, 1, &[1, 1]), "not minimum norm"),
        (matrix(2, 1, &[0, 0]), "does not minimize the residual"),
    ] {
        let mut mutant = good.clone();
        mutant.minimum_norm = minimum_norm;
        let error = verify_least_squares_candidate(&a, &b, mutant, &mut Unbounded).unwrap_err();
        assert!(error.to_string().contains(reason), "{error}");
    }
}

#[test]
fn kernel_proof_original_inputs_and_later_rhs_columns_are_bound_exactly() {
    let a = matrix(2, 2, &[1, 0, 0, 0]);
    let b = matrix(2, 2, &[1, 2, 3, 4]);
    let good = a
        .least_squares_candidate_with_meter(&b, &mut Unbounded)
        .unwrap();
    let mut wrong_kernel = good.clone();
    wrong_kernel.kernel = matrix(2, 2, &[2, 0, 0, 0])
        .affine_solution_candidate_with_meter(&Matrix::zeros(2, 1).unwrap(), &mut Unbounded)
        .unwrap();
    assert!(
        verify_least_squares_candidate(&a, &b, wrong_kernel, &mut Unbounded)
            .unwrap_err()
            .to_string()
            .contains("different coefficients")
    );
    let mut inhomogeneous = good.clone();
    inhomogeneous.kernel = a
        .affine_solution_candidate_with_meter(&matrix(2, 1, &[1, 0]), &mut Unbounded)
        .unwrap();
    assert!(
        verify_least_squares_candidate(&a, &b, inhomogeneous, &mut Unbounded)
            .unwrap_err()
            .to_string()
            .contains("not homogeneous")
    );
    let mut forged_proof = good.clone();
    forged_proof.kernel.witness.reduced[0] = BigRational::from(2_i64);
    assert!(verify_least_squares_candidate(&a, &b, forged_proof, &mut Unbounded).is_err());
    let mut late_rhs = good.clone();
    late_rhs.minimum_norm = matrix(2, 2, &[1, 0, 0, 0]);
    assert!(verify_least_squares_candidate(&a, &b, late_rhs, &mut Unbounded).is_err());
    assert!(
        verify_least_squares_candidate(
            &a,
            &matrix(2, 2, &[2, 2, 3, 4]),
            good.clone(),
            &mut Unbounded
        )
        .is_err()
    );
    let mut wrong_shape = good;
    wrong_shape.minimum_norm = Matrix::zeros(1, 2).unwrap();
    assert!(verify_least_squares_candidate(&a, &b, wrong_shape, &mut Unbounded).is_err());
}

#[test]
fn tiny_exact_pivots_and_mutations_never_use_a_tolerance() {
    let denominator = BigInt::from(10).pow(80);
    let epsilon = Expr::Rational(BigRational::new(1.into(), denominator.clone()));
    let a = Matrix::new(2, 1, vec![epsilon.clone(), Expr::from_i64(0)]).unwrap();
    let b = Matrix::new(2, 1, vec![epsilon, Expr::from_i64(1)]).unwrap();
    assert_eq!(
        a.least_squares_family(&b).unwrap().minimum_norm(),
        &matrix(1, 1, &[1])
    );
    let mut mutant = a
        .least_squares_candidate_with_meter(&b, &mut Unbounded)
        .unwrap();
    mutant.minimum_norm = Matrix::new(
        1,
        1,
        vec![Expr::Rational(BigRational::new(
            &denominator + BigInt::from(1),
            denominator,
        ))],
    )
    .unwrap();
    assert!(verify_least_squares_candidate(&a, &b, mutant, &mut Unbounded).is_err());
}

#[test]
fn whole_original_admission_precedes_zeros_and_resource_amplification() {
    let hidden = Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")]);
    for bad in [
        Expr::symbol("x"),
        hidden,
        Expr::Const(fsym_core::Constant::I),
    ] {
        let a = Matrix::new(1, 1, vec![bad.clone()]).unwrap();
        assert_eq!(
            a.least_squares_family(&Matrix::zeros(1, 1).unwrap())
                .unwrap_err(),
            MatrixError::UnsupportedCertificateDomain
        );
        let b = Matrix::new(1, 2, vec![Expr::from_i64(0), bad]).unwrap();
        assert_eq!(
            Matrix::zeros(1, 1)
                .unwrap()
                .least_squares_family(&b)
                .unwrap_err(),
            MatrixError::UnsupportedCertificateDomain
        );
    }
    let huge = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4096u32)]).unwrap();
    assert!(matches!(
        huge.least_squares_family(&Matrix::zeros(1, 1).unwrap()),
        Err(MatrixError::ResourceLimit(_))
    ));
    for (a, b) in [
        (Matrix::zeros(33, 1).unwrap(), Matrix::zeros(33, 1).unwrap()),
        (Matrix::zeros(0, 33).unwrap(), Matrix::zeros(0, 1).unwrap()),
        (Matrix::zeros(1, 1).unwrap(), Matrix::zeros(1, 33).unwrap()),
    ] {
        assert!(matches!(
            a.least_squares_family(&b),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    assert!(matches!(
        Matrix::eye(2)
            .unwrap()
            .least_squares_family(&Matrix::zeros(3, 1).unwrap()),
        Err(MatrixError::ShapeMismatch(..))
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
fn budgets_are_cumulative_across_gram_solves_all_rhs_and_reference_checks() {
    let a = matrix(3, 3, &[1, 2, 3, 2, 4, 6, 0, 1, 1]);
    let b = matrix(3, 2, &[1, 2, 0, 1, 3, 0]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    a.least_squares_family_with_meters(&b, &mut generator, &mut verifier)
        .unwrap();
    for (dimension, index) in [
        (Dimension::ComputeSteps, 0),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 2),
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = generator.used[index] - 1;
        let mut budget = Budget::new(limits);
        assert!(matches!(
            a.least_squares_family_with_meters(&b, &mut budget, &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = verifier.used[index] - 1;
        let mut budget = Budget::new(limits);
        assert!(matches!(
            a.least_squares_family_with_meters(&b, &mut Unbounded, &mut budget),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut unused = Meter::default();
    assert!(
        a.least_squares_family_with_meters(&b, &mut tiny, &mut unused)
            .is_err()
    );
    assert_eq!(unused.used, [0; 3]);
    assert_eq!(unused.checkpoints, 0);
}

#[test]
fn both_phases_cancel_before_work_during_solves_and_at_final_publication() {
    let a = matrix(3, 3, &[1, 2, 3, 2, 4, 6, 0, 1, 1]);
    let b = matrix(3, 2, &[1, 2, 0, 1, 3, 0]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    a.least_squares_family_with_meters(&b, &mut generator, &mut verifier)
        .unwrap();
    for stop in [
        1,
        generator.checkpoints / 2,
        generator.checkpoints - 1,
        generator.checkpoints,
    ] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert_eq!(
            a.least_squares_family_with_meters(&b, &mut cancelled, &mut Unbounded)
                .unwrap_err(),
            MatrixError::Cancelled,
            "generator {stop}"
        );
    }
    for stop in [
        1,
        verifier.checkpoints / 2,
        verifier.checkpoints - 1,
        verifier.checkpoints,
    ] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert_eq!(
            a.least_squares_family_with_meters(&b, &mut Unbounded, &mut cancelled)
                .unwrap_err(),
            MatrixError::Cancelled,
            "verifier {stop}"
        );
    }
}
