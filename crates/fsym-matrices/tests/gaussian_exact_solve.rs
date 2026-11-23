//! Exact complex systems: independent left-inverse witnesses, original
//! source counterexamples, degenerate shapes and cumulative publication caps.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, Constant, Expr};
use fsym_matrices::{GaussianLinearSystemCandidate, Matrix, MatrixError};
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

fn witness() -> (Matrix, Matrix, GaussianLinearSystemCandidate) {
    // A=[1;i], L=[1,0] is a left inverse that is deliberately NOT A+.
    // This independently authored witness must suffice for an exact solve.
    (
        matrix(2, 1, &[(1, 0), (0, 1)], 1),
        matrix(2, 1, &[(2, -1), (1, 2)], 1),
        GaussianLinearSystemCandidate {
            left_inverse: matrix(1, 2, &[(1, 0), (0, 0)], 1),
            solution: matrix(1, 1, &[(2, -1)], 1),
        },
    )
}

#[test]
fn independently_authored_left_inverse_proves_the_exact_solution() {
    let (a, b, candidate) = witness();
    a.verify_gaussian_linear_system_candidate(&b, &candidate, &mut Unbounded)
        .unwrap();
    assert_eq!(a.solve(&b).unwrap(), candidate.solution);
    assert_eq!(
        a.metered_gaussian_solve(&b, &mut Unbounded).unwrap(),
        candidate.solution
    );
}

#[test]
fn ordinary_square_solve_and_inverse_accept_exact_imaginary_pivots() {
    let a = matrix(2, 2, &[(0, 1), (1, 0), (0, 0), (1, 0)], 1);
    let b = matrix(2, 1, &[(3, 2), (1, 0)], 1);
    let x = matrix(2, 1, &[(2, -2), (1, 0)], 1);
    let inverse = matrix(2, 2, &[(0, -1), (0, 1), (0, 0), (1, 0)], 1);
    assert_eq!(a.solve(&b).unwrap(), x);
    assert_eq!(a.inverse().unwrap(), inverse);
    assert_eq!(a.metered_gaussian_inverse(&mut Unbounded).unwrap(), inverse);
}

#[test]
fn rational_coefficients_and_multiple_complex_rhs_share_the_exact_lane() {
    let a = matrix(2, 2, &[(1, 0), (2, 0), (0, 0), (1, 0)], 1);
    let b = matrix(2, 2, &[(3, 1), (1, -4), (1, 0), (0, 1)], 1);
    assert_eq!(
        a.solve(&b).unwrap(),
        matrix(2, 2, &[(1, 1), (1, -6), (1, 0), (0, 1)], 1)
    );
}

#[test]
fn hand_derived_integer_corpus_exercises_complex_pivots_and_rhs() {
    // Build every RHS using a separate integer-pair implementation, never
    // the candidate's Expr arithmetic, matrix multiplication or verifier.
    let multiply =
        |(ar, ai): (i64, i64), (br, bi): (i64, i64)| (ar * br - ai * bi, ar * bi + ai * br);
    for real in 1..=3 {
        for imaginary in -2..=2 {
            for shift in -1..=1 {
                let diagonal = (real, imaginary);
                let upper = (shift, 1);
                let x0 = (2, -1);
                let x1 = (-1, 2);
                let d0 = multiply(diagonal, x0);
                let u0 = multiply(upper, x1);
                let b1 = multiply((1, -1), x1);
                let a = matrix(2, 2, &[diagonal, upper, (0, 0), (1, -1)], 1);
                let b = matrix(2, 1, &[(d0.0 + u0.0, d0.1 + u0.1), b1], 1);
                assert_eq!(a.solve(&b).unwrap(), matrix(2, 1, &[x0, x1], 1));
            }
        }
    }
}

#[test]
fn least_squares_projection_never_satisfies_an_exact_solve_request() {
    let a = matrix(2, 1, &[(1, 0), (0, 2)], 1);
    let b = matrix(2, 1, &[(1, 0), (0, 0)], 1);
    assert_eq!(
        a.solve_least_squares(&b).unwrap(),
        matrix(1, 1, &[(1, 0)], 5)
    );
    assert_eq!(a.solve(&b), Err(MatrixError::InconsistentLinearSystem));
    let candidate = a
        .gaussian_linear_system_candidate_with_meter(&b, &mut Unbounded)
        .unwrap();
    assert!(matches!(
        a.verify_gaussian_linear_system_candidate(&b, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn residual_only_witnesses_cannot_claim_uniqueness() {
    let a = matrix(1, 2, &[(1, 0), (0, 1)], 1);
    let b = matrix(1, 1, &[(1, 0)], 1);
    // A X=B holds, but there is a complete one-dimensional complex kernel.
    let candidate = GaussianLinearSystemCandidate {
        left_inverse: matrix(2, 1, &[(1, 0), (0, 0)], 1),
        solution: matrix(2, 1, &[(1, 0), (0, 0)], 1),
    };
    assert!(matches!(
        a.verify_gaussian_linear_system_candidate(&b, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    assert_eq!(a.solve(&b), Err(MatrixError::SingularMatrix));
    let singular = matrix(2, 2, &[(1, 0), (0, 1), (2, 0), (0, 2)], 1);
    assert_eq!(singular.inverse(), Err(MatrixError::SingularMatrix));
    assert_eq!(
        singular.solve(&Matrix::zeros(2, 0).unwrap()),
        Err(MatrixError::SingularMatrix)
    );
}

#[test]
fn changes_to_original_sources_solution_and_left_inverse_are_rejected() {
    let (a, b, candidate) = witness();
    for (source, rhs, proposal) in [
        (
            matrix(2, 1, &[(1, 0), (0, 2)], 1),
            b.clone(),
            candidate.clone(),
        ),
        (
            a.clone(),
            matrix(2, 1, &[(2, -1), (2, 2)], 1),
            candidate.clone(),
        ),
        (
            a.clone(),
            b.clone(),
            GaussianLinearSystemCandidate {
                solution: matrix(1, 1, &[(1, -1)], 1),
                ..candidate.clone()
            },
        ),
        (
            a.clone(),
            b.clone(),
            GaussianLinearSystemCandidate {
                left_inverse: matrix(1, 2, &[(1, 0), (1, 0)], 1),
                ..candidate.clone()
            },
        ),
        (
            a,
            b,
            GaussianLinearSystemCandidate {
                left_inverse: Matrix::zeros(2, 1).unwrap(),
                ..candidate
            },
        ),
    ] {
        assert!(
            source
                .verify_gaussian_linear_system_candidate(&rhs, &proposal, &mut Unbounded)
                .is_err()
        );
    }
}

#[test]
fn empty_rhs_keeps_uniqueness_and_empty_unknowns_keep_consistency() {
    let (a, _, _) = witness();
    assert_eq!(
        a.solve(&Matrix::zeros(2, 0).unwrap()).unwrap(),
        Matrix::zeros(1, 0).unwrap()
    );
    for (rows, columns) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
        assert_eq!(
            Matrix::zeros(rows, 0)
                .unwrap()
                .metered_gaussian_solve(&Matrix::zeros(rows, columns).unwrap(), &mut Unbounded)
                .unwrap(),
            Matrix::zeros(0, columns).unwrap()
        );
    }
    assert_eq!(
        Matrix::zeros(2, 0)
            .unwrap()
            .solve(&matrix(2, 1, &[(0, 1), (0, 0)], 1)),
        Err(MatrixError::InconsistentLinearSystem)
    );
    assert_eq!(
        Matrix::zeros(0, 0)
            .unwrap()
            .metered_gaussian_inverse(&mut Unbounded)
            .unwrap(),
        Matrix::zeros(0, 0).unwrap()
    );
}

#[test]
fn original_undefined_sources_cannot_disappear_in_zero_products() {
    let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
    let invalid = Matrix::new(1, 1, vec![Expr::Mul(vec![Expr::from_i64(0), pole])]).unwrap();
    let one = Matrix::eye(1).unwrap();
    assert_eq!(invalid.solve(&one), Err(MatrixError::DivisionByZero));
    assert_eq!(one.solve(&invalid), Err(MatrixError::DivisionByZero));
    assert_eq!(invalid.inverse(), Err(MatrixError::DivisionByZero));
    assert_eq!(
        invalid.solve(&Matrix::zeros(1, 0).unwrap()),
        Err(MatrixError::DivisionByZero)
    );
}

#[test]
fn bounded_syntax_selection_preserves_symbolic_inverse_and_domains() {
    let symbolic = Matrix::new(
        2,
        2,
        vec![
            Expr::symbol("a"),
            Expr::from_i64(1),
            Expr::from_i64(1),
            Expr::from_i64(0),
        ],
    )
    .unwrap();
    assert!(
        !symbolic
            .has_gaussian_rational_syntax_with_meter(&mut Unbounded)
            .unwrap()
    );
    let inverse = symbolic.inverse().unwrap();
    assert_eq!(inverse.get(0, 0).unwrap(), &Expr::from_i64(0));
    assert_eq!(inverse.get(0, 1).unwrap(), &Expr::from_i64(1));
    assert_eq!(inverse.get(1, 0).unwrap(), &Expr::from_i64(1));
    let closed = Matrix::new(
        1,
        1,
        vec![Expr::Pow(
            Arc::new(gaussian(1, 1, 1)),
            Arc::new(Expr::from_i64(-1)),
        )],
    )
    .unwrap();
    assert!(
        closed
            .has_gaussian_rational_syntax_with_meter(&mut Unbounded)
            .unwrap()
    );
    assert_eq!(closed.inverse().unwrap(), matrix(1, 1, &[(1, 1)], 1));
}

#[test]
fn unsupported_malformed_and_oversized_inputs_refuse_before_publication() {
    let oversized = Matrix::new(33, 1, vec![gaussian(0, 1, 1); 33]).unwrap();
    assert!(matches!(
        oversized.solve(&Matrix::zeros(33, 0).unwrap()),
        Err(MatrixError::ResourceLimit(_))
    ));
    let huge = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4097u32)]).unwrap();
    assert!(matches!(
        huge.metered_gaussian_inverse(&mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
    let symbolic = Matrix::new(1, 1, vec![Expr::symbol("x")]).unwrap();
    assert_eq!(
        symbolic.solve(&Matrix::eye(1).unwrap()),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
    assert!(matches!(
        witness().0.solve(&Matrix::zeros(1, 0).unwrap()),
        Err(MatrixError::ShapeMismatch(..))
    ));
    assert!(matches!(
        witness().0.metered_gaussian_inverse(&mut Unbounded),
        Err(MatrixError::NotSquare(..))
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
fn all_exact_solve_and_inverse_work_is_cumulative_through_publication() {
    let a = matrix(2, 2, &[(0, 1), (1, 0), (0, 0), (1, 0)], 1);
    let b = matrix(2, 1, &[(3, 2), (1, 0)], 1);
    for inverse in [false, true] {
        let run = |meter: &mut Recording| {
            if inverse {
                a.metered_gaussian_inverse(meter)
            } else {
                a.metered_gaussian_solve(&b, meter)
            }
        };
        let mut measured = Recording::default();
        let expected = run(&mut measured).unwrap();
        let limits = BudgetLimits {
            dimensions: measured.charges,
            verifier_pool: 0,
        };
        let exact = if inverse {
            a.metered_gaussian_inverse(&mut Budget::new(limits))
        } else {
            a.metered_gaussian_solve(&b, &mut Budget::new(limits))
        };
        assert_eq!(exact.unwrap(), expected);
        for dimension in Dimension::ALL {
            if limits.dimensions[dimension.index()] == 0 {
                continue;
            }
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            let stopped = if inverse {
                a.metered_gaussian_inverse(&mut Budget::new(short))
            } else {
                a.metered_gaussian_solve(&b, &mut Budget::new(short))
            };
            assert!(matches!(stopped, Err(MatrixError::ResourceLimit(_))));
        }
        for cancel_at in [
            1,
            measured.checkpoints / 2,
            measured.checkpoints - 1,
            measured.checkpoints,
        ] {
            assert_eq!(
                run(&mut Recording {
                    cancel_at: Some(cancel_at),
                    ..Default::default()
                }),
                Err(MatrixError::Cancelled)
            );
        }
    }
}

#[test]
fn independent_checker_is_cancelled_even_at_its_final_safe_point() {
    let (a, b, candidate) = witness();
    let mut measured = Recording::default();
    a.verify_gaussian_linear_system_candidate(&b, &candidate, &mut measured)
        .unwrap();
    assert_eq!(
        a.verify_gaussian_linear_system_candidate(
            &b,
            &candidate,
            &mut Recording {
                cancel_at: Some(measured.checkpoints),
                ..Default::default()
            }
        ),
        Err(MatrixError::Cancelled)
    );
    let mut limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    limits.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        a.verify_gaussian_linear_system_candidate(&b, &candidate, &mut Budget::new(limits)),
        Err(MatrixError::ResourceLimit(_))
    ));
}
