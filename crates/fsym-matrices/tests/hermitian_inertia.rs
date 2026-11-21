//! Original complex sources, exact complete signs, and independent mutations.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_matrices::{HermitianMatrixRef, Matrix, MatrixError, verify_hermitian_inertia_candidate};
use fsym_proof_kernel::hermitian_inertia::verify_hermitian_inertia;
use std::sync::Arc;

fn gaussian(real: i64, imaginary: i64) -> Expr {
    Expr::Add(vec![
        Expr::from_i64(real),
        Expr::Mul(vec![Expr::from_i64(imaginary), Expr::Const(Constant::I)]),
    ])
}

fn matrix(n: usize, values: &[(i64, i64)]) -> Matrix {
    Matrix::new(
        n,
        n,
        values
            .iter()
            .map(|&(real, imaginary)| gaussian(real, imaginary))
            .collect(),
    )
    .unwrap()
}

fn example() -> Matrix {
    matrix(2, &[(1, 0), (1, 1), (1, -1), (3, 0)])
}

fn counts(matrix: &Matrix) -> (usize, usize, usize) {
    let result = matrix.hermitian_inertia().unwrap();
    let receipt = result.receipt();
    let replay =
        verify_hermitian_inertia(result.source(), result.witness(), &mut Unbounded).unwrap();
    assert_eq!(receipt, &replay);
    (receipt.positive(), receipt.negative(), receipt.zero())
}

#[test]
fn dense_and_sparse_complex_forms_have_complete_exact_inertia() {
    assert_eq!(counts(&example()), (2, 0, 0));
    let sparse = example().to_sparse().unwrap().hermitian_inertia().unwrap();
    assert_eq!(
        (
            sparse.receipt().positive(),
            sparse.receipt().negative(),
            sparse.receipt().zero()
        ),
        (2, 0, 0)
    );
    assert_eq!(
        counts(&matrix(2, &[(1, 0), (1, 1), (1, -1), (2, 0)])),
        (1, 0, 1)
    );
    assert_eq!(
        counts(&matrix(2, &[(-1, 0), (-1, -1), (-1, 1), (-2, 0)])),
        (0, 1, 1)
    );
    assert_eq!(
        counts(&matrix(2, &[(0, 0), (0, 1), (0, -1), (0, 0)])),
        (1, 1, 0)
    );
    // Arbitrary skew-Hermitian parts do not change Re(z* A z).
    assert_eq!(
        counts(&matrix(2, &[(0, 5), (7, 9), (-7, 9), (0, -3)])),
        (0, 0, 2)
    );
    // A itself has eigenvalues +1,+1; its Hermitian part is indefinite.
    assert_eq!(
        counts(&matrix(2, &[(1, 0), (0, 4), (0, 0), (1, 0)])),
        (1, 1, 0)
    );
    assert_eq!(counts(&Matrix::zeros(0, 0).unwrap()), (0, 0, 0));
    assert_eq!(counts(&Matrix::zeros(4, 4).unwrap()), (0, 0, 4));
}

#[test]
fn exhaustive_small_hermitian_forms_match_independent_trace_determinant_signs() {
    for a in -2i64..=2 {
        for d in -2i64..=2 {
            for real in -1i64..=1 {
                for imaginary in -1i64..=1 {
                    let determinant = a * d - real * real - imaginary * imaginary;
                    let trace = a + d;
                    let expected = if determinant < 0 {
                        (1, 1, 0)
                    } else if determinant > 0 {
                        if trace > 0 { (2, 0, 0) } else { (0, 2, 0) }
                    } else if trace > 0 {
                        (1, 0, 1)
                    } else if trace < 0 {
                        (0, 1, 1)
                    } else {
                        (0, 0, 2)
                    };
                    assert_eq!(
                        counts(&matrix(
                            2,
                            &[(a, 0), (real, imaginary), (real, -imaginary), (d, 0)]
                        )),
                        expected,
                        "a={a},d={d},real={real},imaginary={imaginary}"
                    );
                }
            }
        }
    }
}

#[test]
fn known_complex_congruences_preserve_every_signed_and_null_direction() {
    let n = 5;
    let diagonal = [1i64, -2, 0, 3, 0];
    for seed in 0..8i64 {
        let mut lower = vec![(0i64, 0i64); n * n];
        for row in 0..n {
            for column in 0..=row {
                lower[row * n + column] = if row == column {
                    (1, 0)
                } else {
                    (
                        (seed + row as i64 + 3 * column as i64) % 5 - 2,
                        (seed + 2 * row as i64 + column as i64) % 3 - 1,
                    )
                };
            }
        }
        // Construct U D U* with independent small integer complex arithmetic.
        // Unit triangular U is invertible, so Sylvester's law fixes the signs.
        let mut values = vec![(0i64, 0i64); n * n];
        for row in 0..n {
            for column in 0..n {
                let mut value = (0, 0);
                for k in 0..n {
                    let (a, b) = lower[row * n + k];
                    let (c, d) = lower[column * n + k];
                    value.0 += diagonal[k] * (a * c + b * d);
                    value.1 += diagonal[k] * (b * c - a * d);
                }
                values[(n - row - 1) * n + n - column - 1] = value;
            }
        }
        let original = matrix(n, &values);
        assert_eq!(counts(&original), (2, 1, 2));
        let sparse = original.to_sparse().unwrap().hermitian_inertia().unwrap();
        assert_eq!(
            (
                sparse.receipt().positive(),
                sparse.receipt().negative(),
                sparse.receipt().zero()
            ),
            (2, 1, 2)
        );
    }
}

#[test]
fn tiny_positive_schur_complement_survives_float_rounding() {
    let denominator = BigInt::one() << 128u32;
    let last = BigRational::new(&denominator * 2i64 + 1i64, denominator);
    let original = Matrix::new(
        2,
        2,
        vec![
            Expr::from_i64(1),
            gaussian(1, 1),
            gaussian(1, -1),
            Expr::Rational(last),
        ],
    )
    .unwrap();
    assert_eq!(counts(&original), (2, 0, 0));
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn original_expression_domains_and_dimensions_precede_all_shortcuts() {
    for expression in [
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
        power(power(Expr::from_i64(0), -1), 0),
        power(Expr::from_i64(0), 0),
    ] {
        let original = Matrix::new(1, 1, vec![expression]).unwrap();
        assert!(matches!(
            original.hermitian_inertia(),
            Err(MatrixError::DivisionByZero)
        ));
    }
    for expression in [
        Expr::Sym(Symbol::new("I")),
        Expr::Const(Constant::Infinity),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(Symbol::new("hidden"))]),
    ] {
        let original = Matrix::new(1, 1, vec![expression]).unwrap();
        assert!(matches!(
            original.hermitian_inertia(),
            Err(MatrixError::UnsupportedCertificateDomain)
        ));
    }
    assert!(matches!(
        Matrix::zeros(0, 1).unwrap().hermitian_inertia(),
        Err(MatrixError::NotSquare(0, 1))
    ));
    assert!(matches!(
        Matrix::eye(33).unwrap().hermitian_inertia(),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::eye(33)
            .unwrap()
            .to_sparse()
            .unwrap()
            .hermitian_inertia(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let huge = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4096u32)]).unwrap();
    assert!(matches!(
        huge.hermitian_inertia(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let inverse = power(gaussian(1, 1), -1);
    assert_eq!(
        counts(&Matrix::new(1, 1, vec![inverse]).unwrap()),
        (1, 0, 0)
    );
}

#[test]
fn native_candidate_acceptance_binds_original_skew_parts_and_witness() {
    let original = example();
    let candidate = original
        .hermitian_inertia_candidate_with_meter(&mut Unbounded)
        .unwrap();
    let mut changed = candidate.clone();
    changed.source.imaginary.entries[0].value = BigRational::from_integer(2.into());
    assert!(
        verify_hermitian_inertia_candidate(
            HermitianMatrixRef::Dense(&original),
            changed,
            &mut Unbounded
        )
        .is_err()
    );
    let other = matrix(2, &[(1, 1), (1, 1), (1, -1), (3, 0)]);
    assert!(
        verify_hermitian_inertia_candidate(
            HermitianMatrixRef::Dense(&other),
            candidate.clone(),
            &mut Unbounded
        )
        .is_err()
    );
    let mut changed = candidate.clone();
    changed.witness.diagonal.entries[0].value = -changed.witness.diagonal.entries[0].value.clone();
    assert!(
        verify_hermitian_inertia_candidate(
            HermitianMatrixRef::Dense(&original),
            changed,
            &mut Unbounded
        )
        .is_err()
    );
    let verified = verify_hermitian_inertia_candidate(
        HermitianMatrixRef::Dense(&original),
        candidate,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(verified.receipt().positive(), 2);
}

struct Checkpoints {
    used: usize,
    stop: usize,
}
impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.used += 1;
        if self.used >= self.stop {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn separate_phase_ledgers_cover_conversion_verification_and_publication() {
    let original = example();
    let limits = BudgetLimits::uniform(u64::MAX, 0);
    let (mut generator, mut verifier) = (Budget::new(limits), Budget::new(limits));
    original
        .hermitian_inertia_with_meters(&mut generator, &mut verifier)
        .unwrap();
    let generation = u64::MAX - generator.remaining(Dimension::ComputeSteps);
    let verification = u64::MAX - verifier.remaining(Dimension::ComputeSteps);
    assert!(generation > 100 && verification > 100);
    let mut too_small = limits;
    too_small.dimensions[Dimension::ComputeSteps.index()] = generation - 1;
    assert!(matches!(
        original.hermitian_inertia_with_meters(&mut Budget::new(too_small), &mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
    too_small.dimensions[Dimension::ComputeSteps.index()] = verification - 1;
    assert!(matches!(
        original.hermitian_inertia_with_meters(&mut Unbounded, &mut Budget::new(too_small)),
        Err(MatrixError::ResourceLimit(_))
    ));
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut zero = limits;
        zero.dimensions[dimension.index()] = 0;
        let mut untouched = Budget::new(limits);
        assert!(matches!(
            original.hermitian_inertia_with_meters(&mut Budget::new(zero), &mut untouched),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert_eq!(untouched.remaining(Dimension::ComputeSteps), u64::MAX);
        assert!(matches!(
            original.hermitian_inertia_with_meters(&mut Unbounded, &mut Budget::new(zero)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let (mut generated, mut verified) = (
        Checkpoints {
            used: 0,
            stop: usize::MAX,
        },
        Checkpoints {
            used: 0,
            stop: usize::MAX,
        },
    );
    original
        .hermitian_inertia_with_meters(&mut generated, &mut verified)
        .unwrap();
    for stop in [1, 25, generated.used] {
        assert!(matches!(
            original
                .hermitian_inertia_with_meters(&mut Checkpoints { used: 0, stop }, &mut Unbounded),
            Err(MatrixError::Cancelled)
        ));
    }
    for stop in [1, 25, verified.used] {
        assert!(matches!(
            original
                .hermitian_inertia_with_meters(&mut Unbounded, &mut Checkpoints { used: 0, stop }),
            Err(MatrixError::Cancelled)
        ));
    }
}

#[test]
fn maximum_complex_dimension_preserves_existing_rational_envelope() {
    assert_eq!(counts(&Matrix::eye(32).unwrap()), (32, 0, 0));
    let original = Matrix::eye(64).unwrap();
    assert_eq!(original.inertia().unwrap().receipt().positive(), 64);
}
