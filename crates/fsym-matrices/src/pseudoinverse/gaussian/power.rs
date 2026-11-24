//! Bounded exact integer powers with distinct generation and reference paths.
//!
//! Right-to-left binary exponentiation proposes a result using outer products.
//! A separate left-to-right reference replay re-reads the original expressions
//! and uses dot products instead. Negative powers also carry an inverse whose
//! two products with the original source must equal the identity. Source
//! admission, inversion, both power passes and publication share one ledger.

use super::{BigRational, BudgetMeter, GaussianMatrix, Matrix, MatrixError, Result, Work};
use num_traits::{One, Zero};

struct PowerCandidate {
    inverse: Option<Matrix>,
    power: Matrix,
}

fn invalid(message: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("Gaussian integer power: {message}"))
}

fn identity(size: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<GaussianMatrix> {
    let mut result = GaussianMatrix::zeros(size, size, work)?;
    for index in 0..size {
        work.tick(1)?;
        work.storage(64)?;
        result.entries[index * size + index].0 = BigRational::one();
    }
    Ok(result)
}

/// Generator-only multiplication accumulates rank-one outer products. The
/// reference checker below must never call this implementation.
fn outer_product(
    left: &GaussianMatrix,
    right: &GaussianMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianMatrix> {
    if left.cols != right.rows {
        return Err(MatrixError::ShapeMismatch(
            left.rows, left.cols, right.rows, right.cols,
        ));
    }
    let mut result = GaussianMatrix::zeros(left.rows, right.cols, work)?;
    for inner in 0..left.cols {
        for row in 0..left.rows {
            work.tick(1)?;
            let a = left.get(row, inner);
            if a.0.is_zero() && a.1.is_zero() {
                continue;
            }
            for column in 0..right.cols {
                work.tick(1)?;
                let b = right.get(inner, column);
                if b.0.is_zero() && b.1.is_zero() {
                    continue;
                }
                let product = work.multiply(a, b)?;
                let index = row * right.cols + column;
                result.entries[index] = work.add(&result.entries[index], &product)?;
            }
        }
    }
    Ok(result)
}

fn generate(
    matrix: &Matrix,
    exponent: i64,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<PowerCandidate> {
    // Even A**0 must interpret every original child before returning I.
    let original = GaussianMatrix::read(matrix, work)?;
    if original.rows != original.cols {
        return Err(MatrixError::NotSquare(original.rows, original.cols));
    }
    let mut result = identity(original.rows, work)?;
    let (mut factor, inverse) = if exponent < 0 {
        // Passing the outer Work keeps inverse admission, proposal and both
        // inverse checks inside this operation's cumulative intrinsic limits.
        let inverse = matrix.metered_gaussian_inverse(work)?;
        (GaussianMatrix::read(&inverse, work)?, Some(inverse))
    } else {
        (original, None)
    };
    let mut remaining = exponent.unsigned_abs();
    while remaining != 0 {
        work.tick(1)?;
        if remaining & 1 != 0 {
            result = outer_product(&result, &factor, work)?;
        }
        remaining >>= 1;
        if remaining != 0 {
            factor = outer_product(&factor, &factor, work)?;
        }
    }
    let power = result.into_matrix(work)?;
    work.tick(1)?;
    Ok(PowerCandidate { inverse, power })
}

fn verify_power(
    matrix: &Matrix,
    exponent: i64,
    candidate: &PowerCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    // This pass depends only on original expressions, candidate expressions
    // and bounded scalar arithmetic: no generator loop or product is reused.
    let original = GaussianMatrix::read(matrix, work)?;
    let proposed = GaussianMatrix::read(&candidate.power, work)?;
    if original.rows != original.cols
        || proposed.rows != original.rows
        || proposed.cols != original.cols
    {
        return Err(invalid("source or proposed power has the wrong shape"));
    }
    let mut expected = identity(original.rows, work)?;
    let base = if exponent < 0 {
        let inverse = candidate
            .inverse
            .as_ref()
            .ok_or_else(|| invalid("negative power has no inverse witness"))?;
        let inverse = GaussianMatrix::read(inverse, work)?;
        if inverse.rows != original.rows || inverse.cols != original.cols {
            return Err(invalid("inverse witness has the wrong shape"));
        }
        if !original.multiply(&inverse, work)?.equal(&expected, work)?
            || !inverse.multiply(&original, work)?.equal(&expected, work)?
        {
            return Err(invalid("inverse witness fails an original-source identity"));
        }
        inverse
    } else {
        if candidate.inverse.is_some() {
            return Err(invalid(
                "nonnegative power has an unexpected inverse witness",
            ));
        }
        original
    };
    let magnitude = exponent.unsigned_abs();
    // Scan the exponent in the opposite direction from generation. Each
    // step represents a longer prefix, independently of generated factors.
    for bit in (0..u64::BITS - magnitude.leading_zeros()).rev() {
        work.tick(1)?;
        expected = expected.multiply(&expected, work)?;
        if magnitude & (1u64 << bit) != 0 {
            expected = expected.multiply(&base, work)?;
        }
    }
    if !expected.equal(&proposed, work)? {
        return Err(invalid(
            "proposed power differs from original-source replay",
        ));
    }
    work.tick(1)
}

impl Matrix {
    /// Checked exact Q(i) integer powers, including zero and negative powers.
    ///
    /// Every original cell is admitted before a zero-exponent shortcut.
    /// Negative powers require a checked two-sided inverse. An independent
    /// reference pass uses different exponentiation and matrix-product loops.
    /// All phases share the Gaussian 20M-step/256MiB cumulative envelope:
    /// axes are at most 32, input/output components at most 4096 bits and
    /// intermediate components at most 8192 bits. The full i64 range is
    /// supported subject to these limits. No portable certificate is exported.
    pub fn metered_gaussian_power(
        &self,
        exponent: i64,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self> {
        let mut work = Work::new(meter);
        let candidate = generate(self, exponent, &mut work)?;
        verify_power(self, exponent, &candidate, &mut work)?;
        work.tick(1)?;
        Ok(candidate.power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, DIMENSION_COUNT, Dimension, MeterError, Unbounded};
    use fsym_core::{BigInt, Constant, Expr};
    use std::sync::Arc;

    fn scalar(real: i64, imaginary: i64, denominator: i64) -> Expr {
        let real = Expr::rational(real, denominator).unwrap();
        if imaginary == 0 {
            real
        } else {
            real + Expr::rational(imaginary, denominator).unwrap() * Expr::Const(Constant::I)
        }
    }

    fn matrix(entries: &[(i64, i64)], denominator: i64) -> Matrix {
        Matrix::new(
            2,
            2,
            entries
                .iter()
                .map(|&(real, imaginary)| scalar(real, imaginary, denominator))
                .collect(),
        )
        .unwrap()
    }

    fn assert_power(source: &Matrix, exponent: i64, expected: &Matrix) {
        let actual = source
            .metered_gaussian_power(exponent, &mut Unbounded)
            .unwrap();
        let mut caller = Unbounded;
        let mut work = Work::new(&mut caller);
        let actual = GaussianMatrix::read(&actual, &mut work).unwrap();
        let expected = GaussianMatrix::read(expected, &mut work).unwrap();
        assert!(actual.equal(&expected, &mut work).unwrap());
    }

    #[test]
    fn gaussian_jordan_block_obeys_binomial_identity_for_full_exponent_range() {
        let source = matrix(&[(1, 0), (0, 1), (0, 0), (1, 0)], 1);
        for exponent in [i64::MIN, -19, -1, 0, 1, 2, 17, i64::MAX] {
            assert_power(
                &source,
                exponent,
                &matrix(&[(1, 0), (0, exponent), (0, 0), (1, 0)], 1),
            );
        }
    }

    #[test]
    fn rational_rotation_has_period_four_for_signed_powers() {
        let source = matrix(&[(0, 0), (1, 0), (-1, 0), (0, 0)], 1);
        for exponent in [i64::MIN, -11, -2, -1, 0, 1, 2, 3, 12, i64::MAX] {
            let expected = match exponent.rem_euclid(4) {
                0 => matrix(&[(1, 0), (0, 0), (0, 0), (1, 0)], 1),
                1 => matrix(&[(0, 0), (1, 0), (-1, 0), (0, 0)], 1),
                2 => matrix(&[(-1, 0), (0, 0), (0, 0), (-1, 0)], 1),
                _ => matrix(&[(0, 0), (-1, 0), (1, 0), (0, 0)], 1),
            };
            assert_power(&source, exponent, &expected);
        }
    }

    #[test]
    fn distinct_eigenvalue_triangular_powers_match_closed_form() {
        let source = matrix(&[(2, 0), (0, 1), (0, 0), (3, 0)], 1);
        for exponent in -6i64..=6 {
            let magnitude = exponent.unsigned_abs() as u32;
            let two = 2i64.pow(magnitude);
            let three = 3i64.pow(magnitude);
            let expected = if exponent < 0 {
                matrix(
                    &[(three, 0), (0, two - three), (0, 0), (two, 0)],
                    two * three,
                )
            } else {
                matrix(&[(two, 0), (0, three - two), (0, 0), (three, 0)], 1)
            };
            assert_power(&source, exponent, &expected);
        }
    }

    #[test]
    fn empty_singular_and_rectangular_shapes_have_exact_semantics() {
        let empty = Matrix::zeros(0, 0).unwrap();
        for exponent in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert_power(&empty, exponent, &empty);
        }
        let nilpotent = matrix(&[(0, 0), (0, 1), (0, 0), (0, 0)], 1);
        assert_power(&nilpotent, 0, &Matrix::eye(2).unwrap());
        assert_power(&nilpotent, 2, &Matrix::zeros(2, 2).unwrap());
        assert_eq!(
            nilpotent.metered_gaussian_power(-1, &mut Unbounded),
            Err(MatrixError::SingularMatrix),
        );
        for shape in [(0, 1), (1, 0), (2, 3)] {
            let source = Matrix::zeros(shape.0, shape.1).unwrap();
            for exponent in [-1, 0, 2] {
                assert_eq!(
                    source.metered_gaussian_power(exponent, &mut Unbounded),
                    Err(MatrixError::NotSquare(shape.0, shape.1)),
                );
            }
        }
    }

    #[test]
    fn original_poles_and_unsupported_children_survive_zero_exponent() {
        let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
        for held in [
            Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
            Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
        ] {
            let source = Matrix::new(1, 1, vec![held]).unwrap();
            for exponent in [-1, 0, 1, 2] {
                assert_eq!(
                    source.metered_gaussian_power(exponent, &mut Unbounded),
                    Err(MatrixError::DivisionByZero),
                );
            }
        }
        for entry in [
            Expr::symbol("x"),
            Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("x")]),
        ] {
            let source = Matrix::new(1, 1, vec![entry]).unwrap();
            assert_eq!(
                source.metered_gaussian_power(0, &mut Unbounded),
                Err(MatrixError::UnsupportedCertificateDomain),
            );
        }
        let defined = Matrix::new(
            1,
            1,
            vec![Expr::Add(vec![Expr::from_i64(1), Expr::from_i64(1)])],
        )
        .unwrap();
        assert_power(
            &defined,
            -2,
            &Matrix::new(1, 1, vec![Expr::rational(1, 4).unwrap()]).unwrap(),
        );
    }

    #[test]
    fn source_output_and_intermediate_growth_are_bounded() {
        let over_input =
            Matrix::new(1, 1, vec![Expr::Integer(BigInt::from(1) << 4096u32)]).unwrap();
        assert!(matches!(
            over_input.metered_gaussian_power(0, &mut Unbounded),
            Err(MatrixError::ResourceLimit(_)),
        ));
        let two = Matrix::new(1, 1, vec![Expr::from_i64(2)]).unwrap();
        for exponent in [4096, i64::MAX] {
            assert!(matches!(
                two.metered_gaussian_power(exponent, &mut Unbounded),
                Err(MatrixError::ResourceLimit(_)),
            ));
        }
        assert!(
            Matrix::eye(33)
                .unwrap()
                .metered_gaussian_power(0, &mut Unbounded)
                .is_err()
        );
        let malformed = Matrix {
            rows: 1,
            cols: 1,
            data: Vec::new(),
        };
        assert!(matches!(
            malformed.metered_gaussian_power(0, &mut Unbounded),
            Err(MatrixError::InvalidStorageLength(..)),
        ));
    }

    #[test]
    fn independent_checker_rejects_wrong_power_shape_and_inverse() {
        let source = matrix(&[(2, 0), (0, 1), (0, 0), (3, 0)], 1);
        let mut caller = Unbounded;
        let mut work = Work::new(&mut caller);
        let mut candidate = generate(&source, 3, &mut work).unwrap();
        candidate.power.data[1] = Expr::from_i64(0);
        assert!(matches!(
            verify_power(&source, 3, &candidate, &mut work),
            Err(MatrixError::InvalidCertificate(_)),
        ));
        candidate.power = Matrix::eye(1).unwrap();
        assert!(matches!(
            verify_power(&source, 3, &candidate, &mut work),
            Err(MatrixError::InvalidCertificate(_)),
        ));
        let mut candidate = generate(&source, -2, &mut work).unwrap();
        candidate.inverse = Some(Matrix::eye(2).unwrap());
        assert!(matches!(
            verify_power(&source, -2, &candidate, &mut work),
            Err(MatrixError::InvalidCertificate(_)),
        ));
        candidate.inverse = None;
        assert!(matches!(
            verify_power(&source, -2, &candidate, &mut work),
            Err(MatrixError::InvalidCertificate(_)),
        ));
        let mut candidate = generate(&source, 0, &mut work).unwrap();
        candidate.inverse = Some(Matrix::eye(2).unwrap());
        assert!(matches!(
            verify_power(&source, 0, &candidate, &mut work),
            Err(MatrixError::InvalidCertificate(_)),
        ));
    }

    #[test]
    fn checker_reinterprets_source_and_every_candidate_expression() {
        let source = matrix(&[(1, 0), (0, 1), (0, 0), (1, 0)], 1);
        let mut caller = Unbounded;
        let mut work = Work::new(&mut caller);
        let mut candidate = generate(&source, 0, &mut work).unwrap();
        let pole = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1)));
        let mut invalid_source = source.clone();
        invalid_source.data[1] = Expr::Mul(vec![Expr::from_i64(0), pole.clone()]);
        assert_eq!(
            verify_power(&invalid_source, 0, &candidate, &mut work),
            Err(MatrixError::DivisionByZero),
        );
        candidate.power.data[1] = Expr::Mul(vec![Expr::from_i64(0), pole]);
        assert_eq!(
            verify_power(&source, 0, &candidate, &mut work),
            Err(MatrixError::DivisionByZero),
        );
    }

    #[derive(Default)]
    struct Recording {
        charges: [u64; DIMENSION_COUNT],
        checkpoints: usize,
        cancel_at: Option<usize>,
    }

    impl BudgetMeter for Recording {
        fn charge(
            &mut self,
            dimension: Dimension,
            amount: u64,
        ) -> std::result::Result<(), MeterError> {
            self.charges[dimension.index()] += amount;
            Ok(())
        }

        fn charge_batch(
            &mut self,
            charges: &[(Dimension, u64)],
        ) -> std::result::Result<(), MeterError> {
            for &(dimension, amount) in charges {
                self.charge(dimension, amount)?;
            }
            Ok(())
        }

        fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
            self.checkpoints += 1;
            if self.cancel_at == Some(self.checkpoints) {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn both_passes_inverse_and_publication_share_cumulative_limits() {
        let source = matrix(&[(1, 0), (0, 1), (0, 0), (1, 0)], 1);
        for exponent in [-3, 0, 7] {
            let mut recording = Recording::default();
            source
                .metered_gaussian_power(exponent, &mut recording)
                .unwrap();
            let limits = BudgetLimits {
                dimensions: recording.charges,
                verifier_pool: 0,
            };
            let mut exact = Budget::new(limits);
            source.metered_gaussian_power(exponent, &mut exact).unwrap();
            assert!(source.metered_gaussian_power(exponent, &mut exact).is_err());
            for dimension in [
                Dimension::ComputeSteps,
                Dimension::MemoryBytes,
                Dimension::AllocationCount,
            ] {
                assert!(recording.charges[dimension.index()] > 0);
                let mut short = limits;
                short.dimensions[dimension.index()] -= 1;
                assert!(matches!(
                    source.metered_gaussian_power(exponent, &mut Budget::new(short)),
                    Err(MatrixError::ResourceLimit(_)),
                ));
            }
        }
    }

    #[test]
    fn cancellation_stops_generation_checking_and_final_publication() {
        for source in [
            matrix(&[(1, 0), (0, 1), (0, 0), (1, 0)], 1),
            Matrix::zeros(0, 0).unwrap(),
        ] {
            for exponent in [-3, 0, 7] {
                let mut recording = Recording::default();
                source
                    .metered_gaussian_power(exponent, &mut recording)
                    .unwrap();
                for cancel_at in [
                    1,
                    recording.checkpoints / 3,
                    recording.checkpoints / 2,
                    recording.checkpoints - 1,
                    recording.checkpoints,
                ] {
                    let mut cancelled = Recording {
                        cancel_at: Some(cancel_at),
                        ..Default::default()
                    };
                    assert_eq!(
                        source.metered_gaussian_power(exponent, &mut cancelled),
                        Err(MatrixError::Cancelled),
                    );
                    assert_eq!(cancelled.checkpoints, cancel_at);
                }
            }
        }
    }
}
