//! Complete Jordan chains, independently checked similarity and terminal limits.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{
    MAX_RATIONAL_JORDAN_DIMENSION, Matrix, MatrixError, RationalJordanBlock,
    RationalJordanCandidate, verify_rational_jordan_candidate,
};

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}

fn expr(value: &BigRational) -> Expr {
    if value.is_integer() {
        Expr::Integer(value.to_integer())
    } else {
        Expr::Rational(value.clone())
    }
}

fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        values.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn jordan(blocks: &[(BigRational, usize)]) -> Matrix {
    let n: usize = blocks.iter().map(|(_, size)| size).sum();
    let mut values = vec![Expr::from_i64(0); n * n];
    let mut offset = 0;
    for (eigenvalue, size) in blocks {
        for local in 0..*size {
            let at = offset + local;
            values[at * n + at] = expr(eigenvalue);
            if local + 1 < *size {
                values[at * n + at + 1] = Expr::from_i64(1);
            }
        }
        offset += size;
    }
    Matrix::new(n, n, values).unwrap()
}

/// Two independently invertible integer shears mix chain coordinates.
fn conjugate(source: &Matrix) -> Matrix {
    let n = source.rows();
    let mut upper = Matrix::eye(n).unwrap().data().to_vec();
    let mut lower = upper.clone();
    let mut upper_inverse = upper.clone();
    let mut lower_inverse = upper.clone();
    for index in 1..n {
        let scale = (index % 3 + 1) as i64;
        upper[index] = Expr::from_i64(scale);
        upper_inverse[index] = Expr::from_i64(-scale);
        lower[index * n] = Expr::from_i64(1 - scale);
        lower_inverse[index * n] = Expr::from_i64(scale - 1);
    }
    let upper = Matrix::new(n, n, upper).unwrap();
    let lower = Matrix::new(n, n, lower).unwrap();
    let upper_inverse = Matrix::new(n, n, upper_inverse).unwrap();
    let lower_inverse = Matrix::new(n, n, lower_inverse).unwrap();
    upper
        .matmul(&lower)
        .unwrap()
        .matmul(source)
        .unwrap()
        .matmul(&lower_inverse)
        .unwrap()
        .matmul(&upper_inverse)
        .unwrap()
}

fn check(source: &Matrix, spectrum: &[(BigRational, usize)], expected: &[(BigRational, usize)]) {
    let result = source.rational_jordan(spectrum).unwrap();
    assert_eq!(result.matrix(), source);
    assert_eq!(result.jordan(), &jordan(expected));
    assert_eq!(
        source.matmul(result.transform()).unwrap(),
        result.transform().matmul(result.jordan()).unwrap()
    );
    assert_eq!(
        result.inverse().matmul(result.transform()).unwrap(),
        Matrix::eye(source.rows()).unwrap()
    );
    assert_eq!(
        result
            .transform()
            .matmul(result.jordan())
            .unwrap()
            .matmul(result.inverse())
            .unwrap(),
        *source
    );
    let accepted =
        verify_rational_jordan_candidate(source, result.certificate(), &mut Unbounded).unwrap();
    assert_eq!(accepted, result);
    let (transform, form, inverse) = accepted.into_parts();
    assert_eq!(transform, *result.transform());
    assert_eq!(form, *result.jordan());
    assert_eq!(inverse, *result.inverse());
}

#[test]
fn repeated_nontrivial_blocks_do_not_reuse_the_smaller_kernel_as_independent_vectors() {
    let expected = [(q(0), 2), (q(0), 2)];
    let source = jordan(&expected);
    check(&source, &[(q(0), 4)], &expected);
    check(&conjugate(&source), &[(q(0), 4)], &expected);
}

#[test]
fn mixed_block_sizes_retain_every_generalized_chain_level() {
    let expected = [(q(5), 3), (q(5), 2), (q(5), 1)];
    check(&jordan(&expected), &[(q(5), 6)], &expected);
    check(&conjugate(&jordan(&expected)), &[(q(5), 6)], &expected);
}

fn partitions(
    remaining: usize,
    largest: usize,
    prefix: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        out.push(prefix.clone());
        return;
    }
    for part in (1..=remaining.min(largest)).rev() {
        prefix.push(part);
        partitions(remaining - part, part, prefix, out);
        prefix.pop();
    }
}

#[test]
fn every_nilpotent_block_partition_through_dimension_seven_is_complete() {
    for n in 1..=7 {
        let mut cases = Vec::new();
        partitions(n, n, &mut Vec::new(), &mut cases);
        for partition in cases {
            let expected: Vec<_> = partition.into_iter().map(|size| (q(0), size)).collect();
            check(&conjugate(&jordan(&expected)), &[(q(0), n)], &expected);
        }
    }
}

#[test]
fn distinct_eigenvalues_and_unordered_hints_have_canonical_exact_block_order() {
    let source = jordan(&[(q(3), 2), (q(-2), 1), (q(0), 2)]);
    let hints = [(q(3), 2), (q(0), 2), (q(-2), 1)];
    let expected = [(q(-2), 1), (q(0), 2), (q(3), 2)];
    check(&source, &hints, &expected);
    check(&conjugate(&source), &hints, &expected);
}

#[test]
fn rational_eigenvalues_and_nonsymmetric_rational_coordinates_are_exact() {
    let half = BigRational::new(1.into(), 2.into());
    let third = BigRational::new((-2).into(), 3.into());
    let expected = [(third.clone(), 1), (half.clone(), 2)];
    check(
        &conjugate(&jordan(&expected)),
        &[(half, 2), (third, 1)],
        &expected,
    );
    let scaled = Matrix::new(
        2,
        2,
        vec![
            expr(&BigRational::new(1.into(), 2.into())),
            expr(&BigRational::new(2.into(), 7.into())),
            Expr::from_i64(0),
            expr(&BigRational::new(1.into(), 2.into())),
        ],
    )
    .unwrap();
    check(
        &scaled,
        &[(BigRational::new(1.into(), 2.into()), 2)],
        &[(BigRational::new(1.into(), 2.into()), 2)],
    );
}

#[test]
fn empty_zero_scalar_and_maximum_dimension_matrices_keep_complete_dimensions() {
    check(&Matrix::zeros(0, 0).unwrap(), &[], &[]);
    check(
        &Matrix::zeros(3, 3).unwrap(),
        &[(q(0), 3)],
        &[(q(0), 1), (q(0), 1), (q(0), 1)],
    );
    check(
        &Matrix::eye(2).unwrap(),
        &[(q(1), 2)],
        &[(q(1), 1), (q(1), 1)],
    );
    let n = MAX_RATIONAL_JORDAN_DIMENSION;
    check(&jordan(&[(q(0), n)]), &[(q(0), n)], &[(q(0), n)]);
}

#[test]
fn invalid_or_incomplete_spectral_hints_never_publish_partial_forms() {
    let source = jordan(&[(q(1), 3), (q(2), 1)]);
    for hints in [
        vec![],
        vec![(q(1), 3)],
        vec![(q(1), 5)],
        vec![(q(1), 0), (q(2), 4)],
        vec![(q(1), 2), (q(1), 2)],
        vec![(q(7), 4)],
        vec![(q(1), 2), (q(2), 2)],
        vec![(q(1), usize::MAX)],
    ] {
        assert!(
            matches!(
                source.rational_jordan(&hints),
                Err(MatrixError::InvalidCertificate(_))
            ),
            "{hints:?}"
        );
    }
    // A nonrational spectrum cannot be smuggled in through rational hints.
    assert!(
        matrix(2, 2, &[0, -1, 1, 0])
            .rational_jordan(&[(q(0), 2)])
            .is_err()
    );
}

#[test]
fn independent_checker_accepts_a_valid_alternate_transform() {
    let source = jordan(&[(q(4), 2)]);
    let candidate = RationalJordanCandidate {
        blocks: vec![RationalJordanBlock {
            eigenvalue: q(4),
            size: 2,
        }],
        transform: matrix(2, 2, &[2, 0, 0, 2]),
        inverse: Matrix::new(
            2,
            2,
            vec![
                Expr::rational(1, 2).unwrap(),
                Expr::from_i64(0),
                Expr::from_i64(0),
                Expr::rational(1, 2).unwrap(),
            ],
        )
        .unwrap(),
    };
    let checked = verify_rational_jordan_candidate(&source, &candidate, &mut Unbounded).unwrap();
    assert_eq!(checked.transform(), &candidate.transform);
    assert_eq!(checked.jordan(), &source);
}

#[test]
fn true_chain_relations_without_independence_cannot_certify_completeness() {
    let source = jordan(&[(q(0), 2), (q(0), 2)]);
    let mut forged = source
        .rational_jordan(&[(q(0), 4)])
        .unwrap()
        .certificate()
        .clone();
    // AP=PJ holds trivially for zero P, but it represents no Jordan basis.
    forged.transform = Matrix::zeros(4, 4).unwrap();
    forged.inverse = Matrix::zeros(4, 4).unwrap();
    assert!(verify_rational_jordan_candidate(&source, &forged, &mut Unbounded).is_err());
    // Two copies of the same valid chain also satisfy AP=PJ but are incomplete.
    forged.transform = matrix(4, 4, &[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    forged.inverse = Matrix::eye(4).unwrap();
    assert!(verify_rational_jordan_candidate(&source, &forged, &mut Unbounded).is_err());
}

#[test]
fn checker_rejects_wrong_blocks_original_source_transform_and_inverse() {
    let source = jordan(&[(q(1), 2), (q(1), 2)]);
    let valid = source.rational_jordan(&[(q(1), 4)]).unwrap();
    for change in 0..8 {
        let mut forged = valid.certificate().clone();
        match change {
            0 => forged.blocks.clear(),
            1 => forged.blocks[0].size = 0,
            2 => forged.blocks[0].size = usize::MAX,
            3 => {
                forged.blocks = vec![RationalJordanBlock {
                    eigenvalue: q(1),
                    size: 4,
                }]
            }
            4 => forged.blocks[0].eigenvalue = q(2),
            5 => forged.transform = Matrix::zeros(4, 4).unwrap(),
            6 => forged.inverse = Matrix::zeros(4, 4).unwrap(),
            7 => forged.inverse = Matrix::zeros(3, 3).unwrap(),
            _ => unreachable!(),
        }
        assert!(
            verify_rational_jordan_candidate(&source, &forged, &mut Unbounded).is_err(),
            "mutation {change}"
        );
    }
    let changed_source = jordan(&[(q(2), 2), (q(2), 2)]);
    assert!(
        verify_rational_jordan_candidate(&changed_source, valid.certificate(), &mut Unbounded)
            .is_err()
    );
}

#[test]
fn checker_requires_canonical_block_order_even_for_valid_similarity() {
    for blocks in [vec![(q(1), 1), (q(1), 2)], vec![(q(2), 1), (q(1), 2)]] {
        let source = jordan(&blocks);
        let candidate = RationalJordanCandidate {
            blocks: blocks
                .into_iter()
                .map(|(eigenvalue, size)| RationalJordanBlock { eigenvalue, size })
                .collect(),
            transform: Matrix::eye(3).unwrap(),
            inverse: Matrix::eye(3).unwrap(),
        };
        assert!(verify_rational_jordan_candidate(&source, &candidate, &mut Unbounded).is_err());
    }
}

#[test]
fn source_admission_preserves_held_syntax_shapes_and_coefficient_limits() {
    for entry in [
        Expr::symbol("parameter"),
        Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")]),
        Expr::Const(Constant::I),
    ] {
        let source = Matrix::new(1, 1, vec![entry]).unwrap();
        assert_eq!(
            source.rational_jordan(&[(q(0), 1)]).unwrap_err(),
            MatrixError::UnsupportedCertificateDomain
        );
    }
    for (rows, columns) in [(0, 1), (1, 0), (2, 3)] {
        assert!(matches!(
            Matrix::zeros(rows, columns).unwrap().rational_jordan(&[]),
            Err(MatrixError::NotSquare(..))
        ));
    }
    assert!(matches!(
        Matrix::eye(17).unwrap().rational_jordan(&[(q(1), 17)]),
        Err(MatrixError::ResourceLimit(_))
    ));
    let largest = BigInt::one() << 4095u32;
    let value = BigRational::from_integer(largest.clone());
    check(
        &Matrix::new(1, 1, vec![Expr::Integer(largest)]).unwrap(),
        &[(value.clone(), 1)],
        &[(value, 1)],
    );
    let too_large = BigInt::one() << 4096u32;
    assert!(matches!(
        Matrix::new(1, 1, vec![Expr::Integer(too_large.clone())])
            .unwrap()
            .rational_jordan(&[(q(0), 1)]),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::eye(1)
            .unwrap()
            .rational_jordan(&[(BigRational::from_integer(too_large), 1)]),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn intermediate_coefficient_growth_refuses_instead_of_publishing_unbounded_chains() {
    let denominator = BigInt::one() << 3000u32;
    let small = Expr::Rational(BigRational::new(BigInt::one(), denominator));
    let mut entries = vec![Expr::from_i64(0); 16];
    entries[1] = small.clone();
    entries[6] = small.clone();
    entries[11] = small;
    let source = Matrix::new(4, 4, entries).unwrap();
    assert!(matches!(
        source.rational_jordan(&[(q(0), 4)]),
        Err(MatrixError::ResourceLimit(_))
    ));
    let checked = Matrix::eye(1)
        .unwrap()
        .rational_jordan(&[(q(1), 1)])
        .unwrap();
    let mut forged = checked.certificate().clone();
    forged.blocks[0].eigenvalue = BigRational::from_integer(BigInt::one() << 8192u32);
    assert!(matches!(
        verify_rational_jordan_candidate(checked.matrix(), &forged, &mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[derive(Default)]
struct Meter {
    used: [u64; 3],
    checkpoints: usize,
    cancel_at: Option<usize>,
}
impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
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
fn independent_phase_budgets_cover_admission_arithmetic_and_publication() {
    let source = jordan(&[(q(2), 2)]);
    let spectrum = [(q(2), 2)];
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    source
        .metered_rational_jordan(&spectrum, &mut generator, &mut verifier)
        .unwrap();
    for (dimension, index) in [
        (Dimension::ComputeSteps, 0),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 2),
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = generator.used[index] - 1;
        assert!(matches!(
            source.metered_rational_jordan(&spectrum, &mut Budget::new(limits), &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = verifier.used[index] - 1;
        assert!(matches!(
            source.metered_rational_jordan(&spectrum, &mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let mut untouched = Meter::default();
    assert!(
        source
            .metered_rational_jordan(
                &spectrum,
                &mut Budget::new(BudgetLimits::uniform(1, 0)),
                &mut untouched
            )
            .is_err()
    );
    assert_eq!(untouched.used, [0; 3]);
    assert_eq!(untouched.checkpoints, 0);
}

#[test]
fn cancellation_is_terminal_at_every_safe_point_of_both_phases() {
    let source = Matrix::eye(1).unwrap();
    let spectrum = [(q(1), 1)];
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    source
        .metered_rational_jordan(&spectrum, &mut generator, &mut verifier)
        .unwrap();
    for stop in 1..=generator.checkpoints {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Meter::default()
        };
        assert_eq!(
            source
                .metered_rational_jordan(&spectrum, &mut cancelled, &mut Unbounded)
                .unwrap_err(),
            MatrixError::Cancelled
        );
    }
    for stop in 1..=verifier.checkpoints {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Meter::default()
        };
        assert_eq!(
            source
                .metered_rational_jordan(&spectrum, &mut Unbounded, &mut cancelled)
                .unwrap_err(),
            MatrixError::Cancelled
        );
    }
}
