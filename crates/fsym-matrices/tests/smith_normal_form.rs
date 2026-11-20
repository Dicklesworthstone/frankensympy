//! Integer equivalence, determinantal divisors, source admission and stops.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_matrices::{
    Matrix, MatrixError, generate_smith_normal_form_candidate, verify_smith_normal_form_candidate,
};
use fsym_proof_kernel::integer_normal_form::verify_integer_smith;
use fsym_proof_kernel::sparse_rref::RationalSparseMatrix;

fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        values.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn dense(value: &RationalSparseMatrix) -> Matrix {
    let mut entries = vec![Expr::from_i64(0); value.rows * value.columns];
    for entry in &value.entries {
        assert!(entry.value.is_integer());
        entries[entry.row * value.columns + entry.column] =
            Expr::Integer(entry.value.numer().clone());
    }
    Matrix::new(value.rows, value.columns, entries).unwrap()
}

fn assert_form(source: &Matrix, factors: &[i64]) {
    let result = source.smith_normal_form().unwrap();
    assert_eq!(result.rank(), factors.len());
    assert_eq!(
        result.invariant_factors(),
        factors
            .iter()
            .copied()
            .map(BigInt::from)
            .collect::<Vec<_>>()
    );
    let replay = verify_integer_smith(result.source(), result.witness(), &mut Unbounded).unwrap();
    assert_eq!(result.receipt(), &replay);
    let (left, right) = (dense(result.left()), dense(result.right()));
    assert_eq!(
        left.matmul(source).unwrap().matmul(&right).unwrap(),
        dense(result.diagonal())
    );
    assert_eq!(
        left.matmul(&dense(result.left_inverse())).unwrap(),
        Matrix::eye(source.rows()).unwrap()
    );
    assert_eq!(
        right.matmul(&dense(result.right_inverse())).unwrap(),
        Matrix::eye(source.cols()).unwrap()
    );
    assert_eq!(
        (result.diagonal().rows, result.diagonal().columns),
        (source.rows(), source.cols())
    );
}

#[test]
fn empty_and_rectangular_zero_sources_keep_their_full_shape() {
    for (rows, columns) in [(0, 0), (0, 5), (4, 0), (3, 5), (5, 3)] {
        assert_form(&Matrix::zeros(rows, columns).unwrap(), &[]);
    }
    assert_form(&Matrix::eye(64).unwrap(), &[1; 64]);
}

#[test]
fn divisibility_repair_and_negative_pivots_produce_canonical_factors() {
    assert_form(&matrix(1, 1, &[-6]), &[6]);
    // A diagonal decomposition alone is insufficient: 6 does not divide 10.
    assert_form(&matrix(2, 2, &[6, 0, 0, 10]), &[2, 30]);
    assert_form(
        &matrix(3, 3, &[-6, 0, 0, 0, 10, 0, 0, 0, -15]),
        &[1, 30, 30],
    );
    assert_form(&matrix(3, 3, &[12, 6, 4, 3, 9, 6, 2, 16, 14]), &[1, 10, 30]);
}

#[test]
fn tall_wide_and_rank_deficient_forms_are_complete() {
    let wide = matrix(2, 3, &[2, 4, 4, 6, 6, 12]);
    assert_form(&wide, &[2, 6]);
    assert_form(&wide.transpose(), &[2, 6]);
    assert_form(&matrix(3, 3, &[2, 4, 6, 4, 8, 12, 0, 0, 0]), &[2]);
    assert_form(&matrix(1, 4, &[0, -12, 18, 30]), &[6]);
    assert_form(&matrix(4, 1, &[0, -12, 18, 30]), &[6]);
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[test]
fn exhaustive_two_by_two_forms_match_independent_determinantal_divisors() {
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                for d in -2..=2 {
                    let first = gcd(gcd(a, b), gcd(c, d));
                    let determinant: i64 = a * d - b * c;
                    let factors = if first == 0 {
                        vec![]
                    } else if determinant == 0 {
                        vec![first]
                    } else {
                        vec![first, determinant.abs() / first]
                    };
                    let source = matrix(2, 2, &[a, b, c, d]);
                    let result = source.smith_normal_form().unwrap();
                    assert_eq!(
                        result.invariant_factors(),
                        factors.into_iter().map(BigInt::from).collect::<Vec<_>>(),
                        "[{a}, {b}; {c}, {d}]"
                    );
                }
            }
        }
    }
}

#[test]
fn seeded_integer_equivalences_preserve_every_invariant_factor() {
    for seed in 0..24i64 {
        let (rows, columns) = (4, 5);
        let mut values = vec![0i64; rows * columns];
        for (i, value) in [2, 6, 30].into_iter().enumerate() {
            values[i * columns + i] = value;
        }
        // Known integer-unimodular transformations of a fixed rectangular form.
        for i in 0..3 {
            let multiple = (seed + i as i64 * 3) % 7 - 3;
            for column in 0..columns {
                let source = values[i * columns + column];
                values[(i + 1) * columns + column] += multiple * source;
            }
        }
        for i in (0..4).rev() {
            let multiple = (seed * 3 + i as i64) % 5 - 2;
            for row in 0..rows {
                let source = values[row * columns + i + 1];
                values[row * columns + i] -= multiple * source;
            }
        }
        for column in 0..columns {
            values.swap(column, 3 * columns + column);
        }
        let source = matrix(rows, columns, &values);
        let result = source.smith_normal_form().unwrap();
        assert_eq!(
            result.invariant_factors(),
            &[BigInt::from(2), BigInt::from(6), BigInt::from(30)]
        );
        let sparse = source.to_sparse().unwrap().smith_normal_form().unwrap();
        assert_eq!(result.witness(), sparse.witness());
        assert_eq!(result.receipt(), sparse.receipt());
    }
}

#[test]
fn exact_large_coefficients_never_pass_through_binary64() {
    let n = BigInt::one() << 80u32;
    let source = Matrix::new(
        2,
        2,
        vec![
            Expr::Integer(n.clone()),
            Expr::from_i64(0),
            Expr::from_i64(0),
            Expr::Integer(&n + 1i64),
        ],
    )
    .unwrap();
    let result = source.smith_normal_form().unwrap();
    assert_eq!(
        result.invariant_factors(),
        &[BigInt::one(), &n * (&n + 1i64)]
    );
}

#[test]
fn original_domain_and_size_are_checked_before_zero_rank_shortcuts() {
    for source in [
        Matrix::new(1, 1, vec![Expr::Sym(Symbol::new("x"))]).unwrap(),
        Matrix::new(
            1,
            1,
            vec![Expr::Rational(BigRational::new(1.into(), 2.into()))],
        )
        .unwrap(),
        Matrix::new(
            1,
            1,
            vec![Expr::Mul(vec![
                Expr::from_i64(0),
                Expr::Sym(Symbol::new("x")),
            ])],
        )
        .unwrap(),
    ] {
        assert!(matches!(
            source.smith_normal_form(),
            Err(MatrixError::UnsupportedIntegerDomain)
        ));
    }
    // A literal Rational with denominator one is still an exact integer source.
    let integral_rational = Matrix::new(
        1,
        1,
        vec![Expr::Rational(BigRational::from_integer((-12).into()))],
    )
    .unwrap();
    assert_form(&integral_rational, &[12]);
    for (rows, columns) in [(65, 0), (0, 65), (65, 1), (1, 65)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        assert!(matches!(
            source.smith_normal_form(),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            source.to_sparse().unwrap().smith_normal_form(),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let huge = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4096u32)]).unwrap();
    assert!(matches!(
        huge.smith_normal_form(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn forged_sources_transforms_and_diagonals_do_not_gain_acceptance() {
    let source = matrix(2, 2, &[2, 4, 6, 8]);
    let candidate = generate_smith_normal_form_candidate(&source, &mut Unbounded).unwrap();
    verify_smith_normal_form_candidate(&source, candidate.clone(), &mut Unbounded).unwrap();
    let mut wrong_source = candidate.clone();
    wrong_source.source.entries[0].value = BigRational::from_integer(5.into());
    assert!(matches!(
        verify_smith_normal_form_candidate(&source, wrong_source, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let mut wrong_diagonal = candidate.clone();
    wrong_diagonal.witness.diagonal.entries[0].value = BigRational::from_integer(1.into());
    assert!(verify_smith_normal_form_candidate(&source, wrong_diagonal, &mut Unbounded).is_err());
    let mut wrong_inverse = candidate.clone();
    wrong_inverse.witness.left_inverse.entries[0].value = BigRational::from_integer(123.into());
    assert!(verify_smith_normal_form_candidate(&source, wrong_inverse, &mut Unbounded).is_err());
    // Scaling U and D together preserves U A V = D, but is not unimodular.
    let mut nonunimodular = candidate;
    for entry in &mut nonunimodular.witness.left.entries {
        if entry.row == 0 {
            entry.value *= BigRational::from_integer(2.into());
        }
    }
    nonunimodular.witness.diagonal.entries[0].value *= BigRational::from_integer(2.into());
    assert!(verify_smith_normal_form_candidate(&source, nonunimodular, &mut Unbounded).is_err());
}

struct CancelAfter {
    used: usize,
    stop: usize,
}

impl BudgetMeter for CancelAfter {
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
fn cancellation_at_generation_verification_and_publication_yields_no_form() {
    let source = matrix(3, 3, &[12, 6, 4, 3, 9, 6, 2, 16, 14]);
    let mut generation = CancelAfter {
        used: 0,
        stop: usize::MAX,
    };
    let mut verification = CancelAfter {
        used: 0,
        stop: usize::MAX,
    };
    source
        .metered_smith_normal_form(&mut generation, &mut verification)
        .unwrap();
    for stop in [1, 2, 100, generation.used / 2, generation.used] {
        assert!(matches!(
            source.metered_smith_normal_form(&mut CancelAfter { used: 0, stop }, &mut Unbounded),
            Err(MatrixError::Cancelled)
        ));
    }
    for stop in [1, 2, 100, verification.used / 2, verification.used] {
        assert!(matches!(
            source.metered_smith_normal_form(&mut Unbounded, &mut CancelAfter { used: 0, stop }),
            Err(MatrixError::Cancelled)
        ));
    }
}

#[test]
fn both_phases_honor_independent_compute_memory_and_allocation_limits() {
    let source = matrix(2, 2, &[6, 0, 0, 10]);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(
            source.metered_smith_normal_form(&mut Budget::new(limits), &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            source.metered_smith_normal_form(&mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}
