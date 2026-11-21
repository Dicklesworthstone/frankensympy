//! Exact residue-ring inverses, composite moduli, replay and interruption.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_matrices::{
    Matrix, MatrixError, generate_integer_modular_inverse_candidate, metered_inverse_mod,
    verify_integer_modular_inverse_candidate,
};
use fsym_proof_kernel::integer_modular_inverse::{
    IntegerModularInverseOutcome, verify_integer_modular_inverse,
};
use fsym_proof_kernel::sparse_rref::RationalSparseMatrix;

fn matrix(size: usize, values: &[i64]) -> Matrix {
    Matrix::new(size, size, values.iter().copied().map(Expr::from_i64).collect()).unwrap()
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

fn integer(value: &Expr) -> BigInt {
    match value {
        Expr::Integer(value) => value.clone(),
        Expr::Rational(value) if value.is_integer() => value.numer().clone(),
        _ => panic!("literal integer expected"),
    }
}

fn check_residuals(source: &Matrix, inverse: &Matrix, modulus: &BigInt) {
    for product in [source.matmul(inverse).unwrap(), inverse.matmul(source).unwrap()] {
        for row in 0..source.rows() {
            for column in 0..source.cols() {
                let difference = integer(product.get(row, column).unwrap())
                    - BigInt::from(if row == column { 1i64 } else { 0i64 });
                assert!((&difference % modulus).is_zero());
            }
        }
    }
    for value in inverse.data() {
        let value = integer(value);
        assert!(!value.is_negative() && &value < modulus);
    }
}

#[test]
fn prime_composite_and_hill_cipher_fixtures_match_exact_expected_residues() {
    let fixtures = [
        (matrix(2, &[1, 2, 3, 4]), 5, matrix(2, &[3, 1, 4, 2])),
        (matrix(2, &[2, 3, 3, 2]), 6, matrix(2, &[2, 3, 3, 2])),
        (matrix(2, &[2, 3, 3, 2]), 2, matrix(2, &[0, 1, 1, 0])),
        (matrix(1, &[-3]), 10, matrix(1, &[3])),
        (
            matrix(3, &[6, 24, 1, 13, 16, 10, 20, 17, 15]),
            26,
            matrix(3, &[8, 5, 10, 21, 8, 21, 21, 12, 8]),
        ),
    ];
    for (source, modulus, expected) in fixtures {
        let modulus = BigInt::from(modulus);
        let result = source.inverse_mod(&modulus).unwrap();
        assert!(result.receipt().is_invertible());
        assert_eq!(dense(result.inverse().unwrap()), expected);
        check_residuals(&source, &expected, &modulus);
        let replay = verify_integer_modular_inverse(
            result.source(), &modulus, result.witness(), &mut Unbounded,
        ).unwrap();
        assert_eq!(&replay, result.receipt());
        let sparse = source.to_sparse().unwrap().inverse_mod(&modulus).unwrap();
        assert_eq!(sparse.inverse(), result.inverse());
    }
}

fn gcd(mut left: i64, mut right: i64) -> i64 {
    left = left.abs();
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

#[test]
fn small_matrices_match_independent_determinant_unit_criterion() {
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    let source = matrix(2, &[a, b, c, d]);
                    for modulus in [2, 4, 6, 9, 25] {
                        let result = source.inverse_mod(&modulus.into()).unwrap();
                        assert_eq!(result.inverse().is_some(), gcd(a * d - b * c, modulus) == 1);
                        if let Some(inverse) = result.inverse() {
                            check_residuals(&source, &dense(inverse), &modulus.into());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rank_deficiency_and_nonunit_determinants_are_checked_algebraic_failures() {
    for (source, modulus) in [
        (matrix(2, &[2, 3, 3, 2]), 5),
        (matrix(2, &[2, 0, 0, 1]), 6),
        (matrix(2, &[1, 2, 2, 4]), 7),
        (matrix(2, &[0, 0, 0, 0]), 6),
    ] {
        let result = source.inverse_mod(&modulus.into()).unwrap();
        assert!(result.inverse().is_none());
        assert!(!result.receipt().is_invertible());
        assert!(result.receipt().nonunit_index().is_some());
        verify_integer_modular_inverse(
            result.source(), result.modulus(), result.witness(), &mut Unbounded,
        ).unwrap();
    }
}

#[test]
fn empty_inverse_and_nonsquare_refusals_preserve_shape() {
    let source = Matrix::zeros(0, 0).unwrap();
    let result = source.inverse_mod(&6.into()).unwrap();
    let inverse = result.inverse().unwrap();
    assert_eq!((inverse.rows, inverse.columns), (0, 0));
    for (rows, columns) in [(0, 2), (2, 0), (2, 3)] {
        assert!(matches!(
            Matrix::zeros(rows, columns).unwrap().inverse_mod(&6.into()),
            Err(MatrixError::NotSquare(_, _))
        ));
    }
}

#[test]
fn large_moduli_and_original_integers_stay_exact() {
    let modulus = (BigInt::one() << 256u32) + 1i64;
    let source = Matrix::new(
        2, 2, vec![
            Expr::Integer(&modulus - 2i64), Expr::from_i64(1),
            Expr::from_i64(1), Expr::from_i64(0),
        ],
    ).unwrap();
    let result = source.inverse_mod(&modulus).unwrap();
    let expected = matrix(2, &[0, 1, 1, 2]);
    assert_eq!(dense(result.inverse().unwrap()), expected);
    check_residuals(&source, &expected, &modulus);
}

#[test]
fn changed_original_matrix_modulus_and_recorded_source_are_rejected() {
    let source = matrix(2, &[2, 3, 3, 2]);
    let candidate =
        generate_integer_modular_inverse_candidate((&source).into(), &6.into(), &mut Unbounded)
            .unwrap();
    assert!(verify_integer_modular_inverse_candidate(
        (&matrix(2, &[3, 3, 3, 2])).into(), &6.into(), candidate.clone(), &mut Unbounded,
    ).is_err());
    assert!(verify_integer_modular_inverse_candidate(
        (&source).into(), &7.into(), candidate.clone(), &mut Unbounded,
    ).is_err());
    let mut changed = candidate.clone();
    changed.modulus = 7.into();
    assert!(verify_integer_modular_inverse_candidate(
        (&source).into(), &6.into(), changed, &mut Unbounded,
    ).is_err());
    let mut changed = candidate;
    changed.source.entries[0].value = BigRational::from_integer(3.into());
    assert!(verify_integer_modular_inverse_candidate(
        (&source).into(), &6.into(), changed, &mut Unbounded,
    ).is_err());
}

#[test]
fn forged_residues_fractional_entries_and_smith_transforms_are_rejected() {
    let source = matrix(2, &[2, 3, 3, 2]);
    let candidate =
        generate_integer_modular_inverse_candidate((&source).into(), &6.into(), &mut Unbounded)
            .unwrap();
    for altered in [
        BigRational::from_integer(1.into()),
        BigRational::from_integer(8.into()),
        BigRational::new(1.into(), 2.into()),
    ] {
        let mut bad = candidate.clone();
        let IntegerModularInverseOutcome::Invertible { inverse } = &mut bad.witness.outcome else {
            panic!("invertible fixture");
        };
        inverse.entries[0].value = altered;
        assert!(verify_integer_modular_inverse_candidate(
            (&source).into(), &6.into(), bad, &mut Unbounded,
        ).is_err());
    }
    let mut bad = candidate;
    bad.witness.normal_form.left.entries.clear();
    assert!(verify_integer_modular_inverse_candidate(
        (&source).into(), &6.into(), bad, &mut Unbounded,
    ).is_err());
}

#[test]
fn forged_noninvertibility_requires_a_real_common_divisor() {
    let source = matrix(2, &[2, 3, 3, 2]);
    let candidate =
        generate_integer_modular_inverse_candidate((&source).into(), &6.into(), &mut Unbounded)
            .unwrap();
    for (index, divisor) in [(0, 1), (0, 2), (1, 2), (1, 5), (2, 6)] {
        let mut bad = candidate.clone();
        bad.witness.outcome =
            IntegerModularInverseOutcome::NonUnit { index, divisor: divisor.into() };
        assert!(verify_integer_modular_inverse_candidate(
            (&source).into(), &6.into(), bad, &mut Unbounded,
        ).is_err());
    }
}

#[test]
fn all_original_domains_and_size_bounds_precede_outcome_shortcuts() {
    for value in [
        Expr::Sym(Symbol::new("x")),
        Expr::Rational(BigRational::new(1.into(), 2.into())),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(Symbol::new("x"))]),
    ] {
        let source = Matrix::new(2, 2, vec![
            Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(0), value,
        ]).unwrap();
        assert!(matches!(source.inverse_mod(&6.into()), Err(MatrixError::UnsupportedIntegerDomain)));
    }
    for modulus in [-6, 0, 1] {
        assert!(matches!(
            Matrix::eye(1).unwrap().inverse_mod(&modulus.into()),
            Err(MatrixError::InvalidModulus(_))
        ));
    }
    let large = BigInt::one() << 4096u32;
    assert!(matches!(
        Matrix::eye(1).unwrap().inverse_mod(&large),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::new(1, 1, vec![Expr::Integer(large)]).unwrap().inverse_mod(&6.into()),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::zeros(65, 65).unwrap().inverse_mod(&6.into()),
        Err(MatrixError::ResourceLimit(_))
    ));
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
        if self.used >= self.stop { Err(MeterError::Cancelled) } else { Ok(()) }
    }
}

#[test]
fn cancellation_during_generation_verification_and_publication_never_returns_noninvertible() {
    let source = matrix(2, &[2, 3, 3, 2]);
    let mut generator = CancelAfter { used: 0, stop: usize::MAX };
    let mut verifier = CancelAfter { used: 0, stop: usize::MAX };
    metered_inverse_mod(
        (&source).into(), &6.into(), &mut generator, &mut verifier,
    ).unwrap();
    for stop in [1, 2, generator.used / 2, generator.used] {
        assert!(matches!(
            metered_inverse_mod(
                (&source).into(), &6.into(), &mut CancelAfter { used: 0, stop }, &mut Unbounded,
            ),
            Err(MatrixError::Cancelled)
        ));
    }
    for stop in [1, 2, verifier.used / 2, verifier.used] {
        assert!(matches!(
            metered_inverse_mod(
                (&source).into(), &6.into(), &mut Unbounded, &mut CancelAfter { used: 0, stop },
            ),
            Err(MatrixError::Cancelled)
        ));
    }
}

#[test]
fn both_phases_enforce_compute_memory_and_allocation_allowances() {
    let source = matrix(2, &[2, 3, 3, 2]);
    for dimension in [
        Dimension::ComputeSteps, Dimension::MemoryBytes, Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(
            metered_inverse_mod(
                (&source).into(), &6.into(), &mut Budget::new(limits), &mut Unbounded,
            ),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            metered_inverse_mod(
                (&source).into(), &6.into(), &mut Unbounded, &mut Budget::new(limits),
            ),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}
