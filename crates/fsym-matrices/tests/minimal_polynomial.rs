//! Exact minimality, original-source binding and independent falsification.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_matrices::{
    MAX_MINIMAL_POLYNOMIAL_DIMENSION, Matrix, MatrixError, MinimalPolynomialCandidate,
    SparseMatrix, verify_minimal_polynomial_candidate,
};
use fsym_polys::UnivariatePoly;
use num_traits::Zero;
use std::collections::BTreeMap;

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}
fn coefficients(values: &[i64]) -> Vec<BigRational> {
    values.iter().copied().map(q).collect()
}
fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        values.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}
fn jordan(size: usize, eigenvalue: i64) -> Matrix {
    let mut entries = vec![Expr::from_i64(0); size * size];
    for i in 0..size {
        entries[i * size + i] = Expr::from_i64(eigenvalue);
        if i + 1 < size {
            entries[i * size + i + 1] = Expr::from_i64(1);
        }
    }
    Matrix::new(size, size, entries).unwrap()
}
fn block_diagonal(left: &Matrix, right: &Matrix, couple: bool) -> Matrix {
    let (a, b) = (left.rows(), right.rows());
    let n = a + b;
    let mut entries = vec![Expr::from_i64(0); n * n];
    for (block, offset) in [(left, 0), (right, a)] {
        for row in 0..block.rows() {
            for column in 0..block.cols() {
                entries[(row + offset) * n + column + offset] =
                    block.data()[row * block.cols() + column].clone();
            }
        }
    }
    if couple {
        assert_eq!(a, b);
        for i in 0..a {
            entries[i * n + a + i] = Expr::from_i64(1);
        }
    }
    Matrix::new(n, n, entries).unwrap()
}
fn assert_polynomial(source: &Matrix, expected: &[BigRational]) {
    let checked = source.minimal_polynomial().unwrap();
    assert_eq!(checked.matrix(), source);
    assert_eq!(checked.coefficients(), expected);
    assert_eq!(checked.degree() + 1, expected.len());
    let replayed =
        verify_minimal_polynomial_candidate(source, checked.certificate(), &mut Unbounded).unwrap();
    assert_eq!(replayed.into_coefficients(), expected);
}

#[test]
fn empty_zero_identity_and_rational_scalar_matrices_have_their_least_degree() {
    assert_polynomial(&Matrix::zeros(0, 0).unwrap(), &coefficients(&[1]));
    assert_polynomial(&Matrix::zeros(5, 5).unwrap(), &coefficients(&[0, 1]));
    assert_polynomial(&Matrix::eye(7).unwrap(), &coefficients(&[-1, 1]));
    let scalar = Matrix::new(
        2,
        2,
        vec![
            Expr::rational(3, 2).unwrap(),
            Expr::from_i64(0),
            Expr::from_i64(0),
            Expr::rational(3, 2).unwrap(),
        ],
    )
    .unwrap();
    assert_polynomial(&scalar, &[BigRational::new((-3).into(), 2.into()), q(1)]);
}

#[test]
fn nilpotency_index_uses_every_block_not_a_single_starting_vector() {
    assert_polynomial(&jordan(4, 0), &coefficients(&[0, 0, 0, 0, 1]));
    let hidden_chain = block_diagonal(&Matrix::zeros(1, 1).unwrap(), &jordan(3, 0), false);
    // A Krylov calculation starting only at e_0 would incorrectly return t.
    assert_polynomial(&hidden_chain, &coefficients(&[0, 0, 0, 1]));
    let repeated = block_diagonal(&jordan(4, 0), &jordan(2, 0), false);
    assert_polynomial(&repeated, &coefficients(&[0, 0, 0, 0, 1]));
}

#[test]
fn multiple_jordan_eigenvalues_retain_maximal_block_multiplicities() {
    let source = block_diagonal(&jordan(2, 1), &jordan(3, -2), false);
    // (t-1)^2 (t+2)^3, not a product of distinct eigenvalue factors.
    assert_polynomial(&source, &coefficients(&[8, -4, -10, 1, 4, 1]));
}

#[test]
fn exact_rational_nonsymmetric_entries_do_not_require_eigenvectors() {
    let source = Matrix::new(
        2,
        2,
        vec![
            Expr::rational(1, 2).unwrap(),
            Expr::rational(1, 3).unwrap(),
            Expr::from_i64(0),
            Expr::rational(-2, 5).unwrap(),
        ],
    )
    .unwrap();
    assert_polynomial(
        &source,
        &[
            BigRational::new((-1).into(), 5.into()),
            BigRational::new((-1).into(), 10.into()),
            q(1),
        ],
    );
    assert_polynomial(&matrix(2, 2, &[0, -1, 1, 0]), &coefficients(&[1, 0, 1]));
}

#[test]
fn exhaustive_small_two_by_two_matrices_match_closed_form_minimal_polynomials() {
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    let source = matrix(2, 2, &[a, b, c, d]);
                    let expected = if b == 0 && c == 0 && a == d {
                        coefficients(&[-a, 1])
                    } else {
                        coefficients(&[a * d - b * c, -a - d, 1])
                    };
                    assert_eq!(
                        source.minimal_polynomial().unwrap().coefficients(),
                        expected,
                        "matrix [{a},{b};{c},{d}]"
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_irreducible_quintics_distinguish_semisimple_and_defective_matrices() {
    let companion = matrix(
        5,
        5,
        &[
            0, 0, 0, 0, -54, 1, 0, 0, 0, 31, 0, 1, 0, 0, 82, 0, 0, 1, 0, -66, 0, 0, 0, 1, 15,
        ],
    );
    let p = [54, -31, -82, 66, -15, 1];
    let semisimple = block_diagonal(&companion, &companion, false);
    let defective = block_diagonal(&companion, &companion, true);
    assert_eq!(
        semisimple.char_poly().unwrap(),
        defective.char_poly().unwrap()
    );
    assert_polynomial(&semisimple, &coefficients(&p));
    let mut square = vec![0i64; 11];
    for (i, a) in p.iter().enumerate() {
        for (j, b) in p.iter().enumerate() {
            square[i + j] += a * b;
        }
    }
    assert_polynomial(&defective, &coefficients(&square));
}

#[test]
fn maximum_dimension_supports_full_degree_without_expanding_the_bound() {
    let source = jordan(MAX_MINIMAL_POLYNOMIAL_DIMENSION, 0);
    let mut expected = vec![BigRational::zero(); MAX_MINIMAL_POLYNOMIAL_DIMENSION + 1];
    expected[MAX_MINIMAL_POLYNOMIAL_DIMENSION] = q(1);
    assert_polynomial(&source, &expected);
    assert!(matches!(
        Matrix::eye(MAX_MINIMAL_POLYNOMIAL_DIMENSION + 1)
            .unwrap()
            .minimal_polynomial(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn an_annihilating_characteristic_polynomial_is_not_enough_for_minimality() {
    let source = matrix(2, 2, &[3, 0, 0, 3]);
    let candidate = MinimalPolynomialCandidate {
        coefficients: coefficients(&[9, -6, 1]),
        coordinate_rows: vec![0, 3],
        minor_inverse: coefficients(&[1, 0, 0, 1]),
    };
    assert_eq!(
        source
            .eval_poly(&UnivariatePoly::new(
                Symbol::new("t"),
                candidate.coefficients.clone(),
            ))
            .unwrap(),
        Matrix::zeros(2, 2).unwrap()
    );
    assert!(matches!(
        verify_minimal_polynomial_candidate(&source, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(reason)) if reason.contains("independent")
    ));
}

#[test]
fn tiny_coefficient_and_minor_inverse_mutations_are_rejected_exactly() {
    let source = jordan(3, 0);
    let checked = source.minimal_polynomial().unwrap();
    let epsilon = BigRational::new(1.into(), BigInt::from(10).pow(80));
    let mut polynomial = checked.certificate().clone();
    polynomial.coefficients[0] = epsilon.clone();
    assert!(matches!(
        verify_minimal_polynomial_candidate(&source, &polynomial, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(reason)) if reason.contains("annihilate")
    ));
    let mut independence = checked.certificate().clone();
    independence.minor_inverse[0] += epsilon;
    assert!(matches!(
        verify_minimal_polynomial_candidate(&source, &independence, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(reason)) if reason.contains("independent")
    ));
}

#[test]
fn malformed_nonmonic_and_rebound_claims_never_become_accepted_results() {
    let source = matrix(2, 2, &[1, 0, 0, 2]);
    let checked = source.minimal_polynomial().unwrap();
    let valid = checked.certificate();
    for change in 0..7 {
        let mut bad = valid.clone();
        match change {
            0 => bad.coefficients.clear(),
            1 => bad.coefficients.push(q(1)),
            2 => bad.coefficients[2] = q(2),
            3 => bad.coordinate_rows[1] = bad.coordinate_rows[0],
            4 => bad.coordinate_rows[0] = usize::MAX,
            5 => {
                bad.minor_inverse.pop();
            }
            6 => {
                bad.coordinate_rows.pop();
            }
            _ => unreachable!(),
        }
        assert!(verify_minimal_polynomial_candidate(&source, &bad, &mut Unbounded).is_err());
    }
    assert!(
        verify_minimal_polynomial_candidate(&matrix(2, 2, &[1, 0, 0, 3]), valid, &mut Unbounded,)
            .is_err()
    );
    let mut excessive = valid.clone();
    excessive.minor_inverse[0] = BigRational::from_integer(BigInt::one() << 8192u32);
    assert!(matches!(
        verify_minimal_polynomial_candidate(&source, &excessive, &mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn independent_checker_accepts_a_valid_alternate_coordinate_minor() {
    let source = matrix(2, 2, &[3, 0, 0, 3]);
    let candidate = MinimalPolynomialCandidate {
        coefficients: coefficients(&[-3, 1]),
        coordinate_rows: vec![3],
        minor_inverse: coefficients(&[1]),
    };
    assert_eq!(
        verify_minimal_polynomial_candidate(&source, &candidate, &mut Unbounded)
            .unwrap()
            .coefficients(),
        coefficients(&[-3, 1])
    );
}

#[test]
fn complete_literal_input_admission_precedes_scalar_shortcuts_and_large_copies() {
    for invalid in [
        Expr::symbol("parameter"),
        Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("hidden")]),
        Expr::Const(fsym_core::Constant::I),
    ] {
        let source = Matrix::new(
            2,
            2,
            vec![
                Expr::from_i64(1),
                Expr::from_i64(0),
                Expr::from_i64(0),
                invalid,
            ],
        )
        .unwrap();
        assert_eq!(
            source.minimal_polynomial().unwrap_err(),
            MatrixError::UnsupportedCertificateDomain
        );
    }
    for (rows, columns) in [(0, 1), (1, 0), (2, 3)] {
        assert!(matches!(
            Matrix::zeros(rows, columns).unwrap().minimal_polynomial(),
            Err(MatrixError::NotSquare(..))
        ));
    }
    let largest = BigInt::one() << 4095u32;
    let source = Matrix::new(1, 1, vec![Expr::Integer(largest.clone())]).unwrap();
    assert_polynomial(&source, &[BigRational::from_integer(-largest), q(1)]);
    let oversized = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4096u32)]).unwrap();
    assert!(matches!(
        oversized.minimal_polynomial(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn sparse_inputs_are_bound_before_bounded_fill_and_share_the_exact_polynomial() {
    let source = SparseMatrix::new(
        3,
        3,
        BTreeMap::from([
            ((0, 1), Expr::rational(1, 3).unwrap()),
            ((1, 2), Expr::rational(2, 5).unwrap()),
        ]),
    )
    .unwrap();
    let checked = source.minimal_polynomial().unwrap();
    assert_eq!(checked.coefficients(), coefficients(&[0, 0, 0, 1]));
    assert_eq!(checked.matrix(), &source.to_dense().unwrap());
    assert_eq!(
        SparseMatrix::zeros(0, 0)
            .minimal_polynomial()
            .unwrap()
            .coefficients(),
        coefficients(&[1])
    );
    assert!(matches!(
        SparseMatrix::zeros(usize::MAX, usize::MAX).minimal_polynomial(),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        SparseMatrix::zeros(usize::MAX, 0).minimal_polynomial(),
        Err(MatrixError::NotSquare(..))
    ));
    let symbolic = SparseMatrix::new(2, 2, BTreeMap::from([((1, 1), Expr::symbol("x"))])).unwrap();
    assert_eq!(
        symbolic.minimal_polynomial().unwrap_err(),
        MatrixError::UnsupportedCertificateDomain
    );
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
fn actual_budgets_span_generation_verification_sparse_admission_and_publication() {
    let source = matrix(2, 2, &[0, 1, -2, 3]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    source
        .minimal_polynomial_with_meters(&mut generator, &mut verifier)
        .unwrap();
    for (dimension, index) in [
        (Dimension::ComputeSteps, 0),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 2),
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = generator.used[index] - 1;
        assert!(matches!(
            source.minimal_polynomial_with_meters(&mut Budget::new(limits), &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = verifier.used[index] - 1;
        assert!(matches!(
            source.minimal_polynomial_with_meters(&mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let mut untouched = Meter::default();
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        source
            .minimal_polynomial_with_meters(&mut tiny, &mut untouched)
            .is_err()
    );
    assert_eq!(untouched.used, [0; 3]);
    assert_eq!(untouched.checkpoints, 0);

    let sparse = SparseMatrix::eye(2).unwrap();
    let mut sparse_generator = Meter::default();
    let mut sparse_verifier = Meter::default();
    sparse
        .minimal_polynomial_with_meters(&mut sparse_generator, &mut sparse_verifier)
        .unwrap();
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = sparse_verifier.used[0] - 1;
    assert!(matches!(
        sparse.minimal_polynomial_with_meters(&mut Unbounded, &mut Budget::new(limits)),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn both_phases_honor_cancellation_including_the_final_safe_point() {
    let source = matrix(2, 2, &[0, 1, -2, 3]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    source
        .minimal_polynomial_with_meters(&mut generator, &mut verifier)
        .unwrap();
    for stop in [1, generator.checkpoints / 2, generator.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Meter::default()
        };
        assert_eq!(
            source
                .minimal_polynomial_with_meters(&mut cancelled, &mut Unbounded)
                .unwrap_err(),
            MatrixError::Cancelled
        );
    }
    for stop in [1, verifier.checkpoints / 2, verifier.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Meter::default()
        };
        assert_eq!(
            source
                .minimal_polynomial_with_meters(&mut Unbounded, &mut cancelled)
                .unwrap_err(),
            MatrixError::Cancelled
        );
    }
}
