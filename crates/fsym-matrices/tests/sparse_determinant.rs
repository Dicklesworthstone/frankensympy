//! Reachable sparse determinant APIs and portable independent kernel replay.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    MatrixError, SparseMatrix, generate_sparse_determinant_candidate,
    verify_sparse_determinant_candidate,
};
use fsym_proof_kernel::linear_system::LinearRowOperation;
use fsym_proof_kernel::sparse_determinant::{SparseDeterminantWitness, verify_sparse_determinant};
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION, RationalSparseMatrix,
};
use std::collections::BTreeMap;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}

fn integers(n: usize, values: &[i64]) -> SparseMatrix {
    assert_eq!(n * n, values.len());
    SparseMatrix::new(
        n,
        n,
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(index, &value)| ((index / n, index % n), Expr::from_i64(value)))
            .collect(),
    )
    .unwrap()
}

fn scalar(expression: Expr) -> BigRational {
    match expression {
        Expr::Integer(value) => BigRational::from_integer(value),
        Expr::Rational(value) => value,
        _ => panic!("expected an exact rational"),
    }
}

#[test]
fn public_api_returns_exact_source_bound_portable_evidence() {
    let matrix = integers(3, &[0, 2, 1, 3, 0, 6, 0, 0, -5]);
    let checked = matrix.det_with_portable_certificate().unwrap();
    assert_eq!(checked.determinant(), &q(30));
    assert_eq!(checked.receipt().dimension(), 3);
    assert_eq!(checked.receipt().rank(), 3);
    assert_eq!(scalar(matrix.det().unwrap()), q(30));

    // The artifact is consumable using the proof kernel alone; no call back
    // into the matrix generator or a stored verified flag is needed.
    let source: RationalSparseMatrix =
        serde_json::from_str(&serde_json::to_string(checked.source()).unwrap()).unwrap();
    let witness: SparseDeterminantWitness =
        serde_json::from_str(&serde_json::to_string(checked.witness()).unwrap()).unwrap();
    let receipt = verify_sparse_determinant(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(&receipt, checked.receipt());
}

#[test]
fn all_small_two_by_two_matrices_match_the_closed_form_and_dense_lane() {
    for code in 0..625 {
        let mut remaining = code;
        let mut values = [0; 4];
        for value in &mut values {
            *value = (remaining % 5) - 2;
            remaining /= 5;
        }
        let expected = q(values[0] * values[3] - values[1] * values[2]);
        let matrix = integers(2, &values);
        let accepted = matrix.det_with_portable_certificate().unwrap();
        assert_eq!(accepted.determinant(), &expected, "{values:?}");
        assert_eq!(scalar(matrix.to_dense().unwrap().det().unwrap()), expected);
    }
}

#[test]
fn nontriangular_three_by_three_matrices_match_independent_cofactor_formula() {
    for seed in 0u64..100 {
        let mut state = seed + 1;
        let mut values = [0; 9];
        for value in &mut values {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *value = ((state >> 32) % 11) as i64 - 5;
        }
        let [a, b, c, d, e, f, g, h, i] = values;
        let expected = q(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g));
        assert_eq!(
            integers(3, &values)
                .det_with_portable_certificate()
                .unwrap()
                .determinant(),
            &expected,
            "{values:?}"
        );
    }
}

#[test]
fn rational_singular_and_empty_cases_keep_their_exact_meaning() {
    let rational = SparseMatrix::new(
        2,
        2,
        BTreeMap::from([
            ((0, 0), Expr::rational(1, 2).unwrap()),
            ((0, 1), Expr::rational(1, 3).unwrap()),
            ((1, 1), Expr::rational(-2, 5).unwrap()),
        ]),
    )
    .unwrap();
    assert_eq!(
        rational
            .det_with_portable_certificate()
            .unwrap()
            .determinant(),
        &BigRational::new((-1).into(), 5.into())
    );
    let singular = integers(3, &[1, 2, 3, 2, 4, 6, 0, 1, 1]);
    let checked = singular.det_with_portable_certificate().unwrap();
    assert_eq!(checked.determinant(), &q(0));
    assert_eq!(checked.receipt().rank(), 2);
    assert_eq!(
        SparseMatrix::zeros(0, 0)
            .det_with_portable_certificate()
            .unwrap()
            .determinant(),
        &q(1)
    );
}

#[test]
fn same_determinant_different_source_and_coordinated_replacement_are_rejected() {
    let first = integers(2, &[2, 0, 0, 3]);
    let second = integers(2, &[3, 0, 0, 2]);
    let a = first.det_with_portable_certificate().unwrap();
    let b = second.det_with_portable_certificate().unwrap();
    assert_eq!(a.determinant(), b.determinant());
    assert_ne!(a.receipt().source_digest(), b.receipt().source_digest());
    let replaced = generate_sparse_determinant_candidate(&second, &mut Unbounded).unwrap();
    assert!(matches!(
        verify_sparse_determinant_candidate(&first, replaced, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn corrupted_candidate_value_transcript_and_source_never_publish_evidence() {
    let source = integers(2, &[0, 2, 3, 1]);
    let valid = generate_sparse_determinant_candidate(&source, &mut Unbounded).unwrap();
    for mutation in 0..6 {
        let mut candidate = valid.clone();
        match mutation {
            0 => candidate.witness.determinant = q(6),
            1 => candidate.witness.reduction.operations.clear(),
            2 => {
                candidate.witness.reduction.operations[0] =
                    LinearRowOperation::Swap { left: 0, right: 0 }
            }
            3 => candidate.source.entries[0].value = q(7),
            4 => candidate.witness.reduction.pivots.clear(),
            5 => candidate.witness.schema_version += 1,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                verify_sparse_determinant_candidate(&source, candidate, &mut Unbounded),
                Err(MatrixError::InvalidCertificate(_))
            ),
            "mutation {mutation}"
        );
    }
}

#[test]
fn exact_domain_and_intrinsic_limits_apply_before_zero_shortcuts() {
    let symbolic_singular =
        SparseMatrix::new(2, 2, BTreeMap::from([((0, 0), Expr::symbol("x"))])).unwrap();
    assert!(matches!(
        symbolic_singular.det_with_portable_certificate(),
        Err(MatrixError::UnsupportedCertificateDomain)
    ));
    assert!(matches!(
        SparseMatrix::zeros(2, 3).det_with_portable_certificate(),
        Err(MatrixError::NotSquare(2, 3))
    ));
    let oversized = Expr::Integer(BigInt::one() << MAX_SPARSE_RREF_COEFFICIENT_BITS as u32);
    let source = SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), oversized)])).unwrap();
    assert!(matches!(
        source.det_with_portable_certificate(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let n = MAX_SPARSE_RREF_DIMENSION + 1;
    assert!(matches!(
        SparseMatrix::zeros(n, n).det_with_portable_certificate(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn determinant_product_growth_refuses_even_when_the_final_value_is_small() {
    let power = BigInt::one() << 5_000u32;
    let large = BigRational::from_integer(power.clone());
    let small = BigRational::new(BigInt::one(), power);
    let diagonal = [small.clone(), small, large.clone(), large];
    let matrix = SparseMatrix::new(
        4,
        4,
        diagonal
            .into_iter()
            .enumerate()
            .map(|(i, value)| ((i, i), Expr::Rational(value)))
            .collect(),
    )
    .unwrap();
    // The mathematical determinant is one, but this bounded proposal lane
    // must reject its 10001-bit intermediate instead of bypassing its cap.
    assert!(matches!(
        matrix.det_with_portable_certificate(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    stop: Option<usize>,
    stopped: bool,
    used: [u64; 3],
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "continued after cancellation");
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
        assert!(!self.stopped, "continued after cancellation");
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.stopped = true;
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

#[test]
fn all_generation_verification_and_publication_safe_points_cancel_without_a_receipt() {
    let matrix = integers(2, &[0, 2, 3, 1]);
    let mut measured = Meter::default();
    matrix
        .metered_det_with_portable_certificate(&mut measured)
        .unwrap();
    for stop in 1..=measured.checkpoints {
        let mut meter = Meter {
            stop: Some(stop),
            ..Meter::default()
        };
        assert!(
            matches!(
                matrix.metered_det_with_portable_certificate(&mut meter),
                Err(MatrixError::Cancelled)
            ),
            "safe point {stop}"
        );
    }
}

#[test]
fn one_cumulative_budget_covers_generation_through_accepted_publication() {
    let matrix = integers(2, &[0, 2, 3, 1]);
    let mut measured = Meter::default();
    matrix
        .metered_det_with_portable_certificate(&mut measured)
        .unwrap();
    for (dimension, used) in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ]
    .into_iter()
    .zip(measured.used)
    {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = used;
        assert!(
            matrix
                .metered_det_with_portable_certificate(&mut Budget::new(limits))
                .is_ok()
        );
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = used - 1;
        assert!(matches!(
            matrix.metered_det_with_portable_certificate(&mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = measured.used[0] * 2 - 1;
    let mut shared = Budget::new(limits);
    matrix
        .metered_det_with_portable_certificate(&mut shared)
        .unwrap();
    assert!(matches!(
        matrix.metered_det_with_portable_certificate(&mut shared),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn split_generator_and_verifier_meters_allow_independently_owned_resources() {
    let matrix = integers(2, &[0, 2, 3, 1]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    let proposal = generate_sparse_determinant_candidate(&matrix, &mut generator).unwrap();
    let accepted = verify_sparse_determinant_candidate(&matrix, proposal, &mut verifier).unwrap();
    assert_eq!(accepted.determinant(), &q(-6));
    assert!(generator.used.into_iter().all(|used| used > 0));
    assert!(verifier.used.into_iter().all(|used| used > 0));
}

#[test]
fn thousand_dimensional_diagonal_remains_sparse_for_generation_and_acceptance() {
    let n = 1_024;
    let matrix = SparseMatrix::new(
        n,
        n,
        (0..n)
            .map(|i| ((i, i), Expr::from_i64(if i == 0 { 3 } else { 1 })))
            .collect(),
    )
    .unwrap();
    assert!(
        matrix.to_dense().is_err(),
        "the dense matrix cap must be exceeded"
    );
    let mut limits = BudgetLimits::uniform(2_000_000, 0);
    limits.dimensions[Dimension::MemoryBytes.index()] = 8_000_000;
    let checked = matrix
        .metered_det_with_portable_certificate(&mut Budget::new(limits))
        .unwrap();
    assert_eq!(checked.determinant(), &q(3));
    assert_eq!(checked.source().entries.len(), n);
    assert_eq!(checked.witness().reduction.reduced.entries.len(), n);
}
