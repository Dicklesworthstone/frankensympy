//! Independent exact fixtures and four-identity weakening mutants.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::sparse_pseudoinverse::{
    SPARSE_MINIMUM_NORM_SCHEMA_VERSION, SPARSE_PSEUDOINVERSE_SCHEMA_VERSION,
    SparseMinimumNormWitness, SparsePseudoinverseError, SparsePseudoinverseWitness,
    verify_sparse_minimum_norm, verify_sparse_pseudoinverse,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};
use num_traits::{One, Zero};

fn matrix(
    rows: usize,
    columns: usize,
    numerators: &[i64],
    denominator: i64,
) -> RationalSparseMatrix {
    assert_eq!(numerators.len(), rows * columns);
    RationalSparseMatrix {
        rows,
        columns,
        entries: numerators
            .iter()
            .enumerate()
            .filter(|(_, n)| **n != 0)
            .map(|(index, &n)| SparseRationalEntry {
                row: index / columns,
                column: index % columns,
                value: BigRational::new(n.into(), denominator.into()),
            })
            .collect(),
    }
}

fn witness(pseudoinverse: RationalSparseMatrix) -> SparsePseudoinverseWitness {
    SparsePseudoinverseWitness {
        schema_version: SPARSE_PSEUDOINVERSE_SCHEMA_VERSION,
        pseudoinverse,
    }
}

fn rejected(source: &RationalSparseMatrix, p: RationalSparseMatrix, expected: &str) {
    let error = verify_sparse_pseudoinverse(source, &witness(p), &mut Unbounded).unwrap_err();
    assert!(
        matches!(error, SparsePseudoinverseError::Invalid(message) if message == expected),
        "{error}"
    );
}

#[test]
fn every_penrose_identity_has_an_independent_weakening_mutant() {
    let a = matrix(2, 2, &[1, 0, 0, 0], 1);
    verify_sparse_pseudoinverse(&a, &witness(a.clone()), &mut Unbounded).unwrap();
    // Each mutant satisfies the other three identities exactly.
    rejected(&a, matrix(2, 2, &[0, 0, 0, 0], 1), "A P A != A");
    rejected(&a, matrix(2, 2, &[1, 0, 0, 1], 1), "P A P != P");
    rejected(&a, matrix(2, 2, &[1, 1, 0, 0], 1), "A P is not symmetric");
    rejected(&a, matrix(2, 2, &[1, 0, 1, 0], 1), "P A is not symmetric");
}

#[test]
fn rectangular_rank_deficient_and_zero_area_claims_are_exact() {
    let cases = [
        (
            matrix(2, 2, &[1, 2, 2, 4], 1),
            matrix(2, 2, &[1, 2, 2, 4], 25),
        ),
        (
            matrix(2, 3, &[1, 0, 1, 0, 1, 1], 1),
            matrix(3, 2, &[2, -1, -1, 2, 1, 1], 3),
        ),
        (matrix(2, 3, &[0; 6], 1), matrix(3, 2, &[0; 6], 1)),
        (matrix(0, 3, &[], 1), matrix(3, 0, &[], 1)),
        (matrix(3, 0, &[], 1), matrix(0, 3, &[], 1)),
        (matrix(0, 0, &[], 1), matrix(0, 0, &[], 1)),
    ];
    for (a, p) in cases {
        let receipt = verify_sparse_pseudoinverse(&a, &witness(p), &mut Unbounded).unwrap();
        assert_eq!((receipt.rows(), receipt.columns()), (a.rows, a.columns));
    }
}

#[test]
fn minimum_norm_claim_rejects_a_different_exact_least_squares_minimizer() {
    let a = matrix(2, 2, &[1, 2, 2, 4], 1);
    let b = matrix(2, 2, &[3, 0, 5, 1], 1);
    let mut candidate = SparseMinimumNormWitness {
        schema_version: SPARSE_MINIMUM_NORM_SCHEMA_VERSION,
        pseudoinverse: witness(matrix(2, 2, &[1, 2, 2, 4], 25)),
        solution: matrix(2, 2, &[13, 2, 26, 4], 25),
    };
    let receipt = verify_sparse_minimum_norm(&a, &b, &candidate, &mut Unbounded).unwrap();
    assert_eq!(receipt.rhs_columns(), 2);
    // Adding [2,-1] to the first column leaves A*X unchanged, but increases norm.
    candidate.solution = matrix(2, 2, &[63, 2, 1, 4], 25);
    assert!(matches!(
        verify_sparse_minimum_norm(&a, &b, &candidate, &mut Unbounded),
        Err(SparsePseudoinverseError::Invalid("solution != P B"))
    ));
    candidate.solution = matrix(2, 2, &[13, 2, 26, 4], 25);
    let changed_b = matrix(2, 2, &[3, 0, 4, 1], 1);
    assert!(verify_sparse_minimum_norm(&a, &changed_b, &candidate, &mut Unbounded).is_err());
    candidate.schema_version += 1;
    assert!(verify_sparse_minimum_norm(&a, &b, &candidate, &mut Unbounded).is_err());
}

#[test]
fn empty_rhs_never_skips_pseudoinverse_acceptance() {
    let a = matrix(2, 2, &[1, 0, 0, 0], 1);
    let b = matrix(2, 0, &[], 1);
    let candidate = SparseMinimumNormWitness {
        schema_version: SPARSE_MINIMUM_NORM_SCHEMA_VERSION,
        pseudoinverse: witness(matrix(2, 2, &[1, 0, 0, 1], 1)),
        solution: matrix(2, 0, &[], 1),
    };
    assert!(matches!(
        verify_sparse_minimum_norm(&a, &b, &candidate, &mut Unbounded),
        Err(SparsePseudoinverseError::Invalid("P A P != P"))
    ));
}

#[test]
fn canonical_coordinates_shapes_schema_and_source_are_not_advisory() {
    let a = matrix(2, 2, &[1, 0, 0, 1], 1);
    let p = witness(a.clone());
    let mut mutations = Vec::new();
    let mut changed = p.clone();
    changed.schema_version += 1;
    mutations.push(changed);
    let mut changed = p.clone();
    changed.pseudoinverse.rows = 3;
    mutations.push(changed);
    let mut changed = p.clone();
    changed.pseudoinverse.entries.swap(0, 1);
    mutations.push(changed);
    let mut changed = p.clone();
    changed
        .pseudoinverse
        .entries
        .push(changed.pseudoinverse.entries[1].clone());
    mutations.push(changed);
    let mut changed = p.clone();
    changed.pseudoinverse.entries[0].value = BigRational::zero();
    mutations.push(changed);
    let mut changed = p.clone();
    changed.pseudoinverse.entries[1].column = 2;
    mutations.push(changed);
    for changed in mutations {
        assert!(verify_sparse_pseudoinverse(&a, &changed, &mut Unbounded).is_err());
    }
    let changed_source = matrix(2, 2, &[2, 0, 0, 1], 1);
    assert!(verify_sparse_pseudoinverse(&changed_source, &p, &mut Unbounded).is_err());
    let mut unknown = serde_json::to_value(&p).unwrap();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("verified".into(), true.into());
    assert!(serde_json::from_value::<SparsePseudoinverseWitness>(unknown).is_err());
    let encoded = serde_json::to_vec(&p).unwrap();
    let decoded: SparsePseudoinverseWitness = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, p);
    verify_sparse_pseudoinverse(&a, &decoded, &mut Unbounded).unwrap();
}

#[test]
fn oversized_coefficients_and_dimensions_fail_before_working_storage() {
    let mut a = matrix(1, 1, &[1], 1);
    a.entries[0].value = BigRational::from_integer(BigInt::one() << 8192u32);
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[Dimension::MemoryBytes.index()] = 0;
    let error = verify_sparse_pseudoinverse(
        &a,
        &witness(matrix(1, 1, &[1], 1)),
        &mut Budget::new(limits),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SparsePseudoinverseError::Reference(SparseRrefError::Limit("coefficient height"))
    ));
    let a = RationalSparseMatrix {
        rows: 8193,
        columns: 0,
        entries: vec![],
    };
    assert!(matches!(
        verify_sparse_pseudoinverse(&a, &witness(matrix(0, 0, &[], 1)), &mut Unbounded),
        Err(SparsePseudoinverseError::Reference(SparseRrefError::Limit(
            "matrix dimensions"
        )))
    ));
}

#[test]
fn sparse_input_does_not_authorize_unbounded_projector_fill() {
    let n = 513;
    let a = RationalSparseMatrix {
        rows: n,
        columns: 1,
        entries: (0..n)
            .map(|row| SparseRationalEntry {
                row,
                column: 0,
                value: BigRational::one(),
            })
            .collect(),
    };
    let p = RationalSparseMatrix {
        rows: 1,
        columns: n,
        entries: (0..n)
            .map(|column| SparseRationalEntry {
                row: 0,
                column,
                value: BigRational::new(1.into(), (n as u64).into()),
            })
            .collect(),
    };
    assert!(matches!(
        verify_sparse_pseudoinverse(&a, &witness(p), &mut Unbounded),
        Err(SparsePseudoinverseError::Reference(SparseRrefError::Limit(
            _
        ))) | Err(SparsePseudoinverseError::Reference(SparseRrefError::Meter(
            MeterError::Budget(_)
        )))
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
fn reference_work_and_receipt_publication_are_cumulative_and_cancellable() {
    let a = matrix(1, 1, &[2], 1);
    let b = matrix(1, 1, &[3], 1);
    let candidate = SparseMinimumNormWitness {
        schema_version: SPARSE_MINIMUM_NORM_SCHEMA_VERSION,
        pseudoinverse: witness(matrix(1, 1, &[1], 2)),
        solution: matrix(1, 1, &[3], 2),
    };
    let mut measured = Recording::default();
    verify_sparse_minimum_norm(&a, &b, &candidate, &mut measured).unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    verify_sparse_minimum_norm(&a, &b, &candidate, &mut Budget::new(limits)).unwrap();
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(
                verify_sparse_minimum_norm(&a, &b, &candidate, &mut Budget::new(short)).is_err()
            );
        }
    }
    for cancel_at in 1..=measured.checkpoints {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert!(
            matches!(
                verify_sparse_minimum_norm(&a, &b, &candidate, &mut meter),
                Err(SparsePseudoinverseError::Reference(SparseRrefError::Meter(
                    MeterError::Cancelled
                )))
            ),
            "safe point {cancel_at}"
        );
    }
}
