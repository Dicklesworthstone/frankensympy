//! Complete affine families require the whole nullspace, not just annihilation.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::sparse_pseudoinverse::{
    SPARSE_LEAST_SQUARES_FAMILY_SCHEMA_VERSION, SPARSE_MINIMUM_NORM_SCHEMA_VERSION,
    SPARSE_PSEUDOINVERSE_SCHEMA_VERSION, SparseLeastSquaresFamilyWitness, SparseMinimumNormWitness,
    SparsePseudoinverseError, SparsePseudoinverseWitness, verify_sparse_least_squares_family,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};
use num_traits::{One, Zero};

fn matrix(rows: usize, columns: usize, values: &[i64], denominator: i64) -> RationalSparseMatrix {
    assert_eq!(values.len(), rows * columns);
    RationalSparseMatrix {
        rows,
        columns,
        entries: values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(index, &value)| SparseRationalEntry {
                row: index / columns,
                column: index % columns,
                value: BigRational::new(value.into(), denominator.into()),
            })
            .collect(),
    }
}

fn witness(
    pseudoinverse: RationalSparseMatrix,
    solution: RationalSparseMatrix,
    projector: RationalSparseMatrix,
) -> SparseLeastSquaresFamilyWitness {
    SparseLeastSquaresFamilyWitness {
        schema_version: SPARSE_LEAST_SQUARES_FAMILY_SCHEMA_VERSION,
        minimum_norm: SparseMinimumNormWitness {
            schema_version: SPARSE_MINIMUM_NORM_SCHEMA_VERSION,
            pseudoinverse: SparsePseudoinverseWitness {
                schema_version: SPARSE_PSEUDOINVERSE_SCHEMA_VERSION,
                pseudoinverse,
            },
            solution,
        },
        projector,
    }
}

fn singular_case() -> (
    RationalSparseMatrix,
    RationalSparseMatrix,
    SparseLeastSquaresFamilyWitness,
) {
    (
        matrix(2, 2, &[1, 2, 2, 4], 1),
        matrix(2, 2, &[3, 0, 5, 1], 1),
        witness(
            matrix(2, 2, &[1, 2, 2, 4], 25),
            matrix(2, 2, &[13, 2, 26, 4], 25),
            matrix(2, 2, &[4, -2, -2, 1], 5),
        ),
    )
}

#[test]
fn exact_rank_deficient_inconsistent_family_round_trips_portable_evidence() {
    let (a, b, candidate) = singular_case();
    let receipt = verify_sparse_least_squares_family(&a, &b, &candidate, &mut Unbounded).unwrap();
    assert_eq!(receipt.minimum_norm().rhs_columns(), 2);
    assert_eq!(receipt.minimum_norm().pseudoinverse().columns(), 2);
    assert_ne!(
        receipt.projector_digest(),
        receipt.minimum_norm().solution_digest()
    );
    let encoded = serde_json::to_vec(&candidate).unwrap();
    let decoded: SparseLeastSquaresFamilyWitness = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, candidate);
    assert_eq!(
        verify_sparse_least_squares_family(&a, &b, &decoded, &mut Unbounded).unwrap(),
        receipt
    );
}

#[test]
fn annihilation_symmetry_and_idempotence_cannot_replace_completeness() {
    let a = matrix(3, 3, &[1, 0, 0, 0, 0, 0, 0, 0, 0], 1);
    let b = matrix(3, 1, &[2, 4, 5], 1);
    let correct = witness(
        a.clone(),
        matrix(3, 1, &[2, 0, 0], 1),
        matrix(3, 3, &[0, 0, 0, 0, 1, 0, 0, 0, 1], 1),
    );
    verify_sparse_least_squares_family(&a, &b, &correct, &mut Unbounded).unwrap();
    // The first three mutants are symmetric idempotent nullspace projectors,
    // but omit one or both free coordinates. The last spans the right space
    // with an incorrect scale and still satisfies A*N=0 and symmetry.
    for diagonal in [[0, 0], [1, 0], [0, 1], [2, 2]] {
        let mut changed = correct.clone();
        changed.projector = matrix(3, 3, &[0, 0, 0, 0, diagonal[0], 0, 0, 0, diagonal[1]], 1);
        assert!(matches!(
            verify_sparse_least_squares_family(&a, &b, &changed, &mut Unbounded),
            Err(SparsePseudoinverseError::Invalid("projector + P A != I"))
        ));
    }
}

#[test]
fn families_still_require_original_source_penrose_and_minimum_norm_checks() {
    let (a, b, candidate) = singular_case();
    let mut changed = candidate.clone();
    // A different minimizer is not the required minimum-norm particular.
    changed.minimum_norm.solution = matrix(2, 2, &[63, 2, 1, 4], 25);
    assert!(matches!(
        verify_sparse_least_squares_family(&a, &b, &changed, &mut Unbounded),
        Err(SparsePseudoinverseError::Invalid("solution != P B"))
    ));
    assert!(
        verify_sparse_least_squares_family(
            &matrix(2, 2, &[1, 2, 2, 5], 1),
            &b,
            &candidate,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_sparse_least_squares_family(
            &a,
            &matrix(2, 2, &[4, 0, 5, 1], 1),
            &candidate,
            &mut Unbounded
        )
        .is_err()
    );
    // Even an empty RHS must replay the full four-identity theorem.
    let a = matrix(2, 2, &[1, 0, 0, 0], 1);
    let b = matrix(2, 0, &[], 1);
    let bad_p = witness(
        matrix(2, 2, &[1, 0, 0, 1], 1),
        matrix(2, 0, &[], 1),
        matrix(2, 2, &[0, 0, 0, 1], 1),
    );
    assert!(matches!(
        verify_sparse_least_squares_family(&a, &b, &bad_p, &mut Unbounded),
        Err(SparsePseudoinverseError::Invalid("P A P != P"))
    ));
}

#[test]
fn zero_rank_and_zero_area_families_keep_every_free_coordinate() {
    for (rows, columns) in [(0, 0), (0, 3), (3, 0), (2, 3)] {
        let a = matrix(rows, columns, &vec![0; rows * columns], 1);
        let p = matrix(columns, rows, &vec![0; rows * columns], 1);
        let projector = RationalSparseMatrix {
            rows: columns,
            columns,
            entries: (0..columns)
                .map(|i| SparseRationalEntry {
                    row: i,
                    column: i,
                    value: BigRational::one(),
                })
                .collect(),
        };
        for rhs_columns in [0, 2] {
            let b = matrix(rows, rhs_columns, &vec![7; rows * rhs_columns], 1);
            let x = matrix(columns, rhs_columns, &vec![0; columns * rhs_columns], 1);
            let candidate = witness(p.clone(), x, projector.clone());
            verify_sparse_least_squares_family(&a, &b, &candidate, &mut Unbounded).unwrap();
        }
    }
}

#[test]
fn canonical_projectors_shapes_and_nested_schemas_are_mandatory() {
    let (a, b, candidate) = singular_case();
    let mut mutations = Vec::new();
    let mut changed = candidate.clone();
    changed.schema_version += 1;
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.minimum_norm.schema_version += 1;
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.minimum_norm.pseudoinverse.schema_version += 1;
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.projector.rows = 3;
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.projector.entries.swap(0, 1);
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed
        .projector
        .entries
        .push(changed.projector.entries[3].clone());
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.projector.entries[0].value = BigRational::zero();
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.projector.entries[3].column = 2;
    mutations.push(changed);
    for changed in mutations {
        assert!(verify_sparse_least_squares_family(&a, &b, &changed, &mut Unbounded).is_err());
    }
    let mut encoded = serde_json::to_value(&candidate).unwrap();
    encoded
        .as_object_mut()
        .unwrap()
        .insert("verified".into(), true.into());
    assert!(serde_json::from_value::<SparseLeastSquaresFamilyWitness>(encoded).is_err());
    let mut changed = candidate;
    changed.projector.entries[0].value = BigRational::from_integer(BigInt::one() << 8192u32);
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[Dimension::MemoryBytes.index()] = 0;
    assert!(matches!(
        verify_sparse_least_squares_family(&a, &b, &changed, &mut Budget::new(limits)),
        Err(SparsePseudoinverseError::Reference(SparseRrefError::Limit(
            "coefficient height"
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
fn complete_family_replay_and_receipt_share_one_cancellable_budget() {
    let a = matrix(1, 2, &[1, 0], 1);
    let b = matrix(1, 1, &[2], 1);
    let candidate = witness(
        matrix(2, 1, &[1, 0], 1),
        matrix(2, 1, &[2, 0], 1),
        matrix(2, 2, &[0, 0, 0, 1], 1),
    );
    let mut measured = Recording::default();
    verify_sparse_least_squares_family(&a, &b, &candidate, &mut measured).unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    verify_sparse_least_squares_family(&a, &b, &candidate, &mut Budget::new(limits)).unwrap();
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                verify_sparse_least_squares_family(&a, &b, &candidate, &mut Budget::new(short)),
                Err(SparsePseudoinverseError::Reference(SparseRrefError::Meter(
                    MeterError::Budget(_)
                )))
            ));
        }
    }
    for cancel_at in 1..=measured.checkpoints {
        let mut meter = Recording {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert!(
            matches!(
                verify_sparse_least_squares_family(&a, &b, &candidate, &mut meter),
                Err(SparsePseudoinverseError::Reference(SparseRrefError::Meter(
                    MeterError::Cancelled
                )))
            ),
            "safe point {cancel_at}"
        );
    }
}
