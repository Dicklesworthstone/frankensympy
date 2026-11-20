//! Hand-authored integer Smith witnesses. These tests use no optimizing
//! normal-form generator and distinguish integer invertibility from a valid
//! rational product or a rank-destroying transformation.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::integer_normal_form::{
    INTEGER_SMITH_SCHEMA_VERSION, IntegerSmithError, IntegerSmithWitness,
    MAX_INTEGER_SMITH_COEFFICIENT_BITS, MAX_INTEGER_SMITH_DIMENSION, verify_integer_smith,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}

fn matrix(rows: usize, columns: usize, values: &[(usize, usize, i64)]) -> RationalSparseMatrix {
    RationalSparseMatrix {
        rows,
        columns,
        entries: values
            .iter()
            .map(|&(row, column, value)| SparseRationalEntry {
                row,
                column,
                value: q(value),
            })
            .collect(),
    }
}

fn identity(size: usize) -> RationalSparseMatrix {
    RationalSparseMatrix {
        rows: size,
        columns: size,
        entries: (0..size)
            .map(|index| SparseRationalEntry {
                row: index,
                column: index,
                value: q(1),
            })
            .collect(),
    }
}

fn unchanged(source: &RationalSparseMatrix) -> IntegerSmithWitness {
    IntegerSmithWitness {
        schema_version: INTEGER_SMITH_SCHEMA_VERSION,
        diagonal: source.clone(),
        left: identity(source.rows),
        right: identity(source.columns),
        left_inverse: identity(source.rows),
        right_inverse: identity(source.columns),
    }
}

fn fixture() -> (RationalSparseMatrix, IntegerSmithWitness) {
    // A = [4 6; 3 9]. Subtract the second row from the first, then three
    // times the new first row from the second, then add 3*column 1 to
    // column 2. This gives diag(1,18), with both inverses written by hand.
    let source = matrix(2, 2, &[(0, 0, 4), (0, 1, 6), (1, 0, 3), (1, 1, 9)]);
    let witness = IntegerSmithWitness {
        schema_version: INTEGER_SMITH_SCHEMA_VERSION,
        diagonal: matrix(2, 2, &[(0, 0, 1), (1, 1, 18)]),
        left: matrix(2, 2, &[(0, 0, 1), (0, 1, -1), (1, 0, -3), (1, 1, 4)]),
        right: matrix(2, 2, &[(0, 0, 1), (0, 1, 3), (1, 1, 1)]),
        left_inverse: matrix(2, 2, &[(0, 0, 4), (0, 1, 1), (1, 0, 3), (1, 1, 1)]),
        right_inverse: matrix(2, 2, &[(0, 0, 1), (0, 1, -3), (1, 1, 1)]),
    };
    (source, witness)
}

#[test]
fn exact_integer_normal_form_and_portable_replay_bind_every_matrix() {
    let (source, witness) = fixture();
    let accepted = verify_integer_smith(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (accepted.rows(), accepted.columns(), accepted.rank()),
        (2, 2, 2)
    );
    assert_eq!(
        accepted.invariant_factors(),
        &[BigInt::from(1), BigInt::from(18)]
    );
    let encoded = serde_json::to_string(&witness).unwrap();
    let decoded: IntegerSmithWitness = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, witness);
    assert_eq!(
        accepted,
        verify_integer_smith(&source, &decoded, &mut Unbounded).unwrap()
    );
    assert_ne!(accepted.source_digest(), accepted.diagonal_digest());
    assert_ne!(accepted.left_digest(), accepted.right_digest());
    assert_ne!(accepted.left_digest(), accepted.left_inverse_digest());
    assert_ne!(accepted.right_digest(), accepted.right_inverse_digest());
}

#[test]
fn rectangular_forms_preserve_rank_and_all_coordinates() {
    let (mut source, mut witness) = fixture();
    source.columns = 3;
    witness.diagonal.columns = 3;
    for matrix in [&mut witness.right, &mut witness.right_inverse] {
        matrix.rows = 3;
        matrix.columns = 3;
        matrix.entries.push(SparseRationalEntry {
            row: 2,
            column: 2,
            value: q(1),
        });
    }
    let accepted = verify_integer_smith(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (accepted.rows(), accepted.columns(), accepted.rank()),
        (2, 3, 2)
    );

    let (mut source, mut witness) = fixture();
    source.rows = 3;
    source.entries.extend([
        SparseRationalEntry {
            row: 2,
            column: 0,
            value: q(7),
        },
        SparseRationalEntry {
            row: 2,
            column: 1,
            value: q(15),
        },
    ]);
    witness.diagonal.rows = 3;
    witness.left = matrix(
        3,
        3,
        &[
            (0, 0, 1),
            (0, 1, -1),
            (1, 0, -3),
            (1, 1, 4),
            (2, 0, -1),
            (2, 1, -1),
            (2, 2, 1),
        ],
    );
    witness.left_inverse = matrix(
        3,
        3,
        &[
            (0, 0, 4),
            (0, 1, 1),
            (1, 0, 3),
            (1, 1, 1),
            (2, 0, 7),
            (2, 1, 2),
            (2, 2, 1),
        ],
    );
    let accepted = verify_integer_smith(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (accepted.rows(), accepted.columns(), accepted.rank()),
        (3, 2, 2)
    );
    assert_eq!(
        accepted.invariant_factors(),
        &[BigInt::from(1), BigInt::from(18)]
    );
}

#[test]
fn zero_and_empty_shapes_require_genuine_unimodular_transformations() {
    for (rows, columns) in [(0, 0), (0, 3), (4, 0), (3, 5), (5, 3)] {
        let source = matrix(rows, columns, &[]);
        let witness = unchanged(&source);
        let accepted = verify_integer_smith(&source, &witness, &mut Unbounded).unwrap();
        assert_eq!(
            (accepted.rows(), accepted.columns(), accepted.rank()),
            (rows, columns, 0)
        );
        assert!(accepted.invariant_factors().is_empty());
        if rows > 0 {
            let mut fake = witness.clone();
            fake.left.entries.clear();
            assert!(verify_integer_smith(&source, &fake, &mut Unbounded).is_err());
        }
        if columns > 0 {
            let mut fake = witness;
            fake.right.entries.clear();
            assert!(verify_integer_smith(&source, &fake, &mut Unbounded).is_err());
        }
    }
}

#[test]
fn product_identity_without_integer_invertibility_cannot_destroy_rank_or_scale_the_module() {
    let source = matrix(1, 1, &[(0, 0, 1)]);
    let mut singular = unchanged(&source);
    singular.diagonal.entries.clear();
    singular.left.entries.clear();
    singular.left_inverse.entries.clear();
    assert!(matches!(
        verify_integer_smith(&source, &singular, &mut Unbounded),
        Err(IntegerSmithError::Invalid(_))
    ));
    for left_side in [true, false] {
        let mut scaled = unchanged(&source);
        scaled.diagonal.entries[0].value = q(2);
        let (transform, inverse) = if left_side {
            (&mut scaled.left, &mut scaled.left_inverse)
        } else {
            (&mut scaled.right, &mut scaled.right_inverse)
        };
        transform.entries[0].value = q(2);
        inverse.entries[0].value = BigRational::new(1.into(), 2.into());
        // U A V = D and both inverse identities hold over Q, but the
        // inverse is not integral: the Z-module has been changed.
        assert!(matches!(
            verify_integer_smith(&source, &scaled, &mut Unbounded),
            Err(IntegerSmithError::Invalid("noninteger coefficient"))
        ));
    }
}

#[test]
fn valid_equivalence_is_insufficient_without_positive_divisible_contiguous_diagonal() {
    for source in [
        matrix(2, 2, &[(0, 0, 2), (1, 1, 3)]),
        matrix(2, 2, &[(0, 0, -2), (1, 1, 6)]),
        matrix(2, 2, &[(1, 1, 6)]),
        matrix(2, 2, &[(0, 0, 2), (0, 1, 4), (1, 1, 6)]),
        matrix(3, 3, &[(0, 0, 2), (2, 2, 6)]),
    ] {
        assert!(matches!(
            verify_integer_smith(&source, &unchanged(&source), &mut Unbounded),
            Err(IntegerSmithError::Invalid(_))
        ));
    }
    let source = matrix(4, 5, &[(0, 0, 2), (1, 1, 6), (2, 2, 30)]);
    let accepted = verify_integer_smith(&source, &unchanged(&source), &mut Unbounded).unwrap();
    assert_eq!(accepted.rank(), 3);
    assert_eq!(
        accepted.invariant_factors(),
        &[BigInt::from(2), BigInt::from(6), BigInt::from(30)]
    );
}

#[test]
fn original_source_and_every_witness_component_are_checked() {
    let (source, witness) = fixture();
    for mutation in 0..8 {
        let mut source = source.clone();
        let mut witness = witness.clone();
        match mutation {
            0 => source.entries[0].value = q(5),
            1 => witness.diagonal.entries[1].value = q(36),
            2 => witness.left.entries[0].value = q(2),
            3 => witness.right.entries[0].value = q(2),
            4 => witness.left_inverse.entries[0].value = q(5),
            5 => witness.right_inverse.entries[0].value = q(2),
            6 => witness.schema_version += 1,
            _ => {
                witness.diagonal.entries.pop().unwrap();
            }
        }
        assert!(
            verify_integer_smith(&source, &witness, &mut Unbounded).is_err(),
            "accepted mutation {mutation}"
        );
    }
    let diagonal = matrix(2, 2, &[(0, 0, 1), (1, 1, 18)]);
    let original = verify_integer_smith(&source, &witness, &mut Unbounded).unwrap();
    let unchanged = verify_integer_smith(&diagonal, &unchanged(&diagonal), &mut Unbounded).unwrap();
    assert_ne!(original.source_digest(), unchanged.source_digest());
    assert_eq!(original.diagonal_digest(), unchanged.diagonal_digest());
}

#[test]
fn source_and_witness_storage_cannot_hide_fractional_or_malformed_entries() {
    let (source, witness) = fixture();
    for target in 0..6 {
        for mutation in 0..7 {
            let mut source = source.clone();
            let mut witness = witness.clone();
            let matrix = match target {
                0 => &mut source,
                1 => &mut witness.diagonal,
                2 => &mut witness.left,
                3 => &mut witness.right,
                4 => &mut witness.left_inverse,
                _ => &mut witness.right_inverse,
            };
            match mutation {
                0 => matrix.entries[0].row = matrix.rows,
                1 => matrix.entries[0].column = matrix.columns,
                2 => matrix.entries.swap(0, 1),
                3 => matrix.entries.insert(1, matrix.entries[0].clone()),
                4 => matrix.entries[0].value = q(0),
                5 => matrix.entries[0].value = BigRational::new(1.into(), 2.into()),
                _ => matrix.columns += 1,
            }
            assert!(
                verify_integer_smith(&source, &witness, &mut Unbounded).is_err(),
                "matrix {target} mutation {mutation}"
            );
        }
    }
}

#[test]
fn dimensions_and_coefficient_growth_are_admitted_before_reference_products() {
    let (source, witness) = fixture();
    for dimension in [MAX_INTEGER_SMITH_DIMENSION + 1, usize::MAX] {
        let mut too_large = source.clone();
        too_large.rows = dimension;
        assert!(matches!(
            verify_integer_smith(&too_large, &witness, &mut Unbounded),
            Err(IntegerSmithError::Reference(SparseRrefError::Limit(_)))
        ));
    }
    let mut oversized = source;
    oversized.entries[0].value =
        BigRational::from_integer(BigInt::one() << MAX_INTEGER_SMITH_COEFFICIENT_BITS as u32);
    assert!(matches!(
        verify_integer_smith(&oversized, &witness, &mut Unbounded),
        Err(IntegerSmithError::Reference(SparseRrefError::Limit(_)))
    ));

    let source = matrix(1, 1, &[(0, 0, 1)]);
    let mut growing = unchanged(&source);
    let huge = BigRational::from_integer(BigInt::one() << 5_000);
    growing.left.entries[0].value = huge.clone();
    growing.left_inverse.entries[0].value = huge;
    assert!(matches!(
        verify_integer_smith(&source, &growing, &mut Unbounded),
        Err(IntegerSmithError::Reference(SparseRrefError::Limit(
            "multiplication intermediate"
        )))
    ));
}

#[test]
fn portable_witness_decoder_bounds_matrices_and_rejects_extra_authority() {
    let (_, witness) = fixture();
    let base = serde_json::to_value(&witness).unwrap();
    let mut forged = base.clone();
    forged["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<IntegerSmithWitness>(forged).is_err());
    for field in ["diagonal", "left", "right", "left_inverse", "right_inverse"] {
        let mut oversized = base.clone();
        oversized[field]["rows"] = serde_json::json!(MAX_INTEGER_SMITH_DIMENSION + 1);
        assert!(serde_json::from_value::<IntegerSmithWitness>(oversized).is_err());
        let mut extra = base.clone();
        extra[field]["verified"] = serde_json::json!(true);
        assert!(serde_json::from_value::<IntegerSmithWitness>(extra).is_err());
        let mut fractional = base.clone();
        fractional[field]["entries"][0]["value"] =
            serde_json::to_value(BigRational::new(1.into(), 2.into())).unwrap();
        assert!(serde_json::from_value::<IntegerSmithWitness>(fractional).is_err());
        let mut too_many = base.clone();
        let entry = too_many[field]["entries"][0].clone();
        too_many[field]["entries"] = serde_json::Value::Array(vec![
            entry;
            MAX_INTEGER_SMITH_DIMENSION * MAX_INTEGER_SMITH_DIMENSION
                + 1
        ]);
        assert!(serde_json::from_value::<IntegerSmithWitness>(too_many).is_err());
    }
}

#[derive(Default)]
struct SafePoints {
    seen: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for SafePoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.seen += 1;
        if self.cancel_at == Some(self.seen) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_observed_safe_point_cancels_without_an_accepted_receipt() {
    let source = matrix(1, 1, &[(0, 0, 2)]);
    let witness = unchanged(&source);
    let mut observation = SafePoints::default();
    verify_integer_smith(&source, &witness, &mut observation).unwrap();
    assert!(observation.seen > 20);
    for point in 1..=observation.seen {
        let mut meter = SafePoints {
            seen: 0,
            cancel_at: Some(point),
        };
        assert!(
            matches!(
                verify_integer_smith(&source, &witness, &mut meter),
                Err(IntegerSmithError::Reference(SparseRrefError::Meter(
                    MeterError::Cancelled
                )))
            ),
            "cancellation point {point}"
        );
    }
}

#[test]
fn exhausted_work_storage_and_allocations_remain_terminal() {
    let (source, witness) = fixture();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(100_000_000, 0);
        limits.dimensions[dimension.index()] = 0;
        let mut meter = Budget::new(limits);
        assert!(
            matches!(
                verify_integer_smith(&source, &witness, &mut meter),
                Err(IntegerSmithError::Reference(SparseRrefError::Meter(
                    MeterError::Budget(_)
                )))
            ),
            "budget dimension {dimension:?}"
        );
    }
}

#[test]
fn repeat_verification_consumes_the_same_callers_budget() {
    let (source, witness) = fixture();
    let initial = 100_000_000;
    let mut observation = Budget::new(BudgetLimits::uniform(initial, 0));
    verify_integer_smith(&source, &witness, &mut observation).unwrap();
    let used = initial - observation.remaining(Dimension::ComputeSteps);
    assert!(used > 0);
    let mut limits = BudgetLimits::uniform(initial, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used;
    let mut bounded = Budget::new(limits);
    verify_integer_smith(&source, &witness, &mut bounded).unwrap();
    assert_eq!(bounded.remaining(Dimension::ComputeSteps), 0);
    assert!(matches!(
        verify_integer_smith(&source, &witness, &mut bounded),
        Err(IntegerSmithError::Reference(SparseRrefError::Meter(
            MeterError::Budget(_)
        )))
    ));
}
