//! Hand-authored Hermite witnesses, independent of the matrix generator.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::integer_hermite::{
    INTEGER_HERMITE_SCHEMA_VERSION, IntegerHermiteError, IntegerHermiteWitness,
    MAX_INTEGER_HERMITE_COEFFICIENT_BITS, MAX_INTEGER_HERMITE_DIMENSION, verify_integer_hermite,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}

fn matrix(rows: usize, columns: usize, values: &[i64]) -> RationalSparseMatrix {
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
                value: q(value),
            })
            .collect(),
    }
}

fn identity(n: usize) -> RationalSparseMatrix {
    RationalSparseMatrix {
        rows: n,
        columns: n,
        entries: (0..n)
            .map(|i| SparseRationalEntry {
                row: i,
                column: i,
                value: q(1),
            })
            .collect(),
    }
}

fn unchanged(source: &RationalSparseMatrix) -> IntegerHermiteWitness {
    IntegerHermiteWitness {
        schema_version: INTEGER_HERMITE_SCHEMA_VERSION,
        normal_form: source.clone(),
        right: identity(source.columns),
        right_inverse: identity(source.columns),
    }
}

fn fixture() -> (RationalSparseMatrix, IntegerHermiteWitness) {
    // Three times column 1 minus column 2 is (6,0); original column 1
    // is (4,3). V has determinant one and the inverse is given explicitly.
    let source = matrix(2, 2, &[4, 6, 3, 9]);
    let witness = IntegerHermiteWitness {
        schema_version: INTEGER_HERMITE_SCHEMA_VERSION,
        normal_form: matrix(2, 2, &[6, 4, 0, 3]),
        right: matrix(2, 2, &[3, 1, -1, 0]),
        right_inverse: matrix(2, 2, &[0, -1, 1, 3]),
    };
    (source, witness)
}

#[test]
fn complete_lattice_receipt_binds_source_form_transforms_and_portable_replay() {
    let (source, witness) = fixture();
    let receipt = verify_integer_hermite(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.rows(), receipt.columns(), receipt.rank()),
        (2, 2, 2)
    );
    assert_eq!(receipt.pivot_rows(), &[0, 1]);
    assert_ne!(receipt.source_digest(), receipt.normal_form_digest());
    assert_ne!(receipt.right_digest(), receipt.right_inverse_digest());
    let bytes = serde_json::to_vec(&witness).unwrap();
    let decoded: IntegerHermiteWitness = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(witness, decoded);
    assert_eq!(
        receipt,
        verify_integer_hermite(&source, &decoded, &mut Unbounded).unwrap()
    );
    let other = verify_integer_hermite(
        &witness.normal_form,
        &unchanged(&witness.normal_form),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.normal_form_digest(), other.normal_form_digest());
    assert_ne!(receipt.source_digest(), other.source_digest());
}

#[test]
fn dependent_generators_are_removed_without_losing_the_integer_lattice() {
    let source = matrix(2, 3, &[2, 4, 6, 4, 8, 12]);
    let witness = IntegerHermiteWitness {
        schema_version: INTEGER_HERMITE_SCHEMA_VERSION,
        normal_form: matrix(2, 1, &[2, 4]),
        right: matrix(3, 3, &[-2, -3, 1, 1, 0, 0, 0, 1, 0]),
        right_inverse: matrix(3, 3, &[0, 1, 0, 0, 0, 1, 1, 2, 3]),
    };
    let receipt = verify_integer_hermite(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.rows(), receipt.columns(), receipt.rank()),
        (2, 3, 1)
    );
    assert_eq!(receipt.pivot_rows(), &[1]);
    let mut omitted = witness;
    omitted.normal_form = matrix(2, 0, &[]);
    assert!(verify_integer_hermite(&source, &omitted, &mut Unbounded).is_err());
}

#[test]
fn noncontiguous_pivots_and_arbitrary_nonpivot_rows_are_valid() {
    let source = matrix(4, 2, &[-7, 20, 2, 1, 0, 3, 0, 0]);
    let receipt = verify_integer_hermite(&source, &unchanged(&source), &mut Unbounded).unwrap();
    assert_eq!(receipt.pivot_rows(), &[1, 2]);
}

#[test]
fn zero_and_zero_area_forms_still_require_unimodularity() {
    for (rows, columns) in [(0, 0), (0, 4), (3, 0), (4, 5), (5, 3)] {
        let source = matrix(rows, columns, &vec![0; rows * columns]);
        let mut witness = unchanged(&source);
        witness.normal_form = matrix(rows, 0, &[]);
        let receipt = verify_integer_hermite(&source, &witness, &mut Unbounded).unwrap();
        assert_eq!(
            (receipt.rows(), receipt.columns(), receipt.rank()),
            (rows, columns, 0)
        );
        if columns != 0 {
            witness.right.entries.clear();
            assert!(verify_integer_hermite(&source, &witness, &mut Unbounded).is_err());
        }
    }
}

#[test]
fn correct_rational_span_or_one_way_product_is_not_lattice_equality() {
    let source = matrix(1, 1, &[1]);
    let mut witness = unchanged(&source);
    witness.normal_form = matrix(1, 1, &[2]);
    witness.right = matrix(1, 1, &[2]);
    witness.right_inverse.entries[0].value = BigRational::new(1.into(), 2.into());
    assert!(verify_integer_hermite(&source, &witness, &mut Unbounded).is_err());
    witness.right_inverse = identity(1);
    assert!(verify_integer_hermite(&source, &witness, &mut Unbounded).is_err());

    let source = identity(2);
    let mut witness = unchanged(&source);
    witness.normal_form = matrix(2, 1, &[0, 1]);
    witness.right = matrix(2, 2, &[0, 0, 0, 1]);
    witness.right_inverse = witness.right.clone();
    assert!(verify_integer_hermite(&source, &witness, &mut Unbounded).is_err());
}

#[test]
fn equivalence_alone_does_not_establish_canonical_hermite_form() {
    for source in [
        matrix(2, 2, &[2, 2, 0, 3]),  // remainder equal to pivot
        matrix(2, 2, &[2, -1, 0, 3]), // negative remainder
        matrix(2, 2, &[-2, 1, 0, 3]), // negative pivot
        matrix(2, 2, &[0, 1, 2, 0]),  // reversed pivot rows
        matrix(2, 2, &[1, 1, 2, 3]),  // shared pivot row
        matrix(2, 2, &[0, 1, 0, 3]),  // retained zero column
    ] {
        assert!(matches!(
            verify_integer_hermite(&source, &unchanged(&source), &mut Unbounded),
            Err(IntegerHermiteError::Invalid(_))
        ));
    }
}

#[test]
fn original_source_schema_and_each_witness_component_are_checked() {
    let (source, witness) = fixture();
    for mutation in 0..8 {
        let mut source = source.clone();
        let mut witness = witness.clone();
        match mutation {
            0 => source.entries[0].value = q(5),
            1 => witness.normal_form.entries[0].value = q(7),
            2 => witness.right.entries[0].value = q(4),
            3 => witness.right_inverse.entries[0].value = q(-2),
            4 => witness.schema_version += 1,
            5 => witness.normal_form.rows += 1,
            6 => witness.right.columns += 1,
            _ => {
                witness.normal_form.entries.pop().unwrap();
            }
        }
        assert!(
            verify_integer_hermite(&source, &witness, &mut Unbounded).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn malformed_fractional_and_oversized_storage_is_rejected_before_products() {
    let (source, witness) = fixture();
    for target in 0..4 {
        for mutation in 0..8 {
            let mut source = source.clone();
            let mut witness = witness.clone();
            let matrix = match target {
                0 => &mut source,
                1 => &mut witness.normal_form,
                2 => &mut witness.right,
                _ => &mut witness.right_inverse,
            };
            match mutation {
                0 => matrix.entries[0].row = matrix.rows,
                1 => matrix.entries[0].column = matrix.columns,
                2 => matrix.entries.swap(0, 1),
                3 => matrix.entries.insert(1, matrix.entries[0].clone()),
                4 => matrix.entries[0].value = q(0),
                5 => matrix.entries[0].value = BigRational::new(1.into(), 2.into()),
                6 => matrix.rows = MAX_INTEGER_HERMITE_DIMENSION + 1,
                _ => {
                    matrix.entries[0].value = BigRational::from_integer(
                        BigInt::one() << MAX_INTEGER_HERMITE_COEFFICIENT_BITS as u32,
                    )
                }
            }
            assert!(
                verify_integer_hermite(&source, &witness, &mut Unbounded).is_err(),
                "matrix {target}, mutation {mutation}"
            );
        }
    }
}

#[test]
fn decoder_bounds_nested_matrices_and_cannot_deserialize_a_verified_flag() {
    let (_, witness) = fixture();
    let value = serde_json::to_value(witness).unwrap();
    let mut unknown = value.clone();
    unknown["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<IntegerHermiteWitness>(unknown).is_err());
    for field in ["normal_form", "right", "right_inverse"] {
        let mut oversized = value.clone();
        oversized[field]["rows"] = serde_json::json!(65);
        assert!(serde_json::from_value::<IntegerHermiteWitness>(oversized).is_err());
        let mut fractional = value.clone();
        fractional[field]["entries"][0]["value"] =
            serde_json::to_value(BigRational::new(1.into(), 2.into())).unwrap();
        assert!(serde_json::from_value::<IntegerHermiteWitness>(fractional).is_err());
        let mut unknown = value.clone();
        unknown[field]["verified"] = serde_json::json!(true);
        assert!(serde_json::from_value::<IntegerHermiteWitness>(unknown).is_err());
        let mut too_many = value.clone();
        let entry = too_many[field]["entries"][0].clone();
        too_many[field]["entries"] = serde_json::Value::Array(vec![entry; 4097]);
        assert!(serde_json::from_value::<IntegerHermiteWitness>(too_many).is_err());
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
fn every_safe_point_including_publication_can_cancel_without_receipt() {
    let source = matrix(1, 1, &[2]);
    let witness = unchanged(&source);
    let mut observed = SafePoints::default();
    verify_integer_hermite(&source, &witness, &mut observed).unwrap();
    assert!(observed.seen > 20);
    for point in 1..=observed.seen {
        let mut meter = SafePoints {
            seen: 0,
            cancel_at: Some(point),
        };
        assert!(
            matches!(
                verify_integer_hermite(&source, &witness, &mut meter),
                Err(IntegerHermiteError::Reference(SparseRrefError::Meter(
                    MeterError::Cancelled
                )))
            ),
            "safe point {point}"
        );
    }
}

#[test]
fn budgets_are_terminal_and_cumulative_across_replay() {
    let (source, witness) = fixture();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(100_000_000, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(
            verify_integer_hermite(&source, &witness, &mut Budget::new(limits)),
            Err(IntegerHermiteError::Reference(SparseRrefError::Meter(
                MeterError::Budget(_)
            )))
        ));
    }
    let mut observed = Budget::new(BudgetLimits::uniform(100_000_000, 0));
    verify_integer_hermite(&source, &witness, &mut observed).unwrap();
    let used = 100_000_000 - observed.remaining(Dimension::ComputeSteps);
    let mut limits = BudgetLimits::uniform(100_000_000, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used;
    let mut shared = Budget::new(limits);
    verify_integer_hermite(&source, &witness, &mut shared).unwrap();
    assert_eq!(shared.remaining(Dimension::ComputeSteps), 0);
    assert!(verify_integer_hermite(&source, &witness, &mut shared).is_err());
}
