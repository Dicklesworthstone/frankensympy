//! Hand-authored LLL witnesses, independent of the reduction generator.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::integer_lll::{
    INTEGER_LLL_SCHEMA_VERSION, IntegerLllError, IntegerLllWitness,
    MAX_INTEGER_LLL_COEFFICIENT_BITS, MAX_INTEGER_LLL_COLUMNS, MAX_INTEGER_LLL_ROWS,
    verify_integer_lll,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}

fn ratio(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn delta() -> BigRational {
    ratio(3, 4)
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

fn unchanged(source: &RationalSparseMatrix) -> IntegerLllWitness {
    IntegerLllWitness {
        schema_version: INTEGER_LLL_SCHEMA_VERSION,
        basis: source.clone(),
        transform: identity(source.rows),
        transform_inverse: identity(source.rows),
    }
}

fn fixture() -> (RationalSparseMatrix, IntegerLllWitness) {
    // The reduced rows have squared orthogonal norms 1, 2, 9/2.
    // Their only nonzero Gram--Schmidt coefficient is mu[2,1] = 1/2.
    let source = matrix(3, 3, &[1, 1, 1, -1, 0, 2, 3, 5, 6]);
    let witness = IntegerLllWitness {
        schema_version: INTEGER_LLL_SCHEMA_VERSION,
        basis: matrix(3, 3, &[0, 1, 0, 1, 0, 1, -1, 0, 2]),
        transform: matrix(3, 3, &[-4, -1, 1, 5, 1, -1, 0, 1, 0]),
        transform_inverse: matrix(3, 3, &[1, 1, 0, 0, 0, 1, 5, 4, 1]),
    };
    (source, witness)
}

#[test]
fn receipt_binds_complete_lattice_source_basis_transforms_and_exact_delta() {
    let (source, witness) = fixture();
    let receipt = verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.rows(), receipt.columns(), receipt.rank()),
        (3, 3, 3)
    );
    assert_eq!(receipt.delta(), &delta());
    assert_ne!(receipt.source_digest(), receipt.basis_digest());
    assert_ne!(
        receipt.transform_digest(),
        receipt.transform_inverse_digest()
    );

    let bytes = serde_json::to_vec(&witness).unwrap();
    let decoded: IntegerLllWitness = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded, witness);
    assert_eq!(
        receipt,
        verify_integer_lll(&source, &delta(), &decoded, &mut Unbounded).unwrap()
    );

    let other = verify_integer_lll(
        &witness.basis,
        &delta(),
        &unchanged(&witness.basis),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.basis_digest(), other.basis_digest());
    assert_ne!(receipt.source_digest(), other.source_digest());
    assert_ne!(receipt.transform_digest(), other.transform_digest());

    let weaker = verify_integer_lll(&source, &ratio(3, 5), &witness, &mut Unbounded).unwrap();
    assert_eq!(weaker.delta(), &ratio(3, 5));
    assert_ne!(receipt, weaker);
}

#[test]
fn rectangular_and_maximum_shape_bases_keep_all_row_directions() {
    let source = matrix(2, 4, &[1, 0, 0, 0, 0, 1, 0, 0]);
    let receipt =
        verify_integer_lll(&source, &delta(), &unchanged(&source), &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.rows(), receipt.columns(), receipt.rank()),
        (2, 4, 2)
    );

    let mut source = identity(MAX_INTEGER_LLL_ROWS);
    source.columns = MAX_INTEGER_LLL_COLUMNS;
    let receipt =
        verify_integer_lll(&source, &delta(), &unchanged(&source), &mut Unbounded).unwrap();
    assert_eq!(receipt.rank(), MAX_INTEGER_LLL_ROWS);
    assert_eq!(receipt.columns(), MAX_INTEGER_LLL_COLUMNS);
}

#[test]
fn empty_and_one_zero_row_bases_have_explicit_rank_zero() {
    for (rows, columns) in [(0, 0), (0, 7), (0, 64), (1, 1), (1, 5)] {
        let source = matrix(rows, columns, &vec![0; rows * columns]);
        let witness = unchanged(&source);
        let receipt = verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).unwrap();
        assert_eq!(
            (receipt.rows(), receipt.columns(), receipt.rank()),
            (rows, columns, 0)
        );
        if rows == 1 {
            let mut bad = witness;
            bad.transform.entries.clear();
            assert!(verify_integer_lll(&source, &delta(), &bad, &mut Unbounded).is_err());
        }
    }
}

#[test]
fn sign_changes_are_allowed_and_do_not_claim_a_canonical_basis() {
    let source = matrix(1, 3, &[2, -3, 0]);
    let witness = IntegerLllWitness {
        schema_version: INTEGER_LLL_SCHEMA_VERSION,
        basis: matrix(1, 3, &[-2, 3, 0]),
        transform: matrix(1, 1, &[-1]),
        transform_inverse: matrix(1, 1, &[-1]),
    };
    let receipt = verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.rank(), 1);
}

#[test]
fn dependent_multirow_sources_and_more_rows_than_columns_are_rejected() {
    for source in [
        matrix(2, 2, &[1, 2, 2, 4]),
        matrix(2, 2, &[1, 0, 0, 0]),
        matrix(2, 2, &[0, 0, 0, 0]),
        matrix(2, 1, &[1, 2]),
        matrix(1, 0, &[]),
    ] {
        assert!(
            verify_integer_lll(&source, &delta(), &unchanged(&source), &mut Unbounded).is_err()
        );
    }
}

#[test]
fn lattice_equivalence_alone_does_not_prove_either_reduction_condition() {
    let size_failure = matrix(2, 2, &[1, 0, 1, 1]);
    assert!(matches!(
        verify_integer_lll(
            &size_failure,
            &delta(),
            &unchanged(&size_failure),
            &mut Unbounded,
        ),
        Err(IntegerLllError::Invalid("basis is not size reduced"))
    ));
    let lovasz_failure = matrix(2, 2, &[2, 0, 0, 1]);
    assert!(matches!(
        verify_integer_lll(
            &lovasz_failure,
            &delta(),
            &unchanged(&lovasz_failure),
            &mut Unbounded,
        ),
        Err(IntegerLllError::Invalid("basis violates Lovasz inequality"))
    ));
}

#[test]
fn both_half_ties_and_lovasz_equality_are_accepted_exactly() {
    for sign in [-1, 1] {
        let source = matrix(2, 3, &[2, 0, 0, sign, 1, 1]);
        verify_integer_lll(&source, &delta(), &unchanged(&source), &mut Unbounded).unwrap();
        // This differs from 3/4 by less than one binary64 ulp. Acceptance
        // must use the exact requested rational, not its rounded float.
        let power = BigInt::one() << 100u32;
        let stronger = BigRational::new(
            &power * BigInt::from(3) + BigInt::one(),
            &power * BigInt::from(4),
        );
        assert!(matches!(
            verify_integer_lll(&source, &stronger, &unchanged(&source), &mut Unbounded),
            Err(IntegerLllError::Invalid("basis violates Lovasz inequality"))
        ));
    }
}

#[test]
fn authoritative_delta_cannot_be_weakened_or_substituted_by_a_witness() {
    let source = matrix(2, 2, &[2, 0, 1, 1]);
    let witness = unchanged(&source);
    let receipt = verify_integer_lll(&source, &ratio(1, 2), &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.delta(), &ratio(1, 2));
    assert!(verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).is_err());
    for invalid in [q(-1), q(0), ratio(1, 4), q(1), q(2)] {
        assert!(matches!(
            verify_integer_lll(&source, &invalid, &witness, &mut Unbounded),
            Err(IntegerLllError::Invalid(
                "delta must lie strictly between 1/4 and 1"
            ))
        ));
    }
    let oversized = BigRational::new(
        BigInt::one(),
        BigInt::one() << MAX_INTEGER_LLL_COEFFICIENT_BITS as u32,
    );
    assert!(matches!(
        verify_integer_lll(&source, &oversized, &witness, &mut Unbounded),
        Err(IntegerLllError::Reference(SparseRrefError::Limit(_)))
    ));
}

#[test]
fn coefficients_beyond_machine_integer_and_float_precision_remain_exact() {
    let mut source = matrix(1, 2, &[1, 1]);
    source.entries[0].value = BigRational::from_integer((BigInt::one() << 200u32) + BigInt::one());
    let witness = unchanged(&source);
    let receipt = verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.rank(), 1);
    let mut changed = source;
    changed.entries[0].value = BigRational::from_integer(BigInt::one() << 200u32);
    assert!(verify_integer_lll(&changed, &delta(), &witness, &mut Unbounded).is_err());
}

#[test]
fn proper_sublattices_and_rational_inverse_shortcuts_are_rejected() {
    let source = identity(2);
    let mut witness = unchanged(&source);
    witness.basis = matrix(2, 2, &[2, 0, 0, 2]);
    witness.transform = witness.basis.clone();
    witness.transform_inverse = identity(2);
    assert!(verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).is_err());
    for entry in &mut witness.transform_inverse.entries {
        entry.value = ratio(1, 2);
    }
    // B = T A and both inverse products now hold over Q, but T is not
    // invertible over Z and B generates a proper sublattice.
    assert!(verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).is_err());

    let mut omitted = unchanged(&source);
    omitted.basis = matrix(2, 2, &[1, 0, 0, 0]);
    omitted.transform = omitted.basis.clone();
    omitted.transform_inverse = omitted.basis.clone();
    assert!(verify_integer_lll(&source, &delta(), &omitted, &mut Unbounded).is_err());
}

#[test]
fn source_schema_shapes_and_every_witness_component_are_bound() {
    let (source, witness) = fixture();
    for mutation in 0..10 {
        let mut source = source.clone();
        let mut witness = witness.clone();
        match mutation {
            0 => source.entries[0].value = q(2),
            1 => witness.basis.entries[0].value = q(2),
            2 => witness.transform.entries[0].value = q(-3),
            3 => witness.transform_inverse.entries[0].value = q(2),
            4 => witness.schema_version += 1,
            5 => witness.basis.columns += 1,
            6 => witness.transform.rows += 1,
            7 => witness.transform_inverse.columns += 1,
            8 => {
                witness.basis.entries.pop().unwrap();
            }
            _ => source.columns += 1,
        }
        assert!(
            verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn every_matrix_refuses_noncanonical_fractional_and_oversized_admission() {
    let (source, witness) = fixture();
    for target in 0..4 {
        for mutation in 0..9 {
            let mut source = source.clone();
            let mut witness = witness.clone();
            let matrix = match target {
                0 => &mut source,
                1 => &mut witness.basis,
                2 => &mut witness.transform,
                _ => &mut witness.transform_inverse,
            };
            match mutation {
                0 => matrix.entries[0].row = matrix.rows,
                1 => matrix.entries[0].column = matrix.columns,
                2 => matrix.entries.swap(0, 1),
                3 => matrix.entries.insert(1, matrix.entries[0].clone()),
                4 => matrix.entries[0].value = q(0),
                5 => matrix.entries[0].value = ratio(1, 2),
                6 => matrix.rows = MAX_INTEGER_LLL_ROWS + 1,
                7 => matrix.columns = MAX_INTEGER_LLL_COLUMNS + 1,
                _ => {
                    matrix.entries[0].value = BigRational::from_integer(
                        BigInt::one() << MAX_INTEGER_LLL_COEFFICIENT_BITS as u32,
                    );
                }
            }
            assert!(
                verify_integer_lll(&source, &delta(), &witness, &mut Unbounded).is_err(),
                "matrix {target}, mutation {mutation}"
            );
        }
    }
}

#[test]
fn portable_decoder_bounds_nested_values_shapes_entries_and_unknown_fields() {
    let (_, witness) = fixture();
    let value = serde_json::to_value(witness).unwrap();
    let mut unknown = value.clone();
    unknown["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<IntegerLllWitness>(unknown).is_err());
    let mut substituted_delta = value.clone();
    substituted_delta["delta"] = serde_json::to_value(ratio(1, 2)).unwrap();
    assert!(serde_json::from_value::<IntegerLllWitness>(substituted_delta).is_err());
    for field in ["basis", "transform", "transform_inverse"] {
        for (dimension, bound) in [
            ("rows", MAX_INTEGER_LLL_ROWS),
            ("columns", MAX_INTEGER_LLL_COLUMNS),
        ] {
            let mut oversized = value.clone();
            oversized[field][dimension] = serde_json::json!(bound + 1);
            assert!(serde_json::from_value::<IntegerLllWitness>(oversized).is_err());
        }
        let mut fractional = value.clone();
        fractional[field]["entries"][0]["value"] = serde_json::to_value(ratio(1, 2)).unwrap();
        assert!(serde_json::from_value::<IntegerLllWitness>(fractional).is_err());
        let mut noncanonical = value.clone();
        noncanonical[field]["entries"][0]["value"] =
            serde_json::to_value((BigInt::from(2), BigInt::from(2))).unwrap();
        assert!(serde_json::from_value::<IntegerLllWitness>(noncanonical).is_err());
        let mut oversized = value.clone();
        oversized[field]["entries"][0]["value"] = serde_json::to_value(BigRational::from_integer(
            BigInt::one() << MAX_INTEGER_LLL_COEFFICIENT_BITS as u32,
        ))
        .unwrap();
        assert!(serde_json::from_value::<IntegerLllWitness>(oversized).is_err());
        let mut unknown = value.clone();
        unknown[field]["verified"] = serde_json::json!(true);
        assert!(serde_json::from_value::<IntegerLllWitness>(unknown).is_err());
        let mut too_many = value.clone();
        let entry = too_many[field]["entries"][0].clone();
        too_many[field]["entries"] =
            serde_json::Value::Array(vec![
                entry;
                MAX_INTEGER_LLL_ROWS * MAX_INTEGER_LLL_COLUMNS + 1
            ]);
        assert!(serde_json::from_value::<IntegerLllWitness>(too_many).is_err());
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
fn every_safe_point_including_gram_schmidt_and_publication_is_cancellable() {
    let source = matrix(2, 2, &[1, 0, 0, 1]);
    let witness = unchanged(&source);
    let mut observed = SafePoints::default();
    verify_integer_lll(&source, &delta(), &witness, &mut observed).unwrap();
    assert!(observed.seen > 50);
    for point in 1..=observed.seen {
        let mut meter = SafePoints {
            seen: 0,
            cancel_at: Some(point),
        };
        assert!(
            matches!(
                verify_integer_lll(&source, &delta(), &witness, &mut meter),
                Err(IntegerLllError::Reference(SparseRrefError::Meter(
                    MeterError::Cancelled
                )))
            ),
            "safe point {point}"
        );
    }
}

#[test]
fn all_resource_dimensions_are_terminal_and_shared_work_remains_cumulative() {
    let (source, witness) = fixture();
    let allowance = 100_000_000;
    let mut observed = Budget::new(BudgetLimits::uniform(allowance, 0));
    verify_integer_lll(&source, &delta(), &witness, &mut observed).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let used = allowance - observed.remaining(dimension);
        assert!(used > 0);
        for remaining in [0, used - 1] {
            let mut limits = BudgetLimits::uniform(allowance, 0);
            limits.dimensions[dimension.index()] = remaining;
            assert!(matches!(
                verify_integer_lll(&source, &delta(), &witness, &mut Budget::new(limits),),
                Err(IntegerLllError::Reference(SparseRrefError::Meter(
                    MeterError::Budget(_)
                )))
            ));
        }
    }
    let used = allowance - observed.remaining(Dimension::ComputeSteps);
    let mut limits = BudgetLimits::uniform(allowance, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used;
    let mut shared = Budget::new(limits);
    verify_integer_lll(&source, &delta(), &witness, &mut shared).unwrap();
    assert_eq!(shared.remaining(Dimension::ComputeSteps), 0);
    assert!(verify_integer_lll(&source, &delta(), &witness, &mut shared).is_err());
}
