//! Complete integer-lattice fixtures independent of matrix generators.
//! In particular, a scaled rational nullspace basis is an invalid integer
//! completeness witness even when every reported vector has zero residual.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::integer_linear_system::{
    INTEGER_LINEAR_SCHEMA_VERSION, IntegerLinearError, IntegerLinearOutcome, IntegerLinearWitness,
    verify_integer_linear_system,
};
use fsym_proof_kernel::integer_normal_form::{
    INTEGER_SMITH_SCHEMA_VERSION, IntegerSmithError, IntegerSmithWitness,
    MAX_INTEGER_SMITH_COEFFICIENT_BITS, MAX_INTEGER_SMITH_DIMENSION,
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

fn fixture() -> (
    RationalSparseMatrix,
    RationalSparseMatrix,
    IntegerLinearWitness,
) {
    // 2x + 4y = 6 has every integer solution (3-2t,t).
    let source = matrix(1, 2, &[(0, 0, 2), (0, 1, 4)]);
    let rhs = matrix(1, 1, &[(0, 0, 6)]);
    let witness = IntegerLinearWitness {
        schema_version: INTEGER_LINEAR_SCHEMA_VERSION,
        normal_form: IntegerSmithWitness {
            schema_version: INTEGER_SMITH_SCHEMA_VERSION,
            diagonal: matrix(1, 2, &[(0, 0, 2)]),
            left: identity(1),
            right: matrix(2, 2, &[(0, 0, 1), (0, 1, -2), (1, 1, 1)]),
            left_inverse: identity(1),
            right_inverse: matrix(2, 2, &[(0, 0, 1), (0, 1, 2), (1, 1, 1)]),
        },
        outcome: IntegerLinearOutcome::Solvable {
            particular: matrix(2, 1, &[(0, 0, 3)]),
            kernel: matrix(2, 1, &[(0, 0, -2), (1, 0, 1)]),
        },
    };
    (source, rhs, witness)
}

#[test]
fn complete_primitive_integer_family_and_portable_replay() {
    let (source, rhs, witness) = fixture();
    let receipt = verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).unwrap();
    assert!(receipt.is_consistent());
    assert!(!receipt.is_unique());
    assert_eq!(
        (receipt.rows(), receipt.columns(), receipt.rhs_columns()),
        (1, 2, 1)
    );
    assert_eq!((receipt.rank(), receipt.nullity()), (1, 1));
    assert_eq!(
        receipt.normal_form().invariant_factors(),
        &[BigInt::from(2)]
    );
    assert_eq!(receipt.contradiction(), None);
    assert!(receipt.particular_digest().is_some());
    assert!(receipt.kernel_digest().is_some());
    let decoded: IntegerLinearWitness =
        serde_json::from_str(&serde_json::to_string(&witness).unwrap()).unwrap();
    assert_eq!(decoded, witness);
    assert_eq!(
        receipt,
        verify_integer_linear_system(&source, &rhs, &decoded, &mut Unbounded).unwrap()
    );
}

#[test]
fn rational_solvability_does_not_imply_integer_solvability() {
    let (source, _, mut witness) = fixture();
    let rhs = matrix(1, 1, &[(0, 0, 1)]);
    witness.outcome = IntegerLinearOutcome::Inconsistent {
        row: 0,
        rhs_column: 0,
    };
    let receipt = verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).unwrap();
    assert!(!receipt.is_consistent());
    assert!(!receipt.is_unique());
    assert_eq!(receipt.contradiction(), Some((0, 0)));
    assert_eq!(receipt.particular_digest(), None);
    assert_eq!(receipt.kernel_digest(), None);
    // A rational solution exists (x=1/2,y=0), but gcd(2,4) does not divide 1.
    assert_eq!(q(2) * BigRational::new(1.into(), 2.into()), q(1));
    let (_, integral_rhs, _) = fixture();
    assert!(
        verify_integer_linear_system(&source, &integral_rhs, &witness, &mut Unbounded).is_err()
    );
}

#[test]
fn dependent_rows_require_transformed_zero_rhs_rows() {
    let (_, _, mut witness) = fixture();
    let source = matrix(2, 2, &[(0, 0, 2), (0, 1, 4), (1, 0, 4), (1, 1, 8)]);
    witness.normal_form.diagonal.rows = 2;
    witness.normal_form.left = matrix(2, 2, &[(0, 0, 1), (1, 0, -2), (1, 1, 1)]);
    witness.normal_form.left_inverse = matrix(2, 2, &[(0, 0, 1), (1, 0, 2), (1, 1, 1)]);
    let rhs = matrix(2, 1, &[(0, 0, 6), (1, 0, 12)]);
    assert!(
        verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded)
            .unwrap()
            .is_consistent()
    );
    let contradicted = matrix(2, 1, &[(0, 0, 6), (1, 0, 13)]);
    assert!(
        verify_integer_linear_system(&source, &contradicted, &witness, &mut Unbounded).is_err()
    );
    witness.outcome = IntegerLinearOutcome::Inconsistent {
        row: 1,
        rhs_column: 0,
    };
    let receipt =
        verify_integer_linear_system(&source, &contradicted, &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.contradiction(), Some((1, 0)));
    assert!(verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).is_err());
    witness.outcome = IntegerLinearOutcome::Inconsistent {
        row: 0,
        rhs_column: 0,
    };
    assert!(
        verify_integer_linear_system(&source, &contradicted, &witness, &mut Unbounded).is_err()
    );
}

#[test]
fn each_rhs_column_has_its_own_integer_parameter_and_divisibility_conditions() {
    let (source, _, mut witness) = fixture();
    let rhs = matrix(1, 2, &[(0, 0, 6), (0, 1, 10)]);
    if let IntegerLinearOutcome::Solvable { particular, .. } = &mut witness.outcome {
        *particular = matrix(2, 2, &[(0, 0, 3), (0, 1, 5)]);
    }
    let receipt = verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.rhs_columns(), 2);
    assert_eq!(receipt.nullity(), 1);
    let bad_rhs = matrix(1, 2, &[(0, 0, 6), (0, 1, 7)]);
    assert!(verify_integer_linear_system(&source, &bad_rhs, &witness, &mut Unbounded).is_err());
    witness.outcome = IntegerLinearOutcome::Inconsistent {
        row: 0,
        rhs_column: 1,
    };
    assert_eq!(
        verify_integer_linear_system(&source, &bad_rhs, &witness, &mut Unbounded)
            .unwrap()
            .contradiction(),
        Some((0, 1))
    );
}

#[test]
fn zero_residual_and_full_rational_rank_do_not_certify_a_complete_integer_kernel() {
    let (source, rhs, witness) = fixture();
    // (-4,2) is nonzero and in the nullspace, but spans only even y. It
    // misses the valid integer solution (1,1) of the original equation.
    assert_eq!(2 * (-4) + 4 * 2, 0);
    assert_eq!(2 + 4, 6);
    for bad_kernel in [
        matrix(2, 1, &[(0, 0, -4), (1, 0, 2)]),
        matrix(2, 1, &[]),
        matrix(2, 0, &[]),
        matrix(2, 2, &[(0, 0, -2), (0, 1, -2), (1, 0, 1), (1, 1, 1)]),
    ] {
        let mut changed = witness.clone();
        if let IntegerLinearOutcome::Solvable { kernel, .. } = &mut changed.outcome {
            *kernel = bad_kernel;
        }
        assert!(verify_integer_linear_system(&source, &rhs, &changed, &mut Unbounded).is_err());
    }
}

#[test]
fn full_rank_system_has_one_checked_integer_solution() {
    // D=UAV=diag(1,18) for A=[4 6;3 9]; B=(10,12), C=(-2,18),
    // Y=(-2,1), VY=(1,1). Both transforms and inverses are hand-written.
    let source = matrix(2, 2, &[(0, 0, 4), (0, 1, 6), (1, 0, 3), (1, 1, 9)]);
    let rhs = matrix(2, 1, &[(0, 0, 10), (1, 0, 12)]);
    let witness = IntegerLinearWitness {
        schema_version: INTEGER_LINEAR_SCHEMA_VERSION,
        normal_form: IntegerSmithWitness {
            schema_version: INTEGER_SMITH_SCHEMA_VERSION,
            diagonal: matrix(2, 2, &[(0, 0, 1), (1, 1, 18)]),
            left: matrix(2, 2, &[(0, 0, 1), (0, 1, -1), (1, 0, -3), (1, 1, 4)]),
            right: matrix(2, 2, &[(0, 0, 1), (0, 1, 3), (1, 1, 1)]),
            left_inverse: matrix(2, 2, &[(0, 0, 4), (0, 1, 1), (1, 0, 3), (1, 1, 1)]),
            right_inverse: matrix(2, 2, &[(0, 0, 1), (0, 1, -3), (1, 1, 1)]),
        },
        outcome: IntegerLinearOutcome::Solvable {
            particular: matrix(2, 1, &[(0, 0, 1), (1, 0, 1)]),
            kernel: matrix(2, 0, &[]),
        },
    };
    let receipt = verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).unwrap();
    assert!(receipt.is_unique());
    assert_eq!((receipt.rank(), receipt.nullity()), (2, 0));
    let mut bad = witness.clone();
    if let IntegerLinearOutcome::Solvable { particular, .. } = &mut bad.outcome {
        particular.entries[0].value = q(2);
    }
    assert!(verify_integer_linear_system(&source, &rhs, &bad, &mut Unbounded).is_err());
    let mut altered_rhs = rhs;
    altered_rhs.entries[0].value = q(11);
    assert!(verify_integer_linear_system(&source, &altered_rhs, &witness, &mut Unbounded).is_err());
}

#[test]
fn canonical_particular_and_binding_digests_retain_all_original_rhs_data() {
    let (source, rhs, witness) = fixture();
    let original = verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).unwrap();
    let mut changed = witness.clone();
    if let IntegerLinearOutcome::Solvable { particular, .. } = &mut changed.outcome {
        *particular = matrix(2, 1, &[(0, 0, 1), (1, 0, 1)]);
    }
    // This is a valid particular point, but not the declared canonical
    // representative with all free Smith coordinates equal to zero.
    assert!(verify_integer_linear_system(&source, &rhs, &changed, &mut Unbounded).is_err());
    let rhs = matrix(1, 1, &[(0, 0, -6)]);
    if let IntegerLinearOutcome::Solvable { particular, .. } = &mut changed.outcome {
        *particular = matrix(2, 1, &[(0, 0, -3)]);
    }
    let negative = verify_integer_linear_system(&source, &rhs, &changed, &mut Unbounded).unwrap();
    assert_eq!(original.source_digest(), negative.source_digest());
    assert_eq!(original.kernel_digest(), negative.kernel_digest());
    assert_ne!(original.rhs_digest(), negative.rhs_digest());
    assert_ne!(original.particular_digest(), negative.particular_digest());
}

#[test]
fn zero_area_shapes_and_zero_rhs_columns_keep_their_exact_solution_semantics() {
    for (rows, columns, rhs_columns) in [(0, 0, 0), (0, 3, 2), (4, 0, 2), (2, 3, 0), (2, 3, 2)] {
        let source = matrix(rows, columns, &[]);
        let rhs = matrix(rows, rhs_columns, &[]);
        let witness = IntegerLinearWitness {
            schema_version: INTEGER_LINEAR_SCHEMA_VERSION,
            normal_form: unchanged(&source),
            outcome: IntegerLinearOutcome::Solvable {
                particular: matrix(columns, rhs_columns, &[]),
                kernel: identity(columns),
            },
        };
        let receipt =
            verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).unwrap();
        assert!(receipt.is_consistent());
        assert_eq!(receipt.rank(), 0);
        assert_eq!(receipt.nullity(), columns);
        assert_eq!(receipt.is_unique(), columns == 0 || rhs_columns == 0);
    }
    let source = matrix(2, 0, &[]);
    let rhs = matrix(2, 1, &[(1, 0, 1)]);
    let witness = IntegerLinearWitness {
        schema_version: INTEGER_LINEAR_SCHEMA_VERSION,
        normal_form: unchanged(&source),
        outcome: IntegerLinearOutcome::Inconsistent {
            row: 1,
            rhs_column: 0,
        },
    };
    assert!(
        !verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded)
            .unwrap()
            .is_consistent()
    );
}

#[test]
fn forged_schema_source_normal_form_and_contradiction_indices_fail_closed() {
    let (source, rhs, witness) = fixture();
    for mutation in 0..7 {
        let mut source = source.clone();
        let mut rhs = rhs.clone();
        let mut witness = witness.clone();
        match mutation {
            0 => witness.schema_version += 1,
            1 => witness.normal_form.schema_version += 1,
            2 => witness.normal_form.left_inverse.entries.clear(),
            3 => source.entries[1].value = q(5),
            4 => rhs.rows += 1,
            5 => {
                witness.outcome = IntegerLinearOutcome::Inconsistent {
                    row: usize::MAX,
                    rhs_column: 0,
                }
            }
            _ => {
                witness.outcome = IntegerLinearOutcome::Inconsistent {
                    row: 0,
                    rhs_column: usize::MAX,
                }
            }
        }
        assert!(
            verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn noninteger_or_oversized_data_cannot_be_hidden_by_a_contradiction() {
    let source = matrix(2, 1, &[]);
    let mut rhs = matrix(2, 1, &[(0, 0, 1), (1, 0, 1)]);
    rhs.entries[1].value = BigRational::new(1.into(), 2.into());
    let witness = IntegerLinearWitness {
        schema_version: INTEGER_LINEAR_SCHEMA_VERSION,
        normal_form: unchanged(&source),
        outcome: IntegerLinearOutcome::Inconsistent {
            row: 0,
            rhs_column: 0,
        },
    };
    assert!(verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).is_err());
    rhs.entries[1].value =
        BigRational::from_integer(BigInt::one() << MAX_INTEGER_SMITH_COEFFICIENT_BITS as u32);
    assert!(verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).is_err());
    rhs.columns = MAX_INTEGER_SMITH_DIMENSION + 1;
    assert!(verify_integer_linear_system(&source, &rhs, &witness, &mut Unbounded).is_err());
}

#[test]
fn portable_witness_decoding_bounds_outputs_and_rejects_forged_flags() {
    let (_, _, witness) = fixture();
    let base = serde_json::to_value(&witness).unwrap();
    let mut fake = base.clone();
    fake["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<IntegerLinearWitness>(fake).is_err());
    let mut unknown = base.clone();
    unknown["outcome"] = serde_json::json!({"heuristic": {}});
    assert!(serde_json::from_value::<IntegerLinearWitness>(unknown).is_err());
    for field in ["particular", "kernel"] {
        let mut noninteger = base.clone();
        noninteger["outcome"]["solvable"][field]["entries"][0]["value"] =
            serde_json::to_value(BigRational::new(1.into(), 2.into())).unwrap();
        assert!(serde_json::from_value::<IntegerLinearWitness>(noninteger).is_err());
        let mut oversize = base.clone();
        oversize["outcome"]["solvable"][field]["columns"] =
            serde_json::json!(MAX_INTEGER_SMITH_DIMENSION + 1);
        assert!(serde_json::from_value::<IntegerLinearWitness>(oversize).is_err());
        let mut excessive = base.clone();
        let entry = excessive["outcome"]["solvable"][field]["entries"][0].clone();
        excessive["outcome"]["solvable"][field]["entries"] = serde_json::Value::Array(vec![
            entry;
            MAX_INTEGER_SMITH_DIMENSION * MAX_INTEGER_SMITH_DIMENSION
                + 1
        ]);
        assert!(serde_json::from_value::<IntegerLinearWitness>(excessive).is_err());
    }
}

fn interruption(error: &IntegerLinearError) -> Option<MeterError> {
    match error {
        IntegerLinearError::Reference(SparseRrefError::Meter(error))
        | IntegerLinearError::NormalForm(IntegerSmithError::Reference(SparseRrefError::Meter(
            error,
        ))) => Some(*error),
        _ => None,
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
fn every_phase_cancels_without_publishing_a_solution_or_inconsistency() {
    let source = matrix(1, 1, &[(0, 0, 2)]);
    for consistent in [true, false] {
        let rhs = matrix(1, 1, &[(0, 0, if consistent { 6 } else { 1 })]);
        let witness = IntegerLinearWitness {
            schema_version: INTEGER_LINEAR_SCHEMA_VERSION,
            normal_form: unchanged(&source),
            outcome: if consistent {
                IntegerLinearOutcome::Solvable {
                    particular: matrix(1, 1, &[(0, 0, 3)]),
                    kernel: matrix(1, 0, &[]),
                }
            } else {
                IntegerLinearOutcome::Inconsistent {
                    row: 0,
                    rhs_column: 0,
                }
            },
        };
        let mut observation = SafePoints::default();
        verify_integer_linear_system(&source, &rhs, &witness, &mut observation).unwrap();
        assert!(observation.seen > 20);
        for point in 1..=observation.seen {
            let mut meter = SafePoints {
                seen: 0,
                cancel_at: Some(point),
            };
            let result = verify_integer_linear_system(&source, &rhs, &witness, &mut meter);
            assert!(
                matches!(result, Err(ref error) if interruption(error) == Some(MeterError::Cancelled)),
                "consistent={consistent}, cancellation={point}"
            );
        }
    }
}

#[test]
fn resource_exhaustion_is_not_an_integer_inconsistency_verdict() {
    let (source, rhs, witness) = fixture();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(100_000_000, 0);
        limits.dimensions[dimension.index()] = 0;
        let result =
            verify_integer_linear_system(&source, &rhs, &witness, &mut Budget::new(limits));
        assert!(
            matches!(result, Err(ref error) if matches!(interruption(error), Some(MeterError::Budget(_))))
        );
    }
}

#[test]
fn all_proof_phases_and_repeated_calls_share_the_callers_budget() {
    let (source, rhs, witness) = fixture();
    let initial = 100_000_000;
    let mut observation = Budget::new(BudgetLimits::uniform(initial, 0));
    verify_integer_linear_system(&source, &rhs, &witness, &mut observation).unwrap();
    let used = initial - observation.remaining(Dimension::ComputeSteps);
    assert!(used > 0);
    let mut limits = BudgetLimits::uniform(initial, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used;
    let mut bounded = Budget::new(limits);
    verify_integer_linear_system(&source, &rhs, &witness, &mut bounded).unwrap();
    assert_eq!(bounded.remaining(Dimension::ComputeSteps), 0);
    let result = verify_integer_linear_system(&source, &rhs, &witness, &mut bounded);
    assert!(
        matches!(result, Err(ref error) if matches!(interruption(error), Some(MeterError::Budget(_))))
    );
}
