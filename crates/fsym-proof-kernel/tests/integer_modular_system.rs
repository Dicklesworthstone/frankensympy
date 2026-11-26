//! Independent finite-residue oracles and attacks on modular completeness.
//! This test crate does not depend on the matrix generator.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::integer_modular_system::{
    INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION, IntegerModularSystemError, IntegerModularSystemOutcome,
    IntegerModularSystemWitness, verify_integer_modular_system,
};
use fsym_proof_kernel::integer_normal_form::{
    INTEGER_SMITH_SCHEMA_VERSION, IntegerSmithError, IntegerSmithWitness,
    MAX_INTEGER_SMITH_DIMENSION,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};
use std::collections::BTreeSet;

fn matrix(rows: usize, columns: usize, values: &[i64]) -> RationalSparseMatrix {
    assert_eq!(values.len(), rows * columns);
    RationalSparseMatrix {
        rows,
        columns,
        entries: values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(index, value)| SparseRationalEntry {
                row: index / columns,
                column: index % columns,
                value: BigRational::from_integer((*value).into()),
            })
            .collect(),
    }
}

fn identity(size: usize) -> RationalSparseMatrix {
    let mut values = vec![0; size * size];
    for index in 0..size {
        values[index * size + index] = 1;
    }
    matrix(size, size, &values)
}

fn diagonal_witness(source: &RationalSparseMatrix) -> IntegerSmithWitness {
    IntegerSmithWitness {
        schema_version: INTEGER_SMITH_SCHEMA_VERSION,
        diagonal: source.clone(),
        left: identity(source.rows),
        right: identity(source.columns),
        left_inverse: identity(source.rows),
        right_inverse: identity(source.columns),
    }
}

fn cell(matrix: &RationalSparseMatrix, row: usize, column: usize) -> i64 {
    matrix
        .entries
        .iter()
        .find(|entry| (entry.row, entry.column) == (row, column))
        .map_or(0, |entry| entry.value.numer().to_i64().unwrap())
}

fn verify(
    source: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    modulus: i64,
    witness: &IntegerModularSystemWitness,
) -> bool {
    verify_integer_modular_system(source, rhs, &modulus.into(), witness, &mut Unbounded).is_ok()
}

fn fixture() -> (
    RationalSparseMatrix,
    RationalSparseMatrix,
    IntegerModularSystemWitness,
) {
    // D=diag(2,6), U=I, V=[1 -2;0 1]. Modulo12, C=(4,6),
    // Y=(2,1), P=(0,1). The two generators have orders2 and6.
    let source = matrix(2, 2, &[2, 4, 0, 6]);
    let rhs = matrix(2, 1, &[4, 6]);
    let witness = IntegerModularSystemWitness {
        schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
        normal_form: IntegerSmithWitness {
            schema_version: INTEGER_SMITH_SCHEMA_VERSION,
            diagonal: matrix(2, 2, &[2, 0, 0, 6]),
            left: identity(2),
            right: matrix(2, 2, &[1, -2, 0, 1]),
            left_inverse: identity(2),
            right_inverse: matrix(2, 2, &[1, 2, 0, 1]),
        },
        outcome: IntegerModularSystemOutcome::Solvable {
            particular: matrix(2, 1, &[0, 1]),
            generators: matrix(2, 2, &[6, 8, 0, 2]),
            periods: vec![2.into(), 6.into()],
        },
    };
    (source, rhs, witness)
}

fn family(witness: &IntegerModularSystemWitness, modulus: i64) -> BTreeSet<Vec<i64>> {
    let IntegerModularSystemOutcome::Solvable {
        particular,
        generators,
        periods,
    } = &witness.outcome
    else {
        return BTreeSet::new();
    };
    assert_eq!(particular.columns, 1);
    let count = periods
        .iter()
        .map(|period| period.to_i64().unwrap())
        .product::<i64>();
    let mut result = BTreeSet::new();
    for mut index in 0..count {
        let mut values: Vec<_> = (0..particular.rows)
            .map(|row| cell(particular, row, 0))
            .collect();
        for (column, period) in periods.iter().enumerate() {
            let period = period.to_i64().unwrap();
            let parameter = index % period;
            index /= period;
            for (row, value) in values.iter_mut().enumerate() {
                *value = (*value + cell(generators, row, column) * parameter).rem_euclid(modulus);
            }
        }
        assert!(result.insert(values), "parameterization must be injective");
    }
    assert_eq!(result.len() as i64, count);
    result
}

fn brute_force(
    source: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    modulus: i64,
) -> BTreeSet<Vec<i64>> {
    assert_eq!(rhs.columns, 1);
    let count = modulus.pow(source.columns as u32);
    let mut result = BTreeSet::new();
    for mut index in 0..count {
        let values: Vec<_> = (0..source.columns)
            .map(|_| {
                let value = index % modulus;
                index /= modulus;
                value
            })
            .collect();
        if (0..source.rows).all(|row| {
            let sum: i64 = values
                .iter()
                .enumerate()
                .map(|(column, value)| cell(source, row, column) * value)
                .sum();
            (sum - cell(rhs, row, 0)).rem_euclid(modulus) == 0
        }) {
            result.insert(values);
        }
    }
    result
}

#[test]
fn composite_modulus_family_is_complete_injective_and_bound_to_sources() {
    let (source, rhs, witness) = fixture();
    let receipt =
        verify_integer_modular_system(&source, &rhs, &12.into(), &witness, &mut Unbounded).unwrap();
    assert!(receipt.is_consistent());
    assert!(!receipt.is_unique());
    assert_eq!(
        (receipt.rows(), receipt.columns(), receipt.rhs_columns()),
        (2, 2, 1)
    );
    assert_eq!(receipt.modulus(), &BigInt::from(12));
    assert_eq!(
        receipt.periods(),
        Some([BigInt::from(2), BigInt::from(6)].as_slice())
    );
    assert_eq!(receipt.contradiction(), None);
    assert!(receipt.particular_digest().is_some());
    assert!(receipt.generators_digest().is_some());
    let values = family(&witness, 12);
    assert_eq!(values.len(), 12);
    assert_eq!(values, brute_force(&source, &rhs, 12));
    assert_eq!(
        receipt,
        verify_integer_modular_system(&source, &rhs, &12.into(), &witness, &mut Unbounded).unwrap()
    );
    // Congruent RHS sources prove the same residue family but receive distinct
    // exact-source bindings; a previously accepted digest is never reused.
    let congruent_rhs = matrix(2, 1, &[16, -6]);
    let other = verify_integer_modular_system(
        &source,
        &congruent_rhs,
        &12.into(),
        &witness,
        &mut Unbounded,
    )
    .unwrap();
    assert_ne!(receipt.rhs_digest(), other.rhs_digest());
    assert_eq!(receipt.source_digest(), other.source_digest());
}

#[test]
fn exhaustive_signed_scalar_congruences_use_enumerated_solution_witnesses() {
    for modulus in 2i64..=8 {
        for coefficient in -5i64..=5 {
            for right in 0..modulus {
                let source = matrix(1, 1, &[coefficient]);
                let rhs = matrix(1, 1, &[right]);
                let solutions = brute_force(&source, &rhs, modulus);
                let mut normal_form = diagonal_witness(&matrix(1, 1, &[coefficient.abs()]));
                if coefficient < 0 {
                    normal_form.left = matrix(1, 1, &[-1]);
                    normal_form.left_inverse = matrix(1, 1, &[-1]);
                }
                let outcome = if let Some(first) = solutions.first() {
                    let count = solutions.len() as i64;
                    IntegerModularSystemOutcome::Solvable {
                        particular: matrix(1, 1, &[first[0]]),
                        generators: if count == 1 {
                            matrix(1, 0, &[])
                        } else {
                            matrix(1, 1, &[modulus / count])
                        },
                        periods: if count == 1 {
                            vec![]
                        } else {
                            vec![count.into()]
                        },
                    }
                } else {
                    IntegerModularSystemOutcome::Inconsistent {
                        row: 0,
                        rhs_column: 0,
                    }
                };
                let witness = IntegerModularSystemWitness {
                    schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
                    normal_form,
                    outcome,
                };
                let receipt = verify_integer_modular_system(
                    &source,
                    &rhs,
                    &modulus.into(),
                    &witness,
                    &mut Unbounded,
                )
                .unwrap();
                assert_eq!(receipt.is_consistent(), !solutions.is_empty());
                assert_eq!(receipt.is_unique(), solutions.len() == 1);
                assert_eq!(family(&witness, modulus), solutions);
            }
        }
    }
}

#[test]
fn a_nonzero_integer_diagonal_can_be_zero_modulo_the_modulus() {
    let source = matrix(1, 1, &[6]);
    let rhs = matrix(1, 1, &[0]);
    let witness = IntegerModularSystemWitness {
        schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
        normal_form: diagonal_witness(&source),
        outcome: IntegerModularSystemOutcome::Solvable {
            particular: matrix(1, 1, &[0]),
            generators: identity(1),
            periods: vec![6.into()],
        },
    };
    assert!(verify(&source, &rhs, 6, &witness));
    assert_eq!(family(&witness, 6).len(), 6);
    assert_eq!(family(&witness, 6), brute_force(&source, &rhs, 6));
}

#[test]
fn rectangular_systems_preserve_free_directions_and_transformed_contradictions() {
    let source = matrix(1, 2, &[2, 0]);
    let rhs = matrix(1, 1, &[4]);
    let witness = IntegerModularSystemWitness {
        schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
        normal_form: diagonal_witness(&source),
        outcome: IntegerModularSystemOutcome::Solvable {
            particular: matrix(2, 1, &[2, 0]),
            generators: matrix(2, 2, &[3, 0, 0, 1]),
            periods: vec![2.into(), 6.into()],
        },
    };
    assert!(verify(&source, &rhs, 6, &witness));
    assert_eq!(family(&witness, 6), brute_force(&source, &rhs, 6));

    // Subtracting twice row0 from row1 gives the zero Smith row.
    let source = matrix(2, 1, &[2, 4]);
    let mut witness = IntegerModularSystemWitness {
        schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
        normal_form: IntegerSmithWitness {
            schema_version: INTEGER_SMITH_SCHEMA_VERSION,
            diagonal: matrix(2, 1, &[2, 0]),
            left: matrix(2, 2, &[1, 0, -2, 1]),
            right: identity(1),
            left_inverse: matrix(2, 2, &[1, 0, 2, 1]),
            right_inverse: identity(1),
        },
        outcome: IntegerModularSystemOutcome::Inconsistent {
            row: 1,
            rhs_column: 0,
        },
    };
    let bad_rhs = matrix(2, 1, &[2, 5]);
    assert!(verify(&source, &bad_rhs, 6, &witness));
    assert!(brute_force(&source, &bad_rhs, 6).is_empty());
    let good_rhs = matrix(2, 1, &[2, 10]);
    assert!(!verify(&source, &good_rhs, 6, &witness));
    witness.outcome = IntegerModularSystemOutcome::Solvable {
        particular: matrix(1, 1, &[1]),
        generators: matrix(1, 1, &[3]),
        periods: vec![2.into()],
    };
    assert!(verify(&source, &good_rhs, 6, &witness));
    assert_eq!(family(&witness, 6), brute_force(&source, &good_rhs, 6));
}

#[test]
fn every_rhs_column_has_independent_parameters_and_consistency_obligations() {
    let (source, _, mut witness) = fixture();
    let rhs = matrix(2, 2, &[4, 10, 6, 0]);
    if let IntegerModularSystemOutcome::Solvable { particular, .. } = &mut witness.outcome {
        *particular = matrix(2, 2, &[0, 5, 1, 0]);
    }
    let receipt =
        verify_integer_modular_system(&source, &rhs, &12.into(), &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.rhs_columns(), 2);
    assert!(!receipt.is_unique());
    let left = brute_force(&source, &matrix(2, 1, &[4, 6]), 12);
    let right = brute_force(&source, &matrix(2, 1, &[10, 0]), 12);
    assert_eq!(left.len() * right.len(), 144);
    let inconsistent = matrix(2, 2, &[4, 10, 6, 1]);
    assert!(!verify(&source, &inconsistent, 12, &witness));
    witness.outcome = IntegerModularSystemOutcome::Inconsistent {
        row: 1,
        rhs_column: 1,
    };
    assert!(verify(&source, &inconsistent, 12, &witness));
    witness.outcome = IntegerModularSystemOutcome::Inconsistent {
        row: 0,
        rhs_column: 0,
    };
    assert!(!verify(&source, &inconsistent, 12, &witness));
}

#[test]
fn omitted_scaled_extra_or_wrong_period_generators_are_rejected() {
    let (source, rhs, witness) = fixture();
    for (generators, periods) in [
        (matrix(2, 1, &[8, 2]), vec![6.into()]),
        (matrix(2, 2, &[0, 8, 0, 2]), vec![2.into(), 6.into()]),
        (matrix(2, 2, &[6, 8, 0, 2]), vec![2.into(), 3.into()]),
        (matrix(2, 2, &[6, 8, 0, 2]), vec![4.into(), 6.into()]),
        (matrix(2, 2, &[6, 8, 0, 2]), vec![1.into(), 6.into()]),
        (matrix(2, 2, &[8, 6, 2, 0]), vec![6.into(), 2.into()]),
        (
            matrix(2, 3, &[6, 8, 6, 0, 2, 0]),
            vec![2.into(), 6.into(), 2.into()],
        ),
        (matrix(2, 2, &[6, -4, 0, 2]), vec![2.into(), 6.into()]),
    ] {
        let mut changed = witness.clone();
        if let IntegerModularSystemOutcome::Solvable {
            generators: target,
            periods: target_periods,
            ..
        } = &mut changed.outcome
        {
            *target = generators;
            *target_periods = periods;
        }
        assert!(!verify(&source, &rhs, 12, &changed));
    }
    // Scaling the order-two generator to zero preserves every residual but
    // removes half of the original solutions. Residual-only checking fails.
}

#[test]
fn residual_valid_noncanonical_particular_and_forged_sources_are_rejected() {
    let (source, rhs, witness) = fixture();
    for particular in [
        matrix(2, 1, &[6, 1]),  // A valid solution, but not canonical V^-1P.
        matrix(2, 1, &[0, 0]),  // Incorrect original residual.
        matrix(2, 1, &[12, 1]), // Noncanonical modular residue.
        matrix(2, 1, &[0, -11]),
        matrix(1, 2, &[0, 1]),
    ] {
        let mut changed = witness.clone();
        if let IntegerModularSystemOutcome::Solvable {
            particular: target, ..
        } = &mut changed.outcome
        {
            *target = particular;
        }
        assert!(!verify(&source, &rhs, 12, &changed));
    }
    assert!(!verify(&matrix(2, 2, &[14, 4, 0, 6]), &rhs, 12, &witness));
    assert!(!verify(&source, &matrix(2, 1, &[5, 6]), 12, &witness));
    assert!(!verify(&source, &rhs, 6, &witness));
    let mut changed = witness.clone();
    changed.normal_form.right_inverse = identity(2);
    assert!(!verify(&source, &rhs, 12, &changed));
    changed = witness;
    changed.schema_version += 1;
    assert!(!verify(&source, &rhs, 12, &changed));
}

#[test]
fn empty_dimensions_and_zero_rhs_columns_retain_precise_uniqueness() {
    for (rows, columns, rhs_columns) in [(0, 0, 0), (0, 0, 2), (0, 2, 1), (2, 0, 1), (2, 2, 0)] {
        let source = matrix(rows, columns, &vec![0; rows * columns]);
        let rhs = matrix(rows, rhs_columns, &vec![0; rows * rhs_columns]);
        let witness = IntegerModularSystemWitness {
            schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
            normal_form: diagonal_witness(&source),
            outcome: IntegerModularSystemOutcome::Solvable {
                particular: matrix(columns, rhs_columns, &vec![0; columns * rhs_columns]),
                generators: identity(columns),
                periods: vec![BigInt::from(5); columns],
            },
        };
        let receipt =
            verify_integer_modular_system(&source, &rhs, &5.into(), &witness, &mut Unbounded)
                .unwrap();
        assert!(receipt.is_consistent());
        assert_eq!(receipt.is_unique(), columns == 0 || rhs_columns == 0);
        assert_eq!(receipt.periods().unwrap().len(), columns);
    }
    let source = matrix(2, 0, &[]);
    let rhs = matrix(2, 1, &[0, 1]);
    let witness = IntegerModularSystemWitness {
        schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
        normal_form: diagonal_witness(&source),
        outcome: IntegerModularSystemOutcome::Inconsistent {
            row: 1,
            rhs_column: 0,
        },
    };
    assert!(verify(&source, &rhs, 5, &witness));
}

#[test]
fn invalid_originals_moduli_and_shapes_cannot_hide_behind_inconsistency() {
    let source = matrix(2, 1, &[0, 0]);
    let rhs = matrix(2, 1, &[1, 1]);
    let witness = IntegerModularSystemWitness {
        schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
        normal_form: diagonal_witness(&source),
        outcome: IntegerModularSystemOutcome::Inconsistent {
            row: 0,
            rhs_column: 0,
        },
    };
    for modulus in [-6, 0, 1] {
        assert!(!verify(&source, &rhs, modulus, &witness));
    }
    let mut bad = rhs.clone();
    bad.entries[1].value = BigRational::new(1.into(), 2.into());
    assert!(!verify(&source, &bad, 6, &witness));
    bad = rhs.clone();
    bad.entries[1].value = BigRational::from_integer(BigInt::one() << 4096u32);
    assert!(!verify(&source, &bad, 6, &witness));
    bad = rhs.clone();
    bad.entries[1].row = 0;
    assert!(!verify(&source, &bad, 6, &witness));
    bad = rhs.clone();
    bad.columns = MAX_INTEGER_SMITH_DIMENSION + 1;
    assert!(!verify(&source, &bad, 6, &witness));
    assert!(
        verify_integer_modular_system(
            &source,
            &rhs,
            &(BigInt::one() << 4096u32),
            &witness,
            &mut Unbounded,
        )
        .is_err()
    );
    let mut changed = witness;
    changed.outcome = IntegerModularSystemOutcome::Inconsistent {
        row: 2,
        rhs_column: 0,
    };
    assert!(!verify(&source, &rhs, 6, &changed));
}

fn interruption(error: &IntegerModularSystemError) -> Option<MeterError> {
    match error {
        IntegerModularSystemError::Reference(SparseRrefError::Meter(error))
        | IntegerModularSystemError::NormalForm(IntegerSmithError::Reference(
            SparseRrefError::Meter(error),
        )) => Some(*error),
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
fn every_checkpoint_can_cancel_both_classifications_without_a_receipt() {
    let source = matrix(1, 1, &[2]);
    for consistent in [true, false] {
        let rhs = matrix(1, 1, &[if consistent { 2 } else { 1 }]);
        let witness = IntegerModularSystemWitness {
            schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
            normal_form: diagonal_witness(&source),
            outcome: if consistent {
                IntegerModularSystemOutcome::Solvable {
                    particular: matrix(1, 1, &[1]),
                    generators: matrix(1, 1, &[3]),
                    periods: vec![2.into()],
                }
            } else {
                IntegerModularSystemOutcome::Inconsistent {
                    row: 0,
                    rhs_column: 0,
                }
            },
        };
        let mut observation = SafePoints::default();
        verify_integer_modular_system(&source, &rhs, &6.into(), &witness, &mut observation)
            .unwrap();
        assert!(observation.seen > 20);
        for point in 1..=observation.seen {
            let mut meter = SafePoints {
                seen: 0,
                cancel_at: Some(point),
            };
            let result =
                verify_integer_modular_system(&source, &rhs, &6.into(), &witness, &mut meter);
            assert!(
                matches!(result, Err(ref error) if interruption(error) == Some(MeterError::Cancelled)),
                "consistent={consistent}, cancellation={point}"
            );
            assert_eq!(meter.seen, point);
        }
    }
}

#[test]
fn exact_fit_one_short_and_protected_allowances_are_cumulative() {
    let (source, rhs, witness) = fixture();
    let initial = 100_000_000;
    let mut observation = Budget::new(BudgetLimits::uniform(initial, 59));
    verify_integer_modular_system(&source, &rhs, &12.into(), &witness, &mut observation).unwrap();
    assert_eq!(observation.snapshot().verifier_pool, 59);
    let dimensions = [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ];
    let mut exact = BudgetLimits::uniform(initial, 59);
    for dimension in dimensions {
        let used = initial - observation.remaining(dimension);
        assert!(used > 0);
        exact.dimensions[dimension.index()] = used;
    }
    let mut fit = Budget::new(exact);
    verify_integer_modular_system(&source, &rhs, &12.into(), &witness, &mut fit).unwrap();
    for dimension in dimensions {
        assert_eq!(fit.remaining(dimension), 0);
        let mut short = exact;
        short.dimensions[dimension.index()] -= 1;
        let mut meter = Budget::new(short);
        let result = verify_integer_modular_system(&source, &rhs, &12.into(), &witness, &mut meter);
        assert!(
            matches!(result, Err(ref error) if matches!(interruption(error), Some(MeterError::Budget(_))))
        );
        assert_eq!(meter.snapshot().verifier_pool, 59);
    }
    let result = verify_integer_modular_system(&source, &rhs, &12.into(), &witness, &mut fit);
    assert!(
        matches!(result, Err(ref error) if matches!(interruption(error), Some(MeterError::Budget(_))))
    );

    let mut protected =
        BudgetLimits::uniform(initial, exact.dimensions[Dimension::ComputeSteps.index()]);
    protected.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut ledger = Budget::new(protected);
    let lease = ledger.verifier_lease().unwrap();
    verify_integer_modular_system(
        &source,
        &rhs,
        &12.into(),
        &witness,
        &mut ledger.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(ledger.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(ledger.snapshot().verifier_pool, 0);
}
