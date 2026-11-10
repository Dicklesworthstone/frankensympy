//! Affine certificates written independently of sparse elimination generation.
//! These examples distinguish a complete, canonically parameterized family
//! from one particular solution, a deficient kernel, and a forged contradiction.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::linear_system::LinearRowOperation as Row;
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION, RationalSparseMatrix,
    SPARSE_AFFINE_SCHEMA_VERSION, SPARSE_RREF_SCHEMA_VERSION, SparseAffineOutcome,
    SparseAffineWitness, SparseRationalEntry, SparseRrefError, SparseRrefWitness,
    verify_sparse_affine,
};

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

fn reduction(
    reduced: RationalSparseMatrix,
    pivots: Vec<usize>,
    operations: Vec<Row>,
) -> SparseRrefWitness {
    SparseRrefWitness {
        schema_version: SPARSE_RREF_SCHEMA_VERSION,
        operations,
        reduced,
        pivots,
    }
}

fn fixture() -> (RationalSparseMatrix, RationalSparseMatrix, SparseAffineWitness) {
    // A = [0 2 4 6; 1 1 1 2; 0 4 8 12], B = [-4 8; -1 7; -8 16].
    // P = [1 3; -2 4; 0 0; 0 0] and
    // N = [1 1; -2 -3; 1 0; 0 1] give all X = P + N*T.
    let a = matrix(
        3,
        4,
        &[
            (0, 1, 2), (0, 2, 4), (0, 3, 6),
            (1, 0, 1), (1, 1, 1), (1, 2, 1), (1, 3, 2),
            (2, 1, 4), (2, 2, 8), (2, 3, 12),
        ],
    );
    let b = matrix(3, 2, &[(0, 0, -4), (0, 1, 8), (1, 0, -1), (1, 1, 7), (2, 0, -8), (2, 1, 16)]);
    let witness = SparseAffineWitness {
        schema_version: SPARSE_AFFINE_SCHEMA_VERSION,
        reduction: reduction(
            matrix(
                3,
                6,
                &[
                    (0, 0, 1), (0, 2, -1), (0, 3, -1), (0, 4, 1), (0, 5, 3),
                    (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, -2), (1, 5, 4),
                ],
            ),
            vec![0, 1],
            vec![
                Row::Swap { left: 0, right: 1 },
                Row::Scale { row: 1, factor: BigRational::new(1.into(), 2.into()) },
                Row::Add { target: 0, source: 1, factor: q(-1) },
                Row::Add { target: 2, source: 1, factor: q(-4) },
            ],
        ),
        outcome: SparseAffineOutcome::Solvable {
            particular: matrix(4, 2, &[(0, 0, 1), (0, 1, 3), (1, 0, -2), (1, 1, 4)]),
            kernel: matrix(4, 2, &[(0, 0, 1), (0, 1, 1), (1, 0, -2), (1, 1, -3), (2, 0, 1), (3, 1, 1)]),
            free_columns: vec![2, 3],
        },
    };
    (a, b, witness)
}

fn inconsistent_fixture() -> (RationalSparseMatrix, RationalSparseMatrix, SparseAffineWitness) {
    // The first RHS is consistent, but the second demands x=4 and x=5.
    let a = matrix(2, 1, &[(0, 0, 1), (1, 0, 1)]);
    let b = matrix(2, 2, &[(0, 0, 2), (0, 1, 4), (1, 0, 2), (1, 1, 5)]);
    let witness = SparseAffineWitness {
        schema_version: SPARSE_AFFINE_SCHEMA_VERSION,
        reduction: reduction(
            matrix(2, 3, &[(0, 0, 1), (0, 1, 2), (1, 2, 1)]),
            vec![0, 2],
            vec![
                Row::Add { target: 1, source: 0, factor: q(-1) },
                Row::Add { target: 0, source: 1, factor: q(-4) },
            ],
        ),
        outcome: SparseAffineOutcome::Inconsistent { row: 1 },
    };
    (a, b, witness)
}

fn outputs(witness: &mut SparseAffineWitness) -> (&mut RationalSparseMatrix, &mut RationalSparseMatrix, &mut Vec<usize>) {
    match &mut witness.outcome {
        SparseAffineOutcome::Solvable { particular, kernel, free_columns } => (particular, kernel, free_columns),
        SparseAffineOutcome::Inconsistent { .. } => panic!("fixture must be solvable"),
    }
}

#[test]
fn complete_multiple_rhs_family_has_bound_portable_evidence() {
    let (a, b, witness) = fixture();
    let receipt = verify_sparse_affine(&a, &b, &witness, &mut Unbounded).unwrap();
    assert!(receipt.is_consistent());
    assert_eq!((receipt.rows(), receipt.columns(), receipt.rhs_columns()), (3, 4, 2));
    assert_eq!((receipt.rank(), receipt.augmented_rank()), (2, 2));
    assert_eq!(receipt.free_columns(), &[2, 3]);
    assert_eq!(receipt.contradiction_row(), None);
    assert!(receipt.particular_digest().is_some());
    assert!(receipt.kernel_digest().is_some());
    let decoded: SparseAffineWitness = serde_json::from_str(&serde_json::to_string(&witness).unwrap()).unwrap();
    assert_eq!(decoded, witness);
    assert_eq!(receipt, verify_sparse_affine(&a, &b, &decoded, &mut Unbounded).unwrap());
}

#[test]
fn each_original_source_is_bound_even_when_output_is_unchanged() {
    let (a, b, witness) = fixture();
    let mut changed_a = a.clone();
    changed_a.entries[0].value = q(3);
    assert!(verify_sparse_affine(&changed_a, &b, &witness, &mut Unbounded).is_err());
    let mut changed_b = b.clone();
    changed_b.entries[0].value = q(-3);
    assert!(verify_sparse_affine(&a, &changed_b, &witness, &mut Unbounded).is_err());

    let mut scaled = witness.clone();
    let doubled = |source: &RationalSparseMatrix| RationalSparseMatrix {
        rows: source.rows,
        columns: source.columns,
        entries: source.entries.iter().map(|entry| SparseRationalEntry {
            row: entry.row,
            column: entry.column,
            value: &entry.value * &q(2),
        }).collect(),
    };
    let mut operations = (0..a.rows).map(|row| Row::Scale {
        row, factor: BigRational::new(1.into(), 2.into()),
    }).collect::<Vec<_>>();
    operations.extend(scaled.reduction.operations);
    scaled.reduction.operations = operations;
    let original = verify_sparse_affine(&a, &b, &witness, &mut Unbounded).unwrap();
    let equivalent = verify_sparse_affine(&doubled(&a), &doubled(&b), &scaled, &mut Unbounded).unwrap();
    assert_ne!(original.source_digest(), equivalent.source_digest());
    assert_ne!(original.rhs_digest(), equivalent.rhs_digest());
    assert_eq!(original.particular_digest(), equivalent.particular_digest());
    assert_eq!(original.kernel_digest(), equivalent.kernel_digest());
}

#[test]
fn shifted_particular_is_refused_even_though_it_solves_the_system() {
    let (a, b, mut witness) = fixture();
    // Add the first true kernel direction to the first particular column.
    // AP still equals B; only the promised canonical free coordinates detect it.
    *outputs(&mut witness).0 = matrix(4, 2, &[
        (0, 0, 2), (0, 1, 3), (1, 0, -4), (1, 1, 4), (2, 0, 1),
    ]);
    let error = verify_sparse_affine(&a, &b, &witness, &mut Unbounded).unwrap_err();
    assert!(matches!(error, SparseRrefError::Invalid("particular free coordinate is nonzero")));
}

#[test]
fn free_column_completeness_and_order_cannot_be_forged() {
    let (a, b, valid) = fixture();
    for free in [vec![], vec![2], vec![3, 2], vec![2, 2], vec![1, 3], vec![2, 4], vec![2, 3, 4]] {
        let mut witness = valid.clone();
        *outputs(&mut witness).2 = free;
        assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded), Err(SparseRrefError::Invalid(_))));
    }
}

#[test]
fn missing_dependent_and_noncanonical_kernel_directions_are_rejected() {
    let (a, b, valid) = fixture();
    for mutant in 0..7 {
        let mut witness = valid.clone();
        let kernel = outputs(&mut witness).1;
        match mutant {
            0 => kernel.entries.retain(|entry| entry.column == 0),
            1 => kernel.entries[5].value = q(2),
            2 => kernel.entries[5].column = 0,
            3 => { kernel.entries.remove(1); }
            4 => kernel.entries[0].value = q(2),
            5 => kernel.entries.swap(0, 1),
            6 => kernel.entries.insert(0, kernel.entries[0].clone()),
            _ => unreachable!(),
        }
        assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded), Err(SparseRrefError::Invalid(_))), "kernel mutant {mutant}");
    }
}

#[test]
fn exact_original_residuals_reject_tiny_particular_and_kernel_errors() {
    let (a, b, valid) = fixture();
    let tiny = BigRational::new(1.into(), BigInt::one() << 512u32);
    for change_particular in [true, false] {
        let mut witness = valid.clone();
        let (particular, kernel, _) = outputs(&mut witness);
        let output = if change_particular { particular } else { kernel };
        output.entries[0].value = &output.entries[0].value + &tiny;
        let error = verify_sparse_affine(&a, &b, &witness, &mut Unbounded).unwrap_err();
        assert!(matches!(error, SparseRrefError::Invalid("affine original-source residual")));
    }
}

#[test]
fn output_shapes_and_canonical_coordinates_are_checked() {
    let (a, b, valid) = fixture();
    for mutant in 0..7 {
        let mut witness = valid.clone();
        let (particular, kernel, _) = outputs(&mut witness);
        match mutant {
            0 => particular.rows += 1,
            1 => particular.columns += 1,
            2 => kernel.rows += 1,
            3 => kernel.columns += 1,
            4 => particular.entries[0].value = q(0),
            5 => particular.entries[0].row = particular.rows,
            6 => particular.entries.swap(0, 1),
            _ => unreachable!(),
        }
        assert!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded).is_err(), "output mutant {mutant}");
    }
}

#[test]
fn a_contradiction_in_any_rhs_rejects_the_whole_matrix_equation() {
    let (a, b, mut witness) = inconsistent_fixture();
    let receipt = verify_sparse_affine(&a, &b, &witness, &mut Unbounded).unwrap();
    assert!(!receipt.is_consistent());
    assert_eq!(receipt.contradiction_row(), Some(1));
    assert_eq!((receipt.rank(), receipt.augmented_rank()), (1, 2));
    assert!(receipt.free_columns().is_empty());
    assert_eq!((receipt.particular_digest(), receipt.kernel_digest()), (None, None));
    witness.outcome = SparseAffineOutcome::Solvable {
        particular: matrix(1, 2, &[(0, 0, 2), (0, 1, 4)]),
        kernel: matrix(1, 0, &[]),
        free_columns: vec![],
    };
    assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded), Err(SparseRrefError::Invalid("affine solution omits an inconsistent RHS"))));
}

#[test]
fn contradiction_must_name_an_actual_zero_coefficient_nonzero_rhs_row() {
    let (a, b, valid) = inconsistent_fixture();
    for row in [0, 2, usize::MAX] {
        let mut witness = valid.clone();
        witness.outcome = SparseAffineOutcome::Inconsistent { row };
        assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded), Err(SparseRrefError::Invalid(_))));
    }
    let (a, b, mut witness) = fixture();
    for row in 0..=a.rows {
        witness.outcome = SparseAffineOutcome::Inconsistent { row };
        assert!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded).is_err());
    }
}

#[test]
fn inconsistent_receipt_still_reports_coefficient_rank_and_free_columns() {
    let a = matrix(2, 2, &[(0, 0, 1)]);
    let b = matrix(2, 1, &[(1, 0, 1)]);
    let witness = SparseAffineWitness {
        schema_version: SPARSE_AFFINE_SCHEMA_VERSION,
        reduction: reduction(matrix(2, 3, &[(0, 0, 1), (1, 2, 1)]), vec![0, 2], vec![]),
        outcome: SparseAffineOutcome::Inconsistent { row: 1 },
    };
    let receipt = verify_sparse_affine(&a, &b, &witness, &mut Unbounded).unwrap();
    assert_eq!((receipt.rank(), receipt.augmented_rank()), (1, 2));
    assert_eq!(receipt.free_columns(), &[1]);
}

#[test]
fn empty_shapes_and_zero_rhs_preserve_complete_kernel_information() {
    for (rows, columns, rhs_columns) in [(0, 0, 0), (0, 3, 2), (3, 0, 2), (3, 4, 0), (0, 0, 3), (3, 0, 0)] {
        let a = matrix(rows, columns, &[]);
        let b = matrix(rows, rhs_columns, &[]);
        let diagonal = (0..columns).map(|i| (i, i, 1)).collect::<Vec<_>>();
        let witness = SparseAffineWitness {
            schema_version: SPARSE_AFFINE_SCHEMA_VERSION,
            reduction: reduction(matrix(rows, columns + rhs_columns, &[]), vec![], vec![]),
            outcome: SparseAffineOutcome::Solvable {
                particular: matrix(columns, rhs_columns, &[]),
                kernel: matrix(columns, columns, &diagonal),
                free_columns: (0..columns).collect(),
            },
        };
        let receipt = verify_sparse_affine(&a, &b, &witness, &mut Unbounded).unwrap();
        assert!(receipt.is_consistent());
        assert_eq!(receipt.rank(), 0);
        assert_eq!(receipt.free_columns(), &(0..columns).collect::<Vec<_>>());
        assert_eq!(receipt.rhs_columns(), rhs_columns);
    }
}

#[test]
fn large_rank_deficient_sparse_system_does_not_allocate_dense_area() {
    let rank = 1024;
    let diagonal = (0..rank).map(|i| (i, i, 1)).collect::<Vec<_>>();
    let a = matrix(rank + 1, rank + 2, &diagonal);
    let b = matrix(rank + 1, 2, &[(0, 0, 5), (rank - 1, 1, -7)]);
    let mut reduced = diagonal;
    reduced.extend([(0, rank + 2, 5), (rank - 1, rank + 3, -7)]);
    reduced.sort_by_key(|&(row, column, _)| (row, column));
    let witness = SparseAffineWitness {
        schema_version: SPARSE_AFFINE_SCHEMA_VERSION,
        reduction: reduction(matrix(rank + 1, rank + 4, &reduced), (0..rank).collect(), vec![]),
        outcome: SparseAffineOutcome::Solvable {
            particular: matrix(rank + 2, 2, &[(0, 0, 5), (rank - 1, 1, -7)]),
            kernel: matrix(rank + 2, 2, &[(rank, 0, 1), (rank + 1, 1, 1)]),
            free_columns: vec![rank, rank + 1],
        },
    };
    let mut limits = BudgetLimits::uniform(10_000_000, 0);
    limits.dimensions[Dimension::MemoryBytes.index()] = 8_000_000;
    let receipt = verify_sparse_affine(&a, &b, &witness, &mut Budget::new(limits)).unwrap();
    assert_eq!(receipt.rank(), rank);
    assert_eq!(receipt.free_columns(), &[rank, rank + 1]);
}

#[test]
fn original_shapes_coordinates_and_joint_dimension_are_admitted_before_replay() {
    let (a, b, witness) = fixture();
    for mutant in 0..6 {
        let mut changed_a = a.clone();
        let mut changed_b = b.clone();
        match mutant {
            0 => changed_b.rows += 1,
            1 => changed_a.entries[0].column = changed_a.columns,
            2 => changed_b.entries[0].value = q(0),
            3 => changed_b.entries.swap(0, 1),
            4 => changed_a.rows = usize::MAX,
            5 => changed_b.columns = MAX_SPARSE_RREF_DIMENSION + 1,
            _ => unreachable!(),
        }
        assert!(verify_sparse_affine(&changed_a, &changed_b, &witness, &mut Unbounded).is_err());
    }
    let a = matrix(0, MAX_SPARSE_RREF_DIMENSION, &[]);
    let b = matrix(0, 1, &[]);
    assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded), Err(SparseRrefError::Limit("affine augmented dimensions"))));
}

#[test]
fn affine_wrapper_cannot_bypass_schema_or_invertible_replay() {
    let (a, b, valid) = fixture();
    for mutant in 0..4 {
        let mut witness = valid.clone();
        match mutant {
            0 => witness.schema_version += 1,
            1 => witness.reduction.schema_version += 1,
            2 => witness.reduction.operations.clear(),
            3 => witness.reduction.operations.push(Row::Scale { row: 2, factor: q(0) }),
            _ => unreachable!(),
        }
        assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Unbounded), Err(SparseRrefError::Invalid(_))));
    }
}

#[test]
fn portable_affine_fields_and_sequences_are_strictly_bounded() {
    let (_, _, witness) = fixture();
    let mut wire = serde_json::to_value(&witness).unwrap();
    wire["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<SparseAffineWitness>(wire).is_err());
    let mut wire = serde_json::to_value(&witness).unwrap();
    wire["outcome"]["solvable"]["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<SparseAffineWitness>(wire).is_err());
    let mut wire = serde_json::to_value(&witness).unwrap();
    wire["outcome"]["solvable"]["free_columns"] = serde_json::json!(vec![0usize; MAX_SPARSE_RREF_DIMENSION + 1]);
    assert!(serde_json::from_value::<SparseAffineWitness>(wire).is_err());
    for (numerator, denominator) in [(2i64, 2i64), (1, -2), (1, 0)] {
        let mut wire = serde_json::to_value(&witness).unwrap();
        wire["outcome"]["solvable"]["kernel"]["entries"][0]["value"] = serde_json::json!([
            serde_json::to_value(BigInt::from(numerator)).unwrap(),
            serde_json::to_value(BigInt::from(denominator)).unwrap(),
        ]);
        assert!(serde_json::from_value::<SparseAffineWitness>(wire).is_err());
    }
}

#[test]
fn outcome_decoding_bounds_a_coefficient_before_reading_the_next_entry() {
    let oversized = BigRational::from_integer(BigInt::one() << (MAX_SPARSE_RREF_COEFFICIENT_BITS as u32));
    let encoded = serde_json::to_string(&oversized).unwrap();
    // Externally tagged variants decode bounded fields directly instead of
    // buffering an unrestricted internally tagged JSON payload first.
    let wire = format!("{{\"solvable\":{{\"particular\":{{\"rows\":1,\"columns\":1,\"entries\":[{{\"row\":0,\"column\":0,\"value\":{encoded}}},invalid]}}}}}}");
    let error = serde_json::from_str::<SparseAffineOutcome>(&wire).unwrap_err();
    assert!(error.to_string().contains("height"), "{error}");
}

#[derive(Default)]
struct SafePoints {
    seen: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for SafePoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> { Ok(()) }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> { Ok(()) }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.seen += 1;
        if self.cancel_at == Some(self.seen) { Err(MeterError::Cancelled) } else { Ok(()) }
    }
}

#[test]
fn every_safe_point_cancels_both_affine_outcomes_before_publication() {
    for (a, b, witness) in [fixture(), inconsistent_fixture()] {
        let mut observed = SafePoints::default();
        verify_sparse_affine(&a, &b, &witness, &mut observed).unwrap();
        assert!(observed.seen > 20);
        for point in 1..=observed.seen {
            let mut meter = SafePoints { seen: 0, cancel_at: Some(point) };
            assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut meter), Err(SparseRrefError::Meter(MeterError::Cancelled))), "receipt escaped cancellation at {point}");
        }
    }
}

#[test]
fn exact_cumulative_budgets_cover_replay_residuals_hashing_and_publication() {
    let (a, b, witness) = fixture();
    let limit = 100_000_000;
    let mut observation = Budget::new(BudgetLimits::uniform(limit, 0));
    verify_sparse_affine(&a, &b, &witness, &mut observation).unwrap();
    let used = limit - observation.remaining(Dimension::ComputeSteps);
    assert!(used > 0);
    for dimension in [Dimension::ComputeSteps, Dimension::MemoryBytes, Dimension::AllocationCount] {
        let mut limits = BudgetLimits::uniform(limit, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Budget::new(limits)), Err(SparseRrefError::Meter(MeterError::Budget(_)))));
    }
    let mut limits = BudgetLimits::uniform(limit, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used - 1;
    assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut Budget::new(limits)), Err(SparseRrefError::Meter(MeterError::Budget(_)))));
    limits.dimensions[Dimension::ComputeSteps.index()] = used;
    let mut exact = Budget::new(limits);
    verify_sparse_affine(&a, &b, &witness, &mut exact).unwrap();
    assert_eq!(exact.remaining(Dimension::ComputeSteps), 0);
    assert!(matches!(verify_sparse_affine(&a, &b, &witness, &mut exact), Err(SparseRrefError::Meter(MeterError::Budget(_)))));
}
