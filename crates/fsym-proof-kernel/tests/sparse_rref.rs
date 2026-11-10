//! Hand-authored sparse certificates, independent of the optimizing matrix
//! generator. These adversaries target invertibility, source binding and the
//! complete reduced-row-echelon claim, rather than just residual identities.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::linear_system::LinearRowOperation as Row;
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION, RationalSparseMatrix,
    SPARSE_RREF_SCHEMA_VERSION, SparseRationalEntry, SparseRrefError, SparseRrefWitness,
    verify_sparse_rref,
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

fn identity_witness(source: &RationalSparseMatrix, pivots: Vec<usize>) -> SparseRrefWitness {
    SparseRrefWitness {
        schema_version: SPARSE_RREF_SCHEMA_VERSION,
        operations: vec![],
        reduced: source.clone(),
        pivots,
    }
}

fn fixture() -> (RationalSparseMatrix, SparseRrefWitness) {
    // [0 2 4 6; 1 1 1 2; 0 4 8 12] has rank two and two free columns.
    let source = matrix(
        3,
        4,
        &[
            (0, 1, 2),
            (0, 2, 4),
            (0, 3, 6),
            (1, 0, 1),
            (1, 1, 1),
            (1, 2, 1),
            (1, 3, 2),
            (2, 1, 4),
            (2, 2, 8),
            (2, 3, 12),
        ],
    );
    let witness = SparseRrefWitness {
        schema_version: SPARSE_RREF_SCHEMA_VERSION,
        operations: vec![
            Row::Swap { left: 0, right: 1 },
            Row::Scale {
                row: 1,
                factor: BigRational::new(1.into(), 2.into()),
            },
            Row::Add {
                target: 0,
                source: 1,
                factor: q(-1),
            },
            Row::Add {
                target: 2,
                source: 1,
                factor: q(-4),
            },
        ],
        reduced: matrix(
            3,
            4,
            &[
                (0, 0, 1),
                (0, 2, -1),
                (0, 3, -1),
                (1, 1, 1),
                (1, 2, 2),
                (1, 3, 3),
            ],
        ),
        pivots: vec![0, 1],
    };
    (source, witness)
}

#[test]
fn exact_sparse_trace_checks_rank_and_portable_round_trip() {
    let (source, witness) = fixture();
    let receipt = verify_sparse_rref(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.rows(), 3);
    assert_eq!(receipt.columns(), 4);
    assert_eq!(receipt.rank(), 2);
    assert_eq!(receipt.pivots(), &[0, 1]);
    let decoded_source: RationalSparseMatrix =
        serde_json::from_str(&serde_json::to_string(&source).unwrap()).unwrap();
    let decoded_witness: SparseRrefWitness =
        serde_json::from_str(&serde_json::to_string(&witness).unwrap()).unwrap();
    assert_eq!(decoded_source, source);
    assert_eq!(decoded_witness, witness);
    let replay = verify_sparse_rref(&decoded_source, &decoded_witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.source_digest(), replay.source_digest());
    assert_eq!(receipt.reduced_digest(), replay.reduced_digest());
}

#[test]
fn equivalent_row_spaces_keep_distinct_source_binding() {
    let source = matrix(1, 2, &[(0, 0, 2), (0, 1, 6)]);
    let reduced = matrix(1, 2, &[(0, 0, 1), (0, 1, 3)]);
    let witness = SparseRrefWitness {
        schema_version: SPARSE_RREF_SCHEMA_VERSION,
        operations: vec![Row::Scale {
            row: 0,
            factor: BigRational::new(1.into(), 2.into()),
        }],
        reduced: reduced.clone(),
        pivots: vec![0],
    };
    let scaled = verify_sparse_rref(&source, &witness, &mut Unbounded).unwrap();
    let unchanged = verify_sparse_rref(
        &reduced,
        &identity_witness(&reduced, vec![0]),
        &mut Unbounded,
    )
    .unwrap();
    assert_ne!(scaled.source_digest(), unchanged.source_digest());
    assert_eq!(scaled.reduced_digest(), unchanged.reduced_digest());

    let wider = matrix(1, 3, &[(0, 0, 1), (0, 1, 3)]);
    let wider_receipt =
        verify_sparse_rref(&wider, &identity_witness(&wider, vec![0]), &mut Unbounded).unwrap();
    assert_ne!(unchanged.source_digest(), wider_receipt.source_digest());
    assert_ne!(unchanged.reduced_digest(), wider_receipt.reduced_digest());
}

#[test]
fn zero_scale_and_self_add_cannot_erase_rank() {
    let source = matrix(1, 1, &[(0, 0, 1)]);
    for operation in [
        Row::Scale {
            row: 0,
            factor: q(0),
        },
        Row::Add {
            target: 0,
            source: 0,
            factor: q(-1),
        },
    ] {
        let witness = SparseRrefWitness {
            schema_version: SPARSE_RREF_SCHEMA_VERSION,
            operations: vec![operation],
            reduced: matrix(1, 1, &[]),
            pivots: vec![],
        };
        assert!(matches!(
            verify_sparse_rref(&source, &witness, &mut Unbounded),
            Err(SparseRrefError::Invalid(_))
        ));
    }
}

#[test]
fn transcript_and_reduced_matrix_mutations_fail_closed() {
    let (source, valid) = fixture();
    for mutation in 0..10 {
        let mut witness = valid.clone();
        match mutation {
            0 => witness.operations.clear(),
            1 => witness.operations.swap(0, 1),
            2 => witness.operations[0] = Row::Swap { left: 0, right: 3 },
            3 => {
                witness.operations[1] = Row::Scale {
                    row: 3,
                    factor: q(1),
                }
            }
            4 => {
                witness.operations[3] = Row::Add {
                    target: 2,
                    source: usize::MAX,
                    factor: q(-4),
                }
            }
            5 => witness.reduced.entries[1].value = q(1),
            6 => witness.reduced.rows += 1,
            7 => witness.reduced.columns += 1,
            8 => witness.schema_version += 1,
            _ => {
                witness.reduced.entries.pop();
            }
        }
        assert!(
            verify_sparse_rref(&source, &witness, &mut Unbounded).is_err(),
            "accepted transcript/reduced mutation {mutation}"
        );
    }
    let mut changed = source;
    changed.entries[6].value = q(3);
    assert!(verify_sparse_rref(&changed, &valid, &mut Unbounded).is_err());
}

#[test]
fn exact_tiny_changes_cannot_hide_in_a_numerical_tolerance() {
    let (source, mut witness) = fixture();
    let epsilon = BigRational::new(1.into(), BigInt::one() << 512u32);
    witness.reduced.entries[1].value += epsilon;
    assert!(verify_sparse_rref(&source, &witness, &mut Unbounded).is_err());
}

#[test]
fn sparse_coordinates_are_canonical_and_in_bounds() {
    let (source, witness) = fixture();
    for mutation in 0..6 {
        let mut changed = source.clone();
        match mutation {
            0 => changed.entries[0].row = changed.rows,
            1 => changed.entries[0].column = changed.columns,
            2 => changed.entries.swap(0, 1),
            3 => changed.entries.insert(1, changed.entries[0].clone()),
            4 => changed.entries[0].value = q(0),
            _ => changed.entries[0].row = usize::MAX,
        }
        assert!(
            verify_sparse_rref(&changed, &witness, &mut Unbounded).is_err(),
            "accepted source coordinate mutation {mutation}"
        );
    }
    for mutation in 0..5 {
        let mut changed = witness.clone();
        match mutation {
            0 => changed.reduced.entries[0].row = changed.reduced.rows,
            1 => changed.reduced.entries[0].column = changed.reduced.columns,
            2 => changed.reduced.entries.swap(0, 1),
            3 => changed
                .reduced
                .entries
                .insert(1, changed.reduced.entries[0].clone()),
            _ => changed.reduced.entries[0].value = q(0),
        }
        assert!(
            verify_sparse_rref(&source, &changed, &mut Unbounded).is_err(),
            "accepted reduced coordinate mutation {mutation}"
        );
    }
}

#[test]
fn pivot_list_is_complete_and_ordered() {
    let (source, valid) = fixture();
    for pivots in [
        vec![],
        vec![0],
        vec![1, 0],
        vec![0, 0],
        vec![0, 2],
        vec![0, 1, 2],
        vec![0, usize::MAX],
    ] {
        let mut witness = valid.clone();
        witness.pivots = pivots;
        assert!(verify_sparse_rref(&source, &witness, &mut Unbounded).is_err());
    }
}

#[test]
fn row_equivalence_alone_is_not_a_reduced_echelon_certificate() {
    // Each certificate below honestly describes the unchanged matrix, but
    // violates a different canonical echelon condition. Trace-only acceptance
    // would incorrectly grant its advertised pivots/rank.
    let cases = [
        (matrix(1, 2, &[(0, 0, 2)]), vec![0]), // pivot must be one
        (matrix(1, 2, &[(0, 0, 3), (0, 1, 1)]), vec![1]), // earlier nonzero
        (matrix(2, 2, &[(0, 0, 1), (0, 1, 1), (1, 1, 1)]), vec![0, 1]), // above pivot
        (matrix(2, 2, &[(0, 0, 1), (1, 0, 1), (1, 1, 1)]), vec![0, 1]), // below pivot
        (matrix(2, 2, &[(0, 1, 1), (1, 0, 1)]), vec![1, 0]), // descending pivots
        (matrix(2, 2, &[(1, 1, 1)]), vec![1]), // zero row above a pivot row
        (matrix(2, 3, &[(0, 0, 1), (1, 2, 1)]), vec![0]), // hidden nonzero row
    ];
    for (index, (source, pivots)) in cases.into_iter().enumerate() {
        let witness = identity_witness(&source, pivots);
        assert!(
            verify_sparse_rref(&source, &witness, &mut Unbounded).is_err(),
            "accepted non-RREF case {index}"
        );
    }
}

#[test]
fn free_columns_before_and_between_pivots_are_valid() {
    let source = matrix(
        3,
        6,
        &[(0, 1, 1), (0, 2, 4), (0, 5, -2), (1, 4, 1), (1, 5, 3)],
    );
    let receipt = verify_sparse_rref(
        &source,
        &identity_witness(&source, vec![1, 4]),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.rank(), 2);
    assert_eq!(receipt.pivots(), &[1, 4]);
}

#[test]
fn zero_area_shapes_have_rank_zero_without_fake_pivots() {
    for (rows, columns) in [(0, 0), (0, 3), (4, 0), (4, 7)] {
        let source = matrix(rows, columns, &[]);
        let witness = identity_witness(&source, vec![]);
        let receipt = verify_sparse_rref(&source, &witness, &mut Unbounded).unwrap();
        assert_eq!(receipt.rows(), rows);
        assert_eq!(receipt.columns(), columns);
        assert_eq!(receipt.rank(), 0);
        let mut fake = witness;
        fake.pivots.push(0);
        assert!(verify_sparse_rref(&source, &fake, &mut Unbounded).is_err());
    }
}

#[test]
fn large_rectangular_sparse_input_needs_no_dense_rectangle() {
    let dimension = 1_024;
    let mut source = RationalSparseMatrix {
        rows: dimension + 1,
        columns: dimension + 2,
        entries: (0..dimension)
            .map(|index| SparseRationalEntry {
                row: index,
                column: index,
                value: q(1),
            })
            .collect(),
    };
    // One extra zero row and two free coordinates distinguish this from a
    // square identity. The last nonzero is in a free column of the final row.
    source.entries.push(SparseRationalEntry {
        row: dimension - 1,
        column: dimension + 1,
        value: BigRational::new(2.into(), 3.into()),
    });
    let witness = identity_witness(&source, (0..dimension).collect());
    let mut limits = BudgetLimits::uniform(1_000_000, 0);
    // A dense array of BigRational here would require tens of megabytes.
    limits.dimensions[Dimension::MemoryBytes.index()] = 2_000_000;
    let mut budget = Budget::new(limits);
    let receipt = verify_sparse_rref(&source, &witness, &mut budget).unwrap();
    assert_eq!(receipt.rank(), dimension);
    assert_eq!(receipt.columns() - receipt.rank(), 2);
}

#[test]
fn dimension_and_coefficient_bounds_apply_before_replay() {
    let (source, witness) = fixture();
    for (rows, columns) in [
        (MAX_SPARSE_RREF_DIMENSION + 1, source.columns),
        (source.rows, MAX_SPARSE_RREF_DIMENSION + 1),
        (usize::MAX, usize::MAX),
    ] {
        let mut changed = source.clone();
        changed.rows = rows;
        changed.columns = columns;
        assert!(matches!(
            verify_sparse_rref(&changed, &witness, &mut Unbounded),
            Err(SparseRrefError::Limit(_))
        ));
    }
    let mut changed = source;
    changed.entries[0].value =
        BigRational::from_integer(BigInt::one() << (MAX_SPARSE_RREF_COEFFICIENT_BITS as u32));
    assert!(matches!(
        verify_sparse_rref(&changed, &witness, &mut Unbounded),
        Err(SparseRrefError::Limit(_))
    ));
    let edge = matrix(MAX_SPARSE_RREF_DIMENSION, MAX_SPARSE_RREF_DIMENSION, &[]);
    assert_eq!(
        verify_sparse_rref(&edge, &identity_witness(&edge, vec![]), &mut Unbounded)
            .unwrap()
            .rank(),
        0
    );
}

#[test]
fn unknown_wire_fields_and_noncanonical_rationals_are_refused() {
    let (source, witness) = fixture();
    let mut wire = serde_json::to_value(&source).unwrap();
    wire["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<RationalSparseMatrix>(wire).is_err());
    let mut wire = serde_json::to_value(&witness).unwrap();
    wire["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<SparseRrefWitness>(wire).is_err());
    let mut wire = serde_json::to_value(&source).unwrap();
    wire["entries"][0]["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<RationalSparseMatrix>(wire).is_err());
    for (numerator, denominator) in [(2i64, 2i64), (1, -2), (1, 0)] {
        let mut wire = serde_json::to_value(&source).unwrap();
        wire["entries"][0]["value"] = serde_json::json!([
            serde_json::to_value(BigInt::from(numerator)).unwrap(),
            serde_json::to_value(BigInt::from(denominator)).unwrap(),
        ]);
        assert!(serde_json::from_value::<RationalSparseMatrix>(wire).is_err());
    }
    let mut wire = serde_json::to_value(&witness).unwrap();
    wire["pivots"] = serde_json::json!(vec![0usize; MAX_SPARSE_RREF_DIMENSION + 1]);
    assert!(serde_json::from_value::<SparseRrefWitness>(wire).is_err());
}

#[test]
fn coefficient_height_is_rejected_before_the_next_wire_entry() {
    let oversized =
        BigRational::from_integer(BigInt::one() << (MAX_SPARSE_RREF_COEFFICIENT_BITS as u32));
    let encoded = serde_json::to_string(&oversized).unwrap();
    let wire = format!(
        "{{\"rows\":1,\"columns\":2,\"entries\":[{{\"row\":0,\"column\":0,\"value\":{encoded}}},invalid]}}"
    );
    // The invalid second entry must never be parsed: one oversized scalar is
    // refused before the decoder can retain a whole sequence of them.
    let error = serde_json::from_str::<RationalSparseMatrix>(&wire).unwrap_err();
    assert!(error.to_string().contains("height"), "{error}");

    let edge =
        BigRational::from_integer(BigInt::one() << (MAX_SPARSE_RREF_COEFFICIENT_BITS as u32 - 1));
    let source = RationalSparseMatrix {
        rows: 1,
        columns: 1,
        entries: vec![SparseRationalEntry {
            row: 0,
            column: 0,
            value: edge,
        }],
    };
    let decoded: RationalSparseMatrix =
        serde_json::from_str(&serde_json::to_string(&source).unwrap()).unwrap();
    assert_eq!(decoded, source);
}

#[test]
fn irrelevant_zero_rows_do_not_bypass_factor_admission() {
    let source = matrix(2, 1, &[]);
    let oversized =
        BigRational::from_integer(BigInt::one() << (MAX_SPARSE_RREF_COEFFICIENT_BITS as u32));
    for operation in [
        Row::Scale {
            row: 0,
            factor: oversized.clone(),
        },
        Row::Add {
            target: 1,
            source: 0,
            factor: oversized,
        },
    ] {
        let mut witness = identity_witness(&source, vec![]);
        witness.operations.push(operation);
        assert!(verify_sparse_rref(&source, &witness, &mut Unbounded).is_err());
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
fn every_observed_safe_point_cancels_without_a_receipt() {
    let source = matrix(1, 1, &[(0, 0, 2)]);
    let witness = SparseRrefWitness {
        schema_version: SPARSE_RREF_SCHEMA_VERSION,
        operations: vec![Row::Scale {
            row: 0,
            factor: BigRational::new(1.into(), 2.into()),
        }],
        reduced: matrix(1, 1, &[(0, 0, 1)]),
        pivots: vec![0],
    };
    let mut observation = SafePoints::default();
    verify_sparse_rref(&source, &witness, &mut observation).unwrap();
    assert!(observation.seen > 10);
    for point in 1..=observation.seen {
        let mut meter = SafePoints {
            seen: 0,
            cancel_at: Some(point),
        };
        assert!(
            matches!(
                verify_sparse_rref(&source, &witness, &mut meter),
                Err(SparseRrefError::Meter(MeterError::Cancelled))
            ),
            "accepted after cancellation at safe point {point}"
        );
    }
}

#[test]
fn repeated_verification_does_not_reset_the_callers_ledger() {
    let (source, witness) = fixture();
    let initial = 100_000_000;
    let mut observation = Budget::new(BudgetLimits::uniform(initial, 0));
    verify_sparse_rref(&source, &witness, &mut observation).unwrap();
    let used = initial - observation.remaining(Dimension::ComputeSteps);
    assert!(used > 0);
    let mut limits = BudgetLimits::uniform(initial, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used;
    let mut bounded = Budget::new(limits);
    verify_sparse_rref(&source, &witness, &mut bounded).unwrap();
    assert_eq!(bounded.remaining(Dimension::ComputeSteps), 0);
    assert!(matches!(
        verify_sparse_rref(&source, &witness, &mut bounded),
        Err(SparseRrefError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn protected_verifier_resources_work_after_ordinary_compute_is_exhausted() {
    let (source, witness) = fixture();
    let mut limits = BudgetLimits::uniform(100_000_000, 1_000_000);
    limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut budget = Budget::new(limits);
    assert!(matches!(
        verify_sparse_rref(&source, &witness, &mut budget),
        Err(SparseRrefError::Meter(MeterError::Budget(_)))
    ));
    let lease = budget.verifier_lease().unwrap();
    verify_sparse_rref(
        &source,
        &witness,
        &mut budget.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert!(budget.verifier_remaining() < 1_000_000);
}
