//! Hand-authored determinant witnesses, independent of any matrix generator.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::linear_system::LinearRowOperation as Row;
use fsym_proof_kernel::sparse_determinant::{
    SPARSE_DETERMINANT_SCHEMA_VERSION, SparseDeterminantError, SparseDeterminantWitness,
    verify_sparse_determinant,
};
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION, RationalSparseMatrix,
    SPARSE_RREF_SCHEMA_VERSION, SparseRationalEntry, SparseRrefError, SparseRrefWitness,
};

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}

fn ratio(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn matrix(n: usize, values: &[(usize, usize, i64)]) -> RationalSparseMatrix {
    RationalSparseMatrix {
        rows: n,
        columns: n,
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

fn identity(n: usize) -> RationalSparseMatrix {
    RationalSparseMatrix {
        rows: n,
        columns: n,
        entries: (0..n)
            .map(|index| SparseRationalEntry {
                row: index,
                column: index,
                value: q(1),
            })
            .collect(),
    }
}

fn witness(
    determinant: BigRational,
    operations: Vec<Row>,
    reduced: RationalSparseMatrix,
    pivots: Vec<usize>,
) -> SparseDeterminantWitness {
    SparseDeterminantWitness {
        schema_version: SPARSE_DETERMINANT_SCHEMA_VERSION,
        determinant,
        reduction: SparseRrefWitness {
            schema_version: SPARSE_RREF_SCHEMA_VERSION,
            operations,
            reduced,
            pivots,
        },
    }
}

fn fixture() -> (RationalSparseMatrix, SparseDeterminantWitness) {
    // det([0 2 1; 3 0 6; 0 0 -5]) = 30. This needs a nontrivial
    // permutation, both positive and negative scales, and row additions.
    let source = matrix(3, &[(0, 1, 2), (0, 2, 1), (1, 0, 3), (1, 2, 6), (2, 2, -5)]);
    let proof = witness(
        q(30),
        vec![
            Row::Swap { left: 0, right: 1 },
            Row::Scale {
                row: 0,
                factor: ratio(1, 3),
            },
            Row::Scale {
                row: 1,
                factor: ratio(1, 2),
            },
            Row::Scale {
                row: 2,
                factor: ratio(-1, 5),
            },
            Row::Add {
                target: 0,
                source: 2,
                factor: q(-2),
            },
            Row::Add {
                target: 1,
                source: 2,
                factor: ratio(-1, 2),
            },
        ],
        identity(3),
        vec![0, 1, 2],
    );
    (source, proof)
}

#[test]
fn independent_swap_scale_and_nonsymmetric_certificate_replays_after_transport() {
    let (source, proof) = fixture();
    let receipt = verify_sparse_determinant(&source, &proof, &mut Unbounded).unwrap();
    assert_eq!(receipt.determinant(), &q(30));
    assert_eq!(receipt.dimension(), 3);
    assert_eq!(receipt.rank(), 3);
    let restored: SparseDeterminantWitness =
        serde_json::from_str(&serde_json::to_string(&proof).unwrap()).unwrap();
    let restored_source: RationalSparseMatrix =
        serde_json::from_str(&serde_json::to_string(&source).unwrap()).unwrap();
    let replay = verify_sparse_determinant(&restored_source, &restored, &mut Unbounded).unwrap();
    assert_eq!(receipt, replay);
}

#[test]
fn rational_scales_establish_the_exact_signed_value() {
    let source = RationalSparseMatrix {
        rows: 2,
        columns: 2,
        entries: vec![
            SparseRationalEntry {
                row: 0,
                column: 0,
                value: ratio(1, 2),
            },
            SparseRationalEntry {
                row: 0,
                column: 1,
                value: ratio(1, 3),
            },
            SparseRationalEntry {
                row: 1,
                column: 1,
                value: ratio(-2, 5),
            },
        ],
    };
    let proof = witness(
        ratio(-1, 5),
        vec![
            Row::Scale {
                row: 0,
                factor: q(2),
            },
            Row::Scale {
                row: 1,
                factor: ratio(-5, 2),
            },
            Row::Add {
                target: 0,
                source: 1,
                factor: ratio(-2, 3),
            },
        ],
        identity(2),
        vec![0, 1],
    );
    assert_eq!(
        verify_sparse_determinant(&source, &proof, &mut Unbounded)
            .unwrap()
            .determinant(),
        &ratio(-1, 5),
    );
}

#[test]
fn self_swaps_and_even_permutations_do_not_negate_the_determinant() {
    let source = identity(2);
    for operations in [
        vec![Row::Swap { left: 0, right: 0 }],
        vec![
            Row::Swap { left: 0, right: 1 },
            Row::Swap { left: 0, right: 1 },
        ],
        vec![
            Row::Scale {
                row: 0,
                factor: q(-1),
            },
            Row::Scale {
                row: 0,
                factor: q(-1),
            },
        ],
    ] {
        let proof = witness(q(1), operations, source.clone(), vec![0, 1]);
        assert!(verify_sparse_determinant(&source, &proof, &mut Unbounded).is_ok());
    }
}

#[test]
fn singularity_requires_complete_rank_evidence() {
    let source = matrix(2, &[(0, 0, 2), (0, 1, 4), (1, 0, 1), (1, 1, 2)]);
    let mut proof = witness(
        q(0),
        vec![
            Row::Scale {
                row: 0,
                factor: ratio(1, 2),
            },
            Row::Add {
                target: 1,
                source: 0,
                factor: q(-1),
            },
        ],
        matrix(2, &[(0, 0, 1), (0, 1, 2)]),
        vec![0],
    );
    let receipt = verify_sparse_determinant(&source, &proof, &mut Unbounded).unwrap();
    assert_eq!(receipt.determinant(), &q(0));
    assert_eq!(receipt.rank(), 1);
    proof.determinant = q(1);
    assert!(verify_sparse_determinant(&source, &proof, &mut Unbounded).is_err());
    proof.determinant = q(0);
    proof.reduction.operations.clear();
    assert!(verify_sparse_determinant(&source, &proof, &mut Unbounded).is_err());
    let false_zero = witness(q(0), vec![], identity(2), vec![0]);
    assert!(verify_sparse_determinant(&identity(2), &false_zero, &mut Unbounded).is_err());
}

#[test]
fn empty_determinant_is_one_and_rectangular_inputs_are_refused() {
    let empty = matrix(0, &[]);
    let proof = witness(q(1), vec![], empty.clone(), vec![]);
    assert_eq!(
        verify_sparse_determinant(&empty, &proof, &mut Unbounded)
            .unwrap()
            .determinant(),
        &q(1)
    );
    let wrong = witness(q(0), vec![], empty.clone(), vec![]);
    assert!(verify_sparse_determinant(&empty, &wrong, &mut Unbounded).is_err());
    let rectangular = RationalSparseMatrix {
        rows: 0,
        columns: 1,
        entries: vec![],
    };
    assert!(matches!(
        verify_sparse_determinant(&rectangular, &proof, &mut Unbounded),
        Err(SparseDeterminantError::Invalid(_))
    ));
}

#[test]
fn equal_determinants_never_substitute_for_original_source_binding() {
    let first = matrix(2, &[(0, 0, 2), (1, 1, 3)]);
    let second = matrix(2, &[(0, 0, 3), (1, 1, 2)]);
    let first_proof = witness(
        q(6),
        vec![
            Row::Scale {
                row: 0,
                factor: ratio(1, 2),
            },
            Row::Scale {
                row: 1,
                factor: ratio(1, 3),
            },
        ],
        identity(2),
        vec![0, 1],
    );
    let second_proof = witness(
        q(6),
        vec![
            Row::Scale {
                row: 0,
                factor: ratio(1, 3),
            },
            Row::Scale {
                row: 1,
                factor: ratio(1, 2),
            },
        ],
        identity(2),
        vec![0, 1],
    );
    let a = verify_sparse_determinant(&first, &first_proof, &mut Unbounded).unwrap();
    let b = verify_sparse_determinant(&second, &second_proof, &mut Unbounded).unwrap();
    assert_eq!(a.determinant(), b.determinant());
    assert_ne!(a.source_digest(), b.source_digest());
    assert!(verify_sparse_determinant(&second, &first_proof, &mut Unbounded).is_err());
}

#[test]
fn wrong_determinant_and_malformed_reduction_mutants_are_rejected() {
    let (source, valid) = fixture();
    for mutation in 0..12 {
        let mut proof = valid.clone();
        match mutation {
            0 => proof.determinant = q(-30),
            1 => proof.determinant = q(0),
            2 => proof.schema_version += 1,
            3 => proof.reduction.schema_version += 1,
            4 => {
                proof.reduction.operations.remove(0);
            }
            5 => {
                proof.reduction.operations[1] = Row::Scale {
                    row: 0,
                    factor: q(0),
                }
            }
            6 => {
                proof.reduction.operations[1] = Row::Scale {
                    row: 0,
                    factor: ratio(-1, 3),
                }
            }
            7 => {
                proof.reduction.operations[4] = Row::Add {
                    target: 0,
                    source: 0,
                    factor: q(-1),
                }
            }
            8 => proof.reduction.reduced.entries[0].value = q(2),
            9 => {
                proof.reduction.pivots.pop();
            }
            10 => {
                proof.reduction.operations[0] = Row::Swap {
                    left: 0,
                    right: usize::MAX,
                }
            }
            11 => {
                proof.reduction.reduced.entries.pop();
            }
            _ => unreachable!(),
        }
        assert!(
            verify_sparse_determinant(&source, &proof, &mut Unbounded).is_err(),
            "mutant {mutation}"
        );
    }
}

#[test]
fn malformed_source_and_untrusted_dimensions_fail_closed() {
    let (source, proof) = fixture();
    for mutation in 0..5 {
        let mut changed = source.clone();
        match mutation {
            0 => changed.entries[0].value = q(0),
            1 => changed.entries.swap(0, 1),
            2 => changed.entries[0].column = changed.columns,
            3 => {
                changed.rows = MAX_SPARSE_RREF_DIMENSION + 1;
                changed.columns = changed.rows;
            }
            4 => {
                changed.rows = usize::MAX;
                changed.columns = usize::MAX;
            }
            _ => unreachable!(),
        }
        assert!(verify_sparse_determinant(&changed, &proof, &mut Unbounded).is_err());
    }
}

#[test]
fn scalar_admission_applies_to_claims_and_irrelevant_singular_row_scales() {
    let source = matrix(1, &[]);
    let oversized =
        BigRational::from_integer(BigInt::one() << MAX_SPARSE_RREF_COEFFICIENT_BITS as u32);
    let mut proof = witness(oversized.clone(), vec![], source.clone(), vec![]);
    assert!(verify_sparse_determinant(&source, &proof, &mut Unbounded).is_err());
    proof.determinant = q(0);
    proof.reduction.operations = vec![Row::Scale {
        row: 0,
        factor: oversized,
    }];
    assert!(verify_sparse_determinant(&source, &proof, &mut Unbounded).is_err());
}

#[test]
fn determinant_accumulation_keeps_the_intrinsic_coefficient_limit_after_replay() {
    let power = BigInt::one() << 5_000u32;
    let large = BigRational::from_integer(power.clone());
    let small = BigRational::new(BigInt::one(), power);
    let diagonal = [small.clone(), small.clone(), large.clone(), large.clone()];
    let source = RationalSparseMatrix {
        rows: 4,
        columns: 4,
        entries: diagonal
            .into_iter()
            .enumerate()
            .map(|(index, value)| SparseRationalEntry {
                row: index,
                column: index,
                value,
            })
            .collect(),
    };
    let proof = witness(
        q(1),
        [large.clone(), large, small.clone(), small]
            .into_iter()
            .enumerate()
            .map(|(row, factor)| Row::Scale { row, factor })
            .collect(),
        identity(4),
        vec![0, 1, 2, 3],
    );
    // Every individual row replay stays within the coefficient limit. The
    // determinant's product of scales is a separate bounded computation.
    assert!(
        fsym_proof_kernel::sparse_rref::verify_sparse_rref(
            &source,
            &proof.reduction,
            &mut Unbounded,
        )
        .is_ok()
    );
    assert!(matches!(
        verify_sparse_determinant(&source, &proof, &mut Unbounded),
        Err(SparseDeterminantError::Reduction(SparseRrefError::Limit(_)))
    ));
}

#[test]
fn wire_decoder_rejects_extra_authority_and_noncanonical_claims() {
    let (_, proof) = fixture();
    let mut wire = serde_json::to_value(&proof).unwrap();
    wire["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<SparseDeterminantWitness>(wire).is_err());
    for (n, d) in [(2, 2), (1, -2), (1, 0)] {
        let mut wire = serde_json::to_value(&proof).unwrap();
        wire["determinant"] = serde_json::json!([BigInt::from(n), BigInt::from(d)]);
        assert!(serde_json::from_value::<SparseDeterminantWitness>(wire).is_err());
    }
    let oversized =
        BigRational::from_integer(BigInt::one() << MAX_SPARSE_RREF_COEFFICIENT_BITS as u32);
    let wire = format!(
        "{{\"schema_version\":1,\"determinant\":{},\"reduction\":invalid}}",
        serde_json::to_string(&oversized).unwrap()
    );
    assert!(
        serde_json::from_str::<SparseDeterminantWitness>(&wire)
            .unwrap_err()
            .to_string()
            .contains("height")
    );
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
fn all_replay_arithmetic_and_publication_safe_points_refuse_after_cancellation() {
    let (source, proof) = fixture();
    let mut measured = Meter::default();
    verify_sparse_determinant(&source, &proof, &mut measured).unwrap();
    for stop in 1..=measured.checkpoints {
        let mut meter = Meter {
            stop: Some(stop),
            ..Meter::default()
        };
        assert!(
            matches!(
                verify_sparse_determinant(&source, &proof, &mut meter),
                Err(SparseDeterminantError::Reduction(SparseRrefError::Meter(
                    MeterError::Cancelled
                )))
            ),
            "safe point {stop}"
        );
    }
}

#[test]
fn exact_cumulative_budget_boundaries_cover_final_receipt_publication() {
    let (source, proof) = fixture();
    let mut measured = Meter::default();
    verify_sparse_determinant(&source, &proof, &mut measured).unwrap();
    for (dimension, used) in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ]
    .into_iter()
    .zip(measured.used)
    {
        assert!(used > 0);
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = used;
        assert!(verify_sparse_determinant(&source, &proof, &mut Budget::new(limits)).is_ok());
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = used - 1;
        assert!(verify_sparse_determinant(&source, &proof, &mut Budget::new(limits)).is_err());
    }
}

#[test]
fn large_sparse_identity_verifies_with_less_than_dense_storage() {
    let source = identity(1_024);
    let proof = witness(q(1), vec![], source.clone(), (0..1_024).collect());
    let mut limits = BudgetLimits::uniform(1_000_000, 0);
    limits.dimensions[Dimension::MemoryBytes.index()] = 2_000_000;
    let receipt = verify_sparse_determinant(&source, &proof, &mut Budget::new(limits)).unwrap();
    assert_eq!(receipt.dimension(), 1_024);
    assert_eq!(receipt.determinant(), &q(1));
}
