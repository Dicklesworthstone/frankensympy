//! Hand-built congruences and independent weakening attacks; no generator dependency.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::BigRational;
use fsym_proof_kernel::rational_inertia::{
    RATIONAL_INERTIA_SCHEMA_VERSION, RationalInertiaError, RationalInertiaWitness,
    verify_rational_inertia,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};

fn matrix(n: usize, entries: &[i64]) -> RationalSparseMatrix {
    assert_eq!(entries.len(), n * n);
    RationalSparseMatrix {
        rows: n,
        columns: n,
        entries: entries
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .map(|(i, &v)| SparseRationalEntry {
                row: i / n,
                column: i % n,
                value: BigRational::from_integer(v.into()),
            })
            .collect(),
    }
}
fn example() -> (RationalSparseMatrix, RationalInertiaWitness) {
    // A nontrivial congruence with a 1x1 pivot followed by a zero-diagonal pair.
    (
        matrix(3, &[1, 1, 2, 1, 1, 1, 2, 1, 4]),
        RationalInertiaWitness {
            schema_version: RATIONAL_INERTIA_SCHEMA_VERSION,
            permutation: vec![0, 1, 2],
            lower: matrix(3, &[1, 0, 0, 1, 1, 0, 2, 0, 1]),
            diagonal: matrix(3, &[2, 0, 0, 0, 0, -2, 0, -2, 0]),
        },
    )
}

#[test]
fn scalar_paired_and_zero_blocks_count_complete_inertia() {
    let (source, witness) = example();
    let receipt = verify_rational_inertia(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.positive(), receipt.negative(), receipt.zero()),
        (2, 1, 0)
    );
    let zero = matrix(2, &[0, 0, 0, 0]);
    let zero_witness = RationalInertiaWitness {
        schema_version: 1,
        permutation: vec![0, 1],
        lower: matrix(2, &[1, 0, 0, 1]),
        diagonal: zero.clone(),
    };
    let receipt = verify_rational_inertia(&zero, &zero_witness, &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.positive(), receipt.negative(), receipt.zero()),
        (0, 0, 2)
    );
    let empty = matrix(0, &[]);
    let witness = RationalInertiaWitness {
        schema_version: 1,
        permutation: vec![],
        lower: empty.clone(),
        diagonal: empty.clone(),
    };
    assert_eq!(
        verify_rational_inertia(&empty, &witness, &mut Unbounded)
            .unwrap()
            .dimension(),
        0
    );
}

#[test]
fn original_nonsymmetric_source_uses_its_symmetric_part_and_exact_digest() {
    let source = matrix(2, &[1, 100, -100, 1]);
    let witness = RationalInertiaWitness {
        schema_version: 1,
        permutation: vec![0, 1],
        lower: matrix(2, &[1, 0, 0, 1]),
        diagonal: matrix(2, &[2, 0, 0, 2]),
    };
    let result = verify_rational_inertia(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(result.positive(), 2);
    let same_form = matrix(2, &[1, 0, 0, 1]);
    let other = verify_rational_inertia(&same_form, &witness, &mut Unbounded).unwrap();
    assert_ne!(result.source_digest(), other.source_digest());
    assert!(
        verify_rational_inertia(&matrix(2, &[1, 100, -99, 1]), &witness, &mut Unbounded).is_err()
    );
}

#[test]
fn singular_lower_cannot_hide_negative_or_missing_directions() {
    // Reconstruction alone accepts L=0 and arbitrary D for the zero matrix.
    let source = matrix(2, &[0, 0, 0, 0]);
    let forged = RationalInertiaWitness {
        schema_version: 1,
        permutation: vec![0, 1],
        lower: source.clone(),
        diagonal: matrix(2, &[1, 0, 0, 1]),
    };
    assert!(matches!(
        verify_rational_inertia(&source, &forged, &mut Unbounded),
        Err(RationalInertiaError::Invalid(
            "L is not unit lower triangular"
        ))
    ));
}

#[test]
fn malformed_permutations_blocks_and_reconstruction_are_rejected() {
    let (source, witness) = example();
    let mut forged = witness.clone();
    forged.permutation = vec![0, 0, 2];
    assert!(matches!(
        verify_rational_inertia(&source, &forged, &mut Unbounded),
        Err(RationalInertiaError::Invalid(
            "permutation is not bijective"
        ))
    ));
    forged = witness.clone();
    forged.permutation = vec![0, 1, 3];
    assert!(verify_rational_inertia(&source, &forged, &mut Unbounded).is_err());
    forged = witness.clone();
    forged.lower.entries[0].value = BigRational::from_integer(2.into());
    assert!(verify_rational_inertia(&source, &forged, &mut Unbounded).is_err());
    forged = witness.clone();
    forged.diagonal.entries[1].value = BigRational::from_integer(2.into());
    assert!(matches!(
        verify_rational_inertia(&source, &forged, &mut Unbounded),
        Err(RationalInertiaError::Invalid(
            "D has invalid scalar or paired blocks"
        ))
    ));
    forged = witness.clone();
    forged.diagonal.entries[0].value = BigRational::from_integer((-2).into());
    assert!(matches!(
        verify_rational_inertia(&source, &forged, &mut Unbounded),
        Err(RationalInertiaError::Invalid("P(A+A^T)P^T != L D L^T"))
    ));
    forged = witness;
    forged.schema_version = 2;
    assert!(verify_rational_inertia(&source, &forged, &mut Unbounded).is_err());
}

#[test]
fn wire_round_trip_preserves_the_bounded_untrusted_witness() {
    let (source, witness) = example();
    let encoded = serde_json::to_string(&witness).unwrap();
    let decoded: RationalInertiaWitness = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, witness);
    verify_rational_inertia(&source, &decoded, &mut Unbounded).unwrap();
    let mut malformed = serde_json::to_value(&witness).unwrap();
    malformed["permutation"] = serde_json::json!((0..65).collect::<Vec<_>>());
    assert!(serde_json::from_value::<RationalInertiaWitness>(malformed).is_err());
    let mut malformed = serde_json::to_value(&witness).unwrap();
    malformed["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<RationalInertiaWitness>(malformed).is_err());
}

struct CancelAfter {
    checkpoints: usize,
    stop: usize,
}
impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints >= self.stop {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}
#[test]
fn cancellation_and_independent_verifier_budgets_never_publish_receipts() {
    let (source, witness) = example();
    let mut count = CancelAfter {
        checkpoints: 0,
        stop: usize::MAX,
    };
    verify_rational_inertia(&source, &witness, &mut count).unwrap();
    for stop in [1, 2, 50, count.checkpoints] {
        assert!(matches!(
            verify_rational_inertia(
                &source,
                &witness,
                &mut CancelAfter {
                    checkpoints: 0,
                    stop
                }
            ),
            Err(RationalInertiaError::Reference(SparseRrefError::Meter(
                MeterError::Cancelled
            )))
        ));
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(verify_rational_inertia(&source, &witness, &mut Budget::new(limits)).is_err());
    }
}
