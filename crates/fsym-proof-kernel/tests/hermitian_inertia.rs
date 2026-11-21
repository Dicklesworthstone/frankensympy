//! Hand-derived complex forms and hostile witnesses, with no matrix generator.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_proof_kernel::hermitian_inertia::{
    GaussianRationalMatrix, HermitianInertiaError, read_gaussian_rational_matrix,
    verify_hermitian_inertia, verify_hermitian_inertia_source,
};
use fsym_proof_kernel::rational_inertia::RationalInertiaWitness;
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};
use std::sync::Arc;

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn matrix(n: usize, entries: &[i64]) -> RationalSparseMatrix {
    assert_eq!(entries.len(), n * n);
    RationalSparseMatrix {
        rows: n,
        columns: n,
        entries: entries
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(index, value)| SparseRationalEntry {
                row: index / n,
                column: index % n,
                value: rational(*value, 1),
            })
            .collect(),
    }
}

fn example() -> (GaussianRationalMatrix, RationalInertiaWitness) {
    // A = [[1,1+i],[1-i,3]]. Its determinant is 1, so A is positive definite.
    // Hand-derived LDL for 2R(A), ordered (Re z1,Re z2,Im z1,Im z2):
    // diagonal (2,4,1,2), lower rows (1), (1,1), (0,1/2,1), (-1,1/2,1,1).
    let source = GaussianRationalMatrix {
        real: matrix(2, &[1, 1, 1, 3]),
        imaginary: matrix(2, &[0, 1, -1, 0]),
    };
    let mut lower = matrix(4, &[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, -1, 1, 1, 1]);
    for entry in &mut lower.entries {
        if entry.column == 1 && entry.row > 1 {
            entry.value = rational(1, 2);
        }
    }
    (
        source,
        RationalInertiaWitness {
            schema_version: 1,
            permutation: vec![0, 1, 2, 3],
            lower,
            diagonal: matrix(4, &[2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2]),
        },
    )
}

fn original() -> Vec<Expr> {
    vec![
        Expr::from_i64(1),
        Expr::Add(vec![Expr::from_i64(1), Expr::Const(Constant::I)]),
        Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![Expr::from_i64(-1), Expr::Const(Constant::I)]),
        ]),
        Expr::from_i64(3),
    ]
}

#[test]
fn hand_derived_complex_congruence_has_doubled_real_counts() {
    let (source, witness) = example();
    let receipt = verify_hermitian_inertia(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.positive(), receipt.negative(), receipt.zero()),
        (2, 0, 0)
    );
    assert_eq!(receipt.realification().positive(), 4);
    assert_eq!(receipt.dimension(), 2);
    let cells = original();
    assert_eq!(
        verify_hermitian_inertia_source(
            2,
            2,
            cells
                .iter()
                .enumerate()
                .map(|(index, value)| ((index / 2, index % 2), value)),
            &source,
            &witness,
            &mut Unbounded
        )
        .unwrap(),
        receipt,
    );
}

#[test]
fn skew_hermitian_sources_keep_distinct_original_commitments() {
    let source = GaussianRationalMatrix {
        real: matrix(2, &[1, 7, -7, -2]),
        imaginary: matrix(2, &[5, 9, 9, -3]),
    };
    let witness = RationalInertiaWitness {
        schema_version: 1,
        permutation: vec![0, 1, 2, 3],
        lower: matrix(4, &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        diagonal: matrix(4, &[2, 0, 0, 0, 0, -4, 0, 0, 0, 0, 2, 0, 0, 0, 0, -4]),
    };
    let receipt = verify_hermitian_inertia(&source, &witness, &mut Unbounded).unwrap();
    assert_eq!(
        (receipt.positive(), receipt.negative(), receipt.zero()),
        (1, 1, 0)
    );
    let same_form = GaussianRationalMatrix {
        real: matrix(2, &[1, 0, 0, -2]),
        imaginary: matrix(2, &[0, 0, 0, 0]),
    };
    let second = verify_hermitian_inertia(&same_form, &witness, &mut Unbounded).unwrap();
    assert_ne!(receipt.real_source_digest(), second.real_source_digest());
    assert_ne!(
        receipt.imaginary_source_digest(),
        second.imaginary_source_digest()
    );
    assert_ne!(
        receipt.realification().source_digest(),
        second.realification().source_digest()
    );
}

#[test]
fn sign_coordinate_and_congruence_mutations_do_not_certify() {
    let (source, witness) = example();
    let mut changed = source.clone();
    changed.imaginary.entries[0].value = rational(-1, 1);
    changed.imaginary.entries[1].value = rational(1, 1);
    assert!(verify_hermitian_inertia(&changed, &witness, &mut Unbounded).is_err());
    let mut changed = source.clone();
    changed.real.entries[0].value = rational(2, 1);
    assert!(verify_hermitian_inertia(&changed, &witness, &mut Unbounded).is_err());
    let mut bad = witness.clone();
    bad.diagonal.entries[0].value = rational(-2, 1);
    assert!(verify_hermitian_inertia(&source, &bad, &mut Unbounded).is_err());
    let mut bad = witness.clone();
    bad.lower.entries[0].value = rational(2, 1);
    assert!(verify_hermitian_inertia(&source, &bad, &mut Unbounded).is_err());
    let mut bad = witness.clone();
    bad.permutation.swap(0, 1);
    assert!(verify_hermitian_inertia(&source, &bad, &mut Unbounded).is_err());
    let mut bad = witness;
    bad.schema_version = 2;
    assert!(verify_hermitian_inertia(&source, &bad, &mut Unbounded).is_err());
}

fn read_one(expression: &Expr) -> Result<GaussianRationalMatrix, HermitianInertiaError> {
    read_gaussian_rational_matrix(1, 1, std::iter::once(((0, 0), expression)), &mut Unbounded)
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn exact_reader_handles_gaussian_arithmetic_without_assumptions_or_approximation() {
    let base = Expr::Add(vec![Expr::from_i64(1), Expr::Const(Constant::I)]);
    let inverse = read_one(&power(base.clone(), -1)).unwrap();
    assert_eq!(inverse.real.entries[0].value, rational(1, 2));
    assert_eq!(inverse.imaginary.entries[0].value, rational(-1, 2));
    let squared = read_one(&power(base, 2)).unwrap();
    assert!(squared.real.entries.is_empty());
    assert_eq!(squared.imaginary.entries[0].value, rational(2, 1));
    let fourth = read_one(&power(Expr::Const(Constant::I), -4)).unwrap();
    assert_eq!(fourth.real.entries[0].value, rational(1, 1));
    assert!(fourth.imaginary.entries.is_empty());
    let empty = read_gaussian_rational_matrix(0, 0, std::iter::empty(), &mut Unbounded).unwrap();
    assert!(empty.real.entries.is_empty());
}

#[test]
fn held_poles_and_foreign_nodes_are_checked_before_zero_cancellation() {
    let pole = power(Expr::from_i64(0), -1);
    for expression in [
        pole.clone(),
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        power(pole, 0),
        power(Expr::from_i64(0), 0),
    ] {
        assert!(matches!(
            read_one(&expression),
            Err(HermitianInertiaError::UndefinedSource)
        ));
    }
    for expression in [
        Expr::Sym(Symbol::new("I")),
        Expr::Const(Constant::Infinity),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(Symbol::new("x"))]),
        power(Expr::Sym(Symbol::new("x")), 0),
    ] {
        assert!(matches!(
            read_one(&expression),
            Err(HermitianInertiaError::UnsupportedSource)
        ));
    }
}

#[test]
fn coefficient_snapshot_cannot_replace_original_held_or_skew_source() {
    let (source, witness) = example();
    let mut cells = original();
    // Adding i to a diagonal does not change the Hermitian part or the valid
    // LDL witness. It must still invalidate the generator's original snapshot.
    cells[0] = Expr::Add(vec![Expr::from_i64(1), Expr::Const(Constant::I)]);
    assert!(matches!(
        verify_hermitian_inertia_source(
            2,
            2,
            cells
                .iter()
                .enumerate()
                .map(|(index, value)| ((index / 2, index % 2), value)),
            &source,
            &witness,
            &mut Unbounded
        ),
        Err(HermitianInertiaError::Invalid(
            "coefficient snapshot differs from original source"
        ))
    ));
    cells[0] = Expr::Add(vec![
        Expr::from_i64(1),
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
    ]);
    assert!(matches!(
        verify_hermitian_inertia_source(
            2,
            2,
            cells
                .iter()
                .enumerate()
                .map(|(index, value)| ((index / 2, index % 2), value)),
            &source,
            &witness,
            &mut Unbounded
        ),
        Err(HermitianInertiaError::UndefinedSource)
    ));
}

#[test]
fn malformed_and_oversized_sources_fail_before_congruence_work() {
    let (source, witness) = example();
    let mut bad = source.clone();
    bad.imaginary.columns = 3;
    assert!(verify_hermitian_inertia(&bad, &witness, &mut Unbounded).is_err());
    let mut bad = source.clone();
    bad.real.entries.swap(0, 1);
    assert!(verify_hermitian_inertia(&bad, &witness, &mut Unbounded).is_err());
    let mut bad = source.clone();
    bad.real.entries[0].value = rational(0, 1);
    assert!(verify_hermitian_inertia(&bad, &witness, &mut Unbounded).is_err());
    let mut bad = source;
    bad.real.rows = 33;
    bad.real.columns = 33;
    assert!(verify_hermitian_inertia(&bad, &witness, &mut Unbounded).is_err());
    assert!(read_one(&Expr::Integer(BigInt::one() << 4096u32)).is_err());
    assert!(read_one(&power(Expr::Const(Constant::I), 65)).is_err());
    let mut deep = Expr::from_i64(1);
    for _ in 0..66 {
        deep = Expr::Add(vec![deep]);
    }
    assert!(read_one(&deep).is_err());
    let over = Expr::Add(vec![Expr::from_i64(1); 16_384]);
    assert!(read_one(&over).is_err());
}

struct CancelAfter(usize);
impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.0 == 0 {
            Err(MeterError::Cancelled)
        } else {
            self.0 -= 1;
            Ok(())
        }
    }
}

#[test]
fn composition_and_original_reader_share_cumulative_verifier_budget() {
    let (source, witness) = example();
    let cells = original();
    let mut measured = Budget::new(BudgetLimits::uniform(u64::MAX, 0));
    verify_hermitian_inertia_source(
        2,
        2,
        cells
            .iter()
            .enumerate()
            .map(|(index, value)| ((index / 2, index % 2), value)),
        &source,
        &witness,
        &mut measured,
    )
    .unwrap();
    let used = u64::MAX - measured.remaining(Dimension::ComputeSteps);
    assert!(used > 100);
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used - 1;
    assert!(
        verify_hermitian_inertia_source(
            2,
            2,
            cells
                .iter()
                .enumerate()
                .map(|(index, value)| ((index / 2, index % 2), value)),
            &source,
            &witness,
            &mut Budget::new(limits)
        )
        .is_err()
    );
    for dimension in [Dimension::MemoryBytes, Dimension::AllocationCount] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(verify_hermitian_inertia(&source, &witness, &mut Budget::new(limits)).is_err());
    }
    for stop in [0, 1, 25] {
        let result = verify_hermitian_inertia(&source, &witness, &mut CancelAfter(stop));
        assert!(matches!(
            result,
            Err(HermitianInertiaError::Reference(SparseRrefError::Meter(
                MeterError::Cancelled
            ))) | Err(HermitianInertiaError::Rational(
                fsym_proof_kernel::rational_inertia::RationalInertiaError::Reference(
                    SparseRrefError::Meter(MeterError::Cancelled)
                )
            ))
        ));
    }
}
