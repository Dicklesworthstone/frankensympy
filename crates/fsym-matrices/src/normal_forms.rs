//! Bounded Smith normal forms over the integers, with unimodular witnesses.
//!
//! Deterministic Euclidean row/column operations produce `D = U A V` and
//! maintain both integer inverses. Clearing a pivot row and column is not
//! enough: when the pivot does not divide the remaining block, a row addition
//! exposes an offending entry and another Euclidean pass strictly reduces the
//! pivot. Thus every completed pivot divides the entire remaining block.
//!
//! Acceptance belongs to the independent proof kernel. It checks the original
//! integer source, all inverse and decomposition identities, positive diagonal
//! factors, divisibility and trailing zeros. No diagonal or rank is published
//! as accepted merely because the elimination generator terminated.
//!
//! Each dimension is at most 64. Original entries have at most 4096 bits and
//! intermediate/witness integers at most 8192 bits. Every generation and
//! verification phase has a cumulative 20M-step / 256MiB allocation envelope,
//! including source admission and publication. Arithmetic uses the cooperative
//! integer kernels; resource exhaustion or cancellation yields no accepted form.

mod integer_system;
pub use integer_system::{
    IntegerLinearCandidate, IntegerMatrixRef, VerifiedIntegerLinearSystem,
    generate_integer_linear_candidate, metered_solve_integer, verify_integer_linear_candidate,
};
mod hermite;
pub use hermite::{
    HermiteNormalFormCandidate, VerifiedHermiteNormalForm, generate_hermite_normal_form_candidate,
    verify_hermite_normal_form_candidate,
};

use crate::{Matrix, MatrixError, SparseMatrix};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::{RationalArithmeticError, metered_div_rem, metered_mul, metered_subtract};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_proof_kernel::integer_normal_form::{
    INTEGER_SMITH_SCHEMA_VERSION, IntegerSmithError, IntegerSmithReceipt, IntegerSmithWitness,
    MAX_INTEGER_SMITH_COEFFICIENT_BITS, MAX_INTEGER_SMITH_DIMENSION, verify_integer_smith,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};

const MAX_INPUT_BITS: u64 = 4096;
const MAX_STEPS: u64 = 20_000_000;
const MAX_BYTES: u64 = 256 * 1024 * 1024;
type Result<T> = std::result::Result<T, MatrixError>;

/// An untrusted exact source and its proposed Smith decomposition.
/// Receivers must supply their original matrix separately for acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmithNormalFormCandidate {
    pub source: RationalSparseMatrix,
    pub witness: IntegerSmithWitness,
}

/// An independently checked complete integer equivalence `D = U A V`.
/// All canonical matrices and the receipt remain immutable behind accessors.
#[derive(Debug)]
pub struct VerifiedSmithNormalForm {
    source: RationalSparseMatrix,
    witness: IntegerSmithWitness,
    receipt: IntegerSmithReceipt,
}

impl VerifiedSmithNormalForm {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn diagonal(&self) -> &RationalSparseMatrix {
        &self.witness.diagonal
    }
    /// Left unimodular transform U in `D = U A V`.
    pub fn left(&self) -> &RationalSparseMatrix {
        &self.witness.left
    }
    /// Right unimodular transform V in `D = U A V`.
    pub fn right(&self) -> &RationalSparseMatrix {
        &self.witness.right
    }
    pub fn left_inverse(&self) -> &RationalSparseMatrix {
        &self.witness.left_inverse
    }
    pub fn right_inverse(&self) -> &RationalSparseMatrix {
        &self.witness.right_inverse
    }
    pub fn rank(&self) -> usize {
        self.receipt.rank()
    }
    pub fn invariant_factors(&self) -> &[BigInt] {
        self.receipt.invariant_factors()
    }
    pub fn witness(&self) -> &IntegerSmithWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &IntegerSmithReceipt {
        &self.receipt
    }
    /// Move source-bound evidence without copying its matrix coefficients.
    pub fn into_parts(
        self,
    ) -> (
        RationalSparseMatrix,
        IntegerSmithWitness,
        IntegerSmithReceipt,
    ) {
        (self.source, self.witness, self.receipt)
    }
}

fn limit(reason: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("integer normal form: {reason}"))
}

fn stopped(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => MatrixError::ResourceLimit(error.to_string()),
    }
}

fn verification(error: IntegerSmithError) -> MatrixError {
    match error {
        IntegerSmithError::Reference(SparseRrefError::Meter(error)) => stopped(error),
        IntegerSmithError::Reference(SparseRrefError::Arithmetic(error)) => match error {
            RationalArithmeticError::Meter(error) => stopped(error),
            RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
            RationalArithmeticError::InvariantViolation(reason) => {
                MatrixError::ArithmeticInvariantViolation(reason)
            }
            error => limit(&error.to_string()),
        },
        IntegerSmithError::Reference(SparseRrefError::Limit(reason)) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

fn coefficient(value: &BigInt, bits: u64) -> Result<()> {
    if value.bits() > bits {
        Err(limit("integer coefficient height"))
    } else {
        Ok(())
    }
}

/// Exact literal admission occurs before zero/rank shortcuts or allocation.
fn integer(value: &Expr) -> Result<&BigInt> {
    match value {
        Expr::Integer(value) => Ok(value),
        Expr::Rational(value) if value.is_integer() && value.denom().is_positive() => {
            Ok(value.numer())
        }
        _ => Err(MatrixError::UnsupportedIntegerDomain),
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    used: [u64; 3],
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; 3],
        }
    }
    fn tick(&mut self, count: usize) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge(Dimension::ComputeSteps, count.max(1) as u64)
            .map_err(stopped)
    }
    fn storage(&mut self, bytes: usize) -> Result<()> {
        self.tick(1)?;
        self.charge_batch(&[
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(stopped)
    }
    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>> {
        self.storage(
            count
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| limit("container size overflow"))?,
        )?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| limit("container allocation refused"))?;
        Ok(result)
    }
    fn copy(&mut self, value: &BigInt) -> Result<BigInt> {
        coefficient(value, MAX_INTEGER_SMITH_COEFFICIENT_BITS)?;
        self.tick(value.limb_count() as usize)?;
        self.storage(8 * value.limb_count().max(1) as usize)?;
        Ok(value.clone())
    }
    fn negate(&mut self, value: &BigInt) -> Result<BigInt> {
        Ok(-self.copy(value)?)
    }
    fn scalar(&mut self, value: i64) -> Result<BigInt> {
        self.storage(8)?;
        Ok(BigInt::from(value))
    }
    fn rational(&mut self, value: &BigInt) -> Result<BigRational> {
        // The new canonical denominator is one; the numerator is copied once.
        self.storage(8)?;
        Ok(BigRational::from_integer(self.copy(value)?))
    }
    fn multiply(&mut self, left: &BigInt, right: &BigInt) -> Result<BigInt> {
        if left.bits().saturating_add(right.bits()) > MAX_INTEGER_SMITH_COEFFICIENT_BITS {
            return Err(limit("multiplication intermediate height"));
        }
        let product = metered_mul(left, right, self).map_err(stopped)?;
        coefficient(&product, MAX_INTEGER_SMITH_COEFFICIENT_BITS)?;
        Ok(product)
    }
    fn subtract(&mut self, left: &BigInt, right: &BigInt) -> Result<BigInt> {
        if left.bits().max(right.bits()).saturating_add(1) > MAX_INTEGER_SMITH_COEFFICIENT_BITS {
            return Err(limit("subtraction intermediate height"));
        }
        let difference = metered_subtract(left, right, self).map_err(stopped)?;
        coefficient(&difference, MAX_INTEGER_SMITH_COEFFICIENT_BITS)?;
        Ok(difference)
    }
    fn divide(&mut self, left: &BigInt, right: &BigInt) -> Result<(BigInt, BigInt)> {
        coefficient(left, MAX_INTEGER_SMITH_COEFFICIENT_BITS)?;
        coefficient(right, MAX_INTEGER_SMITH_COEFFICIENT_BITS)?;
        metered_div_rem(left, right, self).map_err(stopped)?.ok_or(
            MatrixError::ArithmeticInvariantViolation(
                "zero integer normal form pivot during Euclidean division",
            ),
        )
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let mut next = self.used;
        for &(dimension, amount) in charges {
            let (index, cap) = match dimension {
                Dimension::ComputeSteps => (0, MAX_STEPS),
                Dimension::MemoryBytes => (1, MAX_BYTES),
                Dimension::AllocationCount => (2, MAX_STEPS),
                _ => continue,
            };
            let remaining = cap.saturating_sub(next[index]);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            next[index] += amount;
        }
        self.meter.charge_batch(charges)?;
        self.used = next;
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

fn admit<'a>(
    rows: usize,
    columns: usize,
    entries: impl Iterator<Item = ((usize, usize), &'a Expr)>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    work.tick(1)?;
    if rows.max(columns) > MAX_INTEGER_SMITH_DIMENSION {
        return Err(limit("dimension exceeds 64"));
    }
    let mut nonzero = 0;
    let mut previous = None;
    for (coordinate, value) in entries {
        work.tick(1)?;
        if coordinate.0 >= rows
            || coordinate.1 >= columns
            || previous.is_some_and(|p| p >= coordinate)
        {
            return Err(MatrixError::InvalidCertificate(
                "invalid integer matrix source coordinates".into(),
            ));
        }
        previous = Some(coordinate);
        let value = integer(value)?;
        coefficient(value, MAX_INPUT_BITS)?;
        nonzero += usize::from(!value.is_zero());
    }
    Ok(nonzero)
}

fn snapshot<'a>(
    rows: usize,
    columns: usize,
    entries: impl Iterator<Item = ((usize, usize), &'a Expr)> + Clone,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalSparseMatrix> {
    let nonzero = admit(rows, columns, entries.clone(), work)?;
    let mut values = work.vector(nonzero)?;
    for ((row, column), value) in entries {
        work.tick(1)?;
        let value = integer(value)?;
        if !value.is_zero() {
            values.push(SparseRationalEntry {
                row,
                column,
                value: work.rational(value)?,
            });
        }
    }
    Ok(RationalSparseMatrix {
        rows,
        columns,
        entries: values,
    })
}

/// A bounded dense generator representation, never itself a certificate.
struct IntegerMatrix {
    rows: usize,
    columns: usize,
    values: Vec<BigInt>,
}

impl IntegerMatrix {
    fn zero(rows: usize, columns: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let mut values = work.vector(rows * columns)?;
        work.tick(rows * columns)?;
        values.resize_with(rows * columns, BigInt::zero);
        Ok(Self {
            rows,
            columns,
            values,
        })
    }
    fn identity(n: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let mut result = Self::zero(n, n, work)?;
        for i in 0..n {
            result.values[i * n + i] = work.scalar(1)?;
        }
        Ok(result)
    }
    fn from_source(
        source: &RationalSparseMatrix,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<Self> {
        let mut result = Self::zero(source.rows, source.columns, work)?;
        for entry in &source.entries {
            result.values[entry.row * source.columns + entry.column] =
                work.copy(entry.value.numer())?;
        }
        Ok(result)
    }
    fn get(&self, row: usize, column: usize) -> &BigInt {
        &self.values[row * self.columns + column]
    }
    fn swap_rows(
        &mut self,
        a: usize,
        b: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        work.tick(self.columns)?;
        for column in 0..self.columns {
            self.values
                .swap(a * self.columns + column, b * self.columns + column);
        }
        Ok(())
    }
    fn swap_columns(
        &mut self,
        a: usize,
        b: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        work.tick(self.rows)?;
        for row in 0..self.rows {
            self.values
                .swap(row * self.columns + a, row * self.columns + b);
        }
        Ok(())
    }
    fn subtract_row(
        &mut self,
        target: usize,
        source: usize,
        multiple: &BigInt,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        work.tick(1)?;
        if multiple.is_zero() {
            return Ok(());
        }
        for column in 0..self.columns {
            let product = work.multiply(multiple, self.get(source, column))?;
            let value = work.subtract(self.get(target, column), &product)?;
            self.values[target * self.columns + column] = value;
        }
        Ok(())
    }
    fn subtract_column(
        &mut self,
        target: usize,
        source: usize,
        multiple: &BigInt,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        work.tick(1)?;
        if multiple.is_zero() {
            return Ok(());
        }
        for row in 0..self.rows {
            let product = work.multiply(multiple, self.get(row, source))?;
            let value = work.subtract(self.get(row, target), &product)?;
            self.values[row * self.columns + target] = value;
        }
        Ok(())
    }
    fn negate_row(&mut self, row: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
        for column in 0..self.columns {
            self.values[row * self.columns + column] = work.negate(self.get(row, column))?;
        }
        Ok(())
    }
    fn negate_column(
        &mut self,
        column: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        for row in 0..self.rows {
            self.values[row * self.columns + column] = work.negate(self.get(row, column))?;
        }
        Ok(())
    }
    fn to_sparse(&self, work: &mut Work<'_, impl BudgetMeter>) -> Result<RationalSparseMatrix> {
        work.tick(self.values.len())?;
        let count = self.values.iter().filter(|value| !value.is_zero()).count();
        let mut entries = work.vector(count)?;
        for (index, value) in self.values.iter().enumerate() {
            work.tick(1)?;
            if !value.is_zero() {
                entries.push(SparseRationalEntry {
                    row: index / self.columns,
                    column: index % self.columns,
                    value: work.rational(value)?,
                });
            }
        }
        Ok(RationalSparseMatrix {
            rows: self.rows,
            columns: self.columns,
            entries,
        })
    }
}

struct Decomposition {
    diagonal: IntegerMatrix,
    left: IntegerMatrix,
    right: IntegerMatrix,
    left_inverse: IntegerMatrix,
    right_inverse: IntegerMatrix,
}

impl Decomposition {
    fn swap_rows(
        &mut self,
        a: usize,
        b: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        self.diagonal.swap_rows(a, b, work)?;
        self.left.swap_rows(a, b, work)?;
        self.left_inverse.swap_columns(a, b, work)
    }
    fn swap_columns(
        &mut self,
        a: usize,
        b: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        self.diagonal.swap_columns(a, b, work)?;
        self.right.swap_columns(a, b, work)?;
        self.right_inverse.swap_rows(a, b, work)
    }
    fn subtract_row(
        &mut self,
        target: usize,
        source: usize,
        multiple: &BigInt,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        self.diagonal.subtract_row(target, source, multiple, work)?;
        self.left.subtract_row(target, source, multiple, work)?;
        let opposite = work.negate(multiple)?;
        // If U' = E U, then U'^-1 = U^-1 E^-1: update the source column.
        self.left_inverse
            .subtract_column(source, target, &opposite, work)
    }
    fn subtract_column(
        &mut self,
        target: usize,
        source: usize,
        multiple: &BigInt,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        self.diagonal
            .subtract_column(target, source, multiple, work)?;
        self.right.subtract_column(target, source, multiple, work)?;
        let opposite = work.negate(multiple)?;
        // If V' = V E, then V'^-1 = E^-1 V^-1: update the source row.
        self.right_inverse
            .subtract_row(source, target, &opposite, work)
    }
    fn negate_row(&mut self, row: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
        self.diagonal.negate_row(row, work)?;
        self.left.negate_row(row, work)?;
        self.left_inverse.negate_column(row, work)
    }
    fn reduce(&mut self, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
        let (rows, columns) = (self.diagonal.rows, self.diagonal.columns);
        for k in 0..rows.min(columns) {
            // Use the smallest bit-height nonzero, resolving ties in row-major
            // order. The result never depends on scheduling or hash iteration.
            let mut pivot = None;
            let mut bits = u64::MAX;
            for row in k..rows {
                for column in k..columns {
                    work.tick(1)?;
                    let value = self.diagonal.get(row, column);
                    if !value.is_zero() && value.bits() < bits {
                        pivot = Some((row, column));
                        bits = value.bits();
                    }
                }
            }
            let Some((row, column)) = pivot else { break };
            self.swap_rows(k, row, work)?;
            self.swap_columns(k, column, work)?;
            loop {
                work.tick(1)?;
                let mut restart = false;
                for row in k + 1..rows {
                    work.tick(1)?;
                    if self.diagonal.get(row, k).is_zero() {
                        continue;
                    }
                    let (quotient, remainder) =
                        work.divide(self.diagonal.get(row, k), self.diagonal.get(k, k))?;
                    self.subtract_row(row, k, &quotient, work)?;
                    if !remainder.is_zero() {
                        self.swap_rows(k, row, work)?;
                        restart = true;
                        break;
                    }
                }
                if restart {
                    continue;
                }
                for column in k + 1..columns {
                    work.tick(1)?;
                    if self.diagonal.get(k, column).is_zero() {
                        continue;
                    }
                    let (quotient, remainder) =
                        work.divide(self.diagonal.get(k, column), self.diagonal.get(k, k))?;
                    self.subtract_column(column, k, &quotient, work)?;
                    if !remainder.is_zero() {
                        self.swap_columns(k, column, work)?;
                        restart = true;
                        break;
                    }
                }
                if restart {
                    continue;
                }
                // The row/column are now cleared. An offending trailing entry
                // must reduce this pivot before it can be an invariant factor.
                let mut offending_row = None;
                'find: for row in k + 1..rows {
                    for column in k + 1..columns {
                        work.tick(1)?;
                        let value = self.diagonal.get(row, column);
                        if !value.is_zero()
                            && !work.divide(value, self.diagonal.get(k, k))?.1.is_zero()
                        {
                            offending_row = Some(row);
                            break 'find;
                        }
                    }
                }
                if let Some(row) = offending_row {
                    let minus_one = work.scalar(-1)?;
                    self.subtract_row(k, row, &minus_one, work)?;
                } else {
                    break;
                }
            }
            if self.diagonal.get(k, k).is_negative() {
                self.negate_row(k, work)?;
            }
        }
        Ok(())
    }
}

fn generate(
    source: RationalSparseMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<SmithNormalFormCandidate> {
    let mut decomposition = Decomposition {
        diagonal: IntegerMatrix::from_source(&source, work)?,
        left: IntegerMatrix::identity(source.rows, work)?,
        right: IntegerMatrix::identity(source.columns, work)?,
        left_inverse: IntegerMatrix::identity(source.rows, work)?,
        right_inverse: IntegerMatrix::identity(source.columns, work)?,
    };
    decomposition.reduce(work)?;
    let witness = IntegerSmithWitness {
        schema_version: INTEGER_SMITH_SCHEMA_VERSION,
        diagonal: decomposition.diagonal.to_sparse(work)?,
        left: decomposition.left.to_sparse(work)?,
        right: decomposition.right.to_sparse(work)?,
        left_inverse: decomposition.left_inverse.to_sparse(work)?,
        right_inverse: decomposition.right_inverse.to_sparse(work)?,
    };
    work.tick(1)?;
    Ok(SmithNormalFormCandidate { source, witness })
}

fn accept(
    original: RationalSparseMatrix,
    candidate: SmithNormalFormCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedSmithNormalForm> {
    // The separately supplied original source is authoritative. Its dimensions
    // and coefficient-height bounds were admitted before these comparisons.
    if original.rows != candidate.source.rows
        || original.columns != candidate.source.columns
        || original.entries.len() != candidate.source.entries.len()
    {
        return Err(MatrixError::InvalidCertificate(
            "Smith source differs from the original matrix".into(),
        ));
    }
    for (expected, supplied) in original.entries.iter().zip(&candidate.source.entries) {
        if !supplied.value.is_integer() || !supplied.value.denom().is_positive() {
            return Err(MatrixError::InvalidCertificate(
                "Smith candidate source contains a noninteger".into(),
            ));
        }
        coefficient(supplied.value.numer(), MAX_INPUT_BITS)?;
        work.tick(expected.value.numer().limb_count().max(1) as usize)?;
        if expected != supplied {
            return Err(MatrixError::InvalidCertificate(
                "Smith source differs from the original matrix".into(),
            ));
        }
    }
    let receipt =
        verify_integer_smith(&original, &candidate.witness, work).map_err(verification)?;
    work.tick(1)?;
    Ok(VerifiedSmithNormalForm {
        source: original,
        witness: candidate.witness,
        receipt,
    })
}

/// Generate an untrusted normal form. Acceptance requires the separate verifier.
pub fn generate_smith_normal_form_candidate(
    source: &Matrix,
    generator: &mut impl BudgetMeter,
) -> Result<SmithNormalFormCandidate> {
    source.smith_normal_form_candidate_with_meter(generator)
}

/// Bind a proposed form to the receiver's original matrix and independently
/// verify all integer equivalence and canonical diagonal obligations.
pub fn verify_smith_normal_form_candidate(
    source: &Matrix,
    candidate: SmithNormalFormCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedSmithNormalForm> {
    source.validate_shape()?;
    let mut work = Work::new(verifier);
    let original = snapshot(
        source.rows,
        source.cols,
        source
            .data
            .iter()
            .enumerate()
            .map(|(i, value)| ((i / source.cols, i % source.cols), value)),
        &mut work,
    )?;
    accept(original, candidate, &mut work)
}

impl Matrix {
    /// Admission-only preflight for bridges, before copying input cells.
    pub fn validate_integer_normal_form_source(&self) -> Result<()> {
        self.validate_shape()?;
        admit(
            self.rows,
            self.cols,
            self.data
                .iter()
                .enumerate()
                .map(|(i, value)| ((i / self.cols, i % self.cols), value)),
            &mut Work::new(&mut Unbounded),
        )?;
        Ok(())
    }
    pub fn smith_normal_form(&self) -> Result<VerifiedSmithNormalForm> {
        self.metered_smith_normal_form(&mut Unbounded, &mut Unbounded)
    }
    pub fn metered_smith_normal_form(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedSmithNormalForm> {
        let candidate = self.smith_normal_form_candidate_with_meter(generator)?;
        generator.checkpoint().map_err(stopped)?;
        let result = verify_smith_normal_form_candidate(self, candidate, verifier)?;
        generator.checkpoint().map_err(stopped)?;
        verifier.checkpoint().map_err(stopped)?;
        Ok(result)
    }
    pub fn smith_normal_form_candidate_with_meter(
        &self,
        generator: &mut impl BudgetMeter,
    ) -> Result<SmithNormalFormCandidate> {
        self.validate_shape()?;
        let mut work = Work::new(generator);
        let source = snapshot(
            self.rows,
            self.cols,
            self.data
                .iter()
                .enumerate()
                .map(|(i, value)| ((i / self.cols, i % self.cols), value)),
            &mut work,
        )?;
        generate(source, &mut work)
    }
}

impl SparseMatrix {
    /// Borrowed source admission for sparse bridges, before copying cells.
    pub fn validate_integer_normal_form_source(&self) -> Result<()> {
        admit(
            self.rows(),
            self.cols(),
            self.entries()
                .iter()
                .map(|(&coordinate, value)| (coordinate, value)),
            &mut Work::new(&mut Unbounded),
        )?;
        Ok(())
    }
    /// Sparse sources are admitted before bounded dense generator storage.
    pub fn smith_normal_form(&self) -> Result<VerifiedSmithNormalForm> {
        self.metered_smith_normal_form(&mut Unbounded, &mut Unbounded)
    }
    pub fn metered_smith_normal_form(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedSmithNormalForm> {
        let mut generation = Work::new(generator);
        let source = snapshot(
            self.rows(),
            self.cols(),
            self.entries()
                .iter()
                .map(|(&coordinate, value)| (coordinate, value)),
            &mut generation,
        )?;
        let candidate = generate(source, &mut generation)?;
        generation.tick(1)?;
        let mut verification = Work::new(verifier);
        let original = snapshot(
            self.rows(),
            self.cols(),
            self.entries()
                .iter()
                .map(|(&coordinate, value)| (coordinate, value)),
            &mut verification,
        )?;
        let result = accept(original, candidate, &mut verification)?;
        generation.tick(1)?;
        verification.tick(1)?;
        Ok(result)
    }
}
