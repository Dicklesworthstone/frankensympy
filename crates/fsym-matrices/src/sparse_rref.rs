//! Sparse exact row reduction with dependency-separated witness acceptance.
//!
//! Generation owns sparse rows and column incidence sets. The proof kernel
//! independently replays invertible row operations against the original input
//! and checks every RREF pivot, zero, and row. Rank and complete nullspaces do
//! not require dense storage. A second sparse residual/free-coordinate check
//! binds extracted nullspace vectors to the original matrix.
//!
//! One-meter operations share cumulative accounting across source admission,
//! generation, verification, extraction, and publication. Separate-meter APIs
//! let an owning runtime reserve verifier resources; this synchronous module
//! does not create such a reservation itself. All limits remain active for an
//! unbounded caller, and operational failures never restart another lane.

use crate::{MatrixError, SparseMatrix};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use fsym_proof_kernel::linear_system::LinearRowOperation;
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION, MAX_SPARSE_RREF_ENTRIES,
    MAX_SPARSE_RREF_OPERATIONS, RationalSparseMatrix, SPARSE_RREF_SCHEMA_VERSION,
    SparseRationalEntry, SparseRrefError, SparseRrefReceipt, SparseRrefWitness, verify_sparse_rref,
};
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};

const MAX_WORK: u64 = 20_000_000;
const MAX_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;
type Row = BTreeMap<usize, BigRational>;

/// Untrusted row-reduction proposal. Its source is checked against the actual
/// matrix again by [`verify_sparse_rref_candidate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseRrefCandidate {
    pub source: RationalSparseMatrix,
    pub witness: SparseRrefWitness,
}

/// Original input, complete RREF witness, and independent acceptance receipt.
/// There is no public unchecked constructor or deserializer.
#[derive(Debug)]
pub struct VerifiedSparseRref {
    source: RationalSparseMatrix,
    witness: SparseRrefWitness,
    receipt: SparseRrefReceipt,
}

impl VerifiedSparseRref {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn witness(&self) -> &SparseRrefWitness {
        &self.witness
    }
    pub fn reduced(&self) -> &RationalSparseMatrix {
        &self.witness.reduced
    }
    pub fn pivots(&self) -> &[usize] {
        &self.witness.pivots
    }
    pub fn rank(&self) -> usize {
        self.receipt.rank()
    }
    pub fn receipt(&self) -> &SparseRrefReceipt {
        &self.receipt
    }
}

fn stopped(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => MatrixError::ResourceLimit(error.to_string()),
    }
}

fn arithmetic_error(error: RationalArithmeticError) -> MatrixError {
    match error {
        RationalArithmeticError::Meter(error) => stopped(error),
        RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
        RationalArithmeticError::InvariantViolation(message) => {
            MatrixError::ArithmeticInvariantViolation(message)
        }
        error => MatrixError::ResourceLimit(error.to_string()),
    }
}

fn verification_error(error: SparseRrefError) -> MatrixError {
    match error {
        SparseRrefError::Invalid(message) => MatrixError::InvalidCertificate(message.into()),
        SparseRrefError::Limit(message) => limit(message),
        SparseRrefError::Meter(error) => stopped(error),
        SparseRrefError::Arithmetic(error) => arithmetic_error(error),
    }
}

fn limit(message: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("sparse row reduction: {message}"))
}

fn invalid(message: &str) -> MatrixError {
    MatrixError::InvalidCertificate(message.into())
}

fn check_value(value: &BigRational) -> Result<(), MatrixError> {
    if value.numer().bits().max(value.denom().bits()) > MAX_SPARSE_RREF_COEFFICIENT_BITS {
        return Err(limit("coefficient exceeds the bit limit"));
    }
    if !value.denom().is_positive() {
        return Err(invalid("rational denominator must be positive"));
    }
    Ok(())
}

struct Work<'a, M> {
    meter: &'a mut M,
    steps: u64,
    bytes: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            steps: 0,
            bytes: 0,
        }
    }

    fn tick(&mut self, amount: u64) -> Result<(), MatrixError> {
        self.checkpoint().map_err(stopped)?;
        if amount != 0 {
            self.charge(Dimension::ComputeSteps, amount)
                .map_err(stopped)?;
        }
        Ok(())
    }

    fn reserve(&mut self, bytes: usize) -> Result<(), MatrixError> {
        self.checkpoint().map_err(stopped)?;
        if bytes != 0 {
            self.charge_batch(&[
                (Dimension::MemoryBytes, bytes as u64),
                (Dimension::AllocationCount, 1),
            ])
            .map_err(stopped)?;
        }
        Ok(())
    }

    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>, MatrixError> {
        self.reserve(
            count
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| limit("vector storage size overflow"))?,
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| limit("vector allocation refused"))?;
        Ok(values)
    }

    fn clone_value(&mut self, value: &BigRational) -> Result<BigRational, MatrixError> {
        check_value(value)?;
        let limbs = (value.numer().limb_count() + value.denom().limb_count()).max(1);
        self.tick(limbs)?;
        self.reserve(limbs as usize * 8)?;
        Ok(value.clone())
    }

    fn multiply(
        &mut self,
        left: &BigRational,
        right: &BigRational,
    ) -> Result<BigRational, MatrixError> {
        self.tick(1)?;
        if left.numer().bits().saturating_add(right.numer().bits())
            > MAX_SPARSE_RREF_COEFFICIENT_BITS
            || left.denom().bits().saturating_add(right.denom().bits())
                > MAX_SPARSE_RREF_COEFFICIENT_BITS
        {
            return Err(limit("rational product intermediate exceeds bit limit"));
        }
        let result = left.metered_mul(right, self).map_err(arithmetic_error)?;
        check_value(&result)?;
        Ok(result)
    }

    fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational, MatrixError> {
        self.tick(1)?;
        let numerator_bits = left
            .numer()
            .bits()
            .saturating_add(right.denom().bits())
            .max(right.numer().bits().saturating_add(left.denom().bits()))
            .saturating_add(1);
        if numerator_bits > MAX_SPARSE_RREF_COEFFICIENT_BITS
            || left.denom().bits().saturating_add(right.denom().bits())
                > MAX_SPARSE_RREF_COEFFICIENT_BITS
        {
            return Err(limit("rational sum intermediate exceeds bit limit"));
        }
        let result = left.metered_add(right, self).map_err(arithmetic_error)?;
        check_value(&result)?;
        Ok(result)
    }

    fn inverse(&mut self, value: &BigRational) -> Result<BigRational, MatrixError> {
        self.tick(1)?;
        let result = BigRational::one()
            .metered_div(value, self)
            .map_err(arithmetic_error)?;
        check_value(&result)?;
        Ok(result)
    }

    fn negate(&mut self, value: &BigRational) -> Result<BigRational, MatrixError> {
        let result = self.clone_value(value)?;
        Ok(-result)
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let (mut steps, mut bytes) = (self.steps, self.bytes);
        for &(dimension, amount) in charges {
            let (counter, maximum) = match dimension {
                Dimension::ComputeSteps => (&mut steps, MAX_WORK),
                Dimension::MemoryBytes => (&mut bytes, MAX_ALLOCATION_BYTES),
                _ => continue,
            };
            let remaining = maximum.saturating_sub(*counter);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *counter += amount;
        }
        self.meter.charge_batch(charges)?;
        self.steps = steps;
        self.bytes = bytes;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

fn source_snapshot<M: BudgetMeter>(
    matrix: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<RationalSparseMatrix, MatrixError> {
    work.tick(1)?;
    if matrix.rows() > MAX_SPARSE_RREF_DIMENSION || matrix.cols() > MAX_SPARSE_RREF_DIMENSION {
        return Err(limit("matrix dimension exceeds 8192"));
    }
    if matrix.nnz() > MAX_SPARSE_RREF_ENTRIES {
        return Err(limit("stored-entry limit exceeded"));
    }
    // Inspect every original entry before any zero, empty, or rank shortcut.
    for (&(row, column), value) in matrix.entries() {
        work.tick(1)?;
        if row >= matrix.rows() || column >= matrix.cols() {
            return Err(invalid("source coordinate lies outside its shape"));
        }
        match value {
            Expr::Integer(value) if value.bits() <= MAX_SPARSE_RREF_COEFFICIENT_BITS => {}
            Expr::Integer(_) => return Err(limit("integer input exceeds coefficient bit limit")),
            Expr::Rational(value) => check_value(value)?,
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        }
    }
    let mut entries = work.vector(matrix.nnz())?;
    for (&(row, column), entry) in matrix.entries() {
        work.tick(1)?;
        let value = match entry {
            Expr::Integer(value) => {
                work.tick(value.limb_count().max(1))?;
                work.reserve((value.limb_count().max(1) as usize + 1) * 8)?;
                BigRational::from_integer(value.clone())
            }
            Expr::Rational(value) => work.clone_value(value)?,
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        };
        if !value.is_zero() {
            entries.push(SparseRationalEntry { row, column, value });
        }
    }
    Ok(RationalSparseMatrix {
        rows: matrix.rows(),
        columns: matrix.cols(),
        entries,
    })
}

fn append_operation<M: BudgetMeter>(
    operations: &mut Vec<LinearRowOperation>,
    operation: LinearRowOperation,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    work.tick(1)?;
    if operations.len() >= MAX_SPARSE_RREF_OPERATIONS {
        return Err(limit("row-operation witness limit exceeded"));
    }
    work.reserve(2 * std::mem::size_of::<LinearRowOperation>())?;
    operations
        .try_reserve(1)
        .map_err(|_| limit("row-operation witness allocation refused"))?;
    operations.push(operation);
    Ok(())
}

fn generate<M: BudgetMeter>(
    source: RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseRrefCandidate, MatrixError> {
    let mut rows: Vec<Row> = work.vector(source.rows)?;
    rows.resize_with(source.rows, Row::new);
    let mut columns: Vec<BTreeSet<usize>> = work.vector(source.columns)?;
    columns.resize_with(source.columns, BTreeSet::new);
    for entry in &source.entries {
        work.tick(1)?;
        work.reserve(384)?;
        rows[entry.row].insert(entry.column, work.clone_value(&entry.value)?);
        columns[entry.column].insert(entry.row);
    }
    let mut fill = source.entries.len();
    let mut operations = Vec::new();
    let mut pivots = work.vector(source.rows.min(source.columns))?;
    let mut pivot_row = 0;
    for pivot_column in 0..source.columns {
        work.tick(1)?;
        if pivot_row == source.rows {
            break;
        }
        let Some(selected) = columns[pivot_column].range(pivot_row..).next().copied() else {
            continue;
        };
        if selected != pivot_row {
            for &column in rows[pivot_row].keys().chain(rows[selected].keys()) {
                work.tick(1)?;
                columns[column].remove(&pivot_row);
                columns[column].remove(&selected);
            }
            rows.swap(pivot_row, selected);
            for &column in rows[pivot_row].keys() {
                work.tick(1)?;
                work.reserve(128)?;
                columns[column].insert(pivot_row);
            }
            for &column in rows[selected].keys() {
                work.tick(1)?;
                work.reserve(128)?;
                columns[column].insert(selected);
            }
            append_operation(
                &mut operations,
                LinearRowOperation::Swap {
                    left: pivot_row,
                    right: selected,
                },
                work,
            )?;
        }
        let pivot = rows[pivot_row]
            .get(&pivot_column)
            .ok_or_else(|| invalid("generator column incidence lost its pivot"))?;
        if !pivot.is_one() {
            let factor = work.inverse(pivot)?;
            for value in rows[pivot_row].values_mut() {
                *value = work.multiply(value, &factor)?;
            }
            append_operation(
                &mut operations,
                LinearRowOperation::Scale {
                    row: pivot_row,
                    factor,
                },
                work,
            )?;
        }
        let mut pivot_entries = work.vector(rows[pivot_row].len())?;
        for (&column, value) in &rows[pivot_row] {
            pivot_entries.push((column, work.clone_value(value)?));
        }
        let mut targets = work.vector(columns[pivot_column].len())?;
        for &row in &columns[pivot_column] {
            work.tick(1)?;
            if row != pivot_row {
                targets.push(row);
            }
        }
        for target in targets {
            work.tick(1)?;
            let factor = work.negate(
                rows[target]
                    .get(&pivot_column)
                    .ok_or_else(|| invalid("generator elimination incidence is stale"))?,
            )?;
            for (column, value) in &pivot_entries {
                let product = work.multiply(&factor, value)?;
                let updated = match rows[target].get(column) {
                    Some(previous) => work.add(previous, &product)?,
                    None => product,
                };
                work.tick(1)?;
                if updated.is_zero() {
                    if rows[target].remove(column).is_some() {
                        columns[*column].remove(&target);
                        fill -= 1;
                    }
                } else {
                    if !rows[target].contains_key(column) {
                        if fill == MAX_SPARSE_RREF_ENTRIES {
                            return Err(limit("intermediate sparse fill limit exceeded"));
                        }
                        work.reserve(384)?;
                        columns[*column].insert(target);
                        fill += 1;
                    }
                    rows[target].insert(*column, updated);
                }
            }
            append_operation(
                &mut operations,
                LinearRowOperation::Add {
                    target,
                    source: pivot_row,
                    factor,
                },
                work,
            )?;
        }
        pivots.push(pivot_column);
        pivot_row += 1;
    }
    let mut entries = work.vector(fill)?;
    for (row, values) in rows.into_iter().enumerate() {
        for (column, value) in values {
            work.tick(1)?;
            entries.push(SparseRationalEntry { row, column, value });
        }
    }
    let reduced = RationalSparseMatrix {
        rows: source.rows,
        columns: source.columns,
        entries,
    };
    work.tick(1)?;
    Ok(SparseRrefCandidate {
        source,
        witness: SparseRrefWitness {
            schema_version: SPARSE_RREF_SCHEMA_VERSION,
            operations,
            reduced,
            pivots,
        },
    })
}

fn accept<M: BudgetMeter>(
    candidate: SparseRrefCandidate,
    work: &mut Work<'_, M>,
) -> Result<VerifiedSparseRref, MatrixError> {
    let receipt = verify_sparse_rref(&candidate.source, &candidate.witness, work)
        .map_err(verification_error)?;
    work.tick(1)?;
    Ok(VerifiedSparseRref {
        source: candidate.source,
        witness: candidate.witness,
        receipt,
    })
}

fn checked<M: BudgetMeter>(
    matrix: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<VerifiedSparseRref, MatrixError> {
    let source = source_snapshot(matrix, work)?;
    let candidate = generate(source, work)?;
    accept(candidate, work)
}

/// Generate a bounded sparse proposal without promoting it to accepted state.
pub fn generate_sparse_rref_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    meter: &mut M,
) -> Result<SparseRrefCandidate, MatrixError> {
    let mut work = Work::new(meter);
    let source = source_snapshot(matrix, &mut work)?;
    generate(source, &mut work)
}

/// Bind a proposal to the actual original matrix and independently accept it.
pub fn verify_sparse_rref_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    candidate: SparseRrefCandidate,
    meter: &mut M,
) -> Result<VerifiedSparseRref, MatrixError> {
    let mut work = Work::new(meter);
    let original = source_snapshot(matrix, &mut work)?;
    if original.rows != candidate.source.rows
        || original.columns != candidate.source.columns
        || original.entries.len() != candidate.source.entries.len()
    {
        return Err(invalid(
            "row-reduction source does not match the original matrix",
        ));
    }
    for (left, right) in original.entries.iter().zip(&candidate.source.entries) {
        work.tick(1 + left.value.numer().limb_count() + left.value.denom().limb_count())?;
        if left != right {
            return Err(invalid(
                "row-reduction source does not match the original matrix",
            ));
        }
    }
    accept(candidate, &mut work)
}

fn lift<M: BudgetMeter>(
    rational: &RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseMatrix, MatrixError> {
    let mut entries = BTreeMap::new();
    for entry in &rational.entries {
        work.tick(1)?;
        // SparseMatrix::new owns a second canonical map; include both maps.
        work.reserve(512)?;
        let value = if entry.value.is_integer() {
            let numerator = entry.value.numer();
            work.tick(numerator.limb_count().max(1))?;
            work.reserve(numerator.limb_count().max(1) as usize * 8)?;
            Expr::Integer(numerator.clone())
        } else {
            Expr::Rational(work.clone_value(&entry.value)?)
        };
        entries.insert((entry.row, entry.column), value);
    }
    let matrix = SparseMatrix::new(rational.rows, rational.columns, entries)?;
    work.tick(1)?;
    Ok(matrix)
}

fn nullspace<M: BudgetMeter>(
    reduced: &VerifiedSparseRref,
    work: &mut Work<'_, M>,
) -> Result<Vec<SparseMatrix>, MatrixError> {
    let columns = reduced.source.columns;
    let mut pivot_rows: Vec<Option<usize>> = work.vector(columns)?;
    pivot_rows.resize(columns, None);
    for (row, &column) in reduced.pivots().iter().enumerate() {
        work.tick(1)?;
        pivot_rows[column] = Some(row);
    }
    let mut free = work.vector(columns - reduced.rank())?;
    let mut free_indices: Vec<Option<usize>> = work.vector(columns)?;
    free_indices.resize(columns, None);
    for column in 0..columns {
        work.tick(1)?;
        if pivot_rows[column].is_none() {
            free_indices[column] = Some(free.len());
            free.push(column);
        }
    }
    let mut output_entries = free.len();
    for entry in &reduced.reduced().entries {
        work.tick(1)?;
        if free_indices[entry.column].is_some() {
            output_entries = output_entries
                .checked_add(1)
                .ok_or_else(|| limit("nullspace output size overflow"))?;
            if output_entries > MAX_SPARSE_RREF_ENTRIES {
                return Err(limit("aggregate nullspace stored-entry limit exceeded"));
            }
        }
    }
    let mut basis: Vec<Row> = work.vector(free.len())?;
    basis.resize_with(free.len(), Row::new);
    for (index, &column) in free.iter().enumerate() {
        work.tick(1)?;
        work.reserve(256)?;
        basis[index].insert(column, BigRational::one());
    }
    for entry in &reduced.reduced().entries {
        work.tick(1)?;
        if let Some(index) = free_indices[entry.column] {
            let pivot_column = reduced.pivots()[entry.row];
            work.reserve(256)?;
            basis[index].insert(pivot_column, work.negate(&entry.value)?);
        }
    }

    // Check original A*N=0 without scanning any implicit zero. Together with
    // the independently checked rank and free-coordinate identity, these
    // independent residuals prove that the entire kernel is spanned.
    let mut source_columns: Vec<Vec<(usize, &BigRational)>> = work.vector(columns)?;
    source_columns.resize_with(columns, Vec::new);
    for entry in &reduced.source.entries {
        work.tick(1)?;
        work.reserve(2 * std::mem::size_of::<(usize, &BigRational)>())?;
        source_columns[entry.column]
            .try_reserve(1)
            .map_err(|_| limit("nullspace source-column allocation refused"))?;
        source_columns[entry.column].push((entry.row, &entry.value));
    }
    for (index, vector) in basis.iter().enumerate() {
        work.tick(1)?;
        if vector.get(&free[index]) != Some(&BigRational::one()) {
            return Err(invalid("nullspace free coordinate is not one"));
        }
        let mut residual: Row = BTreeMap::new();
        for (&column, value) in vector {
            work.tick(1)?;
            if value.is_zero()
                || column >= columns
                || (pivot_rows[column].is_none() && column != free[index])
            {
                return Err(invalid(
                    "nullspace vector violates its free-coordinate identity",
                ));
            }
            for &(row, coefficient) in &source_columns[column] {
                let product = work.multiply(coefficient, value)?;
                let updated = match residual.get(&row) {
                    Some(previous) => work.add(previous, &product)?,
                    None => product,
                };
                work.tick(1)?;
                if updated.is_zero() {
                    residual.remove(&row);
                } else {
                    if !residual.contains_key(&row) {
                        work.reserve(256)?;
                    }
                    residual.insert(row, updated);
                }
            }
        }
        if !residual.is_empty() {
            return Err(invalid(
                "nullspace vector has a nonzero original-source residual",
            ));
        }
    }
    let mut result = work.vector(basis.len())?;
    for vector in basis {
        let mut entries = work.vector(vector.len())?;
        for (row, value) in vector {
            work.tick(1)?;
            entries.push(SparseRationalEntry {
                row,
                column: 0,
                value,
            });
        }
        result.push(lift(
            &RationalSparseMatrix {
                rows: columns,
                columns: 1,
                entries,
            },
            work,
        )?);
    }
    work.tick(1)?;
    Ok(result)
}

fn augmented_source<M: BudgetMeter>(
    coefficients: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<RationalSparseMatrix, MatrixError> {
    let columns = coefficients
        .columns
        .checked_add(rhs.columns)
        .ok_or_else(|| limit("augmented column count overflow"))?;
    if columns > MAX_SPARSE_RREF_DIMENSION {
        return Err(limit("augmented matrix dimension exceeds 8192"));
    }
    let count = coefficients
        .entries
        .len()
        .checked_add(rhs.entries.len())
        .ok_or_else(|| limit("augmented stored-entry count overflow"))?;
    if count > MAX_SPARSE_RREF_ENTRIES {
        return Err(limit("augmented stored-entry limit exceeded"));
    }
    let mut entries = work.vector(count)?;
    let (mut left, mut right) = (0, 0);
    while left < coefficients.entries.len() || right < rhs.entries.len() {
        work.tick(1)?;
        // Within a row every A column precedes every shifted B column. Merging
        // the already canonical sources avoids a dense traversal or a sort.
        let from_left = right == rhs.entries.len()
            || (left < coefficients.entries.len()
                && coefficients.entries[left].row <= rhs.entries[right].row);
        let (entry, offset) = if from_left {
            let entry = &coefficients.entries[left];
            left += 1;
            (entry, 0)
        } else {
            let entry = &rhs.entries[right];
            right += 1;
            (entry, coefficients.columns)
        };
        entries.push(SparseRationalEntry {
            row: entry.row,
            column: entry.column + offset,
            value: work.clone_value(&entry.value)?,
        });
    }
    Ok(RationalSparseMatrix {
        rows: coefficients.rows,
        columns,
        entries,
    })
}

fn sparse_columns<'a, M: BudgetMeter>(
    matrix: &'a RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<Vec<Vec<(usize, &'a BigRational)>>, MatrixError> {
    let mut columns: Vec<Vec<(usize, &BigRational)>> = work.vector(matrix.columns)?;
    columns.resize_with(matrix.columns, Vec::new);
    for entry in &matrix.entries {
        work.tick(1)?;
        work.reserve(2 * std::mem::size_of::<(usize, &BigRational)>())?;
        columns[entry.column]
            .try_reserve(1)
            .map_err(|_| limit("sparse product column allocation refused"))?;
        columns[entry.column].push((entry.row, &entry.value));
    }
    Ok(columns)
}

/// Independent original-source residual check. This does not perform row
/// reduction or use the reduced right-hand side as the expected answer.
/// Uniqueness comes separately from the proof-kernel-checked A pivot coverage.
fn check_rectangular_residual<M: BudgetMeter>(
    coefficients: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    solution: &RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    work.tick(1)?;
    if solution.rows != coefficients.columns
        || solution.columns != rhs.columns
        || coefficients.rows != rhs.rows
    {
        return Err(invalid(
            "sparse solution shape differs from the original system",
        ));
    }
    if solution.entries.len() > MAX_SPARSE_RREF_ENTRIES {
        return Err(limit("sparse solution stored-entry limit exceeded"));
    }
    let mut previous = None;
    for entry in &solution.entries {
        work.tick(1)?;
        let coordinate = (entry.row, entry.column);
        if entry.row >= solution.rows
            || entry.column >= solution.columns
            || previous.is_some_and(|prior| prior >= coordinate)
            || entry.value.is_zero()
        {
            return Err(invalid("sparse solution is not a canonical matrix"));
        }
        check_value(&entry.value)?;
        previous = Some(coordinate);
    }
    let coefficient_columns = sparse_columns(coefficients, work)?;
    let rhs_columns = sparse_columns(rhs, work)?;
    let solution_columns = sparse_columns(solution, work)?;
    for column in 0..rhs.columns {
        work.tick(1)?;
        let mut residual = Row::new();
        for &(row, value) in &rhs_columns[column] {
            work.tick(1)?;
            work.reserve(256)?;
            residual.insert(row, work.negate(value)?);
        }
        for &(variable, value) in &solution_columns[column] {
            work.tick(1)?;
            for &(row, coefficient) in &coefficient_columns[variable] {
                let product = work.multiply(coefficient, value)?;
                let updated = match residual.get(&row) {
                    Some(previous) => work.add(previous, &product)?,
                    None => product,
                };
                work.tick(1)?;
                if updated.is_zero() {
                    residual.remove(&row);
                } else {
                    if !residual.contains_key(&row) {
                        work.reserve(256)?;
                    }
                    residual.insert(row, updated);
                }
            }
        }
        if !residual.is_empty() {
            return Err(invalid(
                "sparse solution has a nonzero original-source AX-B residual",
            ));
        }
    }
    work.tick(1)?;
    Ok(())
}

/// Internal route selected before LU for nonsquare requests. Every admitted
/// right-hand side shares one immutable augmented source and one cumulative
/// meter through proof replay, independent AX=B checking, and publication.
pub(crate) fn solve_rectangular<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    meter: &mut M,
) -> Result<SparseMatrix, MatrixError> {
    let mut work = Work::new(meter);
    work.tick(1)?;
    if matrix.rows() != rhs.rows() {
        return Err(MatrixError::ShapeMismatch(
            matrix.rows(),
            matrix.cols(),
            rhs.rows(),
            rhs.cols(),
        ));
    }
    let coefficients = source_snapshot(matrix, &mut work)?;
    let original_rhs = source_snapshot(rhs, &mut work)?;
    if coefficients.rows < coefficients.columns {
        // No full-column-rank matrix exists in this shape. The sparse API does
        // not select an arbitrary representative of an affine solution family.
        return Err(MatrixError::SingularMatrix);
    }
    let source = augmented_source(&coefficients, &original_rhs, &mut work)?;
    let candidate = generate(source, &mut work)?;
    let verified = accept(candidate, &mut work)?;
    for &column in verified.pivots() {
        work.tick(1)?;
        if column >= coefficients.columns {
            // A pivot in B is a checked row with zero A and nonzero B, even if
            // other right-hand sides happen to be individually consistent.
            return Err(MatrixError::InconsistentLinearSystem);
        }
    }
    if verified.rank() != coefficients.columns {
        // Require uniqueness even for B with no columns: a vacuous residual
        // is not evidence that the coefficient matrix has full column rank.
        return Err(MatrixError::SingularMatrix);
    }
    let mut entries = work.vector(verified.reduced().entries.len())?;
    for entry in &verified.reduced().entries {
        work.tick(1)?;
        if entry.column >= coefficients.columns {
            let variable = *verified.pivots().get(entry.row).ok_or_else(|| {
                invalid("reduced solution row has no checked coefficient pivot")
            })?;
            entries.push(SparseRationalEntry {
                row: variable,
                column: entry.column - coefficients.columns,
                value: work.clone_value(&entry.value)?,
            });
        }
    }
    let solution = RationalSparseMatrix {
        rows: coefficients.columns,
        columns: original_rhs.columns,
        entries,
    };
    check_rectangular_residual(&coefficients, &original_rhs, &solution, &mut work)?;
    lift(&solution, &mut work)
}

impl SparseMatrix {
    /// Independently checked sparse RREF with separately owned phase meters.
    pub fn rref_checked_with_meters<G: BudgetMeter, V: BudgetMeter>(
        &self,
        generator: &mut G,
        verifier: &mut V,
    ) -> Result<VerifiedSparseRref, MatrixError> {
        let candidate = generate_sparse_rref_candidate(self, generator)?;
        verify_sparse_rref_candidate(self, candidate, verifier)
    }

    /// Checked sparse row reduction with one cumulative caller meter.
    pub fn metered_rref<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<(Self, Vec<usize>), MatrixError> {
        let mut work = Work::new(meter);
        let verified = checked(self, &mut work)?;
        let matrix = lift(verified.reduced(), &mut work)?;
        let mut pivots = work.vector(verified.rank())?;
        for &pivot in verified.pivots() {
            work.tick(1)?;
            pivots.push(pivot);
        }
        work.tick(1)?;
        Ok((matrix, pivots))
    }

    /// Exact rational RREF and increasing pivot columns, retaining sparse storage.
    pub fn rref(&self) -> Result<(Self, Vec<usize>), MatrixError> {
        self.metered_rref(&mut Unbounded)
    }

    /// Independently checked rank without materializing a dense matrix.
    pub fn metered_rank<M: BudgetMeter>(&self, meter: &mut M) -> Result<usize, MatrixError> {
        let mut work = Work::new(meter);
        let verified = checked(self, &mut work)?;
        work.tick(1)?;
        Ok(verified.rank())
    }

    pub fn rank(&self) -> Result<usize, MatrixError> {
        self.metered_rank(&mut Unbounded)
    }

    /// Complete exact nullspace with sparse vectors in ascending free-column order.
    /// Original-source residuals and free-coordinate identities are checked before
    /// publication. Aggregate stored output and all transient work remain bounded.
    pub fn metered_nullspace<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<Vec<Self>, MatrixError> {
        let mut work = Work::new(meter);
        let verified = checked(self, &mut work)?;
        nullspace(&verified, &mut work)
    }

    pub fn nullspace(&self) -> Result<Vec<Self>, MatrixError> {
        self.metered_nullspace(&mut Unbounded)
    }

    /// Original source columns selected by independently checked RREF pivots.
    pub fn metered_columnspace<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<Vec<Self>, MatrixError> {
        let mut work = Work::new(meter);
        let verified = checked(self, &mut work)?;
        let mut selected: Vec<Option<usize>> = work.vector(self.cols())?;
        selected.resize(self.cols(), None);
        let mut columns: Vec<Vec<SparseRationalEntry>> = work.vector(verified.rank())?;
        columns.resize_with(verified.rank(), Vec::new);
        for (index, &column) in verified.pivots().iter().enumerate() {
            work.tick(1)?;
            selected[column] = Some(index);
        }
        for entry in &verified.source.entries {
            work.tick(1)?;
            if let Some(index) = selected[entry.column] {
                work.reserve(2 * std::mem::size_of::<SparseRationalEntry>())?;
                columns[index]
                    .try_reserve(1)
                    .map_err(|_| limit("columnspace allocation refused"))?;
                columns[index].push(SparseRationalEntry {
                    row: entry.row,
                    column: 0,
                    value: work.clone_value(&entry.value)?,
                });
            }
        }
        let mut basis = work.vector(columns.len())?;
        for entries in columns {
            basis.push(lift(
                &RationalSparseMatrix {
                    rows: self.rows(),
                    columns: 1,
                    entries,
                },
                &mut work,
            )?);
        }
        work.tick(1)?;
        Ok(basis)
    }

    pub fn columnspace(&self) -> Result<Vec<Self>, MatrixError> {
        self.metered_columnspace(&mut Unbounded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::BigInt;

    fn rational_matrix(rows: usize, columns: usize, values: &[i64]) -> RationalSparseMatrix {
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
                    value: BigRational::from_integer(BigInt::from(value)),
                })
                .collect(),
        }
    }

    #[test]
    fn rectangular_residual_reference_rejects_wrong_and_nearly_right_solutions() {
        let coefficients = rational_matrix(3, 2, &[1, 0, 0, 1, 1, 1]);
        let rhs = rational_matrix(3, 2, &[2, 7, 3, 8, 5, 15]);
        let correct = rational_matrix(2, 2, &[2, 7, 3, 8]);
        check_rectangular_residual(
            &coefficients,
            &rhs,
            &correct,
            &mut Work::new(&mut Unbounded),
        )
        .unwrap();
        let mut mutations = Vec::new();
        let mut changed = correct.clone();
        changed.entries[3].value = BigRational::from_integer(BigInt::from(9));
        mutations.push(changed);
        let mut omitted = correct.clone();
        omitted.entries.pop();
        mutations.push(omitted);
        let mut almost = correct.clone();
        almost.entries[0].value += BigRational::new(BigInt::from(1), BigInt::from(10).pow(80));
        mutations.push(almost);
        let mut duplicate = correct.clone();
        duplicate.entries.insert(1, duplicate.entries[0].clone());
        mutations.push(duplicate);
        let mut wrong_shape = correct.clone();
        wrong_shape.columns = 1;
        mutations.push(wrong_shape);
        for candidate in mutations {
            assert!(matches!(
                check_rectangular_residual(
                    &coefficients,
                    &rhs,
                    &candidate,
                    &mut Work::new(&mut Unbounded)
                ),
                Err(MatrixError::InvalidCertificate(_))
            ));
        }
    }

    #[test]
    fn rectangular_residual_reference_binds_every_original_rhs_row() {
        let coefficients = rational_matrix(3, 2, &[1, 0, 0, 1, 1, 1]);
        let candidate = rational_matrix(2, 2, &[2, 7, 3, 8]);
        // Keeping the first two equations and all but one RHS entry identical
        // cannot bypass the check of the final overdetermined source row.
        let inconsistent_rhs = rational_matrix(3, 2, &[2, 7, 3, 8, 5, 16]);
        assert!(matches!(
            check_rectangular_residual(
                &coefficients,
                &inconsistent_rhs,
                &candidate,
                &mut Work::new(&mut Unbounded)
            ),
            Err(MatrixError::InvalidCertificate(_))
        ));
        let no_unknowns = rational_matrix(3, 0, &[]);
        let nonzero_rhs = rational_matrix(3, 1, &[0, 0, 1]);
        let empty = rational_matrix(0, 1, &[]);
        assert!(matches!(
            check_rectangular_residual(
                &no_unknowns,
                &nonzero_rhs,
                &empty,
                &mut Work::new(&mut Unbounded)
            ),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
}
