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
use fsym_core::{BigRational, Expr, Symbol};
use fsym_proof_kernel::linear_system::LinearRowOperation;
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION, MAX_SPARSE_RREF_ENTRIES,
    MAX_SPARSE_RREF_OPERATIONS, RationalSparseMatrix, SPARSE_AFFINE_SCHEMA_VERSION,
    SPARSE_RREF_SCHEMA_VERSION, SparseAffineOutcome, SparseAffineReceipt, SparseAffineWitness,
    SparseRationalEntry, SparseRrefError, SparseRrefReceipt, SparseRrefWitness,
    verify_sparse_affine, verify_sparse_rref,
};
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

const MAX_WORK: u64 = 20_000_000;
const MAX_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;
const MAX_AFFINE_PARAMETER_TERMS: usize = 1_000_000;
/// Maximum UTF-8 byte length of one caller-owned affine coordinate name.
pub const MAX_SPARSE_AFFINE_PARAMETER_NAME_BYTES: usize = 4_096;
/// Maximum total UTF-8 name bytes in the original ordered coordinate list.
pub const MAX_SPARSE_AFFINE_PARAMETER_TOTAL_BYTES: usize = 1_048_576;
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

/// Untrusted complete-family proposal for the original sparse system `A X = B`.
/// Every right-hand side shares one augmented row-reduction witness and kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseAffineSolutionCandidate {
    pub coefficients: RationalSparseMatrix,
    pub rhs: RationalSparseMatrix,
    pub witness: SparseAffineWitness,
}

/// Complete sparse affine family accepted against both original matrices.
/// The proof kernel checks consistency, coefficient rank, the complete kernel,
/// and every original-source residual before these sparse parts are lifted.
/// This type has no public unchecked constructor or deserializer.
#[derive(Debug)]
pub struct VerifiedSparseAffineSolution {
    coefficients: RationalSparseMatrix,
    rhs: RationalSparseMatrix,
    witness: SparseAffineWitness,
    receipt: SparseAffineReceipt,
    parts: Option<(SparseMatrix, SparseMatrix, Vec<usize>)>,
}

impl VerifiedSparseAffineSolution {
    pub fn coefficients(&self) -> &RationalSparseMatrix {
        &self.coefficients
    }

    pub fn rhs(&self) -> &RationalSparseMatrix {
        &self.rhs
    }

    pub fn witness(&self) -> &SparseAffineWitness {
        &self.witness
    }

    pub fn receipt(&self) -> &SparseAffineReceipt {
        &self.receipt
    }

    pub fn rank(&self) -> usize {
        self.receipt.rank()
    }

    /// Dimension of the coefficient kernel, independently of the RHS count.
    pub fn nullity(&self) -> usize {
        self.receipt.free_columns().len()
    }

    pub fn is_consistent(&self) -> bool {
        self.receipt.is_consistent()
    }

    /// Whether the whole matrix equation has exactly one solution. In
    /// particular, the sole `n x 0` matrix is unique even when A has a kernel.
    pub fn is_unique(&self) -> bool {
        self.is_consistent() && (self.rhs.columns == 0 || self.nullity() == 0)
    }

    pub fn particular(&self) -> Option<&SparseMatrix> {
        self.parts.as_ref().map(|(particular, _, _)| particular)
    }

    pub fn kernel(&self) -> Option<&SparseMatrix> {
        self.parts.as_ref().map(|(_, kernel, _)| kernel)
    }

    pub fn free_columns(&self) -> &[usize] {
        self.receipt.free_columns()
    }

    /// Move the already checked and metered `(P, N, free_columns)` out without
    /// cloning. Every solution is `P + N T`, where T has one row per free
    /// column and one column per original RHS. `None` is checked inconsistency.
    pub fn into_parts(self) -> Option<(SparseMatrix, SparseMatrix, Vec<usize>)> {
        self.parts
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
        let value = rational_expression(&entry.value, work)?;
        entries.insert((entry.row, entry.column), value);
    }
    let matrix = SparseMatrix::new(rational.rows, rational.columns, entries)?;
    work.tick(1)?;
    Ok(matrix)
}

fn rational_expression<M: BudgetMeter>(
    value: &BigRational,
    work: &mut Work<'_, M>,
) -> Result<Expr, MatrixError> {
    if value.is_integer() {
        let numerator = value.numer();
        work.tick(numerator.limb_count().max(1))?;
        work.reserve(numerator.limb_count().max(1) as usize * 8)?;
        Ok(Expr::Integer(numerator.clone()))
    } else {
        Ok(Expr::Rational(work.clone_value(value)?))
    }
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
            let variable = *verified
                .pivots()
                .get(entry.row)
                .ok_or_else(|| invalid("reduced solution row has no checked coefficient pivot"))?;
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

fn affine_sources<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<(RationalSparseMatrix, RationalSparseMatrix), MatrixError> {
    work.tick(1)?;
    if matrix.rows() != rhs.rows() {
        return Err(MatrixError::ShapeMismatch(
            matrix.rows(),
            matrix.cols(),
            rhs.rows(),
            rhs.cols(),
        ));
    }
    let width = matrix
        .cols()
        .checked_add(rhs.cols())
        .ok_or_else(|| limit("affine augmented column count overflow"))?;
    if width > MAX_SPARSE_RREF_DIMENSION {
        return Err(limit("affine augmented dimension exceeds 8192"));
    }
    let entries = matrix
        .nnz()
        .checked_add(rhs.nnz())
        .ok_or_else(|| limit("affine augmented stored-entry count overflow"))?;
    if entries > MAX_SPARSE_RREF_ENTRIES {
        return Err(limit("affine augmented stored-entry limit exceeded"));
    }
    // Both whole original domains are admitted even with zero equations,
    // zero RHS columns, or coefficients which cannot have full column rank.
    Ok((source_snapshot(matrix, work)?, source_snapshot(rhs, work)?))
}

fn affine_outcome<M: BudgetMeter>(
    coefficients: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    reduction: &SparseRrefWitness,
    work: &mut Work<'_, M>,
) -> Result<SparseAffineOutcome, MatrixError> {
    let columns = coefficients.columns;
    let mut previous = None;
    for (row, &column) in reduction.pivots.iter().enumerate() {
        work.tick(1)?;
        if previous.is_some_and(|prior| prior >= column) {
            return Err(invalid(
                "affine generator pivots are not strictly increasing",
            ));
        }
        if column >= columns {
            return Ok(SparseAffineOutcome::Inconsistent { row });
        }
        previous = Some(column);
    }
    let nullity = columns
        .checked_sub(reduction.pivots.len())
        .ok_or_else(|| invalid("affine generator rank exceeds its variable count"))?;
    let mut free_columns = work.vector(nullity)?;
    let mut free_indices: Vec<Option<usize>> = work.vector(columns)?;
    free_indices.resize(columns, None);
    let mut pivot_row = 0;
    for (column, index) in free_indices.iter_mut().enumerate() {
        work.tick(1)?;
        if reduction.pivots.get(pivot_row) == Some(&column) {
            pivot_row += 1;
        } else {
            *index = Some(free_columns.len());
            free_columns.push(column);
        }
    }

    // Bound total stored output before allocating either part. A sparse input
    // may still have a large complete kernel; that fill is never implicit.
    let mut particular_count = 0usize;
    let mut kernel_count = nullity;
    for entry in &reduction.reduced.entries {
        work.tick(1)?;
        let counter = if entry.column >= columns {
            &mut particular_count
        } else if free_indices[entry.column].is_some() {
            &mut kernel_count
        } else {
            continue;
        };
        *counter = counter
            .checked_add(1)
            .ok_or_else(|| limit("affine output stored-entry count overflow"))?;
        if particular_count
            .checked_add(kernel_count)
            .is_none_or(|count| count > MAX_SPARSE_RREF_ENTRIES)
        {
            return Err(limit("aggregate affine output stored-entry limit exceeded"));
        }
    }
    let mut particular_entries = work.vector(particular_count)?;
    let mut kernel_entries = BTreeMap::new();
    work.reserve(16)?;
    let one = BigRational::one();
    for (index, &column) in free_columns.iter().enumerate() {
        work.tick(1)?;
        work.reserve(256)?;
        kernel_entries.insert((column, index), work.clone_value(&one)?);
    }
    for entry in &reduction.reduced.entries {
        work.tick(1)?;
        if entry.column >= columns {
            let variable = *reduction
                .pivots
                .get(entry.row)
                .ok_or_else(|| invalid("affine RHS row has no coefficient pivot"))?;
            particular_entries.push(SparseRationalEntry {
                row: variable,
                column: entry.column - columns,
                value: work.clone_value(&entry.value)?,
            });
        } else if let Some(index) = free_indices[entry.column] {
            let variable = *reduction
                .pivots
                .get(entry.row)
                .ok_or_else(|| invalid("affine kernel row has no coefficient pivot"))?;
            work.reserve(256)?;
            kernel_entries.insert((variable, index), work.negate(&entry.value)?);
        }
    }
    let mut kernel = work.vector(kernel_count)?;
    for ((row, column), value) in kernel_entries {
        work.tick(1)?;
        kernel.push(SparseRationalEntry { row, column, value });
    }
    Ok(SparseAffineOutcome::Solvable {
        particular: RationalSparseMatrix {
            rows: columns,
            columns: rhs.columns,
            entries: particular_entries,
        },
        kernel: RationalSparseMatrix {
            rows: columns,
            columns: nullity,
            entries: kernel,
        },
        free_columns,
    })
}

fn generate_affine<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseAffineSolutionCandidate, MatrixError> {
    let (coefficients, rhs) = affine_sources(matrix, rhs, work)?;
    generate_affine_sources(coefficients, rhs, work)
}

fn generate_affine_sources<M: BudgetMeter>(
    coefficients: RationalSparseMatrix,
    rhs: RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseAffineSolutionCandidate, MatrixError> {
    let source = augmented_source(&coefficients, &rhs, work)?;
    let reduction = generate(source, work)?.witness;
    let outcome = affine_outcome(&coefficients, &rhs, &reduction, work)?;
    work.tick(1)?;
    Ok(SparseAffineSolutionCandidate {
        coefficients,
        rhs,
        witness: SparseAffineWitness {
            schema_version: SPARSE_AFFINE_SCHEMA_VERSION,
            reduction,
            outcome,
        },
    })
}

fn split_augmented_source<M: BudgetMeter>(
    matrix: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<(RationalSparseMatrix, RationalSparseMatrix), MatrixError> {
    let source = source_snapshot(matrix, work)?;
    let columns = source.columns.checked_sub(1).ok_or(
        MatrixError::InvalidAffineParameters("an augmented system requires a right-hand-side column"),
    )?;
    let mut rhs_count = 0;
    for entry in &source.entries {
        work.tick(1)?;
        if entry.column == columns {
            rhs_count += 1;
        }
    }
    let mut coefficients = work.vector(source.entries.len() - rhs_count)?;
    let mut rhs = work.vector(rhs_count)?;
    // Move the admitted exact cells into their two canonical blocks. No dense
    // traversal, reparsing, or second input generation participates in this split.
    for mut entry in source.entries {
        work.tick(1)?;
        if entry.column == columns {
            entry.column = 0;
            rhs.push(entry);
        } else {
            coefficients.push(entry);
        }
    }
    Ok((
        RationalSparseMatrix {
            rows: source.rows,
            columns,
            entries: coefficients,
        },
        RationalSparseMatrix {
            rows: source.rows,
            columns: 1,
            entries: rhs,
        },
    ))
}

fn bind_affine_source<M: BudgetMeter>(
    original: &RationalSparseMatrix,
    proposed: &RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    work.tick(1)?;
    if original.rows != proposed.rows
        || original.columns != proposed.columns
        || original.entries.len() != proposed.entries.len()
    {
        return Err(invalid(
            "affine candidate source differs from the original matrix",
        ));
    }
    for (expected, actual) in original.entries.iter().zip(&proposed.entries) {
        check_value(&actual.value)?;
        work.tick(
            1 + expected.value.numer().limb_count()
                + expected.value.denom().limb_count()
                + actual.value.numer().limb_count()
                + actual.value.denom().limb_count(),
        )?;
        if expected != actual {
            return Err(invalid(
                "affine candidate source differs from the original matrix",
            ));
        }
    }
    Ok(())
}

fn accept_affine<M: BudgetMeter>(
    candidate: SparseAffineSolutionCandidate,
    work: &mut Work<'_, M>,
) -> Result<VerifiedSparseAffineSolution, MatrixError> {
    let receipt = verify_sparse_affine(
        &candidate.coefficients,
        &candidate.rhs,
        &candidate.witness,
        work,
    )
    .map_err(verification_error)?;
    let parts = match &candidate.witness.outcome {
        SparseAffineOutcome::Inconsistent { .. } => None,
        SparseAffineOutcome::Solvable {
            particular,
            kernel,
            free_columns,
        } => {
            let particular = lift(particular, work)?;
            let kernel = lift(kernel, work)?;
            let mut free = work.vector(free_columns.len())?;
            for &column in free_columns {
                work.tick(1)?;
                free.push(column);
            }
            Some((particular, kernel, free))
        }
    };
    work.tick(1)?;
    Ok(VerifiedSparseAffineSolution {
        coefficients: candidate.coefficients,
        rhs: candidate.rhs,
        witness: candidate.witness,
        receipt,
        parts,
    })
}

/// Bind both original matrices before independent complete-family acceptance.
/// Verification, sparse output lifting, and final publication use one ledger.
pub fn verify_sparse_affine_solution_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    candidate: SparseAffineSolutionCandidate,
    meter: &mut M,
) -> Result<VerifiedSparseAffineSolution, MatrixError> {
    let mut work = Work::new(meter);
    let (coefficients, original_rhs) = affine_sources(matrix, rhs, &mut work)?;
    bind_affine_source(&coefficients, &candidate.coefficients, &mut work)?;
    bind_affine_source(&original_rhs, &candidate.rhs, &mut work)?;
    accept_affine(candidate, &mut work)
}

fn copy_affine_symbol<M: BudgetMeter>(
    symbol: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Symbol, MatrixError> {
    work.tick((symbol.name.len() + std::mem::size_of::<Symbol>()) as u64)?;
    work.reserve(symbol.name.len())?;
    let mut name = String::new();
    name.try_reserve_exact(symbol.name.len())
        .map_err(|_| limit("affine coordinate name allocation refused"))?;
    name.push_str(&symbol.name);
    Ok(Symbol {
        name,
        identity: symbol.identity,
    })
}

fn admit_affine_parameters<M: BudgetMeter>(
    columns: usize,
    variables: Option<&[Expr]>,
    work: &mut Work<'_, M>,
) -> Result<Option<Vec<Symbol>>, MatrixError> {
    work.tick(1)?;
    if columns > MAX_SPARSE_RREF_DIMENSION {
        return Err(limit("affine coordinate count exceeds 8192"));
    }
    let Some(variables) = variables else {
        return Ok(None);
    };
    if variables.len() != columns {
        return Err(MatrixError::InvalidAffineParameters(
            "the coordinate count must equal the number of unknowns",
        ));
    }
    // Admit the entire original list before cloning any name or generating a
    // solution. Even pivot-only, unused, and inconsistent-system coordinates
    // must be exact distinct symbol atoms with bounded original UTF-8 names.
    let mut total_bytes = 0usize;
    for variable in variables {
        work.tick(1)?;
        let Expr::Sym(symbol) = variable else {
            return Err(MatrixError::InvalidAffineParameters(
                "every coordinate must be an exact symbol atom",
            ));
        };
        if symbol.name.len() > MAX_SPARSE_AFFINE_PARAMETER_NAME_BYTES {
            return Err(limit("affine coordinate name exceeds 4096 UTF-8 bytes"));
        }
        total_bytes = total_bytes
            .checked_add(symbol.name.len())
            .ok_or_else(|| limit("affine coordinate name storage overflow"))?;
        if total_bytes > MAX_SPARSE_AFFINE_PARAMETER_TOTAL_BYTES {
            return Err(limit("aggregate affine coordinate names exceed 1 MiB"));
        }
    }
    let levels = u64::from(usize::BITS - variables.len().leading_zeros()) + 1;
    let mut unique = BTreeSet::new();
    let mut coordinates = work.vector(variables.len())?;
    for variable in variables {
        let Expr::Sym(symbol) = variable else {
            return Err(MatrixError::InvalidAffineParameters(
                "every coordinate must be an exact symbol atom",
            ));
        };
        // BTreeSet compares the whole typed Symbol, including None versus
        // Some(identity). Charge name-prefix and fixed identity comparisons;
        // neither printed-name equality nor a digest alone defines a duplicate.
        work.tick(4 * levels * (symbol.name.len() + std::mem::size_of::<Symbol>()) as u64)?;
        work.reserve(128)?;
        if !unique.insert(symbol) {
            return Err(MatrixError::InvalidAffineParameters(
                "duplicate typed affine coordinates",
            ));
        }
        coordinates.push(copy_affine_symbol(symbol, work)?);
    }
    Ok(Some(coordinates))
}

fn affine_parameter<M: BudgetMeter>(
    index: usize,
    work: &mut Work<'_, M>,
) -> Result<Expr, MatrixError> {
    let digits = if index == 0 {
        1
    } else {
        index.ilog10() as usize + 1
    };
    work.tick((digits + 3) as u64)?;
    work.reserve(digits + 3)?;
    let mut name = String::new();
    name.try_reserve_exact(digits + 3)
        .map_err(|_| limit("affine parameter name allocation refused"))?;
    write!(name, "tau{index}").map_err(|_| limit("affine parameter name construction failed"))?;
    Ok(Expr::symbol(name))
}

fn selected_affine_parameter<M: BudgetMeter>(
    index: usize,
    free_columns: &[usize],
    coordinates: Option<&[Symbol]>,
    work: &mut Work<'_, M>,
) -> Result<Expr, MatrixError> {
    let Some(coordinates) = coordinates else {
        return affine_parameter(index, work);
    };
    let column = *free_columns
        .get(index)
        .ok_or_else(|| invalid("named affine parameter has no checked free column"))?;
    let symbol = coordinates
        .get(column)
        .ok_or_else(|| invalid("checked free column has no original typed coordinate"))?;
    Ok(Expr::Sym(copy_affine_symbol(symbol, work)?))
}

fn affine_expression<M: BudgetMeter>(
    mut terms: Vec<Expr>,
    coefficient_bits: u64,
    parameter_bytes: u64,
    work: &mut Work<'_, M>,
) -> Result<Expr, MatrixError> {
    work.tick(1)?;
    if terms.len() == 1 {
        return terms
            .pop()
            .ok_or_else(|| invalid("affine expression lost its only term"));
    }
    if terms.is_empty() {
        return Err(invalid("affine expression has no nonzero terms"));
    }
    // All terms have distinct typed parameter symbols, plus at most one
    // rational constant. No collection, expansion, or numeric arithmetic is
    // needed. Account conservatively for the core canonical comparator's
    // temporary scalar/name renderings, typed identity comparisons and sorting
    // before entering this bounded canonicalization step; output terms have
    // depth at most two. The default fixed tau names fit the existing overhead.
    let levels = u64::from(usize::BITS - (terms.len() - 1).leading_zeros());
    let comparisons = 4 * terms.len() as u64 * levels;
    let limbs = coefficient_bits.max(1).div_ceil(64) + 1;
    work.tick(comparisons * (limbs * limbs + 2 * parameter_bytes))?;
    work.charge_batch(&[
        (
            Dimension::MemoryBytes,
            comparisons * (4 * coefficient_bits.max(1) + 2 * parameter_bytes + 128)
                + (terms.len() * std::mem::size_of::<Expr>()) as u64,
        ),
        (
            Dimension::AllocationCount,
            comparisons * (2 * limbs + 8) + 1,
        ),
    ])
    .map_err(stopped)?;
    fsym_core::canonicalize_add_args(&mut terms);
    work.tick(1)?;
    Ok(Expr::Add(terms))
}

fn render_affine<M: BudgetMeter>(
    verified: VerifiedSparseAffineSolution,
    coordinates: Option<&[Symbol]>,
    work: &mut Work<'_, M>,
) -> Result<Option<(SparseMatrix, SparseMatrix, Vec<usize>)>, MatrixError> {
    let SparseAffineOutcome::Solvable {
        particular,
        kernel,
        free_columns,
    } = &verified.witness.outcome
    else {
        work.tick(1)?;
        return Ok(None);
    };
    let rhs_columns = particular.columns;
    let mut parameter_bytes = 0;
    if let Some(coordinates) = coordinates {
        if rhs_columns != 1 || coordinates.len() != particular.rows {
            return Err(invalid("named affine rendering has incompatible coordinate dimensions"));
        }
        for &column in free_columns {
            work.tick(1)?;
            parameter_bytes = parameter_bytes
                .max((coordinates[column].name.len() + std::mem::size_of::<Symbol>()) as u64);
        }
    }
    let parameter_count = free_columns
        .len()
        .checked_mul(rhs_columns)
        .ok_or_else(|| limit("affine parameter count overflow"))?;
    if parameter_count > MAX_SPARSE_RREF_ENTRIES {
        return Err(limit("affine parameter stored-entry limit exceeded"));
    }
    let term_count = kernel
        .entries
        .len()
        .checked_mul(rhs_columns)
        .and_then(|count| count.checked_add(particular.entries.len()))
        .ok_or_else(|| limit("affine symbolic term count overflow"))?;
    if term_count > MAX_AFFINE_PARAMETER_TERMS {
        return Err(limit("affine symbolic term limit exceeded"));
    }
    let mut active_rows: Vec<bool> = work.vector(kernel.rows)?;
    active_rows.resize(kernel.rows, false);
    let mut active_count = 0usize;
    for entry in &kernel.entries {
        work.tick(1)?;
        if !active_rows[entry.row] {
            active_rows[entry.row] = true;
            active_count += 1;
        }
    }
    let mut output_count = active_count
        .checked_mul(rhs_columns)
        .and_then(|count| count.checked_add(parameter_count))
        .ok_or_else(|| limit("affine symbolic output count overflow"))?;
    for entry in &particular.entries {
        work.tick(1)?;
        if !active_rows[entry.row] {
            output_count = output_count
                .checked_add(1)
                .ok_or_else(|| limit("affine symbolic output count overflow"))?;
        }
    }
    if output_count > MAX_SPARSE_RREF_ENTRIES {
        return Err(limit(
            "aggregate affine symbolic stored-entry limit exceeded",
        ));
    }

    let mut parameter_entries = BTreeMap::new();
    for row in 0..free_columns.len() {
        for column in 0..rhs_columns {
            work.tick(1)?;
            work.reserve(512)?;
            parameter_entries.insert(
                (row, column),
                selected_affine_parameter(
                    row * rhs_columns + column,
                    free_columns,
                    coordinates,
                    work,
                )?,
            );
        }
    }
    let parameters = SparseMatrix::new(free_columns.len(), rhs_columns, parameter_entries)?;
    let mut solution_entries = BTreeMap::new();
    let (mut particular_index, mut kernel_index) = (0, 0);
    for row in 0..particular.rows {
        work.tick(1)?;
        let particular_start = particular_index;
        while particular_index < particular.entries.len()
            && particular.entries[particular_index].row == row
        {
            work.tick(1)?;
            particular_index += 1;
        }
        let kernel_start = kernel_index;
        while kernel_index < kernel.entries.len() && kernel.entries[kernel_index].row == row {
            work.tick(1)?;
            kernel_index += 1;
        }
        let particular_row = &particular.entries[particular_start..particular_index];
        let kernel_row = &kernel.entries[kernel_start..kernel_index];
        if kernel_row.is_empty() {
            for entry in particular_row {
                work.tick(1)?;
                work.reserve(512)?;
                solution_entries.insert(
                    (row, entry.column),
                    rational_expression(&entry.value, work)?,
                );
            }
            continue;
        }
        let mut constant_index = 0;
        for column in 0..rhs_columns {
            work.tick(1)?;
            let constant = particular_row
                .get(constant_index)
                .filter(|entry| entry.column == column);
            let mut terms = work.vector(kernel_row.len() + usize::from(constant.is_some()))?;
            let mut coefficient_bits = 1;
            if let Some(entry) = constant {
                coefficient_bits = entry.value.numer().bits().max(entry.value.denom().bits());
                terms.push(rational_expression(&entry.value, work)?);
                constant_index += 1;
            }
            for entry in kernel_row {
                work.tick(1)?;
                let parameter = selected_affine_parameter(
                    entry.column * rhs_columns + column,
                    free_columns,
                    coordinates,
                    work,
                )?;
                if entry.value.is_one() {
                    terms.push(parameter);
                } else {
                    coefficient_bits = coefficient_bits
                        .max(entry.value.numer().bits())
                        .max(entry.value.denom().bits());
                    let mut factors = work.vector(2)?;
                    factors.push(rational_expression(&entry.value, work)?);
                    factors.push(parameter);
                    terms.push(Expr::Mul(factors));
                }
            }
            let expression = affine_expression(terms, coefficient_bits, parameter_bytes, work)?;
            work.reserve(512)?;
            solution_entries.insert((row, column), expression);
        }
    }
    let solution = SparseMatrix::new(particular.rows, rhs_columns, solution_entries)?;
    let (_, _, free_columns) = verified
        .into_parts()
        .ok_or_else(|| invalid("accepted affine family lost its sparse parts"))?;
    work.tick(1)?;
    Ok(Some((solution, parameters, free_columns)))
}

impl SparseMatrix {
    /// Generate a complete-family proposal without an acceptance receipt.
    pub fn affine_solution_candidate_with_meter<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<SparseAffineSolutionCandidate, MatrixError> {
        generate_affine(self, rhs, &mut Work::new(meter))
    }

    /// Complete exact affine solution family with separately owned phase meters.
    /// A runtime may supply a protected verifier meter; this synchronous API
    /// does not create a reservation. Neither phase retries with fresh budgets.
    pub fn solve_affine_with_meters<G: BudgetMeter, V: BudgetMeter>(
        &self,
        rhs: &Self,
        generator: &mut G,
        verifier: &mut V,
    ) -> Result<VerifiedSparseAffineSolution, MatrixError> {
        let candidate = self.affine_solution_candidate_with_meter(rhs, generator)?;
        verify_sparse_affine_solution_candidate(self, rhs, candidate, verifier)
    }

    /// Solve all RHS columns together with one cumulative caller ledger.
    /// Admission, generation, independent verification, and sparse lifting are
    /// included. Singular, rectangular, and empty systems retain their complete
    /// solution sets, and inconsistency is an accepted mathematical outcome.
    pub fn metered_solve_affine<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<VerifiedSparseAffineSolution, MatrixError> {
        let mut work = Work::new(meter);
        let candidate = generate_affine(self, rhs, &mut work)?;
        accept_affine(candidate, &mut work)
    }

    /// Return the complete exact rational family `X = P + N T` without dense
    /// conversion. P is `self.cols() x rhs.cols()` and N is the complete kernel
    /// in ascending free-column order. All sparse and arithmetic limits remain
    /// active even with this unbounded outer meter.
    pub fn solve_affine(&self, rhs: &Self) -> Result<VerifiedSparseAffineSolution, MatrixError> {
        self.metered_solve_affine(rhs, &mut Unbounded)
    }

    /// Complete parameterized sparse solution with one ledger through native
    /// symbol construction. Parameters are plain `tau0`, `tau1`, ... in
    /// row-major order. Parameter, term, and total stored-output amplification
    /// are bounded before construction; no Python arithmetic participates.
    pub fn metered_gauss_jordan_solve<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Option<(Self, Self, Vec<usize>)>, MatrixError> {
        let mut work = Work::new(meter);
        let candidate = generate_affine(self, rhs, &mut work)?;
        let verified = accept_affine(candidate, &mut work)?;
        render_affine(verified, None, &mut work)
    }

    pub fn gauss_jordan_solve(
        &self,
        rhs: &Self,
    ) -> Result<Option<(Self, Self, Vec<usize>)>, MatrixError> {
        self.metered_gauss_jordan_solve(rhs, &mut Unbounded)
    }

    /// Complete single-RHS solution using the caller's ordered typed coordinates.
    /// `Some` supplies exactly one distinct symbol atom for every unknown;
    /// only coordinates selected by the independently checked free columns
    /// become parameters. Names, assumption identities and encoded Dummy labels
    /// are retained exactly. `None` selects the ordinary ascending tau names.
    /// All coordinates are admitted before solving, even for unique, empty or
    /// inconsistent systems. Admission, source work, proof replay and every
    /// repeated symbol copy/comparison share the same caller ledger.
    pub fn metered_solve_affine_parameters<M: BudgetMeter>(
        &self,
        rhs: &Self,
        variables: Option<&[Expr]>,
        meter: &mut M,
    ) -> Result<Option<(Self, Self, Vec<usize>)>, MatrixError> {
        let mut work = Work::new(meter);
        work.tick(1)?;
        if rhs.cols() != 1 {
            return Err(MatrixError::InvalidAffineParameters(
                "a named linear system requires exactly one right-hand-side column",
            ));
        }
        let coordinates = admit_affine_parameters(self.cols(), variables, &mut work)?;
        let candidate = generate_affine(self, rhs, &mut work)?;
        let verified = accept_affine(candidate, &mut work)?;
        render_affine(verified, coordinates.as_deref(), &mut work)
    }

    pub fn solve_affine_parameters(
        &self,
        rhs: &Self,
        variables: Option<&[Expr]>,
    ) -> Result<Option<(Self, Self, Vec<usize>)>, MatrixError> {
        self.metered_solve_affine_parameters(rhs, variables, &mut Unbounded)
    }

    /// Solve a stored sparse augmented matrix `[A | b]`, interpreting its last
    /// column as the single RHS. The split moves admitted stored entries and
    /// never scans the implicit zero area. Coordinate admission, splitting,
    /// independent complete-family verification and rendering share one ledger.
    pub fn metered_solve_augmented_affine_parameters<M: BudgetMeter>(
        &self,
        variables: Option<&[Expr]>,
        meter: &mut M,
    ) -> Result<Option<(Self, Self, Vec<usize>)>, MatrixError> {
        let mut work = Work::new(meter);
        work.tick(1)?;
        let columns = self.cols().checked_sub(1).ok_or(
            MatrixError::InvalidAffineParameters(
                "an augmented system requires a right-hand-side column",
            ),
        )?;
        let coordinates = admit_affine_parameters(columns, variables, &mut work)?;
        let (coefficients, rhs) = split_augmented_source(self, &mut work)?;
        let candidate = generate_affine_sources(coefficients, rhs, &mut work)?;
        let verified = accept_affine(candidate, &mut work)?;
        render_affine(verified, coordinates.as_deref(), &mut work)
    }

    pub fn solve_augmented_affine_parameters(
        &self,
        variables: Option<&[Expr]>,
    ) -> Result<Option<(Self, Self, Vec<usize>)>, MatrixError> {
        self.metered_solve_augmented_affine_parameters(variables, &mut Unbounded)
    }

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
