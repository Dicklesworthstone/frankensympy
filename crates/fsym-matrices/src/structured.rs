//! Sparse exact-rational elimination and independent sparse identity checks.
//!
//! The working rows, pivot-column index, factors, right-hand sides and results
//! remain sparse. Deterministic row pivoting handles zero diagonal entries as
//! well as triangular and banded inputs. Fill, arithmetic work, coefficient
//! growth and cumulative logical allocations have explicit bounds. These are
//! same-crate reference checks, not portable proof-kernel certificates.
//!
//! This lane admits dimensions up to 8192, 262144 stored entries per matrix,
//! 16384 bits per coefficient component (including uncancelled arithmetic
//! intermediates), ten million metered compute units, and 256 MiB of cumulative
//! logical allocation charges. Generation and checking share one caller meter;
//! this synchronous API does not reserve a protected verifier budget.

#![forbid(unsafe_code)]

use crate::{MatrixError, SparseMatrix, from_rational};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};

const MAX_DIMENSION: usize = 8_192;
const MAX_ENTRIES: usize = 262_144;
const MAX_COEFFICIENT_BITS: u64 = 16_384;
const MAX_WORK: u64 = 10_000_000;
const MAX_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;

type Row = BTreeMap<usize, BigRational>;
type Rows = Vec<Row>;

/// Untrusted exact-rational candidate for `P A = L U`.
///
/// Row `i` of `P A` is row `row_permutation[i]` of `A`. `L` must be unit
/// lower triangular and `U` upper triangular. Singular matrices are allowed;
/// nonzero diagonal entries of `U` are a separate invertibility obligation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseLuCertificate {
    pub row_permutation: Vec<usize>,
    pub lower: SparseMatrix,
    pub upper: SparseMatrix,
}

/// Untrusted determinant candidate and its sparse decomposition witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseDeterminantCertificate {
    pub determinant: Expr,
    pub factorization: SparseLuCertificate,
}

/// Untrusted unique-solution candidate, including an invertibility witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseLinearSystemCertificate {
    pub solution: SparseMatrix,
    pub factorization: SparseLuCertificate,
}

fn meter_error(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => MatrixError::ResourceLimit(error.to_string()),
    }
}

fn arithmetic_error(error: RationalArithmeticError) -> MatrixError {
    match error {
        RationalArithmeticError::Meter(error) => meter_error(error),
        RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
        RationalArithmeticError::InvariantViolation(message) => {
            MatrixError::ArithmeticInvariantViolation(message)
        }
        other => MatrixError::ResourceLimit(other.to_string()),
    }
}

fn limit(message: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("sparse exact elimination: {message}"))
}

fn rejected(message: &str) -> MatrixError {
    MatrixError::InvalidCertificate(message.to_string())
}

/// One aggregate counter covers generation, verification and result lifting.
/// Container charges are conservative logical bounds, combined with the exact
/// arithmetic substrate's own charges. They do not measure the process allocator.
struct Work<'a, M> {
    meter: &'a mut M,
    steps: u64,
    allocation_bytes: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            steps: 0,
            allocation_bytes: 0,
        }
    }

    fn tick(&mut self, amount: u64) -> Result<(), MatrixError> {
        self.meter.checkpoint().map_err(meter_error)?;
        self.steps = self
            .steps
            .checked_add(amount)
            .ok_or_else(|| limit("work count overflow"))?;
        if self.steps > MAX_WORK {
            return Err(limit("arithmetic work limit exceeded"));
        }
        if amount != 0 {
            self.meter
                .charge(Dimension::ComputeSteps, amount)
                .map_err(meter_error)?;
        }
        Ok(())
    }

    fn reserve(&mut self, bytes: u64) -> Result<(), MatrixError> {
        self.meter.checkpoint().map_err(meter_error)?;
        self.allocation_bytes = self
            .allocation_bytes
            .checked_add(bytes)
            .ok_or_else(|| limit("allocation accounting overflow"))?;
        if self.allocation_bytes > MAX_ALLOCATION_BYTES {
            return Err(limit("cumulative allocation limit exceeded"));
        }
        if bytes != 0 {
            self.meter
                .charge_batch(&[
                    (Dimension::MemoryBytes, bytes),
                    (Dimension::AllocationCount, 1),
                ])
                .map_err(meter_error)?;
        }
        Ok(())
    }

    fn clone_value(&mut self, value: &BigRational) -> Result<BigRational, MatrixError> {
        check_value(value)?;
        self.tick(1)?;
        self.reserve(
            value
                .numer()
                .limb_count()
                .saturating_add(value.denom().limb_count())
                .max(1)
                * 8,
        )?;
        Ok(value.clone())
    }

    fn arithmetic(
        &mut self,
        left: &BigRational,
        right: &BigRational,
        operation: Arithmetic,
    ) -> Result<BigRational, MatrixError> {
        if matches!(operation, Arithmetic::Divide) && right.is_zero() {
            return Err(MatrixError::DivisionByZero);
        }
        let (ln, ld) = (left.numer().bits(), left.denom().bits());
        let (rn, rd) = (right.numer().bits(), right.denom().bits());
        let (numerator, denominator) = match operation {
            Arithmetic::Add | Arithmetic::Subtract => (
                ln.saturating_add(rd)
                    .max(rn.saturating_add(ld))
                    .saturating_add(1),
                ld.saturating_add(rd),
            ),
            Arithmetic::Multiply => (ln.saturating_add(rn), ld.saturating_add(rd)),
            Arithmetic::Divide => (ln.saturating_add(rd), ld.saturating_add(rn)),
        };
        // Bound cross products before BigRational allocates or cancels them.
        if numerator > MAX_COEFFICIENT_BITS || denominator > MAX_COEFFICIENT_BITS {
            return Err(limit(
                "rational intermediate exceeds the coefficient bit limit",
            ));
        }
        let result = match operation {
            Arithmetic::Add => left.metered_add(right, self),
            Arithmetic::Subtract => left.metered_sub(right, self),
            Arithmetic::Multiply => left.metered_mul(right, self),
            Arithmetic::Divide => left.metered_div(right, self),
        }
        .map_err(arithmetic_error)?;
        check_value(&result)?;
        Ok(result)
    }
}

// Arithmetic primitives charge this adapter, so their internal temporary
// buffers and GCD/division work are included in the same aggregate hard caps.
impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let mut steps = self.steps;
        let mut bytes = self.allocation_bytes;
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
        self.allocation_bytes = bytes;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

#[derive(Clone, Copy)]
enum Arithmetic {
    Add,
    Subtract,
    Multiply,
    Divide,
}

fn check_value(value: &BigRational) -> Result<(), MatrixError> {
    if value.numer().bits() > MAX_COEFFICIENT_BITS || value.denom().bits() > MAX_COEFFICIENT_BITS {
        return Err(limit("input coefficient exceeds the bit limit"));
    }
    Ok(())
}

fn exact_entry(entry: &Expr) -> Result<BigRational, MatrixError> {
    match entry {
        Expr::Integer(value) if value.bits() <= MAX_COEFFICIENT_BITS => {
            Ok(BigRational::from_integer(value.clone()))
        }
        Expr::Integer(_) => Err(limit("input coefficient exceeds the bit limit")),
        Expr::Rational(value) => {
            check_value(value)?;
            Ok(value.clone())
        }
        _ => Err(MatrixError::UnsupportedCertificateDomain),
    }
}

fn copy_entry<M: BudgetMeter>(
    entry: &Expr,
    work: &mut Work<'_, M>,
) -> Result<BigRational, MatrixError> {
    match entry {
        Expr::Integer(value) => {
            if value.bits() > MAX_COEFFICIENT_BITS {
                return Err(limit("input coefficient exceeds the bit limit"));
            }
            work.reserve(value.limb_count().saturating_add(1).saturating_mul(8))?;
        }
        Expr::Rational(value) => {
            check_value(value)?;
            work.reserve(
                value
                    .numer()
                    .limb_count()
                    .saturating_add(value.denom().limb_count())
                    .max(1)
                    .saturating_mul(8),
            )?;
        }
        _ => return Err(MatrixError::UnsupportedCertificateDomain),
    }
    exact_entry(entry)
}

fn square(matrix: &SparseMatrix) -> Result<usize, MatrixError> {
    if matrix.rows() != matrix.cols() {
        return Err(MatrixError::NotSquare(matrix.rows(), matrix.cols()));
    }
    Ok(matrix.rows())
}

fn read_rows<M: BudgetMeter>(
    matrix: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<Rows, MatrixError> {
    work.tick(1)?;
    if matrix.rows() > MAX_DIMENSION || matrix.cols() > MAX_DIMENSION {
        return Err(limit("matrix dimension exceeds 8192"));
    }
    if matrix.nnz() > MAX_ENTRIES {
        return Err(limit("stored entry limit exceeded"));
    }
    // Validate all original entries before simplification, zero shortcuts or
    // arithmetic; unsupported/undefined expressions cannot disappear here.
    for (&(row, column), entry) in matrix.entries() {
        work.tick(1)?;
        if row >= matrix.rows() || column >= matrix.cols() {
            return Err(rejected("sparse coordinate lies outside its shape"));
        }
        let bits = match entry {
            Expr::Integer(value) => {
                if value.bits() > MAX_COEFFICIENT_BITS {
                    return Err(limit("input coefficient exceeds the bit limit"));
                }
                value.bits().saturating_add(1)
            }
            Expr::Rational(value) => {
                check_value(value)?;
                value.numer().bits().saturating_add(value.denom().bits())
            }
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        };
        work.reserve(256 + bits.div_ceil(8))?;
    }
    work.reserve((matrix.rows() as u64).saturating_mul(32))?;
    let mut rows = vec![Row::new(); matrix.rows()];
    for (&(row, column), entry) in matrix.entries() {
        work.tick(1)?;
        let value = exact_entry(entry)?;
        if !value.is_zero() {
            rows[row].insert(column, value);
        }
    }
    Ok(rows)
}

fn write_rows<M: BudgetMeter>(
    rows: Rows,
    columns: usize,
    work: &mut Work<'_, M>,
) -> Result<SparseMatrix, MatrixError> {
    let count: usize = rows.iter().map(Row::len).sum();
    if count > MAX_ENTRIES {
        return Err(limit("result fill limit exceeded"));
    }
    work.reserve((count as u64).saturating_mul(512))?;
    let height = rows.len();
    let mut entries = BTreeMap::new();
    for (row, values) in rows.into_iter().enumerate() {
        for (column, value) in values {
            work.tick(1)?;
            if !value.is_zero() {
                entries.insert((row, column), from_rational(value));
            }
        }
    }
    SparseMatrix::new(height, columns, entries)
}

fn store<M: BudgetMeter>(
    row: &mut Row,
    column: usize,
    value: BigRational,
    count: &mut usize,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    work.tick(1)?;
    if value.is_zero() {
        if row.remove(&column).is_some() {
            *count -= 1;
        }
    } else {
        if !row.contains_key(&column) {
            if *count >= MAX_ENTRIES {
                return Err(limit("intermediate fill limit exceeded"));
            }
            work.reserve(256)?;
            *count += 1;
        }
        row.insert(column, value);
    }
    Ok(())
}

fn clone_row<M: BudgetMeter>(row: &Row, work: &mut Work<'_, M>) -> Result<Row, MatrixError> {
    work.reserve((row.len() as u64).saturating_mul(256))?;
    row.iter()
        .map(|(&column, value)| Ok((column, work.clone_value(value)?)))
        .collect()
}

fn generate_lu<M: BudgetMeter>(
    matrix: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseLuCertificate, MatrixError> {
    let n = square(matrix)?;
    let mut upper = read_rows(matrix, work)?;
    work.reserve((n as u64).saturating_mul(72))?;
    let mut lower = vec![Row::new(); n];
    let mut columns = vec![BTreeSet::new(); n];
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut upper_count = upper.iter().map(Row::len).sum();
    let mut lower_count = 0;
    work.reserve((upper_count as u64).saturating_mul(128))?;
    for (row, values) in upper.iter().enumerate() {
        for &column in values.keys() {
            work.tick(1)?;
            columns[column].insert(row);
        }
    }

    for pivot_column in 0..n {
        work.tick(1)?;
        let Some(pivot_row) = columns[pivot_column].range(pivot_column..).next().copied() else {
            // This column is already zero on and below the diagonal. Keeping
            // its zero diagonal yields a valid singular triangular factor.
            continue;
        };
        if pivot_row != pivot_column {
            for (&column, _) in upper[pivot_column].iter().chain(&upper[pivot_row]) {
                work.tick(1)?;
                columns[column].remove(&pivot_column);
                columns[column].remove(&pivot_row);
            }
            upper.swap(pivot_column, pivot_row);
            lower.swap(pivot_column, pivot_row);
            permutation.swap(pivot_column, pivot_row);
            for &column in upper[pivot_column].keys() {
                work.tick(1)?;
                work.reserve(128)?;
                columns[column].insert(pivot_column);
            }
            for &column in upper[pivot_row].keys() {
                work.tick(1)?;
                work.reserve(128)?;
                columns[column].insert(pivot_row);
            }
        }
        let pivot_values = clone_row(&upper[pivot_column], work)?;
        let pivot = &pivot_values[&pivot_column];
        work.reserve((columns[pivot_column].len() as u64).saturating_mul(8))?;
        let affected: Vec<usize> = columns[pivot_column]
            .range((pivot_column + 1)..)
            .copied()
            .collect();
        for row in affected {
            let factor = work.arithmetic(&upper[row][&pivot_column], pivot, Arithmetic::Divide)?;
            store(
                &mut lower[row],
                pivot_column,
                work.clone_value(&factor)?,
                &mut lower_count,
                work,
            )?;
            for (&column, pivot_value) in &pivot_values {
                let product = work.arithmetic(&factor, pivot_value, Arithmetic::Multiply)?;
                let zero = BigRational::zero();
                let prior = upper[row].get(&column).unwrap_or(&zero);
                let next = work.arithmetic(prior, &product, Arithmetic::Subtract)?;
                let was_stored = upper[row].contains_key(&column);
                if next.is_zero() {
                    columns[column].remove(&row);
                } else if !was_stored {
                    if upper_count >= MAX_ENTRIES {
                        return Err(limit("intermediate fill limit exceeded"));
                    }
                    work.reserve(128)?;
                    columns[column].insert(row);
                }
                store(&mut upper[row], column, next, &mut upper_count, work)?;
            }
        }
    }
    for (row, values) in lower.iter_mut().enumerate() {
        store(values, row, BigRational::one(), &mut lower_count, work)?;
    }
    Ok(SparseLuCertificate {
        row_permutation: permutation,
        lower: write_rows(lower, n, work)?,
        upper: write_rows(upper, n, work)?,
    })
}

fn permutation_parity<M: BudgetMeter>(
    permutation: &[usize],
    n: usize,
    work: &mut Work<'_, M>,
) -> Result<bool, MatrixError> {
    if permutation.len() != n {
        return Err(rejected("sparse LU permutation has the wrong length"));
    }
    work.reserve((n as u64).saturating_mul(2))?;
    let mut seen = vec![false; n];
    for &row in permutation {
        work.tick(1)?;
        if row >= n || seen[row] {
            return Err(MatrixError::InvalidPermutationMatrix);
        }
        seen[row] = true;
    }
    seen.fill(false);
    let mut odd = false;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut row = start;
        let mut length = 0;
        while !seen[row] {
            work.tick(1)?;
            seen[row] = true;
            length += 1;
            row = permutation[row];
        }
        odd ^= length % 2 == 0;
    }
    Ok(odd)
}

/// Exact reference sparse convolution. This checker has no pivoting,
/// elimination, triangular solve or calls to SparseMatrix::matmul.
fn check_product<M: BudgetMeter>(
    left: &Rows,
    right: &Rows,
    expected: &Rows,
    permutation: Option<&[usize]>,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    for (row_index, row) in left.iter().enumerate() {
        work.tick(1)?;
        let mut product = Row::new();
        let mut count = 0;
        for (&inner, left_value) in row {
            for (&column, right_value) in &right[inner] {
                let value = work.arithmetic(left_value, right_value, Arithmetic::Multiply)?;
                let sum = match product.get(&column) {
                    Some(prior) => work.arithmetic(prior, &value, Arithmetic::Add)?,
                    None => value,
                };
                store(&mut product, column, sum, &mut count, work)?;
            }
        }
        let expected_row = permutation.map_or(row_index, |p| p[row_index]);
        if product != expected[expected_row] {
            return Err(rejected("exact sparse matrix product identity failed"));
        }
    }
    work.tick(1)
}

fn check_lu<M: BudgetMeter>(
    matrix: &SparseMatrix,
    certificate: &SparseLuCertificate,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    let n = square(matrix)?;
    let original = read_rows(matrix, work)?;
    if certificate.lower.rows() != n
        || certificate.lower.cols() != n
        || certificate.upper.rows() != n
        || certificate.upper.cols() != n
    {
        return Err(rejected("sparse LU factor shape does not match the input"));
    }
    permutation_parity(&certificate.row_permutation, n, work)?;
    let lower = read_rows(&certificate.lower, work)?;
    let upper = read_rows(&certificate.upper, work)?;
    for row in 0..n {
        work.tick(1)?;
        if lower[row].get(&row) != Some(&BigRational::one())
            || lower[row].keys().any(|&column| column > row)
            || upper[row].keys().any(|&column| column < row)
        {
            return Err(rejected(
                "sparse LU factors do not have the required triangular form",
            ));
        }
    }
    check_product(
        &lower,
        &upper,
        &original,
        Some(&certificate.row_permutation),
        work,
    )
}

fn determinant_from_lu<M: BudgetMeter>(
    certificate: &SparseLuCertificate,
    work: &mut Work<'_, M>,
) -> Result<BigRational, MatrixError> {
    let n = certificate.upper.rows();
    let odd = permutation_parity(&certificate.row_permutation, n, work)?;
    let mut product = if odd {
        -BigRational::one()
    } else {
        BigRational::one()
    };
    for row in 0..n {
        work.tick(1)?;
        let Some(entry) = certificate.upper.entries().get(&(row, row)) else {
            return Ok(BigRational::zero());
        };
        let value = copy_entry(entry, work)?;
        product = work.arithmetic(&product, &value, Arithmetic::Multiply)?;
    }
    Ok(product)
}

fn solve_factors<M: BudgetMeter>(
    right_hand_side: &SparseMatrix,
    certificate: &SparseLuCertificate,
    work: &mut Work<'_, M>,
) -> Result<SparseMatrix, MatrixError> {
    let n = certificate.upper.rows();
    let lower = read_rows(&certificate.lower, work)?;
    let upper = read_rows(&certificate.upper, work)?;
    let rhs = read_rows(right_hand_side, work)?;
    // Refuse singular systems even when the RHS is zero or has no columns:
    // this operation promises a unique solution, not one arbitrary solution.
    for (row, values) in upper.iter().enumerate() {
        if !values.contains_key(&row) {
            return Err(MatrixError::SingularMatrix);
        }
    }
    work.reserve((n as u64).saturating_mul(32))?;
    let mut solution: Rows = Vec::with_capacity(n);
    for &row in &certificate.row_permutation {
        solution.push(clone_row(&rhs[row], work)?);
    }
    let mut count = solution.iter().map(Row::len).sum();
    // L Y = P B, with a unit diagonal.
    for row in 0..n {
        work.tick(1)?;
        for (&inner, coefficient) in lower[row].range(..row) {
            let source = clone_row(&solution[inner], work)?;
            subtract_scaled(&mut solution[row], &source, coefficient, &mut count, work)?;
        }
    }
    // U X = Y.
    for row in (0..n).rev() {
        work.tick(1)?;
        for (&inner, coefficient) in upper[row].range((row + 1)..) {
            let source = clone_row(&solution[inner], work)?;
            subtract_scaled(&mut solution[row], &source, coefficient, &mut count, work)?;
        }
        for value in solution[row].values_mut() {
            *value = work.arithmetic(value, &upper[row][&row], Arithmetic::Divide)?;
        }
    }
    write_rows(solution, right_hand_side.cols(), work)
}

fn subtract_scaled<M: BudgetMeter>(
    target: &mut Row,
    source: &Row,
    scale: &BigRational,
    count: &mut usize,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    for (&column, value) in source {
        let product = work.arithmetic(scale, value, Arithmetic::Multiply)?;
        let zero = BigRational::zero();
        let prior = target.get(&column).unwrap_or(&zero);
        let result = work.arithmetic(prior, &product, Arithmetic::Subtract)?;
        store(target, column, result, count, work)?;
    }
    Ok(())
}

fn check_solve<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    certificate: &SparseLinearSystemCertificate,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    let n = square(matrix)?;
    if rhs.rows() != n
        || certificate.solution.rows() != n
        || certificate.solution.cols() != rhs.cols()
    {
        return Err(rejected("sparse solve input or candidate shape mismatch"));
    }
    check_lu(matrix, &certificate.factorization, work)?;
    // The independent factor identity plus a nonzero diagonal proves
    // invertibility. An AX=B residual alone would not establish uniqueness.
    for row in 0..n {
        work.tick(1)?;
        if !certificate
            .factorization
            .upper
            .entries()
            .contains_key(&(row, row))
        {
            return Err(rejected(
                "sparse solve factorization does not prove uniqueness",
            ));
        }
    }
    let a = read_rows(matrix, work)?;
    let x = read_rows(&certificate.solution, work)?;
    let b = read_rows(rhs, work)?;
    check_product(&a, &x, &b, None, work)
}

impl SparseMatrix {
    /// Sparse rational LU with deterministic row pivots and an independent
    /// `P A = L U` check. Singular inputs still have a valid decomposition.
    pub fn lu(&self) -> Result<SparseLuCertificate, MatrixError> {
        self.metered_lu(&mut Unbounded)
    }

    /// Metered LU; all generation and reference-check work shares this meter.
    pub fn metered_lu<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<SparseLuCertificate, MatrixError> {
        let mut work = Work::new(meter);
        let certificate = generate_lu(self, &mut work)?;
        check_lu(self, &certificate, &mut work)?;
        Ok(certificate)
    }

    /// Exact rational determinant without dense materialization.
    pub fn det(&self) -> Result<Expr, MatrixError> {
        self.metered_det(&mut Unbounded)
    }

    /// Metered determinant, including independent decomposition validation.
    pub fn metered_det<M: BudgetMeter>(&self, meter: &mut M) -> Result<Expr, MatrixError> {
        let mut work = Work::new(meter);
        let factorization = generate_lu(self, &mut work)?;
        check_lu(self, &factorization, &mut work)?;
        let determinant = from_rational(determinant_from_lu(&factorization, &mut work)?);
        work.tick(1)?;
        Ok(determinant)
    }

    /// Determinant and an untrusted, separately checkable decomposition witness.
    pub fn det_with_certificate(&self) -> Result<SparseDeterminantCertificate, MatrixError> {
        let mut meter = Unbounded;
        let mut work = Work::new(&mut meter);
        let factorization = generate_lu(self, &mut work)?;
        check_lu(self, &factorization, &mut work)?;
        let determinant = from_rational(determinant_from_lu(&factorization, &mut work)?);
        work.tick(1)?;
        Ok(SparseDeterminantCertificate {
            determinant,
            factorization,
        })
    }

    /// Solve a square nonsingular exact-rational system with sparse RHS/results.
    /// Singular systems refuse rather than choosing an arbitrary solution.
    pub fn solve(&self, rhs: &SparseMatrix) -> Result<SparseMatrix, MatrixError> {
        self.metered_solve(rhs, &mut Unbounded)
    }

    /// Metered unique solve with one aggregate generation/verification budget.
    pub fn metered_solve<M: BudgetMeter>(
        &self,
        rhs: &SparseMatrix,
        meter: &mut M,
    ) -> Result<SparseMatrix, MatrixError> {
        Ok(self.solve_certificate_with_meter(rhs, meter)?.solution)
    }

    /// Return the solution together with the sparse invertibility witness.
    pub fn solve_with_certificate(
        &self,
        rhs: &SparseMatrix,
    ) -> Result<SparseLinearSystemCertificate, MatrixError> {
        self.solve_certificate_with_meter(rhs, &mut Unbounded)
    }

    fn solve_certificate_with_meter<M: BudgetMeter>(
        &self,
        rhs: &SparseMatrix,
        meter: &mut M,
    ) -> Result<SparseLinearSystemCertificate, MatrixError> {
        let mut work = Work::new(meter);
        let n = square(self)?;
        if rhs.rows() != n {
            return Err(MatrixError::ShapeMismatch(n, n, rhs.rows(), rhs.cols()));
        }
        // Validate RHS even if elimination will discover a singular input.
        read_rows(rhs, &mut work)?;
        let factorization = generate_lu(self, &mut work)?;
        let solution = solve_factors(rhs, &factorization, &mut work)?;
        let certificate = SparseLinearSystemCertificate {
            solution,
            factorization,
        };
        check_solve(self, rhs, &certificate, &mut work)?;
        Ok(certificate)
    }
}

/// Independently check factor shapes, unit/triangular forms, permutation and
/// exact sparse `P A = L U`; no elimination is rerun by this checker.
pub fn verify_sparse_lu_certificate(
    matrix: &SparseMatrix,
    certificate: &SparseLuCertificate,
) -> Result<(), MatrixError> {
    metered_verify_sparse_lu_certificate(matrix, certificate, &mut Unbounded)
}

pub fn metered_verify_sparse_lu_certificate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    certificate: &SparseLuCertificate,
    meter: &mut M,
) -> Result<(), MatrixError> {
    check_lu(matrix, certificate, &mut Work::new(meter))
}

/// Check the determinant formula only after independent decomposition checks.
pub fn verify_sparse_determinant_certificate(
    matrix: &SparseMatrix,
    certificate: &SparseDeterminantCertificate,
) -> Result<(), MatrixError> {
    let mut meter = Unbounded;
    let mut work = Work::new(&mut meter);
    check_lu(matrix, &certificate.factorization, &mut work)?;
    let supplied = copy_entry(&certificate.determinant, &mut work)?;
    if supplied != determinant_from_lu(&certificate.factorization, &mut work)? {
        return Err(rejected(
            "sparse determinant disagrees with its decomposition",
        ));
    }
    Ok(())
}

/// Independently prove both `A X = B` and invertibility from `P A = L U`.
pub fn verify_sparse_linear_system_certificate(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    certificate: &SparseLinearSystemCertificate,
) -> Result<(), MatrixError> {
    metered_verify_sparse_linear_system_certificate(matrix, rhs, certificate, &mut Unbounded)
}

pub fn metered_verify_sparse_linear_system_certificate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    certificate: &SparseLinearSystemCertificate,
    meter: &mut M,
) -> Result<(), MatrixError> {
    check_solve(matrix, rhs, certificate, &mut Work::new(meter))
}
