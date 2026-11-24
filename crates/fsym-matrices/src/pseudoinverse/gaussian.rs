//! Gaussian-rational proposals by realification, checked directly over Q(i).
//!
//! R(A) = [[Re A, -Im A], [Im A, Re A]] preserves products and adjoints.
//! The sparse rational rank-factorization engine proposes R(A)+. Its left
//! blocks propose a complex inverse; no rank, block-structure or accepted
//! flag from that generator is trusted. A distinct dense reference pass
//! reinterprets the original cells and checks all four Hermitian Penrose
//! identities using only bounded exact scalar arithmetic.
//!
//! Each complex axis is at most 32; original and published components have
//! at most 4096 bits. All arithmetic, source interpretation, temporary
//! storage and publication share 20M compute units and 256 MiB cumulative
//! logical allocation. Cancellation is checked throughout scalar work and
//! immediately before return. This synchronous reference-checked interface
//! neither reserves protected verifier resources nor exports a new portable
//! proof-kernel certificate.

mod least_squares;
pub use least_squares::{GaussianLeastSquaresCandidate, VerifiedGaussianLeastSquaresFamily};
mod exact_solve;
pub use exact_solve::GaussianLinearSystemCandidate;
mod qr;
pub use qr::GaussianQrCandidate;
mod lu;
pub use lu::GaussianLuCandidate;
mod rref;
pub use rref::{GaussianAffineCandidate, GaussianRrefCandidate, VerifiedGaussianAffineSolution};

use crate::{Matrix, MatrixError, SparseMatrix, generate_sparse_pseudoinverse_candidate};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Constant, Expr};
use fsym_proof_kernel::hermitian_inertia::{
    GaussianRationalMatrix, HermitianInertiaError, MAX_GAUSSIAN_RATIONAL_INPUT_BITS,
    MAX_HERMITIAN_INERTIA_DIMENSION, read_gaussian_rational_matrix,
};
use fsym_proof_kernel::sparse_rref::SparseRrefError;
use num_traits::{One, Zero};
use std::collections::BTreeMap;

const MAX_STEPS: u64 = 20_000_000;
const MAX_BYTES: u64 = 256 * 1024 * 1024;
const MAX_WORK_BITS: u64 = 8_192;
type Result<T> = std::result::Result<T, MatrixError>;
type Gaussian = (BigRational, BigRational);

fn limit(message: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("Gaussian-rational pseudoinverse: {message}"))
}

fn stopped(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => limit(&error.to_string()),
    }
}

fn arithmetic_error(error: RationalArithmeticError) -> MatrixError {
    match error {
        RationalArithmeticError::Meter(error) => stopped(error),
        RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
        RationalArithmeticError::InvariantViolation(reason) => {
            MatrixError::ArithmeticInvariantViolation(reason)
        }
        error => limit(&error.to_string()),
    }
}

fn source_error(error: HermitianInertiaError) -> MatrixError {
    match error {
        HermitianInertiaError::UnsupportedSource => MatrixError::UnsupportedCertificateDomain,
        HermitianInertiaError::UndefinedSource => MatrixError::DivisionByZero,
        HermitianInertiaError::Reference(SparseRrefError::Meter(error)) => stopped(error),
        HermitianInertiaError::Reference(SparseRrefError::Arithmetic(error)) => {
            arithmetic_error(error)
        }
        HermitianInertiaError::Reference(SparseRrefError::Limit(reason)) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; 3],
            stopped: None,
        }
    }

    fn tick(&mut self, steps: u64) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge(Dimension::ComputeSteps, steps.max(1))
            .map_err(stopped)
    }

    fn storage(&mut self, bytes: u64) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        if bytes == 0 {
            return Ok(());
        }
        self.charge_batch(&[
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(stopped)
    }

    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| limit("vector storage overflow"))?;
        self.storage(bytes)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| limit("vector allocation refused"))?;
        Ok(result)
    }

    fn clone_scalar(&mut self, value: &BigRational) -> Result<BigRational> {
        check_bits(value, MAX_WORK_BITS)?;
        let limbs = value.numer().limb_count() + value.denom().limb_count();
        self.tick(limbs + 1)?;
        self.storage(64 + 8 * limbs)?;
        Ok(value.clone())
    }

    fn negate(&mut self, value: &BigRational) -> Result<BigRational> {
        Ok(-self.clone_scalar(value)?)
    }

    fn scalar_expression(&mut self, value: BigRational) -> Result<Expr> {
        check_bits(&value, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
        self.tick(value.numer().limb_count() + value.denom().limb_count() + 1)?;
        self.storage(64 + 8 * value.numer().limb_count())?;
        Ok(if value.is_integer() {
            Expr::Integer(value.numer().clone())
        } else {
            Expr::Rational(value)
        })
    }

    fn rational(
        &mut self,
        left: &BigRational,
        right: &BigRational,
        operation: Operation,
    ) -> Result<BigRational> {
        check_bits(left, MAX_WORK_BITS)?;
        check_bits(right, MAX_WORK_BITS)?;
        let (ln, ld) = (left.numer().bits(), left.denom().bits());
        let (rn, rd) = (right.numer().bits(), right.denom().bits());
        let (numerator, denominator) = match operation {
            Operation::Add | Operation::Subtract => (
                ln.saturating_add(rd)
                    .max(rn.saturating_add(ld))
                    .saturating_add(1),
                ld.saturating_add(rd),
            ),
            Operation::Multiply => (ln.saturating_add(rn), ld.saturating_add(rd)),
            Operation::Divide => (ln.saturating_add(rd), ld.saturating_add(rn)),
        };
        // Check uncancelled cross-products before allocating their limbs.
        if numerator > MAX_WORK_BITS || denominator > MAX_WORK_BITS {
            return Err(limit("reference intermediate exceeds 8192 bits"));
        }
        self.tick(1)?;
        let value = match operation {
            Operation::Add => left.metered_add(right, self),
            Operation::Subtract => left.metered_sub(right, self),
            Operation::Multiply => left.metered_mul(right, self),
            Operation::Divide => left.metered_div(right, self),
        }
        .map_err(arithmetic_error)?;
        check_bits(&value, MAX_WORK_BITS)?;
        Ok(value)
    }

    fn multiply(&mut self, left: &Gaussian, right: &Gaussian) -> Result<Gaussian> {
        let ac = self.rational(&left.0, &right.0, Operation::Multiply)?;
        let bd = self.rational(&left.1, &right.1, Operation::Multiply)?;
        let ad = self.rational(&left.0, &right.1, Operation::Multiply)?;
        let bc = self.rational(&left.1, &right.0, Operation::Multiply)?;
        Ok((
            self.rational(&ac, &bd, Operation::Subtract)?,
            self.rational(&ad, &bc, Operation::Add)?,
        ))
    }

    fn add(&mut self, left: &Gaussian, right: &Gaussian) -> Result<Gaussian> {
        Ok((
            self.rational(&left.0, &right.0, Operation::Add)?,
            self.rational(&left.1, &right.1, Operation::Add)?,
        ))
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
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = (|| {
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
        })();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = self.meter.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

fn check_bits(value: &BigRational, maximum: u64) -> Result<()> {
    if value.numer().bits() > maximum || value.denom().bits() > maximum {
        return Err(limit("coefficient component exceeds its bit limit"));
    }
    Ok(())
}

fn read_source(
    matrix: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianRationalMatrix> {
    work.tick(1)?;
    matrix.validate_shape()?;
    // The existing source reader accepts square sparse envelopes. Missing
    // padded cells are exactly zero, so this interprets every original cell
    // without copying, simplifying or weakening the reader's domain checks.
    // Its dimension/node/height/storage bounds apply before any narrowing.
    let side = matrix.rows.max(matrix.cols);
    let mut source = read_gaussian_rational_matrix(
        side,
        side,
        matrix
            .data
            .iter()
            .enumerate()
            .map(|(index, value)| ((index / matrix.cols, index % matrix.cols), value)),
        work,
    )
    .map_err(source_error)?;
    // Validated dense storage supplied only coordinates inside the original
    // rectangle. No padded nonzero is retained, and this coefficient snapshot
    // is not an accepted inertia receipt or a mathematical certificate.
    work.tick(4)?;
    source.real.rows = matrix.rows;
    source.real.columns = matrix.cols;
    source.imaginary.rows = matrix.rows;
    source.imaginary.columns = matrix.cols;
    Ok(source)
}

#[derive(Debug, PartialEq, Eq)]
struct GaussianMatrix {
    rows: usize,
    cols: usize,
    entries: Vec<Gaussian>,
}

impl GaussianMatrix {
    fn zeros(rows: usize, cols: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let count = Matrix::checked_element_count(rows, cols)?;
        let mut entries = work.vector(count)?;
        work.tick(count as u64 + 1)?;
        work.storage(128 * count as u64)?;
        entries.resize_with(count, || (BigRational::zero(), BigRational::zero()));
        Ok(Self {
            rows,
            cols,
            entries,
        })
    }

    fn read(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let source = read_source(matrix, work)?;
        let mut result = Self::zeros(matrix.rows, matrix.cols, work)?;
        for entry in source.real.entries {
            work.tick(1)?;
            result.entries[entry.row * matrix.cols + entry.column].0 = entry.value;
        }
        for entry in source.imaginary.entries {
            work.tick(1)?;
            result.entries[entry.row * matrix.cols + entry.column].1 = entry.value;
        }
        Ok(result)
    }

    fn get(&self, row: usize, column: usize) -> &Gaussian {
        &self.entries[row * self.cols + column]
    }

    fn multiply(&self, other: &Self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        if self.cols != other.rows {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        let mut result = Self::zeros(self.rows, other.cols, work)?;
        for row in 0..self.rows {
            for column in 0..other.cols {
                let mut sum = (BigRational::zero(), BigRational::zero());
                work.storage(128)?;
                for inner in 0..self.cols {
                    work.tick(1)?;
                    let product = work.multiply(self.get(row, inner), other.get(inner, column))?;
                    sum = work.add(&sum, &product)?;
                }
                result.entries[row * other.cols + column] = sum;
            }
        }
        Ok(result)
    }

    fn equal(&self, other: &Self, work: &mut Work<'_, impl BudgetMeter>) -> Result<bool> {
        if self.rows != other.rows || self.cols != other.cols {
            return Ok(false);
        }
        for (left, right) in self.entries.iter().zip(&other.entries) {
            let limbs = [&left.0, &left.1, &right.0, &right.1]
                .iter()
                .map(|value| value.numer().limb_count() + value.denom().limb_count())
                .sum::<u64>();
            work.tick(limbs + 1)?;
            if left != right {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn is_hermitian(&self, work: &mut Work<'_, impl BudgetMeter>) -> Result<bool> {
        if self.rows != self.cols {
            return Ok(false);
        }
        for row in 0..self.rows {
            work.tick(1)?;
            // Checking only off-diagonal conjugacy would admit complex
            // diagonal entries, which cannot occur in a Hermitian projector.
            if !self.get(row, row).1.is_zero() {
                return Ok(false);
            }
            for column in 0..row {
                let left = self.get(row, column);
                let right = self.get(column, row);
                work.tick(
                    left.0.numer().limb_count()
                        + left.0.denom().limb_count()
                        + right.0.numer().limb_count()
                        + right.0.denom().limb_count()
                        + 1,
                )?;
                if left.0 != right.0 || !work.rational(&left.1, &right.1, Operation::Add)?.is_zero()
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn into_matrix(self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
        let mut entries = work.vector(self.entries.len())?;
        for (real, imaginary) in self.entries {
            work.tick(1)?;
            check_bits(&real, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
            check_bits(&imaginary, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
            if imaginary.is_zero() {
                entries.push(work.scalar_expression(real)?);
                continue;
            }
            work.storage(256)?;
            let imaginary = if imaginary.is_one() {
                Expr::Const(Constant::I)
            } else {
                Expr::Mul(vec![
                    work.scalar_expression(imaginary)?,
                    Expr::Const(Constant::I),
                ])
            };
            entries.push(if real.is_zero() {
                imaginary
            } else {
                Expr::Add(vec![work.scalar_expression(real)?, imaginary])
            });
        }
        work.tick(1)?;
        Matrix::new(self.rows, self.cols, entries)
    }

    fn realification(&self, work: &mut Work<'_, impl BudgetMeter>) -> Result<SparseMatrix> {
        let (rows, cols) = (self.rows, self.cols);
        let mut entries = BTreeMap::new();
        for row in 0..rows {
            for column in 0..cols {
                work.tick(1)?;
                let (real, imaginary) = self.get(row, column);
                if !real.is_zero() {
                    work.storage(1024)?;
                    let first = work.clone_scalar(real)?;
                    let second = work.clone_scalar(real)?;
                    entries.insert((row, column), work.scalar_expression(first)?);
                    entries.insert((row + rows, column + cols), work.scalar_expression(second)?);
                }
                if !imaginary.is_zero() {
                    work.storage(1024)?;
                    let upper = work.negate(imaginary)?;
                    let lower = work.clone_scalar(imaginary)?;
                    entries.insert((row, column + cols), work.scalar_expression(upper)?);
                    entries.insert((row + rows, column), work.scalar_expression(lower)?);
                }
            }
        }
        work.tick(entries.len() as u64 * 16 + 1)?;
        SparseMatrix::new(rows * 2, cols * 2, entries)
    }
}

fn propose(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
    let source = GaussianMatrix::read(matrix, work)?;
    let (rows, cols) = (source.rows, source.cols);
    let realified = source.realification(work)?;
    let candidate = generate_sparse_pseudoinverse_candidate(&realified, work)?;
    let proposal = candidate.witness.pseudoinverse;
    if proposal.rows != 2 * cols || proposal.columns != 2 * rows {
        return Err(MatrixError::InvalidCertificate(
            "realification candidate shape".into(),
        ));
    }
    let mut inverse = GaussianMatrix::zeros(cols, rows, work)?;
    let mut previous = None;
    for entry in proposal.entries {
        work.tick(1)?;
        if entry.row >= 2 * cols
            || entry.column >= 2 * rows
            || previous.is_some_and(|coordinate| coordinate >= (entry.row, entry.column))
        {
            return Err(MatrixError::InvalidCertificate(
                "realification candidate coordinates".into(),
            ));
        }
        previous = Some((entry.row, entry.column));
        check_bits(&entry.value, MAX_WORK_BITS)?;
        // Only the left blocks propose a complex result. Direct reference
        // checking below establishes the claim against the original A; it
        // does not infer acceptance from the remaining realification blocks.
        if entry.column < rows {
            if entry.row < cols {
                inverse.entries[entry.row * rows + entry.column].0 = entry.value;
            } else {
                inverse.entries[(entry.row - cols) * rows + entry.column].1 = entry.value;
            }
        }
    }
    inverse.into_matrix(work)
}

fn check_penrose(
    a: &GaussianMatrix,
    x: &GaussianMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(GaussianMatrix, GaussianMatrix)> {
    work.tick(1)?;
    if a.rows != x.cols || a.cols != x.rows {
        return Err(MatrixError::ShapeMismatch(a.rows, a.cols, x.rows, x.cols));
    }
    let ax = a.multiply(x, work)?;
    let xa = x.multiply(a, work)?;
    let axa = ax.multiply(a, work)?;
    if !axa.equal(a, work)? {
        return Err(MatrixError::InvalidCertificate("A X A != A".into()));
    }
    let xax = xa.multiply(x, work)?;
    if !xax.equal(x, work)? {
        return Err(MatrixError::InvalidCertificate("X A X != X".into()));
    }
    if !ax.is_hermitian(work)? {
        return Err(MatrixError::InvalidCertificate(
            "A X is not Hermitian".into(),
        ));
    }
    if !xa.is_hermitian(work)? {
        return Err(MatrixError::InvalidCertificate(
            "X A is not Hermitian".into(),
        ));
    }
    work.tick(1)?;
    Ok((ax, xa))
}

fn verify(matrix: &Matrix, inverse: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
    work.tick(1)?;
    matrix.validate_shape()?;
    inverse.validate_shape()?;
    if matrix.rows != inverse.cols || matrix.cols != inverse.rows {
        return Err(MatrixError::ShapeMismatch(
            matrix.rows,
            matrix.cols,
            inverse.rows,
            inverse.cols,
        ));
    }
    // Re-read both complete sources before even a zero-input shortcut.
    let a = GaussianMatrix::read(matrix, work)?;
    let x = GaussianMatrix::read(inverse, work)?;
    check_penrose(&a, &x, work)?;
    work.tick(1)
}

impl Matrix {
    /// Admit every original cell as bounded exact Q(i) arithmetic.
    /// This validates source shape and domains without constructing a result.
    pub fn validate_gaussian_pseudoinverse_source_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        let mut work = Work::new(meter);
        read_source(self, &mut work)?;
        work.tick(1)
    }

    /// Compute a bounded exact Q(i) pseudoinverse, then independently check
    /// all four Hermitian Penrose identities against the original cells.
    /// Source lowering, rational proposal generation, direct verification and
    /// output construction share one cumulative caller meter and intrinsic
    /// 20M-step/256MiB envelope. Each axis is at most 32; published coefficient
    /// components have at most 4096 bits. No protected verifier pool is reserved.
    pub fn metered_gaussian_pseudoinverse(&self, meter: &mut impl BudgetMeter) -> Result<Self> {
        let mut work = Work::new(meter);
        let inverse = propose(self, &mut work)?;
        verify(self, &inverse, &mut work)?;
        work.tick(1)?;
        Ok(inverse)
    }

    /// Independently check an untrusted exact Q(i) candidate, without using
    /// the generator's RREF, sparse products, solves or symbolic simplifier.
    pub fn verify_gaussian_pseudoinverse(
        &self,
        inverse: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        verify(self, inverse, &mut Work::new(meter))
    }
}

impl SparseMatrix {
    /// Bound and admit borrowed original sparse Q(i) cells before any dense
    /// conversion. Missing cells are exactly zero; each axis is at most 32.
    pub fn validate_gaussian_pseudoinverse_source_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        let mut work = Work::new(meter);
        work.tick(1)?;
        let (rows, cols) = (self.rows(), self.cols());
        let side = rows.max(cols);
        if side > MAX_HERMITIAN_INERTIA_DIMENSION {
            return Err(limit("complex dimension exceeds 32"));
        }
        // SparseMatrix keeps its storage private. Check the original
        // rectangle before a padded square could admit a misplaced cell.
        if self.entries().len() > rows * cols {
            return Err(MatrixError::InvalidCertificate(
                "too many original sparse Gaussian cells".into(),
            ));
        }
        for (&(row, column), value) in self.entries() {
            work.tick(1)?;
            if row >= rows || column >= cols {
                return Err(MatrixError::OutOfBounds(row, column, rows, cols));
            }
            if value.is_zero() {
                return Err(MatrixError::InvalidCertificate(
                    "sparse Gaussian source has an explicit zero".into(),
                ));
            }
        }
        read_gaussian_rational_matrix(
            side,
            side,
            self.entries()
                .iter()
                .map(|(&coordinate, value)| (coordinate, value)),
            &mut work,
        )
        .map_err(source_error)?;
        work.tick(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

    #[test]
    fn malformed_private_storage_is_rejected_before_rectangular_indexing() {
        for bad in [
            Matrix {
                rows: 1,
                cols: 1,
                data: Vec::new(),
            },
            Matrix {
                rows: 1,
                cols: 0,
                data: vec![Expr::Const(Constant::I)],
            },
        ] {
            assert!(matches!(
                bad.metered_gaussian_pseudoinverse(&mut Unbounded),
                Err(MatrixError::InvalidStorageLength(..)),
            ));
            assert!(matches!(
                bad.verify_gaussian_pseudoinverse(&Matrix::zeros(0, 1).unwrap(), &mut Unbounded,),
                Err(MatrixError::InvalidStorageLength(..)),
            ));
        }
    }

    #[test]
    fn hermitian_reference_checks_diagonal_reality_and_conjugate_pairs() {
        let one = BigRational::one();
        let zero = BigRational::zero();
        let imaginary_diagonal = GaussianMatrix {
            rows: 1,
            cols: 1,
            entries: vec![(one.clone(), one.clone())],
        };
        assert!(
            !imaginary_diagonal
                .is_hermitian(&mut Work::new(&mut Unbounded))
                .unwrap()
        );
        let symmetric_not_hermitian = GaussianMatrix {
            rows: 2,
            cols: 2,
            entries: vec![
                (one.clone(), zero.clone()),
                (zero.clone(), one.clone()),
                (zero.clone(), one.clone()),
                (one.clone(), zero.clone()),
            ],
        };
        assert!(
            !symmetric_not_hermitian
                .is_hermitian(&mut Work::new(&mut Unbounded))
                .unwrap()
        );
        let hermitian = GaussianMatrix {
            rows: 2,
            cols: 2,
            entries: vec![
                (one.clone(), zero.clone()),
                (zero.clone(), one.clone()),
                (zero.clone(), -one.clone()),
                (one, zero),
            ],
        };
        assert!(
            hermitian
                .is_hermitian(&mut Work::new(&mut Unbounded))
                .unwrap()
        );
    }
}
